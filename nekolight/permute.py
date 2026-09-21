# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Element permutations between two orderings of the same mesh, and the
streaming record permutation used to reorder field and checkpoint files.

Two ``.nmsh`` files written from the same mesh (the original and a
``prepart`` output, or two ``prepart`` outputs) describe the same elements
in different orders with renumbered element ids; what identifies an element
is its point ids (``prepart`` keeps them) or, failing that, its centroid.
:func:`element_permutation` returns, for every element position of the
target mesh, the position of the same element in the source mesh.

:func:`gather_records` copies fixed-size records ``out[q] = in[perm[q]]``
in output chunks, reading the source records of each chunk in ascending
file order through a memory map; memory is one chunk, I/O is one pass over
the output and one (scattered, ascending) pass over the input.
"""
import os
import sys

import numpy as np

from .formats import iter_nmsh_elements, elem_dtype


def _read_header(path):
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype='<i4', count=2)
    if hdr.size != 2 or int(hdr[1]) not in (2, 3):
        sys.exit('Error: %s is not a Neko .nmsh file' % path)
    return int(hdr[0]), int(hdr[1])


def element_keys(path, mode='ids', chunk=1 << 20):
    """Per-element identity keys of a .nmsh in file order.

    ``mode='ids'``: the sorted corner point ids (exact; the same element in
    two prepart outputs has the same ids).  ``mode='centroids'``: the corner
    centroid rounded to 1e-12 of the bounding box (for files that do not
    share point ids, e.g. two independent conversions of the same mesh).
    Returns an (nelv, k) int64 array."""
    nelv, gdim = _read_header(path)
    nv = 8 if gdim == 3 else 4
    if mode == 'ids':
        keys = np.empty((nelv, nv), dtype=np.int64)
        for s, e in iter_nmsh_elements(path, chunk):
            keys[s:s + e.shape[0]] = np.sort(e['v']['idx'].astype(np.int64), axis=1)
        return keys
    if mode == 'centroids':
        cent = np.empty((nelv, 3), dtype=np.float64)
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for s, e in iter_nmsh_elements(path, chunk):
            xyz = np.asarray(e['v']['xyz'], dtype=np.float64)
            cent[s:s + e.shape[0]] = xyz.mean(axis=1)
            lo = np.minimum(lo, xyz.reshape(-1, 3).min(axis=0))
            hi = np.maximum(hi, xyz.reshape(-1, 3).max(axis=0))
        scale = max(float(np.linalg.norm(hi - lo)), 1e-300)
        return np.round((cent - lo) / scale * 1e12).astype(np.int64)
    raise ValueError(mode)


def _void_view(keys):
    k = np.ascontiguousarray(keys, dtype=np.int64)
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


def element_permutation(src_path, dst_path, mode='ids', log=None):
    """For every element position q of ``dst_path`` the position p of the
    same element in ``src_path`` (0-based, so ``data_dst[q] = data_src[perm[q]]``).
    Both files must hold the same set of elements; duplicates or misses
    are errors.  With ``mode='ids'`` a mismatch falls back to centroids
    with a note."""
    n_src, g_src = _read_header(src_path)
    n_dst, g_dst = _read_header(dst_path)
    if (n_src, g_src) != (n_dst, g_dst):
        sys.exit('Error: %s (%d elements, %dD) and %s (%d elements, %dD) are '
                 'not the same mesh' % (src_path, n_src, g_src, dst_path,
                                        n_dst, g_dst))
    for try_mode in ((mode, 'centroids') if mode == 'ids' else (mode,)):
        ks = _void_view(element_keys(src_path, try_mode))
        kd = _void_view(element_keys(dst_path, try_mode))
        order = np.argsort(ks, kind='stable')
        ks_sorted = ks[order]
        if ks_sorted.size > 1 and (ks_sorted[1:] == ks_sorted[:-1]).any():
            if try_mode == 'ids':
                if log:
                    log('        note: repeated point-id keys in %s; matching '
                        'elements by centroid instead' % src_path)
                continue
            sys.exit('Error: %s has elements with identical centroids; cannot '
                     'identify elements' % src_path)
        pos = np.searchsorted(ks_sorted, kd)
        pos[pos >= ks_sorted.size] = 0
        hit = ks_sorted[pos] == kd
        if not hit.all():
            if try_mode == 'ids':
                if log:
                    log('        note: %d element(s) of %s have no point-id '
                        'match in %s; matching by centroid instead'
                        % (int((~hit).sum()), dst_path, src_path))
                continue
            sys.exit('Error: %d element(s) of %s have no counterpart in %s: '
                     'the files are not two orderings of the same mesh'
                     % (int((~hit).sum()), dst_path, src_path))
        perm = order[pos]
        if np.unique(perm).size != perm.size:
            sys.exit('Error: element matching between %s and %s is not '
                     'one-to-one' % (src_path, dst_path))
        return perm
    sys.exit('Error: could not match the elements of %s and %s'
             % (src_path, dst_path))


def gather_records(src, dst, src_offset, rec_bytes, perm, chunk_bytes=1 << 29,
                   progress=None):
    """Append ``out[q] = in[perm[q]]`` for q = 0..len(perm)-1 to the open
    output file ``dst`` (positioned at the block start), reading records
    of ``rec_bytes`` bytes from the memory-mapped source ``src`` (a
    ``np.memmap`` of uint8) starting at ``src_offset``.  Records of one
    output chunk are read in ascending source order."""
    n = perm.size
    if n == 0:
        return
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    view = src[src_offset:src_offset + n * rec_bytes].view(
        np.dtype((np.void, rec_bytes)))
    for q0 in range(0, n, per_chunk):
        idx = perm[q0:q0 + per_chunk]
        srt = np.argsort(idx, kind='stable')
        recs = view[idx[srt]]                        # ascending reads
        inv = np.empty_like(srt)
        inv[srt] = np.arange(srt.size)
        recs[inv].tofile(dst)
        if progress:
            progress(q0 + idx.size, n)


def bucket_records(src_path, dst, src_offset, rec_bytes, perm, scratch_dir,
                   chunk_bytes=1 << 29, progress=None):
    """The same result as :func:`gather_records` with sequential I/O only:
    pass 1 reads the source block once and appends every record, tagged
    with its output position, to the bucket file of its output chunk; pass
    2 loads each bucket, orders it and appends it to ``dst``.  Needs
    scratch space equal to the block size (plus 8 bytes per record)."""
    n = perm.size
    if n == 0:
        return
    inv = np.empty(n, dtype=np.int64)               # source pos -> output pos
    inv[perm] = np.arange(n)
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    nb = (n + per_chunk - 1) // per_chunk
    tagged = np.dtype([('q', '<i8'), ('r', np.void, rec_bytes)])
    files = [open(os.path.join(scratch_dir, 'bucket_%06d.bin' % b), 'wb')
             for b in range(nb)]
    try:
        with open(src_path, 'rb') as f:
            f.seek(src_offset)
            done = 0
            while done < n:
                m = min(per_chunk, n - done)
                recs = np.fromfile(f, dtype=np.dtype((np.void, rec_bytes)), count=m)
                if recs.size != m:
                    sys.exit('Error: truncated source block')
                q = inv[done:done + m]
                b = q // per_chunk
                order = np.argsort(b, kind='stable')
                bs = b[order]
                bounds = np.flatnonzero(np.diff(bs)) + 1
                starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [m]])
                for s, e in zip(starts, ends):
                    t = np.empty(e - s, dtype=tagged)
                    t['q'] = q[order[s:e]]
                    t['r'] = recs[order[s:e]]
                    t.tofile(files[int(bs[s])])
                done += m
                if progress:
                    progress(done, 2 * n)
        for fh in files:
            fh.close()
        for b in range(nb):
            name = os.path.join(scratch_dir, 'bucket_%06d.bin' % b)
            t = np.fromfile(name, dtype=tagged)
            t = t[np.argsort(t['q'], kind='stable')]
            t['r'].tofile(dst)
            os.remove(name)
            if progress:
                progress(n + min((b + 1) * per_chunk, n), 2 * n)
    finally:
        for fh in files:
            if not fh.closed:
                fh.close()
