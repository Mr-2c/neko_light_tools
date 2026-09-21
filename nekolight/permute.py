# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Element permutations between two orderings of the same mesh, and the
streaming record permutation used to reorder field and checkpoint files.

Two ``.nmsh`` files written from the same mesh (the original and a
``prepart`` output, two ``prepart`` outputs, or Neko's own ``_lb_`` mesh)
describe the same elements in different orders with renumbered element
ids; what identifies an element is its point ids (``prepart`` and Neko keep
them) or, failing that, its centroid.  :func:`element_permutation` returns,
for every element position of the target mesh, the position of the same
element in the source mesh, and :func:`verify_corners` checks the match
corner by corner.

:func:`gather_records` copies fixed-size records ``out[q] = in[perm[q]]``
in output chunks, reading the source records of each chunk in ascending
file order through a memory map (memory: two chunk buffers, plus the page
cache of the source); :func:`bucket_records` does the same with sequential
I/O only, through at most 512 bucket files.
"""
import os
import sys

import numpy as np

from .formats import iter_nmsh_elements

MAX_BUCKETS = 512


def _read_header(path):
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype='<i4', count=2)
    if hdr.size != 2 or int(hdr[1]) not in (2, 3):
        sys.exit('Error: %s is not a Neko .nmsh file' % path)
    return int(hdr[0]), int(hdr[1])


def _hash_rows(ids):
    """One uint64 per row from the sorted point ids of a row (FNV-style
    mixing).  A collision between two different elements is caught by the
    corner verification that always follows."""
    s = np.sort(ids.astype(np.uint64), axis=1)
    h = np.full(s.shape[0], 0xcbf29ce484222325, dtype=np.uint64)
    prime = np.uint64(0x100000001b3)
    with np.errstate(over='ignore'):
        for k in range(s.shape[1]):
            h = (h ^ (s[:, k] + np.uint64(0x9e3779b97f4a7c15))) * prime
            h ^= h >> np.uint64(29)
    return h


def id_keys(path, chunk=1 << 19):
    """(nelv,) uint64 hash of the sorted corner point ids of every element."""
    nelv, _ = _read_header(path)
    keys = np.empty(nelv, dtype=np.uint64)
    for s, e in iter_nmsh_elements(path, chunk):
        keys[s:s + e.shape[0]] = _hash_rows(e['v']['idx'])
    return keys


def centroids(path, chunk=1 << 19):
    """(nelv, 3) corner centroids and the bounding-box diagonal."""
    nelv, _ = _read_header(path)
    cent = np.empty((nelv, 3), dtype=np.float64)
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for s, e in iter_nmsh_elements(path, chunk):
        xyz = np.asarray(e['v']['xyz'], dtype=np.float64)
        cent[s:s + e.shape[0]] = xyz.mean(axis=1)
        lo = np.minimum(lo, xyz.reshape(-1, 3).min(axis=0))
        hi = np.maximum(hi, xyz.reshape(-1, 3).max(axis=0))
    return cent, float(np.linalg.norm(hi - lo))


def element_permutation(src_path, dst_path, mode='ids', tol=None, log=None):
    """For every element position q of ``dst_path`` the position p of the
    same element in ``src_path`` (0-based: ``data_dst[q] = data_src[perm[q]]``).
    ``mode='ids'`` matches the hashed sorted point ids; ``'centroids'``
    matches corner centroids within ``tol`` (default 1e-9 of the bounding
    box diagonal).  Both files must hold the same set of elements; a
    non-bijective match is an error.  With ``mode='ids'`` a mismatch falls
    back to centroids with a note."""
    n_src, g_src = _read_header(src_path)
    n_dst, g_dst = _read_header(dst_path)
    if (n_src, g_src) != (n_dst, g_dst):
        sys.exit('Error: %s (%d elements, %dD) and %s (%d elements, %dD) are '
                 'not the same mesh' % (src_path, n_src, g_src, dst_path,
                                        n_dst, g_dst))
    modes = (mode, 'centroids') if mode == 'ids' else (mode,)
    for try_mode in modes:
        if try_mode == 'ids':
            ks, kd = id_keys(src_path), id_keys(dst_path)
            order = np.argsort(ks, kind='stable')
            ks_sorted = ks[order]
            if ks_sorted.size > 1 and (ks_sorted[1:] == ks_sorted[:-1]).any():
                if log:
                    log('        note: repeated point-id keys in %s; matching '
                        'elements by centroid instead' % src_path)
                continue
            pos = np.searchsorted(ks_sorted, kd)
            pos[pos >= ks_sorted.size] = 0
            hit = ks_sorted[pos] == kd
            if not hit.all():
                if log:
                    log('        note: %d element(s) of %s have no point-id '
                        'match in %s; matching by centroid instead'
                        % (int((~hit).sum()), dst_path, src_path))
                continue
            perm = order[pos]
        else:
            from scipy.spatial import cKDTree
            cs, diag = centroids(src_path)
            cd, _ = centroids(dst_path)
            t = tol if tol is not None else 1e-9 * max(diag, 1e-300)
            d, perm = cKDTree(cs).query(cd)
            far = d > t
            if far.any():
                sys.exit('Error: %d element(s) of %s have no counterpart within '
                         '%.2e in %s: the files are not two orderings of the '
                         'same mesh (--tol relaxes the centroid tolerance)'
                         % (int(far.sum()), dst_path, t, src_path))
            perm = perm.astype(np.int64)
        if np.unique(perm).size != perm.size:
            if try_mode == 'ids':
                if log:
                    log('        note: point-id matching is not one-to-one; '
                        'matching by centroid instead')
                continue
            sys.exit('Error: element matching between %s and %s is not '
                     'one-to-one' % (src_path, dst_path))
        return perm
    sys.exit('Error: could not match the elements of %s and %s'
             % (src_path, dst_path))


def corner_mismatch(xs, xd, tol):
    """Largest distance from a corner of dst element to its nearest corner
    of the matched src element, requiring a one-to-one pairing within tol;
    xs, xd are (m, nv, 3).  Returns (worst distance, number of elements
    whose corners do not pair one-to-one)."""
    d = np.linalg.norm(xd[:, :, None, :] - xs[:, None, :, :], axis=3)  # (m,nv,nv)
    nearest = d.min(axis=2)
    k = d.argmin(axis=2)
    ks = np.sort(k, axis=1)
    one2one = (ks[:, 1:] != ks[:, :-1]).all(axis=1)
    bad = (nearest > tol).any(axis=1) | ~one2one
    return float(nearest.max()) if nearest.size else 0.0, int(bad.sum())


def verify_corners(src_path, dst_path, perm, rel_tol=1e-8, chunk=1 << 17):
    """Every dst element's corners must coincide (as a point set, within
    rel_tol of the coordinate scale) with those of src element perm[q]."""
    from .formats import read_nmsh
    ms = read_nmsh(src_path, mmap=True)
    scale = 0.0
    worst = 0.0
    nbad = 0
    for s, e in iter_nmsh_elements(dst_path, chunk):
        xd = np.asarray(e['v']['xyz'], dtype=np.float64)
        idx = perm[s:s + e.shape[0]]
        srt = np.argsort(idx, kind='stable')
        xs = np.asarray(ms.elems['v']['xyz'][idx[srt]], dtype=np.float64)
        inv = np.empty_like(srt); inv[srt] = np.arange(srt.size)
        xs = xs[inv]
        scale = max(scale, float(np.abs(xd).max()) if xd.size else 0.0)
        w, nb = corner_mismatch(xs, xd, rel_tol * max(1.0, scale))
        worst = max(worst, w); nbad += nb
    return worst, nbad, scale


def gather_records(src, dst, src_offset, rec_bytes, perm, chunk_bytes=1 << 29,
                   progress=None):
    """Append ``out[q] = in[perm[q]]`` for q = 0..len(perm)-1 to the open
    output file ``dst``, reading records of ``rec_bytes`` bytes from the
    memory-mapped source ``src`` (a ``np.memmap`` of uint8) starting at
    ``src_offset``.  Records of one output chunk are read in ascending
    source order.  Memory: two chunk buffers."""
    n = perm.size
    if n == 0:
        return
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    view = src[src_offset:src_offset + n * rec_bytes].view(
        np.dtype((np.void, rec_bytes)))
    for q0 in range(0, n, per_chunk):
        idx = perm[q0:q0 + per_chunk]
        srt = np.argsort(idx, kind='stable')
        recs = np.take(view, idx[srt])                 # ascending reads
        out = np.empty_like(recs)
        out[srt] = recs
        del recs
        out.tofile(dst)
        if progress:
            progress(q0 + idx.size, n)


def bucket_records(src_path, dst, src_offset, rec_bytes, perm, scratch_dir,
                   chunk_bytes=1 << 29, progress=None):
    """The same result as :func:`gather_records` with sequential I/O only:
    pass 1 reads the source block once and appends every record, tagged
    with its output position, to the bucket file of its output range; pass
    2 loads each bucket, orders it and appends it to ``dst``.  At most
    MAX_BUCKETS bucket files are open, so a bucket may exceed chunk_bytes
    for blocks larger than MAX_BUCKETS chunks (memory then follows the
    bucket size).  Scratch space: the block size plus 8 bytes per record.
    Bucket files are removed on success and on failure."""
    n = perm.size
    if n == 0:
        return
    inv = np.empty(n, dtype=np.int64)               # source pos -> output pos
    inv[perm] = np.arange(n)
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    nb = (n + per_chunk - 1) // per_chunk
    if nb > MAX_BUCKETS:
        nb = MAX_BUCKETS
        per_chunk = (n + nb - 1) // nb
        nb = (n + per_chunk - 1) // per_chunk
    tagged = np.dtype([('q', '<i8'), ('r', np.void, rec_bytes)])
    names = [os.path.join(scratch_dir, 'bucket_%06d.bin' % b) for b in range(nb)]
    files = []
    try:
        for nm in names:
            files.append(open(nm, 'wb'))
        with open(src_path, 'rb') as f:
            f.seek(src_offset)
            done = 0
            while done < n:
                m = min(per_chunk, n - done)
                recs = np.fromfile(f, dtype=np.dtype((np.void, rec_bytes)), count=m)
                if recs.size != m:
                    sys.exit('Error: truncated source block in %s' % src_path)
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
        for b, nm in enumerate(names):
            t = np.fromfile(nm, dtype=tagged)
            t = t[np.argsort(t['q'], kind='stable')]
            t['r'].tofile(dst)
            os.remove(nm)
            if progress:
                progress(n + min((b + 1) * per_chunk, n), 2 * n)
    finally:
        for fh in files:
            if not fh.closed:
                fh.close()
        for nm in names:
            if os.path.exists(nm):
                os.remove(nm)
