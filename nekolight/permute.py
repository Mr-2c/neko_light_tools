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

Three streaming methods copy fixed-size records ``out[q] = in[perm[q]]``
with two chunk buffers of memory: :func:`gather_records` writes the output
sequentially and reads the source records of each chunk in contiguous runs
(explicit, concurrent reads); :func:`scatter_records` reads the source
sequentially and writes each chunk's records to their destinations in
contiguous runs -- for a prepart ordering (original ids increasing inside
each block) a source chunk has at most one run per block, so this is the
natural direction on a parallel file system; :func:`bucket_records` uses
sequential I/O only, through at most 512 bucket files in scratch space.
:func:`run_stats` counts the runs of the first two for a given
permutation, which is how the tool picks between them.
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
    s = np.ascontiguousarray(np.sort(ids.astype(np.uint64), axis=1).T)   # (nv, n)
    h = np.full(s.shape[1], 0xcbf29ce484222325, dtype=np.uint64)
    prime = np.uint64(0x100000001b3)
    salt = np.uint64(0x9e3779b97f4a7c15)
    sh = np.uint64(29)
    with np.errstate(over='ignore'):
        for k in range(s.shape[0]):
            h ^= s[k] + salt
            h *= prime
            h ^= h >> sh
    return h


def _hash_xyz(xyz):
    """One uint64 per element from the bit patterns of its corner coordinates,
    independent of the corner order (a mixed hash per corner, summed)."""
    u = np.ascontiguousarray(xyz, dtype='<f8').view(np.uint64)   # (n, nv, 3)
    u = np.ascontiguousarray(np.transpose(u, (2, 0, 1)))          # (3, n, nv)
    with np.errstate(over='ignore'):
        h = u[0] * np.uint64(0x9e3779b97f4a7c15)
        h ^= u[1] * np.uint64(0xc2b2ae3d27d4eb4f)
        h ^= u[2] * np.uint64(0x165667b19e3779f9)
        h ^= h >> np.uint64(31)
        h *= np.uint64(0x7fb5d329728ea185)
        h ^= h >> np.uint64(27)
        return h.sum(axis=1, dtype=np.uint64)


class MeshKeys:
    """Per-element keys of a .nmsh collected in ONE sequential pass: ``ids``
    (int32 element ids), ``idhash`` (uint64 of the sorted point ids) and
    ``xyzhash`` (uint64 of the corner coordinate bits), 20 bytes per
    element."""
    def __init__(self, path, chunk=1 << 19):
        self.path = path
        self.nelv, self.gdim = _read_header(path)
        self.ids = np.empty(self.nelv, dtype=np.int32)
        self.idhash = np.empty(self.nelv, dtype=np.uint64)
        self.xyzhash = np.empty(self.nelv, dtype=np.uint64)
        for s, e in iter_nmsh_elements(path, chunk):
            m = e.shape[0]
            self.ids[s:s + m] = e['id']
            self.idhash[s:s + m] = _hash_rows(e['v']['idx'])
            self.xyzhash[s:s + m] = _hash_xyz(e['v']['xyz'])


def id_keys(path, chunk=1 << 19):
    """(nelv,) uint64 hash of the sorted corner point ids of every element."""
    return MeshKeys(path, chunk).idhash


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


def element_permutation(src_path, dst_path, mode='ids', tol=None, log=None,
                        keys=None):
    """For every element position q of ``dst_path`` the position p of the
    same element in ``src_path`` (0-based: ``data_dst[q] = data_src[perm[q]]``).
    ``mode='ids'`` matches the hashed sorted point ids; ``'centroids'``
    matches corner centroids within ``tol`` (default 1e-9 of the bounding
    box diagonal).  Both files must hold the same set of elements; a
    non-bijective match is an error.  With ``mode='ids'`` a mismatch falls
    back to centroids with a note.  ``keys``: optional (MeshKeys of src,
    MeshKeys of dst) already read."""
    if keys is None:
        keys = (MeshKeys(src_path), MeshKeys(dst_path))
    n_src, g_src = keys[0].nelv, keys[0].gdim
    n_dst, g_dst = keys[1].nelv, keys[1].gdim
    if (n_src, g_src) != (n_dst, g_dst):
        sys.exit('Error: %s (%d elements, %dD) and %s (%d elements, %dD) are '
                 'not the same mesh' % (src_path, n_src, g_src, dst_path,
                                        n_dst, g_dst))
    modes = (mode, 'centroids') if mode == 'ids' else (mode,)
    for try_mode in modes:
        if try_mode == 'ids':
            ks, kd = keys[0].idhash, keys[1].idhash
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


def verify_corners(src_path, dst_path, perm, rel_tol=1e-8, chunk=1 << 17, keys=None):
    """Every dst element's corners must coincide (as a point set, within
    rel_tol of the coordinate scale) with those of src element perm[q].
    With ``keys`` (MeshKeys of both files) elements whose corner coordinate
    hashes agree are accepted without reading the files again; only the
    others (none for prepart / Neko _lb_ output, which copy the coordinates
    bit for bit) are read and compared within the tolerance.  Returns
    (worst distance, number of elements that do not pair, coordinate scale)."""
    from .formats import read_nmsh
    if keys is not None:
        todo = np.flatnonzero(keys[0].xyzhash[perm] != keys[1].xyzhash)
        if todo.size == 0:
            return 0.0, 0, 0.0
    else:
        todo = None
    ms = read_nmsh(src_path, mmap=True)
    md = read_nmsh(dst_path, mmap=True)
    scale = 0.0
    worst = 0.0
    nbad = 0
    n = perm.size
    for s in range(0, n if todo is None else todo.size, chunk):
        q = np.arange(s, min(s + chunk, n)) if todo is None else todo[s:s + chunk]
        xd = np.asarray(md.elems['v']['xyz'][q], dtype=np.float64)
        idx = perm[q]
        srt = np.argsort(idx, kind='stable')
        xs = np.asarray(ms.elems['v']['xyz'][idx[srt]], dtype=np.float64)
        inv = np.empty_like(srt); inv[srt] = np.arange(srt.size)
        xs = xs[inv]
        scale = max(scale, float(np.abs(xd).max()) if xd.size else 0.0)
        w, nb = corner_mismatch(xs, xd, rel_tol * max(1.0, scale))
        worst = max(worst, w); nbad += nb
    return worst, nbad, scale


def _runs(sorted_pos):
    """Maximal runs of consecutive positions in a sorted int array: (k0, L)
    with run r covering sorted_pos[k0[r]:k0[r] + L[r]]."""
    if sorted_pos.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = np.flatnonzero(np.diff(sorted_pos) != 1) + 1
    k0 = np.concatenate([[0], brk]).astype(np.int64)
    k1 = np.concatenate([brk, [sorted_pos.size]]).astype(np.int64)
    return k0, k1 - k0


def _pread_into(fd, mv, offset):
    """Fill the byte buffer mv from fd at offset (loops over short reads)."""
    done = 0
    n = len(mv)
    while done < n:
        if hasattr(os, 'preadv'):
            got = os.preadv(fd, [mv[done:]], offset + done)
        else:
            chunk = os.pread(fd, n - done, offset + done)
            got = len(chunk)
            mv[done:done + got] = chunk
        if got <= 0:
            raise IOError('short read at offset %d' % (offset + done))
        done += got


def _pwrite_all(fd, mv, offset):
    done = 0
    n = len(mv)
    while done < n:
        put = os.pwrite(fd, mv[done:], offset + done)
        if put <= 0:
            raise IOError('short write at offset %d' % (offset + done))
        done += put


def _read_full(f, mv):
    done = 0
    n = len(mv)
    while done < n:
        got = f.readinto(mv[done:])
        if not got:
            raise IOError('truncated source block')
        done += got


GATHER_GAP = 1 << 16     # gather reads bridge gaps up to this many bytes
IOV_MAX = 1024


def _preadv_all(fd, iov, offset):
    """Fill the buffers of iov consecutively from fd at offset (loops over
    short reads)."""
    total = sum(len(b) for b in iov)
    done = 0
    while done < total:
        got = os.preadv(fd, iov, offset + done)
        if got <= 0:
            raise IOError('short read at offset %d' % (offset + done))
        done += got
        if done < total:
            skip = got
            while skip >= len(iov[0]):
                skip -= len(iov[0])
                iov = iov[1:]
            if skip:
                iov = [iov[0][skip:]] + iov[1:]


def _read_runs(fd, mv, k0, L, pos, rec, base, r0, r1):
    """Read runs r0..r1-1 of a chunk; runs separated by at most GATHER_GAP
    bytes are fetched by one vectored read that drops the gap bytes into a
    sink buffer."""
    if not hasattr(os, 'preadv'):
        for r in range(r0, r1):
            a = int(k0[r]) * rec
            _pread_into(fd, mv[a:a + int(L[r]) * rec], base + int(pos[k0[r]]) * rec)
        return
    sink = memoryview(bytearray(GATHER_GAP))
    r = r0
    while r < r1:
        a = int(k0[r]) * rec
        iov = [mv[a:a + int(L[r]) * rec]]
        off = base + int(pos[k0[r]]) * rec
        end = int(pos[k0[r]]) + int(L[r])
        r += 1
        while r < r1 and len(iov) < IOV_MAX - 1:
            gap = (int(pos[k0[r]]) - end) * rec
            if gap > GATHER_GAP:
                break
            if gap:
                iov.append(sink[:gap])
            a = int(k0[r]) * rec
            iov.append(mv[a:a + int(L[r]) * rec])
            end = int(pos[k0[r]]) + int(L[r])
            r += 1
        _preadv_all(fd, iov, off)


def _write_runs(fd, mv, k0, L, pos, rec, base, r0, r1):
    for r in range(r0, r1):
        a = int(k0[r]) * rec
        _pwrite_all(fd, mv[a:a + int(L[r]) * rec], base + int(pos[k0[r]]) * rec)


def _run_jobs(pool, fn, fd, mv, k0, L, pos, rec, base, threads):
    """Apply fn(fd, mv, k0, L, pos, rec, base, r0, r1) to `threads * 4`
    groups of consecutive runs, one thread pool task per group."""
    if k0.size == 0:
        return

    def work(r0, r1):
        fn(fd, mv, k0, L, pos, rec, base, r0, r1)

    ngroups = max(1, min(threads * 4, k0.size))
    bounds = np.linspace(0, k0.size, ngroups + 1).astype(int)
    futs = [pool.submit(work, int(bounds[g]), int(bounds[g + 1]))
            for g in range(ngroups) if bounds[g + 1] > bounds[g]]
    try:
        for fut in futs:
            fut.result()
    except BaseException:
        for fut in futs:
            fut.cancel()
        raise


IO_LATENCY_BYTES = 1 << 20   # bytes a stream moves in the time one small I/O takes


def run_stats(perm, rec_bytes, chunk_bytes=1 << 29):
    """I/O statistics of the two streaming directions for records of
    rec_bytes handled in chunks of chunk_bytes: {'gather': (reads, bytes),
    'scatter': (writes, bytes), 'best': name}.  gather counts the vectored
    reads it would issue (runs closer than GATHER_GAP merge, at most
    IOV_MAX segments per call) and the bytes they move, bridged gaps
    included; scatter counts one write per destination run.  'best'
    minimises calls * IO_LATENCY_BYTES + bytes."""
    n = perm.size
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n, dtype=perm.dtype)
    g_calls = g_bytes = s_calls = 0
    for s0 in range(0, n, per_chunk):
        x = np.sort(perm[s0:s0 + per_chunk])
        k0, L = _runs(x)
        gaps = (x[k0[1:]] - (x[k0[:-1]] + L[:-1])) * rec_bytes
        bridged = gaps <= GATHER_GAP
        nb = int(np.count_nonzero(bridged))
        g_calls += k0.size - nb + nb // (IOV_MAX // 2)
        g_bytes += x.size * rec_bytes + int(gaps[bridged].sum())
        y = np.sort(inv[s0:s0 + per_chunk])
        s_calls += 1 + int(np.count_nonzero(np.diff(y) != 1))
    s_bytes = n * rec_bytes
    cost_g = g_calls * IO_LATENCY_BYTES + g_bytes
    cost_s = s_calls * IO_LATENCY_BYTES + s_bytes
    return {'gather': (g_calls, g_bytes), 'scatter': (s_calls, s_bytes),
            'best': 'scatter' if cost_s < cost_g else 'gather'}


def gather_records(src_path, dst, src_offset, rec_bytes, perm, chunk_bytes=1 << 29,
                   threads=8, progress=None):
    """Append ``out[q] = in[perm[q]]`` for q = 0..len(perm)-1 to the open
    output file ``dst`` (positioned at the block start): the output is
    written sequentially chunk by chunk; the source records of a chunk are
    read in ascending file order, one explicit read per contiguous run
    (runs closer than GATHER_GAP bytes share a vectored read), ``threads``
    groups of runs at a time.  Memory: two chunk buffers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    n = perm.size
    if n == 0:
        return
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    rec_dt = np.dtype((np.void, rec_bytes))
    local = threading.local()
    fds = []

    def read_runs(fd_unused, *a):
        # one descriptor per worker thread keeps the kernel's read-ahead
        # state of the threads apart
        if not hasattr(local, 'fd'):
            local.fd = os.open(src_path, os.O_RDONLY)
            fds.append(local.fd)
        _read_runs(local.fd, *a)

    try:
      with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for q0 in range(0, n, per_chunk):
            idx = perm[q0:q0 + per_chunk]
            m = idx.size
            srt = np.argsort(idx, kind='stable')
            ps = idx[srt]
            recs = np.empty(m * rec_bytes, dtype=np.uint8)
            k0, L = _runs(ps)
            _run_jobs(pool, read_runs, None, memoryview(recs), k0, L, ps, rec_bytes,
                      src_offset, threads)
            out = np.empty(m, dtype=rec_dt)
            out[srt] = recs.view(rec_dt)
            del recs
            out.tofile(dst)
            del out
            if progress:
                progress(q0 + m, n)
    finally:
        for fd in fds:
            os.close(fd)


def inverse_permutation(perm):
    """inv with inv[perm[q]] = q; perm must be a permutation of 0..n-1."""
    n = perm.size
    inv = np.full(n, -1, dtype=perm.dtype)
    inv[perm] = np.arange(n, dtype=perm.dtype)
    if n and (inv < 0).any():
        raise ValueError('perm is not a permutation')
    return inv


def scatter_records(src_path, dst, src_offset, dst_offset, rec_bytes, perm,
                    chunk_bytes=1 << 29, threads=8, progress=None, inv=None):
    """The same result with the SOURCE read sequentially: each source chunk
    is read in one piece and its records are written to their destination
    positions (``dst`` must already have the block's full size), one
    explicit write per contiguous run of destinations, ``threads`` runs at
    a time.  For a prepart ordering (original ids increasing inside each
    block) a source chunk has at most one run per block.  Memory: two chunk
    buffers plus 8 bytes per element (``inv``, the inverse permutation,
    may be passed in to share it between blocks)."""
    from concurrent.futures import ThreadPoolExecutor
    n = perm.size
    if n == 0:
        return
    if inv is None:
        inv = inverse_permutation(perm)
    per_chunk = max(1, int(chunk_bytes // rec_bytes))
    rec_dt = np.dtype((np.void, rec_bytes))
    dst.flush()
    fd_out = dst.fileno()
    with open(src_path, 'rb', buffering=0) as f, \
            ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        f.seek(src_offset)
        for p0 in range(0, n, per_chunk):
            m = min(per_chunk, n - p0)
            buf = np.empty(m * rec_bytes, dtype=np.uint8)
            _read_full(f, memoryview(buf))
            q = inv[p0:p0 + m]
            order = np.argsort(q, kind='stable')
            qs = q[order]
            data = np.take(buf.view(rec_dt), order).view(np.uint8)
            del buf
            k0, L = _runs(qs)
            _run_jobs(pool, _write_runs, fd_out, memoryview(data), k0, L, qs, rec_bytes,
                      dst_offset, threads)
            del data
            if progress:
                progress(p0 + m, n)
    dst.seek(dst_offset + n * rec_bytes)


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
    import tempfile
    tdir = tempfile.mkdtemp(prefix='permute_buckets_', dir=scratch_dir)
    names = [os.path.join(tdir, 'bucket_%06d.bin' % b) for b in range(nb)]
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
        try:
            os.rmdir(tdir)
        except OSError:
            pass
