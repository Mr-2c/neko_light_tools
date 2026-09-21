#!/usr/bin/env python3
# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see nekolight/formats.py for the full
# licence text (part of Neko; see also Neko's COPYING).
#
#     _  __  ____  __ __  ____
#    / |/ / / __/ / //_/ / __ \
#   /    / / _/  / ,<   / /_/ /
#  /_/|_/ /___/ /_/|_|  \____/
#
"""permute_fields -- reorder the elements of Neko field files (.fld series,
name0.f00012) and checkpoints (.chkp) from one element ordering of a mesh
to another, so that a run on a re-partitioned mesh (prepart.py, or Neko's
own <mesh>_lb_<ranks>.nmsh) can restart from files written on a different
ordering.

Neko places field and checkpoint data by element POSITION: rank r of a run
owns the positions of its linear element distribution, and the file holds
the elements in the mesh file's order.  Two .nmsh files written from the
same mesh describe the same elements in different orders, so the fix is a
gather of fixed-size per-element records: for every block of the file,
out[q] = in[perm[q]] where perm maps positions of the target mesh to
positions of the source mesh.  Nothing inside a record changes, headers and
the non-element blocks (tlag/dtlag, ALE tracker arrays) are copied
verbatim, and the idx column of a field file gets the element ids of the
target mesh.  A fluid restarted from a permuted checkpoint reproduces the
un-permuted restart bit for bit (Neko copies the fields verbatim when mesh
and polynomial order match); a scalar differs by round-off because Neko
re-applies its gather-scatter on restart.

The permutation comes from the two mesh files: elements are matched by
their point ids (prepart keeps them) and verified by their corner
coordinates; files that do not share point ids (two independent
conversions of one mesh, or Neko's _lb_ mesh with merged periodic ids) are
matched by centroids.

Memory is one chunk (--chunk-mb); the source is read through a memory map,
one output chunk at a time with its source records fetched in ascending
file order, and the output is written sequentially.  With --scratch DIR the
tool instead makes two sequential passes through bucket files (needs
scratch space of the file size), which is faster on file systems that
dislike scattered reads.

Usage:
  permute_fields.py SRC_MESH.nmsh DST_MESH.nmsh -o OUTDIR FILE [FILE ...]
  permute_fields.py old/512.nmsh new/512_96.nmsh -o new/ old/field0.f00003 old/fluid00003.chkp
  permute_fields.py old.nmsh new.nmsh -o new/ old/field0.nek5000 old/field0.f0000*
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

from nekolight import banner, iter_nmsh_elements, read_nmsh
from nekolight.permute import element_permutation, gather_records, bucket_records
from nekolight.fldchkp import fld_layout, chkp_layout, read_fld_idx


def log(msg):
    print(msg, flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        prog='permute_fields.py',
        description='Reorder the elements of Neko .fld / .chkp files from the '
                    'element order of SRC_MESH to that of DST_MESH.')
    ap.add_argument('src_mesh', help='.nmsh the files were written with')
    ap.add_argument('dst_mesh', help='.nmsh of the run that will read them')
    ap.add_argument('files', nargs='+', help='.fld / name0.f00012 / .chkp / '
                    '.nek5000 files (the .nek5000 series index is copied)')
    ap.add_argument('-o', '--outdir', required=True,
                    help='output directory (files keep their names)')
    ap.add_argument('--chunk-mb', type=float, default=512.0,
                    help='working memory per block in MB (default 512)')
    ap.add_argument('--scratch', metavar='DIR',
                    help='two-pass sequential method with bucket files in DIR')
    ap.add_argument('--rp', type=int, choices=(4, 8), default=None,
                    help='real kind of the checkpoint (default: inferred '
                         'from the file size; needed for ALE checkpoints)')
    ap.add_argument('--match', choices=('ids', 'centroids'), default='ids',
                    help='element identity: point ids with coordinate check '
                         '(default) or centroids')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing output files')
    return ap.parse_args()


def element_ids(path):
    ids = []
    for _, e in iter_nmsh_elements(path):
        ids.append(e['id'].astype(np.int64))
    return np.concatenate(ids)


def verify_geometry(src_mesh, dst_mesh, perm, chunk=1 << 18):
    """Corner coordinates of dst element q must equal those of src element
    perm[q] (as sets, to tolerate a different vertex rotation)."""
    ms = read_nmsh(src_mesh, mmap=True)
    worst = 0.0
    scale = 0.0
    for s, e in iter_nmsh_elements(dst_mesh, chunk):
        xd = np.asarray(e['v']['xyz'], dtype=np.float64)
        idx = perm[s:s + e.shape[0]]
        srt = np.argsort(idx, kind='stable')
        xs = np.asarray(ms.elems['v']['xyz'][idx[srt]], dtype=np.float64)
        inv = np.empty_like(srt); inv[srt] = np.arange(srt.size)
        xs = xs[inv]
        # compare sorted corner tuples (lexicographic) per element
        ks = np.sort(xs.reshape(xs.shape[0], -1), axis=1)
        kd = np.sort(xd.reshape(xd.shape[0], -1), axis=1)
        worst = max(worst, float(np.abs(ks - kd).max()) if ks.size else 0.0)
        scale = max(scale, float(np.abs(xd).max()) if xd.size else 0.0)
    return worst, scale


def permute_file(path, out, perm, dst_ids, args, srcmm_cache):
    name = os.path.basename(path)
    low = name.lower()
    if low.endswith('.nek5000'):
        shutil.copyfile(path, out)
        log('  %-40s copied (series index)' % name)
        return
    if low.endswith('.chkp'):
        lay = chkp_layout(path, args.rp)
        if lay.nelgv != perm.size:
            sys.exit('Error: %s holds %d elements, the meshes %d' % (path, lay.nelgv, perm.size))
        kind = 'checkpoint: %d elements, gdim %d, lx %d, %d-byte reals, time %g, blocks %s' % (
            lay.nelgv, lay.gdim, lay.lx, lay.rp, lay.time,
            ' '.join(b.name for b in lay.blocks if b.rec_bytes))
        blocks = lay.blocks
        idx_block = None
    else:
        lay = fld_layout(path)
        if lay.nelgv != perm.size:
            sys.exit('Error: %s holds %d elements, the meshes %d' % (path, lay.nelgv, perm.size))
        kind = 'field file: %d elements, lx %d ly %d lz %d, %d-byte reals, time %g, %s' % (
            lay.nelgv, lay.lx, lay.ly, lay.lz, lay.wdsize, lay.time, lay.rdcode)
        blocks = lay.blocks
        idx_block = lay.blocks[1]
        idx = read_fld_idx(path, lay)
        src_ids = srcmm_cache['src_ids']
        if not np.array_equal(idx, src_ids):
            if np.array_equal(idx, np.arange(1, perm.size + 1)) and \
                    not np.array_equal(src_ids, np.arange(1, perm.size + 1)):
                log('        warning: %s: idx column is 1..N although the '
                    'source mesh ids are not (masked/subset output?); '
                    'proceeding by position' % name)
            else:
                sys.exit('Error: %s: its idx column does not hold the element '
                         'ids of %s -- was this file written on that mesh?'
                         % (path, args.src_mesh))
    log('  %-40s %s' % (name, kind))
    t0 = time.time()
    src = np.memmap(path, dtype=np.uint8, mode='r')
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        for b in blocks:
            if b.rec_bytes == 0:
                f.write(bytes(src[b.offset:b.offset + b.nbytes]))
            elif b is idx_block:
                dst_ids.astype('<i4').tofile(f)
            elif args.scratch:
                bucket_records(path, f, b.offset, b.rec_bytes, perm,
                               args.scratch, int(args.chunk_mb * 2 ** 20))
            else:
                gather_records(src, f, b.offset, b.rec_bytes, perm,
                               int(args.chunk_mb * 2 ** 20))
    if os.path.getsize(tmp) != lay.size:
        os.remove(tmp)
        sys.exit('Error: internal: output size mismatch for %s' % name)
    os.replace(tmp, out)
    log('  %-40s -> %s  (%.1f s, %.1f MB)' % ('', out, time.time() - t0,
                                             lay.size / 2 ** 20))


def main():
    args = parse_args()
    log(banner('permute_fields'))
    log('  source mesh : %s' % args.src_mesh)
    log('  target mesh : %s' % args.dst_mesh)
    if not os.path.isdir(args.outdir):
        sys.exit('Error: output directory %s does not exist' % args.outdir)
    outs = []
    for p in args.files:
        if not os.path.isfile(p):
            sys.exit('Error: %s not found' % p)
        out = os.path.join(args.outdir, os.path.basename(p))
        if os.path.abspath(out) == os.path.abspath(p):
            sys.exit('Error: %s would overwrite the input (choose another -o)' % p)
        if os.path.exists(out) and not args.force:
            sys.exit('Error: %s exists (use --force to overwrite)' % out)
        outs.append(out)
    if args.scratch and not os.path.isdir(args.scratch):
        sys.exit('Error: scratch directory %s does not exist' % args.scratch)
    log('  [1/3] matching elements of the two meshes ...')
    t0 = time.time()
    perm = element_permutation(args.src_mesh, args.dst_mesh, args.match, log)
    worst, scale = verify_geometry(args.src_mesh, args.dst_mesh, perm)
    if worst > 1e-8 * max(1.0, scale):
        if args.match == 'ids':
            log('        note: point ids match but coordinates differ by %.3e; '
                'matching by centroids instead' % worst)
            perm = element_permutation(args.src_mesh, args.dst_mesh, 'centroids', log)
            worst, scale = verify_geometry(args.src_mesh, args.dst_mesh, perm)
        if worst > 1e-8 * max(1.0, scale):
            sys.exit('Error: the two meshes do not hold the same elements '
                     '(corner coordinates differ by up to %.3e)' % worst)
    moved = int((perm != np.arange(perm.size)).sum())
    log('        %d elements, %d change position, corner coordinates agree to '
        '%.1e  (%.1f s)' % (perm.size, moved, worst, time.time() - t0))
    src_ids = element_ids(args.src_mesh)
    dst_ids = element_ids(args.dst_mesh)
    log('  [2/3] permuting %d file(s) -> %s' % (len(args.files), args.outdir))
    cache = {'src_ids': src_ids}
    for p, out in zip(args.files, outs):
        permute_file(p, out, perm, dst_ids, args, cache)
    log('  [3/3] done')
    log('  Restart the target run with the permuted files as usual (no '
        'restart_mesh_file: that would interpolate instead of copying).')


if __name__ == '__main__':
    main()
