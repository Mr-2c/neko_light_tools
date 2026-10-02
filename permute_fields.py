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
positions of the source mesh.  Nothing inside a record changes; headers and
the non-element blocks (tlag/dtlag, ALE tracker arrays) are copied
verbatim, stale bytes after the layout (Neko never truncates an output
file) are dropped, and the idx column of a field file gets the element ids
of the target mesh.  A fluid restarted from a permuted checkpoint
reproduces the un-permuted restart bit for bit (Neko copies the fields
verbatim when mesh and polynomial order match); a scalar differs by
round-off because Neko re-applies its gather-scatter on restart.

Safety: the two meshes must hold the same elements (matched by point ids,
which prepart and Neko keep, or by centroids for files that do not share
ids -- create_periodic_zones output, independent conversions), verified
corner by corner; a field file's idx column must hold the source mesh's
element ids, and when the file carries coordinates (the first file of a
series) they are checked element by element against the source mesh: they
must coincide with its corners or, when the run deformed the mesh
(user_mesh_setup), at least agree wherever the mesh shares a corner between
two elements -- this is what catches swapped mesh arguments.  Checkpoints carry no element
identity at all: for them the source mesh is taken on trust, so give a
coordinate-carrying field file of the same run in the same call when in
doubt.

Output names: every file is written as PREFIX + its name (--prefix,
default 'permuted_'), so an input is never overwritten and the two
orderings cannot be confused.  Neko builds a series' file names from the
base name, so the prefix keeps the files loadable: refer to them in the
case file as "file_name": "permuted_field0.fld" (with "sample_index") or
"restart_file": "permuted_fluid00001.chkp"; the copied .nek5000 index gets
the prefixed template, so ParaView/VisIt open the permuted series.

Streaming: the records of a block are moved in chunks (--chunk-mb, 512
MB) in one of two directions.  gather writes the output sequentially and
reads the source records of each chunk in ascending order, one explicit
read per contiguous run (runs closer than 64 kB share one vectored read);
scatter reads the source sequentially and writes the records of each chunk
to their destinations, one write per contiguous run.  A prepart ordering
keeps the original ids increasing inside each block, so in the scatter
direction a source chunk has at most one run per block -- megabytes at a
time, which is what a parallel file system needs -- while the gather
direction can degrade to runs of a few records.  For every record size
the tool counts the reads and bytes of one direction and the writes of
the other (--method auto, the default, takes the cheaper one, counting a
call as one megabyte; the numbers are logged) and issues the run
reads/writes from --threads (8) threads.  The two-pass --scratch DIR
method (sequential I/O only, through at most 512 bucket files) remains
for permutations whose runs are short in both directions.

Memory: element matching is one sequential pass over each mesh (20 bytes
per element and mesh: ids, a hash of the sorted point ids, a hash of the
corner coordinates -- corners whose hashes agree need no second look),
of which the ids (4 bytes per element and mesh) stay for the run; the
record streaming two chunks (--chunk-mb) plus the permutation and its
inverse (16 bytes per element); the two-pass --scratch method three chunks
(or the bucket size for blocks above 512 chunks) and scratch space of the
file size.  A 41.8-million-element file runs in about 2 GB.

Usage:
  permute_fields.py SRC_MESH.nmsh DST_MESH.nmsh -o OUTDIR [--prefix permuted_]
                    [--method auto|gather|scatter] [--threads 8] [--chunk-mb 512]
                    [--scratch DIR] FILE [FILE ...]
  permute_fields.py old/512.nmsh new/512_96.nmsh -o new/ old/field0.f00000 old/fluid00003.chkp
      -> new/permuted_field0.f00000  new/permuted_fluid00003.chkp
  permute_fields.py old.nmsh new.nmsh -o new/ old/field0.nek5000 old/field0.f0000*
"""
import argparse
import os
import re
import shutil
import sys
import time

import numpy as np

from nekolight import banner, read_nmsh
from nekolight.permute import (MeshKeys, element_permutation, verify_corners,
                               corner_mismatch, gather_records, scatter_records,
                               bucket_records, run_stats, inverse_permutation)
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
                    help='output directory (files are written as PREFIX + name)')
    ap.add_argument('--prefix', default='permuted_',
                    help="prefix of the output names (default permuted_; '' keeps "
                         "the names, which is refused when it would overwrite an input)")
    ap.add_argument('--chunk-mb', type=float, default=512.0,
                    help='working chunk per block in MB (default 512)')
    ap.add_argument('--method', choices=('auto', 'gather', 'scatter'), default='auto',
                    help='streaming direction: gather (sequential output, source read '
                         'in runs), scatter (sequential source, output written in runs) '
                         'or auto (the one with fewer, longer runs; default)')
    ap.add_argument('--threads', type=int, default=8,
                    help='concurrent run reads/writes per chunk (default 8)')
    ap.add_argument('--scratch', metavar='DIR',
                    help='two-pass sequential method with bucket files in DIR '
                         '(instead of --method)')
    ap.add_argument('--rp', type=int, choices=(4, 8), default=None,
                    help='real kind of the checkpoints (default: inferred '
                         'from the file size)')
    ap.add_argument('--match', choices=('ids', 'centroids'), default='ids',
                    help='element identity: point ids (default, falls back to '
                         'centroids) or centroids')
    ap.add_argument('--tol', type=float, default=None,
                    help='centroid matching tolerance (default 1e-9 of the '
                         'bounding-box diagonal)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing output files')
    return ap.parse_args()


class CoordCheck:
    """Result of check_fld_coordinates."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def check_fld_coordinates(path, lay, src_mesh, chunk):
    """Two tests of a field file's X block against the source mesh, element
    by element.  (a) Absolute: the corner GLL nodes of element q coincide
    with the corners of mesh element q (abs_worst, abs_nbad, mscale).  (b)
    Consistency: corner nodes that the mesh shares between elements (same
    point id) carry the same coordinates in the file (con_worst over
    con_nshared shared corners, fscale).  Neko evaluates the coordinates per
    element from its (possibly user-deformed) points, so (b) holds for any
    run-time deformation of the mesh -- user_mesh_setup, apply_deform --
    while it fails for a file written on another element ordering."""
    blk = [b for b in lay.blocks if b.name == 'X'][0]
    lx, ly, lz, g = lay.lx, lay.ly, lay.lz, lay.gdim
    lxyz = lx * ly * lz
    ii = np.array([0, lx - 1]); jj = np.array([0, ly - 1])
    kk = np.array([0, lz - 1]) if g == 3 else np.array([0])
    corner_nodes = np.array([i + lx * (j + ly * k) for k in kk for j in jj for i in ii])
    # nmsh corner order (cyclic) -> GLL corner order (i fastest): Neko's reader swap
    swap = [0, 1, 3, 2, 4, 5, 7, 6] if g == 3 else [0, 1, 3, 2]
    ft = '<f4' if lay.wdsize == 4 else '<f8'
    mm = np.memmap(path, dtype=ft, mode='r', offset=blk.offset,
                   shape=(lay.nelgv, g, lxyz))
    ms = read_nmsh(src_mesh, mmap=True)
    nv = 8 if ms.gdim == 3 else 4
    if (ms.gdim == 3) != (g == 3):
        sys.exit('Error: %s is a %dD field file but %s is a %dD mesh'
                 % (path, g, src_mesh, ms.gdim))
    r = CoordCheck(abs_worst=0.0, abs_nbad=0, mscale=0.0,
                   con_worst=0.0, con_nshared=0, fscale=0.0)
    rel_tol = 1e-5 if lay.wdsize == 4 else 1e-9
    pad = lambda a: (np.concatenate([a, np.zeros(a.shape[:2] + (3 - g,))], axis=2)
                     if g == 2 else a)
    for s in range(0, lay.nelgv, chunk):
        e = min(s + chunk, lay.nelgv)
        xf = np.asarray(mm[s:e][:, :, corner_nodes], dtype=np.float64)   # (m, g, nv)
        xf = np.ascontiguousarray(np.transpose(xf, (0, 2, 1)))           # (m, nv, g)
        xm = np.asarray(ms.elems['v']['xyz'][s:e], dtype=np.float64)[:, :nv, :g]
        r.mscale = max(r.mscale, float(np.abs(xm).max()))
        r.fscale = max(r.fscale, float(np.abs(xf).max()))
        w, nb = corner_mismatch(pad(xm), pad(xf), rel_tol * max(1.0, r.mscale))
        r.abs_worst = max(r.abs_worst, w); r.abs_nbad += nb
        # (b): group the corner slots of this chunk by mesh point id
        ids = np.asarray(ms.elems['v']['idx'][s:e])[:, swap].ravel()
        order = np.argsort(ids, kind='stable')
        sid = ids[order]; sx = xf.reshape(-1, g)[order]
        new = np.r_[True, sid[1:] != sid[:-1]]
        rep = sx[np.flatnonzero(new)][np.cumsum(new) - 1]
        dev = np.linalg.norm(sx - rep, axis=1)
        r.con_worst = max(r.con_worst, float(dev.max()))
        r.con_nshared += int((~new).sum())
    r.rel_tol = rel_tol
    r.con_ok = r.con_nshared > 0 and r.con_worst <= rel_tol * max(1.0, r.fscale)
    return r


def nek5000_members(path, outdir):
    """(template, first, n) of a .nek5000 index and the members missing in
    outdir."""
    txt = open(path).read()
    m1 = re.search(r'filetemplate:\s*(\S+)', txt)
    m2 = re.search(r'firsttimestep:\s*(\d+)', txt)
    m3 = re.search(r'numtimesteps:\s*(\d+)', txt)
    if not (m1 and m2 and m3):
        return None
    tmpl, first, n = m1.group(1), int(m2.group(1)), int(m3.group(1))
    missing = []
    for k in range(first, first + n):
        try:
            name = tmpl % (0, k)
        except TypeError:
            return None
        if not os.path.exists(os.path.join(outdir, name)):
            missing.append(name)
    return missing


def fmt_bytes(b):
    for unit, div in (('GB', 2 ** 30), ('MB', 2 ** 20), ('kB', 2 ** 10)):
        if b >= div:
            return '%.1f %s' % (b / div, unit)
    return '%d B' % b


def choose_method(rec_bytes, names, nbytes, perm, chunk, ctx, args):
    """Streaming direction for the record blocks of one size: --scratch
    selects the bucket method, --method forces a direction, auto takes the
    cheaper one (calls plus bytes, see run_stats).  The statistics are
    logged once per record size and file."""
    key = (rec_bytes, chunk)
    if key not in ctx['stats']:
        ctx['stats'][key] = run_stats(perm, rec_bytes, chunk)
    st = ctx['stats'][key]
    if args.scratch:
        method = 'bucket'
    elif args.method != 'auto':
        method = args.method
    else:
        method = st['best']
    log('        %s records (%s, %s): per block gather %d read(s) of %s, scatter %d '
        'write(s) of %s -> %s'
        % (fmt_bytes(rec_bytes), ', '.join(names), fmt_bytes(nbytes),
           st['gather'][0], fmt_bytes(st['gather'][1] / max(1, st['gather'][0])),
           st['scatter'][0], fmt_bytes(st['scatter'][1] / max(1, st['scatter'][0])), method))
    if (method != 'bucket' and nbytes >= 2 ** 26 and st[method][0] > 1
            and st[method][1] / st[method][0] < 2 ** 20 and not ctx['hinted']):
        log('        hint: the runs are short; on a parallel file system the two-pass '
            '--scratch DIR method (sequential I/O only, scratch space of the file size) '
            'is likely faster')
        ctx['hinted'] = True
    return method


def block_progress(block):
    """A progress callback that logs every 10% of a block of 1 GB or more."""
    if block.nbytes < 2 ** 30:
        return None
    state = {'next': 0.1, 'start': time.time()}

    def prog(done, total):
        frac = done / total
        if frac >= state['next'] or done == total:
            el = time.time() - state['start']
            log('        %-10s %3d%%  %s  %s/s' % (block.name, int(100 * frac),
                                                 fmt_bytes(frac * block.nbytes),
                                                 fmt_bytes(frac * block.nbytes / max(el, 1e-9))))
            while state['next'] <= frac:
                state['next'] += 0.1
    return prog


def permute_file(path, out, perm, ctx, args):
    name = os.path.basename(path)
    low = name.lower()
    if low.endswith('.nek5000'):
        txt = open(path).read()
        m = re.search(r'^(\s*filetemplate:\s*)(\S+)', txt, flags=re.M)
        if m and args.prefix:
            tmpl = os.path.join(os.path.dirname(m.group(2)),
                                args.prefix + os.path.basename(m.group(2)))
            txt = txt[:m.start(2)] + tmpl + txt[m.end(2):]
        else:
            tmpl = m.group(2) if m else '?'
        with open(out, 'w') as f:
            f.write(txt)
        log('  %-40s series index -> %s (template %s)' % (name, os.path.basename(out), tmpl))
        return
    if low.endswith('.chkp'):
        lay = chkp_layout(path, args.rp)
        if lay.nelgv != perm.size:
            sys.exit('Error: %s holds %d elements, the meshes %d' % (path, lay.nelgv, perm.size))
        log('  %-40s checkpoint: %d elements, gdim %d, lx %d, %d-byte reals (%s), '
            'time %g' % (name, lay.nelgv, lay.gdim, lay.lx, lay.rp, lay.rp_note, lay.time))
        log('  %-40s blocks: %s' % ('', ' '.join(b.name for b in lay.blocks if b.rec_bytes)))
        if not ctx['chkp_warned']:
            log('        note: a checkpoint carries no element identity; the source '
                'mesh is taken on trust%s' % (' (the coordinate check of the field '
                                             'file(s) above covers this run)'
                                             if ctx['fld_checked'] else ''))
            ctx['chkp_warned'] = True
        idx_block = None
    else:
        lay = fld_layout(path)
        if lay.nelgv != perm.size:
            sys.exit('Error: %s holds %d elements, the meshes %d' % (path, lay.nelgv, perm.size))
        log('  %-40s field file: %d elements, lx %d ly %d lz %d, %d-byte reals, '
            'time %g, %s' % (name, lay.nelgv, lay.lx, lay.ly, lay.lz, lay.wdsize,
                              lay.time, lay.rdcode))
        idx_block = lay.blocks[1]
        idx = read_fld_idx(path, lay)
        if not np.array_equal(idx, ctx['src_ids']):
            hint = ('the file was written on another mesh ordering, or by a masked '
                    '/ subset writer (whose idx is 1..N and carries no element '
                    'identity: such a file cannot be permuted)')
            sys.exit('Error: %s: its idx column does not hold the element ids of '
                     '%s -- %s' % (path, args.src_mesh, hint))
        if ctx['ids_trivial']:
            log('        idx column holds the source mesh element ids (trivially: both '
                'meshes number their elements 1..N, so this check cannot catch '
                'swapped mesh arguments)')
        else:
            log('        idx column holds the element ids of %s'
                % os.path.basename(args.src_mesh))
        if 'X' in lay.rdcode:
            r = check_fld_coordinates(path, lay, args.src_mesh,
                                      max(1, int(args.chunk_mb * 2 ** 20)
                                          // max(1, lay.blocks[2].rec_bytes)))
            if r.abs_nbad == 0:
                log('        coordinates agree with %s element by element (%.1e)'
                    % (os.path.basename(args.src_mesh), r.abs_worst))
            elif r.con_ok:
                log('        note: the coordinates differ from %s (%d element(s), largest '
                    'corner mismatch %.3e): the mesh was deformed at run time (e.g. '
                    'user_mesh_setup); the %d corners the mesh shares between elements '
                    'agree in the file (%.1e), so the file follows its element ordering'
                    % (os.path.basename(args.src_mesh), r.abs_nbad, r.abs_worst,
                       r.con_nshared, r.con_worst))
            elif r.con_nshared == 0:
                log('        WARNING: the coordinates differ from %s (largest corner '
                    'mismatch %.3e) and the mesh shares no corner between elements, so '
                    'the element ordering cannot be cross-checked'
                    % (os.path.basename(args.src_mesh), r.abs_worst))
            else:
                sys.exit('Error: the coordinates stored in %s do not match %s element '
                         'by element (%d element(s), largest corner mismatch %.3e), and '
                         'corners the mesh shares between elements carry different '
                         'coordinates in the file (up to %.3e over %d shared corners): '
                         'the file was not written on that mesh ordering -- are the '
                         'mesh arguments reversed?'
                         % (path, args.src_mesh, r.abs_nbad, r.abs_worst, r.con_worst,
                            r.con_nshared))
            ctx['fld_checked'] = True
        elif ctx['fld_checked']:
            log('        no coordinates in this file; the coordinate check of the field '
                'file(s) above covers this run')
        else:
            m = re.match(r'^(.*\.f)\d{5}$', name)
            log('        note: this file carries no coordinates, so the source mesh '
                'ordering is taken on trust; give the coordinate-carrying file of the '
                'series (usually the first, %s) in the same call to verify it'
                % ('e.g. ' + m.group(1) + '00000' if m else 'with X in its rdcode'))
    if lay.trailing:
        log('        note: %d stale byte(s) after the layout are dropped (Neko does '
            'not truncate output files it rewrites)' % lay.trailing)
    t0 = time.time()
    chunk = int(args.chunk_mb * 2 ** 20)
    groups = {}
    for b in lay.blocks:
        if b.rec_bytes and b is not idx_block:
            groups.setdefault(b.rec_bytes, [[], 0])
            groups[b.rec_bytes][0].append(b.name)
            groups[b.rec_bytes][1] += b.nbytes
    methods = {rec: choose_method(rec, names, nb, perm, chunk, ctx, args)
               for rec, (names, nb) in groups.items()}
    if 'scatter' in methods.values() and ctx['inv'] is None:
        ctx['inv'] = inverse_permutation(perm)
    tmp = out + '.tmp'
    try:
        with open(path, 'rb') as fsrc, open(tmp, 'wb') as f:
            for b in lay.blocks:
                if b.rec_bytes == 0:
                    fsrc.seek(b.offset)
                    f.write(fsrc.read(b.nbytes))
                elif b is idx_block:
                    ctx['dst_ids'].astype('<i4').tofile(f)
                else:
                    method = methods[b.rec_bytes]
                    prog = block_progress(b)
                    if method == 'bucket':
                        bucket_records(path, f, b.offset, b.rec_bytes, perm,
                                       args.scratch, chunk, progress=prog)
                    elif method == 'scatter':
                        scatter_records(path, f, b.offset, b.offset, b.rec_bytes, perm,
                                        chunk, args.threads, progress=prog, inv=ctx['inv'])
                    else:
                        gather_records(path, f, b.offset, b.rec_bytes, perm, chunk,
                                       args.threads, progress=prog)
                if f.tell() != b.offset + b.nbytes:
                    raise RuntimeError('block %s: %d bytes written instead of %d'
                                       % (b.name, f.tell() - b.offset, b.nbytes))
            f.flush()
            os.fsync(f.fileno())
        if os.path.getsize(tmp) != lay.size:
            raise RuntimeError('output size mismatch')
        os.replace(tmp, out)
    except BaseException as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if isinstance(e, SystemExit):
            raise
        sys.exit('Error: writing %s failed (%s)' % (out, e))
    log('  %-40s -> %s  (%.1f s, %.1f MB)' % ('', out, time.time() - t0,
                                             lay.size / 2 ** 20))


def main():
    args = parse_args()
    log(banner('permute_fields'))
    log('  source mesh : %s' % args.src_mesh)
    log('  target mesh : %s' % args.dst_mesh)
    if args.chunk_mb <= 0:
        sys.exit('Error: --chunk-mb must be positive')
    if not os.path.isdir(args.outdir):
        sys.exit('Error: output directory %s does not exist' % args.outdir)
    outs = []
    for p in args.files:
        if not os.path.isfile(p):
            sys.exit('Error: %s not found' % p)
        out = os.path.join(args.outdir, args.prefix + os.path.basename(p))
        if os.path.realpath(out) == os.path.realpath(p) or (
                os.path.exists(out) and os.path.samefile(out, p)):
            sys.exit('Error: %s would overwrite the input (choose another -o)' % p)
        if out in outs:
            sys.exit('Error: two input files would both be written as %s' % out)
        if os.path.exists(out) and not args.force:
            sys.exit('Error: %s exists (use --force to overwrite)' % out)
        outs.append(out)
    if args.scratch and not os.path.isdir(args.scratch):
        sys.exit('Error: scratch directory %s does not exist' % args.scratch)
    if args.threads < 1:
        sys.exit('Error: --threads must be at least 1')
    log('  [1/3] matching the elements of the two meshes ...')
    t0 = time.time()
    keys = (MeshKeys(args.src_mesh), MeshKeys(args.dst_mesh))
    log('        meshes read (%.1f s)' % (time.time() - t0))
    perm = element_permutation(args.src_mesh, args.dst_mesh, args.match, args.tol, log, keys)
    worst, nbad, scale = verify_corners(args.src_mesh, args.dst_mesh, perm, keys=keys)
    if nbad and args.match == 'ids':
        log('        note: %d element(s) matched by point ids have different '
            'corners; matching by centroids instead' % nbad)
        perm = element_permutation(args.src_mesh, args.dst_mesh, 'centroids', args.tol, log,
                                   keys)
        worst, nbad, scale = verify_corners(args.src_mesh, args.dst_mesh, perm, keys=keys)
    if nbad:
        sys.exit('Error: the two meshes do not hold the same elements: %d matched '
                 'element(s) have different corners (largest mismatch %.3e)' % (nbad, worst))
    moved = int((perm != np.arange(perm.size)).sum())
    log('        %d elements, %d change position, corners agree to %.1e  (%.1f s)'
        % (perm.size, moved, worst, time.time() - t0))
    src_ids, dst_ids = keys[0].ids, keys[1].ids
    del keys
    ctx = dict(src_ids=src_ids, dst_ids=dst_ids, fld_checked=False, chkp_warned=False,
               stats={}, hinted=False, inv=None,
                              ids_trivial=bool(np.array_equal(src_ids, np.arange(1, perm.size + 1))
                                and np.array_equal(dst_ids, np.arange(1, perm.size + 1))))
    log('  [2/3] permuting %d file(s) -> %s' % (len(args.files), args.outdir))
    # coordinate-carrying field files first: they validate the source mesh
    def has_coords(p):
        low = p.lower()
        if low.endswith('.chkp') or low.endswith('.nek5000'):
            return False
        with open(p, 'rb') as f:
            return b'X' in f.read(132)[83:93]

    order = sorted(range(len(args.files)),
                   key=lambda i: (args.files[i].lower().endswith('.chkp'),
                                  args.files[i].lower().endswith('.nek5000'),
                                  not has_coords(args.files[i])))
    for i in order:
        permute_file(args.files[i], outs[i], perm, ctx, args)
    for p, out in zip(args.files, outs):
        if p.lower().endswith('.nek5000'):
            missing = nek5000_members(out, args.outdir)
            if missing:
                log('        warning: %s lists %d field file(s) not present in %s '
                    '(e.g. %s); permute them too or ParaView/VisIt will not open '
                    'the series' % (os.path.basename(out), len(missing), args.outdir,
                                     missing[0]))
    log('  [3/3] done')
    log('  Point the target run at the permuted files as usual, e.g. "restart_file": '
        '"%sfluid00001.chkp" or "initial_condition": {"type": "field", "file_name": '
        '"%sfield0.fld", "sample_index": N}; no restart_mesh_file (that would '
        'interpolate instead of copying).' % (args.prefix, args.prefix))


if __name__ == '__main__':
    main()
