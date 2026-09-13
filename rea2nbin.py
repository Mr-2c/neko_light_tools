#!/usr/bin/env python3
# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see nekolight/formats.py for the full
# licence text.
#
#     _  __  ____  __ __  ____
#    / |/ / / __/ / //_/ / __ \
#   /    / / _/  / ,<   / /_/ /
#  /_/|_/ /___/ /_/|_|  \____/
#
"""rea2nbin -- convert a NEKTON .re2 mesh (2D or 3D) to a Neko .nmsh.

Byte-exact with Neko's contrib/rea2nbin: the same first-appearance point
de-duplication (bit-exact float64 comparison, as Neko's own point table
does), the same boundary-condition classification (MSH/EXO user labels
first, then named BCs W/v/O/SYM/ON/s* with first-appearance internal labels,
'P' pairs to periodic zones via the re2->Neko facet map), the same fixed
3-sweep in-place-min periodic id merge as Neko's re2 reader, and the same
per-element curve-record aggregation ('C' circle -> 3, 'm' midside -> 4; any
other curve type drops ALL curves, as Neko does).  A 2D re2 becomes a 2D
(quad) .nmsh, which Neko extrudes into a one-element slab when it reads it.

Point de-duplication compares coordinates bit-exactly.  Neko's point table
hashes the raw bits too, so this is what real files experience; its
tolerant equality test only matters for near-coincident points that are
not bit-identical, which no sane mesh generator produces.

Everything is read and validated BEFORE the output is opened, and the
output is written atomically (temp file + rename) -- a failed run leaves no
partial .nmsh and can never damage the input.

Usage: rea2nbin.py mesh.re2 [mesh.nmsh]
The periodic match tolerance honours NEKO_PERIODIC_TOL (default 1e-7).
"""

import os
import sys

import numpy as np

from nekolight import (banner, elem_dtype, ZONE_DT, CURVE_DT, facet_table,
                       FACET_MAP, MAX_ZLBLS, read_re2, dedup_points,
                       create_periodic_ids, bc_type_str, write_nmsh)

# named BC -> slot in the first-appearance internal-label table
NAMED_SLOT = {'W': 1, 'v': 2, 'V': 2, 'O': 3, 'o': 3, 'SYM': 4, 'sym': 4,
              'ON': 5, 'on': 5, 's': 6, 'sl': 6, 'sh': 6, 'shl': 6,
              'S': 6, 'SL': 6, 'SH': 6, 'SHL': 6}
MSH_TYPES = ('MSH', 'msh', 'EXO', 'exo')


def log(msg):
    print(msg, flush=True)


def periodic_tol():
    s = os.environ.get('NEKO_PERIODIC_TOL', '')
    if not s:
        return 1e-7
    try:
        tol = float(s.strip().lower().replace('d', 'e'))   # Fortran 1d-6
    except ValueError:
        sys.exit('Error: invalid NEKO_PERIODIC_TOL value: %s' % s)
    if tol <= 0.0:
        sys.exit('Error: invalid NEKO_PERIODIC_TOL value: %s' % s)
    return tol


def classify_bcs(nelv, bcs, gdim):
    """The two-pass BC classification of Neko's re2 reader: (labels, periodic
    facet pairs).  Labels are (e, sym_facet, label) in emission order; pairs
    are (e, sym_facet, partner_e, partner_sym_facet) in file order.  Neko
    prints 'bc type not supported yet' for every other type and moves on;
    they are counted and reported here."""
    nf = 2 * gdim
    types = [bc_type_str(t) for t in bcs['t']]
    nnul = sum(1 for t in bcs['t'] if b'\x00' in bytes(t))
    if nnul:
        log('        note: %d boundary record(s) carry NUL padding in the '
            'type field; treated as blanks here, whereas Neko\'s trim() '
            'would leave them and skip the record' % nnul)
    e = bcs['e'].astype(np.int64)
    fc = bcs['f'].astype(np.int64)
    d1 = bcs['d'][:, 0].astype(np.float64)
    d2 = bcs['d'][:, 1].astype(np.float64)
    d5 = bcs['d'][:, 4].astype(np.float64)

    z_e, z_f, z_lbl = [], [], []
    p = []
    skipped = {}

    def add_zone(el, facet, lbl):
        if lbl < 1 or lbl > MAX_ZLBLS:
            sys.exit('Error: boundary label out of [1,%d]: %d'
                     % (MAX_ZLBLS, lbl))
        z_e.append(el)
        z_f.append(facet)
        z_lbl.append(lbl)

    # pass 1: MSH/EXO user labels (and the max user label as the offset)
    user_off = 0
    for i, t in enumerate(types):
        if t in MSH_TYPES:
            user_off = max(user_off, int(d5[i]))
    for i, t in enumerate(types):
        if t in MSH_TYPES:
            add_zone(int(e[i]), int(FACET_MAP[fc[i] - 1]), int(d5[i]))

    # pass 2: named BCs -> first-appearance internal labels; 'P' -> pairs
    named_map = {}
    cur_internal = [1]

    def named_label(slot):
        if slot not in named_map:
            named_map[slot] = cur_internal[0]
            cur_internal[0] += 1
        return user_off + named_map[slot]

    for i, t in enumerate(types):
        if t in MSH_TYPES:
            continue
        if t in NAMED_SLOT:
            add_zone(int(e[i]), int(FACET_MAP[fc[i] - 1]),
                     named_label(NAMED_SLOT[t]))
        elif t == 'P':
            pe, pf = int(d1[i]), int(d2[i])
            if pe < 1 or pe > nelv or pf < 1 or pf > nf:
                sys.exit('Error: periodic BC references out-of-range partner '
                         'element/face')
            p.append((int(e[i]), int(FACET_MAP[fc[i] - 1]), pe,
                      int(FACET_MAP[pf - 1])))
        else:
            skipped[t] = skipped.get(t, 0) + 1     # as Neko: not supported
    return (np.array(z_e, dtype=np.int64), np.array(z_f, dtype=np.int64),
            np.array(z_lbl, dtype=np.int64), p, skipped)


def aggregate_curves(nelv, curves, gdim):
    """re2 curve records -> nmsh curve records: one record per curved
    element, ascending element order, edge slots filled per record ('C' -> 3,
    'm' -> 4).  A single unsupported type ('s'/'e'/...) makes Neko treat the
    whole mesh as non-curved; we match that (and say so).  A 2D mesh with a
    curve on edges 5..8 is refused, as Neko's mark_curve_element does."""
    if curves.size == 0:
        return np.empty(0, dtype=CURVE_DT), False
    first = np.array([t[:1] for t in curves['t']])       # raw first byte
    ctype = np.zeros(curves.size, dtype=np.int32)
    ctype[first == b'C'] = 3
    ctype[first == b'm'] = 4
    if (ctype == 0).any():
        return np.empty(0, dtype=CURVE_DT), True
    el = curves['e'].astype(np.int64)
    edge = curves['edge'].astype(np.int64)
    if gdim == 2 and ((edge >= 5) & (edge <= 8)).any():
        sys.exit('Error: 2D mesh with a curve on edge 5..8 (Neko: "Invalid '
                 'curve element")')
    uniq = np.unique(el)                                 # ascending
    out = np.zeros(uniq.size, dtype=CURVE_DT)
    out['e'] = uniq.astype(np.int32)
    row = np.searchsorted(uniq, el)
    out['data'][row, edge - 1, :] = curves['d'].astype(np.float64)
    out['type'][row, edge - 1] = ctype
    return out, False


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit('Usage: rea2nbin.py <mesh.re2> [<mesh.nmsh>]')
    fin = sys.argv[1]
    fout = sys.argv[2] if len(sys.argv) == 3 else \
        (fin[:-3] + 'nmsh' if fin.endswith('.re2') else fin + '.nmsh')
    tol = periodic_tol()

    log(banner('rea2nbin  (re2 -> nmsh)'))
    log('  input     : %s' % fin)
    log('  output    : %s' % fout)

    re2 = read_re2(fin)
    nelv, gdim = re2.nelv, re2.gdim
    nv = 8 if gdim == 3 else 4
    log('  mesh      : %d %s elements  (format %s, %dD)'
        % (nelv, 'hex' if gdim == 3 else 'quad', re2.version, gdim))

    log('  [1/3] de-duplicating points ...')
    vid, nuniq = dedup_points(re2.xyz)
    log('        %d unique points' % nuniq)
    # id -> coordinate (ids are first-appearance, so any occurrence writes
    # the identical bits)
    coords = np.empty((nuniq, 3), dtype=np.float64)
    coords[vid.reshape(-1).astype(np.int64) - 1] = re2.xyz.reshape(-1, 3)

    log('  [2/3] classifying boundary conditions and merging periodic '
        'points ...')
    z_e, z_f, z_lbl, pairs, skipped = classify_bcs(nelv, re2.bcs, gdim)
    for t, cnt in sorted(skipped.items()):
        log('        note: %d boundary record(s) of type %r not supported by '
            'Neko -- skipped, as Neko does' % (cnt, t))
    pid = np.arange(1, nuniq + 1, dtype=np.int64)
    if pairs:
        # Neko's re2 reader: 3 sweeps over the 'P' records in file order;
        # its quad branch does not check the match count
        unmatched = create_periodic_ids(pid, vid, coords, pairs, tol,
                                        strict=(gdim == 3))
        if unmatched:
            log('        warning: %d periodic edge corner(s) without exactly '
                'one match (Neko\'s 2D reader does not check this)'
                % unmatched)
    curves_out, curve_skip = aggregate_curves(nelv, re2.curves, gdim)
    if curve_skip:
        log('        note: unsupported curve type (s/e/other); mesh treated '
            'as non-curved (ncurves=0), as Neko also does')

    log('  [3/3] writing %s ...' % fout)
    # element records (input order, ids 1..nelv, vertex order = re2 order)
    elems = np.empty(nelv, dtype=elem_dtype(gdim))
    elems['id'] = np.arange(1, nelv + 1, dtype=np.int32)
    elems['v']['idx'] = vid
    elems['v']['xyz'] = re2.xyz

    # periodic zones (file order of the 'P' records); quads store 2 ids
    ft = facet_table(nv)
    zp = np.zeros(len(pairs), dtype=ZONE_DT)
    for i, (el, f, pe, pf) in enumerate(pairs):
        zp['e'][i], zp['f'][i] = el, f
        zp['p_e'][i], zp['p_f'][i] = pe, pf
        zp['g'][i, :ft.shape[1]] = pid[vid[el - 1, ft[f - 1]].astype(np.int64)
                                       - 1]
    zp['t'] = 5

    # labelled zones, grouped by ascending label (stable within a label)
    zl = np.zeros(z_e.size, dtype=ZONE_DT)
    order = np.argsort(z_lbl, kind='stable') if z_e.size else []
    zl['e'] = z_e[order]
    zl['f'] = z_f[order]
    zl['p_f'] = z_lbl[order]
    zl['t'] = 7

    write_nmsh(fout, elems, (zp, zl), curves_out, inputs=(fin,))
    log('        %d periodic + %d labelled boundary facets, %d curved '
        'elements' % (len(pairs), z_e.size, curves_out.shape[0]))
    log('  done -> %s' % fout)


if __name__ == '__main__':
    main()
