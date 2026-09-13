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
"""create_periodic_zones -- turn pairs of labelled zones of a .nmsh into
periodic zones (the contrib/create_periodic_zones tool).

Each pair (a, b) must be two labelled zones with the same number of facets
that map onto each other by ONE translation.  The translation is the mean
of the facet-centre differences; every facet of zone a must then match
exactly one facet of zone b (centre within the tolerance and all four
corners matching one-to-one under the translation).  The matched facets
become periodic zone records (both directions), the matched corner points
receive common ids through Neko's fixed 3-sweep min merge, and the labelled
records of the converted zones are dropped.  Zones not mentioned and
existing periodic zones are kept.

The default matching tolerance is Neko's: max(1e-10, 1e-8 * max(1, bounding
box diagonal)).  The point-id merge uses NEKO_PERIODIC_TOL (default 1e-7),
exactly as Neko's create_periodic_ids does -- two different tolerances, as
in the Fortran.

Usage: create_periodic_zones.py in.nmsh out.nmsh "(1,2),(3,4)" [--tol X]
The pair list accepts any delimiter mix: "(1,2),(3,4)", "1:2 3:4", "1 2 3 4".
"""

import argparse
import os
import re
import sys

import numpy as np

from nekolight import (banner, read_nmsh, validate_zones, validate_curves,
                       pos_of_elid_map, periodic_replace_merge,
                       create_periodic_ids, write_nmsh, FACE_RE2, ZONE_DT,
                       MAX_ZLBLS)


def log(msg):
    print(msg, flush=True)


def parse_pairs(spec):
    """Neko's parser: every non-digit is a separator; an even number of
    integers is required."""
    vals = [int(v) for v in re.findall(r'\d+', spec)]
    if not vals or len(vals) % 2:
        sys.exit('Error: could not parse periodic zone pairs from %r' % spec)
    pairs = list(zip(vals[0::2], vals[1::2]))
    for a, b in pairs:
        if not (1 <= a <= MAX_ZLBLS and 1 <= b <= MAX_ZLBLS):
            sys.exit('Error: zone indices must be in [1,%d]' % MAX_ZLBLS)
        if a == b:
            sys.exit('Error: periodic zone pair labels must be distinct')
    return pairs


def periodic_tol():
    s = os.environ.get('NEKO_PERIODIC_TOL', '')
    if not s:
        return 1e-7
    try:
        tol = float(s)
    except ValueError:
        sys.exit('Error: invalid NEKO_PERIODIC_TOL value: %s' % s)
    if tol <= 0.0:
        sys.exit('Error: invalid NEKO_PERIODIC_TOL value: %s' % s)
    return tol


def match_facets(ca, xa, cb, xb, offset, tol):
    """For every facet of zone a (centres ca (na,3), corners xa (na,4,3)) the
    index of the unique facet of zone b within tol under the translation,
    with Neko's corner test (each a-corner matches exactly one so far unused
    b-corner).  Errors exactly where Neko does."""
    from scipy.spatial import cKDTree
    tree = cKDTree(cb)
    cand = tree.query_ball_point(ca + offset, r=tol * (1.0 + 1e-12))
    match = np.full(ca.shape[0], -1, dtype=np.int64)
    used_b = np.full(cb.shape[0], -1, dtype=np.int64)
    for i, cl in enumerate(cand):
        hits = []
        for j in sorted(cl):
            if np.linalg.norm(cb[j] - ca[i] - offset) > tol:
                continue
            used = np.zeros(4, dtype=bool)
            ok = True
            for k in range(4):
                d = np.linalg.norm(xb[j] - xa[i, k] - offset, axis=1)
                m = np.flatnonzero((d <= tol) & ~used)
                if m.size != 1:
                    ok = False
                    break
                used[m[0]] = True
            if ok:
                hits.append(j)
        if len(hits) != 1:
            sys.exit('Error: could not determine a unique periodic facet '
                     'match for zone-a facet %d (centre %s): %d candidate(s)'
                     % (i + 1, ca[i], len(hits)))
        j = hits[0]
        if used_b[j] >= 0:
            sys.exit('Error: periodic zone mapping is not one-to-one: zone-b '
                     'facet %d matches zone-a facets %d and %d'
                     % (j + 1, used_b[j] + 1, i + 1))
        used_b[j] = i
        match[i] = j
    return match


def main():
    ap = argparse.ArgumentParser(
        prog='create_periodic_zones.py',
        description='Convert pairs of labelled zones into periodic zones.')
    ap.add_argument('input', help='input .nmsh')
    ap.add_argument('output', help='output .nmsh')
    ap.add_argument('pairs', nargs='+',
                    help='zone pairs, e.g. "(1,2),(3,4)" or 1:2 3:4')
    ap.add_argument('--tol', type=float, default=None,
                    help='facet matching tolerance (default: Neko\'s '
                         'max(1e-10, 1e-8 * max(1, bbox diagonal)))')
    args = ap.parse_args()
    if args.tol is not None and args.tol <= 0.0:
        sys.exit('Error: --tol must be positive')
    pairs = parse_pairs(' '.join(args.pairs))

    log(banner('create_periodic_zones'))
    log('  input     : %s' % args.input)
    log('  output    : %s' % args.output)
    mesh = read_nmsh(args.input)
    if mesh.gdim != 2 + 1:
        sys.exit('Error: create_periodic_zones supports 3D (hex) meshes only')
    validate_zones(mesh.nelv, mesh.zones, args.input, mesh.gdim)
    validate_curves(mesh.nelv, mesh.curves, args.input, mesh.gdim, log)
    pos_of_elid = pos_of_elid_map(mesh.nelv, mesh.elems)
    nelv = mesh.nelv
    elids = mesh.elems['id'].astype(np.int64)
    vidx = mesh.elems['v']['idx'].astype(np.int64)
    xyz = mesh.elems['v']['xyz']

    # point objects: one per raw id; current ids after Neko's read (the
    # existing periodic records applied) -- what get_facet_ids returns
    npt = int(vidx.max())
    coords = np.zeros((npt, 3))
    coords[vidx.ravel() - 1] = xyz.reshape(-1, 3)
    cur = np.arange(1, npt + 1, dtype=np.int64)
    merged = periodic_replace_merge(nelv, vidx, mesh.zones, pos_of_elid)
    cur[vidx.ravel() - 1] = merged.ravel()

    zones = mesh.zones
    z5 = zones[zones['t'] == 5]
    z7 = zones[zones['t'] == 7]
    zx = zones[(zones['t'] != 5) & (zones['t'] != 7)]
    if zx.size:
        log('  note: %d zone records of legacy types are dropped (Neko\'s '
            'reader ignores them, so its create_periodic_zones drops them '
            'too)' % zx.shape[0])
    converted = set()
    for a, b in pairs:
        converted.update((a, b))

    lo, hi = xyz.reshape(-1, 3).min(axis=0), xyz.reshape(-1, 3).max(axis=0)
    tol = args.tol if args.tol is not None else \
        max(1e-10, 1e-8 * max(1.0, float(np.linalg.norm(hi - lo))))
    idtol = periodic_tol()

    def facet_geometry(zrec):
        pos = pos_of_elid[zrec['e'].astype(np.int64)]
        slots = FACE_RE2[zrec['f'].astype(np.int64) - 1]        # (m, 4)
        corners = xyz[pos[:, None], slots]                       # (m, 4, 3)
        return pos, corners, corners.mean(axis=1)

    new_pairs = []             # (el, f, pe, pf) 1-based, both directions
    for a, b in pairs:
        za, zb = z7[z7['p_f'] == a], z7[z7['p_f'] == b]
        if za.shape[0] != zb.shape[0]:
            sys.exit('Error: periodic zone pair (%d, %d) has a different '
                     'number of facets (%d vs %d)'
                     % (a, b, za.shape[0], zb.shape[0]))
        if za.shape[0] == 0:
            sys.exit('Error: periodic zone pair (%d, %d) refers to an empty '
                     'labelled zone' % (a, b))
        pa, xa, ca = facet_geometry(za)
        pb, xb, cb = facet_geometry(zb)
        offset = (cb - ca).mean(axis=0)
        match = match_facets(ca, xa, cb, xb, offset, tol)
        for i in range(za.shape[0]):
            j = int(match[i])
            new_pairs.append((int(elids[pa[i]]), int(za['f'][i]),
                              int(elids[pb[j]]), int(zb['f'][j])))
            new_pairs.append((int(elids[pb[j]]), int(zb['f'][j]),
                              int(elids[pa[i]]), int(za['f'][i])))
        log('  periodic zones %d <-> %d, offset: %s (%d facet pairs)'
            % (a, b, ' '.join('%12.4e' % v for v in offset), za.shape[0]))
    log('  matching tolerance: %12.4e' % tol)

    # the ids the new records store as their originals: the CURRENT ids of
    # their corners (Neko: get_facet_ids after the read-time merge)
    org_new = cur.copy()
    # Neko's create_periodic_ids: 3 sweeps, (a->b, b->a) per matched pair
    pid = cur.copy()
    pairs_by_pos = [(int(pos_of_elid[el]) + 1, f, int(pos_of_elid[pe]) + 1, pf)
                    for (el, f, pe, pf) in new_pairs]
    create_periodic_ids(pid, vidx, coords, pairs_by_pos, idtol, strict=True)

    # ---- write, as Neko's nmsh writer does after reset_periodic_ids ----
    # element vertex ids: every point reset to its 'original' id -- the raw
    # file id for the old records, the read-time merged id for the new ones
    # (later records win, and the new ones come last)
    write_id = np.arange(1, npt + 1, dtype=np.int64)
    for (el, f, pe, pf) in pairs_by_pos:
        raw = vidx[el - 1, FACE_RE2[f - 1]]
        write_id[raw - 1] = org_new[raw - 1]
    elems = mesh.elems.copy()
    elems['v']['idx'] = write_id[vidx.ravel() - 1].reshape(vidx.shape) \
        .astype(np.int32)

    def final_ids(el_pos, f):
        return pid[vidx[el_pos, FACE_RE2[f - 1]] - 1]

    zp = np.zeros(z5.shape[0] + len(new_pairs), dtype=ZONE_DT)
    for i in range(z5.shape[0]):                     # old records, in order
        zp['e'][i], zp['f'][i] = z5['e'][i], z5['f'][i]
        zp['p_e'][i], zp['p_f'][i] = z5['p_e'][i], z5['p_f'][i]
        zp['g'][i] = final_ids(pos_of_elid[int(z5['e'][i])], int(z5['f'][i]))
    for k, (el, f, pe, pf) in enumerate(new_pairs):
        i = z5.shape[0] + k
        zp['e'][i], zp['f'][i], zp['p_e'][i], zp['p_f'][i] = el, f, pe, pf
        zp['g'][i] = final_ids(pos_of_elid[el], f)
    zp['t'] = 5
    keep = np.array([int(lbl) not in converted for lbl in z7['p_f']],
                    dtype=bool) if z7.size else np.zeros(0, dtype=bool)
    zl = z7[keep].copy()
    zl = zl[np.argsort(zl['p_f'], kind='stable')]    # grouped by label
    zl['p_e'] = 0
    zl['g'] = 0
    write_nmsh(args.output, elems, (zp, zl), mesh.curves,
               inputs=(args.input,))
    log('  wrote %d periodic + %d labelled facets -> %s'
        % (zp.shape[0], zl.shape[0], args.output))


if __name__ == '__main__':
    main()
