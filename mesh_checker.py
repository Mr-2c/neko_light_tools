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
"""mesh_checker -- validate and describe a Neko .nmsh.

Reads and validates the ENTIRE file -- elements, zones, curve records and the
exact end-of-file position -- and reports what Neko's own mesh_checker
reports: sizes after the periodic merge (glb_mpts/glb_mfcs/glb_meds), the
bounding box, periodic and labelled zones with the normal alignment of each
labelled zone, unlabelled external faces, plus the curve records and (with
--jacobian) the Jacobian on the 3x3x3 GLL grid of the geometry Neko builds
from the file, curved edges included.  A 2D (quad) file is checked as the
one-element-thick slab Neko extrudes it into.  Any malformed record
(out-of-range reference, bad label, truncated section) is a hard error: this
tool never blesses a file it could not fully parse.

Options:
  --jacobian            also check for negative/zero Jacobians (curved
                        geometry where curve records exist)
  --write-zone-indices  also write <mesh>_zone_indices.fld marking labelled
                        boundary faces by zone index (implies --jacobian)

Exit status 0 means: the file parsed completely and no check failed.
"""

import argparse
import os
import sys

import numpy as np

from nekolight import (banner, read_nmsh, extrude_2d, validate_zones,
                       validate_curves, pos_of_elid_map, merged_vertex_ids,
                       face_multiplicity, count_edges, gll_geometry,
                       jacobian_dets, facet_normals, facet_gll_mask,
                       write_zone_indices_fld, CurveError, MAX_ZLBLS)

AXIS_TOL = 1e-3          # Neko's axis_alignment_tol
CHUNK = 1 << 18


def log(msg):
    print(msg, flush=True)


def zone_alignment(xyz, curves, pos_of_elid, epos, f0):
    """Neko's 'Normal alignment' of a labelled zone: x/y/z when EVERY facet
    normal (at the facet centre, curved geometry) is aligned with that one
    axis to within AXIS_TOL, otherwise 'none'."""
    rows, inv = np.unique(epos, return_inverse=True)
    x27, _ = gll_geometry(xyz, curves, pos_of_elid, rows)
    n = facet_normals(x27)[inv, f0]                        # (m, 3)
    s = np.abs(np.abs(n) - 1.0)                            # Neko's sx,sy,sz
    aligned = s < AXIS_TOL
    one = aligned.sum(axis=1) == 1
    counts = (aligned & one[:, None]).sum(axis=0)          # per axis
    for a, name in enumerate('xyz'):
        if counts[a] == epos.size:
            return name
    return 'none'


def main():
    ap = argparse.ArgumentParser(
        prog='mesh_checker.py',
        description='Validate a Neko .nmsh and report its sizes, zones and '
                    'quality.')
    ap.add_argument('mesh', help='input .nmsh')
    ap.add_argument('--jacobian', action='store_true',
                    help='also check for negative/zero Jacobians')
    ap.add_argument('--write-zone-indices', action='store_true',
                    help='also write <mesh>_zone_indices.fld (implies '
                         '--jacobian)')
    args = ap.parse_args()
    do_jac = args.jacobian or args.write_zone_indices

    log(banner('mesh_checker'))
    failed = False

    log('  [1/3] reading and validating the file ...')
    mesh = read_nmsh(args.mesh)                  # errors on any truncation
    validate_zones(mesh.nelv, mesh.zones, args.mesh, mesh.gdim)
    validate_curves(mesh.nelv, mesh.curves, args.mesh, mesh.gdim, log)
    pos_of_elid = pos_of_elid_map(mesh.nelv, mesh.elems)
    if mesh.trailing:
        log('        note: %d trailing bytes past the curve section '
            '(MPI-IO no-truncate artifact; Neko ignores them)'
            % mesh.trailing)
    is2d = mesh.gdim == 2
    if is2d:
        log('        note: 2D (quad) mesh -- checked as the one-element '
            'slab Neko extrudes it into (z in [0, 1]); sizes below are '
            'the slab\'s, as Neko\'s mesh_checker reports them')
        hexmesh = extrude_2d(mesh)
    else:
        hexmesh = mesh

    xyz = hexmesh.elems['v']['xyz']
    lo, hi = xyz.reshape(-1, 3).min(axis=0), xyz.reshape(-1, 3).max(axis=0)

    # ---- zones: facet marking + Neko's replace-merge ----
    log('  [2/3] applying the periodic merge and counting faces/edges ...')
    zones = hexmesh.zones
    z5 = zones[zones['t'] == 5]
    z7 = zones[zones['t'] == 7]
    zleg = zones[(zones['t'] >= 1) & (zones['t'] <= 4)]
    ftype = np.zeros((mesh.nelv, 6), dtype=np.int8)      # 0 none, 1 lbl,
    flabel = np.zeros((mesh.nelv, 6), dtype=np.int8)     # 2 per, 3 legacy
    if zleg.size:
        # legacy zone types (1..4, pre-labelled-zone Neko): the current Neko
        # reader ignores these records, but they document the boundary --
        # count their facets as accounted for, not as unlabelled
        ftype[pos_of_elid[zleg['e'].astype(np.int64)],
              zleg['f'].astype(np.int64) - 1] = 3
    if z5.size:
        ftype[pos_of_elid[z5['e'].astype(np.int64)],
              z5['f'].astype(np.int64) - 1] = 2
    labeled_cnt = np.zeros(MAX_ZLBLS + 1, dtype=np.int64)
    if z7.size:
        lbl = z7['p_f'].astype(np.int64)                 # label lives in p_f
        labeled_cnt = np.bincount(lbl, minlength=MAX_ZLBLS + 1)
        p = pos_of_elid[z7['e'].astype(np.int64)]
        f0 = z7['f'].astype(np.int64) - 1
        ftype[p, f0] = 1
        flabel[p, f0] = lbl.astype(np.int8)

    merged = merged_vertex_ids(hexmesh, pos_of_elid, extruded=is2d)
    mult, n_faces = face_multiplicity(merged)
    n_edges = count_edges(merged)
    n_points = int(merged.max())                        # Neko's max_pts_id
    n_unlabeled = int(((mult == 1) & (ftype == 0)).sum())

    # ---- report (mirrors Neko's mesh_checker) ----
    log('')
    log(' --------------Size-------------')
    log(' Number of elements: %d' % mesh.nelv)
    log(' Number of points:   %d' % n_points)
    log(' Number of faces:    %d' % n_faces)
    log(' Number of edges:    %d' % n_edges)
    log(' Bounding box:')
    log('    x %14.6g %14.6g' % (lo[0], hi[0]))
    log('    y %14.6g %14.6g' % (lo[1], hi[1]))
    log('    z %14.6g %14.6g' % (lo[2], hi[2]))
    log('')
    log(' --------------Zones------------')
    if is2d:
        log(' Number of periodic faces: %d  (+ %d z-facets of the extruded '
            'slab, which Neko also counts)' % (z5.shape[0], 2 * mesh.nelv))
    else:
        log(' Number of periodic faces: %d' % z5.shape[0])
    if zleg.size:
        log(' Legacy zone records (types 1-4): %d (ignored by Neko\'s '
            'current reader; their facets are treated as documented '
            'boundaries here)' % zleg.shape[0])
    log('')
    log(' Labelled zones:')
    curves = hexmesh.curves
    for i in range(1, MAX_ZLBLS + 1):
        if labeled_cnt[i] > 0:
            sel = z7['p_f'] == i
            epos = pos_of_elid[z7['e'][sel].astype(np.int64)]
            f0 = z7['f'][sel].astype(np.int64) - 1
            try:
                align = zone_alignment(xyz, curves, pos_of_elid, epos, f0)
            except CurveError as ex:
                align = 'n/a (%s)' % ex
            log('    Zone %2d: %d faces. Normal alignment: %s'
                % (i, labeled_cnt[i], align))

    log('')
    log(' -------------Curves------------')
    if curves.size:
        ct = mesh.curves['type']
        log(' Curved elements: %d  (edges: %d circular arcs, %d midside '
            'points%s)' % (curves.shape[0], int((ct == 3).sum()),
                           int((ct == 4).sum()),
                           (', %d unsupported' % int(((ct == 1) | (ct == 2))
                                                     .sum()))
                           if ((ct == 1) | (ct == 2)).any() else ''))
        dup = curves.shape[0] - np.unique(curves['e']).size
        if dup:
            log(' Warning: %d element(s) have more than one curve record; '
                'Neko applies all of them in file order' % dup)
    else:
        log(' No curved elements.')

    if do_jac:
        log('')
        log(' ------------Jacobian----------')
        n_bad, first_bad, jac_min = 0, 0, np.inf
        jac_min_lin = np.inf
        ndef = 0
        curve_err = None
        elids = np.asarray(hexmesh.elems['id'])
        for s in range(0, mesh.nelv, CHUNK):
            rows = np.arange(s, min(s + CHUNK, mesh.nelv))
            try:
                x27, nd = gll_geometry(xyz, curves, pos_of_elid, rows)
            except CurveError as ex:
                curve_err = str(ex)
                x27, nd = gll_geometry(xyz, curves[:0], pos_of_elid, rows)
            ndef += nd
            jm = jacobian_dets(x27).min(axis=1)
            if curves.size:
                jac_min_lin = min(jac_min_lin, float(
                    jacobian_dets(gll_geometry(xyz, curves[:0], pos_of_elid,
                                               rows)[0]).min()))
            jac_min = min(jac_min, float(jm.min()))
            bad = np.flatnonzero(jm <= 0.0)
            if bad.size:
                if n_bad == 0:
                    first_bad = int(elids[s + int(bad[0])])
                n_bad += int(bad.size)
        if curve_err:
            failed = True
            log(' Error: %s -- Neko aborts on this mesh; Jacobians below '
                'are for the straight-sided geometry' % curve_err)
        if curves.size:
            log(' Min Jacobian (curved geometry, %d curved edges applied): '
                '%14.6g' % (ndef, jac_min))
            log(' Min Jacobian (straight-sided):                       '
                '%14.6g' % jac_min_lin)
        else:
            log(' Min Jacobian: %14.6g' % jac_min)
        if is2d:
            log(' (slab Jacobian = 0.5 x the 2D Jacobian)')
        if n_bad > 0:
            failed = True
            log(' Error: Found %d element(s) with a negative/zero Jacobian '
                '(first: element id %d).' % (n_bad, first_bad))
        else:
            log(' No negative/zero Jacobians.')

    if n_unlabeled > 0:
        failed = True
        log(' Error: Found %d unlabelled external faces.' % n_unlabeled)

    if args.write_zone_indices:
        log('')
        log('  [3/3] writing zone-index field ...')
        base = os.path.splitext(args.mesh)[0] + '_zone_indices'
        try:
            gxyz, _ = gll_geometry(xyz, curves, pos_of_elid,
                                   np.arange(mesh.nelv))
        except CurveError:
            gxyz, _ = gll_geometry(xyz, curves[:0], pos_of_elid,
                                   np.arange(mesh.nelv))
        mask = facet_gll_mask()                          # (6, 27)
        sc = np.zeros((mesh.nelv, 27), dtype=np.float32)
        for f0 in range(6):
            v = flabel[:, f0].astype(np.float32)[:, None]  # (nelv, 1)
            on = mask[f0][None, :] & (v > 0)
            np.maximum(sc, np.where(on, v, 0.0), out=sc)   # highest label wins
        # the ACTUAL element ids in record order -- a valid .nmsh may store
        # its records shuffled, and the id list is what maps a block to an
        # element
        write_zone_indices_fld(base, np.asarray(hexmesh.elems['id']), gxyz,
                               sc)
        log('  wrote %s.fld (+ .nek5000 companion)' % base)
    else:
        log('')
        log('  [3/3] no field output requested')

    log(' Done')
    if failed:
        print('Mesh check failed with one or several errors.',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
