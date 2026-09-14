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
"""gmsh2nmsh -- convert a Gmsh mesh (.msh 2.2 or 4.1, ASCII or binary) to a
Neko mesh (.nmsh), optionally extruding a 2D mesh into layers of hexahedra.

This is Nek5000's gmsh2nek and n2to3 (both in Neko's contrib) written
against Neko's format:

* Gmsh hexahedra (8/20/27 nodes) or quadrilaterals (4/8/9 nodes) become
  Neko elements; the Gmsh corner order is Nek's cyclic vertex order.
  Left-handed cells are mirrored.  Mid-edge nodes of second-order cells
  become midside-point curves when they leave the chord by more than
  1e-4 of its length; face- and volume-centre nodes are dropped (Neko
  rebuilds them, exactly as gmsh2nek does).
* Physical groups of the boundary (surfaces in 3D, curves in 2D) become
  labelled zones; the label is the physical tag (Neko accepts 1..20;
  --label renames).  Every boundary facet must be in a physical group
  unless --untagged gives them a label.
* Periodic boundaries come from the file's $Periodic section (translations
  only, as in Neko) and/or from --periodic A:B, which pairs two labelled
  zones by translation like create_periodic_zones.  Periodic facets are
  not labelled.
* A 2D mesh is written as a 2D .nmsh (Neko extrudes it to one layer at
  run time) or, with --extrude, as a 3D mesh of NLAYERS hexahedral layers
  between Z0 and Z1 (uniform, or geometric with --gain, or the planes of
  --zfile).  Layer k of 2D element e is element e + k*nel2d, as in n2to3.
  The two new faces are periodic to each other (--zbc periodic) or
  labelled (--zbc BOTTOM TOP).  2D periodicity and labels are carried to
  every layer; midside curves go to the bottom and top edges of each layer.

The output passes the checks mesh_checker.py applies (zone and curve
validity, positive corner and GLL Jacobians of every element, no facet
shared by more than two elements); the tool refuses to write otherwise.

Usage:
  gmsh2nmsh.py in.msh out.nmsh
  gmsh2nmsh.py in.msh out.nmsh --extrude 0 1 8 --zbc periodic
  gmsh2nmsh.py in.msh out.nmsh --extrude 0 2 10 --gain 1.2 --zbc 5 6
  gmsh2nmsh.py in.msh out.nmsh --zfile planes.txt --zbc periodic
  gmsh2nmsh.py in.msh out.nmsh --periodic 1:2 --label outlet=3
"""
import argparse
import os
import sys

import numpy as np

from nekolight import (banner, read_msh, CellSet, boundary_cells, UnionFind,
                       layer_planes, extrude, Mesh, ZONE_DT, MAX_ZLBLS,
                       extrude_2d, write_nmsh, validate_zones, validate_curves,
                       pos_of_elid_map, periodic_replace_merge, gll_geometry,
                       jacobian_dets, corner_jacobians, GMSH_TYPES,
                       facet_table, face_multiplicity)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        prog='gmsh2nmsh.py',
        description='Convert a Gmsh .msh (hex or quad) mesh to a Neko .nmsh, '
                    'optionally extruding a 2D mesh into hexahedral layers.')
    ap.add_argument('input', help='Gmsh .msh file (2.2 or 4.1, ASCII or binary)')
    ap.add_argument('output', help='output .nmsh')
    ex = ap.add_argument_group('extrusion of a 2D mesh')
    ex.add_argument('--extrude', nargs=3, metavar=('Z0', 'Z1', 'NLAYERS'),
                    help='extrude between z=Z0 and z=Z1 with NLAYERS layers')
    ex.add_argument('--gain', type=float, default=None,
                    help='layer growth ratio for --extrude (default 1 = '
                         'uniform; dz_k proportional to GAIN**k, as n2to3)')
    ex.add_argument('--zfile', metavar='FILE',
                    help='file with the ascending z of every plane '
                         '(NLAYERS+1 values), instead of --extrude')
    ex.add_argument('--zbc', nargs='+', metavar='BC',
                    help='"periodic", or two labels BOTTOM TOP for the '
                         'new z- and z+ faces (required with an extrusion)')
    bc = ap.add_argument_group('boundaries')
    bc.add_argument('--label', action='append', default=[], metavar='OLD=NEW',
                    help='rename a physical tag or physical name to Neko '
                         'label NEW (repeatable)')
    bc.add_argument('--untagged', type=int, metavar='LABEL',
                    help='label for boundary facets in no physical group '
                         '(default: refuse)')
    bc.add_argument('--periodic', action='append', default=[], metavar='A:B',
                    help='pair labelled zones A and B by translation '
                         '(repeatable; labels after --label renaming)')
    bc.add_argument('--no-msh-periodic', action='store_true',
                    help='ignore the $Periodic section of the file')
    bc.add_argument('--tol', type=float, default=None,
                    help='matching tolerance for periodic facets (default: '
                         'max(1e-10, 1e-8 * bounding-box diagonal))')
    return ap.parse_args()


def parse_labels(specs, names_by_dim):
    """--label OLD=NEW entries -> {physical tag: label}."""
    out = {}
    for spec in specs:
        if '=' not in spec:
            sys.exit('Error: --label expects OLD=NEW, got %r' % spec)
        old, new = spec.rsplit('=', 1)
        try:
            new = int(new)
        except ValueError:
            sys.exit('Error: --label: the new label must be an integer (%r)'
                     % spec)
        try:
            tag = int(old)
        except ValueError:
            hits = [t for (d, t), nm in names_by_dim.items() if nm == old]
            if len(hits) != 1:
                sys.exit('Error: --label: physical name %r not found among '
                         'the boundary physical groups (%s)'
                         % (old, ', '.join(sorted(set(names_by_dim.values())))
                            or 'none'))
            tag = hits[0]
        out[tag] = new
    return out


def parse_periodic(specs):
    pairs = []
    for spec in specs:
        parts = spec.replace(',', ':').split(':')
        if len(parts) != 2:
            sys.exit('Error: --periodic expects A:B, got %r' % spec)
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            sys.exit('Error: --periodic expects two integer labels, got %r'
                     % spec)
        if a == b:
            sys.exit('Error: --periodic: a zone cannot be paired with itself')
        pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# periodic pairing
# ---------------------------------------------------------------------------
class PeriodicGroup:
    """One periodic zone pair: facet pairs (slave -> master, 1-based element
    and Neko facet), the point correspondences (dense ids, slave, master)
    and a translation vector for the report."""

    def __init__(self, name, pairs, corr, offset):
        self.name, self.pairs, self.corr, self.offset = name, pairs, corr, offset


def match_by_translation(cells, master_flat, slave_flat, offset, tol, what):
    """Pair every master candidate facet F with the slave candidate facet at
    F + offset: centres within tol, corners one-to-one within tol.  Returns
    (pairs (m,4): slave el, f, master el, f; 1-based), corner
    correspondences (dense ids: slave, master), and the flat indices of the
    matched master and slave facets."""
    from scipy.spatial import cKDTree
    ft = cells.facets
    nfac = ft.nf
    if master_flat.size == 0 or slave_flat.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 4), np.int64), np.zeros((0, 2), np.int64), z, z
    me, mf = master_flat // nfac, master_flat % nfac + 1
    se, sf = slave_flat // nfac, slave_flat % nfac + 1
    mid = ft.facet_ids(cells.vid, me, mf)
    sid = ft.facet_ids(cells.vid, se, sf)
    xm, xs = cells.point_xyz[mid], cells.point_xyz[sid]              # (·, nc, 3)
    tree = cKDTree(xs.mean(axis=1))
    d, j = tree.query(xm.mean(axis=1) + offset)
    hit = d <= tol
    if not hit.any():
        z = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 4), np.int64), np.zeros((0, 2), np.int64), z, z
    m = np.flatnonzero(hit)
    s = j[hit]
    if np.unique(s).size != s.size:
        sys.exit('Error: %s: the translation maps two facets onto the same '
                 'facet' % what)
    dd = np.linalg.norm(xm[m][:, :, None, :] + offset - xs[s][:, None, :, :],
                        axis=3)                                       # (m,nc,nc)
    k = dd.argmin(axis=2)
    ok = (dd.min(axis=2) <= tol).all(axis=1) & \
        np.array([np.unique(r).size == r.size for r in k])
    if not ok.all():
        sys.exit('Error: %s: %d facet pair(s) whose centres match but whose '
                 'corners do not map one-to-one under the translation '
                 '(tolerance %.3e)' % (what, int((~ok).sum()), tol))
    rows = np.arange(m.size)[:, None]
    corr = np.stack([sid[s][rows, k].ravel(), mid[m].ravel()], axis=1)
    pairs = np.stack([se[s] + 1, sf[s], me[m] + 1, mf[m]], axis=1)
    return pairs, np.unique(corr, axis=0), master_flat[m], slave_flat[s]


def periodic_from_msh(gm, cells, dim, tol, b_ent, b_flat):
    """$Periodic links of dimension dim-1 give the periodic translations
    (Gmsh's affine maps master to slave); the facets are then paired
    geometrically, like create_periodic_zones does, because the file's
    node correspondences are split over the surface and its bounding
    curves and points.  Candidates are the boundary facets whose corners
    are all periodic slave (master) nodes of the file, or -- when the file
    carries no node correspondences -- the saved boundary cells of the
    slave (master) entities; every candidate must end up paired.  Links
    with the same translation form one group."""
    ft = cells.facets
    nfac = ft.nf
    bflat = ft.boundary_flat
    b_elem, b_facet = bflat // nfac, bflat % nfac + 1
    b_tags = cells.corner_tags[b_elem[:, None], ft.slots[b_facet - 1]]  # (nb, nc)
    links = [lk for lk in gm.periodic if lk.dim == dim - 1]
    if not links:
        return [], np.zeros(0, np.int64), True
    # node-based candidate sets (all links, all dimensions)
    all_s = np.concatenate([lk.node_map[:, 0] for lk in gm.periodic if lk.node_map.size]
                           or [np.zeros(0, np.int64)])
    all_m = np.concatenate([lk.node_map[:, 1] for lk in gm.periodic if lk.node_map.size]
                           or [np.zeros(0, np.int64)])
    have_nodes = all_s.size > 0
    if have_nodes:
        cand_s = bflat[np.isin(b_tags, all_s).all(axis=1)]
        cand_m = bflat[np.isin(b_tags, all_m).all(axis=1)]
    groups, order = {}, []
    for lk in links:
        if lk.affine.size == 16:
            A = lk.affine.reshape(4, 4)
            if not np.allclose(A[:3, :3], np.eye(3), atol=1e-9):
                sys.exit('Error: $Periodic link (entity %d -> %d) is not a '
                         'pure translation; Neko supports translational '
                         'periodicity only' % (lk.entity, lk.master))
            offset = A[:3, 3].copy()
        elif lk.node_map.shape[0]:
            nidx = gm.node_index()
            xs = gm.xyz[nidx(lk.node_map[:, 0])]
            xm = gm.xyz[nidx(lk.node_map[:, 1])]
            offset = (xs - xm).mean(axis=0)
            if not np.allclose(xs - xm, offset, atol=tol):
                sys.exit('Error: $Periodic link (entity %d -> %d) is not a '
                         'single translation' % (lk.entity, lk.master))
        else:
            sys.exit('Error: $Periodic link (entity %d -> %d) has neither an '
                     'affine transform nor node pairs' % (lk.entity, lk.master))
        if dim == 2:
            offset[2] = 0.0
        if np.linalg.norm(offset) <= tol:
            sys.exit('Error: $Periodic link (entity %d -> %d) has a zero '
                     'translation' % (lk.entity, lk.master))
        key = tuple(np.round(offset / max(tol, 1e-12)).astype(np.int64))
        if key not in groups:
            groups[key] = (offset, [], [], [])
            order.append(key)
        groups[key][1].append('%d->%d' % (lk.entity, lk.master))
        groups[key][2].append(lk.entity)
        groups[key][3].append(lk.master)
    out = []
    matched_s = []
    all_cand_s = []
    verifiable = True
    for key in order:
        offset, ents, s_ents, m_ents = groups[key]
        what = '$Periodic (%s)' % ', '.join(ents)
        # candidates: the saved boundary cells of the slave/master entities
        # (complete when present); else the facets whose corners are all
        # periodic slave/master nodes of the file (Gmsh often writes none
        # for the surfaces themselves); else every boundary facet
        ent_s = b_flat.size and np.isin(b_ent, s_ents).any()
        ent_m = b_flat.size and np.isin(b_ent, m_ents).any()
        if ent_s and ent_m:
            cs = np.unique(b_flat[np.isin(b_ent, s_ents)])
            cm = np.unique(b_flat[np.isin(b_ent, m_ents)])
            group_verifiable = True
        elif have_nodes and cand_s.size and cand_m.size:
            cs, cm = cand_s, cand_m
            group_verifiable = True
        else:
            cs = cm = bflat
            group_verifiable = False
            verifiable = False
        pairs, corr, mflat, sflat = match_by_translation(cells, cm, cs, offset,
                                                         tol, what)
        if pairs.shape[0] == 0:
            sys.exit('Error: %s: no boundary facet maps onto another one under '
                     'the translation (%s), tolerance %.3e'
                     % (what, ', '.join('%g' % v for v in offset), tol))
        matched_s.append(sflat)
        if group_verifiable:
            all_cand_s.append(cs)
        out.append(PeriodicGroup('$Periodic entities %s' % ', '.join(ents),
                                 pairs, corr, offset))
    if all_cand_s:
        cand = np.unique(np.concatenate(all_cand_s))
        got = np.unique(np.concatenate(matched_s))
        miss = np.setdiff1d(cand, got)
        if miss.size:
            e_u = miss // nfac
            sys.exit('Error: $Periodic: %d facet(s) of the periodic surfaces '
                     'have no partner under the translations (e.g. element %d '
                     'facet %d): the periodic surfaces are not exact translates '
                     'of each other within the tolerance %.3e (--tol relaxes it)'
                     % (miss.size, int(e_u[0]) + 1, int(miss[0] % nfac) + 1, tol))
    if not verifiable:
        log('        note: the file has neither saved cells of the periodic '
            'entities nor node pairs for them, so completeness of the '
            'periodic pairing cannot be verified')
    return out, np.zeros(0, np.int64), verifiable


def periodic_from_labels(label_pairs, lab_elem, lab_facet, lab_label, cells,
                         tol):
    """--periodic A:B: pair the facets of zone A (masters) with those of
    zone B (slaves) by the mean translation (create_periodic_zones' rule);
    every facet of both zones must be paired."""
    groups = []
    nfac = cells.facets.nf
    for a, b in label_pairs:
        sa, sb = lab_label == a, lab_label == b
        if not sa.any() or not sb.any():
            sys.exit('Error: --periodic %d:%d refers to an empty labelled zone'
                     % (a, b))
        if sa.sum() != sb.sum():
            sys.exit('Error: --periodic %d:%d: the zones have %d and %d facets'
                     % (a, b, int(sa.sum()), int(sb.sum())))
        fa = lab_elem[sa] * nfac + lab_facet[sa] - 1
        fb = lab_elem[sb] * nfac + lab_facet[sb] - 1
        ida = cells.facets.facet_ids(cells.vid, lab_elem[sa], lab_facet[sa])
        idb = cells.facets.facet_ids(cells.vid, lab_elem[sb], lab_facet[sb])
        offset = cells.point_xyz[idb].mean(axis=(0, 1)) - \
            cells.point_xyz[ida].mean(axis=(0, 1))
        if np.linalg.norm(offset) <= tol:
            sys.exit('Error: --periodic %d:%d: the two zones coincide (zero '
                     'translation)' % (a, b))
        pairs, corr, _, _ = match_by_translation(cells, fa, fb, offset, tol,
                                                 '--periodic %d:%d' % (a, b))
        if pairs.shape[0] != sa.sum():
            sys.exit('Error: --periodic %d:%d: only %d of %d facets map onto '
                     'the other zone under the translation (%s), tolerance %.3e'
                     % (a, b, pairs.shape[0], int(sa.sum()),
                        ', '.join('%g' % v for v in offset), tol))
        groups.append(PeriodicGroup('zones %d <-> %d' % (a, b), pairs, corr,
                                    offset))
    return groups


def coincident_boundary_facets(cells, flat, tol):
    """Number of pairs of distinct boundary facets with the same centre (a
    cracked mesh: touching volumes that were not fragmented in Gmsh)."""
    from scipy.spatial import cKDTree
    if flat.size < 2:
        return 0
    ft = cells.facets
    e, f = flat // ft.nf, flat % ft.nf + 1
    c = cells.point_xyz[ft.facet_ids(cells.vid, e, f)].mean(axis=1)
    pairs = cKDTree(c).query_pairs(tol)
    return len(pairs)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    log(banner('gmsh2nmsh'))
    log('  input     : %s' % args.input)
    log('  output    : %s' % args.output)
    outdir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(outdir) or not os.access(outdir, os.W_OK):
        sys.exit('Error: cannot write to the output directory %s' % outdir)
    gm = read_msh(args.input)
    log('  format    : msh %s %s, %d nodes, %d element blocks'
        % (gm.version, 'binary' if gm.binary else 'ASCII', gm.node_tags.size,
           len(gm.blocks)))
    counts = {}
    for b in gm.blocks:
        nm = GMSH_TYPES[b.etype][0]
        counts[nm] = counts.get(nm, 0) + b.nodes.shape[0]
    log('  cells     : %s' % ', '.join('%d %s' % (c, k) for k, c in sorted(counts.items())))
    has_hex = any(b.etype in (5, 17, 12) for b in gm.blocks)
    has_quad = any(b.etype in (3, 16, 10) for b in gm.blocks)
    if has_hex:
        dim = 3
    elif has_quad:
        dim = 2
    else:
        sys.exit('Error: the mesh contains neither hexahedra nor quadrilaterals '
                 '(cells: %s); Neko needs an all-hex (3D) or all-quad (2D) '
                 'mesh -- in Gmsh use Recombine / Mesh.RecombineAll, '
                 'transfinite or extruded meshing'
                 % ', '.join('%d %s' % (c, k) for k, c in sorted(counts.items())))
    extruding = args.extrude is not None or args.zfile is not None
    if extruding and dim == 3:
        sys.exit('Error: --extrude/--zfile apply to 2D (quad) meshes only')
    if args.extrude is not None and args.zfile is not None:
        sys.exit('Error: give either --extrude or --zfile, not both')
    if args.gain is not None and args.zfile is not None:
        sys.exit('Error: --gain has no effect with --zfile (the file gives the '
                 'planes)')
    if args.gain is not None and args.extrude is None:
        sys.exit('Error: --gain only applies together with --extrude')
    gain = 1.0 if args.gain is None else args.gain
    if extruding and not args.zbc:
        sys.exit('Error: an extrusion needs --zbc periodic or --zbc BOTTOM TOP')
    if args.zbc and not extruding:
        sys.exit('Error: --zbc only applies together with --extrude/--zfile')
    zplanes = None
    zbc = None
    if extruding:
        if args.zfile:
            try:
                zplanes = np.loadtxt(args.zfile, dtype=np.float64).ravel()
            except (OSError, ValueError) as e:
                sys.exit('Error: cannot read the z planes from %s (%s)'
                         % (args.zfile, e))
            if zplanes.size < 2 or not (np.diff(zplanes) > 0).all():
                sys.exit('Error: --zfile must hold at least two strictly '
                         'ascending z values (one per plane)')
        else:
            try:
                z0, z1, nlev = float(args.extrude[0]), float(args.extrude[1]), \
                    int(args.extrude[2])
            except ValueError:
                sys.exit('Error: --extrude expects Z0 Z1 NLAYERS')
            zplanes = layer_planes(z0, z1, nlev, gain)
        if len(args.zbc) == 1 and args.zbc[0].lower() in ('periodic', 'p'):
            zbc = 'periodic'
            if zplanes.size - 1 == 2:
                sys.exit('Error: 2 layers with periodic z faces are not a valid '
                         'mesh: the lateral faces of the two layers get identical '
                         'corner ids after the periodic merge (n2to3 requires at '
                         'least 3 layers); use 1 or 3 or more layers')
        elif len(args.zbc) == 2:
            try:
                zbc = (int(args.zbc[0]), int(args.zbc[1]))
            except ValueError:
                sys.exit('Error: --zbc expects "periodic" or two integer labels')
        else:
            sys.exit('Error: --zbc expects "periodic" or two integer labels')

    log('  [1/4] converting %s cells ...' % ('hexahedral' if dim == 3 else 'quadrilateral'))
    cells = CellSet(gm, dim, log)
    log('        %d elements, %d corner points, %d curved edges on %d elements'
        % (cells.n, cells.npts, cells.n_curved_edges, cells.curves.shape[0]))
    if (cells.orders == 27).any() or (cells.orders == 9).any():
        log('        note: face/volume-centre nodes of the second-order cells '
            'are dropped (Neko reconstructs them), as gmsh2nek does')

    # ---- boundary facets -------------------------------------------------
    log('  [2/4] boundary zones ...')
    names = {(d, t): nm for (d, t), nm in gm.physical_names.items() if d == dim - 1}
    relabel = parse_labels(args.label, names)
    b_corners, b_phys, b_ent, b_tags = boundary_cells(gm, dim)
    # 4.1: an entity with several physical groups is ambiguous (2.2 carries
    # the physical tag per element, so no ambiguity is possible there)
    if gm.version.startswith('4'):
        multi = [(e, p) for (d, e), p in gm.entity_physical.items()
                 if d == dim - 1 and len(p) > 1]
        if multi:
            sys.exit('Error: boundary entity(ies) with several physical groups '
                     '(ambiguous label): %s' % ', '.join('%d: %s' % m for m in multi))
    ft = cells.facets
    nfac = ft.nf
    bflat = ft.boundary_flat
    lab_elem = np.zeros(0, dtype=np.int64)
    lab_facet = np.zeros(0, dtype=np.int64)
    lab_label = np.zeros(0, dtype=np.int64)
    tagged_flat = np.zeros(0, dtype=np.int64)
    b_flat_all = np.zeros(0, dtype=np.int64)     # facet of every boundary cell
    b_ent_all = np.zeros(0, dtype=np.int64)
    if b_corners.shape[0]:
        ids = cells.tag_to_id(b_corners)
        ok = (ids >= 0).all(axis=1)
        e, f, cnt = ft.lookup(np.where(ok[:, None], ids, 1))
        e[~ok] = -1
        found = (e >= 0) & (cnt == 1)
        b_flat_all = e[found] * nfac + f[found] - 1
        b_ent_all = b_ent[found]
        tagged = found & (b_phys != 0)
        nomatch = (b_phys != 0) & ~ok | ((b_phys != 0) & ok & (e < 0))
        internal = (b_phys != 0) & (e >= 0) & (cnt == 2)
        if nomatch.any():
            log('        warning: %d tagged boundary cell(s) match no element '
                'facet (e.g. Gmsh elements %s) -- ignored'
                % (int(nomatch.sum()), ', '.join(str(int(t)) for t in b_tags[nomatch][:5])))
        if internal.any():
            log('        warning: %d tagged cell(s) lie on interior facets '
                '(shared by two elements) -- ignored, Neko boundaries are '
                'exterior' % int(internal.sum()))
        lab_elem, lab_facet = e[tagged], f[tagged]
        lab_label = np.array([relabel.get(int(p), int(p)) for p in b_phys[tagged]],
                             dtype=np.int64)
        tagged_flat = lab_elem * nfac + (lab_facet - 1)
        # a facet tagged by two boundary cells must carry one label
        if tagged_flat.size:
            uq, inv = np.unique(tagged_flat, return_inverse=True)
            lo = np.full(uq.size, np.iinfo(np.int64).max); hi = np.zeros(uq.size, np.int64)
            np.minimum.at(lo, inv, lab_label); np.maximum.at(hi, inv, lab_label)
            conflict = lo != hi
            if conflict.any():
                ex = np.flatnonzero(conflict)[:3]
                sys.exit('Error: %d boundary facet(s) carry two different '
                         'labels, e.g. %s' % (int(conflict.sum()),
                                              ['%d/%d' % (lo[i], hi[i]) for i in ex]))
            first = np.unique(tagged_flat, return_index=True)[1]
            lab_elem, lab_facet, lab_label, tagged_flat = \
                lab_elem[first], lab_facet[first], lab_label[first], tagged_flat[first]

    # ---- periodic groups --------------------------------------------------
    groups = []
    tol = args.tol
    if tol is None:
        lo = cells.point_xyz[1:].min(axis=0); hi = cells.point_xyz[1:].max(axis=0)
        tol = max(1e-10, 1e-8 * max(1.0, float(np.linalg.norm(hi - lo))))
    if not args.no_msh_periodic:
        g_msh, _, _ = periodic_from_msh(gm, cells, dim, tol, b_ent_all, b_flat_all)
        groups += g_msh
    label_pairs = parse_periodic(args.periodic)
    if label_pairs:
        groups += periodic_from_labels(label_pairs, lab_elem, lab_facet,
                                       lab_label, cells, tol)
    # periodic facets are not labelled zones
    per_flat = np.zeros(0, dtype=np.int64)
    if groups:
        allp = np.concatenate([g.pairs for g in groups])
        s_flat = (allp[:, 0] - 1) * nfac + allp[:, 1] - 1
        m_flat = (allp[:, 2] - 1) * nfac + allp[:, 3] - 1
        both = np.concatenate([s_flat, m_flat])
        if np.unique(both).size != both.size:
            sys.exit('Error: a facet appears in two periodic pairings')
        per_flat = np.unique(both)
        if lab_elem.size:
            drop = np.isin(tagged_flat, per_flat)
            if drop.any():
                log('        %d labelled facet(s) are periodic and lose their '
                    'label (labels %s)' % (int(drop.sum()), sorted(set(lab_label[drop].tolist()))))
            lab_elem, lab_facet, lab_label, tagged_flat = \
                lab_elem[~drop], lab_facet[~drop], lab_label[~drop], tagged_flat[~drop]

    # ---- untagged boundary facets ------------------------------------------
    covered = np.zeros(cells.n * nfac, dtype=bool)
    covered[tagged_flat] = True
    covered[per_flat] = True
    untagged = bflat[~covered[bflat]]
    if untagged.size:
        cracks = coincident_boundary_facets(cells, untagged, tol)
        if cracks:
            sys.exit('Error: %d pair(s) of boundary facets coincide: the mesh '
                     'has duplicated points along an interior interface '
                     '(volumes that touch but were not fragmented -- use '
                     'BooleanFragments / Coherence in Gmsh)' % cracks)
        if args.untagged is None:
            e_u = untagged // nfac
            hint = ('tag every boundary surface in Gmsh with a physical group '
                    '(Gmsh saves only entities in physical groups, unless '
                    'Mesh.SaveAll is set -- which in format 2.2 drops the '
                    'physical tags, so use format 4.1 with SaveAll)')
            sys.exit('Error: %d boundary facet(s) belong to no physical group '
                     '(e.g. element %d facet %d); %s, or give them a label with '
                     '--untagged LABEL'
                     % (untagged.size, int(e_u[0]) + 1, int(untagged[0] % nfac) + 1,
                        hint))
        log('        %d untagged boundary facet(s) get label %d'
            % (untagged.size, args.untagged))
        lab_elem = np.concatenate([lab_elem, untagged // nfac])
        lab_facet = np.concatenate([lab_facet, untagged % nfac + 1])
        lab_label = np.concatenate([lab_label, np.full(untagged.size, args.untagged)])
    if lab_label.size and ((lab_label < 1) | (lab_label > MAX_ZLBLS)).any():
        bad = sorted(set(lab_label[(lab_label < 1) | (lab_label > MAX_ZLBLS)].tolist()))
        sys.exit('Error: Neko zone labels must be in 1..%d; physical tags %s '
                 'are outside (rename them with --label OLD=NEW)' % (MAX_ZLBLS, bad))

    # ---- extrusion --------------------------------------------------------------
    elems = cells.element_records()
    curves = cells.curves
    vid = cells.vid.astype(np.int64)
    npts = cells.npts
    nelv = cells.n
    if extruding:
        nlev = zplanes.size - 1
        log('  [3/4] extruding %d layers, z = %g .. %g%s ...'
            % (nlev, zplanes[0], zplanes[-1],
               ' (gain %g)' % gain if args.gain is not None else ''))
        elems, curves, npts2d = extrude(vid, cells.xyz_full, zplanes, curves)
        vid = elems['v']['idx'].astype(np.int64)
        npts = npts2d * (nlev + 1)
        n2 = nelv
        nelv = elems.shape[0]
        lab_elem = np.concatenate([lab_elem + k * n2 for k in range(nlev)])
        lab_facet = np.tile(lab_facet, nlev)
        lab_label = np.tile(lab_label, nlev)
        new_groups = []
        for g in groups:
            pairs = np.concatenate([g.pairs + np.array([k * n2, 0, k * n2, 0])
                                    for k in range(nlev)])
            corr = np.concatenate([g.corr + k * npts2d for k in range(nlev + 1)])
            new_groups.append(PeriodicGroup(g.name, pairs, corr, g.offset))
        groups = new_groups
        e2 = np.arange(1, n2 + 1, dtype=np.int64)
        if zbc == 'periodic':
            pairs = np.stack([e2, np.full(n2, 5), e2 + (nlev - 1) * n2,
                              np.full(n2, 6)], axis=1)
            p = np.arange(1, npts2d + 1, dtype=np.int64)
            corr = np.stack([p + nlev * npts2d, p], axis=1)
            groups.append(PeriodicGroup('z planes (facets 5 <-> 6)', pairs, corr,
                                        np.array([0.0, 0.0, zplanes[0] - zplanes[-1]])))
        else:
            lb, lt = zbc
            if not (1 <= lb <= MAX_ZLBLS and 1 <= lt <= MAX_ZLBLS):
                sys.exit('Error: --zbc labels must be in 1..%d' % MAX_ZLBLS)
            lab_elem = np.concatenate([lab_elem, e2 - 1, e2 - 1 + (nlev - 1) * n2])
            lab_facet = np.concatenate([lab_facet, np.full(n2, 5), np.full(n2, 6)])
            lab_label = np.concatenate([lab_label, np.full(n2, lb), np.full(n2, lt)])
        gdim = 3
    else:
        log('  [3/4] no extrusion (%dD output)' % dim)
        gdim = dim

    # ---- zone records ---------------------------------------------------------
    log('  [4/4] zones, checks and output ...')
    fts = facet_table(8 if gdim == 3 else 4)
    nc = fts.shape[1]
    uf = UnionFind(npts)
    for g in groups:
        uf.union(g.corr[:, 0], g.corr[:, 1])
    labels = uf.labels()
    zp_list = []
    for g in groups:
        for (el, f, pe, pf) in g.pairs:
            zp_list.append((el, f, pe, pf))
            zp_list.append((pe, pf, el, f))
    zp = np.zeros(len(zp_list), dtype=ZONE_DT)
    for i, (el, f, pe, pf) in enumerate(zp_list):
        zp['e'][i], zp['f'][i], zp['p_e'][i], zp['p_f'][i] = el, f, pe, pf
        zp['g'][i, :nc] = labels[vid[el - 1, fts[f - 1]]]
    zp['t'] = 5
    zl = np.zeros(lab_elem.size, dtype=ZONE_DT)
    order = np.argsort(lab_label, kind='stable') if lab_elem.size else []
    zl['e'] = (lab_elem[order] + 1)
    zl['f'] = lab_facet[order]
    zl['p_f'] = lab_label[order]
    zl['t'] = 7
    zones = np.concatenate([zp, zl])
    validate_zones(nelv, zones, args.output, gdim)
    validate_curves(nelv, curves, args.output, gdim, log)
    # Neko's replacement semantics must realise every correspondence
    pos = pos_of_elid_map(nelv, elems)
    merged = periodic_replace_merge(nelv, vid, zones, pos)
    if groups:
        fin = np.zeros(npts + 1, dtype=np.int64)
        fin[vid.ravel()] = merged.ravel()
        for g in groups:
            if (fin[g.corr[:, 0]] != fin[g.corr[:, 1]]).any():
                sys.exit('Error: internal: periodic correspondence not realised '
                         'for %s' % g.name)
        # a periodic direction two elements thick makes distinct faces share
        # all their corner ids (n2to3 refuses nlev < 3 for the same reason)
        mult, _ = face_multiplicity(merged)
        if (mult > 2).any():
            sys.exit('Error: after the periodic merge %d facet(s) are shared by '
                     'more than two elements: a periodic direction is only two '
                     'elements thick, so distinct faces get identical corner '
                     'ids; use one or at least three elements across it'
                     % int((mult > 2).sum()))
    # geometry checks: corner Jacobians, then the GLL Jacobian of every
    # element (curves applied), chunked as mesh_checker does
    det = corner_jacobians(elems['v']['xyz'] if gdim == 3 else elems['v']['xyz'][:, :, :2])
    if (det <= 0).any():
        sys.exit('Error: internal: %d element(s) with non-positive corner '
                 'Jacobian after conversion' % int((det <= 0).any(axis=1).sum()))
    m3 = Mesh(nelv, elems, zones, curves, 0, gdim)
    if gdim == 2:
        m3 = extrude_2d(m3)
    minj = np.inf
    nbad = 0
    first_bad = None
    chunk = 1 << 16
    for start in range(0, nelv, chunk):
        rows = np.arange(start, min(start + chunk, nelv))
        x27, _ = gll_geometry(m3.elems['v']['xyz'], m3.curves, pos, rows)
        dets = jacobian_dets(x27)
        minj = min(minj, float(dets.min()))
        bad = (dets <= 0).any(axis=1)
        if bad.any():
            nbad += int(bad.sum())
            if first_bad is None:
                first_bad = int(elems['id'][rows[np.flatnonzero(bad)[0]]])
    if nbad:
        sys.exit('Error: %d element(s) have a non-positive GLL Jacobian (e.g. '
                 'element %d; minimum %.3e): the cell is too distorted, or its '
                 'mid-edge nodes deviate too much from the chord; refine or '
                 'straighten the Gmsh mesh' % (nbad, first_bad, minj))
    try:
        write_nmsh(args.output, elems, (zp, zl), curves, inputs=(args.input,))
    except OSError as e:
        sys.exit('Error: cannot write %s (%s)' % (args.output, e))

    # ---- report -----------------------------------------------------------------
    log('  elements  : %d (%dD), %d points, %d curved elements'
        % (nelv, gdim, int(np.unique(merged).size), curves.shape[0]))
    if lab_label.size:
        log('  labelled zones:')
        pname = {}
        for (d, t), nm in gm.physical_names.items():
            if d == dim - 1:
                pname[relabel.get(t, t)] = nm
        for lb in sorted(set(lab_label.tolist())):
            n = int((lab_label == lb).sum())
            if zbc not in (None, 'periodic') and lb in zbc:
                extra = 'z-face' if lb not in pname else pname[lb] + ' / z-face'
            elif args.untagged is not None and lb == args.untagged:
                extra = pname.get(lb, '') + (' / ' if lb in pname else '') + 'untagged'
            else:
                extra = pname.get(lb, '')
            log('    label %2d : %8d facets  %s' % (lb, n, extra))
    for g in groups:
        log('  periodic  : %-32s %6d facet pairs, translation (%s)'
            % (g.name, g.pairs.shape[0], ', '.join('%g' % v for v in g.offset)))
    log('  GLL Jacobian minimum: %.4g%s'
        % (minj, ' (curved geometry)' if curves.size else ''))
    log('  done -> %s' % args.output)


if __name__ == '__main__':
    main()
