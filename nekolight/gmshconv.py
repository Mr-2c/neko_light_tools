# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Gmsh cells -> Neko element records.

The conventions are those of Nek5000's gmsh2nek (shipped in Neko's
``contrib/gmsh2nek``): Gmsh's hexahedron corner order is Nek's cyclic
(preprocessor) vertex order, so corners map one-to-one onto the ``.nmsh``
vertex slots; mid-edge nodes of second-order elements become midside-point
curves (type 4) when they deviate from the chord midpoint by more than
``1e-4`` of the chord length; face-centre and volume-centre nodes are
dropped (Neko rebuilds them); boundary cells are matched to element facets
by node-id set equality.  Unlike gmsh2nek, first-order cells are accepted
and left-handed hexahedra are mirrored (gmsh2nek mirrors quads only).
"""
import sys

import numpy as np

from .formats import EL_DT, QUAD_DT, CURVE_DT, FACE_RE2, QFACE_RE2
from .topology import EDGE_CYC
from .gmsh import TagLookup

# Gmsh element type -> node count, per family
HEX_TYPES = {5: 8, 17: 20, 12: 27}
QUAD_TYPES = {3: 4, 16: 8, 10: 9}
LINE_TYPES = {1: 2, 8: 3, 26: 4, 27: 5}
TRI_TYPES = {2: 3, 9: 6, 20: 9, 21: 10}
OTHER_3D = {4: 'tetrahedron', 6: 'prism', 7: 'pyramid', 11: 'tetrahedron',
            13: 'prism', 14: 'pyramid', 18: 'prism', 19: 'pyramid'}
# mid-edge node k (file node 8+k of a hex20/27) joins these two corners
GMSH_HEX_EDGE = np.array([[0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3],
                          [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]])
# mid-edge node k (file node 4+k of a quad8/9) joins these two corners
GMSH_QUAD_EDGE = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
# quad curve slots 1..4 as vertex-slot pairs (cyclic order, like EDGE_CYC)
QEDGE_CYC = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
# reflection of the first reference coordinate: turns a left-handed cell
# into a right-handed one (gmsh2nek's msh_to_nek_left for quads)
MIRROR = {8: np.array([1, 0, 3, 2, 5, 4, 7, 6]), 4: np.array([1, 0, 3, 2])}
# vertex slots in (ix fastest, then iy, then iz) order
SYM_ORDER = np.array([0, 1, 3, 2, 4, 5, 7, 6])
CURVE_TOL = 1.0e-4          # gmsh2nek: midside deviation / chord length


def corner_jacobians(xyz):
    """Jacobian determinant of the (bi/tri)linear map at every corner.
    ``xyz`` is (n, 8, 3) or (n, 4, 2|3) in .nmsh slot order.  Positive
    everywhere for a right-handed element."""
    n, nv = xyz.shape[:2]
    if nv == 8:
        P = xyz[:, SYM_ORDER].reshape(n, 2, 2, 2, 3)        # [iz, iy, ix]
        dxi = P[:, :, :, 1] - P[:, :, :, 0]                  # (n, iz, iy, 3)
        deta = P[:, :, 1] - P[:, :, 0]                       # (n, iz, ix, 3)
        dzeta = P[:, 1] - P[:, 0]                            # (n, iy, ix, 3)
        A = dxi[:, :, :, None, :]                            # (n,iz,iy,1,3)
        B = deta[:, :, None, :, :]                           # (n,iz,1,ix,3)
        C = dzeta[:, None, :, :, :]                          # (n,1,iy,ix,3)
        det = np.einsum('nabci,nabci->nabc', A, np.cross(B, C))
        return det.reshape(n, 8)
    P = xyz[:, [0, 1, 3, 2], :2].reshape(n, 2, 2, 2)         # [iy, ix]
    dxi = P[:, :, 1] - P[:, :, 0]                            # (n, iy, 2)
    deta = P[:, 1] - P[:, 0]                                 # (n, ix, 2)
    det = (dxi[:, :, None, 0] * deta[:, None, :, 1]
           - dxi[:, :, None, 1] * deta[:, None, :, 0])
    return det.reshape(n, 4)


def orient(corner_xyz, corner_tags, elem_tags, log=None):
    """Make every cell right-handed.  Cells whose corner Jacobians are all
    negative are mirrored (first reference coordinate reflected); mixed
    signs or zeros are an invalid cell and a hard error.  Returns the
    permuted (xyz, tags) and the per-cell permutation array (n, nv)."""
    n, nv = corner_tags.shape
    det = corner_jacobians(corner_xyz)
    neg = (det < 0).all(axis=1)
    pos = (det > 0).all(axis=1)
    bad = ~(neg | pos)
    if bad.any():
        idx = np.flatnonzero(bad)
        sys.exit('Error: %d cell(s) have a sign-changing or zero corner '
                 'Jacobian (degenerate or self-intersecting), e.g. Gmsh '
                 'element(s) %s' % (idx.size, ', '.join(
                     str(int(elem_tags[i])) for i in idx[:10])))
    perm = np.tile(np.arange(nv), (n, 1))
    if neg.any():
        perm[neg] = MIRROR[nv]
        if log:
            log('        %d left-handed cell(s) mirrored to right-handed'
                % int(neg.sum()))
    rows = np.arange(n)[:, None]
    return corner_xyz[rows, perm], corner_tags[rows, perm], perm


def _sorted_key_view(keys):
    """(m, nc) int64 -> (m,) structured void view of the row-sorted keys,
    so rows can be compared with np.unique / searchsorted."""
    k = np.ascontiguousarray(np.sort(keys.astype(np.int64), axis=1))
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


class FacetTable:
    """All facets of all cells keyed by their sorted corner ids: the
    lookup gmsh2nek does with node_hex/node_quad + ifquadmatch, vectorised.
    Facet numbers are Neko's (FACE_RE2 / QFACE_RE2 order, 1-based)."""

    def __init__(self, vid):
        n, nv = vid.shape
        self.slots = FACE_RE2 if nv == 8 else QFACE_RE2
        self.nf = self.slots.shape[0]
        keys = vid[:, self.slots].reshape(n * self.nf, self.slots.shape[1])
        kv = _sorted_key_view(keys)
        self.uniq, first, inv, cnt = np.unique(kv, return_index=True,
                                               return_inverse=True,
                                               return_counts=True)
        self.first = first                  # flat facet index of 1st occurrence
        self.count = cnt
        self.inv = inv.ravel()
        self.n = n
        # boundary facets: multiplicity 1
        bnd = np.flatnonzero(cnt == 1)
        self.boundary_flat = first[bnd]     # flat indices (elem * nf + f)
        if (cnt > 2).any():
            m = int((cnt > 2).sum())
            sys.exit('Error: %d facet(s) are shared by more than two cells '
                     '(the mesh is not a valid conforming hex/quad mesh)' % m)

    def lookup(self, corner_ids):
        """(m, nc) corner ids -> (elem_pos, facet 1-based, multiplicity);
        elem_pos = -1 where no facet has this corner set."""
        kv = _sorted_key_view(corner_ids)
        pos = np.searchsorted(self.uniq, kv)
        pos[pos >= self.uniq.size] = 0
        hit = self.uniq[pos] == kv
        flat = np.where(hit, self.first[pos], -1)
        cnt = np.where(hit, self.count[pos], 0)
        elem = np.where(hit, flat // self.nf, -1)
        facet = np.where(hit, flat % self.nf + 1, 0)
        return elem, facet, cnt

    def facet_ids(self, vid, elem, facet):
        """Corner ids of facets (elem_pos, facet) in FACE order (m, nc)."""
        return vid[np.asarray(elem)[:, None], self.slots[np.asarray(facet) - 1]]


class CellSet:
    """The volume cells of one dimension converted to .nmsh conventions."""

    def __init__(self, gm, dim, log=None):
        self.dim = dim
        fam = HEX_TYPES if dim == 3 else QUAD_TYPES
        nv = 8 if dim == 3 else 4
        vol = [b for b in gm.blocks if b.etype in fam]
        wrong = {}
        for b in gm.blocks:
            if b.dim == dim and b.etype not in fam:
                name = OTHER_3D.get(b.etype) or ('triangle' if b.etype in TRI_TYPES
                                                 else 'type %d' % b.etype)
                wrong[name] = wrong.get(name, 0) + b.nodes.shape[0]
        if wrong:
            sys.exit('Error: the %dD mesh contains non-%s cells: %s.  Neko '
                     'needs an all-%s mesh (Gmsh: Mesh.RecombineAll / '
                     'Recombine Surface, and transfinite or Mesh.Algorithm3D '
                     '= 9 (R-tree) / extrusion for hexahedra)'
                     % (dim, 'hex' if dim == 3 else 'quad',
                        ', '.join('%d %s' % (c, k) for k, c in sorted(wrong.items())),
                        'hex' if dim == 3 else 'quad'))
        if not vol:
            sys.exit('Error: no %s cells in the mesh' % ('hexahedral' if dim == 3
                                                        else 'quadrilateral'))
        self.nidx = gm.node_index()                 # TagLookup: tag -> xyz row
        corners, mids, tags, orders = [], [], [], []
        for b in vol:
            nn = fam[b.etype]
            corners.append(b.nodes[:, :nv])
            if nn > nv:
                mids.append(b.nodes[:, nv:nv + (12 if dim == 3 else 4)])
            else:
                mids.append(np.zeros((b.nodes.shape[0], 12 if dim == 3 else 4),
                                     dtype=np.int64))
            tags.append(b.tags)
            orders.append(nn)
        self.corner_tags = np.concatenate(corners)
        self.mid_tags = np.concatenate(mids)
        self.elem_tags = np.concatenate(tags)
        self.orders = np.repeat(orders, [c.shape[0] for c in corners])
        self.n = self.corner_tags.shape[0]
        crow = self.nidx(self.corner_tags)
        if (crow < 0).any():
            sys.exit('Error: a cell references a node tag that is not in '
                     '$Nodes')
        xyz = gm.xyz[crow]
        if dim == 2:
            zr = xyz[:, :, 2]
            span = float(zr.max() - zr.min()) if zr.size else 0.0
            scale = float(np.abs(xyz[:, :, :2]).max()) or 1.0
            if span > 1e-8 * scale and log:
                log('        warning: 2D mesh is not planar in z (z range %g), '
                    'z is dropped' % span)
        xyz, ctags, self.perm = orient(xyz, self.corner_tags, self.elem_tags,
                                       log)
        self.corner_tags = ctags
        self.flipped = (self.perm != np.arange(nv)).any(axis=1)
        # dense point ids 1..npts in order of first appearance
        uniq, inv = np.unique(self.corner_tags.ravel(), return_inverse=True)
        first = np.full(uniq.size, self.corner_tags.size, dtype=np.int64)
        np.minimum.at(first, inv.ravel(), np.arange(self.corner_tags.size))
        order = np.argsort(first, kind='stable')
        rank = np.empty(uniq.size, dtype=np.int64)
        rank[order] = np.arange(1, uniq.size + 1)
        self.tag_to_id = TagLookup(uniq, rank)      # corner tag -> id, -1 else
        self.vid = rank[inv].reshape(self.corner_tags.shape)     # (n, nv)
        self.npts = uniq.size
        self.xyz = xyz if dim == 3 else xyz[:, :, :2]
        self.xyz_full = xyz                                       # (n, nv, 3)
        self.point_xyz = np.zeros((self.npts + 1, 3))
        self.point_xyz[self.vid.ravel()] = xyz.reshape(-1, 3)
        self.facets = FacetTable(self.vid)
        self.curves, self.n_curved_edges = self._midside_curves(gm)

    # -- curves ------------------------------------------------------------
    def _midside_curves(self, gm):
        nv = 8 if self.dim == 3 else 4
        cyc = EDGE_CYC if self.dim == 3 else QEDGE_CYC
        gedge = GMSH_HEX_EDGE if self.dim == 3 else GMSH_QUAD_EDGE
        has = self.orders > nv
        if not has.any():
            return np.empty(0, dtype=CURVE_DT), 0
        rows = np.flatnonzero(has)
        ne = cyc.shape[0]
        # which Gmsh mid-edge node sits on cyclic edge j, for each permutation
        def edge_map(perm):
            m = np.empty(ne, dtype=np.int64)
            for j, (s1, s2) in enumerate(cyc):
                pair = {int(perm[s1]), int(perm[s2])}
                k = [i for i, e in enumerate(gedge) if set(e.tolist()) == pair]
                assert len(k) == 1
                m[j] = k[0]
            return m
        emap_id = edge_map(np.arange(nv))
        emap_mir = edge_map(MIRROR[nv])
        emap = np.where(self.flipped[rows, None], emap_mir[None, :],
                        emap_id[None, :])                              # (m, ne)
        mid_tags = self.mid_tags[rows[:, None], emap]                  # (m, ne)
        mrow = self.nidx(mid_tags)
        if (mrow < 0).any():
            sys.exit('Error: a second-order cell references a mid-edge node '
                     'tag that is not in $Nodes')
        mid = gm.xyz[mrow]                                             # (m, ne, 3)
        p1 = self.xyz_full[rows[:, None], cyc[None, :, 0]]
        p2 = self.xyz_full[rows[:, None], cyc[None, :, 1]]
        if self.dim == 2:
            mid = mid.copy(); mid[:, :, 2] = 0.0
            p1 = p1.copy(); p1[:, :, 2] = 0.0
            p2 = p2.copy(); p2[:, :, 2] = 0.0
        dev = np.sum((mid - 0.5 * (p1 + p2)) ** 2, axis=2)
        chord = np.sum((p2 - p1) ** 2, axis=2)
        curved = dev > CURVE_TOL ** 2 * chord
        any_c = curved.any(axis=1)
        out = np.zeros(int(any_c.sum()), dtype=CURVE_DT)
        sel = np.flatnonzero(any_c)
        out['e'] = (rows[sel] + 1).astype(np.int32)
        d = np.zeros((sel.size, 12, 5))
        d[:, :ne, :3] = np.where(curved[sel][:, :, None], mid[sel], 0.0)
        out['data'] = d
        t = np.zeros((sel.size, 12), dtype=np.int32)
        t[:, :ne] = np.where(curved[sel], 4, 0)
        out['type'] = t
        return out, int(curved.sum())

    # -- records -----------------------------------------------------------
    def element_records(self):
        el = np.empty(self.n, dtype=EL_DT if self.dim == 3 else QUAD_DT)
        el['id'] = np.arange(1, self.n + 1, dtype=np.int32)
        el['v']['idx'] = self.vid.astype(np.int32)
        xyz = self.xyz_full.copy()
        if self.dim == 2:
            xyz[:, :, 2] = 0.0                    # a 2D .nmsh stores z = 0
        el['v']['xyz'] = xyz
        return el


def boundary_cells(gm, dim):
    """The (dim-1)-cells of the file: corner node tags (m, nc), physical
    tag, entity tag, Gmsh element tag.  Triangles in 3D are an error (a hex
    mesh has quad faces)."""
    fam = QUAD_TYPES if dim == 3 else LINE_TYPES
    nc = 4 if dim == 3 else 2
    corners, phys, ent, tags = [], [], [], []
    for b in gm.blocks:
        if b.dim != dim - 1:
            continue
        if b.etype in fam:
            corners.append(b.nodes[:, :nc]); phys.append(b.physical)
            ent.append(np.full(b.nodes.shape[0], b.entity, dtype=np.int64))
            tags.append(b.tags)
        elif b.etype in TRI_TYPES and dim == 3:
            sys.exit('Error: the boundary contains %d triangle(s); the faces '
                     'of a hexahedral mesh are quadrilaterals' % b.nodes.shape[0])
    if not corners:
        z = np.zeros((0, nc), dtype=np.int64)
        return z, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return (np.concatenate(corners), np.concatenate(phys),
            np.concatenate(ent), np.concatenate(tags))


class UnionFind:
    """Union-find over point ids 0..n; classes are labelled by their
    smallest member, which is what the periodic zone records store."""

    def __init__(self, n):
        self.parent = np.arange(n + 1, dtype=np.int64)

    def find(self, a):
        a = np.asarray(a, dtype=np.int64)
        p = self.parent
        r = a.copy()
        while True:
            q = p[r]
            done = q == r
            if done.all():
                return r
            r = np.where(done, r, p[q])          # path halving

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        # merge smaller root into larger? we want min-labelled classes: link
        # the larger root under the smaller one, iterate until stable
        while True:
            m = ra != rb
            if not m.any():
                break
            lo = np.minimum(ra[m], rb[m]); hi = np.maximum(ra[m], rb[m])
            self.parent[hi] = lo
            ra, rb = self.find(a), self.find(b)

    def labels(self):
        return self.find(np.arange(self.parent.size))
