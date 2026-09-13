# Copyright (c) 2026, The Neko Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
#   * Neither the name of the authors nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#     _  __  ____  __ __  ____
#    / |/ / / __/ / //_/ / __ \
#   /    / / _/  / ,<   / /_/ /
#  /_/|_/ /___/ /_/|_|  \____/
#
"""Hex-element geometry on the 3x3x3 GLL grid -- the grid Neko's own
mesh_checker uses (``Xh%init(1, 3, 3, 3)``).

* :func:`gll_xyz` is the trilinear (straight-sided) map at the 27 nodes,
  Neko's ``dofmap_xyzlin``.
* :func:`apply_curves` reproduces ``dofmap_generate_xyz`` for the curve types
  Neko constructs geometry for: midside points (type 4, re2 'm') through the
  vertex+edge Gordon-Hall blend of ``dofmap_xyzquad``/``gh_face_extend_3d``,
  then circular arcs (type 3, re2 'C') through ``arc_surface``, edge by edge,
  in the same order.  With lx = 3 the interpolation steps of the Fortran are
  identities, so the node coordinates -- and hence the Jacobians and facet
  normals below -- are the ones Neko's checker computes.
* :func:`jacobian_gll` differentiates the quadratic interpolant of the node
  coordinates (exact for the trilinear map and for the curved maps, which
  are quadratic on this grid).

Node p of an element is ``ir + 3*is + 9*it`` (r fastest), the nek fld order
and Neko's column-major ``(i, j, k)``.
"""

import numpy as np

from .formats import SIJK

# the 3-point GLL nodes on [-1, 1] and the Lagrange differentiation matrix
GLL3 = np.array([-1.0, 0.0, 1.0])
D3 = np.array([[-1.5, 2.0, -0.5],
               [-0.5, 0.0, 0.5],
               [0.5, -2.0, 1.5]])
# linear blending weights of the two ends along one axis (Neko's compute_h):
# H[a, i] = weight of end a (0: -1 end, 1: +1 end) at node i
H = np.array([(1.0 - GLL3) * 0.5, (1.0 + GLL3) * 0.5])            # (2, 3)

# corner signs of the 8 nmsh vertex slots (2 * SIJK - 1)
_RST = 2.0 * SIJK.astype(np.float64) - 1.0                        # (8, 3)

# the 27 trilinear shape functions at the GLL nodes, in node order
_t, _s, _r = np.meshgrid(GLL3, GLL3, GLL3, indexing='ij')
_pts = np.stack([_r.ravel(), _s.ravel(), _t.ravel()], axis=1)     # (27, 3)
SHAPE_N = 0.125 * (1.0 + _pts[:, None, :] * _RST[None, :, :]).prod(axis=2)
del _t, _s, _r, _pts

# --- Neko's numbering tables needed by the curve code -----------------------
# symmetric vertex 1..8 -> nmsh file slot (0-based): the [1,2,4,3,5,6,8,7] swap
SYM2SLOT = np.array([0, 1, 3, 2, 4, 5, 7, 6])
# hex.f90 edge_nodes: the 12 symmetric edges as symmetric vertex pairs
SYM_EDGE_NODES = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 3], [2, 4],
                           [5, 7], [6, 8], [1, 5], [2, 6], [3, 7], [4, 8]])
# dofmap.f90 arc_surface: cyclic (re2) edge -> symmetric edge / face
ECYC_TO_SYM = np.array([1, 6, 2, 5, 3, 8, 4, 7, 9, 10, 12, 11])
FCYC_TO_SYM = np.array([3, 2, 4, 1, 5, 6])
# dofmap.f90 eindx: 0-based node of the midpoint of cyclic edge 1..12
EDGE_MID_NODE = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
# node at the centre of facet 1..6
FACET_CENTRE_NODE = np.array([12, 14, 10, 16, 4, 22])


class CurveError(ValueError):
    """A curve record Neko's geometry generation would refuse."""


def gll_xyz(corners):
    """GLL-node coordinates of the trilinear map: (m, 8, 3) -> (m, 27, 3)."""
    return np.einsum('pk,mkc->mpc', SHAPE_N, corners)


def _grid(xyz27):
    """(m, 27, 3) -> view (m, it, is, ir, 3)."""
    return xyz27.reshape(xyz27.shape[0], 3, 3, 3, 3)


def jacobian_gll(xyz27):
    """Jacobian matrices at the 27 nodes: (m, 27, 3) -> (m, 27, 3, 3) with
    ``J[..., c, d] = d x_c / d r_d`` (r_0 = r, r_1 = s, r_2 = t)."""
    g = _grid(xyz27)
    J = np.empty(xyz27.shape[:2] + (3, 3))
    J[..., 0] = np.einsum('ab,mtsbc->mtsac', D3, g).reshape(-1, 27, 3)
    J[..., 1] = np.einsum('ab,mtbrc->mtarc', D3, g).reshape(-1, 27, 3)
    J[..., 2] = np.einsum('ab,mbsrc->masrc', D3, g).reshape(-1, 27, 3)
    return J


def jacobian_dets(xyz27):
    """det J at the 27 nodes: (m, 27, 3) -> (m, 27).  <= 0 anywhere flags an
    inverted or degenerate element (Neko's coef%jac sign convention)."""
    J = jacobian_gll(xyz27)
    return (J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2]
                            - J[..., 1, 2] * J[..., 2, 1])
            - J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2]
                              - J[..., 1, 2] * J[..., 2, 0])
            + J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1]
                              - J[..., 1, 1] * J[..., 2, 0]))


def min_jacobian(corners):
    """Minimum Jacobian over the 27 GLL nodes of the STRAIGHT-SIDED map:
    (m, 8, 3) -> (m,).  Apply :func:`apply_curves` to ``gll_xyz(corners)``
    first for the curved geometry."""
    return jacobian_dets(gll_xyz(corners)).min(axis=1)


def facet_normals(xyz27):
    """Unit normals at the centre node of facets 1..6: (m, 27, 3) ->
    (m, 6, 3).  Orientation is not normalised (Neko's alignment test uses
    absolute components)."""
    J = jacobian_gll(xyz27)
    n = np.empty((xyz27.shape[0], 6, 3))
    for f, node in enumerate(FACET_CENTRE_NODE):
        Jf = J[:, node]                                     # (m, 3, 3)
        if f < 2:
            v = np.cross(Jf[:, :, 1], Jf[:, :, 2])
        elif f < 4:
            v = np.cross(Jf[:, :, 2], Jf[:, :, 0])
        else:
            v = np.cross(Jf[:, :, 0], Jf[:, :, 1])
        nrm = np.linalg.norm(v, axis=1)
        nrm[nrm == 0.0] = 1.0
        n[:, f] = v / nrm[:, None]
    return n


def facet_gll_mask():
    """(6, 27) bool: which of the 27 GLL nodes lie on facet 1..6."""
    idx = np.arange(27)
    ir, is_, it = idx % 3, (idx // 3) % 3, idx // 9
    m = np.zeros((6, 27), dtype=bool)
    m[0], m[1] = ir == 0, ir == 2
    m[2], m[3] = is_ == 0, is_ == 2
    m[4], m[5] = it == 0, it == 2
    return m


# ---------------------------------------------------------------------------
# Curved geometry (dofmap_generate_xyz on the 3x3x3 grid)
# ---------------------------------------------------------------------------
def _gh_vertex_edge_extend(g):
    """Neko's ``gh_face_extend_3d(x, zg, 3, gh_type=2)`` on (m, 3, 3, 3, c)
    grids: rebuild every node from the 8 corners (trilinear) plus the linear
    blend of the 12 edge deviations.  Face centres and the volume centre of
    the input are discarded, exactly as in the Fortran."""
    c = g[:, ::2, ::2, ::2, :]                          # corners (m,2,2,2,c)
    v = np.einsum('ak,bj,ci,mabcx->mkjix', H, H, H, c)  # vertex interpolant
    d = g - v                                           # edge deviations
    # grid axes are (m, k=t, j=s, i=r, x); an edge line keeps one axis free
    e = np.einsum('ak,bj,mabix->mkjix', H, H, d[:, ::2, ::2, :, :])   # r-edges
    e += np.einsum('ak,ci,majcx->mkjix', H, H, d[:, ::2, :, ::2, :])  # s-edges
    e += np.einsum('bj,ci,mkbcx->mkjix', H, H, d[:, :, ::2, ::2, :])  # t-edges
    return v + e


def apply_curves(xyz27, corners, curves, pos_of_elid=None, rows=None):
    """Deform the straight-sided GLL coordinates ``xyz27`` (m, 27, 3, modified
    in place) according to the curve records, as Neko's dofmap does:

    1. every element with at least one type-4 edge gets the midside points
       of those edges replaced by ``curve_data(1:3)`` and the whole element
       rebuilt by the vertex+edge Gordon-Hall blend;
    2. every type-3 edge among edges 1..8 (the planar ones -- Neko never
       arcs the vertical edges) adds ``arc_surface``'s perturbation to x, y.

    ``corners`` are the (m, 8, 3) element corners (needed by the arcs),
    ``curves`` the CURVE_DT records and ``pos_of_elid`` the id -> row map
    into ``xyz27``/``corners`` (or pass the rows directly as ``rows``).
    Raises :class:`CurveError` where Neko's ``arc_surface`` would abort
    ('Radius to small for arced element surface').  Returns the number of
    (element, edge) pairs deformed.
    """
    if curves.size == 0:
        return 0
    if rows is None:
        rows = pos_of_elid[curves['e'].astype(np.int64)]
    ctype = curves['type']
    cdata = curves['data']                              # (nc, 12, 5)
    ndef = 0

    # -- 1. midside points --------------------------------------------------
    has_mid = (ctype == 4).any(axis=1)
    if has_mid.any():
        r = rows[has_mid]
        g = _grid(xyz27[r]).copy()                      # (k, 3,3,3, 3)
        flat = g.reshape(r.size, 27, 3)
        t4 = ctype[has_mid] == 4                        # (k, 12)
        k_idx, e_idx = np.nonzero(t4)
        flat[k_idx, EDGE_MID_NODE[e_idx], :] = cdata[has_mid][k_idx, e_idx, :3]
        xyz27[r] = _gh_vertex_edge_extend(g).reshape(r.size, 27, 3)
        ndef += int(t4.sum())

    # -- 2. circular arcs, cyclic edges 1..8 --------------------------------
    for isid in range(1, 9):
        sel = ctype[:, isid - 1] == 3
        if not sel.any():
            continue
        r = rows[sel]
        radius = cdata[sel, isid - 1, 0]
        itmp = ECYC_TO_SYM[isid - 1]
        a, b = SYM_EDGE_NODES[itmp - 1]
        if isid in (1, 2, 5, 6):
            s1, s2 = SYM2SLOT[a - 1], SYM2SLOT[b - 1]
        else:
            s1, s2 = SYM2SLOT[b - 1], SYM2SLOT[a - 1]
        pt1 = corners[r, s1, :2]                         # (k, 2)
        pt2 = corners[r, s2, :2]
        xs = pt2[:, 1] - pt1[:, 1]
        ys = pt1[:, 0] - pt2[:, 0]
        xys = np.sqrt(xs ** 2 + ys ** 2)
        bad = np.abs(2.0 * radius) <= xys * 1.00001
        if bad.any():
            which = int(np.flatnonzero(bad)[0])
            eid = int(curves['e'][sel][which])
            raise CurveError('Radius to small for arced element surface '
                             '(element %d, edge %d, radius %g, chord %g)'
                             % (eid, isid, radius[which], xys[which]))
        dtheta = np.abs(np.arcsin(0.5 * xys / radius))
        pt12 = 0.5 * (pt1 + pt2)
        xcenn = pt12[:, 0] - xs / xys * radius * np.cos(dtheta)
        ycenn = pt12[:, 1] - ys / xys * radius * np.cos(dtheta)
        theta0 = np.arctan2(pt12[:, 1] - ycenn, pt12[:, 0] - xcenn)
        isid1 = (isid - 1) % 4 + 1
        dtheta = np.where(radius < 0.0, -dtheta, dtheta)
        ang = theta0[:, None] + GLL3[None, :] * dtheta[:, None]      # (k, 3)
        lin_x = H[0][None, :] * pt1[:, 0:1] + H[1][None, :] * pt2[:, 0:1]
        lin_y = H[0][None, :] * pt1[:, 1:2] + H[1][None, :] * pt2[:, 1:2]
        xcrved = xcenn[:, None] + np.abs(radius)[:, None] * np.cos(ang) - lin_x
        ycrved = ycenn[:, None] + np.abs(radius)[:, None] * np.sin(ang) - lin_y
        if isid1 > 2:                                   # ixt = lx + 1 - ix
            xcrved = xcrved[:, ::-1]
            ycrved = ycrved[:, ::-1]
        isid1s = FCYC_TO_SYM[isid1 - 1]
        ht = H[(isid - 1) // 4]                         # izt: bottom / top
        if isid1s <= 2:
            hr = H[isid1s - 1]                          # weight along r
            px = np.einsum('i,kj,l->klji', hr, xcrved, ht)
            py = np.einsum('i,kj,l->klji', hr, ycrved, ht)
        else:
            hs = H[isid1s - 3]                          # weight along s
            px = np.einsum('ki,j,l->klji', xcrved, hs, ht)
            py = np.einsum('ki,j,l->klji', ycrved, hs, ht)
        g = _grid(xyz27[r]).copy()
        g[..., 0] += px
        g[..., 1] += py
        xyz27[r] = g.reshape(r.size, 27, 3)
        ndef += int(sel.sum())
    return ndef


def gll_geometry(xyz, curves, pos_of_elid, rows):
    """Neko's GLL-node geometry (3x3x3) for the element records ``rows``
    (sorted, unique record positions): the trilinear map of their corners
    ``xyz[rows]`` with the curve records that reference them applied.
    Returns (xyz27 (len(rows), 27, 3), n_deformed_edges)."""
    corners = np.asarray(xyz[rows])
    x27 = gll_xyz(corners)
    ndef = 0
    if curves.size:
        crow = pos_of_elid[curves['e'].astype(np.int64)]
        loc = np.searchsorted(rows, crow)
        loc[loc >= rows.size] = 0
        hit = rows[loc] == crow
        if hit.any():
            ndef = apply_curves(x27, corners, curves[hit], rows=loc[hit])
    return x27, ndef
