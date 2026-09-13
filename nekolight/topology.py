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
"""Mesh topology: point de-duplication, the periodic merge, face/edge tables,
external surface (skin) extraction and the element dual graph.

Everything is a plain function over numpy arrays and works for hex meshes
(8 vertices, 6 facets) and quad meshes (4 vertices, 4 facets; a facet is an
edge).  All indices along the nv*nelv corner axis are int64 (8 * 3e8 corners
overflows int32); packed sort keys are uint64 (defined wrap-around).
"""

import sys

import numpy as np

from .formats import FACE_RE2, EDGE_RE2, QFACE_RE2


def facet_table(nv):
    """Corner slots of every facet: (6, 4) for hexes, (4, 2) for quads."""
    if nv == 8:
        return FACE_RE2
    if nv == 4:
        return QFACE_RE2
    raise ValueError('elements must have 8 (hex) or 4 (quad) vertices')


# ---------------------------------------------------------------------------
# Element id maps
# ---------------------------------------------------------------------------
def pos_of_elid_map(nelv, elems):
    """0-based record position of each global element id (1-based index).

    A valid .nmsh may store its element records in any order; the ids must be
    a permutation of 1..nelv.  Anything else is a hard error.
    """
    elids = np.asarray(elems['id']).astype(np.int64)
    if elids.size and (elids.min() < 1 or elids.max() > nelv):
        sys.exit('Error: element id out of [1,%d]' % nelv)
    pos = np.full(nelv + 1, -1, dtype=np.int64)
    pos[elids] = np.arange(nelv)
    if (pos[1:] < 0).any():
        sys.exit('Error: element ids are not a permutation of 1..nelv '
                 '(duplicate or missing id)')
    return pos


# ---------------------------------------------------------------------------
# Point de-duplication (bit-exact, first-appearance order)
# ---------------------------------------------------------------------------
def dedup_points(xyz):
    """Assign each distinct corner coordinate a 1-based id in order of first
    appearance, comparing float64 triples bit-exactly.

    Neko's re2 reader does the same through ``htable_pt_t``: the hash is
    computed from the raw bits of the three coordinates, so two corners only
    ever meet in the table when their bits agree (its equality test,
    ``abscmp``, tolerates a few ulps, but a probe sequence reaching a
    near-equal point stored under a different hash is a load-factor accident
    no writer should rely on).  Bit-exact comparison is therefore what real
    files experience.

    ``xyz`` is (nelv, nv, 3) f64.  Returns (vid (nelv, nv) int32, n_unique).

    Memory note: this views the corners as 24-byte keys and sorts them, so
    the transient cost is roughly 45 bytes per corner (int64 indices
    included).  At 3e8 elements that is a fat-node job; below ~1e7 elements
    it is instant.
    """
    nelv, nv = xyz.shape[0], xyz.shape[1]
    flat = np.ascontiguousarray(xyz).reshape(nelv * nv, 3)
    keys = flat.view([('', 'V24')]).ravel()
    _, first, inverse = np.unique(keys, return_index=True,
                                  return_inverse=True)
    # renumber the (byte-sorted) unique keys to first-appearance order
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(first.size)
    vid = (rank[inverse.ravel()] + 1).reshape(nelv, nv)
    if first.size > np.iinfo(np.int32).max:
        sys.exit('Error: more than 2^31 unique points -- the .nmsh format '
                 'stores 32-bit point ids')
    return vid.astype(np.int32), int(first.size)


# ---------------------------------------------------------------------------
# The periodic merge (Neko's apply_periodic_facet, vectorised)
# ---------------------------------------------------------------------------
def periodic_replace_merge(nelv, vidx, zones, pos_of_elid):
    """Corner k of every periodic facet takes ``glb_pt_ids(k)`` directly, the
    last record in file order winning on conflicts -- exactly what Neko's
    reader does when it calls ``apply_periodic_facet`` for each type-5 zone
    record.  Used by the checker AND the partitioners, so both see the
    connectivity Neko sees.

    ``vidx`` is (nelv, nv) int64 vertex ids; a corner is identified by its
    id, so all elements sharing a point follow the point (Neko's points are
    shared objects).  Returns merged ids (same shape).  Zone records must
    already be validated (see formats.validate_zones).
    """
    z5 = zones[zones['t'] == 5]
    if not z5.size:
        return vidx
    nv = vidx.shape[1]
    ft = facet_table(nv)
    nc = ft.shape[1]
    pos = pos_of_elid[z5['e'].astype(np.int64)]
    slots = ft[z5['f'].astype(np.int64) - 1]                  # (nz5, nc)
    keys = vidx[pos[:, None], slots].ravel()
    vals = z5['g'][:, :nc].astype(np.int64).ravel()
    # last record wins: unique on the reversed sequence keeps the LAST
    # occurrence of each key
    rk, first_rev = np.unique(keys[::-1], return_index=True)
    rv = vals[::-1][first_rev]
    flat = vidx.ravel()
    idx = np.searchsorted(rk, flat)
    idx[idx >= rk.size] = 0
    hit = rk[idx] == flat
    out = flat.copy()
    out[hit] = rv[idx[hit]]
    return out.reshape(vidx.shape)


def create_periodic_ids(pid, vid, coords, pairs, tol, nsweeps=3, strict=True):
    """Neko's ``mesh_create_periodic_ids``, line for line, over a fixed list
    of (el, f, pe, pf) facet pairs (1-based element ids and facets): THREE
    sweeps (Neko's re2 reader and create_periodic_zones both use 3), each
    setting ``pid = min(pid_i, pid_j)`` in place for every corner of facet
    (el, f) that matches a corner of (pe, pf) under the facet-mean
    translation, to ``tol``.  The fixed sweep count and the in-place
    sequential minimum are what make the resulting ids byte-identical to
    Neko's -- do not 'improve' this into a union-find.

    ``pid`` (n_points,) holds the current id of every point object (indexed
    by the raw first-appearance id - 1) and is updated in place; ``vid``
    (nelv, nv) are the raw corner ids, ``coords`` (n_points, 3) the point
    coordinates.  With ``strict`` (hex meshes) anything but exactly one
    match per corner is a hard error, as in Neko; Neko's quad branch does
    no such check, so pass ``strict=False`` for quads -- unmatched corners
    are then counted and returned instead.
    """
    ft = facet_table(vid.shape[1])
    unmatched = 0
    for sweep in range(nsweeps):
        for (el, f, pe, pf) in pairs:
            ii = vid[el - 1, ft[f - 1]].astype(np.int64)         # corner ids
            jj = vid[pe - 1, ft[pf - 1]].astype(np.int64)
            a = coords[ii - 1]
            b = coords[jj - 1]
            L = (a - b).mean(axis=0)
            d = np.linalg.norm(a[:, None, :] - b[None, :, :] - L, axis=2)
            for k in range(ii.size):
                hits = np.flatnonzero(d[k] < tol)
                if hits.size != 1:
                    if strict:
                        sys.exit('Error: periodic facet corner has %d matches '
                                 '(expected 1) between element %d facet %d '
                                 'and element %d facet %d -- malformed '
                                 'periodic pairing' % (hits.size, el, f, pe, pf))
                    if sweep == 0:
                        unmatched += 1
                for j in hits:                                   # all matches
                    j = int(j)
                    m = min(pid[ii[k] - 1], pid[jj[j] - 1])
                    pid[ii[k] - 1] = m
                    pid[jj[j] - 1] = m
    return unmatched


def merged_vertex_ids(mesh, pos_of_elid, extruded=False):
    """The vertex ids Neko works with after reading ``mesh``: the periodic
    merge applied and, for a slab built by ``formats.extrude_2d``
    (``extruded=True``), every top point given the id of the bottom point
    below it -- Neko marks facets 5/6 of each extruded element periodic to
    each other and applies that AFTER the file's own zone records (and, as
    in Neko, only when the file has a zone section at all).
    """
    vidx = np.asarray(mesh.elems['v']['idx']).astype(np.int64)
    merged = periodic_replace_merge(mesh.nelv, vidx, mesh.zones, pos_of_elid)
    if extruded and mesh.zones.size:       # Neko does this inside nzones > 0
        if merged is vidx:
            merged = vidx.copy()
        merged[:, 4:8] = merged[:, 0:4]
    return merged


def compress_ids(vidx):
    """Compress arbitrary vertex ids to dense 0..n-1 (sorted-id order)."""
    _, cell = np.unique(vidx.ravel(), return_inverse=True)
    return cell.reshape(vidx.shape).astype(np.int64)


# ---------------------------------------------------------------------------
# Face / edge tables (packed-key sort; shared by the checker and the skin)
# ---------------------------------------------------------------------------
def _pack_facets(vidx):
    """Canonical key per (element, facet): the sorted corner ids of each
    facet packed into uint64 words.  Hex faces (4 ids) become two words,
    quad edges (2 ids) one word.  Returns (nfacets*nelv, nwords)."""
    ft = facet_table(vidx.shape[1])
    f = vidx[:, ft].reshape(-1, ft.shape[1]).astype(np.uint64)
    f.sort(axis=1)
    if ft.shape[1] == 4:
        keys = np.empty((f.shape[0], 2), dtype=np.uint64)
        keys[:, 0] = (f[:, 0] << np.uint64(32)) | f[:, 1]
        keys[:, 1] = (f[:, 2] << np.uint64(32)) | f[:, 3]
        return keys
    return ((f[:, 0] << np.uint64(32)) | f[:, 1]).reshape(-1, 1)


def face_multiplicity(vidx):
    """For every (element, facet): how many times its canonical facet occurs
    in the whole mesh.  Returns (counts (nelv, nfacets), n_unique_facets)."""
    keys = _pack_facets(vidx)
    nf = facet_table(vidx.shape[1]).shape[0]
    kv = keys.view([('', 'V%d' % (8 * keys.shape[1]))]).ravel()
    _, inverse, counts = np.unique(kv, return_inverse=True,
                                   return_counts=True)
    return counts[inverse.ravel()].reshape(-1, nf), int(counts.size)


def count_edges(vidx):
    """Number of unique edges: the 12 hex edges (sorted 2-tuples) or, for
    quads, the 4 sides (which are the facets)."""
    if vidx.shape[1] == 4:
        return face_multiplicity(vidx)[1]
    e = vidx[:, EDGE_RE2].reshape(-1, 2).astype(np.uint64)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    return int(np.unique((lo << np.uint64(32)) | hi).size)


def skin(vidx):
    """External surface: the (element, facet) pairs whose facet occurs
    exactly once.  Returns (elem_pos (m,), facet0 (m,)) int64 arrays.

    Use RAW (unmerged) vertex ids for viewing -- periodic boundaries are then
    part of the skin, which is what you want to look at; use merged ids to
    reproduce the checker's topological externality.
    """
    mult, _ = face_multiplicity(vidx)
    epos, fct = np.nonzero(mult == 1)
    return epos.astype(np.int64), fct.astype(np.int64)


# ---------------------------------------------------------------------------
# Element dual graph
# ---------------------------------------------------------------------------
def dual_graph(cell):
    """Weighted element dual graph A = E.E^T with the diagonal removed:
    A[i,j] = number of shared (merged) vertices between elements i and j.
    ``cell`` is (nelv, nv) dense 0-based ids.  Needs scipy."""
    import scipy.sparse as sp
    nelv, nv = cell.shape
    npts = int(cell.max()) + 1
    rows = np.repeat(np.arange(nelv, dtype=np.int64), nv)
    E = sp.csr_matrix((np.ones(nelv * nv), (rows, cell.ravel())),
                      shape=(nelv, npts))
    A = (E @ E.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.sum_duplicates()
    return A
