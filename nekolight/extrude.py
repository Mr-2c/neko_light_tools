# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Extrusion of a quad mesh into layers of hexahedra (Nek5000's n2to3).

Layer index varies slowest: 2D element e (1-based) of layer k (0-based)
becomes element ``e + k * nel2d``; quad corners 1-4 become hex corners 1-4
at the lower plane and 5-8 at the upper plane; 2D facets 1-4 become hex
facets 1-4 (Neko numbers both x-, x+, y-, y+); the new facets 5 (bottom)
and 6 (top) are either periodic to each other across the whole stack or
labelled; midside curves of 2D edge k are carried to hex edge k (lower
plane z) and k+4 (upper plane z); vertical edges stay straight.  The
2D quads must be counter-clockwise and the planes ascending, which makes
every hexahedron right-handed.
"""
import sys

import numpy as np

from .formats import EL_DT, CURVE_DT


def layer_planes(z0, z1, nlev, gain=1.0):
    """n2to3's layer distribution: dz_i proportional to gain**(i-1),
    scaled to span [z0, z1]; gain = 1 is uniform."""
    if nlev < 1:
        sys.exit('Error: the number of layers must be at least 1')
    if not z1 > z0:
        sys.exit('Error: z1 must be larger than z0 (got %g, %g)' % (z0, z1))
    if gain <= 0.0:
        sys.exit('Error: --gain must be positive (1 = uniform layers)')
    dz = gain ** np.arange(nlev, dtype=np.float64)
    dz *= (z1 - z0) / dz.sum()
    z = np.concatenate([[z0], z0 + np.cumsum(dz)])
    z[-1] = z1
    return z


def extrude(vid2d, xy2d, zplanes, curves2d=None, elem_ids=None):
    """Hex records for the quad mesh (``vid2d`` (n, 4) dense point ids
    1..npts, ``xy2d`` (n, 4, 2|3)) between the planes ``zplanes`` (nlev+1
    ascending values).  Point id of 2D point p on plane k is
    ``p + k * npts``.  Returns (elems (EL_DT), curves (CURVE_DT), npts2d)."""
    z = np.asarray(zplanes, dtype=np.float64)
    if z.ndim != 1 or z.size < 2 or not (np.diff(z) > 0).all():
        sys.exit('Error: the z planes must be at least two strictly '
                 'ascending values')
    n = vid2d.shape[0]
    nlev = z.size - 1
    npts = int(vid2d.max())
    if (nlev + 1) * npts > np.iinfo(np.int32).max or nlev * n > np.iinfo(np.int32).max:
        sys.exit('Error: the extruded mesh is too large for 32-bit ids')
    el = np.empty(n * nlev, dtype=EL_DT)
    e2 = np.arange(1, n + 1, dtype=np.int64)
    for k in range(nlev):
        blk = el[k * n:(k + 1) * n]
        blk['id'] = (e2 + k * n).astype(np.int32)
        blk['v']['idx'][:, :4] = (vid2d + k * npts).astype(np.int32)
        blk['v']['idx'][:, 4:] = (vid2d + (k + 1) * npts).astype(np.int32)
        blk['v']['xyz'][:, :4, :2] = xy2d[:, :, :2]
        blk['v']['xyz'][:, 4:, :2] = xy2d[:, :, :2]
        blk['v']['xyz'][:, :4, 2] = z[k]
        blk['v']['xyz'][:, 4:, 2] = z[k + 1]
    curves = np.empty(0, dtype=CURVE_DT)
    if curves2d is not None and curves2d.size:
        c2 = curves2d
        m = c2.shape[0]
        curves = np.zeros(m * nlev, dtype=CURVE_DT)
        for k in range(nlev):
            blk = curves[k * m:(k + 1) * m]
            blk['e'] = (c2['e'].astype(np.int64) + k * n).astype(np.int32)
            blk['data'][:, 0:4, :] = c2['data'][:, 0:4, :]
            blk['data'][:, 4:8, :] = c2['data'][:, 0:4, :]
            blk['type'][:, 0:4] = c2['type'][:, 0:4]
            blk['type'][:, 4:8] = c2['type'][:, 0:4]
            mid_lo = blk['type'][:, 0:4] == 4
            mid_hi = blk['type'][:, 4:8] == 4
            d = blk['data']
            lo = d[:, 0:4, 2]; lo[mid_lo] = z[k]; d[:, 0:4, 2] = lo
            hi = d[:, 4:8, 2]; hi[mid_hi] = z[k + 1]; d[:, 4:8, 2] = hi
            blk['data'] = d
    return el, curves, npts
