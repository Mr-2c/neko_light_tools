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
"""Every binary layout and topology table used by the Neko light tools.

This module is the single home for the format knowledge shared by all the
tools; nothing else in the package hard-codes a byte offset or a
vertex-ordering table.

The Neko binary mesh (`.nmsh`, little-endian, no record markers)::

    header    2 x int32                      nelv, gdim
    element   int32 + 8 x (int32 + 3 x f64)  el_idx, 8 x (v_idx, x, y, z)   228 B
              (gdim = 2: int32 + 4 x (...)   quad records, 116 B)
    (nzones)  int32
    zone      9 x int32                       36 B
              e, f, p_e, p_f, glb_pt_ids(4), type   (5 = periodic, 7 = labelled)
    (ncurves) int32
    curve     int32 + 60 x f64 + 12 x int32  el, curve_data(5,12), type(12) 532 B

Real Neko meshes may carry trailing bytes past the curve section (an MPI-IO
no-truncate artifact); Neko's reader ignores them and so do we (the checker
reports them).

A ``gdim = 2`` file stores quads.  Neko has no native 2D solver: its reader
(``nmsh_file_read_2d``) extrudes every quad into one hex layer, z = 0 at the
bottom and z = 1 at the top, gives the top copy of point id ``p`` the id
``p + 8*nelv``, and makes facets 5/6 of every element periodic to each other.
:func:`extrude_2d` reproduces exactly that slab so the 3D topology code gives
the numbers Neko itself reports for a 2D mesh.

The NEKTON re2 mesh (little-endian)::

    header    80 ASCII chars: '#v001'..'#v004' + counts (fixed widths below)
    endian    f32 = 6.54321  (byte-swapped files are rejected)
    element   v2+: f64 group + 8 x f64 x/y/z          200 B   (v1: f32, 100 B)
              2D:  f64 group + 4 x f64 x/y             72 B   (v1: f32,  36 B)
    (ncurve)  v2+: f64  (v1: int32)
    curve     v2+: 2 x f64 + 5 x f64 + char(8)         64 B   (v1: 32 B)
    (nbc)     v2+: f64  (v1: int32)
    bc        v2+: 2 x f64 + 5 x f64 + char(8)         64 B   (v1: 32 B)

Topology tables (all validated byte-exact against Neko's own writers by the
golden tests): ``FACE_RE2`` gives the 4 corners of Neko facet 1..6 as nmsh
vertex slots -- Neko's ``face_nodes`` composed with the nmsh->mesh read swap
[1,2,4,3,5,6,8,7], which nets to identity on the file layout.  ``EDGE_RE2``
likewise for the 12 hex edges, ``QFACE_RE2`` gives the 2 corners of quad
facet 1..4 (Neko's quad ``edge_nodes`` composed with the [1,2,4,3] swap),
``FACET_MAP`` maps an re2 face to Neko's symmetric facet, and ``SIJK`` maps an
nmsh vertex slot to a corner of the reference cube.
"""

import os
import sys
import tempfile
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# nmsh record dtypes
# ---------------------------------------------------------------------------
EL_DT = np.dtype([('id', '<i4'),
                  ('v', [('idx', '<i4'), ('xyz', '<f8', (3,))], (8,))])
QUAD_DT = np.dtype([('id', '<i4'),
                    ('v', [('idx', '<i4'), ('xyz', '<f8', (3,))], (4,))])
ZONE_DT = np.dtype([('e', '<i4'), ('f', '<i4'), ('p_e', '<i4'), ('p_f', '<i4'),
                    ('g', '<i4', (4,)), ('t', '<i4')])
CURVE_DT = np.dtype([('e', '<i4'), ('data', '<f8', (12, 5)),
                     ('type', '<i4', (12,))])
assert EL_DT.itemsize == 228 and QUAD_DT.itemsize == 116 \
    and ZONE_DT.itemsize == 36 and CURVE_DT.itemsize == 532


def elem_dtype(gdim):
    """The element record dtype of a gdim-dimensional .nmsh."""
    if gdim == 3:
        return EL_DT
    if gdim == 2:
        return QUAD_DT
    sys.exit('Error: unsupported mesh dimension gdim=%d (valid: 2, 3)' % gdim)

# ---------------------------------------------------------------------------
# re2 record dtypes (v2+ = double precision body, v1 = single precision)
# ---------------------------------------------------------------------------
RE2_EL_DT = {True: np.dtype([('rg', '<f8'), ('x', '<f8', (8,)),
                             ('y', '<f8', (8,)), ('z', '<f8', (8,))]),
             False: np.dtype([('rg', '<f4'), ('x', '<f4', (8,)),
                              ('y', '<f4', (8,)), ('z', '<f4', (8,))])}
RE2_EL2D_DT = {True: np.dtype([('rg', '<f8'), ('x', '<f8', (4,)),
                               ('y', '<f8', (4,))]),
               False: np.dtype([('rg', '<f4'), ('x', '<f4', (4,)),
                                ('y', '<f4', (4,))])}
RE2_CURVE_DT = {True: np.dtype([('e', '<f8'), ('edge', '<f8'),
                                ('d', '<f8', (5,)), ('t', 'S8')]),
                False: np.dtype([('e', '<i4'), ('edge', '<i4'),
                                 ('d', '<f4', (5,)), ('t', 'S4')])}
RE2_BC_DT = {True: np.dtype([('e', '<f8'), ('f', '<f8'),
                             ('d', '<f8', (5,)), ('t', 'S8')]),
             False: np.dtype([('e', '<i4'), ('f', '<i4'),
                              ('d', '<f4', (5,)), ('t', 'S4')])}
RE2_ENDIAN_TEST = np.float32(6.54321)

# ---------------------------------------------------------------------------
# Topology tables (0-based numpy versions of the validated Fortran tables)
# ---------------------------------------------------------------------------
# corners of Neko facet 1..6 as nmsh vertex slots
FACE_RE2 = np.array([[1, 5, 8, 4], [2, 6, 7, 3], [1, 2, 6, 5],
                     [4, 3, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]],
                    dtype=np.int64) - 1
# the 12 hex edges as nmsh vertex slot pairs
EDGE_RE2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 4], [2, 3],
                     [5, 8], [6, 7], [1, 5], [2, 6], [4, 8], [3, 7]],
                    dtype=np.int64) - 1
# corners of Neko quad facet 1..4 as nmsh (quad) vertex slots: quad
# edge_nodes [[1,3],[2,4],[1,2],[3,4]] composed with the [1,2,4,3] read swap
QFACE_RE2 = np.array([[1, 4], [2, 3], [1, 2], [4, 3]], dtype=np.int64) - 1
# re2 face (1..6) -> Neko symmetric facet (1..6); index with [face - 1]
# (a 2D re2 uses faces 1..4 of the same table)
FACET_MAP = np.array([3, 2, 4, 1, 5, 6], dtype=np.int64)
# nmsh vertex slot (0..7) -> (ix, iy, iz) corner of the reference cube
SIJK = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
# labels outside [1, NEKO_MSH_MAX_ZLBLS] are rejected, exactly as Neko does
MAX_ZLBLS = 20

BANNER = r"""
     _  __  ____  __ __  ____
    / |/ / / __/ / //_/ / __ \
   /    / / _/  / ,<   / /_/ /
  /_/|_/ /___/ /_/|_|  \____/   %s
"""


def banner(name):
    """The NEKO banner with a tool name attached."""
    return BANNER % name


class Mesh(NamedTuple):
    """A whole .nmsh in memory: the raw structured record arrays.

    ``elems['id']`` are the global element ids, ``elems['v']['idx']`` the
    (nelv, nv) vertex ids and ``elems['v']['xyz']`` the (nelv, nv, 3) corner
    coordinates, with nv = 8 for a 3D (hex) file and 4 for a 2D (quad) file.
    Zones and curves are kept in file layout so a write is a plain
    ``tofile``.  ``elems`` may be a read-only ``np.memmap`` (see
    :func:`read_nmsh`).
    """
    nelv: int
    elems: np.ndarray     # EL_DT / QUAD_DT (nelv,)
    zones: np.ndarray     # ZONE_DT
    curves: np.ndarray    # CURVE_DT
    trailing: int = 0     # bytes past the curve section (MPI-IO artifact)
    gdim: int = 3

    @property
    def nv(self):
        """Vertices per element (8 for hexes, 4 for quads)."""
        return 8 if self.gdim == 3 else 4

    @property
    def nfacets(self):
        """Facets per element (6 for hexes, 4 for quads)."""
        return 2 * self.gdim


# ---------------------------------------------------------------------------
# Atomic output writing (never clobber inputs, never leave partial files)
# ---------------------------------------------------------------------------
class atomic_output:
    """Write to a temporary file and rename it into place only on success.

    Guards against ``out == in`` (which would truncate the input) and
    guarantees no partial output survives an error -- any exception unlinks
    the temporary file and the destination is never touched.
    """

    def __init__(self, path, inputs=()):
        rp = os.path.realpath(path)
        for src in inputs:
            if os.path.exists(src) and os.path.realpath(src) == rp:
                sys.exit('Error: output %s would overwrite an input file'
                         % path)
        self.path = path
        d = os.path.dirname(rp) or '.'
        fd, self.tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                                        suffix='.tmp', dir=d)
        # mkstemp creates 0600; give the file the ordinary umask-based mode
        umask = os.umask(0)
        os.umask(umask)
        try:
            os.chmod(self.tmp, 0o666 & ~umask)
        except OSError:
            pass
        self.f = os.fdopen(fd, 'wb')

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc, tb):
        self.f.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            try:
                os.unlink(self.tmp)
            except OSError:
                pass
        return False


# ---------------------------------------------------------------------------
# nmsh read / write
# ---------------------------------------------------------------------------
def _count(f, what, path, itemsize=None, fsize=None):
    a = np.fromfile(f, dtype='<i4', count=1)
    if a.size != 1:
        sys.exit('Error: %s: truncated file (missing %s count)' % (path, what))
    n = int(a[0])
    if n < 0:
        sys.exit('Error: %s: negative %s count (%d) -- corrupt file?'
                 % (path, what, n))
    if itemsize is not None and fsize is not None \
       and n * itemsize > fsize - f.tell():
        sys.exit('Error: %s: %s count %d exceeds what the file can hold '
                 '(%d bytes left) -- corrupt or mis-framed file?'
                 % (path, what, n, fsize - f.tell()))
    return n


def read_nmsh(path, mmap=False):
    """Read a whole .nmsh into a :class:`Mesh` (validating as it goes).

    With ``mmap=True`` the element section is returned as a read-only
    ``np.memmap`` instead of being loaded: the operating system pages it in
    on demand, so a tool that only needs a few fields per element (the
    geometric partitioners, streaming reductions) never holds the full
    228 B/element in RAM.  Every fancy-indexing operation on it still
    materialises its result, so consumers gather in chunks.
    """
    try:
        f = open(path, 'rb')
    except OSError as ex:
        sys.exit('Error: cannot open %s (%s)' % (path, ex.strerror))
    with f:
        hdr = np.fromfile(f, dtype='<i4', count=2)
        if hdr.size != 2:
            sys.exit('Error: %s is not a Neko .nmsh file (short header)'
                     % path)
        nelv, gdim = int(hdr[0]), int(hdr[1])
        if gdim not in (2, 3):
            sys.exit('Error: %s: gdim=%d is not a valid Neko mesh dimension '
                     '(2 = quads, 3 = hexes)' % (path, gdim))
        if nelv < 1:
            sys.exit('Error: %s: non-positive element count (%d)'
                     % (path, nelv))
        dt = elem_dtype(gdim)
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        el_end = 8 + nelv * dt.itemsize
        if fsize < el_end:
            sys.exit('Error: %s: truncated element section (%d of %d records)'
                     % (path, max(0, (fsize - 8) // dt.itemsize), nelv))
        if mmap:
            elems = np.memmap(path, dtype=dt, mode='r', offset=8,
                              shape=(nelv,))
        else:
            f.seek(8)
            elems = np.fromfile(f, dtype=dt, count=nelv)
        # every vertex id must be a positive int32 (Neko: "Invalid point id")
        for s in range(0, nelv, 1 << 20):
            if (np.asarray(elems['v']['idx'][s:s + (1 << 20)]) < 1).any():
                sys.exit('Error: %s: element record with a vertex id < 1 '
                         '(Neko refuses this: "Invalid point id")' % path)
        f.seek(el_end)
        nz = _count(f, 'zone', path, ZONE_DT.itemsize, fsize)
        zones = np.fromfile(f, dtype=ZONE_DT, count=nz)
        if zones.size != nz:
            sys.exit('Error: %s: truncated zone section (%d of %d records)'
                     % (path, zones.size, nz))
        nc = _count(f, 'curve', path, CURVE_DT.itemsize, fsize)
        curves = np.fromfile(f, dtype=CURVE_DT, count=nc)
        if curves.size != nc:
            sys.exit('Error: %s: truncated curve section (%d of %d records)'
                     % (path, curves.size, nc))
        trailing = fsize - f.tell()
    return Mesh(nelv, elems, zones, curves, trailing, gdim)


def extrude_2d(mesh):
    """The thin hex slab Neko's reader builds from a 2D (quad) mesh.

    Exactly ``nmsh_file_read_2d``: quad slot k becomes hex slot k at z = 0
    and hex slot k+4 at z = 1 with point id ``idx + 8*nelv``; periodic zone
    records get their two stored point ids extruded to four
    (``nmsh_file_extrude_periodic_ids``); labelled records are unchanged;
    curve records get edges 1-4 mirrored onto 5-8 with the midside z set to
    the slab depth (``nmsh_file_read_curves`` with ``extrusion_depth``).

    The reader then also makes facets 5 and 6 of every element periodic to
    each other and applies that AFTER the file's zone records, so every top
    point ends up with the (merged) id of the bottom point below it.  That
    step is a plain slice copy and is left to the caller
    (:func:`nekolight.topology.merged_vertex_ids` does it), because it must
    come after the zone merge.
    """
    if mesh.gdim != 2:
        return mesh
    n = mesh.nelv
    q = mesh.elems
    if 8 * n + int(np.asarray(q['v']['idx']).max()) > np.iinfo(np.int32).max:
        sys.exit('Error: 2D mesh too large to extrude: the top-layer ids '
                 'idx + 8*nelv overflow int32 (Neko aborts with "Invalid '
                 'point id" on this mesh)')
    h = np.empty(n, dtype=EL_DT)
    h['id'] = q['id']
    h['v']['idx'][:, :4] = q['v']['idx']
    h['v']['idx'][:, 4:] = q['v']['idx'].astype(np.int64) + 8 * n
    h['v']['xyz'][:, :4, :] = q['v']['xyz']
    h['v']['xyz'][:, :4, 2] = 0.0
    h['v']['xyz'][:, 4:, :] = q['v']['xyz']
    h['v']['xyz'][:, 4:, 2] = 1.0
    zones = mesh.zones.copy()
    z5 = zones['t'] == 5
    if z5.any():
        g = zones['g'][z5].astype(np.int64)
        f = zones['f'][z5]
        pt1, pt2 = g[:, 0], g[:, 1]
        pt3, pt4 = pt1 + 8 * n, pt2 + 8 * n
        ext = np.where((f[:, None] == 1) | (f[:, None] == 2),
                       np.stack([pt1, pt3, pt4, pt2], axis=1),
                       np.stack([pt1, pt2, pt4, pt3], axis=1))
        zones['g'][z5] = ext.astype(np.int32)
    curves = mesh.curves.copy()
    if curves.size:
        curves['data'][:, 4:8, :] = curves['data'][:, 0:4, :]
        curves['type'][:, 4:8] = curves['type'][:, 0:4]
        top_mid = curves['type'][:, 4:8] == 4
        d = curves['data'][:, 4:8, 2]
        d[top_mid] = 1.0
        curves['data'][:, 4:8, 2] = d
    return Mesh(n, h, zones, curves, mesh.trailing, 3)


def iter_nmsh_elements(path, chunk=1 << 21):
    """Yield (start, elems_chunk) over the element section of a .nmsh.

    The low-memory alternative to :func:`read_nmsh` for reductions that never
    need the whole mesh at once (bounding boxes, Jacobian scans, ...).
    """
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype='<i4', count=2)
        if hdr.size != 2 or int(hdr[1]) not in (2, 3):
            sys.exit('Error: %s is not a Neko .nmsh file' % path)
        nelv = int(hdr[0])
        dt = elem_dtype(int(hdr[1]))
        done = 0
        while done < nelv:
            n = min(chunk, nelv - done)
            e = np.fromfile(f, dtype=dt, count=n)
            if e.size != n:
                sys.exit('Error: %s: truncated element section '
                         '(%d of %d records)' % (path, done + e.size, nelv))
            yield done, e
            done += n


def write_nmsh(path, elems, zone_arrays, curves, inputs=()):
    """Write a .nmsh atomically.  ``zone_arrays`` is a sequence of ZONE_DT
    arrays written in order (periodic first, then labelled, then any legacy
    types -- the order Neko's own writer uses).  The header dimension follows
    the element dtype (EL_DT -> 3, QUAD_DT -> 2)."""
    gdim = 3 if elems.dtype == EL_DT else 2
    if elems.dtype not in (EL_DT, QUAD_DT):
        raise TypeError('write_nmsh: elems must be EL_DT or QUAD_DT records')
    nz = sum(int(z.shape[0]) for z in zone_arrays)
    with atomic_output(path, inputs) as f:
        np.array([elems.shape[0], gdim], dtype='<i4').tofile(f)
        np.ascontiguousarray(elems).tofile(f)
        np.array([nz], dtype='<i4').tofile(f)
        for z in zone_arrays:
            z.tofile(f)
        np.array([curves.shape[0]], dtype='<i4').tofile(f)
        curves.tofile(f)


def validate_zones(nelv, zones, path='', gdim=3):
    """Refuse malformed zone records instead of silently repairing them.

    Checks: plausible type, element/facet ranges (facets 1..2*gdim),
    periodic partner ranges and labelled-zone label range.  Any violation is
    a hard error -- no tool in this package drops or rewrites a bad record
    while exiting 0.
    """
    where = (' in %s' % path) if path else ''
    if zones.size == 0:
        return
    nf = 2 * gdim
    t = zones['t']
    if ((t < 1) | (t > 7)).any():
        sys.exit('Error: zone record with implausible type (valid: 1..7)%s '
                 '-- corrupt or mis-framed zone section?' % where)
    bad = (zones['e'] < 1) | (zones['e'] > nelv) \
        | (zones['f'] < 1) | (zones['f'] > nf)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        sys.exit('Error: zone record %d references element %d facet %d, '
                 'outside [1,%d]x[1,%d]%s'
                 % (i + 1, int(zones['e'][i]), int(zones['f'][i]), nelv, nf,
                    where))
    z5 = zones[t == 5]
    if z5.size:
        bad = (z5['p_e'] < 1) | (z5['p_e'] > nelv) \
            | (z5['p_f'] < 1) | (z5['p_f'] > nf)
        if bad.any():
            sys.exit('Error: periodic zone record references partner '
                     'element/facet out of range%s' % where)
        nc = 4 if gdim == 3 else 2
        if (z5['g'][:, :nc] < 1).any():
            sys.exit('Error: periodic zone record with a point id < 1 in '
                     'glb_pt_ids (Neko refuses this: "Invalid point id")%s'
                     % where)
    z7 = zones[t == 7]
    if z7.size:
        lbl = z7['p_f']       # the label lives in the p_f field
        if ((lbl < 1) | (lbl > MAX_ZLBLS)).any():
            sys.exit('Error: labelled zone with label outside [1,%d]%s'
                     % (MAX_ZLBLS, where))


def validate_curves(nelv, curves, path='', gdim=3, log=None):
    """Refuse corrupt curve records; warn about ones Neko reads but ignores.

    Element references must be in range.  Edge types: 0 (none), 3 (circular
    arc, 'C') and 4 (midside point, 'm') are what Neko's geometry generation
    uses; 1 and 2 ('s'/'e') are accepted by Neko's reader and then ignored by
    dofmap, so they are carried through with a note rather than refused; any
    other value cannot come from a Neko writer and is treated as corruption.
    A 2D mesh may not curve edges 5..8 (Neko's mark_curve_element refuses
    it).  Returns the number of edges with unsupported types 1/2.
    """
    where = (' in %s' % path) if path else ''
    if curves.size == 0:
        return 0
    if ((curves['e'] < 1) | (curves['e'] > nelv)).any():
        sys.exit('Error: curve record references element outside [1,%d]%s'
                 % (nelv, where))
    ct = curves['type']
    if ((ct < 0) | (ct > 4)).any():
        sys.exit('Error: curve record with impossible edge type (valid: 0, '
                 '1/2 = unsupported s/e, 3 = circle, 4 = midside)%s -- '
                 'corrupt or mis-framed curve section?' % where)
    if gdim == 2 and (ct[:, 4:8] != 0).any():
        sys.exit('Error: 2D mesh with curved edges 5..8 (Neko refuses '
                 'this: "Invalid curve element")%s' % where)
    n12 = int(((ct == 1) | (ct == 2)).sum())
    if n12 and log is not None:
        log('        note: %d curved edge(s) of type 1/2 (re2 \'s\'/\'e\'); '
            'Neko reads them but generates no geometry for them' % n12)
    return n12


# ---------------------------------------------------------------------------
# re2 read
# ---------------------------------------------------------------------------
class Re2(NamedTuple):
    """A whole .re2 in memory: raw coordinates + curve/BC records."""
    nelv: int
    version: str
    xyz: np.ndarray       # (nelv, nv, 3) f64 corner coordinates (z = 0 in 2D)
    curves: np.ndarray    # RE2_CURVE_DT records (as read)
    bcs: np.ndarray       # RE2_BC_DT records (as read)
    gdim: int = 3


def read_re2(path, chunk=1 << 21):
    """Read a NEKTON .re2 (versions #v001..#v004, little-endian, 2D or 3D).

    2D files hold 4-corner (x, y) records; the corners are returned with
    z = 0 exactly as Neko's reader initialises them (``p%init(x, y, 0.0d0)``),
    so the bit-exact point de-duplication sees the same triples.
    """
    try:
        f = open(path, 'rb')
    except OSError as ex:
        sys.exit('Error: cannot open %s (%s)' % (path, ex.strerror))
    with f:
        hdr = f.read(80)
        if len(hdr) != 80:
            sys.exit('Error: %s is not a .re2 file (short header)' % path)
        ver = hdr[:5].decode('latin-1')
        try:
            if ver == '#v004':
                nel = int(hdr[5:21]); ndim = int(hdr[21:24])
                nelv = int(hdr[24:40])
            elif ver in ('#v001', '#v002', '#v003'):
                nel = int(hdr[5:14]); ndim = int(hdr[14:17])
                nelv = int(hdr[17:26])
            else:
                sys.exit('Error: unknown re2 version %r' % ver)
        except ValueError:
            sys.exit('Error: cannot parse re2 header of %s' % path)
        v2 = ver != '#v001'
        endian = np.fromfile(f, dtype='<f4', count=1)
        if endian.size != 1 or abs(float(endian[0]) - 6.54321) > 1e-4:
            sys.exit('Error: byte-swapped or corrupt re2 (endian tag)')
        if ndim not in (2, 3):
            sys.exit('Error: re2 header dimension %d is not 2 or 3' % ndim)
        if nelv < 1:
            sys.exit('Error: non-positive element count in re2 header')
        del nel
        nv = 8 if ndim == 3 else 4
        rec_dt = RE2_EL_DT[v2] if ndim == 3 else RE2_EL2D_DT[v2]
        fsize = os.fstat(f.fileno()).st_size
        if nelv * rec_dt.itemsize > fsize - f.tell():
            sys.exit('Error: %s: header element count %d exceeds what the '
                     'file can hold -- corrupt header?' % (path, nelv))

        # elements, chunked (v1 is f32 and upcast)
        xyz = np.zeros((nelv, nv, 3), dtype=np.float64)
        done = 0
        while done < nelv:
            n = min(chunk, nelv - done)
            rec = np.fromfile(f, dtype=rec_dt, count=n)
            if rec.size != n:
                sys.exit('Error: truncated or corrupt .re2 file (element '
                         'section, record %d of %d)' % (done + rec.size, nelv))
            xyz[done:done + n, :, 0] = rec['x']
            xyz[done:done + n, :, 1] = rec['y']
            if ndim == 3:
                xyz[done:done + n, :, 2] = rec['z']
            done += n
        if not np.isfinite(xyz).all():
            sys.exit('Error: non-finite coordinate in %s' % path)

        ncurve = _re2_count(f, v2, 'curve', RE2_CURVE_DT[v2].itemsize, fsize)
        curves = np.fromfile(f, dtype=RE2_CURVE_DT[v2], count=ncurve)
        if curves.size != ncurve:
            sys.exit('Error: truncated or corrupt .re2 file (curve section)')
        nbc = _re2_count(f, v2, 'boundary-condition', RE2_BC_DT[v2].itemsize,
                         fsize)
        bcs = np.fromfile(f, dtype=RE2_BC_DT[v2], count=nbc)
        if bcs.size != nbc:
            sys.exit('Error: truncated or corrupt .re2 file (BC section)')

    # bounds validation up front (validate-and-refuse; nothing written yet)
    nf = 2 * ndim
    ce = curves['e'].astype(np.int64)
    cz = curves['edge'].astype(np.int64)
    if curves.size and (((ce < 1) | (ce > nelv)).any()):
        sys.exit('Error: curve record references element out of range')
    if curves.size and (((cz < 1) | (cz > 12)).any()):
        sys.exit('Error: curve record edge index out of [1,12]')
    be = bcs['e'].astype(np.int64)
    bf = bcs['f'].astype(np.int64)
    if bcs.size and ((be < 1) | (be > nelv)).any():
        sys.exit('Error: BC record references element out of range')
    if bcs.size and ((bf < 1) | (bf > nf)).any():
        sys.exit('Error: BC record face out of [1,%d]' % nf)
    return Re2(nelv, ver, xyz, curves, bcs, ndim)


def _re2_count(f, v2, what, itemsize, fsize):
    if v2:
        a = np.fromfile(f, dtype='<f8', count=1)
    else:
        a = np.fromfile(f, dtype='<i4', count=1)
    if a.size != 1:
        sys.exit('Error: truncated or corrupt .re2 file (missing %s count)'
                 % what)
    if not np.isfinite(a[0]):
        sys.exit('Error: corrupt %s count in .re2' % what)
    n = int(a[0])
    if n < 0:
        sys.exit('Error: negative %s count in .re2' % what)
    if n * itemsize > fsize - f.tell():
        sys.exit('Error: %s count %d exceeds what the .re2 file can hold '
                 '-- corrupt or mis-framed file?' % (what, n))
    return n


def bc_type_str(raw):
    """The BC/curve type field as Neko's ``trim(type)`` sees it: trailing
    blanks removed, leading blanks kept.  NUL padding (which no Nek5000
    writer produces, but some third-party ones might) is treated as blank;
    Neko itself would leave the NULs in and match nothing."""
    return raw.decode('latin-1').replace('\x00', ' ').rstrip()


# ---------------------------------------------------------------------------
# Distribution CSV files (genmeshbox), matching Neko's csv reader semantics
# ---------------------------------------------------------------------------
def read_dist_csv(path, n):
    """Read n+1 grid coordinates from a genmeshbox distribution file.

    Exactly Neko's ``csv_file_read_vector``: if the file has more than one
    line the first is treated as a header and skipped; commas, spaces and
    newlines all act as separators.
    """
    try:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
    except OSError as ex:
        sys.exit('Error: cannot open distribution file %s (%s)'
                 % (path, ex.strerror))
    body = lines[1:] if len(lines) > 1 else lines
    toks = ' '.join(body).replace(',', ' ').split()
    if len(toks) < n + 1:
        sys.exit('Error: expected %d grid values in %s (Neko treats the '
                 'first line as a header when the file has >1 line)'
                 % (n + 1, path))
    try:
        return np.array([float(t) for t in toks[:n + 1]], dtype=np.float64)
    except ValueError:
        sys.exit('Error: cannot parse grid values in %s' % path)


# ---------------------------------------------------------------------------
# fld writing (single precision NEKTON/Neko field file, lx=ly=lz=3)
# ---------------------------------------------------------------------------
def write_zone_indices_fld(out_base, elids, gll_xyz, scal):
    """Write ``<out_base>.fld`` (+ ``.nek5000`` companion): the given 3x3x3
    GLL-node geometry (straight-sided or curved) plus one scalar field.

    ``elids`` are the ACTUAL global element ids in record order (a valid
    .nmsh may store its records in any order -- the id list is what maps a
    block to an element), ``gll_xyz`` is (nelv, 27, 3) f64 and ``scal`` is
    (nelv, 27) f32-compatible.
    """
    nel = len(elids)
    hdr = ('#std %1d %2d %2d %2d %10d %10d %20.13E %9d %6d %6d %-10s'
           % (4, 3, 3, 3, nel, nel, 0.0, 1, 1, 1, 'XS01')).ljust(132)
    assert len(hdr) == 132
    gx = gll_xyz.astype(np.float32)
    sc = np.asarray(scal, dtype=np.float32)
    with atomic_output(out_base + '.fld') as f:
        f.write(hdr.encode('ascii'))
        np.float32(6.54321).tofile(f)
        np.asarray(elids, dtype='<i4').tofile(f)
        # geometry block: per element gx(27), gy(27), gz(27)
        np.ascontiguousarray(gx.transpose(0, 2, 1)).tofile(f)
        sc.tofile(f)
        # per-element geometry bounding-box metadata (xmin,xmax,...,zmin,zmax)
        bb = np.empty((nel, 6), dtype=np.float32)
        bb[:, 0::2] = gx.min(axis=1)
        bb[:, 1::2] = gx.max(axis=1)
        bb.tofile(f)
        # per-element (min, max) metadata for the scalar field
        mm = np.stack([sc.min(axis=1), sc.max(axis=1)], axis=1)
        mm.astype(np.float32).tofile(f)
    with atomic_output(out_base + '.nek5000') as f:
        f.write(('filetemplate: %s.fld\nfirsttimestep: 1\nnumtimesteps: 1\n'
                 % os.path.basename(out_base)).encode('ascii'))
