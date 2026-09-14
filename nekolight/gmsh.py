# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Reader for Gmsh ``.msh`` files (format 2.2 and 4.1, ASCII and binary).

Only what a mesh converter needs is read: the node coordinates, the
elements with their node tags grouped by Gmsh element type, the physical
group of every element (from the element tags in 2.2, from the entity
block in 4.1), the physical names, and the ``$Periodic`` node
correspondences.  Everything else ($NodeData, $PartitionedEntities, ...)
is skipped.  Numbers follow Gmsh's documented layout
(https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format).
"""
import re
import sys
from typing import NamedTuple

import numpy as np

# Gmsh element type number -> (name, number of nodes, dimension)
GMSH_TYPES = {
    1: ('line2', 2, 1), 2: ('tri3', 3, 2), 3: ('quad4', 4, 2),
    4: ('tet4', 4, 3), 5: ('hex8', 8, 3), 6: ('prism6', 6, 3),
    7: ('pyr5', 5, 3), 8: ('line3', 3, 1), 9: ('tri6', 6, 2),
    10: ('quad9', 9, 2), 11: ('tet10', 10, 3), 12: ('hex27', 27, 3),
    13: ('prism18', 18, 3), 14: ('pyr14', 14, 3), 15: ('point', 1, 0),
    16: ('quad8', 8, 2), 17: ('hex20', 20, 3), 18: ('prism15', 15, 3),
    19: ('pyr13', 13, 3), 20: ('tri9', 9, 2), 21: ('tri10', 10, 2),
    26: ('line4', 4, 1), 27: ('line5', 5, 1), 36: ('quad16', 16, 2),
    37: ('quad25', 25, 2), 92: ('hex64', 64, 3), 93: ('hex125', 125, 3),
}


class ElementBlock(NamedTuple):
    etype: int            # Gmsh element type number
    dim: int              # entity dimension
    entity: int           # elementary entity tag
    physical: np.ndarray  # (n,) physical tag per element (0 = none)
    tags: np.ndarray      # (n,) element tags
    nodes: np.ndarray     # (n, nnodes) node tags


class PeriodicLink(NamedTuple):
    dim: int
    entity: int           # slave entity tag
    master: int           # master entity tag
    affine: np.ndarray    # (16,) or empty
    node_map: np.ndarray  # (m, 2): slave node tag, master node tag


class GmshMesh(NamedTuple):
    version: str
    binary: bool
    node_tags: np.ndarray       # (nn,)
    xyz: np.ndarray             # (nn, 3)
    blocks: list                # [ElementBlock]
    physical_names: dict        # (dim, tag) -> name
    entity_physical: dict       # (dim, entity tag) -> [physical tags]
    periodic: list              # [PeriodicLink]

    def node_index(self):
        """Lookup object: ``rows(tags)`` gives the row of xyz for each node
        tag (-1 for unknown tags).  Sorted search, so sparse or offset tags
        cost nothing extra."""
        return TagLookup(self.node_tags, np.arange(self.node_tags.size))


class TagLookup:
    """Map integer tags to values by sorted search (memory O(n), not
    O(max tag)); unknown tags map to ``missing``."""

    def __init__(self, tags, values, missing=-1):
        order = np.argsort(tags, kind='stable')
        self.tags = np.asarray(tags, dtype=np.int64)[order]
        self.values = np.asarray(values)[order]
        self.missing = missing
        if self.tags.size and (np.diff(self.tags) == 0).any():
            raise ValueError('duplicate tags')

    def __call__(self, q):
        return self.rows(q)

    def rows(self, q):
        q = np.asarray(q, dtype=np.int64)
        pos = np.searchsorted(self.tags, q.ravel())
        pos[pos >= self.tags.size] = 0
        hit = self.tags[pos] == q.ravel() if self.tags.size else np.zeros(q.size, bool)
        out = np.where(hit, self.values[pos], self.missing)
        return out.reshape(q.shape)

    @property
    def size(self):
        return self.tags.size


class _Cursor:
    """Byte cursor over the file with helpers for the mixed ASCII / binary
    layout of .msh files."""

    def __init__(self, data, path):
        self.d = data
        self.p = 0
        self.path = path

    def line(self):
        e = self.d.find(b'\n', self.p)
        if e < 0:
            e = len(self.d)
        s = self.d[self.p:e]
        self.p = e + 1
        return s.strip()

    def peek_line(self):
        e = self.d.find(b'\n', self.p)
        return self.d[self.p:(e if e >= 0 else len(self.d))].strip()

    def skip_ws(self):
        while self.p < len(self.d) and self.d[self.p] in b' \t\r\n':
            self.p += 1

    def find_section_end(self, name):
        e = self.d.find(b'$End' + name, self.p)
        if e < 0:
            sys.exit('Error: %s: missing $End%s' % (self.path, name.decode()))
        return e

    def ascii_block(self, name):
        """The ASCII text between the cursor and $End<name>; advances past it."""
        e = self.find_section_end(name)
        txt = self.d[self.p:e]
        self.p = e
        self.line()                 # consume the $End line
        return txt

    def bin(self, dtype, count):
        n = int(count)
        dt = np.dtype(dtype)
        nbytes = n * dt.itemsize
        if self.p + nbytes > len(self.d):
            sys.exit('Error: %s: truncated binary section' % self.path)
        a = np.frombuffer(self.d, dtype=dt, count=n, offset=self.p)
        self.p += nbytes
        return a

    def end_section(self, name):
        self.skip_ws()
        ln = self.line()
        if ln != b'$End' + name:
            sys.exit('Error: %s: expected $End%s, found %r'
                     % (self.path, name.decode(), ln[:40]))


def _ints(txt):
    return np.array(txt.split(), dtype=np.int64) if txt.strip() else \
        np.zeros(0, dtype=np.int64)


def read_msh(path):
    """Read a Gmsh .msh file (2.2 / 4.1, ASCII or binary)."""
    with open(path, 'rb') as f:
        data = f.read()
    c = _Cursor(data, path)
    if c.line() != b'$MeshFormat':
        sys.exit('Error: %s is not a Gmsh .msh file ($MeshFormat missing)'
                 % path)
    parts = c.line().split()
    if len(parts) != 3:
        sys.exit('Error: %s: malformed $MeshFormat line' % path)
    version, ftype, dsize = parts[0].decode(), int(parts[1]), int(parts[2])
    binary = ftype == 1
    if dsize != 8:
        sys.exit('Error: %s: data size %d not supported (expected 8)'
                 % (path, dsize))
    if binary:
        one = c.bin('<i4', 1)[0]
        if one != 1:
            sys.exit('Error: %s: binary .msh with a non-native byte order is '
                     'not supported' % path)
    c.end_section(b'MeshFormat')
    major = version.split('.')[0]
    if version.startswith('2.'):
        reader = _read_v2
    elif version in ('4.1',):
        reader = _read_v41
    else:
        sys.exit('Error: %s: .msh format version %s is not supported (use '
                 'Gmsh format 2.2 or 4.1, e.g. "gmsh -format msh41" or '
                 'Mesh.MshFileVersion)' % (path, version))
    return reader(c, version, binary, path)


def _physical_names(c):
    txt = c.ascii_block(b'PhysicalNames').decode('utf-8', 'replace')
    lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    n = int(lines[0])
    names = {}
    for ln in lines[1:1 + n]:
        m = re.match(r'\s*(\d+)\s+(-?\d+)\s+"(.*)"\s*$', ln)
        if not m:
            sys.exit('Error: %s: malformed $PhysicalNames line %r'
                     % (c.path, ln))
        names[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    return names


# ---------------------------------------------------------------------------
# format 2.2
# ---------------------------------------------------------------------------
def _read_v2(c, version, binary, path):
    names, node_tags, xyz, blocks, periodic = {}, None, None, [], []
    while c.p < len(c.d):
        c.skip_ws()
        if c.p >= len(c.d):
            break
        sec = c.line()
        if not sec.startswith(b'$'):
            sys.exit('Error: %s: unexpected text %r where a section was '
                     'expected' % (path, sec[:40]))
        name = sec[1:]
        if name == b'PhysicalNames':
            names = _physical_names(c)
        elif name == b'Nodes':
            n = int(c.line())
            if binary:
                rec = c.bin(np.dtype([('t', '<i4'), ('x', '<f8', (3,))]), n)
                node_tags = rec['t'].astype(np.int64)
                xyz = rec['x'].astype(np.float64)
                c.end_section(name)
            else:
                txt = c.ascii_block(name)
                a = np.array(txt.split(), dtype=np.float64).reshape(n, 4)
                node_tags = a[:, 0].astype(np.int64)
                xyz = a[:, 1:4].copy()
        elif name == b'ParametricNodes':
            # Mesh.SaveParametric with format 2.2: tag x y z dim entity [u [v]]
            n = int(c.line())
            if binary:
                sys.exit('Error: %s: binary $ParametricNodes (Mesh.SaveParametric '
                         'with format 2.2) is not supported; save without '
                         'parametric coordinates or use format 4.1' % path)
            txt = c.ascii_block(name)
            rows = [r.split() for r in txt.split(b'\n') if r.strip()]
            if len(rows) != n:
                sys.exit('Error: %s: $ParametricNodes announces %d nodes, found '
                         '%d lines' % (path, n, len(rows)))
            node_tags = np.array([r[0] for r in rows], dtype=np.int64)
            xyz = np.array([r[1:4] for r in rows], dtype=np.float64)
        elif name == b'Elements':
            n = int(c.line())
            if binary:
                done = 0
                while done < n:
                    etype, nel, ntags = (int(v) for v in c.bin('<i4', 3))
                    nn = _nnodes(etype, path)
                    rec = c.bin('<i4', nel * (1 + ntags + nn)).reshape(
                        nel, 1 + ntags + nn).astype(np.int64)
                    phys = rec[:, 1] if ntags >= 1 else np.zeros(nel, np.int64)
                    ent = rec[:, 2] if ntags >= 2 else np.full(nel, -1, np.int64)
                    _add_v2_blocks(blocks, etype, phys, ent, rec[:, 0],
                                   rec[:, 1 + ntags:])
                    done += nel
                c.end_section(name)
            else:
                txt = c.ascii_block(name)
                rows = txt.split(b'\n')
                rows = [r for r in rows if r.strip()]
                if len(rows) != n:
                    sys.exit('Error: %s: $Elements announces %d elements, '
                             'found %d lines' % (path, n, len(rows)))
                # group by (type, ntags) so each group parses as one array
                head = np.array([r.split(None, 3)[:3] for r in rows],
                                dtype=np.int64)
                etypes, ntagss = head[:, 1], head[:, 2]
                for key in np.unique(np.stack([etypes, ntagss], 1), axis=0):
                    sel = np.flatnonzero((etypes == key[0]) & (ntagss == key[1]))
                    etype, ntags = int(key[0]), int(key[1])
                    nn = _nnodes(etype, path)
                    a = np.array(b' '.join(rows[i] for i in sel).split(),
                                 dtype=np.int64)
                    if a.size != sel.size * (3 + ntags + nn):
                        sys.exit('Error: %s: element type %d records have an '
                                 'unexpected number of entries' % (path, etype))
                    a = a.reshape(sel.size, 3 + ntags + nn)
                    phys = a[:, 3] if ntags >= 1 else np.zeros(sel.size, np.int64)
                    ent = a[:, 4] if ntags >= 2 else np.full(sel.size, -1, np.int64)
                    _add_v2_blocks(blocks, etype, phys, ent, a[:, 0],
                                   a[:, 3 + ntags:])
        elif name == b'Periodic':
            # Gmsh 2.x writes $Periodic as text even in binary files
            periodic = _read_periodic(c, False, version, path)
        else:
            # any other section: skip to its end marker
            c.ascii_block(name) if not binary else _skip_binary_section(c, name)
    if node_tags is None or not blocks:
        sys.exit('Error: %s: no $Nodes / $Elements section found' % path)
    # 2.2 has no entity section: the physical tag is per element and never
    # ambiguous, so the entity -> physical map is only informative (entity
    # -1 = elements without an elementary tag)
    ent_phys = {}
    for b in blocks:
        for p in np.unique(b.physical):
            if p and b.entity >= 0:
                ent_phys.setdefault((b.dim, b.entity), [])
                if p not in ent_phys[(b.dim, b.entity)]:
                    ent_phys[(b.dim, b.entity)].append(int(p))
    return GmshMesh(version, binary, node_tags, xyz, blocks, names, ent_phys,
                    periodic)


def _skip_binary_section(c, name):
    e = c.find_section_end(name)
    c.p = e
    c.line()


def _nnodes(etype, path):
    if etype not in GMSH_TYPES:
        sys.exit('Error: %s: Gmsh element type %d is not known to this '
                 'reader' % (path, etype))
    return GMSH_TYPES[etype][1]


def _add_v2_blocks(blocks, etype, phys, ent, tags, nodes):
    """2.2 has no entity blocks; group by (entity, physical) so the block
    structure matches 4.1."""
    dim = GMSH_TYPES[etype][2]
    keys = np.stack([ent, phys], axis=1)
    for key in np.unique(keys, axis=0):
        sel = np.flatnonzero((ent == key[0]) & (phys == key[1]))
        blocks.append(ElementBlock(etype, dim, int(key[0]),
                                   phys[sel].copy(), tags[sel].copy(),
                                   nodes[sel].copy()))


# ---------------------------------------------------------------------------
# format 4.1
# ---------------------------------------------------------------------------
def _read_v41(c, version, binary, path):
    names, node_tags, xyz, blocks, periodic = {}, None, None, [], []
    ent_phys = {}
    while c.p < len(c.d):
        c.skip_ws()
        if c.p >= len(c.d):
            break
        sec = c.line()
        if not sec.startswith(b'$'):
            sys.exit('Error: %s: unexpected text %r where a section was '
                     'expected' % (path, sec[:40]))
        name = sec[1:]
        if name == b'PhysicalNames':
            names = _physical_names(c)
        elif name == b'Entities':
            ent_phys = _read_entities41(c, binary, path)
        elif name == b'Nodes':
            node_tags, xyz = _read_nodes41(c, binary, path)
        elif name == b'Elements':
            blocks = _read_elements41(c, binary, path, ent_phys)
        elif name == b'Periodic':
            periodic = _read_periodic(c, binary, version, path)
        else:
            if binary:
                _skip_binary_section(c, name)
            else:
                c.ascii_block(name)
    if node_tags is None or not blocks:
        sys.exit('Error: %s: no $Nodes / $Elements section found' % path)
    return GmshMesh(version, binary, node_tags, xyz, blocks, names, ent_phys,
                    periodic)


def _read_entities41(c, binary, path):
    ent_phys = {}
    if binary:
        counts = c.bin('<u8', 4)
        for dim, n in enumerate(counts):
            for _ in range(int(n)):
                tag = int(c.bin('<i4', 1)[0])
                c.bin('<f8', 3 if dim == 0 else 6)
                nphys = int(c.bin('<u8', 1)[0])
                phys = c.bin('<i4', nphys).astype(np.int64).tolist()
                if dim > 0:
                    nb = int(c.bin('<u8', 1)[0])
                    c.bin('<i4', nb)
                ent_phys[(dim, tag)] = phys
        c.end_section(b'Entities')
    else:
        txt = c.ascii_block(b'Entities')
        toks = txt.split()
        pos = 0
        counts = [int(t) for t in toks[:4]]
        pos = 4
        for dim, n in enumerate(counts):
            for _ in range(n):
                tag = int(toks[pos]); pos += 1
                pos += 3 if dim == 0 else 6
                nphys = int(toks[pos]); pos += 1
                phys = [int(t) for t in toks[pos:pos + nphys]]; pos += nphys
                if dim > 0:
                    nb = int(toks[pos]); pos += 1
                    pos += nb
                ent_phys[(dim, tag)] = phys
    return ent_phys


def _read_nodes41(c, binary, path):
    if binary:
        nblk, nn, _, _ = (int(v) for v in c.bin('<u8', 4))
        tags, coords = [], []
        for _ in range(nblk):
            dim, ent, param = (int(v) for v in c.bin('<i4', 3))
            m = int(c.bin('<u8', 1)[0])
            tags.append(c.bin('<u8', m).astype(np.int64))
            ncomp = 3 + (dim if param else 0)
            coords.append(c.bin('<f8', m * ncomp).reshape(m, ncomp)[:, :3])
        c.end_section(b'Nodes')
    else:
        txt = c.ascii_block(b'Nodes')
        toks = txt.split()
        nblk, nn = int(toks[0]), int(toks[1])
        pos = 4
        tags, coords = [], []
        for _ in range(nblk):
            dim, ent, param, m = (int(t) for t in toks[pos:pos + 4])
            pos += 4
            tags.append(np.array(toks[pos:pos + m], dtype=np.int64))
            pos += m
            ncomp = 3 + (dim if param else 0)
            a = np.array(toks[pos:pos + m * ncomp], dtype=np.float64)
            pos += m * ncomp
            coords.append(a.reshape(m, ncomp)[:, :3])
    node_tags = np.concatenate(tags) if tags else np.zeros(0, np.int64)
    xyz = np.concatenate(coords) if coords else np.zeros((0, 3))
    if node_tags.size != nn:
        sys.exit('Error: %s: $Nodes announces %d nodes, blocks hold %d'
                 % (path, nn, node_tags.size))
    return node_tags, np.ascontiguousarray(xyz, dtype=np.float64)


def _read_elements41(c, binary, path, ent_phys):
    blocks = []
    if binary:
        nblk, ne, _, _ = (int(v) for v in c.bin('<u8', 4))
        for _ in range(nblk):
            dim, ent, etype = (int(v) for v in c.bin('<i4', 3))
            m = int(c.bin('<u8', 1)[0])
            nn = _nnodes(etype, path)
            rec = c.bin('<u8', m * (1 + nn)).reshape(m, 1 + nn).astype(np.int64)
            blocks.append(_block41(etype, dim, ent, rec, ent_phys))
        c.end_section(b'Elements')
    else:
        txt = c.ascii_block(b'Elements')
        toks = txt.split()
        nblk = int(toks[0])
        pos = 4
        for _ in range(nblk):
            dim, ent, etype, m = (int(t) for t in toks[pos:pos + 4])
            pos += 4
            nn = _nnodes(etype, path)
            rec = np.array(toks[pos:pos + m * (1 + nn)], dtype=np.int64)
            pos += m * (1 + nn)
            blocks.append(_block41(etype, dim, ent, rec.reshape(m, 1 + nn),
                                   ent_phys))
    return blocks


def _block41(etype, dim, ent, rec, ent_phys):
    phys_list = ent_phys.get((dim, ent), [])
    # an entity may carry several physical tags; the converter is told all
    # of them through entity_physical, the block records the first
    phys = np.full(rec.shape[0], phys_list[0] if phys_list else 0,
                   dtype=np.int64)
    return ElementBlock(etype, dim, ent, phys, rec[:, 0].copy(),
                        rec[:, 1:].copy())


def _read_periodic(c, binary, version, path):
    links = []
    v4 = version.startswith('4')
    if binary:
        if v4:
            n = int(c.bin('<u8', 1)[0])
            for _ in range(n):
                dim, ent, master = (int(v) for v in c.bin('<i4', 3))
                na = int(c.bin('<u8', 1)[0])
                affine = c.bin('<f8', na).astype(np.float64)
                m = int(c.bin('<u8', 1)[0])
                nm = c.bin('<u8', 2 * m).reshape(m, 2).astype(np.int64)
                links.append(PeriodicLink(dim, ent, master, affine, nm))
        else:
            n = int(c.bin('<i4', 1)[0])
            for _ in range(n):
                dim, ent, master = (int(v) for v in c.bin('<i4', 3))
                # 2.2 binary: optional affine block preceded by its count
                na = int(c.bin('<i4', 1)[0])
                affine = c.bin('<f8', na).astype(np.float64)
                m = int(c.bin('<i4', 1)[0])
                nm = c.bin('<i4', 2 * m).reshape(m, 2).astype(np.int64)
                links.append(PeriodicLink(dim, ent, master, affine, nm))
        c.end_section(b'Periodic')
        return links
    txt = c.ascii_block(b'Periodic').decode('ascii', 'replace')
    toks = txt.split()
    pos = 0
    n = int(toks[pos]); pos += 1
    for _ in range(n):
        dim, ent, master = (int(t) for t in toks[pos:pos + 3]); pos += 3
        affine = np.zeros(0)
        if v4:
            na = int(toks[pos]); pos += 1
            affine = np.array(toks[pos:pos + na], dtype=np.float64); pos += na
        elif toks[pos].lower() == 'affine':
            pos += 1
            affine = np.array(toks[pos:pos + 16], dtype=np.float64); pos += 16
        m = int(toks[pos]); pos += 1
        nm = np.array(toks[pos:pos + 2 * m], dtype=np.int64).reshape(m, 2)
        pos += 2 * m
        links.append(PeriodicLink(dim, ent, master, affine, nm))
    return links
