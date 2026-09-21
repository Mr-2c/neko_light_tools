# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see formats.py for the full licence text.
"""Layouts of Neko's field files (``.fld`` / ``name0.f00012``, the Nek5000
"std" format as written by ``src/io/fld_file.f90``) and checkpoint files
(``.chkp``, ``src/io/chkp_file.f90``), as lists of fixed-size element-major
blocks plus the bytes that are not per element.  Both files are flat
MPI-IO byte streams: an element's record inside a block sits at
``block_start + position * record_bytes``, and a run owns the positions of
its linear element distribution, so reordering the elements of a mesh is a
gather of fixed-size records in every block.

Nothing here interprets the numbers; the byte content of a record is copied
as it is.
"""
import os
import sys
from typing import NamedTuple

import numpy as np

FLD_HEADER = 132
FLD_TEST_PATTERN = np.float32(6.54321)
CHKP_HEADER = 24


class Block(NamedTuple):
    name: str
    offset: int        # byte offset of the block in the file
    rec_bytes: int     # bytes per element (0 for a verbatim-copied block)
    nbytes: int        # total bytes of the block


class FldLayout(NamedTuple):
    header: bytes
    wdsize: int
    lx: int
    ly: int
    lz: int
    nelgv: int
    time: float
    step: int
    rdcode: str
    gdim: int
    n_scalars: int
    blocks: list       # [Block], in file order, covering the whole file
    size: int


def fld_layout(path):
    """Parse the header of a Neko field file and lay out its blocks
    (``fld_file_write``: 132-byte header, 4-byte test pattern, int32 idx per
    element, then per rdcode letter a data block -- X and U hold gdim*lxyz
    values per element (component-major inside the element), P, T and each
    S one lxyz -- and, in 3D only, one single-precision min/max block per
    field group).  The layout must account for the file size exactly."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        hdr = f.read(FLD_HEADER)
        tp = np.frombuffer(f.read(4), dtype='<f4')
    if len(hdr) != FLD_HEADER or not hdr.startswith(b'#std'):
        sys.exit('Error: %s is not a Neko/Nek5000 field file (no #std header)'
                 % path)
    if tp.size != 1 or abs(float(tp[0]) - float(FLD_TEST_PATTERN)) > 1e-5:
        sys.exit('Error: %s: endianness test pattern mismatch (%s); the file '
                 'was written with the other byte order' % (path, tp))
    toks = hdr.decode('ascii', 'replace').split()
    # #std wdsize lx ly lz nelv nelgv time step fid nfid rdcode
    try:
        wdsize, lx, ly, lz = int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4])
        nelgv = int(toks[6])
        time = float(toks[7])
        step = int(toks[8])
    except (IndexError, ValueError):
        sys.exit('Error: %s: cannot parse the field-file header %r' % (path, hdr))
    rdcode = hdr[83:93].decode('ascii', 'replace').strip()
    if wdsize not in (4, 8):
        sys.exit('Error: %s: word size %d in header' % (path, wdsize))
    gdim = 2 if lz == 1 else 3
    lxyz = lx * ly * lz
    n_s = 0
    if 'S' in rdcode:
        i = rdcode.index('S')
        try:
            n_s = int(rdcode[i + 1:i + 3])
        except ValueError:
            sys.exit('Error: %s: cannot parse the scalar count in rdcode %r'
                     % (path, rdcode))
    blocks = [Block('header', 0, 0, FLD_HEADER + 4),
              Block('idx', FLD_HEADER + 4, 4, 4 * nelgv)]
    off = FLD_HEADER + 4 + 4 * nelgv
    groups = []                       # (name, ncomp) in rdcode order
    if 'X' in rdcode:
        groups.append(('X', gdim))
    if 'U' in rdcode:
        groups.append(('U', gdim))
    if 'P' in rdcode:
        groups.append(('P', 1))
    if 'T' in rdcode:
        groups.append(('T', 1))
    for i in range(n_s):
        groups.append(('S%02d' % (i + 1), 1))
    for name, nc in groups:
        rb = nc * lxyz * wdsize
        blocks.append(Block(name, off, rb, rb * nelgv))
        off += rb * nelgv
    if gdim == 3:
        for name, nc in groups:
            rb = 2 * nc * 4
            blocks.append(Block(name + ' min/max', off, rb, rb * nelgv))
            off += rb * nelgv
    if off != size:
        sys.exit('Error: %s: the header (lx=%d ly=%d lz=%d, %d elements, word '
                 'size %d, rdcode %r) implies %d bytes but the file has %d'
                 % (path, lx, ly, lz, nelgv, wdsize, rdcode, off, size))
    return FldLayout(hdr, wdsize, lx, ly, lz, nelgv, time, step, rdcode, gdim,
                     n_s, blocks, size)


class ChkpLayout(NamedTuple):
    header: bytes
    nelgv: int
    gdim: int
    lx: int
    optional: int
    time: float
    rp: int
    blocks: list
    size: int
    flags: dict


def chkp_layout(path, rp=None):
    """Parse a Neko checkpoint (``chkp_file_write``): 24-byte header
    (glb_nelv, gdim, lx, optional_fields as int32, time as float64), then
    element-major blocks of lxyz reals each -- u, v, w, p; with bit 0 the six
    velocity lags; bit 1 the scalar; bit 2 the 160-byte tlag/dtlag block
    (not per element); bit 3 the six abx/aby/abz; bit 4 the two scalar lags
    and abs1, abs2; bit 5 the fourteen ALE blocks followed by tracker
    arrays of unrecorded length (copied verbatim to the end of file).  The
    real kind ``rp`` (4 or 8 bytes) is not stored; it is inferred from the
    file size unless given."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        hdr = f.read(CHKP_HEADER)
    if len(hdr) != CHKP_HEADER:
        sys.exit('Error: %s is too short to be a Neko checkpoint' % path)
    nelgv, gdim, lx, opt = (int(v) for v in np.frombuffer(hdr[:16], dtype='<i4'))
    time = float(np.frombuffer(hdr[16:24], dtype='<f8')[0])
    if gdim not in (2, 3) or lx < 2 or nelgv < 1 or opt < 0 or opt >= 64:
        sys.exit('Error: %s does not look like a Neko checkpoint (glb_nelv=%d, '
                 'gdim=%d, lx=%d, optional_fields=%d)' % (path, nelgv, gdim, lx, opt))
    lxyz = lx ** gdim
    flags = dict(lag=bool(opt & 1), scalar=bool(opt & 2), dtlag=bool(opt & 4),
                 abvel=bool(opt & 8), scalarlag=bool(opt & 16), ale=bool(opt & 32))
    names = ['u', 'v', 'w', 'p']
    if flags['lag']:
        names += ['ulag1', 'ulag2', 'vlag1', 'vlag2', 'wlag1', 'wlag2']
    if flags['scalar']:
        names += ['s']
    if flags['dtlag']:
        names += ['tlag/dtlag']
    if flags['abvel']:
        names += ['abx1', 'abx2', 'aby1', 'aby2', 'abz1', 'abz2']
    if flags['scalarlag']:
        names += ['slag1', 'slag2', 'abs1', 'abs2']
    if flags['ale']:
        names += ['msh_x', 'msh_y', 'msh_z', 'wm_x', 'wm_y', 'wm_z',
                  'wm_x_lag1', 'wm_x_lag2', 'wm_y_lag1', 'wm_y_lag2',
                  'wm_z_lag1', 'wm_z_lag2', 'Blag', 'Blaglag']
    nblk = sum(1 for n in names if n != 'tlag/dtlag')
    fixed = CHKP_HEADER + (160 if flags['dtlag'] else 0)
    if rp is None:
        fits = [r for r in (8, 4) if fixed + nblk * nelgv * lxyz * r
                + (0 if not flags['ale'] else 0) <= size
                and (flags['ale'] or fixed + nblk * nelgv * lxyz * r == size)]
        if flags['ale'] and len(fits) > 1:
            sys.exit('Error: %s: ALE checkpoint; the real kind cannot be '
                     'inferred from the file size, give --rp 8 or --rp 4' % path)
        if len(fits) != 1:
            sys.exit('Error: %s: no real kind (4 or 8 bytes) makes the header '
                     '(%d elements, gdim %d, lx %d, optional_fields %d -> %d '
                     'blocks) fit the file size %d' % (path, nelgv, gdim, lx, opt,
                                                        nblk, size))
        rp = fits[0]
    expect = fixed + nblk * nelgv * lxyz * rp
    if (not flags['ale'] and expect != size) or expect > size:
        sys.exit('Error: %s: with %d-byte reals the layout needs %d bytes, the '
                 'file has %d' % (path, rp, expect, size))
    blocks = [Block('header', 0, 0, CHKP_HEADER)]
    off = CHKP_HEADER
    for n in names:
        if n == 'tlag/dtlag':
            blocks.append(Block(n, off, 0, 160)); off += 160
        else:
            rb = lxyz * rp
            blocks.append(Block(n, off, rb, rb * nelgv)); off += rb * nelgv
    if off < size:
        blocks.append(Block('ALE trackers', off, 0, size - off))
    return ChkpLayout(hdr, nelgv, gdim, lx, opt, time, rp, blocks, size, flags)


def read_fld_idx(path, layout):
    b = layout.blocks[1]
    return np.fromfile(path, dtype='<i4', count=layout.nelgv, offset=b.offset)
