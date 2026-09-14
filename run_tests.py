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
"""run_tests.py -- the validation suite for the Neko light tools.

    python3 run_tests.py [/path/to/neko]

The Neko checkout supplies the golden files (examples/hemi/hemi.re2 with the
hemi.nmsh Neko's rea2nbin wrote from it, the shipped box meshes) and the
mesh corpus; it defaults to $NEKO_DIR, then the checkout this directory
lives in (contrib/<tools>/../..), then a sibling ``neko`` directory.  Work
files go to a temporary directory.  Principles:

  * oracles are Neko's own source (datadist.f90's block formula, the
    closed-form point/face/edge counts of a box, the .re2 layout written by
    hand here) or Neko-written golden files, never the tools' own code;
  * missing optional dependencies SKIP their tests, they do not fail them;
  * every generated mesh is re-read by mesh_checker.py before it counts as
    a pass.

Exit status 0 iff no test failed (skips are fine).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nekolight import (read_nmsh, write_nmsh, EL_DT, QUAD_DT,   # noqa: E402
                       ZONE_DT)


def find_neko():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    for c in (os.environ.get('NEKO_DIR', ''),
              os.path.join(HERE, '..', 'neko'),
              os.path.join(HERE, '..', '..'),
              os.path.join(HERE, '..', '..', '..')):
        if c and os.path.isdir(os.path.join(c, 'examples')) \
           and os.path.isdir(os.path.join(c, 'contrib')):
            return os.path.abspath(c)
    sys.exit('Usage: run_tests.py /path/to/neko  (or set NEKO_DIR)')


NEKO = find_neko()
WORK = tempfile.mkdtemp(prefix='nekolight_tests.')
RESULTS = []


def tool(name, *args, env=None):
    """Run one of the CLIs; returns (exit_code, stdout+stderr)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, os.path.join(HERE, name)]
                       + [str(a) for a in args],
                       capture_output=True, text=True, cwd=WORK, env=e)
    return r.returncode, r.stdout + r.stderr


def report(name, ok, detail=''):
    RESULTS.append((name, ok))
    print('  %-58s %s %s' % (name, 'PASS' if ok else 'FAIL',
                             detail if not ok else ''), flush=True)


def skip(name, why):
    print('  %-58s SKIP (%s)' % (name, why), flush=True)


def wpath(name):
    return os.path.join(WORK, name)


def have(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def all_meshes():
    out = []
    for root in ('examples', 'tests'):
        for d, _, files in os.walk(os.path.join(NEKO, root)):
            out += [os.path.join(d, f) for f in files if f.endswith('.nmsh')]
    return sorted(out)


def neko_linear_sizes_oracle(M, P):
    """datadist.f90: Ip(rank) = floor((M + P - rank - 1) / P)."""
    return np.array([(M + P - r - 1) // P for r in range(P)], dtype=np.int64)


def elem_keys(elems):
    """Order-independent element identity: the raw (vidx,xyz) payload."""
    return set(elems['v'][i].tobytes() for i in range(elems.size))


def key_to_id(elems):
    return {elems['v'][i].tobytes(): int(elems['id'][i])
            for i in range(elems.size)}


def zone_sig(m):
    pos = np.empty(m.nelv + 1, dtype=np.int64)
    pos[m.elems['id']] = np.arange(m.nelv)
    sig = set()
    for z in m.zones:
        anchor = m.elems['v'][pos[z['e']]].tobytes()
        partner = (m.elems['v'][pos[z['p_e']]].tobytes()
                   if z['t'] == 5 else b'')
        sig.add((anchor, int(z['f']), partner, int(z['p_f']),
                 z['g'].tobytes() if z['t'] == 5 else b'', int(z['t'])))
    return sig


def curve_sig(m):
    pos = np.empty(m.nelv + 1, dtype=np.int64)
    pos[m.elems['id']] = np.arange(m.nelv)
    return set((m.elems['v'][pos[c['e']]].tobytes(), c['data'].tobytes(),
                c['type'].tobytes()) for c in m.curves)


def golden_compare(a_path, b_path):
    """Byte-compare two .nmsh allowing diffs ONLY in the labelled-zone
    p_e/glb_pt_ids fields (which Neko's writers leave uninitialised)."""
    a, b = open(a_path, 'rb').read(), open(b_path, 'rb').read()
    if len(a) != len(b):
        return False, 'sizes differ'
    nelv, gdim = np.frombuffer(a[:8], '<i4')
    dt = EL_DT if gdim == 3 else QUAD_DT
    off = 8 + int(nelv) * dt.itemsize
    if a[:off + 4] != b[:off + 4]:
        return False, 'element section differs'
    nz = int(np.frombuffer(a[off:off + 4], '<i4')[0])
    za = np.frombuffer(a[off + 4:off + 4 + nz * 36], ZONE_DT)
    zb = np.frombuffer(b[off + 4:off + 4 + nz * 36], ZONE_DT)
    if not all(np.array_equal(za[k], zb[k]) for k in ('e', 'f', 'p_f', 't')):
        return False, 'zone core fields differ'
    n7 = za['t'] != 7
    if not (np.array_equal(za['p_e'][n7], zb['p_e'][n7])
            and np.array_equal(za['g'][n7], zb['g'][n7])):
        return False, 'non-labelled zone p_e/g differ'
    if a[off + 4 + nz * 36:] != b[off + 4 + nz * 36:]:
        return False, 'curve section / tail differs'
    return True, ''


def checker_numbers(out):
    """(points, faces, edges) parsed from mesh_checker output."""
    g = lambda k: int(re.search(r'Number of %s:\s+(\d+)' % k, out).group(1))
    return g('points'), g('faces'), g('edges')


def prepart_contract(name, src, out_path, P, gdim=3):
    """The prepart contract on one output: permutation, zone/curve binding,
    Neko's exact linear-read block sizes, original order kept inside every
    block, checker passes."""
    m = read_nmsh(out_path)
    ok = m.gdim == gdim \
        and np.array_equal(np.sort(m.elems['id']), np.arange(1, m.nelv + 1)) \
        and np.array_equal(m.elems['id'], np.arange(1, m.nelv + 1)) \
        and elem_keys(m.elems) == elem_keys(src.elems) \
        and zone_sig(m) == zone_sig(src) and curve_sig(m) == curve_sig(src)
    report('%s: permutation + zone/curve binding' % name, ok)
    oracle = neko_linear_sizes_oracle(src.nelv, P)
    bounds = np.concatenate([[0], np.cumsum(oracle)])
    old_id = key_to_id(src.elems)
    blocks_ok, mono_ok = True, True
    seen = set()
    for r in range(P):
        blk = m.elems[bounds[r]:bounds[r + 1]]
        keys = [blk['v'][i].tobytes() for i in range(blk.size)]
        if len(keys) != oracle[r] or seen & set(keys):
            blocks_ok = False
        seen |= set(keys)
        ids = [old_id[k] for k in keys]
        if any(b <= a for a, b in zip(ids, ids[1:])):
            mono_ok = False
    report('%s: linear-read blocks have Neko\'s exact sizes' % name,
           blocks_ok)
    report('%s: original ids strictly increasing inside each block' % name,
           mono_ok)
    rc, out = tool('mesh_checker.py', out_path)
    report('%s: checker passes on output' % name, rc == 0, out[-300:])
    return m


def write_re2_2d(path, nx, ny, x1, y1, periodic_x=True, curve=True):
    """A #v002 2D .re2 of nx x ny quads on [0,x1] x [0,y1], laid out exactly
    as Neko's re2 module defines the records (re2v2_xy_t: f64 group + 4 f64
    x + 4 f64 y; curve/BC: f64 el, f64 side, 5 f64, char(8)).  Periodic 'P'
    between the x sides (face 4 <-> face 2), 'W' on the y sides, optionally
    one 'm' curve on edge 1 of element 1."""
    nel = nx * ny
    hdr = ('#v002%9d%3d%9d' % (nel, 2, nel)).ljust(80)
    with open(path, 'wb') as f:
        f.write(hdr.encode('ascii'))
        np.float32(6.54321).tofile(f)
        gx = np.linspace(0.0, x1, nx + 1)
        gy = np.linspace(0.0, y1, ny + 1)
        for ey in range(ny):
            for ex in range(nx):
                np.float64(1.0).tofile(f)
                xs = [gx[ex], gx[ex + 1], gx[ex + 1], gx[ex]]
                ys = [gy[ey], gy[ey], gy[ey + 1], gy[ey + 1]]
                np.array(xs + ys, dtype='<f8').tofile(f)
        curves = []
        if curve:
            curves.append((1, 1, [0.5 * gx[1], -0.1, 0.0, 0.0, 0.0], b'm'))
        np.float64(len(curves)).tofile(f)
        for (e, edge, d, t) in curves:
            np.array([e, edge] + d, dtype='<f8').tofile(f)
            f.write(t.ljust(8))
        bcs = []
        for ey in range(ny):
            for ex in range(nx):
                e = 1 + ex + nx * ey
                if ey == 0:
                    bcs.append((e, 1, [0.0] * 5, b'W'))
                if ey == ny - 1:
                    bcs.append((e, 3, [0.0] * 5, b'W'))
                if ex == 0:
                    pe = 1 + (nx - 1) + nx * ey
                    bcs.append((e, 4, [pe, 2, 0, 0, 0]
                                if periodic_x else [0.0] * 5,
                                b'P' if periodic_x else b'W'))
                if ex == nx - 1:
                    pe = 1 + nx * ey
                    bcs.append((e, 2, [pe, 4, 0, 0, 0]
                                if periodic_x else [0.0] * 5,
                                b'P' if periodic_x else b'W'))
        np.float64(len(bcs)).tofile(f)
        for (e, face, d, t) in bcs:
            np.array([e, face] + [float(v) for v in d], dtype='<f8').tofile(f)
            f.write(t.ljust(8))


# ===========================================================================
print('Neko light tools test suite  (Neko checkout: %s)' % NEKO)
print('work dir: %s\n' % WORK)
HAVE_SCIPY = have('scipy')
HAVE_METIS = have('pymetis')

# ---- T1: rea2nbin golden (hemi) -------------------------------------------
print('[T1] rea2nbin: hemi golden pair (Neko-written .nmsh from hemi.re2)')
hemi_re2 = os.path.join(NEKO, 'examples', 'hemi', 'hemi.re2')
hemi_ref = os.path.join(NEKO, 'examples', 'hemi', 'hemi.nmsh')
if os.path.exists(hemi_re2) and os.path.exists(hemi_ref):
    rc, out = tool('rea2nbin.py', hemi_re2, wpath('hemi.nmsh'))
    ok = rc == 0
    if ok:
        ok, why = golden_compare(wpath('hemi.nmsh'), hemi_ref)
        report('hemi.re2 -> nmsh byte-exact (mod uninit fields)', ok, why)
    else:
        report('hemi conversion runs', False, out[-200:])
else:
    sys.exit('examples/hemi not found in %s -- not a Neko checkout?' % NEKO)
src = read_nmsh(hemi_ref)

# ---- T1b: rea2nbin 2D (hand-written re2, closed-form expectations) -------
print('[T1b] rea2nbin: 2D re2 (3x2 quads, x-periodic, W walls, one m curve)')
write_re2_2d(wpath('q.re2'), 3, 2, 3.0, 2.0)
rc, out = tool('rea2nbin.py', 'q.re2', 'q.nmsh')
if rc != 0:
    report('2D conversion runs', False, out[-300:])
else:
    q = read_nmsh(wpath('q.nmsh'))
    z5 = q.zones[q.zones['t'] == 5]
    z7 = q.zones[q.zones['t'] == 7]
    # first-appearance ids of the 12 grid points, element by element
    exp_vid = np.array([[1, 2, 3, 4], [2, 5, 6, 3], [5, 7, 8, 6],
                        [4, 3, 9, 10], [3, 6, 11, 9], [6, 8, 12, 11]])
    ok = q.gdim == 2 and q.nelv == 6 \
        and np.array_equal(q.elems['v']['idx'], exp_vid) \
        and (q.elems['v']['xyz'][:, :, 2] == 0.0).all()
    report('2D: quad records, first-appearance point ids, z = 0', ok)
    # 'P' records in file order: (el1 f4->sym1, el3 f2->sym2, el4, el6);
    # min-merged ids: 7->1, 8->4, 12->10
    exp_p = [(1, 1, 3, 2, (1, 4)), (3, 2, 1, 1, (1, 4)),
             (4, 1, 6, 2, (4, 10)), (6, 2, 4, 1, (4, 10))]
    ok = z5.shape[0] == 4 and all(
        (int(z['e']), int(z['f']), int(z['p_e']), int(z['p_f']),
         tuple(int(v) for v in z['g'][:2])) == exp
        and (z['g'][2:] == 0).all() for z, exp in zip(z5, exp_p))
    report('2D: periodic records, 2 merged ids per edge facet', ok)
    ok = z7.shape[0] == 6 and (z7['p_f'] == 1).all() \
        and sorted((int(z['e']), int(z['f'])) for z in z7) \
        == [(1, 3), (2, 3), (3, 3), (4, 4), (5, 4), (6, 4)]
    report('2D: W walls -> label 1 on sym facets 3/4', ok)
    ok = q.curves.shape[0] == 1 and int(q.curves['e'][0]) == 1 \
        and int(q.curves['type'][0, 0]) == 4 \
        and abs(float(q.curves['data'][0, 0, 1]) + 0.1) < 1e-15
    report('2D: midside curve record carried', ok)
    rc, out = tool('mesh_checker.py', 'q.nmsh', '--jacobian')
    nums = checker_numbers(out) if rc == 0 else None
    # Neko's slab: glb_mpts is the largest id registered at read time, the
    # top copy of point 12 (12 + 8*6 = 60); faces = 15 unique 2D edges + 6
    # tops; edges = 15 + 9 distinct merged points (degenerate vertical edges)
    report('2D: checker on the slab: points 60, faces 21, edges 24',
           rc == 0 and nums == (60, 21, 24), out[-300:])
    write_re2_2d(wpath('q2.re2'), 3, 2, 3.0, 2.0, periodic_x=False,
                 curve=False)
    rc, out = tool('rea2nbin.py', 'q2.re2', 'q2.nmsh')
    q2 = read_nmsh(wpath('q2.nmsh')) if rc == 0 else None
    report('2D: all-W box -> 10 labelled facets, no periodic, no curves',
           rc == 0 and q2.zones.shape[0] == 10
           and (q2.zones['t'] == 7).all() and q2.curves.shape[0] == 0)

# ---- T2: genmeshbox vs a shipped box --------------------------------------
print('[T2] genmeshbox: shipped turb_channel box')
box_ref = os.path.join(NEKO, 'examples', 'turb_channel', 'box.nmsh')
if os.path.exists(box_ref):
    m = read_nmsh(box_ref)
    xyz = m.elems['v']['xyz'].reshape(-1, 3)
    g = [np.unique(xyz[:, c]) for c in range(3)]
    z5 = m.zones[m.zones['t'] == 5]
    per = ['.true.' if (z5['f'] == f).any() else '.false.' for f in (1, 3, 5)]
    lim = [repr(float(v)) for gg in g for v in (gg[0], gg[-1])]
    n = [str(len(gg) - 1) for gg in g]
    rc, out = tool('genmeshbox.py', *(lim[0:2] + lim[2:4] + lim[4:6] + n
                                      + per + ['box.nmsh', '--direct']))
    ok = rc == 0
    if ok:
        ok, why = golden_compare(wpath('box.nmsh'), box_ref)
        report('regenerated box byte-exact (mod uninit fields)', ok, why)
    else:
        report('box generation runs', False, out[-200:])
    rc, _ = tool('mesh_checker.py', 'box.nmsh')
    report('checker passes on generated box', rc == 0)
else:
    skip('turb_channel box', 'not found')

cyl = os.path.join(NEKO, 'examples', 'cylinder', 'cylinder.nmsh')
cyl2d = os.path.join(NEKO, 'examples', '2d_cylinder', '2d_cylinder.nmsh')

# ---- T2b: checker counts vs closed forms on boxes --------------------------
print('[T2b] mesh_checker: closed-form point/face/edge counts of boxes')
nx, ny, nz = 3, 4, 5
for per in ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)):
    flags = ['.true.' if p else '.false.' for p in per]
    name = 'box_%d%d%d.nmsh' % per
    rc, _ = tool('genmeshbox.py', 0, 1, 0, 1, 0, 1, nx, ny, nz, *flags,
                 'uniform', 'uniform', 'uniform', name)
    rc2, out = tool('mesh_checker.py', name, '--jacobian')
    Px, Py, Pz = [n if p else n + 1 for n, p in zip((nx, ny, nz), per)]
    faces = Px * ny * nz + nx * Py * nz + nx * ny * Pz
    edges = nx * Py * Pz + Px * ny * Pz + Px * Py * nz
    # 'points' is Neko's max_pts_id: the largest RAW point id registered
    # when the elements are added -- the merge never lowers it
    points = (nx + 1) * (ny + 1) * (nz + 1)
    nums = checker_numbers(out) if rc2 == 0 else None
    report('box periodic=%s: points/faces/edges %s' % (per, (points, faces,
                                                             edges)),
           rc == 0 and rc2 == 0 and nums == (points, faces, edges),
           str(nums))
rc, out = tool('genmeshbox.py', 1, 0, 0, 1, 0, 1, 2, 2, 2, 'inv.nmsh')
rc2, out2 = tool('mesh_checker.py', 'inv.nmsh', '--jacobian')
report('inverted box (x1 < x0): checker flags negative Jacobians, exit 1',
       rc == 0 and rc2 == 1 and 'negative/zero Jacobian' in out2)
# a legacy zone type: Neko's reader ignores the record, its checker then
# sees an unlabelled external facet and fails -- same verdict here
bm = read_nmsh(wpath('box_000.nmsh'))
zl = bm.zones.copy()
zl['t'][0] = 1
write_nmsh(wpath('legacy.nmsh'), bm.elems, (zl,), bm.curves)
rc, out = tool('mesh_checker.py', 'legacy.nmsh')
report('legacy zone type -> unlabelled facet, exit 1 (as Neko)',
       rc == 1 and 'Legacy zone records' in out
       and 'unlabelled external faces' in out)
# a curved element Neko refuses (arc radius below half the chord)
from nekolight import CURVE_DT as _CDT   # noqa: E402
cv = np.zeros(1, dtype=_CDT)
cv['e'] = 1
cv['type'][0, 0] = 3
cv['data'][0, 0, 0] = 0.05
write_nmsh(wpath('badarc.nmsh'), bm.elems, (bm.zones,), cv)
rc, out = tool('mesh_checker.py', 'badarc.nmsh')
report('arc radius too small: checker exits 1 without --jacobian',
       rc == 1 and 'Radius' in out)
# bounding box follows a midside point that bulges past the corner hull
cv = np.zeros(1, dtype=_CDT)
cv['e'] = 1
cv['type'][0, 0] = 4
cv['data'][0, 0, :3] = [0.5 / nx, -0.75, 0.0]
write_nmsh(wpath('bulge.nmsh'), bm.elems, (bm.zones,), cv)
rc, out = tool('mesh_checker.py', 'bulge.nmsh')
report('bounding box includes the curved geometry (y min = -0.75)',
       rc == 0 and re.search(r'y\s+-0\.75\s', out) is not None, out[-300:])

# ---- T2c: structural defects Neko does not detect --------------------------
print('[T2c] mesh_checker: point-id/coordinate and shared-edge midside checks')
bm = read_nmsh(wpath('box_000.nmsh'))
e2 = bm.elems.copy()
e2['v']['xyz'][0, 6, 0] += 1e-3        # interior corner (shared), same id
write_nmsh(wpath('idconf.nmsh'), e2, (bm.zones,), bm.curves)
rc, out = tool('mesh_checker.py', 'idconf.nmsh')
report('point id with two coordinates -> error',
       rc == 1 and 'more than one coordinate' in out, out[-300:])
if os.path.exists(cyl):
    from nekolight import EDGE_CYC  # noqa: E402
    cm = read_nmsh(cyl)
    cpos = np.empty(cm.nelv + 1, dtype=np.int64)
    cpos[cm.elems['id']] = np.arange(cm.nelv)
    vv = cm.elems['v']['idx'].astype(np.int64)
    mult = {}
    for e in range(cm.nelv):
        for kk in range(12):
            key = tuple(sorted(vv[e, EDGE_CYC[kk]]))
            mult[key] = mult.get(key, 0) + 1
    # a curved edge shared by at least two elements
    rec, k = next((i, kk) for i in range(cm.curves.shape[0])
                  for kk in range(12) if cm.curves['type'][i, kk] == 4
                  and mult[tuple(sorted(vv[cpos[cm.curves['e'][i]],
                                           EDGE_CYC[kk]]))] >= 2)
    cv = cm.curves.copy()
    cv['data'][rec, k, 1] += 1e-4                   # one side of a shared edge
    write_nmsh(wpath('midconf.nmsh'), cm.elems, (cm.zones,), cv)
    rc, out = tool('mesh_checker.py', 'midconf.nmsh')
    report('midside points disagreeing across a shared edge -> error',
           rc == 1 and 'midside points disagree' in out, out[-300:])
    cv = cm.curves.copy()
    cv['type'][rec, k] = 0                          # curved here, straight there
    write_nmsh(wpath('mixed.nmsh'), cm.elems, (cm.zones,), cv)
    rc, out = tool('mesh_checker.py', 'mixed.nmsh')
    report('edge curved in one element only -> error',
           rc == 1 and 'straight in a neighbour' in out, out[-300:])
    rc, out = tool('mesh_checker.py', cyl)
    report('cylinder mesh itself passes both checks', rc == 0, out[-300:])

# ---- T3: checker corpus sweep ---------------------------------------------
print('[T3] mesh_checker: every shipped .nmsh (2D and 3D)')
meshes = all_meshes()
bad, legacy = [], []
for f in meshes:
    rc, out = tool('mesh_checker.py', f)
    if rc != 0:
        # the nekbone/poisson meshes carry pre-labelled-zone records of
        # type 1..4; Neko's reader ignores those, so Neko's own checker
        # reports their facets as unlabelled and fails -- same verdict here
        if 'Legacy zone records' in out and 'unlabelled external' in out:
            legacy.append(f)
        else:
            bad.append(f)
report('corpus sweep (%d meshes, %d with legacy zone types failing as '
       'in Neko)' % (len(meshes), len(legacy)), not bad, '; '.join(bad[:3]))

# ---- T4: prepart contract -------------------------------------------------
print('[T4] prepart: contract on hemi (odd sizes) per backend')
backends = ['geometric', 'grid']
if HAVE_SCIPY:
    backends.insert(0, 'spectral')
    if HAVE_METIS:
        backends.insert(1, 'metis')
    else:
        skip('metis backend', 'pymetis not installed')
else:
    skip('spectral/metis backends', 'scipy not installed')
NP = 8
for be in backends:
    extra = ['--grid', '2,2,2'] if be == 'grid' else ['--backend', be]
    rc, out = tool('prepart.py', hemi_ref, NP, 'h_%s.nmsh' % be, *extra)
    if rc != 0:
        report('%s: runs' % be, False, out[-300:])
        continue
    prepart_contract(be, src, wpath('h_%s.nmsh' % be), NP)
    rc2, _ = tool('prepart.py', hemi_ref, NP, 'h2_%s.nmsh' % be, *extra)
    same = open(wpath('h_%s.nmsh' % be), 'rb').read() == \
        open(wpath('h2_%s.nmsh' % be), 'rb').read()
    report('%s: deterministic output' % be, rc2 == 0 and same)

# ---- T4a: exact-size repair must converge (METIS) --------------------------
print('[T4a] prepart: METIS size repair on many parts / disconnected mesh')
if HAVE_METIS:
    rc, _ = tool('genmeshbox.py', 0, 1, 0, 1, 0, 1, 6, 6, 6, 'b666.nmsh')
    b666 = read_nmsh(wpath('b666.nmsh'))
    for P in (100, 50, 27):
        rc, out = tool('prepart.py', 'b666.nmsh', P, 'b666_%d.nmsh' % P,
                       '--metis')
        if rc != 0:
            report('metis 6x6x6 box P=%d: runs' % P, False, out[-300:])
        else:
            prepart_contract('metis 6x6x6 box P=%d' % P, b666,
                             wpath('b666_%d.nmsh' % P), P)
    # two disconnected boxes in one file
    tool('genmeshbox.py', 0, 1, 0, 1, 0, 1, 4, 4, 4, 'b444.nmsh')
    b1 = read_nmsh(wpath('b444.nmsh'))
    e2 = b1.elems.copy()
    e2['id'] += b1.nelv
    e2['v']['idx'] += int(b1.elems['v']['idx'].max())
    e2['v']['xyz'][:, :, 0] += 2.0
    z2 = b1.zones.copy()
    z2['e'] += b1.nelv
    write_nmsh(wpath('two.nmsh'), np.concatenate([b1.elems, e2]),
               (np.concatenate([b1.zones, z2]),), b1.curves)
    two = read_nmsh(wpath('two.nmsh'))
    for P in (29, 47):
        rc, out = tool('prepart.py', 'two.nmsh', P, 'two_%d.nmsh' % P,
                       '--metis')
        if rc != 0:
            report('metis two boxes P=%d: runs' % P, False, out[-300:])
        else:
            prepart_contract('metis two boxes P=%d' % P, two,
                             wpath('two_%d.nmsh' % P), P)
    rc, out = tool('prepart.py', 'b444.nmsh', '--metis', '13', 'after.nmsh')
    report('nparts and output accepted after the options',
           rc == 0 and os.path.exists(wpath('after.nmsh')), out[-200:])
    rc, out = tool('prepart.py', 'b444.nmsh', '--grid', '2,2,2', '-o',
                   'grid_o.nmsh')
    report('-o output with --grid', rc == 0
           and os.path.exists(wpath('grid_o.nmsh')), out[-200:])
else:
    skip('metis size repair', 'pymetis not installed')

# ---- T4b: non-power-of-2 P and ranks-per-node ------------------------------
print('[T4b] prepart: non-power-of-2 nparts, --ranks-per-node, --relabel')
tgv = os.path.join(NEKO, 'examples', 'tgv', '512.nmsh')
if os.path.exists(tgv):
    tsrc = read_nmsh(tgv)
    for be in [b for b in backends if b != 'grid']:
        for P in (7, 12):
            rc, out = tool('prepart.py', tgv, P, 't_%s_%d.nmsh' % (be, P),
                           '--backend', be)
            if rc != 0:
                report('%s P=%d: runs' % (be, P), False, out[-300:])
                continue
            prepart_contract('%s P=%d' % (be, P), tsrc,
                             wpath('t_%s_%d.nmsh' % (be, P)), P)
    if HAVE_SCIPY:
        from nekolight import (pos_of_elid_map, merged_vertex_ids,
                               compress_ids, dual_graph)
        from scipy.sparse.csgraph import connected_components
        pos = pos_of_elid_map(tsrc.nelv, tsrc.elems)
        A = dual_graph(compress_ids(merged_vertex_ids(tsrc, pos)))
        old_id = key_to_id(tsrc.elems)
        for be in ['spectral'] + (['metis'] if HAVE_METIS else []):
            P, N = 12, 4
            rc, out = tool('prepart.py', tgv, P, 'rpn_%s.nmsh' % be,
                           '--backend', be, '--ranks-per-node', N)
            if rc != 0:
                report('%s rpn: runs' % be, False, out[-300:])
                continue
            m = prepart_contract('%s P=%d N=%d' % (be, P, N), tsrc,
                                 wpath('rpn_%s.nmsh' % be), P)
            # every node group (N consecutive ranks) is one connected region
            oracle = neko_linear_sizes_oracle(tsrc.nelv, P)
            bounds = np.concatenate([[0], np.cumsum(oracle)])
            conn = True
            for g in range(P // N):
                blk = m.elems[bounds[g * N]:bounds[(g + 1) * N]]
                idx = np.array([pos[old_id[blk['v'][i].tobytes()]]
                                for i in range(blk.size)])
                ncomp, _ = connected_components(A[idx][:, idx],
                                                directed=False)
                conn = conn and ncomp == 1
            report('%s: each node group of %d ranks is connected' % (be, N),
                   conn)
            report('%s: report states intra-node share' % be,
                   'stays inside a node' in out)
        rc, out = tool('prepart.py', tgv, 12, 'rel.nmsh', '--relabel',
                       'fiedler', '--ranks-per-node', 4)
        if rc == 0:
            prepart_contract('spectral + fiedler relabel', tsrc,
                             wpath('rel.nmsh'), 12)
        else:
            report('spectral + fiedler relabel: runs', False, out[-300:])
    # low-memory (memmap) path gives the identical file
    rc1, _ = tool('prepart.py', tgv, 5, 'lm1.nmsh', '--geometric',
                  '--no-stats', '--low-memory')
    rc2, _ = tool('prepart.py', tgv, 5, 'lm2.nmsh', '--geometric',
                  '--no-stats')
    report('--low-memory (memmap) output identical to in-memory',
           rc1 == 0 and rc2 == 0 and open(wpath('lm1.nmsh'), 'rb').read()
           == open(wpath('lm2.nmsh'), 'rb').read())
    # grid: NX*NY*NZ implied, mismatch refused, cuts reported
    rc, out = tool('prepart.py', tgv, 'grid.nmsh', '--grid', '2,3,2')
    ok = rc == 0 and 'x cuts:' in out and 'y cuts' in out
    if ok:
        prepart_contract('grid 2x3x2 (nparts implied)', tsrc,
                         wpath('grid.nmsh'), 12)
    else:
        report('grid 2x3x2 runs and reports cuts', False, out[-300:])
    rc, _ = tool('prepart.py', tgv, 11, 'gridbad.nmsh', '--grid', '2,3,2')
    report('grid: nparts != NX*NY*NZ refused',
           rc != 0 and not os.path.exists(wpath('gridbad.nmsh')))
else:
    skip('tgv/512', 'not found')

# ---- T5: curved + periodic through every graph backend ---------------------
print('[T5] prepart: curved+periodic mesh (turb_pipe) through each backend')
pipe = os.path.join(NEKO, 'examples', 'turb_pipe', 'turb_pipe.nmsh')
if os.path.exists(pipe):
    psrc = read_nmsh(pipe)
    for be in [b for b in backends if b != 'grid']:
        rc, out = tool('prepart.py', pipe, 16, 'pipe_%s.nmsh' % be,
                       '--backend', be, '--no-stats')
        if rc != 0:
            report('%s: runs on turb_pipe' % be, False, out[-300:])
            continue
        o = read_nmsh(wpath('pipe_%s.nmsh' % be))
        report('%s: curves + periodic zones bound through reorder' % be,
               curve_sig(psrc) == curve_sig(o) and zone_sig(psrc)
               == zone_sig(o))
else:
    skip('turb_pipe', 'not found')

# ---- T5b: 2D mesh through prepart -----------------------------------------
print('[T5b] prepart: 2D (quad) mesh stays 2D')
if os.path.exists(cyl2d):
    csrc = read_nmsh(cyl2d)
    for be in [b for b in backends if b != 'grid']:
        rc, out = tool('prepart.py', cyl2d, 6, 'c2d_%s.nmsh' % be,
                       '--backend', be)
        if rc != 0:
            report('%s: runs on 2D' % be, False, out[-300:])
            continue
        prepart_contract('2D %s' % be, csrc, wpath('c2d_%s.nmsh' % be), 6,
                         gdim=2)
else:
    skip('2d_cylinder', 'not found')

# ---- T6: robustness fixtures ----------------------------------------------
print('[T6] robustness (atomic writes, validate-and-refuse)')
shutil.copy(hemi_re2, wpath('same.re2'))
before = open(wpath('same.re2'), 'rb').read()
rc, out = tool('rea2nbin.py', 'same.re2', 'same.re2')
report('same in/out path refused, input intact',
       rc != 0 and open(wpath('same.re2'), 'rb').read() == before)
with open(wpath('trunc.re2'), 'wb') as f:
    f.write(open(hemi_re2, 'rb').read()[:100000])
rc, _ = tool('rea2nbin.py', 'trunc.re2', 'tr.nmsh')
leftovers = [f for f in os.listdir(WORK) if f.endswith('.tmp')]
report('truncated re2: error, no partial output, no temp',
       rc != 0 and not os.path.exists(wpath('tr.nmsh')) and not leftovers)
data = open(hemi_ref, 'rb').read()
open(wpath('nocc.nmsh'), 'wb').write(data[:-4])
rc, _ = tool('mesh_checker.py', 'nocc.nmsh')
report('truncated nmsh (missing curve count) rejected', rc != 0)
open(wpath('trel.nmsh'), 'wb').write(data[:8 + 100 * EL_DT.itemsize + 17])
rc, _ = tool('mesh_checker.py', 'trel.nmsh')
report('truncated nmsh (element section) rejected', rc != 0)
buf = bytearray(data)
off = 8 + src.nelv * EL_DT.itemsize + 4
z = np.frombuffer(bytes(buf[off:off + 36]), ZONE_DT).copy()
z['e'] = src.nelv + 7
buf[off:off + 36] = z.tobytes()
open(wpath('badz.nmsh'), 'wb').write(bytes(buf))
rc1, _ = tool('mesh_checker.py', 'badz.nmsh')
rc2, _ = tool('prepart.py', 'badz.nmsh', 4, 'badz4.nmsh', '--geometric')
report('out-of-range zone ref rejected by checker + prepart',
       rc1 != 0 and rc2 != 0 and not os.path.exists(wpath('badz4.nmsh')))
rc, _ = tool('prepart.py', hemi_ref, src.nelv + 1, 'toomany.nmsh',
             '--geometric')
report('nparts > nelv refused', rc != 0)
# absurd record counts are refused before any allocation
buf = bytearray(data)
buf[off - 4:off] = np.array([2**31 - 1], dtype='<i4').tobytes()
open(wpath('hugez.nmsh'), 'wb').write(bytes(buf))
rc, out = tool('mesh_checker.py', 'hugez.nmsh')
report('zone count larger than the file refused cleanly',
       rc != 0 and 'exceeds what the file can hold' in out)
buf = bytearray(data)
buf[12:16] = np.array([0], dtype='<i4').tobytes()       # first vertex id 0
open(wpath('vid0.nmsh'), 'wb').write(bytes(buf))
rc, out = tool('mesh_checker.py', 'vid0.nmsh')
report('vertex id 0 refused (Neko: Invalid point id)',
       rc != 0 and 'vertex id < 1' in out)
rc, out = tool('rea2nbin.py', hemi_re2, 'tolD.nmsh',
               env={'NEKO_PERIODIC_TOL': '1d-7'})
report('NEKO_PERIODIC_TOL accepts Fortran 1d-7', rc == 0)
mode = os.stat(wpath('hemi.nmsh')).st_mode & 0o777
report('output files get the umask mode, not 0600', mode != 0o600)

# ---- T7: fld ids on a shuffled (valid) mesh --------------------------------
print('[T7] zone-index fld: shuffled element records')
rng = np.random.default_rng(7)
sh = src.elems[rng.permutation(src.nelv)]
z5 = src.zones[src.zones['t'] == 5]
z7 = src.zones[src.zones['t'] == 7]
zx = src.zones[(src.zones['t'] != 5) & (src.zones['t'] != 7)]
write_nmsh(wpath('shuf.nmsh'), sh, (z5, z7, zx), src.curves)
rc, out = tool('mesh_checker.py', 'shuf.nmsh', '--write-zone-indices')
ok = rc == 0
if ok:
    raw = open(wpath('shuf_zone_indices.fld'), 'rb').read()
    ids = np.frombuffer(raw[136:136 + 4 * src.nelv], '<i4')
    ok = np.array_equal(ids, sh['id'])
report('fld id list carries the actual element ids', ok)
rc, out2 = tool('mesh_checker.py', hemi_ref)
report('shuffled records give the same sizes as the original',
       rc == 0 and checker_numbers(out) == checker_numbers(out2))
if HAVE_SCIPY:
    rc, _ = tool('prepart.py', 'shuf.nmsh', 6, 'shuf6.nmsh')
    if rc == 0:
        prepart_contract('shuffled input, spectral', src, wpath('shuf6.nmsh'),
                         6)

# ---- T8: create_periodic_zones --------------------------------------------
print('[T8] create_periodic_zones vs genmeshbox\'s own periodic boxes')
if HAVE_SCIPY:
    args = [0, 1, 0, 2, 0, 3, 4, 5, 6]
    tool('genmeshbox.py', *args, '.false.', '.false.', '.false.',
         'uniform', 'uniform', 'uniform', 'lab.nmsh')
    tool('genmeshbox.py', *args, '.true.', '.true.', '.false.',
         'uniform', 'uniform', 'uniform', 'per.nmsh')
    tool('genmeshbox.py', *args, '.true.', '.false.', '.false.',
         'uniform', 'uniform', 'uniform', 'perx.nmsh')
    rc, out = tool('create_periodic_zones.py', 'lab.nmsh', 'cpz.nmsh',
                   '(1,2),(3,4)')
    rcc, outc = tool('mesh_checker.py', 'cpz.nmsh')
    rcp, outp = tool('mesh_checker.py', 'per.nmsh')
    ok = rc == 0 and rcc == 0 and rcp == 0 \
        and checker_numbers(outc) == checker_numbers(outp)
    report('labelled box -> (1,2),(3,4): checker sizes == genmeshbox '
           'periodic box', ok, out[-200:])
    if rc == 0:
        a, b = read_nmsh(wpath('cpz.nmsh')), read_nmsh(wpath('per.nmsh'))
        report('element records identical to genmeshbox\'s periodic box',
               np.array_equal(a.elems, b.elems))
        report('zone counts: 108 periodic + 40 labelled (5, 6 kept)',
               (a.zones['t'] == 5).sum() == 108
               and sorted(set(int(v) for v in a.zones['p_f'][a.zones['t']
                                                              == 7]))
               == [5, 6] and (a.zones['t'] == 7).sum() == 40)
    rc, out = tool('create_periodic_zones.py', 'perx.nmsh', 'cpz2.nmsh',
                   '3:4')
    rcc, outc2 = tool('mesh_checker.py', 'cpz2.nmsh')
    report('existing periodic zones kept, (3,4) added: same sizes',
           rc == 0 and rcc == 0
           and checker_numbers(outc2) == checker_numbers(outp))
    # one-shot == sequential, and one point id per physical corner
    rc1, _ = tool('create_periodic_zones.py', 'lab.nmsh', 's1.nmsh', '(1,2)')
    rc2, _ = tool('create_periodic_zones.py', 's1.nmsh', 's2.nmsh', '(3,4)')
    ok = rc1 == 0 and rc2 == 0 and open(wpath('s2.nmsh'), 'rb').read() \
        == open(wpath('cpz.nmsh'), 'rb').read()
    report('converting (1,2) then (3,4) == converting both at once', ok)

    def aliased_ids(path):
        m = read_nmsh(path)
        ids = m.elems['v']['idx'].ravel()
        xyz = m.elems['v']['xyz'].reshape(-1, 3)
        key = xyz.view([('', 'V24')]).ravel()
        pairs = np.unique(np.rec.fromarrays([ids, key]))
        return int(pairs.size - np.unique(pairs.f0).size)
    report('no vertex id carries two different coordinates',
           aliased_ids(wpath('cpz.nmsh')) == 0
           and aliased_ids(wpath('cpz2.nmsh')) == 0)
    rc, _ = tool('create_periodic_zones.py', 'lab.nmsh', 'bad.nmsh', '(1,3)')
    report('mismatched facet counts refused, no output',
           rc != 0 and not os.path.exists(wpath('bad.nmsh')))
    rc, _ = tool('create_periodic_zones.py', 'lab.nmsh', 'bad2.nmsh', '(1,1)')
    report('identical labels refused', rc != 0)
else:
    skip('create_periodic_zones', 'scipy not installed')

# ---- T9: curved geometry against Neko's formulas ---------------------------
print('[T9] curved geometry (dofmap port): loop-port, midpoints, arcs')
from nekolight.geometry import (_gh_vertex_edge_extend, gll_xyz,   # noqa
                                apply_curves, GLL3, EDGE_MID_NODE, SYM2SLOT,
                                SYM_EDGE_NODES, ECYC_TO_SYM)
from nekolight import CURVE_DT, SIJK, jacobian_dets, CurveError  # noqa


def gh_loop(x):
    """gh_face_extend_3d(x, zg, n=3, gh_type=2), the Fortran loops verbatim
    on one scalar field x(i,j,k)."""
    n, zg = 3, GLL3

    def s(ii, i):
        return 0.5 * ((n - ii) * (1 - zg[i]) + (ii - 1) * (1 + zg[i])) / (n - 1)
    v = np.zeros((n, n, n))
    e = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ii in (1, n):
                    for jj in (1, n):
                        for kk in (1, n):
                            v[i, j, k] += s(ii, i) * s(jj, j) * s(kk, k) \
                                * x[ii - 1, jj - 1, kk - 1]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for jj in (1, n):
                    for kk in (1, n):
                        e[i, j, k] += s(jj, j) * s(kk, k) * (
                            x[i, jj - 1, kk - 1] - v[i, jj - 1, kk - 1])
                for ii in (1, n):
                    for kk in (1, n):
                        e[i, j, k] += s(ii, i) * s(kk, k) * (
                            x[ii - 1, j, kk - 1] - v[ii - 1, j, kk - 1])
                for ii in (1, n):
                    for jj in (1, n):
                        e[i, j, k] += s(ii, i) * s(jj, j) * (
                            x[ii - 1, jj - 1, k] - v[ii - 1, jj - 1, k])
    return e + v


rng = np.random.default_rng(3)
x = rng.standard_normal((3, 3, 3, 3))
ref = np.stack([gh_loop(x[..., c]) for c in range(3)], axis=-1)
out = _gh_vertex_edge_extend(x.transpose(2, 1, 0, 3)[None])[0] \
    .transpose(2, 1, 0, 3)
report('Gordon-Hall edge blend == Fortran loop port',
       np.abs(out - ref).max() < 1e-13)
corners = (2.0 * SIJK - 1.0).astype(float)[None] \
    + 0.1 * rng.standard_normal((1, 8, 3))
x27 = gll_xyz(corners)
cur = np.zeros(1, dtype=CURVE_DT)
cur['e'] = 1
for edge in (2, 7, 11):
    cur['type'][0, edge - 1] = 4
    cur['data'][0, edge - 1, :3] = rng.standard_normal(3)
apply_curves(x27, corners, cur, rows=np.array([0]))
corner_nodes = [0, 2, 8, 6, 18, 20, 26, 24]
report('midside points land on the edge midpoints, corners fixed',
       all(np.allclose(x27[0, EDGE_MID_NODE[e - 1]], cur['data'][0, e - 1, :3])
           for e in (2, 7, 11))
       and np.allclose(x27[0, corner_nodes], gll_xyz(corners)[0, corner_nodes]))
corners = (2.0 * SIJK - 1.0).astype(float)[None]
ok_all = True
for isid in range(1, 9):
    for R in (1.5, -1.5):
        x27 = gll_xyz(corners)
        cur = np.zeros(1, dtype=CURVE_DT)
        cur['e'] = 1
        cur['type'][0, isid - 1] = 3
        cur['data'][0, isid - 1, 0] = R
        apply_curves(x27, corners, cur, rows=np.array([0]))
        itmp = ECYC_TO_SYM[isid - 1]
        a, b = SYM_EDGE_NODES[itmp - 1]
        p1, p2 = corners[0, SYM2SLOT[a - 1]], corners[0, SYM2SLOT[b - 1]]
        mid = x27[0, EDGE_MID_NODE[isid - 1]]
        chord = p2[:2] - p1[:2]
        c = 0.5 * (p1[:2] + p2[:2])
        L = np.linalg.norm(chord)
        h = np.sqrt(R * R - (L / 2) ** 2)
        nrm = np.array([-chord[1], chord[0]]) / L
        on_circle = any(abs(np.linalg.norm(mid[:2] - (c + sg * h * nrm))
                            - abs(R)) < 1e-12 for sg in (1, -1))
        outward = np.sign(np.dot(mid[:2], nrm) - np.dot(c, nrm)) * \
            np.sign(np.dot(c, nrm)) > 0          # away from the centre
        ok_all = ok_all and on_circle and (outward == (R > 0)) \
            and np.allclose(x27[0, :, 2], gll_xyz(corners)[0, :, 2]) \
            and np.allclose(x27[0, corner_nodes],
                            gll_xyz(corners)[0, corner_nodes])
report('arcs: midpoint on the circle, R>0 bulges outward, planar, '
       'corners fixed', ok_all)
x27 = gll_xyz(corners)
cur = np.zeros(1, dtype=CURVE_DT)
cur['e'] = 1
cur['type'][0, 0] = 3
cur['data'][0, 0, 0] = 0.5                       # radius < half chord
try:
    apply_curves(x27, corners, cur, rows=np.array([0]))
    ok = False
except CurveError:
    ok = True
report('radius smaller than the chord raises CurveError (Neko aborts)', ok)
report('straight-sided unit cube: det J == 1 at all 27 nodes',
       np.allclose(jacobian_dets(gll_xyz(corners)), 1.0))
# two records arcing the same edge add up (Neko's addtnsr accumulates)
x_a = gll_xyz(corners)
cur2 = np.zeros(2, dtype=CURVE_DT)
cur2['e'] = 1
cur2['type'][:, 0] = 3
cur2['data'][0, 0, 0] = 3.0
cur2['data'][1, 0, 0] = 5.0
apply_curves(x_a, corners, cur2, rows=np.array([0, 0]))
x_b = gll_xyz(corners)
apply_curves(x_b, corners, cur2[:1], rows=np.array([0]))
x_c = gll_xyz(corners)
apply_curves(x_c, corners, cur2[1:], rows=np.array([0]))
report('duplicate arc records accumulate their perturbations',
       np.allclose(x_a - gll_xyz(corners),
                   (x_b - gll_xyz(corners)) + (x_c - gll_xyz(corners))))
from nekolight import facet_normals  # noqa: E402
nrm = facet_normals(gll_xyz(corners))[0]
report('facet normals point outward on the unit cube',
       np.allclose(nrm, [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                         [0, 0, -1], [0, 0, 1]]))

# ---- T10: meshview exports ------------------------------------------------
print('[T10] meshview: .vtu export (3D, 2D slab, --curved)')
for name, path, extra in (('hemi', hemi_ref, []),
                          ('2d_cylinder slab', cyl2d, []),
                          ('cylinder --curved', cyl, ['--curved'])):
    if not os.path.exists(path):
        skip('meshview %s' % name, 'mesh not found')
        continue
    rc, out = tool('meshview.py', path, '--export', 'skin.vtu', *extra)
    ok = rc == 0 and os.path.exists(wpath('skin.vtu'))
    if ok:
        txt = open(wpath('skin.vtu'), 'rb').read(400).decode('ascii', 'replace')
        ok = 'NumberOfCells' in txt
    report('meshview %s: skin export' % name, ok, out[-200:])

# ---- T11: gmsh2nmsh ---------------------------------------------------------
print('[T11] gmsh2nmsh: Gmsh .msh -> .nmsh (formats, orientation, curves, '
      'zones, periodicity, extrusion)')
from nekolight import (read_msh, merged_vertex_ids, pos_of_elid_map,   # noqa
                       dual_graph, extrude_2d, FACE_RE2, layer_planes)
GM = os.path.join(HERE, 'tests', 'gmsh')


def gconv(msh, out, *extra):
    return tool('gmsh2nmsh.py', os.path.join(GM, msh), wpath(out), *extra)


def checker(path, *extra):
    rc, out = tool('mesh_checker.py', wpath(path), *extra)
    return rc == 0 and 'Done' in out, out


def label_counts(out):
    return {int(a): int(b) for a, b in
            re.findall(r'label\s+(\d+) :\s+(\d+) facets', out)}


def connectivity(path):
    """(unique merged points, sorted dual-graph degrees, periodic facets,
    {label: sorted facet centres}) of a .nmsh after Neko's periodic merge."""
    m = read_nmsh(wpath(path))
    pos = pos_of_elid_map(m.nelv, m.elems)
    ext = m.gdim == 2
    mm = extrude_2d(m) if ext else m
    merged = merged_vertex_ids(mm, pos, extruded=ext)
    A = dual_graph(merged)
    deg = np.sort(np.diff(A.indptr)) if hasattr(A, 'indptr') else None
    xyz = mm.elems['v']['xyz']
    cent = {}
    z7 = m.zones[m.zones['t'] == 7]
    for lb in np.unique(z7['p_f']):
        r = z7[z7['p_f'] == lb]
        slots = FACE_RE2[r['f'].astype(np.int64) - 1] if not ext else \
            FACE_RE2[r['f'].astype(np.int64) - 1]
        c = xyz[pos[r['e'].astype(np.int64)][:, None], slots].mean(axis=1)
        cent[int(lb)] = np.round(np.sort(c, axis=0), 9)
    return int(np.unique(merged).size), deg, int((m.zones['t'] == 5).sum()), cent


# 1. the four format variants of the same box give one file
outs = []
for v in ('v22a', 'v22b', 'v41a', 'v41b'):
    rc, out = gconv('box8_%s.msh' % v, 'gbox_%s.nmsh' % v)
    outs.append((rc, out))
ok_all = all(rc == 0 for rc, _ in outs)


def same_mesh(a, b, rtol=1e-12):
    """Same records up to the last bits of the coordinates (Gmsh writes 16
    significant digits in ASCII files, exact doubles in binary ones)."""
    A, B = read_nmsh(wpath(a)), read_nmsh(wpath(b))
    return A.nelv == B.nelv and np.array_equal(A.elems['id'], B.elems['id']) \
        and np.array_equal(A.elems['v']['idx'], B.elems['v']['idx']) \
        and np.allclose(A.elems['v']['xyz'], B.elems['v']['xyz'], rtol=rtol, atol=1e-13) \
        and np.array_equal(A.zones, B.zones) and A.curves.shape == B.curves.shape \
        and np.array_equal(A.curves['e'], B.curves['e']) \
        and np.array_equal(A.curves['type'], B.curves['type']) \
        and np.allclose(A.curves['data'], B.curves['data'], rtol=rtol, atol=1e-13)


same = ok_all and open(wpath('gbox_v22a.nmsh'), 'rb').read() == \
    open(wpath('gbox_v41a.nmsh'), 'rb').read() and \
    all(same_mesh('gbox_%s.nmsh' % v, 'gbox_v41a.nmsh') for v in ('v22b', 'v41b'))
report('gmsh2nmsh: msh 2.2/4.1 ASCII/binary give the same mesh (ASCII '
       'byte-identical, binary to the last bit)', same,
       ''.join(o[-300:] for _, o in outs if _ != 0))
ok, cout = checker('gbox_v41a.nmsh', '--jacobian')
lc = label_counts(outs[2][1])
report('gmsh2nmsh: box passes the checker with labels 1..6 x 9 facets',
       ok and lc == {i: 9 for i in range(1, 7)}, cout[-300:] + str(lc))
# 2. the same box from genmeshbox: identical connectivity and zone geometry
rc, gout = tool('genmeshbox.py', 0, 3, 0, 2, 0, 1, 3, 3, 3, wpath('genbox.nmsh'))
if rc == 0:
    a = connectivity('gbox_v41a.nmsh'); b = connectivity('genbox.nmsh')
    ok = a[0] == b[0] == 64 and np.array_equal(a[1], b[1]) and \
        set(a[3]) == set(b[3]) and all(np.allclose(a[3][k], b[3][k]) for k in a[3])
    report('gmsh2nmsh: box equals genmeshbox (points, dual graph, zone '
           'facet centres)', ok, '%s vs %s' % (a[:3], b[:3]))
else:
    skip('gmsh2nmsh vs genmeshbox', 'genmeshbox failed: ' + gout[-100:])
# 3. periodic box: $Periodic (x and z) == genmeshbox periodic == --periodic pairs
rc1, o1 = gconv('boxper8_v41a.msh', 'gboxper.nmsh')
rc2, o2 = gconv('boxper27_v22b.msh', 'gboxper27.nmsh')
rc3, o3 = gconv('box8_v41a.msh', 'gboxlab.nmsh', '--periodic', '1:2', '--periodic', '5:6')
rc4, o4 = tool('genmeshbox.py', 0, 3, 0, 2, 0, 1, 3, 3, 3, '.true.', '.false.',
               '.true.', wpath('genboxper.nmsh'))
if rc1 == 0 and rc2 == 0 and rc3 == 0 and rc4 == 0:
    c = [connectivity(p) for p in ('gboxper.nmsh', 'gboxper27.nmsh', 'gboxlab.nmsh',
                                  'genboxper.nmsh')]
    ok = all(x[0] == 36 and x[2] == 36 and np.array_equal(x[1], c[0][1]) for x in c)
    ok = ok and all(set(x[3]) == {3, 4} and np.allclose(x[3][3], c[0][3][3])
                    and np.allclose(x[3][4], c[0][3][4]) for x in c)
    report('gmsh2nmsh: periodic box ($Periodic, hex27 2.2-binary, --periodic '
           'pairs) equals genmeshbox px,pz', ok, str([x[:3] for x in c]))
    ok, cout = checker('gboxper.nmsh', '--jacobian')
    report('gmsh2nmsh: periodic box passes the checker', ok, cout[-300:])
else:
    report('gmsh2nmsh: periodic box conversions run', False,
           (o1 + o2 + o3 + o4)[-400:])
# 4. curved annulus: hex27 and hex20 identical; extruded 2D quad9 equals the
#    Gmsh-extruded 3D mesh (points, curved edges, zone geometry, Jacobians)
rc1, o1 = gconv('annulus3d_h27_v41b.msh', 'ann27.nmsh')
rc2, o2 = gconv('annulus3d_h20_v41a.msh', 'ann20.nmsh')
rc3, o3 = gconv('annulus2d_q9_v22a.msh', 'ann2dx.nmsh', '--extrude', 0, 0.5, 2,
                '--zbc', 5, 6)
rc4, o4 = gconv('annulus3d_h8_v41a.msh', 'ann8.nmsh')
ok = rc1 == 0 and rc2 == 0 and same_mesh('ann27.nmsh', 'ann20.nmsh', rtol=1e-9)
report('gmsh2nmsh: hex27 and hex20 versions give the same mesh', ok,
       (o1 + o2)[-300:])
if rc1 == 0 and rc3 == 0 and rc4 == 0:
    ok27, c27 = checker('ann27.nmsh', '--jacobian')
    okx, cx = checker('ann2dx.nmsh', '--jacobian')
    ok8, c8 = checker('ann8.nmsh', '--jacobian')
    jac = lambda out: float(re.search(r'curved geometry.*?:\s+([-0-9.eE+]+)', out).group(1))
    a, b = connectivity('ann27.nmsh'), connectivity('ann2dx.nmsh')
    ok = ok27 and okx and ok8 and a[0] == b[0] == 72 and np.array_equal(a[1], b[1]) \
        and set(a[3]) == set(b[3]) and all(np.allclose(a[3][k], b[3][k]) for k in a[3]) \
        and abs(jac(c27) - jac(cx)) < 1e-12 and 'Curved elements: 20' in c27 \
        and 'Curved elements: 20' in cx and '40 midside' in cx
    report('gmsh2nmsh: extruded 2D quad9 == Gmsh-extruded hex27 (geometry, '
           'zones, curved Jacobian)', ok, (c27 + cx)[-400:])
    ok = 'Min Jacobian (curved' in c27 and jac(c27) < float(
        re.search(r'straight-sided\):\s+([-0-9.eE+]+)', c27).group(1))
    report('gmsh2nmsh: midside curves are applied (curved Jacobian differs '
           'from straight)', ok, c27[-300:])
else:
    report('gmsh2nmsh: annulus conversions run', False, (o1 + o3 + o4)[-400:])
# 5. 2D output: Neko's own one-layer extrusion of the 2D file equals
#    --extrude with one periodic layer
rc1, o1 = gconv('rect2d_per_q4_v41a.msh', 'rect2d.nmsh')
rc2, o2 = gconv('rect2d_per_q4_v41a.msh', 'rect3d1.nmsh', '--extrude', 0, 1, 1,
                '--zbc', 'periodic')
if rc1 == 0 and rc2 == 0:
    m2 = read_nmsh(wpath('rect2d.nmsh'))
    a, b = connectivity('rect2d.nmsh'), connectivity('rect3d1.nmsh')
    ok = m2.gdim == 2 and a[0] == b[0] == 12 and np.array_equal(a[1], b[1]) \
        and a[2] == 4 and b[2] == 4 + 16 and set(a[3]) == set(b[3]) == {3, 4}
    report('gmsh2nmsh: 2D file (Neko slab) == one periodic extruded layer',
           ok, '%s %s' % (a[:3], b[:3]))
    ok2, c2 = checker('rect2d.nmsh')
    report('gmsh2nmsh: 2D periodic rectangle passes the checker',
           ok2 and 'periodic faces: 4' in c2, c2[-300:])
else:
    report('gmsh2nmsh: 2D conversions run', False, (o1 + o2)[-400:])
# 6. layer distributions and labelled z-faces
rc, out = gconv('rect2d_per_q9_v41a.msh', 'rect3d.nmsh', '--extrude', 0, 2, 3,
                '--gain', 2, '--zbc', 5, 6, '--no-msh-periodic', '--periodic', '1:2')
if rc == 0:
    m3 = read_nmsh(wpath('rect3d.nmsh'))
    z = np.unique(np.round(m3.elems['v']['xyz'][:, :, 2], 12))
    ok = np.allclose(z, layer_planes(0, 2, 3, 2.0)) and m3.nelv == 24 and \
        label_counts(out) == {3: 12, 4: 12, 5: 8, 6: 8} and \
        (m3.zones['t'] == 5).sum() == 12 and \
        np.array_equal(m3.elems['id'], np.arange(1, 25))
    okc, cout = checker('rect3d.nmsh', '--jacobian')
    report('gmsh2nmsh: --gain planes, labelled z-faces, --periodic per layer',
           ok and okc, out[-300:] + cout[-200:])
else:
    report('gmsh2nmsh: --gain extrusion runs', False, out[-400:])
with open(wpath('planes.txt'), 'w') as f:
    f.write('0\n0.1\n0.3\n0.6\n1\n')
rc, out = gconv('rect2d_per_q4_v41a.msh', 'rectz.nmsh', '--zfile', wpath('planes.txt'),
                '--zbc', 'periodic')
if rc == 0:
    m3 = read_nmsh(wpath('rectz.nmsh'))
    z = np.unique(np.round(m3.elems['v']['xyz'][:, :, 2], 12))
    c = connectivity('rectz.nmsh')
    report('gmsh2nmsh: --zfile planes and periodic z with lateral periodicity',
           np.allclose(z, [0, 0.1, 0.3, 0.6, 1]) and c[0] == 12 * 4 and
           c[2] == 2 * (2 * 4 + 8), str(c[:3]))
else:
    report('gmsh2nmsh: --zfile extrusion runs', False, out[-400:])
# 7. error paths
rc, out = gconv('box8_untagged_v41a.msh', 'bu.nmsh')
report('gmsh2nmsh: untagged boundary facets are refused',
       rc != 0 and 'no physical group' in out and not os.path.exists(wpath('bu.nmsh')),
       out[-300:])
rc, out = gconv('box8_untagged_v41a.msh', 'bu.nmsh', '--untagged', 6)
report('gmsh2nmsh: --untagged labels them', rc == 0 and label_counts(out).get(6) == 9,
       out[-300:])
rc, out = gconv('tets_v41a.msh', 't.nmsh')
report('gmsh2nmsh: a tet mesh is refused with the cell types named',
       rc != 0 and 'tet4' in out, out[-300:])
rc, out = gconv('box8_v41a.msh', 'x.nmsh', '--extrude', 0, 1, 2, '--zbc', 'periodic')
report('gmsh2nmsh: --extrude on a 3D mesh is refused', rc != 0, out[-200:])
rc, out = gconv('box8_v41a.msh', 'x.nmsh', '--label', 'outlet=25')
report('gmsh2nmsh: labels outside 1..20 are refused', rc != 0 and '1..20' in out,
       out[-200:])
rc, out = gconv('box8_v41a.msh', 'x.nmsh', '--periodic', '1:3')
report('gmsh2nmsh: --periodic between non-translated zones is refused', rc != 0,
       out[-200:])
# 8. left-handed cells (mirrored coordinates) are fixed; reader agrees with meshio
if have('meshio'):
    import meshio
    mm = meshio.read(os.path.join(GM, 'box8_v41a.msh'))
    mm.points[:, 0] *= -1.0
    meshio.write(wpath('mirror.msh'), mm, file_format='gmsh', binary=False)
    rc, out = tool('gmsh2nmsh.py', wpath('mirror.msh'), wpath('mirror.nmsh'))
    okc, cout = checker('mirror.nmsh', '--jacobian')
    report('gmsh2nmsh: left-handed hexahedra are mirrored and pass the checker',
           rc == 0 and 'mirrored' in out and okc and label_counts(out) == {i: 9 for i in range(1, 7)},
           (out + cout)[-400:])
    ok = True
    for v in ('v22a', 'v22b', 'v41a', 'v41b'):
        g = read_msh(os.path.join(GM, 'box8_%s.msh' % v))
        ref = meshio.read(os.path.join(GM, 'box8_%s.msh' % v))
        idx = g.node_index()
        hx = np.concatenate([b.nodes for b in g.blocks if b.etype == 5])
        ok = ok and np.allclose(g.xyz[idx(g.node_tags)], ref.points) and \
            np.array_equal(np.sort(idx(hx), axis=0), np.sort(ref.cells_dict['hexahedron'], axis=0))
    report('gmsh reader agrees with meshio on all four format variants', ok)
else:
    skip('gmsh2nmsh mirrored / meshio comparison', 'meshio not installed')

# 9. review cases: genuine hex20/quad8 fixtures, refusals that keep Neko safe
rc, out = gconv('annulus3d_h20_v41a.msh', 'h20check.nmsh')
report('gmsh2nmsh: the hex20 fixture really is hex20/quad8', rc == 0 and 'hex20' in out
       and 'quad8' in out, out[-200:])
rc1, o1 = gconv('annulus2d_q8_v41a.msh', 'a2q8.nmsh')
rc2, o2 = gconv('annulus2d_q9_v22a.msh', 'a2q9.nmsh')
report('gmsh2nmsh: quad8 and quad9 versions give the same 2D mesh',
       rc1 == 0 and rc2 == 0 and 'quad8' in o1 and same_mesh('a2q8.nmsh', 'a2q9.nmsh', rtol=1e-9),
       (o1 + o2)[-300:])
rc, out = gconv('box2z_per_v41a.msh', 'b2z.nmsh')
report('gmsh2nmsh: a periodic direction two elements thick is refused',
       rc != 0 and 'more than two elements' in out and not os.path.exists(wpath('b2z.nmsh')),
       out[-300:])
rc, out = gconv('rect2d_per_q4_v41a.msh', 'x2.nmsh', '--extrude', 0, 1, 2, '--zbc', 'periodic')
report('gmsh2nmsh: two periodic layers are refused (n2to3 needs three)',
       rc != 0 and 'layers' in out, out[-300:])
rc, out = gconv('nofrag_v41a.msh', 'nf.nmsh', '--untagged', 7)
report('gmsh2nmsh: a cracked interface (unfragmented volumes) is refused',
       rc != 0 and 'coincide' in out and not os.path.exists(wpath('nf.nmsh')), out[-300:])
rc, out = gconv('badhex_v22a.msh', 'bh.nmsh', '--untagged', 1)
report('gmsh2nmsh: positive corner but negative GLL Jacobian is refused',
       rc != 0 and 'GLL Jacobian' in out and not os.path.exists(wpath('bh.nmsh')), out[-300:])
txt = open(os.path.join(GM, 'boxper8_v41a.msh')).read()
txt = txt.replace('16 1 0 0 3 0 1 0 0 0 0 1 0 0 0 0 1', '16 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')
open(wpath('identity.msh'), 'w').write(txt)
rc, out = tool('gmsh2nmsh.py', wpath('identity.msh'), wpath('identity.nmsh'))
report('gmsh2nmsh: a $Periodic link with zero translation is refused',
       rc != 0 and 'zero translation' in out, out[-300:])
# 2.2 file with one (physical) tag per element, and sparse node tags
lines = open(os.path.join(GM, 'box8_v22a.msh')).read().split('\n')
out1, out2, sec = [], [], None
S = 100000000
for ln in lines:
    if ln.startswith('$'):
        sec = ln; out1.append(ln); out2.append(ln); continue
    p = ln.split()
    l1, l2 = ln, ln
    if sec == '$Elements' and len(p) > 3:
        nt = int(p[2])
        if nt == 2:
            l1 = ' '.join(p[:2] + ['1', p[3]] + p[5:])
        l2 = ' '.join(p[:3 + nt] + [str(int(v) + S) for v in p[3 + nt:]])
    elif sec == '$Nodes' and len(p) == 4:
        l2 = ' '.join([str(int(p[0]) + S)] + p[1:])
    out1.append(l1); out2.append(l2)
open(wpath('ntags1.msh'), 'w').write('\n'.join(out1))
open(wpath('sparse.msh'), 'w').write('\n'.join(out2))
rc, out = tool('gmsh2nmsh.py', wpath('ntags1.msh'), wpath('ntags1.nmsh'))
report('gmsh2nmsh: msh 2.2 with only the physical tag per element converts',
       rc == 0 and label_counts(out) == {i: 9 for i in range(1, 7)}, out[-300:])
import time as _time
t0 = _time.time()
rc, out = tool('gmsh2nmsh.py', wpath('sparse.msh'), wpath('sparse.nmsh'))
report('gmsh2nmsh: node tags offset by 1e8 cost nothing (sorted lookup)',
       rc == 0 and _time.time() - t0 < 10.0 and same_mesh('sparse.nmsh', 'gbox_v22a.nmsh'),
       out[-200:] + ' %.1fs' % (_time.time() - t0))
rc, out = gconv('rect2d_per_q4_v41a.msh', 'x.nmsh', '--zfile', wpath('nosuch.txt'), '--zbc', 5, 6)
report('gmsh2nmsh: a missing --zfile is a clean error', rc != 0 and 'Traceback' not in out
       and 'cannot read' in out, out[-200:])
rc, out = gconv('box8_v41a.msh', '../nonexistent_dir/x.nmsh')
report('gmsh2nmsh: an unwritable output directory is a clean error',
       rc != 0 and 'Traceback' not in out and 'output directory' in out, out[-200:])
rc, out = gconv('rect2d_per_q4_v41a.msh', 'x.nmsh', '--zfile', wpath('planes.txt'), '--gain', 2,
                '--zbc', 5, 6)
report('gmsh2nmsh: --gain together with --zfile is refused', rc != 0 and 'gain' in out,
       out[-200:])
if have('meshio'):
    mm = meshio.read(os.path.join(GM, 'boxper8_v41a.msh'))
    sel = np.flatnonzero((np.abs(mm.points[:, 0] - 3) < 1e-9) & (mm.points[:, 1] > 0.1)
                         & (mm.points[:, 1] < 1.9) & (mm.points[:, 2] > 0.1) & (mm.points[:, 2] < 0.9))
    mm.points[sel[0], 1] += 1e-5
    meshio.write(wpath('pert.msh'), mm, file_format='gmsh', binary=False)
    rc, out = tool('gmsh2nmsh.py', wpath('pert.msh'), wpath('pert.nmsh'))
    report('gmsh2nmsh: a partly matching periodic surface is refused, not '
           'turned into a wall', rc != 0 and 'no partner' in out, out[-300:])

# ---- summary ---------------------------------------------------------------
nfail = sum(1 for _, ok in RESULTS if not ok)
print('\n%d checks, %d failed' % (len(RESULTS), nfail))
print('RESULT: %s' % ('PASS' if nfail == 0 else 'FAIL'))
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if nfail else 0)
