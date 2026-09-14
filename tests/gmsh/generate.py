"""Generate the Gmsh fixtures of run_tests.py with the Gmsh Python API (pip install gmsh); run from this directory.
import gmsh, numpy as np, sys
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.Verbosity", 1)

def write_all(base, versions=((2.2, 0, 'v22a'), (2.2, 1, 'v22b'), (4.1, 0, 'v41a'), (4.1, 1, 'v41b'))):
    for ver, binary, suf in versions:
        gmsh.option.setNumber("Mesh.MshFileVersion", ver)
        gmsh.option.setNumber("Mesh.Binary", binary)
        gmsh.write('%s_%s.msh' % (base, suf))

def tag_box_surfaces(lx, ly, lz, names=('inlet', 'outlet', 'ymin', 'ymax', 'zmin', 'zmax'), skip=()):
    for dim, tag in gmsh.model.getEntities(2):
        c = gmsh.model.occ.getCenterOfMass(2, tag)
        if abs(c[0]) < 1e-9: p = 1
        elif abs(c[0] - lx) < 1e-9: p = 2
        elif abs(c[1]) < 1e-9: p = 3
        elif abs(c[1] - ly) < 1e-9: p = 4
        elif abs(c[2]) < 1e-9: p = 5
        else: p = 6
        if p in skip: continue
        gmsh.model.addPhysicalGroup(2, [tag], p, name=names[p - 1])

def transfinite_all(n):
    for dim, tag in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(tag, n + 1)
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)
    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(tag)

# (a) box, first order hex8, all six sides tagged 1..6
gmsh.model.add("box"); b = gmsh.model.occ.addBox(0, 0, 0, 3, 2, 1); gmsh.model.occ.synchronize()
transfinite_all(3); tag_box_surfaces(3, 2, 1); gmsh.model.addPhysicalGroup(3, [b], 10, name='fluid')
gmsh.model.mesh.generate(3); write_all('box8')
# (a2) same box, one side (zmax) left untagged
gmsh.model.add("box_untagged"); b = gmsh.model.occ.addBox(0, 0, 0, 3, 2, 1); gmsh.model.occ.synchronize()
transfinite_all(3); tag_box_surfaces(3, 2, 1, skip=(6,)); gmsh.model.addPhysicalGroup(3, [b], 10, name='fluid')
gmsh.model.mesh.generate(3); write_all('box8_untagged', versions=((4.1, 0, 'v41a'),))
# (a3) periodic box: x-periodic (outlet -> inlet) and z-periodic, via setPeriodic; periodic sides also tagged
gmsh.model.add("boxper"); b = gmsh.model.occ.addBox(0, 0, 0, 3, 2, 1); gmsh.model.occ.synchronize()
transfinite_all(3); tag_box_surfaces(3, 2, 1); gmsh.model.addPhysicalGroup(3, [b], 10, name='fluid')
def surf_at(axis, val):
    for dim, tag in gmsh.model.getEntities(2):
        c = gmsh.model.occ.getCenterOfMass(2, tag)
        if abs(c[axis] - val) < 1e-9: return tag
gmsh.model.mesh.setPeriodic(2, [surf_at(0, 3.0)], [surf_at(0, 0.0)], [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
gmsh.model.mesh.setPeriodic(2, [surf_at(2, 1.0)], [surf_at(2, 0.0)], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
gmsh.model.mesh.generate(3); write_all('boxper8')
gmsh.model.mesh.setOrder(2); write_all('boxper27', versions=((4.1, 0, 'v41a'), (2.2, 1, 'v22b')))
# (b) curved: quarter annulus (r 1..2) in 2D, then extruded to hex, second order complete (27) and incomplete (20)
def annulus2d(name):
    gmsh.model.add(name)
    p0 = gmsh.model.occ.addPoint(0, 0, 0); p1 = gmsh.model.occ.addPoint(1, 0, 0); p2 = gmsh.model.occ.addPoint(2, 0, 0)
    p3 = gmsh.model.occ.addPoint(0, 2, 0); p4 = gmsh.model.occ.addPoint(0, 1, 0)
    l1 = gmsh.model.occ.addLine(p1, p2); a1 = gmsh.model.occ.addCircleArc(p2, p0, p3); l2 = gmsh.model.occ.addLine(p3, p4); a2 = gmsh.model.occ.addCircleArc(p4, p0, p1)
    cl = gmsh.model.occ.addCurveLoop([l1, a1, l2, a2]); s = gmsh.model.occ.addPlaneSurface([cl]); gmsh.model.occ.synchronize()
    for t, n in ((l1, 3), (l2, 3), (a1, 5), (a2, 5)): gmsh.model.mesh.setTransfiniteCurve(t, n + 1)
    gmsh.model.mesh.setTransfiniteSurface(s); gmsh.model.mesh.setRecombine(2, s)
    return s, (l1, a1, l2, a2)
s, (l1, a1, l2, a2) = annulus2d("annulus2d")
gmsh.model.addPhysicalGroup(1, [l1], 1, name='bottom'); gmsh.model.addPhysicalGroup(1, [a1], 2, name='outer'); gmsh.model.addPhysicalGroup(1, [l2], 3, name='left'); gmsh.model.addPhysicalGroup(1, [a2], 4, name='inner')
gmsh.model.addPhysicalGroup(2, [s], 10, name='fluid')
gmsh.model.mesh.generate(2); write_all('annulus2d_q4', versions=((4.1, 0, 'v41a'), (2.2, 0, 'v22a')))
gmsh.model.mesh.setOrder(2); write_all('annulus2d_q9', versions=((4.1, 0, 'v41a'), (2.2, 0, 'v22a'), (4.1, 1, 'v41b')))
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1); gmsh.model.mesh.setOrder(2); write_all('annulus2d_q8', versions=((4.1, 0, 'v41a'),)); gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
# 3D extruded annulus (gmsh extrusion, 2 layers), hex27 and hex20
s, (l1, a1, l2, a2) = annulus2d("annulus3d")
ext = gmsh.model.occ.extrude([(2, s)], 0, 0, 0.5, numElements=[2], recombine=True); gmsh.model.occ.synchronize()
vol = [e[1] for e in ext if e[0] == 3][0]
for dim, tag in gmsh.model.getEntities(1):
    if tag not in (l1, a1, l2, a2):
        gmsh.model.mesh.setTransfiniteCurve(tag, 4)
for dim, tag in gmsh.model.getEntities(2): gmsh.model.mesh.setTransfiniteSurface(tag); gmsh.model.mesh.setRecombine(2, tag)
gmsh.model.mesh.setTransfiniteVolume(vol)
# physical surfaces by centre of mass: bottom (z=0)=5, top=6, inner cyl (r~1)=4, outer (r~2)=2, y=0 plane=1, x=0 plane=3
for dim, tag in gmsh.model.getEntities(2):
    c = gmsh.model.occ.getCenterOfMass(2, tag); r = np.hypot(c[0], c[1])
    if abs(c[2]) < 1e-9: p = 5
    elif abs(c[2] - 0.5) < 1e-9: p = 6
    elif abs(c[1]) < 1e-9: p = 1
    elif abs(c[0]) < 1e-9: p = 3
    elif r < 1.5: p = 4
    else: p = 2
    gmsh.model.addPhysicalGroup(2, [tag], p, name=['ymin', 'outer', 'xmin', 'inner', 'zmin', 'zmax'][p - 1])
gmsh.model.addPhysicalGroup(3, [vol], 10, name='fluid')
gmsh.model.mesh.generate(3); write_all('annulus3d_h8', versions=((4.1, 0, 'v41a'),))
gmsh.model.mesh.setOrder(2); write_all('annulus3d_h27', versions=((4.1, 0, 'v41a'), (4.1, 1, 'v41b'), (2.2, 0, 'v22a'), (2.2, 1, 'v22b')))
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1); gmsh.model.mesh.setOrder(2); write_all('annulus3d_h20', versions=((4.1, 0, 'v41a'),)); gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
# (d) 2D rectangle, x-periodic via setPeriodic on curves, tagged sides: 1 left 2 right 3 bottom 4 top
gmsh.model.add("rect2d"); r = gmsh.model.occ.addRectangle(0, 0, 0, 4, 1); gmsh.model.occ.synchronize()
for dim, tag in gmsh.model.getEntities(1):
    c = gmsh.model.occ.getCenterOfMass(1, tag)
    n = 4 if abs(c[1]) < 1e-9 or abs(c[1] - 1) < 1e-9 else 2
    gmsh.model.mesh.setTransfiniteCurve(tag, n + 1)
    p = 1 if abs(c[0]) < 1e-9 else 2 if abs(c[0] - 4) < 1e-9 else 3 if abs(c[1]) < 1e-9 else 4
    gmsh.model.addPhysicalGroup(1, [tag], p, name=['left', 'right', 'bottom', 'top'][p - 1])
gmsh.model.mesh.setTransfiniteSurface(r); gmsh.model.mesh.setRecombine(2, r); gmsh.model.addPhysicalGroup(2, [r], 10, name='fluid')
def curve_at(axis, val):
    for dim, tag in gmsh.model.getEntities(1):
        c = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(c[axis] - val) < 1e-9: return tag
gmsh.model.mesh.setPeriodic(1, [curve_at(0, 4.0)], [curve_at(0, 0.0)], [1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
gmsh.model.mesh.generate(2); write_all('rect2d_per_q4', versions=((4.1, 0, 'v41a'), (2.2, 0, 'v22a')))
gmsh.model.mesh.setOrder(2); write_all('rect2d_per_q9', versions=((4.1, 0, 'v41a'),))
# (e) a tet mesh for the error path
gmsh.model.add("tets"); b = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1); gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5); gmsh.model.addPhysicalGroup(3, [b], 10); gmsh.model.mesh.generate(3); write_all('tets', versions=((4.1, 0, 'v41a'),))
gmsh.finalize()
print('meshes written')
