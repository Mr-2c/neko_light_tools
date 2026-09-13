```text
     _  __  ____  __ __  ____
    / |/ / / __/ / //_/ / __ \
   /    / / _/  / ,<   / /_/ /
  /_/|_/ /___/ /_/|_|  \____/
```

# Neko light tools

Standalone, CPU-only mesh utilities for [Neko](https://github.com/ExtremeFLOW/neko),
in scientific Python.  They reproduce the mesh tools shipped under Neko's
`contrib/` -- byte-exact where a Neko-written reference exists -- without
a Neko build, MPI or a Fortran compiler: `python3` with numpy is the only
hard requirement (scipy for partitioning and periodic-zone creation).
One shared library (`nekolight/`) holds every byte layout, topology table
and algorithm exactly once; six thin command-line tools sit on top of it.

| Tool | What it does | Needs |
|---|---|---|
| `rea2nbin.py` | NEKTON `.re2` (2D or 3D) → `.nmsh`, byte-exact with `contrib/rea2nbin` (periodic BCs, curved elements) | numpy |
| `genmeshbox.py` | Box-mesh generator, byte-exact with `contrib/genmeshbox`; O(chunk + nelx+nely+nelz) memory | numpy |
| `mesh_checker.py` | `.nmsh` validation and diagnostics as `contrib/mesh_checker` reports them: sizes after the periodic merge, zones with normal alignment, unlabelled external faces, curve records, `--jacobian` on Neko's curved geometry, `--write-zone-indices` | numpy |
| `prepart.py` | Mesh partitioner (spectral / METIS / geometric / grid) writing a reordered `.nmsh` whose linear read reproduces the partition exactly | numpy + scipy (optional: pymetis) |
| `create_periodic_zones.py` | Turn pairs of labelled zones into periodic zones, as `contrib/create_periodic_zones` | numpy + scipy |
| `meshview.py` | Interactive mesh viewer: external-surface extraction + PyVista, `.vtu` export for ParaView, matplotlib fallback, `--curved` | numpy (optional: pyvista, matplotlib) |

All tools accept 2D (quad) as well as 3D (hex) meshes, exactly as Neko
does: a 2D file is partitioned and written as quads, and checked/viewed as
the one-element-thick slab Neko's reader extrudes it into.

Install nothing: run the scripts from a checkout (`python3 prepart.py ...`),
or `pip install .` to get them on your `PATH` (`pip install '.[all]'` adds
the optional extras).

## Validation

```
python3 run_tests.py /path/to/neko
```

runs the suite against a Neko checkout (its example meshes are the corpus
and the golden files: `examples/hemi/hemi.re2` with the `hemi.nmsh` Neko
wrote from it, the shipped box meshes).  Oracles are Neko's own source --
`datadist.f90`'s block formula, the closed-form point/face/edge counts of a
box, the `.re2` record layout -- never the tools' own code.  Optional
dependencies that are missing skip their tests rather than failing them.
Every generated mesh is re-read by `mesh_checker.py` before it counts as a
pass.  The suite makes it easy to add a mesh you trust and confirm the
tools reproduce what Neko does with it.

## Design

* **Functions over numpy arrays.**  The one shared data structure is the
  `Mesh` named tuple (the raw `.nmsh` record arrays); there are no class
  hierarchies.  Anyone with basic numpy should be able to read any module
  top to bottom.
* **All format knowledge lives in `nekolight/formats.py`** -- the `.nmsh`,
  `.re2` and `.fld` byte layouts and the vertex-ordering tables
  (`FACE_RE2`, `QFACE_RE2`, `EDGE_RE2`, `FACET_MAP`, `SIJK`), each defined
  exactly once and documented where it is defined.  `nekolight/geometry.py`
  carries the numbering tables of Neko's curved-geometry code.
* **Atomic outputs.**  Every writer refuses an output path that names one
  of its inputs, writes to a temporary file, and renames it into place only
  after the entire run (including validation) has succeeded.  A failed run
  can never truncate an input or leave a partial output behind.
* **Validate and refuse.**  Out-of-range zone/curve references, bad labels,
  non-permutation element ids and truncated sections are hard errors in
  every tool.  Nothing silently drops or rewrites a malformed record.
  Records Neko accepts but ignores (legacy zone types, curve types 1/2) are
  carried through with a note.
* **Exact Neko semantics where it matters.**  The converter reproduces
  Neko's first-appearance, bit-exact point numbering and its fixed 3-sweep
  periodic id merge (byte-exact output); the checker applies the periodic
  ids exactly as Neko's reader does (`apply_periodic_facet`, last record
  wins) and reports the same numbers; the partitioners produce block sizes
  exactly matching Neko's `linear_dist_t` (`L = nelv/P`, the first
  `nelv mod P` ranks get `L+1`), so rank *i* of a *P*-rank run receives
  exactly partition *i* -- including when `nelv mod P /= 0`.
* **int64 where 8·nelv can overflow, uint64 for packed sort keys.**

## The partitioner

```
prepart.py mesh.nmsh P [out.nmsh] [--backend spectral|metis|geometric] [options]
prepart.py mesh.nmsh [out.nmsh] --grid NX,NY,NZ
```

All backends share the periodic-merged, shared-vertex-weighted element dual
graph (one sparse E·Eᵀ product) and write contiguous per-partition blocks in
Neko's exact linear-read sizes.  Inside every block the input order is kept,
so a rank's elements carry a strictly increasing run of new global ids that
also follows the original numbering.

* `spectral` (default): recursive spectral bisection.  Every sub-graph is
  split by the best of its lowest Laplacian eigenvectors (the one giving
  the smallest cut with both sides connected), computed by LOBPCG with an
  aggregation-multigrid preconditioner and a coarse-grid starting vector
  (numpy/scipy only, no random numbers, so the output is reproducible).
  A 260k-element mesh into 256 parts takes about a minute on a laptop.
* `metis`: multilevel k-way via pymetis, then an exact-size repair pass
  that moves a few boundary elements to match the linear shares (the moves
  are reported).  Usually the best cut on large meshes.
* `geometric`: recursive coordinate bisection on element centroids.  With
  `--no-stats` it needs neither scipy nor the vertex merge -- the numpy-only
  fast path for very large meshes (add `--low-memory` to memory-map the
  input instead of loading it).  Ignores periodic wrap-around.
* `grid` (`--grid NX,NY,NZ`): user-chosen slabs.  The mesh is cut into NX
  slabs along x at element-count quantiles, every slab into NY strips along
  y, every strip into NZ boxes along z, each part receiving exactly its
  linear share; the cut coordinates are printed.  Rank of box (ix,iy,iz)
  is `(ix·NY + iy)·NZ + iz`.

Non-power-of-two part counts are handled by splitting on exact rank-share
sums at every level (the left child gets `q//2` of the `q` ranks and
exactly their elements), so no rounding accumulates.

**Rank locality.**  The recursive backends number the parts by their
position in the bisection tree, so spatially adjacent parts get adjacent
numbers; METIS's own numbering follows its internal recursive bisection
and is kept.  Two options go further:

* `--ranks-per-node N` makes every top-level split fall on a multiple of N
  ranks (METIS: a two-level partition), so the N ranks of one node always
  own one compact, connected region and most communication stays inside
  the node.  This is the systematic way to keep neighbouring partitions on
  the same node.
* `--relabel fiedler` renumbers the parts along the Fiedler vector of their
  quotient graph (the spectral relaxation of the minimum linear
  arrangement), which shortens the rank distance between communicating
  ranks on open domains.  On ring-like periodic domains a linear order has
  to fold the ring and the option can hurt, so it is opt-in; with
  `--ranks-per-node` it orders the node groups and the ranks inside each
  group.

The report printed after partitioning -- cut size, neighbours per rank,
rank distance of communicating pairs, the share of communication inside a
node -- lets you check the effect of either option on your mesh.

Differences from `contrib/prepart`: Neko's tool partitions with ParMETIS
after `reset_periodic_ids`, i.e. without periodic connectivity, and writes
whatever block sizes ParMETIS produced, so its output only reproduces the
partition when read on a rank count whose linear shares happen to match.
The Python partitioner merges periodic points first (elements across a
periodic boundary do communicate) and always writes the exact linear
shares.

If a balanced connected bisection does not exist (e.g. a star-shaped dual
graph), one side of that bisection stays disconnected and a note is printed;
the output remains a valid permutation -- only cut quality is affected.

## The checker

`mesh_checker.py mesh.nmsh [--jacobian] [--write-zone-indices]`

Reports what Neko's `mesh_checker` reports for the same file: element,
point, face and edge counts after the periodic merge (`glb_mpts` is the
highest point id in use, as in Neko), the bounding box, periodic faces,
labelled zones with their normal alignment (x/y/z/none, Neko's facet-centre
test), unlabelled external faces, plus the curve records.  `--jacobian`
evaluates the Jacobian on the 3×3×3 GLL grid of the geometry Neko
constructs from the file -- the trilinear map deformed by the curve
records (circular arcs and midside points, ports of `arc_surface` and
`gh_face_extend_3d`) -- so curved elements are checked as Neko sees them.

## The viewer

`meshview.py mesh.nmsh [--color zone|partition|jacobian] [--nparts P]
[--curved] [--export skin.vtu] [--screenshot out.png] [--matplotlib]`

The volume is never rendered: the external surface (skin) is extracted in
numpy -- an n-element mesh has O(n^(2/3)) boundary quads, so even a
100M-element mesh reduces to about a million quads.  `--curved` splits each
skin face into 2×2 through the GLL nodes of the curved geometry so arcs and
midside points are visible.  PyVista (VTK, ParaView's rendering engine;
`pip install pyvista`) gives smooth camera interaction, cell picking and
clipping widgets; since VTK ≥ 9.4 the same wheel renders headless
(`--screenshot` on cluster nodes).  Without pyvista, `--export skin.vtu`
writes a file ParaView opens directly, and `--matplotlib` handles small
meshes (≲ 10⁴ elements) with no further dependencies.

## Periodic zones from labelled zones

`create_periodic_zones.py in.nmsh out.nmsh "(1,2),(3,4)" [--tol X]`

Each pair must be two labelled zones with the same number of facets that
map onto each other by one translation (inferred from the facet centres,
verified corner by corner).  The matched facets become periodic records,
the matched points get common ids through Neko's 3-sweep merge, existing
zones are kept.  The default tolerance and the two-tolerance scheme (facet
matching vs. `NEKO_PERIODIC_TOL` for the id merge) are Neko's.

## Memory

The whole-mesh readers hold the raw records in memory: 228 B per element
for a 3D `.nmsh` (so ~2.3 GB at 10⁷ elements; a fat node at 10⁸+), 116 B
for a 2D one.  The two sort-based passes are the peak consumers: point
de-duplication costs about 45 B per corner (≈ 360 B/element transient) and
face matching about 160 B/element.  The dual graph of the graph-based
partitioners is ~27 neighbours per element (≈ 330 B/element in CSR).
`iter_nmsh_elements` streams the element section in chunks for reductions
that do not need the whole mesh; `read_nmsh(..., mmap=True)` (prepart's
`--low-memory`) memory-maps it, and the reordered output is gathered and
written in chunks.  `genmeshbox.py` streams its output and needs only the
three 1-D grid-line arrays plus one chunk.  Curved meshes: a `.nmsh` curve
record is 532 B per curved element.

## Licence

BSD-3-Clause (see `LICENSE`).  The curved-geometry construction in
`nekolight/geometry.py` is a port of Neko routines that Neko itself
derives from Nek5000; the Nek5000 notice reproduced in `LICENSE` applies to
that code.  The spectral partitioner is an independent implementation of
recursive spectral bisection and contains no Nek5000 (genmap) code.

## Scope and known differences

Hex and quad meshes.  Point de-duplication is bit-exact (Neko's point table
hashes the raw bits, so this is what real files experience; its tolerant
equality test only ever matters for near-coincident points that are not
bit-identical).  Legacy zone types (1-4, pre-labelled-zone Neko) are carried
through verbatim by the partitioner and treated as documented boundaries by
the checker; Neko's current reader ignores them.  Neko's own
`create_periodic_zones` indexes elements by their global id, which is only
correct when ids equal record positions; the Python tool uses record
positions and therefore also handles shuffled files.  All tools are
validated against the meshes shipped with Neko (`run_tests.py`), but it
remains your responsibility to confirm the result is correct for your own
mesh -- the suite makes that easy to do on a case you trust.
