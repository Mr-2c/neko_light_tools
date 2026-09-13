#!/usr/bin/env python3
# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see nekolight/formats.py for the full
# licence text.
#
#     _  __  ____  __ __  ____
#    / |/ / / __/ / //_/ / __ \
#   /    / / _/  / ,<   / /_/ /
#  /_/|_/ /___/ /_/|_|  \____/
#
"""prepart -- partition a Neko .nmsh and write a reordered .nmsh whose linear
read reproduces the partition exactly (the contrib/prepart contract).

The written element order is grouped into contiguous blocks whose sizes
exactly match Neko's linear_dist_t shares (L = nelv // P, first nelv % P
ranks get L+1), so rank i of a P-rank run receives exactly partition i.
Inside every block the original element order is kept, so the new global
ids of a rank's elements are a strictly increasing run AND follow the
input's numbering.  Curves and zones are carried through with only their
element references renumbered; point ids and coordinates are never
changed.  2D (quad) meshes are partitioned and written as quads.

Backends (--backend):
  spectral   (default) recursive spectral bisection of the periodic-merged,
             shared-vertex-weighted element dual graph (multilevel LOBPCG
             eigensolver, numpy/scipy only, deterministic).
  metis      multilevel k-way via pymetis (pip install pymetis) on the same
             graph, relabelled for rank locality and repaired to the exact
             linear block sizes; usually the best cut.
  geometric  recursive coordinate bisection on element centroids: numpy
             only, no graph, near-instant on very large meshes.  Ignores
             periodic wrap-around.
  grid       --grid NX,NY,NZ: NX slabs along x, each cut into NY along y,
             each of those into NZ along z, at element-count quantiles (all
             parts get exactly their linear share); the cut coordinates are
             reported.  Rank of box (ix,iy,iz) = (ix*NY + iy)*NZ + iz.

Rank locality: --ranks-per-node N makes every top-level split fall on a
multiple of N ranks, so the N ranks of one node always own one compact,
connected region (spectral/geometric/grid by construction, METIS through a
two-level partition).  --relabel fiedler additionally renumbers the parts
along the Fiedler vector of their quotient graph, which shortens the rank
distance between communicating ranks on open domains (it can hurt on
ring-like periodic domains, hence opt-in).  The report lists how much of
the communication stays inside a node and how far apart (in rank number)
communicating ranks are, so the effect of either option can be checked.

Usage: prepart.py mesh.nmsh [nparts] [out.nmsh] [options]
Default output is <base>_<nparts>.nmsh; nparts and the output name may also
come after the options (or use -o out.nmsh).
"""

import argparse
import sys
import time

import numpy as np

from nekolight import (banner, read_nmsh, validate_zones, validate_curves,
                       pos_of_elid_map, merged_vertex_ids, compress_ids,
                       dual_graph, neko_linear_sizes, spectral_partition,
                       metis_partition, geometric_partition, grid_partition,
                       fiedler_relabel, repair_sizes, communication_report,
                       reorder_and_write)


def log(msg):
    print(msg, flush=True)


def centroids(mesh, chunk=1 << 20):
    """Element centroids, chunked (works on a memory-mapped element array)."""
    cent = np.empty((mesh.nelv, 3))
    for s in range(0, mesh.nelv, chunk):
        cent[s:s + chunk] = np.asarray(
            mesh.elems['v']['xyz'][s:s + chunk]).mean(axis=1)
    return cent


def main():
    ap = argparse.ArgumentParser(
        prog='prepart.py',
        description='Partition a Neko .nmsh and write a reordered .nmsh '
                    'whose linear read reproduces the partition.')
    ap.add_argument('mesh', help='input .nmsh')
    ap.add_argument('rest', nargs='*', metavar='nparts [out.nmsh]',
                    help='number of partitions (implied by --grid) and the '
                         'output .nmsh (default <base>_<nparts>.nmsh); both '
                         'may also follow the options')
    ap.add_argument('-o', '--out', default=None, help='output .nmsh')
    ap.add_argument('--backend',
                    choices=('spectral', 'metis', 'geometric', 'grid'),
                    default=None)
    ap.add_argument('--metis', dest='backend', action='store_const',
                    const='metis', help='shorthand for --backend metis')
    ap.add_argument('--geometric', dest='backend', action='store_const',
                    const='geometric', help='shorthand for --backend geometric')
    ap.add_argument('--grid', metavar='NX,NY,NZ', default=None,
                    help='partitions per direction (implies --backend grid)')
    ap.add_argument('--ranks-per-node', type=int, metavar='N', default=None,
                    help='keep the N ranks of one node on one compact region')
    ap.add_argument('--relabel', choices=('none', 'fiedler'), default='none',
                    help='fiedler: renumber the parts along the Fiedler '
                         'vector of their quotient graph so communicating '
                         'ranks get close numbers (node groups kept with '
                         '--ranks-per-node); needs the dual graph')
    ap.add_argument('--kmodes', type=int, default=4,
                    help='spectral: number of low modes tried per bisection '
                         '(default 4)')
    ap.add_argument('--no-stats', action='store_true',
                    help='skip the edge-cut/communication report (with the '
                         'geometric and grid backends this also skips the '
                         'vertex merge and dual graph entirely -- the '
                         'numpy-only fast path)')
    ap.add_argument('--low-memory', action='store_true',
                    help='memory-map the element section instead of loading '
                         'it (geometric/grid: peak memory ~ 60 B/element)')
    args, extra = ap.parse_known_args()
    # positional: [nparts] [out]; nparts may be omitted with --grid, and
    # argparse only matches positionals before the first option, so
    # anything left over after the options is folded in here
    bad = [a for a in extra if a.startswith('-')]
    if bad:
        ap.error('unrecognised arguments: %s' % ' '.join(bad))
    rest = list(args.rest) + list(extra)
    args.nparts = None
    if rest and rest[0].lstrip('+-').isdigit():
        args.nparts = int(rest.pop(0))
    if rest:
        if args.out is not None:
            ap.error('output given twice (%s and %s)' % (args.out, rest[0]))
        args.out = rest.pop(0)
    if rest:
        ap.error('unrecognised arguments: %s' % ' '.join(rest))

    grid = None
    if args.grid is not None:
        try:
            grid = tuple(int(v) for v in args.grid.split(','))
        except ValueError:
            sys.exit('Error: --grid expects three integers NX,NY,NZ')
        if len(grid) != 3 or min(grid) < 1:
            sys.exit('Error: --grid expects three positive integers NX,NY,NZ')
        if args.backend not in (None, 'grid'):
            sys.exit('Error: --grid implies --backend grid')
        args.backend = 'grid'
        gp = grid[0] * grid[1] * grid[2]
        if args.nparts is None:
            args.nparts = gp
        elif args.nparts != gp:
            sys.exit('Error: nparts (%d) does not equal NX*NY*NZ (%d)'
                     % (args.nparts, gp))
    if args.backend is None:
        args.backend = 'spectral'
    if args.backend == 'grid' and grid is None:
        sys.exit('Error: --backend grid needs --grid NX,NY,NZ')
    if args.nparts is None:
        sys.exit('Error: nparts is required (or give --grid NX,NY,NZ)')
    if args.nparts < 1:
        sys.exit('Error: nparts must be a positive integer')
    if args.ranks_per_node is not None and args.ranks_per_node < 1:
        sys.exit('Error: --ranks-per-node must be a positive integer')
    if args.kmodes < 1:
        sys.exit('Error: --kmodes must be >= 1')
    if args.out is None:
        base = args.mesh
        i, j = base.rfind('.'), base.rfind('/')
        base = base[:i] if i > j else base
        args.out = '%s_%d.nmsh' % (base, args.nparts)

    log(banner('prepart  (mesh partitioner)'))
    log('  input     : %s' % args.mesh)
    log('  output    : %s' % args.out)
    log('  nparts    : %d%s' % (args.nparts,
                                (' = %dx%dx%d' % grid) if grid else ''))
    log('  backend   : %s' % args.backend)
    if args.ranks_per_node:
        log('  ranks/node: %d' % args.ranks_per_node)

    t0 = time.time()
    log('  [1/4] reading mesh ...')
    mesh = read_nmsh(args.mesh, mmap=args.low_memory)
    validate_zones(mesh.nelv, mesh.zones, args.mesh, mesh.gdim)
    validate_curves(mesh.nelv, mesh.curves, args.mesh, mesh.gdim, log)
    log('        %d %s elements, %d zones, %d curved elements'
        % (mesh.nelv, 'hex' if mesh.gdim == 3 else 'quad',
           mesh.zones.shape[0], mesh.curves.shape[0]))
    if args.nparts > mesh.nelv:
        sys.exit('Error: nparts (%d) > number of elements (%d)'
                 % (args.nparts, mesh.nelv))
    pos_of_elid = pos_of_elid_map(mesh.nelv, mesh.elems)

    need_graph = (args.backend in ('spectral', 'metis')) or not args.no_stats \
        or args.relabel != 'none'
    A = None
    if need_graph:
        log('  [2/4] building element dual graph (periodic-merged) ...')
        cell = compress_ids(merged_vertex_ids(mesh, pos_of_elid))
        A = dual_graph(cell)
        del cell
    else:
        log('  [2/4] skipping dual graph (--no-stats, %s backend)'
            % args.backend)

    log('  [3/4] partitioning (%s) ...' % args.backend)
    cuts = None
    if args.backend == 'spectral':
        part = spectral_partition(A, mesh.nelv, args.nparts, args.kmodes,
                                  args.ranks_per_node, log)
    elif args.backend == 'metis':
        part = metis_partition(A, mesh.nelv, args.nparts,
                               args.ranks_per_node, log)
    elif args.backend == 'geometric':
        part = geometric_partition(centroids(mesh), mesh.nelv, args.nparts,
                                   args.ranks_per_node)
    else:
        part, cuts = grid_partition(centroids(mesh), mesh.nelv, grid, log)

    want = neko_linear_sizes(mesh.nelv, args.nparts)
    if args.relabel == 'fiedler' and args.nparts > 2:
        log('        relabelling parts along the quotient-graph Fiedler '
            'vector ...')
        part = fiedler_relabel(A, part, args.nparts, args.ranks_per_node)
        part = repair_sizes(A, part, want, log)       # L/L+1 blocks moved
    sizes = np.bincount(part, minlength=args.nparts)
    if not np.array_equal(sizes, want):
        sys.exit('Error: internal error -- block sizes %s do not match '
                 'Neko\'s linear distribution %s' % (sizes, want))
    log('        part sizes: min %d / max %d (exact linear_dist blocks)'
        % (sizes.min(), sizes.max()))
    if cuts is not None:
        report_cuts(grid, cuts)
    if A is not None and not args.no_stats:
        rep = communication_report(A, part, args.nparts, args.ranks_per_node)
        log('        edge cut: %d / %d shared-vertex links cut (%7.3f%%)'
            % (rep['cut'], rep['total'], rep['cut_pct']))
        if args.nparts > 1:
            log('        neighbours per rank: min %d / mean %.1f / max %d'
                % (rep['neigh_min'], rep['neigh_mean'], rep['neigh_max']))
        if 'dist_max' in rep:
            log('        rank distance of communicating pairs: mean %.1f, '
                'max %d; %.1f%% of the cut is between adjacent ranks'
                % (rep['dist_mean'], rep['dist_max'], rep['adjacent_pct']))
        if 'intranode_pct' in rep:
            log('        %.1f%% of the cut stays inside a node of %d ranks'
                % (rep['intranode_pct'], args.ranks_per_node))

    log('  [4/4] renumbering and writing reordered mesh ...')
    reorder_and_write(args.out, mesh, part, pos_of_elid,
                      inputs=(args.mesh,), log=log)
    log('  done -> %s   (%.1f s)' % (args.out, time.time() - t0))


def report_cuts(grid, cuts):
    cut_x, cut_y, cut_z = cuts
    names = 'xyz'
    for axis, c in enumerate((cut_x, cut_y, cut_z)):
        c = np.asarray(c)
        if c.size == 0:
            continue
        flat = c.ravel()
        flat = flat[np.isfinite(flat)]
        if c.ndim == 1 or c.shape[0] == 1:
            log('        %s cuts: %s' % (names[axis],
                                        ' '.join('%.6g' % v for v in flat)))
        elif flat.size <= 24:
            for i, row in enumerate(c):
                log('        %s cuts (%s-block %d): %s'
                    % (names[axis], names[axis - 1], i,
                       ' '.join('%.6g' % v for v in row if np.isfinite(v))))
        else:
            log('        %s cuts: %d values in [%.6g, %.6g] (vary per %s '
                'block)' % (names[axis], flat.size, flat.min(), flat.max(),
                            names[axis - 1]))


if __name__ == '__main__':
    main()
