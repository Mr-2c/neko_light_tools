# Copyright (c) 2026, The Neko Authors
# All rights reserved.  BSD-3-Clause: see any module in this package for the
# full licence text.
#
#     _  __  ____  __ __  ____
#    / |/ / / __/ / //_/ / __ \
#   /    / / _/  / ,<   / /_/ /
#  /_/|_/ /___/ /_/|_|  \____/
#
"""nekolight -- the shared library behind the Neko light tools.

Plain functions over numpy arrays; the only data structure is the
:class:`~nekolight.formats.Mesh` named tuple.  Core dependencies are numpy
(everywhere) and scipy (partitioning and the create_periodic_zones matcher);
pymetis, pyvista and matplotlib are optional extras.
"""

from .formats import (Mesh, Re2, EL_DT, QUAD_DT, ZONE_DT, CURVE_DT, FACE_RE2,
                      EDGE_RE2, QFACE_RE2, FACET_MAP, SIJK, MAX_ZLBLS,
                      elem_dtype, banner, atomic_output, read_nmsh,
                      extrude_2d, iter_nmsh_elements, write_nmsh,
                      validate_zones, validate_curves, read_re2,
                      read_dist_csv, write_zone_indices_fld, bc_type_str)
from .topology import (facet_table, pos_of_elid_map, dedup_points,
                       periodic_replace_merge, create_periodic_ids,
                       merged_vertex_ids,
                       compress_ids, face_multiplicity, count_edges, skin,
                       dual_graph, point_coordinate_conflicts,
                       midside_conflicts, EDGE_CYC)
from .geometry import (gll_xyz, jacobian_gll, jacobian_dets, min_jacobian,
                       facet_normals, facet_gll_mask, apply_curves,
                       gll_geometry, CurveError, GLL3)
from .gmsh import read_msh, GmshMesh, GMSH_TYPES
from .gmshconv import (CellSet, FacetTable, UnionFind, boundary_cells,
                       corner_jacobians, orient, CURVE_TOL)
from .extrude import layer_planes, extrude
from .partition import (neko_linear_sizes, spectral_partition,
                        metis_partition, geometric_partition, grid_partition,
                        repair_sizes, weighted_cut, quotient_graph,
                        fiedler_relabel, communication_report,
                        reorder_and_write)

__all__ = [n for n in dir() if not n.startswith('_')]
__version__ = '0.2.0'
