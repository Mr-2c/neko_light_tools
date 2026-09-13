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
"""Mesh partitioning: four backends over the same weighted element dual
graph, all producing block sizes that EXACTLY match how Neko distributes
elements when it reads the file.

The block model (the whole point of the reordering contract): Neko's reader
splits nelv elements over P ranks with ``linear_dist_t``
(src/common/datadist.f90): ``L = nelv // P``, ``R = nelv % P``, and the
first R ranks get L+1 elements.  The written element order must therefore be
composed of blocks of exactly those sizes, one block per rank, or a linear
read does not reproduce the partition.  The recursive backends split on
exact rank-share sums (no rounding); METIS output is repaired to the exact
sizes by moving a few boundary elements.

Backends:
  spectral   recursive spectral bisection: at every level the sub-graph is
             split by the best of the lowest Laplacian eigenvectors, computed
             by LOBPCG with an aggregation-multigrid preconditioner and a
             coarse-grid initial guess (numpy/scipy only; deterministic).
  metis      multilevel k-way via pymetis on the same weighted graph (two
             levels when --ranks-per-node is given), repaired to the exact
             sizes.  METIS's own part numbers already follow its internal
             recursive bisection, so they are kept.
  geometric  recursive coordinate bisection on element centroids: no graph,
             no eigensolve, numpy only.
  grid       user-given NX x NY x NZ slabs: the mesh is cut into NX equal
             slabs along x, each slab into NY along y, each of those into NZ
             along z, always at element-count quantiles.

Rank locality: the recursive backends label ranks by their position in the
bisection tree, so spatially adjacent parts get adjacent numbers; with
``ranks_per_node = N`` the top-level splits are made at multiples of N, so
the ranks of one node always form one compact, connected region.
:func:`fiedler_relabel` additionally orders the parts (and, with
ranks_per_node, the node groups and the ranks inside each group) along the
Fiedler vector of their quotient graph, which minimises the weighted
rank distance of communicating pairs -- good on open domains, but on
ring-like (periodic) domains a linear order necessarily folds the ring, so
it is opt-in.
"""

import sys
import warnings

import numpy as np


def _noop(msg):
    pass


# ---------------------------------------------------------------------------
# Neko's linear distribution (the only block model in this package)
# ---------------------------------------------------------------------------
def neko_linear_sizes(nelv, nparts):
    """Per-rank element counts of Neko's linear_dist_t: the first
    ``nelv % nparts`` ranks get one extra element."""
    L, R = divmod(nelv, nparts)
    sizes = np.full(nparts, L, dtype=np.int64)
    sizes[:R] += 1
    return sizes


def _share(nelv, nparts, base, q1):
    """Sum of the linear shares of ranks base .. base+q1-1 (closed form)."""
    L, R = divmod(nelv, nparts)
    return q1 * L + max(0, min(base + q1, R) - base)


def _split_count(q, ranks_per_node):
    """How many of q consecutive ranks go to the left child.  Plain halving,
    or -- with ranks_per_node = N -- halving of the number of N-rank node
    groups, so that node boundaries are never straddled by a split."""
    if ranks_per_node is None or q <= ranks_per_node:
        return q // 2
    groups = -(-q // ranks_per_node)
    return (groups // 2) * ranks_per_node


def weighted_cut(A, part):
    C = A.tocoo()
    mask = part[C.row] != part[C.col]
    return float(C.data[mask].sum()) / 2.0, float(C.data.sum()) / 2.0


# ---------------------------------------------------------------------------
# Low Laplacian modes: dense for small graphs, multilevel LOBPCG otherwise
# ---------------------------------------------------------------------------
DENSE_N = 600          # dense eigh below this size (about 20 ms)
COARSE_N = 300         # coarsest multigrid level (dense pseudo-inverse)


def _aggregate(L):
    """Deterministic heavy-edge aggregation of a graph Laplacian: mutual
    heaviest-neighbour pairs form aggregates, every other node joins the
    aggregate its heaviest neighbour ended up in (chains resolved
    iteratively), leftovers stay singletons.  Returns (agg (n,), n_agg)."""
    n = L.shape[0]
    C = L.tocoo()
    off = C.row != C.col
    row, col, w = C.row[off], C.col[off], -C.data[off]
    if row.size == 0:
        return np.arange(n), n
    # symmetric, deterministic tie-break so the heaviest edge is unique
    lo = np.minimum(row, col).astype(np.int64)
    hi = np.maximum(row, col).astype(np.int64)
    w = w * (1.0 + 1e-9 * (((lo * 7919 + hi) % 1009) / 1009.0))
    order = np.lexsort((col, -w, row))
    row_s = row[order]
    first = np.unique(row_s, return_index=True)[1]
    best = np.full(n, -1, dtype=np.int64)
    best[row_s[first]] = col[order][first]
    agg = np.full(n, -1, dtype=np.int64)
    has = best >= 0
    idx = np.arange(n)
    mutual = has.copy()
    mutual[has] = best[best[has]] == idx[has]
    lead = mutual & (idx < best)
    nlead = int(lead.sum())
    agg[lead] = np.arange(nlead)
    agg[best[lead]] = np.arange(nlead)
    nagg = nlead
    # chains: join the aggregate of the heaviest neighbour once it has one
    while True:
        free = has & (agg < 0)
        join = free.copy()
        join[free] = agg[best[free]] >= 0
        if not join.any():
            break
        agg[join] = agg[best[join]]
    rest = np.flatnonzero(agg < 0)
    agg[rest] = nagg + np.arange(rest.size)
    return agg, nagg + rest.size


class _MultigridLaplacian:
    """Aggregation multigrid on a graph Laplacian: a symmetric V-cycle
    (weighted Jacobi smoothing, dense pseudo-inverse on the coarsest level)
    used as the LOBPCG preconditioner, plus the coarsest-level eigenvectors
    prolonged to the fine level as the initial guess."""

    def __init__(self, L, log=_noop):
        import scipy.sparse as sp
        self.levels = [L.tocsr()]
        self.P = []
        while self.levels[-1].shape[0] > COARSE_N:
            Lc = self.levels[-1]
            n = Lc.shape[0]
            agg, nagg = _aggregate(Lc)
            if nagg > 0.9 * n:
                break
            P = sp.csr_matrix((np.ones(n), (np.arange(n), agg)),
                              shape=(n, nagg))
            self.P.append(P)
            self.levels.append((P.T @ Lc @ P).tocsr())
        self.dinv = []
        for Ll in self.levels:
            d = Ll.diagonal()
            di = np.zeros_like(d)
            nz = d > 0
            di[nz] = 1.0 / d[nz]
            self.dinv.append(di)
        Lc = self.levels[-1].toarray()
        vals, vecs = np.linalg.eigh(Lc)
        keep = vals > 1e-10 * max(vals.max(), 1e-300)
        self.coarse_pinv = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T
        self.coarse_vals, self.coarse_vecs = vals, vecs
        self.n_coarse = Lc.shape[0]

    def _cycle(self, l, r):
        L = self.levels[l]
        if l == len(self.levels) - 1:
            return self.coarse_pinv @ r
        omega = 2.0 / 3.0
        di = self.dinv[l]
        x = omega * di[:, None] * r
        x += omega * di[:, None] * (r - L @ x)
        P = self.P[l]
        x += P @ self._cycle(l + 1, P.T @ (r - L @ x))
        x += omega * di[:, None] * (r - L @ x)
        x += omega * di[:, None] * (r - L @ x)
        return x

    def precondition(self, r):
        r = np.asarray(r)
        return self._cycle(0, r.reshape(r.shape[0], -1)).reshape(r.shape)

    def initial_guess(self, k):
        """The k lowest non-trivial coarsest eigenvectors, prolonged."""
        X = self.coarse_vecs[:, 1:1 + k]
        for P in reversed(self.P):
            X = P @ X
        return X


def low_modes(subA, n, kmodes, log=_noop):
    """The kmodes lowest non-trivial eigenvectors of the Laplacian of subA
    (n x n).  Dense below DENSE_N; otherwise LOBPCG constrained to the
    complement of the constant vector, preconditioned by aggregation
    multigrid and started from the coarse-level eigenvectors.  Deterministic
    (no random numbers anywhere)."""
    from scipy.sparse.csgraph import laplacian
    L = laplacian(subA).tocsr().astype(np.float64)
    k = min(kmodes, n - 1)
    if n <= DENSE_N:
        vals, vecs = np.linalg.eigh(L.toarray())
        return vecs[:, 1:1 + k]
    from scipy.sparse.linalg import lobpcg, LinearOperator
    mg = _MultigridLaplacian(L, log)
    if mg.n_coarse > 4 * COARSE_N:
        log('  note: multigrid coarsening stalled at %d nodes; using '
            'shift-invert Lanczos on a %d-element subset (slow)'
            % (mg.n_coarse, n))
        return _shift_invert_modes(L, n, k)
    X0 = mg.initial_guess(k)
    ones = np.ones((n, 1)) / np.sqrt(n)
    X0 -= ones @ (ones.T @ X0)
    # make sure the block is well conditioned (degenerate coarse modes)
    X0, _ = np.linalg.qr(X0)
    M = LinearOperator((n, n), matvec=mg.precondition,
                       matmat=mg.precondition, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vals, vecs = lobpcg(L, X0, M=M, Y=ones, largest=False,
                            tol=1e-5 * np.sqrt(n), maxiter=60)
    order = np.argsort(vals)
    return vecs[:, order[:k]]


def _shift_invert_modes(L, n, k):
    from scipy.sparse.linalg import eigsh
    v0 = np.linspace(-1.0, 1.0, n) + 1e-3 * np.cos(np.arange(n))
    vals, vecs = eigsh(L.tocsc(), k=k + 1, sigma=-1e-6, which='LM', v0=v0)
    order = np.argsort(vals)
    return vecs[:, order[1:1 + k]]


# ---------------------------------------------------------------------------
# Bisection helpers
# ---------------------------------------------------------------------------
def _sides_connected(subA, side):
    from scipy.sparse.csgraph import connected_components
    for s in (0, 1):
        idx = np.flatnonzero(side == s)
        if idx.size == 0:
            continue
        ncomp, _ = connected_components(subA[idx][:, idx], directed=False)
        if ncomp > 1:
            return False
    return True


def _region_grow(subA, n, n1, log=_noop):
    """Deterministic balanced split by BFS region growing.  Tries a BFS from
    the first node (connected side 0) and, if the complement comes out
    disconnected, the symmetric grow from the last-reached node (connected
    side 1).  On graphs where no balanced connected split exists (e.g. a
    star), returns the first split with a note -- the reordering contract
    is unaffected, only cut quality."""
    from scipy.sparse.csgraph import breadth_first_order

    def grow(start, take_n, claimed_side):
        order = breadth_first_order(subA, start, directed=False,
                                    return_predecessors=False)
        side = np.full(n, 1 - claimed_side, dtype=np.int8)
        take = order[:min(take_n, order.size)]
        side[take] = claimed_side
        short = take_n - take.size
        if short > 0:      # disconnected input: top up deterministically
            rest = np.flatnonzero(side != claimed_side)[:short]
            side[rest] = claimed_side
        return side, order

    side, order = grow(0, n1, 0)
    if _sides_connected(subA, side):
        return side
    alt, _ = grow(int(order[-1]), n - n1, 1)
    if _sides_connected(subA, alt):
        return alt
    log('  note: no balanced connected bisection exists for a %d-element '
        'sub-graph; one side stays disconnected (cut quality only)' % n)
    return side


def spectral_bisect(subA, n, n1, kmodes=4, log=_noop):
    """Split the n nodes of subA into sides 0 (exactly n1 nodes) and 1 using
    the best of the kmodes lowest Laplacian modes (smallest cut; connected
    sides preferred; region growing if no mode gives connected sides)."""
    F = low_modes(subA, n, min(kmodes, n - 1), log)
    C = subA.tocoo()
    best = None
    for c in range(F.shape[1]):
        f = F[:, c]
        j = int(np.argmax(np.abs(f)))
        if f[j] < 0:
            f = -f                                      # sign convention
        order = np.argsort(f, kind='stable')
        side = np.ones(n, dtype=np.int8)
        side[order[:n1]] = 0
        cut = float(C.data[side[C.row] != side[C.col]].sum()) / 2
        conn = _sides_connected(subA, side)
        score = cut if conn else cut + 1e12             # prefer connected
        if best is None or score < best[0]:
            best = (score, side, conn)
    side = best[1]
    if not best[2]:
        side = _region_grow(subA, n, n1, log)
    return side


# ---------------------------------------------------------------------------
# Backend: spectral (recursive spectral bisection)
# ---------------------------------------------------------------------------
def spectral_partition(A, nelv, nparts, kmodes=4, ranks_per_node=None,
                       log=_noop):
    part = np.zeros(nelv, dtype=np.int64)
    A = A.tocsr()

    def rec(idx, q, base):
        n = idx.size
        if q <= 1 or n == 0:
            part[idx] = base
            return
        q1 = _split_count(q, ranks_per_node)
        n1 = _share(nelv, nparts, base, q1)   # exact linear share, left ranks
        if n <= 2 or n1 == 0 or n1 == n:
            left, right = idx[:n1], idx[n1:]
        else:
            subA = A[idx][:, idx].tocsr()
            side = spectral_bisect(subA, n, n1, kmodes, log)
            left = idx[side == 0]
            right = idx[side == 1]
        rec(left, q1, base)
        rec(right, q - q1, base + q1)

    rec(np.arange(nelv, dtype=np.int64), nparts, 0)
    return part


# ---------------------------------------------------------------------------
# Backend: geometric (recursive coordinate bisection; numpy only)
# ---------------------------------------------------------------------------
def geometric_partition(cent, nelv, nparts, ranks_per_node=None):
    """Recursive coordinate bisection on element centroids (``cent`` is
    (nelv, 3)): each sub-domain is cut across its longest extent at the
    element-count quantile that gives the left ranks exactly their linear
    share.  Ignores periodic wrap-around (a known quality trade-off)."""
    part = np.zeros(nelv, dtype=np.int64)

    def rec(idx, q, base):
        n = idx.size
        if q <= 1 or n == 0:
            part[idx] = base
            return
        q1 = _split_count(q, ranks_per_node)
        n1 = _share(nelv, nparts, base, q1)
        c = cent[idx]
        axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        order = np.argsort(c[:, axis], kind='stable')
        rec(idx[order[:n1]], q1, base)
        rec(idx[order[n1:]], q - q1, base + q1)

    rec(np.arange(nelv, dtype=np.int64), nparts, 0)
    return part


# ---------------------------------------------------------------------------
# Backend: grid (user-specified NX x NY x NZ slabs at count quantiles)
# ---------------------------------------------------------------------------
def grid_partition(cent, nelv, grid):
    """Cut the mesh into ``grid = (NX, NY, NZ)`` boxes by element count: NX
    slabs along x (each holding exactly the linear shares of its NY*NZ
    ranks), every slab into NY strips along y, every strip into NZ boxes
    along z.  Rank of box (ix, iy, iz) is ``(ix*NY + iy)*NZ + iz``, so
    consecutive ranks are z-neighbours, then y-neighbours.

    Returns (part, cuts) where ``cuts`` lists, per split axis in order,
    the cut coordinates found: for x one array of NX-1 values; for y an
    (NX, NY-1) array (one row per x-slab); for z an (NX*NY, NZ-1) array.
    A cut coordinate is the midpoint between the two centroids it
    separates."""
    nx, ny, nz = [int(g) for g in grid]
    if min(nx, ny, nz) < 1:
        sys.exit('Error: --grid counts must all be >= 1')
    nparts = nx * ny * nz
    part = np.empty(nelv, dtype=np.int64)
    cut_x = np.full(nx - 1, np.nan)
    cut_y = np.full((nx, ny - 1), np.nan)
    cut_z = np.full((nx * ny, nz - 1), np.nan)

    def split(idx, axis, nsplit, base, ranks_each, cuts_out):
        """Split idx along axis into nsplit consecutive groups, group g
        getting exactly the shares of ranks base+g*ranks_each ...; returns
        the list of index groups and records the cut coordinates."""
        order = idx[np.argsort(cent[idx, axis], kind='stable')]
        groups, s = [], 0
        for g in range(nsplit):
            m = _share(nelv, nparts, base + g * ranks_each, ranks_each)
            groups.append(order[s:s + m])
            if g < nsplit - 1 and cuts_out is not None:
                if s + m < order.size and m > 0:
                    a = cent[order[s + m - 1], axis]
                    b = cent[order[s + m], axis]
                    cuts_out[g] = 0.5 * (a + b)
            s += m
        return groups

    slabs = split(np.arange(nelv, dtype=np.int64), 0, nx, 0, ny * nz, cut_x)
    for ix, sl in enumerate(slabs):
        strips = split(sl, 1, ny, ix * ny * nz, nz, cut_y[ix])
        for iy, st in enumerate(strips):
            base = (ix * ny + iy) * nz
            boxes = split(st, 2, nz, base, 1, cut_z[ix * ny + iy])
            for iz, bx in enumerate(boxes):
                part[bx] = base + iz
    return part, (cut_x, cut_y, cut_z)


# ---------------------------------------------------------------------------
# Backend: metis (multilevel k-way, two-level with ranks_per_node)
# ---------------------------------------------------------------------------
def _pymetis_part(Ai, nparts, tpwgts=None, recursive=False, log=_noop):
    import pymetis
    xadj = Ai.indptr.astype(np.int32)
    adjncy = Ai.indices.astype(np.int32)
    eweights = np.rint(Ai.data).astype(np.int32)
    kw = dict(eweights=eweights, recursive=recursive)
    if tpwgts is not None:
        kw['tpwgts'] = [float(t) for t in tpwgts]
    try:
        opts = pymetis.Options()
        opts.seed = 12345
        opts.ufactor = 5              # 0.5 % imbalance: few repair moves
        kw['options'] = opts
    except Exception:
        pass
    try:                                  # pymetis >= 2025.2
        adj = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
        _, membership = pymetis.part_graph(nparts, adjacency=adj, **kw)
    except (AttributeError, TypeError):   # older pymetis
        _, membership = pymetis.part_graph(nparts, xadj=xadj, adjncy=adjncy,
                                          **kw)
    part = np.asarray(membership, dtype=np.int64)
    if np.unique(part).size < nparts and not recursive:
        log('  note: k-way METIS returned %d non-empty parts; retrying with '
            'recursive bisection' % np.unique(part).size)
        return _pymetis_part(Ai, nparts, tpwgts, True, log)
    if np.unique(part).size < nparts:
        sys.exit('Error: METIS produced only %d non-empty parts of %d '
                 'requested; use --backend spectral or geometric for this '
                 'nparts/nelv ratio.' % (np.unique(part).size, nparts))
    return part


def _metis_level(A, idx, nparts_total, nelv, base, q, log):
    """METIS-partition the elements idx into q consecutive ranks base..base+q-1
    with exactly their linear shares: k-way with target weights, Fiedler
    relabelling for rank locality, then the exact-size repair."""
    if q == 1:
        return np.full(idx.size, base, dtype=np.int64)
    Ai = A[idx][:, idx].tocsr()
    target = np.array([_share(nelv, nparts_total, base + r, 1)
                       for r in range(q)], dtype=np.int64)
    tpw = target / target.sum()
    part = _pymetis_part(Ai, q, tpw, log=log)
    part = repair_sizes(Ai, part, target, log)
    return part + base


def metis_partition(A, nelv, nparts, ranks_per_node=None, log=_noop):
    try:
        import pymetis  # noqa: F401
    except ImportError:
        sys.exit('Error: --backend metis needs pymetis (pip install pymetis)')
    if nparts == 1:
        return np.zeros(nelv, dtype=np.int64)
    A = A.tocsr()
    if A.nnz >= 2**31:
        sys.exit('Error: dual graph too large for METIS 32-bit indices '
                 '(%d edges); use --backend geometric' % A.nnz)
    allidx = np.arange(nelv, dtype=np.int64)
    if ranks_per_node is None or nparts <= ranks_per_node:
        return _metis_level(A, allidx, nparts, nelv, 0, nparts, log)
    # two levels: node groups first (exact group shares), then ranks inside
    N = ranks_per_node
    groups = -(-nparts // N)
    gsize = np.array([_share(nelv, nparts, g * N, min(N, nparts - g * N))
                      for g in range(groups)], dtype=np.int64)
    gpart = _pymetis_part(A, groups, gsize / gsize.sum(), log=log)
    gpart = repair_sizes(A, gpart, gsize, log)
    part = np.empty(nelv, dtype=np.int64)
    for g in range(groups):
        idx = np.flatnonzero(gpart == g)
        q = min(N, nparts - g * N)
        part[idx] = _metis_level(A, idx, nparts, nelv, g * N, q, log)
    return part


# ---------------------------------------------------------------------------
# Quotient graph, Fiedler relabelling and the communication report
# ---------------------------------------------------------------------------
def quotient_graph(A, part, nparts):
    """Partition-to-partition connectivity Q (nparts x nparts, symmetric,
    zero diagonal): Q[p, q] = summed dual-graph weight between p and q."""
    import scipy.sparse as sp
    n = part.size
    O = sp.csr_matrix((np.ones(n), (np.arange(n), part)), shape=(n, nparts))
    Q = (O.T @ A @ O).tocsr()
    Q.setdiag(0)
    Q.eliminate_zeros()
    return Q


def _fiedler_order(Q, n):
    """Permutation new -> old of the n nodes of quotient graph Q along its
    Fiedler vector (identity when Q has no edges)."""
    if n < 3 or Q.nnz == 0:
        return np.arange(n)
    f = low_modes(Q, n, 1)[:, 0]
    j = int(np.argmax(np.abs(f)))
    if f[j] < 0:
        f = -f
    return np.argsort(f, kind='stable')


def fiedler_relabel(A, part, nparts, ranks_per_node=None):
    """Relabel the parts so that strongly communicating parts get nearby
    numbers: parts are sorted by the Fiedler vector of their quotient graph
    (the spectral relaxation of the minimum linear arrangement, which
    minimises sum w_pq (p - q)^2).  With ``ranks_per_node = N`` the node
    groups (ranks kN..kN+N-1) are kept together: the groups are ordered by
    the Fiedler vector of the group quotient graph and the ranks inside each
    group by that of the intra-group quotient graph.  Deterministic.

    Sizes: relabelling permutes the part sizes, so the caller must repair
    the sizes afterwards (or, for the recursive backends whose blocks all
    have size L or L+1, accept that the L+1 blocks move to other ranks --
    which is why the tools apply it BEFORE the exact-size repair)."""
    if nparts < 3:
        return part
    Q = quotient_graph(A, part, nparts)
    if ranks_per_node is None or nparts <= ranks_per_node:
        order = _fiedler_order(Q, nparts)               # new -> old
        relabel = np.empty(nparts, dtype=np.int64)
        relabel[order] = np.arange(nparts)
        return relabel[part]
    N = ranks_per_node
    groups = -(-nparts // N)
    import scipy.sparse as sp
    gid = np.arange(nparts) // N
    G = sp.csr_matrix((np.ones(nparts), (np.arange(nparts), gid)),
                      shape=(nparts, groups))
    QG = (G.T @ Q @ G).tocsr()
    QG.setdiag(0)
    QG.eliminate_zeros()
    gorder = _fiedler_order(QG, groups)                 # new group -> old
    relabel = np.empty(nparts, dtype=np.int64)
    nxt = 0
    for g_new, g_old in enumerate(gorder):
        ranks = np.arange(g_old * N, min((g_old + 1) * N, nparts))
        Qg = Q[ranks][:, ranks].tocsr()
        sub = _fiedler_order(Qg, ranks.size)            # new -> old (local)
        for r_old in ranks[sub]:
            relabel[r_old] = nxt
            nxt += 1
    assert nxt == nparts
    return relabel[part]


def communication_report(A, part, nparts, ranks_per_node=None):
    """Statistics of the inter-partition communication pattern implied by a
    partition: neighbours per rank, cut weight, weighted rank distance and
    (with ranks_per_node) the share of cut weight that stays inside a node.
    Returns a dict of plain numbers."""
    Q = quotient_graph(A, part, nparts).tocoo()
    total = float(A.sum()) / 2.0
    cut = float(Q.data.sum()) / 2.0
    deg = np.bincount(Q.row, minlength=nparts)
    rep = dict(cut=cut, total=total,
               cut_pct=100.0 * cut / max(total, 1.0),
               neigh_min=int(deg.min()) if nparts > 1 else 0,
               neigh_mean=float(deg.mean()) if nparts > 1 else 0.0,
               neigh_max=int(deg.max()) if nparts > 1 else 0)
    if Q.nnz:
        dist = np.abs(Q.row.astype(np.int64) - Q.col.astype(np.int64))
        rep['dist_max'] = int(dist.max())
        rep['dist_mean'] = float((dist * Q.data).sum() / Q.data.sum())
        rep['adjacent_pct'] = 100.0 * float(Q.data[dist == 1].sum()) \
            / float(Q.data.sum())
        if ranks_per_node:
            same = (Q.row // ranks_per_node) == (Q.col // ranks_per_node)
            rep['intranode_pct'] = 100.0 * float(Q.data[same].sum()) \
                / float(Q.data.sum())
    return rep


# ---------------------------------------------------------------------------
# Exact-size repair (METIS output -> Neko's linear shares)
# ---------------------------------------------------------------------------
def repair_sizes(A, part, target, log=_noop):
    """Move elements between parts until the part sizes exactly equal
    ``target``.  METIS is near-balanced, so this typically moves a few
    elements per part.  Each round computes, in one sparse product, the
    connectivity of every element of an over-full part to every part, then
    greedily moves the best-connected boundary elements into under-full
    parts (never overfilling one, never moving an element twice).  An
    over-full part with no under-full neighbour pushes its best boundary
    element into an exactly-full neighbour instead, which then becomes the
    over-full one next round -- excess flows along the quotient graph until
    it reaches a deficit.  Relabelling first (largest parts onto the L+1
    ranks) minimises the number of moves."""
    import scipy.sparse as sp
    nparts = target.size
    part = part.copy()
    sizes = np.bincount(part, minlength=nparts)
    # relabel so the size ranking matches the target ranking
    order_p = np.argsort(-sizes, kind='stable')
    order_r = np.argsort(-target, kind='stable')
    relabel = np.empty(nparts, dtype=np.int64)
    relabel[order_p] = order_r
    part = relabel[part]
    sizes = np.bincount(part, minlength=nparts)
    if np.array_equal(sizes, target):
        return part
    Ac = A.tocsr()
    n = part.size
    moved_flag = np.zeros(n, dtype=bool)
    moved = 0
    rounds = 0
    while True:
        excess = sizes - target
        if not (excess != 0).any():
            break
        rounds += 1
        if rounds > 4 * nparts + 16:
            sys.exit('Error: size-repair pass failed to converge (bug)')
        over = np.flatnonzero(excess > 0)
        cand = np.flatnonzero(np.isin(part, over) & ~moved_flag)
        if cand.size == 0:
            cand = np.flatnonzero(np.isin(part, over))
            moved_flag[cand] = False
        O = sp.csr_matrix((np.ones(n), (np.arange(n), part)),
                          shape=(n, nparts))
        G = (Ac[cand] @ O).tocoo()             # (n_cand, nparts) gains
        gsrc = part[cand[G.row]]
        keep = (G.col != gsrc)
        gr, gc, gw = G.row[keep], G.col[keep], G.data[keep]
        # best moves first; element index as a deterministic tie-break
        o = np.lexsort((cand[gr], gc, -gw))
        gr, gc, gw = gr[o], gc[o], gw[o]
        moved_round = 0
        # pass 1: into under-full parts
        for r, c in zip(gr, gc):
            e = cand[r]
            s = part[e]
            if excess[s] <= 0 or excess[c] >= 0 or moved_flag[e]:
                continue
            part[e] = c
            excess[s] -= 1
            excess[c] += 1
            moved_flag[e] = True
            moved_round += 1
        # pass 2: parts still over-full with no under-full neighbour push
        # one element each into their best exactly-full neighbour
        still = np.flatnonzero(excess > 0)
        if still.size:
            pushed = set()
            for r, c in zip(gr, gc):
                e = cand[r]
                s = part[e]
                if s in pushed or excess[s] <= 0 or excess[c] != 0 \
                   or moved_flag[e]:
                    continue
                part[e] = c
                excess[s] -= 1
                excess[c] += 1
                moved_flag[e] = True
                pushed.add(s)
                moved_round += 1
        if moved_round == 0:
            # no boundary candidates at all (isolated part): move arbitrary
            # elements -- connectivity cannot be preserved here anyway
            s = int(over[0])
            e = int(np.flatnonzero(part == s)[0])
            c = int(np.flatnonzero(excess < 0)[0])
            part[e] = c
            excess[s] -= 1
            excess[c] += 1
            moved_round = 1
        moved += moved_round
        sizes = target + excess
    if moved:
        log('  note: moved %d element(s) to match Neko\'s exact linear '
            'block sizes' % moved)
    return part


# ---------------------------------------------------------------------------
# Reorder + write (the prepart contract)
# ---------------------------------------------------------------------------
def reorder_and_write(path, mesh, part, pos_of_elid, inputs=(), log=_noop,
                      chunk=1 << 19):
    """Write the reordered .nmsh: contiguous per-partition blocks, inside
    each block the elements in increasing ORIGINAL global id (for every
    Neko-written file ids equal record positions, so this is also the
    input order), element ids renumbered 1..nelv, zone/curve element
    references remapped, all payloads verbatim.

    The element section is gathered and written in chunks, so a
    memory-mapped input (``read_nmsh(..., mmap=True)``) is never
    materialised in full.
    """
    from .formats import atomic_output, ZONE_DT
    nelv = mesh.nelv
    elids = np.asarray(mesh.elems['id']).astype(np.int64)
    order = np.lexsort((elids, part))                   # new position -> old pos
    newid_of_pos = np.empty(nelv, dtype=np.int64)       # old pos -> new 1-based id
    newid_of_pos[order] = np.arange(1, nelv + 1)
    # strictly increasing original ids inside every block -- the property a
    # user relies on for locality; assert it
    if nelv > 1:
        same_block = part[order[1:]] == part[order[:-1]]
        assert (elids[order[1:]][same_block]
                > elids[order[:-1]][same_block]).all()

    def remap_el(ids):
        return newid_of_pos[pos_of_elid[ids.astype(np.int64)]].astype(np.int32)

    zones, curves = mesh.zones, mesh.curves
    z5 = zones[zones['t'] == 5].copy()
    z7 = zones[zones['t'] == 7].copy()
    zx = zones[(zones['t'] != 5) & (zones['t'] != 7)].copy()
    if z5.size:
        z5['e'] = remap_el(z5['e'])
        z5['p_e'] = remap_el(z5['p_e'])
    if z7.size:
        z7['e'] = remap_el(z7['e'])
        z7['p_e'] = 0                                   # Neko leaves these unset;
        z7['g'] = 0                                     # write deterministic zeros
    if zx.size:
        # Legacy zone types (e.g. type 1/2 in older meshes).  Neko's current
        # reader ignores them, but they are part of the file -- carry them
        # through verbatim with only the element id renumbered.
        log('  note: carrying %d zone records of other types %s through '
            '(element ids renumbered, payload verbatim)'
            % (zx.shape[0], sorted(set(int(t) for t in zx['t']))))
        zx['e'] = remap_el(zx['e'])
    curves_out = curves.copy()
    if curves_out.size:
        curves_out['e'] = remap_el(curves_out['e'])
    nz = z5.shape[0] + z7.shape[0] + zx.shape[0]

    with atomic_output(path, inputs) as f:
        np.array([nelv, mesh.gdim], dtype='<i4').tofile(f)
        for s in range(0, nelv, chunk):
            sel = order[s:s + chunk]
            # sorted gather is much faster on a memmap; restore block order
            srt = np.argsort(sel, kind='stable')
            rec = np.asarray(mesh.elems[sel[srt]])
            inv = np.empty_like(srt)
            inv[srt] = np.arange(srt.size)
            rec = rec[inv]
            rec['id'] = np.arange(s + 1, s + 1 + sel.size, dtype=np.int32)
            rec.tofile(f)
        np.array([nz], dtype='<i4').tofile(f)
        for z in (z5, z7, zx):
            z.astype(ZONE_DT).tofile(f)
        np.array([curves_out.shape[0]], dtype='<i4').tofile(f)
        curves_out.tofile(f)
