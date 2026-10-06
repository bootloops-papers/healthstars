# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Ward's minimum-variance agglomerative clustering for 1-D data, exact arithmetic,
with tie-branch enumeration, efficient at realistic n.

Role in the paper: this is the published-side arm — a semantics-faithful stand-in for
SAS PROC CLUSTER METHOD=WARD as used by the CMS star-ratings cut-point pipeline.
The DP arm (dp_kmeans.py) is the exact optimum; the paper's claim is about the
gap between the two through the full pipeline.

Residuals not fixed by the public documents, handled by enumeration:
  1. Input order / tie-breaking: when two merge candidates have exactly equal Ward
     distance, SAS's choice depends on observation-order conventions. We branch on
     every exact tie and return the SET of reachable final partitions.
  2. Float-vs-exact near-ties: SAS computes in IEEE double; candidates whose exact
     Ward distances differ by less than double noise may compare either way in SAS.
     Optional rel_eps branching treats candidates with d <= d_min*(1+rel_eps) as
     co-minimal, covering every IEEE-double realization. rel_eps=0 -> exact ties.

Ward merge cost between clusters (sizes a,b; means m_A,m_B), Lance-Williams:
    d(A,B) = a*b/(a+b) * (m_A - m_B)^2
Global scalings of d do not change any argmin or tie structure.

Efficiency design (a fully exact O(n^3) greedy is avoided at n~700):
  - DUPLICATE PRE-AGGREGATION LEMMA: equal values sit at pairwise Ward distance 0;
    a global-min greedy merges ALL zero-distance pairs before any positive-distance
    pair (0 < d strictly), and the state after the zero phase is the same for every
    tie order: each duplicate group fully merged. [Proof: merging two clusters with
    equal means leaves their union's mean unchanged, so all intra-group distances
    stay 0 and cross distances stay positive; the zero phase can only end when no
    two clusters share a mean, i.e. groups are fully merged. Any order reaches that
    unique state.] So we start from weighted distinct values, losslessly.
    CAVEAT (SAS fidelity): this lemma covers any exact greedy min-merge. If SAS's
    float Ward ever computes a nonzero distance for equal values (it cannot:
    (m_A-m_B)^2 = 0 exactly in IEEE too), the lemma still holds in SAS arithmetic.
  - Exact Fractions only where decisions are made: candidate distances are mirrored
    in float for screening; the argmin set is confirmed exactly among candidates
    within a conservative float window (SCREEN_REL = 1e-6 relative; float error on
    this formula is ~1e-15 relative, so the screen is safe by ~9 orders — asserted
    in the self-test by full exact scans on random instances).
  - Per-branch state is O(m) cluster stats; distances recomputed per step over
    active clusters: O(m^2) float ops + O(#screened) exact ops per step, O(m^3)
    float worst case overall — floats are cheap; exact ops stay near O(m^2).

m = number of DISTINCT values; CMS measure scores are published at 0-2 decimals so
m <= ~1001 and typically a few hundred.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.3 (Ward's method
    with the documented tie rule, exact arithmetic).
Run:  cd src && python3 ward1d.py   (self-tests; imported by ward_cluster.py)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction
from dp_kmeans import _to_fractions
import sys

SCREEN_REL = 1e-6


class _Cluster:
    __slots__ = ("idxs", "size", "s")  # idxs: tuple of original sorted indices

    def __init__(self, idxs, size, s):
        self.idxs = idxs
        self.size = size
        self.s = s

    def key(self):
        return self.idxs


def _ward_d_exact(ca, cb):
    ma = Fraction(ca.s, 1) / ca.size
    mb = Fraction(cb.s, 1) / cb.size
    diff = ma - mb
    return Fraction(ca.size * cb.size, ca.size + cb.size) * diff * diff


def _ward_d_float(ca, cb):
    ma = float(ca.s) / ca.size
    mb = float(cb.s) / cb.size
    diff = ma - mb
    return (ca.size * cb.size) / (ca.size + cb.size) * diff * diff


def ward_partitions(xs, k, rel_eps=Fraction(0), max_branches=256):
    """Greedy global-min Ward from singletons down to k clusters, all tie branches.

    Returns dict {sorted_x, partitions: [{blocks, clusters, contiguous}]} where
    blocks are (start, end) over the sorted ORIGINAL data (duplicates included),
    clusters are value lists, contiguous flags whether every cluster is an interval
    of the sorted data. Raises RuntimeError on branch overflow (overflow is
    reported, never truncated).
    """
    xs_sorted = sorted(_to_fractions(xs))
    n = len(xs_sorted)
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")

    # duplicate pre-aggregation (lemma above): one weighted cluster per distinct value
    clusters = []
    i = 0
    while i < n:
        j = i
        while j < n and xs_sorted[j] == xs_sorted[i]:
            j += 1
        clusters.append(_Cluster(tuple(range(i, j)), j - i, xs_sorted[i] * (j - i)))
        i = j
    m0 = len(clusters)
    if k > m0:
        # fewer distinct values than clusters requested: every distinct value its
        # own cluster is the unique 0-SSQ outcome at k' = m0; caller must handle.
        raise ValueError(f"k={k} exceeds distinct values m={m0}")

    states = {tuple(clusters)}  # each state: tuple of _Cluster sorted by first idx

    while len(next(iter(states))) > k:
        next_states = {}
        for state in states:
            m = len(state)
            # float screen
            best_f = None
            for a in range(m):
                for b in range(a + 1, m):
                    d = _ward_d_float(state[a], state[b])
                    if best_f is None or d < best_f:
                        best_f = d
            window = best_f * (1.0 + SCREEN_REL) + 1e-300
            screened = []
            for a in range(m):
                for b in range(a + 1, m):
                    if _ward_d_float(state[a], state[b]) <= window:
                        screened.append((a, b))
            # exact argmin among screened
            dex = {(a, b): _ward_d_exact(state[a], state[b]) for a, b in screened}
            dmin = min(dex.values())
            thresh = dmin * (1 + rel_eps) if rel_eps else dmin
            winners = [ab for ab, d in dex.items() if d <= thresh]
            for a, b in winners:
                ca, cb = state[a], state[b]
                merged = _Cluster(tuple(sorted(ca.idxs + cb.idxs)),
                                  ca.size + cb.size, ca.s + cb.s)
                ns = tuple(sorted(
                    [c for t, c in enumerate(state) if t != a and t != b] + [merged],
                    key=lambda c: c.idxs[0]))
                next_states[tuple(c.idxs for c in ns)] = ns
        if len(next_states) > max_branches:
            raise RuntimeError(
                f"tie-branch overflow: {len(next_states)} branches "
                f"(cap {max_branches})")
        states = set(next_states.values())

    out = []
    seen = set()
    for state in states:
        key = tuple(c.idxs for c in state)
        if key in seen:
            continue
        seen.add(key)
        blocks = []
        contiguous = True
        for c in state:
            if list(c.idxs) != list(range(c.idxs[0], c.idxs[-1] + 1)):
                contiguous = False
                blocks = None
                break
            blocks.append((c.idxs[0], c.idxs[-1] + 1))
        out.append({
            "blocks": blocks,
            "clusters": [[xs_sorted[i] for i in c.idxs] for c in state],
            "contiguous": contiguous,
        })
    return {"sorted_x": xs_sorted, "partitions": out}


def _exact_greedy_reference(xs, k):
    """Fully exact greedy (single branch, first-tie order) for testing."""
    xs_sorted = sorted(_to_fractions(xs))
    clusters = [_Cluster((i,), 1, xs_sorted[i]) for i in range(len(xs_sorted))]
    while len(clusters) > k:
        best = None
        arg = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = _ward_d_exact(clusters[a], clusters[b])
                if best is None or d < best:
                    best, arg = d, (a, b)
        a, b = arg
        ca, cb = clusters[a], clusters[b]
        merged = _Cluster(tuple(sorted(ca.idxs + cb.idxs)), ca.size + cb.size, ca.s + cb.s)
        clusters = [c for t, c in enumerate(clusters) if t not in (a, b)] + [merged]
        clusters.sort(key=lambda c: c.idxs[0])
    return tuple(c.idxs for c in clusters)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random
    from dp_kmeans import dp_optimal_partition, partition_ssq

    random.seed(1)
    # 1) the reachable set contains the fully-exact greedy reference outcome
    for trial in range(60):
        n = random.randint(6, 40)
        k = random.randint(2, 5)
        if k > n:
            continue
        xs = [str(random.randint(0, 300) / 10) for _ in range(n)]
        try:
            w = ward_partitions(xs, k)
        except ValueError:
            continue  # k > distinct
        ref = _exact_greedy_reference(xs, k)
        keys = set()
        for p in w["partitions"]:
            if p["blocks"] is not None:
                keys.add(tuple(tuple(range(b[0], b[1])) for b in p["blocks"]))
        # reference uses same index space (sorted) — must be among branches
        assert ref in keys, (trial, xs, k)
    print("ward1d: greedy-reference containment OK (60 trials)")

    # 2) Ward branches are >= DP optimum; count strict suboptimality
    strict = total = 0
    for trial in range(60):
        n = random.randint(10, 60)
        k = random.randint(2, 5)
        xs = [str(random.randint(0, 1000) / 10) for _ in range(n)]
        try:
            w = ward_partitions(xs, k)
        except ValueError:
            continue
        dp = dp_optimal_partition(xs, k)
        for p in w["partitions"]:
            flat = [(v, ci) for ci, cl in enumerate(p["clusters"]) for v in cl]
            ssq = partition_ssq([v for v, _ in flat], [c for _, c in flat])
            assert ssq >= dp["ssq"]
            total += 1
            if ssq > dp["ssq"]:
                strict += 1
    print(f"ward1d: SSQ >= DP on all {total} branch-checks; strictly > on {strict}")

    # 3) realistic scale with duplicate-heavy values: tie branches are enumerated
    #    exhaustively, which is exponential on such data, so this check runs at a
    #    size where the enumeration completes (the production path, ward_cluster_tie,
    #    uses the documented tie rule instead of enumeration).
    for n in (120,):
        xs = [str(random.randint(0, 1000) / 10) for _ in range(n)]
        w = ward_partitions(xs, 5, max_branches=20000)
        dp = dp_optimal_partition(xs, 5)
        print(f"n={n}: ward {len(w['partitions'])} branch; dp ssq {dp['ssq']}")
