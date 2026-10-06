# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Exact 1-D k-means (within-cluster sum of squares minimization) by dynamic programming.

This is the exact-optimum layer for the Medicare Advantage cut-point analysis.
Wang & Song (2011, The R Journal; CRAN Ckmeans.1d.dp) prove the DP recurrence
delivers the global SSQ minimum for univariate data. It is re-implemented here
in exact rational arithmetic (fractions.Fraction) so the optimum and every SSQ
comparison is exact: no floating-point step anywhere in the objective.

Conventions:
- Input: a list of numeric scores (ints/decimal strings/Fractions). Duplicates allowed.
- A k-partition of the SORTED data into contiguous blocks (contiguity is WLOG for
  the SSQ objective in 1-D: any optimal partition can be reordered to contiguous
  blocks of the sorted sequence; proof: exchange argument on cluster means).
- Output: optimal SSQ (Fraction), block boundaries, and per-cluster means.

Direction convention for cut points is the caller's problem (CMS cut points are
stated as thresholds on the measure score; higher-is-better vs lower-is-better
measures differ). This module only partitions.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.3 (the exact optimum
    by dynamic programming) and Appendix C.4 (300 random instances against
    exhaustive enumeration).
Run:  cd src && python3 dp_kmeans.py   (self-tests; imported by the census scripts)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction
from itertools import combinations
import sys


def _to_fractions(xs):
    out = []
    for x in xs:
        if isinstance(x, Fraction):
            out.append(x)
        elif isinstance(x, int):
            out.append(Fraction(x))
        elif isinstance(x, str):
            out.append(Fraction(x))  # exact decimal-string parse, e.g. "87.5" -> 175/2
        elif isinstance(x, float):
            # Floats are refused: the caller must decide the decimal
            # precision of the published data (scores are published at fixed
            # decimal precision; parse them as strings).
            raise TypeError(
                f"float input {x!r} refused - pass the published decimal as a string"
            )
        else:
            raise TypeError(f"unsupported type {type(x)}")
    return out


def _prefix_sums(xs):
    """Prefix sums of x and x^2 (exact)."""
    ps = [Fraction(0)]
    ps2 = [Fraction(0)]
    for x in xs:
        ps.append(ps[-1] + x)
        ps2.append(ps2[-1] + x * x)
    return ps, ps2


def _block_ssq(ps, ps2, i, j):
    """SSQ of sorted block xs[i:j] (0-indexed, j exclusive) about its mean. Exact."""
    n = j - i
    if n <= 1:
        return Fraction(0)
    s = ps[j] - ps[i]
    s2 = ps2[j] - ps2[i]
    return s2 - s * s / n


def dp_optimal_partition(xs, k):
    """Globally optimal k-partition of xs (any order) minimizing within-cluster SSQ.

    Returns dict with:
      ssq        : Fraction, the exact optimal within-cluster sum of squares
      boundaries : list of k+1 indices into the sorted array (block b = sorted[bnds[b]:bnds[b+1]])
      blocks     : list of k lists of sorted values
      means      : list of k Fractions
      sorted_x   : the sorted values (Fractions)

    O(n^2 k) exact DP. n ~ 800, k = 5 is fine. If k > #distinct values, the optimum
    is 0 with some empty-block convention refused: we require k <= n and raise if
    k exceeds the number of points (CMS handles small-n measures separately; the
    caller must not ask for more clusters than points).
    """
    xs = sorted(_to_fractions(xs))
    n = len(xs)
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    ps, ps2 = _prefix_sums(xs)

    INF = None  # sentinel: unreachable
    # D[m][j] = optimal SSQ of first j points in m blocks (each nonempty)
    D = [[INF] * (n + 1) for _ in range(k + 1)]
    B = [[0] * (n + 1) for _ in range(k + 1)]  # backpointer: start of last block
    D[0][0] = Fraction(0)
    for m in range(1, k + 1):
        for j in range(m, n + 1):
            best = INF
            best_i = -1
            # last block is xs[i:j]; i ranges over [m-1, j-1]
            for i in range(m - 1, j):
                prev = D[m - 1][i]
                if prev is INF:
                    continue
                cand = prev + _block_ssq(ps, ps2, i, j)
                if best is INF or cand < best:
                    best = cand
                    best_i = i
            D[m][j] = best
            B[m][j] = best_i

    boundaries = [n]
    j = n
    for m in range(k, 0, -1):
        i = B[m][j]
        boundaries.append(i)
        j = i
    boundaries.reverse()

    blocks = [xs[boundaries[b]:boundaries[b + 1]] for b in range(k)]
    means = [sum(b) / len(b) for b in blocks]
    return {
        "ssq": D[k][n],
        "boundaries": boundaries,
        "blocks": blocks,
        "means": means,
        "sorted_x": xs,
    }


def dp_optimal_partitions_all(xs, k, cap=64):
    """ALL optimal contiguous k-partitions (exact ties in total SSQ).

    Returns (ssq, list-of-boundary-lists). Needed for the counterfactual
    records: if the SSQ optimum is attained by several partitions, the
    'optimal cut points' are set-valued and must be reported as a set.
    Note contiguity is WLOG for the OPTIMAL VALUE, and every optimal partition
    has a contiguous representative with identical multiset of clusters; a
    non-contiguous optimum can only differ by exchanging equal values across
    adjacent clusters, which does not change cluster min/max boundaries beyond
    what the contiguous set already spans. Cap guards pathological tie blowup
    (raises RuntimeError rather than truncating silently).
    """
    xs = sorted(_to_fractions(xs))
    n = len(xs)
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    ps, ps2 = _prefix_sums(xs)
    D = [[None] * (n + 1) for _ in range(k + 1)]
    ARGS = [[None] * (n + 1) for _ in range(k + 1)]  # all argmin i lists
    D[0][0] = Fraction(0)
    for m in range(1, k + 1):
        for j in range(m, n + 1):
            best = None
            args = []
            for i in range(m - 1, j):
                prev = D[m - 1][i]
                if prev is None:
                    continue
                cand = prev + _block_ssq(ps, ps2, i, j)
                if best is None or cand < best:
                    best = cand
                    args = [i]
                elif cand == best:
                    args.append(i)
            D[m][j] = best
            ARGS[m][j] = args
    # enumerate boundary sets by DFS over tied backpointers
    out = []

    def rec(m, j, tail):
        if len(out) > cap:
            raise RuntimeError(f"optimal-partition tie blowup: >{cap} optima")
        if m == 0:
            if j == 0:
                out.append([0] + tail)
            return
        for i in ARGS[m][j]:
            rec(m - 1, i, [j] + tail)

    rec(k, n, [])
    return D[k][n], out


def partition_ssq(xs, labels):
    """Exact SSQ of an arbitrary labeled partition (labels parallel to xs).

    Used to score the published-method (Ward) partition against the exact
    optimum: gap = partition_ssq(x, ward_labels) - dp_optimal_partition(x, k)['ssq']
    >= 0, with equality iff Ward found an optimum. Exact, so the sign of the gap
    is an exact statement, not a numerical claim.
    """
    xs = _to_fractions(xs)
    groups = {}
    for x, l in zip(xs, labels):
        groups.setdefault(l, []).append(x)
    total = Fraction(0)
    for g in groups.values():
        n = len(g)
        s = sum(g)
        s2 = sum(v * v for v in g)
        total += s2 - s * s / n
    return total


def brute_force_optimal(xs, k):
    """Exhaustive check over all contiguous k-partitions of sorted data. Small n only."""
    xs = sorted(_to_fractions(xs))
    n = len(xs)
    ps, ps2 = _prefix_sums(xs)
    best = None
    best_bnds = None
    for cuts in combinations(range(1, n), k - 1):
        bnds = [0, *cuts, n]
        ssq = sum(_block_ssq(ps, ps2, bnds[b], bnds[b + 1]) for b in range(k))
        if best is None or ssq < best:
            best = ssq
            best_bnds = bnds
    return best, list(best_bnds)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random

    random.seed(0)
    # DP vs brute force on random small instances, exact equality required
    for trial in range(200):
        n = random.randint(3, 14)
        k = random.randint(1, min(5, n))
        xs = [str(random.randint(0, 1000) / 10) for _ in range(n)]
        dp = dp_optimal_partition(xs, k)
        bf, _ = brute_force_optimal(xs, k)
        assert dp["ssq"] == bf, (trial, xs, k, dp["ssq"], bf)
    # non-contiguous labeled partition is never better than DP optimum
    for trial in range(200):
        n = random.randint(5, 12)
        k = random.randint(2, min(5, n - 1))
        xs = [str(random.randint(0, 1000) / 10) for _ in range(n)]
        labels = [random.randint(0, k - 1) for _ in range(n)]
        if len(set(labels)) < k:
            continue
        assert partition_ssq(xs, labels) >= dp_optimal_partition(xs, k)["ssq"]
    # uniqueness enumeration: brute-force count of SSQ-tied contiguous partitions
    for trial in range(100):
        n = random.randint(4, 12)
        k = random.randint(2, min(4, n))
        xs = [str(random.randint(0, 60) / 10) for _ in range(n)]
        ssq, opts = dp_optimal_partitions_all(xs, k)
        bf, _ = brute_force_optimal(xs, k)
        assert ssq == bf
        # count all optimal contiguous partitions by brute force
        sx = sorted(_to_fractions(xs))
        ps, ps2 = _prefix_sums(sx)
        cnt = 0
        for cuts in combinations(range(1, n), k - 1):
            bnds = [0, *cuts, n]
            s = sum(_block_ssq(ps, ps2, bnds[b], bnds[b + 1]) for b in range(k))
            if s == bf:
                cnt += 1
        assert cnt == len(opts), (xs, k, cnt, len(opts))
    print("dp_kmeans: all self-tests passed (incl. tie enumeration vs brute force)")
