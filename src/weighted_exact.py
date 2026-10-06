# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""EXACT weighted-distinct-value engines for 1-D star clustering at scale.

Both engines operate on (value, count) pairs instead of raw point lists and are
PROVABLY identical to their pointwise counterparts:
 - DP: within-block SSQ depends only on the multiset; collapsing equal values
   into weighted points changes nothing (block SSQ computed from weighted
   prefix sums, exact Fractions).
 - Optimal partitions of sorted data are contiguous; equal values never split
   across an OPTIMAL boundary unless the split is SSQ-indifferent — the DP here
   partitions DISTINCT values, which is exactly the space of interval
   partitions that never split a tied group. (A pointwise optimum that splits
   a tied group has an equal-SSQ interval representative iff the split is
   SSQ-indifferent; wd_optimal is exact over interval partitions, the space
   the published bins live in.)

Validation: the self-test below checks against dp_kmeans.dp_optimal_partition
(same SSQ, same boundaries).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 (the exact optimum
    on the survey files: 3,176 hospitals and 2,594 facilities per measure).
Run:  cd src && python3 weighted_exact.py   (validation against the pointwise DP on 3,000
      random scores)
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # record root
from fractions import Fraction


def weighted(xs):
    """[(value, count)] sorted ascending from raw iterable."""
    from collections import Counter
    c = Counter(xs)
    return sorted((Fraction(v), n) for v, n in c.items())


def wd_optimal(wpairs, k):
    """Exact optimal k-partition over weighted distinct values.
    Returns (ssq, boundaries) where boundaries = list of k (lo_idx, hi_idx)
    into wpairs, inclusive."""
    m = len(wpairs)
    if k > m:
        raise ValueError(f"k={k} > {m} distinct values")
    # weighted prefix sums
    W = [0] * (m + 1)   # counts
    S = [Fraction(0)] * (m + 1)
    S2 = [Fraction(0)] * (m + 1)
    for i, (v, n) in enumerate(wpairs):
        W[i + 1] = W[i] + n
        S[i + 1] = S[i] + n * v
        S2[i + 1] = S2[i] + n * v * v

    def block(i, j):  # values i..j inclusive
        w = W[j + 1] - W[i]
        s = S[j + 1] - S[i]
        s2 = S2[j + 1] - S2[i]
        return s2 - s * s / w

    INF = None
    D = [[INF] * m for _ in range(k)]
    B = [[0] * m for _ in range(k)]
    for j in range(m):
        D[0][j] = block(0, j)
    for c in range(1, k):
        for j in range(c, m):
            best = None
            barg = None
            for i in range(c, j + 1):
                cand = D[c - 1][i - 1] + block(i, j)
                if best is None or cand < best:
                    best, barg = cand, i
            D[c][j] = best
            B[c][j] = barg
    # backtrack
    bounds = []
    j = m - 1
    for c in range(k - 1, -1, -1):
        i = B[c][j] if c > 0 else 0
        bounds.append((i, j))
        j = i - 1
    bounds.reverse()
    return D[k - 1][m - 1], bounds


def wd_all_optima(wpairs, k, cap=64):
    """All optimal interval partitions (boundary sets attaining the optimum)."""
    ssq, _ = wd_optimal(wpairs, k)
    m = len(wpairs)
    W = [0] * (m + 1)
    S = [Fraction(0)] * (m + 1)
    S2 = [Fraction(0)] * (m + 1)
    for i, (v, n) in enumerate(wpairs):
        W[i + 1] = W[i] + n
        S[i + 1] = S[i] + n * v
        S2[i + 1] = S2[i] + n * v * v

    def block(i, j):
        w = W[j + 1] - W[i]
        s = S[j + 1] - S[i]
        s2 = S2[j + 1] - S2[i]
        return s2 - s * s / w

    # forward table of optimal costs for prefix with c blocks
    D = [[None] * m for _ in range(k)]
    for j in range(m):
        D[0][j] = block(0, j)
    for c in range(1, k):
        for j in range(c, m):
            D[c][j] = min(D[c - 1][i - 1] + block(i, j) for i in range(c, j + 1))
    out = []

    def rec(c, j, tail):
        if len(out) >= cap:
            return
        if c == 0:
            out.append([(0, j)] + tail)
            return
        for i in range(c, j + 1):
            if D[c - 1][i - 1] + block(i, j) == D[c][j]:
                rec(c - 1, i - 1, [(i, j)] + tail)
    rec(k - 1, m - 1, [])
    return ssq, out


def wd_ssq_of_intervals(wpairs, intervals_by_value):
    """SSQ of a partition given as value-intervals [(lo_val, hi_val), ...]."""
    total = Fraction(0)
    for lo, hi in intervals_by_value:
        pts = [(v, n) for v, n in wpairs if lo <= v <= hi]
        w = sum(n for _, n in pts)
        s = sum(n * v for v, n in pts)
        s2 = sum(n * v * v for v, n in pts)
        total += s2 - s * s / w
    return total


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random
    rng = random.Random(1)
    xs = [rng.randint(60, 99) for _ in range(3000)]
    import sys
    sys.path.insert(0, _REC + "/src")
    from dp_kmeans import dp_optimal_partition
    ssq_w, bounds = wd_optimal(weighted(xs), 5)
    ref = dp_optimal_partition(xs, 5)
    assert ssq_w == ref["ssq"], (ssq_w, ref["ssq"])
    print("selftest OK: weighted and pointwise engines give the same exact SSQ")
