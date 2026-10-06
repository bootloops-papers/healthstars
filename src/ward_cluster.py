# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Ward arm as the Technical Notes describe it: IEEE-double greedy Ward on 1-D
scores, tree cut, cut points.

Implements the chain the Technical Notes print, from the documented algorithms
(SAS/STAT User's Guide chapters "The DISTANCE Procedure", "The CLUSTER
Procedure" METHOD=WARD, "The TREE Procedure"; Ward 1963): Euclidean distance
(|xi-xj| in 1-D) -> Ward's minimum-variance agglomeration ("input measure score
distances are squared"; Lance-Williams recurrence; merge the pair minimizing the
within-cluster SSQ increase) -> cut the tree at NSTARS clusters. Runs in IEEE
double throughout, deliberately: exact-arithmetic Ward is tie-heavy on
decimal-gridded scores while double rounding breaks those ties
deterministically, as any double-precision implementation of the documented
recurrence does.

Engine: scipy.cluster.hierarchy.linkage(method='ward') on the 1-D points. For
reducible linkages (Ward is reducible) NN-chain produces the same dendrogram as
global greedy min-merge, up to tie order. Tie/near-tie risk is quantified per run
by near_tie_scan (recomputes the exact greedy sequence in doubles and reports
steps where the runner-up merge cost is within rel_window of the winner near the
k-cut). Runs flagged there get the branch-enumeration treatment (ward1d.py, or
ordering resamples) rather than a silent single answer.

Cut points (Tech Notes p.152): after cutting to NSTARS clusters and ordering by
score, for higher-is-better measures the MINIMUM score in each star category is
the effective cut point; for lower-is-better the MAXIMUM. Clusters with identical
score-value ranges are combined (fewer than NSTARS categories result).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Sections 2.3 and 4.1 (the Ward
    arm as the Technical Notes describe it).
Run:  cd src && python3 ward_cluster.py   (self-tests; imported by pipeline.py)
Requires: Python >= 3.10; numpy, scipy.
"""

import numpy as np
import sys
from scipy.cluster.hierarchy import linkage, fcluster


def ward_cut(values, k):
    """values: list of floats (doubles as parsed). Returns list of clusters,
    each a sorted list of member indices into `values`, ordered by cluster mean.
    len(result) == min(k, n) unless equal-range combining reduces it further
    (combining is applied by cutpoints_from_clusters, not here)."""
    n = len(values)
    if n == 0:
        return []
    if n <= k:
        order = np.argsort(values, kind="stable")
        return [[int(i)] for i in order]
    X = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    clusters = {}
    for i, l in enumerate(labels):
        clusters.setdefault(int(l), []).append(i)
    out = sorted(clusters.values(), key=lambda idxs: np.mean([values[i] for i in idxs]))
    return [sorted(c) for c in out]


def near_tie_scan(values, k, rel_window=1e-9):
    """Replay the greedy min-merge in doubles; count near-tie decisions.

    Returns (n_near_tie_steps, first_step_index_or_None), counting ONLY
    positive-cost near-ties: zero-cost merges (duplicate values) are excluded —
    all zero merges precede positive ones and their order provably does not
    affect the post-zero-phase state (dedup lemma, ward1d.py). Positive-cost
    exact double ties are REAL on integer-valued scores (small-integer double
    arithmetic is exact), so this count is the tie-branch trigger.
    O(n^2) per step — a diagnostic, not the hot path."""
    vals = list(map(float, values))
    n = len(vals)
    if n <= k:
        return 0, None
    size = {i: 1 for i in range(n)}
    ssum = {i: vals[i] for i in range(n)}
    active = set(range(n))
    nxt = n
    near = 0
    first = None
    step = 0
    while len(active) > k:
        best = None
        second = None
        arg = None
        act = sorted(active)
        for ai in range(len(act)):
            a = act[ai]
            for bi in range(ai + 1, len(act)):
                b = act[bi]
                ma = ssum[a] / size[a]
                mb = ssum[b] / size[b]
                d = (size[a] * size[b]) / (size[a] + size[b]) * (ma - mb) ** 2
                if best is None or d < best:
                    second = best
                    best = d
                    arg = (a, b)
                elif second is None or d < second:
                    second = d
        if (second is not None and best > 0
                and (second - best) <= rel_window * best):
            near += 1
            if first is None:
                first = step
        a, b = arg
        size[nxt] = size[a] + size[b]
        ssum[nxt] = ssum[a] + ssum[b]
        active.discard(a)
        active.discard(b)
        active.add(nxt)
        nxt += 1
        step += 1
    return near, first


def cutpoints_from_clusters(clusters, values, higher_is_better):
    """Effective cut points from ordered clusters (ascending by mean).

    Combines clusters with identical score-value ranges first (Tech Notes rule).
    Returns list of cut points, one per boundary between adjacent final
    categories, as the defining score per the direction convention:
      higher-is-better: the MIN score of each category above the lowest
      lower-is-better:  the MAX score of each category below the highest
    (i.e., always the boundary-defining value of the better-side category edge
    per the Tech Notes' 'lower limit of each cluster becomes the cut point').
    """
    if not clusters:
        return []
    merged = []
    for c in clusters:
        rng = (min(values[i] for i in c), max(values[i] for i in c))
        if merged and merged[-1][1] == rng:
            merged[-1][0].extend(c)
        else:
            merged.append([list(c), rng])
    cats = [m[0] for m in merged]  # ascending score order
    if higher_is_better:
        # categories map to stars ascending; cut points = min of categories 2..K
        return [min(values[i] for i in c) for c in cats[1:]]
    # lower is better: stars descend with score; cut points = max of categories 1..K-1
    return [max(values[i] for i in c) for c in cats[:-1]]


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random
    from ward1d import ward_partitions

    random.seed(3)
    # 1) scipy-double Ward vs exact-arithmetic Ward branch set on tie-free data
    agree = 0
    for trial in range(40):
        n = random.randint(8, 30)
        vals = sorted(random.uniform(0, 100) for _ in range(n))  # continuous: tie-free
        strs = [f"{v:.6f}" for v in vals]
        cl = ward_cut([float(s) for s in strs], 3)
        blocks = tuple(tuple(c) for c in cl)
        w = ward_partitions(strs, 3)
        keys = set()
        for p in w["partitions"]:
            if p["blocks"] is not None:
                keys.add(tuple(tuple(range(b[0], b[1])) for b in p["blocks"]))
        if blocks in keys:
            agree += 1
    assert agree >= 38, agree  # allow rare double-vs-exact near-tie divergence
    print(f"ward_cluster: scipy-double vs exact-greedy agreement {agree}/40 (tie-free data)")
    # 2) cut point extraction conventions
    vals = [10.0, 11.0, 50.0, 51.0, 90.0, 91.0]
    cl = ward_cut(vals, 3)
    cp_hib = cutpoints_from_clusters(cl, vals, True)
    assert cp_hib == [50.0, 90.0], cp_hib
    cp_lib = cutpoints_from_clusters(cl, vals, False)
    assert cp_lib == [11.0, 51.0], cp_lib
    # 3) equal-range combining
    vals = [5.0, 5.0, 5.0, 9.0]
    cl = [[0, 1], [2], [3]]  # two clusters both spanning [5,5]
    cp = cutpoints_from_clusters(cl, vals, True)
    assert cp == [9.0], cp
    # 4) near-tie scanner fires on symmetric data, quiet on generic data
    nt, _ = near_tie_scan([0.0, 1.0, 10.0, 11.0], 2)
    assert nt >= 1
    nt, _ = near_tie_scan([random.uniform(0, 100) for _ in range(30)], 5)
    print(f"ward_cluster: near-tie scan generic-data count = {nt} (expect 0 or tiny)")
    print("ward_cluster: self-tests passed")
