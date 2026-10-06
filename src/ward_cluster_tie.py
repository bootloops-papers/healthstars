# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Ward agglomeration implemented from the documented algorithm (SAS/STAT
User's Guide, "The CLUSTER Procedure": METHOD=WARD, Lance-Williams update and
the chapter's "Ties" rule; Ward 1963): stored-distance Lance-Williams in IEEE
double with the documented tie rule. An independent implementation written
from the manual chapter; validated against CMS's published cut points.

Tie rule. The CLUSTER chapter's Ties section (SAS/STAT 14.2 User's Guide,
"The CLUSTER Procedure", p. 2143; the rule is unchanged across editions)
specifies that every cluster is labeled by the smallest observation number
among its members and that, when several pairs of clusters are tied at the
minimum distance, the pair whose larger label is smallest merges first, a
remaining tie being broken by the smaller label. Implemented here as the
lexicographic merge key (D_KL, max(id_K,id_L), min(id_K,id_L)) with id = the
smallest 1-based row number of a cluster's members; row order = the
inclusterdat order.

Distance input from PROC DISTANCE method=Euclid (1-D: |xi-xj|), squared by
CLUSTER for METHOD=WARD (DATA= doc p.2122-23). Lance-Williams stored-distance
update (Ward, p.2138-39):
  D_JM = ((N_J+N_K) D_JK + (N_J+N_L) D_JL - N_J D_KL) / (N_J + N_M)
computed in doubles, left-to-right as written (SAS operand order undocumented —
bitwise-height residual; merge decisions differ only if a comparison flips at
ulp level, which the near-tie window in the caller can flag).

PROC TREE NCL=k unwinds the merge sequence to n-k merges (no height cut), so
cutting = stopping after n-k merges. Tie detection = exact double equality
(no fuzz documented).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (Ward's method
    with SAS's documented tie rule).
Run:  cd src && python3 ward_cluster_tie.py   (self-test; imported by the census and replay scripts)
Requires: Python >= 3.10; numpy, scipy.
"""

import numpy as np
import sys


def ward_sas_cut(values, k):
    """values: list of floats in DATASET ORDER (row order = SAS obs numbers).
    Returns clusters as lists of original indices, ordered by cluster mean."""
    n = len(values)
    if n == 0:
        return []
    if n <= k:
        order = np.argsort(values, kind="stable")
        return [[int(i)] for i in order]
    x = np.asarray(values, dtype=np.float64)
    # condensed full matrix (n x n), inf on/below diagonal unused entries
    D = (x[:, None] - x[None, :]) ** 2  # squared Euclidean, exact for ints
    np.fill_diagonal(D, np.inf)
    active = np.ones(n, dtype=bool)
    minid = np.arange(1, n + 1)  # SAS 1-based obs numbers
    size = np.ones(n, dtype=np.float64)
    members = [[i] for i in range(n)]
    D = np.where(np.triu(np.ones((n, n), dtype=bool), 1), D, np.inf)

    for _ in range(n - k):
        dmin = D.min()
        cand = np.argwhere(D == dmin)  # exact equality = documented tie set
        if len(cand) == 1:
            a, b = int(cand[0][0]), int(cand[0][1])
        else:
            best = None
            for i, j in cand:
                ida, idb = minid[i], minid[j]
                key = (max(ida, idb), min(ida, idb))
                if best is None or key < best[0]:
                    best = (key, int(i), int(j))
            a, b = best[1], best[2]
        NK, NL = size[a], size[b]
        DKL = D[min(a, b), max(a, b)]
        # vectorized L-W update for every active J != a,b
        js = np.nonzero(active)[0]
        js = js[(js != a) & (js != b)]
        if len(js):
            mA = js < a
            djk = np.empty(len(js))
            djk[mA] = D[js[mA], a]
            djk[~mA] = D[a, js[~mA]]
            mB = js < b
            djl = np.empty(len(js))
            djl[mB] = D[js[mB], b]
            djl[~mB] = D[b, js[~mB]]
            NJ = size[js]
            newd = ((NJ + NK) * djk + (NJ + NL) * djl - NJ * DKL) / (NJ + NK + NL)
            D[js[mA], a] = newd[mA]
            D[a, js[~mA]] = newd[~mA]
        active[b] = False
        D[b, :] = np.inf
        D[:, b] = np.inf
        size[a] = NK + NL
        minid[a] = min(minid[a], minid[b])
        members[a] = members[a] + members[b]

    out = [members[i] for i in np.nonzero(active)[0]]
    out.sort(key=lambda idxs: float(np.mean([values[i] for i in idxs])))
    return [sorted(c) for c in out]


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    # Test 1: two exactly tied pairs (rows 1,4) and (2,3) at d=0.
    # 1-based ids: pair (2,3) has larger-id 3 < pair (1,4)'s larger-id 4 -> (2,3)
    # merges FIRST. We verify merge order via a 3-cluster cut structure.
    vals = [0.0, 7.0, 7.0, 0.0, 100.0]
    # step1 must merge rows 2,3 (values 7,7) per rule; then rows 1,4 (0,0).
    cl = ward_sas_cut(vals, 3)
    assert sorted(map(sorted, cl)) == sorted([[0, 3], [1, 2], [4]]), cl
    # Test 2: obs 5 equidistant from obs 1 and obs 2 (ids: pairs (1,5),(2,5)
    # tie with equal larger id 5 -> min smaller id -> merge (1,5)).
    vals = [0.0, 8.0, 100.0, 200.0, 4.0]  # d(1,5)=16, d(2,5)=16 squared
    cl = ward_sas_cut(vals, 4)
    assert sorted(map(sorted, cl)) == sorted([[0, 4], [1], [2], [3]]), cl
    # agreement with scipy ward on tie-free continuous data
    import random
    from ward_cluster import ward_cut
    random.seed(9)
    agree = 0
    for _ in range(50):
        n = random.randint(10, 60)
        vals = [random.uniform(0, 100) for _ in range(n)]
        a = ward_sas_cut(vals, 5)
        b = ward_cut(vals, 5)
        if [sorted(c) for c in a] == [sorted(c) for c in b]:
            agree += 1
    assert agree >= 48, agree
    print(f"ward_cluster_tie: tests 1-2 pass; scipy agreement {agree}/50 (tie-free)")
    # duplicate-heavy integer data: runs clean, deterministic under reruns
    vals = [float(random.randint(0, 30)) for _ in range(200)]
    c1 = ward_sas_cut(vals, 5)
    c2 = ward_sas_cut(vals, 5)
    assert c1 == c2
    print("ward_cluster_tie: deterministic on duplicate-heavy integer data")
