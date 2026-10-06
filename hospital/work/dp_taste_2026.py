#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Exact-optimum vs published census for the 2026 release on the decimal-string
register.

Per peer group (2026 release): compare the replayed CMS k-means star assignment
(pre-safety-cap; the replay reproduces every published star on all 3,182 jointly
rated hospitals) against the exact 1-D k-means optimum (dp_kmeans, exact
rational arithmetic).

Register: scores enter as the decimal strings printed by R's write.csv (about
15 significant digits); the comparison is exact on that register. The
binary-double register is in exact_census_2026.py.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.2 and Appendix C.1
    (the decimal-string register of the 2026 comparison).
Run:  cd hospital/work && python3 dp_taste_2026.py
Requires: Python >= 3.10, standard library only.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json, sys
sys.path.insert(0, _REC + "/src")
from fractions import Fraction
from dp_kmeans import dp_optimal_partition, dp_optimal_partitions_all, partition_ssq

BASE = _REC + "/hospital"
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

rows = list(csv.DictReader(open(f"{BASE}/work/R_output/Star_precap_2026apr.csv")))
rated = [r for r in rows if r["report_indicator"] == "1" and r["star"] not in ("NA", "")]
print("rated:", len(rated))

out = {"run": "dp_taste_2026",
       "register": "decimal-string register of R write.csv (about 15 significant digits)",
       "peer_groups": {}, "total_rated": len(rated)}
total_diff = 0
for grp_label, key in [("peer3", "1) # of groups=3"), ("peer4", "2) # of groups=4"),
                       ("peer5", "3) # of groups=5")]:
    sub = [r for r in rated if r["cnt_grp"] == key]
    xs = [r["summary_score"] for r in sub]
    stars_cms = [int(r["star"]) for r in sub]
    n = len(sub)
    dp = dp_optimal_partition(xs, 5)
    ssq_opt, all_opt = dp_optimal_partitions_all(xs, 5)
    # DP blocks are ascending -> block b (0-indexed) = star b+1
    bnds = dp["boundaries"]
    sorted_x = dp["sorted_x"]
    # value-interval thresholds: block b covers sorted_x[bnds[b]:bnds[b+1]]
    # assign each hospital by locating its exact value among block ranges;
    # ambiguity only possible if equal values straddle a boundary — check.
    ambiguous = 0
    for b in range(1, 5):
        if sorted_x[bnds[b] - 1] == sorted_x[bnds[b]]:
            ambiguous += 1
    # assign via sorted position of value (stable: use bisect on block max)
    block_max = [sorted_x[bnds[b + 1] - 1] for b in range(5)]
    def dp_star(v):
        f = Fraction(v)
        for b in range(5):
            if f <= block_max[b]:
                return b + 1
        return 5
    stars_dp = [dp_star(v) for v in xs]
    ssq_cms = partition_ssq(xs, stars_cms)
    ndiff = sum(1 for a, b in zip(stars_cms, stars_dp) if a != b)
    total_diff += ndiff
    moved = {}
    for a, b in zip(stars_cms, stars_dp):
        if a != b:
            moved[f"{a}->{b}"] = moved.get(f"{a}->{b}", 0) + 1
    out["peer_groups"][grp_label] = {
        "n": n,
        "cms_star_counts": {s: stars_cms.count(s) for s in range(1, 6)},
        "dp_star_counts": {s: stars_dp.count(s) for s in range(1, 6)},
        "ssq_cms_float": float(ssq_cms),
        "ssq_optimal_float": float(dp["ssq"]),
        "ssq_gap_float": float(ssq_cms - dp["ssq"]),
        "cms_is_optimal": ssq_cms == dp["ssq"],
        "n_optimal_partitions": len(all_opt),
        "boundary_value_ties": ambiguous,
        "hospitals_star_differs": ndiff,
        "moves": moved,
    }
    print(grp_label, "n=", n, "diff=", ndiff, "gap=", float(ssq_cms - dp["ssq"]),
          "opt-unique:", len(all_opt) == 1)

out["total_hospitals_star_differs_precap"] = total_diff
with open(f"{BASE}/runs/dp_taste_2026.json", "w") as f:
    json.dump(out, f, indent=2)
print("TOTAL star-differs (pre-cap):", total_diff, "of", len(rated))
print("wrote", f"{BASE}/runs/dp_taste_2026.json")
