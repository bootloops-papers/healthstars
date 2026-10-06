# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Star year 2023 optimality-gap table on the full score list.

Mirrors src/exact_gaps.py except for the 2023 methodology facts
(spec/SPEC_NOTES_2023.md): no outlier trim (2023 has none), membership =
documented-rules-only hypothesis cost+d60r with the HEDIS-HOS 60%-waiver
(inputs_2023). No resampling groups, no RNG.

Fields per row:
  ward_ssq, dp_ssq, gap : exact (Fraction strings)
  off_optimum           : exact boolean (gap > 0) given the stated input
  dp_n_optima           : exact (count of SSQ-tied optimal partitions, cap 32)
  membership            : 'hypothesis:cost+d60r(hoswaiver)' always; 2023 has
                          no published outlier bounds, so no row can be
                          fence-exact; the row claim is conditional on the
                          stated input.
Each row states that the published partition of the stated input does not
minimize the within-cluster sum of squares.

Run:  python3 exact_gaps_2023.py [workers]  -> runs/exact_gaps_2023.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1, Table 2 note a
    (star year 2023: 36 of 38 computations miss the minimum).
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import sys
from fractions import Fraction as F

import inputs_2023 as I23
from dp_kmeans import dp_optimal_partition, dp_optimal_partitions_all, partition_ssq
from ward_cluster import cutpoints_from_clusters
from ward_cluster_tie import ward_sas_cut

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "runs", "exact_gaps_2023.json")


def _run(j):
    scores = [s for _, s in j["kept"]]          # NO Tukey trim in 2023
    vals = [float(s) for s in scores]
    k = min(5, len(set(vals)))
    ward_cl = ward_sas_cut(vals, k)
    labels = [0] * len(vals)
    for ci, c in enumerate(ward_cl):
        for i in c:
            labels[i] = ci
    ward_ssq = partition_ssq(scores, labels)
    dp = dp_optimal_partition(scores, k)
    gap = ward_ssq - dp["ssq"]
    try:
        _, opts = dp_optimal_partitions_all(scores, k, cap=32)
        n_opt = len(opts)
    except RuntimeError:
        n_opt = ">32"
    ward_cps = cutpoints_from_clusters(ward_cl, vals, j["higher"])
    dpb = dp["boundaries"]
    if j["higher"]:
        dp_cps = [dp["sorted_x"][b] for b in dpb[1:-1]]
    else:
        dp_cps = [dp["sorted_x"][b - 1] for b in dpb[1:-1]]

    def stars(cps, x, higher):
        xs = F(x)
        if higher:
            return 1 + sum(1 for t in cps if xs >= F(str(t)))
        return 5 - sum(1 for t in cps if xs > F(str(t)))

    n_star_delta = sum(
        1 for _, s in j["pool"]
        if stars(ward_cps, s, j["higher"]) != stars(dp_cps, s, j["higher"]))
    return {"year": "2023", "mid": j["mid"], "org": j["org"],
            "name": j["name"], "n_input": len(scores),
            "n_trimmed": len(scores),          # no trim: equals n_input
            "n_pool": len(j["pool"]), "k": k,
            "membership": j["membership"],
            "tukey": "none (2023 methodology)",
            "ward_ssq": str(ward_ssq), "dp_ssq": str(dp["ssq"]),
            "gap": str(gap), "gap_float": float(gap),
            "off_optimum": gap > 0, "dp_n_optima": n_opt,
            "ward_cutpoints": [str(c) for c in ward_cps],
            "dp_cutpoints": [str(c) for c in dp_cps],
            "n_star_delta_fullsample": n_star_delta}


def main(workers=6):
    jobs = I23.build_splits()
    print(f"jobs: {len(jobs)}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    off = [r for r in rows if r["off_optimum"]]
    uniq = [r for r in rows if r["dp_n_optima"] == 1]
    tot_delta = sum(r["n_star_delta_fullsample"] for r in rows)
    rows.sort(key=lambda r: -r["gap_float"])
    out = {"year": "2023",
           "membership": I23.HYP_LABEL,
           "membership_note": "hypothesis basis: 2023 has no outlier trim, "
                              "hence no published outlier bounds; rows are "
                              "exact given this input.",
           "tukey": "none (2023 methodology - see spec/SPEC_NOTES_2023.md)",
           "n_splits": len(rows),
           "off_optimum": len(off),
           "dp_unique": len(uniq),
           "max_gap": rows[0]["gap_float"] if rows else None,
           "max_gap_split": (f"{rows[0]['mid']}/{rows[0]['org']}"
                             if rows else None),
           "total_star_delta_fullsample": tot_delta,
           "rows": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nEXACT GIVEN INPUT (full score list): {len(off)}/{len(rows)} "
          f"splits off the SSQ optimum; unique DP optimum: {len(uniq)}/{len(rows)}")
    print(f"full-score-list star deltas: {tot_delta} contract-measure cells")
    for r in rows[:10]:
        print(f"  {r['mid']} {r['org']:>5} {r['name'][:40]:40} "
              f"gap={r['gap_float']:.4f} stars_moved={r['n_star_delta_fullsample']:3d} "
              f"optima={r['dp_n_optima']}")
    print(f"saved {OUT}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
