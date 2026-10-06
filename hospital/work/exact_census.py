#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Exact-optimum vs published census for one SAS-era hospital-stars release.

Per peer group: the exact 1-D k-means optimum (src/dp_kmeans, exact rational
arithmetic) vs the published FASTCLUS star assignment (replayed by
work/sas_replay.py, which reproduces the published Care Compare snapshot with
zero mismatches; record runs/sas_replay_<year>.json).

Register: scores enter as Fraction(float) of the replay pipeline's IEEE doubles
(exact binary rationals). This is a tighter register than the decimal-string
register of dp_taste_2026 (R write.csv, about 15 significant digits). The SAS
era has no safety cap, so the published side is the published assignment.

Usage: python3 exact_census.py <release>    e.g. 2021-04
Writes runs/exact_census_<year>.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.2 and Table 1
    (hospital stars 2021-2025); Appendix C.1.
Exit codes: 0 on success; 2 on a missing argument or a missing input (one line
    on stderr names it).
Requires: Python >= 3.10, standard library only.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import json
import sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, _REC + "/src")
from dp_kmeans import (dp_optimal_partition, dp_optimal_partitions_all,
                       partition_ssq)  # noqa: E402

BASE = _REC + "/hospital"


def main(release):
    year = release.split("-")[0]
    rows = list(csv.DictReader(
        open(f"{BASE}/work/sas_replay_out/star_{release}.csv")))
    rated = [r for r in rows
             if r["report_indicator"] == "1" and r["star"] not in ("NA", "")]
    out = {
        "run": f"exact_census_{year}",
        "register": ("Exact comparison in binary-double arithmetic: Fraction(float) "
                     "of the replay doubles; the replay matches the published stars "
                     f"with zero mismatches (runs/sas_replay_{year}.json); there is no "
                     "safety cap in the SAS era, so the replayed assignment equals the "
                     "published assignment."),
        "peer_groups": {},
        "total_rated": len(rated),
    }
    total_diff = 0
    moves_total = Counter()
    detail = []
    for tg, label in ((3, "peer3"), (4, "peer4"), (5, "peer5")):
        sub = [r for r in rated if r["Total_measure_group_cnt"] == str(tg)]
        if not sub:
            continue
        xs = [Fraction(float(r["summary_score"])) for r in sub]  # exact binary
        stars_cms = [int(r["star"]) for r in sub]
        dp = dp_optimal_partition(xs, 5)
        ssq_opt, all_opt = dp_optimal_partitions_all(xs, 5)
        bnds, sorted_x = dp["boundaries"], dp["sorted_x"]
        boundary_ties = sum(1 for b in range(1, 5)
                            if sorted_x[bnds[b] - 1] == sorted_x[bnds[b]])
        block_max = [sorted_x[bnds[b + 1] - 1] for b in range(5)]

        def dp_star(v):
            for b in range(5):
                if v <= block_max[b]:
                    return b + 1
            return 5

        stars_dp = [dp_star(v) for v in xs]
        ssq_cms = partition_ssq(xs, stars_cms)
        gap = ssq_cms - dp["ssq"]
        ndiff = 0
        moves = Counter()
        for r, a, b in zip(sub, stars_cms, stars_dp):
            if a != b:
                ndiff += 1
                moves[f"{a}->{b}"] += 1
                detail.append({"provider_id": r["PROVIDER_ID"], "peer": label,
                               "shipped": a, "dp_optimum": b})
        total_diff += ndiff
        moves_total.update(moves)
        out["peer_groups"][label] = {
            "n": len(sub),
            "cms_star_counts": {s: stars_cms.count(s) for s in range(1, 6)},
            "dp_star_counts": {s: stars_dp.count(s) for s in range(1, 6)},
            "ssq_cms_float": float(ssq_cms),
            "ssq_optimal_float": float(dp["ssq"]),
            "ssq_gap_float": float(gap),
            "ssq_gap_exact": f"{gap.numerator}/{gap.denominator}",
            "cms_is_optimal": ssq_cms == dp["ssq"],
            "n_optimal_partitions": len(all_opt),
            "boundary_value_ties": boundary_ties,
            "hospitals_star_differs": ndiff,
            "moves": dict(moves),
        }
        print(f"[{release}] {label} n={len(sub)} diff={ndiff} "
              f"gap={float(gap):.6g} unique={len(all_opt) == 1}", flush=True)
    ups = sum(v for k, v in moves_total.items()
              if int(k.split("->")[1]) > int(k.split("->")[0]))
    downs = sum(v for k, v in moves_total.items()
                if int(k.split("->")[1]) < int(k.split("->")[0]))
    out["total_hospitals_star_differs"] = total_diff
    out["moves_total"] = dict(moves_total)
    out["moves_up"] = ups
    out["moves_down"] = downs
    out["star_differs_detail"] = detail
    dest = f"{BASE}/runs/exact_census_{year}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{release}] TOTAL differs={total_diff} of {len(rated)} "
          f"(up={ups}, down={downs}) -> {dest}", flush=True)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and len(sys.argv) < 2:
    print("usage: python3 exact_census.py <release>   (2021-04, 2022-07, 2023-07, 2024-07 or 2025-07)",
          file=sys.stderr)
    sys.exit(2)
if __name__ == "__main__" and not _os.path.exists(f"{BASE}/work/sas_replay_out/star_{sys.argv[1]}.csv"):
    print("exact_census: missing input hospital/work/sas_replay_out/star_%s.csv (the replay table "
          "sas_replay.py writes; the releases are 2021-04, 2022-07, 2023-07, 2024-07 and "
          "2025-07)" % sys.argv[1],
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1])
