# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""HCAHPS: all 8 starred measures on the current Provider Data Catalog refresh
(discharges 07/2024-06/2025). Per measure: the published partition (checked
for interval consistency) vs the exact SSQ optimum (weighted exact DP plus
exhaustive enumeration of all optimal interval partitions), the gap and the
migration table. Output: HCAHPS_CENSUS_CURRENT.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 (the May 2026
    HCAHPS refresh: 3,363 stars move, 2,118 up and 1,245 down).
Run:  cd stars_census/legb_hcahps && python3 census_current.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10, standard library only.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction

sys.path.insert(0, _REC + "/src")
from weighted_exact import weighted, wd_optimal, wd_all_optima, \
    wd_ssq_of_intervals  # noqa: E402

CSV = "data/HCAHPS-Hospital.csv"
OUT = "HCAHPS_CENSUS_CURRENT.json"
MEASURES = ["H_COMP_1", "H_COMP_2", "H_COMP_5", "H_COMP_6",
            "H_CLEAN", "H_QUIET", "H_HSP_RATING", "H_RECMND"]


def main():
    lin = defaultdict(dict)
    star = defaultdict(dict)
    for r in csv.DictReader(open(CSV, newline="", encoding="utf-8-sig")):
        mid, ccn = r["HCAHPS Measure ID"], r["Facility ID"]
        for m in MEASURES:
            if mid == f"{m}_LINEAR_SCORE" and r["HCAHPS Linear Mean Value"] \
                    not in ("Not Applicable", "Not Available", ""):
                lin[m][ccn] = int(r["HCAHPS Linear Mean Value"])
            elif mid == f"{m}_STAR_RATING" and r["Patient Survey Star Rating"] \
                    not in ("Not Applicable", "Not Available", ""):
                star[m][ccn] = int(r["Patient Survey Star Rating"])
    rows = {}
    for m in MEASURES:
        common = sorted(set(lin[m]) & set(star[m]))
        vals = [lin[m][c] for c in common]
        pubs = [star[m][c] for c in common]
        by_score = defaultdict(set)
        for v, s in zip(vals, pubs):
            by_score[v].add(s)
        consistent = all(len(s) == 1 for s in by_score.values())
        if not consistent:
            rows[m] = {"n": len(vals), "error": "published stars not "
                       "interval-consistent",
                       "offending_scores": {v: sorted(s) for v, s in
                                            by_score.items() if len(s) > 1}}
            continue
        iv = {}
        for v, s in sorted(by_score.items()):
            s = next(iter(s))
            lo, hi = iv.get(s, (v, v))
            iv[s] = (min(lo, v), max(hi, v))
        pub_intervals = [list(iv[s]) for s in sorted(iv)]
        wp = weighted(vals)
        opt_ssq, bounds = wd_optimal(wp, 5)
        _, optima = wd_all_optima(wp, 5, cap=64)
        opt_intervals = [[int(wp[i][0]), int(wp[j][0])] for i, j in bounds]
        pub_ssq = wd_ssq_of_intervals(
            wp, [(Fraction(a), Fraction(b)) for a, b in pub_intervals])
        gap = pub_ssq - opt_ssq

        def star_of(v, ivs):
            for s, (a, b) in enumerate(ivs, 1):
                if a <= v <= b:
                    return s
            raise ValueError((v, ivs))
        mig = Counter()
        for v in vals:
            a, b = star_of(v, pub_intervals), star_of(v, opt_intervals)
            if a != b:
                mig[(a, b)] += 1
        rows[m] = {
            "n": len(vals),
            "published_intervals": pub_intervals,
            "optimal_intervals": opt_intervals,
            "published_ssq": str(pub_ssq), "optimal_ssq": str(opt_ssq),
            "gap": str(gap), "gap_float": float(gap),
            "published_is_optimal": gap == 0,
            "n_optimal_partitions": len(optima),
            "migration": {f"{a}->{b}": n for (a, b), n in sorted(mig.items())},
            "n_migrating": sum(mig.values()),
        }
    n_off = sum(1 for r in rows.values()
                if "error" not in r and not r["published_is_optimal"])
    bank = {
        "register": ("HCAHPS: all 8 starred measures, Provider Data Catalog "
                     "refresh covering discharges 07/2024-06/2025; published "
                     "star partition vs the exact SSQ optimum on the published "
                     "integer linear mean scores (the Technical Notes' "
                     "prescribed inputs), with all optimal interval "
                     "partitions enumerated per measure."),
        "data_sha256_16": hashlib.sha256(open(CSV, "rb").read()).hexdigest()[:16],
        "measures": rows,
        "headline": {"n_measures": len(MEASURES),
                     "off_optimum": n_off,
                     "total_migrating_hospitals": sum(
                         r.get("n_migrating", 0) for r in rows.values())},
    }
    with open(OUT, "w") as f:
        json.dump(bank, f, indent=1)
        f.write("\n")
    for m, r in rows.items():
        if "error" in r:
            print(m, "ERROR:", r["error"])
        else:
            print(f"{m:14s} n={r['n']} gap={r['gap_float']:.3f} "
                  f"optimal={r['published_is_optimal']} "
                  f"n_optima={r['n_optimal_partitions']} mig={r['migration']}")
    print("off-optimum measures:", n_off, "/ 8; total migrating:",
          bank["headline"]["total_migrating_hospitals"])
    print("output sha256-16:",
          hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16])


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not _os.path.exists(CSV):
    print("census_current: missing input data/HCAHPS-Hospital.csv (run from stars_census/legb_hcahps/; the file is the "
          "Provider Data Catalog download, not included; re-fetch it by the "
          "URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
