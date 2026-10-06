# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""ICH CAHPS six-measure census: the published partition vs the exact SSQ
optimum on the published linearized scores (the April 2026 methodology
document's own clustered inputs; Ward's minimum variance, 5 clusters).
Interval consistency is checked per measure; all optimal interval partitions
are enumerated; migrations are tabulated.
Output: ICH_CENSUS.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 and Figure 4
    (dialysis survey: none of the six published groupings attains the minimum;
    4,061 facility-measure stars move).
Run:  cd stars_census/legc_ich && python3 census_ich.py
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

CSV = "data/ICH_CAHPS_FACILITY.csv"
OUT = "ICH_CENSUS.json"
MEASURES = [
    ("nephrologists' communication and caring",
     "Linearized score of nephrologists' communication and caring",
     "Star rating of nephrologists' communication and caring"),
    ("quality of dialysis center care and operations",
     "Linearized score of quality of dialysis center care and operations",
     "Star rating of quality of dialysis center care and operations"),
    ("providing information to patients",
     "Linearized score of providing information to patients",
     "Star rating of providing information to patients"),
    ("rating of the nephrologist",
     "Linearized score of rating of the nephrologist",
     "Star rating of the nephrologist"),
    ("rating of the dialysis center staff",
     "Linearized score of rating of the dialysis center staff",
     "Star rating of the dialysis center staff"),
    ("rating of the dialysis facility",
     "Linearized score of rating of the dialysis facility",
     "Star rating of the dialysis facility"),
]
MISSING = ("", "Not Available", "Not Applicable")


def main():
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8-sig")))
    out = {}
    for name, lc, sc in MEASURES:
        pts = []
        for r in rows:
            v, s = r[lc].strip(), r[sc].strip()
            if v not in MISSING and s not in MISSING:
                sf = Fraction(s)
                assert sf.denominator == 1, f"non-integer star {s!r}"
                pts.append((Fraction(v), int(sf)))
        by_score = defaultdict(set)
        for v, s in pts:
            by_score[v].add(s)
        if not all(len(s) == 1 for s in by_score.values()):
            out[name] = {"n": len(pts), "error": "published stars NOT interval-"
                         "consistent", "offending": {str(v): sorted(s) for v, s
                                                     in by_score.items()
                                                     if len(s) > 1}}
            continue
        iv = {}
        for v, s in sorted(by_score.items()):
            s = next(iter(s))
            lo, hi = iv.get(s, (v, v))
            iv[s] = (min(lo, v), max(hi, v))
        pub_intervals = [[str(iv[s][0]), str(iv[s][1])] for s in sorted(iv)]
        vals = [v for v, _ in pts]
        wp = weighted(vals)
        opt_ssq, bounds = wd_optimal(wp, 5)
        _, optima = wd_all_optima(wp, 5, cap=64)
        opt_intervals = [[str(wp[i][0]), str(wp[j][0])] for i, j in bounds]
        pub_ssq = wd_ssq_of_intervals(
            wp, [(Fraction(a), Fraction(b)) for a, b in pub_intervals])
        gap = pub_ssq - opt_ssq

        def star_of(v, ivs):
            for s, (a, b) in enumerate(ivs, 1):
                if Fraction(a) <= v <= Fraction(b):
                    return s
            raise ValueError((v, ivs))
        mig = Counter()
        for v, _ in pts:
            a, b = star_of(v, pub_intervals), star_of(v, opt_intervals)
            if a != b:
                mig[(a, b)] += 1
        out[name] = {
            "n": len(pts),
            "published_intervals": pub_intervals,
            "optimal_intervals": opt_intervals,
            "published_ssq": str(pub_ssq), "optimal_ssq": str(opt_ssq),
            "gap": str(gap), "gap_float": float(gap),
            "published_is_optimal": gap == 0,
            "n_optimal_partitions": len(optima),
            "migration": {f"{a}->{b}": n for (a, b), n in sorted(mig.items())},
            "n_migrating": sum(mig.values()),
        }
    n_off = sum(1 for r in out.values()
                if "error" not in r and not r["published_is_optimal"])
    bank = {
        "register": ("ICH CAHPS six-measure census, Provider Data Catalog "
                     "facility file (surveys Fall 2024 and Spring 2025, April "
                     "2026 reporting): published stars vs the unique exact "
                     "SSQ optimum on the published linearized scores (the "
                     "methodology document's own clustered inputs), with all "
                     "optimal interval partitions enumerated per measure."),
        "data_sha256_16": hashlib.sha256(open(CSV, "rb").read()).hexdigest()[:16],
        "measures": out,
        "headline": {"n_measures": len(MEASURES), "off_optimum": n_off,
                     "total_migrating": sum(r.get("n_migrating", 0)
                                            for r in out.values())},
    }
    with open(OUT, "w") as f:
        json.dump(bank, f, indent=1)
        f.write("\n")
    for m, r in out.items():
        if "error" in r:
            print(m, "ERROR", r.get("offending"))
        else:
            print(f"{m[:44]:44s} n={r['n']} gap={r['gap_float']:.3f} "
                  f"opt={r['published_is_optimal']} "
                  f"n_optima={r['n_optimal_partitions']} mig={r['migration']}")
    print("off-optimum:", n_off, "/6; migrating:",
          bank["headline"]["total_migrating"])
    print("output sha256-16:",
          hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16])


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not _os.path.exists(CSV):
    print("census_ich: missing input data/ICH_CAHPS_FACILITY.csv (run from stars_census/legc_ich/; the file is the "
          "Provider Data Catalog download, not included; re-fetch it by the "
          "URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
