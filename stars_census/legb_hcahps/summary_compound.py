# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""HCAHPS Summary Star under the published vs the exact-optimal per-measure
stars.

Per the Technical Notes (pp. 4-5): Summary Star = normal-rounded simple
average of 6 components: the 4 composite stars + (average of Clean, Quiet
stars) + (average of Rating, Recommend stars). Each hospital's summary star
is recomputed twice, from the published per-measure stars and from the
exact-optimal per-measure stars (the five-refresh census), and summary-level
changes are counted. Consistency check: the published-side recompute vs the
published summary star column (H_STAR_RATING), with mismatches reported.
Rounding = half-up at .5 ('normal rounding rules'), exact Fractions.
Output: HCAHPS_SUMMARY_COMPOUND.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 (the summary star
    changes for 413 of 3,176 hospitals in May 2026 and 316 of 3,183 in February
    2026).
Run:  cd stars_census/legb_hcahps && python3 summary_compound.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10, standard library only.
"""
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction

COMPOSITES = ["H_COMP_1", "H_COMP_2", "H_COMP_5", "H_COMP_6"]
INDIV = ["H_CLEAN", "H_QUIET"]
GLOBAL = ["H_HSP_RATING", "H_RECMND"]
ALL8 = COMPOSITES + INDIV + GLOBAL
MISSING = ("Not Applicable", "Not Available", "")
VINTAGES = [
    ("2025-04-30", "data/archive/2025-04-30/HCAHPS-Hospital.csv"),
    ("2025-08-14", "data/archive/2025-08-14/HCAHPS-Hospital.csv"),
    ("2025-11-26", "data/archive/2025-11-26/HCAHPS-Hospital.csv"),
    ("2026-02-25", "data/archive/2026-02-25/HCAHPS-Hospital.csv"),
    ("current-2026-05-13", "data/HCAHPS-Hospital.csv"),
]
BANK = "HCAHPS_CENSUS_ALLVINTAGES.json"
OUT = "HCAHPS_SUMMARY_COMPOUND.json"


def round_half_up(x):
    return int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))


def summary_star(stars):
    comps = [Fraction(stars[m]) for m in COMPOSITES]
    iv = (Fraction(stars["H_CLEAN"]) + Fraction(stars["H_QUIET"])) / 2
    gl = (Fraction(stars["H_HSP_RATING"]) + Fraction(stars["H_RECMND"])) / 2
    avg = (sum(comps) + iv + gl) / 6
    return round_half_up(avg)


def main():
    census = json.load(open(BANK))
    vout = {}
    for tag, path in VINTAGES:
        cs = census["vintages"][tag]["measures"]
        opt_iv = {m: cs[m]["optimal_intervals"] for m in ALL8
                  if "error" not in cs[m]}
        lin = defaultdict(dict)
        star = defaultdict(dict)
        pub_summary = {}
        for r in csv.DictReader(open(path, newline="", encoding="utf-8-sig")):
            mid, ccn = r["HCAHPS Measure ID"], r["Facility ID"]
            if mid == "H_STAR_RATING" and \
                    r["Patient Survey Star Rating"] not in MISSING:
                pub_summary[ccn] = int(r["Patient Survey Star Rating"])
            for m in ALL8:
                if mid == f"{m}_LINEAR_SCORE" and \
                        r["HCAHPS Linear Mean Value"] not in MISSING:
                    lin[m][ccn] = int(r["HCAHPS Linear Mean Value"])
                elif mid == f"{m}_STAR_RATING" and \
                        r["Patient Survey Star Rating"] not in MISSING:
                    star[m][ccn] = int(r["Patient Survey Star Rating"])
        ccns = set.intersection(*(set(star[m]) for m in ALL8))

        def star_of(v, ivs):
            for s, (a, b) in enumerate(ivs, 1):
                if a <= v <= b:
                    return s
            raise ValueError((v, ivs))
        n_pubmatch = n_pubmismatch = 0
        mig = Counter()
        changed = 0
        for c in sorted(ccns):
            pub_stars = {m: star[m][c] for m in ALL8}
            s_pub = summary_star(pub_stars)
            if c in pub_summary:
                if s_pub == pub_summary[c]:
                    n_pubmatch += 1
                else:
                    n_pubmismatch += 1
            opt_stars = {m: star_of(lin[m][c], opt_iv[m]) for m in ALL8}
            s_opt = summary_star(opt_stars)
            if s_opt != s_pub:
                changed += 1
                mig[(s_pub, s_opt)] += 1
        vout[tag] = {
            "n_hospitals_all8": len(ccns),
            "published_side_recompute_matches_published_summary":
                f"{n_pubmatch}/{n_pubmatch + n_pubmismatch}",
            "summary_star_changes": changed,
            "migration": {f"{a}->{b}": n for (a, b), n in sorted(mig.items())},
        }
        print(f"{tag}: n={len(ccns)} recompute-check {vout[tag]['published_side_recompute_matches_published_summary']} "
              f"summary changes {changed} mig {dict(mig)}")
    bank = {"register": ("HCAHPS Summary Star recomputed per hospital under "
                         "the published and the exact-optimal per-measure "
                         "stars (6-component simple average, normal rounding, "
                         "Technical Notes pp. 4-5); the published-side "
                         "recompute is checked against the published "
                         "H_STAR_RATING column per refresh. Uses the "
                         "five-refresh census in "
                         "HCAHPS_CENSUS_ALLVINTAGES.json."),
            "census_bank_sha256_16": hashlib.sha256(
                open(BANK, "rb").read()).hexdigest()[:16],
            "vintages": vout}
    with open(OUT, "w") as f:
        json.dump(bank, f, indent=1)
        f.write("\n")
    print("output sha256-16:",
          hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16])


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(p) for _, p in VINTAGES):
    print("summary_compound: missing input file(s) among data/HCAHPS-Hospital.csv and data/archive/<refresh>/HCAHPS-Hospital.csv "
          "(run from stars_census/legb_hcahps/; the files are Provider Data Catalog downloads, "
          "not included; re-fetch each by the URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
