# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""HCAHPS cut points in the following quarter (the rider test).

Inputs: HCAHPS_CENSUS_ALLVINTAGES.json (data-derived published bins and exact
optimal bins per refresh) and APPENDIX_C_CUTS.json (the five editions' printed
cut points).

V) VALIDATION: for the four (data refresh, matching edition) pairs, the
   edition's printed star2..star5 minima must equal the data-derived published
   lower edges, cross-checking the PDF extraction and the data parse.
R) RIDER: for each measure and consecutive edition pair (t -> t+1) where
   published(t) != optimal(t) on boundary b, record whether the printed cut at
   t+1 moved toward the refresh-t optimal value, away, or stayed. (t+1 is
   computed on newer data: movement is covariation with the optimum, not a
   causal claim.)
Output: RIDER_TEST.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.5 (HCAHPS cut
    points in the following quarter: 17, 10, 46 and 15 under the July 2026
    pairing).
Run:  cd stars_census/legb_hcahps && python3 rider_test.py
Requires: Python >= 3.10, standard library only.
"""
import hashlib
import json
import sys

MEAS_KEY = {"H_COMP_1": "nurses", "H_COMP_2": "doctors",
            "H_COMP_5": "medicines", "H_COMP_6": "discharge",
            "H_CLEAN": "clean", "H_QUIET": "quiet",
            "H_HSP_RATING": "rating", "H_RECMND": "recommend"}
# data vintage tag -> edition index in APPENDIX_C_CUTS (period-matched)
PAIRING = {"2025-04-30": 0, "2025-08-14": 1, "2025-11-26": 2, "2026-02-25": 3}
NEXT_EDITION = {"2025-04-30": 1, "2025-08-14": 2, "2025-11-26": 3,
                "2026-02-25": 4, "current-2026-05-13": 4}
# the current refresh has no matching edition in APPENDIX_C_CUTS.json (the
# April 2026 notes were not acquired); its next edition is July 2026 (index 4).


def mins_from_intervals(ivs):
    """[star2_min..star5_min] = lower edges of bins 2..5."""
    return [ivs[i][0] for i in range(1, 5)]


def main():
    census = json.load(open("HCAHPS_CENSUS_ALLVINTAGES.json"))["vintages"]
    eds = json.load(open("APPENDIX_C_CUTS.json"))["editions"]

    validation = {}
    v_ok = v_bad = 0
    for tag, ei in PAIRING.items():
        cuts = eds[ei]["cutpoints"]
        rows = {}
        for m, key in MEAS_KEY.items():
            cs = census[tag]["measures"][m]
            if "error" in cs or key not in cuts:
                rows[m] = "n/a"
                continue
            data_mins = mins_from_intervals(cs["published_intervals"])
            pdf_mins = [cuts[key][f"star{s}_min"] for s in (2, 3, 4, 5)]
            ok = data_mins == pdf_mins
            rows[m] = {"data_derived": data_mins, "printed": pdf_mins,
                       "match": ok}
            v_ok += ok
            v_bad += not ok
        validation[tag] = rows

    rider = []
    for tag, nei in NEXT_EDITION.items():
        next_cuts = eds[nei]["cutpoints"]
        for m, key in MEAS_KEY.items():
            cs = census[tag]["measures"][m]
            if "error" in cs or key not in next_cuts:
                continue
            pub = mins_from_intervals(cs["published_intervals"])
            opt = mins_from_intervals(cs["optimal_intervals"])
            nxt = [next_cuts[key][f"star{s}_min"] for s in (2, 3, 4, 5)]
            for si, (p, o, x) in enumerate(zip(pub, opt, nxt), start=2):
                if p == o:
                    continue          # boundary already optimal at t
                if x == o:
                    verdict = "MOVED-TO-OPTIMUM-VALUE"
                elif (o > p and x > p) or (o < p and x < p):
                    verdict = "moved-toward"
                elif x == p:
                    verdict = "stayed"
                else:
                    verdict = "moved-away"
                rider.append({"vintage": tag, "measure": m,
                              "boundary": f"star{si}_min",
                              "published_t": p, "optimal_t": o,
                              "next_edition_cut": x, "verdict": verdict})
    tally = {}
    for r in rider:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1

    out = {"register": ("Rider test: where a refresh's published boundary "
                        "differs from its exact optimum, does the next "
                        "edition's printed cut point move toward the optimal "
                        "value? Movement is computed on the next edition's "
                        "own newer data: covariation, not causation. The "
                        "validation section cross-checks the PDF extraction "
                        "against the data-derived bins on the four "
                        "period-matched pairs."),
           "validation": validation,
           "validation_tally": {"match": v_ok, "mismatch": v_bad},
           "rider_rows": rider,
           "rider_tally": tally}
    with open("RIDER_TEST.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("validation:", v_ok, "match /", v_bad, "mismatch")
    print("rider tally:", tally)
    for r in rider:
        if r["verdict"] == "MOVED-TO-OPTIMUM-VALUE":
            print("  EXACT:", r["vintage"], r["measure"], r["boundary"],
                  f"pub {r['published_t']} opt {r['optimal_t']} next {r['next_edition_cut']}")
    print("output sha256-16:", hashlib.sha256(
        open("RIDER_TEST.json", "rb").read()).hexdigest()[:16])


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
