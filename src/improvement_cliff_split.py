# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Cliff-adjacency split by scenario class. Splits the cliff-adjacency union by scenario class from the stored
improvement_sensitivity.json crossed_contracts rows (each row carries its
scenario; the sensitivity driver is not rerun).
Output: runs/improvement_cliff_split.json
Run: python3 improvement_cliff_split.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Appendix C.6
    (improvement stars on the recomputed side).
Requires: Python >= 3.10, standard library only.
"""
import csv
import hashlib
import json
import os
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE = ("C-1", "C+1", "D-1", "D+1")


def main():
    src_path = os.path.join(BASE, "runs/improvement_sensitivity.json")
    d = json.load(open(src_path))
    rows = d["q2_cliff_adjacency"]["crossed_contracts"]
    mapd = [r for r in rows if r["org_class"] == "MA-PD"]
    allc = {(r["year"], r["contract_id"]) for r in mapd}
    single = {(r["year"], r["contract_id"]) for r in mapd
              if r["scenario"] in SINGLE}
    summ = {}
    for y, v in (("2024", "recalc"), ("2025", "current"), ("2026", "current")):
        with open(os.path.join(BASE, f"data/parsed/summary_{y}_{v}.csv"),
                  newline="", encoding="utf-8-sig") as f:
            summ[y] = {r["contract_id"]: r["overall_raw"].strip()
                       for r in csv.DictReader(f)}
    pub = {(y, c) for (y, c) in single
           if summ[y].get(c, "").replace(".", "").isdigit()}
    out = {
        "register": ("The cliff-adjacency union split by scenario class. The "
                     "printed 104 (26/37/41) is the union including the "
                     "both-measures-together scenarios (CD+-1); the count for "
                     "a single-star move of one improvement measure is the "
                     "single-measure union."),
        "generator": "src/improvement_cliff_split.py",
        "union_all_scenarios_mapd": len(allc),
        "union_single_measure_mapd": len(single),
        "single_by_year": dict(Counter(y for y, _ in single)),
        "single_with_published_overall": len(pub),
        "single_pub_by_year": dict(Counter(y for y, _ in pub)),
        "scenario_row_counts": dict(Counter(r["scenario"] for r in rows)),
        "note": ("baseline = the published current/recalc ratings as in the "
                 "stored sensitivity record; the effect of the later CMS "
                 "overlay on this program-scale count is not included"),
        "input_pin_sha256_16": hashlib.sha256(
            open(src_path, "rb").read()).hexdigest()[:16],
    }
    p = os.path.join(BASE, "runs/improvement_cliff_split.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print(json.dumps({k: out[k] for k in
                      ("union_all_scenarios_mapd", "union_single_measure_mapd",
                       "single_by_year", "single_with_published_overall")}))
    print("->", p)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
