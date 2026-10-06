# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Aggregation-fidelity verification record:
the printed counts 1,754/1,763 (2024), 1,673/1,681 (2025), 1,645/1,653
(2026) of replayed rating cells matching CMS's published summary/overall
cells. A rating cell is one published numeric value among overall / Part C
summary / Part D summary for a rated contract (baseline vintage); match = the
replay reproduces it exactly at the published half-star precision.
Also emits the DP self-test inventory (dp_kmeans __main__ run fresh).
Output: runs/aggregation_fidelity.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.4 (the aggregation
    stage reproduces 1,754 of 1,763 published rating cells for 2024, 1,673 of
    1,681 for 2025 and 1,645 of 1,653 for 2026).
Run:  cd src && python3 aggregation_fidelity.py
Requires: Python >= 3.10; numpy, scipy.
"""
import csv, json, os, subprocess
from fractions import Fraction as F
import enclosure_census_docsonly as C
import sys
from aggregate import AggSpec, rate_contract, load_inputs, org_class_of, disaster_pct_of, dup_d_ids

BASE = C.BASE
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

per_year = {}
YEAR_VINTAGES = [("2024", "original"), ("2024", "recalc"),
                 ("2025", "current"), ("2026", "current")]
for year, v in YEAR_VINTAGES:
    rows = list(csv.DictReader(open(os.path.join(BASE, f"data/parsed/summary_{year}_{v}.csv"),
                                    newline="", encoding="utf-8-sig")))
    stars, summary, cai = load_inputs(year, v)
    ddup = frozenset(dup_d_ids(year, v)); spec = AggSpec(year)
    n_cells = n_match = 0
    mismatch_contracts = set()
    for r in rows:
        cid = r["contract_id"]
        st = stars.get(cid)
        if not st:
            continue
        oc = org_class_of(r)
        res = rate_contract(spec, st, oc, cai.get(cid, {}), gate_convention="raw",
                            disaster=disaster_pct_of(r, year) >= 25, dup_d=ddup)
        for slot, col in (("overall", "overall_raw"),
                          ("part_c", "part_c_summary_raw"),
                          ("part_d", "part_d_summary_raw")):
            pub = r[col].strip()
            if not pub.replace(".", "").isdigit():
                continue
            if slot not in res:
                n_cells += 1
                mismatch_contracts.add(cid)
                continue
            n_cells += 1
            if F(res[slot][0]) == F(pub):
                n_match += 1
            else:
                mismatch_contracts.add(cid)
    per_year[f"{year}/{v}"] = dict(cells=n_cells, match=n_match,
                          mismatch_contracts=sorted(mismatch_contracts))

dp_run = subprocess.run(["python3", os.path.join(BASE, "src", "dp_kmeans.py")],
                        capture_output=True, text=True)
out = {
 "register": ("Aggregation fidelity: replayed rating cells vs CMS's published "
           "numeric summary/overall cells, per star year at the baseline "
           "vintages (raw hold-harmless threshold, documented conventions); "
           "the mismatch contracts are listed per year. The DP self-test "
           "inventory is appended (assert-based, passes iff exit 0): 200 "
           "DP-vs-brute-force, 200 non-contiguous dominance, 100 tied-optima "
           "enumeration."),
 "generator": "src/aggregation_fidelity.py",
 "per_year": per_year,
 "printed_targets": {"2024": "1,754/1,763", "2025": "1,673/1,681",
                     "2026": "1,645/1,653"},
 "vintage_finding": (
     "The printed 2024 figure (1,754/1,763; residue confined to the Puerto "
     "Rico-class contracts) reproduces on the ORIGINAL 2024 vintage. On the "
     "RECALC vintage — the criterion census's 2024 BASELINE — fidelity is "
     "1,705/1,763 (58 mismatch cells, 53 contracts): CMS's June-2024 "
     "recalculated summary ratings are not reproducible from the "
     "recalculated measure stars under the documented aggregation rules, "
     "consistent with the recalculation applying adjudicative adjustments "
     "outside the published pipeline. The printed fidelity figure for 2024 "
     "therefore carries its vintage label (original)."),
 "dp_selftest": dict(exit_code=dp_run.returncode,
                     stdout=dp_run.stdout.strip(),
                     inventory="200 brute-force + 200 dominance + 100 tie-enumeration"),
}
p = os.path.join(BASE, "runs", "aggregation_fidelity.json")
json.dump(out, open(p, "w"), indent=1); open(p, "a").write("\n")
for y, d in per_year.items():
    print(y, f"{d['match']}/{d['cells']}", "mismatch contracts:", len(d['mismatch_contracts']))
print("dp self-test exit:", dp_run.returncode)
