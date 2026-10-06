# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Six-row within-groups dollar record: the six contract-years in the dollar
estimate whose bonus status changes under every one of the 28 sampled
orderings of the exact-within-groups construction, at the standard dollar
conventions.
Inputs: runs/classed_ambiguity.json (stable rows + classes), the enrollment
files at the payment-year months, the criterion census for the direction
cross-check.
Output: runs/within_groups_core_dollar.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (the eleven
    contract-years that change status under every sampled order, six with no
    other caveat; conditional on the assumed readout).
Run:  cd src && python3 within_groups_core_dollar.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""
import csv, hashlib, json, os
import enclosure_census_docsonly as C
import sys

BASE = C.BASE
VALUE = {"lo": 400, "mid": 500, "hi": 600}
EFILE = {"2024": "data/raw/enrollment/Monthly_Report_By_Contract_2025_12/Monthly_Report_By_Contract_2025_12.csv",
         "2025": "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/Monthly_Report_By_Contract_2026_07.csv",
         "2026": "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/Monthly_Report_By_Contract_2026_07.csv"}

def sha16(rel):
    return hashlib.sha256(open(os.path.join(BASE, rel), "rb").read()).hexdigest()[:16]

if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(os.path.join(BASE, rel)) for rel in EFILE.values()):
    print("within_groups_core_dollar: missing CMS monthly enrollment file(s) under data/raw/enrollment/ (not included; "
          "re-fetch each by the URL and sha256 in CMS_RAW_PINS.json; see "
          "data/raw/enrollment/SOURCES_enrollment.txt)",
          file=sys.stderr)
    sys.exit(2)

ca = json.load(open(os.path.join(BASE, "runs/classed_ambiguity.json")))
stable = ca["stable_all_28"]["rows"]
emap = {}
for y, rel in EFILE.items():
    with open(os.path.join(BASE, rel), newline="", encoding="utf-8-sig") as fh:
        emap[y] = {}
        for row in csv.DictReader(fh):
            v = row["Enrollment"].replace(",", "")
            emap[y][row["Contract Number"]] = int(v) if v.isdigit() else 0

rows, counted = [], []
for r in stable:
    e = emap[r["year"]].get(r["contract_id"], 0)
    entry = dict(year=r["year"], contract_id=r["contract_id"],
                 direction=r["direction"], enrollment=e,
                 enrollment_data_month={"2024": "2025-12"}.get(r["year"], "2026-07"))
    if r["cls"] == "clean":
        entry["est_annual_qbp_revenue_delta_usd"] = {
            k: e * v * (1 if r["direction"] == "UP" else -1)
            for k, v in VALUE.items()}
        rows.append(entry)
    else:
        entry["reason"] = ("no published overall rating: counted, not in the "
                           "dollar estimate (as in the criterion census)")
        counted.append(entry)
enr = sum(r["enrollment"] for r in rows)
up = [r for r in rows if r["direction"] == "UP"]
dn = [r for r in rows if r["direction"] == "DOWN"]
out = {
 "register": ("Six-row within-groups dollar record: the contract-years in the "
              "dollar estimate whose exact-within-groups rating differs from "
              "the published rating under every one of the 28 sampled "
              "orderings (a 29th ordering could remove a row), at enrollment "
              "x $400-600 for single payment-year months as an estimate; "
              "five further all-orderings-stable rows have no published "
              "overall rating and are counted but not in the dollar estimate; "
              "the census's conditions apply, including the assumed "
              "group-assignment rule of the paper's Appendix C (see README, "
              "'Assumed readout')."),
 "generator": "src/within_groups_core_dollar.py",
 "input_pins_sha256_16": {p: sha16(p) for p in
     ("runs/classed_ambiguity.json", "runs/criterion_census.json",
      EFILE["2024"], EFILE["2025"])},
 "in_dollar_estimate_rows": rows,
 "counted_not_in_dollar_estimate": counted,
 "summary": dict(
     n_in_dollar_estimate=len(rows), n_counted=len(counted),
     n_up=len(up), n_down=len(dn),
     enrollment=enr,
     up_enrollment=sum(r["enrollment"] for r in up),
     down_enrollment=sum(r["enrollment"] for r in dn),
     gross_usd={k: enr * v for k, v in VALUE.items()}),
}
p = os.path.join(BASE, "runs/within_groups_core_dollar.json")
json.dump(out, open(p, "w"), indent=1); open(p, "a").write("\n")
s = out["summary"]
print(f"in dollar estimate: {s['n_in_dollar_estimate']} rows ({s['n_up']}u/{s['n_down']}d), enr {s['enrollment']:,}, "
      f"gross [{s['gross_usd']['lo']/1e6:.1f}, {s['gross_usd']['hi']/1e6:.1f}]M mid {s['gross_usd']['mid']/1e6:.1f}")
print("counted:", s["n_counted"])
