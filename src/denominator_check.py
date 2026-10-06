# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""The printed denominators.
- n_published_headline per star year (the 1,724 ratings CMS issued): every
  contract whose headline rating is published numeric (overall for MA-PD,
  Part C summary for MA-only, Part D summary for PDP).
- n_rated per star year (2,180): contracts the replay's aggregation rates
  (headline rating computable, raw hold-harmless threshold).
- the printed ratio.
Output: runs/denominator_check.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.2, Table 3 (the
    1,724 published contract-year ratings of star years 2024-2026).
Run:  cd src && python3 denominator_check.py
Requires: Python >= 3.10; numpy, scipy.
"""
import csv, json, os
import enclosure_census_docsonly as C
import sys
from aggregate import AggSpec, rate_contract, load_inputs, org_class_of, disaster_pct_of, dup_d_ids

BASE = C.BASE
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

out_years = {}
for year in C.YEARS:
    v = C.BASELINE_VINTAGE[year]
    with open(os.path.join(BASE, f"data/parsed/summary_{year}_{v}.csv"),
              newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    stars, summary, cai = load_inputs(year, v)
    ddup = frozenset(dup_d_ids(year, v))
    spec = AggSpec(year)
    n_pub = n_rated = 0
    for r in rows:
        cid = r["contract_id"]
        oc = org_class_of(r)
        headline_raw = {"MA-PD": r["overall_raw"],
                        "MA-only": r["part_c_summary_raw"],
                        "PDP": r["part_d_summary_raw"]}[oc].strip()
        if headline_raw.replace(".", "").isdigit():
            n_pub += 1
        st = stars.get(cid)
        if st:
            hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
            res = rate_contract(spec, st, oc, cai.get(cid, {}),
                                gate_convention="raw",
                                disaster=disaster_pct_of(r, year) >= 25,
                                dup_d=ddup)
            if hh in res:
                n_rated += 1
    out_years[year] = dict(n_published_headline=n_pub, n_rated=n_rated)

tot_pub = sum(y["n_published_headline"] for y in out_years.values())
tot_rated = sum(y["n_rated"] for y in out_years.values())
out = {
 "register": ("Printed denominators: the 1,724 counts every published numeric "
           "headline rating (overall for MA-PD, Part C summary for MA-only, "
           "Part D summary for PDP; cost contracts included, since CMS "
           "issues them ratings), at the baseline vintages; n_rated = the "
           "replay aggregation produces the headline (raw hold-harmless "
           "threshold), all classes. The numerator is the set of "
           "contract-years in the dollar estimate while the denominator is "
           "all issued ratings."),
 "generator": "src/denominator_check.py",
 "per_year": out_years,
 "totals": dict(n_published_headline=tot_pub, n_rated=tot_rated),
 # the numerator in the paper is the set in the dollar estimate (34
 # contract-years, dual-baseline condition, endpoint-conditional pair
 # excluded); the July-22-2026 file leaves the denominators unchanged (516
 # numeric 2026 overalls on both vintages; one contract each way, net zero).
 "ratio_of_record": dict(numerator=34, denominator=tot_pub,
                         pct=round(100.0*34/tot_pub, 1),
                         beside_all_rated=dict(numerator=34, denominator=2180,
                                               pct=round(100.0*34/2180, 1))),
 "printed_targets": dict(published="597/566/561=1724",
                         rated="781/715/684=2180", ratio="2.8%"),
}
p = os.path.join(BASE, "runs/denominator_check.json")
json.dump(out, open(p, "w"), indent=1); open(p, "a").write("\n")
print(json.dumps({**out["per_year"], "totals": out["totals"],
                  "ratio": out["ratio_of_record"]}, indent=1))
