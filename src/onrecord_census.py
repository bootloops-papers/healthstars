# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Bonus-status census at record-reproducing input orders.

The rule: a split is REPRODUCED when at least one sampled realization
(4 order conventions + 24 uniform-random permutations, all under the
assumed front-swap mapping) reproduces all four of CMS's published final
boundaries bit-for-bit (pub_match_ward == 4). For a reproduced split the
published side is CMS's printed number, reproduced at an exhibited ordering.
A star change counts only if it is identical across every reproducing
realization of its split; contracts with divergent changes across
reproducing realizations are dropped from the headline and counted as
reproduced-but-ambiguous. Splits with no reproducing realization contribute
no changes (a conservative, one-sided undercount, the same logic as
improvement/CAHPS held at published). Splits that errored in any of the 28
realizations are excluded everywhere (the stored census error policy).

The headline object: contract-years whose bonus-line crossing is derivable
entirely from reproduced splits at record-reproducing orderings.

Inputs : runs/enclosure_collapse_frontswap.json (4 conventions)
         runs/enclosure_collapse_frontswap_rperm.json (24 rperms)
Output : runs/onrecord_census.json
Run    : python3 onrecord_census.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (an input order
    that reproduces all four published cut points exists for 50 of 114
    computations).
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""
import csv
import json
import os
from collections import defaultdict

import enclosure_census_docsonly as C
import sys

BASE = C.BASE
OUT = os.path.join(BASE, "runs", "onrecord_census.json")
SRC = [("conventions", "runs/enclosure_collapse_frontswap.json"),
       ("rperm", "runs/enclosure_collapse_frontswap_rperm.json")]

ENROLL = {  # star year -> (payment year, stored monthly file, data month)
    "2024": ("CY2025", "data/raw/enrollment/Monthly_Report_By_Contract_2025_12/"
                       "Monthly_Report_By_Contract_2025_12.csv", "2025-12"),
    "2025": ("CY2026", "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
                       "Monthly_Report_By_Contract_2026_07.csv", "2026-07"),
    "2026": ("CY2027", "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
                       "Monthly_Report_By_Contract_2026_07.csv", "2026-07"),
}
VALUE = {"lo": 400, "mid": 500, "hi": 600}   # per-enrollee annual QBP value band


def load_rows():
    rows = []
    for tag, rel in SRC:
        rows += json.load(open(os.path.join(BASE, rel)))["measure_results"]
    return rows


def main():
    rows = load_rows()
    err_splits = {(r["year"], r["mid"], r["org"]) for r in rows if "error" in r}
    ok = [r for r in rows if "error" not in r
          and (r["year"], r["mid"], r["org"]) not in err_splits]
    n_real = len({r["realization"] for r in ok})

    # reproduced splits: realizations reproducing all 4 published boundaries
    repro = defaultdict(list)
    for r in ok:
        if r["pub_match_ward"] == 4:
            repro[(r["year"], r["mid"], r["org"])].append(r["realization"])

    # stable deltas inside the reproduced set
    synth, ambiguous_dropped = [], []
    for key, rns in sorted(repro.items()):
        year, mid, org = key
        reps = [r for r in ok if (r["year"], r["mid"], r["org"]) == key
                and r["realization"] in rns]
        per_contract = defaultdict(dict)      # cid -> {realization: (sw, sd)}
        for r in reps:
            for cid, sw, sd in r["deltas"]:
                per_contract[cid][r["realization"]] = (int(sw), int(sd))
        stable = []
        for cid, seen in sorted(per_contract.items()):
            vals = set(seen.values())
            if len(vals) == 1 and len(seen) == len(reps):
                sw, sd = vals.pop()
                stable.append((cid, sw, sd))
            else:
                ambiguous_dropped.append(
                    dict(year=year, mid=mid, org=org, contract_id=cid,
                         n_reproducing=len(reps),
                         n_present=len(seen),
                         variants=sorted({f"{a}->{b}" for a, b in seen.values()})))
        synth.append(dict(year=year, mid=mid, org=org,
                          name=f"{mid}/{org}", realization="onrecord",
                          deltas=stable, n_star_changes=len(stable),
                          reproducing_realizations=sorted(rns)))

    agg = C.enclosure_aggregate(synth, ["onrecord"])
    flips = [c for c in agg["contracts"] if c["n_flip_realizations"] == 1]

    # dollar estimate (stored estimate conventions; MA classes only; PDP has no QBP)
    enroll_cache = {}
    def enrollment_of(cid, star_year):
        _, path, _ = ENROLL[star_year]
        if path not in enroll_cache:
            m = {}
            with open(os.path.join(BASE, path), newline="",
                      encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    v = row["Enrollment"].replace(",", "")
                    m[row["Contract Number"]] = int(v) if v.isdigit() else 0
            enroll_cache[path] = m
        return enroll_cache[path].get(cid)

    per_contract, agg_years = [], defaultdict(lambda: defaultdict(int))
    for c in sorted(flips, key=lambda x: (x["year"], x["contract_id"])):
        py, _, month = ENROLL[c["year"]]
        qbp = c["qbp_relevant"]
        e = enrollment_of(c["contract_id"], c["year"]) if qbp else None
        note = (None if not qbp and e is None else
                "PDP: no QBP rides on Part D stand-alone ratings; counted, not in the dollar estimate"
                if not qbp else
                "not present in payment-year enrollment month; $0 in the dollar estimate, enrollees in no total"
                if e is None else None)
        e_val = e or 0
        band = {k: e_val * v * (1 if c["direction"] == "UP" else -1)
                for k, v in VALUE.items()} if qbp else None
        per_contract.append(dict(
            year=c["year"], payment_year=py, contract_id=c["contract_id"],
            org_class=c["org_class"], direction=c["direction"],
            baseline=c["baseline"],
            onrecord_rating=c["ratings_by_realization"]["onrecord"],
            qbp_relevant=qbp, enrollment=(e_val if qbp else None),
            enrollment_data_month=(month if qbp else None),
            est_annual_qbp_revenue_delta_usd=band, note=note))
        if qbp:
            agg_years[py]["enrollment"] += e_val
            for k, v in VALUE.items():
                agg_years[py][f"gross_{k}"] += e_val * v

    qbp_rows = [p for p in per_contract if p["qbp_relevant"]]
    totals = {k: sum(a[f"gross_{k}"] for a in agg_years.values())
              for k in VALUE}
    result = {
        "register": (
            "Published-boundary flip census: the published side of every counted "
            "computation is CMS's published boundary values, reproduced "
            "bit-for-bit by the documented method at an exhibited input ordering "
            "(record-reproducing realizations named per split); the exact side "
            "is the stated criterion's result at the same reproducing "
            "ordering(s); a star change counts only if identical at every "
            "reproducing ordering of its split. Splits with no reproducing "
            "ordering contribute nothing (one-sided undercount). Dollar "
            "estimate: enrollment x $400-600/enrollee-year, single payment-year "
            "month per star year, per-year figures are annual, the three-year "
            "total is a sum not a rate; outward rounding on displayed ends; PDP "
            "contracts counted, never in the dollar estimate (no QBP). The "
            "reproducing set can only grow with more sampled orderings (4/4 "
            "reproduction is monotone evidence); changes could be re-classed "
            "ambiguous by a new reproducing ordering."),
        "inputs": {tag: rel for tag, rel in SRC},
        "n_realizations_sampled": n_real,
        "n_splits_total_ok": len({(r["year"], r["mid"], r["org"]) for r in ok}),
        "n_splits_reproduced": len(repro),
        "errored_splits_excluded": sorted(err_splits),
        "reproduced_splits": [dict(year=s["year"], mid=s["mid"], org=s["org"],
                                   n_stable_deltas=s["n_star_changes"],
                                   reproducing_realizations=s["reproducing_realizations"])
                              for s in synth],
        "reproduced_but_ambiguous_deltas": ambiguous_dropped,
        "flips_onrecord": per_contract,
        "summary": {
            "n_flips_onrecord": len(per_contract),
            "n_qbp_relevant": len(qbp_rows),
            "n_up": sum(1 for p in qbp_rows if p["direction"] == "UP"),
            "n_down": sum(1 for p in qbp_rows if p["direction"] == "DOWN"),
            "total_enrollment": sum(p["enrollment"] for p in qbp_rows),
            "gross_total_3yr_usd": totals,
            "per_payment_year": {py: dict(a) for py, a in sorted(agg_years.items())},
            "n_ambiguous_dropped": len(ambiguous_dropped),
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps(result["summary"], indent=1))
    print("flips:", [(p["year"], p["contract_id"], p["direction"],
                      p["qbp_relevant"]) for p in per_contract])
    print("->", OUT)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(os.path.join(BASE, v[1])) for v in ENROLL.values()):
    print("onrecord_census: missing CMS monthly enrollment file(s) under data/raw/enrollment/ (not included in the package; "
          "re-fetch each by the URL and sha256 in CMS_RAW_PINS.json; see "
          "data/raw/enrollment/SOURCES_enrollment.txt)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
