# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Dollar estimate for the bonus-status changes under the exact optimum.

Writes runs/criterion_dollar_estimate.json from the bonus-status census
(runs/criterion_census.json, independently recounted in
runs/criterion_recount_stored.json) in the schema of
runs/dollar_estimate_collapse23.json, so every aggregate is at a
named JSON path: per-payment-year and total up/down/gross/net,
largest-single-contract band, prior-basis comparison. Estimate basis
(per-enrollee $400-600 band, public reference figures) carried from the
prior-basis record.
Run: python3 criterion_dollar_estimate_build.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Table 2 block
    b (34 flips in the dollar estimate; gross band $916-1,375 million).
Requires: Python >= 3.10, standard library only.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENSUS = os.path.join(BASE, "runs", "criterion_census.json")
PRIOR = os.path.join(BASE, "runs", "dollar_estimate_collapse23.json")
OUT = os.path.join(BASE, "runs", "criterion_dollar_estimate.json")

VALUE = {"lo": 400, "mid": 500, "hi": 600}


def band(enr, sign=1):
    return {k: float(sign * enr * v) for k, v in VALUE.items()}


def agg_rows(rows):
    up = [p for p in rows if p["direction"] == "UP"]
    dn = [p for p in rows if p["direction"] == "DOWN"]
    ue = sum(p["enrollment"] for p in up)
    de = sum(p["enrollment"] for p in dn)
    out = {
        "n_contract_years": len(rows),
        "total_enrollment": ue + de,
        "up_direction": {"n": len(up), "enrollment": ue,
                         "gross_annual_usd": band(ue)},
        "down_direction": {"n": len(dn), "enrollment": de,
                           "gross_annual_usd": band(de)},
        "gross_total_usd": {
            "note": ("gross magnitude in both directions (payments to the "
                     "wrong insurers in both directions), never a net-cost "
                     "claim"),
            **{k: float((ue + de) * v) for k, v in VALUE.items()}},
        "net_usd_common_value": {
            "note": "net = (up_enrollment - down_enrollment) x v, v swept over the band; not printed in the paper",
            "mid": float((ue - de) * 500),
            "lo": float(min((ue - de) * 400, (ue - de) * 600)),
            "hi": float(max((ue - de) * 400, (ue - de) * 600))},
    }
    return out


DEMOTED = {("2024", "H1304"), ("2025", "H5273")}   # endpoint-conditional
# (excluded: their bonus status changes only if the prior-year boundary sits
#  exactly at the closed endpoint of its published display interval; excluded
#  from every printed figure, $8.9M mid together)


def main():
    census = json.load(open(CENSUS))
    prior = json.load(open(PRIOR))
    allq = [p for p in census["flips_criterion"] if p.get("qbp_priceable")]
    qbp = [p for p in allq if (p["year"], p["contract_id"]) not in DEMOTED]
    demoted = [p for p in allq if (p["year"], p["contract_id"]) in DEMOTED]
    pdp = [p for p in census["flips_criterion"] if not p["qbp_relevant"]]
    cost = [p for p in census["flips_criterion"] if p["qbp_relevant"]
            and "Cost" in p.get("org_type", "")]
    noover = [p for p in census["flips_criterion"] if p["qbp_relevant"]
              and not p.get("published_overall_exists")
              and "Cost" not in p.get("org_type", "")]
    def _doubled(rows):
        from collections import Counter as _C
        cnt = _C(r["contract_id"] for r in rows)
        return sorted(c for c, n in cnt.items() if n > 1)

    per_year = {}
    for py in ("CY2025", "CY2026", "CY2027"):
        per_year[py] = agg_rows([p for p in qbp if p["payment_year"] == py])

    total = agg_rows(qbp)
    largest = max(qbp, key=lambda p: p["enrollment"])

    result = {
        "generator": "src/criterion_dollar_estimate_build.py",
        "register": (
            "Dollar estimate attached to the bonus-status census: published "
            "stars vs the stars under the exact optimum on the full trimmed "
            "score list (runs/criterion_census.json, independently recounted "
            "in runs/criterion_recount_stored.json). Everything "
            "dollar-denominated is an estimate: single-month enrollment "
            "snapshots joined by contract ID; $500/enrollee/yr midpoint with "
            "a $400-$600 band from public actuarial/policy commentary. The "
            "estimate is gross and two-directional (payments to the wrong "
            "insurers), never a net-cost-to-taxpayer claim; net fields are "
            "stored here and not printed."),
        "inputs": {
            "flip_set": ("runs/criterion_census.json flips_criterion "
                         "(qbp_priceable rows), verified by "
                         "runs/criterion_recount_stored.json"),
            "prior_basis": ("runs/dollar_estimate_collapse23.json (the 23 "
                            "flips exact at the collapse envelope; prior "
                            "basis)"),
            "enrollment_files": prior["inputs"]["enrollment_files"],
            "enrollment_provenance": prior["inputs"]["enrollment_provenance"],
        },
        "estimate_basis": prior["estimate_basis"],
        "reference_figure_notes": prior["reference_figure_notes"],
        "framing": {
            "unit": ("counts are the machine fields n_flips_of_record_priced, "
                     "priced_distinct_contracts, priced_multi_year_contracts "
                     "and n_rows_counted_never_priced; enrollment totals are "
                     "enrollee-payment-years; H8547 flips down in CY2025 "
                     "(1,792) and up in CY2026 (1,553)"),
            "time_scope": ("gross/net totals sum three payment years "
                           "(CY2025+CY2026+CY2027); they are not an annual "
                           "rate; totals are printed as summed over payment "
                           "years 2025 through 2027 with the per-year split"),
            "direction_convention": ("UP = the criterion's answer gains bonus "
                                     "status vs published (denied money); "
                                     "DOWN = loses it (payments above the "
                                     "criterion's answer); both are "
                                     "misallocation, neither is printed as a "
                                     "taxpayer cost"),
            "temporal_blend": ("CY2025 is a completed payment year; CY2027 is "
                               "prospective (estimated on the 2026-07 proxy "
                               "month); the 6/17/2026 upward-only 2027-QBP "
                               "recalculation is not decidable at the contract "
                               "level from the public record (stored check, "
                               "data/raw/sy2026_recalc2027qbp/)"),
            "full_set_split": ("18 UP / 62 DOWN QBP-relevant (printed grain; 2 "
                               "endpoint-conditional UP candidates demoted, "
                               "listed); 8 PDP counted"),
            "conditionality": ("improvement/CAHPS held at published stars; "
                               "guardrail vs published prior-year boundaries "
                               "(this-year-corrected, history as published); "
                               "documented-rules score list; the 2026 D07 "
                               "MA-PD computation (fewer than five distinct "
                               "scores: all trimmed scores identical) held at "
                               "published"),
        },
        "aggregates_per_payment_year": per_year,
        "aggregates": {
            "priced_of_record": total,
            "largest_single_contract_band": {
                "contract_id": largest["contract_id"],
                "star_year": largest["year"],
                "direction": largest["direction"],
                "enrollment": largest["enrollment"],
                "share_of_total_enrollment": round(
                    largest["enrollment"] / total["total_enrollment"], 4),
                "est_qbp_revenue_delta_usd": band(
                    largest["enrollment"],
                    1 if largest["direction"] == "UP" else -1),
                "thin_margin_note": ("published rating sits ~0.030 above the "
                                     "3.75 rounding edge (recount chain); "
                                     "printed with the largest-row "
                                     "disclosure"),
            },
        },
        "n_flips_of_record_priced": len(qbp),
        "n_rows_counted_never_priced": dict(
            pdp=len(pdp), cost_contracts=len(cost),
            no_published_overall=len(noover)),
        "counted_never_priced_cost": [
            dict(year=p["year"], contract_id=p["contract_id"],
                 direction=p["direction"], enrollment=p["enrollment"],
                 reason=p["unpriceable_reason"]) for p in cost],
        "counted_never_priced_no_published_overall": [
            dict(year=p["year"], contract_id=p["contract_id"],
                 direction=p["direction"], enrollment=p["enrollment"],
                 published=p["published_overall"],
                 **({"note": ("enrollment cell also CMS-SUPPRESSED ('*' in "
                              "the raw monthly CSV; contract present); the "
                              "row is out of the dollar estimate for the "
                              "no-overall reason")}
                    if p["contract_id"] == "H8095" else {}))
            for p in noover],
        "ex_CY2027_subtotal": {
            "note": ("CY2025 and CY2026 only (the completed/current payment "
                     "years). The CY2027 part is estimated against the "
                     "July-22-2026 updated Summary Ratings (posted by CMS on "
                     "August 17, 2026); the four 2026 rows whose bonus "
                     "status changes against only one baseline are carried "
                     "as june17-overlay-undecidable and never enter the "
                     "estimate."),
            "n_contract_years": (per_year["CY2025"]["n_contract_years"]
                                 + per_year["CY2026"]["n_contract_years"]),
            "total_enrollment": (per_year["CY2025"]["total_enrollment"]
                                 + per_year["CY2026"]["total_enrollment"]),
            **{k: (per_year["CY2025"]["gross_total_usd"][k]
                   + per_year["CY2026"]["gross_total_usd"][k])
               for k in ("lo", "mid", "hi")}},
        "endpoint_convention_note": (
            "lo/hi in per-row and gross blocks are the $400/$600 sweep ends "
            "(DOWN rows therefore carry lo numerically greater than hi); "
            "lo/hi in net_usd_common_value blocks are numerically ordered. "
            "Both labeled; net is not printed."),
        "priced_distinct_contracts": len({p["contract_id"] for p in qbp}),
        "priced_multi_year_contracts": _doubled(qbp),
        "demoted_endpoint_conditional": [
            dict(year=p["year"], contract_id=p["contract_id"],
                 direction=p["direction"], enrollment=p["enrollment"],
                 mid_usd=abs(p["est_annual_qbp_revenue_delta_usd"]["mid"]))
            for p in demoted],
        "zero_priced_rows": [
            dict(year=p["year"], contract_id=p["contract_id"],
                 direction=p["direction"],
                 reason=("CMS-SUPPRESSED enrollment ('*' in the raw monthly "
                         "CSV; contract present)" if p["contract_id"] == "H8095"
                         else "absent from the payment-year enrollment month"))
            for p in qbp if p["enrollment"] == 0],
        "pdp_counted_never_priced": [
            dict(year=p["year"], contract_id=p["contract_id"],
                 direction=p["direction"]) for p in pdp],
        "prior_basis_bands_for_comparison": {
            "certified_23_gross_total": prior["aggregates"]["certified_23"][
                "gross_total_annual_usd"],
            "certified_23_enrollment": prior["aggregates"]["certified_23"][
                "total_enrollment"],
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    t = result["aggregates"]["priced_of_record"]
    print(json.dumps({
        "n": t["n_contract_years"], "enr": t["total_enrollment"],
        "gross": t["gross_total_usd"],
        "up": t["up_direction"]["enrollment"],
        "down": t["down_direction"]["enrollment"]}, indent=1))
    print("->", OUT)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
