# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Bonus-status census under the exact optimum (order-free).

The comparison is between the published stars and the stars under the exact
optimum on the full trimmed score list; no resampling and no input-order
dependence enter the criterion side. Published side = the published stars and
ratings (no replay). Criterion side = the stars under the stated minimum
computed on the full trimmed score list (the unique exact DP optimum stored
per computation in runs/exact_gaps_criterionvintage.json), carried through
the documented guardrail (published prior-year boundaries), display rounding,
and the stored aggregation. The CFR/Tech-Notes criterion names a minimum; the
minimum on scores on a line is deterministic and order-free; CMS's 10-group
mean resampling is the agency's instrument for approximating it and belongs
to the published computation (whose order-dependence is the separate
replication-gap finding). Conditionality (stated with the number):
improvement and CAHPS measures held at published stars (not clustered / not
public); guardrail priors taken at published display precision (sensitivity
checked in runs/guardrail_base_check_receipt.json); score list per the
documented-rules reconstruction (identical on both sample bases).

Inputs : runs/exact_gaps_criterionvintage.json (dp_cutpoints)
         C.build_jobs() (pool, guardrail prior/cap, display digits)
         aggregate loaders at BASELINE_VINTAGE (published stars/ratings)
Output : runs/criterion_census.json
Run    : python3 criterion_census.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Table 2 block
    b (60 contract-years change bonus status, 34 of them in the dollar estimate).
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""
import csv
import json
import os
from collections import defaultdict
from fractions import Fraction as F

import enclosure_census_docsonly as C
import sys
from aggregate import load_inputs
from census_measures import (disaster_pcts, prior_published_star_map,
                             disaster_higher_of)
from falsify_groups import round_display
from guardrail import guardrail_boundary

BASE = C.BASE
OUT = os.path.join(BASE, "runs", "criterion_census.json")
# the census runs at one operative vintage per star year on both sides
# (2024 recalc / 2025 current / 2026 current): CMS's published 2025 cut
# points are identical original->current (CMS re-scored 144 D01 contracts
# 98->100 at the Dec-2024 update but did not re-cluster); mixed vintages
# (cluster original / baseline current) would manufacture score-correction
# deltas.
GAPS = os.path.join(BASE, "runs", "exact_gaps_criterionvintage.json")
CRIT_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}
# CMS replaced the 2026 data-tables zip on 2026-08-17 with
# the July-22-2026 updated Summary Ratings (50 contracts' overall changed,
# 48 up). The operative CY2027 bonus baseline is that file.
JUL22 = os.path.join(BASE, "data", "parsed", "summary_2026_jul22_overall.csv")

ENROLL = {
    "2024": ("CY2025", "data/raw/enrollment/Monthly_Report_By_Contract_2025_12/"
                       "Monthly_Report_By_Contract_2025_12.csv", "2025-12"),
    "2025": ("CY2026", "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
                       "Monthly_Report_By_Contract_2026_07.csv", "2026-07"),
    "2026": ("CY2027", "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
                       "Monthly_Report_By_Contract_2026_07.csv", "2026-07"),
}
VALUE = {"lo": 400, "mid": 500, "hi": 600}


def main():
    gaps = {(g["year"], g["mid"], g["org"]): g for g in json.load(open(GAPS))}
    jobs, skips = C.build_jobs(cluster_vintage=CRIT_VINTAGE)
    # disaster percentages + prior-year published stars (name+part matched)
    # per star year
    dis = {y: disaster_pcts(y, C.BASELINE_VINTAGE[y]) for y in C.YEARS}
    prior_maps = {y: prior_published_star_map(y) for y in C.YEARS}
    hold_harmless_applied = []

    # published measure stars at the aggregator's own baseline vintage
    baseline_stars = {}
    for year in C.YEARS:
        stars, _, _ = load_inputs(year, C.BASELINE_VINTAGE[year])
        baseline_stars[year] = stars

    synth, split_log, missing_gap, degenerate = [], [], [], []
    for job in jobs:
        key = (job["year"], job["mid"], job["org"])
        g = gaps.get(key)
        if g is None:
            missing_gap.append(key)
            continue
        assert g["dp_n_optima"] == 1, key
        if len(g["dp_cutpoints"]) != 4:
            # a split with fewer than five distinct scores (e.g. 2026 D07
            # MA-PD, all 336 trimmed scores identical, k=1, empty
            # dp_cutpoints) admits no 5-star partition; held at the published
            # stars, as in the resampled replay.
            degenerate.append(dict(year=key[0], mid=key[1], org=key[2],
                                   k=g["k"], n_trimmed=g["n_trimmed"],
                                   dp_cutpoints=g["dp_cutpoints"]))
            continue
        vals = [F(v) for v in g["dp_cutpoints"]]
        if job["gr_note"] == "applied":
            cap = F(job["cap"])
            fin = [guardrail_boundary(v, F(p), cap)[0]
                   for v, p in zip(vals, job["prior"])]
        else:
            fin = vals
        bd = [F(round_display(v, job["dd"])) for v in fin]
        st_pub = baseline_stars[job["year"]]
        deltas = []
        for cid, s in job["pool"]:
            sw = st_pub.get(cid, {}).get(job["mid"])
            if sw is None:
                continue                      # not rated at baseline vintage
            x = F(s)
            if job["higher"]:
                sd = 1 + sum(1 for t in bd if x >= t)
            else:
                sd = 5 - sum(1 for t in bd if x > t)
            # CMS's >=25% disaster measure-star hold-harmless applied on the
            # criterion side (higher of the criterion star and the prior-year
            # published star; measure classes and name+part matching in
            # census_measures).
            part = "C" if job["mid"].startswith("C") else "D"
            sd_hh = disaster_higher_of(job["name"], part, cid, sd,
                                       dis[job["year"]], prior_maps[job["year"]])
            if sd_hh != sd:
                hold_harmless_applied.append(
                    dict(year=job["year"], mid=job["mid"], cid=cid,
                         criterion_raw=int(sd), held_to=int(sd_hh)))
                sd = sd_hh
            if sd != sw:
                deltas.append((cid, int(sw), int(sd)))
        synth.append(dict(year=job["year"], mid=job["mid"], org=job["org"],
                          name=job["name"], realization="criterion",
                          deltas=deltas, n_star_changes=len(deltas),
                          dp_final_boundaries=[str(b) for b in bd],
                          guardrail=job["gr_note"]))
        split_log.append(dict(year=job["year"], mid=job["mid"], org=job["org"],
                              n_star_deltas_vs_published=len(deltas),
                              guardrail=job["gr_note"]))

    agg = C.enclosure_aggregate(synth, ["criterion"])
    flips = [c for c in agg["contracts"] if c["n_flip_realizations"] == 1]

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

    # H2461/H2462 are 1876 Cost contracts: no QBP per 83 FR
    # (data/raw/fr/FR-2018-04-16_2018-07179.pdf, "no Quality Bonus Payments
    # (QBP) will be associated with the ratings for 1876 cost contracts");
    # 29 further QBP-classed rows have no published overall rating ("Not
    # enough data available"/"Plan too new to be measured"; none has a
    # published Part C summary either): their census baseline is the
    # replayed rating, and no QBP determination is readable from the printed
    # record. Priceable = qbp_relevant and published overall exists and
    # org_type is not a Cost contract.
    import csv as _csv
    summ = {}
    for _y in C.YEARS:
        _v = C.BASELINE_VINTAGE[_y]
        with open(os.path.join(BASE, f"data/parsed/summary_{_y}_{_v}.csv"),
                  newline="", encoding="utf-8-sig") as _fh:
            summ[_y] = {r["contract_id"]: r for r in _csv.DictReader(_fh)}
    # July-22 operative overall for star year 2026 (CY2027 bonus baseline)
    jul22 = {}
    with open(JUL22, newline="", encoding="utf-8-sig") as _fh:
        for r in _csv.DictReader(_fh):
            jul22[r["contract_id"]] = r["overall_jul22_raw"].strip()
    per_contract, agg_years = [], defaultdict(lambda: defaultdict(int))
    for c in sorted(flips, key=lambda x: (x["year"], x["contract_id"])):
        py, _, month = ENROLL[c["year"]]
        qbp = c["qbp_relevant"]
        e = enrollment_of(c["contract_id"], c["year"]) if qbp else None
        e_val = e or 0
        note = ("PDP: no QBP rides on Part D stand-alone ratings; counted, "
                "not priced" if not qbp else
                "not present in payment-year enrollment month; priced $0, "
                "enrollees in no total" if e is None else None)
        srow = summ[c["year"]].get(c["contract_id"], {})
        org_type = srow.get("org_type", "?")
        pub_overall = srow.get("overall_raw", "?").strip()
        # for star year 2026 the operative bonus baseline is the
        # July-22-2026 updated determination when one exists
        if c["year"] == "2026":
            j = jul22.get(c["contract_id"], "")
            if j.replace(".", "").isdigit():
                pub_overall = j
        overall_exists = pub_overall.replace(".", "").isdigit()
        is_cost = "Cost" in org_type
        # Dual-baseline condition: a row enters the dollar estimate only if
        # its bonus status differs from both the published operative overall
        # and the re-rated baseline; single-baseline flips are carried beside
        # and never enter the estimate (the June-17 upward-only overlay and
        # the baseline-fidelity residue cannot masquerade as optimizer
        # effects).
        bonus_crit = F(c["ratings_by_realization"]["criterion"]) >= 4
        bonus_rerated = F(c["baseline"]) >= 4
        bonus_pub = overall_exists and F(pub_overall) >= 4
        dual_ok = overall_exists and (bonus_crit != bonus_pub) and \
            (bonus_crit != bonus_rerated)
        direction = ("UP" if bonus_crit and not bonus_pub else
                     "DOWN") if overall_exists and bonus_crit != bonus_pub \
            else c["direction"]
        band = {k: e_val * v * (1 if direction == "UP" else -1)
                for k, v in VALUE.items()} if qbp else None
        priceable = qbp and overall_exists and not is_cost and dual_ok
        unpriceable_reason = (None if priceable or not qbp else
            ("COST-CONTRACT: 1876 Cost, no QBP attaches (83 FR 16520, "
             "stored FR-2018-04-16 PDF)" if is_cost else
             (f"NO-PUBLISHED-OVERALL: CMS printed '{pub_overall}'; baseline "
              "is the replayed rating; no QBP determination readable from "
              "the printed record (no published Part C summary either)")
             if not overall_exists else
             ("JUNE17-OVERLAY-UNDECIDABLE: bonus status flips against only "
              "one of the two baselines (published operative overall vs "
              "re-rated baseline); the June-17 upward-only recalc overlay / "
              "baseline-fidelity residue cannot be separated from the "
              "optimizer effect on the public record" if c["year"] == "2026"
              else "BASELINE-FIDELITY-MISMATCH: bonus status flips against "
              "only one of the two baselines; not attributable to the "
              "optimizer on the public record")))
        per_contract.append(dict(
            year=c["year"], payment_year=py, contract_id=c["contract_id"],
            org_class=c["org_class"], org_type=org_type,
            direction=direction,
            direction_vs_rerated_baseline=c["direction"],
            bonus_published=bool(bonus_pub) if overall_exists else None,
            bonus_criterion=bool(bonus_crit),
            bonus_rerated_baseline=bool(bonus_rerated),
            jul22_overlaid=(c["year"] == "2026"
                            and c["contract_id"] in jul22),
            baseline=c["baseline"],
            published_overall=pub_overall,
            published_overall_exists=overall_exists,
            criterion_rating=c["ratings_by_realization"]["criterion"],
            qbp_relevant=qbp, qbp_priceable=priceable,
            unpriceable_reason=unpriceable_reason,
            enrollment=(e_val if qbp else None),
            enrollment_data_month=(month if qbp else None),
            est_annual_qbp_revenue_delta_usd=(band if priceable else None),
            note=note))
        if priceable:
            agg_years[py]["enrollment"] += e_val
            agg_years[py]["n_contracts"] += 1
            for k, v in VALUE.items():
                agg_years[py][f"gross_{k}"] += e_val * v

    qbp_rows = [p for p in per_contract if p.get("qbp_priceable")]
    cost_rows = [p for p in per_contract if p["qbp_relevant"]
                 and "Cost" in p["org_type"]]
    noover_rows = [p for p in per_contract if p["qbp_relevant"]
                   and not p["published_overall_exists"]
                   and "Cost" not in p["org_type"]]
    totals = {k: sum(a[f"gross_{k}"] for a in agg_years.values()) for k in VALUE}
    n_star_cells = sum(s["n_star_changes"] for s in synth)
    result = {
        "register": (
            "Bonus-status census under the exact optimum (order-free): "
            "published stars vs the stars under the stated minimum computed "
            "exactly on the full trimmed score list (unique DP optimum per "
            "computation, stored in exact_gaps_criterionvintage.json), "
            "guardrailed against published prior-year boundaries, display-"
            "rounded, aggregated by the stored machinery. No resampling on "
            "the criterion side and no input-ordering dependence anywhere in "
            "this census; the published computation's order-dependence is "
            "the separate replication-gap finding. Conditionality: "
            "improvement/CAHPS held at published stars; guardrail priors at "
            "published display precision; documented-rules score list. "
            "Dollar estimate: enrollment x $400-600/enrollee-year, single "
            "payment-year month per star year; the three-year total is a "
            "sum, not an annual rate; outward rounding on displayed ends; "
            "PDP counted, never in the dollar estimate."),
        "conventions": (
            "(1) One operative vintage per star year on both sides: 2024 "
            "recalc / 2025 current / 2026 current (CMS's 2025 cut points are "
            "identical across vintages: re-scored, not re-clustered); (2) "
            "guardrail priors keyed by measure name+part (the C28/D01 and "
            "C24/D03 cross-part twins do not collide) and MPF Price Accuracy "
            "guardrail-exempt in 2024; (3) CMS's >=25% disaster measure-star "
            "hold-harmless applied on the criterion side (higher of criterion "
            "star and prior-year published star; HOS-class keys on the "
            "earlier disaster year, call center never adjusted, CAHPS/"
            "improvement held at published throughout; the new-measure "
            "with/without overall-level provision is not modeled); (4) all "
            "123 published computations enter (no n<30 filter); (5) star "
            "year 2026's operative bonus baseline is the July-22-2026 "
            "updated Summary Ratings; (6) dual-baseline condition: a row "
            "enters the dollar estimate only if its bonus status differs "
            "from both the published operative overall and the re-rated "
            "baseline; single-baseline flips are carried beside with named "
            "reasons and never enter the estimate."),
        "inputs": {
            "dp_boundaries": "runs/exact_gaps_criterionvintage.json",
            "jobs": ("enclosure_census_docsonly.build_jobs(cluster_vintage="
                     "CRIT_VINTAGE) (documented-rules basis; membership layer "
                     "carried from the vintages it was determined on)"),
            "jul22_overall_overlay": "data/parsed/summary_2026_jul22_overall.csv",
            "baseline_vintages": dict(C.BASELINE_VINTAGE),
            "cluster_vintages": dict(CRIT_VINTAGE)},
        "disaster_hold_harmless_applied": dict(
            n_cells=len(hold_harmless_applied),
            rows=hold_harmless_applied),
        "n_splits": len(synth),
        "skipped_splits": skips,
        "missing_gap_rows": missing_gap,
        "degenerate_splits_held_at_published": degenerate,
        "n_star_cells_moved": n_star_cells,
        "per_split": split_log,
        "flips_criterion": per_contract,
        "summary": {
            "n_flips": len(per_contract),
            "n_qbp_priceable": len(qbp_rows),
            "n_cost_counted_never_priced": len(cost_rows),
            "n_no_published_overall_counted_never_priced": len(noover_rows),
            "n_up": sum(1 for p in qbp_rows if p["direction"] == "UP"),
            "n_down": sum(1 for p in qbp_rows if p["direction"] == "DOWN"),
            "total_enrollment": sum(p["enrollment"] for p in qbp_rows),
            "gross_total_3yr_usd": totals,
            "per_payment_year": {py: dict(a)
                                 for py, a in sorted(agg_years.items())},
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps(result["summary"], indent=1))
    print("n_star_cells_moved:", n_star_cells, "| splits:", len(synth),
          "| missing_gap:", len(missing_gap))
    print("->", OUT)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(os.path.join(BASE, v[1])) for v in ENROLL.values()):
    print("criterion_census: missing CMS monthly enrollment file(s) under data/raw/enrollment/ (not included; "
          "re-fetch each by the URL and sha256 in CMS_RAW_PINS.json; see "
          "data/raw/enrollment/SOURCES_enrollment.txt)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
