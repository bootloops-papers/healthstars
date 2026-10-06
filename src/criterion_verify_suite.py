# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Verification suite for the bonus-status census.

Independence scope: it shares the stored primitives (build_jobs data
loading, guardrail_boundary, round_display, rate_contract) with the census,
and reimplements everything above them: boundary assembly, delta
derivation, the counterfactual application loop (direct rate_contract, no
enclosure_aggregate), flip classification, the dollar estimate, and the
mechanism/shock layer.

Modes (each writes its own record):
  recount    -> runs/criterion_recount_stored.json
                independent 90-row flip recount vs runs/criterion_census.json
                (the full row list is emitted so row-list identity can be
                checked), dollar-estimate recount, and the disclosure
                counts block (distinct/doubled contracts, top-8 share).
  chains     -> runs/criterion_toprow_chains.json
  mechanism  -> runs/criterion_mechanism_v2.json (net-weighted-shock bands
                on the census basis, with a comparison against the
                earlier-basis cells, and the star-cell lineage raw ->
                guardrail -> display)

Run:  python3 criterion_verify_suite.py recount|chains|mechanism|all

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Appendix C.4
    (independent re-derivation of the flip rows, their chains and the
    mechanism).
Requires: Python >= 3.10; numpy, scipy.
Exit codes: 0 on success; 2 when the recount mode's enrollment inputs are missing
    (one line on stderr names them; the chains and mechanism modes do not need them).
"""
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction as F

import enclosure_census_docsonly as C
from aggregate import (AggSpec, rate_contract, load_inputs, org_class_of,
                       disaster_pct_of, dup_d_ids)
from falsify_groups import round_display
from census_measures import (disaster_pcts, prior_published_star_map,
                             disaster_higher_of)
from guardrail import guardrail_boundary

BASE = C.BASE
GAPS = os.path.join(BASE, "runs", "exact_gaps_criterionvintage.json")
# criterion vintages + hold-harmless
CRIT_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}
_DIS = {y: disaster_pcts(y, {"2024": "recalc", "2025": "current", "2026": "current"}[y]) for y in ("2024", "2025", "2026")}
_PRIORMAP = {y: prior_published_star_map(y) for y in ("2024", "2025", "2026")}
CENSUS = os.path.join(BASE, "runs", "criterion_census.json")
V1RC = os.path.join(BASE, "runs", "criterion_v1_recount.json")

ENROLL_FILE = {
    "2024": "data/raw/enrollment/Monthly_Report_By_Contract_2025_12/"
            "Monthly_Report_By_Contract_2025_12.csv",
    "2025": "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
            "Monthly_Report_By_Contract_2026_07.csv",
    "2026": "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/"
            "Monthly_Report_By_Contract_2026_07.csv",
}
VALUE = {"lo": 400, "mid": 500, "hi": 600}


# ---------------------------------------------------------------- shared layer
def load_world():
    gaps = {(g["year"], g["mid"], g["org"]): g for g in json.load(open(GAPS))}
    jobs, skips = C.build_jobs(cluster_vintage=CRIT_VINTAGE)
    baseline = {}
    for year in C.YEARS:
        stars, summary, cai = load_inputs(year, C.BASELINE_VINTAGE[year])
        baseline[year] = dict(stars=stars, summary=summary, cai=cai,
                              dup_d=frozenset(dup_d_ids(year, C.BASELINE_VINTAGE[year])))
    return gaps, jobs, skips, baseline


def split_boundaries(job, g):
    """(raw dp Fractions, guardrailed Fractions or None, display-rounded
    Fractions); splits with fewer than five distinct scores return
    (None, None, None)."""
    if len(g["dp_cutpoints"]) != 4:
        return None, None, None
    raw = [F(v) for v in g["dp_cutpoints"]]
    if job["gr_note"] == "applied":
        cap = F(job["cap"])
        clamped = [guardrail_boundary(v, F(p), cap)[0]
                   for v, p in zip(raw, job["prior"])]
    else:
        clamped = list(raw)
    final = [F(round_display(v, job["dd"])) for v in clamped]
    return raw, clamped, final


def star_at(x, bd, higher):
    if higher:
        return 1 + sum(1 for t in bd if x >= t)
    return 5 - sum(1 for t in bd if x > t)


def delta_sets(gaps, jobs, baseline, *, include_degenerate, stage="final"):
    """{(year,mid,org): [(cid, sw, sd), ...]}. stage: 'raw'|'clamped'|'final'
    picks which boundary stage assigns the criterion star (lineage use).
    include_degenerate=True replicates the earlier empty-boundary rule."""
    out = {}
    for job in jobs:
        key = (job["year"], job["mid"], job["org"])
        g = gaps.get(key)
        if g is None:
            continue
        raw, clamped, final = split_boundaries(job, g)
        if raw is None:
            if not include_degenerate:
                continue
            bd = []                        # earlier rule: empty boundary list
        else:
            bd = {"raw": raw, "clamped": clamped, "final": final}[stage]
        st_pub = baseline[job["year"]]["stars"]
        deltas = []
        _dis = _DIS[job["year"]]
        _pm = _PRIORMAP[job["year"]]
        _part = "C" if job["mid"].startswith("C") else "D"
        for cid, s in job["pool"]:
            sw = st_pub.get(cid, {}).get(job["mid"])
            if sw is None:
                continue
            sd = star_at(F(s), bd, job["higher"])
            # disaster measure-star hold-harmless (same helper the census
            # uses; the recount's independence is the application/rating/
            # dollar-estimate path)
            sd = disaster_higher_of(job["name"], _part, cid, sd, _dis, _pm)
            if sd != sw:
                deltas.append((cid, int(sw), int(sd)))
        out[key] = deltas
    return out


def apply_and_rate(baseline, dsets):
    """Independent application loop: per contract, substitute criterion stars
    and re-rate directly with rate_contract. Returns rows for contracts whose
    rating crosses 4.0."""
    idx = defaultdict(dict)                # (year,cid) -> {mid: sd}
    for (year, mid, org), deltas in dsets.items():
        for cid, sw, sd in deltas:
            idx[(year, cid)][mid] = sd
    flips = []
    for year in C.YEARS:
        bl = baseline[year]
        spec = AggSpec(year)
        for cid, srow in bl["summary"].items():
            st = bl["stars"].get(cid)
            if not st:
                continue
            moved = {m: v for m, v in idx.get((year, cid), {}).items() if m in st}
            if not moved:
                continue
            oc = org_class_of(srow)
            hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
            kw = dict(gate_convention="raw",
                      disaster=disaster_pct_of(srow, year) >= 25,
                      dup_d=bl["dup_d"])
            b_out = rate_contract(spec, st, oc, bl["cai"].get(cid, {}), **kw)
            if hh not in b_out:
                continue
            b = F(b_out[hh][0])
            st_cf = dict(st)
            for m, v in moved.items():
                st_cf[m] = max(1, min(5, v))
            c_out = rate_contract(spec, st_cf, oc, bl["cai"].get(cid, {}), **kw)
            if hh not in c_out:
                continue
            c = F(c_out[hh][0])
            if (b >= 4) != (c >= 4):
                flips.append(dict(
                    year=year, contract_id=cid, org_class=oc,
                    qbp_relevant=oc != "PDP",
                    direction="DOWN" if b >= 4 else "UP",
                    baseline=str(b), criterion_rating=str(c),
                    baseline_raw=str(b_out[hh][1]),
                    criterion_raw=str(c_out[hh][1]),
                    n_measure_moves=len(moved)))
    return sorted(flips, key=lambda r: (r["year"], r["contract_id"]))


def enrollment_map(year):
    import csv
    m = {}
    with open(os.path.join(BASE, ENROLL_FILE[year]), newline="",
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            v = row["Enrollment"].replace(",", "")
            m[row["Contract Number"]] = int(v) if v.isdigit() else 0
    return m


# ------------------------------------------------------------------ recount
def mode_recount():
    gaps, jobs, skips, baseline = load_world()
    dsets = delta_sets(gaps, jobs, baseline, include_degenerate=False)
    flips = apply_and_rate(baseline, dsets)
    census = json.load(open(CENSUS))
    cref = {(p["year"], p["contract_id"]):
            (p["direction"], p["baseline"], p["criterion_rating"])
            for p in census["flips_criterion"]}
    mine = {(r["year"], r["contract_id"]):
            (r["direction"], r["baseline"], r["criterion_rating"])
            for r in flips}
    mismatches = []
    for k in sorted(set(cref) | set(mine)):
        if cref.get(k) != mine.get(k):
            mismatches.append(dict(key=list(k), census=cref.get(k),
                                   recount=mine.get(k)))
    # dollar-estimate recount: priceable rows minus the demoted pair, with
    # the all-qbp-rows definition kept as a labeled diagnostic
    emaps = {y: enrollment_map(y) for y in C.YEARS}
    cen_rows = {(p["year"], p["contract_id"]): p
                for p in census["flips_criterion"]}
    qbp_all = [r for r in flips if r["qbp_relevant"]]
    for r in qbp_all:
        r["enrollment"] = emaps[r["year"]].get(r["contract_id"], 0)
    DEM = {("2024", "H1304"), ("2025", "H5273")}
    qbp = [r for r in qbp_all
           if cen_rows.get((r["year"], r["contract_id"]), {}).get("qbp_priceable")
           and (r["year"], r["contract_id"]) not in DEM]
    enr = sum(r["enrollment"] for r in qbp)
    gross = {k: enr * v for k, v in VALUE.items()}
    enr_all = sum(r["enrollment"] for r in qbp_all)
    # disclosure counts
    def dcount(rows):
        seen = defaultdict(list)
        for r in rows:
            seen[r["contract_id"]].append(r["year"])
        doubled = sorted(c for c, ys in seen.items() if len(ys) > 1)
        return len(seen), doubled
    n_qbp_distinct, qbp_doubled = dcount(qbp)
    pdp = [r for r in flips if not r["qbp_relevant"]]
    n_pdp_distinct, pdp_doubled = dcount(pdp)
    n_all_distinct, all_doubled = dcount(flips)
    top8 = sorted(qbp, key=lambda r: -r["enrollment"])[:8]
    top8_enr = sum(r["enrollment"] for r in top8)
    disclosure = dict(
        n_rows_qbp=len(qbp), n_distinct_qbp_contracts=n_qbp_distinct,
        qbp_multi_year_contracts=qbp_doubled,
        n_rows_pdp=len(pdp), n_distinct_pdp_contracts=n_pdp_distinct,
        pdp_multi_year_contracts=pdp_doubled,
        n_rows_total=len(flips), n_distinct_contracts_total=n_all_distinct,
        multi_year_contracts_total=all_doubled,
        n_multi_year_contracts_total=len(all_doubled),
        top8_rows=[dict(year=r["year"], contract_id=r["contract_id"],
                        enrollment=r["enrollment"]) for r in top8],
        top8_enrollment=top8_enr,
        top8_share_pct=round(100.0 * top8_enr / enr, 1),
        top8_distinct_contracts=len({r["contract_id"] for r in top8}))
    out = {
        "register": ("Independent recount of runs/criterion_census.json: a "
                  "separate application loop over the stored primitives; the "
                  "full flip-row list is emitted so row-list identity can be "
                  "checked. Disclosure counts emitted."),
        "generator": "src/criterion_verify_suite.py mode=recount",
        "verdict": "CONFIRMED" if not mismatches else "MISMATCH",
        "n_flips": len(flips),
        "n_mismatches_vs_census": len(mismatches),
        "mismatches": mismatches,
        "pricing_recount": dict(
            grain=("priceable rows minus the two endpoint-conditional "
                   "contract-years; compare against the dollar-estimate "
                   "aggregates, not the census summary (which counts the "
                   "pre-demotion priceable rows)"),
            n_rows_priced=len(qbp),
            total_enrollment=enr, gross_total_3yr_usd=gross,
            all_qbp_grain_diagnostic=dict(
                note="all-qbp-rows definition (diagnostic only)",
                total_enrollment=enr_all),
            matches_dollar_record=None),
        "disclosure_counts": disclosure,
        "disclosure_counts_of_record": (lambda cen: (lambda pr:
            (lambda top8p, enrp: dict(
                note=("Priced = qbp_priceable rows (cost and no-published-"
                      "overall rows counted, never in the estimate) minus the "
                      "two endpoint-conditional contract-years"),
                n_rows_priced=len(pr),
                n_distinct_priced_contracts=len({r["contract_id"] for r in pr}),
                total_enrollment=enrp,
                top8_enrollment=sum(r["enrollment"] for r in top8p),
                top8_share_pct=round(100.0 * sum(r["enrollment"]
                                                 for r in top8p) / enrp, 1),
                top8_distinct_contracts=len({r["contract_id"]
                                             for r in top8p})))(
                sorted(pr, key=lambda r: -r["enrollment"])[:8],
                sum(r["enrollment"] for r in pr)))(
            [r for r in cen["flips_criterion"] if r.get("qbp_priceable")
             and (r["year"], r["contract_id"]) not in
             {("2024", "H1304"), ("2025", "H5273")}]))(
            json.load(open(CENSUS))),
        "disclosure_counts_printed_grain": (lambda qbp2:
            (lambda top8p, enrp: dict(
                note="printed definition (H1304/H5273 excluded)",
                n_rows_qbp=len(qbp2),
                n_distinct_qbp_contracts=len({r["contract_id"] for r in qbp2}),
                total_enrollment=enrp,
                top8_enrollment=sum(r["enrollment"] for r in top8p),
                top8_share_pct=round(100.0 * sum(r["enrollment"]
                                                 for r in top8p) / enrp, 1),
                top8_distinct_contracts=len({r["contract_id"]
                                             for r in top8p})))(
                sorted(qbp2, key=lambda r: -r["enrollment"])[:8],
                sum(r["enrollment"] for r in qbp2)))(
            [r for r in qbp if (r["year"], r["contract_id"]) not in
             {("2024", "H1304"), ("2025", "H5273")}]),
        "flip_rows": flips,
    }
    try:
        dv = json.load(open(os.path.join(BASE, "runs",
                                         "criterion_dollar_estimate.json")))
        agg_key = ("priced_of_record" if "priced_of_record" in dv["aggregates"]
                   else "criterion_80")
        A = dv["aggregates"][agg_key]
        out["pricing_recount"]["matches_dollar_record"] = (
            enr == A["total_enrollment"]
            and gross["mid"] == int(A["gross_total_usd"]["mid"]))
    except Exception as e:
        out["pricing_recount"]["matches_dollar_record"] = f"unavailable: {e!r}"
    p = os.path.join(BASE, "runs", "criterion_recount_stored.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print(json.dumps({k: out[k] for k in ("verdict", "n_flips",
                                          "n_mismatches_vs_census")}, indent=1))
    print(json.dumps(disclosure, indent=1))
    print("->", p)


# ------------------------------------------------------------------- chains
TOP_KEYS = [("2024", "H0524"), ("2026", "H2406"), ("2024", "H8768"),
            ("2025", "H3805"), ("2024", "H3805"), ("2025", "H5322"),
            ("2024", "H3815"), ("2026", "H4514")]


def mode_chains():
    gaps, jobs, skips, baseline = load_world()
    dsets = delta_sets(gaps, jobs, baseline, include_degenerate=False)
    names = {(j["year"], j["mid"], j["org"]): j["name"] for j in jobs}
    bnd = {}
    for job in jobs:
        g = gaps.get((job["year"], job["mid"], job["org"]))
        if g:
            bnd[(job["year"], job["mid"], job["org"])] = (
                split_boundaries(job, g), job["gr_note"])
    idx = defaultdict(dict)
    for (year, mid, org), deltas in dsets.items():
        for cid, sw, sd in deltas:
            idx[(year, cid)][mid] = (sw, sd, org)
    entries = []
    for key in TOP_KEYS + [("2025", "H3655")]:
        year, cid = key
        bl = baseline[year]
        st = bl["stars"].get(cid)
        srow = bl["summary"].get(cid)
        oc = org_class_of(srow)
        hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
        kw = dict(gate_convention="raw",
                  disaster=disaster_pct_of(srow, year) >= 25,
                  dup_d=bl["dup_d"])
        spec = AggSpec(year)
        moved = {m: v for m, v in idx.get((year, cid), {}).items() if m in st}
        moves = []
        for m, (sw, sd, org) in sorted(moved.items()):
            (raw, clamped, final), gr = bnd.get((year, m, org),
                                                ((None, None, None), "?"))
            moves.append(dict(
                measure=m, name=names.get((year, m, org), m),
                published_star=sw, criterion_star=sd,
                raw_dp_boundaries=[str(v) for v in raw] if raw else None,
                guardrail=gr,
                clamped_boundaries=([str(v) for v in clamped]
                                    if clamped and gr == "applied" else None),
                display_final_boundaries=[str(v) for v in final] if final else None))
        b_out = rate_contract(spec, st, oc, bl["cai"].get(cid, {}), **kw)
        st_cf = dict(st)
        for m, (sw, sd, org) in moved.items():
            st_cf[m] = max(1, min(5, sd))
        c_out = rate_contract(spec, st_cf, oc, bl["cai"].get(cid, {}), **kw)
        b, c = F(b_out[hh][0]), F(c_out[hh][0])
        n_dn = sum(1 for m in moved if moved[m][1] < moved[m][0])
        n_up = sum(1 for m in moved if moved[m][1] > moved[m][0])
        e = dict(year=year, contract_id=cid, org_class=oc,
                 rating_type=hh,
                 published_rating=str(b), criterion_rating=str(c),
                 published_raw=str(b_out[hh][1]), criterion_raw=str(c_out[hh][1]),
                 crosses_4=( (b >= 4) != (c >= 4) ),
                 n_moves=len(moved), n_down_moves=n_dn, n_up_moves=n_up,
                 measure_moves=moves)
        if key == ("2025", "H3655"):
            e["flag"] = ("NOT-A-FLIP-ROW: both ratings sit below 4.0; no "
                         "direction is claimed.")
        entries.append(e)
    out = {
        "register": ("Boundary chains for the eight largest-enrollment rows of "
                  "the bonus-status census: raw exact-optimum cut point -> "
                  "guardrail-clamped (where applied) -> display-rounded; "
                  "ratings as exact fractions; crossing = the 4.0 bonus line."),
        "generator": "src/criterion_verify_suite.py mode=chains",
        "entries": entries,
    }
    p = os.path.join(BASE, "runs", "criterion_toprow_chains.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    for e in entries:
        print(f"{e['year']} {e['contract_id']}: {e['published_rating']} -> "
              f"{e['criterion_rating']} moves {e['n_moves']} "
              f"(dn {e['n_down_moves']}/up {e['n_up_moves']}) "
              f"crosses={e['crosses_4']}" + (" [FLAGGED]" if "flag" in e else ""))
    print("->", p)


# ---------------------------------------------------------------- mechanism
def shock_bands(baseline, dsets, allow_sentinel=False):
    """Net weighted shock per contract (sum of weight x star delta over its
    delta measures), banded by baseline rating: ge4 / 3.5 / lt3.5 per year.
    allow_sentinel: entries for measures absent from the stars table are kept
    as zero-shock population membership markers (fixed-population variant)."""
    idx = defaultdict(dict)
    for (year, mid, org), deltas in dsets.items():
        for cid, sw, sd in deltas:
            idx[(year, cid)][mid] = sd - sw
    bands = {}
    for year in C.YEARS:
        bl = baseline[year]
        spec = AggSpec(year)
        cells = defaultdict(lambda: dict(n=0, neg=0, pos=0, zero=0))
        for cid, srow in bl["summary"].items():
            st = bl["stars"].get(cid)
            if not st:
                continue
            allm = idx.get((year, cid), {})
            moved = {m: d for m, d in allm.items() if m in st}
            if not moved and not (allow_sentinel and allm):
                continue
            oc = org_class_of(srow)
            hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
            kw = dict(gate_convention="raw",
                      disaster=disaster_pct_of(srow, year) >= 25,
                      dup_d=bl["dup_d"])
            b_out = rate_contract(spec, st, oc, bl["cai"].get(cid, {}), **kw)
            if hh not in b_out:
                continue
            b = F(b_out[hh][0])
            band = "ge4" if b >= 4 else ("3.5" if b == F(7, 2) else "lt3.5")
            shock = sum(spec.weights.get(m, F(0)) * d for m, d in moved.items())
            cell = cells[f"{year}/{band}"]
            cell["n"] += 1
            cell["neg" if shock < 0 else "pos" if shock > 0 else "zero"] += 1
        bands.update(cells)
    return {k: dict(v) for k, v in bands.items()}


def mode_mechanism():
    gaps, jobs, skips, baseline = load_world()
    v1rc = json.load(open(V1RC))
    stored_v1 = v1rc["direction_mechanism"]["net_weighted_shock_by_rating_band"]
    # comparison: the earlier basis (fewer-than-five-distinct-score splits
    # included) against the nine stored cells
    d_v1 = delta_sets(gaps, jobs, baseline, include_degenerate=True)
    cal = shock_bands(baseline, d_v1)
    cal_ok, cal_diff = True, {}
    for k, v in stored_v1.items():
        mine = cal.get(k, {})
        if {x: mine.get(x) for x in ("n", "neg", "pos", "zero")} != \
           {x: v.get(x) for x in ("n", "neg", "pos", "zero")}:
            cal_ok = False
            cal_diff[k] = dict(stored=v, recomputed=mine)
    # census basis, two population conventions, both emitted:
    # (a) census population: contracts with >=1 census-basis move (this
    #     driver's native convention);
    # (b) fixed population: contracts with >=1 earlier-basis move, shocks from
    #     census-basis deltas (D07-only movers become zero-shock rows).
    #     neg/pos are identical between the two by construction; only n/zero
    #     differ.
    d_v2 = delta_sets(gaps, jobs, baseline, include_degenerate=False)
    v2 = shock_bands(baseline, d_v2)
    d_v2_fixedpop = {}
    v2_idx = {}
    for key, deltas in d_v2.items():
        for cid, sw, sd in deltas:
            v2_idx.setdefault((key[0], cid), {})[key[1]] = (cid, sw, sd)
    # rebuild: for every (year,cid) in the earlier index, emit its census-basis deltas
    # (possibly none) under a sentinel zero-delta so the contract stays in
    # the population with shock contribution 0.
    v1_pop = set()
    for key, deltas in d_v1.items():
        for cid, sw, sd in deltas:
            v1_pop.add((key[0], cid))
    d_v2_fixedpop = {k: list(v) for k, v in d_v2.items()}
    sentinel = []
    v2_pop = {(k[0], cid) for k, ds in d_v2.items() for cid, _, _ in ds}
    for (year, cid) in sorted(v1_pop - v2_pop):
        sentinel.append((year, cid))
    # zero-delta sentinel measure: use a per-year pseudo split with d=0
    for (year, cid) in sentinel:
        d_v2_fixedpop.setdefault((year, "__SENTINEL__", "ALL"), []).append(
            (cid, 3, 3))  # sw==sd -> would be filtered; use explicit 0-shock
    # shock_bands filters sw==sd nowhere (deltas lists carry sd!=sw normally);
    # a (3,3) entry yields d=0 and weight lookup 0 -> shock 0, and keeps the
    # contract in the population. "__SENTINEL__" is in no stars table, so it
    # must bypass the `m in st` filter: handled inside shock_bands via the
    # sentinel name.
    v2_fixedpop = shock_bands(baseline, d_v2_fixedpop, allow_sentinel=True)
    # star-cell lineage
    def cellcount(stage):
        ds = delta_sets(gaps, jobs, baseline, include_degenerate=False,
                        stage=stage)
        return sum(len(v) for v in ds.values())
    pool_cells = 0
    for job in jobs:
        g = gaps.get((job["year"], job["mid"], job["org"]))
        if g is None or len(g["dp_cutpoints"]) != 4:
            continue
        st_pub = baseline[job["year"]]["stars"]
        pool_cells += sum(1 for cid, s in job["pool"]
                          if st_pub.get(cid, {}).get(job["mid"]) is not None)
    lineage = dict(pool_star_cells=pool_cells,
                   raw_stage_moved=cellcount("raw"),
                   guardrail_stage_moved=cellcount("clamped"),
                   display_stage_moved=cellcount("final"))
    out = {
        "register": ("Net-weighted-shock bands on the bonus-status census basis "
                  "(criterion vintages, name+part guardrail priors, disaster "
                  "hold-harmless, 120 computations with at least five "
                  "distinct scores). Shock = sum over a contract's changed "
                  "measures of AggSpec weight x (exact-optimum star - "
                  "published star); bands by baseline rating ge4/3.5/lt3.5; "
                  "population = contracts with >=1 delta on the stated "
                  "basis. The earlier-basis cells differ from the "
                  "shock_bands_v2_basis cells (all_nine_cells_reproduced "
                  "records the comparison). The star-cell lineage counts "
                  "pool cells, cells moved at the raw stage, after the "
                  "guardrail and after display rounding (the last equals "
                  "the census n_star_cells_moved)."),
        "generator": "src/criterion_verify_suite.py mode=mechanism",
        "calibration_v1_basis": dict(all_nine_cells_reproduced=cal_ok,
                                     differences=cal_diff, recomputed=cal),
        "shock_bands_v2_basis": v2,
        "shock_bands_v2_basis_v1_fixed_population": v2_fixedpop,
        "star_cell_lineage_v2": lineage,
    }
    p = os.path.join(BASE, "runs", "criterion_mechanism_v2.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print("earlier-basis cells reproduced:", cal_ok)
    for y in ("2024", "2025", "2026"):
        print(f"{y}/ge4:", v2.get(f"{y}/ge4"))
    print("lineage:", lineage)
    print("->", p)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__" and (sys.argv[1] if len(sys.argv) > 1 else "all") in ("recount", "all") \
        and not all(os.path.exists(os.path.join(BASE, p)) for p in ENROLL_FILE.values()):
    print("criterion_verify_suite: missing CMS monthly enrollment file(s) under data/raw/enrollment/ "
          "(not included; re-fetch each by the URL and sha256 in CMS_RAW_PINS.json; see "
          "data/raw/enrollment/SOURCES_enrollment.txt); the recount mode needs them",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("recount", "all"):
        mode_recount()
    if mode in ("chains", "all"):
        mode_chains()
    if mode in ("mechanism", "all"):
        mode_mechanism()
