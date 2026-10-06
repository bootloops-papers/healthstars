# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Improvement-star sensitivity bound on the undercount.

The flip census holds improvement measures (C27/C30, D04) at published stars
in BOTH arms because their input scores are not public (Tech Notes: the
improvement measure is computed from year-over-year measure-score movement,
whose inputs CMS does not release at contract level). The paper states this
as an undercount. This run bounds how conservative the undercount is by
direct sensitivity: perturb the improvement star(s) by +/-1 (clamped [1,5]),
identically in baseline and counterfactual, and recompute the QBP-relevant
rating with the exact production aggregation (aggregate.rate_contract,
gate_convention="raw", post-revision, flip-census kwargs).

SCENARIOS (6): delta in {-1,+1} applied to (a) the Part C improvement measure
only (C27 in SY2024/2025, C30 in SY2026), (b) D04 only, (c) both. Perturbation
is a no-op on a contract that is not rated on the perturbed measure; a clamp
at 1 or 5 is recorded as effective delta 0 for that measure.

THREE QUESTIONS:
 Q1 STABILITY OF THE EXACT SET: for the 15 contracts of the census's exact
    set (10 stress survivors + 5 near-tier demotions), replay all 84 stored
    realizations (24 census + 60 stress; per-realization deltas reconstructed
    from runs/enclosure_census_docsonly.json + runs/stress_exact_docsonly
    .json measure_results, errored splits excluded exactly as production)
    under each scenario in TWO arm conventions:
      BOTH-ARMS (scenario names "C+1" etc.): perturbation applied identically
      to baseline and counterfactual star vectors. Shifts both weighted means
      identically, but the reward-factor thresholds and the improvement
      hold-harmless are nonlinear — computed, not assumed. An undo with
      reason baseline_crossed is a cliff-adjacency fact about the contract
      (in that hypothetical world it holds/loses QBP in BOTH arms), not a
      flaw of the census.
      CF-ONLY (scenario names "C+1:cf" etc.): baseline held at published
      stars (they ARE the published record — no uncertainty on that arm);
      only the counterfactual arm's improvement star is perturbed. This is
      the strict model of the held-fixed convention: the unreplayable
      object is what the improvement star WOULD BE under the substituted
      cut-point rule, and +/-1 brackets it at the granularity every other
      measure exhibited across realizations.
    A crossing is UNDONE in a realization if the identity scenario flips
    there but the perturbed scenario does not.
 Q2 CLIFF-ADJACENCY: for every rated QBP-relevant contract (MA-PD primary;
    MA-only tallied separately; PDPs get no QBP, counted only), does a single
    improvement-star step move the BASELINE rating across 4.0? This bounds
    how much the improvement layer COULD matter for any flip census.
 Q3 DIRECTION: per exact-set/near-tier contract and scenario, can the
    perturbation only ADD flips, or does it REMOVE any (n_flip under scenario
    < n_flip under identity)? Also replayed for the realization-dependent
    QBP set over the 24 census realizations.

INTEGRITY: the identity scenario must reproduce the stored
ratings_by_realization strings for all 15 stress contracts (24 census + 60
stress values each) and the stored n_flip_realizations for the realization-
dependent set; any mismatch is a hard error.

Output: runs/improvement_sensitivity.json.  Run: python3 improvement_sensitivity.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Appendix C.6
    (improvement stars on the recomputed side).
Run:  cd src && python3 improvement_sensitivity.py
Requires: Python >= 3.10, standard library only.
"""

import json
import os
import sys
from collections import defaultdict
from fractions import Fraction as F

from aggregate import (AggSpec, rate_contract, load_inputs, org_class_of,
                       disaster_pct_of, dup_d_ids)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs")
BASELINE_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}
YEARS = ("2024", "2025", "2026")
SCENARIOS = [("C", -1), ("C", +1), ("D", -1), ("D", +1), ("CD", -1), ("CD", +1)]


def scen_name(which, delta):
    return f"{which}{'+1' if delta > 0 else '-1'}"


class YearCtx:
    def __init__(self, year):
        self.year = year
        self.spec = AggSpec(year)
        vin = BASELINE_VINTAGE[year]
        self.stars, self.summary, self.cai = load_inputs(year, vin)
        self.dd = frozenset(dup_d_ids(year, vin))
        self.imp_c = next(m for m in self.spec.imp_ids if m.startswith("C"))
        self.imp_d = next(m for m in self.spec.imp_ids if m.startswith("D"))


def perturb(st, ctx, which, delta):
    """Return (perturbed star dict, {mid: effective_delta}). No-op mids absent
    from st are skipped; clamping records effective delta 0."""
    mids = []
    if "C" in which:
        mids.append(ctx.imp_c)
    if "D" in which:
        mids.append(ctx.imp_d)
    st2, eff = dict(st), {}
    for m in mids:
        if m in st2:
            nv = max(1, min(5, st2[m] + delta))
            eff[m] = nv - st2[m]
            st2[m] = nv
    return st2, eff


class Rater:
    """Memoized QBP-relevant rating for one contract (fixed measure set)."""

    def __init__(self, ctx, cid, srow, st):
        self.ctx, self.st = ctx, st
        self.oc = org_class_of(srow)
        self.hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[self.oc]
        self.kw = dict(gate_convention="raw",
                       disaster=disaster_pct_of(srow, ctx.year) >= 25,
                       dup_d=ctx.dd)
        self.cai = ctx.cai.get(cid, {})
        self.mids = sorted(st)
        self.memo = {}

    def rating(self, st):
        key = tuple(st[m] for m in self.mids)
        if key not in self.memo:
            res = rate_contract(self.ctx.spec, st, self.oc, self.cai, **self.kw)
            self.memo[key] = F(res[self.hh][0]) if self.hh in res else None
        return self.memo[key]


# ---------------------------------------------------------------- Q2: cliff adjacency
def cliff_adjacency(ctxs):
    out = {y: {} for y in YEARS}
    crossed = []
    pop = {y: {"MA-PD": 0, "MA-only": 0, "PDP_rated_excluded": 0} for y in YEARS}
    union_crossed = defaultdict(set)  # (year, oc) -> set of cids crossed in >=1 scenario
    for year in YEARS:
        ctx = ctxs[year]
        raters = {}
        for cid, srow in ctx.summary.items():
            st = ctx.stars.get(cid)
            if not st:
                continue
            r = Rater(ctx, cid, srow, st)
            b0 = r.rating(st)
            if b0 is None:
                continue
            if r.oc == "PDP":
                pop[year]["PDP_rated_excluded"] += 1
                continue
            pop[year][r.oc] += 1
            raters[cid] = (r, b0)
        for which, delta in SCENARIOS:
            sn = scen_name(which, delta)
            cnt = {"MA-PD": 0, "MA-only": 0, "noop": 0}
            for cid, (r, b0) in raters.items():
                st_s, eff = perturb(r.st, ctx, which, delta)
                if not eff or all(v == 0 for v in eff.values()):
                    cnt["noop"] += 1
                    continue
                b_s = r.rating(st_s)
                if b_s is None:
                    continue
                if (b0 >= 4) != (b_s >= 4):
                    cnt[r.oc] += 1
                    union_crossed[(year, r.oc)].add(cid)
                    crossed.append({
                        "year": year, "contract_id": cid, "org_class": r.oc,
                        "scenario": sn, "baseline": str(b0),
                        "perturbed": str(b_s),
                        "crossing": "up" if b_s >= 4 else "down",
                        "effective_deltas": eff})
            out[year][sn] = cnt
    union = {y: {"MA-PD": len(union_crossed[(y, "MA-PD")]),
                 "MA-only": len(union_crossed[(y, "MA-only")])} for y in YEARS}
    return {"population_rated_qbp": pop, "counts": out,
            "union_any_scenario": union,
            "union_any_scenario_total_mapd": sum(u["MA-PD"] for u in union.values()),
            "union_any_scenario_total_maonly": sum(u["MA-only"] for u in union.values()),
            "crossed_contracts": crossed}


# ---------------------------------------------------------------- delta reconstruction
def build_delta_idx(doc, errored_splits):
    err = {tuple(e) for e in errored_splits}
    idx = defaultdict(dict)  # (year,cid,mid) -> {realization: delta}
    for r in doc["measure_results"]:
        if "error" in r or (r["year"], r["mid"], r["org"]) in err:
            continue
        for cid, sw, sd in r["deltas"]:
            idx[(r["year"], cid, r["mid"])][r["realization"]] = int(sd) - int(sw)
    return idx


def cf_stars(st, year, cid, didx, rn):
    tm = [m for m in st if (year, cid, m) in didx]
    st_cf = dict(st)
    for m in tm:
        d = didx[(year, cid, m)].get(rn, 0)
        if d:
            st_cf[m] = max(1, min(5, st[m] + d))
    return st_cf


# ---------------------------------------------------------------- Q1/Q3 replay
def replay_contract(ctx, cid, didx_by_reg, realizations_by_reg, stored):
    """realizations_by_reg: [('census', [rn...]), ('stress', [rn...])] (stress
    optional). stored: {reg: {rn: rating_str_or_None}} for integrity check.
    Returns per-scenario record incl. identity."""
    srow = ctx.summary[cid]
    st = ctx.stars[cid]
    r = Rater(ctx, cid, srow, st)
    # per-realization counterfactual star vectors (identity layer)
    cfv = {}
    for reg, rns in realizations_by_reg:
        for rn in rns:
            cfv[(reg, rn)] = cf_stars(st, ctx.year, cid, didx_by_reg[reg], rn)
    total = len(cfv)
    mism = []
    out = {}
    ident_flips = {}
    modes = ([("none", 0, "both")]
             + [(w, d, "both") for w, d in SCENARIOS]
             + [(w, d, "cf") for w, d in SCENARIOS])
    for which, delta, arm in modes:
        sn = ("identity" if which == "none"
              else scen_name(which, delta) + ("" if arm == "both" else ":cf"))
        if which == "none":
            st_b, eff = st, {}
        elif arm == "cf":
            st_b = st                      # baseline held at published stars
            eff = perturb(st, ctx, which, delta)[1]   # same clamp math (cf
            # improvement stars are the published ones in every realization)
        else:
            st_b, eff = perturb(st, ctx, which, delta)
        b_s = r.rating(st_b)
        flips, ratings = {}, {}
        for (reg, rn), stc in cfv.items():
            stc_s = stc if which == "none" else perturb(stc, ctx, which, delta)[0]
            c = r.rating(stc_s)
            ratings[(reg, rn)] = c
            flips[(reg, rn)] = (c is not None and b_s is not None
                                and ((b_s >= 4 and c < 4) or (b_s < 4 and c >= 4)))
        if which == "none":
            ident_flips = dict(flips)
            ident_ratings = dict(ratings)
            b0 = b_s
            for reg, bank in (stored or {}).items():
                for rn, want in bank.items():
                    got = ratings.get((reg, rn))
                    gots = str(got) if got is not None else None
                    if gots != want:
                        mism.append((reg, rn, gots, want))
        nf = sum(flips.values())
        undone = sorted(f"{reg}:{rn}" for (reg, rn), fl in flips.items()
                        if ident_flips.get((reg, rn)) and not fl)
        added = sorted(f"{reg}:{rn}" for (reg, rn), fl in flips.items()
                       if fl and not ident_flips.get((reg, rn)))
        reasons = {}
        if undone:
            base_moved = (b_s is not None and b0 is not None
                          and (b_s >= 4) != (b0 >= 4))
            for tag in undone:
                reg, rn = tag.split(":", 1)
                c_s, c_i = ratings.get((reg, rn)), ident_ratings.get((reg, rn))
                cf_moved = (c_s is not None and c_i is not None
                            and (c_s >= 4) != (c_i >= 4))
                reasons[tag] = ("both_crossed" if base_moved and cf_moved
                                else "baseline_crossed" if base_moved
                                else "cf_crossed")
        rec = {"baseline_rating": str(b_s) if b_s is not None else None,
               "n_flip_realizations": nf, "k_realizations": total,
               "flips_all": nf == total,
               "counterfactual_set": sorted({str(c) for c in ratings.values()
                                             if c is not None})}
        if which != "none":
            rec["arm"] = arm
            rec["effective_deltas"] = eff
            rec["noop"] = (not eff) or all(v == 0 for v in eff.values())
            rec["n_undone"] = len(undone)
            rec["undone_realizations"] = undone
            rec["undone_reasons"] = reasons
            rec["n_added"] = len(added)
            rec["added_realizations"] = added if len(added) <= 90 else added[:90]
        out[sn] = rec
    return out, mism


def main():
    ctxs = {y: YearCtx(y) for y in YEARS}
    enc = json.load(open(os.path.join(RUNS, "enclosure_census_docsonly.json")))
    strs = json.load(open(os.path.join(RUNS, "stress_exact_docsonly.json")))
    census_rns = enc["realizations"]
    stress_rns = strs["realizations"]
    didx = {"census": build_delta_idx(enc, enc.get("errored_splits", [])),
            "stress": build_delta_idx(strs, strs.get("errored_splits", []))}

    print("Q2 cliff adjacency...", flush=True)
    q2 = cliff_adjacency(ctxs)
    print(json.dumps(q2["counts"], indent=1), flush=True)

    breaker_ids = {(b["year"], b["contract_id"]) for b in strs["summary"]["breakers"]}
    exact15 = [c for c in enc["contracts"] if c["class"] == "exact-within-groups"]
    assert len(exact15) == 15
    stress_by_id = {(c["year"], c["contract_id"]): c for c in strs["contracts"]}

    print("Q1 exact-set + near-tier replay (84 realizations x 7 scenarios)...",
          flush=True)
    q1_rows, all_mism = [], []
    for c in exact15:
        year, cid = c["year"], c["contract_id"]
        sc = stress_by_id[(year, cid)]
        stored = {"census": {rn: c["ratings_by_realization"][rn] for rn in census_rns},
                  "stress": {rn: sc["ratings_by_realization"][rn] for rn in stress_rns}}
        rec, mism = replay_contract(
            ctxs[year], cid, didx,
            [("census", census_rns), ("stress", stress_rns)], stored)
        all_mism.extend([(year, cid) + m for m in mism])
        tier = "near-tier" if (year, cid) in breaker_ids else "exact-set"
        removed = {sn: r["n_undone"] for sn, r in rec.items()
                   if sn != "identity" and r["n_undone"] > 0}
        q1_rows.append({"year": year, "contract_id": cid, "tier": tier,
                        "direction": c["direction"],
                        "published_baseline": c["baseline"],
                        "scenarios": rec,
                        "any_scenario_removes_flips": bool(removed),
                        "scenarios_removing_flips": removed})
        print(f"  {year} {cid} [{tier}] identity nf="
              f"{rec['identity']['n_flip_realizations']}/84 "
              f"removed_by={removed or 'NONE'}", flush=True)

    print("Q3 realization-dependent QBP set replay (24 realizations x 7 scenarios)...",
          flush=True)
    dep = [c for c in enc["contracts"]
           if c["class"] == "realization-dependent" and c["qbp_relevant"]]
    q3_rows = []
    for c in dep:
        year, cid = c["year"], c["contract_id"]
        stored = {"census": {rn: c["ratings_by_realization"][rn] for rn in census_rns}}
        rec, mism = replay_contract(ctxs[year], cid, didx,
                                    [("census", census_rns)], stored)
        all_mism.extend([(year, cid) + m for m in mism])
        if rec["identity"]["n_flip_realizations"] != c["n_flip_realizations"]:
            all_mism.append((year, cid, "n_flip", rec["identity"]["n_flip_realizations"],
                             c["n_flip_realizations"]))
        q3_rows.append({"year": year, "contract_id": cid,
                        "direction": c["direction"],
                        "published_baseline": c["baseline"],
                        "identity_n_flip": rec["identity"]["n_flip_realizations"],
                        "scenarios": {sn: {"n_flip": r["n_flip_realizations"],
                                           "n_undone": r.get("n_undone", 0),
                                           "n_added": r.get("n_added", 0),
                                           "noop": r.get("noop"),
                                           "baseline_rating": r["baseline_rating"]}
                                      for sn, r in rec.items() if sn != "identity"}})

    if all_mism:
        print(f"INTEGRITY FAILURE: {len(all_mism)} mismatches vs stored records")
        for m in all_mism[:20]:
            print("  ", m)
        raise SystemExit(1)
    print("integrity: identity scenario reproduces ALL stored per-realization "
          "ratings and flip counts (15 x 84 + %d x 24)" % len(dep), flush=True)

    # ------------------------------------------------------------- summaries
    base_scen = [scen_name(w, d) for w, d in SCENARIOS]
    all_scen = base_scen + [s + ":cf" for s in base_scen]
    exact_rows = [r for r in q1_rows if r["tier"] == "exact-set"]
    near_rows = [r for r in q1_rows if r["tier"] == "near-tier"]

    def removal_list(rows, suffix_cf):
        out = []
        for r in rows:
            rem = {sn: n for sn, n in r["scenarios_removing_flips"].items()
                   if sn.endswith(":cf") == suffix_cf}
            if rem:
                out.append({"year": r["year"], "contract_id": r["contract_id"],
                            "direction": r["direction"],
                            "scenarios_removing_flips": rem,
                            "undone_reasons": {sn: sorted(set(
                                r["scenarios"][sn]["undone_reasons"].values()))
                                for sn in rem}})
        return out

    dep_removed = {sn: sum(1 for r in q3_rows
                           if r["scenarios"][sn]["n_undone"] > 0)
                   for sn in all_scen}
    dep_added = {sn: sum(1 for r in q3_rows
                         if r["scenarios"][sn]["n_added"] > 0)
                 for sn in all_scen}
    dep_newly_exact = {sn: [
        {"year": r["year"], "contract_id": r["contract_id"]}
        for r in q3_rows if r["scenarios"][sn]["n_flip"] == 24]
        for sn in all_scen}

    summary = {
        "q1_any_exact_set_flip_undone_both_arms":
            bool(removal_list(exact_rows, False)),
        "q1_any_exact_set_flip_undone_cf_only":
            bool(removal_list(exact_rows, True)),
        "q1_exact_set_removals_both_arms": removal_list(exact_rows, False),
        "q1_exact_set_removals_cf_only": removal_list(exact_rows, True),
        "q1_near_tier_removals_both_arms": removal_list(near_rows, False),
        "q1_near_tier_removals_cf_only": removal_list(near_rows, True),
        "q1_exact_set_held_all_scenarios_both_conventions":
            [{"year": r["year"], "contract_id": r["contract_id"]}
             for r in exact_rows if not r["any_scenario_removes_flips"]],
        "q2_cliff_adjacent_mapd_by_year": q2["union_any_scenario"],
        "q2_cliff_adjacent_mapd_total": q2["union_any_scenario_total_mapd"],
        "q2_cliff_adjacent_maonly_total": q2["union_any_scenario_total_maonly"],
        "q3_realization_dependent_qbp_n": len(q3_rows),
        "q3_dep_contracts_with_flips_removed_per_scenario": dep_removed,
        "q3_dep_contracts_with_flips_added_per_scenario": dep_added,
        "q3_dep_newly_exact_set_per_scenario": {k: v for k, v in
                                                sorted(dep_newly_exact.items())
                                                if v},
    }
    scen_field = all_scen
    out = {
        "register": (
            "Improvement-star sensitivity bound on the held-fixed convention "
            "(C27/C30, D04 at published stars, both arms). 6 scenarios: delta "
            "in {-1,+1} on C-improvement / D04 / both, clamped [1,5], applied "
            "IDENTICALLY to baseline and counterfactual star vectors; QBP-"
            "relevant rating recomputed with aggregate.rate_contract exactly "
            "as the flip census (gate_convention raw, post-revision, disaster, dup_d). "
            "Exact-set and near-tier contracts replayed over all 84 stored realizations "
            "(24 census + 60 stress; per-realization deltas rebuilt from the "
            "stored measure_results, errored splits excluded as production); "
            "realization-dependent QBP set over the 24 census realizations. "
            "Identity scenario reproduced every stored per-realization rating "
            "(hard-checked). Cliff-adjacency computed on BASELINE ratings of "
            "every rated QBP-relevant contract (MA-PD primary, MA-only "
            "separate; PDPs excluded: no QBP). TWO arm conventions for the "
            "replay scenarios: both-arms (suffix-free names; identical "
            "perturbation both arms — an undo via baseline_crossed is a "
            "cliff-adjacency fact, the contract holds/loses QBP in BOTH "
            "worlds) and cf-only (':cf' suffix; baseline held at the "
            "published record, only the counterfactual arm's improvement "
            "star perturbed — the strict model of the held-fixed "
            "convention, since the unreplayable object is the improvement "
            "star under the substituted cut-point rule)."),
        "sources": ["runs/enclosure_census_docsonly.json",
                    "runs/stress_exact_docsonly.json"],
        "baseline_vintage": BASELINE_VINTAGE,
        "improvement_ids": {y: sorted(ctxs[y].spec.imp_ids) for y in YEARS},
        "scenarios": scen_field,
        "summary": summary,
        "q1_exact_set_and_near_tier": q1_rows,
        "q2_cliff_adjacency": q2,
        "q3_realization_dependent": q3_rows,
        "integrity": {"mismatches": 0,
                      "stored_records_checked":
                          {"stress_contracts_x_84": len(exact15),
                           "dependent_contracts_x_24": len(dep)}},
    }
    path = os.path.join(RUNS, "improvement_sensitivity.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
