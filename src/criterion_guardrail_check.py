# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Guardrail check for the bonus-status census.

PART 1: near-base check on the criterion boundaries (the method of
guardrail_base_check.py: live comparison path, half-display-unit prior
uncertainty). The bonus-status census applies the guardrail to the exact
DP boundaries on the full score list using published prior-year boundaries
at display precision (point values). Question: could a prior
differing by up to half a display unit change any guardrail clamp decision
in a way that moves a star cell of any of the 90 flip contracts? Method:
for every guardrail-applied split with 4 dp cutpoints, recompute the
guardrailed+rounded criterion boundary with the prior perturbed to BOTH
closed ends of its half-display-unit interval (prior +/- 0.5*10^-dd, dd =
the job's display digits); flag any boundary whose DISPLAYED value changes
under either perturbation; for each flagged (boundary, end) scenario,
recompute the affected contracts' criterion stars and re-rate them with the
single-measure override to test whether any of the 90 stored flips
(runs/criterion_census.json flips_criterion) changes flip status.

PART 2: guardrail dependence of the 90 changes (two counts; the 2026 D07
MA-PD computation, with fewer than five distinct scores, is held at
published):
 (a) how many of the 90 involve >=1 measure whose criterion boundary was
     clamped by the guardrail (displayed guardrailed value != displayed raw
     dp value on >=1 of the 4 boundaries of a split the contract's criterion
     star actually moved on);
 (b) how many of the 90 change flip status if the guardrail is not applied
     at all (raw dp boundaries, display-rounded, same aggregation).

Integrity: the nominal synth is rebuilt with the census's own code path and
must reproduce every stored per_split delta count and the stored 90-flip set
exactly, else abort.

Run   : python3 criterion_guardrail_check.py
Output: runs/criterion_guardrail_check.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.5, conventions (2)
    and (3), and Appendix C.4 (the guardrail against the published prior-year
    cut points).
Requires: Python >= 3.10; numpy, scipy.
"""
import hashlib
import json
import os
from collections import defaultdict
from fractions import Fraction as F

import enclosure_census_docsonly as C
import sys
from aggregate import (AggSpec, rate_contract, load_inputs, org_class_of,
                       disaster_pct_of, dup_d_ids)
from falsify_groups import round_display
from census_measures import (disaster_pcts, prior_published_star_map,
                             disaster_higher_of)
_DIS = {y: disaster_pcts(y, {"2024": "recalc", "2025": "current", "2026": "current"}[y]) for y in ("2024", "2025", "2026")}
_PRIORMAP = {y: prior_published_star_map(y) for y in ("2024", "2025", "2026")}
from guardrail import guardrail_boundary

BASE = C.BASE
OUT = os.path.join(BASE, "runs", "criterion_guardrail_check.json")
GAPS = os.path.join(BASE, "runs", "exact_gaps_criterionvintage.json")
# criterion vintages
CRIT_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}
CENSUS = os.path.join(BASE, "runs", "criterion_census.json")

HH_OF = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def stars_from_bd(x, bd, higher):
    if higher:
        return 1 + sum(1 for t in bd if x >= t)
    return 5 - sum(1 for t in bd if x > t)


def build_synth(jobs, gaps, baseline_stars, guardrail_on=True):
    """Rebuild the criterion census's synth rows (census code path verbatim);
    guardrail_on=False gives the no-guardrail counterfactual (raw dp
    boundaries, display-rounded). Splits with fewer than five distinct scores
    (len(dp_cutpoints)!=4) are held at published (skipped), as in the census."""
    synth = []
    for job in jobs:
        key = (job["year"], job["mid"], job["org"])
        g = gaps.get(key)
        if g is None or len(g["dp_cutpoints"]) != 4:
            continue
        vals = [F(v) for v in g["dp_cutpoints"]]
        if guardrail_on and job["gr_note"] == "applied":
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
                continue
            sd = stars_from_bd(F(s), bd, job["higher"])
            # disaster measure-star hold-harmless (census convention)
            sd = disaster_higher_of(job["name"],
                                    "C" if job["mid"].startswith("C") else "D",
                                    cid, sd, _DIS[job["year"]],
                                    _PRIORMAP[job["year"]])
            if sd != sw:
                deltas.append((cid, int(sw), int(sd)))
        synth.append(dict(year=job["year"], mid=job["mid"], org=job["org"],
                          name=job["name"], realization="criterion",
                          deltas=deltas, bd=bd, guardrail=job["gr_note"]))
    return synth


class Rater:
    """Targeted per-contract rating recompute (enclosure_aggregate's inner
    loop, one contract at a time), with the nominal criterion delta index."""

    def __init__(self, delta_idx):
        self.delta_idx = delta_idx     # (year, cid) -> {mid: sd - sw}
        self.ctx = {}
        self.base_cache = {}

    def _year_ctx(self, year):
        if year not in self.ctx:
            bv = C.BASELINE_VINTAGE[year]
            spec = AggSpec(year)
            stars, summary, cai = load_inputs(year, bv)
            ddup = frozenset(dup_d_ids(year, bv))
            self.ctx[year] = (spec, stars, summary, cai, ddup)
        return self.ctx[year]

    def base(self, year, cid):
        """(b_rounded_F, oc, hh, kw, st) or None if unrated."""
        key = (year, cid)
        if key in self.base_cache:
            return self.base_cache[key]
        spec, stars, summary, cai, ddup = self._year_ctx(year)
        out = None
        st = stars.get(cid)
        srow = summary.get(cid)
        if st and srow is not None:
            oc = org_class_of(srow)
            hh = HH_OF[oc]
            kw = dict(gate_convention="raw",
                      disaster=disaster_pct_of(srow, year) >= 25, dup_d=ddup)
            b = rate_contract(spec, st, oc, cai.get(cid, {}), **kw)
            if hh in b:
                out = (F(b[hh][0]), oc, hh, kw, st)
        self.base_cache[key] = out
        return out

    def rating_with_override(self, year, cid, override):
        """Rounded rating with nominal criterion deltas applied and
        override = {mid: star} imposed on top. None if unrated."""
        info = self.base(year, cid)
        if info is None:
            return None
        b, oc, hh, kw, st = info
        spec, _, _, cai, _ = self._year_ctx(year)
        st_cf = dict(st)
        for m, dv in self.delta_idx.get((year, cid), {}).items():
            st_cf[m] = max(1, min(5, st[m] + dv))
        for m, v in override.items():
            st_cf[m] = v
        cf = rate_contract(spec, st_cf, oc, cai.get(cid, {}), **kw)
        return F(cf[hh][0]) if hh in cf else None


def is_flip(b, c):
    return c is not None and ((b >= 4 and c < 4) or (b < 4 and c >= 4))


def main():
    gaps = {(g["year"], g["mid"], g["org"]): g for g in json.load(open(GAPS))}
    census = json.load(open(CENSUS))
    jobs, _ = C.build_jobs(cluster_vintage=CRIT_VINTAGE)
    jmap = {(j["year"], j["mid"], j["org"]): j for j in jobs}

    baseline_stars = {}
    for year in C.YEARS:
        stars, _, _ = load_inputs(year, C.BASELINE_VINTAGE[year])
        baseline_stars[year] = stars

    # ---- nominal synth rebuild + integrity vs the stored census -------------
    synth = build_synth(jobs, gaps, baseline_stars, guardrail_on=True)
    stored_counts = {(r["year"], r["mid"], r["org"]):
                     r["n_star_deltas_vs_published"]
                     for r in census["per_split"]}
    assert len(synth) == len(stored_counts), (len(synth), len(stored_counts))
    for r in synth:
        k = (r["year"], r["mid"], r["org"])
        assert len(r["deltas"]) == stored_counts[k], (k, len(r["deltas"]),
                                                      stored_counts[k])
    agg_rows = [dict(year=r["year"], mid=r["mid"], org=r["org"],
                     realization="criterion", deltas=r["deltas"])
                for r in synth]
    agg = C.enclosure_aggregate(agg_rows, ["criterion"])
    flips_nom = {(c["year"], c["contract_id"]): c for c in agg["contracts"]
                 if c["n_flip_realizations"] == 1}
    stored_flips = {(p["year"], p["contract_id"]): p
                    for p in census["flips_criterion"]}
    assert set(flips_nom) == set(stored_flips), \
        ("flip-set mismatch", sorted(set(flips_nom) ^ set(stored_flips)))
    print(f"integrity: {len(synth)} splits, {len(stored_flips)} flips "
          "reproduce the stored census exactly")

    delta_idx = defaultdict(dict)   # (year, cid) -> {mid: delta}
    touch = defaultdict(set)        # (year, cid) -> {(mid, org)}
    star_nom = {}                   # (year, mid, org) -> {cid: criterion star}
    for r in synth:
        k = (r["year"], r["mid"], r["org"])
        job = jmap[k]
        st_pub = baseline_stars[r["year"]]
        sn = {}
        for cid, s in job["pool"]:
            if st_pub.get(cid, {}).get(r["mid"]) is None:
                continue
            sn[cid] = stars_from_bd(F(s), r["bd"], job["higher"])
        star_nom[k] = sn
        for cid, sw, sd in r["deltas"]:
            delta_idx[(r["year"], cid)][r["mid"]] = sd - sw
            touch[(r["year"], cid)].add((r["mid"], r["org"]))
    rater = Rater(delta_idx)

    # ---- PART 1: near-base check on every applied boundary ------------------
    n_checked = 0
    flagged = []
    affected = {}                   # (year, cid) -> record (90-flip status change)
    new_flips_outside = {}          # (year, cid) -> record (not among the 90)
    nom_rating_cache = {}
    for r in synth:
        k = (r["year"], r["mid"], r["org"])
        job = jmap[k]
        if job["gr_note"] != "applied":
            continue
        vals = [F(v) for v in gaps[k]["dp_cutpoints"]]
        cap = F(job["cap"])
        u = F(1, 2 * 10 ** job["dd"])
        bd0 = r["bd"]
        u_int = u - u / 1000            # strictly interior test point
        for i, (v, p) in enumerate(zip(vals, job["prior"])):
            n_checked += 1
            p = F(p)
            moved_ends = []
            for lab, sgn in (("+", 1), ("-", -1)):
                fp = F(round_display(guardrail_boundary(v, p + sgn * u, cap)[0],
                                     job["dd"]))
                if fp != bd0[i]:
                    fi = F(round_display(
                        guardrail_boundary(v, p + sgn * u_int, cap)[0],
                        job["dd"]))
                    moved_ends.append((lab, fp, fi != bd0[i]))
            if not moved_ends:
                continue
            frec = dict(year=r["year"], mid=r["mid"], org=r["org"],
                        boundary_index=i, prior=str(p), dp_raw=str(v),
                        nominal_final=str(bd0[i]),
                        ends_moved=[dict(end=lab, perturbed_final=str(fp),
                                         interior_sensitive=inter)
                                    for lab, fp, inter in moved_ends],
                        n_affected_cells=0, n_flip_contracts_checked=0)
            for lab, fp, inter in moved_ends:
                bd_p = list(bd0)
                bd_p[i] = fp
                for cid, s in job["pool"]:
                    sn = star_nom[k].get(cid)
                    if sn is None:
                        continue
                    sp = stars_from_bd(F(s), bd_p, job["higher"])
                    if sp == sn:
                        continue
                    frec["n_affected_cells"] += 1
                    ck = (r["year"], cid)
                    info = rater.base(r["year"], cid)
                    if info is None:
                        continue
                    b = info[0]
                    if ck not in nom_rating_cache:
                        c_nom = rater.rating_with_override(r["year"], cid, {})
                        nom_rating_cache[ck] = c_nom
                        # consistency: nominal flip status must match the census
                        assert is_flip(b, c_nom) == (ck in stored_flips), \
                            ("nominal flip mismatch", ck)
                    c_nom = nom_rating_cache[ck]
                    c_p = rater.rating_with_override(r["year"], cid,
                                                     {r["mid"]: sp})
                    fl_nom, fl_p = is_flip(b, c_nom), is_flip(b, c_p)
                    if ck in stored_flips:
                        frec["n_flip_contracts_checked"] += 1
                    if fl_p == fl_nom:
                        continue
                    rec = dict(year=r["year"], contract_id=cid,
                               mid=r["mid"], org=r["org"], boundary_index=i,
                               end=lab, perturbed_final=str(fp),
                               interior_sensitive=inter,
                               star_nominal=sn, star_perturbed=sp,
                               baseline=str(b), rating_nominal=str(c_nom),
                               rating_perturbed=str(c_p))
                    if ck in stored_flips:
                        rec["direction"] = stored_flips[ck]["direction"]
                        rec["change"] = "flip-withdrawn"
                        affected.setdefault(ck, rec)
                    else:
                        rec["change"] = "new-flip-appears"
                        new_flips_outside.setdefault(ck, rec)
            flagged.append(frec)
    n_flagged_interior = sum(1 for f in flagged
                             if any(e["interior_sensitive"]
                                    for e in f["ends_moved"]))
    n_aff_interior = sum(1 for a in affected.values()
                         if a["interior_sensitive"])
    print(f"part1: checked={n_checked} flagged={len(flagged)} "
          f"(interior-sensitive {n_flagged_interior}) "
          f"flips_affected={len(affected)} "
          f"(interior-sensitive {n_aff_interior}) "
          f"new_flips_outside_90={len(new_flips_outside)}")

    # ---- PART 2a: how many of the 90 touch a clamped measure ----------------
    clamped_split = {}
    for r in synth:
        k = (r["year"], r["mid"], r["org"])
        job = jmap[k]
        if job["gr_note"] != "applied":
            clamped_split[k] = False
            continue
        vals = [F(v) for v in gaps[k]["dp_cutpoints"]]
        raw_disp = [F(round_display(v, job["dd"])) for v in vals]
        clamped_split[k] = any(b != rw for b, rw in zip(r["bd"], raw_disp))
    touch_clamped = []
    for (year, cid), p in sorted(stored_flips.items()):
        hit = sorted(f"{m}/{o}" for (m, o) in touch[(year, cid)]
                     if clamped_split[(year, m, o)])
        if hit:
            touch_clamped.append(dict(year=year, contract_id=cid,
                                      direction=p["direction"],
                                      clamped_measures=hit))
    n_clamped_splits = sum(1 for v in clamped_split.values() if v)
    print(f"part2a: clamped splits={n_clamped_splits}/{len(clamped_split)}; "
          f"flips touching >=1 clamped measure: {len(touch_clamped)}/"
          f"{len(stored_flips)}")

    # ---- PART 2b: no-guardrail counterfactual -------------------------------
    synth_ng = build_synth(jobs, gaps, baseline_stars, guardrail_on=False)
    agg_ng = C.enclosure_aggregate(
        [dict(year=r["year"], mid=r["mid"], org=r["org"],
              realization="criterion", deltas=r["deltas"]) for r in synth_ng],
        ["criterion"])
    flips_ng = {(c["year"], c["contract_id"]): c for c in agg_ng["contracts"]
                if c["n_flip_realizations"] == 1}
    ng_delta_idx = defaultdict(dict)
    for r in synth_ng:
        for cid, sw, sd in r["deltas"]:
            ng_delta_idx[(r["year"], cid)][r["mid"]] = sd - sw
    ng_rater = Rater(ng_delta_idx)
    ng_rater.ctx = rater.ctx            # share loaded inputs
    ng_rater.base_cache = rater.base_cache
    dirs = {(c["year"], c["contract_id"]): c["direction"]
            for c in json.load(open(os.path.join(
                BASE, "runs/criterion_census.json")))["flips_criterion"]}
    fail_list = []
    for (year, cid), p in sorted(stored_flips.items()):
        if (year, cid) in flips_ng:
            continue
        c_ng = ng_rater.rating_with_override(year, cid, {})
        fail_list.append(dict(year=year, contract_id=cid,
                              direction=p["direction"],
                              baseline=p["baseline"],
                              criterion_rating=p["criterion_rating"],
                              no_guardrail_rating=(str(c_ng)
                                                   if c_ng is not None
                                                   else None)))
    new_ng = [dict(year=y, contract_id=c,
                   direction=flips_ng[(y, c)]["direction"],
                   baseline=flips_ng[(y, c)]["baseline"],
                   no_guardrail_counterfactual=flips_ng[(y, c)]
                   ["ratings_by_realization"]["criterion"])
              for (y, c) in sorted(set(flips_ng) - set(stored_flips))]
    print(f"part2b: without guardrail {len(flips_ng)} flips; of the 90: "
          f"{len(fail_list)} fail, {len(stored_flips) - len(fail_list)} "
          f"survive; new flips outside the 90: {len(new_ng)}")
    for f in fail_list:
        print("  FAIL-WITHOUT-GUARDRAIL:", f["year"], f["contract_id"],
              f["direction"], f["baseline"], "->", f["no_guardrail_rating"])

    result = {
        "register": (
            "CRITERION GUARDRAIL CHECK: the bonus-status census (one "
            "operative vintage per star year, name+part guardrail priors, "
            "disaster hold-harmless, 123 computations; the stored flip set, "
            "with the 2026 D07 MA-PD computation held at published). PART 1 "
            "applies the guardrail_base_check method to the criterion "
            "boundaries: every guardrail-applied exact DP boundary is recomputed "
            "with its published prior perturbed to both closed ends of the "
            "half-display-unit interval (prior +/- 0.5*10^-dd, dd = the "
            "job's display digits); a boundary is FLAGGED if its displayed "
            "guardrailed value changes under either endpoint. Because a "
            "clamped boundary equals prior +/- cap, a half-unit prior shift "
            "moves it a half display unit and generically crosses a rounding "
            "edge, so most clamped boundaries flag by construction; the "
            "operative test is the second stage, which recomputes the "
            "affected contracts' criterion stars under each flagged "
            "single-boundary perturbation and re-rates them (nominal deltas "
            "elsewhere, stored aggregation kwargs) to test whether any of "
            "the stored flips changes flip status. Closed endpoints are "
            "conservative: at a grid-aligned clamped boundary the +half-unit "
            "endpoint lands EXACTLY on the display rounding tie, which is "
            "unattainable if CMS rounds the published prior half-up (the "
            "attainable interval is half-open), so each flag and each "
            "affected flip also carries interior_sensitive = whether the "
            "displayed move persists at a strictly interior perturbation "
            "(0.999 of a half unit) — endpoint-only flags are rounding-tie "
            "artifacts of the closed-interval convention. Perturbations are "
            "single-boundary (one prior at a time; joint perturbations out "
            "of scope, as in the stored check); the restricted-range cap is "
            "held at its point value (cap uncertainty from prior trimmed "
            "scores is out of scope here, as in the stored check). PART 2 "
            "gives the guardrail-dependence counts: (a) a flip TOUCHES a "
            "clamped measure iff its criterion star moved on >=1 split "
            "where the displayed guardrailed boundary differs from the "
            "displayed raw dp boundary; (b) the no-guardrail counterfactual "
            "re-runs the census with raw dp boundaries (display-rounded, "
            "identical aggregation) and lists the stored flips that no "
            "longer flip. Integrity: the rebuilt nominal synth reproduced "
            "every stored per_split delta count and the stored 90-flip set "
            "exactly before any boundary check."),
        "inputs": {
            "criterion_census": {"path": "runs/criterion_census.json",
                                 "sha256_16": sha16(CENSUS)},
            "dp_boundaries": {"path": GAPS.replace(BASE + "/", ""),
                              "sha256_16": sha16(GAPS)},
            "jobs": "enclosure_census_docsonly.build_jobs() (documented-rules basis)"},
        "integrity": {
            "per_split_counts_reproduced": len(synth),
            "stored_flip_set_reproduced": len(stored_flips)},
        "part1": {
            "n_checked": n_checked,
            "n_flagged": len(flagged),
            "n_flagged_interior_sensitive": n_flagged_interior,
            "n_flagged_boundaries_also_clamped": sum(
                1 for f in flagged
                if clamped_split[(f["year"], f["mid"], f["org"])]),
            "flagged": flagged,
            "n_flips_affected": len(affected),
            "n_flips_affected_interior_sensitive": n_aff_interior,
            "affected": sorted(affected.values(),
                               key=lambda x: (x["year"], x["contract_id"])),
            "n_new_flips_outside_stored": len(new_flips_outside),
            "new_flips_outside_stored": sorted(
                new_flips_outside.values(),
                key=lambda x: (x["year"], x["contract_id"]))},
        "part2": {
            "n_touch_clamped": len(touch_clamped),
            "n_flips": len(stored_flips),
            "n_clamped_splits": n_clamped_splits,
            "touch_clamped_list": touch_clamped,
            "n_fail_without_guardrail": len(fail_list),
            "fail_list": fail_list,
            # the decomposition by dollar-estimate class that the guardrail
            # caveat prints, emitted rather than derived.
            "priced_grain": (lambda cen: (lambda cls, DEM: (lambda tset, fset: dict(
                register=("classes per the census's qbp_priceable field (which "
                       "encodes the dual-baseline condition: the four "
                       "june17-overlay-undecidable rows are not in the dollar "
                       "estimate); priced = qbp_priceable minus the demoted "
                       "endpoint-conditional pair"),
                touch_priced=sum(1 for k in tset if cls.get(k) == "priced"),
                touch_priced_up=sum(1 for k in tset if cls.get(k) == "priced"
                                    and dirs.get(k) == "UP"),
                touch_priced_down=sum(1 for k in tset if cls.get(k) == "priced"
                                      and dirs.get(k) == "DOWN"),
                fails_by_class={c: sum(1 for k in fset if cls.get(k) == c)
                                for c in ("priced", "no_published_overall",
                                          "demoted", "pdp", "cost",
                                          "overlay_undecidable")},
                ))(
                {(x["year"], x["contract_id"]) for x in touch_clamped},
                {(x["year"], x["contract_id"]) for x in fail_list}))(
                {(p["year"], p["contract_id"]):
                 ("pdp" if not p["qbp_relevant"] else
                  "cost" if "Cost" in p.get("org_type", "") else
                  "no_published_overall" if not p.get("published_overall_exists")
                  else "demoted" if (p["year"], p["contract_id"]) in
                  {("2024", "H1304"), ("2025", "H5273")} else
                  "overlay_undecidable" if not p.get("qbp_priceable")
                  else "priced")
                 for p in cen["flips_criterion"]},
                {("2024", "H1304"), ("2025", "H5273")}))(
                json.load(open(os.path.join(BASE, "runs/criterion_census.json")))),
            "n_flips_without_guardrail_total": len(flips_ng),
            "n_new_flips_without_guardrail": len(new_ng),
            "new_flips_without_guardrail": new_ng},
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
        f.write("\n")
    print("->", OUT)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
