# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Adversarial search over alternative filled-in scores and score-list
configurations for the 34 bonus-status changes in the dollar estimate
(runs/criterion_census.json, qbp_priceable rows minus the endpoint-conditional
pair H1304-2024 / H5273-2025). Starts with H0524 (2024) and H0473 (2024) and
reports which changes survive every configuration tested.

Scope: configurations actually tested, never "all feasible reconstructions".
  * For every published clustering computation whose reconstructed score
    list carries synthetic added contracts (the filled-in layer of
    runs/membership_delta_sets_docsonly.json), it generates alternative add
    value multisets with the same count k and the same removals, requires
    both published outlier bounds reproduced exactly (fences_capped /
    QNTLDEF=5 in Fraction arithmetic, caps as the target carries, verified on
    the vintage base the filled-in layer was solved on), values on the
    measure's data grid inside the score range. Candidates: (i) every
    structural fill the constructive solver
    (membership_stacker_docsonly.solve_adds_k) emits when run without a
    cap (all quartile-pair candidates x all count windows), (ii) for each
    structure, free-region value variation of the adds that are not pinned
    at a quartile-critical order statistic: interval extremes, all-equal
    placements, placements at / just below / just above the score of every
    contract in the dollar estimate in that computation, placements
    adjacent to the canonical DP cut points, and a seeded random sample
    (seed 20260927); when the whole (structure x free-value) family is small
    it is enumerated exhaustively and flagged so. Coupled C/D families
    (complaints, members-leaving) are generated at the add-contract level
    (shared MA-PD adds carry one value on both sides; both bounds exact).
  * For each bounds-exact candidate: rebuild the computation's criterion-
    vintage score list (real kept scores + candidate adds - removals), trim
    exactly as exact_gaps_docsonly._run, exact DP optimum (weighted-
    distinct-value DP in Fractions, calibrated against every stored
    dp_cutpoints/dp_ssq row it touches), guardrail + display rounding as
    criterion_census, criterion star of every contract in the dollar
    estimate in the pool (higher_is_better + disaster hold-harmless as
    criterion_census).
  * Per bonus-status change: cells = every computation of that year in which
    the contract is scored; moved cells (criterion star != published) and
    unmoved cells in filled-in computations are both channels. Any candidate
    changing a cell's star -> the contract's overall is re-rated
    (aggregate.rate_contract path exactly as criterion_verify_suite, all
    other criterion cells applied) and the dual-baseline condition
    re-evaluated; joint combinations across independent computations are
    enumerated (cartesian over achieved star tuples, capped and logged).
  * Score-list configurations beyond the canonical delta set: R24 = star
    year 2024 with no removals (the canonical 2024 removals were a cost-axis
    choice; every 2024 miss re-solved add-only and assembled by
    assemble_year), ALTB25 = the stored ALT-B removal family for 2025 C04
    (removal value 55, the 8 stored ALT-B contracts) transported to the
    documented-rules basis (global safety re-verified on the documented-rules
    targets; adds re-solved/assembled). Both must pass the full outlier-bound
    battery (every evaluable target exact) before use; both get the same
    value-level search on their filled-in computations.
Not done (stated in the record): add counts k other than the delta set's;
removal families other than {canonical, R24, ALTB25}; structures outside the
solver's quartile-pair grid; CAHPS/improvement (held at published, as the
census). Exact arithmetic everywhere an outlier bound, an SSQ comparison, a
cut point side, or a rating threshold is decided; floats nowhere in a
decision.

Run:  cd src && python3 fill_search_adversarial.py [workers]
      then the independent cross-check of every ERASABLE row (appends the
      'crosscheck' block to the same record):
      python3 fill_search_adversarial.py verify [workers]

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.6, filled-in scores
    (125,063 alternative profiles, 58 sweeps; 20 of the 34 changes in the
    dollar estimate survive every configuration).
Requires: Python >= 3.10; numpy, scipy.
"""
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction as F

import criterion_verify_suite as V
import enclosure_census_docsonly as C
import exact_gaps_criterionvintage as _GCV     # patches G.SCORE_VINTAGE
import membership_stacker_docsonly as S
from aggregate import AggSpec, rate_contract, org_class_of, disaster_pct_of
from census_measures import disaster_higher_of
from exact_gaps_docsonly import frac_to_decimal_str
from falsify_groups import round_display
from guardrail import guardrail_boundary
from tukey import tukey_trim

G = _GCV.G
BASE = C.BASE
OUT = os.path.join(BASE, "runs", "fill_search_adversarial.json")
SEED = 20260927
BUDGET = 240            # bounds-exact candidates wanted per computation/group
STRUCT_CAP = 4000       # structural fills per computation (logged if hit)
EXH_CAP = 6000          # enumerate the whole family when it is this small
JOINT_CAP = 200000      # cartesian joint combinations per flip per basis
JOINT_CAND_CAP = 2500   # joint (C,D) candidates evaluated per coupled group (logged)
DEMOTED = {("2024", "H1304"), ("2025", "H5273")}
FIRST = [("2024", "H0524"), ("2024", "H0473")]
COUPLED = {"2024": [("C25/ALL", "D02/MA-PD"), ("C26/ALL", "D03/MA-PD")],
           "2025": [("C25/ALL", "D02/MA-PD"), ("C26/ALL", "D03/MA-PD")],
           "2026": [("C28/ALL", "D02/MA-PD"), ("C29/ALL", "D03/MA-PD")]}
ALTB_FILE = os.path.join(BASE, "runs", "membership_delta_sets_altb.json")
INPUTS = ["runs/criterion_census.json",
          "runs/exact_gaps_criterionvintage.json",
          "runs/membership_delta_sets_docsonly.json",
          "runs/membership_delta_sets_altb.json",
          "runs/criterion_dollar_estimate.json",
          "data/parsed/summary_2026_jul22_overall.csv"]


def sha16(rel):
    return hashlib.sha256(open(os.path.join(BASE, rel), "rb").read()).hexdigest()[:16]


def fs(v):
    return str(v) if not isinstance(v, F) or v.denominator != 1 else str(v.numerator)


# ----------------------------------------------------------- exact weighted DP
def wdp(values, k, cap=8):
    """Exact optimal k-partition of a multiset of Fractions (weighted distinct
    values). Returns (ssq, [list of interior boundary index lists into the
    DISTINCT sorted values (start index of blocks 2..k)], n_optima_capped).
    Tie-break of the first optimum = smallest start index at every level
    (dp_kmeans.dp_optimal_partition's convention)."""
    cnt = Counter(values)
    dv = sorted(cnt)
    m = len(dv)
    W = [0] * (m + 1)
    S1 = [F(0)] * (m + 1)
    S2 = [F(0)] * (m + 1)
    for i, v in enumerate(dv):
        n = cnt[v]
        W[i + 1] = W[i] + n
        S1[i + 1] = S1[i] + n * v
        S2[i + 1] = S2[i] + n * v * v

    def block(i, j):          # distinct indices i..j-1 (j exclusive)
        w = W[j] - W[i]
        s = S1[j] - S1[i]
        return (S2[j] - S2[i]) - s * s / w

    D = [[None] * (m + 1) for _ in range(k + 1)]
    ARGS = [[None] * (m + 1) for _ in range(k + 1)]
    D[0][0] = F(0)
    for c in range(1, k + 1):
        for j in range(c, m + 1):
            best, args = None, []
            for i in range(c - 1, j):
                prev = D[c - 1][i]
                if prev is None:
                    continue
                cand = prev + block(i, j)
                if best is None or cand < best:
                    best, args = cand, [i]
                elif cand == best:
                    args.append(i)
            D[c][j] = best
            ARGS[c][j] = args
    outs = []

    def rec(c, j, tail):
        if len(outs) >= cap:
            return
        if c == 0:
            if j == 0:
                outs.append(list(tail))
            return
        for i in ARGS[c][j]:
            rec(c - 1, i, [i] + tail)
    rec(k, m, [])
    # outs: each = [0?..] starts of blocks 1..k (first is 0); interior = [1:]
    res = [o[1:] for o in outs]
    return D[k][m], res, len(outs), dv


def dp_cutpoints(scores_str, cap_hi, higher):
    """exact_gaps_docsonly._run recipe: trim (cap_lo='0'), k=min(5,
    distinct), exact optimum, cut point readout. Returns dict."""
    mask, lo, hi = tukey_trim(scores_str, cap_lo="0", cap_hi=cap_hi)
    trimmed = [F(s) for s, mm in zip(scores_str, mask) if mm]
    k = min(5, len(set(trimmed)))
    out = dict(n_input=len(scores_str), n_trimmed=len(trimmed), k=k,
               trim_lo=fs(lo), trim_hi=fs(hi))
    if k < 5:
        out["degenerate"] = True
        return out
    ssq, opts, nopt, dv = wdp(trimmed, k)
    cps_all = []
    for interior in opts:
        if higher:
            cps = [dv[b] for b in interior]
        else:
            cps = [dv[b - 1] for b in interior]
        cps_all.append([fs(c) for c in cps])
    out.update(ssq=str(ssq), n_optima=nopt, cutpoints=cps_all[0],
               cutpoints_all_optima=cps_all)
    return out


def _dp_worker(task):
    key, base_scores, cap_hi, higher, adds_list = task
    res = []
    for adds in adds_list:
        scores = list(base_scores) + list(adds)
        r = dp_cutpoints(scores, cap_hi, higher)
        res.append((adds, r))
    return key, res


# ------------------------------------------------------------ world helpers
def final_bd(raw_cps, jobC):
    vals = [F(v) for v in raw_cps]
    if jobC["gr_note"] == "applied":
        cap = F(jobC["cap"])
        fin = [guardrail_boundary(v, F(p), cap)[0]
               for v, p in zip(vals, jobC["prior"])]
    else:
        fin = vals
    return [F(round_display(v, jobC["dd"])) for v in fin]


def cell_star(jobC, bd, cid, score_str):
    sd = V.star_at(F(score_str), bd, jobC["higher"])
    part = "C" if jobC["mid"].startswith("C") else "D"
    return disaster_higher_of(jobC["name"], part, cid, sd,
                              V._DIS[jobC["year"]], V._PRIORMAP[jobC["year"]])


def split_deltas(jobC, bd, baseline):
    st_pub = baseline[jobC["year"]]["stars"]
    out = []
    for cid, s in jobC["pool"]:
        sw = st_pub.get(cid, {}).get(jobC["mid"])
        if sw is None:
            continue
        sd = cell_star(jobC, bd, cid, s)
        if sd != sw:
            out.append((cid, int(sw), int(sd)))
    return out


class Rater:
    def __init__(self, baseline, census_rows):
        self.baseline = baseline
        self.rows = {(r["year"], r["contract_id"]): r for r in census_rows}
        self.spec = {y: AggSpec(y) for y in C.YEARS}

    def rate(self, year, cid, cell_stars):
        """cell_stars: {mid: criterion star} for EVERY moved cell of the
        contract (others at published). Returns (display Fraction, raw)."""
        bl = self.baseline[year]
        st = bl["stars"][cid]
        srow = bl["summary"][cid]
        oc = org_class_of(srow)
        hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
        kw = dict(gate_convention="raw",
                  disaster=disaster_pct_of(srow, year) >= 25,
                  dup_d=bl["dup_d"])
        st_cf = dict(st)
        for m, v in cell_stars.items():
            if m in st_cf:
                st_cf[m] = max(1, min(5, int(v)))
        out = rate_contract(self.spec[year], st_cf, oc, bl["cai"].get(cid, {}), **kw)
        return F(out[hh][0]), out[hh][1]

    def priced(self, year, cid, rating):
        """criterion_census dual-baseline condition on a counterfactual rating."""
        r = self.rows[(year, cid)]
        bonus_crit = rating >= 4
        bonus_pub = bool(r["bonus_published"])
        bonus_rer = bool(r["bonus_rerated_baseline"])
        return (bonus_crit != bonus_pub) and (bonus_crit != bonus_rer)


# --------------------------------------------------------- basis construction
def canonical_ds():
    return json.load(open(os.path.join(BASE, "runs",
                                       "membership_delta_sets_docsonly.json")))


def solve_addonly(t, removed, k_cap=S.K_CAP, max_w=8):
    vals = sorted(v for c, v in t["pairs"] if c not in removed)
    bc = S.BaseCounts(vals)
    for k in range(0, k_cap + 1):
        ws = S.solve_adds_k(bc, k, t["pub_lo"], t["pub_hi"], t["cap_lo"],
                            t["cap_hi"], t["reg"], t["score_hi"], max_witnesses=max_w)
        if ws:
            return {"k": k, "witnesses": [[str(v) for v in w] for w in ws]}
    return None


def build_R24(log):
    """2024 with no removals: every miss solved add-only, assembled by
    assemble_year (coupling, slots, battery)."""
    year = "2024"
    targets, meta = S.load_year(year)
    search = {}
    for t in targets:
        if t["status"] != "miss":
            continue
        ao = solve_addonly(t, set())
        search[S.target_key(t)] = {"res": {"add_only": ao, "with_rems": []}}
    res = S.assemble_year(year, targets, meta, search, log)
    ok = (res["exact_after"] == res["evaluable"] and not res["removals"]
          and not res["unsolved"])
    info = dict(basis="R24_NO_REMOVALS", year=year,
                add_only_k={f"{k[0]}/{k[1]}": (v["res"]["add_only"]["k"]
                                              if v["res"]["add_only"] else None)
                            for k, v in search.items()},
                n_adds=len(res["adds"]), n_removals=len(res["removals"]),
                battery_exact_after=res["exact_after"],
                battery_evaluable=res["evaluable"],
                unsolved=res["unsolved"], coupling=res["coupling"],
                accepted=ok)
    return res, info, targets


def build_ALTB(log):
    """ALT-B (C04-2025 removal value 55, the 8 stored ALT-B contracts)
    transported to the documented-rules basis."""
    year = "2025"
    altb = json.load(open(ALTB_FILE))
    stored8 = list(altb["alt"]["removed_contracts"])
    targets, meta = S.load_year(year)
    tmap = {S.target_key(t): t for t in targets}
    t04 = tmap[("C04", "ALL")]
    v55 = F(55)
    present = {c: (v == v55) for c, v in t04["pairs"] if c in stored8}
    all_present_55 = len(present) == 8 and all(present.values())
    safe, culprit = S.removal_is_globally_safe(targets, set(stored8))
    search = {}
    for t in targets:
        if t["status"] != "miss":
            continue
        ao = solve_addonly(t, set())
        wr = []
        if S.target_key(t) == ("C04", "ALL"):
            red = solve_addonly(t, set(stored8))
            if red:
                wr = [{"rems": ["55"] * 8, "k": red["k"],
                       "witnesses": red["witnesses"]}]
        search[S.target_key(t)] = {"res": {"add_only": ao, "with_rems": wr}}
    c04_addonly = search[("C04", "ALL")]["res"]["add_only"]
    used_stored = [False]
    orig_sel = S._select_removal_contracts

    def sel(t, targets_, rem_vals, already):
        if (S.target_key(t) == ("C04", "ALL") and all_present_55 and safe
                and Counter(rem_vals) == Counter([v55] * 8) and not already):
            used_stored[0] = True
            return list(stored8)
        return orig_sel(t, targets_, rem_vals, already)
    S._select_removal_contracts = sel
    try:
        res = S.assemble_year(year, targets, meta, search, log)
    finally:
        S._select_removal_contracts = orig_sel
    rem_ids = [r["contract_id"] for r in res["removals"]]
    ok = (res["exact_after"] == res["evaluable"] and not res["unsolved"]
          and len(rem_ids) == 8)
    info = dict(basis="ALTB25_TRANSPORTED", year=year,
                stored_altb_contracts=stored8,
                stored_altb_hypothesis=altb.get("hypothesis"),
                all_8_present_with_C04_eq_55_on_docsonly=all_present_55,
                globally_safe_on_docsonly_targets=bool(safe),
                global_safety_culprit=culprit,
                stored_8_used=used_stored[0], removals_assembled=rem_ids,
                c04_add_only_solution_k_le_28=(c04_addonly["k"]
                                               if c04_addonly else None),
                c04_k_after_altb_removal=(search[("C04", "ALL")]["res"]
                                          ["with_rems"][0]["k"]
                                          if search[("C04", "ALL")]["res"]
                                          ["with_rems"] else None),
                n_adds=len(res["adds"]), n_removals=len(res["removals"]),
                battery_exact_after=res["exact_after"],
                battery_evaluable=res["evaluable"],
                unsolved=res["unsolved"], coupling=res["coupling"],
                removal_notes=res["removal_notes"], accepted=ok)
    return res, info, targets


def ds_with_year(ds0, year, res):
    ds = json.loads(json.dumps(ds0))
    ds["years"][year] = {"adds": res["adds"], "removals": res["removals"],
                         "removal_notes": res.get("removal_notes", []),
                         "coupling": res.get("coupling", []),
                         "unsolved": res.get("unsolved", []),
                         "battery": res.get("battery", [])}
    return ds


def build_jobs_for(ds):
    """G (clustering input, criterion vintage) and C (pool/guardrail) jobs
    under a given delta set — runtime override of the loaders only."""
    gl, cl = G.load_delta_sets, C.load_delta_sets
    G.load_delta_sets = lambda: ds
    C.load_delta_sets = lambda: ds
    try:
        jobsG = G.build_jobs()
        jobsC, skips = C.build_jobs(cluster_vintage=V.CRIT_VINTAGE)
    finally:
        G.load_delta_sets, C.load_delta_sets = gl, cl
    return ({(j["year"], j["mid"], j["org"]): j for j in jobsG},
            {(j["year"], j["mid"], j["org"]): j for j in jobsC}, skips)


# ------------------------------------------------------- candidate generation
def crit_positions(n):
    if n % 4 == 0:
        j1, j3 = n // 4, 3 * n // 4
        return [j1, j1 + 1, j3, j3 + 1]
    return [math.ceil(n / 4), math.ceil(3 * n / 4)]


def free_regions(base_sorted, adds, reg, U):
    """For fill `adds` on base: per add value -> None (pinned at a
    quartile-critical order statistic) or (glo, ghi) grid interval it may
    roam without moving any critical order statistic."""
    M = sorted(base_sorted + list(adds))
    crit = sorted({M[p - 1] for p in crit_positions(len(M))})
    out = {}
    for v in set(adds):
        if v in crit:
            out[v] = None
            continue
        below = [c for c in crit if c < v]
        above = [c for c in crit if c > v]
        glo = (max(below) + reg) if below else F(0)
        # snap glo up to grid
        glo = F(math.ceil(glo / reg)) * reg
        ghi = (min(above) - reg) if above else U
        ghi = F(math.floor(ghi / reg)) * reg
        if ghi < glo:               # no room: treat as pinned
            out[v] = None
        else:
            out[v] = (glo, ghi)
    return out


def grid_points(glo, ghi, reg):
    n = int((ghi - glo) / reg) + 1
    return [glo + i * reg for i in range(n)]


def gen_singleton(t, base_sorted, k, canon_adds, targets_x, cut_targets, rng,
                  budget=BUDGET):
    """Returns (accepted: list of sorted tuples of Fractions (bounds-exact),
    info dict)."""
    bc = S.BaseCounts(base_sorted)
    pub_lo, pub_hi = t["pub_lo"], t["pub_hi"]
    cap_lo, cap_hi, reg, score_hi = t["cap_lo"], t["cap_hi"], t["reg"], t["score_hi"]
    U = score_hi if score_hi is not None else max(base_sorted + list(canon_adds))
    U = F(math.floor(U / reg)) * reg
    canon = tuple(sorted(canon_adds))
    info = dict(k=k, n_base=len(base_sorted), reg=str(reg), U=fs(U),
                cap_hi=(fs(cap_hi) if cap_hi is not None else None),
                pub=[pub_lo, pub_hi])
    if k == 0:
        info.update(n_structural=1, exhaustive=True, family_size=1,
                    n_generated=1, n_fence_exact=1, note="k=0: singleton")
        return [()], info
    struct = S.solve_adds_k(bc, k, pub_lo, pub_hi, cap_lo, cap_hi, reg,
                            score_hi, max_witnesses=STRUCT_CAP)
    struct = [tuple(sorted(w)) for w in struct]
    info["n_structural"] = len(struct)
    info["structural_cap_hit"] = len(struct) >= STRUCT_CAP
    info["canonical_in_structural"] = canon in struct
    if canon not in struct:
        struct.append(canon)
    seen = set()
    accepted = []
    n_rej = 0

    def check(adds):
        nonlocal n_rej
        key = tuple(sorted(adds))
        if key in seen:
            return
        seen.add(key)
        if len(key) != k or any(v < 0 or v > U or (v / reg).denominator != 1
                                for v in key):
            n_rej += 1
            return
        lo, hi = bc.fences_with(list(key), cap_lo, cap_hi)
        if not S.fences_match(lo, hi, pub_lo, pub_hi):
            n_rej += 1
            return
        flo, fhi = S.fences_capped(sorted(base_sorted + list(key)), cap_lo, cap_hi)
        assert (flo, fhi) == (lo, hi)
        accepted.append(key)

    # family size (structure x free values) and optional exhaustive enumeration
    fam = 0
    plans = []
    for W in struct:
        regs = free_regions(base_sorted, W, reg, U)
        cntW = Counter(W)
        groups = []     # (interval, count, grid)
        pinned = []
        for v, c in cntW.items():
            if regs[v] is None:
                pinned += [v] * c
            else:
                groups.append((regs[v], c))
        # merge groups sharing the same interval
        gm = defaultdict(int)
        for iv, c in groups:
            gm[iv] += c
        groups = [(iv, c, grid_points(iv[0], iv[1], reg)) for iv, c in gm.items()]
        size = 1
        for iv, c, grid in groups:
            size *= math.comb(len(grid) + c - 1, c)
        fam += size
        plans.append((W, pinned, groups))
    info["family_size_structure_x_free_values"] = fam
    exhaustive = fam <= EXH_CAP
    info["exhaustive"] = exhaustive
    if exhaustive:
        for W, pinned, groups in plans:
            pools = [list(itertools.combinations_with_replacement(grid, c))
                     for iv, c, grid in groups]
            for combo in itertools.product(*pools):
                adds = list(pinned)
                for part in combo:
                    adds += list(part)
                check(adds)
    else:
        tvals = set()
        for x in targets_x:
            for d in (-1, 0, 1):
                tvals.add(x + d * reg)
        for cpt in cut_targets:
            for d in (-1, 0, 1):
                tvals.add(cpt + d * reg)
        tvals = sorted(v for v in tvals if 0 <= v <= U)
        for W, pinned, groups in plans:
            check(W)
            if not groups:
                continue

            def place(assign):      # assign: list per group of list of values
                adds = list(pinned)
                for part in assign:
                    adds += part
                check(adds)
            # per-group canonical values
            cparts = []
            regsW = free_regions(base_sorted, W, reg, U)
            for iv, c, grid in groups:
                cparts.append([v for v in W if regsW[v] == iv])
            # extremes
            place([[grid[0]] * c for iv, c, grid in groups])
            place([[grid[-1]] * c for iv, c, grid in groups])
            for gi, (iv, c, grid) in enumerate(groups):
                for gv in (grid[0], grid[-1]):
                    a = [list(p) for p in cparts]
                    a[gi] = [gv] * c
                    place(a)
                # evenly spaced all-equal placements of this group
                npts = min(9, len(grid))
                for s in range(npts):
                    gv = grid[(s * (len(grid) - 1)) // max(1, npts - 1)] if npts > 1 else grid[0]
                    a = [list(p) for p in cparts]
                    a[gi] = [gv] * c
                    place(a)
                # split placements: half low / half high
                a = [list(p) for p in cparts]
                a[gi] = [grid[0]] * (c // 2) + [grid[-1]] * (c - c // 2)
                place(a)
            # targeted placements
            for tv in tvals:
                clamped = []
                for iv, c, grid in groups:
                    cv = min(max(tv, grid[0]), grid[-1])
                    clamped.append([cv] * c)
                place(clamped)
                for gi, (iv, c, grid) in enumerate(groups):
                    if grid[0] <= tv <= grid[-1]:
                        a = [list(p) for p in cparts]
                        a[gi] = [tv] * c
                        place(a)
                        for m_ in range(1, c):
                            a = [list(p) for p in cparts]
                            a[gi] = [tv] * m_ + cparts[gi][m_:]
                            place(a)
        # seeded random until budget
        attempts = 0
        while len(accepted) < budget and attempts < 60 * budget:
            W, pinned, groups = plans[rng.randrange(len(plans))]
            attempts += 1
            if not groups:
                continue
            adds = list(pinned)
            mode = rng.random()
            for iv, c, grid in groups:
                if mode < 0.5:
                    adds += [grid[rng.randrange(len(grid))] for _ in range(c)]
                elif mode < 0.8:      # clustered: one random value per group
                    gv = grid[rng.randrange(len(grid))]
                    adds += [gv] * c
                else:                 # two-point mixtures
                    g1 = grid[rng.randrange(len(grid))]
                    g2 = grid[rng.randrange(len(grid))]
                    m_ = rng.randrange(c + 1)
                    adds += [g1] * m_ + [g2] * (c - m_)
            check(adds)
        info["random_attempts"] = attempts
    info["n_generated"] = len(seen)
    info["n_fence_exact"] = len(accepted)
    info["n_rejected_not_fence_exact_or_off_grid"] = n_rej
    assert canon in accepted, "canonical fill must be bounds-exact"
    return accepted, info


def sub_multisets_of_size(ms, r, cap=400):
    """Distinct sub-multisets of size r of sorted tuple ms (capped)."""
    cnt = sorted(Counter(ms).items())
    out = []

    def rec(i, need, acc):
        if len(out) >= cap:
            return
        if need == 0:
            out.append(tuple(sorted(acc)))
            return
        if i == len(cnt):
            return
        v, c = cnt[i]
        rest = sum(cc for _, cc in cnt[i + 1:])
        for take in range(min(c, need), -1, -1):
            if need - take > rest:
                break
            rec(i + 1, need - take, acc + [v] * take)
    rec(0, r, [])
    return out


def gen_coupled(tC, baseC, kC, canonC, tD, baseD, kD, canonD, n_shared,
                conly_canon, targets_xC, targets_xD, cutC, cutD, rng,
                budget=BUDGET):
    """Joint (C multiset, D multiset) candidates: D = shared values, C =
    shared + C-only values; both outlier bounds exact."""
    accC, infoC = gen_singleton(tC, baseC, kC, canonC, targets_xC, cutC, rng, budget)
    accD, infoD = gen_singleton(tD, baseD, kD, canonD, targets_xD, cutD, rng, budget)
    setC, setD = set(accC), set(accD)
    bcC, bcD = S.BaseCounts(baseC), S.BaseCounts(baseD)

    def okC(ms):
        if ms in setC:
            return True
        lo, hi = bcC.fences_with(list(ms), tC["cap_lo"], tC["cap_hi"])
        if S.fences_match(lo, hi, tC["pub_lo"], tC["pub_hi"]):
            flo, fhi = S.fences_capped(sorted(baseC + list(ms)), tC["cap_lo"], tC["cap_hi"])
            assert (flo, fhi) == (lo, hi)
            setC.add(ms)
            return True
        return False

    def okD(ms):
        if ms in setD:
            return True
        lo, hi = bcD.fences_with(list(ms), tD["cap_lo"], tD["cap_hi"])
        if S.fences_match(lo, hi, tD["pub_lo"], tD["pub_hi"]):
            flo, fhi = S.fences_capped(sorted(baseD + list(ms)), tD["cap_lo"], tD["cap_hi"])
            assert (flo, fhi) == (lo, hi)
            setD.add(ms)
            return True
        return False

    UC = F(infoC["U"])
    regC = tC["reg"]
    joint = set()
    canon_joint = (tuple(sorted(canonC)), tuple(sorted(canonD)))
    # from C side: choose shared sub-multiset
    for ms in accC:
        for sub in sub_multisets_of_size(ms, n_shared, cap=24):
            if okD(sub):
                joint.add((ms, sub))
    # from D side: extend with C-only values
    conly_sets = [tuple(sorted(conly_canon))]
    nco = kC - n_shared
    if nco > 0:
        gridC = grid_points(F(0), UC, regC)
        conly_sets.append(tuple([gridC[0]] * nco))
        conly_sets.append(tuple([gridC[-1]] * nco))
        for _ in range(40):
            conly_sets.append(tuple(sorted(gridC[rng.randrange(len(gridC))]
                                           for _ in range(nco))))
        for x in targets_xC:
            for d in (-1, 0, 1):
                v = x + d * regC
                if 0 <= v <= UC:
                    conly_sets.append(tuple([v] * nco))
    for msD in accD:
        for co in conly_sets:
            msC = tuple(sorted(list(msD) + list(co)))
            if len(msC) == kC and all(0 <= v <= UC for v in msC) and okC(msC):
                joint.add((msC, msD))
    assert canon_joint in joint or (okC(canon_joint[0]) and okD(canon_joint[1]))
    joint.add(canon_joint)
    n_joint_all = len(joint)
    cap_applied = None
    if len(joint) > JOINT_CAND_CAP:
        # deterministic seeded subsample keeping the canonical pair and at
        # least one joint per distinct C multiset and per distinct D multiset
        keep = {canon_joint}
        byC, byD = {}, {}
        for pr in sorted(joint):
            byC.setdefault(pr[0], pr)
            byD.setdefault(pr[1], pr)
        keep |= set(byC.values()) | set(byD.values())
        rest = sorted(joint - keep)
        rng.shuffle(rest)
        need = max(0, JOINT_CAND_CAP - len(keep))
        keep |= set(rest[:need])
        cap_applied = dict(n_joint_generated=n_joint_all, n_kept=len(keep),
                           rule=("canonical + one per distinct C multiset + one per "
                                 "distinct D multiset + seeded random fill to cap"))
        joint = keep
    info = dict(C=infoC, D=infoD, n_shared_adds=n_shared, n_c_only_adds=nco,
                n_joint_fence_exact=len(joint), n_joint_generated_before_cap=n_joint_all,
                joint_cap_applied=cap_applied,
                exhaustive=False,
                note=("joint candidates = C-side candidates x shared "
                      "sub-multiset choices that keep D exact, plus D-side "
                      "candidates extended by C-only value sets that keep C "
                      "exact; not an exhaustive product"))
    return sorted(joint), info


# ------------------------------------------------------------------- main
def main(workers=4):
    rng = random.Random(SEED)
    logs = []
    log = lambda s: (logs.append(s), print(s, flush=True))
    census = json.load(open(os.path.join(BASE, "runs", "criterion_census.json")))
    rows36 = [r for r in census["flips_criterion"] if r["qbp_priceable"]]
    rows = ([r for k in FIRST for r in rows36 if (r["year"], r["contract_id"]) == k]
            + sorted((r for r in rows36 if (r["year"], r["contract_id"]) not in FIRST),
                     key=lambda r: (r["year"], r["contract_id"])))
    gaps, jobsC0, skips0, baseline = V.load_world()
    assert G.SCORE_VINTAGE == V.CRIT_VINTAGE, G.SCORE_VINTAGE
    assert S.SCORE_VINTAGE == C.FITVINTAGE, S.SCORE_VINTAGE
    rater = Rater(baseline, rows36)
    ds0 = canonical_ds()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(min(4, workers))

    # ---------------- bases
    bases = {"CANON": dict(ds=ds0, years=set(C.YEARS),
                           info=dict(basis="CANON", note="runs/membership_delta_sets_docsonly.json as stored"))}
    log("building R24 (2024 no-removal) basis ...")
    res24, info24, _ = build_R24(lambda s: log("  [R24] " + s))
    log(f"  R24 accepted={info24['accepted']} battery {info24['battery_exact_after']}/{info24['battery_evaluable']} adds={info24['n_adds']}")
    if info24["accepted"]:
        bases["R24"] = dict(ds=ds_with_year(ds0, "2024", res24), years={"2024"}, info=info24)
    log("building ALTB25 (2025 ALT-B transported) basis ...")
    res25, info25, _ = build_ALTB(lambda s: log("  [ALTB] " + s))
    log(f"  ALTB accepted={info25['accepted']} stored8_used={info25['stored_8_used']} battery {info25['battery_exact_after']}/{info25['battery_evaluable']}")
    if info25["accepted"]:
        bases["ALTB25"] = dict(ds=ds_with_year(ds0, "2025", res25), years={"2025"}, info=info25)
    rejected_bases = {k: v for k, v in (("R24", info24), ("ALTB25", info25)) if not v["accepted"]}

    # outlier-bound targets (fit vintage) per year
    stk = {y: {S.target_key(t): t for t in S.load_year(y)[0]} for y in C.YEARS}

    # ---------------- per basis: jobs, basis-canonical DP for every computation
    calib = dict(rows_checked=0, rows_matched=0, mismatches=[])
    basis_data = {}
    for bname, b in bases.items():
        jobsG, jobsC, skipsC = build_jobs_for(b["ds"])
        keys = [k for k in jobsC if k[0] in b["years"] and k in jobsG]
        missing_G = [k for k in jobsC if k[0] in b["years"] and k not in jobsG]
        tasks = []
        for key in keys:
            jg, jc = jobsG[key], jobsC[key]
            assert jg["higher"] == jc["higher"], key
            tasks.append((key, [s for _, s in jg["kept"]], jg["cap_hi"], jg["higher"], [()]))
        dpres = {}
        for key, res in pool.imap_unordered(_dp_worker, tasks, chunksize=2):
            dpres[key] = res[0][1]
        log(f"[{bname}] basis-canonical DP for {len(tasks)} computations")
        # calibration against stored gaps (CANON only: same inputs)
        if bname == "CANON":
            for key in keys:
                g = gaps.get(key)
                if g is None:
                    continue
                calib["rows_checked"] += 1
                r = dpres[key]
                ok = (r.get("n_trimmed") == g["n_trimmed"] and r["k"] == g["k"]
                      and ((r.get("degenerate") and len(g["dp_cutpoints"]) != 4)
                           or (not r.get("degenerate")
                               and [F(x) for x in r["cutpoints"]] == [F(x) for x in g["dp_cutpoints"]]
                               and F(r["ssq"]) == F(g["dp_ssq"])
                               and r["n_optima"] == g["dp_n_optima"])))
                if ok:
                    calib["rows_matched"] += 1
                else:
                    calib["mismatches"].append(dict(key=list(key), mine=r,
                                                    stored=dict(n_trimmed=g["n_trimmed"], k=g["k"],
                                                                dp_cutpoints=g["dp_cutpoints"], dp_ssq=g["dp_ssq"])))
        # delta sets under this basis (basis years recomputed; others canonical)
        dsets = {}
        canon_d = V.delta_sets(gaps, jobsC0, baseline, include_degenerate=False)
        for key, dl in canon_d.items():
            if key[0] not in b["years"]:
                dsets[key] = dl
        bd_basis = {}
        for key in keys:
            r = dpres[key]
            if r.get("degenerate"):
                continue
            bd = final_bd(r["cutpoints"], jobsC[key])
            bd_basis[key] = bd
            dsets[key] = split_deltas(jobsC[key], bd, baseline)
        if bname == "CANON":
            # must equal the census's own delta sets exactly
            same = all(sorted(dsets.get(k, [])) == sorted(canon_d.get(k, []))
                       for k in set(dsets) | set(canon_d))
            calib["canon_delta_sets_identical_to_census_machinery"] = same
        basis_data[bname] = dict(jobsG=jobsG, jobsC=jobsC, dpres=dpres, dsets=dsets,
                                 bd=bd_basis, missing_G=missing_G)
    calib["all_matched"] = calib["rows_checked"] == calib["rows_matched"] and calib["rows_checked"] > 0
    log(f"CALIBRATION: {calib['rows_matched']}/{calib['rows_checked']} stored DP rows reproduced; delta sets identical: {calib.get('canon_delta_sets_identical_to_census_machinery')}")
    assert calib["all_matched"], calib["mismatches"][:2]
    assert calib["canon_delta_sets_identical_to_census_machinery"]

    # canonical rating calibration for the 36 rows
    rate_cal = []
    for r in rows36:
        y, cid = r["year"], r["contract_id"]
        cells = {k[1]: sd for k, dl in basis_data["CANON"]["dsets"].items() if k[0] == y
                 for c, sw, sd in dl if c == cid}
        disp, raw = rater.rate(y, cid, cells)
        rate_cal.append(dict(year=y, contract_id=cid, census=r["criterion_rating"],
                             mine=str(disp), raw=str(raw), match=F(r["criterion_rating"]) == disp,
                             priced=rater.priced(y, cid, disp)))
    assert all(x["match"] and x["priced"] for x in rate_cal), [x for x in rate_cal if not (x["match"] and x["priced"])]
    log("rating calibration: 36/36 census criterion ratings reproduced, all in the estimate under the dual-baseline condition")

    # ---------------- candidate generation per basis x fitted group
    flip_keys_by_year = defaultdict(list)
    for r in rows36:
        flip_keys_by_year[r["year"]].append(r["contract_id"])
    group_results = {}      # (bname, group tuple of mkeys) -> dict
    for bname, b in bases.items():
        ds = b["ds"]
        BD = basis_data[bname]
        for year in sorted(b["years"]):
            d = ds["years"][year]
            rems = {x["contract_id"] for x in d["removals"]}
            addmap = defaultdict(list)      # mkey -> [(id, F)]
            for a in d["adds"]:
                for mk, v in a["measures"].items():
                    addmap[mk].append((a["id"], F(v)))
            coupled = []
            for cmk, dmk in COUPLED[year]:
                if cmk in addmap and dmk in addmap:
                    coupled.append((cmk, dmk))
            in_pair = {m for p in coupled for m in p}
            groups = [tuple(p) for p in coupled] + [(mk,) for mk in sorted(addmap) if mk not in in_pair]
            for grp in groups:
                comp = {}
                for mk in grp:
                    mid, org = mk.split("/")
                    key = (year, mid, org)
                    t = stk[year][(mid, org)]
                    base_fit = sorted(v for c, v in t["pairs"] if c not in rems)
                    canon = sorted(v for _, v in addmap[mk])
                    jg, jc = BD["jobsG"].get(key), BD["jobsC"].get(key)
                    if jg is None or jc is None:
                        comp[mk] = None
                        continue
                    base_crit = [s for c, s in jg["kept"] if not str(c).startswith("ADD-")]
                    n_add_in_kept = sum(1 for c, s in jg["kept"] if str(c).startswith("ADD-"))
                    assert n_add_in_kept == len(canon), (bname, key, n_add_in_kept, len(canon))
                    # scores of the dollar-estimate contracts in this pool + canonical cut points
                    tx = [F(s) for c, s in jc["pool"] if c in flip_keys_by_year[year]]
                    cuts = [F(x) for x in BD["dpres"][key].get("cutpoints", [])]
                    comp[mk] = dict(key=key, t=t, base_fit=base_fit, canon=canon, jg=jg, jc=jc,
                                    base_crit=base_crit, tx=tx, cuts=cuts, k=len(canon))
                if any(v is None for v in comp.values()):
                    group_results[(bname, year, grp)] = dict(skipped="computation not built (n<5 or no published rows)")
                    continue
                if len(grp) == 1:
                    mk = grp[0]
                    cm = comp[mk]
                    acc, info = gen_singleton(cm["t"], cm["base_fit"], cm["k"], cm["canon"],
                                              cm["tx"], cm["cuts"], rng)
                    cand = [{mk: a} for a in acc]
                    ginfo = {mk: info}
                    exhaustive = info["exhaustive"]
                else:
                    cmk, dmk = grp
                    cc, dd_ = comp[cmk], comp[dmk]
                    ids_c = {i for i, _ in addmap[cmk]}
                    ids_d = {i for i, _ in addmap[dmk]}
                    shared = ids_c & ids_d
                    vc = dict(addmap[cmk]); vd = dict(addmap[dmk])
                    assert all(vc[i] == vd[i] for i in shared), (bname, grp, "shared adds not value-identical")
                    assert not (ids_d - ids_c), (bname, grp, "D-only adds present")
                    conly = sorted(vc[i] for i in ids_c - shared)
                    joint, info = gen_coupled(cc["t"], cc["base_fit"], cc["k"], cc["canon"],
                                              dd_["t"], dd_["base_fit"], dd_["k"], dd_["canon"],
                                              len(shared), conly, cc["tx"], dd_["tx"],
                                              cc["cuts"], dd_["cuts"], rng)
                    cand = [{cmk: a, dmk: bb} for a, bb in joint]
                    ginfo = info
                    exhaustive = False
                joint_cap_applied = ginfo.get("joint_cap_applied") if len(grp) == 2 else None
                # evaluate DPs per computation (distinct multisets)
                evals = {}
                for mk in grp:
                    cm = comp[mk]
                    ms_list = sorted({c_[mk] for c_ in cand})
                    adds_str = [tuple(frac_to_decimal_str(v) for v in ms) for ms in ms_list]
                    chunks = [adds_str[i:i + 25] for i in range(0, len(adds_str), 25)]
                    tasks = [(cm["key"], cm["base_crit"], cm["jg"]["cap_hi"], cm["jg"]["higher"], ch)
                             for ch in chunks]
                    resmap = {}
                    for key, res in pool.imap_unordered(_dp_worker, tasks, chunksize=1):
                        for adds, r in res:
                            resmap[tuple(F(a) for a in adds)] = r
                    evals[mk] = resmap
                # canonical check: canonical multiset reproduces basis dp
                for mk in grp:
                    cm = comp[mk]
                    rc = evals[mk][tuple(cm["canon"])]
                    rb = BD["dpres"][cm["key"]]
                    assert rc.get("cutpoints") == rb.get("cutpoints") and rc.get("ssq") == rb.get("ssq"), (bname, grp, mk)
                # per candidate: final bd per computation, stars of flip contracts
                stars_by_cand = []
                n_tied = 0
                bd_variety = defaultdict(set)
                for c_ in cand:
                    entry = {}
                    for mk in grp:
                        cm = comp[mk]
                        r = evals[mk][c_[mk]]
                        jc = cm["jc"]
                        if r.get("degenerate"):
                            entry[mk] = dict(degenerate=True, stars={})
                            continue
                        if r["n_optima"] > 1:
                            n_tied += 1
                        # stars under every tied optimum (adversary may pick)
                        star_sets = defaultdict(set)
                        bds = []
                        for cps in r["cutpoints_all_optima"]:
                            bd = final_bd(cps, jc)
                            bds.append(tuple(fs(x) for x in bd))
                            for cid, s in jc["pool"]:
                                if cid in flip_keys_by_year[year] and baseline[year]["stars"].get(cid, {}).get(jc["mid"]) is not None:
                                    star_sets[cid].add(cell_star(jc, bd, cid, s))
                        bd_variety[mk].add(bds[0])
                        tlo, thi = F(r["trim_lo"]), F(r["trim_hi"])
                        n_in = sum(1 for v in c_[mk] if tlo <= v <= thi)
                        entry[mk] = dict(stars={cid: sorted(v) for cid, v in star_sets.items()},
                                         bd=bds[0], n_optima=r["n_optima"], raw=r["cutpoints"],
                                         trim=[r["trim_lo"], r["trim_hi"]], n_adds_inside_trim=n_in,
                                         n_trimmed=r["n_trimmed"])
                    stars_by_cand.append(entry)
                group_results[(bname, year, grp)] = dict(
                    basis=bname, year=year, group=list(grp), cand=cand, stars=stars_by_cand,
                    final_boundary_sets={mk: sorted(list(x) for x in v) for mk, v in bd_variety.items()},
                    info=ginfo, exhaustive=exhaustive, n_candidates=len(cand),
                    n_candidates_tied_optimum=n_tied,
                    n_distinct_final_boundary_sets={mk: len(v) for mk, v in bd_variety.items()},
                    joint_cap_applied=joint_cap_applied,
                    comp={mk: dict(key=list(comp[mk]["key"]), k=comp[mk]["k"],
                                   canonical_adds=[fs(v) for v in comp[mk]["canon"]],
                                   n_base_fit=len(comp[mk]["base_fit"]),
                                   n_base_crit=len(comp[mk]["base_crit"]))
                          for mk in grp})
                log(f"[{bname}] {year} {grp}: {len(cand)} bounds-exact candidates "
                    f"(exhaustive={exhaustive}) tied={n_tied} "
                    f"bd-sets={ {mk: len(v) for mk, v in bd_variety.items()} }")
    pool.close(); pool.join()

    # ---------------- per flip analysis
    per_flip = []
    for r in rows:
        y, cid = r["year"], r["contract_id"]
        of_record = (y, cid) not in DEMOTED
        st_pub = baseline[y]["stars"][cid]
        fl = dict(year=y, contract_id=cid, direction=r["direction"], of_record_34=of_record,
                  published_overall=r["published_overall"], baseline=r["baseline"],
                  criterion_rating=r["criterion_rating"], enrollment=r["enrollment"],
                  est_annual_qbp_revenue_delta_usd_mid=r["est_annual_qbp_revenue_delta_usd"]["mid"],
                  bases={})
        erasing = []
        any_star_change = False
        for bname, b in bases.items():
            if y not in b["years"]:
                continue
            BD = basis_data[bname]
            # all cells of this contract this year under the basis
            basis_cells = {}
            for key, jc in BD["jobsC"].items():
                if key[0] != y or key not in BD["bd"]:
                    continue
                sw = st_pub.get(jc["mid"])
                if sw is None:
                    continue
                sc = dict(jc["pool"]).get(cid)
                if sc is None:
                    continue
                sd = cell_star(jc, BD["bd"][key], cid, sc)
                basis_cells[key] = dict(mid=jc["mid"], org=jc["org"], name=jc["name"],
                                        score=sc, published_star=int(sw), basis_star=int(sd))
            moved = {k: v for k, v in basis_cells.items() if v["basis_star"] != v["published_star"]}
            cells_for_rating = {v["mid"]: v["basis_star"] for v in basis_cells.values()}
            disp, raw = rater.rate(y, cid, cells_for_rating)
            priced_basis = rater.priced(y, cid, disp)
            bres = dict(basis=bname, n_cells=len(basis_cells), n_moved_cells=len(moved),
                        moved_cells=[dict(comp=f"{v['mid']}/{v['org']}", name=v["name"], score=v["score"],
                                          published=v["published_star"], criterion=v["basis_star"])
                                     for v in moved.values()],
                        rating_under_basis=str(disp), raw_under_basis=str(raw),
                        raw_minus_3p75=str(raw - F(15, 4)),
                        flip_priced_under_basis=priced_basis)
            if not priced_basis:
                erasing.append(dict(basis=bname, kind="SCORE-LIST-CONFIGURATION",
                                    detail="the basis's canonical fill alone removes the flip from the estimate",
                                    rating=str(disp), raw=str(raw)))
            # filled-in groups touching this contract
            gres = []
            achievable = []     # per group: dict tuple(cells)->example candidate
            for (bn, gy, grp), GR in group_results.items():
                if bn != bname or gy != y or "cand" not in GR:
                    continue
                cellkeys = [(mk, tuple([y] + mk.split("/"))) for mk in grp]
                present = [(mk, k) for mk, k in cellkeys if k in basis_cells]
                if not present:
                    continue
                basis_tuple = tuple(basis_cells[k]["basis_star"] for mk, k in present)
                ach = {}
                n_change = 0
                n_tested = 0
                for c_, stv in zip(GR["cand"], GR["stars"]):
                    n_tested += 1
                    # adversary may pick among tied optima: enumerate star choices
                    opts = []
                    for mk, k in present:
                        e = stv[mk]
                        if e.get("degenerate"):
                            opts.append([basis_cells[k]["published_star"]])   # held at published
                        else:
                            opts.append(e["stars"].get(cid, [basis_cells[k]["basis_star"]]))
                    for tup in itertools.product(*opts):
                        tup = tuple(int(x) for x in tup)
                        if tup != basis_tuple:
                            n_change += 1
                        if tup not in ach:
                            ach[tup] = {mk: dict(adds=[fs(v) for v in c_[mk]],
                                                 canonical_adds=GR["comp"][mk]["canonical_adds"],
                                                 final_boundaries=list(stv[mk].get("bd", [])),
                                                 basis_final_boundaries=[fs(x) for x in BD["bd"].get(tuple([y] + mk.split("/")), [])],
                                                 raw_dp_cutpoints=stv[mk].get("raw"),
                                                 trim_fences=stv[mk].get("trim"),
                                                 n_adds_inside_trim_fences=stv[mk].get("n_adds_inside_trim"),
                                                 n_optima=stv[mk].get("n_optima"))
                                        for mk in grp}
                if basis_tuple not in ach:
                    ach[basis_tuple] = {mk: dict(adds=GR["comp"][mk]["canonical_adds"], canonical=True) for mk in grp}
                changed_tuples = [t for t in ach if t != basis_tuple]
                if changed_tuples:
                    any_star_change = True
                # single-group overrides
                single = []
                for tup in changed_tuples:
                    cf = dict(cells_for_rating)
                    for (mk, k), sv in zip(present, tup):
                        cf[basis_cells[k]["mid"]] = sv
                    d2, r2 = rater.rate(y, cid, cf)
                    pr = rater.priced(y, cid, d2)
                    single.append(dict(stars=list(tup), rating=str(d2), raw=str(r2), flip_priced=pr,
                                       witness=ach[tup]))
                    if not pr:
                        erasing.append(dict(basis=bname, kind="SINGLE-COMPUTATION-WITNESS",
                                            group=list(grp), stars=list(tup), rating=str(d2), raw=str(r2),
                                            witness=ach[tup]))
                gres.append(dict(group=list(grp),
                                 cells=[dict(comp=mk, score=basis_cells[k]["score"],
                                             published=basis_cells[k]["published_star"],
                                             basis_star=basis_cells[k]["basis_star"],
                                             moved=basis_cells[k]["basis_star"] != basis_cells[k]["published_star"])
                                        for mk, k in present],
                                 n_candidates_fence_exact_tested=GR["n_candidates"],
                                 exhaustive_family=GR["exhaustive"],
                                 n_candidate_star_readouts_differing=n_change,
                                 achievable_star_tuples=[list(t) for t in sorted(ach)],
                                 n_achievable_tuples_that_change_a_star=len(changed_tuples),
                                 single_group_overrides=single))
                achievable.append((grp, present, ach))
            # joint enumeration across groups (independent computations)
            sizes = [len(a[2]) for a in achievable]
            prod = 1
            for s_ in sizes:
                prod *= s_
            joint = dict(n_groups=len(achievable), product_size=prod, enumerated=None,
                         n_combos_erasing=0, example=None)
            if achievable and prod > 1:
                if prod <= JOINT_CAP:
                    joint["enumerated"] = "full"
                    n_er = 0
                    for combo in itertools.product(*[list(a[2].items()) for a in achievable]):
                        cf = dict(cells_for_rating)
                        for (grp, present, ach), (tup, wit) in zip(achievable, combo):
                            for (mk, k), sv in zip(present, tup):
                                cf[basis_cells[k]["mid"]] = sv
                        d2, r2 = rater.rate(y, cid, cf)
                        if not rater.priced(y, cid, d2):
                            n_er += 1
                            if joint["example"] is None:
                                joint["example"] = dict(rating=str(d2), raw=str(r2),
                                                        groups=[dict(group=list(grp), stars=list(tup), witness=wit)
                                                                for (grp, present, ach), (tup, wit) in zip(achievable, combo)
                                                                if tup != tuple(basis_cells[k]["basis_star"] for mk, k in present)])
                    joint["n_combos_erasing"] = n_er
                    if n_er:
                        erasing.append(dict(basis=bname, kind="JOINT-WITNESS", **joint["example"]))
                else:
                    # directional greedy: push every group toward the erasing direction
                    joint["enumerated"] = "greedy-directional (product over cap)"
                    want_up = not (F(r["criterion_rating"]) >= 4)   # DOWN flip: need rating up
                    cf = dict(cells_for_rating)
                    used = []
                    for grp, present, ach in achievable:
                        best = max(ach, key=lambda t: (sum(t) if want_up else -sum(t)))
                        for (mk, k), sv in zip(present, best):
                            cf[basis_cells[k]["mid"]] = sv
                        used.append(dict(group=list(grp), stars=list(best), witness=ach[best]))
                    d2, r2 = rater.rate(y, cid, cf)
                    if not rater.priced(y, cid, d2):
                        joint["n_combos_erasing"] = 1
                        joint["example"] = dict(rating=str(d2), raw=str(r2), groups=used)
                        erasing.append(dict(basis=bname, kind="JOINT-WITNESS(greedy)", **joint["example"]))
            bres["fitted_groups"] = gres
            bres["joint"] = joint
            bres["n_fitted_groups_touching_contract"] = len(gres)
            bres["n_moved_cells_in_fitted_computations"] = sum(
                1 for g_ in gres for c_ in g_["cells"] if c_["moved"])
            fl["bases"][bname] = bres
        canon_b = fl["bases"]["CANON"]
        cls = ("A: moved cell(s) in fitted computations (value search on moved + unmoved fitted cells)"
               if canon_b["n_moved_cells_in_fitted_computations"] > 0 else
               "B: no moved cell is fitted; unmoved cells in fitted computations value-searched"
               if canon_b["n_fitted_groups_touching_contract"] > 0 else
               "C: no fitted computation scores this contract")
        fl["class"] = cls
        fl["n_candidates_tested_total"] = sum(g_["n_candidates_fence_exact_tested"]
                                              for b_ in fl["bases"].values() for g_ in b_["fitted_groups"])
        fl["any_candidate_changes_a_measure_star"] = any_star_change
        fl["n_star_changing_configurations"] = sum(
            g_["n_achievable_tuples_that_change_a_star"] for b_ in fl["bases"].values()
            for g_ in b_["fitted_groups"])
        fl["erasing_witnesses"] = erasing
        fl["n_erasing"] = len(erasing)
        fl["verdict"] = "ERASABLE" if erasing else "SURVIVES-ALL-TESTED"
        fl["roster_bases_tested"] = sorted(fl["bases"])
        per_flip.append(fl)
        log(f"{y} {cid} {r['direction']}: {fl['verdict']}  class={cls[:1]} "
            f"cands={fl['n_candidates_tested_total']} star-changing={fl['n_star_changing_configurations']} "
            f"erasing={len(erasing)} bases={fl['roster_bases_tested']}")

    rec34 = [f for f in per_flip if f["of_record_34"]]
    surv34 = [f for f in rec34 if f["verdict"] == "SURVIVES-ALL-TESTED"]
    eras34 = [f for f in rec34 if f["verdict"] == "ERASABLE"]
    h0524 = next(f for f in per_flip if (f["year"], f["contract_id"]) == ("2024", "H0524"))
    h0473 = next(f for f in per_flip if (f["year"], f["contract_id"]) == ("2024", "H0473"))
    n_cands_total = sum(GR["n_candidates"] for GR in group_results.values() if "cand" in GR)
    group_table = []
    for (bn, gy, grp), GR in group_results.items():
        if "cand" not in GR:
            group_table.append(dict(basis=bn, group=list(grp), skipped=GR.get("skipped")))
            continue
        inf = GR["info"]
        group_table.append(dict(
            basis=bn, year=GR["year"], group=list(grp), n_candidates_fence_exact=GR["n_candidates"],
            exhaustive_family=GR["exhaustive"], n_candidates_tied_optimum=GR["n_candidates_tied_optimum"],
            n_distinct_final_boundary_sets=GR["n_distinct_final_boundary_sets"],
            final_boundary_sets=GR["final_boundary_sets"],
            joint_cap_applied=GR.get("joint_cap_applied"),
            comp=GR["comp"],
            generator_info=json.loads(json.dumps(inf, default=str))))
    enroll_surv = sum(f["enrollment"] for f in surv34)
    enroll_eras = sum(f["enrollment"] for f in eras34)
    summary = dict(
        n_valued_flips_of_record=len(rec34),
        n_survive_all_tested=len(surv34),
        n_erasable=len(eras34),
        erasable_rows=[dict(year=f["year"], contract_id=f["contract_id"], direction=f["direction"],
                            enrollment=f["enrollment"], kinds=sorted({e["kind"] for e in f["erasing_witnesses"]}),
                            bases=sorted({e["basis"] for e in f["erasing_witnesses"]}))
                       for f in eras34],
        survivors=[[f["year"], f["contract_id"], f["direction"]] for f in surv34],
        enrollment_surviving=enroll_surv, enrollment_erasable=enroll_eras,
        demoted_pair_results={f"{f['year']}/{f['contract_id']}": f["verdict"] for f in per_flip if not f["of_record_34"]},
        H0524_2024=dict(verdict=h0524["verdict"], **{f"{b}_raw_minus_3p75": h0524["bases"][b]["raw_minus_3p75"] for b in h0524["bases"]},
                        n_candidates=h0524["n_candidates_tested_total"], cls=h0524["class"],
                        n_star_changing=h0524["n_star_changing_configurations"],
                        moved_cells=h0524["bases"]["CANON"]["moved_cells"]),
        H0473_2024=dict(verdict=h0473["verdict"], **{f"{b}_raw_minus_3p75": h0473["bases"][b]["raw_minus_3p75"] for b in h0473["bases"]},
                        n_candidates=h0473["n_candidates_tested_total"], cls=h0473["class"],
                        n_star_changing=h0473["n_star_changing_configurations"],
                        moved_cells=h0473["bases"]["CANON"]["moved_cells"]),
        n_fence_exact_candidates_evaluated_total=n_cands_total,
        n_fitted_groups_swept=sum(1 for GR in group_results.values() if "cand" in GR),
        bases_tested=sorted(bases), bases_rejected=rejected_bases,
        class_counts=dict(Counter(f["class"][:1] for f in rec34)))
    out = {
        "register": (
            "Adversarial search over configurations actually tested, never 'all feasible "
            "reconstructions'. Tested: for every clustering computation carrying synthetic added "
            "contracts (filled-in layer), alternative add-value multisets with the delta set's own "
            "count k and removals, both published outlier bounds reproduced exactly (QNTLDEF=5, "
            "Fraction arithmetic, fit-vintage base), values on the data grid in [0, score range]; "
            "structures = the solver's full uncapped fill enumeration, values = free-region "
            "variation (extremes, all-equal, at/adjacent to the score of each contract in the dollar "
            "estimate, adjacent to canonical DP cut points, seeded random seed=20260927), whole family "
            "enumerated when small (flagged exhaustive per computation); coupled C/D families "
            "generated at the add-contract level. Each candidate -> criterion-vintage score list -> "
            "exact outlier trim -> exact DP optimum (weighted-distinct DP, calibrated against every "
            "stored dp row) -> guardrail/rounding -> measure star (higher_is_better + disaster "
            "hold-harmless) -> overall re-rated by aggregate.rate_contract with all other criterion "
            "cells applied -> dual-baseline condition. Cells searched: moved and unmoved cells of the "
            "contract in filled-in computations; joint combinations across independent computations "
            "enumerated (capped, logged). Score-list configurations: CANON (stored delta set), R24 "
            "(2024 without the cost-axis removals, adds re-solved and assembled), ALTB25 (stored "
            "ALT-B C04-2025 removal family transported to the documented-rules basis, global safety "
            "re-verified, adds re-solved); every basis passes the full outlier-bound battery before "
            "use. Not tested: add counts other than the delta set's minimal k; removal families "
            "beyond these three; fill structures outside the solver's quartile-pair grid; "
            "CAHPS/improvement measures (held at published, as the census); other contracts' flips "
            "gained/lost under a fill are not the object here."),
        "generator": "src/fill_search_adversarial.py",
        "input_pins_sha256_16": {p: sha16(p) for p in INPUTS},
        "seed": SEED, "budget_per_computation": BUDGET, "struct_cap": STRUCT_CAP,
        "exhaustive_cap": EXH_CAP, "joint_cap": JOINT_CAP,
        "calibration": dict(dp=calib, rating=dict(n=len(rate_cal), all_match=all(x["match"] for x in rate_cal),
                                                  rows=rate_cal)),
        "bases": {k: json.loads(json.dumps(v["info"], default=str)) for k, v in bases.items()},
        "bases_rejected": json.loads(json.dumps(rejected_bases, default=str)),
        "summary": summary,
        "per_flip": json.loads(json.dumps(per_flip, default=str)),
        "fitted_groups": group_table,
        "log": logs,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps(summary, indent=1, default=str))
    print("->", OUT)


def _pointwise_run(job):
    """exact_gaps_docsonly._run itself (pointwise exact DP + Ward),
    the independent cross-check engine."""
    return G._run(job)


def verify(workers=4):
    """Cross-check mode: re-derive every ERASABLE verdict's first erasing
    fill through the original engines, exact_gaps_docsonly._run
    (pointwise exact DP, dp_kmeans) on the fill computation(s) and
    criterion_verify_suite.delta_sets/apply_and_rate (the independent
    application loop) for the flip list, and append a 'crosscheck' block to
    the record. Also re-verifies the fill's outlier bounds with
    apply_delta_to_target."""
    rec = json.load(open(OUT))
    gaps, jobsC0, skips0, baseline = V.load_world()
    ds0 = canonical_ds()
    dss = {"CANON": ds0}
    if "R24" in rec["bases"]:
        res24, info24, _ = build_R24(lambda s: None)
        dss["R24"] = ds_with_year(ds0, "2024", res24)
    if "ALTB25" in rec["bases"]:
        res25, info25, _ = build_ALTB(lambda s: None)
        dss["ALTB25"] = ds_with_year(ds0, "2025", res25)
    stk = {y: {S.target_key(t): t for t in S.load_year(y)[0]} for y in C.YEARS}
    valued = {(f["year"], f["contract_id"]) for f in rec["per_flip"]}
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(min(4, workers))
    checks = []
    jobs_cache = {}
    for f in rec["per_flip"]:
        if not f["erasing_witnesses"]:
            continue
        y, cid = f["year"], f["contract_id"]
        w = f["erasing_witnesses"][0]
        bname = w["basis"]
        overrides = {}
        if w["kind"].startswith("SINGLE"):
            for mk, d in w["witness"].items():
                overrides[mk] = [F(v) for v in (d["adds"] if isinstance(d, dict) else d)]
        elif w["kind"].startswith("JOINT"):
            for g_ in w["groups"]:
                for mk, d in g_["witness"].items():
                    overrides[mk] = [F(v) for v in (d["adds"] if isinstance(d, dict) else d)]
        ds = json.loads(json.dumps(dss[bname]))
        d_y = ds["years"][y]
        # assign fill multisets to add ids (coupled: shared ids get D values)
        by_mk = defaultdict(list)
        for a in d_y["adds"]:
            for mk in a["measures"]:
                by_mk[mk].append(a)
        done = set()
        for cmk, dmk in COUPLED[y]:
            if cmk in overrides and dmk in overrides:
                Cv, Dv = Counter(overrides[cmk]), Counter(overrides[dmk])
                conly = sorted((Cv - Dv).elements())
                shared = [a for a in by_mk[dmk]]
                conly_ids = [a for a in by_mk[cmk] if dmk not in a["measures"]]
                assert len(shared) == sum(Dv.values()) and len(conly_ids) == len(conly), (cmk, dmk)
                for a, v in zip(shared, sorted(Dv.elements())):
                    a["measures"][dmk] = str(v)
                    a["measures"][cmk] = str(v)
                for a, v in zip(conly_ids, conly):
                    a["measures"][cmk] = str(v)
                done |= {cmk, dmk}
        for mk, vals in overrides.items():
            if mk in done:
                continue
            ids = by_mk[mk]
            assert len(ids) == len(vals), mk
            for a, v in zip(ids, sorted(vals)):
                a["measures"][mk] = str(v)
        # outlier bounds re-verified with the solver's own routine
        rems = {r["contract_id"] for r in d_y["removals"]}
        fence_ok = {}
        for mk in overrides:
            mid, org = mk.split("/")
            t = stk[y][(mid, org)]
            adds = [F(a["measures"][mk]) for a in d_y["adds"] if mk in a["measures"]]
            lo, hi = S.apply_delta_to_target(t, rems, adds)
            fence_ok[mk] = S.fences_match(lo, hi, t["pub_lo"], t["pub_hi"])
        jobsG, jobsC, _ = build_jobs_for(ds)
        touched = [(y,) + tuple(mk.split("/")) for mk in overrides]
        # pointwise stored engine on touched computations; calibrated wdp elsewhere
        results = pool.map(_pointwise_run, [jobsG[k] for k in touched])
        prow = {(r["year"], r["mid"], r["org"]): r for r in results}
        gaps_mod = {}
        for key, g in gaps.items():
            if key[0] != y:
                gaps_mod[key] = g
        for key, jg in jobsG.items():
            if key[0] != y:
                continue
            if key in prow:
                r = prow[key]
                gaps_mod[key] = dict(dp_cutpoints=r["dp_cutpoints"], dp_n_optima=r["dp_n_optima"],
                                     k=r["k"], n_trimmed=r["n_trimmed"])
            else:
                ck = (bname, key)
                if ck not in jobs_cache:
                    r = dp_cutpoints([s for _, s in jg["kept"]], jg["cap_hi"], jg["higher"])
                    jobs_cache[ck] = dict(dp_cutpoints=r.get("cutpoints", []), dp_n_optima=r.get("n_optima", 1),
                                          k=r["k"], n_trimmed=r["n_trimmed"])
                gaps_mod[key] = jobs_cache[ck]
        dsets = V.delta_sets(gaps_mod, list(jobsC.values()), baseline, include_degenerate=False)
        flips = {(r["year"], r["contract_id"]): r for r in V.apply_and_rate(baseline, dsets)}
        erased = (y, cid) not in flips
        # collateral among the dollar-estimate rows of the same year under this fill
        lost = sorted(k for k in valued if k[0] == y and k not in flips)
        row = flips.get((y, cid))
        checks.append(dict(year=y, contract_id=cid, basis=bname, witness_kind=w["kind"],
                           overrides={mk: [fs(v) for v in sorted(vals)] for mk, vals in overrides.items()},
                           fences_exact_stacker_recheck=fence_ok,
                           pointwise_dp_cutpoints={f"{k[1]}/{k[2]}": prow[k]["dp_cutpoints"] for k in touched},
                           pointwise_dp_n_optima={f"{k[1]}/{k[2]}": prow[k]["dp_n_optima"] for k in touched},
                           search_engine_raw_cutpoints={mk: (d.get("raw_dp_cutpoints") if isinstance(d, dict) else None)
                                                        for g_ in (w.get("groups") or [dict(witness=w.get("witness", {}))])
                                                        for mk, d in g_["witness"].items()},
                           flip_absent_under_stored_application_loop=erased,
                           rating_if_still_present=(row["criterion_rating"] if row else None),
                           valued_rows_of_this_year_absent_under_this_witness=[list(k) for k in lost],
                           confirmed=bool(erased and all(fence_ok.values()))))
        print(f"verify {y} {cid} [{bname} {w['kind']}]: erased={erased} fences={fence_ok} "
              f"collateral_absent={lost}", flush=True)
    pool.close(); pool.join()
    rec["crosscheck"] = dict(
        register=("Independent re-derivation of each ERASABLE row's first erasing fill: fill add values "
               "written into the delta set by add id, outlier bounds re-verified with "
               "membership_stacker_docsonly.apply_delta_to_target, the fill computation(s) re-clustered by "
               "exact_gaps_docsonly._run (pointwise exact DP of dp_kmeans), all other computations by "
               "the calibrated weighted DP, and the flip list re-derived by criterion_verify_suite.delta_sets "
               "+ apply_and_rate (the independent application loop). confirmed = flip absent and bounds exact."),
        generator="src/fill_search_adversarial.py verify",
        n_checked=len(checks), n_confirmed=sum(1 for c in checks if c["confirmed"]),
        rows=checks)
    rec["summary"]["crosscheck_n_checked"] = len(checks)
    rec["summary"]["crosscheck_n_confirmed"] = sum(1 for c in checks if c["confirmed"])
    # ---- summary enrichment (all derived from this record + the stored delta set)
    VALUE = {"lo": 400, "mid": 500, "hi": 600}
    rec34 = [f for f in rec["per_flip"] if f["of_record_34"]]
    surv = [f for f in rec34 if f["verdict"] == "SURVIVES-ALL-TESTED"]
    eras = [f for f in rec34 if f["verdict"] == "ERASABLE"]
    e_s, e_e = sum(f["enrollment"] for f in surv), sum(f["enrollment"] for f in eras)
    rec["summary"]["dollar_bands_usd_3yr_sum"] = dict(
        note=("enrollment x $400/500/600 per enrollee-year, same estimate class as "
              "criterion_dollar_estimate.json (gross, both directions, 3 payment years summed)"),
        surviving={k: e_s * v for k, v in VALUE.items()},
        erasable={k: e_e * v for k, v in VALUE.items()},
        all_34={k: (e_s + e_e) * v for k, v in VALUE.items()},
        surviving_share_of_enrollment=str(F(e_s, e_s + e_e)),
        surviving_share_of_enrollment_float=round(e_s / (e_s + e_e), 4))
    rec["summary"]["survivors_by_direction"] = dict(Counter(f["direction"] for f in surv))
    rec["summary"]["erasable_by_direction"] = dict(Counter(f["direction"] for f in eras))
    rec["summary"]["erasable_by_year"] = dict(Counter(f["year"] for f in eras))
    rec["summary"]["survivors_by_year"] = dict(Counter(f["year"] for f in surv))
    ra = {}
    for b in ("R24", "ALTB25"):
        rows_b = [f for f in rec["per_flip"] if b in f["bases"]]
        if rows_b:
            ra[b] = dict(n_valued_rows_of_that_year=len(rows_b),
                        n_of_record=sum(1 for f in rows_b if f["of_record_34"]),
                        n_still_priced_under_roster_configuration_alone=sum(
                            1 for f in rows_b if f["bases"][b]["flip_priced_under_basis"]),
                        rows_unpriced_by_roster_alone=[f["contract_id"] for f in rows_b
                                                       if not f["bases"][b]["flip_priced_under_basis"]])
    rec["summary"]["roster_configuration_alone"] = ra
    bat = {}
    for y in C.YEARS:
        for b in ds0["years"][y]["battery"]:
            bat[(y, b["mid"], b["org"])] = b.get("before_exact")
    for tag, key in (("H0524_2024", ("2024", "H0524")), ("H0473_2024", ("2024", "H0473"))):
        f = next(x for x in rec["per_flip"] if (x["year"], x["contract_id"]) == key)
        cb = f["bases"]["CANON"]
        rec["summary"][tag]["moved_cells_all_in_parameter_free_exact_computations"] = all(
            bat.get((key[0],) + tuple(m["comp"].split("/"))) for m in cb["moved_cells"])
        rec["summary"][tag]["moved_cells_parameter_free_flags"] = {
            m["comp"]: bat.get((key[0],) + tuple(m["comp"].split("/"))) for m in cb["moved_cells"]}
        rec["summary"][tag]["fitted_computations_scoring_contract_searched"] = {
            bn: [dict(group=g["group"], n_candidates=g["n_candidates_fence_exact_tested"],
                      cells=[f"{c['comp']}:{c['published']}->{c['basis_star']}" for c in g["cells"]],
                      n_star_changing=g["n_achievable_tuples_that_change_a_star"])
                 for g in f["bases"][bn]["fitted_groups"]] for bn in f["bases"]}
        rec["summary"][tag]["rating_under_each_basis"] = {bn: f["bases"][bn]["rating_under_basis"]
                                                          for bn in f["bases"]}
        rec["summary"][tag]["raw_minus_3p75_float"] = {bn: float(F(f["bases"][bn]["raw_minus_3p75"]))
                                                        for bn in f["bases"]}
    mech = []
    for f in rec["per_flip"]:
        if not f["erasing_witnesses"]:
            continue
        w = f["erasing_witnesses"][0]
        groups = w.get("groups") or [dict(group=w.get("group"), stars=w.get("stars"), witness=w.get("witness", {}))]
        comps = []
        for g_ in groups:
            for mk, dd in g_["witness"].items():
                comps.append(dict(comp=mk, canonical_adds=dict(Counter(dd.get("canonical_adds", []))),
                                  witness_adds=dict(Counter(dd["adds"])),
                                  n_witness_adds_inside_trim_fences=dd.get("n_adds_inside_trim_fences"),
                                  final_cutpoints_canonical=dd.get("basis_final_boundaries"),
                                  final_cutpoints_witness=dd.get("final_boundaries")))
        cb = f["bases"][w["basis"]]
        mech.append(dict(year=f["year"], contract_id=f["contract_id"], direction=f["direction"],
                         of_record_34=f["of_record_34"], enrollment=f["enrollment"],
                         basis=w["basis"], kind=w["kind"],
                         rating_canonical=cb["rating_under_basis"], raw_minus_3p75_canonical=cb["raw_minus_3p75"],
                         rating_witness=w.get("rating"), raw_witness=w.get("raw"),
                         cell_stars_witness={",".join(g_["group"]): g_["stars"] for g_ in groups},
                         computations=comps,
                         n_erasing_configurations_found=f["n_erasing"],
                         erasing_kinds=sorted({e["kind"] for e in f["erasing_witnesses"]}),
                         erasing_bases=sorted({e["basis"] for e in f["erasing_witnesses"]})))
    rec["summary"]["erasing_mechanisms_first_witness"] = mech
    rec["summary"]["erasing_computations_involved"] = dict(Counter(
        f"{m['year']} {c['comp']}" for m in mech if m["of_record_34"] for c in m["computations"]))
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
        fh.write("\n")
    print(f"crosscheck: {rec['summary']['crosscheck_n_confirmed']}/{len(checks)} confirmed -> {OUT}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "verify":
        verify(workers=min(4, int(args[1]) if len(args) > 1 else 4))
    else:
        w = int(args[0]) if args else 4
        main(workers=min(4, w))
