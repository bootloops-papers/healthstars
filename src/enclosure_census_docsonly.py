# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Flip census over resampling-group assignment and input order.

The single-realization flip census used one resampling-group
realization per run. PROC SURVEYSELECT GROUPS=10's assignment algorithm is
undocumented, so here the group assignment is varied within the documented
family ("10 equal-as-possible groups, random assignment"), and, because the
documented PROC CLUSTER tie rule and the group assignment both consume row
order (ward_tie display outputs move by one or two display units across input
orderings on each of five sampled measures), the input-order convention is a
second axis. K = 24 sampled realizations:

  16 on cid_asc order (the baseline dataset-order reading):
     the 11 structurally distinct implemented candidates in
     groups_assign.CANDIDATES (seed 8675309 where deterministic) plus 5
     uniform-random equal-split assignments (random.Random(1000+i) shuffles
     of the size-quota block labels), and
   8 ordering variants: {seq_quota_last, sort_blocks_first} x {cid_desc,
     score_asc (score ascending, contract_id tiebreak), score_desc} = 6,
     plus 2 uniform-random draws (random.Random(1000+5), Random(1000+6)) on
     score_asc order. Orderings via order_grid_screen.order_pairs().

The same 24 realizations (algorithm+seed+ordering, index-aligned) are used for
every measure-split. A 4.0-line flip holds across the sampled realizations only
if it holds for every one of the 24.

Scope: a range over 24 sampled realizations of the documented randomization
family x input-order conventions, not a proof over all realizations of
GROUPS=. A flip that survives all 24 is robust to group assignment and ordering
within the sample, but remains falsifiable by a 25th realization.

Vintage split:
  * Clustering inputs (scores fed to outlier trimming -> resampling groups ->
    Ward/DP) use the K-table basis vintages {"2024": "original", "2025":
    "original", "2026": "current"}: the clustering replays CMS's original
    computation (the K-5/K-6 outlier-bound tables were produced from the
    original run's data). This differs from census_measures.py, which
    clustered the 2024 recalc scores.
  * The published stars/ratings baseline consumed by the aggregation stays
    the operative published vintages {"2024": "recalc", "2025": "current",
    "2026": "current"} (what bonus payments were paid on). Deltas (dp_star -
    ward_star), computed on the replay side, are applied to the operative
    baseline by (contract, measure).
  * Guardrail bases: published prior-year final boundaries at display
    precision, identical treatment both arms, exactly as in census_measures.py
    (prior_final_boundaries / prior_trimmed_scores / NEW_BY_YEAR exemptions).

Membership: stage-1 reconciled (runs/membership_delta_sets_docsonly.json,
hypothesis cost+d60r): hypothesis-filtered clustering input, removal contracts
dropped, synthetic adds appended (Fraction-string scores converted exactly to
decimal strings). Star-delta assignment pool = every real contract with a
numeric published score for the split (pre-hypothesis); CMS assigns stars to
all scored contracts from cut points computed on the clustering input.

Arms: ward_tie (SAS-documented tie-rule Ward, IEEE double; the published-method
replay) vs dp (exact-rational SSQ optimum). Identical resampling groups,
outlier trimming, guardrails and display rounding in both arms: the delta
isolates the clustering step.

Aggregation: deltas across measures are correlated through the shared
realization, so the aggregation is per realization: for each realization r
the contract's full counterfactual rating is computed with all measures'
delta_r applied simultaneously (aggregate.rate_contract, same kwargs as
the single-realization census). Counterfactual rating set = {rating_r : r in realizations}.
Per-measure delta intervals [min_r, max_r] are reported as diagnostics only;
classification never enumerates measure combinations that no realization
produces.

Error policy: any (measure-split, realization) error is recorded and reported
(D07 has fewer than five distinct scores: resampling-group cut-point counts
differ). A split that errors in any realization is excluded from delta
application in all realizations (keeps the aggregation passes comparable;
held at published stars, like CAHPS/improvement measures).

Run:  python3 enclosure_census_docsonly.py pilot    [workers]  # realization 0 end-to-end
      python3 enclosure_census_docsonly.py full     [workers]  # the 16 cid_asc realizations
      python3 enclosure_census_docsonly.py extend24 [workers]  # +8 ordering realizations:
        reuses the stored 16-realization measure results (deltas reconstructed
        exactly from saved display-rounded finals x assignment pool, checked
        against saved n_star_changes), runs only the 8 new realizations,
        classifies over all 24, and reports the K=16 vs K=24 difference.
Output: runs/enclosure_census_docsonly.json (K=24 after extend24; the K=16
result is preserved as runs/enclosure_census_k16_docsonly.json) /
runs/enclosure_census_pilot_docsonly.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (the flip census over resampling-group assignment and input order under
    the assumed readout, documented-rules basis).
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction as F

import groups_assign
from aggregate import (AggSpec, rate_contract, load_inputs, org_class_of,
                       disaster_pct_of, dup_d_ids)
from census_measures import (CAHPS_NAMES, IMPROVEMENT_NAMES,
                             RESTRICTED_RANGE_NAMES, NEW_BY_YEAR,
                             measure_registry, prior_final_boundaries,
                             prior_trimmed_scores)
from exact_gaps_docsonly import frac_to_decimal_str, load_delta_sets
from falsify_groups import published_boundaries, round_display
from guardrail import guardrail_boundary, restricted_range_cap
from order_grid_screen import order_pairs
from pipeline import measure_cutpoints
from tukey_battery import scores_by_measure, load_summary_meta, apply_hypothesis

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")

HYP = "cost+d60r"                          # documented-rules-only basis (employer axis dropped)
CLUSTER_VINTAGE = {"2024": "original", "2025": "original", "2026": "current"}
BASELINE_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}
# the vintages the membership delta sets were determined on (fixed)
FITVINTAGE = {"2024": "original", "2025": "original", "2026": "current"}
YEARS = ("2024", "2025", "2026")
SEED = 8675309
N_GROUPS = 10

# ---------------------------------------------------------------- realizations
# 11 implemented candidates, in groups_assign definition order (index-aligned
# across every measure-split), + 5 uniform-random equal-split draws.
_LEGACY_11 = ["seq_quota_last", "seq_quota_first", "sort_blocks_last",
              "sort_blocks_first", "sort_deal", "floyd_last", "floyd_first",
              "fy_down_blocks", "fy_up_blocks", "fy_down_deal", "fy_up_deal"]
_CAND_NAMES = [c for c in groups_assign.CANDIDATES if c in _LEGACY_11]
# (listed explicitly: groups_assign also carries the assumed "frontswap"
#  readout; the stored censuses' 11-candidate family must stay index-aligned)
assert len(_CAND_NAMES) == 11, _CAND_NAMES


def _urand_factory(i):
    """Pure-random equal-as-possible split: shuffle the size-quota block labels
    with random.Random(1000+i). Deterministic given (i, n); the SURVEYSELECT
    seed argument is ignored (this realization samples the documented family,
    not a specific SAS stream)."""
    def f(n, g, seed):
        rng = random.Random(1000 + i)
        sizes = groups_assign.group_sizes(n, g, "last")
        ids = [gid for gid, sz in enumerate(sizes, 1) for _ in range(sz)]
        rng.shuffle(ids)
        return ids
    return f


for _i in range(7):
    # runtime-only registration (groups_assign.py itself is not modified);
    # urand_0..4 serve the cid_asc block, urand_5/6 the score_asc ordering block
    groups_assign.CANDIDATES[f"urand_{_i}"] = _urand_factory(_i)

# realization name encoding: "<candidate>" = cid_asc order; "<candidate>@<order>"
# applies order_grid_screen.order_pairs(<order>) to the reconciled input first.
REALIZATIONS16 = _CAND_NAMES + [f"urand_{i}" for i in range(5)]   # cid_asc block
ORDER_EXT = ([f"{c}@{o}" for c in ("seq_quota_last", "sort_blocks_first")
              for o in ("cid_desc", "score_asc", "score_desc")]
             + ["urand_5@score_asc", "urand_6@score_asc"])        # ordering block
REALIZATIONS24 = REALIZATIONS16 + ORDER_EXT                       # K = 24


def _split_rname(rname):
    if "@" in rname:
        cand, order = rname.split("@", 1)
    else:
        cand, order = rname, "cid_asc"
    return cand, order


# ------------------------------------------------------------------- job build
def _pub_rows(year, vintage, mid, oc):
    """Published cut-point rows for one measure(-org split): {star: (lo, lo_incl,
    hi, hi_incl)}. oc=None -> rows whose org is blank/ALL."""
    import csv
    path = os.path.join(PARSED, f"cutpoints_{year}_{vintage}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["measure_id"] != mid:
                continue
            row_oc = r.get("org_type", "").strip() or "ALL"
            if (oc is None and row_oc == "ALL") or (oc is not None and row_oc == oc):
                out[int(r["star_level"])] = (r["lo"], r["lo_incl"], r["hi"], r["hi_incl"])
    return out


def build_jobs(cluster_vintage=None, baseline_vintage=None):
    """Static per-(year, measure, split) data; realizations are crossed in later.
    Returns (jobs, skips). Everything a worker needs is embedded (no file I/O
    in workers).

    Vintages are parameters. The membership layer (removals/adds) was
    determined at the replay vintages {2024 original / 2025 original / 2026
    current}; the assert checks that, not the job vintage: membership is
    contract-level and carries across score vintages (stated in every
    consuming record)."""
    ds = load_delta_sets()
    if ds is None:
        raise RuntimeError("runs/membership_delta_sets_docsonly.json missing; stage-1 "
                           "reconciled membership is required for this census")
    assert ds["hypothesis"] == HYP, ds["hypothesis"]
    assert ds["score_vintage"] == FITVINTAGE, ds["score_vintage"]
    cv = cluster_vintage or CLUSTER_VINTAGE
    bvv = baseline_vintage or BASELINE_VINTAGE
    jobs, skips = [], []
    for year in YEARS:
        kv = cv[year]
        bv = bvv[year]
        reg = measure_registry(year, bv)      # operative registry (names, splits)
        sc = scores_by_measure(year, kv)      # clustering-input vintage scores
        meta = load_summary_meta(year, kv)
        disaster_active = year in ("2024", "2025")
        d = ds["years"][year]
        rems = {r["contract_id"] for r in d["removals"]}
        for mid in sorted(reg):
            name = reg[mid]["name"]
            if name in CAHPS_NAMES or name in IMPROVEMENT_NAMES:
                continue
            pairs = sc.get(mid)
            # an undocumented n<30 filter would drop six published PDP
            # computations (n = 19-25); the guard is k-viability (n >= 5).
            if not pairs or len(pairs) < 5:
                skips.append((year, mid, "ALL", "n<5 in cluster vintage"))
                continue
            pairs = sorted(pairs)             # dataset order: contract_id ascending
            splits = ([("ALL", None)] if reg[mid]["org_types"] == {"ALL"}
                      else [("MA-PD", "MA-PD"), ("PDP", "PDP")])
            for label, oc in splits:
                if oc is None:
                    sub = pairs
                else:
                    sub = [(c, s) for c, s in pairs
                           if ("PDP" in meta.get(c, ("",))[0]) == (oc == "PDP")]
                if len(sub) < 5:
                    skips.append((year, mid, label, "split n<5"))
                    continue
                # published boundaries: cluster vintage primary (replay target),
                # baseline vintage fallback (direction/decimals identical)
                pub = _pub_rows(year, kv, mid, oc) or _pub_rows(year, bv, mid, oc)
                if len(pub) < 5:
                    skips.append((year, mid, label, "no 5-row published cutpoints"))
                    continue
                target, higher = published_boundaries(pub)
                if target is None:
                    skips.append((year, mid, label, "unparsable boundaries"))
                    continue
                dd = max(len(v.split(".")[1]) if "." in v else 0 for v in target)
                # stage-1 reconciled clustering input
                kept = apply_hypothesis(sub, meta, HYP, disaster_active)
                kept = [(c, s) for c, s in kept if c not in rems]
                mkey = f"{mid}/{label}"
                for a in d["adds"]:
                    if mkey in a["measures"]:
                        kept.append((a["id"], frac_to_decimal_str(a["measures"][mkey])))
                if len(kept) < 5:
                    skips.append((year, mid, label, "reconciled n<5"))
                    continue
                restricted = name in RESTRICTED_RANGE_NAMES
                cap_lo, cap_hi = ("0", None) if restricted else ("0", "100")
                # guardrail pinning (identical to census_measures.run_measure)
                exempt = name in NEW_BY_YEAR.get(year, set())
                gr_note = "exempt-new" if exempt else "applied"
                prior = None if exempt else prior_final_boundaries(year, name, label, part=reg[mid]["part"])
                if not exempt and prior is None:
                    gr_note = "no-prior->treated-exempt"
                cap = None
                if gr_note == "applied":
                    if restricted:
                        pts = prior_trimmed_scores(year, name, part=reg[mid]["part"])
                        cap = str(restricted_range_cap(pts)) if pts else None
                        if cap is None:
                            gr_note = "no-prior-range->exempt"
                    else:
                        cap = "5"
                jobs.append({
                    "year": year, "mid": mid, "org": label, "name": name,
                    "higher": higher, "dd": dd, "target": target,
                    "cap_lo": cap_lo, "cap_hi": cap_hi,
                    "gr_note": gr_note, "prior": prior, "cap": cap,
                    "kept": kept,        # reconciled clustering input (cid, score)
                    "pool": sub,         # star-assignment pool (cid, score)
                })
    return jobs, skips


# --------------------------------------------------------------------- workers
def _worker(task):
    """(job, realization_name) -> measure-level result for BOTH arms.
    Errors are returned as rows (counted by the caller), never swallowed."""
    job, rname = task
    out = {"year": job["year"], "mid": job["mid"], "org": job["org"],
           "name": job["name"], "realization": rname,
           "n_cluster": len(job["kept"]), "n_pool": len(job["pool"]),
           "guardrail": job["gr_note"]}
    try:
        cand, order = _split_rname(rname)
        scores = [s for _, s in order_pairs(job["kept"], order)]
        arms = {}
        for arm in ("ward_tie", "dp"):
            r = measure_cutpoints(scores, higher_is_better=job["higher"],
                                  tukey=True, cap_lo=job["cap_lo"],
                                  cap_hi=job["cap_hi"], groups_candidate=cand,
                                  seed=SEED, n_groups=N_GROUPS, arm=arm)
            if "error" in r:
                out["error"] = f"{arm}: {r['error']}"
                return out
            if len(r["mean_cutpoints"]) != 4:
                out["error"] = f"{arm}: cutpoint count {len(r['mean_cutpoints'])} != 4"
                return out
            arms[arm] = r["mean_cutpoints"]
        finals = {}
        for arm in ("ward_tie", "dp"):
            vals = arms[arm]
            if job["gr_note"] == "applied":
                cap = F(job["cap"])
                fin = [guardrail_boundary(v, F(p), cap)[0]
                       for v, p in zip(vals, job["prior"])]
            else:
                fin = vals
            finals[arm] = [round_display(v, job["dd"]) for v in fin]
        bw = [F(x) for x in finals["ward_tie"]]
        bd = [F(x) for x in finals["dp"]]
        deltas = []
        for c, s in job["pool"]:
            x = F(s)
            if job["higher"]:
                sw = 1 + sum(1 for t in bw if x >= t)
                sd = 1 + sum(1 for t in bd if x >= t)
            else:
                sw = 5 - sum(1 for t in bw if x > t)
                sd = 5 - sum(1 for t in bd if x > t)
            if sw != sd:
                deltas.append((c, sw, sd))
        out.update(
            ward_mean=[str(v) for v in arms["ward_tie"]],
            dp_mean=[str(v) for v in arms["dp"]],
            ward_final=finals["ward_tie"], dp_final=finals["dp"],
            pub_match_ward=sum(1 for a, b in zip(finals["ward_tie"], job["target"])
                               if F(a) == F(b)),
            n_star_changes=len(deltas), deltas=deltas)
        return out
    except Exception as e:                    # reported, not swallowed
        out["error"] = f"exception: {e!r}"
        return out


# ---------------------------------------------------------------- aggregation
def enclosure_aggregate(measure_rows, realizations):
    """One aggregation pass per realization (correlated deltas: each pass
    applies one realization's deltas to all measures simultaneously), then the
    classification per contract."""
    errors = [r for r in measure_rows if "error" in r]
    err_splits = {(r["year"], r["mid"], r["org"]) for r in errors}
    # deltas[(year,cid,mid)][rname] = dp_star - ward_star  (0 if absent)
    delta_idx = defaultdict(dict)
    for r in measure_rows:
        if "error" in r:
            continue
        if (r["year"], r["mid"], r["org"]) in err_splits:
            continue          # excluded everywhere for cross-realization comparability
        for cid, sw, sd in r["deltas"]:
            delta_idx[(r["year"], cid, r["mid"])][r["realization"]] = int(sd) - int(sw)

    per_r = {rn: Counter() for rn in realizations}   # up/down/changed per realization
    per_r_year = {rn: defaultdict(Counter) for rn in realizations}
    contracts = []
    n_rated_total = 0
    for year in YEARS:
        bv = BASELINE_VINTAGE[year]
        spec = AggSpec(year)
        stars, summary, cai = load_inputs(year, bv)
        ddup = frozenset(dup_d_ids(year, bv))
        for cid, srow in summary.items():
            st = stars.get(cid)
            if not st:
                continue
            oc = org_class_of(srow)
            hh = {"MA-PD": "overall", "MA-only": "part_c", "PDP": "part_d"}[oc]
            kw = dict(gate_convention="raw",
                      disaster=disaster_pct_of(srow, year) >= 25, dup_d=ddup)
            base = rate_contract(spec, st, oc, cai.get(cid, {}), **kw)
            if hh not in base:
                continue
            n_rated_total += 1
            tmids = [mid for mid in st if (year, cid, mid) in delta_idx]
            if not tmids:
                continue
            b = F(base[hh][0])
            memo = {}                          # cf star-vector -> rating Fraction
            ratings, flips, changed = {}, {}, {}
            for rn in realizations:
                sig = tuple(max(1, min(5, st[m] + delta_idx[(year, cid, m)].get(rn, 0)))
                            for m in tmids)
                if sig in memo:
                    c = memo[sig]
                else:
                    if all(sig[i] == st[tmids[i]] for i in range(len(tmids))):
                        c = b                  # untouched under this realization
                    else:
                        st_cf = dict(st)
                        for m, v in zip(tmids, sig):
                            st_cf[m] = v
                        cf = rate_contract(spec, st_cf, oc, cai.get(cid, {}), **kw)
                        c = F(cf[hh][0]) if hh in cf else None
                    memo[sig] = c
                ratings[rn] = c
                flips[rn] = (c is not None
                             and ((b >= 4 and c < 4) or (b < 4 and c >= 4)))
                changed[rn] = c is not None and c != b
                if flips[rn]:
                    di = "down" if b >= 4 else "up"
                    per_r[rn][di] += 1
                    per_r_year[rn][year][di] += 1
                if changed[rn]:
                    per_r[rn]["changed"] += 1
            nf = sum(flips.values())
            if nf == 0:
                continue
            direction = "DOWN" if b >= 4 else "UP"
            ivals = {m: [min(delta_idx[(year, cid, m)].get(rn, 0) for rn in realizations),
                         max(delta_idx[(year, cid, m)].get(rn, 0) for rn in realizations)]
                     for m in tmids}
            contracts.append({
                "year": year, "contract_id": cid, "org_class": oc,
                "qbp_relevant": oc != "PDP", "rating_type": hh,
                "baseline": str(b),
                "counterfactual_set": sorted({str(ratings[rn]) for rn in realizations
                                              if ratings[rn] is not None}),
                "ratings_by_realization": {rn: (str(ratings[rn])
                                                if ratings[rn] is not None else None)
                                           for rn in realizations},
                "n_flip_realizations": nf, "k_realizations": len(realizations),
                "class": ("exact-within-groups" if nf == len(realizations)
                          else "realization-dependent"),
                "direction": direction,
                "delta_intervals": ivals,
                "n_uncertain_measures": sum(1 for v in ivals.values() if v[0] != v[1]),
            })
    return {"contracts": contracts, "per_realization": per_r,
            "per_realization_year": per_r_year, "errors": errors,
            "errored_splits": sorted(err_splits), "n_rated": n_rated_total}


# ---------------------------------------------------------------------- driver
def run_measure_phase(jobs, realizations, workers):
    tasks = [(j, rn) for rn in realizations for j in jobs]
    print(f"measure-splits: {len(jobs)}  realizations: {len(realizations)}  "
          f"tasks: {len(tasks)}", flush=True)
    ctx = mp.get_context("spawn")            # fork can deadlock with BLAS thread pools
    rows = []
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} measure tasks", flush=True)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"measure phase: errors: {n_err}", flush=True)
    return rows


def reconstruct_prior_rows(prior, jobs):
    """Rehydrate stored measure rows without recomputing clustering: per-contract
    star deltas are a deterministic function of the saved display-rounded finals
    and the (rebuilt) assignment pool. The reconstructed delta count must equal
    the stored n_star_changes for every row, else abort."""
    jmap = {(j["year"], j["mid"], j["org"]): j for j in jobs}
    rows = []
    for r in prior["measure_results"]:
        j = jmap[(r["year"], r["mid"], r["org"])]
        if "deltas" in r:                     # newer schema: deltas persisted
            rows.append(r)
            continue
        bw = [F(x) for x in r["ward_final"]]
        bd = [F(x) for x in r["dp_final"]]
        deltas = []
        for c, s in j["pool"]:
            x = F(s)
            if j["higher"]:
                sw = 1 + sum(1 for t in bw if x >= t)
                sd = 1 + sum(1 for t in bd if x >= t)
            else:
                sw = 5 - sum(1 for t in bw if x > t)
                sd = 5 - sum(1 for t in bd if x > t)
            if sw != sd:
                deltas.append((c, sw, sd))
        if len(deltas) != r["n_star_changes"]:
            raise RuntimeError(
                f"delta reconstruction mismatch {r['year']}/{r['mid']}/{r['org']}"
                f"/{r['realization']}: {len(deltas)} vs stored {r['n_star_changes']}")
        rr = dict(r)
        rr["deltas"] = deltas
        rows.append(rr)
    return rows


def _scope_text(k):
    axes = ("SURVEYSELECT GROUPS=10 randomization family x input-order "
            "conventions" if k > 16 else "SURVEYSELECT GROUPS=10 family")
    parts = [
        f"Range over {k} sampled realizations of the documented {axes} "
        "(11 structurally distinct candidate algorithms, seed 8675309, + 5 "
        "uniform-random equal-split draws random.Random(1000+i), on cid_asc "
        "dataset order"]
    if k > 16:
        parts.append(
            "; + 8 input-order variants: {seq_quota_last, sort_blocks_first} x "
            "{cid_desc, score_asc, score_desc} and 2 uniform-random draws "
            "(Random(1005), Random(1006)) on score_asc; the ordering feeds "
            "both the group assignment and the ward_tie observation numbering")
    parts.append(
        "). Not a proof over all realizations. Clustering inputs: K-table "
        f"basis vintages (2024/2025 original, 2026 current), stage-1 "
        f"reconciled membership ({HYP}; removals dropped, adds "
        "appended). Baseline stars/ratings: operative published record "
        "(2024 recalc, 2025/2026 current). Arms: ward_tie (SAS tie-rule "
        "Ward replay) vs dp (exact SSQ optimum); identical resampling groups, "
        "guardrails, rounding. CAHPS, improvement, and errored "
        "measure-splits held at published stars (an undercount). "
        "Errored splits excluded from all realizations.")
    return "".join(parts)


def main(mode="full", workers=os.cpu_count()):
    jobs, skips = build_jobs()
    extra_summary = {}
    reused_rows = []
    if mode == "pilot":
        realizations = REALIZATIONS16[:1]
        rows = run_measure_phase(jobs, realizations, workers)
    elif mode == "full":
        realizations = REALIZATIONS16
        rows = run_measure_phase(jobs, realizations, workers)
    elif mode == "extend24":
        realizations = REALIZATIONS24
        prior_path = os.path.join(BASE, "runs", "enclosure_census_docsonly.json")
        prior = json.load(open(prior_path))
        if prior["realizations"] == REALIZATIONS16:
            pending = ORDER_EXT
        elif prior["realizations"] == REALIZATIONS24:
            print("prior run already K=24; re-aggregating stored rows only")
            pending = []
        else:
            raise RuntimeError(f"unexpected prior realizations: "
                               f"{prior['realizations']}")
        reused_rows = reconstruct_prior_rows(prior, jobs) + prior["errors"]
        n_prior_err = len(prior["errors"])
        print(f"reused stored rows: {len(reused_rows) - n_prior_err} "
              f"(+{n_prior_err} stored errors); deltas reconstructed and "
              f"checked", flush=True)
        # preserve the K=16 result before overwriting in place
        k16_path = os.path.join(BASE, "runs", "enclosure_census_k16_docsonly.json")
        if prior["realizations"] == REALIZATIONS16 and not os.path.exists(k16_path):
            json.dump(prior, open(k16_path, "w"), indent=1)
            print(f"K=16 result preserved at {k16_path}", flush=True)
        if pending:
            rows = run_measure_phase(jobs, pending, workers)
        else:
            rows = []
        rows = reused_rows + rows
        # K=16 vs K=24 difference (K=16 classified with its own error set,
        # exactly as originally computed)
        rows16 = [r for r in rows if r["realization"] in REALIZATIONS16]
        agg16 = enclosure_aggregate(rows16, REALIZATIONS16)
        cert16 = sorted((c["year"], c["contract_id"], c["direction"])
                        for c in agg16["contracts"]
                        if c["class"] == "exact-within-groups")
        extra_summary["k16_exact"] = cert16
    else:
        raise ValueError(mode)

    agg = enclosure_aggregate(rows, realizations)
    print(f"aggregation phase ({len(realizations)} passes) done", flush=True)

    exact_rows = [c for c in agg["contracts"] if c["class"] == "exact-within-groups"]
    dependent = [c for c in agg["contracts"] if c["class"] == "realization-dependent"]
    totals = {rn: agg["per_realization"][rn]["up"] + agg["per_realization"][rn]["down"]
              for rn in realizations}
    tv = sorted(totals.values())

    if "k16_exact" in extra_summary:
        cert24 = sorted((c["year"], c["contract_id"], c["direction"])
                        for c in exact_rows)
        c16set = {(y, c) for y, c, _ in extra_summary["k16_exact"]}
        c24set = {(y, c) for y, c, _ in cert24}
        extra_summary["k16_vs_k24"] = {
            "k16_exact_count": len(extra_summary["k16_exact"]),
            "k24_exact_count": len(cert24),
            "k24_exact": cert24,
            "survived_ordering_axis": sorted(c16set & c24set),
            "dropped_by_ordering_axis": sorted(c16set - c24set),
            "gained": sorted(c24set - c16set),   # impossible by monotonicity; checked
        }

    out = {
        "register": _scope_text(len(realizations)),
        "hypothesis": HYP,
        "cluster_vintage": CLUSTER_VINTAGE,
        "baseline_vintage": BASELINE_VINTAGE,
        "seed": SEED,
        "realizations": realizations,
        "n_measure_splits": len(jobs),
        "skipped_splits": skips,
        "errors": agg["errors"],
        "errored_splits": agg["errored_splits"],
        "measure_results": [
            {k: r[k] for k in ("year", "mid", "org", "name", "realization",
                               "n_cluster", "n_pool", "guardrail", "ward_mean",
                               "dp_mean", "ward_final", "dp_final",
                               "pub_match_ward", "n_star_changes", "deltas")
             if k in r}
            for r in rows if "error" not in r],
        "contracts": agg["contracts"],
        "per_realization_flips": {
            rn: {"up": agg["per_realization"][rn]["up"],
                 "down": agg["per_realization"][rn]["down"],
                 "total": totals[rn],
                 "highest_rating_changed": agg["per_realization"][rn]["changed"],
                 "by_year": {y: dict(agg["per_realization_year"][rn][y])
                             for y in YEARS}}
            for rn in realizations},
        "summary": {
            "n_rated_contract_years": agg["n_rated"],
            "exact_within_groups_flips": len(exact_rows),
            "realization_dependent_flips": len(dependent),
            "per_realization_total_flips": {
                "min": tv[0] if tv else 0,
                "median": tv[len(tv) // 2] if tv else 0,
                "max": tv[-1] if tv else 0},
            "stored_single_realization_reference": {
                "seq_quota_last": 55, "sort_blocks_first": 48},
            "n_rows_reused_from_stored": len(reused_rows),
            **extra_summary,
        },
    }
    outpath = os.path.join(
        BASE, "runs",
        "enclosure_census_pilot_docsonly.json" if mode == "pilot" else "enclosure_census_docsonly.json")
    json.dump(out, open(outpath, "w"), indent=1)
    print(f"\nsaved {outpath}")

    print(f"\n=== SAMPLED-REALIZATION SUMMARY (K={len(realizations)}) ===")
    print(f"rated contract-years: {agg['n_rated']}")
    print(f"flips holding in all realizations: {len(exact_rows)}")
    for c in exact_rows:
        print(f"  {c['year']} {c['contract_id']} {c['org_class']:>7} "
              f"{c['direction']:>4} {c['baseline']} -> "
              f"{{{','.join(c['counterfactual_set'])}}} QBP={c['qbp_relevant']}")
    print(f"REALIZATION-DEPENDENT flips: {len(dependent)}")
    dist = Counter(c["n_flip_realizations"] for c in dependent)
    print(f"  flip-count distribution (n_realizations -> contracts): "
          f"{dict(sorted(dist.items()))}")
    print(f"per-realization total flips: min={tv[0] if tv else 0} "
          f"median={tv[len(tv) // 2] if tv else 0} max={tv[-1] if tv else 0}")
    for rn in realizations:
        p = agg["per_realization"][rn]
        print(f"  {rn:>18}: UP {p['up']:3d}  DOWN {p['down']:3d}  "
              f"total {p['up'] + p['down']:3d}  changed {p['changed']:3d}")
    if agg["errors"]:
        errc = Counter((e["year"], e["mid"], e["org"]) for e in agg["errors"])
        print(f"errors: {len(agg['errors'])} task(s) over "
              f"{len(errc)} measure-split(s) [excluded from all realizations]:")
        for (y, m, o), n in sorted(errc.items()):
            sample = next(e["error"] for e in agg["errors"]
                          if (e["year"], e["mid"], e["org"]) == (y, m, o))
            print(f"  {y} {m} {o}: {n}/{len(realizations)} realizations "
                  f"[{sample[:90]}]")
    if "k16_vs_k24" in extra_summary:
        kk = extra_summary["k16_vs_k24"]
        print(f"\n=== K=16 vs K=24 DIFFERENCE ===")
        print(f"K=16 all-realization flips: {kk['k16_exact_count']}  "
              f"K=24 all-realization flips: {kk['k24_exact_count']}  "
              f"survived ordering axis: {len(kk['survived_ordering_axis'])}")
        print(f"dropped by ordering axis: {kk['dropped_by_ordering_axis']}")
        if kk["gained"]:
            print(f"GAINED (impossible by monotonicity): {kk['gained']}")
    return out


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    assert mode in ("pilot", "full", "extend24"), mode
    main(mode, workers)
