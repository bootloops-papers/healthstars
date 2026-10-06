# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Optimality without filled-in scores, and provenance of the criterion flips.
Matching the published outlier bounds does not pin down the Medicare Advantage
score list, so optimality is reported separately for the computations that
need no filled-in scores.

WHAT THIS COMPUTES:
 (A) Splits the 123 cut-point computations into
       PARAMETER_FREE_STRICT : published fences exact from the documented
                               rules alone (battery before_exact True) AND no
                               synthetic add touches the MID/ORG AND no fitted
                               removal contract sits in its documented score list;
       REMOVAL_ONLY          : before_exact True, no add, but >=1 fitted
                               removal contract is dropped from its score list
                               (removals are applied YEAR-WIDE by
                               enclosure_census_docsonly.build_jobs /
                               exact_gaps_docsonly.build_jobs);
       FITTED                : an add touches it or before_exact is False;
     and reports off-optimum, zero-gap and fewer-than-five-score counts per
     class per year on BOTH stored gap tables (runs/exact_gaps_fullsample_docsonly
     .json = replay vintages; runs/exact_gaps_criterionvintage.json =
     criterion vintages). It also RECOMPUTES Ward vs exact-DP on the pure
     documented-rules score list (no adds, no removals) for every computation at
     both vintages (exact SSQ comparison), so optimality can be quoted with
     the fitted layer stripped entirely.
 (B) Provenance of the 60 criterion flips (34 in the dollar estimate): per flip row of
     runs/criterion_census.json, re-derives the measure-level cells where the
     criterion star differs from the published star (criterion_verify_suite
     load_world/delta_sets, identical hold-harmless), classifies each moved
     cell's computation, classifies each flip ALL_PARAMETER_FREE_STRICT /
     PARAMETER_FREE_WITH_REMOVAL / TOUCHES_FITTED, confirms the contract's
     own score in every moved measure is a real published score, and sums
     enrollment and |mid| dollars of the flips in the dollar estimate per class.
WHAT IT DOES NOT COMPUTE: it does not pin down CMS's actual score list;
parameter-free score lists are documented-rules reconstructions from
public files whose outlier bounds equal the published ones (corroboration,
not proof of score list identity).

Output: runs/optimality_without_fills.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.6, filled-in scores
    and the Medicare Advantage results (82 computations reproduced with nothing
    filled in: 65 miss the minimum, 14 attain it).
Run:  cd src && python3 optimality_without_fills.py [workers]
Requires: Python >= 3.10; numpy, scipy.
"""
import builtins
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "runs", "optimality_without_fills.json")

# ---- log every file opened for reading under BASE (input pins) -------------
_OPENED = set()
_real_open = builtins.open


def _logging_open(file, mode="r", *a, **k):
    try:
        if isinstance(file, (str, bytes, os.PathLike)) and ("r" in mode
                                                              and "w" not in mode
                                                              and "a" not in mode
                                                              and "+" not in mode):
            p = os.path.abspath(os.fsdecode(file))
            if p.startswith(BASE + os.sep):
                _OPENED.add(p)
    except Exception:
        pass
    return _real_open(file, mode, *a, **k)


builtins.open = _logging_open

import exact_gaps_docsonly as G                      # noqa: E402
import criterion_verify_suite as V                       # noqa: E402
import enclosure_census_docsonly as C                    # noqa: E402
from criterion_dollar_estimate_build import DEMOTED             # noqa: E402

GAPS_FILES = {
    "replay_vintage": "runs/exact_gaps_fullsample_docsonly.json",
    "criterion_vintage": "runs/exact_gaps_criterionvintage.json",
}
VINTAGES = {
    "replay_vintage": dict(C.FITVINTAGE),
    "criterion_vintage": dict(V.CRIT_VINTAGE),
}
MEMB = "runs/membership_delta_sets_docsonly.json"
CENSUS = "runs/criterion_census.json"
DOLLAR = "runs/criterion_dollar_estimate.json"


def sha16(path):
    with _real_open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def has_fewer_than_five(g):
    return len(g["dp_cutpoints"]) != 4


# --------------------------------------------------------------------------
def membership_layer():
    ds = json.load(open(os.path.join(BASE, MEMB)))
    layer = {}
    for year, d in ds["years"].items():
        bex = {}
        for r in d["battery"]:
            if "before_exact" in r:
                bex[(r["mid"], r["org"])] = (bool(r["before_exact"]),
                                             bool(r["after_exact"]))
        adds_by_key = defaultdict(list)
        for a in d["adds"]:
            for mkey in a["measures"]:
                adds_by_key[mkey].append(a["id"])
        rems = [r["contract_id"] for r in d["removals"]]
        layer[year] = dict(before_exact=bex, adds_by_key=adds_by_key,
                           removals=rems, n_adds=len(d["adds"]),
                           n_removals=len(rems),
                           n_add_cells=sum(len(a["measures"]) for a in d["adds"]))
    return ds, layer


def documented_jobs(vintage_map):
    """exact_gaps_docsonly.build_jobs at a vintage: reconciled (stored
    score list) and unreconciled (pure documented-rules score list)."""
    G.SCORE_VINTAGE = dict(vintage_map)
    rec = {(j["year"], j["mid"], j["org"]): j for j in G.build_jobs(reconciled=True)}
    doc = {(j["year"], j["mid"], j["org"]): j for j in G.build_jobs(reconciled=False)}
    return rec, doc


def classify_computations(layer, doc_jobs):
    """key -> dict(class, before_exact, adds, removals_in_score_list, ...)."""
    out = {}
    for key, j in doc_jobs.items():
        year, mid, org = key
        L = layer[year]
        mkey = f"{mid}/{org}"
        adds = L["adds_by_key"].get(mkey, [])
        score_list_ids = {c for c, _ in j["kept"]}
        rem_in = [c for c in L["removals"] if c in score_list_ids]
        be = L["before_exact"].get((mid, org))
        if be is None:
            cls = "NO_BATTERY_ROW"
        elif be[0] and not adds and not rem_in:
            cls = "PARAMETER_FREE_STRICT"
        elif be[0] and not adds and rem_in:
            cls = "REMOVAL_ONLY"
        else:
            cls = "FITTED"
        out[key] = dict(year=year, mid=mid, org=org, name=j["name"],
                        battery_before_exact=(be[0] if be else None),
                        battery_after_exact=(be[1] if be else None),
                        synthetic_adds_in_score_list=adds,
                        n_synthetic_cells_in_score_list=len(adds),
                        fitted_removals_in_documented_score_list=rem_in,
                        n_documented_score_list=len(j["kept"]),
                        documented_score_list_fence_exact_recomputed=(
                            j["membership"] == "fence-exact"),
                        cls=cls)
    return out


def tally(classes, gaps, label):
    """per class per year: n, off_optimum, zero_gap, fewer_than_five_scores."""
    t = defaultdict(lambda: defaultdict(lambda: Counter()))
    rows = []
    for key, c in classes.items():
        g = gaps.get(key)
        if g is None:
            rows.append(dict(key=list(key), missing_in=label))
            continue
        deg = has_fewer_than_five(g)
        off = F(g["gap"]) > 0
        assert off == bool(g["off_optimum"]), key
        zero = F(g["gap"]) == 0
        for yk in (c["year"], "ALL"):
            cc = t[c["cls"]][yk]
            cc["n"] += 1
            cc["off_optimum"] += int(off)
            cc["zero_gap"] += int(zero)
            cc["fewer_than_five_scores"] += int(deg)
            cc["zero_gap_with_five_scores"] += int(zero and not deg)
            cc["dp_unique"] += int(g["dp_n_optima"] == 1)
    out = {cls: {y: dict(v) for y, v in sorted(d.items())} for cls, d in t.items()}
    # the paper's definition: "parameter-free" = fences exact with nothing fitted
    # (battery before_exact True, no add) = STRICT + REMOVAL_ONLY
    roll = defaultdict(Counter)
    for cls in ("PARAMETER_FREE_STRICT", "REMOVAL_ONLY"):
        for y, v in t.get(cls, {}).items():
            roll[y].update(v)
    out["PAPER_GRAIN_82_before_exact_no_add"] = {y: dict(v) for y, v in sorted(roll.items())}
    return out, rows


def _run_doc(job):
    """worker: Ward vs exact DP on a pinned score list (reuses the stored _run)."""
    return G._run(job)


# --------------------------------------------------------------------------
def main(workers=None):
    workers = workers or os.cpu_count() or 1
    ds, layer = membership_layer()

    # ---------------- (A) classification + stored-table tallies ------------
    A = {}
    doc_recompute_jobs = []
    classes_by_v = {}
    recjobs_by_v = {}
    for vlabel, vm in VINTAGES.items():
        rec, doc = documented_jobs(vm)
        gaps = {(g["year"], g["mid"], g["org"]): g
                for g in json.load(open(os.path.join(BASE, GAPS_FILES[vlabel])))}
        classes = classify_computations(layer, doc)
        classes_by_v[vlabel] = classes
        recjobs_by_v[vlabel] = rec
        keys_gaps, keys_doc, keys_rec = set(gaps), set(doc), set(rec)
        tallies, missing = tally(classes, gaps, vlabel)
        # battery cross-check: recomputed documented score list fence-exact flag
        # vs battery before_exact (battery was computed at the FIT vintages)
        xcheck = Counter()
        for key, c in classes.items():
            xcheck[(c["battery_before_exact"],
                    c["documented_score_list_fence_exact_recomputed"])] += 1
        A[vlabel] = dict(
            vintages=vm, gaps_file=GAPS_FILES[vlabel],
            n_gap_rows=len(gaps), n_documented_jobs=len(doc),
            n_reconciled_jobs=len(rec),
            keyset_equal_gaps_doc=(keys_gaps == keys_doc),
            keyset_equal_gaps_rec=(keys_gaps == keys_rec),
            keys_in_gaps_not_doc=sorted(map(list, keys_gaps - keys_doc)),
            keys_in_doc_not_gaps=sorted(map(list, keys_doc - keys_gaps)),
            class_counts_per_year={
                y: dict(Counter(c["cls"] for c in classes.values()
                                if c["year"] == y)) for y in C.YEARS},
            class_counts_all=dict(Counter(c["cls"] for c in classes.values())),
            stored_table_tallies_per_class=tallies,
            missing_rows=missing,
            battery_before_exact_vs_recomputed_doc_score_list_fence_exact={
                f"battery={a}|recomputed={b}": n for (a, b), n in xcheck.items()},
            computations=[dict(c, stored_off_optimum=(bool(gaps[k]["off_optimum"])
                                                      if k in gaps else None),
                               stored_gap=(gaps[k]["gap"] if k in gaps else None),
                               stored_fewer_than_five_scores=(has_fewer_than_five(gaps[k])
                                                  if k in gaps else None),
                               stored_n_trimmed=(gaps[k]["n_trimmed"]
                                                 if k in gaps else None))
                          for k, c in sorted(classes.items())])
        for key, j in doc.items():
            jj = dict(j)
            jj["_vlabel"] = vlabel
            doc_recompute_jobs.append(jj)

    # ------------- (A2) recompute on the pure documented-rules score list -------
    print(f"documented-score-list recompute: {len(doc_recompute_jobs)} jobs, "
          f"{workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    doc_recompute_jobs.sort(key=lambda j: -len(j["kept"]))
    with ctx.Pool(workers) as pool:
        rows = pool.map(_run_doc, doc_recompute_jobs, chunksize=1)
    RE = {}
    for vlabel in VINTAGES:
        gaps = {(g["year"], g["mid"], g["org"]): g
                for g in json.load(open(os.path.join(BASE, GAPS_FILES[vlabel])))}
        classes = classes_by_v[vlabel]
        mine = [r for r, j in zip(rows, doc_recompute_jobs) if j["_vlabel"] == vlabel]
        per = defaultdict(lambda: defaultdict(Counter))
        detail = []
        strict_repro = Counter()
        for r in mine:
            key = (r["year"], r["mid"], r["org"])
            cls = classes[key]["cls"]
            g = gaps.get(key)
            deg = len(r["dp_cutpoints"]) != 4
            off = F(r["gap"]) > 0
            same_dp = (g is not None and r["dp_cutpoints"] == g["dp_cutpoints"])
            same_off = (g is not None and bool(g["off_optimum"]) == off)
            for yk in (r["year"], "ALL"):
                cc = per[cls][yk]
                cc["n"] += 1
                cc["off_optimum"] += int(off)
                cc["zero_gap"] += int(not off)
                cc["zero_gap_with_five_scores"] += int((not off) and not deg)
                cc["fewer_than_five_scores"] += int(deg)
                cc["dp_unique"] += int(r["dp_n_optima"] == 1)
                cc["dp_cutpoints_equal_stored"] += int(same_dp)
                cc["off_optimum_status_equal_stored"] += int(same_off)
            if cls == "PARAMETER_FREE_STRICT":
                # identical score list -> must reproduce the stored row exactly
                strict_repro[(same_dp, r["gap"] == (g or {}).get("gap"),
                              r["n_trimmed"] == (g or {}).get("n_trimmed"))] += 1
            detail.append(dict(year=r["year"], mid=r["mid"], org=r["org"],
                               cls=cls, n_input=r["n_input"],
                               n_trimmed=r["n_trimmed"], k=r["k"],
                               fence_exact=r["membership"] == "fence-exact",
                               ward_ssq=r["ward_ssq"], dp_ssq=r["dp_ssq"],
                               gap=r["gap"], off_optimum=off,
                               dp_n_optima=r["dp_n_optima"],
                               dp_cutpoints=r["dp_cutpoints"],
                               ward_cutpoints=r["ward_cutpoints"],
                               stored_dp_cutpoints=(g["dp_cutpoints"] if g else None),
                               stored_gap=(g["gap"] if g else None),
                               dp_cutpoints_equal_stored=same_dp,
                               off_optimum_status_equal_stored=same_off))
        tot = Counter()
        for r in mine:
            _deg = len(r["dp_cutpoints"]) != 4
            _off = F(r["gap"]) > 0
            tot["n"] += 1
            tot["off_optimum"] += int(_off)
            tot["zero_gap_with_five_scores"] += int((not _off) and not _deg)
            tot["fewer_than_five_scores"] += int(_deg)
            tot["dp_unique"] += int(r["dp_n_optima"] == 1)
            tot["fence_exact_on_documented_score_list"] += int(
                r["membership"] == "fence-exact")
        RE[vlabel] = dict(
            register=("Ward (SAS tie rule, IEEE double) vs exact DP optimum on "
                      "the PURE DOCUMENTED-RULES score list (cost+d60r hypothesis; "
                      "NO synthetic adds, NO fitted removals) after Tukey "
                      "trim; exact SSQ comparison (Fraction). For "
                      "PARAMETER_FREE_STRICT computations this score list IS the "
                      "stored score list, so the row must reproduce the stored "
                      "table (self-check). For REMOVAL_ONLY/FITTED it is a "
                      "different, stated score list whose fences do NOT all match "
                      "the published fences."),
            totals=dict(tot),
            per_class=({cls: {y: dict(v) for y, v in sorted(d.items())}
                        for cls, d in per.items()}),
            strict_class_reproduces_stored_row={
                f"dp_cut={a}|gap={b}|n_trimmed={c}": n
                for (a, b, c), n in strict_repro.items()},
            rows=sorted(detail, key=lambda d: (d["year"], d["mid"], d["org"])))

    # ---------------- (B) provenance of the 60 flips ------------------------
    gapsC, jobsC, skipsC, baseline = V.load_world()
    dsets = V.delta_sets(gapsC, jobsC, baseline, include_degenerate=False)
    rerated = {(r["year"], r["contract_id"]): r
               for r in V.apply_and_rate(baseline, dsets)}
    jobmap = {(j["year"], j["mid"], j["org"]): j for j in jobsC}
    classesC = classes_by_v["criterion_vintage"]
    classesR = classes_by_v["replay_vintage"]
    idx = defaultdict(dict)
    for (year, mid, org), deltas in dsets.items():
        for cid, sw, sd in deltas:
            assert mid not in idx[(year, cid)], (year, cid, mid)
            idx[(year, cid)][mid] = (int(sw), int(sd), org)
    census = json.load(open(os.path.join(BASE, CENSUS)))
    dollar = json.load(open(os.path.join(BASE, DOLLAR)))
    flips = census["flips_criterion"]
    out_flips = []
    agg = defaultdict(lambda: Counter())
    mismatch = []
    own_score_anoms = []
    for row in flips:
        year, cid = row["year"], row["contract_id"]
        st = baseline[year]["stars"].get(cid, {})
        moved_all = idx.get((year, cid), {})
        moved = {m: v for m, v in moved_all.items() if m in st}
        cells = []
        touched_cls = Counter()
        fitted_comps, removal_comps = [], []
        for m, (sw, sd, org) in sorted(moved.items()):
            key = (year, m, org)
            j = jobmap[key]
            cC = classesC[key]
            cR = classesR.get(key)
            pool_ids = {c for c, _ in j["pool"]}
            kept_ids = {c for c, _ in j["kept"]}
            own_score = dict(j["pool"]).get(cid)
            own_real = (cid in pool_ids) and not cid.startswith("ADD-")
            if not own_real:
                own_score_anoms.append(dict(year=year, cid=cid, mid=m))
            g = gapsC[key]
            cell = dict(measure=m, org=org, name=j["name"],
                        published_star=sw, criterion_star=sd,
                        own_score=own_score, own_score_is_real_published=own_real,
                        own_score_in_clustering_score_list=(cid in kept_ids),
                        computation_class=cC["cls"],
                        computation_class_replay_vintage=(cR["cls"] if cR else None),
                        synthetic_adds_in_score_list=cC["synthetic_adds_in_score_list"],
                        n_synthetic_cells_in_score_list=cC["n_synthetic_cells_in_score_list"],
                        fitted_removals_in_documented_score_list=cC[
                            "fitted_removals_in_documented_score_list"],
                        score_list_n_input=g["n_input"], score_list_n_trimmed=g["n_trimmed"],
                        computation_off_optimum=bool(g["off_optimum"]),
                        computation_gap=g["gap"], guardrail=j["gr_note"])
            cells.append(cell)
            touched_cls[cC["cls"]] += 1
            if cC["cls"] == "FITTED":
                fitted_comps.append(dict(
                    computation=f"{year}/{m}/{org}",
                    n_synthetic_cells=cC["n_synthetic_cells_in_score_list"],
                    n_removals=len(cC["fitted_removals_in_documented_score_list"]),
                    battery_before_exact=cC["battery_before_exact"],
                    score_list_n_input=g["n_input"]))
            elif cC["cls"] == "REMOVAL_ONLY":
                removal_comps.append(dict(
                    computation=f"{year}/{m}/{org}",
                    removals=cC["fitted_removals_in_documented_score_list"],
                    score_list_n_input=g["n_input"]))
        if fitted_comps:
            fcls = "TOUCHES_FITTED"
        elif removal_comps:
            fcls = "PARAMETER_FREE_WITH_REMOVAL"
        else:
            fcls = "ALL_PARAMETER_FREE_STRICT"
        rr = rerated.get((year, cid))
        ok = (rr is not None and rr["criterion_rating"] == row["criterion_rating"]
              and rr["baseline"] == row["baseline"])
        if not ok:
            mismatch.append(dict(year=year, cid=cid, census=(row["baseline"],
                                 row["criterion_rating"]),
                                 rederived=(rr["baseline"], rr["criterion_rating"])
                                 if rr else None))
        valued = bool(row["qbp_priceable"]) and (year, cid) not in DEMOTED
        mid_usd = (abs(row["est_annual_qbp_revenue_delta_usd"]["mid"])
                   if row.get("est_annual_qbp_revenue_delta_usd") else 0)
        out_flips.append(dict(
            year=year, contract_id=cid, direction=row["direction"],
            published_overall=row["published_overall"], baseline=row["baseline"],
            criterion_rating=row["criterion_rating"],
            qbp_priceable=bool(row["qbp_priceable"]),
            demoted_endpoint_conditional=(year, cid) in DEMOTED,
            in_dollar_estimate=valued, enrollment=row.get("enrollment"),
            mid_usd_abs=(mid_usd if valued else None),
            rederived_rating_matches_census=ok,
            n_moved_cells=len(cells),
            n_moved_cells_by_computation_class=dict(touched_cls),
            flip_class=fcls,
            fitted_computations=fitted_comps,
            removal_only_computations=removal_comps,
            all_own_scores_real_published=all(c["own_score_is_real_published"]
                                              for c in cells),
            n_own_scores_not_in_clustering_score_list=sum(
                1 for c in cells if not c["own_score_in_clustering_score_list"]),
            moved_cells=cells))
        for grp in ("all60", "dollar34" if valued else None):
            if grp is None:
                continue
            a = agg[(grp, fcls)]
            a["n_flips"] += 1
            a["n_moved_cells"] += len(cells)
            if valued:
                a["enrollment"] += int(row["enrollment"] or 0)
                a["mid_usd_abs"] += int(mid_usd)
            for y in C.YEARS:
                if y == year:
                    a[f"n_{y}"] += 1
    agg_out = defaultdict(dict)
    for (grp, fcls), a in agg.items():
        agg_out[grp][fcls] = dict(a)
    # totals for cross-check against the dollar estimate record
    v34 = [f for f in out_flips if f["in_dollar_estimate"]]
    tot34 = dict(n=len(v34), enrollment=sum(f["enrollment"] for f in v34),
                 mid_usd_abs=sum(f["mid_usd_abs"] for f in v34))
    dv = dollar["aggregates"]["priced_of_record"]
    tot34["matches_dollar_estimate"] = (
        tot34["n"] == dv["n_contract_years"]
        and tot34["enrollment"] == dv["total_enrollment"]
        and tot34["mid_usd_abs"] == int(dv["gross_total_usd"]["mid"]))
    cells_all = [c for f in out_flips for c in f["moved_cells"]]
    cell_cls = Counter(c["computation_class"] for c in cells_all)
    # distinct computations feeding the flips, by class
    comp_sets = defaultdict(set)
    for f in out_flips:
        for c in f["moved_cells"]:
            comp_sets[c["computation_class"]].add((f["year"], c["measure"], c["org"]))
    # ---- (B2) fitted layer stripped: re-derive every flip from the DP cut
    # points of the PURE documented-rules score lists (criterion vintage) --------
    gaps_doc = {(r["year"], r["mid"], r["org"]): {"dp_cutpoints": r["dp_cutpoints"]}
                for r in RE["criterion_vintage"]["rows"]}
    dsets_doc = V.delta_sets(gaps_doc, jobsC, baseline, include_degenerate=False)
    flips_doc = {(r["year"], r["contract_id"]): r
                 for r in V.apply_and_rate(baseline, dsets_doc)}
    idx_doc = defaultdict(dict)
    for (year, mid, org), deltas in dsets_doc.items():
        for cid, sw, sd in deltas:
            idx_doc[(year, cid)][mid] = (int(sw), int(sd), org)
    surv = Counter()
    surv_rows = []
    cellset_diff_rows = []
    for f in out_flips:
        key = (f["year"], f["contract_id"])
        d = flips_doc.get(key)
        rr = rerated.get(key)
        st = baseline[f["year"]]["stars"].get(f["contract_id"], {})
        moved_doc = {m: v for m, v in idx_doc.get(key, {}).items() if m in st}
        moved_bank = {c["measure"]: (c["published_star"], c["criterion_star"], c["org"])
                      for c in f["moved_cells"]}
        same_cells = moved_doc == moved_bank
        same_side = d is not None and ((F(d["criterion_rating"]) >= 4)
                                       == (F(f["criterion_rating"]) >= 4))
        same_rating = d is not None and d["criterion_rating"] == f["criterion_rating"]
        same_raw = (d is not None and rr is not None
                    and d["criterion_raw"] == rr["criterion_raw"])
        f["documented_score_list_only"] = dict(
            still_crosses_4=bool(d is not None), same_bonus_side=bool(same_side),
            criterion_rating_documented_score_list=(d["criterion_rating"] if d else None),
            criterion_raw_documented_score_list=(d["criterion_raw"] if d else None),
            criterion_raw_stored_score_list=(rr["criterion_raw"] if rr else None),
            same_criterion_rating_display=bool(same_rating),
            same_criterion_raw=bool(same_raw),
            same_moved_cell_set=bool(same_cells),
            moved_cells_documented_score_list={m: dict(published_star=v[0],
                                                   criterion_star=v[1], org=v[2])
                                           for m, v in sorted(moved_doc.items())})
        if not same_cells:
            cellset_diff_rows.append(dict(
                year=f["year"], contract_id=f["contract_id"], in_dollar_estimate=f["in_dollar_estimate"],
                cells_only_stored=sorted(set(moved_bank) - set(moved_doc)),
                cells_only_documented=sorted(set(moved_doc) - set(moved_bank)),
                cells_star_differs=sorted(m for m in set(moved_bank) & set(moved_doc)
                                          if moved_bank[m][:2] != moved_doc[m][:2]),
                display_rating_stored=f["criterion_rating"],
                display_rating_documented=(d["criterion_rating"] if d else None)))
        for grp in (["all60"] + (["dollar34"] if f["in_dollar_estimate"] else [])):
            surv[(grp, "n")] += 1
            surv[(grp, "still_crosses_same_side")] += int(bool(same_side))
            surv[(grp, "same_criterion_rating_display")] += int(bool(same_rating))
            surv[(grp, "same_criterion_raw")] += int(bool(same_raw))
            surv[(grp, "same_moved_cell_set")] += int(bool(same_cells))
            if f["in_dollar_estimate"] and grp == "dollar34" and same_side:
                surv[(grp, "enrollment_surviving")] += int(f["enrollment"] or 0)
                surv[(grp, "mid_usd_abs_surviving")] += int(f["mid_usd_abs"] or 0)
        if not same_side:
            surv_rows.append(dict(year=f["year"], contract_id=f["contract_id"],
                                  in_dollar_estimate=f["in_dollar_estimate"], flip_class=f["flip_class"],
                                  census_rating=f["criterion_rating"],
                                  documented_score_list_rating=(d["criterion_rating"]
                                                            if d else "no 4.0 crossing")))
    census_keys = {(f["year"], f["contract_id"]) for f in out_flips}
    new_doc = sorted([dict(year=k[0], contract_id=k[1], direction=r["direction"],
                       baseline=r["baseline"], criterion_rating=r["criterion_rating"],
                       org_class=r["org_class"])
                      for k, r in flips_doc.items() if k not in census_keys],
                     key=lambda x: (x["year"], x["contract_id"]))
    n_cells_doc = sum(len(v) for v in dsets_doc.values())
    n_cells_stored = sum(len(v) for v in dsets.values())
    B2 = dict(
        register=("FITTED LAYER STRIPPED: every computation's exact DP cut "
                  "points recomputed on the pure documented-rules score list "
                  "(criterion vintage; no synthetic adds, no removals; 41 of "
                  "these score lists do NOT reproduce the published fences), then "
                  "the identical guardrail/display/hold-harmless/aggregation "
                  "path (criterion_verify_suite). Reports which of the 60 "
                  "census flips (34 in the dollar estimate) still cross 4.0 on the same side. "
                  "New crossings that appear only under the stripped score lists "
                  "are listed but neither put through the dual-baseline test nor "
                  "included in the dollar estimate."),
        n_star_cells_moved_stored_score_list=n_cells_stored,
        n_star_cells_moved_documented_score_list=n_cells_doc,
        survival={f"{g}.{k}": v for (g, k), v in sorted(surv.items())},
        flips_not_surviving=surv_rows,
        flips_whose_moved_cell_set_differs=cellset_diff_rows,
        n_computations_dp_cutpoints_changed_vs_stored=sum(
            1 for x in RE["criterion_vintage"]["rows"]
            if not x["dp_cutpoints_equal_stored"]),
        n_new_crossings_not_in_census=len(new_doc),
        new_crossings_not_in_census=new_doc)

    named = {}
    for key in (("2024", "H0524"), ("2024", "H0473")):
        f = next((x for x in out_flips if (x["year"], x["contract_id"]) == key), None)
        named[f"{key[1]}-{key[0]}"] = f

    B = dict(
        register=("Per flip row of runs/criterion_census.json (60 rows; 34 "
                  "in the dollar estimate = qbp_priceable minus the demoted endpoint-"
                  "conditional pair H1304-2024/H5273-2025): moved cells = "
                  "measures where the criterion star (exact DP cut points at "
                  "criterion vintage -> guardrail -> display rounding -> "
                  "disaster hold-harmless, via criterion_verify_suite."
                  "delta_sets) differs from the published star, restricted to "
                  "measures present in the contract's published star row (the "
                  "aggregator's convention). Each cell's computation is "
                  "classed by (A). Ratings re-derived with "
                  "criterion_verify_suite.apply_and_rate and compared to the "
                  "census row."),
        n_flip_rows=len(out_flips), n_in_dollar_estimate=len(v34),
        n_rederived_rating_mismatches=len(mismatch), mismatches=mismatch,
        dollar_estimate_totals=tot34,
        flip_class_aggregates={k: v for k, v in agg_out.items()},
        moved_cells_total=len(cells_all),
        moved_cells_by_computation_class=dict(cell_cls),
        distinct_computations_feeding_flips_by_class={
            k: len(v) for k, v in comp_sets.items()},
        distinct_fitted_computations_feeding_flips=sorted(
            f"{y}/{m}/{o}" for (y, m, o) in comp_sets.get("FITTED", set())),
        n_own_score_not_real_published=len(own_score_anoms),
        own_score_anomalies=own_score_anoms,
        n_moved_cells_own_score_not_in_clustering_score_list=sum(
            1 for c in cells_all if not c["own_score_in_clustering_score_list"]),
        named_examples=named,
        flips=out_flips)

    result = dict(
        generator="src/optimality_without_fills.py",
        register=(
            "COMPUTED: (A) a three-way split of the 123 cut-point computations "
            "by how their score list was obtained — PARAMETER_FREE_STRICT "
            "(published outlier bounds reproduced by the documented exclusion "
            "rules alone, battery before_exact True, AND no synthetic add "
            "score in the score list AND no fitted removal contract present in the "
            "documented score list), REMOVAL_ONLY (fences exact before fitting and "
            "no add, but a fitted removal — applied year-wide by build_jobs — "
            "drops >=1 real contract from the score list), FITTED (a synthetic add "
            "touches it or the fences needed fitting); off-optimum, zero-gap and "
            "fewer-than-five-score counts per class per year on both STORED gap tables, "
            "plus a fresh Ward-vs-exact-DP census on the pure documented-rules "
            "score list (fitted layer stripped) at both vintages, exact SSQ "
            "comparison. (B) for each of the 60 criterion flip rows (34 "
            "in the dollar estimate), the measure cells actually moved (published star != "
            "criterion star) re-derived with the stored verify-suite "
            "machinery, each cell's computation class, the flip class, the "
            "check that the contract's own score in every moved measure is a "
            "real published score (synthetic adds are other, hypothetical "
            "contracts), and enrollment/|mid|-dollar sums of the flips in the dollar estimate "
            "per class. NOT COMPUTED / RESIDUAL: none of this pins down CMS's "
            "actual score list. Even a PARAMETER_FREE_STRICT score list is a "
            "documented-rules reconstruction from the public files; equality of "
            "its two outlier-bound values with the published ones corroborates "
            "but does not prove score list identity (distinct score lists can share "
            "quartiles). Configurations reported are the ones actually tested "
            "(two vintages x 123 computations), not all feasible score lists."),
        input_pins_sha256_16=None,
        membership_layer_summary={
            y: dict(n_adds=L["n_adds"], n_removals=L["n_removals"],
                    n_add_cells=L["n_add_cells"], removals=L["removals"],
                    n_battery_before_exact=sum(1 for v in L["before_exact"].values() if v[0]),
                    n_battery_evaluable=len(L["before_exact"]))
            for y, L in layer.items()},
        removal_scope_note=(
            "enclosure_census_docsonly.build_jobs and exact_gaps_docsonly."
            "build_jobs drop every removal contract of a star year from EVERY "
            "computation of that year in which it has a documented score list "
            "score (rems applied before adds; not scoped to the target that "
            "forced the removal). Hence REMOVAL_ONLY computations exist in "
            "2024 (H0332, H1894) and 2025 (8 contracts); 2026 has no removals."),
        A_classification_and_stored_tallies=A,
        A2_documented_rules_score_list_recompute=RE,
        B_flip_provenance=B,
        B2_flips_with_fitted_layer_stripped=B2,
        dropped_or_capped="none: all 123 computations x 2 vintages recomputed; all 60 flip rows traced",
    )
    pins = {}
    for p in sorted(_OPENED):
        if os.path.abspath(p) == os.path.abspath(OUT):
            continue
        if os.path.isfile(p):
            pins[os.path.relpath(p, BASE)] = sha16(p)
    result["input_pins_sha256_16"] = pins
    with _real_open(OUT, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
        fh.write("\n")
    # console summary
    print(json.dumps({v: A[v]["class_counts_all"] for v in A}, indent=1))
    for v in A:
        print(v, "stored tallies:",
              json.dumps({cls: d.get("ALL") for cls, d in
                          A[v]["stored_table_tallies_per_class"].items()}))
        print(v, "documented-score-list recompute totals:", RE[v]["totals"],
              "strict repro:", RE[v]["strict_class_reproduces_stored_row"])
    print("B aggregates:", json.dumps(B["flip_class_aggregates"], indent=1))
    print("dollar-estimate totals:", tot34, "| mismatches:", len(mismatch),
          "| own-score anomalies:", len(own_score_anoms))
    print("B2 survival:", B2["survival"], "| new crossings:",
          B2["n_new_crossings_not_in_census"])
    for k, f in named.items():
        if f:
            print(k, f["flip_class"], f["n_moved_cells_by_computation_class"],
                  [c["computation"] for c in f["fitted_computations"]])
    print("->", OUT)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else None)
