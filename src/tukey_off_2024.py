# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Outlier-deletion-off 2024 counter-census: how much of the 2023->2024 published
cut-point movement is attributable to the documented introduction of outlier
deletion in SY2024 (vs distribution change / group-assignment spread)?

Method:
  * Re-run the 2024 per-measure pre-guardrail mean cut points with outlier
    deletion disabled (pipeline.measure_cutpoints tukey=False) for the 35 name-matched
    2023->2024 identities (runs/sampled_realizations_2023_2024.json pairs), ward_tie arm
    (published-side replay), reconciled docsonly membership, the same 16
    cid_asc realizations as the sampled-realization censuses
    (enclosure_census_docsonly.REALIZATIONS16, seed 8675309).
  * Outlier-deletion-on side: the stored resampled-rerun census means
    (runs/enclosure_census_docsonly.json ward_mean) — same jobs, same
    realizations; only the tukey flag differs. Integrity check: two splits x
    16 realizations are recomputed tukey=True in the pilot and must reproduce
    the stored ward_mean Fraction-strings exactly.
  * Decomposition per boundary (35 identities x 4 boundaries, exact
    Fractions until output):
      deletion effect  T = median over the 16 same-realization pairs of
                    (mean_cp_on(r) - mean_cp_off(r)), display-rounded at
                    the 2024 published precision (the same-realization
                    pairing cancels the group-assignment spread within the
                    pair);
      residual      R = published 2023->2024 move (pre-guardrail precision
                    where the guardrail bound, as stored) - T;
      null          the stored 2023-based same-year cross-realization null
                    (recomputed exactly from runs/sampled_realizations_2023.json,
                    cross-checked against the stored row verdicts).
    R is then tested against the null: |R| <= max|D_2023| (and q95) — the
    group-assignment-spread plus distribution-change residual after removing
    the measured outlier-deletion component.

Scope: replay-based attribution on a sampled family. The effect is measured
on the replay pipeline (ward_tie, docsonly membership, 16 sampled realizations
of the undocumented GROUPS= family), not on CMS's record: CMS never computed a
2024 without outlier deletion. The replay family carries the documented
family-choice bias (stage 2: no candidate matches CMS exactly); the
same-realization on/off pairing removes the group-assignment spread from the
estimate but not the family bias. 16 realizations sample the family; the
2023-based null is a measured lower bound on the spread support. 2023
membership is hypothesis-grade (no outlier-bound instrument exists in 2023).

Run:  python3 tukey_off_2024.py pilot   [workers]  # 2 splits x 16, off and on
                                                   # (integrity check)
      python3 tukey_off_2024.py full    [workers]  # all 35 splits (checkpointed)
      python3 tukey_off_2024.py analyze            # decomposition -> record
Output: runs/tukey_decomposition_2023_2024.json
        (+ runs/tukey_off_2024_pilot.json, runs/tukey_off_2024.ckpt.jsonl)

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (star year 2023
    had no outlier deletion; how much of the 2023-2024 cut-point
    movement its introduction explains; a record-only check).
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import statistics
import sys
from collections import defaultdict
from fractions import Fraction as F

import enclosure_census_docsonly as ec        # registers urand_0..6
from falsify_groups import round_display
from pipeline import measure_cutpoints

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs")
PAIRS_PATH = os.path.join(RUNS, "sampled_realizations_2023_2024.json")
BANK24_PATH = os.path.join(RUNS, "enclosure_census_docsonly.json")
BANK23_PATH = os.path.join(RUNS, "sampled_realizations_2023.json")
CKPT = os.path.join(RUNS, "tukey_off_2024.ckpt.jsonl")
OUT = os.path.join(RUNS, "tukey_decomposition_2023_2024.json")
OUT_PILOT = os.path.join(RUNS, "tukey_off_2024_pilot.json")

R16 = ec.REALIZATIONS16
SEED = ec.SEED

# pilot splits: one C-side ALL (typical n) + one D-side MA-PD (large n);
# chosen from the paired set, integrity-checked tukey=ON vs the bank.
PILOT_SPLITS = [("C01", "ALL"), ("D08", "MA-PD")]


def load_pairs():
    w = json.load(open(PAIRS_PATH))
    return w["pairs"], w        # [name, side, org, mid23, mid24]


def paired_jobs():
    """2024 build_jobs rows restricted to the 35 paired identities; the pool
    is stripped (no star-delta stage here)."""
    pairs, _ = load_pairs()
    need = {(p[4], p[2]) for p in pairs}
    jobs, _ = ec.build_jobs()
    out = [dict(j, pool=None) for j in jobs
           if j["year"] == "2024" and (j["mid"], j["org"]) in need]
    got = {(j["mid"], j["org"]) for j in out}
    missing = need - got
    if missing:
        raise RuntimeError(f"paired 2024 splits missing from build_jobs: "
                           f"{sorted(missing)}")
    return out, pairs


def _worker(task):
    """(job, rname, also_on) -> outlier-deletion-off ward_tie mean cut points
    (+ deletion-on when also_on, for the integrity check against the stored
    census). Errors are returned as rows."""
    job, rname, also_on = task
    out = {"year": job["year"], "mid": job["mid"], "org": job["org"],
           "name": job["name"], "realization": rname,
           "n_cluster": len(job["kept"]), "guardrail": job["gr_note"]}
    try:
        scores = [s for _, s in job["kept"]]          # cid_asc dataset order
        modes = [("ward_mean_off", False)] + ([("ward_mean_on", True)]
                                              if also_on else [])
        for field, tk in modes:
            r = measure_cutpoints(scores, higher_is_better=job["higher"],
                                  tukey=tk, cap_lo=job["cap_lo"],
                                  cap_hi=job["cap_hi"], groups_candidate=rname,
                                  seed=SEED, n_groups=ec.N_GROUPS,
                                  arm="ward_tie")
            if "error" in r:
                out["error"] = f"tukey={tk}: {r['error']}"
                break
            if len(r["mean_cutpoints"]) != 4:
                out["error"] = (f"tukey={tk}: cutpoint count "
                                f"{len(r['mean_cutpoints'])} != 4")
                break
            out[field] = [str(v) for v in r["mean_cutpoints"]]
    except Exception as e:                            # reported as a row
        out["error"] = f"exception: {e!r}"
    return out


def _load_ckpt():
    rows = []
    if os.path.exists(CKPT):
        for ln in open(CKPT):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    keys = {(r["mid"], r["org"], r["realization"]) for r in rows}
    if len(keys) != len(rows):
        raise RuntimeError(f"duplicate checkpoint rows: {len(rows)} rows, "
                           f"{len(keys)} keys")
    return rows


def run_compute(mode, workers):
    jobs, pairs = paired_jobs()
    if mode == "pilot":
        jobs = [j for j in jobs if (j["mid"], j["org"]) in PILOT_SPLITS]
        assert len(jobs) == 2, [(j["mid"], j["org"]) for j in jobs]
    rows = _load_ckpt()
    done = {(r["mid"], r["org"], r["realization"]) for r in rows}
    pilot_set = set(PILOT_SPLITS)
    tasks = [(j, rn, (j["mid"], j["org"]) in pilot_set and mode == "pilot")
             for j in jobs for rn in R16
             if (j["mid"], j["org"], rn) not in done]
    print(f"{mode}: {len(jobs)} splits x {len(R16)} realizations, "
          f"{len(rows)} stored, {len(tasks)} tasks",
          flush=True)
    new_rows = []
    if tasks:
        ctx = mp.get_context("spawn")     # fork can deadlock with BLAS thread pools
        with open(CKPT, "a") as ck:
            with ctx.Pool(workers) as pool:
                for n, r in enumerate(pool.imap_unordered(_worker, tasks,
                                                          chunksize=1)):
                    new_rows.append(r)
                    ck.write(json.dumps(r) + "\n")    # parent = single writer
                    ck.flush()
                    if (n + 1) % 50 == 0:
                        print(f"  {n + 1}/{len(tasks)}", flush=True)
    rows += new_rows
    errs = [r for r in rows if "error" in r]
    print(f"{mode}: {len(new_rows)} new rows, errors {len(errs)}", flush=True)

    if mode == "pilot":
        # -------- integrity check: tukey=ON recompute vs stored census means
        bank = json.load(open(BANK24_PATH))
        bidx = {(r["mid"], r["org"], r["realization"]): r
                for r in bank["measure_results"] if r["year"] == "2024"}
        mism, checked = [], 0
        for r in rows:
            if "ward_mean_on" not in r:
                continue
            checked += 1
            stored = bidx[(r["mid"], r["org"], r["realization"])]["ward_mean"]
            if r["ward_mean_on"] != stored:
                mism.append({"key": [r["mid"], r["org"], r["realization"]],
                             "fresh_on": r["ward_mean_on"], "stored": stored})
        record = {
            "mode": "pilot", "splits": PILOT_SPLITS,
            "n_rows": len(new_rows), "n_errors": len(errs),
            "integrity_tukey_on_vs_bank": {
                "checked": checked, "expected": 32, "mismatches": mism,
                "pass": checked == 32 and not mism},
        }
        json.dump(record, open(OUT_PILOT, "w"), indent=1)
        print(f"integrity tukey=True vs stored census: checked={checked}/32 "
              f"mismatches={len(mism)}", flush=True)
        print(f"saved {OUT_PILOT}")
        if mism:
            raise SystemExit("INTEGRITY FAILURE: tukey=True recompute does not "
                             "reproduce the stored census means")
        return record
    return rows


# -------------------------------------------------------------------- analyze
def _median16(vals):
    s = sorted(vals)
    assert len(s) == 16
    return (s[7] + s[8]) / 2


def analyze():
    pairs, w = load_pairs()
    wrows = {(r["measure"], r["org"], r["boundary_index"]): r
             for r in w["rows"] if r["arm"] == "ward_tie"}
    assert len(wrows) == 140, len(wrows)

    bank24 = json.load(open(BANK24_PATH))
    on_idx = {(r["mid"], r["org"], r["realization"]): r
              for r in bank24["measure_results"] if r["year"] == "2024"}
    bank23 = json.load(open(BANK23_PATH))
    idx23 = {(r["mid"], r["org"], r["realization"]): r
             for r in bank23["measure_results"]}
    off_rows = [r for r in _load_ckpt() if "error" not in r]
    off_idx = {(r["mid"], r["org"], r["realization"]): r for r in off_rows}

    rows, crosscheck_fail = [], []
    for name, side, org, mid23, mid24 in pairs:
        for rn in R16:
            for idx, key, what in ((off_idx, (mid24, org, rn), "tukey-off"),
                                   (on_idx, (mid24, org, rn), "bank-on"),
                                   (idx23, (mid23, org, rn), "bank-2023")):
                if key not in idx:
                    raise RuntimeError(f"{what} row missing: {key}")
        for i in range(4):
            wr = wrows[(name, org, i)]
            assert wr["mids"] == [mid23, mid24], (name, org, wr["mids"])
            ddA, ddB = wr["display_decimals"]
            # exact published movement on the stored register choice
            fA, fB = wr["published_final"]
            pA, pB = wr["published_pre_guardrail"]
            if wr["move_register_used"] == "pre-guardrail":
                mv_eff = F(pB) - F(pA)
            else:
                mv_eff = F(fB) - F(fA)
            assert abs(float(mv_eff) - wr["published_move_used"]) < 1e-9

            on_r = {rn: F(on_idx[(mid24, org, rn)]["ward_mean"][i])
                    for rn in R16}
            off_r = {rn: F(off_idx[(mid24, org, rn)]["ward_mean_off"][i])
                     for rn in R16}
            t_raw = [on_r[rn] - off_r[rn] for rn in R16]
            t_rnd = [F(round_display(on_r[rn], ddB))
                     - F(round_display(off_r[rn], ddB)) for rn in R16]
            t_med_raw = _median16(t_raw)
            t_med = _median16(t_rnd)

            # 2023-based null recomputed exactly (240 ordered pairs, rounded
            # at the 2023 register), cross-checked vs the stored verdict
            v23 = [F(idx23[(mid23, org, rn)]["ward_mean"][i]) for rn in R16]
            r23 = [F(round_display(v, ddA)) for v in v23]
            D = sorted(abs(x - y) for j, x in enumerate(r23)
                       for k, y in enumerate(r23) if j != k)
            nmax, nq95 = D[-1], D[227]
            within_pub = abs(mv_eff) <= nmax
            if within_pub != wr["within_wobble_null"]:
                crosscheck_fail.append([name, org, i, "within_null",
                                        within_pub, wr["within_wobble_null"]])

            residual = mv_eff - t_med
            share_signed = (float(t_med / mv_eff) if mv_eff != 0 else None)
            share_capped = (min(1.0, max(0.0, share_signed))
                            if share_signed is not None else None)
            rows.append({
                "measure": name, "side": side, "org": org,
                "mids": [mid23, mid24], "boundary_index": i,
                "boundary_stars": wr["boundary_stars"],
                "display_decimals": [ddA, ddB],
                "guardrail_bound": wr["guardrail_bound"],
                "move_register_used": wr["move_register_used"],
                "published_move_used": float(mv_eff),
                "published_move_used_exact": str(mv_eff),
                "tukey_effect_median_rounded": float(t_med),
                "tukey_effect_median_rounded_exact": str(t_med),
                "tukey_effect_median_raw": float(t_med_raw),
                "tukey_effect_range_rounded": [float(min(t_rnd)),
                                               float(max(t_rnd))],
                "tukey_effect_range_raw": [float(min(t_raw)),
                                           float(max(t_raw))],
                "residual_move": float(residual),
                "residual_move_exact": str(residual),
                "null_max_abs_rounded_2023": float(nmax),
                "null_q95_abs_rounded_2023": float(nq95),
                "published_within_null": bool(within_pub),
                "published_within_null_q95": bool(abs(mv_eff) <= nq95),
                "residual_within_null": bool(abs(residual) <= nmax),
                "residual_within_null_q95": bool(abs(residual) <= nq95),
                "tukey_share_signed": share_signed,
                "tukey_share_capped": share_capped,
            })
    if crosscheck_fail:
        raise SystemExit(f"CROSS-CHECK FAILURE vs stored pair analysis: "
                         f"{crosscheck_fail[:5]}")

    # ------------------------------------------------------------- summaries
    nz = [r for r in rows if r["published_move_used"] != 0]
    def frac(n, d):
        return round(n / d, 3) if d else None
    tmags = sorted(abs(r["tukey_effect_median_rounded"]) for r in rows)
    tdist = {"zero": sum(1 for t in tmags if t == 0),
             "0_to_1": sum(1 for t in tmags if 0 < t <= 1),
             "1_to_2": sum(1 for t in tmags if 1 < t <= 2),
             "over_2": sum(1 for t in tmags if t > 2),
             "max": tmags[-1], "median": statistics.median(tmags)}
    shares = sorted(r["tukey_share_capped"] for r in nz)
    n_pub_full = sum(1 for r in nz if r["published_within_null"])
    n_pub_q95 = sum(1 for r in nz if r["published_within_null_q95"])
    n_res_full = sum(1 for r in nz if r["residual_within_null"])
    n_res_q95 = sum(1 for r in nz if r["residual_within_null_q95"])
    entered = [r for r in nz
               if r["residual_within_null"] and not r["published_within_null"]]
    left = [r for r in nz
            if r["published_within_null"] and not r["residual_within_null"]]

    fam = defaultdict(list)
    for r in rows:
        fam[(r["measure"], r["side"], r["org"])].append(r)
    family_table = []
    for (name, side, org), rr in sorted(fam.items()):
        ts = [abs(r["tukey_effect_median_rounded"]) for r in rr]
        family_table.append({
            "measure": name, "side": side, "org": org,
            "max_abs_tukey_effect_rounded": max(ts),
            "median_abs_tukey_effect_rounded": statistics.median(ts),
            "max_abs_published_move": max(abs(r["published_move_used"])
                                          for r in rr),
            "n_nonzero_moves": sum(1 for r in rr
                                   if r["published_move_used"] != 0),
            "n_residual_within_null": sum(r["residual_within_null"]
                                          for r in rr
                                          if r["published_move_used"] != 0),
        })
    family_table.sort(key=lambda f: -f["max_abs_tukey_effect_rounded"])

    big = sorted((r for r in nz if abs(r["published_move_used"]) >= 20),
                 key=lambda r: -abs(r["published_move_used"]))
    big_movers = [{k: r[k] for k in
                   ("measure", "org", "boundary_stars", "published_move_used",
                    "tukey_effect_median_rounded", "residual_move",
                    "null_max_abs_rounded_2023", "tukey_share_capped",
                    "residual_within_null")} for r in big]

    pilot = (json.load(open(OUT_PILOT)) if os.path.exists(OUT_PILOT) else None)
    ck_rows = _load_ckpt()
    out = {
        "register": (
            "Replay-based attribution on a sampled family: the outlier-"
            "deletion component of each 2023->2024 published boundary move "
            "is the median over 16 same-realization pairs of the 2024 "
            "ward_tie mean cut point with outlier deletion on minus off, "
            "display-rounded at the 2024 published precision; the on side "
            "is the stored resampled-rerun census (docsonly membership), the "
            "off side a recompute identical except pipeline tukey=False, "
            "and the residual (published move minus the deletion component) "
            "is tested against the stored 2023-based same-year "
            "cross-realization null; this is not an attribution on CMS's "
            "record, since CMS never computed a 2024 without outlier "
            "deletion, the replay family carries the documented "
            "family-choice bias, the 16 realizations sample the undocumented "
            "GROUPS= family, 2023 membership is hypothesis-grade, and the "
            "movement is measured pre-guardrail where the guardrail bound in "
            "either year and at the published final otherwise."),
        "sources": {"pairs_and_null": "runs/sampled_realizations_2023_2024.json",
                    "tukey_on_2024": "runs/enclosure_census_docsonly.json",
                    "bank_2023": "runs/sampled_realizations_2023.json",
                    "tukey_off_2024": "runs/tukey_off_2024.ckpt.jsonl"},
        "realizations": R16, "seed": SEED, "arm": "ward_tie",
        "n_identities": len(pairs),
        "n_boundaries": len(rows),
        "pilot_record": pilot,
        "checkpoint_rows": {"n_off_rows": len([r for r in ck_rows
                                               if "error" not in r]),
                            "n_errors": len([r for r in ck_rows
                                             if "error" in r])},
        "summary": {
            "n_nonzero_published_moves": len(nz),
            "published_within_null_full": n_pub_full,
            "published_within_null_q95": n_pub_q95,
            "published_frac_within_null_full": frac(n_pub_full, len(nz)),
            "residual_within_null_full": n_res_full,
            "residual_within_null_q95": n_res_q95,
            "residual_frac_within_null_full": frac(n_res_full, len(nz)),
            "residual_frac_within_null_q95": frac(n_res_q95, len(nz)),
            "n_entered_null_after_tukey_removal": len(entered),
            "n_left_null_after_tukey_removal": len(left),
            "left_null_rows": [{k: r[k] for k in
                                ("measure", "org", "boundary_stars",
                                 "published_move_used",
                                 "tukey_effect_median_rounded",
                                 "residual_move")} for r in left],
            "tukey_effect_magnitude_distribution_rounded_140": tdist,
            "tukey_share_capped_quartiles_nonzero_moves":
                [shares[0], shares[len(shares) // 4],
                 statistics.median(shares), shares[3 * len(shares) // 4],
                 shares[-1]],
            "n_share_ge_half": sum(1 for s in shares if s >= 0.5),
            "n_share_full": sum(1 for s in shares if s == 1.0),
            "big_movers_ge_20_units": big_movers,
        },
        "family_table": family_table,
        "rows": rows,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    s = out["summary"]
    print(f"saved {OUT}")
    print(f"nonzero moves: {len(nz)}/140")
    print(f"within 2023-null full: published {n_pub_full} "
          f"({s['published_frac_within_null_full']}) -> residual "
          f"{n_res_full} ({s['residual_frac_within_null_full']}); "
          f"q95 {n_pub_q95} -> {n_res_q95}")
    print(f"entered null after removing the deletion component: {len(entered)}, left: "
          f"{len(left)}")
    print(f"outlier-deletion |effect| dist (140 boundaries): {tdist}")
    print(f"share quartiles (capped): "
          f"{s['tukey_share_capped_quartiles_nonzero_moves']}")
    print("families most moved by outlier deletion:")
    for f_ in family_table[:8]:
        print(f"  {f_['measure'][:52]:52} {f_['org']:>5} "
              f"maxT={f_['max_abs_tukey_effect_rounded']:5.2f} "
              f"maxMove={f_['max_abs_published_move']:5.2f}")
    return out


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode in ("pilot", "full"):
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 1)
        run_compute(mode, workers)
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode}")
