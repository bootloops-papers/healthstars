# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Resampled rerun under the assumed group-assignment rule at the four
input-order conventions.

The 132-realization sampled census treats SURVEYSELECT GROUPS='s
uniform->GroupID readout as unknown. This driver instead assumes one readout
(front-swap Fisher-Yates over the documented Mersenne-twister stream, seed
8675309; the Technical Notes print the SURVEYSELECT call, seed and group count
but not the readout, so every result below is conditional on that assumption).
Given the assumed rule the resampling-group assignment is a function of the
input order alone; the input-order convention is stated in no Technical Notes
edition.

This driver runs the docsonly census machinery unchanged (same jobs, arms,
guardrails, aggregation) over the four frontswap realizations:
    frontswap            (cid_asc, the baseline dataset-order reading)
    frontswap@cid_desc
    frontswap@score_asc
    frontswap@score_desc
and stores:
  1. the order fingerprint: per-realization pub_match_ward totals (published
     final boundaries matched by the ward-arm replay) vs the stored K=24
     census field;
  2. the flip census: per-realization exact flip sets (each a point statement
     conditional on its order convention), with the four-order envelope as
     the residual range;
  3. the comparison to the stored 132-realization contract set.

Output: runs/enclosure_collapse_frontswap.json
Run:    python3 collapse_census.py pilot [workers]   # 12-job pilot
        python3 collapse_census.py full  [workers]   # all jobs x 4 realizations

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (the resampled replay under the assumed readout at four input-order
    conventions).
Requires: Python >= 3.10; numpy, scipy.
"""
import json
import os
import sys

import enclosure_census_docsonly as C

REALIZATIONS = ["frontswap", "frontswap@cid_desc",
                "frontswap@score_asc", "frontswap@score_desc"]
RANUNI_REALIZATIONS = ["frontswap_ranuni", "frontswap_ranuni@cid_desc",
                       "frontswap_ranuni@score_asc",
                       "frontswap_ranuni@score_desc"]
OUT = os.path.join(C.BASE, "runs", "enclosure_collapse_frontswap.json")
OUT_RANUNI = os.path.join(C.BASE, "runs", "enclosure_collapse_ranuni_branch.json")


def fingerprint(rows):
    """pub_match_ward totals per realization, split by guardrail status."""
    fp = {}
    for rn in REALIZATIONS:
        rr = [r for r in rows if r["realization"] == rn and "error" not in r]
        fp[rn] = {
            "splits": len(rr),
            "pub_match_total": sum(r["pub_match_ward"] for r in rr),
            "pub_match_possible": 4 * len(rr),
            "splits_4of4": sum(1 for r in rr if r["pub_match_ward"] == 4),
            "by_guardrail": {},
        }
        for g in sorted({r["guardrail"] for r in rr}):
            gg = [r for r in rr if r["guardrail"] == g]
            fp[rn]["by_guardrail"][g] = {
                "splits": len(gg),
                "pub_match_total": sum(r["pub_match_ward"] for r in gg),
                "splits_4of4": sum(1 for r in gg if r["pub_match_ward"] == 4),
            }
    return fp


def stored_field_fingerprint():
    """The stored K=24 census's pub_match_ward field, for comparison."""
    path = os.path.join(C.BASE, "runs", "enclosure_census_docsonly.json")
    if not os.path.exists(path):
        return None
    prior = json.load(open(path))
    field = {}
    for r in prior.get("measure_results", []):
        if "error" in r or "pub_match_ward" not in r:
            continue
        f = field.setdefault(r["realization"], {"splits": 0, "pub_match_total": 0,
                                                "splits_4of4": 0})
        f["splits"] += 1
        f["pub_match_total"] += r["pub_match_ward"]
        f["splits_4of4"] += r["pub_match_ward"] == 4
    return field


def main(mode, workers):
    jobs, skips = C.build_jobs()
    if mode == "pilot":
        jobs = jobs[:12]
    if mode == "ranuni":
        # the other branch of the 2024/2025 generator-family two-way
        # (fingerprint only; the SURVEYSELECT default was Mersenne twister for 2026).
        jobs = [j for j in jobs if j["year"] in ("2024", "2025")]
        rows = C.run_measure_phase(jobs, RANUNI_REALIZATIONS, workers)
        fp = {}
        for rn in RANUNI_REALIZATIONS:
            rr = [r for r in rows if r["realization"] == rn and "error" not in r]
            fp[rn] = {"splits": len(rr),
                      "pub_match_total": sum(r["pub_match_ward"] for r in rr),
                      "pub_match_possible": 4 * len(rr),
                      "splits_4of4": sum(1 for r in rr
                                         if r["pub_match_ward"] == 4)}
        # MT-branch comparison on the same 2024/2025 subset, from the full run
        mt = None
        if os.path.exists(OUT):
            full = json.load(open(OUT))
            mt = {}
            for rn in REALIZATIONS:
                rr = [r for r in full["measure_results"]
                      if r["realization"] == rn and "error" not in r
                      and r["year"] in ("2024", "2025")]
                mt[rn] = {"splits": len(rr),
                          "pub_match_total": sum(r["pub_match_ward"] for r in rr),
                          "splits_4of4": sum(1 for r in rr
                                             if r["pub_match_ward"] == 4)}
        result = {
            "register": (
                "RANUNI-branch fingerprint for the 2024/2025 generator-family "
                "two-way: the assumed front-swap readout fed by the legacy "
                "RANUNI (Fishman-Moore) stream instead of the Mersenne "
                "twister; the readout is an assumption on either stream. "
                "Compared against the MT branch on the identical 2024/2025 "
                "splits."),
            "mode": mode, "realizations": RANUNI_REALIZATIONS,
            "n_jobs": len(jobs),
            "fingerprint_ranuni": fp,
            "fingerprint_mt_same_subset": mt,
            "measure_results": rows,
        }
        with open(OUT_RANUNI, "w") as f:
            json.dump(result, f, indent=1, default=str)
            f.write("\n")
        print(json.dumps({"ranuni": fp, "mt_same_subset": mt}, indent=1))
        print(f"-> {OUT_RANUNI}")
        return
    rows = C.run_measure_phase(jobs, REALIZATIONS, workers)
    fp = fingerprint(rows)
    result = {
        "register": (
            "Resampled rerun under the assumed group-assignment rule: front-swap "
            "Fisher-Yates readout of the documented Mersenne-twister stream, "
            "seed 8675309 (the Technical Notes print the SURVEYSELECT call, "
            "seed and group count but not the readout; results that depend on "
            "it are conditional on this assumption). Given the assumed rule "
            "the resampling-group assignment is a function of the input order "
            "alone; the four realizations here are the four input-order "
            "conventions under that one rule. Each realization's flip set is "
            "a point statement conditional on its order convention; the "
            "four-order envelope is the residual range (the order is stated "
            "in no Technical Notes edition). Everything else (jobs, "
            "membership, vintages, guardrails, arms, aggregation) is "
            "identical to the stored docsonly census."),
        "mode": mode,
        "realizations": REALIZATIONS,
        "n_jobs": len(jobs),
        "skipped_splits": skips,
        "fingerprint_frontswap": fp,
        "fingerprint_stored_field": stored_field_fingerprint(),
        "measure_results": rows,
    }
    if mode == "full":
        agg = C.enclosure_aggregate(rows, REALIZATIONS)
        # per-realization exact flip lists (point statements)
        point_sets = {}
        for rn in REALIZATIONS:
            flips = [dict(year=c["year"], contract_id=c["contract_id"],
                          org_class=c["org_class"], direction=c["direction"],
                          baseline=c["baseline"],
                          rating=c["ratings_by_realization"][rn])
                     for c in agg["contracts"]
                     if c["ratings_by_realization"].get(rn) is not None
                     and c["ratings_by_realization"][rn] != c["baseline"]
                     and ((C.F(c["baseline"]) >= 4)
                          != (C.F(c["ratings_by_realization"][rn]) >= 4))]
            point_sets[rn] = flips
        envelope = [dict(year=c["year"], contract_id=c["contract_id"],
                         direction=c["direction"], n_flip=c["n_flip_realizations"],
                         cls=c["class"])
                    for c in agg["contracts"]]
        prior132 = None
        p132 = os.path.join(C.BASE, "runs", "exact_at_132.json")
        if os.path.exists(p132):
            prior132 = json.load(open(p132)).get("exact_at_132")
        result.update(
            aggregation={k: agg[k] for k in ("per_realization",
                                             "per_realization_year",
                                             "errored_splits", "n_rated")},
            contracts=agg["contracts"],
            point_flip_sets=point_sets,
            four_order_envelope=envelope,
            stored_exact_at_132=prior132,
        )
    with open(OUT if mode == "full" else OUT.replace(".json", "_pilot.json"),
              "w") as f:
        json.dump(result, f, indent=1, default=str)
        f.write("\n")
    print(json.dumps(fp, indent=1))
    print(f"-> {OUT if mode == 'full' else OUT.replace('.json', '_pilot.json')}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    assert mode in ("pilot", "full", "ranuni"), mode
    main(mode, workers)
