# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Decomposition record: published cut points -> plain Ward on the full score
list -> the exact optimum on the full score list, all three through the
identical pipeline (name+part guardrail, hold-harmless, display, aggregation,
dual-baseline check, demotion rule), using the stored ward_cutpoints of the
criterion-vintage gap file.
Output: runs/ward_dp_decomposition.json
Run: python3 ward_dp_decomposition.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 (17 of the 34
    bonus-status changes in the dollar estimate already occur under a single
    run of Ward's method on the full score list).
Requires: Python >= 3.10; numpy, scipy.
"""
import hashlib
import json
import os
from fractions import Fraction as F

import criterion_verify_suite as V
import sys
from census_measures import disaster_higher_of
from falsify_groups import round_display
from guardrail import guardrail_boundary

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    gaps, jobs, skips, baseline = V.load_world()

    def delta_sets_arm(arm_key):
        out = {}
        for job in jobs:
            g = gaps.get((job["year"], job["mid"], job["org"]))
            if g is None or len(g.get(arm_key, [])) != 4:
                continue
            vals = [F(v) for v in g[arm_key]]
            if job["gr_note"] == "applied":
                cap = F(job["cap"])
                fin = [guardrail_boundary(v, F(p), cap)[0]
                       for v, p in zip(vals, job["prior"])]
            else:
                fin = vals
            bd = [F(round_display(v, job["dd"])) for v in fin]
            st_pub = baseline[job["year"]]["stars"]
            deltas = []
            part = "C" if job["mid"].startswith("C") else "D"
            for cid, s in job["pool"]:
                sw = st_pub.get(cid, {}).get(job["mid"])
                if sw is None:
                    continue
                sd = V.star_at(F(s), bd, job["higher"])
                sd = disaster_higher_of(job["name"], part, cid, sd,
                                        V._DIS[job["year"]],
                                        V._PRIORMAP[job["year"]])
                if sd != sw:
                    deltas.append((cid, int(sw), int(sd)))
            out[(job["year"], job["mid"], job["org"])] = deltas
        return out

    ward_flips = V.apply_and_rate(baseline, delta_sets_arm("ward_cutpoints"))
    wset = {(r["year"], r["contract_id"]): r["direction"] for r in ward_flips}
    cen = json.load(open(os.path.join(BASE, "runs/criterion_census.json")))
    dol = json.load(open(os.path.join(BASE, "runs/criterion_dollar_estimate.json")))
    dem = {(r["year"], r["contract_id"])
           for r in dol["demoted_endpoint_conditional"]}
    in_estimate = [(p["year"], p["contract_id"], p["direction"], p["enrollment"])
                   for p in cen["flips_criterion"]
                   if p["qbp_priceable"] and (p["year"], p["contract_id"]) not in dem]
    shared = [(y, c) for y, c, dr, e in in_estimate
              if (y, c) in wset and wset[(y, c)] == dr]
    dp_only = [(y, c) for y, c, dr, e in in_estimate
               if (y, c) not in wset or wset[(y, c)] != dr]
    enr = {(y, c): e for y, c, dr, e in in_estimate}
    e_sh = sum(enr[k] for k in shared)
    e_dp = sum(enr[k] for k in dp_only)

    def sha16(rel):
        return hashlib.sha256(
            open(os.path.join(BASE, rel), "rb").read()).hexdigest()[:16]

    out = {
        "register": ("Decomposition of the bonus-status changes in the dollar "
                     "estimate: published cut points, plain Ward on the full "
                     "score list and the exact optimum on the full score "
                     "list, all three through the identical pipeline; "
                     "'shared' are the changes that already occur with plain "
                     "full-score-list Ward cut points against the published "
                     "baseline, 'dp_only' the changes requiring the "
                     "exact-optimum step."),
        "generator": "src/ward_dp_decomposition.py",
        "n_ward_full_candidates": len(ward_flips),
        "in_dollar_estimate_34_decomposition": {
            "already_flip_under_plain_ward": len(shared), "enrollment": e_sh,
            "require_the_optimal_step": len(dp_only), "enrollment_dp": e_dp,
            "dp_only_rows": sorted(f"{y}-{c}" for y, c in dp_only),
            "dollars_mid_dp_only_usd": e_dp * 500,
        },
        "input_pins_sha256_16": {
            p: sha16(p) for p in ("runs/exact_gaps_criterionvintage.json",
                                  "runs/criterion_census.json")},
    }
    p = os.path.join(BASE, "runs/ward_dp_decomposition.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print("shared", len(shared), e_sh, "| dp_only", len(dp_only), e_dp)
    print("->", p, sha16("runs/ward_dp_decomposition.json"))


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
