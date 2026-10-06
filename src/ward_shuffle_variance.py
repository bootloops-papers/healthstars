# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Shuffle variance, two arms.

Arm A: the published Ward-within-groups procedure. Arm B: the exact solver
inside the same ten groups (the stored dp arm). Each is replayed at the 28
sampled orderings (stored ward_final/dp_final boundaries) and compared with
itself across orderings: a lottery row is a contract-year in the dollar
estimate whose bonus status varies across the orderings. No optimality claim;
dollar-estimate scope (published overall rating exists, org type not Cost);
estimate-class dollars ($400-600 per enrollee); sampled (N=28; classes can
only grow).
Output: runs/ward_shuffle_variance.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (expected payments
    at stake per draw over the 28 sampled orders: $787.2 million with Ward's
    method, $98.4 million with the exact optimum).
Run:  cd src && python3 ward_shuffle_variance.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""
import csv
import hashlib
import json
import os
from fractions import Fraction as F

import enclosure_census_docsonly as C
import criterion_verify_suite as V
import sys

BASE = C.BASE
ARMS = {"ward": "ward_final", "dp": "dp_final"}
INPUTS = [
    "runs/enclosure_collapse_frontswap_refinal.json",
    "runs/enclosure_collapse_frontswap_rperm_refinal.json",
    "runs/exact_gaps_fullsample_docsonly.json",
    "data/parsed/summary_2024_recalc.csv",
    "data/parsed/summary_2025_current.csv",
    "data/parsed/summary_2026_current.csv",
    "data/raw/enrollment/Monthly_Report_By_Contract_2025_12/Monthly_Report_By_Contract_2025_12.csv",
    "data/raw/enrollment/Monthly_Report_By_Contract_2026_07/Monthly_Report_By_Contract_2026_07.csv",
]


def sha16(rel):
    return hashlib.sha256(open(os.path.join(BASE, rel), "rb").read()).hexdigest()[:16]


def main():
    b = json.load(open(os.path.join(BASE, INPUTS[0])))
    r = json.load(open(os.path.join(BASE, INPUTS[1])))
    allrows = b["measure_results"] + r["measure_results"]
    err = {(x["year"], x["mid"], x["org"]) for x in allrows if "error" in x}
    rows = [x for x in allrows if "error" not in x
            and (x["year"], x["mid"], x["org"]) not in err]
    reals = sorted({x["realization"] for x in rows})
    assert len(reals) == 28

    # replay inputs built directly: V.load_world serves the criterion
    # computation (CRIT_VINTAGE jobs); the shuffle pair replays the
    # resampled computation and keeps its score vintages.
    jobs, skips = C.build_jobs()
    jobmap = {(j["year"], j["mid"], j["org"]): j for j in jobs}
    from aggregate import load_inputs as _li
    baseline = {}
    for _y in C.YEARS:
        _st, _, _ = _li(_y, C.BASELINE_VINTAGE[_y])
        baseline[_y] = dict(stars=_st)
    from census_measures import (disaster_pcts as _dp,
                                 prior_published_star_map as _pm,
                                 disaster_higher_of as _hh)
    _dis = {y: _dp(y, C.BASELINE_VINTAGE[y]) for y in C.YEARS}
    _pmaps = {y: _pm(y) for y in C.YEARS}

    synth = []
    for x in rows:
        job = jobmap.get((x["year"], x["mid"], x["org"]))
        if job is None:
            continue
        st_pub = baseline[x["year"]]["stars"]
        for arm, fld in ARMS.items():
            bd = [F(v) for v in x[fld]]
            deltas = []
            _part = "C" if x["mid"].startswith("C") else "D"
            for cid, s in job["pool"]:
                sw = st_pub.get(cid, {}).get(x["mid"])
                if sw is None:
                    continue
                sd = V.star_at(F(s), bd, job["higher"])
                sd = _hh(job["name"], _part, cid, sd, _dis[x["year"]],
                         _pmaps[x["year"]])
                if sd != sw:
                    deltas.append((cid, int(sw), int(sd)))
            synth.append(dict(year=x["year"], mid=x["mid"], org=x["org"],
                              name=x["name"],
                              realization=f"{arm}@{x['realization']}",
                              deltas=deltas, n_star_changes=len(deltas)))

    agg = C.enclosure_aggregate(synth, [f"{a}@{rn}" for a in ARMS for rn in reals])

    summ, emap = {}, {}
    for y, v in (("2024", "recalc"), ("2025", "current"), ("2026", "current")):
        with open(os.path.join(BASE, f"data/parsed/summary_{y}_{v}.csv"),
                  newline="", encoding="utf-8-sig") as fh:
            summ[y] = {row["contract_id"]: row for row in csv.DictReader(fh)}
    for y, rel in (("2024", INPUTS[6]), ("2025", INPUTS[7]), ("2026", INPUTS[7])):
        with open(os.path.join(BASE, rel), newline="", encoding="utf-8-sig") as fh:
            emap[y] = {}
            for row in csv.DictReader(fh):
                vv = row["Enrollment"].replace(",", "")
                emap[y][row["Contract Number"]] = int(vv) if vv.isdigit() else 0

    def arm_results(arm):
        lottery, per_real = [], {rn: 0 for rn in reals}
        for c in agg["contracts"]:
            year, cid = c["year"], c["contract_id"]
            srow = summ[year].get(cid, {})
            ov = srow.get("overall_raw", "").strip()
            priceable = (c["qbp_relevant"] and ov.replace(".", "").isdigit()
                         and "Cost" not in srow.get("org_type", ""))
            stats = {}
            for rn in reals:
                val = c["ratings_by_realization"].get(f"{arm}@{rn}")
                if val is not None:
                    stats[rn] = F(val) >= 4
            if not stats:
                continue
            bl = F(c["baseline"]) >= 4
            for rn, s in stats.items():
                if s != bl:
                    per_real[rn] += 1
            if len(set(stats.values())) > 1 and priceable:
                e = emap[year].get(cid, 0)
                k, N = sum(stats.values()), len(stats)
                lottery.append(dict(year=year, contract_id=cid, enrollment=e,
                                    published=ov, bonus_in=f"{k}/{N}",
                                    k=k, N=N, baseline=c["baseline"]))
        enr = sum(x["enrollment"] for x in lottery)
        E = {p: sum(x["enrollment"] * v * min(x["k"], x["N"] - x["k"]) / x["N"]
                    for x in lottery)
             for p, v in (("lo", 400), ("mid", 500), ("hi", 600))}
        coin_incl = [x for x in lottery if 0.25 <= x["k"] / x["N"] <= 0.75]
        coin_open = [x for x in lottery if 0.25 < x["k"] / x["N"] < 0.75]
        zero_ey = [dict(year=x["year"], contract_id=x["contract_id"],
                        bonus_in=x["bonus_in"]) for x in lottery
                   if x["enrollment"] == 0]
        return dict(
            n_lottery_priceable=len(lottery),
            lottery_enrollment=enr,
            lottery_gross_envelope_usd={p: enr * v for p, v in
                                        (("lo", 400), ("mid", 500), ("hi", 600))},
            expected_per_draw_deviation_usd=dict(
                formula=("E = sum over lottery rows of enrollment x price x "
                         "min(k, N-k)/N; ASSUMES the N sampled orderings are "
                         "equiprobable; SAMPLED (N=28), not a population "
                         "quantity"),
                **E),
            n_near_coinflip_inclusive_25_75=len(coin_incl),
            n_near_coinflip_open_interval=len(coin_open),
            near_coinflip_convention=("headline count uses the INCLUSIVE "
                                      "[25%, 75%] convention; open-interval "
                                      "count carried beside it"),
            zero_enrollment_lottery_rows=zero_ey,
            flips_vs_published_per_ordering=dict(
                min=min(per_real.values()), max=max(per_real.values())),
            lottery_contract_years=sorted(lottery,
                                          key=lambda x: -x["enrollment"]),
        )

    arm_A = arm_results("ward")
    arm_B = arm_results("dp")
    out = {
        "register": ("Shuffle variance, two arms: arm A is the published "
                     "Ward-within-groups procedure and arm B the exact solver "
                     "inside the same ten groups, each replayed at the 28 "
                     "sampled orderings and compared with itself across "
                     "orderings, with no optimality claim, estimate-class "
                     "dollars, a sampled set of orderings whose classes can "
                     "only grow, and conditional on the assumed "
                     "group-assignment rule of the paper's Appendix C (see "
                     "README, 'Assumed readout')."),
        "generator": "src/ward_shuffle_variance.py",
        "input_pins_sha256_16": {p: sha16(p) for p in INPUTS},
        "n_realizations": 28,
        "arm_A_ward_within_groups": arm_A,
        "arm_B_exact_within_groups": arm_B,
        "pair_summary": {
            "E_mid_ward": arm_A["expected_per_draw_deviation_usd"]["mid"],
            "E_mid_exact": arm_B["expected_per_draw_deviation_usd"]["mid"],
            "ratio_exact_over_ward": (
                arm_B["expected_per_draw_deviation_usd"]["mid"]
                / arm_A["expected_per_draw_deviation_usd"]["mid"]),
        },
    }
    p = os.path.join(BASE, "runs/ward_shuffle_variance.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    open(p, "a").write("\n")
    print(json.dumps(out["pair_summary"], indent=1))
    for nm, a in (("WARD ", arm_A), ("EXACT", arm_B)):
        E = a["expected_per_draw_deviation_usd"]
        print(f"{nm}: lottery {a['n_lottery_priceable']:>3} rows / "
              f"{a['lottery_enrollment']:>9,} EY | E/draw mid ${E['mid']/1e6:>5.0f}M "
              f"[{E['lo']/1e6:.0f}, {E['hi']/1e6:.0f}] | coinflip "
              f"{a['n_near_coinflip_inclusive_25_75']} "
              f"(open {a['n_near_coinflip_open_interval']}) | "
              f"zero-EY {len(a['zero_enrollment_lottery_rows'])}")
    print("->", p)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(os.path.join(BASE, rel)) for rel in INPUTS):
    print("ward_shuffle_variance: missing input(s): " + ", ".join(rel for rel in INPUTS
                                                          if not os.path.exists(os.path.join(BASE, rel)))
          + " (the enrollment files are raw CMS downloads, not included; re-fetch each by the URL and "
          "sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
