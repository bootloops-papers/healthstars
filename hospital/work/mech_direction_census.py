#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""mech_direction_census.py — per-group-year score skewness and direction
structure of the census deltas, all six years x three peer groups, from the
stored replay score outputs and the stored exact-optimum censuses.

Skewness estimator: adjusted Fisher-Pearson standardized moment coefficient
G1 = g1 * sqrt(n(n-1))/(n-2),  g1 = m3 / m2^(3/2)  with biased sample moments
m_k = mean((x-xbar)^k). Both g1 (the "moment skew" quoted in
runs/mechanism_allupward.json) and G1 are stored.

Direction per group-year from the stored exact_census moves (exact optimum
vs published): all-up / all-down / mixed. The canonical universe is 18
group-years (6 years x 3 peer groups, one census per year, 2026 pre-cap); a
count of 21 arises only by also counting the 2026 group-years at a second
register (post-cap). Both tallies are stored.

Writes runs/mechanism_direction_census.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.3 (mechanism and
    direction of the differences).
Run:  cd hospital/work && python3 mech_direction_census.py
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json, math

BASE = _REC + "/hospital"
RUNS = f"{BASE}/runs"

def skew(xs):
    n = len(xs)
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    m3 = sum((x - mean) ** 3 for x in xs) / n
    g1 = m3 / m2 ** 1.5
    G1 = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    return g1, G1

def moves_updown(moves):
    up = down = 0
    for k, v in moves.items():
        a, b = k.split("->")
        if int(b) > int(a):
            up += v
        else:
            down += v
    return up, down

# ---- scores per group-year ----
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

scores = {}   # (year, peer) -> list of floats
for year, tag in [("2021", "2021-04"), ("2022", "2022-07"), ("2023", "2023-07"),
                  ("2024", "2024-07"), ("2025", "2025-07")]:
    with open(f"{BASE}/work/sas_replay_out/star_{tag}.csv") as f:
        for r in csv.DictReader(f):
            if r["star"] in ("NA", ""):
                continue
            peer = f"peer{r['Total_measure_group_cnt']}"
            scores.setdefault((year, peer), []).append(float(r["summary_score"]))
with open(f"{BASE}/work/R_output/fullprec_2026apr.csv") as f:
    for r in csv.DictReader(f):
        if r["star_precap"] in ("NA", ""):
            continue
        peer = f"peer{r['Total_measure_group_cnt']}"
        scores.setdefault(("2026", peer), []).append(float(r["summary_score_17g"]))

# ---- direction per group-year from the stored censuses ----
rows = []
for year in ["2021", "2022", "2023", "2024", "2025", "2026"]:
    cc = json.load(open(f"{RUNS}/exact_census_{year}.json"))
    for peer in ["peer3", "peer4", "peer5"]:
        pg = cc["peer_groups"][peer]
        xs = scores[(year, peer)]
        assert len(xs) == pg["n"], (year, peer, len(xs), pg["n"])
        g1, G1 = skew(xs)
        up, down = moves_updown(pg["moves"])
        assert up + down == pg["hospitals_star_differs"]
        direction = ("all-up" if down == 0 else
                     "all-down" if up == 0 else "mixed")
        rows.append({"year": year, "peer": peer, "n": pg["n"],
                     "skew_g1_moment": round(g1, 6),
                     "skew_G1_adjusted_fisher_pearson": round(G1, 6),
                     "left_skewed": g1 < 0,
                     "differs": pg["hospitals_star_differs"],
                     "up": up, "down": down, "direction": direction,
                     "moves": pg["moves"]})

# 2026 post-cap register group-year rows (second register; NOT in canonical count)
cap = json.load(open(f"{RUNS}/safety_cap_check_2026.json"))
cc26 = json.load(open(f"{RUNS}/exact_census_2026.json"))
erased = set(cap["postcap_differ_census"]["differs_erased_by_cap"])
postcap_rows = []
for peer in ["peer3", "peer4", "peer5"]:
    ids = [h for h in cc26["star_differs_detail"] if h["peer"] == peer]
    kept = [h for h in ids if h["provider_id"] not in erased]
    up = sum(1 for h in kept if h["dp_optimum"] > h["shipped"])
    down = len(kept) - up
    postcap_rows.append({"year": "2026", "peer": peer, "register": "post-cap",
                         "differs": len(kept), "up": up, "down": down,
                         "direction": "all-up" if down == 0 else
                                      "all-down" if up == 0 else "mixed"})

def tally(rr):
    c = {"all-up": 0, "all-down": 0, "mixed": 0}
    for r in rr:
        c[r["direction"]] += 1
    n = len(rr)
    return {"group_years": n, **c, "one_signed": c["all-up"] + c["all-down"],
            "one_signed_over_total": f"{c['all-up'] + c['all-down']}/{n}"}

canonical = tally(rows)                    # 18 group-years
sas_only = tally([r for r in rows if r["year"] != "2026"])   # 15
both_reg = tally(rows + postcap_rows)      # 21 (2026 counted at both registers)

mixed_named = [{"year": r["year"], "peer": r["peer"], "up": r["up"],
                "down": r["down"]} for r in rows if r["direction"] == "mixed"]
alldown = [r for r in rows if r["direction"] == "all-down"]
alldown_leftskew = [{"year": r["year"], "peer": r["peer"],
                     "skew_g1": r["skew_g1_moment"], "left_skewed": r["left_skewed"],
                     "down": r["down"]} for r in alldown]

# cross-check the skews quoted in runs/mechanism_allupward.json
mech = json.load(open(f"{RUNS}/mechanism_allupward.json"))
qs = mech["stage3_skew_and_worked_example"]
xchk = {}
for peer in ["peer3", "peer4", "peer5"]:
    mine26 = next(r for r in rows if r["year"] == "2026" and r["peer"] == peer)
    mine25 = next(r for r in rows if r["year"] == "2025" and r["peer"] == peer)
    xchk[f"2026_{peer}"] = {
        "stored_mechanism_record_g1": qs["skew_per_peer_group_2026"][peer]["moment_skewness"],
        "recomputed_g1": mine26["skew_g1_moment"],
        "agrees_4dp": abs(qs["skew_per_peer_group_2026"][peer]["moment_skewness"]
                          - mine26["skew_g1_moment"]) < 5e-5}
    xchk[f"2025_{peer}"] = {
        "stored_mechanism_record_g1": qs["skew_per_peer_group_2025"][peer]["moment_skewness"],
        "recomputed_g1": mine25["skew_g1_moment"],
        "agrees_4dp": abs(qs["skew_per_peer_group_2025"][peer]["moment_skewness"]
                          - mine25["skew_g1_moment"]) < 5e-5}

record = {
    "run": "mechanism_direction_census",
    "register": ("skew from the stored replay score outputs (SAS era: "
                 "work/sas_replay_out/star_*.csv summary_score doubles, replays that "
                 "reproduce the published stars with zero mismatches; 2026: work/R_output/"
                 "fullprec_2026apr.csv %.17g doubles of the byte-unmodified R replay); "
                 "direction from the stored censuses (exact binary-double register; "
                 "the 2026 CMS side is pre-cap; after the safety cap the census is "
                 "208 of 3,203, still all upward, runs/safety_cap_check_2026.json). "
                 "Skewness estimator: adjusted Fisher-Pearson G1 = g1*sqrt(n(n-1))/(n-2); "
                 "the quoted 'moment skew' values are g1 (both stored per group-year)."),
    "table": rows,
    "postcap_2026_rows_second_register": postcap_rows,
    "direction_structure": {
        "sas_era_15_group_years": sas_only,
        "canonical_18_group_years_one_census_per_year_2026_precap": canonical,
        "with_2026_postcap_register_also_counted_21": both_reg,
        "mixed_group_years_named": mixed_named,
        "universe_note": ("21 group-year instances arise only when the 2026 census "
                          "is counted at two registers (pre-cap and post-cap); the "
                          "one-census-per-year count is 16/18 one-signed with the "
                          "same two mixed group-years. Both tallies are stored."),
    },
    "supported_register_checks": {
        "one_signed_within_group_year_is_the_norm": canonical["one_signed_over_total"],
        "side_is_year_and_group_dependent": {
            "all_up_group_years": canonical["all-up"],
            "all_down_group_years": canonical["all-down"],
            "all_down_named": [{"year": r["year"], "peer": r["peer"], "down": r["down"]}
                               for r in alldown]},
        "skew_sign_does_NOT_pick_the_side_across_years": {
            "claim": "three left-skewed group-years move all-down",
            "computed": alldown_leftskew,
            "holds": all(r["left_skewed"] for r in alldown_leftskew)
                     and len(alldown_leftskew) == 3,
            "reading": ("every group-year in the study is left-skewed (g1 < 0 in "
                        "18/18) yet three of them move all-down, so the skew sign "
                        "cannot be the cross-year side-selector") if all(
                            r["left_skewed"] for r in rows) else
                       "not all group-years left-skewed; see table",
        },
        "all_left_skewed_18_of_18": all(r["left_skewed"] for r in rows),
    },
    "cross_check_vs_stored_mechanism_record": xchk,
    "inputs": {
        "scores": ["work/sas_replay_out/star_{2021-04,2022-07,2023-07,2024-07,2025-07}.csv",
                   "work/R_output/fullprec_2026apr.csv"],
        "censuses": [f"runs/exact_census_{y}.json" for y in
                     ["2021", "2022", "2023", "2024", "2025", "2026"]],
        "cap": "runs/safety_cap_check_2026.json",
    },
}
out = f"{RUNS}/mechanism_direction_census.json"
with open(out, "w") as f:
    json.dump(record, f, indent=1)

print("=== DIRECTION / SKEW CENSUS (estimator: adjusted Fisher-Pearson G1; g1 also shown) ===")
print(f"{'yr':4s} {'peer':5s} {'n':>5s} {'g1':>8s} {'G1':>8s} {'differs':>7s} "
      f"{'up':>4s} {'down':>4s}  direction")
for r in rows:
    print(f"{r['year']:4s} {r['peer']:5s} {r['n']:5d} {r['skew_g1_moment']:8.3f} "
          f"{r['skew_G1_adjusted_fisher_pearson']:8.3f} {r['differs']:7d} "
          f"{r['up']:4d} {r['down']:4d}  {r['direction']}")
print("\nSAS-era:", sas_only)
print("canonical 18:", canonical)
print("with 2026 post-cap register (21):", both_reg)
print("mixed named:", mixed_named)
print("all-down (all left-skewed?):", alldown_leftskew)
print("skew cross-check vs runs/mechanism_allupward.json:",
      all(v["agrees_4dp"] for v in xchk.values()))
print("wrote", out)
