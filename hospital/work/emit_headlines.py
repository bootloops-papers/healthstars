#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""emit_headlines.py — recompute every hospital headline number from the stored
run files (replay_2026_pilot, exact_census_2021..2026, sas_replay_2021..2025,
safety_cap_check_2026, mechanism_allupward, dp_taste_2026, coverage_split_2026,
mechanism_direction_census) and compare each with the value quoted in the text.
Every discrepancy is listed as a finding.

Outputs: printed table and runs/emitted_headlines.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3 and Table 1 (every
    hospital headline count assembled from the stored run files).
Run:  cd hospital/work && python3 emit_headlines.py
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json

BASE = _REC + "/hospital"
RUNS = f"{BASE}/runs"
YEARS = ["2021", "2022", "2023", "2024", "2025"]

if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

R = {}
for name in (["replay_2026_pilot", "dp_taste_2026", "safety_cap_check_2026",
              "mechanism_allupward", "coverage_split_2026",
              "mechanism_direction_census", "exact_census_2026"]
             + [f"exact_census_{y}" for y in YEARS]
             + [f"sas_replay_{y}" for y in YEARS]):
    R[name] = json.load(open(f"{RUNS}/{name}.json"))

def fmt_like(quote, value):
    """Format value with the same decimal places as the quoted string."""
    q = str(quote).replace(",", "").rstrip("%")
    if "." in q:
        dec = len(q.split(".")[1])
        return f"{value:.{dec}f}"
    return str(int(round(value)))

rows = []      # every emitted headline
findings = []  # every discrepancy

def emit(section, label, emitted, quoted=None, note=""):
    if quoted is None:
        status = "no quoted value"
    else:
        qs = str(quoted).replace(",", "").rstrip("%")
        es = fmt_like(quoted, emitted) if isinstance(emitted, (int, float)) \
            else str(emitted).replace(",", "").rstrip("%")
        status = "MATCH" if es == qs else "FINDING"
        if status == "FINDING":
            findings.append({"section": section, "label": label,
                             "emitted": emitted, "note_quoted": quoted,
                             "note": note})
    rows.append({"section": section, "label": label, "emitted": emitted,
                 "note_quoted": quoted, "status": status, "note": note})

# ---------------- 2026 replay ----------------
c = R["replay_2026_pilot"]["comparison"]
emit("replay-2026", "2026 jointly-rated matched exactly", c["matched_exactly"], "3182")
emit("replay-2026", "2026 mismatches", c["mismatched"], "0")
emit("replay-2026", "2026 replay-rated footnote-suppressed", c["replay_rated_published_NotAvailable"], "14")
emit("replay-2026", "2026 match rate of jointly rated", c["match_rate_of_jointly_rated"], "1.0")
xr = R["replay_2026_pilot"].get("cross_checks", {})
emit("replay-2026", "peer group sizes replay = methodology fig5",
     "/".join(str(xr["peer_group_sizes_replay"][k]) for k in ["3-group", "4-group", "5-group"]),
     "177/749/2277")
emit("replay-2026", "input hospitals (replay = v5.1)", xr["input_hospitals_replay"], "4569")

# ---------------- 2026 census (binary-double register) ----------------
cc = R["exact_census_2026"]
emit("census-2026", "2026 rated hospitals", cc["total_rated"], "3203")
emit("census-2026", "2026 census differs (pre-cap)", cc["total_hospitals_star_differs"], "213",
     note="pre-cap; 208 of 3,203 after the safety cap, all upward (runs/safety_cap_check_2026.json)")
emit("census-2026", "2026 census percent (pre-cap)",
     100 * cc["total_hospitals_star_differs"] / cc["total_rated"], "6.7%",
     note="pre-cap")
emit("census-2026", "2026 per-peer differs",
     "/".join(str(cc["peer_groups"][p]["hospitals_star_differs"]) for p in ["peer3", "peer4", "peer5"]),
     "55/68/90")
mt = cc["moves_total"]
emit("census-2026", "2026 moves (pre-cap)",
     f"1->2:{mt['1->2']}, 2->3:{mt['2->3']}, 3->4:{mt['3->4']}, 4->5:{mt['4->5']}",
     "1->2:20, 2->3:68, 3->4:92, 4->5:33")
emit("census-2026", "2026 all upward (pre-cap)", f"{cc['moves_up']}/{cc['moves_down']}", "213/0")
for p, q in [("peer3", "0.3767"), ("peer4", "0.0234"), ("peer5", "0.0181")]:
    emit("census-2026", f"2026 {p} SSQ gap", cc["peer_groups"][p]["ssq_gap_float"], q)
emit("census-2026", "2026 optima unique (groups)",
     sum(cc["peer_groups"][p]["n_optimal_partitions"] == 1 for p in ["peer3", "peer4", "peer5"]), "3")
emit("census-2026", "2026 boundary value ties",
     sum(cc["peer_groups"][p]["boundary_value_ties"] for p in ["peer3", "peer4", "peer5"]), "0")
rs = cc["register_stability"]
emit("register-stability-2026", "decimal register reproduces dp_taste_2026",
     str(rs["decimal_register_reproduces_dp_taste_2026"]), "True")
nd = sum(int(rs["input_doubles_differing_15digit_vs_binary"][p].split("/")[0])
         for p in ["peer3", "peer4", "peer5"])
emit("register-stability-2026", "input doubles differing 15-digit vs binary", nd, "2978")
emit("register-stability-2026", "doubles differing percent", 100 * nd / 3203, "93")

# ---------------- Mechanism addendum ----------------
m = R["mechanism_allupward"]
s1 = m["stage1_boundary_dissection"]
emit("mechanism", "boundaries CMS above DP", f"{s1['summary']['shipped_above_dp']}/12", "12/12")
for p, q in [("peer3", "9+19+22+5"), ("peer4", "10+28+27+3"), ("peer5", "1+21+43+25")]:
    xs = "+".join(str(b["hospitals_crossing"]) for b in s1["per_group"][p]["boundaries"])
    emit("mechanism", f"crossing counts {p}", xs, q)
mir = m["stage2a_necessity_tests"]["mirror_test"]
emit("mechanism", "negation flips peer3 corrections down",
     f"{mir['peer3']['moves_down']}", "55")
emit("mechanism", "negation flips peer4 corrections down",
     f"{mir['peer4']['moves_down']}", "68")
sb = m["stage2a_necessity_tests"]["skew_direction_batch"]
emit("mechanism", "synthetic non-optimal datasets followed skew sign",
     f"{sb['left_skew']['nonoptimal'] + sb['right_skew']['nonoptimal']}/33 "
     f"(mixed {sb['left_skew']['mixed_direction_datasets'] + sb['right_skew']['mixed_direction_datasets']})",
     "33/33 (mixed 0)")
emit("mechanism", "synthetic moves left-skew up / right-skew down",
     f"{sb['left_skew']['moves_up']}/{sb['right_skew']['moves_down']}", "627/989")
w = m["stage3_skew_and_worked_example"]["worked_example"]
emit("mechanism", "worked example SSQ CMS vs optimal",
     f"{w['shipped_ssq']:.3f} vs {w['optimal_ssq']:.3f}", "1.498 vs 0.132")
emit("mechanism", "worked example points moved up", f"{w['moves_up']}/{w['n']}", "9/15")
s2b = m["stage2b_2025_cross_year"]
emit("mechanism", "2025 cross-year check differs / jointly-published-rated",
     f"{s2b['total_stars_differ']}/"
     f"{sum(s2b[p]['n_jointly_published_rated'] for p in ['peer3', 'peer4', 'peer5'])}",
     "38/2872", note="cross-year check register: DP vs PUBLISHED stars; census register "
                     "denominator is 2,891 replay-rated (both stored)")
emit("mechanism", "2025 cross-year check up/down", f"{s2b['moves_up_total']}/{s2b['moves_down_total']}", "33/5")
for p, q in [("peer3", "-1.21"), ("peer4", "-0.08"), ("peer5", "-0.09")]:
    emit("mechanism", f"2025 {p} moment skew (g1)", s2b[p]["moment_skewness"], q)
for p, q in [("peer3", "-0.80"), ("peer4", "-0.69"), ("peer5", "-0.31")]:
    emit("mechanism", f"2026 {p} moment skew (g1)",
         m["stage3_skew_and_worked_example"]["skew_per_peer_group_2026"][p]["moment_skewness"], q)

# ---------------- SAS-era replays ----------------
SNAP_QUOTED = {"2021": ("hospitals_2021-04-28.zip", "3339", "6"),
             "2022": ("hospitals_2022-07-06.zip", "3094", "10"),
             "2023": ("hospitals_2023-07-06.zip", "3061", "7"),
             "2024": ("hospitals_2024-07-31.zip", "2834", "1"),
             "2025": ("hospitals_2025-08-14.zip", "2872", "10")}
purus, iters, stricts, residue = [], [], [], 0
for y in YEARS:
    sr = R[f"sas_replay_{y}"]
    snap, qm, qs = SNAP_QUOTED[y]
    cb = sr["comparisons"][snap]
    emit("sas-era-replay", f"{y} matched exactly ({snap})", cb["matched_exactly"], qm)
    emit("sas-era-replay", f"{y} mismatched", cb["mismatched"], "0")
    emit("sas-era-replay", f"{y} suppressed", cb["replay_rated_published_NotAvailable"], qs)
    emit("sas-era-replay", f"{y} match rate", cb["match_rate_of_jointly_rated"], "1.0")
    purus.append(cb["published_rated_replay_unrated"])
    for p in ["peer3", "peer4", "peer5"]:
        pg = sr["peer_groups"][p]
        iters += [pg["stage1_iters"], pg["stage2_iters"]]
        stricts.append(pg["strict_excluded_reincluded"])
        residue += (pg["tie_events_total"] + pg["mean_sort_ties"]
                    + pg["duplicate_seed_values"])
emit("sas-era-replay", "published_rated_replay_unrated (five matching snapshots)",
     "/".join(map(str, purus)), "0/0/0/0/0",
     note="zero at the five matching snapshots; the alternate 2022-10-06 snapshot "
          "carries 6 (runs/sas_replay_2022.json)")
emit("sas-era-replay", "2023 both snapshots identical matched",
     f"{R['sas_replay_2023']['comparisons']['hospitals_2023-07-06.zip']['matched_exactly']}="
     f"{R['sas_replay_2023']['comparisons']['hospitals_2023-10-06.zip']['matched_exactly']}",
     "3061=3061")
emit("sas-era-replay", "2025-04-30 mismatches (vintage contrast)",
     R["sas_replay_2025"]["comparisons"]["hospitals_2025-04-30.zip"]["mismatched"], "1245")
emit("sas-era-replay", "FASTCLUS residue counters total (15 group-runs)", residue, "0")
emit("sas-era-replay", "max FASTCLUS iterations (MAXITER 1000)", max(iters), "38",
     note="quoted as '<= 38'")
nz = [s for s in stricts if s > 0]
strict_detail = {f"{y}-{p}": R[f"sas_replay_{y}"]["peer_groups"][p]["strict_excluded_reincluded"]
                 for y in YEARS for p in ["peer3", "peer4", "peer5"]}
emit("sas-era-replay", "STRICT=1 exclusions per group (15 groups)",
     f"nonzero in {len(nz)}/15 groups, range {min(nz)}-{max(nz)} where nonzero; "
     f"{stricts.count(0)} groups had 0",
     "1-4 per group",
     note="per-group values: " + json.dumps(strict_detail)
          + "; four groups had 0 and one had 5, so the quoted range 1-4 is wrong")

# ---------------- SAS-era censuses ----------------
QUOTED_SAS = {  # year: rated, differs, pct, up, down, per-peer, gaps(3/4/5)
    "2021": ("3355", "99", "3.0", "40", "59", "67/26/6", ["0.789", "0.380", "0.0038"]),
    "2022": ("3121", "153", "4.9", "107", "46", "46/74/33", ["0.776", "0.362", "0.0159"]),
    "2023": ("3076", "370", "12.0", "136", "234", "118/249/3", ["0.765", "1.740", "0.00016"]),
    "2024": ("2847", "273", "9.6", "15", "258", "7/258/8", ["0.279", "0.0213", "0.0022"]),
    "2025": ("2891", "38", "1.3", "33", "5", "11/5/22", ["0.159", "0.0040", "0.0061"]),
}
tot5 = 0
uniq = ties = 0
for y in YEARS:
    ccy = R[f"exact_census_{y}"]
    qr, qd, qp, qu, qdn, qpp, qg = QUOTED_SAS[y]
    emit("sas-era-census", f"{y} rated", ccy["total_rated"], qr)
    emit("sas-era-census", f"{y} differs", ccy["total_hospitals_star_differs"], qd)
    emit("sas-era-census", f"{y} percent",
         100 * ccy["total_hospitals_star_differs"] / ccy["total_rated"], qp)
    emit("sas-era-census", f"{y} up/down", f"{ccy['moves_up']}/{ccy['moves_down']}", f"{qu}/{qdn}")
    emit("sas-era-census", f"{y} per-peer differs",
         "/".join(str(ccy["peer_groups"][p]["hospitals_star_differs"])
                  for p in ["peer3", "peer4", "peer5"]), qpp)
    for p, q in zip(["peer3", "peer4", "peer5"], qg):
        emit("sas-era-census", f"{y} {p} SSQ gap", ccy["peer_groups"][p]["ssq_gap_float"], q)
    emit("sas-era-census", f"{y} moves table (from the stored census)",
         json.dumps(ccy["moves_total"], sort_keys=True))
    tot5 += ccy["total_hospitals_star_differs"]
    uniq += sum(ccy["peer_groups"][p]["n_optimal_partitions"] == 1
                for p in ["peer3", "peer4", "peer5"])
    ties += sum(ccy["peer_groups"][p]["boundary_value_ties"]
                for p in ["peer3", "peer4", "peer5"])
emit("sas-era-census", "five-year SAS-era total differs", tot5, "933")
emit("sas-era-census", "unique optima (SAS 15 instances)", f"{uniq}/15", "15/15")
emit("sas-era-census", "boundary ties (SAS 15 instances)", ties, "0")
emit("sas-era-census", "2023 peer4 share of peer group",
     f"{R['exact_census_2023']['peer_groups']['peer4']['hospitals_star_differs']} of "
     f"{R['exact_census_2023']['peer_groups']['peer4']['n']} = "
     f"{100 * R['exact_census_2023']['peer_groups']['peer4']['hospitals_star_differs'] / R['exact_census_2023']['peer_groups']['peer4']['n']:.0f}%",
     "249 of 462 = 54%")

# ---------------- Six-year totals ----------------
emit("six-year-totals", "pre-cap census total (933 + 213)",
     tot5 + cc["total_hospitals_star_differs"], "1146",
     note="pre-cap; 208 of 3,203 after the safety cap, all upward (runs/safety_cap_check_2026.json)")

# ---------------- Safety cap ----------------
cap = R["safety_cap_check_2026"]
emit("safety-cap-2026", "cap replay mismatches vs R star_p3",
     cap["cap_replay_validation"]["python_replay_mismatches_vs_R_star_p3"], "0")
emit("safety-cap-2026", "cap replay rows checked", cap["cap_replay_validation"]["starred_rows_checked"], "3203")
emit("safety-cap-2026", "CMS capped 5->4", cap["cap_replay_validation"]["shipped_capped_5_to_4"], "15")
emit("safety-cap-2026", "DP corrections survive cap", cap["survive_cap_unchanged"], "208")
emit("safety-cap-2026", "DP corrections erased by cap", cap["modified_by_cap"], "5")
emit("safety-cap-2026", "erased are peer5 4->5 movers",
     sum(1 for h in cap["modified_detail"]
         if h["peer"] == "peer5" and h["dp_precap"] == 5 and h["shipped_precap"] == 4), "5")
emit("safety-cap-2026", "post-cap census percent",
     100 * cap["postcap_differ_census"]["dp_postcap_vs_shipped_postcap_differs"] / 3203, "6.5%")
emit("safety-cap-2026", "new differs created by cap",
     len(cap["postcap_differ_census"]["new_differs_created_by_cap"]), "0")
emit("safety-cap-2026", "CMS-capped hospitals among the 213",
     len(cap["shipped_capped_hospitals_in_dp_differ_set"]), "0")
# post-cap moves table, emitted from census detail minus erased
erased = set(cap["postcap_differ_census"]["differs_erased_by_cap"])
pmoves = {}
for h in cc["star_differs_detail"]:
    if h["provider_id"] in erased:
        continue
    k = f"{h['shipped']}->{h['dp_optimum']}"
    pmoves[k] = pmoves.get(k, 0) + 1
emit("safety-cap-2026", "post-cap moves",
     f"1->2:{pmoves['1->2']}, 2->3:{pmoves['2->3']}, 3->4:{pmoves['3->4']}, 4->5:{pmoves['4->5']}",
     "1->2:20, 2->3:68, 3->4:92, 4->5:28")
emit("six-year-totals", "post-cap-2026 six-year total (933 + 208)",
     tot5 + cap["postcap_differ_census"]["dp_postcap_vs_shipped_postcap_differs"], "1141")
# CY2027 trigger set — recomputed from the stored full-precision dump
trig = 0
with open(f"{BASE}/work/R_output/fullprec_2026apr.csv") as f:
    for r in csv.DictReader(f):
        if (r["star_precap"] not in ("NA", "") and r["Safe_q"] == "1"
                and r["Outcomes_Safety_cnt"] not in ("NA", "")
                and int(r["Outcomes_Safety_cnt"]) >= 3):
            trig += 1
emit("safety-cap-2026", "CY2027 trigger set (rated, safety Q1, >=3 safety measures)", trig, "558",
     note="recomputed from work/R_output/fullprec_2026apr.csv")

# ---------------- Direction structure ----------------
dc = R["mechanism_direction_census"]["direction_structure"]
emit("direction", "SAS-era one-signed group-years",
     dc["sas_era_15_group_years"]["one_signed_over_total"], "13/15")
emit("direction", "mixed group-years named",
     "; ".join(f"{x['year']}-{x['peer']} +{x['up']}/-{x['down']}"
               for x in dc["mixed_group_years_named"]),
     "2021-peer3 +8/-59; 2023-peer4 +15/-234")
emit("direction", "one-signed, canonical universe (one census per year, 2026 pre-cap)",
     dc["canonical_18_group_years_one_census_per_year_2026_precap"]["one_signed_over_total"],
     "19/21",
     note="21 instances arise only when the 2026 census is counted at both registers "
          "(pre-cap and post-cap), which gives 19/21 (next row); one census per year "
          "gives 16/18 with the same two mixed group-years")
emit("direction", "one-signed with 2026 counted at both registers",
     dc["with_2026_postcap_register_also_counted_21"]["one_signed_over_total"], "19/21")
sup = R["mechanism_direction_census"]["supported_register_checks"]
emit("direction", "three left-skewed group-years move all-down",
     f"{len(sup['skew_sign_does_NOT_pick_the_side_across_years']['computed'])} "
     f"({'; '.join(x['year'] + '-' + x['peer'] + ' g1=' + str(x['skew_g1']) for x in sup['skew_sign_does_NOT_pick_the_side_across_years']['computed'])})",
     None, note="claim holds: " + str(sup["skew_sign_does_NOT_pick_the_side_across_years"]["holds"]))
emit("direction", "all 18 group-years left-skewed (g1<0)",
     str(sup["all_left_skewed_18_of_18"]), "True")

# ---------------- Coverage split ----------------
cov = R["coverage_split_2026"]
t26 = cov["year_2026"]["split"]
emit("display-coverage", "2026 accounting", t26["accounting"],
     "3182 matched + 0 mismatched + 14 suppressed + 7 absent = 3203")
emit("display-coverage", "2026 suppression footnotes",
     json.dumps(t26["suppressed_by_footnote"], sort_keys=True),
     json.dumps({"11": 1, "5": 13}, sort_keys=True))
emit("display-coverage", "suppression range across six years (hospitals/year)",
     f"{min(r['suppressed'] for r in cov['era_uniform_table'])}-"
     f"{max(r['suppressed'] for r in cov['era_uniform_table'])}", "1-14")
fi = cov["FINDINGS"]["1_interaction_not_zero"]
emit("display-coverage", "census interaction with suppressed/absent classes",
     f"3 of 213 (and 3 of 208) not publicly displayed; display-visible post-cap "
     f"corrections {fi['display_visible_corrections_postcap']}/{fi['displayed_ratings_denominator']}",
     "0",
     note="from runs/coverage_split_2026.json: 2 footnote-5-suppressed and 1 "
          "absent-from-snapshot member of the 213/208; six-year rollup 12 of 1,146. "
          "The census sets are unchanged; this is the display-visibility layer only.")

# ---------------- write the record ----------------
n_match = sum(1 for r in rows if r["status"] == "MATCH")
n_find = len(findings)
record = {
    "run": "emitted_headlines",
    "register": ("Every quoted headline number re-emitted from the stored records and "
                 "compared at the quoted printed precision. Census figures are at the "
                 "pre-cap register; the published-visible post-cap census is 208/3,203 "
                 "= 6.5%, still all-upward (runs/safety_cap_check_2026.json); data "
                 "provenance per runs/coverage_split_2026.json."),
    "headlines_emitted": len(rows),
    "matches": n_match,
    "findings_count": n_find,
    "FINDINGS": findings,
    "table": rows,
}
out = f"{RUNS}/emitted_headlines.json"
with open(out, "w") as f:
    json.dump(record, f, indent=1)

for r in rows:
    q = "" if r["note_quoted"] is None else f"  [quoted: {r['note_quoted']}]"
    print(f"{r['status']:8s} {r['section']:24s} {r['label']}: {r['emitted']}{q}")
    if r["note"]:
        print(f"         note: {r['note']}")
print(f"\n{len(rows)} headlines emitted; {n_match} MATCH; {n_find} FINDINGS")
for i, f_ in enumerate(findings, 1):
    print(f"FINDING {i}: [{f_['section']}] {f_['label']} — emitted {f_['emitted']} "
          f"vs quoted {f_['note_quoted']}. {f_['note']}")
print("wrote", out)
