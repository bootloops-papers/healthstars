#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
# Derived work: a Python port of stages 0-2 of CMS's published Overall Hospital Quality Star Rating SAS package (see the docstring); that package is a public CMS document and keeps its own terms.

"""Mechanism analysis, stage 2b: 2025 SAS-era cross-check.

Port of SAS pack stages 0-2 (standardize -> group scores -> summary score +
peer groups) from data/raw/2025-07/alldata_2025jul.sas7bdat, in pandas floats
(register: binary doubles of this port; the SAS float-accumulation order is
not reproduced). The assignment step (FASTCLUS) is not replayed here (see
sas_replay.py): the script compares the exact 1-D k-means optimum on the
ported summary scores against the published 2025 stars (Care Compare snapshot
2025-08-14). The 2025 pipeline has no safety cap.

Port validation checks: measure volume filter (expected empty), peer-group
sizes, published per-group measure-count columns vs ported counts,
contiguity of published stars in ported-score order.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.3 (mechanism and
    direction of the differences; the 2025 cross-check).
Run:  cd hospital/work && python3 mech_2025.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, pandas.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json, sys
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, _REC + "/src")
from dp_kmeans import dp_optimal_partition, dp_optimal_partitions_all

BASE = _REC + "/hospital"

MORT = ["MORT_30_AMI", "MORT_30_CABG", "MORT_30_COPD", "MORT_30_HF",
        "MORT_30_PN", "MORT_30_STK", "PSI_4_SURG_COMP"]
SAFE = ["COMP_HIP_KNEE", "HAI_1", "HAI_2", "HAI_3", "HAI_4", "HAI_5", "HAI_6",
        "PSI_90_SAFETY"]
READM = ["EDAC_30_AMI", "EDAC_30_HF", "EDAC_30_PN", "OP_32", "READM_30_CABG",
         "READM_30_COPD", "READM_30_HIP_KNEE", "READM_30_HOSP_WIDE",
         "OP_35_ADM", "OP_35_ED", "OP_36"]
PTEXP = ["H_COMP_1_STAR_RATING", "H_COMP_2_STAR_RATING", "H_COMP_3_STAR_RATING",
         "H_COMP_5_STAR_RATING", "H_COMP_6_STAR_RATING", "H_COMP_7_STAR_RATING",
         "H_GLOB_STAR_RATING", "H_INDI_STAR_RATING"]
PROC = ["HCP_COVID_19", "IMM_3", "OP_10", "OP_13", "OP_18B", "OP_22", "OP_23",
        "OP_29", "OP_8", "PC_01", "SAFE_USE_OF_OPIOIDS", "SEP_1"]
ALL = MORT + SAFE + READM + PTEXP + PROC
FLIP = MORT + SAFE + READM + ["OP_22", "PC_01", "OP_18B", "OP_8", "OP_10",
                              "OP_13", "SAFE_USE_OF_OPIOIDS"]
GROUPS = {"Mortality": (MORT, 0.22), "Safety": (SAFE, 0.22),
          "Readmission": (READM, 0.22), "PatientExp": (PTEXP, 0.22),
          "Process": (PROC, 0.12)}

def zstd(s):
    return (s - s.mean()) / s.std(ddof=1)   # PROC STANDARD mean=0 std=1

def port_2025():
    df = pd.read_sas(f"{BASE}/data/raw/2025-07/alldata_2025jul.sas7bdat")
    df["PROVIDER_ID"] = df["PROVIDER_ID"].str.decode("utf-8")
    # stage 0: volume filter (N<=100 excluded)
    vol = {m: int(df[m].notna().sum()) for m in ALL}
    excluded = [m for m, v in vol.items() if v <= 100]
    meas = [m for m in ALL if m not in excluded]
    # keep hospitals with >=1 included measure
    df = df[df[meas].notna().any(axis=1)].reset_index(drop=True)
    # standardize then flip
    for m in meas:
        df[m] = zstd(df[m])
        if m in FLIP:
            df[m] = -df[m]
    # stage 1: group scores
    out = pd.DataFrame({"PROVIDER_ID": df["PROVIDER_ID"]})
    for g, (mm, _) in GROUPS.items():
        mm2 = [m for m in mm if m in meas]
        cnt = df[mm2].notna().sum(axis=1)
        avg = df[mm2].sum(axis=1, skipna=True) / cnt   # NaN where cnt==0
        avg[cnt == 0] = np.nan
        out[f"{g}_cnt"] = cnt
        out[f"{g}_grp"] = zstd(avg)                    # standardized over nonmissing
    # stage 2: weighted summary score with redistribution
    w = np.zeros(len(out))
    s = np.zeros(len(out))
    tot_w = np.zeros(len(out))
    for g, (_, wt) in GROUPS.items():
        present = out[f"{g}_grp"].notna().to_numpy()
        tot_w += np.where(present, wt, 0.0)
    for g, (_, wt) in GROUPS.items():
        v = out[f"{g}_grp"].to_numpy()
        present = ~np.isnan(v)
        s += np.where(present, (wt / tot_w) * np.where(present, v, 0.0), 0.0)
    out["summary_score"] = s
    # report criteria + peer group
    D = sum((out[f"{g}_cnt"] >= 3).astype(int) for g in GROUPS)
    out["Total_measure_group_cnt"] = D
    out["MortSafe_Group_cnt"] = (out["Mortality_cnt"] >= 3).astype(int) + \
                                (out["Safety_cnt"] >= 3).astype(int)
    out["report_indicator"] = ((out["MortSafe_Group_cnt"] >= 1) & (D >= 3)).astype(int)
    return out, vol, excluded

def load_published():
    pub = {}
    with open(f"{BASE}/work/cc_2025_08/Hospital_General_Information.csv",
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pub[r["Facility ID"]] = r
    return pub

def dp_group(args):
    label, xs = args
    dp = dp_optimal_partition(xs, 5)
    ssq, all_opt = dp_optimal_partitions_all(xs, 5)
    return label, dp, len(all_opt)

_RAW_NEEDED = (f"{BASE}/data/raw/2025-07/alldata_2025jul.sas7bdat",
               f"{BASE}/work/cc_2025_08/Hospital_General_Information.csv")
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(_os.path.exists(p) for p in _RAW_NEEDED):
    print("mech_2025: missing raw input(s): " + ", ".join(_os.path.relpath(p, _REC) for p in _RAW_NEEDED
                                                  if not _os.path.exists(p))
          + " (the 2025 SAS input file and the Care Compare snapshot member, not included in the package; re-fetch each "
          "by the URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    port, vol, excluded = port_2025()
    pub = load_published()
    rated = port[port["report_indicator"] == 1].copy()
    res = {"run": "mech_2025_dp_vs_published",
           "register": "ported summary scores (pandas float64, SAS semantics: "
                       "n-1 std, missing-skip sums, weight redistribution); "
                       "DP on exact binary doubles of the port; published stars "
                       "= Care Compare 2025-08-14 snapshot; FASTCLUS not replayed",
           "volume_filter_excluded": excluded,
           "n_hospitals_with_any_measure": int(len(port)),
           "n_report_indicator_1": int(len(rated))}

    # validation: ported per-group measure counts vs published count columns
    colmap = {"Mortality": "Count of Facility MORT Measures",
              "Safety": "Count of Facility Safety Measures",
              "Readmission": "Count of Facility READM Measures",
              "PatientExp": "Count of Facility Pt Exp Measures",
              "Process": "Count of Facility TE Measures"}
    cnt_checked = cnt_mismatch = 0
    for _, r in port.iterrows():
        p = pub.get(r["PROVIDER_ID"])
        if not p:
            continue
        for g, col in colmap.items():
            v = p.get(col, "").strip()
            if v.isdigit():
                cnt_checked += 1
                if int(v) != int(r[f"{g}_cnt"]):
                    cnt_mismatch += 1
    res["validation_group_measure_counts"] = {
        "checked": cnt_checked, "mismatched": cnt_mismatch}

    # peer groups
    peer = {3: "peer3", 4: "peer4", 5: "peer5"}
    jobs = []
    groups = {}
    for tg, label in peer.items():
        sub = rated[rated["Total_measure_group_cnt"] == tg]
        pairs = []
        for _, r in sub.iterrows():
            p = pub.get(r["PROVIDER_ID"])
            star = None
            if p and p["Hospital overall rating"] not in ("Not Available", ""):
                star = int(p["Hospital overall rating"])
            pairs.append((float(r["summary_score"]), star, r["PROVIDER_ID"]))
        groups[label] = pairs
        jobs.append((label, [Fraction(x) for x, _, _ in pairs]))
        print(label, "n=", len(pairs), "published-rated:",
              sum(1 for _, s, _ in pairs if s is not None))

    with Pool(3) as pool:
        dps = pool.map(dp_group, jobs)

    total_diff = 0
    moves_total = {}
    for label, dp, nopt in dps:
        pairs = groups[label]
        bnds, sx = dp["boundaries"], dp["sorted_x"]
        block_max = [sx[bnds[b + 1] - 1] for b in range(5)]
        def dp_star(v):
            f = Fraction(v)
            for b in range(5):
                if f <= block_max[b]:
                    return b + 1
            return 5
        ties = sum(1 for b in range(1, 5) if sx[bnds[b] - 1] == sx[bnds[b]])
        joint = [(x, s, dp_star(x)) for x, s, _ in pairs if s is not None]
        # contiguity of published stars in ported-score order
        joint_sorted = sorted(joint)
        viol = sum(1 for i in range(1, len(joint_sorted))
                   if joint_sorted[i][1] < joint_sorted[i - 1][1])
        moves = {}
        for _, s_pub, s_dp in joint:
            if s_pub != s_dp:
                moves[f"{s_pub}->{s_dp}"] = moves.get(f"{s_pub}->{s_dp}", 0) + 1
                moves_total[f"{s_pub}->{s_dp}"] = moves_total.get(f"{s_pub}->{s_dp}", 0) + 1
        ndiff = sum(moves.values())
        total_diff += ndiff
        # boundary shifts (published cut = first index where published star exceeds b+1)
        stars_sorted = [s for _, s, _ in joint_sorted]
        xs_sorted = [x for x, _, _ in joint_sorted]
        bshifts = []
        if viol == 0:
            # published cut index within the joint set; dp cut = count of joint values <= block_max[b]
            for b in range(4):
                try:
                    pi = next(i for i, s in enumerate(stars_sorted) if s >= b + 2)
                except StopIteration:
                    pi = len(stars_sorted)
                di = sum(1 for x in xs_sorted if Fraction(x) <= block_max[b])
                bshifts.append({"cut": f"{b+1}|{b+2}",
                                "published_minus_dp_index": pi - di,
                                "hospitals_crossing": abs(pi - di),
                                "direction": "published ABOVE dp" if pi > di
                                             else ("published BELOW dp" if pi < di else "equal")})
        # skew
        fx = sorted(x for x, _, _ in pairs)
        n = len(fx); m = sum(fx) / n
        m2 = sum((v - m) ** 2 for v in fx) / n
        m3 = sum((v - m) ** 3 for v in fx) / n
        med = fx[n // 2] if n % 2 else (fx[n // 2 - 1] + fx[n // 2]) / 2
        res[label] = {
            "n_clustered": n,
            "n_jointly_published_rated": len(joint),
            "published_contiguity_violations_in_ported_score_order": viol,
            "n_optimal_partitions": nopt,
            "boundary_value_ties": ties,
            "moment_skewness": m3 / m2 ** 1.5,
            "mean_minus_median": m - med,
            "stars_differ_published_vs_dp": ndiff,
            "moves": moves,
            "boundary_shifts": bshifts,
        }
        print(label, "diff=", ndiff, moves, "viol=", viol, "skew=%.3f" % (m3 / m2 ** 1.5))

    up = sum(v for k, v in moves_total.items() if int(k[0]) < int(k[-1]))
    dn = sum(v for k, v in moves_total.items() if int(k[0]) > int(k[-1]))
    res["moves_total"] = moves_total
    res["total_stars_differ"] = total_diff
    res["moves_up_total"] = up
    res["moves_down_total"] = dn
    with open(f"{BASE}/work/mech_stage2b_2025.json", "w") as f:
        json.dump(res, f, indent=2)
    print("TOTAL diff:", total_diff, "up:", up, "down:", dn)
    print("wrote", f"{BASE}/work/mech_stage2b_2025.json")
