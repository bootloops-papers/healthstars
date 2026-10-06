#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
# Reimplements the DATA-step chain of CMS's published Overall Hospital Quality Star Rating SAS packages (2021-2025), which are public CMS documents and keep their own terms; the k-means step is this record's own src/hospital_kmeans.py.

"""SAS-era hospital stars replay (2021-2025): the packs' DATA-step chain in
Python doubles + src/hospital_kmeans.py for the two-stage PROC FASTCLUS star
step.

Arithmetic: IEEE-double pipeline on purpose (the reproduced SAS pipeline is
double); accumulations sequential in dataset order (see the Notes in
src/hospital_kmeans.py); the k-means step is the record's independent
implementation of the FASTCLUS procedure as the SAS/STAT User's Guide
documents it (chapter "The FASTCLUS Procedure"), with the points the
documentation leaves open carried as counters in the output record.

Choices that follow CMS's published pack for the year (program 0/1/2 +
Star_Macros %kmeans / %grp_score / %report / %keep_hos, all public documents):
  - measure volume rule: nonmissing count > 100 keeps the measure
  - %keep_hos: hospitals with zero included measures dropped BEFORE
    standardization
  - PROC STANDARD mean=0 std=1: mean/sd over nonmissing, sd with n-1,
    missing left missing (two-pass, sequential accumulation)
  - direction flips AFTER standardization (program 0 RE-DIRECT step)
  - group score: measure_wt = 1/total_cnt then avg = sum*measure_wt
    (multiply by reciprocal, as coded — not a division)
  - summary score: weight redistribution W_k / (1 - I1*W1 - ... - I5*W5)
    left-to-right; summary = ((((swa1+swa2)+swa3)+swa4)+swa5)
  - peer groups by Total_measure_group_cnt in {3,4,5}; report_indicator =
    (MortSafe>=1) and (Total>=3); %kmeans per peer group
    (hospital_kmeans.kmeans_star)
  - rows sorted by PROVIDER_ID (program 2 sorts before the star step; the
    module's dataset-order conventions then run on that order)

Usage: python3 sas_replay.py <release>   e.g. 2021-04
Writes work/sas_replay_out/star_<release>.csv and runs/sas_replay_<year>.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.1 and Appendix C.1
    (reproducing the 2021-2025 SAS packages).
Exit codes: 0 on success; 2 on a missing argument or a missing input (one line
    on stderr names it).
Requires: Python >= 3.10; numpy, pyreadstat.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter

import numpy as np
import pyreadstat

# The k-means star step: src/hospital_kmeans.py in this record, an independent
# implementation of the FASTCLUS procedure as documented in the SAS/STAT
# User's Guide (chapter "The FASTCLUS Procedure") and of CMS's published
# %kmeans macro.
sys.path.insert(0, os.path.join(_REC, "src"))
from hospital_kmeans import kmeans_star  # noqa: E402

BASE = _REC + "/hospital"

GROUP_DATASETS = {  # DATA-step name in program 0 -> our group key
    "outcomes_mortality": "Mortality",
    "outcomes_safety": "Safety",
    "outcomes_readmission": "Readmission",
    "ptexp": "PtExp",
    "process": "Process",
}
# program 2 array orders (as listed in the pack's summary_score DATA step):
#   std_weight: PatientExperience .22, Readmission .22, Mortality .22,
#               safety .22, Process .12
#   score:      PtExp, Readmission, Mortality, Safety, Process
WEIGHT_ORDER = ["PtExp", "Readmission", "Mortality", "Safety", "Process"]
WEIGHTS = [0.22, 0.22, 0.22, 0.22, 0.12]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_sas_comments(text):
    """Remove /* ... */ block comments and lines that are '*...;' statement
    comments (the packs use both)."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        if re.match(r"^\s*\*", line):  # '*comment;' statement (possibly multiline
            continue                   # in the packs it is always one line)
        out.append(line)
    return "\n".join(out)


def expand_range(tok):
    """SAS numbered range list HAI_1-HAI_6 -> HAI_1..HAI_6."""
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*?)(\d+)-\1(\d+)$", tok)
    if not m:
        return [tok]
    stem, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    return [f"{stem}{i}" for i in range(a, b + 1)]


def parse_pack(release):
    """Read the year's program 0: measure_all, per-group in-lists, flips."""
    pack_dir = None
    for cand in ("sas_package_mirror_rush", "sas_package_mirror_kos"):
        d = os.path.join(BASE, "data/raw", release, cand)
        if os.path.isdir(d):
            pack_dir = d
            break
    prog0 = [f for f in os.listdir(pack_dir) if f.startswith("0 -")][0]
    raw = open(os.path.join(pack_dir, prog0), encoding="latin-1").read()
    text = strip_sas_comments(raw)

    m = re.search(r"%LET\s+measure_all\s*=\s*(.*?);", text, flags=re.S | re.I)
    toks = m.group(1).split()
    measure_all = [t.upper() for tok in toks for t in expand_range(tok)]

    groups = {}
    for ds, key in GROUP_DATASETS.items():
        m = re.search(
            rf"DATA\s+{ds}\b.*?if\s+measure_in_name\s+in\s*\((.*?)\)\s*;",
            text, flags=re.S | re.I)
        names = re.findall(r"'([^']+)'", m.group(1))
        groups[key] = [n.upper() for n in names]

    flips = [mm.group(1).upper()
             for mm in re.finditer(
                 r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*-\s*\1\s*;",
                 text, flags=re.M | re.I)]

    ym = re.search(r"%LET\s+year\s*=\s*(\w+)\s*;", text, flags=re.I)
    qm = re.search(r"%LET\s+quarter\s*=\s*(\w+)\s*;", text, flags=re.I)
    return {
        "pack_dir": pack_dir, "prog0": prog0,
        "measure_all": measure_all, "groups": groups, "flips": flips,
        "year": ym.group(1) if ym else release[:4],
        "quarter": qm.group(1) if qm else "",
    }


def seq_mean_sd(vals):
    """Two-pass mean/sd (n-1), sequential dataset-order accumulation."""
    n = len(vals)
    s = 0.0
    for v in vals:
        s += v
    m = s / n
    q = 0.0
    for v in vals:
        d = v - m
        q += d * d
    sd = (q / (n - 1)) ** 0.5 if n > 1 else float("nan")
    return m, sd


def run_pipeline(release):
    cfg = parse_pack(release)
    raw_dir = os.path.join(BASE, "data/raw", release)
    sas_files = [f for f in os.listdir(raw_dir) if f.endswith(".sas7bdat")
                 and "kos" not in f]
    assert len(sas_files) == 1, sas_files
    sas_path = os.path.join(raw_dir, sas_files[0])
    df, meta = pyreadstat.read_sas7bdat(sas_path)
    df.columns = [c.upper() for c in df.columns]

    # program 2 sorts by provider_id before the star step; the whole SAS
    # chain is provider-keyed merges, so dataset order = PROVIDER_ID order
    df = df.sort_values("PROVIDER_ID", kind="stable").reset_index(drop=True)

    missing_cols = [m for m in cfg["measure_all"] if m not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{release}: measures absent from alldata: {missing_cols}")

    # --- volume rule: keep nonmissing count > 100 ---
    counts = {m: int(df[m].notna().sum()) for m in cfg["measure_all"]}
    measure_in = [m for m in cfg["measure_all"] if counts[m] > 100]
    excluded = [m for m in cfg["measure_all"] if counts[m] <= 100]

    # --- %keep_hos: drop hospitals with zero included measures ---
    keep = df[measure_in].notna().sum(axis=1) >= 1
    df = df[keep].reset_index(drop=True)

    # --- PROC STANDARD mean=0 std=1 then RE-DIRECT flips ---
    Z = {}
    for mcol in measure_in:
        x = df[mcol].to_numpy(dtype=np.float64)
        nn = ~np.isnan(x)
        m, sd = seq_mean_sd(list(x[nn]))
        z = (x - m) / sd
        if mcol in cfg["flips"]:
            z = -z
        Z[mcol] = z
    n_hosp = len(df)

    # --- %grp_score per group: total_cnt, avg, then standardize avg ---
    grp_score = {}
    grp_cnt = {}
    for key in WEIGHT_ORDER:
        vlist = [m for m in cfg["measure_all"] if m in cfg["groups"][key]
                 and m in measure_in]
        cnt = np.zeros(n_hosp, dtype=int)
        avg = np.full(n_hosp, np.nan)
        for i in range(n_hosp):
            s = 0.0
            c = 0
            for mcol in vlist:  # sum(of varlist) skips missing, varlist order
                v = Z[mcol][i]
                if not np.isnan(v):
                    s += v
                    c += 1
            cnt[i] = c
            if c > 0:
                avg[i] = s * (1.0 / c)  # measure_wt=1/total_cnt; avg=sum*wt
        nn = ~np.isnan(avg)
        m, sd = seq_mean_sd(list(avg[nn]))
        gs = (avg - m) / sd
        grp_score[key] = gs
        grp_cnt[key] = cnt

    # --- summary score (program 2 DATA step, arrays in WEIGHT_ORDER) ---
    summary = np.full(n_hosp, np.nan)
    for i in range(n_hosp):
        scores = [grp_score[k][i] for k in WEIGHT_ORDER]
        Ind = [1.0 if np.isnan(s) else 0.0 for s in scores]
        denom = 1.0
        for k in range(5):  # 1 - I1*W1 - I2*W2 - ... left-to-right
            denom = denom - Ind[k] * WEIGHTS[k]
        if all(Ind):          # all five group scores missing -> summary missing
            summary[i] = np.nan
            continue
        s = 0.0
        for k in range(5):  # sum(of swa1-swa5), swa=0 for missing groups
            swa = 0.0 if Ind[k] else (WEIGHTS[k] / denom) * scores[k]
            s += swa
        summary[i] = s

    # NOTE the weight expression: SAS computes weight[k]=W[k]/(1-...); then
    # sum_weight_ave[k]=weight[k]*score[k]; replicated exactly above.

    # --- %report indicator ---
    total_grp_cnt = np.zeros(n_hosp, dtype=int)
    for key in WEIGHT_ORDER:
        total_grp_cnt += (grp_cnt[key] >= 3).astype(int)
    mortsafe = (grp_cnt["Mortality"] >= 3).astype(int) + \
               (grp_cnt["Safety"] >= 3).astype(int)
    report = ((mortsafe >= 1) & (total_grp_cnt >= 3)).astype(int)

    # --- peer groups + %kmeans (hospital_kmeans two-stage driver) ---
    star = [None] * n_hosp
    peer_records = {}
    for tg, label in ((3, "peer3"), (4, "peer4"), (5, "peer5")):
        idx = [i for i in range(n_hosp)
               if report[i] == 1 and total_grp_cnt[i] == tg]
        if not idx:
            peer_records[label] = {"n": 0}
            continue
        xs = [float(summary[i]) for i in idx]
        r = kmeans_star(xs)
        for j, i in enumerate(idx):
            star[i] = r.star[j]
        peer_records[label] = {
            "n": len(idx),
            "stage1_iters": r.stage1.n_iter, "stage1_converged": r.stage1.converged,
            "stage2_iters": r.stage2.n_iter, "stage2_converged": r.stage2.converged,
            "strict_excluded_reincluded": r.strict_excluded,
            "tie_events_total": r.tie_events_total,
            "mean_sort_ties": r.mean_sort_ties,
            "empty_cluster_events_stage2": r.stage2.empty_cluster_events,
            "duplicate_seed_values": (r.stage1.duplicate_seed_values
                                      + r.stage2.duplicate_seed_values),
            "seeds1": r.seeds1, "seeds2": r.seeds2,
            "star_counts": dict(Counter(s for s in
                                        (r.star[j] for j in range(len(idx)))
                                        if s is not None)),
        }

    out_rows = []
    for i in range(n_hosp):
        out_rows.append({
            "PROVIDER_ID": df["PROVIDER_ID"].iloc[i],
            "report_indicator": int(report[i]),
            "Total_measure_group_cnt": int(total_grp_cnt[i]),
            "summary_score": (repr(float(summary[i]))
                              if not np.isnan(summary[i]) else ""),
            "star": star[i] if star[i] is not None else "NA",
        })
    outdir = os.path.join(BASE, "work/sas_replay_out")
    os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, f"star_{release}.csv")
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    return {
        "cfg": cfg, "sas_path": sas_path, "outcsv": outcsv,
        "counts": counts, "measure_in": measure_in, "excluded": excluded,
        "n_hospitals_kept": n_hosp,
        "peer_records": peer_records,
        "replay": {r["PROVIDER_ID"]: r for r in out_rows},
    }


def load_snapshot(zpath):
    """Hospital_General_Information.csv from a Care Compare snapshot zip."""
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist()
                if n.endswith("Hospital_General_Information.csv")][0]
        with z.open(name) as f:
            txt = io.TextIOWrapper(f, encoding="utf-8-sig")
            rows = list(csv.DictReader(txt))
    idcol = ("Facility ID" if "Facility ID" in rows[0]
             else "Provider ID" if "Provider ID" in rows[0] else None)
    rcol = "Hospital overall rating"
    fcol = next((c for c in rows[0] if c.startswith("Hospital overall rating foot")),
                None)
    pub = {}
    for r in rows:
        pub[r[idcol]] = (r[rcol], r.get(fcol, "") if fcol else "")
    return pub


def compare(replay, pub):
    ids_r, ids_p = set(replay), set(pub)
    both = ids_r & ids_p
    match = mismatch = both_unrated = 0
    mismatches, r_rated_p_na, p_rated_r_na = [], [], []
    for pid in sorted(both):
        rs = replay[pid]["star"]
        ps, foot = pub[pid]
        rr = rs not in ("NA", "")
        pr = ps not in ("Not Available", "")
        if rr and pr:
            if int(rs) == int(ps):
                match += 1
            else:
                mismatch += 1
                mismatches.append({"provider_id": pid, "replay_star": int(rs),
                                   "published_star": int(ps), "footnote": foot})
        elif rr:
            r_rated_p_na.append({"provider_id": pid, "replay_star": int(rs),
                                 "footnote": foot})
        elif pr:
            p_rated_r_na.append({"provider_id": pid, "published_star": int(ps)})
        else:
            both_unrated += 1
    only_pub_rated = [p for p in ids_p - ids_r
                      if pub[p][0] not in ("Not Available", "")]
    return {
        "ids_in_both": len(both),
        "matched_exactly": match,
        "mismatched": mismatch,
        "both_unrated": both_unrated,
        "replay_rated_published_NotAvailable": len(r_rated_p_na),
        "published_rated_replay_unrated": len(p_rated_r_na),
        "ids_only_in_replay_input": len(ids_r - ids_p),
        "ids_only_in_published": len(ids_p - ids_r),
        "ids_only_in_published_that_are_rated": len(only_pub_rated),
        "match_rate_of_jointly_rated": (round(match / (match + mismatch), 6)
                                        if match + mismatch else None),
        "_mismatches": mismatches,
        "_r_rated_p_na_footnotes": dict(Counter(x["footnote"]
                                                for x in r_rated_p_na)),
        "_p_rated_r_na": p_rated_r_na,
    }


def main(release):
    res = run_pipeline(release)
    raw_dir = os.path.join(BASE, "data/raw", release)
    snapdir = os.path.join(raw_dir, "care_compare_snapshots")
    comparisons = {}
    for z in sorted(os.listdir(snapdir)):
        pub = load_snapshot(os.path.join(snapdir, z))
        c = compare(res["replay"], pub)
        c["snapshot_hospitals"] = len(pub)
        c["zip_md5"] = md5(os.path.join(snapdir, z))
        comparisons[z] = c
    # primary snapshot = the one with the most exact matches
    primary = max(comparisons, key=lambda z: comparisons[z]["matched_exactly"])

    dist = Counter(str(r["star"]) for r in res["replay"].values())
    record = {
        "run": f"sas_replay_{release}",
        "code": ("Python replay of the year's SAS pack (programs 0/1/2 + "
                 "Star_Macros %kmeans) with src/hospital_kmeans.py for the "
                 "PROC FASTCLUS step (independent implementation of the "
                 "procedure as documented in the SAS/STAT User's Guide); "
                 "IEEE-double arithmetic; open points of the documentation "
                 "carried as counters (Notes 1-7 in src/hospital_kmeans.py)"),
        "pack": {"dir": os.path.relpath(res["cfg"]["pack_dir"], BASE),
                 "prog0": res["cfg"]["prog0"],
                 "year": res["cfg"]["year"], "quarter": res["cfg"]["quarter"]},
        "input": {"file": os.path.relpath(res["sas_path"], BASE),
                  "md5": md5(res["sas_path"])},
        "measures": {"listed": len(res["cfg"]["measure_all"]),
                     "included_gt100": len(res["measure_in"]),
                     "excluded_le100": res["excluded"]},
        "hospitals_kept": res["n_hospitals_kept"],
        "star_distribution_replay": dict(sorted(dist.items())),
        "peer_groups": res["peer_records"],
        "fastclus_residue_rollup": {
            "tie_events_total": sum(p.get("tie_events_total", 0)
                                    for p in res["peer_records"].values()),
            "mean_sort_ties": sum(p.get("mean_sort_ties", 0)
                                  for p in res["peer_records"].values()),
            "duplicate_seed_values": sum(p.get("duplicate_seed_values", 0)
                                         for p in res["peer_records"].values()),
            "note": ("all zero => the points the FASTCLUS documentation "
                     "leaves open (equidistant ties, duplicate seeds, "
                     "star-order mean ties; Notes in src/hospital_kmeans.py) "
                     "were never exercised on this year's data"),
        },
        "comparisons": {z: {k: v for k, v in c.items() if not k.startswith("_")}
                        for z, c in comparisons.items()},
        "primary_snapshot": primary,
        "mismatch_detail_primary": comparisons[primary]["_mismatches"][:50],
        "replay_rated_pub_na_footnotes_primary":
            comparisons[primary]["_r_rated_p_na_footnotes"],
        "pub_rated_replay_na_primary": comparisons[primary]["_p_rated_r_na"][:50],
        "output_csv": os.path.relpath(res["outcsv"], BASE),
    }
    year = release.split("-")[0]
    out = os.path.join(BASE, "runs", f"sas_replay_{year}.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{release}] kept={res['n_hospitals_kept']}")
    for z, c in comparisons.items():
        print(f"  {z}: match={c['matched_exactly']} mism={c['mismatched']} "
              f"r-rated-p-NA={c['replay_rated_published_NotAvailable']} "
              f"p-rated-r-NA={c['published_rated_replay_unrated']}")
    print("wrote", out)
    return record


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and len(sys.argv) < 2:
    print("usage: python3 sas_replay.py <release>   (2021-04, 2022-07, 2023-07, 2024-07 or 2025-07)",
          file=sys.stderr)
    sys.exit(2)
if __name__ == "__main__" and not os.path.isdir(os.path.join(BASE, "data/raw", sys.argv[1])):
    print("sas_replay: missing input directory hospital/data/raw/%s/ (the release's SAS package, .sas7bdat "
          "input file and Care Compare snapshots are not included in the package; re-fetch each by the URL and sha256 "
          "in CMS_RAW_PINS.json)" % sys.argv[1],
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1])
