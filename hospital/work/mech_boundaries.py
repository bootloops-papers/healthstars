#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""Mechanism adjudication, stage 1: dissect published vs exact-DP cluster
boundaries per peer group (2026 release, pre-cap register of dp_taste_2026).

Published boundaries come from the replayed (== published) star labels on the
decimal-string scores of R write.csv; DP boundaries from dp_kmeans exact DP.
A cut between adjacent clusters is reported as the OPEN INTERVAL
(max of lower block, min of upper block); shift = published midpoint - DP
midpoint; crossing count = # hospitals strictly between the two cuts
(= # reassigned across that boundary).
"""
import csv, json, sys
from fractions import Fraction
from multiprocessing import Pool

import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REC + "/src")
from dp_kmeans import dp_optimal_partition

BASE = _REC + "/hospital"
GROUPS = [("peer3", "1) # of groups=3"), ("peer4", "2) # of groups=4"),
          ("peer5", "3) # of groups=5")]

def load():
    rows = list(csv.DictReader(open(f"{BASE}/work/R_output/Star_precap_2026apr.csv")))
    rated = [r for r in rows if r["report_indicator"] == "1" and r["star"] not in ("NA", "")]
    out = {}
    for label, key in GROUPS:
        sub = [(r["summary_score"], int(r["star"])) for r in rated if r["cnt_grp"] == key]
        out[label] = sub
    return out

def one_group(args):
    label, sub = args
    xs = [s for s, _ in sub]
    dp = dp_optimal_partition(xs, 5)
    pairs = sorted((Fraction(s), st) for s, st in sub)
    sx = [p[0] for p in pairs]
    stars = [p[1] for p in pairs]
    n = len(sx)
    # published cut indices: first index where star increments
    ship_idx = []
    for s in range(1, 5):
        i = next(i for i in range(n) if stars[i] == s + 1)
        ship_idx.append(i)
    dp_idx = dp["boundaries"][1:5]
    bnds = []
    for b in range(4):
        si, di = ship_idx[b], dp_idx[b]
        ship_lo, ship_hi = sx[si - 1], sx[si]
        dp_lo, dp_hi = sx[di - 1], sx[di]
        ship_mid = (ship_lo + ship_hi) / 2
        dp_mid = (dp_lo + dp_hi) / 2
        bnds.append({
            "cut": f"{b+1}|{b+2}",
            "published_cut_interval": [float(ship_lo), float(ship_hi)],
            "dp_cut_interval": [float(dp_lo), float(dp_hi)],
            "shift_published_minus_dp": float(ship_mid - dp_mid),
            "published_minus_dp_index": si - di,
            "hospitals_crossing": abs(si - di),
            "direction": "published ABOVE dp" if si > di else
                         ("published BELOW dp" if si < di else "equal"),
        })
    # skew stats (floats fine for descriptive stats)
    fx = [float(v) for v in sx]
    m = sum(fx) / n
    m2 = sum((v - m) ** 2 for v in fx) / n
    m3 = sum((v - m) ** 3 for v in fx) / n
    med = (fx[n // 2] if n % 2 else (fx[n // 2 - 1] + fx[n // 2]) / 2)
    p = lambda q: fx[min(n - 1, int(q * (n - 1)))]
    skew = {
        "n": n,
        "moment_skewness": m3 / m2 ** 1.5,
        "mean": m, "median": med, "mean_minus_median": m - med,
        "low_tail_span_P0_P20": p(0.20) - fx[0],
        "high_tail_span_P80_P100": fx[-1] - p(0.80),
    }
    return label, bnds, skew, {
        "dp_boundary_indices": dp_idx, "published_boundary_indices": ship_idx,
        "dp_block_sizes": [dp["boundaries"][b+1]-dp["boundaries"][b] for b in range(5)],
        "published_block_sizes": [([0]+ship_idx+[n])[b+1]-([0]+ship_idx+[n])[b] for b in range(5)],
        "dp_centers": [float(mu) for mu in dp["means"]],
    }

if __name__ == "__main__":
    data = load()
    with Pool(3) as pool:
        res = pool.map(one_group, [(l, data[l]) for l, _ in GROUPS])
    out = {}
    n_up = n_dn = n_eq = 0
    for label, bnds, skew, extra in res:
        out[label] = {"boundaries": bnds, "skew": skew, **extra}
        for b in bnds:
            if b["published_minus_dp_index"] > 0: n_up += 1
            elif b["published_minus_dp_index"] < 0: n_dn += 1
            else: n_eq += 1
    out["_summary"] = {"boundaries_total": 12, "published_above_dp": n_up,
                       "published_below_dp": n_dn, "equal": n_eq}
    with open(f"{BASE}/work/mech_stage1_boundaries.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["_summary"]))
    for label, bnds, skew, extra in res:
        print(label, "skew=%.3f" % skew["moment_skewness"])
        for b in bnds:
            print("  cut", b["cut"], "shift=%+.4f" % b["shift_published_minus_dp"],
                  "crossing=%d" % b["hospitals_crossing"], b["direction"])
