#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""Mechanism adjudication, stage 2a: is the all-upward correction a NECESSITY
of quintile-median-seeded two-stage Hartigan-Wong, or contingent on data shape?

Records produced:
  A. replica validation: the R replica of fn_kmeans reproduces the published
     stars exactly on all three 2026 peer groups (else everything downstream is void);
  B. MIRROR test: negate the real peer3/peer4 scores, run the SAME published
     procedure -> if every corrected star now moves DOWN, upward-only is not a
     property of the algorithm (the procedure is equivariant under x -> -x);
  C. fresh synthetic counterexamples: small right-skewed datasets where the
     published procedure lands boundaries BELOW the optimum (correction moves DOWN);
  D. skew-direction batch: random left/right-skewed samples, tally move directions;
  E. DP-center probe: Hartigan-Wong seeded AT the exact-DP centers stays at the
     optimum (the algorithm can hold the optimum; the seeding picks the basin).
"""
import csv, json, math, os, random, subprocess, sys
from fractions import Fraction

import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REC + "/src")
from dp_kmeans import dp_optimal_partition, partition_ssq

BASE = _REC + "/hospital"
SCRATCH = f"{BASE}/work/mech_tmp"
os.makedirs(SCRATCH, exist_ok=True)
RSCRIPT = f"{BASE}/work/mech_fn_kmeans.R"

def run_replica(xs_str, tag, seedfile=None):
    """xs_str: list of decimal strings. Returns (stars list aligned to xs, stderr text)."""
    inp = f"{SCRATCH}/{tag}_in.csv"; outp = f"{SCRATCH}/{tag}_out.csv"
    with open(inp, "w") as f:
        f.write("summary_score\n")
        f.write("\n".join(xs_str) + "\n")
    cmd = ["Rscript", RSCRIPT, inp, outp] + ([seedfile] if seedfile else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    stars = [int(row["star"]) for row in csv.DictReader(open(outp))]
    return stars, r.stderr.strip()

def dp_stars(xs_str, k=5):
    dp = dp_optimal_partition(xs_str, k)
    bnds, sx = dp["boundaries"], dp["sorted_x"]
    block_max = [sx[bnds[b + 1] - 1] for b in range(k)]
    def star(v):
        f = Fraction(v)
        for b in range(k):
            if f <= block_max[b]:
                return b + 1
        return k
    return [star(v) for v in xs_str], dp

def compare(tag, xs_str, note=""):
    ship, rlog = run_replica(xs_str, tag)
    dps, dp = dp_stars(xs_str)
    ssq_ship = partition_ssq(xs_str, ship)
    moves = {}
    for a, b in zip(ship, dps):
        if a != b:
            moves[f"{a}->{b}"] = moves.get(f"{a}->{b}", 0) + 1
    n_up = sum(v for k_, v in moves.items() if int(k_[0]) < int(k_[-1]))
    n_dn = sum(v for k_, v in moves.items() if int(k_[0]) > int(k_[-1]))
    xs_f = [float(Fraction(v)) for v in xs_str]
    n = len(xs_f); m = sum(xs_f) / n
    m2 = sum((v - m) ** 2 for v in xs_f) / n
    m3 = sum((v - m) ** 3 for v in xs_f) / n
    res = {
        "n": n, "note": note,
        "moment_skewness": round(m3 / m2 ** 1.5, 4) if m2 > 0 else 0.0,
        "published_ssq": float(ssq_ship), "optimal_ssq": float(dp["ssq"]),
        "published_is_optimal": ssq_ship == dp["ssq"],
        "moves": moves, "moves_up": n_up, "moves_down": n_dn,
        "replica_log": rlog.splitlines(),
    }
    print(f"[{tag}] skew={res['moment_skewness']:+.3f} moves_up={n_up} moves_down={n_dn} {moves}")
    return res

def load_2026():
    rows = list(csv.DictReader(open(f"{BASE}/work/R_output/Star_precap_2026apr.csv")))
    rated = [r for r in rows if r["report_indicator"] == "1" and r["star"] not in ("NA", "")]
    out = {}
    for label, key in [("peer3", "1) # of groups=3"), ("peer4", "2) # of groups=4"),
                       ("peer5", "3) # of groups=5")]:
        out[label] = [(r["summary_score"], int(r["star"])) for r in rated if r["cnt_grp"] == key]
    return out

def neg_str(s):
    return s[1:] if s.startswith("-") else "-" + s

out = {"A_replica_validation": {}, "B_mirror": {}, "C_counterexamples": [],
       "D_skew_batch": {}, "E_dp_center_probe": {}}
data = load_2026()

# --- A: replica must reproduce published stars exactly ---
for label in ["peer3", "peer4", "peer5"]:
    xs = [s for s, _ in data[label]]
    ship_pub = [st for _, st in data[label]]
    rep, rlog = run_replica(xs, f"val_{label}")
    mism = sum(1 for a, b in zip(rep, ship_pub) if a != b)
    out["A_replica_validation"][label] = {
        "n": len(xs), "mismatch_vs_published": mism, "replica_log": rlog.splitlines()}
    print(f"[A {label}] replica-vs-published mismatches: {mism}")

# --- B: mirror test on real data (peer3, peer4) ---
for label in ["peer3", "peer4"]:
    xs = [neg_str(s) for s, _ in data[label]]
    out["B_mirror"][label] = compare(f"mirror_{label}", xs,
        note=f"negated real 2026 {label} scores; same published procedure, same DP")

# --- C: fresh small right-skewed counterexamples ---
random.seed(20260719)
found = 0
trial = 0
while found < 2 and trial < 60:
    trial += 1
    n = random.choice([60, 80, 100])
    shape = random.uniform(1.2, 2.5)
    xs = [f"{random.gammavariate(shape, 1.0):.6f}" for _ in range(n)]
    res = compare(f"cex_{trial}", xs, note=f"gamma(shape={shape:.2f}) right-skewed, n={n}, seed trial {trial}")
    if res["moves_down"] > 0 and res["moves_up"] == 0 and not res["published_is_optimal"]:
        res["trial"] = trial
        res["values"] = xs
        out["C_counterexamples"].append(res)
        found += 1
print(f"[C] counterexamples found: {found} in {trial} trials")

# --- D: skew-direction batch ---
random.seed(4242)
batch = {"left_skew": {"datasets": 0, "nonoptimal": 0, "moves_up": 0, "moves_down": 0, "mixed_direction_datasets": 0},
         "right_skew": {"datasets": 0, "nonoptimal": 0, "moves_up": 0, "moves_down": 0, "mixed_direction_datasets": 0}}
for i in range(40):
    side = "right_skew" if i % 2 == 0 else "left_skew"
    n = 120
    g = [random.gammavariate(1.8, 1.0) for _ in range(n)]
    xs = [f"{(v if side=='right_skew' else -v):.6f}" for v in g]
    res = compare(f"batch_{i}", xs, note=side)
    b = batch[side]
    b["datasets"] += 1
    if not res["published_is_optimal"]:
        b["nonoptimal"] += 1
        b["moves_up"] += res["moves_up"]
        b["moves_down"] += res["moves_down"]
        if res["moves_up"] > 0 and res["moves_down"] > 0:
            b["mixed_direction_datasets"] += 1
out["D_skew_batch"] = {"design": "40 gamma(1.8) samples n=120, alternating sign; "
                                 "left_skew = negated (mass at high values, like the 2026 scores)",
                       **batch}

# --- E: H-W seeded at the exact-DP centers (peer3) stays at the optimum ---
xs = [s for s, _ in data["peer3"]]
dps, dp = dp_stars(xs)
seedf = f"{SCRATCH}/dp_centers_peer3.txt"
with open(seedf, "w") as f:
    f.write("\n".join(repr(float(mu)) for mu in dp["means"]) + "\n")
ship_seeded, rlog = run_replica(xs, "dpseed_peer3", seedfile=seedf)
mism = sum(1 for a, b in zip(ship_seeded, dps) if a != b)
ssq_seeded = partition_ssq(xs, ship_seeded)
out["E_dp_center_probe"] = {
    "group": "peer3",
    "labels_match_dp_optimum": mism == 0,
    "mismatches": mism,
    "ssq_seeded_run": float(ssq_seeded),
    "ssq_optimum": float(dp["ssq"]),
    "seeded_run_is_optimal": ssq_seeded == dp["ssq"],
    "replica_log": rlog.splitlines(),
    "reading": "Hartigan-Wong holds the exact optimum when seeded there: "
               "the published result is a seeding-selected local optimum, not an "
               "algorithmic ceiling."}
print("[E] dp-center probe: match_dp =", mism == 0)

with open(f"{BASE}/work/mech_stage2a_synthetic.json", "w") as f:
    json.dump(out, f, indent=2)
print("record ->", f"{BASE}/work/mech_stage2a_synthetic.json")
