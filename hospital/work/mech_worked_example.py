#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Mechanism analysis, stage 3: minimal worked example.

Find a small left-skewed dataset where the CMS procedure (quintile-median
seeded Hartigan-Wong, two-stage, k=5; the R replica) terminates at a
non-optimal fixed point, and report every number: seeds, converged centers,
both partitions, both SSQs, move directions. Then the same dataset negated
(right-skew) to show that the direction flips.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.3 (mechanism and
    direction of the differences).
Run:  cd hospital/work && python3 mech_worked_example.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; R 4.x with Rscript on PATH.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json, subprocess, sys, random
from fractions import Fraction

sys.path.insert(0, _REC + "/src")
from dp_kmeans import dp_optimal_partition, partition_ssq

BASE = _REC + "/hospital"
SCRATCH = f"{BASE}/work/mech_tmp"
RSCRIPT = f"{BASE}/work/mech_fn_kmeans.R"

def run_replica(xs_str, tag):
    inp = f"{SCRATCH}/{tag}_in.csv"; outp = f"{SCRATCH}/{tag}_out.csv"
    with open(inp, "w") as f:
        f.write("summary_score\n" + "\n".join(xs_str) + "\n")
    r = subprocess.run(["Rscript", RSCRIPT, inp, outp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    stars = [int(row["star"]) for row in csv.DictReader(open(outp))]
    return stars, r.stderr.strip()

def analyze(xs_str, tag):
    ship, rlog = run_replica(xs_str, tag)
    dp = dp_optimal_partition(xs_str, 5)
    bnds, sx = dp["boundaries"], dp["sorted_x"]
    block_max = [sx[bnds[b + 1] - 1] for b in range(5)]
    def dp_star(v):
        f = Fraction(v)
        for b in range(5):
            if f <= block_max[b]:
                return b + 1
        return 5
    dps = [dp_star(v) for v in xs_str]
    moves = {}
    for a, b in zip(ship, dps):
        if a != b:
            moves[f"{a}->{b}"] = moves.get(f"{a}->{b}", 0) + 1
    return {
        "values_sorted": sorted(xs_str, key=lambda s: Fraction(s)),
        "shipped_stars": ship, "dp_stars": dps,
        "shipped_ssq": float(partition_ssq(xs_str, ship)),
        "optimal_ssq": float(dp["ssq"]),
        "shipped_partition_sorted": sorted_partition(xs_str, ship),
        "optimal_partition_sorted": [[str(v) for v in blk] for blk in
                                     [[float(x) for x in b] for b in dp["blocks"]]],
        "dp_centers": [float(m) for m in dp["means"]],
        "moves": moves,
        "moves_up": sum(v for k, v in moves.items() if int(k[0]) < int(k[-1])),
        "moves_down": sum(v for k, v in moves.items() if int(k[0]) > int(k[-1])),
        "replica_log": rlog.splitlines(),
    }

def sorted_partition(xs_str, labels):
    pairs = sorted(zip([Fraction(v) for v in xs_str], labels))
    out = {s: [] for s in sorted(set(labels))}
    for v, l in pairs:
        out[l].append(float(v))
    return [out[s] for s in sorted(out)]

def neg(xs):
    return [s[1:] if s.startswith("-") else "-" + s for s in xs]

if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not _os.path.isdir(SCRATCH):
    print("mech_worked_example: scratch directory hospital/work/mech_tmp/ is missing (mech_synthetic.py creates it; "
          "run that stage first, or create the directory)",
          file=sys.stderr)
    sys.exit(2)

random.seed(7)
best = None
for trial in range(400):
    n = random.choice([15, 18, 20])
    # left-skew: dense mass high, sparse tail low  (negated gamma, 1 decimal)
    xs = sorted(round(10 - random.gammavariate(1.5, 1.2), 1) for _ in range(n))
    xs_str = [f"{v:.1f}" for v in xs]
    if len(set(xs_str)) < 10:
        continue
    try:
        res = analyze(xs_str, f"wex_{trial}")
    except RuntimeError:
        continue
    if res["moves_up"] > 0 and res["moves_down"] == 0 and \
       res["shipped_ssq"] > res["optimal_ssq"]:
        # prefer the smallest instance with the fewest moves
        score = (len(xs_str), sum(res["moves"].values()))
        if best is None or score < best[0]:
            best = (score, xs_str, res, trial)
            if len(xs_str) == 15 and sum(res["moves"].values()) <= 3:
                break

(_, xs_str, res, trial) = best
mirror = analyze(neg(xs_str), "wex_mirror")
out = {"worked_example": {"n": len(xs_str), "search_trial": trial, **res},
       "worked_example_mirror": {"n": len(xs_str), **mirror}}
with open(f"{BASE}/work/mech_stage3_worked.json", "w") as f:
    json.dump(out, f, indent=2)
print("n =", len(xs_str), "trial", trial)
print("values:", res["values_sorted"])
print("CMS partition:", res["shipped_partition_sorted"])
print("optimal partition:", res["optimal_partition_sorted"])
print("ssq CMS/optimal: %.6f / %.6f" % (res["shipped_ssq"], res["optimal_ssq"]))
print("moves:", res["moves"])
print("replica log:", *res["replica_log"], sep="\n  ")
print("mirror moves:", mirror["moves"], "up=%d down=%d" % (mirror["moves_up"], mirror["moves_down"]))
