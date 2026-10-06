#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Exact-optimum vs published census for the 2026 hospital-stars release,
binary-double register.

Register: scores enter as Fraction(float) of the R replay's IEEE doubles, dumped
at %.17g by work/dump_fullprec_2026.R (round-trip asserted in R). This is the
same register as the SAS-era records (exact_census_2021..2025). The CMS
side is the pre-cap k-means star (Star_ / star_precap): the 2026 safety cap is
a post-assignment step, examined separately in runs/safety_cap_check_2026.json.
The replay reproduces the published post-cap stars on all 3,182 jointly rated
hospitals (runs/replay_2026_pilot.json).

Register-stability cross-check: the decimal register (15-significant-digit
strings of R write.csv, the dp_taste_2026 register) is recomputed with provider
IDs from the same rows, verified to reproduce runs/dp_taste_2026.json, and the
differ sets are compared by provider ID.

Usage:
  python3 exact_census_2026.py pilot   # peer3 only, both registers
  python3 exact_census_2026.py full    # all three peer groups, both registers

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.2, Figure 1 and
    Table 1 (2026: 213 of 3,203 before the safety cap); Appendix C.1.
Exit codes: 0 on success; 2 on a missing or malformed argument (one line on
    stderr says what to pass).
Requires: Python >= 3.10, standard library only.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import json
import sys
from collections import Counter
from fractions import Fraction

sys.path.insert(0, _REC + "/src")

BASE = _REC + "/hospital"
PEERS = {3: "peer3", 4: "peer4", 5: "peer5"}


def load_rows():
    """Aligned rated rows: (provider_id, peer_cnt, star_precap, s17, s15)."""
    dump = {r["PROVIDER_ID"]: r for r in csv.DictReader(
        open(f"{BASE}/work/R_output/fullprec_2026apr.csv"))}
    prec = {r["PROVIDER_ID"]: r for r in csv.DictReader(
        open(f"{BASE}/work/R_output/Star_precap_2026apr.csv"))}
    assert set(dump) == set(prec)
    rows = []
    for pid in sorted(dump):
        d = dump[pid]
        if d["report_indicator"] == "1" and d["star_precap"] not in ("NA", ""):
            assert d["star_precap"] == prec[pid]["star"]
            rows.append((pid, int(d["Total_measure_group_cnt"]),
                         int(d["star_precap"]), d["summary_score_17g"],
                         prec[pid]["summary_score"]))
    return rows


def census_one(task):
    """One (register, peer) census. task = (register, peer_cnt, rows-subset)."""
    from dp_kmeans import (dp_optimal_partition, dp_optimal_partitions_all,
                           partition_ssq)
    register, tg, sub = task
    label = PEERS[tg]
    if register == "binary":
        xs = [Fraction(float(s17)) for (_, _, _, s17, _) in sub]
    else:  # decimal: exact rational of the 15-digit write.csv string (dp_taste register)
        xs = [Fraction(s15) for (_, _, _, _, s15) in sub]
    stars_cms = [s for (_, _, s, _, _) in sub]
    pids = [p for (p, _, _, _, _) in sub]
    dp = dp_optimal_partition(xs, 5)
    ssq_opt, all_opt = dp_optimal_partitions_all(xs, 5)
    bnds, sorted_x = dp["boundaries"], dp["sorted_x"]
    boundary_ties = sum(1 for b in range(1, 5)
                        if sorted_x[bnds[b] - 1] == sorted_x[bnds[b]])
    block_max = [sorted_x[bnds[b + 1] - 1] for b in range(5)]

    def dp_star(v):
        for b in range(5):
            if v <= block_max[b]:
                return b + 1
        return 5

    stars_dp = [dp_star(v) for v in xs]
    ssq_cms = partition_ssq(xs, stars_cms)
    gap = ssq_cms - dp["ssq"]
    moves = Counter()
    detail = []
    for pid, a, b in zip(pids, stars_cms, stars_dp):
        if a != b:
            moves[f"{a}->{b}"] += 1
            detail.append({"provider_id": pid, "peer": label,
                           "shipped": a, "dp_optimum": b})
    return {
        "register": register, "peer": label,
        "n": len(sub),
        "cms_star_counts": {s: stars_cms.count(s) for s in range(1, 6)},
        "dp_star_counts": {s: stars_dp.count(s) for s in range(1, 6)},
        "ssq_cms_float": float(ssq_cms),
        "ssq_optimal_float": float(dp["ssq"]),
        "ssq_gap_float": float(gap),
        "ssq_gap_exact": f"{gap.numerator}/{gap.denominator}",
        "cms_is_optimal": ssq_cms == dp["ssq"],
        "n_optimal_partitions": len(all_opt),
        "boundary_value_ties": boundary_ties,
        "hospitals_star_differs": len(detail),
        "moves": dict(moves),
        "detail": detail,
    }


def main(mode):
    rows = load_rows()
    print(f"rated: {len(rows)}", flush=True)
    by_peer = {tg: [r for r in rows if r[1] == tg] for tg in (3, 4, 5)}
    reg_diff = {tg: sum(1 for (_, _, _, s17, s15) in by_peer[tg]
                        if float(s17) != float(s15)) for tg in (3, 4, 5)}

    if mode == "pilot":
        tasks = [("binary", 3, by_peer[3]), ("decimal", 3, by_peer[3])]
        results = [census_one(t) for t in tasks]
        for r in results:
            print(f"[pilot] {r['register']}/{r['peer']} n={r['n']} "
                  f"diff={r['hospitals_star_differs']} moves={r['moves']} "
                  f"gap={r['ssq_gap_float']:.6g} "
                  f"unique={r['n_optimal_partitions'] == 1}", flush=True)
        b, d = results
        same = ({x["provider_id"] for x in b["detail"]} ==
                {x["provider_id"] for x in d["detail"]})
        print(f"[pilot] peer3 differ-sets identical across registers: {same}")
        print(f"[pilot] input doubles differing between registers in peer3: "
              f"{reg_diff[3]}/{len(by_peer[3])}", flush=True)
        return

    # full: the six (register, peer group) tasks run in a process pool
    import multiprocessing as mp
    tasks = [(reg, tg, by_peer[tg]) for reg in ("binary", "decimal")
             for tg in (3, 4, 5)]
    with mp.get_context("spawn").Pool(6) as pool:
        results = pool.map(census_one, tasks)
    res = {(r["register"], r["peer"]): r for r in results}
    for r in results:
        print(f"[full] {r['register']}/{r['peer']} n={r['n']} "
              f"diff={r['hospitals_star_differs']} gap={r['ssq_gap_float']:.6g} "
              f"unique={r['n_optimal_partitions'] == 1}", flush=True)

    # ---- assemble the record (binary register), same schema as the SAS-era files ----
    out = {
        "run": "exact_census_2026",
        "register": ("Exact comparison in binary-double arithmetic: Fraction(float) "
                     "of the R replay's IEEE doubles (work/dump_fullprec_2026.R, %.17g "
                     "round-trip asserted in R), the same arithmetic as the SAS-era "
                     "records. The CMS side is the pre-cap k-means star (Star_); the "
                     "2026 safety cap is a post-assignment adjustment checked in "
                     "runs/safety_cap_check_2026.json. The replay matches the published "
                     "post-cap stars with zero mismatches (runs/replay_2026_pilot.json)."),
        "peer_groups": {},
        "total_rated": len(rows),
    }
    total_diff = 0
    moves_total = Counter()
    detail_all = []
    for tg in (3, 4, 5):
        r = res[("binary", PEERS[tg])]
        total_diff += r["hospitals_star_differs"]
        moves_total.update(r["moves"])
        detail_all.extend(r["detail"])
        out["peer_groups"][PEERS[tg]] = {k: r[k] for k in (
            "n", "cms_star_counts", "dp_star_counts", "ssq_cms_float",
            "ssq_optimal_float", "ssq_gap_float", "ssq_gap_exact",
            "cms_is_optimal", "n_optimal_partitions", "boundary_value_ties",
            "hospitals_star_differs", "moves")}
    ups = sum(v for k, v in moves_total.items()
              if int(k.split("->")[1]) > int(k.split("->")[0]))
    downs = sum(v for k, v in moves_total.items()
                if int(k.split("->")[1]) < int(k.split("->")[0]))
    out["total_hospitals_star_differs"] = total_diff
    out["moves_total"] = dict(moves_total)
    out["moves_up"] = ups
    out["moves_down"] = downs
    out["star_differs_detail"] = detail_all

    # ---- register stability vs the decimal (dp_taste) register ----
    taste = json.load(open(f"{BASE}/runs/dp_taste_2026.json"))
    stab = {"decimal_register_reproduces_dp_taste_2026": True,
            "per_peer": {}, "input_doubles_differing_15digit_vs_binary": {}}
    for tg in (3, 4, 5):
        lb = PEERS[tg]
        db = res[("binary", lb)]
        dd = res[("decimal", lb)]
        tp = taste["peer_groups"][lb]
        repro = (dd["hospitals_star_differs"] == tp["hospitals_star_differs"]
                 and dd["moves"] == tp["moves"]
                 and dd["n_optimal_partitions"] == tp["n_optimal_partitions"]
                 and dd["cms_star_counts"] ==
                 {int(k): v for k, v in tp["cms_star_counts"].items()}
                 and dd["dp_star_counts"] ==
                 {int(k): v for k, v in tp["dp_star_counts"].items()})
        if not repro:
            stab["decimal_register_reproduces_dp_taste_2026"] = False
        set_b = {x["provider_id"]: x["dp_optimum"] for x in db["detail"]}
        set_d = {x["provider_id"]: x["dp_optimum"] for x in dd["detail"]}
        entering = sorted(set(set_b) - set(set_d))  # differ under binary only
        leaving = sorted(set(set_d) - set(set_b))   # differ under decimal only
        move_changed = sorted(p for p in set(set_b) & set(set_d)
                              if set_b[p] != set_d[p])
        stab["per_peer"][lb] = {
            "decimal_reproduces_dp_taste": repro,
            "differs_binary": db["hospitals_star_differs"],
            "differs_decimal": dd["hospitals_star_differs"],
            "hospitals_entering_at_binary_register": entering,
            "hospitals_leaving_at_binary_register": leaving,
            "hospitals_same_but_dp_star_changed": move_changed,
            "ssq_gap_float_binary": db["ssq_gap_float"],
            "ssq_gap_float_decimal": dd["ssq_gap_float"],
            "n_optimal_partitions_binary": db["n_optimal_partitions"],
            "n_optimal_partitions_decimal": dd["n_optimal_partitions"],
            "boundary_value_ties_binary": db["boundary_value_ties"],
        }
        stab["input_doubles_differing_15digit_vs_binary"][lb] = (
            f"{reg_diff[tg]}/{len(by_peer[tg])}")
    stab["all_upward_binary"] = downs == 0
    out["register_stability"] = stab

    dest = f"{BASE}/runs/exact_census_2026.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[full] TOTAL differs={total_diff} of {len(rows)} (up={ups}, "
          f"down={downs}); register-stable={stab}", flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and len(sys.argv) < 2:
    print("usage: python3 exact_census_2026.py <mode>   (pilot: peer group 3 only; full: all three peer groups)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1])
