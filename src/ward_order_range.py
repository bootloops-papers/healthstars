# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Order dependence of the full-score-list Ward comparator. Per computation:
the documented-tie-rule Ward partition is recomputed under 8 input orderings
(stored ascending order, reversed, and six seeded uniform permutations), its
exact SSQ compared to the stored exact optimum, and the off-optimum verdict
recorded per ordering. Output:
runs/ward_order_range.json (verdict-stable splits, the off-count range, and
the splits whose verdict is order-dependent).
Run: python3 ward_order_range.py [workers]

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (eight input
    orders: 103 to 106 computations above the minimum).
Requires: Python >= 3.10; numpy, scipy.
"""
import json
import multiprocessing as mp
import os
import random
import sys
from fractions import Fraction as F

import exact_gaps_docsonly as G
from ward_cluster_tie import ward_sas_cut
from dp_kmeans import partition_ssq

BASE = G.BASE
N_PERM = 6
SEED0 = 20260911


def _run(j):
    from tukey import tukey_trim
    key = (j["year"], j["mid"], j["org"])
    gaps = {(g["year"], g["mid"], g["org"]): g for g in
            json.load(open(os.path.join(BASE, "runs/exact_gaps_fullsample_docsonly.json")))}
    g = gaps.get(key)
    if g is None or len(g.get("dp_cutpoints", [])) != 4:
        return dict(key=list(key), skip="fewer-than-five-distinct-scores-or-missing")
    dp_ssq = F(g["dp_ssq"])
    kept = j["kept"]
    orders = {"asc": sorted(kept), "desc": sorted(kept, reverse=True)}
    for i in range(N_PERM):
        rng = random.Random(SEED0 + i)
        p = sorted(kept)
        rng.shuffle(p)
        orders[f"perm{i}"] = p
    out = {}
    for name, orderd in orders.items():
        # mirror exact_gaps_docsonly._run exactly, on this input order
        scores = [s for _, s in orderd]
        mask, _, _ = tukey_trim(scores, cap_lo="0", cap_hi=j["cap_hi"])
        trimmed = [s for s, m in zip(scores, mask) if m]
        vals = [float(s) for s in trimmed]
        k = min(5, len(set(vals)))
        try:
            ward_cl = ward_sas_cut(vals, k)
            labels = [0] * len(vals)
            for ci, c in enumerate(ward_cl):
                for i2 in c:
                    labels[i2] = ci
            ssq = partition_ssq(trimmed, labels)
            out[name] = (str(ssq), ssq > dp_ssq)
        except Exception as e:
            out[name] = f"error:{type(e).__name__}"
    verdicts = {v[1] for v in out.values() if isinstance(v, tuple)}
    return dict(key=list(key),
                per_order={kk: (v if isinstance(v, str) else [v[0], v[1]])
                           for kk, v in out.items()},
                verdict_stable=(len(verdicts) == 1),
                n_orders_ok=sum(1 for v in out.values() if isinstance(v, tuple)))


def main(workers=None):
    if workers is None:
        workers = os.cpu_count() or 1
    jobs = G.build_jobs()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    ok = [r for r in rows if "skip" not in r]
    unstable = [r for r in ok if not r["verdict_stable"]]
    # off-count per ordering name
    names = ["asc", "desc"] + [f"perm{i}" for i in range(N_PERM)]
    counts = {}
    for nm in names:
        c = 0
        for r in ok:
            v = r["per_order"].get(nm)
            if isinstance(v, list) and v[1]:
                c += 1
        counts[nm] = c
    out = dict(
        register=("Order dependence of the full-score-list Ward comparator: "
                  "the exact-optimum side is order-free by construction, and "
                  "this record gives the Ward side's off-optimum verdict per "
                  "computation under 8 input orderings and the off-count "
                  "range."),
        generator="src/ward_order_range.py",
        n_computations=len(ok), n_orderings=len(names),
        off_count_by_ordering=counts,
        off_count_range=[min(counts.values()), max(counts.values())],
        n_verdict_order_dependent=len(unstable),
        order_dependent_keys=[r["key"] for r in unstable],
        rows=rows)
    p = os.path.join(BASE, "runs/ward_order_range.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print("off range:", out["off_count_range"], "| order-dependent verdicts:",
          len(unstable), [r["key"] for r in unstable][:8])
    print("->", p)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
