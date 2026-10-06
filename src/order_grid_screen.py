# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""3-axis candidate screen: GROUPS= algorithm x RNG stream x input order,
on the sharpest fence-exact discriminating targets. The documented PROC CLUSTER
tie rule makes row order enter twice (group assignment and Ward observation
numbering), so input orders are screened as candidates.

Scoring: snap = |group-mean - published| < 0.5 per boundary. Survivors (>=3/4 on
the primary target) get validated on the secondary targets inline.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (candidate reconstructions of the group-assignment step).
Run:  cd src && python3 order_grid_screen.py C01 2024
Exit codes: 0 on success; 2 on a missing or malformed argument (one line on
    stderr says what to pass).
Requires: Python >= 3.10; numpy, scipy.
"""

import multiprocessing as mp
import os
import sys
from fractions import Fraction

from stage2_battery import build_worklist
from pipeline import measure_cutpoints
import groups_assign
import re
from groups_assign import group_sizes, floyd_sample, fisher_yates
from mt19937_stream import SasMT, SasRanuni

ALGS = ["seq_quota_last", "sort_blocks_last", "sort_blocks_first", "sort_deal",
        "floyd_last", "floyd_first", "fy_down_blocks", "fy_up_blocks",
        "fy_down_deal", "fy_up_deal"]
STREAMS = {"MT": SasMT, "RANUNI": SasRanuni}
ORDERS = ["cid_asc", "cid_desc", "score_asc", "score_desc"]


def gen_gids(alg, rng_cls, n, seed=8675309, g=10):
    rng = rng_cls(seed)
    if alg.startswith("seq_quota"):
        remainder = alg.split("_")[-1]
        remaining = group_sizes(n, g, remainder)[:]
        out = []
        for _ in range(n):
            u = rng.uniform()
            total = sum(remaining)
            x = u * total
            acc = 0
            gid = None
            for j in range(g):
                nxt = acc + remaining[j]
                if x < nxt or (j == g - 1 and gid is None):
                    gid = j
                    break
                acc = nxt
            remaining[gid] -= 1
            out.append(gid + 1)
        return out
    if alg.startswith("sort_blocks") or alg == "sort_deal":
        us = [rng.uniform() for _ in range(n)]
        order = sorted(range(n), key=us.__getitem__)
        out = [0] * n
        if alg == "sort_deal":
            for rank, i in enumerate(order):
                out[i] = (rank % g) + 1
            return out
        remainder = alg.split("_")[-1]
        sizes = group_sizes(n, g, remainder)
        pos = 0
        for gid, sz in enumerate(sizes, 1):
            for i in order[pos:pos + sz]:
                out[i] = gid
            pos += sz
        return out
    if alg.startswith("floyd"):
        remainder = alg.split("_")[-1]
        sizes = group_sizes(n, g, remainder)
        remaining = list(range(n))
        out = [0] * n
        for gid, sz in enumerate(sizes, 1):
            pick = floyd_sample(rng, len(remaining), sz)
            for i in [remaining[i - 1] for i in sorted(pick)]:
                out[i] = gid
            remaining = [i for i in remaining if out[i] == 0]
        return out
    if alg.startswith("fy"):
        _, direction, mode = alg.split("_")
        a = fisher_yates(rng, n, direction)
        out = [0] * n
        if mode == "deal":
            for p, idx in enumerate(a):
                out[idx] = (p % g) + 1
            return out
        sizes = group_sizes(n, g, "last")
        pos = 0
        for gid, sz in enumerate(sizes, 1):
            for p in range(pos, pos + sz):
                out[a[p]] = gid
            pos += sz
        return out
    raise ValueError(alg)


def order_pairs(kept, order):
    if order == "cid_asc":
        return kept
    if order == "cid_desc":
        return kept[::-1]
    if order == "score_asc":
        return sorted(kept, key=lambda t: (Fraction(t[1]), t[0]))
    if order == "score_desc":
        return sorted(kept, key=lambda t: (-Fraction(t[1]), t[0]))
    raise ValueError(order)


def run_combo(args):
    w, alg, stream, order = args
    pairs = order_pairs(w["kept"], order)
    scores = [s for _, s in pairs]
    rng_cls = STREAMS[stream]

    def cand(n, g, s):
        return gen_gids(alg, rng_cls, n, s, g)

    key = f"__grid_{alg}_{stream}_{order}"
    groups_assign.CANDIDATES[key] = cand
    try:
        r = measure_cutpoints(scores, higher_is_better=w["higher"], tukey=True,
                              cap_lo="0", cap_hi=w["cap_hi"],
                              groups_candidate=key, arm="ward_tie")
    finally:
        del groups_assign.CANDIDATES[key]
    if "error" in r or len(r["mean_cutpoints"]) != 4:
        return (w["mid"], alg, stream, order, -1, None)
    means = [float(c) for c in r["mean_cutpoints"]]
    snap = sum(1 for m, t in zip(means, w["target"]) if abs(m - float(t)) < 0.5)
    return (w["mid"], alg, stream, order, snap, means)


def main(primary_mid="C01", year="2024", threshold=3, workers=None):
    workers = workers or os.cpu_count() or 1
    work = build_worklist((year,))
    prim = next(x for x in work if x["mid"] == primary_mid)
    combos = [(prim, a, s, o) for a in ALGS for s in STREAMS for o in ORDERS]
    print(f"primary {primary_mid}-{year} pub={prim['target']}: {len(combos)} combos")
    with mp.Pool(workers) as pool:
        res = pool.map(run_combo, combos)
    res.sort(key=lambda r: -r[4])
    survivors = []
    for mid, alg, stream, order, snap, means in res[:15]:
        line = f"  {alg:>18} {stream:>6} {order:>10}: snap {snap}/4"
        if means:
            line += f" {[f'{m:.1f}' for m in means]}"
        print(line)
        if snap >= threshold:
            survivors.append((alg, stream, order))
    if not survivors:
        print("no survivors at threshold", threshold)
        return
    print(f"\nvalidating {len(survivors)} survivors on secondary targets:")
    secondary = [x for x in work if x["mid"] in ("C02", "C04", "C08", "C06", "C14")]
    jobs = [(w, a, s, o) for w in secondary for (a, s, o) in survivors]
    with mp.Pool(workers) as pool:
        res2 = pool.map(run_combo, jobs)
    from collections import Counter
    score = Counter()
    for mid, alg, stream, order, snap, means in res2:
        score[(alg, stream, order)] += max(snap, 0)
        print(f"  {mid}: {alg}/{stream}/{order} snap {snap}/4 "
              f"{[f'{m:.1f}' for m in means] if means else ''}")
    print("\naggregate over secondaries:")
    for k, v in score.most_common():
        print(f"  {k}: {v}/{4 * len(secondary)}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and len(sys.argv) > 1 and not (re.fullmatch(r"[CD]\d{2}", sys.argv[1])
                            and (len(sys.argv) < 3 or sys.argv[2] in ("2024", "2025", "2026"))):
    print("usage: python3 order_grid_screen.py [MEASURE_ID [STAR_YEAR]]   e.g. C01 2024 (a Part C/D measure "
          "id among the fence-exact targets of stage2_battery.build_worklist and a star year 2024-2026)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["C01", "2024"]))
