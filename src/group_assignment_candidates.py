# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Full 80-combination candidate grid for the group-assignment step (10 readouts x
MT/RANUNI x 4 input orders), screened against the published cut points.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (the 80-combination candidate grid for the group-assignment step).
Run:  cd src && python3 group_assignment_candidates.py
Requires: Python >= 3.10; numpy, scipy.
"""
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from fractions import Fraction

def main(workers=None):
    from stage2_battery import build_worklist
    from falsify_groups import round_display
    from order_grid_screen import run_combo, ALGS, STREAMS, ORDERS
    work = build_worklist(("2024", "2025", "2026"))
    jobs = [(w, a, s, o) for w in work for a in ALGS for s in STREAMS for o in ORDERS]
    print(f"targets: {len(work)}; jobs: {len(jobs)}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers or os.cpu_count() or 1) as pool:
        res = pool.map(run_combo, jobs)
    tally = Counter(); denom = Counter(); perfect = Counter(); rows = []
    for (w, a, s, o), (mid, alg, stream, order, snap, means) in zip(jobs, res):
        key = f"{alg}|{stream}|{order}"
        if means is None:
            rows.append({"year": w["year"], "mid": w["mid"], "org": w["org"], "combo": key, "error": True})
            continue
        dd = max(len(v.split(".")[1]) if "." in v else 0 for v in w["target"])
        exact = sum(1 for m, t in zip(means, w["target"])
                    if Fraction(round_display(Fraction(str(m)).limit_denominator(10**9), dd)) == Fraction(t))
        tally[key] += exact; denom[key] += 4
        if exact == 4: perfect[key] += 1
        rows.append({"year": w["year"], "mid": w["mid"], "org": w["org"], "combo": key,
                     "means": means, "target": w["target"], "n_exact": exact})
    summary = {k: {"boundaries": f"{tally[k]}/{denom[k]}", "targets_4of4": perfect[k]}
               for k in sorted(tally, key=lambda kk: -tally[kk])}
    json.dump({"register": "Full 80-combination grid (10 readouts x MT/RANUNI x 4 input orders), ward_tie arm, reconciled membership, all fence-exact targets; the screening criterion is agreement across targets (a single target at 4/4 is chance-level)",
               "n_targets": len(work), "n_combos": 80, "summary": summary, "rows": rows},
              open("../runs/stage2_full_grid.json", "w"), indent=1)
    print("saved; best 4/4-target counts:", sorted(perfect.values(), reverse=True)[:5], flush=True)

if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
