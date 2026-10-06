# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Optimality-gap table on the full score list, ALT-B membership basis.

Identical to exact_gaps.py (HYP = cost+d60r+emp) except that
the 2025 delta set is the ALT-B family (runs/membership_delta_sets_altb.json:
C04-2025 removal value 55 instead of the canonical 54); 2024 and 2026 delta
sets are the canonical ones from runs/membership_delta_sets.json. Output:
runs/exact_gaps_fullsample_altb.json (same row shape as the canonical
table). Purpose: record the upper edge of the off-optimum count across
membership bases/orderings.

Run: cd src && python3 exact_gaps_altb.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (106 of 123 when
    one forced removal is carried out differently).
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os

from exact_gaps import _run, build_jobs
import exact_gaps
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_delta_sets_altb():
    """Canonical 2024/2026 + ALT-B 2025."""
    canon = json.load(open(os.path.join(BASE, "runs",
                                        "membership_delta_sets.json")))
    altb = json.load(open(os.path.join(BASE, "runs",
                                       "membership_delta_sets_altb.json")))
    merged = dict(canon)
    merged["years"] = {"2024": canon["years"]["2024"],
                       "2025": altb["years"]["2025"],
                       "2026": canon["years"]["2026"]}
    merged["register"] = ("merged basis: canonical 2024/2026 "
                          "(membership_delta_sets.json) + ALT-B 2025 "
                          "(membership_delta_sets_altb.json)")
    return merged


def main(workers=os.cpu_count()):
    exact_gaps.load_delta_sets = load_delta_sets_altb
    jobs = build_jobs()
    print(f"jobs: {len(jobs)}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    out = os.path.join(BASE, "runs", "exact_gaps_fullsample_altb.json")
    json.dump(rows, open(out, "w"), indent=1)
    off = [r for r in rows if r["off_optimum"]]
    fe = [r for r in rows if r["membership"] == "fence-exact"]
    fe25 = [r for r in rows if r["year"] == "2025"
            and r["membership"] == "fence-exact"]
    n25 = sum(1 for r in rows if r["year"] == "2025")
    print(f"ALT-B basis: {len(off)}/{len(rows)} off the SSQ optimum; "
          f"fence-exact {len(fe)}/{len(rows)}; 2025 fence-exact {len(fe25)}/{n25}")
    print("saved", out)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
