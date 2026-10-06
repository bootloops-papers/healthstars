# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Inverse-membership solution families for every missed outlier-bound target
(all years, best hypothesis cost+d60r+emp). Output: runs/membership_families.json.

Per target: minimal-k families from inverse_membership.solve_measure with
per-precision grids (integer measures: 0..100 integer grid; 2dp measures: 2dp
grid over [0, observed_max + 0.5]). k_max=3 for the 2dp complaints family (the
stage-2 discriminators), k_max=2 elsewhere.

Cross-measure stacking happens in the stacking scripts; this script stores the
families. The families are exact-solution sets, minimal k only.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (reconstructing
    the score list: filled-in scores, Table C1).
Run:  cd src && python3 solve_all_misses.py
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import sys
from fractions import Fraction as F

from inverse_membership2 import solve_profiles
from tukey import tukey_fences
from tukey_battery import scores_by_measure, load_summary_meta, apply_hypothesis

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(BASE, "spec")
HYP = "cost+d60r+emp"
SCORE_VINTAGE = {"2024": "original", "2025": "original", "2026": "current"}
COMPLAINTS = {"Complaints"}  # name fragment for 2dp family


def build_miss_list():
    misses = []
    for year in ("2024", "2025", "2026"):
        vintage = SCORE_VINTAGE[year]
        vt = json.load(open(os.path.join(SPEC, f"validation_targets_{year}.json")))
        sc = scores_by_measure(year, vintage)
        meta = load_summary_meta(year, vintage)
        disaster = year in ("2024", "2025")
        for r in vt["tukey_cutoffs"]["rows"]:
            mid = r["measure_id"]
            org_key = r.get("org_type") or "ALL"
            pairs = sc.get(mid)
            if not pairs:
                continue
            if org_key in ("MA-PD", "PDP"):
                sub = [(c, s) for c, s in pairs
                       if ("PDP" in meta.get(c, ("",))[0]) == (org_key == "PDP")]
            else:
                sub = pairs
            kept = apply_hypothesis(sub, meta, HYP, disaster)
            if len(kept) < 30:
                continue
            plo, phi = r["lower"], r["upper"]
            dd = max(len(v.split(".")[1]) if v and "." in v else 0
                     for v in (plo, phi) if v)
            is_pct = dd == 0 and phi is not None and F(phi) <= 100
            cap_hi = F(100) if is_pct else None
            lo, hi = tukey_fences([s for _, s in kept], cap_lo="0",
                                  cap_hi=str(cap_hi) if cap_hi else None)
            ok = (plo is None or lo == F(plo)) and (phi is None or hi == F(phi))
            if ok:
                continue
            misses.append({"year": year, "vintage": vintage, "mid": mid,
                           "org": org_key, "dd": dd,
                           "values": [str(s) for _, s in kept],
                           "pub_lo": plo, "pub_hi": phi,
                           "cap_lo": "0", "cap_hi": str(cap_hi) if cap_hi else None})
    return misses


def _solve(m):
    vals = [F(s) for s in m["values"]]
    try:
        sols = solve_profiles(vals, m["pub_lo"], m["pub_hi"],
                              cap_lo=F(m["cap_lo"]),
                              cap_hi=F(m["cap_hi"]) if m["cap_hi"] else None,
                              k_max=4)
    except Exception as e:
        return {**{k: m[k] for k in ("year", "mid", "org")},
                "error": repr(e)}
    return {**{k: m[k] for k in ("year", "mid", "org")},
            "n": len(m["values"]), "pub": [m["pub_lo"], m["pub_hi"]],
            "n_profiles": len(sols),
            "profiles": [{kk: s[kk] for kk in
                          ("k_add", "k_rem", "adds", "rems", "constraints")}
                         for s in sols]}


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    misses = build_miss_list()
    print(f"missed outlier-bound targets: {len(misses)}")
    for m in misses:
        print(f"  {m['year']} {m['mid']} {m['org']} dd={m['dd']} "
              f"pub=({m['pub_lo']},{m['pub_hi']}) n={len(m['values'])}")
    ctx = mp.get_context("spawn")  # spawn: fork is unsafe with BLAS threads
    with ctx.Pool(min(os.cpu_count() or 1, len(misses))) as pool:
        out = pool.map(_solve, misses)
    path = os.path.join(BASE, "runs", "membership_families.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nresults:")
    for r in out:
        if "error" in r:
            print(f"  {r['year']} {r['mid']} {r['org']}: ERROR {r['error']}")
        else:
            profs = [(p["adds"], p["rems"]) for p in r["profiles"]]
            print(f"  {r['year']} {r['mid']} {r['org']}: {r['n_profiles']} "
                  f"profiles {profs[:6]}")
    print("saved", path)
