# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""ALT-B membership delta set — C04-2025 removal value 55 (canonical: 54).

An alternative removal family exists for the one forced removal branch of
the employer-basis reconstruction (C04-2025, "Members Choosing to
Leave"-era C04 = Care for Older Adults-independent 2025 ID; see
spec/measures_2025.json): 8 removals of real contracts with C04 score 55
instead of the canonical 54, globally safe, full 37/37 battery. This driver
reruns membership_stacker's 2025 stage with the C04/ALL removal branch pinned
to value 55 and writes runs/membership_delta_sets_altb.json (2025 only; the
2024/2026 delta sets are untouched by the C04-2025 branch and remain those of
runs/membership_delta_sets.json).

Everything else is the canonical machinery, imported unmodified: phase-1
add-only search on all 2025 misses, greedy globally-safe removal-contract
selection, post-removal re-solve, family coupling, slot assembly, full
battery. Register: exact; no float in any accept/reject decision.

Run: cd src && python3 membership_stacker_altb.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (reconstructing
    the score list; the alternative reconstruction with one forced
    removal carried out differently).
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import sys
from fractions import Fraction as F

from membership_stacker import (BaseCounts, K_CAP, REM_CAP, RUNS, WORKERS,
                                _search_worker, assemble_year, load_year,
                                make_job, solve_adds_k, target_key)
from solve_all_misses import HYP, SCORE_VINTAGE

ALT_TARGET = ("C04", "ALL")
ALT_VALUE = "55"
ALT_COUNT = 8
CANONICAL_VALUE = "54"


def main():
    year = "2025"
    targets, meta = load_year(year)
    misses = [t for t in targets if t["status"] == "miss"]
    print(f"SY{year}: {sum(1 for t in targets if t['status'] == 'exact')} exact, "
          f"{len(misses)} misses")

    jobs = [make_job(t, k_cap=K_CAP, rem_limit=0) for t in misses]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(WORKERS, len(jobs))) as pool:
        results = pool.map(_search_worker, jobs)
    search = {(r["mid"], r["org"]): r for r in results}

    c04 = search[ALT_TARGET]
    if c04["res"]["add_only"] is not None:
        raise SystemExit("C04/ALL unexpectedly has an add-only solution; "
                         "the forced removal branch would not engage — abort")

    t04 = next(t for t in targets if target_key(t) == ALT_TARGET)
    av = F(ALT_VALUE)
    n_at = sum(1 for _, v in t04["pairs"] if v == av)
    if n_at < ALT_COUNT:
        raise SystemExit(f"only {n_at} contracts at C04={ALT_VALUE}; "
                         f"need {ALT_COUNT} — abort")
    reduced = sorted(v for _, v in t04["pairs"])
    for _ in range(ALT_COUNT):
        reduced.remove(av)
    bc = BaseCounts(reduced)
    k55, ws55 = None, None
    for k in range(K_CAP + 1):
        ws = solve_adds_k(bc, k, t04["pub_lo"], t04["pub_hi"], t04["cap_lo"],
                          t04["cap_hi"], t04["reg"], t04["score_hi"],
                          max_witnesses=3)
        if ws:
            k55, ws55 = k, ws
            break
    if k55 is None:
        raise SystemExit(f"no add-only solution at k<={K_CAP} on the "
                         f"{ALT_COUNT}x{ALT_VALUE}-reduced C04 base — abort")
    print(f"ALT-B option: remove {ALT_COUNT} x {ALT_VALUE}, then k={k55} adds "
          f"e.g. {[str(v) for v in ws55[0]]}")
    c04["res"]["with_rems"] = [{
        "rems": [ALT_VALUE] * ALT_COUNT, "k": k55,
        "witnesses": [[str(v) for v in w] for w in ws55]}]

    print(f"\n=== SY{year} ALT-B assembly ===")
    res = assemble_year(year, targets, meta, search, print)

    # hard verification — errors are raised, never swallowed
    errs = []
    if res["unsolved"]:
        errs.append(f"unsolved targets: {res['unsolved']}")
    # SY2025 evaluable is 40 (no n<30 size filter; three stand-alone Part D
    # targets included).
    if not (res["exact_after"] == res["evaluable"] == 40):
        errs.append(f"battery {res['exact_after']}/{res['evaluable']} "
                    f"(expected 40/40)")
    rn = res["removal_notes"]
    if not (len(rn) == 1 and rn[0]["target"] == "C04/ALL"
            and rn[0]["values"] == [ALT_VALUE] * ALT_COUNT):
        errs.append(f"removal notes not the ALT-B branch: {rn}")
    if len(res["removals"]) != ALT_COUNT:
        errs.append(f"{len(res['removals'])} removals (expected {ALT_COUNT})")
    canon = json.load(open(os.path.join(RUNS, "membership_delta_sets.json")))
    canon_rem = {r["contract_id"] for r in canon["years"][year]["removals"]}
    alt_rem = {r["contract_id"] for r in res["removals"]}
    regressions = [(r["mid"], r["org"]) for r in res["battery"]
                   if r.get("before_exact") and not r.get("after_exact")]
    if regressions:
        errs.append(f"regressions: {regressions}")
    if errs:
        raise SystemExit("ALT-B verification FAILED: " + "; ".join(errs))

    out = {
        "register": ("ALT-B delta set: the one forced removal branch of the "
                     "employer-basis reconstruction (C04-2025) pinned to removal value "
                     f"{ALT_VALUE} (canonical {CANONICAL_VALUE}); all other "
                     "machinery canonical (membership_stacker.py, imported "
                     "unmodified); 2025 only — 2024/2026 unchanged from "
                     "runs/membership_delta_sets.json"),
        "hypothesis": HYP, "score_vintage": SCORE_VINTAGE,
        "k_cap": K_CAP, "rem_cap": REM_CAP,
        "alt": {"target": "C04/ALL", "removal_value": ALT_VALUE,
                "n_removals": ALT_COUNT,
                "canonical_removal_value": CANONICAL_VALUE,
                "removed_contracts": sorted(alt_rem),
                "canonical_removed_contracts": sorted(canon_rem),
                "overlap_with_canonical": sorted(alt_rem & canon_rem)},
        "years": {year: res},
    }
    path = os.path.join(RUNS, "membership_delta_sets_altb.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nALT-B battery: {res['exact_after']}/{res['evaluable']} exact; "
          f"removals {sorted(alt_rem)}; adds={len(res['adds'])}; saved {path}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
