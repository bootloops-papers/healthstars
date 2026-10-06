#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Safety-cap interaction check for the 2026 exact-optimum stars.

The 2026 pipeline (R era, methodology v5.1 Step 9) applies a post-assignment
safety cap — CMS code: work/'R pack'/2 - Second Stage_Weighted Average and
Categorize Star_2026Apr.R, lines 254-280:
    Safe_q  = ntile(Std_Outcomes_Safety_score, 4)          # quartiles over Star_
    star_p3 = ifelse(Outcomes_Safety_cnt >= 3 & Safe_q == 1 & star > 4, 4, star)
i.e. hospitals in the lowest Safety-of-Care quartile with >=3 safety measures
are capped at 4 stars. Documented: Star_Rtngs_CompMthdlgy_v5.1.pdf (Step 9;
finalized CY2026 OPPS/ASC Final Rule CMS-1834-FC).

This script replays the cap on the exact-optimum assignment
(runs/exact_census_2026.json, binary-double register) using R's own Safe_q
values (dumped from the post-program-2 global `qq` by dump_fullprec_2026.R;
ntile is not reimplemented), and reports how many of the corrections survive
the cap unchanged. Writes runs/safety_cap_check_2026.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.2 (the safety cap
    returns five of the 213 hospitals to their published star, leaving 208).
Run:  cd hospital/work && python3 safety_cap_check_2026.py
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import json

BASE = _REC + "/hospital"


def cap(star, safe_q, safety_cnt):
    """CMS's rule: >=3 safety measures, lowest safety quartile, star>4 -> 4."""
    if star is None:
        return None
    if safety_cnt >= 3 and safe_q == 1 and star > 4:
        return 4
    return star


def main():
    dump = {r["PROVIDER_ID"]: r for r in csv.DictReader(
        open(f"{BASE}/work/R_output/fullprec_2026apr.csv"))}

    # --- 1) validate the Python cap replay against R's own star_p3, all rows ---
    n_checked = n_capped = n_mismatch = 0
    for r in dump.values():
        if r["star_precap"] in ("NA", ""):
            continue
        star = int(r["star_precap"])
        safe_q = int(r["Safe_q"]) if r["Safe_q"] not in ("NA", "") else None
        cnt = int(r["Outcomes_Safety_cnt"]) if r["Outcomes_Safety_cnt"] not in ("NA", "") else 0
        got = cap(star, safe_q if safe_q is not None else 0, cnt)
        want = int(r["star_postcap"])
        if got != want:
            n_mismatch += 1
        assert got == want, (r["PROVIDER_ID"], star, safe_q, cnt, got, want)
        n_checked += 1
        if got != star:
            n_capped += 1
    print(f"cap replay validated on {n_checked} starred rows; "
          f"CMS-capped 5->4: {n_capped}", flush=True)

    census = json.load(open(f"{BASE}/runs/exact_census_2026.json"))
    differs = census["star_differs_detail"]

    # --- 2) cap the DP assignment for each of the DP-corrected hospitals ---
    survive = []
    modified = []
    for d in differs:
        r = dump[d["provider_id"]]
        safe_q = int(r["Safe_q"]) if r["Safe_q"] not in ("NA", "") else 0
        cnt = int(r["Outcomes_Safety_cnt"]) if r["Outcomes_Safety_cnt"] not in ("NA", "") else 0
        shipped_pre = d["shipped"]
        shipped_post = int(r["star_postcap"])
        dp_pre = d["dp_optimum"]
        dp_post = cap(dp_pre, safe_q, cnt)
        row = {"provider_id": d["provider_id"], "peer": d["peer"],
               "shipped_precap": shipped_pre, "shipped_postcap": shipped_post,
               "dp_precap": dp_pre, "dp_postcap": dp_post,
               "safe_q": safe_q, "safety_cnt": cnt,
               "cap_touches_shipped": shipped_post != shipped_pre,
               "cap_touches_dp": dp_post != dp_pre}
        if row["cap_touches_shipped"] or row["cap_touches_dp"]:
            modified.append(row)
        else:
            survive.append(row)

    # --- 3) post-cap differ census over ALL rated (does the cap create/erase differs?) ---
    dp_by_pid = {d["provider_id"]: d["dp_optimum"] for d in differs}
    post_differs = []
    for pid, r in dump.items():
        if r["report_indicator"] != "1" or r["star_precap"] in ("NA", ""):
            continue
        shipped_pre = int(r["star_precap"])
        shipped_post = int(r["star_postcap"])
        dp_pre = dp_by_pid.get(pid, shipped_pre)  # non-differs: DP == CMS pre-cap
        safe_q = int(r["Safe_q"]) if r["Safe_q"] not in ("NA", "") else 0
        cnt = int(r["Outcomes_Safety_cnt"]) if r["Outcomes_Safety_cnt"] not in ("NA", "") else 0
        dp_post = cap(dp_pre, safe_q, cnt)
        if dp_post != shipped_post:
            post_differs.append(pid)
    new_differs = sorted(set(post_differs) - set(dp_by_pid))

    # --- 4) the 15 hospitals CMS capped vs the DP differ set ---
    shipped_capped = sorted(pid for pid, r in dump.items()
                            if r["star_precap"] not in ("NA", "")
                            and r["star_postcap"] not in ("NA", "")
                            and int(r["star_precap"]) != int(r["star_postcap"]))
    capped_in_differ_set = sorted(set(shipped_capped) & set(dp_by_pid))

    dp5_capped = [m for m in modified if m["cap_touches_dp"]]
    out = {
        "run": "safety_cap_check_2026",
        "cap_rule": {
            "source_code": "work/R pack/2 - Second Stage_Weighted Average and "
                           "Categorize Star_2026Apr.R, lines 254-280 "
                           "(Safe_q = ntile(Std_Outcomes_Safety_score, 4); "
                           "star_p3 = ifelse(Outcomes_Safety_cnt >= 3 & "
                           "Safe_q == 1 & star > 4, 4, star))",
            "methodology": "Star_Rtngs_CompMthdlgy_v5.1.pdf Step 9: 4-star cap, "
                           "lowest Safety-of-Care quartile with >=3 safety "
                           "measures; finalized CY2026 OPPS/ASC Final Rule "
                           "(CMS-1834-FC); CY2027 anticipated replacement: "
                           "blanket 1-star reduction (same trigger)",
            "safe_q_provenance": "R's own ntile output, dumped from the "
                                 "post-program-2 global qq by "
                                 "work/dump_fullprec_2026.R (no reimplementation)",
        },
        "cap_replay_validation": {
            "starred_rows_checked": n_checked,
            "python_replay_mismatches_vs_R_star_p3": n_mismatch,
            "shipped_capped_5_to_4": n_capped,
        },
        "dp_corrected_hospitals": len(differs),
        "survive_cap_unchanged": len(survive),
        "modified_by_cap": len(modified),
        "modified_detail": modified,
        "dp5_capped_detail": dp5_capped,
        "postcap_differ_census": {
            "dp_postcap_vs_shipped_postcap_differs": len(post_differs),
            "new_differs_created_by_cap": new_differs,
            "differs_erased_by_cap": sorted(set(dp_by_pid) - set(post_differs)),
        },
        "shipped_capped_hospitals_in_dp_differ_set": capped_in_differ_set,
    }
    dest = f"{BASE}/runs/safety_cap_check_2026.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"of {len(differs)} DP-corrected: survive={len(survive)} "
          f"modified={len(modified)}; post-cap differs="
          f"{len(post_differs)}; new-differs={len(new_differs)}", flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
