# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""SY2023 shared input builder for the 2023 extension of the exact chain.

2023 differs from 2024-2026 (see spec/SPEC_NOTES_2023.md):
  - NO Tukey outlier deletion (tukey=False everywhere);
  - documented-rules-only membership hypothesis 'cost+d60r' with the
    2023-only HEDIS-HOS waiver: the >=60% 2021-disaster exclusion is NOT
    applied to C04 (Monitoring Physical Activity), C13 (Reducing the Risk of
    Falling), C14 (Improving Bladder Control);
  - guardrail base = published 2022 finals (trend-doc rows, stored in
    spec/validation_targets_2023.json guardrail_basis_thresholds), with the
    falsified-subset caveat from the spec; exempt: improvement C25/D04, new
    C12 + D07;
  - restricted-range measures (complaints C23/D02): the published cap
    is not reproducible from public data (spec (a)3-4) -> guardrail register
    for those splits is 'unpinned' and rows are flagged.

MEMBERSHIP REGISTER: hypothesis-grade ONLY. 2023 has no fence instrument, so
no split can ever be labeled fence-exact. All downstream rows must carry
membership='hypothesis:cost+d60r(hoswaiver)'.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1, Table 2 note a
    (star year 2023, different documented rules).
Run:  cd src && python3 inputs_2023.py   (prints the 2023 clusterable splits; imported by the
      *_2023.py scripts)
Requires: Python >= 3.10; numpy, scipy.
"""

import csv
import json
import os
import sys
from collections import defaultdict

from tukey_battery import scores_by_measure, load_summary_meta
from falsify_groups import published_boundaries

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")
SPEC = os.path.join(BASE, "spec")

YEAR = "2023"
VINTAGE = "oct2022"
HYP = "cost+d60r"                      # documented-rules-only (docsonly basis)
HYP_LABEL = "hypothesis:cost+d60r(hoswaiver)"
HOS_WAIVER_MIDS = {"C04", "C13", "C14"}    # 2023-only 60%-rule waiver
IMPROVEMENT_2023 = {"C25", "D04"}
NEW_EXEMPT_2023 = {"C12", "D07"}           # guardrail-exempt (new-measure)
RESTRICTED_2023 = {"C23", "D02"}           # complaints: restricted-range
CAHPS_2023 = {"C03", "C17", "C18", "C19", "C20", "C21", "C22", "D05", "D06"}


def apply_hypothesis_2023(mid, pairs, meta, hyp=HYP):
    """2023 hypothesis filter: 'cost' excludes 1876 Cost; 'd60r' excludes
    >=60% in the 2021 disaster column EXCEPT for the 3 HEDIS-HOS waiver
    measures; 'emp' excludes Employer/Union-only org types."""
    keep = []
    waived = mid in HOS_WAIVER_MIDS and "nowaiver" not in hyp
    for cid, s in pairs:
        org, recent, either = meta.get(cid, ("", 0, 0))
        if "cost" in hyp and "1876" in org:
            continue
        if "emp" in hyp and "Employer" in org:
            continue
        if "d60r" in hyp and not waived and recent >= 60:
            continue
        keep.append((cid, s))
    return keep


def pub_rows_2023(mid, org):
    """published final cut-point rows {star: (lo, lo_incl, hi, hi_incl)}."""
    out = {}
    path = os.path.join(PARSED, f"cutpoints_{YEAR}_{VINTAGE}.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["measure_id"] != mid:
                continue
            row_oc = (r.get("org_type") or "").strip() or "ALL"
            if row_oc == org:
                out[int(r["star_level"])] = (r["lo"], r["lo_incl"],
                                             r["hi"], r["hi_incl"])
    return out


def j_targets():
    """(mid, org) -> (4 ascending pre-guardrail boundary strings, higher)."""
    vt = json.load(open(os.path.join(SPEC, "validation_targets_2023.json")))
    rows = defaultdict(dict)
    for r in vt["pre_guardrail_thresholds"]["rows"]:
        org = r["org_type"] or "ALL"
        rows[(r["measure_id"], org)][r["star_level"]] = r
    out = {}
    for key, by_star in rows.items():
        if len(by_star) < 5:
            continue
        r1 = by_star[1]
        higher = r1["lo"] is None and r1["hi"] is not None
        if higher:
            vals = [by_star[s]["lo"] for s in (2, 3, 4, 5)]
        else:
            vals = [by_star[s]["hi"] for s in (5, 4, 3, 2)]
        if any(v is None for v in vals):
            continue
        out[key] = (vals, higher)
    return out


def guardrail_base_2023():
    """(mid, org) -> 4 ascending published-2022 boundary strings (or None)."""
    vt = json.load(open(os.path.join(SPEC, "validation_targets_2023.json")))
    rows = defaultdict(dict)
    for r in vt["guardrail_basis_thresholds"]["rows"]:
        org = r["org_type"] or "ALL"
        rows[(r["measure_id"], org)][r["star_level"]] = r
    out = {}
    for key, by_star in rows.items():
        if len(by_star) < 5:
            continue
        r1 = by_star[1]
        higher = r1["lo"] is None and r1["hi"] is not None
        if higher:
            vals = [by_star[s]["lo"] for s in (2, 3, 4, 5)]
        else:
            vals = [by_star[s]["hi"] for s in (5, 4, 3, 2)]
        if any(v is None for v in vals):
            continue
        out[key] = vals
    return out


def measure_names_2023():
    names = {}
    with open(os.path.join(PARSED, f"cutpoints_{YEAR}_{VINTAGE}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            names.setdefault(r["measure_id"], r["measure_name"].strip())
    return names


def build_splits(hyp=HYP, min_n=30):
    """All clusterable 2023 measure-splits with hypothesis-filtered input.

    Returns list of dicts: {mid, org, name, higher, dd, kept, pool, target
    (published final boundaries), pre (J boundaries or None), guardrail:
    {'status': exempt/applied/unpinned, 'base': [...] or None, 'cap': '5' or
    None}}. Improvement + CAHPS measures excluded (no clusterable public
    scores / different method).
    """
    sc = scores_by_measure(YEAR, VINTAGE)
    meta = load_summary_meta(YEAR, VINTAGE)
    names = measure_names_2023()
    pre_map = j_targets()
    base_map = guardrail_base_2023()
    splits = []
    for mid in sorted(names):
        if mid in IMPROVEMENT_2023 or mid in CAHPS_2023:
            continue
        pairs = sc.get(mid)
        if not pairs:
            continue
        pairs = sorted(pairs)                      # dataset order: cid asc
        orgs = ["MA-PD", "PDP"] if mid.startswith("D") else ["ALL"]
        for org in orgs:
            if org == "ALL":
                sub = pairs
            else:
                sub = [(c, s) for c, s in pairs
                       if ("PDP" in meta.get(c, ("",))[0]) == (org == "PDP")]
            if len(sub) < min_n:
                continue
            pub = pub_rows_2023(mid, org)
            if len(pub) < 5:
                continue
            target, higher = published_boundaries(pub)
            if target is None:
                continue
            dd = max(len(v.split(".")[1]) if "." in v else 0 for v in target)
            kept = apply_hypothesis_2023(mid, sub, meta, hyp)
            if len(kept) < min_n:
                continue
            if mid in NEW_EXEMPT_2023:
                gr = {"status": "exempt-new", "base": None, "cap": None}
            elif mid in RESTRICTED_2023:
                gr = {"status": "unpinned-restricted",
                      "base": base_map.get((mid, org)), "cap": None}
            else:
                base = base_map.get((mid, org))
                gr = ({"status": "applied", "base": base, "cap": "5"}
                      if base else
                      {"status": "no-base->exempt", "base": None, "cap": None})
            splits.append({
                "mid": mid, "org": org, "name": names[mid], "higher": higher,
                "dd": dd, "kept": kept, "pool": sub, "target": target,
                "pre": pre_map.get((mid, org), (None, None))[0],
                "guardrail": gr,
                "membership": HYP_LABEL,
            })
    return splits


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    sp = build_splits()
    print(f"2023 clusterable splits: {len(sp)}")
    from collections import Counter
    print(Counter(s["guardrail"]["status"] for s in sp))
    for s in sp[:8]:
        print(f"  {s['mid']}/{s['org']:>5} n_kept={len(s['kept'])} "
              f"n_pool={len(s['pool'])} dd={s['dd']} higher={s['higher']} "
              f"gr={s['guardrail']['status']}")
