# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Star assignment from published cut points, validated against published stars.

For every measure-year-vintage: take published thresholds (parsed cut points
CSV, with inclusive/exclusive flags) + published scores, assign measure stars,
compare to the published Measure Stars file. Exact Fraction comparisons.

What mismatches mean:
  - systematic per-measure -> our boundary-convention or parsing bug;
  - isolated contracts     -> CMS special handling (disaster rules, data
    integrity flags) — exactly the contracts to exclude/flag when
    reconstructing inclusterdat for the replay fidelity check.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.4 (reproduction of
    the later stages: stars from the published cut points against the published
    stars).
Run:  cd src && python3 assign_stars.py                (every published year/vintage pair)
      cd src && python3 assign_stars.py 2026 current
Requires: Python >= 3.10, standard library only.
"""

import csv
import os
import sys
from collections import defaultdict
from fractions import Fraction

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")


def load_cutpoint_rows(year, vintage):
    """Keyed (measure_id, org_class) — Part D thresholds differ MA-PD vs PDP."""
    path = os.path.join(PARSED, f"cutpoints_{year}_{vintage}.csv")
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            oc = r.get("org_type", "").strip() or "ALL"
            out[(r["measure_id"], oc)][int(r["star_level"])] = r
    return out


def load_org_class(year, vintage):
    """contract_id -> 'PDP' or 'MA-PD' from the summary file's org_type."""
    path = os.path.join(PARSED, f"summary_{year}_{vintage}.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ot = r.get("org_type", "")
            out[r["contract_id"]] = "PDP" if "PDP" in ot else "MA-PD"
    return out


def assign_star(score, rows):
    """rows: {star_level: parsed row}. Returns star or None if no row matches."""
    x = Fraction(score)
    for lvl, r in rows.items():
        lo, lo_incl, hi, hi_incl = r["lo"], r["lo_incl"], r["hi"], r["hi_incl"]
        ok = True
        if lo:
            l = Fraction(lo)
            ok &= (x >= l) if lo_incl == "1" else (x > l)
        if hi:
            h = Fraction(hi)
            ok &= (x <= h) if hi_incl == "1" else (x < h)
        if lo == "" and hi == "":
            ok = False
        if ok:
            return lvl
    return None


def validate(year, vintage, skip_measures=()):
    cps = load_cutpoint_rows(year, vintage)
    org = load_org_class(year, vintage)
    scores = {}
    with open(os.path.join(PARSED, f"scores_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["score"].strip():
                scores[(r["contract_id"], r["measure_id"])] = r["score"].strip()
    pub = {}
    with open(os.path.join(PARSED, f"stars_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stars"].strip():
                pub[(r["contract_id"], r["measure_id"])] = int(r["stars"])
    per_measure = defaultdict(lambda: [0, 0, 0, 0])  # ok, mismatch, no_rule, no_pub
    mismatches = []
    for (cid, mid), s in scores.items():
        if mid in skip_measures:
            continue
        key = (mid, "ALL")
        if key not in cps:
            key = (mid, org.get(cid, "MA-PD"))
        if key not in cps:
            continue
        got = assign_star(s, cps[key])
        want = pub.get((cid, mid))
        pm = per_measure[mid]
        if want is None:
            pm[3] += 1
            continue
        if got is None:
            pm[2] += 1
            mismatches.append((cid, mid, s, None, want))
        elif got == want:
            pm[0] += 1
        else:
            pm[1] += 1
            mismatches.append((cid, mid, s, got, want))
    return per_measure, mismatches


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import sys
    combos = [("2026", "current"), ("2025", "current"), ("2025", "original"),
              ("2024", "recalc"), ("2024", "original"), ("2023", "oct2022")]
    if len(sys.argv) > 2:
        combos = [(sys.argv[1], sys.argv[2])]
    for year, vintage in combos:
        try:
            pm, mm = validate(year, vintage)
        except FileNotFoundError as e:
            print(f"{year}/{vintage}: missing file ({e.filename})")
            continue
        tot_ok = sum(v[0] for v in pm.values())
        tot_bad = sum(v[1] for v in pm.values())
        tot_norule = sum(v[2] for v in pm.values())
        print(f"\n=== {year}/{vintage}: OK {tot_ok}, mismatch {tot_bad}, "
              f"no-rule {tot_norule} ===")
        worst = sorted(pm.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:8]
        for mid, (ok, bad, norule, nopub) in worst:
            if bad or norule:
                print(f"  {mid}: ok={ok} mismatch={bad} no-rule={norule}")
        for cid, mid, s, got, want in mm[:12]:
            print(f"    e.g. {cid} {mid} score={s} ours={got} published={want}")
