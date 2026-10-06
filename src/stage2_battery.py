# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Stage 2: GROUPS= candidate screen on the outlier-bound-exact measure subset.

Instrument: Tech Notes K-3/K-4 "Mean Resampling Estimated Thresholds" (published
pre-guardrail, spec/validation_targets_<year>.json) — the only published numbers
that see the group assignment but not the guardrails. Restricting to measure-org
targets whose K-5/K-6 outlier bounds are reproduced exactly under the best membership
hypothesis (cost+d60r+emp) removes the membership confound: on this subset any
remaining mismatch is (candidate, ordering) — the random-number question — plus Ward-arm
double semantics and tie residuals (flagged separately by near-tie counts).

Precision: published K-3/K-4 values are display-precision strings; comparison via
round_display at the per-measure precision inferred from the strings themselves.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (candidate reconstructions screened on the computations whose outlier
    bounds are reproduced exactly).
Run:  cd src && python3 stage2_battery.py        (all targets)
      cd src && python3 stage2_battery.py 5      (pilot: the first five targets)
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction

from falsify_groups import round_display
from pipeline import measure_cutpoints
from tukey import tukey_fences
from tukey_battery import scores_by_measure, load_summary_meta, apply_hypothesis

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(BASE, "spec")
HYP = "cost+d60r+emp"
CANDIDATES = ["seq_quota_last", "seq_quota_first", "sort_blocks_first"]
ORDERINGS = ["cid_asc", "cid_desc"]
SCORE_VINTAGE = {"2024": "original", "2025": "original", "2026": "current"}
IMPROVEMENT = {"2024": {"C27", "D04"}, "2025": {"C27", "D04"}, "2026": {"C30", "D04"}}


def load_targets(year):
    vt = json.load(open(os.path.join(SPEC, f"validation_targets_{year}.json")))
    fences = {}
    for r in vt["tukey_cutoffs"]["rows"]:
        fences[(r["measure_id"], r.get("org_type") or "ALL")] = (r["lower"], r["upper"])
    pre = defaultdict(dict)
    for r in vt["pre_guardrail_thresholds"]["rows"]:
        pre[(r["measure_id"], r.get("org_type") or "ALL")][r["star_level"]] = r
    return fences, pre


def boundaries_from_rows(rows):
    """4 ascending boundary strings + direction from K-table star rows."""
    r1 = rows.get(1)
    if r1 is None or len(rows) < 5:
        return None, None
    higher = r1.get("lo") is None and r1.get("hi") is not None
    if higher:
        vals = [rows[s].get("lo") for s in (2, 3, 4, 5)]
    else:
        vals = [rows[s].get("hi") for s in (5, 4, 3, 2)]
    if any(v is None for v in vals):
        return None, None
    return vals, higher


def build_worklist(years):
    work = []
    for year in years:
        vintage = SCORE_VINTAGE[year]
        fences, pre = load_targets(year)
        sc = scores_by_measure(year, vintage)
        meta = load_summary_meta(year, vintage)
        disaster = year in ("2024", "2025")
        for key, frows in fences.items():
            mid, org_key = key
            if mid in IMPROVEMENT[year]:
                continue
            pairs = sc.get(mid)
            if not pairs or key not in pre:
                continue
            if org_key in ("MA-PD", "PDP"):
                sub = [(c, s) for c, s in pairs
                       if ("PDP" in meta.get(c, ("",))[0]) == (org_key == "PDP")]
            else:
                sub = pairs
            kept = apply_hypothesis(sub, meta, HYP, disaster)
            if len(kept) < 30:
                continue
            plo, phi = frows
            dd_f = max(len(v.split(".")[1]) if v and "." in v else 0
                       for v in (plo, phi) if v)
            is_pct = dd_f == 0 and phi is not None and Fraction(phi) <= 100
            cap_hi = "100" if is_pct else None
            lo, hi = tukey_fences([s for _, s in kept], cap_lo="0", cap_hi=cap_hi)
            ok_lo = plo is None or lo == Fraction(plo)
            ok_hi = phi is None or hi == Fraction(phi)
            if not (ok_lo and ok_hi):
                continue  # membership not yet reconciled for this target
            target, higher = boundaries_from_rows(pre[key])
            if target is None:
                continue
            work.append({"year": year, "vintage": vintage, "mid": mid,
                         "org": org_key, "kept": kept, "target": target,
                         "higher": higher, "cap_hi": cap_hi})
    return work


def round_display_even(x, d):
    """banker's rounding at d decimals; returns string."""
    q = Fraction(10) ** d
    y = x * q
    n, den = y.numerator, y.denominator
    fl, rem = divmod(n, den)  # floor for negatives too
    if 2 * rem > den or (2 * rem == den and fl % 2 != 0):
        fl += 1
    s = str(abs(fl)).rjust(d + 1, "0")
    sign = "-" if fl < 0 else ""
    return sign + (s[:-d] if d else s) + (("." + s[-d:]) if d else "")


def round_display_trunc(x, d):
    q = Fraction(10) ** d
    y = x * q
    n, den = y.numerator, y.denominator
    t = n // den if n >= 0 else -((-n) // den)
    s = str(abs(t)).rjust(d + 1, "0")
    sign = "-" if t < 0 else ""
    return sign + (s[:-d] if d else s) + (("." + s[-d:]) if d else "")


ROUNDERS = {"half_away": round_display, "half_even": round_display_even,
            "trunc": round_display_trunc}


def _run_one(args):
    w, cand, ordering = args
    kept = w["kept"] if ordering == "cid_asc" else w["kept"][::-1]
    scores = [s for _, s in kept]
    dd = max(len(v.split(".")[1]) if "." in v else 0 for v in w["target"])
    r = measure_cutpoints(scores, higher_is_better=w["higher"], tukey=True,
                          cap_lo="0", cap_hi=w["cap_hi"],
                          groups_candidate=cand, arm="ward_tie", scan_ties=False)
    if "error" in r or len(r["mean_cutpoints"]) != 4:
        return (w["year"], w["mid"], w["org"], cand, ordering, None, None)
    matches = {}
    ours_by = {}
    for rname, rfn in ROUNDERS.items():
        ours = [rfn(c, dd) for c in r["mean_cutpoints"]]
        matches[rname] = sum(1 for a, b in zip(ours, w["target"])
                             if Fraction(a) == Fraction(b))
        ours_by[rname] = ours
    raw = [str(c) for c in r["mean_cutpoints"]]
    return (w["year"], w["mid"], w["org"], cand, ordering, matches,
            {"raw": raw, "ours": ours_by["half_away"]})


def main(years=("2024", "2025", "2026"), workers=None, pilot=0):
    if workers is None:
        workers = os.cpu_count() or 1
    work = build_worklist(years)
    print(f"outlier-bound-exact targets: {len(work)} "
          f"({Counter(w['year'] for w in work)})")
    if pilot:
        work = work[:pilot]
    jobs = [(w, c, o) for w in work for c in CANDIDATES for o in ORDERINGS]
    print(f"jobs: {len(jobs)} ({len(work)} targets x {len(CANDIDATES)} cands x "
          f"{len(ORDERINGS)} orderings)")
    with mp.Pool(workers) as pool:
        results = pool.map(_run_one, jobs)
    tally = defaultdict(Counter)
    denom = defaultdict(Counter)
    perfect = defaultdict(Counter)
    detail = defaultdict(dict)
    for year, mid, org, cand, ordering, matches, extra in results:
        if matches is None:
            continue
        key = (cand, ordering)
        for rname, m in matches.items():
            tally[rname][key] += m
            denom[rname][key] += 4
            if m == 4:
                perfect[rname][key] += 1
        detail[(year, mid, org)][key] = (matches, extra)
    for rname in ROUNDERS:
        print(f"\n=== TALLY [{rname}] ===")
        for key in sorted(tally[rname], key=lambda k: -tally[rname][k]):
            print(f"  {key[0]:>18} {key[1]:>8}: "
                  f"{tally[rname][key]}/{denom[rname][key]} boundaries | "
                  f"{perfect[rname][key]} targets 4/4")
    out = os.path.join(BASE, "runs", "stage2_battery.json")
    json.dump({rname: {f"{k[0]}|{k[1]}": [tally[rname][k], denom[rname][k],
                                          perfect[rname][k]]
                       for k in tally[rname]} for rname in ROUNDERS} |
              {"detail": {f"{y}|{m}|{o}": {f"{k[0]}|{k[1]}":
                                           [v[0], v[1]] for k, v in res.items()}
                          for (y, m, o), res in detail.items()}},
              open(out, "w"), indent=1, default=str)
    print("saved", out)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    pilot = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(pilot=pilot)
