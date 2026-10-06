# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Stage-1 membership reconciliation: published outlier-bound cutoffs
(Tech Notes K-5/K-6 tables, spec/validation_targets_<year>.json) vs the exact
bounds under membership hypotheses. No random-number dependence: the bounds
depend only on the clustering-input membership, so this isolates the score-list
(inclusterdat) reconstruction from the GROUPS= readout question entirely.

Hypotheses (per SPEC_NOTES (c) + Attachment K narrative):
  all        : every contract with a numeric published score (baseline)
  d60r       : exclude contracts with >= 60% in the MOST RECENT disaster-year
               column (2024: '2022 Disaster %'; 2025: '2023 Disaster %') —
               2024/2025 only (rule removed for 2026)
  d60e       : exclude >= 60% in EITHER published disaster column
  cost       : exclude 1876 Cost org types ('voluntary contract scores' reading)
  cost+d60r, cost+d60e : combinations
Comparison precision: published K-5/K-6 values are display-precision strings;
the test is round_display(exact bound, dd) == published, with near-misses
(|diff| <= 1 display unit) separately. Capping per Tech Notes: percent 0-100
displays capped at [0,100]; rates capped below at 0.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (the documented
    rules alone reproduce the published outlier bounds on 82 of 123 targets).
Run:  cd src && python3 tukey_battery.py 2024 2025 2026
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction

from tukey import tukey_fences
from falsify_groups import round_display

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")
SPEC = os.path.join(BASE, "spec")

# score-file vintages to test per star year (K-tables were computed on the
# ORIGINAL run's data; 2024 recalc changed 50 score cells)
SCORE_VINTAGES = {
    "2024": ["original", "original_mar2024", "recalc"],
    "2025": ["original"],
    "2026": ["current"],
}
DISASTER_YEARS = {"2024": True, "2025": True, "2026": False}


def load_summary_meta(year, vintage_for_summary):
    """contract_id -> (org_type, disaster_recent_pct, disaster_either_max)."""
    path = os.path.join(PARSED, f"summary_{year}_{vintage_for_summary}.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vals = []
            for i in (1, 2):
                v = (r.get(f"disaster_col{i}_raw") or "").strip()
                try:
                    vals.append(int(v))
                except ValueError:
                    vals.append(None)
            # col order in files: older year first, recent year second
            recent = vals[1] if vals[1] is not None else 0
            either = max([v for v in vals if v is not None] or [0])
            out[r["contract_id"]] = (r.get("org_type", ""), recent, either)
    return out


def scores_by_measure(year, vintage):
    out = defaultdict(list)  # mid -> [(cid, score_str)]
    with open(os.path.join(PARSED, f"scores_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["score"].strip():
                out[r["measure_id"]].append((r["contract_id"], r["score"].strip()))
    return out


def apply_hypothesis(pairs, meta, hyp, disaster_active):
    keep = []
    for cid, s in pairs:
        org, recent, either = meta.get(cid, ("", 0, 0))
        if "cost" in hyp and "1876" in org:
            continue
        if "emp" in hyp and "Employer" in org:
            continue
        if disaster_active:
            if "d60r" in hyp and recent >= 60:
                continue
            if "d60e" in hyp and either >= 60:
                continue
        keep.append((cid, s))
    return keep


def run_battery(years=("2024", "2025", "2026"), verbose=True):
    grand = {}
    for year in years:
        vt = json.load(open(os.path.join(SPEC, f"validation_targets_{year}.json")))
        tk = vt["tukey_cutoffs"]
        rows = tk["rows"] if isinstance(tk, dict) else tk
        targets = {}
        for r in rows:
            targets.setdefault(r["measure_id"], {})[r.get("org_type") or "ALL"] = (
                r["lower"], r["upper"])
        disaster_active = DISASTER_YEARS[year]
        hyps = ["all", "cost", "emp", "cost+emp"]
        if disaster_active:
            hyps += ["d60r", "cost+d60r", "cost+d60r+emp", "emp+d60r"]
        for vintage in SCORE_VINTAGES[year]:
            try:
                sc = scores_by_measure(year, vintage)
            except FileNotFoundError:
                continue
            sum_vintage = vintage if os.path.exists(
                os.path.join(PARSED, f"summary_{year}_{vintage}.csv")) else \
                SCORE_VINTAGES[year][-1]
            meta = load_summary_meta(year, sum_vintage)
            tally = Counter()
            denom = Counter()
            miss_detail = defaultdict(list)
            for mid, tmap in targets.items():
                pairs = sc.get(mid)
                if not pairs:
                    continue
                for org_key, (plo, phi) in tmap.items():
                    if org_key == "MA-PD" or org_key == "PDP":
                        sub = [(c, s) for c, s in pairs
                               if ("PDP" in meta.get(c, ("",))[0]) == (org_key == "PDP")]
                    else:
                        sub = pairs
                    if len(sub) < 30:
                        continue
                    for hyp in hyps:
                        kept = apply_hypothesis(sub, meta, hyp, disaster_active)
                        if len(kept) < 30:
                            continue
                        # caps: percent displays capped [0,100]; others lo 0
                        dd = max(len(v.split(".")[1]) if v and "." in v else 0
                                 for v in (plo, phi) if v)
                        is_pct = dd == 0 and phi is not None and Fraction(phi) <= 100
                        cap_hi = "100" if is_pct else None
                        lo, hi = tukey_fences([s for _, s in kept],
                                              cap_lo="0", cap_hi=cap_hi)
                        for ours, pub, side in ((lo, plo, "lo"), (hi, phi, "hi")):
                            if pub is None:
                                continue
                            denom[hyp] += 1
                            od = round_display(ours, dd)
                            if Fraction(od) == Fraction(pub):
                                tally[hyp] += 1
                            else:
                                gap = abs(Fraction(od) - Fraction(pub))
                                miss_detail[hyp].append(
                                    (mid, org_key, side, od, pub, float(gap)))
            key = (year, vintage)
            grand[key] = (tally, denom, miss_detail)
            if verbose:
                print(f"\n=== SY{year} scores={vintage} ===")
                for hyp in hyps:
                    if denom[hyp]:
                        near = sum(1 for m in miss_detail[hyp] if m[5] <= 1.0)
                        print(f"  {hyp:>10}: {tally[hyp]}/{denom[hyp]} exact "
                              f"(+{near} within 1 display unit)")
    return grand


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(os.path.exists(os.path.join(SPEC, f"validation_targets_{y}.json"))
            for y in (sys.argv[1:] or ["2024", "2025", "2026"])):
    print("tukey_battery: no spec/validation_targets_<year>.json for one of the requested star years %s "
          "(included: 2024, 2025, 2026)" % (sys.argv[1:] or ["2024", "2025", "2026"]),
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    years = sys.argv[1:] or ["2024", "2025", "2026"]
    g = run_battery(tuple(years))
    # show worst misses for the best hypothesis of each year/vintage
    for (year, vintage), (tally, denom, md) in g.items():
        if not denom:
            continue
        best = max(tally, key=lambda h: tally[h] / max(1, denom[h]))
        print(f"\nSY{year}/{vintage} best={best} misses:")
        for m in sorted(md[best], key=lambda m: -m[5])[:10]:
            print("   ", m)
