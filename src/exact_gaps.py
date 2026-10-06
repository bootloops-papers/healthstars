# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Optimality-gap table: the exact optimum against the published-method
partition on the full score list.

For every clusterable measure-year(-org): outlier-trimmed clustering input
(best membership hypothesis), published-method Ward partition on the full score
list (documented SAS tie rule, IEEE double), exact-rational SSQ of that
partition, exact DP optimum with uniqueness/tie enumeration, exact gap, and the
star-assignment delta census between the two partitions' cut points on the
full score list.

No RNG anywhere: this table does not use the resampling groups. Fields per row:
  ward_ssq, dp_ssq, gap : exact (Fraction strings)
  off_optimum           : exact boolean (gap > 0)
  dp_n_optima           : exact (count of SSQ-tied optimal partitions)
  membership            : 'fence-exact' (the published outlier bounds
                          reproduced exactly) or 'pending' (membership
                          delta unresolved; the row is exact given the
                          stated input)
Each row states that the published partition of the stated input does not
minimize the within-cluster sum of squares and gives the gap to the exact
minimum.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (optimality gaps
    at the clustering step on the employer-rule basis: 105 of 123).
Run:  cd src && python3 exact_gaps.py
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import multiprocessing as mp
import os
import sys
from fractions import Fraction as F

from dp_kmeans import dp_optimal_partition, dp_optimal_partitions_all, partition_ssq
from tukey import tukey_fences
from tukey_battery import scores_by_measure, load_summary_meta, apply_hypothesis
from ward_cluster_tie import ward_sas_cut
from ward_cluster import cutpoints_from_clusters
from stage2_battery import load_targets, boundaries_from_rows, IMPROVEMENT

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYP = "cost+d60r+emp"
SCORE_VINTAGE = {"2024": "original", "2025": "original", "2026": "current"}
CAHPS_NAMES = {"Annual Flu Vaccine", "Getting Needed Care",
               "Getting Appointments and Care Quickly", "Customer Service",
               "Rating of Health Care Quality", "Rating of Health Plan",
               "Care Coordination", "Getting Needed Prescription Drugs",
               "Rating of Drug Plan"}


def measure_names(year, vintage):
    import csv
    names = {}
    with open(os.path.join(BASE, "data", "parsed",
                           f"scores_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            names.setdefault(r["measure_id"], r["measure_name"].strip())
    return names


def frac_to_decimal_str(v, max_dd=6):
    """Exact Fraction -> decimal string (raises if not a decimal fraction)."""
    v = F(v)
    for d in range(max_dd + 1):
        scaled = v * 10 ** d
        if scaled.denominator == 1:
            n = scaled.numerator
            if d == 0:
                return str(n)
            sign = "-" if n < 0 else ""
            s = str(abs(n)).rjust(d + 1, "0")
            return f"{sign}{s[:-d]}.{s[-d:]}"
    raise ValueError(f"{v} not a decimal fraction at <= {max_dd} dp")


def load_delta_sets():
    """Reconciled-membership deltas (stage-1 result), or None."""
    path = os.path.join(BASE, "runs", "membership_delta_sets.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def build_jobs(reconciled=True):
    ds = load_delta_sets() if reconciled else None
    jobs = []
    for year in ("2024", "2025", "2026"):
        vintage = SCORE_VINTAGE[year]
        fences, pre = load_targets(year)
        sc = scores_by_measure(year, vintage)
        meta = load_summary_meta(year, vintage)
        names = measure_names(year, vintage)
        disaster = year in ("2024", "2025")
        d = ds["years"][year] if ds else None
        rems = {r["contract_id"] for r in d["removals"]} if d else set()
        for key in fences:
            mid, org_key = key
            name = names.get(mid, "")
            if mid in IMPROVEMENT[year] or name in CAHPS_NAMES:
                continue
            pairs = sc.get(mid)
            if not pairs:
                continue
            if org_key in ("MA-PD", "PDP"):
                sub = [(c, s) for c, s in pairs
                       if ("PDP" in meta.get(c, ("",))[0]) == (org_key == "PDP")]
            else:
                sub = pairs
            kept = apply_hypothesis(sub, meta, HYP, disaster)
            if d:
                kept = [(c, s) for c, s in kept if c not in rems]
                mkey = f"{mid}/{org_key}"
                for a in d["adds"]:
                    if mkey in a["measures"]:
                        kept = kept + [(a["id"],
                                        frac_to_decimal_str(a["measures"][mkey]))]
            # k-viability guard (n >= 5), not an n<30 size filter
            if len(kept) < 5:
                continue
            plo, phi = fences[key]
            dd = max(len(v.split(".")[1]) if v and "." in v else 0
                     for v in (plo, phi) if v)
            is_pct = dd == 0 and phi is not None and F(phi) <= 100
            cap_hi = "100" if is_pct else None
            lo, hi = tukey_fences([s for _, s in kept], cap_lo="0", cap_hi=cap_hi)
            fence_ok = ((plo is None or lo == F(plo))
                        and (phi is None or hi == F(phi)))
            # direction from pre-guardrail rows when available
            higher = True
            if key in pre:
                t, h = boundaries_from_rows(pre[key])
                if h is not None:
                    higher = h
            jobs.append({"year": year, "vintage": vintage, "mid": mid,
                         "org": org_key, "name": name, "higher": higher,
                         "kept": kept, "cap_hi": cap_hi,
                         "membership": "fence-exact" if fence_ok else "pending",
                         "fence_lo": str(lo), "fence_hi": str(hi)})
    return jobs


def _run(j):
    from tukey import tukey_trim
    scores = [s for _, s in j["kept"]]
    mask, _, _ = tukey_trim(scores, cap_lo="0", cap_hi=j["cap_hi"])
    trimmed = [s for s, m in zip(scores, mask) if m]
    vals = [float(s) for s in trimmed]
    k = min(5, len(set(vals)))
    ward_cl = ward_sas_cut(vals, k)
    # exact SSQ of ward partition: labels parallel to trimmed
    labels = [0] * len(vals)
    for ci, c in enumerate(ward_cl):
        for i in c:
            labels[i] = ci
    ward_ssq = partition_ssq(trimmed, labels)
    dp = dp_optimal_partition(trimmed, k)
    gap = ward_ssq - dp["ssq"]
    try:
        _, opts = dp_optimal_partitions_all(trimmed, k, cap=32)
        n_opt = len(opts)
    except RuntimeError:
        n_opt = ">32"
    ward_cps = cutpoints_from_clusters(ward_cl, vals, j["higher"])
    dpb = dp["boundaries"]
    if j["higher"]:
        dp_cps = [dp["sorted_x"][b] for b in dpb[1:-1]]
    else:
        dp_cps = [dp["sorted_x"][b - 1] for b in dpb[1:-1]]
    # star deltas on the full score list between the two partitions' cut points
    def stars(cps, x, higher):
        xs = F(x)
        if higher:
            return 1 + sum(1 for t in cps if xs >= F(str(t)))
        return 5 - sum(1 for t in cps if xs > F(str(t)))
    n_star_delta = sum(
        1 for s in trimmed
        if stars(ward_cps, s, j["higher"]) != stars(dp_cps, s, j["higher"]))
    return {"year": j["year"], "mid": j["mid"], "org": j["org"],
            "name": j["name"], "n_input": len(scores),
            "n_trimmed": len(trimmed), "k": k,
            "membership": j["membership"],
            "ward_ssq": str(ward_ssq), "dp_ssq": str(dp["ssq"]),
            "gap": str(gap), "gap_float": float(gap),
            "off_optimum": gap > 0, "dp_n_optima": n_opt,
            "ward_cutpoints": [str(c) for c in ward_cps],
            "dp_cutpoints": [str(c) for c in dp_cps],
            "n_star_delta_fullsample": n_star_delta}


def main(workers=os.cpu_count(), out_name="exact_gaps_fullsample.json"):
    jobs = build_jobs()
    print(f"jobs: {len(jobs)}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_run, jobs)
    out = os.path.join(BASE, "runs", out_name)
    json.dump(rows, open(out, "w"), indent=1)
    off = [r for r in rows if r["off_optimum"]]
    fe = [r for r in rows if r["membership"] == "fence-exact"]
    fe_off = [r for r in fe if r["off_optimum"]]
    print(f"\nEXACT (full score list): {len(off)}/{len(rows)} measure-splits "
          f"off the SSQ optimum; fence-exact subset: {len(fe_off)}/{len(fe)}")
    tot_delta = sum(r["n_star_delta_fullsample"] for r in rows)
    print(f"full-score-list star deltas (ward vs exact-optimal cut points): "
          f"{tot_delta} contract-measure cells across all years")
    rows.sort(key=lambda r: -r["gap_float"])
    for r in rows[:12]:
        print(f"  {r['year']} {r['mid']} {r['org']:>5} {r['name'][:38]:38} "
              f"gap={r['gap_float']:.3f} stars_moved={r['n_star_delta_fullsample']:3d} "
              f"optima={r['dp_n_optima']} [{r['membership']}]")
    nonuniq = [r for r in rows if r["dp_n_optima"] not in (1,)]
    print(f"targets with non-unique DP optimum: {len(nonuniq)}")
    print("saved", out)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
