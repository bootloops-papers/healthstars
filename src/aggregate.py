# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Measure stars -> Part C/D summary and overall ratings, per Tech Notes spec
(spec/aggregation_<year>.json). Exact Fractions; rounding LAST via Table 22.

Pipeline per rating type (Part C summary / Part D summary / overall):
  1. raw = sum(w_i * star_i)/sum(w_i), with and without improvement measure(s)
  2. + reward factor per run (mean vs Table 8, weighted variance vs Table 9;
     thresholds published — never re-derived)
  3. +/- CAI (per-contract final adjustment category from the CAI data table,
     category value from Tables 12/15/18/21)
  4. improvement hold-harmless on the contract's HIGHEST rating: if step-2
     (without-improvement, fully adjusted) >= threshold, final = max(step2, step3);
     else step3. Threshold convention (gate_convention): 'rounded' (round(step2) >= 4.0) or
     'raw' (step2 >= 3.75); both implemented.
     For MA-PD summaries improvement is simply included (hold-harmless applies
     to overall only); MA-only: Part C summary; PDP: Part D summary.
  5. Table 22 half-star rounding, band lower bounds inclusive.

Eligibility (Tables 6/7 minimum measure counts) is NOT re-derived: we compute
ratings only for contracts that have a published rating (fidelity check) or
had one in the Ward arm (flip census).

Reward-factor 'New Measures' dimension: standard runs use the With-new columns
(a contract's rated set naturally includes its new measures). The Without-new
columns exist for the 25%+ disaster new-measure hold-harmless (2024/2026):
for disaster-affected contracts (public: 'Disaster %' column in the summary
data table) we also compute the without-new variant and take the better final.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.5 and Appendix C.4
    (the aggregation stage: reward factor, categorical adjustment, hold-
    harmless, Table 22 rounding).
Run:  cd src && python3 aggregate.py 2026 current   (fidelity check of the aggregation stage;
      the star year and vintage are optional)
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10, standard library only.
"""

import csv
import json
import os
import sys
from fractions import Fraction

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")
SPEC = os.path.join(BASE, "spec")

IMPROVEMENT_IDS = {"2024": ("C27", "D04"), "2025": ("C27", "D04"), "2026": ("C30", "D04")}
# reward-table "new measures" sets (NOT the guardrail-new sets)
REWARD_NEW = {"2024": {"C15", "C17", "C18"}, "2025": set(), "2026": {"C04", "C05", "C13"}}
# D-part twins of C-part measures (same data source), counted ONCE in the
# overall rating: drop the D instance from the overall pool (Tech Notes,
# 'CMS includes only one instance of each of these two measures').
DUP_D_NAMES = {"Complaints about the Drug Plan", "Members Choosing to Leave the Plan"}


def dup_d_ids(year, vintage):
    import csv as _csv
    path = os.path.join(PARSED, f"scores_{year}_{vintage}.csv")
    out = set()
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            mid = r["measure_id"]
            if mid in seen or not mid.startswith("D"):
                continue
            seen.add(mid)
            if r["measure_name"].strip() in DUP_D_NAMES:
                out.add(mid)
    return out

F = Fraction


class AggSpec:
    def __init__(self, year):
        self.year = str(year)
        d = json.load(open(os.path.join(SPEC, f"aggregation_{self.year}.json")))
        self.raw = d
        w = {}
        for part in ("part_c", "part_d"):
            for mid, info in d[f"measure_weights_{self.year}"][part].items():
                w[mid] = F(info["weight"])
        self.weights = w
        self.imp_ids = set(IMPROVEMENT_IDS[self.year])
        self.reward_new = REWARD_NEW[self.year]
        # thresholds keyed (imp, new, pct) -> {col: Fraction}
        self.t8 = {}
        for r in d["reward_factor"]["table_8_performance_summary_thresholds"]:
            key = (r["improvement"], r.get("new_measures", "With"), r["percentile"])
            self.t8[key] = {c: F(r[c]) for c in ("part_c", "part_d_mapd", "part_d_pdp", "overall") if r.get(c)}
        self.t9 = {}
        for r in d["reward_factor"]["table_9_variance_thresholds"]:
            key = (r["improvement"], r.get("new_measures", "With"), r["percentile"])
            self.t9[key] = {c: F(r[c]) for c in ("part_c", "part_d_mapd", "part_d_pdp", "overall") if r.get(c)}
        # CAI category -> value per rating type
        fac = d["cai"]["final_adjustment_categories"]
        self.cai = {
            "overall": {r["final_adjustment_category"]: F(r["cai_value"]) for r in fac["overall_table12"]},
            "part_c": {r["final_adjustment_category"]: F(r["cai_value"]) for r in fac["part_c_table15"]},
            "part_d_mapd": {r["final_adjustment_category"]: F(r["cai_value"]) for r in fac["mapd_part_d_table18"]},
            "part_d_pdp": {r["final_adjustment_category"]: F(r["cai_value"]) for r in fac["pdp_part_d_table21"]},
        }
        self.bands = [(F(b["raw_ge"]), F(b["raw_lt"]) if b.get("raw_lt") else None, b["final"])
                      for b in d["summary_rating_rule"]["rounding"]["table_22_bands"]]

    def round22(self, x):
        for lo, hi, fin in self.bands:
            if x >= lo and (hi is None or x < hi):
                return fin
        return "0" if x < 0 else "5.0"


def weighted_mean_var(stars, weights, include):
    """stars: {mid: int}; include: set of mids. Returns (mean, var, n) Fractions."""
    items = [(mid, s) for mid, s in stars.items() if mid in include]
    n = len(items)
    if n == 0:
        return None, None, 0
    W = sum(weights[mid] for mid, _ in items)
    mean = sum(weights[mid] * s for mid, s in items) / W
    if n == 1:
        return mean, F(0), 1
    var = F(n, (n - 1)) / W * sum(weights[mid] * (s - mean) ** 2 for mid, s in items)
    return mean, var, n


def _round6(x):
    """Round Fraction to 6 decimals, half away from zero, the precision of the
    published Table 8/9 thresholds (which are 6-dp roundings of attained
    values). An exact-vs-rounded comparison would deny rewards CMS awarded for
    contracts whose exact mean IS the attained threshold value (e.g. 137/37 vs
    '3.702703')."""
    q = 10 ** 6
    n, den = (x * q).numerator, (x * q).denominator
    sign = -1 if n < 0 else 1
    n = abs(n)
    w, rem = divmod(n, den)
    if 2 * rem >= den:
        w += 1
    return F(sign * w, q)


def reward(spec, mean, var, imp_label, new_label, col):
    t8h = spec.t8[(imp_label, new_label, "85th")][col]
    t8m = spec.t8[(imp_label, new_label, "65th")][col]
    t9lo = spec.t9[(imp_label, new_label, "30th")][col]
    t9hi = spec.t9[(imp_label, new_label, "70th")][col]
    mean = _round6(mean)
    var = _round6(var)
    if var < t9lo:
        vcat = "low"
    elif var < t9hi:
        vcat = "medium"
    else:
        vcat = "high"
    if mean >= t8h:
        mcat = "high"
    elif mean >= t8m:
        mcat = "relhigh"
    else:
        mcat = "other"
    if vcat == "low" and mcat == "high":
        return F("0.4")
    if vcat == "medium" and mcat == "high":
        return F("0.3")
    if vcat == "low" and mcat == "relhigh":
        return F("0.2")
    if vcat == "medium" and mcat == "relhigh":
        return F("0.1")
    return F(0)


def rate_contract(spec, stars, org_class, cai_cats, *, gate_convention="raw",
                  disaster=False, dup_d=frozenset(), hh_scope="highest"):
    """stars: {mid: star int} (published or arm-substituted). cai_cats: dict with
    keys overall/part_c/part_d_mapd/part_d_pdp -> category string or None.
    org_class: 'MA-PD' | 'MA-only' | 'PDP'.
    hh_scope: 'highest' = hold-harmless max(with,without) applied only to the
    highest rating (Tech Notes steps read literally); 'all' = when the gate
    passes on the highest rating, apply max(with,without) to EVERY rating type
    (422.166(g)-style reading). Both testable against published ratings.
    Returns {rating_type: (final_str, raw_final_Fraction)}."""
    out = {}
    cmids = {m for m in stars if m.startswith("C")}
    dmids = {m for m in stars if m.startswith("D")}
    types = []
    if org_class == "MA-PD":
        types = [("part_c", cmids, "part_c"), ("part_d", dmids, "part_d_mapd"),
                 ("overall", (cmids | dmids) - dup_d, "overall")]
        hh_type = "overall"
    elif org_class == "MA-only":
        types = [("part_c", cmids, "part_c")]
        hh_type = "part_c"
    else:
        types = [("part_d", dmids, "part_d_pdp")]
        hh_type = "part_d"

    def adjusted(pool, col, include, imp_label, new_label):
        cai_cat = cai_cats.get(col)
        cai_val = spec.cai[col].get(cai_cat, F(0)) if cai_cat else F(0)
        mean, var, n = weighted_mean_var(stars, spec.weights, include)
        if mean is None:
            return None
        r = reward(spec, mean, var, imp_label, new_label, col)
        return mean + r + cai_val

    def all_finals(newset_excluded, new_label):
        vals = {}
        for rtype, pool, col in types:
            pe = pool - newset_excluded
            w = adjusted(pool, col, pe, "With", new_label)
            wo = adjusted(pool, col, pe - spec.imp_ids, "Without", new_label)
            if w is None or wo is None:
                continue
            vals[rtype] = (w, wo)
        if hh_type not in vals:
            return {r: v[0] for r, v in vals.items()}
        w_h, wo_h = vals[hh_type]
        if gate_convention == "raw":
            gate_ok = wo_h >= F("3.75")
        else:
            gate_ok = F(spec.round22(wo_h)) >= 4
        fin = {}
        for rtype, (w, wo) in vals.items():
            if rtype == hh_type:
                fin[rtype] = max(wo, w) if gate_ok else w
            elif hh_scope == "all" and gate_ok:
                fin[rtype] = max(wo, w)
            else:
                fin[rtype] = w
        return fin

    finals = all_finals(set(), "With")
    if disaster and spec.reward_new:
        alt = all_finals(spec.reward_new, "Without")
        for rtype, v in alt.items():
            if rtype in finals and v > finals[rtype]:
                finals[rtype] = v
    for rtype, v in finals.items():
        out[rtype] = (spec.round22(v), v)
    return out


def load_inputs(year, vintage):
    """Published stars, org class, CAI categories, disaster flags, published ratings."""
    stars = {}
    with open(os.path.join(PARSED, f"stars_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stars"].strip():
                stars.setdefault(r["contract_id"], {})[r["measure_id"]] = int(r["stars"])
    summary = {}
    with open(os.path.join(PARSED, f"summary_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            summary[r["contract_id"]] = r
    cai = {}
    with open(os.path.join(PARSED, f"cai_{year}_{vintage}.csv"),
              newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            def clean(x):
                x = (x or "").strip()
                return x if x and x not in ("N/A", "NA", "") else None
            cai[r["contract_id"]] = {
                "overall": clean(r.get("overall_fac_raw")),
                "part_c": clean(r.get("part_c_fac_raw")),
                "part_d_mapd": clean(r.get("part_d_mapd_fac_raw")),
                "part_d_pdp": clean(r.get("part_d_pdp_fac_raw")),
            }
    return stars, summary, cai


def org_class_of(summary_row):
    ot = summary_row.get("org_type", "")
    if "PDP" in ot:
        return "PDP"
    pc = (summary_row.get("part_c_summary_raw") or "").strip()
    pd_ = (summary_row.get("part_d_summary_raw") or "").strip()
    if pd_.startswith("Not Applicable") or pd_ == "":
        return "MA-only"
    return "MA-PD"


def disaster_pct_of(summary_row, year):
    """Operative disaster percentage: the provisions key on the MOST RECENT
    disaster year's column only (SY2024: '2022 Disaster %'; SY2025: 2023;
    SY2026: 2024) — NOT the max over both published columns.

    Taking the max over both columns would over-apply the new-measure
    hold-harmless to prior-prior-year disaster contracts. The column matching year-2
    (data year - 1 relative to star year label) is selected by header."""
    want = str(int(year) - 2)
    for i in (1, 2):
        h = summary_row.get(f"disaster_col{i}_header", "") or ""
        if want in h:
            v = (summary_row.get(f"disaster_col{i}_raw") or "").strip()
            try:
                return int(v)
            except ValueError:
                return 0
    # header not found: fall back to the second (most recent) column
    v = (summary_row.get("disaster_col2_raw") or "").strip()
    try:
        return int(v)
    except ValueError:
        return 0


def fidelity(year, vintage, gate_convention="raw", verbose=True):
    spec = AggSpec(year)
    stars, summary, cai = load_inputs(year, vintage)
    dd = frozenset(dup_d_ids(year, vintage))
    ok = bad = skip = 0
    mism = []
    for cid, srow in summary.items():
        pub_overall = (srow.get("overall") or "").strip()
        pub_c = (srow.get("part_c_summary") or "").strip()
        pub_d = (srow.get("part_d_summary") or "").strip()
        st = stars.get(cid)
        if not st:
            continue
        oc = org_class_of(srow)
        res = rate_contract(spec, st, oc, cai.get(cid, {}),
                            gate_convention=gate_convention,
                            disaster=disaster_pct_of(srow, year) >= 25,
                            dup_d=dd)
        for rtype, pub in (("overall", pub_overall), ("part_c", pub_c), ("part_d", pub_d)):
            if not pub:
                continue
            if rtype not in res:
                skip += 1
                continue
            got = res[rtype][0]
            if F(got) == F(pub):
                ok += 1
            else:
                bad += 1
                mism.append((cid, rtype, got, pub, float(res[rtype][1])))
    if verbose:
        print(f"{year}/{vintage} gate={gate_convention}: ratings OK={ok} mismatch={bad} skipped={skip}")
        for m in mism[:15]:
            print("   ", m)
    return ok, bad, mism


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not (os.path.exists(os.path.join(SPEC, "aggregation_%s.json" % (sys.argv[1] if len(sys.argv) > 1 else "2026")))
        and os.path.exists(os.path.join(PARSED, "scores_%s_%s.csv" % (sys.argv[1] if len(sys.argv) > 1 else "2026",
                                                                   sys.argv[2] if len(sys.argv) > 2 else "current")))):
    print("aggregate: no spec/aggregation_<year>.json or data/parsed/scores_<year>_<vintage>.csv for "
          "year %s, vintage %s (published star years 2024-2026; vintages as named in data/parsed/)"
          % (sys.argv[1] if len(sys.argv) > 1 else "2026", sys.argv[2] if len(sys.argv) > 2 else "current"),
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    import sys
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    vintage = sys.argv[2] if len(sys.argv) > 2 else "current"
    for gc in ("raw", "rounded"):
        fidelity(year, vintage, gc)
