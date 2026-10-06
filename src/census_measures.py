# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Measure-level two-arm census: Ward-replay vs DP-substituted cut points through
the full pre-star pipeline (outlier trimming -> resampling groups -> cluster ->
mean -> guardrail -> display rounding), star deltas per contract per measure.

Scope (stated in every output): resampling-group assignment uses the stated
group-assignment rule (see groups_assign.py); guardrail bases are published
prior-year finals at display precision (not internal precision); the score list
is all published numeric scores. Both arms share resampling groups, bases and
membership, so the arm deltas isolate the clustering-step substitution.

CAHPS measures and improvement measures are held at published stars in both arms
(CAHPS: not clustered at all; improvement: input scores not public). This
UNDERCOUNTS counterfactual movement (conservative for the flip census).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 and Appendix C.4
    (the two-arm census through guardrail, rounding and aggregation).
Run:  cd src && python3 census_measures.py seq_quota_last
Requires: Python >= 3.10; numpy, scipy.
"""

import csv
import json
import os
import sys
from fractions import Fraction

from pipeline import measure_cutpoints
from guardrail import guardrail_boundary, restricted_range_cap
from falsify_groups import load_scores, load_published_cutpoints, \
    published_boundaries, round_display
from tukey import tukey_trim
from assign_stars import load_org_class

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")

CAHPS_NAMES = {
    "Annual Flu Vaccine", "Getting Needed Care", "Getting Appointments and Care Quickly",
    "Customer Service", "Rating of Health Care Quality", "Rating of Health Plan",
    "Care Coordination", "Getting Needed Prescription Drugs", "Rating of Drug Plan",
}
IMPROVEMENT_NAMES = {"Health Plan Quality Improvement", "Drug Plan Quality Improvement"}
# complaints are rate-per-1000-style (not 0-100 percent): restricted-range guardrail
RESTRICTED_RANGE_NAMES = {"Complaints about the Health Plan", "Complaints about the Drug Plan"}
# measures new in year (guardrail-exempt, by name), derived from cross-year diffs
NEW_BY_YEAR = {
    # 'MPF Price Accuracy' is new/respecified through 2024: CMS's 2024 final
    # D07 equals the pre-guardrail values (SPEC_NOTES concurs), so it is not
    # clamped against 2023's boundaries.
    "2024": {"Controlling High Blood Pressure", "Plan All-Cause Readmissions",
             "Transitions of Care", "MPF Price Accuracy",
             "Follow-up after Emergency Department Visit for People with Multiple High-Risk Chronic Conditions"},
    "2025": {"Controlling High Blood Pressure", "Plan All-Cause Readmissions",
             "Transitions of Care",
             "Follow-up after Emergency Department Visit for People with Multiple High-Risk Chronic Conditions"},
    "2026": {"Controlling High Blood Pressure", "Plan All-Cause Readmissions",
             "Transitions of Care",
             "Follow-up after Emergency Department Visit for People with Multiple High-Risk Chronic Conditions",
             "Improving or Maintaining Physical Health", "Improving or Maintaining Mental Health",
             "Kidney Health Evaluation for Patients with Diabetes"},
}

VINTAGE = {"2023": "oct2022", "2024": "recalc", "2025": "current", "2026": "current"}
PRIOR_YEAR = {"2024": "2023", "2025": "2024", "2026": "2025"}


def measure_registry(year, vintage):
    """measure_id -> dict(name, part, org_split) from the cutpoints file."""
    path = os.path.join(PARSED, f"cutpoints_{year}_{vintage}.csv")
    reg = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m = reg.setdefault(r["measure_id"], {
                "name": r["measure_name"].strip(), "part": r["part"],
                "org_types": set()})
            m["org_types"].add(r.get("org_type", "").strip() or "ALL")
    return reg


def _prior_pid(reg, mid_name, part):
    """Resolve a prior-year measure id by NAME + PART.

    Keying by name alone would collide the cross-part twins ('Call Center ...' = 2023 C28/D01,
    'Members Choosing to Leave the Plan' = C24/D03), guardrailing Part D
    measures against Part C priors. Part is therefore REQUIRED; ambiguity
    raises rather than guessing.
    """
    cands = [k for k, v in reg.items()
             if v["name"] == mid_name and (part is None or v["part"] == part)]
    if len(cands) > 1:
        raise AssertionError(
            f"ambiguous prior measure for {mid_name!r} part={part}: {cands}")
    return cands[0] if cands else None


def prior_final_boundaries(year, mid_name, org_type, part=None):
    """Published prior-year final boundaries (ascending) by NAME + PART."""
    py = PRIOR_YEAR.get(year)
    if py is None:
        return None
    pv = VINTAGE[py]
    reg = measure_registry(py, pv)
    pid = _prior_pid(reg, mid_name, part)
    if pid is None:
        return None
    path = os.path.join(PARSED, f"cutpoints_{py}_{pv}.csv")
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            oc = r.get("org_type", "").strip() or "ALL"
            if r["measure_id"] == pid and oc in (org_type, "ALL"):
                rows[int(r["star_level"])] = (r["lo"], r["lo_incl"], r["hi"], r["hi_incl"])
    if len(rows) < 5:
        return None
    return published_boundaries(rows)[0]


def prior_trimmed_scores(year, mid_name, part=None):
    py = PRIOR_YEAR.get(year)
    pv = VINTAGE[py]
    reg = measure_registry(py, pv)
    pid = _prior_pid(reg, mid_name, part)   # keyed by name+part
    if pid is None:
        return None
    scores, _ = load_scores(py, pv, pid)
    if not scores:
        return None
    mask, _, _ = tukey_trim(scores, cap_lo="0")
    return [s for s, m in zip(scores, mask) if m]


def run_measure(year, vintage, mid, reg, candidate="seq_quota_last"):
    """Both arms for one measure(-org split). Returns list of result dicts."""
    name = reg[mid]["name"]
    if name in CAHPS_NAMES or name in IMPROVEMENT_NAMES:
        return []
    scores_all, cids_all = load_scores(year, vintage, mid)
    if len(scores_all) < 30:
        return []
    org = load_org_class(year, vintage)
    splits = ([("ALL", None)] if reg[mid]["org_types"] == {"ALL"}
              else [("MA-PD", "MA-PD"), ("PDP", "PDP")])
    out = []
    for label, oc in splits:
        if oc is None:
            scores = scores_all
            cids = cids_all
        else:
            pairs = [(c, s) for c, s in zip(cids_all, scores_all) if org.get(c) == oc]
            cids = [c for c, _ in pairs]
            scores = [s for _, s in pairs]
        if len(scores) < 30:
            continue
        pub = load_published_cutpoints(year, vintage, mid)
        # restrict published rows to this org split when split applies
        if oc is not None:
            path = os.path.join(PARSED, f"cutpoints_{year}_{vintage}.csv")
            pub = {}
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r["measure_id"] == mid and (r.get("org_type", "").strip() or "ALL") == oc:
                        pub[int(r["star_level"])] = (r["lo"], r["lo_incl"], r["hi"], r["hi_incl"])
        if len(pub) < 5:
            continue
        target, higher = published_boundaries(pub)
        if target is None:
            continue
        dd = max(len(v.split(".")[1]) if "." in v else 0 for v in target)
        restricted = name in RESTRICTED_RANGE_NAMES
        cap_lo, cap_hi = ("0", None) if restricted else ("0", "100")
        arms = {}
        err = None
        for arm in ("ward", "dp"):
            r = measure_cutpoints(scores, higher_is_better=higher, tukey=True,
                                  cap_lo=cap_lo, cap_hi=cap_hi,
                                  groups_candidate=candidate, arm=arm)
            if "error" in r or len(r["mean_cutpoints"]) != 4:
                err = r.get("error", "cutpoint count != 4")
                break
            arms[arm] = r["mean_cutpoints"]
        if err:
            out.append({"year": year, "mid": mid, "org": label, "name": name,
                        "error": err})
            continue
        # guardrails (identical treatment both arms)
        exempt = name in NEW_BY_YEAR.get(year, set())
        gr_note = "exempt-new" if exempt else "applied"
        prior = None if exempt else prior_final_boundaries(year, name, label, part=reg[mid]["part"])
        if not exempt and prior is None:
            gr_note = "no-prior->treated-exempt"
        finals = {}
        for arm in ("ward", "dp"):
            vals = arms[arm]
            if gr_note == "applied":
                if restricted:
                    pts = prior_trimmed_scores(year, name, part=reg[mid]["part"])
                    cap = restricted_range_cap(pts) if pts else None
                else:
                    cap = Fraction(5)
                if cap is None:
                    gr_note = "no-prior-range->exempt"
                    fin = vals
                else:
                    fin = [guardrail_boundary(v, Fraction(p), cap)[0]
                           for v, p in zip(vals, prior)]
            else:
                fin = vals
            finals[arm] = [round_display(v, dd) for v in fin]
        # star assignment per contract under each arm's boundaries
        deltas = []
        for c, s in zip(cids, scores):
            st = {}
            for arm in ("ward", "dp"):
                b = [Fraction(x) for x in finals[arm]]
                x = Fraction(s)
                if higher:
                    star = 1 + sum(1 for t in b if x >= t)
                else:
                    star = 5 - sum(1 for t in b if x > t)
                st[arm] = star
            if st["ward"] != st["dp"]:
                deltas.append((c, s, st["ward"], st["dp"]))
        pub_match = sum(1 for a, b in zip(finals["ward"], target)
                        if Fraction(a) == Fraction(b))
        out.append({"year": year, "mid": mid, "org": label, "name": name,
                    "n": len(scores), "higher": higher, "dd": dd,
                    "guardrail": gr_note,
                    "ward_final": finals["ward"], "dp_final": finals["dp"],
                    "published": target, "pub_match_ward": pub_match,
                    "n_star_changes": len(deltas),
                    "deltas": deltas})
    return out


def _worker(args):
    year, vintage, mid, candidate = args
    reg = measure_registry(year, vintage)
    try:
        return run_measure(year, vintage, mid, reg, candidate)
    except Exception as e:
        return [{"year": year, "mid": mid, "error": repr(e)}]


def main(candidate="seq_quota_last", years=("2024", "2025", "2026"), workers=8):
    import multiprocessing as mp
    tasks = []
    for year in years:
        vintage = VINTAGE[year]
        reg = measure_registry(year, vintage)
        tasks.extend((year, vintage, mid, candidate) for mid in sorted(reg))
    results = []
    with mp.Pool(workers) as pool:
        for chunk in pool.imap_unordered(_worker, tasks):
            results.extend(chunk)
    outpath = os.path.join(BASE, "runs", f"census_measures_{candidate}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=1, default=str)
    # summary
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    print(f"measure-splits computed: {len(ok)}, errors: {len(errs)}")
    for year in years:
        yr = [r for r in ok if r["year"] == year]
        chg = sum(r["n_star_changes"] for r in yr)
        n_meas = len(yr)
        moved = sum(1 for r in yr if r["n_star_changes"] > 0)
        fid = sum(r["pub_match_ward"] for r in yr)
        fid_of = 4 * n_meas
        print(f"  SY{year}: {n_meas} measure-splits | ward-arm matches published "
              f"{fid}/{fid_of} boundaries | {moved} splits with star changes | "
              f"{chg} contract-measure star changes (ward vs dp)")
    for r in sorted(ok, key=lambda r: -r["n_star_changes"])[:15]:
        print(f"    {r['year']} {r['mid']} {r['org']:>5} {r['name'][:40]:40} "
              f"gr={r['guardrail'][:12]:12} changes={r['n_star_changes']:3d} "
              f"ward={r['ward_final']} dp={r['dp_final']}")
    if errs:
        print("  errors:")
        for r in errs[:8]:
            print("   ", r["year"], r["mid"], str(r["error"])[:100])
    return outpath


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    cand = sys.argv[1] if len(sys.argv) > 1 else "seq_quota_last"
    main(cand)


# ---------------------------------------------------------------------------
# CMS's extreme-and-uncontrollable-
# circumstances MEASURE-star hold-harmless (2026 Tech Notes pp. 7-8):
# contracts with >= 25% affected enrollees receive the
# HIGHER of the current vs prior-year measure star. Trigger year by class:
# HOS / HEDIS-HOS measures key on the EARLIER disaster column (year-3),
# everything else on the LATER (year-2); Call Center gets no adjustment
# ("For all contracts, no adjustments were made"); CAHPS / improvement are
# held at published throughout this pipeline anyway. The new-measure
# with/without overall-level provision is not modeled (stated in every
# consuming record). Prior-year stars matched by NAME + PART (ids drift).
HOS_CLASS_NAMES = {
    "Monitoring Physical Activity", "Reducing the Risk of Falling",
    "Improving Bladder Control", "Improving or Maintaining Physical Health",
    "Improving or Maintaining Mental Health",
}
CALL_CENTER_NAMES = {
    "Call Center \u2013 Foreign Language Interpreter and TTY Availability",
}


def disaster_pcts(year, vintage):
    """{contract_id: (early_pct, late_pct)} from the year's summary table."""
    path = os.path.join(PARSED, f"summary_{year}_{vintage}.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            def _p(v):
                v = (v or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return 0.0
            out[r["contract_id"]] = (_p(r.get("disaster_col1_raw")),
                                     _p(r.get("disaster_col2_raw")))
    return out


def prior_published_star_map(year):
    """{(name, part): {cid: star_int}} for the prior star year at the
    operative published vintage (2023 oct2022 / 2024 recalc / 2025 current)."""
    py = PRIOR_YEAR.get(year)
    if py is None:
        return {}
    pv = VINTAGE[py]
    reg = measure_registry(py, pv)
    out = {}
    with open(os.path.join(PARSED, f"stars_{py}_{pv}.csv"), newline="",
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            e = reg.get(r["measure_id"])
            if not e:
                continue
            v = (r["stars"] or "").strip()
            if v.replace(".", "").isdigit():
                out.setdefault((e["name"], e["part"]), {})[r["contract_id"]] = \
                    int(float(v))
    return out


def disaster_higher_of(name, part, cid, sd, dis, prior_map):
    """Hold-harmless: the criterion star for an affected contract is at
    least its prior-year published star (scope per the measure classes above)."""
    if (name in CALL_CENTER_NAMES or name in CAHPS_NAMES
            or name in IMPROVEMENT_NAMES):
        return sd
    pct = dis.get(cid)
    if not pct:
        return sd
    trig = pct[0] if name in HOS_CLASS_NAMES else pct[1]
    if trig < 25:
        return sd
    ps = prior_map.get((name, part), {}).get(cid)
    return sd if ps is None else max(sd, ps)
