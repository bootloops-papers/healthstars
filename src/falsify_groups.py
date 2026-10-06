# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""End-to-end falsification of GROUPS= candidates against published cut points.

Primary targets: improvement measures (C30, D-part improvement) — guardrail-EXEMPT
(published final cut points == pre-guardrail mean-resampling values) and published
at 6 decimals: a correct (candidate, input-order, tukey-flag) triple must
reproduce dozens of 6-decimal values exactly; a wrong one has essentially zero
chance of matching any of them.

Secondary targets (after a winner emerges): all non-CAHPS, non-new measures vs
the PRE-guardrail thresholds from the Tech Notes tables (spec/validation_targets),
and vs final published cut points after applying guardrails.

Display rounding: SAS w.d format rounds half away from zero. We compare
round_display(exact mean, d) == published string, and ALSO report near-misses
(|ours - published| <= 10^-d) to distinguish "wrong algorithm" from "right
algorithm, boundary rounding or one-contract discrepancies".

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (candidate group-assignment rules tested against the published cut points).
Run:  cd src && python3 falsify_groups.py 2026 current C28
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""

import csv
import os
from fractions import Fraction

from pipeline import measure_cutpoints
import groups_assign
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(BASE, "data", "parsed")


def round_display(x, d):
    """SAS w.d display rounding: nearest, half away from zero. Returns string."""
    q = Fraction(10) ** d
    y = x * q
    n, den = y.numerator, y.denominator
    sign = -1 if n < 0 else 1
    n = abs(n)
    whole, rem = divmod(n, den)
    if 2 * rem >= den:
        whole += 1
    val = sign * whole
    s = str(abs(val)).rjust(d + 1, "0")
    out = (("-" if val < 0 else "")
           + (s[:-d] if d else s)
           + (("." + s[-d:]) if d else ""))
    return out


def load_scores(year, vintage, measure_id):
    """Scores for one measure, contract_id ascending, as decimal strings."""
    path = os.path.join(PARSED, f"scores_{year}_{vintage}.csv")
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["measure_id"] == measure_id and r["score"].strip():
                rows.append((r["contract_id"], r["score"].strip()))
    rows.sort(key=lambda t: t[0])
    return [s for _, s in rows], [c for c, _ in rows]


def load_published_cutpoints(year, vintage, measure_id):
    """Published thresholds: {star_level: (lo, lo_incl, hi, hi_incl)} as strings."""
    path = os.path.join(PARSED, f"cutpoints_{year}_{vintage}.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["measure_id"] == measure_id:
                out[int(r["star_level"])] = (r["lo"], r["lo_incl"], r["hi"], r["hi_incl"])
    return out


def improvement_published_cuts(pub):
    """Extract [cut12, cut34, cut45] decimal strings from published improvement rows.
    Rows: 1star: < c12 ; 2star: >= c12 to < 0 ; 3star: >= 0 to < c34 ;
    4star: >= c34 to < c45 ; 5star: >= c45."""
    c12 = pub[2][0]
    c34 = pub[4][0]
    c45 = pub[5][0]
    assert pub[3][0] in ("0", "0.0", "0.000000"), pub[3]
    return [c12, c34, c45]


def falsify_improvement(year, vintage, measure_id, display_decimals=6,
                        candidates=None, orders=("cid_asc", "cid_desc"),
                        tukey_options=(True, False), verbose=True):
    """Try all (candidate, order, tukey) combos on one improvement measure-year.
    Returns list of dicts with per-combo match results."""
    scores, cids = load_scores(year, vintage, measure_id)
    pub = load_published_cutpoints(year, vintage, measure_id)
    target = improvement_published_cuts(pub)
    if candidates is None:
        candidates = list(groups_assign.CANDIDATES)
    results = []
    for order in orders:
        ss = scores if order == "cid_asc" else scores[::-1]
        for tk in tukey_options:
            for cand in candidates:
                r = measure_cutpoints(ss, improvement=True, tukey=tk,
                                      groups_candidate=cand, arm="ward")
                if "error" in r:
                    results.append({"cand": cand, "order": order, "tukey": tk,
                                    "match": 0, "of": len(target), "note": r["error"]})
                    continue
                ours = [round_display(c, display_decimals) for c in r["mean_cutpoints"]]
                if len(ours) != len(target):
                    results.append({"cand": cand, "order": order, "tukey": tk,
                                    "match": 0, "of": len(target),
                                    "note": f"count {len(ours)} vs {len(target)}",
                                    "ours": ours})
                    continue
                m = sum(1 for a, b in zip(ours, target)
                        if Fraction(a) == Fraction(b))
                near = sum(1 for a, b in zip(ours, target)
                           if abs(Fraction(a) - Fraction(b)) <= Fraction(1, 10 ** display_decimals))
                results.append({"cand": cand, "order": order, "tukey": tk,
                                "match": m, "near": near, "of": len(target),
                                "ours": ours, "target": target})
    if verbose:
        print(f"\n=== {year}/{vintage} {measure_id} (n={len(scores)}) target={target} ===")
        for r in sorted(results, key=lambda r: -r["match"]):
            print(f"  {r['cand']:>18} {r['order']:>8} tukey={str(r['tukey']):>5} "
                  f"match {r['match']}/{r['of']}"
                  + (f" near {r.get('near')}" if 'near' in r else "")
                  + (f" ours={r.get('ours')}" if r["match"] < r["of"] else " EXACT")
                  + (f" [{r['note']}]" if r.get("note") else ""))
    return results


def published_boundaries(pub):
    """4 boundary values ascending + direction from published star rows.

    higher-is-better: 1star row has hi only -> boundaries = lo of rows 2..5.
    lower-is-better:  1star row has lo only -> boundaries = lo of rows 4,3,2,1
    (ascending = [4star lo, 3star lo, 2star lo, 1star lo])."""
    r1 = pub[1]
    higher = r1[0] == "" and r1[2] != ""
    if higher:
        vals = [pub[s][0] for s in (2, 3, 4, 5)]
    else:
        vals = [pub[s][0] for s in (4, 3, 2, 1)]
    if any(v == "" for v in vals):
        return None, higher
    return vals, higher


def falsify_measure(year, vintage, measure_id, cap_lo=None, cap_hi=None,
                    candidates=None, orders=("cid_asc",),
                    verbose=True):
    """Falsification on a generic non-improvement measure: compare the
    pre-guardrail mean cut points (display-rounded at the published precision)
    to the published final cut points. Guardrail-bound targets show up as
    systematic near-misses; guardrail-slack ones must match exactly under the
    right candidate."""
    scores, cids = load_scores(year, vintage, measure_id)
    pub = load_published_cutpoints(year, vintage, measure_id)
    if not pub or len(pub) < 5:
        if verbose:
            print(f"{year}/{vintage} {measure_id}: no 5-row published cut points; skip")
        return []
    target, higher = published_boundaries(pub)
    if target is None:
        if verbose:
            print(f"{year}/{vintage} {measure_id}: unparsable boundaries; skip")
        return []
    dd = max(len(v.split(".")[1]) if "." in v else 0 for v in target)
    if candidates is None:
        candidates = list(groups_assign.CANDIDATES)
    results = []
    for order in orders:
        ss = scores if order == "cid_asc" else scores[::-1]
        for cand in candidates:
            r = measure_cutpoints(ss, higher_is_better=higher, tukey=True,
                                  cap_lo=cap_lo, cap_hi=cap_hi,
                                  groups_candidate=cand, arm="ward")
            if "error" in r or len(r["mean_cutpoints"]) != 4:
                results.append({"cand": cand, "order": order, "match": 0, "of": 4,
                                "note": r.get("error", f"{len(r.get('mean_cutpoints', []))} cps")})
                continue
            ours = [round_display(c, dd) for c in r["mean_cutpoints"]]
            m = sum(1 for a, b in zip(ours, target) if Fraction(a) == Fraction(b))
            results.append({"cand": cand, "order": order, "match": m, "of": 4,
                            "ours": ours, "target": target, "decimals": dd,
                            "higher": higher, "n": len(ss)})
    if verbose:
        print(f"\n=== {year}/{vintage} {measure_id} n={len(scores)} "
              f"{'HIB' if higher else 'LIB'} dd={dd} target={target} ===")
        for r in sorted(results, key=lambda r: -r["match"]):
            print(f"  {r['cand']:>18} {r['order']:>8} match {r['match']}/4"
                  + (f" ours={r['ours']}" if r["match"] < 4 else " EXACT")
                  + (f" [{r['note']}]" if r.get("note") else ""))
    return results


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not os.path.exists(os.path.join(PARSED, "scores_%s_%s.csv" % (sys.argv[1] if len(sys.argv) > 1 else "2026",
                                                            sys.argv[2] if len(sys.argv) > 2 else "current"))):
    print("falsify_groups: no data/parsed/scores_<year>_<vintage>.csv for year %s, vintage %s (published star "
          "years 2023-2026; vintages as named in data/parsed/)"
          % (sys.argv[1] if len(sys.argv) > 1 else "2026", sys.argv[2] if len(sys.argv) > 2 else "current"),
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    import sys
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    vintage = sys.argv[2] if len(sys.argv) > 2 else "current"
    mid = sys.argv[3] if len(sys.argv) > 3 else "C28"
    if mid in ("C30",):
        falsify_improvement(year, vintage, mid)
    else:
        cl = "0"
        ch = None if mid in ("C28", "D02") else "100"
        falsify_measure(year, vintage, mid, cap_lo=cl, cap_hi=ch)
