# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Cross-measure membership stacker for the CMS Star Ratings audit.

Goal: per star year, find a single small DELTA SET — hypothetical ADDED
contracts (org_class in {MA-PD, MA-only, PDP}, per-measure participation +
exact in-register score) plus REMOVED real contracts — that turns every
missed published outlier-bound target exact while leaving every
currently-exact target exactly unchanged.  All accept/reject decisions in
exact Fraction arithmetic; published K-5/K-6 fences are exact unrounded
values of CMS's clustering input under membership hypothesis cost+d60r+emp.

Method
------
1. Per missed target, a constructive profile solver (extension of
   inverse_membership2 to large k): for k = 1..K_CAP additions (optionally on
   a removal-reduced base), enumerate candidate new-quartile pairs (q1, q3)
   from the fence equations
       lo = 4*q1 - 3*q3,  hi = 4*q3 - 3*q1   (capped),
   then solve the QNTLDEF=5 placement problem exactly: with n' = n + k,
   both quartile positions are 'avg' iff n' % 4 == 0 else 'at' ceil(n'p);
   count windows for adds below/at/between/above the required order
   statistics follow from linear inequalities in the add counts.  Witness
   add values are kept in the measure's data register (integers for percent
   measures, 2dp for complaints).  Every witness is re-verified with
   tukey.tukey_fences on the fully materialized value list (exact).
2. Cross-measure stacking: one added contract serves many measures (a score
   per measure, participation a free choice).  Complaints and
   members-choosing-to-leave C/D counterparts are value-coupled in the data
   (identical per MA-PD contract, verified), so the D-row witness multiset is
   forced to be a sub-multiset of the C-row multiset carried by the same
   MA-PD adds; C-only surplus values ride on MA-only adds.
3. Removals only where additions cannot reach the target (flagged axis);
   removal candidates are real contracts, and the removal is accepted only if
   every exact target of that year stays exactly unchanged (full battery).
4. Output: runs/membership_delta_sets.json with per-year adds, removals and
   the full before/after fence battery.

Register: exact.  No float ever participates in an accept/reject decision.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (reconstructing
    the score list on the documented-rules basis: filled-in scores,
    Table C1).
Run:  cd src && python3 membership_stacker_docsonly.py
Requires: Python >= 3.10; numpy, scipy.
"""

import json
import math
import multiprocessing as mp
import os
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import combinations_with_replacement

from tukey import qntldef5
from inverse_membership2 import required_qs
from tukey_battery import scores_by_measure, load_summary_meta, apply_hypothesis
from solve_all_misses import SCORE_VINTAGE
HYP = "cost+d60r"   # DOCUMENTED-RULES-ONLY basis: employer axis dropped

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(BASE, "spec")
RUNS = os.path.join(BASE, "runs")

K_CAP = 28          # max additions per single target
REM_CAP = 8         # max removals per single target
WORKERS = os.cpu_count() or 1   # worker processes

# C-side / D-side measure families that are value-coupled per MA-PD contract
# (verified in-data before being enforced).
FAMILY_FRAGMENTS = ("Complaints about", "Members Choosing to Leave",
                    "Call Center")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_year(year):
    """Return (targets, meta): targets = list of dicts for every published
    K-table row (deduped improvement rows dropped: no public score data)."""
    vintage = SCORE_VINTAGE[year]
    vt = json.load(open(os.path.join(SPEC, f"validation_targets_{year}.json")))
    sc = scores_by_measure(year, vintage)
    meta = load_summary_meta(year, vintage)
    disaster = year in ("2024", "2025")
    measures = json.load(open(os.path.join(SPEC, f"measures_{year}.json")))
    names = {m["measure_id"]: m["name"] for m in measures["measures"]}
    targets, seen = [], set()
    for r in vt["tukey_cutoffs"]["rows"]:
        mid = r["measure_id"]
        org = r.get("org_type") or "ALL"
        if (mid, org) in seen:          # improvement decline-group twin row
            continue
        seen.add((mid, org))
        pairs = sc.get(mid)
        if not pairs:
            targets.append({"year": year, "mid": mid, "org": org,
                            "name": names.get(mid, ""), "status": "no_data",
                            "pub_lo": r["lower"], "pub_hi": r["upper"]})
            continue
        if org in ("MA-PD", "PDP"):
            sub = [(c, s) for c, s in pairs
                   if ("PDP" in meta.get(c, ("",))[0]) == (org == "PDP")]
        else:
            sub = pairs
        kept = apply_hypothesis(sub, meta, HYP, disaster)
        # no n<30 size filter: fences are quartile arithmetic, evaluable at
        # any n. The n<5 guard matches the guard of the resampled rerun
        # against fewer than five scores.
        if len(kept) < 5:
            targets.append({"year": year, "mid": mid, "org": org,
                            "name": names.get(mid, ""), "status": "small_n",
                            "pub_lo": r["lower"], "pub_hi": r["upper"]})
            continue
        plo, phi = r["lower"], r["upper"]
        dd = max(len(v.split(".")[1]) if v and "." in v else 0
                 for v in (plo, phi) if v)
        is_pct = dd == 0 and phi is not None and F(phi) <= 100
        # data register: max decimals seen in the scores of this measure
        reg_dd = max(len(s.split(".")[1]) if "." in s else 0 for _, s in kept)
        t = {"year": year, "mid": mid, "org": org, "name": names.get(mid, ""),
             "pub_lo": plo, "pub_hi": phi, "dd": dd,
             "cap_lo": F(0), "cap_hi": F(100) if is_pct else None,
             "reg": F(1, 10 ** reg_dd),
             "score_hi": F(100) if is_pct else None,
             "pairs": [(c, F(s)) for c, s in kept]}
        vals = sorted(v for _, v in t["pairs"])
        lo, hi = fences_capped(vals, t["cap_lo"], t["cap_hi"])
        t["before"] = (lo, hi)
        t["status"] = "exact" if fences_match(lo, hi, plo, phi) else "miss"
        targets.append(t)
    return targets, meta


def fences_capped(sorted_vals, cap_lo, cap_hi):
    q1 = qntldef5(sorted_vals, F(1, 4))
    q3 = qntldef5(sorted_vals, F(3, 4))
    lo, hi = 4 * q1 - 3 * q3, 4 * q3 - 3 * q1
    if cap_lo is not None:
        lo = max(lo, cap_lo)
    if cap_hi is not None:
        hi = min(hi, cap_hi)
    return lo, hi


def fences_match(lo, hi, pub_lo, pub_hi):
    return ((pub_lo is None or lo == F(pub_lo)) and
            (pub_hi is None or hi == F(pub_hi)))


# --------------------------------------------------------------------------
# fast exact quartiles of  base-values (+ adds multiset)  via counting
# --------------------------------------------------------------------------

class BaseCounts:
    """Sorted distinct values + cumulative counts of a base value list."""

    def __init__(self, sorted_vals):
        self.vals = sorted_vals
        self.n = len(sorted_vals)
        dv, cum = [], []
        for v in sorted_vals:
            if not dv or v != dv[-1]:
                dv.append(v)
                cum.append(1 if not cum else cum[-1] + 1)
            else:
                cum[-1] += 1
        self.dv, self.cum = dv, cum

    def cnt_le(self, v):
        i = bisect_right(self.dv, v)
        return self.cum[i - 1] if i else 0

    def cnt_lt(self, v):
        i = bisect_left(self.dv, v)
        return self.cum[i - 1] if i else 0

    def merged_kth(self, adds_sorted, j):
        """j-th (1-based) order stat of base ∪ adds (adds: sorted list)."""
        def cnt(v):
            return self.cnt_le(v) + bisect_right(adds_sorted, v)
        best = None
        lo, hi = 0, len(self.dv) - 1
        if self.dv and cnt(self.dv[-1]) >= j:
            while lo < hi:
                mid = (lo + hi) // 2
                if cnt(self.dv[mid]) >= j:
                    hi = mid
                else:
                    lo = mid + 1
            best = self.dv[lo]
        for a in adds_sorted:
            if cnt(a) >= j and (best is None or a < best):
                best = a
                break
        return best

    def fences_with(self, adds_sorted, cap_lo, cap_hi):
        n = self.n + len(adds_sorted)
        if n % 4 == 0:
            j1, j3 = n // 4, 3 * n // 4
            q1 = (self.merged_kth(adds_sorted, j1) +
                  self.merged_kth(adds_sorted, j1 + 1)) / 2
            q3 = (self.merged_kth(adds_sorted, j3) +
                  self.merged_kth(adds_sorted, j3 + 1)) / 2
        else:
            q1 = self.merged_kth(adds_sorted, math.ceil(n / 4))
            q3 = self.merged_kth(adds_sorted, math.ceil(3 * n / 4))
        lo, hi = 4 * q1 - 3 * q3, 4 * q3 - 3 * q1
        if cap_lo is not None:
            lo = max(lo, cap_lo)
        if cap_hi is not None:
            hi = min(hi, cap_hi)
        return lo, hi


# --------------------------------------------------------------------------
# constructive add-only solver for one target and one k
# --------------------------------------------------------------------------

def _reg_valid(v, reg, score_hi):
    if v < 0 or (score_hi is not None and v > score_hi):
        return False
    r = v / reg
    return r.denominator == 1


def _q_pair_candidates(bc, k, pub_lo, pub_hi, cap_lo, cap_hi, reg):
    """Candidate (q1, q3) new-quartile pairs consistent with the published
    fences.  One-sided modes are gridded along the free quartile."""
    q1r, q3r, mode = required_qs(pub_lo, pub_hi, cap_lo, cap_hi)
    n = bc.n
    n_new = n + k
    if n_new % 4 == 0:
        j1, j3 = n_new // 4, 3 * n_new // 4
    else:
        j1, j3 = math.ceil(n_new / 4), math.ceil(3 * n_new / 4)

    def window_vals(j):
        i_lo = max(0, j - k - 4)
        i_hi = min(n - 1, j + k + 3)
        vlo, vhi = bc.vals[i_lo], bc.vals[i_hi]
        out = set()
        i0 = bisect_left(bc.dv, vlo)
        i1 = bisect_right(bc.dv, vhi)
        exist = bc.dv[i0:i1]
        out.update(exist)
        for a, b in zip(exist, exist[1:]):          # adjacent-pair averages
            out.add((a + b) / 2)
        # register and half-register multiples in [vlo, vhi]
        step = reg / 2
        start = math.ceil(vlo / step)
        stop = math.floor(vhi / step)
        if stop - start <= 800:
            out.update(m * step for m in range(start, stop + 1))
        return out

    pairs = set()
    if mode == "both":
        pairs.add((q1r, q3r))
    elif mode == "lo_only":
        L = F(pub_lo)
        for q3 in window_vals(j3):
            q1 = (L + 3 * q3) / 4
            if q1 <= q3:
                pairs.add((q1, q3))
        for q1 in window_vals(j1):
            q3 = (4 * q1 - L) / 3
            if q1 <= q3:
                pairs.add((q1, q3))
    elif mode == "hi_only":
        U = F(pub_hi)
        for q3 in window_vals(j3):
            q1 = (4 * q3 - U) / 3
            if 0 <= q1 <= q3:
                pairs.add((q1, q3))
        for q1 in window_vals(j1):
            q3 = (U + 3 * q1) / 4
            if q1 <= q3:
                pairs.add((q1, q3))
    else:  # 'none': both sides capped; grid both windows (rare)
        for q1 in window_vals(j1):
            for q3 in window_vals(j3):
                if q1 <= q3:
                    pairs.add((q1, q3))
    return pairs, (j1, j3, n_new % 4 == 0)


def _quartile_realizations(bc, q, j, avg, k, reg, score_hi):
    """Ways to make the new quartile at 1-based position j (and j+1 if avg)
    equal q.  Yields dicts:
      {'crits': [(value, count), ...],   # adds pinned at critical values
       'lo_rng': (cmin, cmax),           # allowed count of adds strictly
       'v_lo_below': value}              # below `below` = min critical value
    Counts of pinned adds are enumerated 0..2 (position windows only ever
    need the value block widened by <= 2)."""
    outs = []
    addable = lambda v: _reg_valid(v, reg, score_hi)

    if not avg:
        # y_j = q
        for e in range(0, 3):
            if e and not addable(q):
                break
            lo_min = j - bc.cnt_le(q) - e
            lo_max = j - 1 - bc.cnt_lt(q)
            if lo_max < 0 or lo_min > k:
                continue
            outs.append({"crits": [(q, e)] if e else [],
                         "lo_rng": (max(0, lo_min), lo_max),
                         "below": q})
        return outs

    # avg: y_j + y_{j+1} = 2q
    # (a) both equal q
    for e in range(0, 3):
        if e and not addable(q):
            break
        lo_min = j + 1 - bc.cnt_le(q) - e
        lo_max = j - 1 - bc.cnt_lt(q)
        if lo_max < 0 or lo_min > k:
            continue
        outs.append({"crits": [(q, e)] if e else [],
                     "lo_rng": (max(0, lo_min), lo_max),
                     "below": q})
    # (b) straddle pairs (s, s') with s < q < s', s + s' = 2q, and no value
    #     strictly between s and s' in the merged array.
    s_cands = set()
    if _reg_valid(q - reg / 2, reg, score_hi):
        for m in range(0, 3):
            s_cands.add(q - reg / 2 - m * reg)
    i_hi = bisect_left(bc.dv, q)
    for i in range(max(0, i_hi - (k + 4)), i_hi):     # existing values < q
        s_cands.add(bc.dv[i])
    for s in sorted(s_cands, reverse=True):
        sp = 2 * q - s
        if s >= q or s < 0:
            continue
        s_exists = bc.cnt_le(s) - bc.cnt_lt(s) > 0
        sp_exists = bc.cnt_le(sp) - bc.cnt_lt(sp) > 0
        if bc.cnt_lt(sp) != bc.cnt_le(s):   # data strictly between s, s'
            continue
        for es in range(0, 3):
            if es and not addable(s):
                break
            if not (s_exists or es):
                continue
            for esp in range(0, 3):
                if esp and not addable(sp):
                    break
                if not (sp_exists or esp):
                    continue
                # cnt_le(s) must equal j exactly:
                c_lo = j - bc.cnt_le(s) - es
                if c_lo < 0 or c_lo > k:
                    continue
                if bc.cnt_lt(s) + c_lo > j - 1:
                    continue
                if bc.cnt_le(sp) + c_lo + es + esp < j + 1:
                    continue
                outs.append({"crits": [(s, es), (sp, esp)],
                             "lo_rng": (c_lo, c_lo),
                             "below": s})
    return outs


def solve_adds_k(bc, k, pub_lo, pub_hi, cap_lo, cap_hi, reg, score_hi,
                 max_witnesses=6):
    """All-additions witness multisets of size exactly k (list of lists of
    Fractions), exact-verified.  Returns up to max_witnesses distinct ones."""
    if k == 0:
        lo, hi = fences_capped(bc.vals, cap_lo, cap_hi)
        return [[]] if fences_match(lo, hi, pub_lo, pub_hi) else []
    pairs, (j1, j3, avg) = _q_pair_candidates(
        bc, k, pub_lo, pub_hi, cap_lo, cap_hi, reg)
    found = []
    seen = set()
    for q1, q3 in sorted(pairs):
        reals1 = _quartile_realizations(bc, q1, j1, avg, k, reg, score_hi)
        if not reals1:
            continue
        reals3 = _quartile_realizations(bc, q3, j3, avg, k, reg, score_hi)
        for r1 in reals1:
            crit1 = r1["crits"]
            n1 = sum(c for _, c in crit1)
            hi1_val = max([v for v, _ in crit1], default=q1)  # top of Q1 crit
            for r3 in reals3:
                crit3 = r3["crits"]
                n3 = sum(c for _, c in crit3)
                lo3_val = r3["below"]
                if hi1_val > lo3_val and q1 != q3:
                    continue
                base_used = n1 + n3
                if base_used > k:
                    continue
                # filler values
                v_lo = None
                b1 = r1["below"]
                for cand in (bc.vals[0], b1 - reg):
                    if 0 <= cand < b1 and _reg_valid(cand, reg, score_hi):
                        v_lo = cand
                        break
                v_mid = None
                m = (hi1_val / reg).__floor__() + 1
                while m * reg < lo3_val:
                    if _reg_valid(m * reg, reg, score_hi) and m * reg > hi1_val:
                        v_mid = m * reg
                        break
                    m += 1
                hi3_val = max([v for v, _ in crit3], default=q3)
                v_hi = None
                m = (hi3_val / reg).__floor__() + 1
                if _reg_valid(m * reg, reg, score_hi) and m * reg > hi3_val:
                    v_hi = m * reg
                # count of adds strictly below Q1-critical bottom
                c1min, c1max = r1["lo_rng"]
                if v_lo is None:
                    if c1min > 0:
                        continue
                    c1max = 0
                for c_lo in range(c1min, min(c1max, k - base_used) + 1):
                    # adds strictly below Q3 bottom = c_lo + crit1 + c_mid
                    c3min, c3max = r3["lo_rng"]
                    mid_min = c3min - c_lo - n1
                    mid_max = c3max - c_lo - n1
                    if v_mid is None:
                        if mid_min > 0:
                            continue
                        mid_max = min(mid_max, 0)
                    mid_lo = max(0, mid_min)
                    mid_hi = min(mid_max, k - base_used - c_lo)
                    for c_mid in range(mid_lo, mid_hi + 1):
                        c_hi = k - base_used - c_lo - c_mid
                        if c_hi < 0:
                            continue
                        if c_hi > 0 and v_hi is None:
                            continue
                        adds = ([v_lo] * c_lo +
                                [v for v, c in crit1 for _ in range(c)] +
                                [v_mid] * c_mid +
                                [v for v, c in crit3 for _ in range(c)] +
                                [v_hi] * c_hi)
                        adds.sort()
                        key = tuple(adds)
                        if key in seen:
                            continue
                        seen.add(key)
                        lo, hi = bc.fences_with(adds, cap_lo, cap_hi)
                        if fences_match(lo, hi, pub_lo, pub_hi):
                            # authoritative re-verify on materialized list
                            flo, fhi = fences_capped(
                                sorted(bc.vals + adds), cap_lo, cap_hi)
                            assert (flo, fhi) == (lo, hi)
                            found.append(adds)
                            if len(found) >= max_witnesses:
                                return found
    return found


def solve_target(vals_sorted, pub_lo, pub_hi, cap_lo, cap_hi, reg, score_hi,
                 k_cap=K_CAP, rem_limit=0, max_witnesses=6):
    """Escalating search.  Returns dict:
       {'add_only': {'k': k, 'witnesses': [...]}, or None,
        'with_rems': [{'rems': [...], 'k': k, 'witnesses': [...]}, ...]}"""
    bc = BaseCounts(vals_sorted)
    out = {"add_only": None, "with_rems": []}
    for k in range(0, k_cap + 1):
        ws = solve_adds_k(bc, k, pub_lo, pub_hi, cap_lo, cap_hi, reg,
                          score_hi, max_witnesses)
        if ws:
            out["add_only"] = {"k": k, "witnesses": ws}
            break
    if rem_limit:
        n = len(vals_sorted)
        n4 = math.ceil(n / 4)
        n34 = math.ceil(3 * n / 4)
        wnd = set()
        for j in (n4, n34):
            for i in range(max(0, j - rem_limit - 3),
                           min(n, j + rem_limit + 3)):
                wnd.add(vals_sorted[i])
        wnd = sorted(wnd)
        avail = Counter(vals_sorted)
        best_k = out["add_only"]["k"] if out["add_only"] else k_cap + 1
        for r in range(1, rem_limit + 1):
            n_r = 0                   # keep a few (min-k) options per r level
            homog = [(v,) * r for v in wnd]      # homogeneous sets first
            mixed = [c for c in combinations_with_replacement(wnd, r)
                     if c not in set(homog)]
            for rem in homog + mixed:
                cnt = Counter(rem)
                if any(cnt[v] > avail[v] for v in cnt):
                    continue
                reduced = list(vals_sorted)
                for v in rem:
                    reduced.remove(v)
                bcr = BaseCounts(reduced)
                for k in range(0, min(best_k - 1, k_cap) + 1):
                    ws = solve_adds_k(bcr, k, pub_lo, pub_hi, cap_lo, cap_hi,
                                      reg, score_hi, max_witnesses=3)
                    if ws:
                        out["with_rems"].append(
                            {"rems": list(rem), "k": k, "witnesses": ws})
                        n_r += 1
                        break
                if n_r >= 6:
                    break
    return out


# --------------------------------------------------------------------------
# per-target worker (spawn pool)
# --------------------------------------------------------------------------

def _search_worker(job):
    vals = sorted(F(s) for s in job["values"])
    res = solve_target(vals, job["pub_lo"], job["pub_hi"],
                       F(job["cap_lo"]) if job["cap_lo"] is not None else None,
                       F(job["cap_hi"]) if job["cap_hi"] is not None else None,
                       F(job["reg"]),
                       F(job["score_hi"]) if job["score_hi"] is not None else None,
                       k_cap=job.get("k_cap", K_CAP),
                       rem_limit=job.get("rem_limit", 0))
    ser = {"add_only": None, "with_rems": []}
    if res["add_only"]:
        ser["add_only"] = {"k": res["add_only"]["k"],
                           "witnesses": [[str(v) for v in w]
                                         for w in res["add_only"]["witnesses"]]}
    for wr in res["with_rems"]:
        ser["with_rems"].append({"rems": [str(v) for v in wr["rems"]],
                                 "k": wr["k"],
                                 "witnesses": [[str(v) for v in w]
                                               for w in wr["witnesses"]]})
    return {"mid": job["mid"], "org": job["org"], "year": job["year"],
            "res": ser}


def make_job(t, k_cap=K_CAP, rem_limit=0):
    return {"year": t["year"], "mid": t["mid"], "org": t["org"],
            "values": [str(v) for _, v in t["pairs"]],
            "pub_lo": t["pub_lo"], "pub_hi": t["pub_hi"],
            "cap_lo": str(t["cap_lo"]),
            "cap_hi": str(t["cap_hi"]) if t["cap_hi"] is not None else None,
            "reg": str(t["reg"]),
            "score_hi": str(t["score_hi"]) if t["score_hi"] is not None else None,
            "k_cap": k_cap, "rem_limit": rem_limit}


# --------------------------------------------------------------------------
# stacking / assembly
# --------------------------------------------------------------------------

def detect_families(targets):
    """(C-mid, D-mid) pairs coupled by name fragment (complaints, leaving,
    call center)."""
    byname = defaultdict(dict)
    for t in targets:
        for frag in FAMILY_FRAGMENTS:
            if frag in t.get("name", ""):
                byname[frag][t["mid"][0]] = t["mid"]
    return {frag: (d.get("C"), d.get("D")) for frag, d in byname.items()
            if "C" in d and "D" in d}


def verify_coupling(year, cmid, dmid):
    """Check in-data that C-side and D-side values are identical per contract."""
    sc = scores_by_measure(year, SCORE_VINTAGE[year])
    cd = {c: s for c, s in sc.get(cmid, [])}
    for c, s in sc.get(dmid, []):
        if c in cd and cd[c] != s:
            return False
    return True


def sub_multisets(values):
    """All sub-multisets of a small multiset, largest first, as sorted tuples."""
    cnt = sorted(Counter(values).items())
    def rec(i):
        if i == len(cnt):
            yield ()
            return
        v, c = cnt[i]
        for rest in rec(i + 1):
            for take in range(c, -1, -1):
                yield (v,) * take + rest
    subs = {tuple(sorted(s)) for s in rec(0)}
    return sorted(subs, key=lambda s: -len(s))


def target_key(t):
    return (t["mid"], t["org"])


def apply_delta_to_target(t, removed_cids, add_values):
    vals = sorted([v for c, v in t["pairs"] if c not in removed_cids] +
                  list(add_values))
    return fences_capped(vals, t["cap_lo"], t["cap_hi"])


def battery(targets, removed_cids, participation):
    """participation: {(mid, org): [Fraction add values]}.  Returns per-target
    after-status list and exact counts."""
    rows = []
    n_exact = 0
    n_eval = 0
    for t in targets:
        if t["status"] in ("no_data", "small_n"):
            rows.append({**{k: t[k] for k in ("mid", "org", "pub_lo", "pub_hi")},
                         "status": t["status"]})
            continue
        n_eval += 1
        adds = participation.get(target_key(t), [])
        lo, hi = apply_delta_to_target(t, removed_cids, adds)
        ok = fences_match(lo, hi, t["pub_lo"], t["pub_hi"])
        n_exact += ok
        rows.append({"mid": t["mid"], "org": t["org"],
                     "pub_lo": t["pub_lo"], "pub_hi": t["pub_hi"],
                     "before_exact": t["status"] == "exact",
                     "after_exact": ok,
                     "before": [str(x) for x in t["before"]],
                     "after": [str(lo), str(hi)]})
    return rows, n_exact, n_eval


def removal_is_globally_safe(targets, removed_cids):
    """Every exact target must stay exactly unchanged under the removals."""
    for t in targets:
        if t.get("status") != "exact":
            continue
        if not any(c in removed_cids for c, _ in t["pairs"]):
            continue
        lo, hi = apply_delta_to_target(t, removed_cids, [])
        if not fences_match(lo, hi, t["pub_lo"], t["pub_hi"]):
            return False, target_key(t)
    return True, None


def _select_removal_contracts(t, targets, rem_vals, already):
    """Pick real contracts realizing rem_vals in target t, jointly keeping
    every exact target of the year exactly unchanged.  Returns list or None."""
    pool = defaultdict(list)
    for c, v in t["pairs"]:
        pool[v].append(c)
    chosen = []
    for v, cnt in Counter(rem_vals).items():
        cands = [c for c in pool[v] if c not in already and c not in chosen]
        picked = 0
        for c in cands:
            trial = set(already) | set(chosen) | {c}
            ok, _ = removal_is_globally_safe(targets, trial)
            if ok:
                chosen.append(c)
                picked += 1
                if picked == cnt:
                    break
        if picked < cnt:
            return None
    return chosen


def assemble_year(year, targets, meta, search, log):
    """Build the delta set for one year from per-target search results.
    search: {(mid, org): result-dict from _search_worker}."""
    misses = [t for t in targets if t["status"] == "miss"]
    families = detect_families(targets)
    tmap = {target_key(t): t for t in targets}

    # ---- step 1: removal axis ----------------------------------------------
    # (a) targets with NO add-only solution at k <= K_CAP are forced onto the
    #     removal axis;  (b) the year's unique add-only argmax target may take
    #     a removal branch iff it strictly lowers  max_k + #removals.
    removed_cids = []
    removal_notes = []
    unsolved = []

    ao_k = {}
    for t in misses:
        res = search[target_key(t)]["res"]
        ao_k[target_key(t)] = res["add_only"]["k"] if res["add_only"] else None

    forced = [t for t in misses if ao_k[target_key(t)] is None]
    optional = []
    solved_ks = sorted((k for k in ao_k.values() if k is not None),
                       reverse=True)
    if len(solved_ks) >= 2 and solved_ks[0] > solved_ks[1]:
        argmax = [t for t in misses if ao_k[target_key(t)] == solved_ks[0]]
        if len(argmax) == 1 and search[target_key(argmax[0])]["res"]["with_rems"]:
            optional.append(argmax[0])

    for t in forced + optional:
        res = search[target_key(t)]["res"]
        if not res["with_rems"]:
            log(f"  UNSOLVED {t['mid']} {t['org']}: no solution "
                f"(k<={K_CAP}, r<={REM_CAP})")
            unsolved.append(target_key(t))
            continue
        # cost of an option = max(add count of the other targets, k) + r;
        # add-only (if it exists) competes with cost = max(B_others, k_ao).
        b_others = max((k for kk, k in ao_k.items()
                        if k is not None and kk != target_key(t)), default=0)
        opts = sorted(res["with_rems"],
                      key=lambda o: (max(b_others, o["k"]) + len(o["rems"]),
                                     len(o["rems"])))
        k_ao = ao_k[target_key(t)]
        if k_ao is not None and max(b_others, k_ao) <= \
                max(b_others, opts[0]["k"]) + len(opts[0]["rems"]):
            continue     # add-only is at least as cheap: no removal branch
        done = False
        for opt in opts[:16]:
            if k_ao is not None and \
                    max(b_others, opt["k"]) + len(opt["rems"]) >= \
                    max(b_others, k_ao):
                break    # remaining options cannot beat add-only
            rem_vals = [F(s) for s in opt["rems"]]
            chosen = _select_removal_contracts(t, targets, rem_vals,
                                               removed_cids)
            if chosen is not None:
                removed_cids.extend(chosen)
                removal_notes.append(
                    {"target": f"{t['mid']}/{t['org']}",
                     "axis": "forced" if t in forced else "cost",
                     "contracts": chosen,
                     "values": [str(v) for v in rem_vals],
                     "k_add_after": opt["k"]})
                log(f"  removals for {t['mid']} {t['org']} "
                    f"({'forced' if t in forced else 'cost'}): {chosen} "
                    f"(values {[str(v) for v in rem_vals]}) "
                    f"+ k={opt['k']} adds")
                done = True
                break
        if not done and k_ao is None:
            log(f"  removal selection FAILED for {t['mid']} {t['org']}: "
                f"no globally safe contract set")
            unsolved.append(target_key(t))
    removed_cids = list(dict.fromkeys(removed_cids))

    # ---- step 2: re-solve every miss on the removal-reduced base -----------
    final_adds = {}       # (mid, org) -> list of witness multisets (Fractions)
    for t in misses:
        vals = sorted(v for c, v in t["pairs"] if c not in removed_cids)
        bc = BaseCounts(vals)
        sols = None
        for k in range(0, K_CAP + 1):
            ws = solve_adds_k(bc, k, t["pub_lo"], t["pub_hi"], t["cap_lo"],
                              t["cap_hi"], t["reg"], t["score_hi"],
                              max_witnesses=8)
            if ws:
                sols = ws
                break
        if sols is None:
            log(f"  POST-REMOVAL unsolved: {t['mid']} {t['org']}")
            if target_key(t) not in unsolved:
                unsolved.append(target_key(t))
            continue
        final_adds[target_key(t)] = sols

    # ---- step 3: family coupling (D multiset must be sub-multiset of C) ----
    coupled_choice = {}   # (mid, org) -> chosen multiset (Fractions)
    coupling_report = []
    for frag, (cmid, dmid) in families.items():
        ck, dk = (cmid, "ALL"), (dmid, "MA-PD")
        c_miss, d_miss = ck in final_adds, dk in final_adds
        if not (c_miss or d_miss):
            continue
        if not verify_coupling(year, cmid, dmid):
            coupling_report.append({"family": frag, "status":
                                    "data not value-coupled; left free"})
            continue
        if c_miss and d_miss:
            tC, tD = tmap[ck], tmap[dk]
            valsD = sorted(v for c, v in tD["pairs"] if c not in removed_cids)
            bcD = BaseCounts(valsD)
            match = None
            for M in final_adds[ck]:
                for S in sub_multisets(M):
                    lo, hi = bcD.fences_with(sorted(S), tD["cap_lo"],
                                             tD["cap_hi"])
                    if fences_match(lo, hi, tD["pub_lo"], tD["pub_hi"]):
                        match = (M, list(S))
                        break
                if match:
                    break
            if match is None:
                # force-include: solve C row with D witness pre-added; keep
                # the S whose completion needs the fewest extra C-only adds
                tCvals = sorted(v for c, v in tC["pairs"]
                                if c not in removed_cids)
                best = None
                for S in final_adds[dk]:
                    bcC = BaseCounts(sorted(tCvals + S))
                    for k in range(0, K_CAP + 1 - len(S)):
                        if best is not None and k >= best[0]:
                            break
                        ws = solve_adds_k(bcC, k, tC["pub_lo"], tC["pub_hi"],
                                          tC["cap_lo"], tC["cap_hi"],
                                          tC["reg"], tC["score_hi"],
                                          max_witnesses=3)
                        if ws:
                            best = (k, sorted(S + ws[0]), list(S))
                            break
                if best is not None:
                    match = (best[1], best[2])
            if match:
                M, S = match
                coupled_choice[ck] = sorted(M)
                coupled_choice[dk] = sorted(S)
                coupling_report.append(
                    {"family": frag, "C": cmid, "D": dmid,
                     "C_multiset": [str(v) for v in sorted(M)],
                     "D_submultiset": [str(v) for v in sorted(S)],
                     "status": "coupled (D sub-multiset of C on shared "
                               "MA-PD adds)"})
            else:
                coupling_report.append(
                    {"family": frag, "status": "COUPLING FAILED; left free "
                     "(C and D witnesses independent)"})
        else:
            # one side misses, the other is exact: try NEUTRAL participation
            # of the miss-side witness values in the exact row.
            miss_key, exact_key = (ck, dk) if c_miss else (dk, ck)
            tE = tmap.get(exact_key)
            S = final_adds[miss_key][0]
            note = "partner row not evaluable"
            if tE and tE.get("status") == "exact":
                valsE = sorted(v for c, v in tE["pairs"]
                               if c not in removed_cids)
                bcE = BaseCounts(valsE)
                lo, hi = bcE.fences_with(sorted(S), tE["cap_lo"], tE["cap_hi"])
                if fences_match(lo, hi, tE["pub_lo"], tE["pub_hi"]):
                    coupled_choice[exact_key] = sorted(S)  # neutral
                    note = (f"{miss_key[0]} adds also participate in exact "
                            f"{exact_key[0]} row (values neutral there)")
                else:
                    note = (f"{miss_key[0]} adds do NOT participate in exact "
                            f"{exact_key[0]} row (values not neutral; "
                            f"participation freedom invoked)")
            coupling_report.append({"family": frag, "C": cmid, "D": dmid,
                                    "miss_multiset":
                                        [str(v) for v in sorted(S)],
                                    "status": note})

    # ---- step 4: final multiset per target, slot assignment ----------------
    participation = {}
    for key, sols in final_adds.items():
        participation[key] = coupled_choice.get(key, sols[0])
    for key, S in coupled_choice.items():
        if key not in participation:
            participation[key] = S            # neutral participation rows

    slot_mapd = defaultdict(dict)     # slot index -> {(mid,org): value}
    slot_maonly = defaultdict(dict)
    slot_pdp = defaultdict(dict)
    handled = set()

    for frag, (cmid, dmid) in families.items():
        ck, dk = (cmid, "ALL"), (dmid, "MA-PD")
        in_c, in_d = ck in participation, dk in participation
        if not (in_c or in_d):
            continue
        handled |= {ck, dk}
        if in_c and in_d and ck in coupled_choice and dk in coupled_choice:
            S = sorted(coupled_choice[dk])
            M = sorted(coupled_choice[ck])
            for i, v in enumerate(S):                 # shared MA-PD adds
                slot_mapd[i][dk] = v
                slot_mapd[i][ck] = v
            surplus = sorted((Counter(M) - Counter(S)).elements())
            for i, v in enumerate(surplus):           # C-only -> MA-only adds
                slot_maonly[i][ck] = v
        elif in_d:
            # D-row miss (C exact or decoupled): values on MA-PD slots;
            # neutral C participation where the values are neutral there.
            vals = sorted(participation[dk])
            for i, v in enumerate(vals):
                slot_mapd[i][dk] = v
                if ck in coupled_choice and i < len(coupled_choice[ck]):
                    slot_mapd[i][ck] = coupled_choice[ck][i]
            if in_c and dk not in coupled_choice:     # decoupled C side
                for i, v in enumerate(sorted(participation[ck])):
                    slot_maonly[i][ck] = v
        elif in_c:
            # C-row miss, D exact: reuse MA-PD slots; neutral D participation
            # where neutral there (else participation freedom, caveated).
            vals = sorted(participation[ck])
            for i, v in enumerate(vals):
                slot_mapd[i][ck] = v
                if dk in coupled_choice and i < len(coupled_choice[dk]):
                    slot_mapd[i][dk] = coupled_choice[dk][i]

    for key in sorted(participation):
        if key in handled:
            continue
        vals = sorted(participation[key])
        pool = slot_pdp if tmap[key]["org"] == "PDP" else slot_mapd
        for i, v in enumerate(vals):
            pool[i][key] = v

    # consistency: per-target multiset over slots == chosen participation
    slot_totals = defaultdict(list)
    for pool in (slot_mapd, slot_maonly, slot_pdp):
        for i in pool:
            for key, v in pool[i].items():
                slot_totals[key].append(v)
    for key, vals in participation.items():
        assert Counter(slot_totals[key]) == Counter(vals), \
            f"slot assignment inconsistent for {key}"

    adds_out = []
    for pool, cls, tag in ((slot_mapd, "MA-PD", "MAPD"),
                           (slot_maonly, "MA-only", "MAONLY"),
                           (slot_pdp, "PDP", "PDP")):
        for i in sorted(pool):
            adds_out.append({"id": f"ADD-{year}-{tag}-{i+1:02d}",
                             "org_class": cls,
                             "measures": {f"{k[0]}/{k[1]}": str(v)
                                          for k, v in sorted(pool[i].items())}})

    # ---- step 5: final full battery ----------------------------------------
    rows, n_exact, n_eval = battery(targets, set(removed_cids), participation)
    regressions = [r for r in rows if r.get("before_exact")
                   and not r.get("after_exact")]
    if regressions:
        log(f"  REGRESSIONS: {[(r['mid'], r['org']) for r in regressions]}")

    removed_out = []
    for c in removed_cids:
        mm = sorted(f"{t['mid']}/{t['org']}" for t in targets
                    if t["status"] not in ("no_data", "small_n")
                    and any(cc == c for cc, _ in t["pairs"]))
        removed_out.append({"contract_id": c,
                            "org_type": meta.get(c, ("?",))[0],
                            "reported_targets": mm})

    return {"adds": adds_out, "removals": removed_out,
            "removal_notes": removal_notes,
            "coupling": coupling_report,
            "unsolved": [f"{m}/{o}" for m, o in unsolved],
            "battery": rows,
            "exact_before": sum(1 for t in targets if t["status"] == "exact"),
            "exact_after": n_exact, "evaluable": n_eval}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    report = {}
    all_jobs, jobmap = [], {}
    year_data = {}
    for year in ("2024", "2025", "2026"):
        targets, meta = load_year(year)
        year_data[year] = (targets, meta)
        for t in targets:
            if t["status"] != "miss":
                continue
            job = make_job(t, k_cap=K_CAP, rem_limit=0)
            all_jobs.append(job)
            jobmap[(year, t["mid"], t["org"])] = job
        n_ev = sum(1 for t in targets if t["status"] in ("exact", "miss"))
        n_ex = sum(1 for t in targets if t["status"] == "exact")
        print(f"SY{year}: {n_ex}/{n_ev} exact before, "
              f"{n_ev - n_ex} misses, vintage={SCORE_VINTAGE[year]}")

    print(f"\nphase 1: add-only search on {len(all_jobs)} targets "
          f"({WORKERS} workers, spawn) ...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(WORKERS, len(all_jobs))) as pool:
        results = pool.map(_search_worker, all_jobs)
    search = {(r["year"], r["mid"], r["org"]): r for r in results}
    print("phase 1 done")
    for r in results:
        ao = r["res"]["add_only"]
        print(f"  {r['year']} {r['mid']:>4} {r['org']:>5}: "
              + (f"k={ao['k']} adds e.g. {ao['witnesses'][0]}"
                 if ao else "NO add-only solution"))

    # phase 2: removal search where add-only failed, plus the per-year unique
    # add-only argmax target (candidate for a cost-reducing removal branch)
    rem_keys = set()
    for r in results:
        if r["res"]["add_only"] is None:
            rem_keys.add((r["year"], r["mid"], r["org"]))
    for year in ("2024", "2025", "2026"):
        ks = sorted(((r["res"]["add_only"]["k"],
                      (r["year"], r["mid"], r["org"]))
                     for r in results if r["year"] == year
                     and r["res"]["add_only"] is not None), reverse=True)
        if len(ks) >= 2 and ks[0][0] > ks[1][0]:
            rem_keys.add(ks[0][1])
    rem_jobs = []
    for key in sorted(rem_keys):
        job = dict(jobmap[key])
        job["rem_limit"] = REM_CAP
        rem_jobs.append(job)
    if rem_jobs:
        print(f"\nphase 2: removal-assisted search on {len(rem_jobs)} "
              f"targets ...")
        with ctx.Pool(min(WORKERS, len(rem_jobs))) as pool:
            rres = pool.map(_search_worker, rem_jobs)
        for r in rres:
            search[(r["year"], r["mid"], r["org"])]["res"]["with_rems"] = \
                r["res"]["with_rems"]
            wr = r["res"]["with_rems"]
            print(f"  {r['year']} {r['mid']:>4} {r['org']:>5}: "
                  + (f"r={len(wr[0]['rems'])} rems {wr[0]['rems']} "
                     f"+ k={wr[0]['k']} adds" if wr else "STILL UNSOLVED"))

    # phase 3: per-year assembly + full battery
    print("\nphase 3: assembly ...")
    out = {"hypothesis": HYP, "score_vintage": SCORE_VINTAGE,
           "k_cap": K_CAP, "rem_cap": REM_CAP, "years": {}}
    for year in ("2024", "2025", "2026"):
        targets, meta = year_data[year]
        ysearch = {(m, o): search[(year, m, o)]
                   for (yy, m, o) in search if yy == year}
        print(f"\n=== SY{year} assembly ===")
        res = assemble_year(year, targets, meta, ysearch,
                            lambda s: print(s))
        out["years"][year] = res
        print(f"  SY{year}: exact {res['exact_before']}/{res['evaluable']} "
              f"-> {res['exact_after']}/{res['evaluable']};  "
              f"adds={len(res['adds'])} "
              f"({Counter(a['org_class'] for a in res['adds'])}), "
              f"removals={len(res['removals'])}")

    path = os.path.join(RUNS, "membership_delta_sets_docsonly.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
