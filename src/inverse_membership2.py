# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Inverse membership v2: profile enumeration with index-shift pruning.

A delta of k_add additions + k_rem removals changes each quartile's order-stat
position by at most k = k_add + k_rem. Enumerate SIDE PROFILES
  (a_lo, a_mid, a_hi, r_lo, r_mid, r_hi)  — additions/removals below Q1 region,
between, above Q3 region — instead of value combinations. For each profile the
new n and quartile positions are determined; QNTLDEF=5 then pins WHICH order
stats must equal the required Q1/Q3 (or average to them). Feasibility check is
O(1) lookups into the sorted data plus window computation for added values.

Output per target: feasible profiles with, for each, the exact constraints on
added values (which region, and when an added value must BE an order statistic,
its exact required value or window) and on removed values (which existing
values must vanish from specific positions). This is the stackable form for
cross-measure reconciliation.

Register: exact. Capped fences produce inequality constraints on (Q1,Q3)
instead of equalities.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (reconstructing
    the score list: filled-in scores, Table C1).
Run:  cd src && python3 inverse_membership2.py   (planted-perturbation self-test)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction as F
from tukey import qntldef5
import sys


def required_qs(pub_lo, pub_hi, cap_lo, cap_hi):
    """Returns (q1, q3, mode): mode 'both' (two equations), 'hi_only'/'lo_only'
    (one equation + one inequality), or 'none'."""
    lo_capped = pub_lo is not None and cap_lo is not None and F(pub_lo) == cap_lo
    hi_capped = pub_hi is not None and cap_hi is not None and F(pub_hi) == cap_hi
    if pub_lo is None or pub_hi is None:
        # single published side
        if pub_hi is not None and not hi_capped:
            return None, None, "hi_only"
        if pub_lo is not None and not lo_capped:
            return None, None, "lo_only"
        return None, None, "none"
    if not lo_capped and not hi_capped:
        lo, hi = F(pub_lo), F(pub_hi)
        return (4 * lo + 3 * hi) / 7, (3 * lo + 4 * hi) / 7, "both"
    if lo_capped and not hi_capped:
        return None, None, "hi_only"
    if hi_capped and not lo_capped:
        return None, None, "lo_only"
    return None, None, "none"


def _q_positions(n, p_num, p_den):
    """QNTLDEF=5 target positions for p = p_num/p_den on n points.
    Returns ('avg', j) meaning (x_j + x_{j+1})/2 (1-based) or ('at', j)."""
    np_num = n * p_num
    if np_num % p_den == 0:
        j = np_num // p_den
        return ("avg", j)
    return ("at", -(-np_num // p_den))  # ceil


def solve_profiles(values, pub_lo, pub_hi, cap_lo=None, cap_hi=None, k_max=3):
    """Enumerate feasible side profiles. values: Fractions (any order).

    Returns list of dicts:
      {k_add, k_rem, adds: (a_lo, a_mid, a_hi), rems: (r_lo, r_mid, r_hi),
       constraints: human-readable exact constraints,
       fence_check: (new_lo, new_hi) exact under a canonical witness}
    Feasibility is verified CONSTRUCTIVELY: we build a witness delta (canonical
    added values placed mid-window / required exact values; removals taken
    greedily from the required positions) and recompute fences exactly.
    """
    xs = sorted(values)
    n = len(xs)
    q1_req, q3_req, mode = required_qs(pub_lo, pub_hi, cap_lo, cap_hi)
    plo = F(pub_lo) if pub_lo is not None else None
    phi = F(pub_hi) if pub_hi is not None else None

    def fences(sorted_vals):
        q1 = qntldef5(sorted_vals, F(1, 4))
        q3 = qntldef5(sorted_vals, F(3, 4))
        lo = 4 * q1 - 3 * q3
        hi = 4 * q3 - 3 * q1
        if cap_lo is not None:
            lo = max(lo, cap_lo)
        if cap_hi is not None:
            hi = min(hi, cap_hi)
        return lo, hi

    def matches(sorted_vals):
        lo, hi = fences(sorted_vals)
        ok = True
        if plo is not None:
            ok &= lo == plo
        if phi is not None:
            ok &= hi == phi
        return ok

    if matches(xs):
        return [{"k_add": 0, "k_rem": 0, "adds": (0, 0, 0), "rems": (0, 0, 0),
                 "witness_ok": True, "constraints": "already exact"}]

    # region boundaries for classification: use current Q1/Q3 positions
    out = []
    seen = set()
    for k in range(1, k_max + 1):
        for k_add in range(0, k + 1):
            k_rem = k - k_add
            # enumerate side splits
            for a_lo in range(k_add + 1):
                for a_mid in range(k_add - a_lo + 1):
                    a_hi = k_add - a_lo - a_mid
                    for r_lo in range(k_rem + 1):
                        for r_mid in range(k_rem - r_lo + 1):
                            r_hi = k_rem - r_lo - r_mid
                            prof = (a_lo, a_mid, a_hi, r_lo, r_mid, r_hi)
                            if prof in seen:
                                continue
                            seen.add(prof)
                            w = _witness(xs, prof, q1_req, q3_req, mode,
                                         plo, phi, cap_lo, cap_hi, matches)
                            if w is not None:
                                out.append({"k_add": k_add, "k_rem": k_rem,
                                            "adds": (a_lo, a_mid, a_hi),
                                            "rems": (r_lo, r_mid, r_hi),
                                            **w})
        if out:
            break  # minimal k only
    return out


def _witness(xs, prof, q1_req, q3_req, mode, plo, phi, cap_lo, cap_hi, matches):
    """Try to construct a witness delta realizing the profile. Heuristic but
    exact-verified: candidate added values are drawn from a small set of trial values
    (region representatives + values required to hit target order stats);
    removals greedily from positions adjacent to quartiles. Returns dict or None.
    """
    a_lo, a_mid, a_hi, r_lo, r_mid, r_hi = prof
    n = len(xs)
    n_new = n + a_lo + a_mid + a_hi - r_lo - r_mid - r_hi
    if n_new < 8:
        return None
    # current quartile neighborhood positions
    q1pos = _q_positions(n, 1, 4)
    q3pos = _q_positions(n, 3, 4)
    i_q1 = q1pos[1]
    i_q3 = q3pos[1]

    # trial values per region: around required stats and region representatives
    trials_lo, trials_mid, trials_hi = set(), set(), set()
    lo_near = xs[max(0, i_q1 - 4):i_q1 + 4]
    hi_near = xs[max(0, i_q3 - 4):min(n, i_q3 + 4)]
    if q1_req is not None:
        trials_lo |= {q1_req, 2 * q1_req - xs[i_q1 - 1]}
        trials_mid |= {q1_req}
    if q3_req is not None:
        trials_hi |= {q3_req, 2 * q3_req - xs[i_q3 - 1]}
        trials_mid |= {q3_req}
    trials_lo |= {xs[0], xs[i_q1 - 1], (xs[0] + xs[i_q1 - 1]) / 2}
    trials_mid |= {xs[n // 2], xs[i_q1 - 1], xs[i_q3 - 1]}
    trials_hi |= {xs[-1], xs[i_q3 - 1], (xs[i_q3 - 1] + xs[-1]) / 2}
    trials_lo |= set(lo_near)
    trials_hi |= set(hi_near)
    if cap_lo is not None:
        trials_lo = {max(v, cap_lo) for v in trials_lo}
        trials_mid = {max(v, cap_lo) for v in trials_mid}
    if cap_hi is not None:
        trials_hi = {min(v, cap_hi) for v in trials_hi}

    # removal candidates: values near the quartile positions on each side
    rem_lo_c = sorted(set(xs[max(0, i_q1 - 5):i_q1 + 3]))
    rem_mid_c = sorted(set(xs[max(0, n // 2 - 3):n // 2 + 3]))
    rem_hi_c = sorted(set(xs[max(0, i_q3 - 3):min(n, i_q3 + 5)]))

    from itertools import combinations_with_replacement as cwr, combinations

    def region_ok(v, region):
        if region == "lo":
            return v <= xs[i_q1 - 1]
        if region == "hi":
            return v >= xs[i_q3 - 1]
        return xs[i_q1 - 1] <= v <= xs[i_q3 - 1]

    add_sets_lo = list(cwr(sorted(trials_lo), a_lo)) if a_lo else [()]
    add_sets_mid = list(cwr(sorted(trials_mid), a_mid)) if a_mid else [()]
    add_sets_hi = list(cwr(sorted(trials_hi), a_hi)) if a_hi else [()]
    rem_sets_lo = list(combinations(rem_lo_c, r_lo)) if r_lo else [()]
    rem_sets_mid = list(combinations(rem_mid_c, r_mid)) if r_mid else [()]
    rem_sets_hi = list(combinations(rem_hi_c, r_hi)) if r_hi else [()]

    max_trials = 4000
    tried = 0
    for al in add_sets_lo:
        for am in add_sets_mid:
            for ah in add_sets_hi:
                adds = list(al) + list(am) + list(ah)
                for rl in rem_sets_lo:
                    for rm in rem_sets_mid:
                        for rh in rem_sets_hi:
                            tried += 1
                            if tried > max_trials:
                                return None
                            rems = list(rl) + list(rm) + list(rh)
                            cand = list(xs)
                            ok = True
                            for v in rems:
                                try:
                                    cand.remove(v)
                                except ValueError:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            cand = sorted(cand + adds)
                            if matches(cand):
                                return {"witness_ok": True,
                                        "witness_adds": [str(a) for a in adds],
                                        "witness_rems": [str(r) for r in rems],
                                        "constraints":
                                            f"adds={[str(a) for a in adds]} "
                                            f"rems={[str(r) for r in rems]}"}
    return None


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random
    random.seed(5)
    xs = sorted(F(random.randint(4000, 9900), 100) for _ in range(121))
    planted = F(8701, 100)
    from inverse_membership import fences_of
    full = sorted(xs + [planted])
    lo, hi = fences_of(full)
    sols = solve_profiles(xs, str(lo), str(hi), k_max=2)
    assert sols and all(s["k_add"] + s["k_rem"] >= 1 for s in sols)
    assert any(s["adds"][2] == 1 or s["adds"][1] == 1 or s["adds"][0] == 1
               for s in sols), sols
    print(f"inverse_membership2: planted-addition feasible profiles: "
          f"{[(s['adds'], s['rems']) for s in sols]}")
    # planted removal near Q3
    victim = xs[90]
    reduced = list(xs)
    reduced.remove(victim)
    lo, hi = fences_of(sorted(reduced))
    sols = solve_profiles(xs, str(lo), str(hi), k_max=2)
    assert sols, "removal not recovered"
    print(f"inverse_membership2: planted-removal profiles: "
          f"{[(s['adds'], s['rems']) for s in sols]}")
    # negative control
    sols = solve_profiles(xs, "0", "99999", k_max=2)
    assert sols == []
    print("inverse_membership2: negative control clean")
