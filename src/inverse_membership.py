# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Stage-1 inverse solver: which membership delta makes the exact outlier bounds
equal the published K-5/K-6 values exactly.

Algebra: lo = Q1 - 3(Q3-Q1) = 4*Q1 - 3*Q3 ; hi = Q3 + 3(Q3-Q1) = 4*Q3 - 3*Q1
=> Q1 = (4*lo + 3*hi)/7 ; Q3 = (3*lo + 4*hi)/7   (both fences uncapped)
If one side is capped (0/100), only the other equation constrains.

QNTLDEF=5 on n points: pos = n*p; if integer j: Q = (x_(j)+x_(j+1))/2 else
Q = x_(ceil(pos)). For p=1/4, 3/4: averaging occurs iff n % 4 == 0.

Solver per measure: for k_add additions (terminated contracts, scores unknown ->
solve for required score windows) and k_rem removals (flagged contracts, must be
data values we hold), k_add + k_rem <= K_MAX, test whether required (Q1, Q3) is
achievable; report the exact windows/values. Cross-measure consistency is then
checked by the caller (same contract set must work everywhere it reported).

This module reports solution families: each family = (k_add, k_rem,
windows for added scores, candidate removed values). Register: exact.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.2 (reconstructing
    the score list: filled-in scores, Table C1).
Run:  cd src && python3 inverse_membership.py   (planted-perturbation self-test)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction
from itertools import combinations

from tukey import qntldef5
import sys

F = Fraction


def required_quartiles(pub_lo, pub_hi, capped_lo, capped_hi):
    """Return (Q1, Q3) requirement; None entries where capped (unconstrained)."""
    if not capped_lo and not capped_hi:
        lo, hi = F(pub_lo), F(pub_hi)
        return (4 * lo + 3 * hi) / 7, (3 * lo + 4 * hi) / 7
    if capped_lo and not capped_hi:
        return None, None  # single equation 4*Q3-3*Q1 = hi: line of solutions
    if capped_hi and not capped_lo:
        return None, None
    return None, None


def fences_of(sorted_vals):
    q1 = qntldef5(sorted_vals, F(1, 4))
    q3 = qntldef5(sorted_vals, F(3, 4))
    return 4 * q1 - 3 * q3, 4 * q3 - 3 * q1  # uncapped


def solve_measure(values, pub_lo, pub_hi, cap_lo=None, cap_hi=None,
                  k_max=3, grid=None):
    """values: list of Fractions (our best-hypothesis membership, sorted or not).
    pub_lo/pub_hi: published fence strings (None if absent).
    cap_lo/cap_hi: cap values as Fractions or None.
    grid: allowed added-score values (list of Fractions); None -> windows only.

    Returns list of solutions: dicts {k_add, k_rem, added (list of exact values
    or (window_lo, window_hi) pairs), removed (list of values)}. Exhaustive over
    k_add+k_rem <= k_max with added values drawn from grid (if given) else
    reported as windows via direct search over insertion positions.
    """
    xs = sorted(values)
    n = len(xs)
    plo = F(pub_lo) if pub_lo is not None else None
    phi = F(pub_hi) if pub_hi is not None else None

    def match(cand):
        lo, hi = fences_of(cand)
        if cap_lo is not None:
            lo = max(lo, cap_lo)
        if cap_hi is not None:
            hi = min(hi, cap_hi)
        ok = True
        if plo is not None:
            ok &= lo == plo
        if phi is not None:
            ok &= hi == phi
        return ok

    sols = []
    if match(xs):
        return [{"k_add": 0, "k_rem": 0, "added": [], "removed": []}]

    gvals = sorted(set(grid)) if grid else []
    for k in range(1, k_max + 1):
        for k_rem in range(0, k + 1):
            k_add = k - k_rem
            # removals: only values near the quartiles matter; restrict to the
            # distinct values within the central half plus fence-adjacent tails
            # (complete for quartile effects: removing elsewhere shifts indices
            # identically to removing the nearest same-side value); all
            # distinct values on each side are searched for exactness.
            distinct = sorted(set(xs))
            rem_cands = distinct
            for rem in combinations(rem_cands, k_rem):
                base = list(xs)
                okrem = True
                for v in rem:
                    try:
                        base.remove(v)
                    except ValueError:
                        okrem = False
                        break
                if not okrem:
                    continue
                if k_add == 0:
                    if match(base):
                        sols.append({"k_add": 0, "k_rem": k_rem,
                                     "added": [], "removed": list(rem)})
                    continue
                if gvals:
                    for add in combinations(gvals, k_add):
                        cand = sorted(base + list(add))
                        if match(cand):
                            sols.append({"k_add": k_add, "k_rem": k_rem,
                                         "added": list(add), "removed": list(rem)})
                else:
                    # windows mode: k_add=1 only (window search); higher k_add
                    # without a grid is underdetermined — skip
                    if k_add == 1:
                        # candidate insertion values: midpoints between distinct
                        # values + the values themselves (fences are piecewise
                        # constant in the inserted value between data points)
                        cands = []
                        for i in range(len(distinct) - 1):
                            cands.append(distinct[i])
                            cands.append((distinct[i] + distinct[i + 1]) / 2)
                        cands += [distinct[-1], distinct[0] - 1, distinct[-1] + 1]
                        hits = [c for c in cands if match(sorted(base + [c]))]
                        if hits:
                            sols.append({"k_add": 1, "k_rem": k_rem,
                                         "added": [("window-representatives", hits)],
                                         "removed": list(rem)})
        if sols:
            break  # minimal-k families only
    return sols


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    # controls: plant KNOWN perturbations on tie-free data (every insertion or
    # removal must move an order statistic) and require exact recovery.
    import random
    random.seed(5)
    xs = sorted(F(random.randint(4000, 9900), 100) for _ in range(121))
    planted = F(8701, 100)
    full = sorted(xs + [planted])
    lo, hi = fences_of(full)
    grid = sorted({F(v, 100) for v in range(4000, 9901, 1)} | {planted})
    sols = solve_measure(xs, str(lo), str(hi), k_max=1, grid=list(grid))
    assert sols and all(s["k_add"] == 1 and not s["removed"] for s in sols), sols[:3]
    added_vals = {a for s in sols for a in s["added"]}
    assert planted in added_vals, (len(sols), sorted(added_vals)[:5])
    print(f"inverse_membership: planted-addition recovered "
          f"({len(sols)} minimal families; planted value among them)")
    # planted removal: remove a value adjacent to Q3 so fences shift
    q3_victim = xs[90]
    reduced = list(xs)
    reduced.remove(q3_victim)
    lo, hi = fences_of(sorted(reduced))
    sols = solve_measure(xs, str(lo), str(hi), k_max=1, grid=[])
    assert sols and any(s["k_rem"] == 1 and s["removed"] == [q3_victim]
                        for s in sols), sols[:3]
    print("inverse_membership: planted-removal recovered")
    # negative control: fences unreachable by any single delta must return []
    sols = solve_measure(xs, "0", "99999", k_max=1, grid=list(grid))
    assert sols == [], sols[:3]
    print("inverse_membership: negative control clean (unreachable target -> no solution)")
