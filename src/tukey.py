# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Outlier deletion by the outlier bounds, CMS star-ratings convention, exact arithmetic.

Tech Notes (2024 final, p.149-150; identical language 2025/2026): outlier bounds =
Q1 - 3.0*IQR, Q3 + 3.0*IQR; quartiles "can be obtained by using the MEANS
procedure in SAS"; outliers removed BEFORE mean resampling; cutoffs capped at
[0,100] for percent-no-decimal measures displayed 0-100, and at any measure
range restriction (e.g. lower bound 0).

SAS PROC MEANS default quartile definition is QNTLDEF=5 (empirical distribution
function with averaging): with sorted x_(1..n) and target p, let np = n*p,
j = floor(np). If np is an integer, Q = (x_(j) + x_(j+1))/2; else Q = x_(j+1).
Implemented in exact Fraction arithmetic on decimal-string scores, so the bound
comparison (score strictly outside [lo, hi] -> deleted) is exact. "Outside the
bounds" reads as strict inequality: boundary-equal scores are retained.

Validation target: the Tech Notes publish the per-measure outlier-bound cutoffs
(Tables K-5/K-6 in 2024); tukey_battery.py compares the computed bounds to the
published ones per measure-year.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 and Appendix C.2
    (the outlier-bounds trim with SAS's default quartile definition).
Run:  cd src && python3 tukey.py   (self-test; imported by the census scripts)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction
from dp_kmeans import _to_fractions
import sys


def qntldef5(sorted_x, p):
    """SAS QNTLDEF=5 quantile of pre-sorted Fractions, p as Fraction."""
    n = len(sorted_x)
    np_ = n * p
    j = np_.numerator // np_.denominator  # floor
    if np_ == j:  # integer
        if j == 0:
            return sorted_x[0]
        if j >= n:
            return sorted_x[-1]
        return (sorted_x[j - 1] + sorted_x[j]) / 2
    k = j + 1  # ceil(np)
    k = min(max(k, 1), n)
    return sorted_x[k - 1]


def tukey_fences(scores, cap_lo=None, cap_hi=None, mult=Fraction(3)):
    """Outlier-bound cutoffs (exact). scores: decimal strings/ints/Fractions.

    cap_lo/cap_hi: measure range restrictions (e.g. 0 and 100 for percent
    no-decimal 0-100 displays; 0 alone for nonnegative rates), as parseable
    decimal strings or None.
    Returns (lo, hi) Fractions.
    """
    xs = sorted(_to_fractions(scores))
    q1 = qntldef5(xs, Fraction(1, 4))
    q3 = qntldef5(xs, Fraction(3, 4))
    iqr = q3 - q1
    lo = q1 - mult * iqr
    hi = q3 + mult * iqr
    if cap_lo is not None:
        lo = max(lo, Fraction(str(cap_lo)))
    if cap_hi is not None:
        hi = min(hi, Fraction(str(cap_hi)))
    return lo, hi


def tukey_trim(scores, cap_lo=None, cap_hi=None, mult=Fraction(3)):
    """Return (kept_mask, lo, hi): mask parallel to input order; strict-outside deleted."""
    xs = _to_fractions(scores)
    lo, hi = tukey_fences(xs, cap_lo, cap_hi, mult)
    mask = [not (x < lo or x > hi) for x in xs]
    return mask, lo, hi


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    # QNTLDEF=5 spot checks (hand-computed):
    xs = _to_fractions(["1", "2", "3", "4"])  # n=4, p=.25 -> np=1 integer -> (x1+x2)/2
    assert qntldef5(sorted(xs), Fraction(1, 4)) == Fraction(3, 2)
    assert qntldef5(sorted(xs), Fraction(3, 4)) == Fraction(7, 2)
    xs = _to_fractions(["1", "2", "3", "4", "5"])  # n=5, np=1.25 -> ceil=2 -> x2
    assert qntldef5(sorted(xs), Fraction(1, 4)) == 2
    assert qntldef5(sorted(xs), Fraction(3, 4)) == 4
    # bounds + caps + strict-outside retention semantics
    scores = ["10"] * 3 + ["50"] * 94 + ["90"] * 3
    mask, lo, hi = tukey_trim(scores, cap_lo="0", cap_hi="100")
    assert lo == 50 and hi == 50  # IQR=0 -> bounds collapse to Q1=Q3=50
    assert sum(mask) == 94  # boundary-equal (=50) retained, 10s and 90s deleted
    print("tukey: self-tests passed")
