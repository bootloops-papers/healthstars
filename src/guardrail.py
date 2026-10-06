# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Guardrail application (Tech Notes, 'Guardrails are then applied...', 2026 p.147;
same rule 2024/2025), with interval support for uncertain bases.

Rule per boundary (non-CAHPS, non-improvement, non-new measures):
  cap = 5 (percentage points)                     if measure is on a 0-100 scale
      = 0.05 * (prior-year trimmed max - min)     otherwise
        (trimmed = excluding the prior year's outliers beyond the outlier bounds)
  final = current            if |current - prior| <= cap
        = prior +/- cap      otherwise (toward current)

Register subtlety: CMS computes with >=6-digit internal precision; the public
record shows display-rounded priors. Where an internal prior is unavailable, we
carry an INTERVAL [prior_display - u, prior_display + u] (u = half display unit)
and propagate: final becomes an interval; a star flip claim survives only if it
holds for EVERY realization in the interval (exact and conservative).

Intervals are (lo, hi) Fraction pairs; scalars are Fractions treated as
point intervals (lo == hi).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.5, conventions (2)
    and (3), and Appendix C.4 (the guardrail, with interval priors).
Run:  cd src && python3 guardrail.py   (self-test; imported by the census scripts)
Requires: Python >= 3.10, standard library only.
"""

from fractions import Fraction
import sys


def _iv(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def guardrail_boundary(current, prior, cap):
    """Apply the guardrail to one boundary. All args scalar-or-interval.
    Returns interval (lo, hi) of possible final values."""
    clo, chi = _iv(current)
    plo, phi = _iv(prior)
    klo, khi = _iv(cap)
    finals = []
    # endpoint analysis: the map is piecewise monotone in (c, p, k); evaluating
    # on corner combinations bounds the range because for fixed (p, k) the map
    # c -> clamp(c, p-k, p+k) is monotone nondecreasing, and for fixed c it is
    # monotone in p and in k on each branch. Corners suffice.
    for c in (clo, chi):
        for p in (plo, phi):
            for k in (klo, khi):
                lo_b, hi_b = p - k, p + k
                f = c if lo_b <= c <= hi_b else (lo_b if c < lo_b else hi_b)
                finals.append(f)
    return (min(finals), max(finals))


def restricted_range_cap(prior_trimmed_scores):
    """cap = 0.05 * (max - min) of the prior-year clustering input after outlier trimming."""
    xs = [Fraction(s) if not isinstance(s, Fraction) else s
          for s in prior_trimmed_scores]
    return Fraction(5, 100) * (max(xs) - min(xs))


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    F = Fraction
    # slack: current within cap of prior -> final = current
    assert guardrail_boundary(F("0.11"), F("0.12"), F("0.05")) == (F("0.11"), F("0.11"))
    # bound: current far below prior -> final = prior - cap
    assert guardrail_boundary(F("0.60"), F("1.39"), F("0.05")) == (F("1.34"), F("1.34"))
    # interval prior (display-rounded base +/- 0.005), bound case
    lo, hi = guardrail_boundary(F("0.60"), (F("1.385"), F("1.395")), F("0.05"))
    assert (lo, hi) == (F("1.335"), F("1.345"))
    # interval cap
    lo, hi = guardrail_boundary(F("0.60"), F("1.39"), (F("0.0445"), F("0.055")))
    assert (lo, hi) == (F("1.335"), F("1.3455"))
    # scalar passthrough when unconstrained both sides
    assert guardrail_boundary(F("52"), F("50"), F("5")) == (F("52"), F("52"))
    assert guardrail_boundary(F("58"), F("50"), F("5")) == (F("55"), F("55"))
    print("guardrail: self-tests passed")
