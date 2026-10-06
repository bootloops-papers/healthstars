#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""stars-evaluate.py — recompute the Medicare Advantage cut-point census.

Reads ma-cutpoint-rows.json (beside this script): the 123 clusterable
measure-year(-org) rows of the paper's exact optimality-gap table, each
carrying its sorted score list (after the outlier bounds) as exact decimal
strings, the published cut points, the cut points at the exact optimum, and
the recorded objective values as explicit fractions.

For every row this script, in exact rational arithmetic (no floating point
anywhere in the objective):
  1. scores the PUBLISHED cut points on the score list -> within-cluster sum
     of squares as an explicit fraction, asserted equal to the recorded
     published-grouping value (ward_ssq);
  2. runs Fisher's dynamic program for the globally optimal k=5 grouping
     (Wang & Song 2011 prove the recurrence exact in 1-D) -> optimal SSQ,
     asserted equal to the recorded dp_ssq, and counts ALL optimal groupings,
     asserting the optimum is unique (dp_n_optima == 1);
  3. checks gap = ward_ssq - dp_ssq against the recorded fraction and the
     off-optimum verdict (gap > 0);
  4. recounts the star-assignment delta census between the two cut-point
     sets, asserted equal to the recorded n_star_delta_fullsample.

Default run: the 43 star-year-2026 rows (the most recent star year). --full
runs all 123 rows and additionally asserts the census tallies: 104 of 123
rows off the exact optimum, 120 of 120 rows with five or more distinct scores
having a unique optimum, 47,752 scores in the score lists, and 9,809
measure-level star assignments moved between the two groupings.

Rows with fewer than five distinct scores (3 of the 123): the score list
after the outlier bounds carries fewer than five distinct score values, so no
five-level grouping exists and the row carries k < 5 with k - 1 cut points
(none when k = 1), zero objective values and the on-optimum verdict. For such
a row checks 1-3 are skipped; instead k is asserted equal to the number of
distinct score values, both cut-point lists must hold k - 1 entries, the
recorded gap must be 0 with off_optimum false and one optimum, and the
star-delta recount (4) still runs. Such a row therefore never counts as
off-optimum or as non-unique.

Run:  python3 stars-evaluate.py            # the 43 star-year-2026 rows
      python3 stars-evaluate.py --full     # all 123 rows + census tallies
      python3 stars-evaluate.py --check    # mutation control (must fail)
Exit code 0 only if every check agrees (or, under --check, the control
fired); 1 otherwise; 2 when ma-cutpoint-rows.json is not beside the script
or does not hold 123 rows.
Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering"
    (M. D. Schwartz, 2026): the optimality-gap table (104 of 123 cut-point
    computations off the exact minimum; 120 of 120 rows with five or more
    distinct scores have a unique optimum; 9,809 star assignments moved).
Requires: Python >= 3.10, standard library only.

--check runs a mutation control that must FAIL: one off-optimum row's
published cut points are replaced by its optimal ones, and the recorded
verdict for that row has to break (the row scores as on-optimum). If the
row still matches its recorded values, the control did not fire and the
script exits 1.
"""

import argparse
import json
import os
import sys
from fractions import Fraction as F

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_JSON = os.path.join(HERE, "ma-cutpoint-rows.json")

DEFAULT_YEAR = "2026"   # default subset: the star-year-2026 rows

# ---------------------------------------------------------------- exact DP
#
# Values are scaled to integers (the samples are decimal strings of bounded
# precision), so every block cost is the integer pair
#     cost(i, j) = (c * Q - S^2, c),   c = block size,
# in units of 10^(-2d).  DP values are kept as unreduced (num, den) integer
# pairs — denominators are products of at most k block sizes, so they stay
# below n^k and every comparison is one integer cross-multiplication.


def dp_optimal(xints, k):
    """Fisher's DP on sorted integers. Returns (ssq_num, ssq_den, n_optima,
    boundaries of one optimal grouping). Exact; counts ALL optimal groupings
    by argmin-path multiplicity (each distinct boundary sequence is one
    grouping)."""
    n = len(xints)
    ps = [0] * (n + 1)
    qs = [0] * (n + 1)
    for i, x in enumerate(xints):
        ps[i + 1] = ps[i] + x
        qs[i + 1] = qs[i] + x * x
    # level m=1: one block covering [0:j]
    num = [0] * (n + 1)
    den = [1] * (n + 1)
    cnt = [1] * (n + 1)
    for j in range(1, n + 1):
        s = ps[j]
        num[j] = j * qs[j] - s * s
        den[j] = j
    prev_num, prev_den, prev_cnt = num, den, cnt
    backs = []
    for m in range(2, k + 1):
        cur_num = [0] * (n + 1)
        cur_den = [1] * (n + 1)
        cur_cnt = [0] * (n + 1)
        cur_back = [0] * (n + 1)
        for j in range(m, n + 1):
            bn = None
            bd = 1
            bc = 0
            bi = -1
            pj = ps[j]
            qj = qs[j]
            for i in range(m - 1, j):
                pn = prev_num[i]
                pd = prev_den[i]
                c = j - i
                s = pj - ps[i]
                costn = c * (qj - qs[i]) - s * s
                cn = pn * c + costn * pd
                cd = pd * c
                if bn is None:
                    bn, bd, bc, bi = cn, cd, prev_cnt[i], i
                else:
                    lhs = cn * bd
                    rhs = bn * cd
                    if lhs < rhs:
                        bn, bd, bc, bi = cn, cd, prev_cnt[i], i
                    elif lhs == rhs:
                        bc += prev_cnt[i]
            cur_num[j] = bn
            cur_den[j] = bd
            cur_cnt[j] = bc
            cur_back[j] = bi
        prev_num, prev_den, prev_cnt = cur_num, cur_den, cur_cnt
        backs.append(cur_back)
    # recover one optimal boundary set (indices into the sorted array)
    bounds = [n]
    j = n
    for m in range(k, 1, -1):
        j = backs[m - 2][j]
        bounds.append(j)
    bounds.append(0)
    bounds.reverse()
    return prev_num[n], prev_den[n], prev_cnt[n], bounds


def star_of(cps_scaled, x, higher):
    """Star level of scaled score x under scaled cut points (the data file's
    convention: higher-is-better star = 1 + #{cut points <= x}; lower-is-
    better star = 5 - #{cut points < x})."""
    if higher:
        return 1 + sum(1 for t in cps_scaled if x >= t)
    return 5 - sum(1 for t in cps_scaled if x > t)


def score_cutpoints(xints, cps_scaled, higher):
    """Exact SSQ (num, den in scaled^2 units) of the grouping induced by the
    cut points, plus the number of nonempty groups."""
    groups = {}
    for x in xints:
        groups.setdefault(star_of(cps_scaled, x, higher), []).append(x)
    num = 0
    den = 1
    for g in groups.values():
        c = len(g)
        s = sum(g)
        q = sum(x * x for x in g)
        gn = c * q - s * s
        num = num * c + gn * den
        den = den * c
    return num, den, len(groups)


def check_row(r, mutate=False):
    """Recompute one row against its recorded values. Returns a list of
    disagreement strings (empty = row fully reproduced)."""
    bad = []
    sample = r["sample"]
    if len(sample) != r["n_trimmed"]:
        bad.append("sample size %d != n_trimmed %d" % (len(sample), r["n_trimmed"]))
        return bad
    fr = [F(s) for s in sample]
    if fr != sorted(fr):
        bad.append("sample not sorted")
        return bad
    # common decimal register -> integers
    dmax = max((len(s.split(".")[1]) if "." in s else 0) for s in sample)
    scale = 10 ** dmax
    xints = [int(v * scale) for v in fr]
    if any(F(x, scale) != v for x, v in zip(xints, fr)):
        bad.append("sample not exactly decimal at %d places" % dmax)
        return bad
    higher = r["higher_is_better"]
    k = r["k"]
    wcps = [F(c) * scale for c in r["ward_cutpoints"]]
    if mutate:
        wcps = [F(c) * scale for c in r["dp_cutpoints"]]
    dcps = [F(c) * scale for c in r["dp_cutpoints"]]

    if k < 5:
        # Fewer than five distinct score values after the outlier bounds,
        # so no five-level grouping exists. Checks 1-3 (SSQ, DP, gap) are
        # skipped; k is asserted against the score list instead, both
        # cut-point lists must hold k - 1 entries, and the row must carry
        # gap 0, off_optimum false and one optimum. The star-delta recount
        # (check 4) still runs below.
        ndistinct = len(set(fr))
        print("       fewer than five distinct scores: %s %s %s (%s): %d distinct score values, "
              "%d cut points; k = %d asserted, SSQ/DP checks skipped"
              % (r["year"], r["mid"], r["org"], r["name"], ndistinct,
                 len(r["ward_cutpoints"]), k))
        if ndistinct != k:
            bad.append("row with fewer than five distinct scores: %d distinct score values != k = %d"
                       % (ndistinct, k))
        if len(r["ward_cutpoints"]) != k - 1 or len(r["dp_cutpoints"]) != k - 1:
            bad.append("row with fewer than five distinct scores: cut-point lists hold %d / %d entries, "
                       "k - 1 = %d" % (len(r["ward_cutpoints"]),
                                        len(r["dp_cutpoints"]), k - 1))
        if F(r["gap"]) != 0 or r["off_optimum"] or r["dp_n_optima"] != 1:
            bad.append("row with fewer than five distinct scores: recorded gap %s / off_optimum %s / "
                       "dp_n_optima %s, must be 0 / False / 1"
                       % (r["gap"], r["off_optimum"], r["dp_n_optima"]))
        ndelta = sum(1 for x in xints
                     if star_of(wcps, x, higher) != star_of(dcps, x, higher))
        if ndelta != r["n_star_delta_fullsample"]:
            bad.append("star-delta census %d != recorded %d"
                       % (ndelta, r["n_star_delta_fullsample"]))
        return bad

    unscale = F(1, scale * scale)
    wn, wd, wk = score_cutpoints(xints, wcps, higher)
    ward_ssq = F(wn, wd) * unscale
    if ward_ssq != F(r["ward_ssq"]):
        bad.append("published-grouping SSQ %s != recorded ward_ssq %s"
                   % (ward_ssq, r["ward_ssq"]))
    if wk != k:
        bad.append("published cut points induce %d groups, k = %d" % (wk, k))

    on_, od, nopt, _bounds = dp_optimal(xints, k)
    dp_ssq = F(on_, od) * unscale
    if dp_ssq != F(r["dp_ssq"]):
        bad.append("DP optimal SSQ %s != recorded dp_ssq %s" % (dp_ssq, r["dp_ssq"]))
    if nopt != r["dp_n_optima"] or nopt != 1:
        bad.append("DP optimum multiplicity %s != recorded dp_n_optima %s (must be 1)"
                   % (nopt, r["dp_n_optima"]))
    # the recorded optimal cut points must score exactly the optimum
    dn, dd, _dk = score_cutpoints(xints, dcps, higher)
    if F(dn, dd) * unscale != dp_ssq:
        bad.append("recorded dp_cutpoints do not score the DP optimum")

    gap = ward_ssq - dp_ssq
    if gap != F(r["gap"]):
        bad.append("gap %s != recorded %s" % (gap, r["gap"]))
    if (gap > 0) != r["off_optimum"]:
        bad.append("off_optimum verdict %s != recorded %s" % (gap > 0, r["off_optimum"]))

    ndelta = sum(1 for x in xints
                 if star_of(wcps, x, higher) != star_of(dcps, x, higher))
    if ndelta != r["n_star_delta_fullsample"]:
        bad.append("star-delta census %d != recorded %d"
                   % (ndelta, r["n_star_delta_fullsample"]))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="all 123 rows + census tallies (default: the 43 "
                         "star-year-%s rows)" % DEFAULT_YEAR)
    ap.add_argument("--check", action="store_true",
                    help="mutation control: swap one off-optimum row's published "
                         "cut points for its optimal ones; the row check MUST fail")
    args = ap.parse_args()

    print("stars-evaluate: exact recomputation of the MA cut-point census"
          + (" (--check control)" if args.check else
             " (all 123 rows)" if args.full else
             " (default subset: star-year-%s rows)" % DEFAULT_YEAR))
    doc = json.load(open(ROWS_JSON))
    rows = doc["rows"]
    if len(rows) != 123:
        print("FAIL: expected 123 rows, found %d" % len(rows))
        return 2

    if args.check:
        target = next(r for r in rows if r["off_optimum"])
        print("control: row %s %s %s (%s) — published cut points replaced by "
              "the optimal ones; its recorded verdict must now break"
              % (target["year"], target["mid"], target["org"], target["name"]))
        bad = check_row(target, mutate=True)
        if bad:
            for b in bad:
                print("  broke as required:", b)
            print("CONTROL FIRED as required (%d disagreements) — PASS" % len(bad))
            return 0
        print("CONTROL DID NOT FIRE: mutated row still matches its recorded values — FAIL")
        return 1

    todo = rows if args.full else [r for r in rows if r["year"] == DEFAULT_YEAR]
    print("rows to recompute: %d; score lists carry %d scores"
          % (len(todo), sum(r["n_trimmed"] for r in todo)))
    nbad = 0
    for i, r in enumerate(todo):
        bad = check_row(r)
        tag = "ok" if not bad else "DISAGREES"
        if bad or (i + 1) % 10 == 0 or i + 1 == len(todo):
            print("  [%3d/%d] %s %s %-5s n=%-3d gap=%s %s"
                  % (i + 1, len(todo), r["year"], r["mid"], r["org"],
                     r["n_trimmed"], r["gap"][:24] + ("…" if len(r["gap"]) > 24 else ""),
                     tag))
        for b in bad:
            print("      ", b)
        nbad += bool(bad)
    if nbad:
        print("FAIL: %d of %d rows disagree with the recorded values" % (nbad, len(todo)))
        return 1

    n_deg = sum(1 for r in todo if r["k"] < 5)
    print("all %d rows reproduced exactly: per-row published-grouping SSQ, "
          "optimal SSQ, gap, uniqueness, star-delta census (%d rows with fewer "
          "than five distinct scores: k, cut-point count, zero gap, star-delta census)"
          % (len(todo), n_deg))
    n_off = sum(1 for r in todo if r["off_optimum"])
    n_nondeg = len(todo) - n_deg
    n_uni = sum(1 for r in todo if r["k"] >= 5 and r["dp_n_optima"] == 1)
    n_sc = sum(r["n_trimmed"] for r in todo)
    n_mv = sum(r["n_star_delta_fullsample"] for r in todo)
    print("subset tallies: %d/%d off-optimum, %d/%d rows with five or more distinct "
          "scores have a unique optimum, %s scores, %s star assignments moved"
          % (n_off, len(todo), n_uni, n_nondeg,
             format(n_sc, ","), format(n_mv, ",")))
    if args.full:
        ok = (n_off == 104 and n_nondeg == 120 and n_uni == 120 and n_sc == 47752
              and n_mv == 9809)
        print("census tallies %s: 104/123 off-optimum, 120/120 rows with five or more "
              "distinct scores unique, 47,752 scores, 9,809 star assignments moved"
              % ("CONFIRMED" if ok else "BROKEN"))
        if not ok:
            return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__" and not os.path.exists(ROWS_JSON):
    print("stars-evaluate: ma-cutpoint-rows.json not found beside stars-evaluate.py (looked for %s); the data "
          "file must sit in the same folder as this script" % ROWS_JSON,
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    sys.exit(main())
