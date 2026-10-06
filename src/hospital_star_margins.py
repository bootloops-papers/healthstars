#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Constrained-DP star margins for the 2026
hospital program (peer groups 3/4/5, k=5, summary scores at the exact
binary-double register Fraction(float) of the R doubles, exactly as
hospital/work/exact_census_2026.py loads them).

For EVERY rated hospital i in each peer group:
  penalty(i, s) = [min SSQ over contiguous 5-partitions of the sorted scores in
                   which hospital i falls in star group s] - SSQ_opt
  margin(i)     = min over s != s_opt(i) of penalty(i, s)
Method: float prefix/suffix DP tables (search) -> for star s and sorted
position i, min over a<=i<b of P[s-1][a] + cost(a,b) + S[5-s][b]; every
candidate (s,a,b) within 1e-9 relative of the float minimum is evaluated in
exact Fraction arithmetic using EXACT prefix/suffix DP tables (so the exact
value of a candidate is the true minimum over all partitions whose block s is
[a,b)); reported penalties/margins are exact differences of exact SSQs.
Hospitals with identical score values share a margin (min over positions).
Also: per peer group SSQ_cms, SSQ_opt, gap (exact + percent), SSQ of the
2nd-best contiguous partition (runner-up gap), CMS-partition contiguity check,
and distributions of margins among the reassigned (star_differs) hospitals.

Writes runs/hospital_star_margins.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.6, near-optimal
    hospital groupings (the criterion is flat around the unique optimum).
Run:  cd src && python3 hospital_star_margins.py
Requires: Python >= 3.10; numpy.
"""
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                 # record root
APPS = BASE                                  # pins are keyed relative to the record root (hospital/ sits inside the record)
BASE_H = os.path.join(BASE, "hospital")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BASE_H, "work"))

K = 5
REL_TOL = 1e-9
ABS_TOL = 1e-12
PEERS = {3: "peer3", 4: "peer4", 5: "peer5"}

INPUTS = [
    os.path.join(BASE_H, "work", "R_output", "fullprec_2026apr.csv"),
    os.path.join(BASE_H, "work", "R_output", "Star_precap_2026apr.csv"),
    os.path.join(BASE_H, "work", "exact_census_2026.py"),
    os.path.join(BASE_H, "runs", "exact_census_2026.json"),
    os.path.join(HERE, "dp_kmeans.py"),
]


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def fstr(q):
    return f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------- exact DP
def exact_prefix_tables(xs, gmax_full, need_top):
    """xs sorted Fractions. Returns (P, cost) with P[g][j] = min SSQ of xs[:j]
    in g nonempty contiguous blocks for g=0..gmax_full (all j), plus
    P[need_top][n] if need_top > gmax_full. None = infeasible."""
    n = len(xs)
    ps = [Fraction(0)]
    ps2 = [Fraction(0)]
    for x in xs:
        ps.append(ps[-1] + x)
        ps2.append(ps2[-1] + x * x)

    def cost(i, j):
        m = j - i
        if m <= 1:
            return Fraction(0)
        s = ps[j] - ps[i]
        return (ps2[j] - ps2[i]) - s * s / m

    gtop = max(gmax_full, need_top)
    P = [[None] * (n + 1) for _ in range(gtop + 1)]
    P[0][0] = Fraction(0)
    for g in range(1, gtop + 1):
        js = range(g, n + 1) if g <= gmax_full else [n]
        for j in js:
            best = None
            row = P[g - 1]
            for i in range(g - 1, j):
                prev = row[i]
                if prev is None:
                    continue
                c = prev + cost(i, j)
                if best is None or c < best:
                    best = c
            P[g][j] = best
    return P, cost


# ---------------------------------------------------------------- float DP
def float_tables(xf):
    """xf: sorted, centered float array. Returns P (K+1 x n+1), S (K+1 x n+1),
    cost matrix C (n+1 x n+1, inf where a>=b)."""
    n = len(xf)
    ps = np.concatenate([[0.0], np.cumsum(xf)])
    ps2 = np.concatenate([[0.0], np.cumsum(xf * xf)])
    idx = np.arange(n + 1)
    cnt = idx[None, :] - idx[:, None]            # b - a
    d = ps[None, :] - ps[:, None]
    d2 = ps2[None, :] - ps2[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(cnt > 0, d2 - d * d / np.where(cnt > 0, cnt, 1), np.inf)
    INF = np.inf
    P = np.full((K + 1, n + 1), INF)
    P[0, 0] = 0.0
    for g in range(1, K + 1):
        # P[g][j] = min_i P[g-1][i] + C[i][j]
        M = P[g - 1][:, None] + C           # (i, j)
        P[g] = M.min(axis=0)
        P[g, :g] = INF
    S = np.full((K + 1, n + 1), INF)
    S[0, n] = 0.0
    for g in range(1, K + 1):
        # S[g][b] = min_c C[b][c] + S[g-1][c]
        M = C + S[g - 1][None, :]           # (b, c)
        S[g] = M.min(axis=1)
        S[g, n - g + 1:] = INF
    return P, S, C


def peer_job(task):
    tg, sub = task
    label = PEERS[tg]
    # rows: (pid, peer, star_precap, s17, s15) ; register = Fraction(float(s17))
    recs = [(pid, Fraction(float(s17)), int(st), s17) for (pid, _, st, s17, _) in sub]
    order = sorted(range(len(recs)), key=lambda r: (recs[r][1], recs[r][0]))
    xs = [recs[r][1] for r in order]
    pids = [recs[r][0] for r in order]
    s_cms = [recs[r][2] for r in order]
    s17s = [recs[r][3] for r in order]
    n = len(xs)

    # CMS partition contiguity in sorted order
    cms_contiguous = all(s_cms[i] <= s_cms[i + 1] for i in range(n - 1))

    # exact tables: prefix P_ex[g][a] g=0..4 (all a) + P_ex[5][n]; suffix via reversed
    P_ex, cost_ex = exact_prefix_tables(xs, K - 1, K)
    xr = [-x for x in reversed(xs)]  # reversed & negated keeps sorted ascending
    R_ex, _ = exact_prefix_tables(xr, K - 1, 0)
    S_ex = [[R_ex[g][n - b] for b in range(n + 1)] for g in range(K)]
    ssq_opt = P_ex[K][n]

    # exact optimal boundaries by exact search over last block then recursion:
    # recover via: for g=K..1 find i with P[g-1][i]+cost(i,j)==P[g][j]
    # (P_ex[K] only has [n]; use value ssq_opt)
    bnds = [n]
    j = n
    target = ssq_opt
    opt_count_chain = 1
    for g in range(K, 0, -1):
        found = []
        for i in range(g - 1, j):
            prev = P_ex[g - 1][i]
            if prev is not None and prev + cost_ex(i, j) == target:
                found.append(i)
        opt_count_chain *= len(found)
        i = found[0]
        bnds.append(i)
        target = P_ex[g - 1][i]
        j = i
    bnds.reverse()  # [0,b1,..,b4,n]
    s_opt = [0] * n
    for b in range(K):
        for i in range(bnds[b], bnds[b + 1]):
            s_opt[i] = b + 1
    opt_blocks = {s: (bnds[s - 1], bnds[s]) for s in range(1, K + 1)}

    # exact SSQ_cms
    from dp_kmeans import partition_ssq
    ssq_cms = partition_ssq(xs, s_cms)
    gap = ssq_cms - ssq_opt

    # float tables
    xf = np.array([float(x) for x in xs])
    xf = xf - xf.mean()
    P_f, S_f, C_f = float_tables(xf)
    float_opt = P_f[K, n]

    exact_cache = {}
    disc = [0.0]  # max |float - exact| over exactly evaluated entries (in un-centered SSQ units; SSQ shift-invariant)

    def exact_value(s, a, b, fval):
        key = (s, a, b)
        if key not in exact_cache:
            pv = P_ex[s - 1][a]
            sv = S_ex[K - s][b]
            assert pv is not None and sv is not None, key
            v = pv + cost_ex(a, b) + sv
            exact_cache[key] = v
            dd = abs(float(v) - fval)
            if dd > disc[0]:
                disc[0] = dd
        return exact_cache[key]

    # per star s: F_s, M_s, and exact evaluation of the candidates per position
    pen_exact = [[None] * (K + 1) for _ in range(n)]   # pen_exact[i][s] = exact min SSQ with i in s
    pen_arg = [[None] * (K + 1) for _ in range(n)]
    pen_ncand = [[0] * (K + 1) for _ in range(n)]
    second_best = None
    second_arg = None
    max_cands = 0
    for s in range(1, K + 1):
        F = P_f[s - 1][:, None] + C_f + S_f[K - s][None, :]   # (a, b)
        # suffix-min over b, prefix-min over a
        G = np.minimum.accumulate(F[:, ::-1], axis=1)[:, ::-1]
        H = np.minimum.accumulate(G, axis=0)
        # runner-up search: exclude the optimal block entry for this s
        ao, bo = opt_blocks[s]
        F2 = F.copy()
        F2[ao, bo] = np.inf
        m2 = F2.min()
        cand2 = np.argwhere(F2 <= m2 + REL_TOL * abs(m2) + ABS_TOL)
        for (a, b) in cand2:
            v = exact_value(s, int(a), int(b), float(F[a, b]))
            if second_best is None or v < second_best:
                second_best = v
                second_arg = (s, int(a), int(b))
        for i in range(n):
            m = H[i, i + 1]
            if not np.isfinite(m):
                # infeasible: i in block s leaves too few points for the other blocks
                # (i < s-1 or i > n-1-(K-s)); no contiguous 5-partition exists
                pen_exact[i][s] = None
                pen_arg[i][s] = None
                pen_ncand[i][s] = 0
                continue
            sub_m = F[:i + 1, i + 1:]
            cands = np.argwhere(sub_m <= m + REL_TOL * abs(m) + ABS_TOL)
            max_cands = max(max_cands, len(cands))
            best = None
            barg = None
            for (a, cb) in cands:
                a = int(a)
                b = int(cb) + i + 1
                v = exact_value(s, a, b, float(F[a, b]))
                if best is None or v < best:
                    best = v
                    barg = (a, b)
            pen_exact[i][s] = best
            pen_arg[i][s] = barg
            pen_ncand[i][s] = int(len(cands))
        del F, G, H, F2

    # check: for s_opt(i) the exact minimum must equal ssq_opt
    sopt_consistent = all(pen_exact[i][s_opt[i]] == ssq_opt for i in range(n))

    # value ties: share margins across identical values (min over positions)
    # group consecutive equal values
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        groups.append((i, j))
        i = j + 1
    n_tied_values = sum(1 for (a, b) in groups if b > a)
    pen_val = {}  # position -> dict s -> exact min SSQ (min over tied positions)
    for (a, b) in groups:
        d = {}
        for s in range(1, K + 1):
            vals = [pen_exact[p][s] for p in range(a, b + 1) if pen_exact[p][s] is not None]
            d[s] = min(vals) if vals else None
        for p in range(a, b + 1):
            pen_val[p] = d

    hospitals = []
    n_infeasible_cells = 0
    for i in range(n):
        so = s_opt[i]
        pens = {}
        for s in range(1, K + 1):
            if s == so:
                continue
            if pen_val[i][s] is None:
                n_infeasible_cells += 1
                pens[str(s)] = {"penalty_exact": None, "penalty_float": None,
                                "infeasible_no_contiguous_partition": True}
                continue
            q = pen_val[i][s] - ssq_opt
            pens[str(s)] = {"penalty_exact": fstr(q), "penalty_float": float(q),
                            "argmin_block_s": list(pen_arg[i][s]) if pen_arg[i][s] else None,
                            "n_exact_candidates": pen_ncand[i][s]}
        feas = [s for s in pens if pen_val[i][int(s)] is not None]
        rec = {"provider_id": pids[i], "summary_score_17g": s17s[i],
               "sorted_pos": i, "star_cms": s_cms[i], "star_opt": so,
               "reassigned": s_cms[i] != so}
        if feas:
            ms = min(feas, key=lambda s: pen_val[i][int(s)])
            mq = pen_val[i][int(ms)] - ssq_opt
            rec.update({"margin_exact": fstr(mq), "margin_float": float(mq),
                        "margin_star": int(ms),
                        "margin_over_gap": float(mq / gap) if gap != 0 else None,
                        "margin_over_ssq_opt": float(mq / ssq_opt)})
        else:
            # sorted position 0 / n-1: every alternative star infeasible under
            # contiguous 5-partitions with nonempty blocks (margin = +inf)
            rec.update({"margin_exact": None, "margin_float": None, "margin_star": None,
                        "margin_infinite_all_alternatives_infeasible": True})
        rec["penalties"] = pens
        if s_cms[i] != so:
            pc = pen_val[i][s_cms[i]] - ssq_opt
            rec["penalty_cms_star_exact"] = fstr(pc)
            rec["penalty_cms_star_float"] = float(pc)
            rec["penalty_cms_star_over_gap"] = float(pc / gap)
            rec["penalty_cms_star_cmp_gap"] = ("<" if pc < gap else "==" if pc == gap else ">")
        hospitals.append(rec)

    def dist(vals_q):
        if not vals_q:
            return None
        fl = np.array([float(v) for v in vals_q])
        qs = np.percentile(fl, [0, 25, 50, 75, 100]).tolist()
        return {"n": len(vals_q), "min": qs[0], "q1": qs[1], "median": qs[2],
                "q3": qs[3], "max": qs[4],
                "min_exact": fstr(min(vals_q)), "max_exact": fstr(max(vals_q))}

    def dist_block(sel):
        m = [Fraction(h["margin_exact"]) for h in sel if h["margin_exact"] is not None]
        out = {"n_hospitals": len(sel),
               "n_margin_infinite": sum(1 for h in sel if h["margin_exact"] is None),
               "margin_ssq_units": dist(m),
               "margin_over_gap": dist([q / gap for q in m]),
               "margin_over_ssq_opt_percent": dist([q / ssq_opt * 100 for q in m]),
               "n_margin_lt_gap": sum(1 for q in m if q < gap),
               "n_margin_eq_gap": sum(1 for q in m if q == gap),
               "n_margin_gt_gap": sum(1 for q in m if q > gap)}
        return out

    reass = [h for h in hospitals if h["reassigned"]]
    keep = [h for h in hospitals if not h["reassigned"]]
    pc = [Fraction(h["penalty_cms_star_exact"]) for h in reass]
    summary = {
        "reassigned": dist_block(reass),
        "reassigned_penalty_at_cms_star": {
            "ssq_units": dist(pc),
            "over_gap": dist([q / gap for q in pc]),
            "over_ssq_opt_percent": dist([q / ssq_opt * 100 for q in pc]),
            "n_lt_gap": sum(1 for q in pc if q < gap),
            "n_eq_gap": sum(1 for q in pc if q == gap),
            "n_gt_gap": sum(1 for q in pc if q > gap),
            "n_margin_star_is_cms_star": sum(1 for h in reass if h["margin_star"] == h["star_cms"]),
        },
        "not_reassigned": dist_block(keep),
        "all": dist_block(hospitals),
    }
    # threshold counts over ALL hospitals: star undetermined at tolerance t
    thr = {}
    INFQ = None
    def mq_of(h):
        return Fraction(h["margin_exact"]) if h["margin_exact"] is not None else INFQ
    allm = [mq_of(h) for h in hospitals if h["margin_exact"] is not None]  # +inf margins never <= t
    thr["n_all_margin_le_gap"] = sum(1 for q in allm if q <= gap)
    thr["n_notreassigned_margin_le_gap"] = sum(
        1 for h in keep if h["margin_exact"] is not None and Fraction(h["margin_exact"]) <= gap)
    for pct in ("0.01", "0.046", "0.05", "0.097", "0.1", "0.5", "1", "4.67", "5"):
        t = ssq_opt * Fraction(pct) / 100
        thr[f"n_all_margin_le_{pct}pct_of_ssq_opt"] = sum(1 for q in allm if q <= t)
    # fractions of reassigned with margin below fractions of gap
    for f in ("1/10", "1/4", "1/2", "3/4"):
        thr[f"n_reassigned_margin_le_{f}_gap"] = sum(
            1 for h in reass if h["margin_exact"] is not None
            and Fraction(h["margin_exact"]) <= gap * Fraction(f))
        thr[f"n_reassigned_penalty_cms_le_{f}_gap"] = sum(
            1 for q in pc if q <= gap * Fraction(f))

    runner_gap = second_best - ssq_opt
    res = {
        "peer": label, "n": n,
        "cms_star_counts": {str(s): s_cms.count(s) for s in range(1, K + 1)},
        "opt_star_counts": {str(s): s_opt.count(s) for s in range(1, K + 1)},
        "opt_boundaries_sorted_index": bnds,
        "cms_partition_contiguous_in_sorted_scores": cms_contiguous,
        "n_value_tie_groups": n_tied_values,
        "n_infeasible_star_cells_edge_positions": n_infeasible_cells,
        "ssq_cms_exact": fstr(ssq_cms), "ssq_cms_float": float(ssq_cms),
        "ssq_opt_exact": fstr(ssq_opt), "ssq_opt_float": float(ssq_opt),
        "gap_exact": fstr(gap), "gap_float": float(gap),
        "reduction_percent_of_ssq_cms_exact": fstr(gap / ssq_cms * 100),
        "reduction_percent_of_ssq_cms_float": float(gap / ssq_cms * 100),
        "gap_percent_of_ssq_opt_float": float(gap / ssq_opt * 100),
        "n_optimal_partitions_backpointer_chain": opt_count_chain,
        "second_best_ssq_exact": fstr(second_best), "second_best_ssq_float": float(second_best),
        "runner_up_gap_exact": fstr(runner_gap), "runner_up_gap_float": float(runner_gap),
        "runner_up_gap_over_cms_gap": float(runner_gap / gap),
        "runner_up_gap_percent_of_ssq_opt": float(runner_gap / ssq_opt * 100),
        "runner_up_block_change": {"star": second_arg[0], "block": [second_arg[1], second_arg[2]],
                                    "opt_block": list(opt_blocks[second_arg[0]])},
        "optimum_unique_among_contiguous": runner_gap > 0,
        "float_search": {"float_opt_minus_exact_opt": float(float_opt - float(ssq_opt)),
                         "max_abs_float_minus_exact_over_exact_entries": disc[0],
                         "n_exact_entries": len(exact_cache),
                         "max_candidates_in_one_query": max_cands,
                         "rel_tol": REL_TOL, "abs_tol": ABS_TOL,
                         "sopt_selfcheck_exact": sopt_consistent},
        "hospitals_star_differs": len(reass),
        "summary": summary,
        "threshold_counts": thr,
        "hospitals": hospitals,
    }
    print(f"[{label}] n={n} differs={len(reass)} gap={float(gap):.6g} "
          f"runner={float(runner_gap):.3g} disc={disc[0]:.2e}", flush=True)
    return res


def main():
    import exact_census_2026 as CC
    rows = CC.load_rows()
    by_peer = {tg: [r for r in rows if r[1] == tg] for tg in (3, 4, 5)}
    tasks = [(tg, by_peer[tg]) for tg in (5, 4, 3)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(3) as pool:
        results = pool.map(peer_job, tasks)
    res = {r["peer"]: r for r in results}

    # cross-check against the stored census record
    cen = json.load(open(os.path.join(BASE_H, "runs", "exact_census_2026.json")))
    xc = {}
    for tg in (3, 4, 5):
        lb = PEERS[tg]
        r = res[lb]
        c = cen["peer_groups"][lb]
        det = {d["provider_id"]: (d["shipped"], d["dp_optimum"])
               for d in cen["star_differs_detail"] if d["peer"] == lb}
        mine = {h["provider_id"]: (h["star_cms"], h["star_opt"])
                for h in r["hospitals"] if h["reassigned"]}
        xc[lb] = {"gap_exact_matches": r["gap_exact"] == c["ssq_gap_exact"],
                  "n_matches": r["n"] == c["n"],
                  "star_differs_detail_matches": det == mine,
                  "differs": [len(mine), c["hospitals_star_differs"]]}

    out = {
        "generator": os.path.relpath(os.path.abspath(__file__), APPS),
        "input_pins_sha256_16": {os.path.relpath(p, APPS): sha16(p) for p in INPUTS},
        "register": (
            "COMPUTED: for every rated 2026 hospital (3,203; peer groups 3/4/5 as loaded by "
            "hospital/work/exact_census_2026.load_rows, scores = Fraction(float) of the "
            "%.17g R doubles, k=5, published star = pre-cap k-means star), penalty(i,s) = "
            "[min SSQ over CONTIGUOUS 5-partitions of the sorted peer-group scores with "
            "hospital i in star block s] - SSQ_opt, for every s != s_opt(i); margin(i) = min "
            "over feasible s. The counterfactual class is ALTERNATIVE CUT POINTS: 5 nonempty "
            "score intervals (contiguous blocks of the sorted scores), the object the paper "
            "audits. This is NOT the minimum over arbitrary labelings: letting hospital i alone "
            "be labeled out of score order could cost less SSQ but is not a cut-point scheme, "
            "and was NOT computed. Consequence: for the s-1 lowest / 5-s highest sorted "
            "positions star s is infeasible (reported as infeasible cells; the single lowest and "
            "single highest hospital of each peer group have no feasible alternative star, "
            "margin = +inf, excluded from quantiles and counted separately). "
            "Search: float64 prefix/suffix DP; exact evaluation: "
            "every (s,a,b) candidate within 1e-9 relative (+1e-12 abs) of the float minimum "
            "is evaluated EXACTLY as P_exact[s-1][a] + cost_exact(a,b) + S_exact[5-s][b] with "
            "exact Fraction prefix/suffix DP tables, i.e. the exact minimum over all partitions "
            "whose block s is [a,b); reported penalties are exact differences of exact SSQs. "
            "Minimality over (a,b) pairs farther than 1e-9 relative from the float minimum rests "
            "on float64 table error (max observed |float-exact| over all exactly evaluated entries is "
            "reported per peer group, ~13 orders below the tolerance*SSQ). Hospitals with "
            "identical score values share the min over their positions. SSQ_cms, SSQ_opt, gap, "
            "percent reduction, 2nd-best contiguous partition SSQ: exact. Quartiles are numpy "
            "linear-interpolation percentiles of the floats of exact values. NOT COMPUTED: "
            "post-cap (safety cap) stars; years other than 2026; non-interval (out-of-order) "
            "constrained labelings (see above); any k other than 5."),
        "cross_check_vs_exact_census_2026": xc,
        "peer_groups": {PEERS[tg]: res[PEERS[tg]] for tg in (3, 4, 5)},
    }
    dest = os.path.join(BASE, "runs", "hospital_star_margins.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"record -> {dest}", flush=True)
    print(json.dumps(xc), flush=True)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
