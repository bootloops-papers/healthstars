#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Input-order and tie-rule sweep of the HCAHPS Ward replay on the public file
(stars_census/legb_hcahps/hcahps_ward_replay.py; stored result 1 of 8 measures
reproduce, H_COMP_1), with the ICH CAHPS replay
(stars_census/legc_ich/ich_ward_replay.py; stored 5 of 6) as the positive
control.

A failed Ward rerun does not by itself show that the input set differs: Ward
can depend on merge ties and ordering. This script measures that dependence on
the public starred sets.

Per measure (8 HCAHPS starred measures of the May-2026 refresh, n=3,176; 6 ICH
CAHPS measures of the April-2026 file, n=2,594), k=5:
  DOCUMENTED tie rule (src/ward_cluster_tie.ward_sas_cut = SAS PROC CLUSTER's
  documented tie rule, stored-distance Lance-Williams in IEEE double) under
  4 named input orders (file order, reversed file order, ascending score,
  descending score) + N_PERM seeded uniform permutations of the file order
  (random.Random(SEED0+i).shuffle, SEED0=20260927), each compared with CMS's
  published per-hospital stars (mismatch count; 0 = reproduces).
  ALTERNATIVE tie rules at file order and ascending order:
    (a) scipy NN-chain Ward (src/ward_cluster.ward_cut);
    (b) exact-arithmetic Ward with EXHAUSTIVE exact-tie branch enumeration
        (src/ward1d.ward_partitions, rel_eps=0): the set of k=5 partitions
        reachable under ANY tie-breaking order -- order-free by construction;
    (c) the same with a 1e-9 relative near-tie window, which encloses every
        IEEE-double rounding realization of the merge comparisons.
  Tie structure: n distinct public score values vs n; whether CMS's published
  stars are interval-consistent on the public scores (no two hospitals with
  the same public score carry different stars; stars monotone in score) and
  the implied cut intervals between adjacent distinct public values.

The sweep runs a row-minimum-cached mirror of ward_sas_cut written here (same
distance init, same candidate set = exact double equality with the global
minimum, same (max id, min id) tie key, same Lance-Williams expression
evaluated by the same numpy elementwise ops in the same operand order, same
survivor slot).  The mirror is checked bit-for-bit (identical member
partitions) against the imported ward_sas_cut on (i) seeded random
duplicate-heavy and tie-free instances, (ii) full-size reruns of the original
engine on the real (measure, ordering) configurations listed in the record
(which ones, agree/disagree), and (iii) the stored replay records' per-measure
mismatch counts at their own input order.  Any disagreement is written to the
record.

Writes runs/hcahps_order_tie_sweep.json .

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 and Appendix C.6
    (Ward's method on the public survey files under every input order and tie
    rule tried).
Run:  cd src && python3 hcahps_order_tie_sweep.py [workers] [permutations]
Requires: Python >= 3.10; numpy, scipy.
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
"""
import os
import csv
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                      # package root
sys.path.insert(0, HERE)
from ward_cluster_tie import ward_sas_cut  # noqa: E402  (the stored engine)

HC_DIR = os.path.join(BASE, "stars_census", "legb_hcahps")
ICH_DIR = os.path.join(BASE, "stars_census", "legc_ich")
HC_CSV = os.path.join(HC_DIR, "data", "HCAHPS-Hospital.csv")
ICH_CSV = os.path.join(ICH_DIR, "data", "ICH_CAHPS_FACILITY.csv")
HC_STORED = os.path.join(HC_DIR, "HCAHPS_WARD_REPLAY.json")
ICH_STORED = os.path.join(ICH_DIR, "ICH_WARD_REPLAY.json")
OUT = os.path.join(BASE, "runs", "hcahps_order_tie_sweep.json")

HC_MEASURES = ["H_COMP_1", "H_COMP_2", "H_COMP_5", "H_COMP_6",
               "H_CLEAN", "H_QUIET", "H_HSP_RATING", "H_RECMND"]
ICH_MEASURES = [
    ("nephrologists' communication and caring",
     "Linearized score of nephrologists' communication and caring",
     "Star rating of nephrologists' communication and caring"),
    ("quality of dialysis center care and operations",
     "Linearized score of quality of dialysis center care and operations",
     "Star rating of quality of dialysis center care and operations"),
    ("providing information to patients",
     "Linearized score of providing information to patients",
     "Star rating of providing information to patients"),
    ("rating of the nephrologist",
     "Linearized score of rating of the nephrologist",
     "Star rating of the nephrologist"),
    ("rating of the dialysis center staff",
     "Linearized score of rating of the dialysis center staff",
     "Star rating of the dialysis center staff"),
    ("rating of the dialysis facility",
     "Linearized score of rating of the dialysis facility",
     "Star rating of the dialysis facility"),
]
ICH_MISSING = ("", "Not Available", "Not Applicable")
K = 5
SEED0 = 20260927
CODE_PINS = ["src/ward_cluster_tie.py", "src/ward_cluster.py", "src/ward1d.py",
             "src/dp_kmeans.py", "src/hcahps_order_tie_sweep.py",
             "stars_census/legb_hcahps/hcahps_ward_replay.py",
             "stars_census/legc_ich/ich_ward_replay.py"]


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Row-minimum-cached mirror of ward_cluster_tie.ward_sas_cut (intended to agree exactly).
# --------------------------------------------------------------------------
def ward_sas_cut_mirror(values, k, check=False):
    """Same contract as ward_sas_cut(values, k): values = floats in DATASET
    ORDER (row i = SAS obs i+1); returns clusters (sorted index lists) ordered
    by cluster mean.  Differences from the original are bookkeeping only:
    a full symmetric matrix instead of the upper triangle (same stored values),
    and a cached per-row minimum so the global minimum and its exact-equality
    candidate set are found without scanning n^2 entries per merge.
    check=True additionally asserts, every merge, that the chosen pair is at
    the global minimum and wins the (max id, min id) key among ALL entries equal
    to it (an O(n^2) audit; used by the self-test only)."""
    n = len(values)
    if n == 0:
        return []
    if n <= k:
        order = np.argsort(values, kind="stable")
        return [[int(i)] for i in order]
    x = np.asarray(values, dtype=np.float64)
    M = (x[:, None] - x[None, :]) ** 2
    np.fill_diagonal(M, np.inf)
    active = np.ones(n, dtype=bool)
    minid = np.arange(1, n + 1)
    size = np.ones(n, dtype=np.float64)
    members = [[i] for i in range(n)]
    rowmin = M.min(axis=1)
    # zero phase schedule (see module docstring of ward1d for the dedup lemma;
    # here we only use it to FIND the tie-rule winner fast, then verify it):
    # an entry is exactly 0.0 iff the two active clusters carry the same value;
    # the (max id, min id) key with the invariant minid[i] == i+1 selects, among
    # zero pairs, the smallest larger index j, partnered with its value group's
    # first index.  So the zero merges are (g[0], g[t]) over all value groups g,
    # in increasing order of g[t].
    groups = defaultdict(list)
    for i, v in enumerate(values):
        groups[v].append(i)
    zero_sched = sorted((g[t], g[0]) for g in groups.values()
                        for t in range(1, len(g)))
    zptr = 0
    INF = np.inf
    if check:
        iu = np.triu_indices(n, 1)

        def _audit(a, b):
            u = M[iu]
            tgt = u.min()
            sel = np.nonzero(u == tgt)[0]
            keys = sorted((max(minid[iu[0][s]], minid[iu[1][s]]),
                           min(minid[iu[0][s]], minid[iu[1][s]]),
                           int(iu[0][s]), int(iu[1][s])) for s in sel)
            assert M[a, b] == tgt and (keys[0][2], keys[0][3]) == (a, b), (keys[0], a, b)
    for _ in range(n - k):
        dmin = rowmin.min()
        if dmin == 0.0 and zptr < len(zero_sched):
            b, a = zero_sched[zptr]
            zptr += 1
            # verify: this is a zero (hence global-min) pair, both active, and
            # the id invariant that makes it the tie-rule winner holds
            if not (active[a] and active[b] and M[a, b] == 0.0
                    and minid[a] == a + 1 and minid[b] == b + 1):
                raise AssertionError("zero-phase schedule violated")
            if check:
                _audit(a, b)
        else:
            if dmin == 0.0:
                raise AssertionError("zero distance outside the zero schedule")
            rows = np.nonzero(rowmin == dmin)[0]
            best = None
            for i in rows:
                cols = np.nonzero(M[i] == dmin)[0]
                for j in cols:
                    j = int(j)
                    i_, j_ = (int(i), j) if i < j else (j, int(i))
                    ida, idb = minid[i_], minid[j_]
                    key = (max(ida, idb), min(ida, idb))
                    if best is None or key < best[0]:
                        best = (key, i_, j_)
            a, b = best[1], best[2]
            if check:
                _audit(a, b)
        NK, NL = size[a], size[b]
        DKL = M[a, b]
        js = np.nonzero(active)[0]
        js = js[(js != a) & (js != b)]
        if len(js):
            djk = M[js, a]
            djl = M[js, b]
            NJ = size[js]
            # identical expression / operand order to ward_sas_cut
            newd = ((NJ + NK) * djk + (NJ + NL) * djl - NJ * DKL) / (NJ + NK + NL)
            old_rm = rowmin[js]
            M[js, a] = newd
            M[a, js] = newd
        active[b] = False
        M[b, :] = INF
        M[:, b] = INF
        rowmin[b] = INF
        size[a] = NK + NL
        minid[a] = min(minid[a], minid[b])
        members[a] = members[a] + members[b]
        if len(js):
            rowmin[a] = newd.min()
            lower = newd <= old_rm
            rowmin[js[lower]] = newd[lower]
            need = (~lower) & ((djk == old_rm) | (djl == old_rm))
            for J in js[need]:
                rowmin[J] = M[J].min()
        else:
            rowmin[a] = INF
        if check:
            act = np.nonzero(active)[0]
            for J in act:
                assert rowmin[J] == M[J].min(), "rowmin invariant"
    out = [members[i] for i in np.nonzero(active)[0]]
    out.sort(key=lambda idxs: float(np.mean([values[i] for i in idxs])))
    return [sorted(c) for c in out]


def selftest_mirror(n_trials=160, seed=SEED0):
    """Bit-for-bit agreement of the mirror with the imported ward_sas_cut on
    seeded random instances: duplicate-heavy integers (HCAHPS-like), one- and
    two-decimal grids, and tie-free continuous data, in random input order."""
    rng = random.Random(seed)
    agree = 0
    cases = []
    for t in range(n_trials):
        kind = t % 4
        if kind == 0:
            n = rng.randint(40, 420)
            vals = [float(rng.randint(60, 100)) for _ in range(n)]
        elif kind == 1:
            n = rng.randint(40, 300)
            vals = [rng.randint(500, 1000) / 10 for _ in range(n)]
        elif kind == 2:
            n = rng.randint(20, 200)
            vals = [rng.uniform(0, 100) for _ in range(n)]
        else:
            n = rng.randint(40, 300)
            base = [float(rng.randint(46, 99)) for _ in range(n)]
            vals = sorted(base, reverse=bool(t % 8 == 3))
        k = rng.choice([3, 5, 5, 5])
        a = ward_sas_cut(vals, k)
        b = ward_sas_cut_mirror(vals, k, check=(t % 10 == 0))
        ok = (a == b)
        agree += ok
        if not ok:
            cases.append(dict(trial=t, kind=kind, n=n, k=k))
    return dict(n_trials=n_trials, n_agree=agree, disagreements=cases,
                kinds="0: ints 60..100 n<=420; 1: one-decimal 50.0..100.0; "
                      "2: continuous tie-free; 3: sorted/reversed ints 46..99",
                seed=seed)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_hcahps():
    """{measure: [(facility_id, int score, int star), ...] in FILE order of the
    measure's LINEAR_SCORE rows, restricted to hospitals carrying both}."""
    lin = defaultdict(dict)
    lin_order = defaultdict(list)
    star = defaultdict(dict)
    want_lin = {f"{m}_LINEAR_SCORE": m for m in HC_MEASURES}
    want_star = {f"{m}_STAR_RATING": m for m in HC_MEASURES}
    with open(HC_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            mid = r.get("HCAHPS Measure ID", "")
            if mid in want_lin:
                v = r.get("HCAHPS Linear Mean Value", "").strip()
                if v.isdigit():
                    m = want_lin[mid]
                    lin[m][r["Facility ID"]] = int(v)
                    lin_order[m].append(r["Facility ID"])
            elif mid in want_star:
                v = r.get("Patient Survey Star Rating", "").strip()
                if v.isdigit():
                    star[want_star[mid]][r["Facility ID"]] = int(v)
    out = {}
    for m in HC_MEASURES:
        seen = set()
        rows = []
        for fid in lin_order[m]:
            if fid in star[m] and fid not in seen:
                seen.add(fid)
                rows.append((fid, lin[m][fid], star[m][fid]))
        out[m] = rows
    return out


def load_ich():
    """{measure: [(ccn, score_str, int star), ...] in FILE row order}."""
    out = {name: [] for name, _, _ in ICH_MEASURES}
    with open(ICH_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ccn_col = "CMS Certification Number (CCN)"
    for r in rows:
        for name, lc, sc in ICH_MEASURES:
            v, s = r[lc].strip(), r[sc].strip()
            if v not in ICH_MISSING and s not in ICH_MISSING:
                sf = Fraction(s)
                assert sf.denominator == 1
                out[name].append((r.get(ccn_col, ""), v, int(sf)))
    return out


def published_structure(rows):
    """rows: [(id, score_str_or_int, star)].  Interval consistency of the
    published stars on the public scores + implied cut intervals."""
    by_score = defaultdict(set)
    for _, v, s in rows:
        by_score[Fraction(v)].add(s)
    split_values = {str(v): sorted(s) for v, s in by_score.items() if len(s) > 1}
    distinct = sorted(by_score)
    single = {v: min(s) for v, s in by_score.items()}
    monotone = all(single[distinct[i]] <= single[distinct[i + 1]]
                   for i in range(len(distinct) - 1))
    iv = {}
    for v in distinct:
        for s in by_score[v]:
            lo, hi = iv.get(s, (v, v))
            iv[s] = (min(lo, v), max(hi, v))
    intervals = [[str(iv[s][0]), str(iv[s][1])] for s in sorted(iv)]
    implied = []
    ss = sorted(iv)
    for a, b in zip(ss, ss[1:]):
        implied.append(dict(between_stars=[a, b], above=str(iv[a][1]),
                            below=str(iv[b][0]),
                            open_cut_interval=[str(iv[a][1]), str(iv[b][0])]))
    consistent = (not split_values) and monotone
    return dict(n=len(rows), n_distinct_scores=len(distinct),
                score_min=str(distinct[0]), score_max=str(distinct[-1]),
                stars_present=sorted(iv),
                published_interval_consistent=consistent,
                published_monotone_in_score=monotone,
                scores_split_across_stars=split_values,
                published_intervals=intervals,
                implied_cut_intervals=implied,
                star_counts={str(s): c for s, c in
                             sorted(Counter(s for _, _, s in rows).items())})


def orderings(rows, n_perm, id_sorted_asc):
    """name -> list of row indices (the input order handed to Ward)."""
    n = len(rows)
    file_ix = list(range(n))
    if id_sorted_asc:   # HCAHPS stored replay: ids sorted, then by (score, star)
        asc = sorted(file_ix, key=lambda i: (Fraction(rows[i][1]), rows[i][2], rows[i][0]))
    else:               # ICH stored replay: sorted(pts) = by (score, star), stable
        asc = sorted(file_ix, key=lambda i: (Fraction(rows[i][1]), rows[i][2]))
    out = {"file": file_ix, "file_rev": file_ix[::-1], "asc": asc, "desc": asc[::-1]}
    seeds = []
    for i in range(n_perm):
        sd = SEED0 + i
        p = list(file_ix)
        random.Random(sd).shuffle(p)
        out[f"perm_seed{sd}"] = p
        seeds.append(sd)
    return out, seeds


def stars_from_clusters(clusters, exact_vals):
    """clusters: index lists; exact_vals: Fractions per index.  Rank clusters
    by exact mean (ties by min value) -> star 1..len."""
    info = []
    for ci, c in enumerate(clusters):
        xs = [exact_vals[i] for i in c]
        info.append((sum(xs, Fraction(0)) / len(xs), min(xs), ci))
    means = [m for m, _, _ in info]
    tie = len(set(means)) != len(means)
    rank = {ci: r + 1 for r, (_, _, ci) in enumerate(sorted(info))}
    st = {}
    for ci, c in enumerate(clusters):
        for i in c:
            st[i] = rank[ci]
    return st, tie


def intervals_of(st, exact_vals):
    iv = {}
    for i, s in st.items():
        v = exact_vals[i]
        lo, hi = iv.get(s, (v, v))
        iv[s] = (min(lo, v), max(hi, v))
    ivs = [[str(iv[s][0]), str(iv[s][1])] for s in sorted(iv)]
    # contiguous iff intervals do not overlap
    fr = [(iv[s][0], iv[s][1]) for s in sorted(iv)]
    contiguous = all(fr[i][1] < fr[i + 1][0] for i in range(len(fr) - 1))
    return ivs, contiguous


def n_boundaries_matched(ivs, pub_ivs):
    a = {(x[1], y[0]) for x, y in zip(ivs, ivs[1:])}
    b = {(x[1], y[0]) for x, y in zip(pub_ivs, pub_ivs[1:])}
    return len(a & b)


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
_DATA = {}


def _task(args):
    """(program, measure, ordering name, engine) -> result dict."""
    prog, meas, oname, engine = args
    rows = _DATA[prog][meas]["rows"]
    perm = _DATA[prog][meas]["orders"][oname]
    pub_ivs = _DATA[prog][meas]["pub_ivs"]
    vals = [float(Fraction(rows[i][1])) if prog == "ICH" else float(rows[i][1])
            for i in perm]
    exact = [Fraction(rows[i][1]) for i in perm]
    pub = [rows[i][2] for i in perm]
    if engine == "mirror":
        cl = ward_sas_cut_mirror(vals, K)
    elif engine == "original":
        cl = ward_sas_cut(vals, K)
    elif engine == "scipy_nnchain":
        from ward_cluster import ward_cut
        cl = ward_cut(vals, K)
    else:
        raise ValueError(engine)
    st, tie = stars_from_clusters(cl, exact)
    ws = [st[i] for i in range(len(perm))]
    mism = sum(1 for a, b in zip(ws, pub) if a != b)
    ivs, contiguous = intervals_of(st, exact)
    # canonical partition key independent of input order: member sets by id
    part_by_value = tuple(tuple(x) for x in ivs)
    memb = sorted(tuple(sorted(rows[perm[i]][0] for i in c)) for c in cl)
    memb_sha = hashlib.sha256(json.dumps(memb).encode()).hexdigest()[:16]
    return dict(program=prog, measure=meas, ordering=oname, engine=engine,
                n=len(perm), n_mismatched=mism, reproduces=(mism == 0),
                ward_intervals=ivs, ward_contiguous=contiguous,
                cluster_mean_tie=tie, n_clusters=len(cl),
                n_boundaries_matched_of_4=n_boundaries_matched(ivs, pub_ivs),
                partition_key=part_by_value, membership_sha16=memb_sha)


def exact_enumeration(rows, pub_by_id, rel_eps, max_branches=4096):
    """ward1d exhaustive tie-branch enumeration on the multiset of public scores
    (order-free).  Returns summary incl. whether ANY branch reproduces."""
    from ward1d import ward_partitions
    xs = [Fraction(v) for _, v, _ in rows]
    pub_of_value = {}
    for _, v, s in rows:
        pub_of_value.setdefault(Fraction(v), s)
    cnt = Counter(Fraction(v) for _, v, _ in rows)
    try:
        w = ward_partitions(xs, K, rel_eps=rel_eps, max_branches=max_branches)
    except RuntimeError as e:
        return dict(rel_eps=str(rel_eps), overflow=True, error=str(e),
                    max_branches=max_branches)
    branches = []
    for p in w["partitions"]:
        info = []
        for ci, cl in enumerate(p["clusters"]):
            info.append((sum(cl, Fraction(0)) / len(cl), min(cl), ci))
        rank = {ci: r + 1 for r, (_, _, ci) in enumerate(sorted(info))}
        star_of_value = {}
        for ci, cl in enumerate(p["clusters"]):
            for v in cl:
                star_of_value[v] = rank[ci]
        mism = sum(cnt[v] for v in cnt if star_of_value[v] != pub_of_value[v])
        iv = {}
        for v, s in star_of_value.items():
            lo, hi = iv.get(s, (v, v))
            iv[s] = (min(lo, v), max(hi, v))
        ivs = [[str(iv[s][0]), str(iv[s][1])] for s in sorted(iv)]
        branches.append(dict(intervals=ivs, contiguous=bool(p["contiguous"]),
                             n_mismatched=mism))
    branches.sort(key=lambda b: b["n_mismatched"])
    return dict(rel_eps=str(rel_eps), overflow=False, n_branches=len(branches),
                published_reachable=any(b["n_mismatched"] == 0 for b in branches),
                min_mismatched=branches[0]["n_mismatched"] if branches else None,
                branches=branches[:12], branches_listed="up to 12, best first")


def main(workers=None, n_perm=64):
    workers = workers or os.cpu_count() or 1
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    log("selftest mirror vs stored engine ...")
    st = selftest_mirror()
    log("selftest:", st["n_agree"], "/", st["n_trials"], "disagree:", st["disagreements"])
    log("loading HCAHPS csv ...")
    hc = load_hcahps()
    log("loading ICH csv ...")
    ich = load_ich()
    data = {"HCAHPS": {}, "ICH": {}}
    seeds_used = None
    for prog, src in (("HCAHPS", hc), ("ICH", ich)):
        for meas, rows in src.items():
            ps = published_structure(rows)
            ords, seeds = orderings(rows, n_perm, id_sorted_asc=(prog == "HCAHPS"))
            seeds_used = seeds
            ids = [r[0] for r in rows]
            data[prog][meas] = dict(rows=rows, orders=ords, pub=ps,
                                    pub_ivs=ps["published_intervals"],
                                    file_order_is_id_sorted=(ids == sorted(ids)),
                                    ids_unique=(len(ids) == len(set(ids))))
            log(prog, meas, "n=", ps["n"], "distinct=", ps["n_distinct_scores"],
                "consistent=", ps["published_interval_consistent"])
    _DATA.update(data)
    order_names = list(next(iter(data["HCAHPS"].values()))["orders"].keys())

    # ---- phase 1: documented tie rule (mirror engine), all orderings ----------
    tasks = [(prog, meas, on, "mirror") for prog in data for meas in data[prog]
             for on in order_names]
    # ---- alt arm (a): scipy NN-chain at file + asc
    tasks += [(prog, meas, on, "scipy_nnchain") for prog in data
              for meas in data[prog] for on in ("file", "asc")]
    log("phase 1:", len(tasks), "tasks on", workers, "workers")
    ctx = mp.get_context("fork")
    res = []
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 50 == 0 or r["engine"] != "mirror":
                log(f"  {i+1}/{len(tasks)}", r["program"], r["measure"][:24],
                    r["ordering"][:14], r["engine"], "mism", r["n_mismatched"])
    log("phase 1 done")

    # ---- alt arms (b),(c): exact tie-branch enumeration (order-free) ----------
    enum = {}
    for prog in data:
        for meas, d in data[prog].items():
            pub_by_id = {r[0]: r[2] for r in d["rows"]}
            e0 = exact_enumeration(d["rows"], pub_by_id, Fraction(0))
            e9 = exact_enumeration(d["rows"], pub_by_id, Fraction(1, 10**9))
            enum[(prog, meas)] = {"exact_ties_rel_eps_0": e0,
                                  "near_ties_rel_eps_1e-9": e9}
            log("enum", prog, meas[:30], "b:", e0.get("n_branches"), e0.get("min_mismatched"),
                "c:", e9.get("n_branches", "overflow"), e9.get("min_mismatched"))

    # ---- phase 3 queue: original-engine check of the mirror on full-size data -
    # (ordered by n ascending: ICH n=2,594 before HCAHPS n=3,176)
    check_order = ([("ICH", m[0], "asc", "original") for m in ICH_MEASURES]
                  + [("HCAHPS", m, "asc", "original") for m in
                     ("H_COMP_1", "H_RECMND", "H_QUIET", "H_COMP_5",
                      "H_CLEAN", "H_COMP_6", "H_HSP_RATING", "H_COMP_2")]
                  + [("ICH", m[0], "file", "original") for m in ICH_MEASURES]
                  + [("HCAHPS", m, "file", "original") for m in HC_MEASURES])
    ctx_args = dict(data=data, res=res, enum=enum, st=st, order_names=order_names,
                    n_perm=n_perm, seeds_used=seeds_used, log=log)
    # checkpoint record BEFORE the original-engine phase (rewritten after)
    write_record(check=[], pending_check=[dict(program=t[0], measure=t[1], ordering=t[2])
                                          for t in check_order],
                 phase3_state="pending (checkpoint written before phase 3)", **ctx_args)
    log("phase 3: original engine on full-size data;", len(check_order), "queued")
    check = []
    with ctx.Pool(workers) as pool:
        pending = [pool.apply_async(_task, (t,)) for t in check_order]
        done = [False] * len(pending)
        last_ckpt = time.time()
        while True:
            got = False
            for i, p in enumerate(pending):
                if not done[i] and p.ready():
                    done[i] = True
                    got = True
                    r = p.get()
                    check.append(r)
                    log("  orig", r["program"], r["measure"][:24], r["ordering"],
                        "mism", r["n_mismatched"])
            if got and time.time() - last_ckpt > 300:
                last_ckpt = time.time()
                write_record(check=list(check),
                             pending_check=[dict(program=t[0], measure=t[1], ordering=t[2])
                                            for t, d in zip(check_order, done) if not d],
                             phase3_state="running (intermediate checkpoint)", **ctx_args)
            if all(done):
                break
            time.sleep(5)
        pool.terminate()
    pending_check = [dict(program=t[0], measure=t[1], ordering=t[2])
                     for t, d in zip(check_order, done) if not d]
    log("phase 3 done; completed", len(check), "pending", len(pending_check))
    write_record(check=check, pending_check=pending_check,
                 phase3_state=("complete" if not pending_check else
                               f"{len(pending_check)} queued configurations not run"),
                 **ctx_args)


def write_record(*, data, res, enum, st, order_names, n_perm, seeds_used,
                 log, check, pending_check, phase3_state):
    # ---- assemble ------------------------------------------------------------
    by = defaultdict(list)
    for r in res:
        by[(r["program"], r["measure"], r["engine"])].append(r)
    mirror_at = {(r["program"], r["measure"], r["ordering"]): r
                 for r in res if r["engine"] == "mirror"}
    stored = {"HCAHPS": json.load(open(HC_STORED)), "ICH": json.load(open(ICH_STORED))}

    live_checks = []
    for r in check:
        m = mirror_at[(r["program"], r["measure"], r["ordering"])]
        live_checks.append(dict(program=r["program"], measure=r["measure"],
                                ordering=r["ordering"], n=r["n"],
                                original_membership_sha16=r["membership_sha16"],
                                mirror_membership_sha16=m["membership_sha16"],
                                identical_partition=(r["membership_sha16"] == m["membership_sha16"]),
                                original_n_mismatched=r["n_mismatched"],
                                mirror_n_mismatched=m["n_mismatched"]))
    stored_checks = []
    for prog in data:
        bm = stored[prog]["measures"]
        for meas in data[prog]:
            b = bm.get(meas, {})
            bn = b.get("n_mismatched", b.get("n_mismatched_facilities"))
            m = mirror_at[(prog, meas, "asc")]
            stored_checks.append(dict(program=prog, measure=meas,
                                      stored_n_mismatched=bn,
                                      stored_reproduces=b.get("ward_reproduces_published"),
                                      mirror_asc_n_mismatched=m["n_mismatched"],
                                      agree=(bn == m["n_mismatched"]
                                             and b.get("ward_reproduces_published") == m["reproduces"])))

    programs = {}
    for prog in data:
        pm = {}
        for meas, d in data[prog].items():
            rs = sorted(by[(prog, meas, "mirror")], key=lambda r: order_names.index(r["ordering"]))
            hist = Counter(r["partition_key"] for r in rs)
            first_of = {}
            for r in rs:
                first_of.setdefault(r["partition_key"], r)
            best = min(rs, key=lambda r: (r["n_mismatched"], order_names.index(r["ordering"])))
            named = {on: {kk: mirror_at[(prog, meas, on)][kk] for kk in
                          ("n_mismatched", "reproduces", "ward_intervals",
                           "n_boundaries_matched_of_4", "ward_contiguous")}
                     for on in ("file", "file_rev", "asc", "desc")}
            perm_rs = [r for r in rs if r["ordering"].startswith("perm_seed")]
            sci = {r["ordering"]: {kk: r[kk] for kk in ("n_mismatched", "reproduces",
                                                        "ward_intervals",
                                                        "n_boundaries_matched_of_4")}
                   for r in by[(prog, meas, "scipy_nnchain")]}
            en = enum[(prog, meas)]
            any_alt = (any(v["reproduces"] for v in sci.values())
                       or bool(en["exact_ties_rel_eps_0"].get("published_reachable"))
                       or bool(en["near_ties_rel_eps_1e-9"].get("published_reachable")))
            alt_min = min([v["n_mismatched"] for v in sci.values()]
                          + [e["min_mismatched"] for e in en.values()
                             if e.get("min_mismatched") is not None])
            pm[meas] = dict(
                n=d["pub"]["n"], n_distinct_scores=d["pub"]["n_distinct_scores"],
                score_range=[d["pub"]["score_min"], d["pub"]["score_max"]],
                file_order_is_id_sorted=d["file_order_is_id_sorted"],
                ids_unique=d["ids_unique"],
                published=d["pub"],
                documented_rule=dict(
                    n_orderings_tested=len(rs),
                    n_orderings_reproducing=sum(r["reproduces"] for r in rs),
                    n_seeded_permutations=len(perm_rs),
                    n_permutations_reproducing=sum(r["reproduces"] for r in perm_rs),
                    min_mismatched=best["n_mismatched"],
                    argmin_ordering=best["ordering"],
                    argmin_ward_intervals=best["ward_intervals"],
                    argmin_boundaries_matched_of_4=best["n_boundaries_matched_of_4"],
                    max_mismatched=max(r["n_mismatched"] for r in rs),
                    mismatch_by_named_ordering=named,
                    permutation_mismatch_min_median_max=(
                        [min(r["n_mismatched"] for r in perm_rs),
                         sorted(r["n_mismatched"] for r in perm_rs)[len(perm_rs) // 2],
                         max(r["n_mismatched"] for r in perm_rs)] if perm_rs else None),
                    n_distinct_partitions_reached=len(hist),
                    partitions_reached=[dict(ward_intervals=[list(x) for x in key], n_orderings=c,
                                             n_mismatched=first_of[key]["n_mismatched"],
                                             first_ordering=first_of[key]["ordering"])
                                        for key, c in hist.most_common()],
                    any_noncontiguous=any(not r["ward_contiguous"] for r in rs),
                    any_cluster_mean_tie=any(r["cluster_mean_tie"] for r in rs),
                    per_ordering=[dict(ordering=r["ordering"], n_mismatched=r["n_mismatched"],
                                       membership_sha16=r["membership_sha16"])
                                  for r in rs]),
                alt_tie_rules=dict(scipy_nnchain=sci, **en),
                reproduces_under_any_tested_ordering_or_tie_rule=(
                    any(r["reproduces"] for r in rs) or any_alt),
                best_case_mismatched_over_everything=min(best["n_mismatched"], alt_min))
        programs[prog] = pm

    def summ(prog):
        pm = programs[prog]
        n_meas = len(pm)
        return dict(
            n_measures=n_meas,
            n_orderings_per_measure_documented_rule=len(order_names),
            n_tie_rule_arms=4,
            tie_rule_arms=["documented tie rule (ward_cluster_tie) x all orderings",
                           "scipy NN-chain Ward x {file, asc}",
                           "exact-arithmetic Ward, exhaustive exact-tie branches (order-free)",
                           "exact-arithmetic Ward, 1e-9 near-tie branches (order-free)"],
            n_configurations_per_measure=len(order_names) + 2 + 2,
            n_reproduce_at_stored_order_asc=sum(
                pm[m]["documented_rule"]["mismatch_by_named_ordering"]["asc"]["reproduces"] for m in pm),
            n_reproduce_at_file_order=sum(
                pm[m]["documented_rule"]["mismatch_by_named_ordering"]["file"]["reproduces"] for m in pm),
            n_reproduce_under_some_ordering_documented_rule=sum(
                pm[m]["documented_rule"]["n_orderings_reproducing"] > 0 for m in pm),
            n_reproduce_under_any_ordering_or_tie_rule=sum(
                pm[m]["reproduces_under_any_tested_ordering_or_tie_rule"] for m in pm),
            measures_reproducing_somewhere=[m for m in pm if
                                            pm[m]["reproduces_under_any_tested_ordering_or_tie_rule"]],
            best_case_mismatched_by_measure={m: pm[m]["best_case_mismatched_over_everything"]
                                             for m in pm},
            documented_rule_min_mismatched_by_measure={m: pm[m]["documented_rule"]["min_mismatched"]
                                                    for m in pm},
            documented_rule_orderings_reproducing_by_measure={
                m: f'{pm[m]["documented_rule"]["n_orderings_reproducing"]}/{pm[m]["documented_rule"]["n_orderings_tested"]}'
                for m in pm},
            n_distinct_partitions_reached_by_measure={
                m: pm[m]["documented_rule"]["n_distinct_partitions_reached"] for m in pm},
            n_distinct_scores_by_measure={m: pm[m]["n_distinct_scores"] for m in pm},
            all_published_interval_consistent=all(
                pm[m]["published"]["published_interval_consistent"] for m in pm))

    hc_s, ich_s = summ("HCAHPS"), summ("ICH")
    pins = [HC_CSV, ICH_CSV, HC_STORED, ICH_STORED] + [os.path.join(BASE, p) for p in CODE_PINS]
    check_ok = (st["n_agree"] == st["n_trials"]
                and len(live_checks) > 0
                and all(c["identical_partition"] for c in live_checks)
                and all(c["agree"] for c in stored_checks))
    record = dict(
        generator="src/hcahps_order_tie_sweep.py",
        input_pins_sha256_16={os.path.relpath(p, BASE): sha16(p) for p in pins},
        register=(
            "COMPUTED: for each of the 8 HCAHPS starred measures (public PDC file, May-2026 "
            "refresh, hospitals carrying both an integer linear mean value and a star; n per "
            "measure in the rows) and each of the 6 ICH CAHPS measures (April-2026 facility "
            "file), Ward k=5 under the documented SAS tie rule for exactly the input orderings "
            f"listed in config.orderings ({len(order_names)} per measure: file, file_rev, asc, "
            f"desc, {n_perm} seeded permutations of file order), plus scipy NN-chain Ward at "
            "file and asc order, plus exact-arithmetic Ward with exhaustive exact-tie branch "
            "enumeration (rel_eps=0) and 1e-9 near-tie enumeration (order-free; a branch set, "
            "not an ordering sample), each compared hospital-by-hospital with CMS's published "
            "stars. The documented-rule sweep ran on a row-min-cached MIRROR of ward_sas_cut; "
            "its bit-for-bit identity with the stored engine is checked only on the "
            "configurations listed under kernel_check (random seeded instances, the "
            "full-size original-engine reruns that completed, and the stored replay "
            "records at their own input order) -- configurations actually tested, not all "
            "orderings. NOT COMPUTED: orderings other than those listed; tie rules other "
            "than the four arms; any statement about CMS's production input set beyond "
            "what the public file shows; SSQ optimality (see the census records)."),
        config=dict(k=K, n_seeded_permutations=n_perm, seed0=SEED0, seeds=seeds_used,
                    orderings=order_names,
                    ordering_definitions=dict(
                        file="row order of the measure's score rows in the public csv",
                        file_rev="file reversed",
                        asc="HCAHPS: sorted by (score, star, facility id) = the stored replay's "
                            "input order; ICH: stable sort of file order by (score, star) = the "
                            "stored replay's input order",
                        desc="asc reversed",
                        perm_seedS="random.Random(S).shuffle of the file-order index list")),
        kernel_check=dict(
            all_checks_pass=check_ok,
            random_instances=st,
            live_original_engine_phase=phase3_state,
            live_original_engine_full_n=live_checks,
            live_original_engine_n_completed=len(live_checks),
            live_original_engine_n_identical=sum(c["identical_partition"] for c in live_checks),
            live_original_engine_not_run=pending_check,
            stored_replay_crosscheck_at_asc=stored_checks,
            stored_replay_crosscheck_n_agree=f'{sum(c["agree"] for c in stored_checks)}/{len(stored_checks)}'),
        programs=programs,
        summary=dict(HCAHPS=hc_s, ICH_control=ich_s),
        headline_numbers=dict(
            hcahps_orderings_per_measure=len(order_names),
            hcahps_configurations_per_measure=len(order_names) + 4,
            hcahps_measures=8,
            hcahps_reproduce_somewhere=hc_s["n_reproduce_under_any_ordering_or_tie_rule"],
            hcahps_reproduce_file_order=hc_s["n_reproduce_at_file_order"],
            hcahps_reproduce_stored_order=hc_s["n_reproduce_at_stored_order_asc"],
            hcahps_best_case_mismatched=hc_s["best_case_mismatched_by_measure"],
            ich_measures=6,
            ich_reproduce_stored_order=ich_s["n_reproduce_at_stored_order_asc"],
            ich_reproduce_file_order=ich_s["n_reproduce_at_file_order"],
            ich_reproduce_somewhere=ich_s["n_reproduce_under_any_ordering_or_tie_rule"],
            ich_best_case_mismatched=ich_s["best_case_mismatched_by_measure"]),
        dropped_or_capped=dict(
            live_original_engine_configs_not_run=len(pending_check),
            live_original_engine_phase=phase3_state,
            note=("the original-engine reruns check the mirror only; the sweep itself "
                  "(all orderings x all measures, both programs) ran in full")))
    with open(OUT, "w") as f:
        json.dump(record, f, indent=1, default=str)
        f.write("\n")
    log("wrote", OUT, "sha16", sha16(OUT))
    log("HCAHPS:", json.dumps(hc_s["documented_rule_orderings_reproducing_by_measure"]))
    log("HCAHPS best-case:", json.dumps(hc_s["best_case_mismatched_by_measure"]))
    log("ICH:", json.dumps(ich_s["documented_rule_orderings_reproducing_by_measure"]))
    log("ICH best-case:", json.dumps(ich_s["best_case_mismatched_by_measure"]))
    log("check pass:", check_ok)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__" and not (os.path.exists(HC_CSV) and os.path.exists(ICH_CSV)):
    print("hcahps_order_tie_sweep: missing input stars_census/legb_hcahps/data/HCAHPS-Hospital.csv or "
          "stars_census/legc_ich/data/ICH_CAHPS_FACILITY.csv (raw CMS survey files, not included in the package; "
          "re-fetch each by the URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    a = sys.argv[1:]
    main(workers=int(a[0]) if len(a) > 0 else None,
         n_perm=int(a[1]) if len(a) > 1 else 64)
