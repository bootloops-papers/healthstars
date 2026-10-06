# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""One-dimensional k-means for the hospital star step.

One-dimensional k-means (nearest-centroid sorting) with the seed-selection
and seed-replacement rules, the MAXITER/CONVERGE fixed-point iteration and
the STRICT exclusion option specified for the FASTCLUS procedure in the
SAS/STAT User's Guide (chapter "The FASTCLUS Procedure", sections on
background/computational method and on the REPLACE=, RADIUS=, CONVERGE=,
MAXITER=, STRICT= options), implemented independently in Python; see also
MacQueen (1967) and Hartigan and Wong (1979) for the algorithm family. The
two-stage driver kmeans_star() follows the %kmeans macro that CMS publishes
in its Hospital Star Ratings SAS package (quintile-median seeding, a first
pass, cluster centers as seeds for a STRICT second pass, re-inclusion, stars
by cluster-mean order). Validated against CMS's published hospital star
assignments for 2021-2025 (see hospital/runs/sas_replay_<year>.json).

What the documentation specifies, as implemented here (SAS/STAT 14.1 User's
Guide, chapter "The FASTCLUS Procedure"; page numbers as printed there):

  * Computational method ("Background", pp. 2430-2431). Seeds are chosen
    first; then, if the iteration limit is positive, every observation is
    assigned to its nearest seed and each seed is replaced by the mean of
    its cluster, repeatedly, until the seeds stop moving by more than the
    convergence criterion; a final pass assigns every observation to its
    nearest seed. With CONVERGE=0 and a large MAXITER= the procedure runs to
    complete convergence, at which point the final seeds equal the cluster
    means. The CMS packs call the procedure with converge=0 maxiter=1000.
  * Initial seed selection (p. 2431; RADIUS= p. 2440, default 0; REPLACE=
    p. 2444, default FULL). The first complete observation of the seed data
    set is the first seed; each later complete observation becomes a new
    seed while fewer than the maximum number of seeds are held and it lies
    at least the radius away from every seed held so far. Once the maximum
    is reached two replacement tests apply in turn: (1) if the observation
    is farther from its closest seed than the two closest seeds are from
    each other, it replaces one of those two, namely the one that would lie
    closer to the remaining seeds if the other were replaced by the
    observation; (2) otherwise it replaces its nearest seed if its smallest
    distance to the other seeds exceeds that seed's smallest distance to the
    other seeds. REPLACE=PART applies test 1 only, REPLACE=NONE neither.
  * SEED= data set (p. 2445): initial seeds are selected from its rows, in
    data-set order, by the same rules. For the packs' seed data sets (five
    quintile medians; then at most five first-pass cluster centers) with
    radius 0 the selection is the identity, but the rules are implemented in
    full (select_seeds) and exercised by the synthetic self-test below.
  * CONVERGE= (pp. 2440-2441): iteration stops when the largest seed
    movement, divided by the minimum distance among the initial seeds, is at
    most the criterion.
  * MAXITER= (p. 2443): each iteration assigns observations to the nearest
    seed and recomputes seeds as cluster means; MAXITER=0 skips the
    iterative phase.
  * STRICT=s (p. 2445; OUT= data set pp. 2448-2449): an observation whose
    distance to the nearest seed exceeds s is not assigned; in the output it
    carries the negative of the nearest cluster's number. Observations with
    a missing analysis variable get a missing cluster.

Arithmetic. The pipeline is IEEE double on purpose: the computation being
reproduced is a double-precision one. Exact statements about the resulting
partitions (optimality, sums of squares) belong to the exact layer in
src/dp_kmeans.py, which works in rational arithmetic.

Notes (points the documentation leaves open, each handled by a stated
convention and reported by a counter in every result so that a replay can say
whether the point was ever exercised on real data):

  1. Equidistant ties. An observation exactly equidistant (double equality)
     from two or more nearest seeds: not specified in the documentation; a
     stated convention (lower cluster index) with a counter that reports
     whether any tie occurred in a replay (tie_events_*). tie_rule="high" is
     the opposite convention, kept as a negative control for tests only.
  2. STRICT= during the iterative phase. Whether observations beyond the
     STRICT= distance are also left out of the iterative-phase means is not
     stated. Convention: STRICT applies in every assignment pass and
     excluded points contribute to no mean (strict_phase="all");
     strict_phase="final" is the alternative, kept so the difference can be
     measured. In the CMS driver the second pass starts from converged
     first-pass centers and the two conventions rarely separate.
  3. Empty cluster during iteration. No seeds are deleted by default
     (DELETE=, p. 2441) but the mean of an empty cluster is undefined.
     Convention: the seed is carried forward unchanged; counted in
     empty_cluster_events.
  4. Duplicate seeds with radius 0. "At least the radius" with radius 0
     admits a duplicate value as a seed. Implemented as documented (>=);
     duplicates are counted in duplicate_seed_values, and with converge=0 a
     duplicate seed yields an empty cluster (note 3) or a tie (note 1), both
     counted.
  5. Accumulation order. Cluster means are accumulated sequentially in
     data-set order in IEEE double; any other internal order would differ
     at the ulp level only.
  6. Quintile integrality test. Percentile definition 5 needs the test
     "fractional part of n*p equals 0"; it is evaluated exactly here
     (fractions.Fraction), which agrees with a double product for
     p in {1/5, 2/5, 3/5, 4/5, 1/2} and any realistic n.
  7. Distance comparisons. One-dimensional Euclidean distance is |x - seed|
     computed directly; comparing squared instead of unsquared distances
     could differ only at an ulp-level near-tie, which the tie counter of
     note 1 reports in practice.

The two-stage driver kmeans_star() follows CMS's published %kmeans macro
(Star_Macros.sas in the 2021-2025 Hospital Star Ratings SAS packages, the
same macro in all five years):
  - first-pass seeds: the 20th/40th/60th/80th sample percentiles (percentile
    definition 5, the PROC UNIVARIATE default) cut the scores into five
    groups (x <= P20, P20 < x <= P40, ..., P80 < x); the seed rows are the
    per-group medians (same definition) in ascending group order;
  - first pass: maxclusters=5 converge=0 maxiter=1000;
  - second-pass seeds: the first pass's cluster centers in cluster order;
    second pass: maxclusters=5 converge=0 maxiter=1000 strict=1;
  - re-inclusion: cluster = abs(cluster) for STRICT-excluded observations;
  - stars 1..k by ascending second-pass cluster mean (a stable sort, so
    exactly equal means keep cluster order; counted in mean_sort_ties).

References
  SAS Institute Inc., SAS/STAT 14.1 User's Guide, chapter "The FASTCLUS
    Procedure"; Base SAS Procedures Guide, "Calculating Percentiles"
    (percentile definitions).
  MacQueen, J. (1967). Some methods for classification and analysis of
    multivariate observations. Proc. Fifth Berkeley Symp. Math. Statist.
    Prob., 1, 281-297.
  Hartigan, J. A. and Wong, M. A. (1979). Algorithm AS 136: a k-means
    clustering algorithm. Applied Statistics 28, 100-108.
  Centers for Medicare & Medicaid Services, Overall Hospital Quality Star
    Rating SAS packages, 2021-2025 (Star_Macros.sas, %kmeans).

Run `python3 hospital_kmeans.py` for a self-test on synthetic data.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.1 and Appendix C.1
    (the k-means star step of the 2021-2025 packages, with the three unspecified
    cases carried as counters).
Run:  cd src && python3 hospital_kmeans.py   (synthetic self-test; imported by hospital/work/sas_replay.py)
Requires: Python >= 3.10; numpy.
"""

from fractions import Fraction

import numpy as np
import sys

__all__ = [
    "select_seeds", "fastclus", "quintile_median_seeds", "kmeans_star",
    "pctldef5_double", "FastclusResult",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seq_mean(values):
    """Sequential data-set-order accumulation in IEEE double (note 5)."""
    s = 0.0
    for v in values:
        s += v
    return s / len(values)


def _as_float_array(x, name):
    """Floats in, NaN = missing. Non-numeric input is refused with an
    explicit error, never silently coerced."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (this module implements the "
                         f"1-D star-assignment step), got shape {arr.shape}")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains +-inf; not a valid score")
    return arr


# ---------------------------------------------------------------------------
# seed selection (Background p. 2431; REPLACE= p. 2444; RADIUS= p. 2440)
# ---------------------------------------------------------------------------

def select_seeds(candidates, maxc, radius=0.0, replace="FULL"):
    """Initial seed selection over `candidates` in data-set order.

    candidates: values of the analysis variable in the seed (or input) data
    set, data-set order; NaN rows are incomplete and are never seeds (the
    documentation requires seeds to be complete observations, p. 2431).
    Returns (seeds, n_duplicate_seed_values). Seeds keep selection order:
    cluster numbers 1..k are seed order, NOT sorted order.

    replace: "FULL" (default; both documented replacement tests), "PART"
    (test 1 only), "NONE" (no replacement). REPLACE=RANDOM is refused: the
    CMS packs never use it and it would need a random-number stream.
    """
    if replace not in ("FULL", "PART", "NONE"):
        raise ValueError(f"REPLACE={replace!r} not implemented (FULL/PART/NONE)")
    if maxc < 1:
        raise ValueError(f"maxc must be >= 1, got {maxc}")
    cand = _as_float_array(candidates, "candidates")
    seeds = []
    for v in cand:
        if np.isnan(v):
            continue                      # incomplete observation: never a seed
        if not seeds:
            seeds.append(float(v))        # the first complete observation is
            continue                      # always the first seed
        d = [abs(v - s) for s in seeds]
        if len(seeds) < maxc:
            # a new seed must lie at least the radius from all previous seeds
            if min(d) >= radius:
                seeds.append(float(v))
            continue
        if replace == "NONE":
            continue
        # replacement test 1 (p. 2431, paraphrased in the module docstring):
        # an old seed is replaced if dist(obs, closest seed) exceeds the
        # minimum separation between the current seeds
        k = len(seeds)
        sep = [(abs(seeds[i] - seeds[j]), i, j)
               for i in range(k) for j in range(i + 1, k)]
        min_sep, pi, pj = min(sep)
        if min(d) > min_sep:
            # replace one of the two closest seeds (pi, pj): the one with the
            # SHORTER distance to the closest of the remaining seeds when the
            # OTHER is replaced by the current observation
            def dist_to_rest(keep, drop):
                rest = ([seeds[t] for t in range(k) if t not in (keep, drop)]
                        + [float(v)])
                return min(abs(seeds[keep] - r) for r in rest)
            di = dist_to_rest(pi, pj)     # pi stays, pj replaced by obs
            dj = dist_to_rest(pj, pi)     # pj stays, pi replaced by obs
            # the one with the SHORTER such distance is the one replaced;
            # di == dj is not specified; convention: later of the pair
            drop = pi if di < dj else pj
            seeds[drop] = float(v)
            continue
        if replace == "PART":
            continue
        # replacement test 2: obs replaces its NEAREST seed if its smallest
        # distance to all OTHER seeds exceeds that seed's shortest distance
        # to the other seeds
        near = int(np.argmin(d))          # first minimum: tie convention, note 1
        d_others = [d[t] for t in range(k) if t != near]
        seed_to_others = [abs(seeds[near] - seeds[t]) for t in range(k) if t != near]
        if d_others and min(d_others) > min(seed_to_others):
            seeds[near] = float(v)
    dup = len(seeds) - len(set(seeds))
    return seeds, dup


# ---------------------------------------------------------------------------
# core k-means pass (the FASTCLUS computational method, 1-D)
# ---------------------------------------------------------------------------

class FastclusResult(dict):
    """Dict with attribute access; keys documented in fastclus()."""
    __getattr__ = dict.__getitem__


def _assign_pass(xv, seeds, tie_rule):
    """Nearest-seed assignment. Returns (idx0, dist, n_ties).
    idx0: 0-based nearest-seed index; ties -> lowest cluster number
    (numpy argmin first occurrence), the stated convention of note 1;
    tie_rule="high" is the negative control (highest cluster number wins)."""
    d = np.abs(xv[:, None] - seeds[None, :])
    if tie_rule == "low":
        idx = np.argmin(d, axis=1)
    elif tie_rule == "high":
        idx = (d.shape[1] - 1) - np.argmin(d[:, ::-1], axis=1)
    else:
        raise ValueError(f"tie_rule must be 'low' or 'high', got {tie_rule!r}")
    dist = d[np.arange(len(xv)), idx]
    n_ties = int(((d == dist[:, None]).sum(axis=1) > 1).sum())
    return idx, dist, n_ties


def fastclus(x, seeds, maxc=None, radius=0.0, replace="FULL",
             converge=0.0, maxiter=1, strict=None,
             tie_rule="low", strict_phase="all"):
    """One FASTCLUS-style k-means run on 1-D data.

    x       : analysis-variable values in data-set order (floats; NaN =
              missing -> cluster None, excluded from everything).
    seeds   : seed data-set values in data-set order (in the CMS driver: the
              quintile medians, then the first-pass cluster centers). The
              seed SELECTION rules run over these rows (select_seeds).
    maxc    : MAXCLUSTERS=; required explicitly when len(seeds) exceeds the
              desired k (the CMS packs always say maxc=5). Defaults to
              len(selected seeds).
    converge/maxiter : CONVERGE= and MAXITER= (CMS packs: 0 and 1000). The
              default maxiter=1 is the documented default without LEAST=.
    strict  : STRICT= value or None (option absent). Distance > strict
              (STRICTLY greater, "exceeds") -> not assigned; output cluster
              is -(nearest cluster). strict_phase: "all" (convention, note
              2) applies it in iterative passes too; "final" = alternative.
    tie_rule: "low" (convention, note 1) / "high" (negative control).

    Returns FastclusResult with:
      cluster   : list, per x row: +c assigned to cluster c (1-based),
                  -c STRICT-excluded with nearest cluster c, None missing x
      distance  : |x - final seed of nearest cluster| (None for missing x)
      initial_seeds, final_seeds : lists, cluster order
      means     : per-cluster means of POSITIVELY assigned observations in
                  the final pass (the "cluster centers"; equal to
                  final_seeds at complete convergence); an empty final
                  cluster carries its seed (note 3)
      freqs     : per-cluster positively-assigned counts (final pass)
      n_iter, converged : iterative-phase status
      tie_events_iter, tie_events_final : note-1 counters
      strict_excluded : count of negative clusters in the final pass
      empty_cluster_events : note-3 counter (iterative phase)
      duplicate_seed_values : note-4 counter
    """
    xa = _as_float_array(x, "x")
    if strict is not None and strict < 0:
        raise ValueError(f"strict must be nonnegative, got {strict}")
    if converge < 0:
        raise ValueError(f"converge must be nonnegative, got {converge}")
    sel, dup = select_seeds(seeds, maxc if maxc is not None else len(seeds),
                            radius=radius, replace=replace)
    if not sel:
        raise ValueError("no complete observation available as a seed")
    k = len(sel)
    seeds0 = np.asarray(sel, dtype=np.float64)
    obs = ~np.isnan(xa)
    xv = xa[obs]
    if xv.size == 0:
        raise ValueError("all observations missing; nothing to cluster")

    # CONVERGE= scaling factor: the minimum distance between the initial
    # seeds (p. 2441). k==1 or duplicate seeds -> scale 0 -> the test
    # delta <= converge*scale reduces to the exact fixed point, which is
    # also precisely what converge=0 (the CMS packs' setting) asks for.
    if k >= 2:
        scale = min(abs(seeds0[i] - seeds0[j])
                    for i in range(k) for j in range(i + 1, k))
    else:
        scale = 0.0

    seeds_cur = seeds0.copy()
    n_iter = 0
    converged = (maxiter == 0)  # MAXITER=0: iterative phase skipped entirely
    tie_iter = 0
    empty_events = 0
    for _ in range(maxiter):
        idx, dist, nt = _assign_pass(xv, seeds_cur, tie_rule)
        tie_iter += nt
        if strict is not None and strict_phase == "all":
            in_cluster = dist <= strict     # "exceeds" = strictly greater
        else:
            in_cluster = np.ones(len(xv), dtype=bool)
        new_seeds = seeds_cur.copy()
        for c in range(k):
            m = in_cluster & (idx == c)
            if m.any():
                new_seeds[c] = _seq_mean(xv[m])   # seeds recomputed as the
            else:                                 # means of the clusters
                empty_events += 1                 # note 3: carry seed
        n_iter += 1
        delta = float(np.max(np.abs(new_seeds - seeds_cur)))
        seeds_cur = new_seeds
        if delta <= converge * scale:
            converged = True
            break

    # final pass: every observation assigned to the nearest seed
    idx, dist, tie_final = _assign_pass(xv, seeds_cur, tie_rule)
    cluster_v = idx + 1
    if strict is not None:
        excl = dist > strict
        cluster_v = np.where(excl, -cluster_v, cluster_v)
    else:
        excl = np.zeros(len(xv), dtype=bool)

    means = []
    freqs = []
    for c in range(k):
        m = (~excl) & (idx == c)
        freqs.append(int(m.sum()))
        means.append(_seq_mean(xv[m]) if m.any() else float(seeds_cur[c]))

    cluster = [None] * len(xa)
    distance = [None] * len(xa)
    pos = np.nonzero(obs)[0]
    for i, ci, di in zip(pos, cluster_v, dist):
        cluster[int(i)] = int(ci)
        distance[int(i)] = float(di)

    return FastclusResult(
        cluster=cluster, distance=distance,
        initial_seeds=[float(s) for s in seeds0],
        final_seeds=[float(s) for s in seeds_cur],
        means=means, freqs=freqs,
        n_iter=n_iter, converged=bool(converged),
        tie_events_iter=int(tie_iter), tie_events_final=int(tie_final),
        strict_excluded=int(excl.sum()),
        empty_cluster_events=int(empty_events),
        duplicate_seed_values=int(dup),
    )


# ---------------------------------------------------------------------------
# %kmeans first-pass seeding (CMS Star_Macros.sas: quintile groups, medians)
# ---------------------------------------------------------------------------

def pctldef5_double(sorted_vals, p_num, p_den):
    """Sample percentile, definition 5 of the Base SAS Procedures Guide
    ("Calculating Percentiles"), on IEEE doubles: write n*p = j + g with j
    the integer part; the value is (x_j + x_{j+1})/2 if g == 0, else
    x_{j+1} (1-based order statistics). The integrality test g == 0 is
    evaluated exactly (Fraction n*p; note 6); the average is taken in
    double because the reproduced pipeline is double. sorted_vals:
    ascending nonmissing doubles."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("no nonmissing values")
    np_exact = Fraction(n) * Fraction(p_num, p_den)
    j = int(np_exact)           # integer part
    g = np_exact - j            # fractional part
    if g == 0:
        hi = min(j + 1, n)      # x_{n+1} guard: p<1 keeps j+1<=n; stated
        return (sorted_vals[j - 1] + sorted_vals[hi - 1]) / 2.0
    return sorted_vals[j]       # x_{j+1}, 1-based


def quintile_median_seeds(x):
    """First-pass seed rows as CMS's %kmeans macro builds them: the 20th to
    80th percentiles (definition 5, the PROC UNIVARIATE default) of the
    nonmissing scores cut them into five groups with the macro's boundary
    conditions ( x <= P20 | P20 < x <= P40 | P40 < x <= P60 |
    P60 < x <= P80 | P80 < x ); the seed for each group is its median (same
    definition, as PROC MEANS computes it per class level), groups in
    ascending order.
    Returns (seeds, record): seeds in group order, EMPTY groups omitted
    (an absent class level produces no seed row); record carries the P
    values and group sizes."""
    xa = _as_float_array(x, "x")
    xv = np.sort(xa[~np.isnan(xa)])
    if xv.size == 0:
        raise ValueError("all scores missing")
    P = {q: pctldef5_double(xv, q, 100) for q in (20, 40, 60, 80)}
    grp = np.full(len(xa), -1, dtype=int)
    nn = ~np.isnan(xa)
    grp[nn & (xa <= P[20])] = 1
    grp[nn & (P[20] < xa) & (xa <= P[40])] = 2
    grp[nn & (P[40] < xa) & (xa <= P[60])] = 3
    grp[nn & (P[60] < xa) & (xa <= P[80])] = 4
    grp[nn & (P[80] < xa)] = 5
    seeds = []
    sizes = {}
    for g in (1, 2, 3, 4, 5):
        vals = np.sort(xa[grp == g])
        sizes[g] = int(vals.size)
        if vals.size:
            seeds.append(pctldef5_double(vals, 1, 2))  # median = p=1/2
    return seeds, {"P20": P[20], "P40": P[40], "P60": P[60], "P80": P[80],
                   "grp_sizes": sizes}


# ---------------------------------------------------------------------------
# %kmeans two-stage driver (CMS Star_Macros.sas, 2021-2025 SAS packages)
# ---------------------------------------------------------------------------

def kmeans_star(x, tie_rule="low", strict_phase="all"):
    """The CMS %kmeans macro end to end: scores -> stars 1..k.

    Stage 1: fastclus(seeds=quintile medians, maxc=5, converge=0,
             maxiter=1000)
    Stage 2: fastclus(seeds=stage-1 cluster centers, maxc=5, converge=0,
             maxiter=1000, strict=1)
    Post:    cluster = abs(cluster) re-inclusion of STRICT exclusions;
             stars = 1..k by ascending stage-2 cluster mean (a stable sort:
             exactly equal means keep cluster order).

    Returns FastclusResult with: star (list, per x row; None for missing x),
    stage1, stage2 (full fastclus results), the seed record, and the note
    counters rolled up (tie_events_total, mean_sort_ties).
    """
    seeds1, seed_rec = quintile_median_seeds(x)
    st1 = fastclus(x, seeds=seeds1, maxc=5, converge=0.0, maxiter=1000,
                   tie_rule=tie_rule)
    seeds2 = st1.means                       # cluster centers, cluster order
    st2 = fastclus(x, seeds=seeds2, maxc=5, converge=0.0, maxiter=1000,
                   strict=1.0, tie_rule=tie_rule, strict_phase=strict_phase)
    k = len(st2.means)
    order = sorted(range(k), key=lambda c: (st2.means[c], c))  # stable
    mean_sort_ties = len(st2.means) - len(set(st2.means))
    star_of_cluster = {c + 1: r + 1 for r, c in enumerate(order)}
    star = [None if c is None else star_of_cluster[abs(c)]  # abs() re-inclusion
            for c in st2.cluster]
    return FastclusResult(
        star=star, stage1=st1, stage2=st2,
        seeds1=seeds1, seed_record=seed_rec,
        seeds2=[float(s) for s in seeds2],
        star_of_cluster=star_of_cluster,
        tie_events_total=(st1.tie_events_iter + st1.tie_events_final
                          + st2.tie_events_iter + st2.tie_events_final),
        mean_sort_ties=int(mean_sort_ties),
        strict_excluded=st2.strict_excluded,
        converged=bool(st1.converged and st2.converged),
    )


# ---------------------------------------------------------------------------
# self-test on synthetic data (no external inputs)
# ---------------------------------------------------------------------------

def _selftest():
    """Synthetic checks of the documented rules and the stated conventions.
    Uses made-up numbers only; prints PASS and returns 0 on success."""
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    # 1. seed selection: identity below maxc; NaN skipped; radius; the two
    #    replacement tests
    s, dup = select_seeds([1.0, float("nan"), 2.0, 3.0], maxc=5)
    ok("select identity, NaN skipped", s == [1.0, 2.0, 3.0] and dup == 0)
    s, dup = select_seeds([1.0, 1.0, 2.0], maxc=3)
    ok("radius 0 admits a duplicate seed, counted", s == [1.0, 1.0, 2.0] and dup == 1)
    s, _ = select_seeds([1.0, 1.5, 4.0], maxc=3, radius=1.0)
    ok("radius excludes a close candidate", s == [1.0, 4.0])
    # test 1: seeds 0,1,10 (min separation 1); obs 20 is 10 from its closest
    # seed > 1 -> replaces one of {0,1}: keeping 0 and dropping 1 leaves 0 at
    # distance 10 from the rest; keeping 1 leaves 1 at distance 9; the one
    # with the shorter distance (1) is replaced
    s, _ = select_seeds([0.0, 1.0, 10.0, 20.0], maxc=3)
    ok("replacement test 1", s == [0.0, 20.0, 10.0])
    s, _ = select_seeds([0.0, 1.0, 10.0, 20.0], maxc=3, replace="NONE")
    ok("REPLACE=NONE keeps the first maxc seeds", s == [0.0, 1.0, 10.0])
    # test 2: seeds 0,3,10 (min separation 3); obs 6 has distances (6,3,4):
    # its closest seed is at 3, not > 3, so test 1 does not fire; its
    # nearest seed is 3, its smallest distance to the other seeds is
    # min(6,4)=4, and seed 3's smallest distance to the other seeds is
    # min(3,7)=3; 4 > 3, so the observation replaces seed 3
    s, _ = select_seeds([0.0, 3.0, 10.0, 6.0], maxc=3)
    ok("replacement test 2", s == [0.0, 6.0, 10.0])
    s, _ = select_seeds([0.0, 3.0, 10.0, 6.0], maxc=3, replace="PART")
    ok("REPLACE=PART skips test 2", s == [0.0, 3.0, 10.0])

    # 2. one k-means run converging on three separated groups
    x = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 9.0, 9.2, float("nan")]
    r = fastclus(x, seeds=[0.0, 5.0, 9.0], maxc=3, converge=0.0, maxiter=100)
    ok("converged", r.converged and r.n_iter >= 1)
    ok("clusters", r.cluster == [1, 1, 1, 2, 2, 2, 3, 3, None])
    ok("means equal final seeds at convergence", r.means == r.final_seeds)
    ok("freqs", r.freqs == [3, 3, 2])
    ok("no ties, no empties", r.tie_events_final == 0 and r.empty_cluster_events == 0)
    # MAXITER=0: no iteration, final pass on the initial seeds
    r0 = fastclus(x, seeds=[0.0, 5.0, 9.0], maxc=3, maxiter=0)
    ok("maxiter 0", r0.n_iter == 0 and r0.final_seeds == [0.0, 5.0, 9.0])

    # 3. STRICT: a far point gets a negative cluster and is left out of the mean
    xs = [0.0, 0.2, 5.0, 5.2, 8.0]
    r = fastclus(xs, seeds=[0.1, 5.1], maxc=2, converge=0.0, maxiter=100, strict=1.0)
    ok("strict flags the far point", r.cluster == [1, 1, 2, 2, -2] and r.strict_excluded == 1)
    ok("strict point excluded from the mean", abs(r.means[1] - 5.1) < 1e-12)
    rf = fastclus(xs, seeds=[0.1, 5.1], maxc=2, converge=0.0, maxiter=100,
                  strict=1.0, strict_phase="final")
    ok("strict_phase final differs only through the iterative means",
       rf.cluster[4] == -2 and rf.final_seeds[1] != r.final_seeds[1])

    # 4. tie convention and its counter
    rt = fastclus([1.0, 3.0, 2.0], seeds=[1.0, 3.0], maxc=2, maxiter=0)
    ok("tie -> lower cluster index, counted", rt.cluster == [1, 2, 1] and rt.tie_events_final == 1)
    rh = fastclus([1.0, 3.0, 2.0], seeds=[1.0, 3.0], maxc=2, maxiter=0, tie_rule="high")
    ok("negative control assigns the tie the other way", rh.cluster == [1, 2, 2])

    # 5. percentile definition 5 and quintile-median seeds
    v = [float(i) for i in range(1, 11)]           # n=10
    ok("pctl def 5, g==0 averages", pctldef5_double(v, 20, 100) == 2.5)
    ok("pctl def 5, g>0 takes x_{j+1}", pctldef5_double(v[:9], 20, 100) == 2.0)
    ok("median", pctldef5_double(v, 1, 2) == 5.5 and pctldef5_double(v[:9], 1, 2) == 5.0)
    seeds, rec = quintile_median_seeds(v)
    ok("quintile medians of 1..10", seeds == [1.5, 3.5, 5.5, 7.5, 9.5]
       and rec["grp_sizes"] == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})

    # 6. two-stage driver on a synthetic 50-point sample: five ordered stars
    rng = np.random.RandomState(12345)
    centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
    sample = np.concatenate([c + 0.15 * rng.standard_normal(10) for c in centers])
    rng.shuffle(sample)
    sample = list(sample) + [float("nan")]
    ks = kmeans_star(sample)
    stars = ks.star
    ok("driver: 5 clusters, both stages converged", len(ks.stage2.means) == 5 and ks.converged)
    ok("driver: missing score -> no star", stars[-1] is None)
    ok("driver: every star 1..5 used", sorted(set(s for s in stars if s is not None)) == [1, 2, 3, 4, 5])
    # stars must be monotone in the score (contiguous clusters, ordered means)
    pairs = sorted((sc, st) for sc, st in zip(sample, stars) if st is not None)
    ok("driver: stars nondecreasing in score", all(a[1] <= b[1] for a, b in zip(pairs, pairs[1:])))
    ok("driver: star k <-> k-th smallest stage-2 mean",
       [ks.star_of_cluster[c + 1] for c in np.argsort(ks.stage2.means)] == [1, 2, 3, 4, 5])
    ok("driver: counters present and zero on separated data",
       ks.tie_events_total == 0 and ks.mean_sort_ties == 0)

    failed = [n for n, c in checks if not c]
    for n, c in checks:
        print(("ok   " if c else "FAIL ") + n)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks; " + ("PASS" if not failed else "FAIL"))
    return 0 if not failed else 1


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
