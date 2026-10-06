# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Per-measure cut-point pipeline: outlier-bounds trim -> GROUPS=10 resampling
groups -> per-group clustering -> cut points -> mean over groups. Two arms:

  ward arm : IEEE-double Ward from the documented algorithm (ward_cluster) — the published-side replay.
  dp arm   : exact-rational optimal 1-D SSQ partition (dp_kmeans) —
             the counterfactual in which the clustering step minimizes
             the within-cluster sum of squares the Tech Notes describe.

Everything before and after the clustering step is identical between
arms (same outlier trim, same group assignment, same cut-point extraction, same
mean), so any final difference is attributable to the clustering step alone.

Improvement measures: group assignment on the whole measure dataset, then within
each group the scores are split at zero; >=0 side clustered with NSTARS=3 (stars
3/4/5), <0 side with NSTARS=2 (stars 1/2); the 3-star lower threshold is 0 by
construction (Tech Notes p.151-152). Reading: SURVEYSELECT runs once per
measure before the leave-one-out loop; the split is inside the per-group steps.
(Alternative readings are tested by the fidelity battery, not here.)

Cut-point precision: group cut points are data values (exact decimal strings ->
Fraction). The Ward arm produces them via double clustering but the values are
published decimals, re-lifted to Fraction for the mean, so both arms' means are
exact Fractions and the arm comparison is exact.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Sections 2.4 and 4.3 (the
    resampled cut-point pipeline with Ward's method or the exact optimum inside
    the ten groups).
Run:  cd src && python3 pipeline.py   (synthetic end-to-end self-test; imported by the census
      scripts)
Requires: Python >= 3.10; numpy, scipy.
"""

from fractions import Fraction

from dp_kmeans import dp_optimal_partition, _to_fractions
from tukey import tukey_trim
from ward_cluster import ward_cut, cutpoints_from_clusters, near_tie_scan
from ward_cluster_tie import ward_sas_cut
import groups_assign
import sys


def _fold_cutpoints_ward(fold_scores_str, k, higher_is_better, scan_ties=False,
                         engine="scipy"):
    """engine='scipy': NN-chain double Ward (tie order not SAS's).
    engine='sas_tie': documented SAS tie rule (D, maxID, minID) on the group's
    dataset order — the published-side arm."""
    vals = [float(s) for s in fold_scores_str]
    if engine == "sas_tie":
        clusters = ward_sas_cut(vals, k)
    else:
        clusters = ward_cut(vals, k)
    cps_f = cutpoints_from_clusters(clusters, vals, higher_is_better)
    # map back to exact decimal strings: each cut point is some data value
    cps = []
    for c in cps_f:
        # find a source string equal to this double (must exist)
        m = next(s for s, v in zip(fold_scores_str, vals) if v == c)
        cps.append(Fraction(m))
    nt = near_tie_scan(vals, k)[0] if scan_ties else None
    return cps, nt


def _fold_cutpoints_dp(fold_scores_str, k, higher_is_better):
    xs = _to_fractions(fold_scores_str)
    kk = min(k, len(set(xs)))
    if kk == 0:
        return []
    r = dp_optimal_partition(xs, kk)
    blocks = r["blocks"]  # ascending
    if higher_is_better:
        return [b[0] for b in blocks[1:]]  # min of categories 2..K
    return [b[-1] for b in blocks[:-1]]  # max of categories 1..K-1


def _cluster_one_fold(fold_scores_str, arm, k, higher_is_better, improvement, scan_ties):
    """Returns (list of cut points as Fractions, near_tie_count or None).

    Non-improvement: k categories -> k-1 cut points.
    Improvement: [neg-side 1|2 cut] + [0 implicit] + [pos-side 3|4, 4|5 cuts];
    returned as [cut12, cut34, cut45] (the 2|3 boundary is the constant 0)."""
    ward_engine = "sas_tie" if arm == "ward_tie" else "scipy"
    is_ward = arm in ("ward", "ward_tie")
    if not improvement:
        if is_ward:
            return _fold_cutpoints_ward(fold_scores_str, k, higher_is_better,
                                        scan_ties, engine=ward_engine)
        return _fold_cutpoints_dp(fold_scores_str, k, higher_is_better), None
    neg = [s for s in fold_scores_str if Fraction(s) < 0]
    pos = [s for s in fold_scores_str if Fraction(s) >= 0]
    nt_total = 0
    out = []
    for side, kk in ((neg, 2), (pos, 3)):
        if len(side) == 0:
            continue
        kk_eff = min(kk, len(set(side)))
        if is_ward:
            cps, nt = _fold_cutpoints_ward(side, kk_eff, True, scan_ties,
                                           engine=ward_engine)
            nt_total += nt or 0
        else:
            cps = _fold_cutpoints_dp(side, kk_eff, True)
        out.extend(cps)
    return out, (nt_total if is_ward and scan_ties else None)


def measure_cutpoints(scores_str, *, higher_is_better=True, k=5, improvement=False,
                      cap_lo=None, cap_hi=None, tukey=True,
                      groups_candidate="seq_quota_last", seed=8675309, n_groups=10,
                      arm="ward", scan_ties=False):
    """Full pre-guardrail pipeline for one measure.

    scores_str: decimal-string scores in dataset order (the order SURVEYSELECT
    sees — normally contract_id ascending; the order is itself a tested
    input). Returns dict with group cut points, mean cut points (Fractions),
    kept mask, outlier bounds, near-tie counts.
    """
    if tukey:
        mask, lo, hi = tukey_trim(scores_str, cap_lo, cap_hi)
    else:
        mask, lo, hi = [True] * len(scores_str), None, None
    kept = [s for s, m in zip(scores_str, mask) if m]
    n = len(kept)
    gfn = groups_assign.CANDIDATES[groups_candidate]
    gids = gfn(n, n_groups, seed)
    folds = []
    near_ties = []
    for g in range(1, n_groups + 1):
        fold = [s for s, gid in zip(kept, gids) if gid != g]
        cps, nt = _cluster_one_fold(fold, arm, k, higher_is_better, improvement, scan_ties)
        folds.append(cps)
        near_ties.append(nt)
    # mean per boundary; groups may (rarely) yield differing counts if a group's
    # equal-range combining removed a boundary — flag rather than average
    counts = {len(f) for f in folds}
    if len(counts) != 1:
        return {"error": f"group cut-point counts differ: {sorted(counts)}",
                "folds": folds, "kept_mask": mask, "fences": (lo, hi)}
    nb = counts.pop()
    means = [sum(f[i] for f in folds) / len(folds) for i in range(nb)]
    return {"fold_cutpoints": folds, "mean_cutpoints": means,
            "kept_mask": mask, "fences": (lo, hi), "near_ties": near_ties}


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    import random

    random.seed(11)
    # synthetic end-to-end: 500 contracts, 5 latent bands, integer percent display
    scores = []
    for _ in range(500):
        band = random.choice([55, 65, 75, 85, 93])
        scores.append(str(max(0, min(100, band + random.randint(-6, 6)))))
    rw = measure_cutpoints(scores, arm="ward", scan_ties=True)
    rd = measure_cutpoints(scores, arm="dp")
    assert "error" not in rw and "error" not in rd
    assert len(rw["mean_cutpoints"]) == 4 and len(rd["mean_cutpoints"]) == 4
    print("ward mean cps:", [f"{float(c):.2f}" for c in rw["mean_cutpoints"]],
          "near-ties/group:", rw["near_ties"])
    print("dp   mean cps:", [f"{float(c):.2f}" for c in rd["mean_cutpoints"]])
    # improvement-measure plumbing
    imp = [f"{random.uniform(-0.4, 0.6):.6f}" for _ in range(400)]
    ri = measure_cutpoints(imp, improvement=True, arm="ward")
    rj = measure_cutpoints(imp, improvement=True, arm="dp")
    assert "error" not in ri and len(ri["mean_cutpoints"]) == 3
    assert "error" not in rj and len(rj["mean_cutpoints"]) == 3
    print("improvement ward:", [f"{float(c):.6f}" for c in ri["mean_cutpoints"]])
    print("improvement dp  :", [f"{float(c):.6f}" for c in rj["mean_cutpoints"]])
    # verify exact comparability of the two arms
    diffs = [float(a - b) for a, b in zip(rw["mean_cutpoints"], rd["mean_cutpoints"])]
    print("ward-dp mean cut deltas:", [f"{d:+.3f}" for d in diffs])
    print("pipeline: synthetic end-to-end OK")
