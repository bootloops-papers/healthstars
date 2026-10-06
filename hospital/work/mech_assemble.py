#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Assemble runs/mechanism_allupward.json from the stage output files.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.3 (mechanism and
    direction of the differences).
Run:  cd hospital/work && python3 mech_assemble.py   (after the four mech_*.py stage scripts)
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import json

BASE = _REC + "/hospital"
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

s1 = json.load(open(f"{BASE}/work/mech_stage1_boundaries.json"))
s2a = json.load(open(f"{BASE}/work/mech_stage2a_synthetic.json"))
s2b = json.load(open(f"{BASE}/work/mech_stage2b_2025.json"))
s3 = json.load(open(f"{BASE}/work/mech_stage3_worked.json"))
s3c = json.load(open(f"{BASE}/work/mech_stage3_worked_clean.json"))

out = {
    "run": "mechanism_allupward",
    "question": "why do all 213 star differences move upward when the exact 1-D "
                "k-means optimum replaces CMS's two-stage Hartigan-Wong "
                "k-means (2026 release): necessity, seeding artifact, or "
                "data-shape coincidence?",
    "verdict": {
        "one_line": "Not a necessity of the algorithm and not a coincidence: "
                    "CMS's seeded local optimizer displaces every cut coherently "
                    "to one side of the exact optimum, and in 2026 all three "
                    "score distributions are left-skewed enough that the displaced "
                    "side is the high side in every group, so correcting to the "
                    "optimum can only lower cuts and raise stars. Mirrored "
                    "(right-skewed) data reverses the direction, and the weakly "
                    "skewed 2025 groups show the contingency: per-group "
                    "one-directional deltas but not globally upward (33 up, 5 down).",
        "mechanism_class": "seeding-selected local optimum x one-signed skew of the "
                           "score distribution (deterministic given the data; no RNG)",
        "necessity": False,
        "coincidence": False,
        "per_group_coherence": "in all 6 group-years examined (3x2026, 3x2025) the "
            "cut displacements are one-signed within the group (never mixed), so "
            "corrections within a group all move the same way; the sign follows "
            "the skew when |skew| is large (2026: -0.80/-0.69/-0.31 all up; 2025 "
            "peer3: -1.21 all up) and is small and either-signed near symmetry "
            "(2025 peer4 -0.08: down; peer5 -0.09: up)",
        "evidence_keys": ["stage1_boundary_dissection", "stage2a_necessity_tests",
                          "stage2b_2025_cross_year", "stage3_skew_and_worked_example"],
    },

    "shipped_algorithm_characterization": {
        "source": "R pack '2 - Second Stage_Weighted Average and Categorize "
                  "Star_2026Apr.R', fn_kmeans (lines 136-179)",
        "procedure": [
            "stage-1 seeds: medians of the five quintile groups of summary_score "
            "(quantile() default type 7; groups cut at P20/P40/P60/P80 with <=)",
            "stage 1: kmeans(x, seeds, iter.max=1000), Hartigan-Wong default",
            "stage 2: kmeans(x, stage1 centers, iter.max=1000)",
            "clusters ordered by center -> stars 1..5; safety cap applied later "
            "(outside this comparison)"],
        "note_2025_variant": "the strict=1-exclusion-then-abs()-re-inclusion detail "
            "belongs to the 2025 SAS-pack variant (CMS Star_Macros.sas %kmeans, PROC FASTCLUS "
            "STRICT= option; per-group counts in hospital/runs/sas_replay_2025.json "
            "peer_groups.*.strict_excluded_reincluded), not the 2026 R code; the 2026 pack dropped STRICT",
        "replica_validation": {g: s2a["A_replica_validation"][g]["mismatch_vs_shipped"]
                               for g in ["peer3", "peer4", "peer5"]},
        "two_stage_is_inert_2026": "stage 2 changes nothing in any peer group: "
            "identical centers, 0 label changes (replica logs); stage 1 already "
            "converges after 2 Hartigan-Wong iterations, so the published partition is "
            "the Hartigan-Wong fixed point nearest the quintile-median seeds",
        "replica_logs": {g: s2a["A_replica_validation"][g]["replica_log"]
                         for g in ["peer3", "peer4", "peer5"]},
    },

    "stage1_boundary_dissection": {
        "register": "decimal-string scores of R write.csv (same as dp_taste_2026); "
                    "cut = open interval (max of lower block, min of upper block), "
                    "shift = published midpoint - DP midpoint",
        "summary": s1["_summary"],
        "all_12_boundaries_shifted": "up (published above DP), 12 of 12; crossing "
            "counts reconcile exactly with the 213 (55+68+90)",
        "per_group": {g: {"boundaries": s1[g]["boundaries"],
                          "shipped_block_sizes": s1[g]["shipped_block_sizes"],
                          "dp_block_sizes": s1[g]["dp_block_sizes"],
                          "dp_centers": s1[g]["dp_centers"]}
                      for g in ["peer3", "peer4", "peer5"]},
        "centers_reading": "all 15 converged published centers sit above the "
            "corresponding DP-optimal centers; the quintile-median seeds start the "
            "bottom centers far above the optimal bottom centers (peer3 seed1 is "
            "+1.14 above DP center1) because equal-mass seeding places centers "
            "proportional to data density while the SSQ optimum pushes centers out "
            "into the sparse low tail",
    },

    "stage2a_necessity_tests": {
        "mirror_test": {
            "design": "negate the real 2026 peer3/peer4 scores (exact sign flip), "
                      "run the same CMS procedure (R replica) and the same DP",
            "peer3": {k: s2a["B_mirror"]["peer3"][k] for k in
                      ["moment_skewness", "moves", "moves_up", "moves_down"]},
            "peer4": {k: s2a["B_mirror"]["peer4"][k] for k in
                      ["moment_skewness", "moves", "moves_up", "moves_down"]},
            "reading": "all 55 (peer3) and 68 (peer4) corrections now move down, "
                       "the exact mirror of the real data's all-up. Constructive "
                       "disproof of necessity: the procedure is equivariant under "
                       "x -> -x, so 'all corrections upward' cannot be a property "
                       "of the seeding alone; it is a property of the seeding "
                       "together with the skew direction of the data.",
        },
        "fresh_counterexamples": [
            {k: c[k] for k in ["note", "n", "moment_skewness", "moves",
                               "moves_up", "moves_down", "shipped_ssq",
                               "optimal_ssq", "values"]}
            for c in s2a["C_counterexamples"]],
        "skew_direction_batch": s2a["D_skew_batch"],
        "skew_batch_reading": "direction of all corrections followed the skew sign "
            "in 33/33 non-optimal synthetic datasets (0 mixed-direction datasets); "
            "left-skew (mass high, like CMS scores): 627 moves all up; right-skew: "
            "989 moves all down. 4/20 and 3/20 datasets landed exactly optimal.",
        "dp_center_check": {k: s2a["E_dp_center_check"][k] for k in
                            ["group", "labels_match_dp_optimum",
                             "seeded_run_is_optimal", "reading"]},
    },

    "stage2b_2025_cross_year": s2b,

    "stage3_skew_and_worked_example": {
        "skew_per_peer_group_2026": {g: s1[g]["skew"] for g in
                                     ["peer3", "peer4", "peer5"]},
        "skew_per_peer_group_2025": {g: {"moment_skewness": s2b[g]["moment_skewness"],
                                         "mean_minus_median": s2b[g]["mean_minus_median"]}
                                     for g in ["peer3", "peer4", "peer5"]},
        "mechanism_statement": [
            "1. The summary-score distributions are left-skewed in every peer group "
            "and both years (2026 moment skew -0.80/-0.69/-0.31; 2025 similar): a "
            "long sparse tail of low scores under a dense mass of high scores "
            "(low-tail span P0-P20 is 1.7-2.2x the high-tail span P80-P100).",
            "2. Quintile-median seeding places the 5 initial centers at equal-count "
            "positions - center density proportional to the data density f. The SSQ "
            "optimum instead spreads centers toward sparse regions (1-D quantization: "
            "optimal center density ~ f^(1/3), flatter than f), so the optimal bottom "
            "centers/cuts sit far deeper in the low tail than any equal-mass position.",
            "3. Hartigan-Wong is a descent method: it stops at the first fixed point "
            "downhill from its seeds. Seeded on the dense side of the optimum it "
            "converges (in 2 iterations here) to a local optimum whose centers and "
            "cuts all remain on the dense side; with one-signed skew, all cuts are "
            "displaced the same way (12/12 above the optimum).",
            "4. Star direction is the mirror of cut displacement: every hospital "
            "between a published cut and the lower optimal cut gains a star when the "
            "optimum replaces the published partition. Left-skew => dense side is the "
            "high-score side => published cuts too high => corrections all up. On "
            "right-skewed data the same procedure errs low and corrections all go "
            "down (mirror test + 2 explicit counterexamples + 17/17 batch).",
            "5. The 2025 release confirms the contingency on real data: its peer4 "
            "and peer5 scores are nearly symmetric (skew -0.08/-0.09), the "
            "published-vs-optimum deltas shrink to 1.3% (38/2,872 vs 6.7% in "
            "2026), and the direction is no longer globally pinned: peer4's 5 "
            "deltas all move down while peer3 (-1.21) and peer5 move up. "
            "Within-group one-directionality persists in all six group-years.",
        ],
        "worked_example": {
            "note": "n=15, all values distinct, deep left tail (6.237 under dense "
                    "mass 9.0-9.8); the CMS procedure (R kmeans, replica "
                    "script) converges in 1 iteration to SSQ 1.4979 vs optimum "
                    "0.1322: equal-mass seeding forces the bottom cluster to span "
                    "tail+shoulder (3 points) while the optimum spends a whole "
                    "cluster on the lone tail point; 9 of 15 points move, all up; "
                    "the negated dataset moves the same 9 all down.",
            **s3c["clean_worked_example"]},
        "worked_example_mirror_moves": {k: s3c["clean_mirror"][k] for k in
                                        ["moves", "moves_up", "moves_down"]},
        "small_n_tie_caveat": {
            "note": "the skew law is a density-level regularity, not a per-instance "
                    "theorem: a 1-decimal n=15 instance with tied values "
                    "(work/mech_stage3_worked.json) has a mirror whose "
                    "quintile groups shift on the ties (the <= group cuts are not "
                    "negation-equivariant on ties), landing a different local "
                    "optimum whose correction moves UP on right-skewed data. On "
                    "tie-free data with pronounced skew (|skew| >= 0.3) the "
                    "direction followed the skew sign in every observed case: "
                    "3 real 2026 groups + 2025 peer3 (up under left skew), 2 "
                    "real-data mirrors (down), 2 explicit counterexamples (down "
                    "under right skew), 33/33 synthetic batch, clean worked "
                    "example + its mirror = 42/42. Near-symmetric groups "
                    "(2025 peer4/peer5, |skew| < 0.1) have small either-signed "
                    "displacement: one-signed within the group, direction not "
                    "pinned by skew.",
            "tied_instance_moves": s3["worked_example"]["moves"],
            "tied_instance_mirror_moves": s3["worked_example_mirror"]["moves"]},
    },

    "registers_and_limits": [
        "2026 comparisons on the decimal-string register of R write.csv (same as "
        "dp_taste_2026); the binary-double register is in "
        "runs/exact_census_2026.json",
        "2025 summary scores are a pandas float64 port of SAS stages 0-2 (SAS "
        "accumulation order not reproduced); FASTCLUS itself is not replayed here: "
        "the 2025 comparison is the DP optimum vs the published Care Compare stars "
        "(2025-08-14 snapshot); the port is validated three ways: all 9 published "
        "national averages of Table 7 match at printed precision, 18,965 published "
        "per-group measure counts match with 0 mismatches, and published stars are "
        "contiguous in ported-score order (0 violations in all 3 peer groups); "
        "published-vs-optimum deltas in 2025 therefore combine FASTCLUS-vs-optimum "
        "with any strict=1 corner semantics (the FASTCLUS replay of the SAS years "
        "is in runs/sas_replay_2021..2025.json)",
        "synthetic/necessity runs use R 4.3.3 stats::kmeans itself (not a Python "
        "imitation) via a replica script that reproduces the published stars on all "
        "three real peer groups with zero mismatches",
        "no error attribution: the procedure is CMS's published choice; "
        "this record only addresses why exact optimization moves stars one way "
        "on this data",
    ],
    "inputs": {
        "stage1": "work/mech_stage1_boundaries.json",
        "stage2a": "work/mech_stage2a_synthetic.json",
        "stage2b": "work/mech_stage2b_2025.json",
        "stage3": "work/mech_stage3_worked.json",
        "scripts": ["work/mech_boundaries.py", "work/mech_fn_kmeans.R",
                    "work/mech_synthetic.py", "work/mech_2025.py",
                    "work/mech_worked_example.py", "work/mech_assemble.py"],
    },
}

with open(f"{BASE}/runs/mechanism_allupward.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote", f"{BASE}/runs/mechanism_allupward.json")
print(json.dumps(out["verdict"], indent=2))
