# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Write the data file of the stand-alone evaluator (ma-cutpoint-rows.json)
from the 123-computation optimality-gap table. Rows are joined from the gap
table (exact_gaps_fullsample_docsonly.json) with the sorted
post-outlier-bound sample reproduced through the table's own trim path.
Computations with fewer than five distinct scores (k < 5: no five-cluster
partition) are carried with their k - 1 cut points (none when k = 1),
exactly as the gap table records them. Typed values throughout; every row
carries a resolved membership label.
Output: runs/ma-cutpoint-rows-123.json
Run: python3 export_cutpoint_rows.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.1 (the
    123-computation census; writes the evaluator's data file ma-cutpoint-
    rows.json).
Requires: Python >= 3.10; numpy, scipy.
"""
import hashlib
import json
import os

import exact_gaps_docsonly as G
import sys
from tukey import tukey_trim

BASE = G.BASE


def main():
    gaps = {(g["year"], g["mid"], g["org"]): g for g in
            json.load(open(os.path.join(BASE, "runs/exact_gaps_fullsample_docsonly.json")))}
    jobs = G.build_jobs()
    rows = []
    for j in jobs:
        key = (j["year"], j["mid"], j["org"])
        g = gaps.get(key)
        if g is None:
            continue
        scores = [s for _, s in sorted(j["kept"])]
        mask, _, _ = tukey_trim(scores, cap_lo="0", cap_hi=j["cap_hi"])
        trimmed = sorted((s for s, m in zip(scores, mask) if m), key=lambda x: __import__("fractions").Fraction(x))
        assert len(trimmed) == g["n_trimmed"], (key, len(trimmed), g["n_trimmed"])
        # typed pass-through: the evaluator consumes the table's row types
        # verbatim; no str() wraps
        rows.append(dict(
            year=g["year"], mid=g["mid"], org=g["org"], name=g["name"],
            n_input=g["n_input"], n_trimmed=g["n_trimmed"],
            k=g["k"], membership=g["membership"],
            ward_ssq=g["ward_ssq"], dp_ssq=g["dp_ssq"],
            gap=g["gap"], off_optimum=g["off_optimum"],
            dp_n_optima=g["dp_n_optima"],
            ward_cutpoints=g["ward_cutpoints"],
            dp_cutpoints=g["dp_cutpoints"],
            n_star_delta_fullsample=g["n_star_delta_fullsample"],
            higher_is_better=j["higher"], sample=trimmed))
    assert len(rows) == 123, len(rows)
    n_off = sum(1 for r in rows if r["off_optimum"])
    n_scores = sum(r["n_trimmed"] for r in rows)
    n_delta = sum(r["n_star_delta_fullsample"] for r in rows)
    # fewer than five distinct scores: k < 5 and len(dp_cutpoints) == k - 1
    # (two k=1 rows carry no cut points, the k=4 row carries three)
    n_deg = sum(1 for r in rows if len(r["dp_cutpoints"]) != 4)
    assert n_deg == sum(1 for r in rows if r["k"] < 5)
    assert not any(r["membership"] == "pending" for r in rows), (
        "every row must carry a resolved membership label")
    out = dict(
        _what=("Medicare Advantage cut-point table: the 123 clusterable "
               "measure-year(-org) computations of the paper's "
               "optimality-gap table (no n<30 filter; three computations "
               "with fewer than five distinct scores, k < 5, carried with "
               "k - 1 cut points, none when k = 1), each with its sorted "
               "post-outlier-bound score sample as exact decimal strings, "
               "the published-method (SAS Ward) cut points, the "
               "exact-optimal cut points, and the recorded objective values "
               "as explicit fractions."),
        _source="runs/exact_gaps_fullsample_docsonly.json",
        _source_sha256_16=hashlib.sha256(open(os.path.join(
            BASE, "runs/exact_gaps_fullsample_docsonly.json"),
            "rb").read()).hexdigest()[:16],
        _generator="src/export_cutpoint_rows.py",
        _tallies=dict(n_rows=123, n_off_optimum=n_off, n_degenerate=n_deg,
                      n_scores_total=n_scores, n_star_deltas_total=n_delta),
        rows=rows)
    p = os.path.join(BASE, "runs/ma-cutpoint-rows-123.json")
    json.dump(out, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print(f"123 rows: off {n_off}, fewer than five distinct scores {n_deg}, scores {n_scores:,}, "
          f"deltas {n_delta:,} -> {p}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
