# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Within-groups re-final.

The stored collapse records (enclosure_collapse_frontswap.json /
_rperm.json) carry the pre-guardrail per-realization group means (ward_mean /
dp_mean) beside the finals. The guardrail-keying corrections invalidate the
stored finals and deltas, not the means, so this script recomputes:
  finals  = guardrail(stored mean, corrected prior via name+part, cap) -> display
  deltas  = star assignment of the replay pools (the default score vintages
            of the resampled computation; these records replay the
            computation under the assumed readout, so the criterion-vintage
            correction does not apply here) against the published baseline
            stars, with the disaster measure-star hold-harmless applied
            identically in both arms.
The six small-n PDP computations recovered by the small-n correction have no
stored realizations (the resampled runs predate that correction) and enter
the within-groups records only prospectively, as recorded in the output.
Outputs: runs/enclosure_collapse_frontswap_refinal.json,
         runs/enclosure_collapse_frontswap_rperm_refinal.json (same schema,
         'refinal_from' block added).
Run: python3 within_groups_refinal.py

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (the exact optimum
    inside the ten resampling groups, under the assumed readout of Appendix
    C.3).
Requires: Python >= 3.10; numpy, scipy.
"""
import hashlib
import json
import os
from fractions import Fraction as F

import enclosure_census_docsonly as C
import sys
from aggregate import load_inputs
from census_measures import (disaster_pcts, prior_published_star_map,
                             disaster_higher_of)
from falsify_groups import round_display
from guardrail import guardrail_boundary

BASE = C.BASE
PAIRS = [("runs/enclosure_collapse_frontswap.json",
          "runs/enclosure_collapse_frontswap_refinal.json"),
         ("runs/enclosure_collapse_frontswap_rperm.json",
          "runs/enclosure_collapse_frontswap_rperm_refinal.json")]


def sha16(rel):
    return hashlib.sha256(
        open(os.path.join(BASE, rel), "rb").read()).hexdigest()[:16]


def main():
    # replay inputs: the default score vintages of the resampled computation
    # (2024 original / 2025 original / 2026 current)
    jobs, _ = C.build_jobs()
    jmap = {(j["year"], j["mid"], j["org"]): j for j in jobs}
    baseline = {}
    for year in C.YEARS:
        stars, _, _ = load_inputs(year, C.BASELINE_VINTAGE[year])
        baseline[year] = stars
    dis = {y: disaster_pcts(y, C.BASELINE_VINTAGE[y]) for y in C.YEARS}
    pmaps = {y: prior_published_star_map(y) for y in C.YEARS}
    for src_rel, out_rel in PAIRS:
        b = json.load(open(os.path.join(BASE, src_rel)))
        n_changed = 0
        for r in b["measure_results"]:
            if "error" in r:
                continue
            key = (r["year"], r["mid"], r["org"])
            job = jmap.get(key)
            if job is None:
                r["error"] = "no-job-after-refinal (split absent from the corrected jobs)"
                continue
            part = "C" if r["mid"].startswith("C") else "D"
            newfin = {}
            for arm, mkey in (("ward_final", "ward_mean"),
                              ("dp_final", "dp_mean")):
                vals = [F(v) for v in r[mkey]]
                if job["gr_note"] == "applied":
                    cap = F(job["cap"])
                    fin = [guardrail_boundary(v, F(p), cap)[0]
                           for v, p in zip(vals, job["prior"])]
                else:
                    fin = vals
                newfin[arm] = [str(F(round_display(v, job["dd"])))
                               for v in fin]
            if (newfin["ward_final"] != r["ward_final"]
                    or newfin["dp_final"] != r["dp_final"]):
                n_changed += 1
            r["ward_final"], r["dp_final"] = (newfin["ward_final"],
                                              newfin["dp_final"])
            # deltas re-derived from the replay pool for the WARD arm (the
            # collapse schema's deltas column is the ward-arm delta list)
            st_pub = baseline[r["year"]]
            deltas = []
            bd = [F(v) for v in r["ward_final"]]
            for cid, s in job["pool"]:
                sw = st_pub.get(cid, {}).get(r["mid"])
                if sw is None:
                    continue
                x = F(s)
                if job["higher"]:
                    sd = 1 + sum(1 for t in bd if x >= t)
                else:
                    sd = 5 - sum(1 for t in bd if x > t)
                sd = disaster_higher_of(job["name"], part, cid, sd,
                                        dis[r["year"]], pmaps[r["year"]])
                if sd != sw:
                    deltas.append([cid, int(sw), int(sd)])
            r["deltas"] = deltas
            r["n_star_changes"] = len(deltas)
        b["refinal_from"] = dict(
            source=src_rel, source_sha256_16=sha16(src_rel),
            generator="src/within_groups_refinal.py",
            register=("Finals and deltas recomputed from the stored "
                      "pre-guardrail group means with the corrected guardrail "
                      "keying (name+part, MPF-2024 exempt) and the disaster "
                      "measure-star hold-harmless, keeping the replay score "
                      "vintages under the assumed readout; the six recovered "
                      "small-n PDP computations have no stored realizations "
                      "and enter the within-groups records only prospectively."),
            n_rows_refinal_changed=n_changed)
        out = os.path.join(BASE, out_rel)
        json.dump(b, open(out, "w"), indent=1)
        open(out, "a").write("\n")
        print(f"{src_rel}: {n_changed} rows re-finaled -> {out_rel}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main()
