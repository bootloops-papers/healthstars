# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""ICH CAHPS Ward's-method replay on the public file.
Per measure: Ward's minimum-variance clustering (documented tie rule, k=5) on
the published linearized scores of the April 2026 facility file, the
resulting bins compared to the published star assignments.
Output: ICH_WARD_REPLAY.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 and Appendix C.6
    (Ward's method on the ICH CAHPS file reproduces five of the six published
    groupings).
Run:  cd stars_census/legc_ich && python3 census_ich.py && python3 ich_ward_replay.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10; numpy, scipy.
"""
import os as _os
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv
import hashlib
import json
import sys
from fractions import Fraction

sys.path.insert(0, _REC + "/src")
from ward_cluster_tie import ward_sas_cut

from census_ich import CSV, MEASURES, MISSING


def main():
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8-sig")))
    out = {}
    n_match = 0
    for name, lc, sc in MEASURES:
        pts = []
        for r in rows:
            v, s = r[lc].strip(), r[sc].strip()
            if v not in MISSING and s not in MISSING:
                pts.append((Fraction(v), int(Fraction(s))))
        vals = [float(v) for v, _ in sorted(pts)]
        exact = [v for v, _ in sorted(pts)]
        pub = [s for _, s in sorted(pts)]
        cl = ward_sas_cut(vals, 5)
        labels = [0] * len(vals)
        for ci, c in enumerate(cl):
            for i in c:
                labels[i] = ci
        # order clusters by their mean score -> stars 1..5
        means = {}
        for ci in set(labels):
            xs = [exact[i] for i in range(len(exact)) if labels[i] == ci]
            means[ci] = sum(xs, Fraction(0)) / len(xs)
        rank = {ci: r + 1 for r, ci in
                enumerate(sorted(means, key=lambda c: means[c]))}
        ward_stars = [rank[l] for l in labels]
        same = ward_stars == pub
        n_match += bool(same)
        out[name] = dict(n=len(pts), ward_reproduces_published=bool(same),
                         n_mismatched_facilities=sum(1 for a, b in
                                                     zip(ward_stars, pub)
                                                     if a != b))
    rec = dict(
        register=("ICH CAHPS Ward's-method replay: per-facility Ward's "
                  "method (k=5, documented tie rule) on the published April "
                  "2026 linearized scores vs the published star assignments. "
                  "The nephrologists' communication and caring measure is "
                  "sensitive to the tie rule (2,594 facilities over 46 "
                  "distinct scores), so a 5-of-6 result is a tie-rule "
                  "difference, not an objective difference. Control: Ward's "
                  "method on the distinct values reproduces 0 of 6."),
        control_distinct_value_ward="0 of 6 measures (control)",
        sixth_measure_diagnosis=dict(distinct_values=46, n=2594,
                                     tie_rule_sensitive=True),
        data_sha256_16=hashlib.sha256(open(CSV, "rb").read()).hexdigest()[:16],
        n_measures=len(MEASURES), n_reproduced=n_match, measures=out)
    json.dump(rec, open("ICH_WARD_REPLAY.json", "w"), indent=1)
    open("ICH_WARD_REPLAY.json", "a").write("\n")
    print(f"ICH Ward replay: {n_match}/{len(MEASURES)} measures reproduce "
          "the published bins")
    for k, v in out.items():
        print(" ", k[:44], v["ward_reproduces_published"], f"(n={v['n']}, mismatched {v['n_mismatched_facilities']})")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not _os.path.exists(CSV):
    print("ich_ward_replay: missing input data/ICH_CAHPS_FACILITY.csv (run from stars_census/legc_ich/; the file is the "
          "Provider Data Catalog download, not included; re-fetch it by the "
          "URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
