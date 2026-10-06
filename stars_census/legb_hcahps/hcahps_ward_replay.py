# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""HCAHPS Ward's-method replay on the public file (current refresh).
Per starred measure: per-hospital Ward's method (k=5, documented tie rule) on
the public file's linear scores vs the published star assignments.
Output: HCAHPS_WARD_REPLAY.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 5.1 and Appendix C.6
    (Ward's method on the public HCAHPS file recovers one of the eight published
    groupings).
Run:  cd stars_census/legb_hcahps && python3 hcahps_ward_replay.py
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

sys.path.insert(0, _REC + "/src")
from ward_cluster_tie import ward_sas_cut

CSV_PATH = "data/HCAHPS-Hospital.csv"
MEASURES = ["H_COMP_1", "H_COMP_2", "H_COMP_5", "H_COMP_6",
            "H_CLEAN", "H_QUIET", "H_HSP_RATING", "H_RECMND"]


def main():
    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8-sig")))
    lin, star = {}, {}
    for r in rows:
        mid = r.get("HCAHPS Measure ID", "")
        for m in MEASURES:
            if mid == f"{m}_LINEAR_SCORE":
                v = r.get("HCAHPS Linear Mean Value", "").strip()
                if v.isdigit():
                    lin.setdefault(m, {})[r["Facility ID"]] = int(v)
            if mid == f"{m}_STAR_RATING":
                v = r.get("Patient Survey Star Rating", "").strip()
                if v.isdigit():
                    star.setdefault(m, {})[r["Facility ID"]] = int(v)
    out = {}
    n_match = 0
    for m in MEASURES:
        ids = sorted(set(lin.get(m, {})) & set(star.get(m, {})))
        pts = sorted((lin[m][i], star[m][i]) for i in ids)
        vals = [float(v) for v, _ in pts]
        pub = [s for _, s in pts]
        cl = ward_sas_cut(vals, 5)
        labels = [0] * len(vals)
        for ci, c in enumerate(cl):
            for i in c:
                labels[i] = ci
        means = {ci: sum(vals[i] for i in range(len(vals)) if labels[i] == ci)
                 / labels.count(ci) for ci in set(labels)}
        rank = {ci: r + 1 for r, ci in
                enumerate(sorted(means, key=lambda c: means[c]))}
        ws = [rank[l] for l in labels]
        same = ws == pub
        n_match += bool(same)
        out[m] = dict(n=len(pts),
                      ward_reproduces_published=bool(same),
                      n_mismatched=sum(1 for a, b in zip(ws, pub) if a != b))
        print(m, same, f"n={len(pts)} mismatched={out[m]['n_mismatched']}", flush=True)
    rec = dict(
        register=("HCAHPS Ward's-method replay on the public file, current "
                  "refresh: per-hospital Ward's method (k=5, documented tie "
                  "rule) on the public file's linear mean values vs the "
                  "published star assignments, per starred measure. CMS's "
                  "own tables give a clustering N different from the public "
                  "file's starred set, so a failed replay indicates an "
                  "input-set difference and/or a tie-rule difference, not "
                  "optimizer suboptimality alone."),
        data_sha256_16=hashlib.sha256(open(CSV_PATH, "rb").read()).hexdigest()[:16],
        n_measures=len(MEASURES), n_reproduced=n_match, measures=out)
    json.dump(rec, open("HCAHPS_WARD_REPLAY.json", "w"), indent=1)
    open("HCAHPS_WARD_REPLAY.json", "a").write("\n")
    print(f"HCAHPS Ward replay: {n_match}/{len(MEASURES)}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not _os.path.exists(CSV_PATH):
    print("hcahps_ward_replay: missing input data/HCAHPS-Hospital.csv (run from stars_census/legb_hcahps/; the file is the "
          "Provider Data Catalog download, not included; re-fetch it by the "
          "URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
    main()
