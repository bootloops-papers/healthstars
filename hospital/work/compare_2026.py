#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Compare the replayed 2026 star ratings (CMS R pack, unmodified) against the
published Care Compare 2026-05-13 snapshot. Writes runs/replay_2026_pilot.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.1 (every published
    2026 star reproduced, 3,182 of 3,182); Appendix C.1.
Run:  cd hospital/work && python3 compare_2026.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, json, hashlib
from collections import Counter

BASE = _REC + "/hospital"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# --- replay output ---
_RAW_NEEDED = (f"{BASE}/work/cc_2026_05/Hospital_General_Information.csv",
               f"{BASE}/data/raw/2026-05/stars_alldata_2026apr.csv",
               f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip")
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(_os.path.exists(p) for p in _RAW_NEEDED):
    print("compare_2026: missing raw input(s): " + ", ".join(_os.path.relpath(p, _REC) for p in _RAW_NEEDED
                                                     if not _os.path.exists(p))
          + " (not included in the package; re-fetch each by the URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

replay = {}
with open(f"{BASE}/work/R_output/Star_2026apr.csv") as f:
    for r in csv.DictReader(f):
        replay[r["PROVIDER_ID"]] = r

# --- published snapshot ---
pub = {}
with open(f"{BASE}/work/cc_2026_05/Hospital_General_Information.csv",
          encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        pub[r["Facility ID"]] = (r["Hospital overall rating"],
                                 r.get("Hospital overall rating footnote", ""))

ids_replay = set(replay)
ids_pub = set(pub)
both = ids_replay & ids_pub

match = mismatch = 0
both_unrated = 0
replay_rated_pub_na = []
pub_rated_replay_na = []
mismatches = []
for pid in sorted(both):
    rs = replay[pid]["star"]
    ps, foot = pub[pid]
    r_rated = rs not in ("NA", "")
    p_rated = ps not in ("Not Available", "")
    if r_rated and p_rated:
        if int(rs) == int(ps):
            match += 1
        else:
            mismatch += 1
            mismatches.append({"provider_id": pid, "replay_star": int(rs),
                               "published_star": int(ps), "footnote": foot})
    elif r_rated and not p_rated:
        replay_rated_pub_na.append({"provider_id": pid, "replay_star": int(rs),
                                    "footnote": foot})
    elif p_rated and not r_rated:
        pub_rated_replay_na.append({"provider_id": pid, "published_star": int(ps)})
    else:
        both_unrated += 1

only_replay = sorted(ids_replay - ids_pub)
only_pub = sorted(ids_pub - ids_replay)
only_pub_rated = [p for p in only_pub if pub[p][0] not in ("Not Available", "")]

na_footnotes = Counter(x["footnote"] for x in replay_rated_pub_na)

dist_replay = Counter(r["star"] for r in replay.values())
dist_pub = Counter(v[0] for v in pub.values())


# --- cross-checks against the methodology report and the pre-cap replay ---
def _grp_col(row):
    for k in row:
        if k.lower() in ("measure_group_cnt", "grp_cnt", "group_cnt", "cnt_grp", "n_groups", "peer_group", "grp"):
            return k
    return None
_gc = _grp_col(next(iter(replay.values())))
import re as _re
def _ngroups(v):
    m = _re.search(r"groups=(\d)", str(v))
    return int(m.group(1)) if m else str(v)
peer_sizes = Counter(_ngroups(replay[p][_gc]) for p in replay if replay[p]["star"] not in ("NA", "")) if _gc else {}
precap = {}
with open(f"{BASE}/work/R_output/Star_precap_2026apr.csv") as f:
    for r in csv.DictReader(f):
        precap[r["PROVIDER_ID"]] = r["star"]
capped_5to4 = sum(1 for p, r in replay.items() if precap.get(p) == "5" and r["star"] == "4")
cross_checks = {
    "peer_group_sizes_replay": {f"{k}-group": v for k, v in sorted(peer_sizes.items(), key=lambda kv: str(kv[0]))},
    "peer_group_sizes_methodology_report_v51_fig5": {"3-group": 177, "4-group": 749, "5-group": 2277},
    "peer_groups_match": {f"{k}-group": v for k, v in sorted(peer_sizes.items(), key=lambda kv: str(kv[0]))} == {"3-group": 177, "4-group": 749, "5-group": 2277},
    "input_hospitals_replay": len(replay),
    "input_hospitals_v51": 4569,
    "safety_cap_hospitals_5to4": capped_5to4,
}
result = {
    "run": "replay_2026_pilot",
    "code": "CMS R_Pack_2026Apr.zip, 3 programs byte-unmodified; a thin driver "
            "(work/driver.R) remaps the two hardcoded Windows paths via an "
            "attached file.path shim environment",
    "r_version": "R 4.3.3; packages readr, psych, dplyr",
    "input": {
        "file": "data/raw/2026-05/stars_alldata_2026apr.csv",
        "md5": md5(f"{BASE}/data/raw/2026-05/stars_alldata_2026apr.csv"),
        "rows_incl_header": 4570,
    },
    "published_reference": {
        "file": "data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip"
                " :: Hospital_General_Information.csv",
        "zip_md5": md5(f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip"),
        "column": "Hospital overall rating",
        "hospitals": len(pub),
    },
    "star_distribution": {
        "replay_post_cap": dict(sorted(dist_replay.items())),
        "published": dict(sorted(dist_pub.items())),
    },
    "comparison": {
        "ids_in_both": len(both),
        "matched_exactly": match,
        "mismatched": mismatch,
        "both_unrated": both_unrated,
        "replay_rated_published_NotAvailable": len(replay_rated_pub_na),
        "published_rated_replay_unrated": len(pub_rated_replay_na),
        "ids_only_in_replay_input": len(only_replay),
        "ids_only_in_published": len(only_pub),
        "ids_only_in_published_that_are_rated": len(only_pub_rated),
        "match_rate_of_jointly_rated": round(match / (match + mismatch), 6)
            if match + mismatch else None,
    },
    "cross_checks": cross_checks,
    "mismatch_detail": mismatches,
    "replay_rated_pub_na_footnotes": dict(na_footnotes),
    "pub_rated_replay_na_detail": pub_rated_replay_na,
}
out = f"{BASE}/runs/replay_2026_pilot.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result["comparison"], indent=2))
print("mismatches:", mismatches[:10])
print("NA footnotes on suppressed-but-computable:", dict(na_footnotes))
print("wrote", out)
