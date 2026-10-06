#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""coverage_split.py — exact decomposition of the replay-rated hospitals vs the
published Care Compare snapshots, all six years, uniform schema.

2026: the replay rates 3,203; the published comparison covers 3,182. This script
accounts for every one of the 3,203: rated-published-matched /
suppressed-by-footnote-N / absent-from-snapshot (the seven absent provider IDs
are enumerated), pins provider 200052 (input vs published vs replay), and
computes the overlap of the suppressed and absent classes with the 213
(pre-cap) and 208 (post-cap) censuses.

SAS years 2021-2025: the same decomposition per year (primary = the snapshot that
matches the release; the alternate snapshot is also decomposed), consolidated
here so the era is uniform. Cross-checked against the stored sas_replay_*.json /
replay_2026_pilot.json comparison blocks (which are not modified).

Writes runs/coverage_split_2026.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 3.2 (205 of 3,182
    displayed hospitals; 208 of 3,203 after the cap).
Run:  cd hospital/work && python3 coverage_split.py
Exit codes: 0 on success; 2 when a required input is missing (one line on stderr
    names it).
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # record root
import csv, io, json, zipfile
from collections import Counter

BASE = _REC + "/hospital"
RUNS = f"{BASE}/runs"

PRECAP_NOTE = ("pre-cap register; the published-visible post-cap census is "
               "208/3,203 = 6.5%, still all-upward (runs/safety_cap_check_2026.json)")
MIRROR_NOTE = ("the qualitynet-hosted pipeline artifacts were obtained from two public "
               "GitHub mirrors (RUSH klocey/stars-data-builder f27704d4; KOS "
               "Kentucky-Open-Science/CMS-Star-Rankings ad0db1f6); 2025 inputs "
               "md5-identical across both mirrors; 2021-2024 and 2026 pack artifacts on "
               "the RUSH mirror only; Care Compare snapshots, Federal Register PDFs and "
               "eCFR text directly from CMS, govinfo and eCFR (data/raw/SOURCES.txt, "
               "MANIFEST_MD5.txt)")
PROVENANCE = ("Data provenance: the pipeline artifacts hosted on qualitynet (SAS packs "
              "2021-2025, alldata sas7bdat inputs, R_Pack_2026Apr.zip, "
              "stars_alldata_2026apr.csv, methodology v5.1 PDF, pack docs) were obtained "
              "from two public GitHub mirrors (RUSH klocey/stars-data-builder f27704d4; "
              "KOS Kentucky-Open-Science/CMS-Star-Rankings ad0db1f6); the 2025 inputs are "
              "md5-identical across both mirrors (alldata_2025jul.sas7bdat "
              "d9bf0fbc9c27db46d7cd8cd5f32574e3) and the 2025 .sas programs are "
              "byte-identical up to CRLF/LF, while the 2021-2024 and 2026 pack artifacts "
              "are present on one mirror only (RUSH); Care Compare snapshot zips, Federal "
              "Register PDFs and eCFR text came directly from CMS, govinfo and eCFR "
              "(data/raw/SOURCES.txt, MANIFEST_MD5.txt).")

def read_zip_csv(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(member)]
        if not name:
            return None
        with z.open(name[0]) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))

def load_published(zpath):
    rows = read_zip_csv(zpath, "Hospital_General_Information.csv")
    hdr = rows[0].keys()
    idcol = "Facility ID" if "Facility ID" in hdr else "Provider ID"
    ratecol = "Hospital overall rating"
    footcol = None
    for c in hdr:
        if c.lower().startswith("hospital overall rating") and "footnote" in c.lower():
            footcol = c
    assert ratecol in hdr, (zpath, list(hdr)[:8])
    pub = {}
    for r in rows:
        pub[r[idcol]] = (r[ratecol].strip(), (r.get(footcol) or "").strip() if footcol else "")
    cross = {}
    xrows = read_zip_csv(zpath, "Footnote_Crosswalk.csv")
    if xrows:
        for r in xrows:
            vals = list(r.values())
            if len(vals) >= 2 and vals[0]:
                cross[vals[0].strip()] = vals[1].strip()
    return pub, cross

def load_replay(path, idcol="PROVIDER_ID", starcol="star"):
    rep = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rep[r[idcol]] = r
    rated = {p: r for p, r in rep.items() if r[starcol] not in ("NA", "", "NaN")}
    return rep, rated

def decompose(rated, pub, cross):
    matched = mismatched = 0
    mism_ids, suppressed, absent = [], [], []
    for pid in sorted(rated):
        star = int(rated[pid]["star"])
        if pid not in pub:
            absent.append(pid)
            continue
        ps, foot = pub[pid]
        if ps in ("Not Available", ""):
            suppressed.append({"provider_id": pid, "replay_star": star,
                               "footnote": foot or "(blank)"})
        elif int(ps) == star:
            matched += 1
        else:
            mismatched += 1
            mism_ids.append(pid)
    foot_counts = Counter(s["footnote"] for s in suppressed)
    total = matched + mismatched + len(suppressed) + len(absent)
    return {
        "replay_rated_total": len(rated),
        "rated_published_matched": matched,
        "rated_published_MISMATCHED": mismatched,
        "mismatched_ids": mism_ids,
        "suppressed_by_footnote": dict(sorted(foot_counts.items())),
        "suppressed_footnote_text": {k: cross.get(k, "(not in crosswalk)")
                                     for k in foot_counts},
        "suppressed_count": len(suppressed),
        "suppressed_detail": suppressed,
        "absent_from_snapshot_count": len(absent),
        "absent_provider_ids": absent,
        "all_accounted": total == len(rated),
        "accounting": f"{matched} matched + {mismatched} mismatched + "
                      f"{len(suppressed)} suppressed + {len(absent)} absent = {total}",
    }

_RAW_NEEDED = (f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip",
               f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-02-25.zip",
               f"{BASE}/data/raw/2026-05/stars_alldata_2026apr.csv",
               f"{BASE}/data/raw/2021-04/care_compare_snapshots",
               f"{BASE}/data/raw/2022-07/care_compare_snapshots",
               f"{BASE}/data/raw/2023-07/care_compare_snapshots",
               f"{BASE}/data/raw/2024-07/care_compare_snapshots",
               f"{BASE}/data/raw/2025-07/care_compare_snapshots")
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)
if __name__ == "__main__" and not all(_os.path.exists(p) for p in _RAW_NEEDED):
    print("coverage_split: missing raw input(s): " + ", ".join(_os.path.relpath(p, _REC) for p in _RAW_NEEDED
                                                       if not _os.path.exists(p))
          + " (Care Compare snapshot archives and the 2026 input file, not included in the package; re-fetch each by "
          "the URL and sha256 in CMS_RAW_PINS.json)",
          file=sys.stderr)
    sys.exit(2)

# ---------------- 2026 ----------------
_, rated26 = load_replay(f"{BASE}/work/R_output/Star_2026apr.csv")
full26 = {}
with open(f"{BASE}/work/R_output/fullprec_2026apr.csv") as f:
    for r in csv.DictReader(f):
        full26[r["PROVIDER_ID"]] = r
zp26 = f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip"
pub26, cross26 = load_published(zp26)
d26 = decompose(rated26, pub26, cross26)

# census sets
cc26 = json.load(open(f"{RUNS}/exact_census_2026.json"))
cap = json.load(open(f"{RUNS}/safety_cap_check_2026.json"))
set213 = {h["provider_id"] for h in cc26["star_differs_detail"]}
erased = set(cap["postcap_differ_census"]["differs_erased_by_cap"])
set208 = set213 - erased
assert len(set213) == 213 and len(set208) == 208
sup_ids = {s["provider_id"] for s in d26["suppressed_detail"]}
abs_ids = set(d26["absent_provider_ids"])

def inter(a, b):
    x = sorted(a & b)
    return {"count": len(x), "ids": x}

interaction = {
    "note": "membership interaction of the display-side classes (suppressed/absent) "
            "with the census sets",
    "census_213_precap": {"in_suppressed_14": inter(set213, sup_ids),
                          "in_absent_7": inter(set213, abs_ids)},
    "census_208_postcap": {"in_suppressed_14": inter(set208, sup_ids),
                           "in_absent_7": inter(set208, abs_ids)},
    "erased_5": {"in_suppressed_14": inter(erased, sup_ids),
                 "in_absent_7": inter(erased, abs_ids)},
    "provider_200052": {"in_213": "200052" in set213, "in_208": "200052" in set208,
                        "in_erased_5": "200052" in erased,
                        "in_suppressed_14": "200052" in sup_ids,
                        "in_absent_class": "200052" in abs_ids},
}
tot_inter = (interaction["census_213_precap"]["in_suppressed_14"]["count"]
             + interaction["census_213_precap"]["in_absent_7"]["count"]
             + interaction["census_208_postcap"]["in_suppressed_14"]["count"]
             + interaction["census_208_postcap"]["in_absent_7"]["count"])
interaction["total_interaction_count"] = tot_inter
interaction["census_unaffected"] = (tot_inter == 0)
cc_by_id = {h["provider_id"]: h for h in cc26["star_differs_detail"]}
disp_invisible_226 = sorted((set213 & (sup_ids | abs_ids)))
interaction["display_visibility"] = {
    "detail": ("3 of the 213 pre-cap corrections (and the same 3 of the 208 post-cap "
               "survivors) attach to hospitals with no publicly displayed star in the "
               "2026-05-13 snapshot: 2 footnote-5-suppressed and 1 absent from the "
               "snapshot. The census sets are unchanged (the DP-vs-CMS comparison runs "
               "on the 3,203 replay-rated universe and suppression or absence is "
               "display-side only), but the display-visible correction count is 205, "
               "not 208: 205/3,182 displayed ratings = 6.44%."),
    "members": [{**cc_by_id[i],
                 "class": "absent-from-snapshot" if i in abs_ids else "suppressed-footnote-"
                 + next(s["footnote"] for s in d26["suppressed_detail"]
                        if s["provider_id"] == i)}
                for i in disp_invisible_226],
    "display_visible_corrections_postcap": len(set208 - sup_ids - abs_ids),
    "displayed_ratings_denominator": d26["rated_published_matched"],
}

# provider 200052 pin
p = "200052"
in_input = False
with open(f"{BASE}/data/raw/2026-05/stars_alldata_2026apr.csv") as f:
    rd = csv.reader(f)
    next(rd)
    for row in rd:
        if row[0].strip('"') == p or row[0] == p:
            in_input = True
            break
r26 = rated26.get(p)
fp = full26.get(p)
zp26pre = f"{BASE}/data/raw/2026-05/care_compare_snapshots/hospitals_2026-02-25.zip"
pub26pre, _ = load_published(zp26pre)
pin = {
    "provider_id": p,
    "input_data": {"present_in_stars_alldata_2026apr.csv": in_input},
    "replay": None if r26 is None else {
        "rated": True,
        "peer_group": f"peer{r26['Total_measure_group_cnt']}",
        "summary_score_17g": fp["summary_score_17g"],
        "star_precap": int(fp["star_precap"]),
        "star_postcap": int(fp["star_postcap"]),
        "safety_quartile_Safe_q": fp["Safe_q"],
        "safety_measure_cnt": fp["Outcomes_Safety_cnt"],
    },
    "published_2026-05-13": ("ABSENT from snapshot (no row for this Facility ID)"
                             if p not in pub26 else
                             {"rating": pub26[p][0], "footnote": pub26[p][1]}),
    "published_2026-02-25_prerelease": ("ABSENT from snapshot"
                                        if p not in pub26pre else
                                        {"rating": pub26pre[p][0],
                                         "footnote": pub26pre[p][1]}),
    "class": ("absent-from-snapshot" if p in abs_ids else
              "suppressed" if p in sup_ids else "matched"),
    "census_membership": interaction["provider_200052"],
    "census_entry": cc_by_id.get(p),
    "cap_erased": p in erased,
}

# cross-check against the stored replay record (not modified)
rp = json.load(open(f"{RUNS}/replay_2026_pilot.json"))["comparison"]
xchk26 = {
    "stored_record": "runs/replay_2026_pilot.json",
    "matched_agrees": rp["matched_exactly"] == d26["rated_published_matched"],
    "suppressed_agrees": rp["replay_rated_published_NotAvailable"] == d26["suppressed_count"],
    "mismatched_agrees": rp["mismatched"] == d26["rated_published_MISMATCHED"] == 0,
}

# ---------------- SAS years ----------------
MATCH_SNAPSHOT = {   # (release dir, the snapshot that carries the release's stars, the alternate)
    "2021": ("2021-04", "hospitals_2021-04-28.zip", "hospitals_2021-07-21.zip"),
    "2022": ("2022-07", "hospitals_2022-07-06.zip", "hospitals_2022-10-06.zip"),
    "2023": ("2023-07", "hospitals_2023-07-06.zip", "hospitals_2023-10-06.zip"),
    "2024": ("2024-07", "hospitals_2024-07-31.zip", "hospitals_2024-10-30.zip"),
    "2025": ("2025-07", "hospitals_2025-08-14.zip", None),
}
ALT_NOTE = {"2025": ("hospitals_2025-04-30.zip carries the previous release's stars "
                     "(1,245 wholesale mismatches, a data-vintage contrast; "
                     "runs/sas_replay_2025.json); no coverage split computed on it")}

sas = {}
for year, (ddir, primary, alternate) in MATCH_SNAPSHOT.items():
    _, rated = load_replay(f"{BASE}/work/sas_replay_out/star_{ddir}.csv")
    stored = json.load(open(f"{RUNS}/sas_replay_{year}.json"))["comparisons"]
    cens = json.load(open(f"{RUNS}/exact_census_{year}.json"))
    dset = {h["provider_id"] for h in cens["star_differs_detail"]}
    ydat = {"replay_rated_total": len(rated),
            "exact_census_total_rated_agrees": cens["total_rated"] == len(rated)}
    for tag, zname in (("primary", primary), ("alternate", alternate)):
        if zname is None:
            continue
        zp = f"{BASE}/data/raw/{ddir}/care_compare_snapshots/{zname}"
        pub, cross = load_published(zp)
        d = decompose(rated, pub, cross)
        bk = stored[zname]
        d["snapshot"] = zname
        d["stored_comparison_agrees"] = {
            "matched": bk["matched_exactly"] == d["rated_published_matched"],
            "suppressed": bk["replay_rated_published_NotAvailable"] == d["suppressed_count"],
            "mismatched_zero": bk["mismatched"] == d["rated_published_MISMATCHED"] == 0,
        }
        sup = {s["provider_id"] for s in d["suppressed_detail"]}
        ab = set(d["absent_provider_ids"])
        d["census_interaction"] = {
            "differ_set_in_suppressed": inter(dset, sup),
            "differ_set_in_absent": inter(dset, ab),
        }
        ydat[tag] = d
    if year in ALT_NOTE:
        ydat["alternate_note"] = ALT_NOTE[year]
    sas[year] = ydat

# era-uniform table (primary snapshots + 2026)
table = []
for year in ["2021", "2022", "2023", "2024", "2025"]:
    d = sas[year]["primary"]
    table.append({"year": year, "snapshot": d["snapshot"],
                  "replay_rated": d["replay_rated_total"],
                  "matched": d["rated_published_matched"],
                  "suppressed": d["suppressed_count"],
                  "suppressed_by_footnote": d["suppressed_by_footnote"],
                  "absent": d["absent_from_snapshot_count"],
                  "all_accounted": d["all_accounted"]})
table.append({"year": "2026", "snapshot": "hospitals_2026-05-13.zip",
              "replay_rated": d26["replay_rated_total"],
              "matched": d26["rated_published_matched"],
              "suppressed": d26["suppressed_count"],
              "suppressed_by_footnote": d26["suppressed_by_footnote"],
              "absent": d26["absent_from_snapshot_count"],
              "all_accounted": d26["all_accounted"]})

record = {
    "run": "coverage_split_2026",
    "register": ("Display-coverage decomposition of the replay-rated universe against the "
                 "published Care Compare snapshots; the 2026 replay side is the post-cap "
                 "stars (the published register). Census interactions are quoted at both "
                 "censuses: 213 at the pre-cap register and 208/3,203 = 6.5% at the "
                 "published-visible post-cap register, still all-upward "
                 "(runs/safety_cap_check_2026.json). " + PROVENANCE),
    "notes": {"precap": PRECAP_NOTE, "mirror_provenance": MIRROR_NOTE},
    "year_2026": {
        "replay_rated_total": d26["replay_rated_total"],
        "published_snapshot": {"zip": "data/raw/2026-05/care_compare_snapshots/hospitals_2026-05-13.zip"},
        "split": d26,
        "stored_comparison_agrees": xchk26,
        "census_interaction": interaction,
        "provider_200052_pin": pin,
    },
    "sas_years": sas,
    "era_uniform_table": table,
}
# six-year display-visibility rollup (primary snapshots)
rollup = []
for year in ["2021", "2022", "2023", "2024", "2025"]:
    ci = sas[year]["primary"]["census_interaction"]
    rollup.append({"year": year,
                   "differ_set_not_displayed": ci["differ_set_in_suppressed"]["count"]
                   + ci["differ_set_in_absent"]["count"],
                   "ids": ci["differ_set_in_suppressed"]["ids"]
                   + ci["differ_set_in_absent"]["ids"]})
rollup.append({"year": "2026", "differ_set_not_displayed": len(disp_invisible_226),
               "ids": disp_invisible_226})
record["FINDINGS"] = {
    "1_interaction_not_zero": interaction["display_visibility"],
    "2_six_year_display_visibility_rollup": {
        "note": ("corrections attached to hospitals with no publicly displayed star in "
                 "the matching snapshot (suppressed or absent), per year; the census sets "
                 "themselves are unchanged (display-side layer only)"),
        "per_year": rollup,
        "total_of_1146": sum(r["differ_set_not_displayed"] for r in rollup),
    },
}
out = f"{RUNS}/coverage_split_2026.json"
with open(out, "w") as f:
    json.dump(record, f, indent=1)

print("=== ERA-UNIFORM COVERAGE SPLIT ===")
for r in table:
    print(f"{r['year']}  {r['snapshot']:28s} rated {r['replay_rated']:5d} = "
          f"matched {r['matched']:5d} + suppressed {r['suppressed']:2d} "
          f"{r['suppressed_by_footnote']} + absent {r['absent']:2d}   "
          f"accounted={r['all_accounted']}")
print("\n2026 absent class:", d26["absent_provider_ids"])
print("2026 suppressed footnotes:", d26["suppressed_by_footnote"],
      d26["suppressed_footnote_text"])
print("\n200052 pin:", json.dumps(pin, indent=1))
print("\ninteraction:", json.dumps({k: v for k, v in interaction.items()
                                    if k != 'note'}, indent=1))
print("wrote", out)
