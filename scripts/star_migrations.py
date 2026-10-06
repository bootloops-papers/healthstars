# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Star-migration tables (published star group -> star group at the exact
optimum) for Medicare Advantage and hospital stars, derived from the stored
census files:
 - MA stars: the contracts whose overall rating differs
   (runs/enclosure_collapse_frontswap.json), half-star groups, 6 cases =
   star year x direction; after = the rating under the frontswap@cid_asc
   contract order (order sensitivity flagged per contract).
 - Hospital stars: the yearly hospital census files
   (hospital/runs/exact_census_2021..2026.json), integer 1-5 groups,
   published -> dp_optimum.
Output: runs/star_migrations.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering"
    (M. D. Schwartz, 2026): the migration tables between star groups.
Run:  python3 scripts/star_migrations.py   (from the package root)
Requires: Python >= 3.10, standard library only.
"""
import os as _os
import sys
_REC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # package root
import json
from collections import Counter

BASE = _REC
HOSP = _REC + "/hospital"
OUT = f"{BASE}/runs/star_migrations.json"


def ma_block():
    d = json.load(open(f"{BASE}/runs/enclosure_collapse_frontswap.json"))
    cert = [c for c in d["contracts"] if c["class"] == "exact-within-groups"]
    cases = {}
    for yr in ("2024", "2025", "2026"):
        for dr in ("UP", "DOWN"):
            rows = [c for c in cert if c["year"] == yr and c["direction"] == dr]
            mig = Counter((c["baseline"], c["ratings_by_realization"]["frontswap"])
                          for c in rows)
            cases[f"{yr}-{dr}"] = {
                "n": len(rows),
                "migrations": {f"{a}->{b}": n for (a, b), n in sorted(mig.items())},
                "before_bins": dict(Counter(c["baseline"] for c in rows)),
                "after_bins": dict(Counter(c["ratings_by_realization"]["frontswap"]
                                           for c in rows)),
            }
    order_sensitive = [
        {"contract_id": c["contract_id"], "year": c["year"],
         "values_by_order": c["ratings_by_realization"]}
        for c in cert
        if len({v for v in c["ratings_by_realization"].values() if v}) > 1]
    return {"register": ("Contracts whose overall rating differs under the exact "
                         "optimum, with the rating under the frontswap@cid_asc "
                         "contract order; ratings on the half-star ladder as "
                         "fractions (7/2 = 3.5); the change holds under all four "
                         "contract orders, and the value depends on the order only "
                         "where flagged."),
            "cases": cases, "order_sensitive_values": order_sensitive}


def hospital_block():
    years = {}
    for y in range(2021, 2027):
        d = json.load(open(f"{HOSP}/runs/exact_census_{y}.json"))
        det = d["star_differs_detail"]
        mig = Counter((r["shipped"], r["dp_optimum"]) for r in det)
        years[str(y)] = {
            "n_flips": len(det),
            "migrations": {f"{a}->{b}": n for (a, b), n in sorted(mig.items())},
            "before_bins": dict(sorted(Counter(r["shipped"] for r in det).items())),
            "after_bins": dict(sorted(Counter(r["dp_optimum"] for r in det).items())),
        }
    return {"register": ("Published star -> star at the exact optimum, integer 1-5 "
                         "groups; the 2026 row compares stars before the safety cap "
                         "(the CY2026 4-star safety cap modifies 5 of the 213 on "
                         "the optimum side; see runs/safety_cap_check_2026.json)."),
            "years": years}


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    bank = {"generator": "scripts/star_migrations.py",
            "sources": {"ma": "runs/enclosure_collapse_frontswap.json (sha256 in PINS_XREF.json)",
                        "hospital": "hospital/runs/exact_census_2021..2026.json (sha256 in PINS_XREF.json)"},
            "ma_stars": ma_block(),
            "hospital_stars": hospital_block()}
    with open(OUT, "w") as f:
        json.dump(bank, f, indent=1)
        f.write("\n")
    print("wrote", OUT)
    for case, v in bank["ma_stars"]["cases"].items():
        print(f"MA {case}: n={v['n']} {v['migrations']}")
    for y, v in bank["hospital_stars"]["years"].items():
        print(f"HOSP {y}: n={v['n_flips']} {v['migrations']}")
