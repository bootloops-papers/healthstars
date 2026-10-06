#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""fig_migrations.py — the migration figure of the star-ratings paper
(fig:orion-migrations): every rated hospital of one peer group as a dot on
the score line, the published star groups and the groups at the exact
optimum as two rulers, and a histogram of the group populations.

One panel, peer group 3, in the two-ruler format (black uniform dots
y-jittered, published ruler above, optimal ruler below, cut points attached
to their own ruler and reaching 2/3 of the way into the dot band, optimal
cut points dashed, mean-score ticks on the rulers, the sum-of-squares pair
at the right in the ruler colors), plus the peer-3 star-group population
histogram below. With --single-panel the histogram is omitted. Peer groups
4 and 5 render in the same format in the appendix figure
(fig_allbins_medicare.py).

DATA (every printed number is checked against these files):
  scores    hospital/work/R_output/fullprec_2026apr.csv
  census    hospital/runs/exact_census_2026.json  (assignments, star
            changes, sums of squares)
  cuts      hospital/runs/mechanism_allupward.json    (cut intervals, centers)
  totals    runs/star_migrations.json                 (sha256-16 pinned below)

Stars are compared before CMS's safety cap.
Deterministic: fixed SOURCE_DATE_EPOCH and fixed jitter seed; a rerun is
byte-identical. Exit 0 only if every assert passes.
Output: ../../figures/fig_migrations.pdf (relative to this script), or
fig_migrations_single.pdf with --single-panel.
"""
import csv
import hashlib
import json as _json


def canon_sha16(path):
    return hashlib.sha256(_json.dumps(_json.load(open(path)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
import json
import math
import os
import re
import sys

os.environ["SOURCE_DATE_EPOCH"] = "1754092800"  # fixed build date for a byte-identical PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SINGLE_PANEL = "--single-panel" in sys.argv
OUT = os.path.join(HERE, "..", "..", "figures", "fig_migrations_single.pdf" if SINGLE_PANEL else "fig_migrations.pdf")

_ROOT = os.path.normpath(os.path.join(HERE, ".."))
CSV = os.path.join(_ROOT, "hospital", "work", "R_output", "fullprec_2026apr.csv")
CENSUS = os.path.join(_ROOT, "hospital", "runs", "exact_census_2026.json")
MECH = os.path.join(_ROOT, "hospital", "runs", "mechanism_allupward.json")
BANK = os.path.join(_ROOT, "runs", "star_migrations.json")
BANK_SHA16 = "7980c47f21a76209"

GROUPS = ["peer3", "peer4", "peer5"]  # by measure-group count 3/4/5

# colors live in binlib (the shared two-ruler format)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    # -- totals file identity -----------------------------------------
    assert canon_sha16(BANK) == BANK_SHA16, "star_migrations.json sha256 mismatch"
    bank = json.load(open(BANK))
    census = json.load(open(CENSUS))
    mech = json.load(open(MECH))

    # -- hospitals: score and published star (before the safety cap) per peer group
    rows = list(csv.DictReader(open(CSV)))
    rated = [r for r in rows if r["star_precap"] not in ("", "NA")]
    assert len(rated) == census["total_rated"] == 3203
    data = {g: {"pid": [], "score": [], "ship": []} for g in GROUPS}
    for r in rated:
        g = "peer" + r["Total_measure_group_cnt"]
        data[g]["pid"].append(r["PROVIDER_ID"])
        data[g]["score"].append(float(r["summary_score_17g"]))
        data[g]["ship"].append(int(r["star_precap"]))

    # -- optimal assignment = published, except the hospitals whose star differs
    detail = census["star_differs_detail"]
    assert len(detail) == census["total_hospitals_star_differs"] == 213
    flips = {d["provider_id"]: d for d in detail}
    for g in GROUPS:
        dd = data[g]
        dd["dp"] = list(dd["ship"])
        for i, pid in enumerate(dd["pid"]):
            if pid in flips:
                d = flips[pid]
                assert d["peer"] == g and d["shipped"] == dd["ship"][i]
                assert d["dp_optimum"] == d["shipped"] + 1  # all one step up
                dd["dp"][i] = d["dp_optimum"]

    # -- check every printed count against the census and totals files
    b26 = bank["hospital_stars"]["years"]["2026"]
    assert b26["n_flips"] == 213 and census["moves_up"] == 213
    assert census["moves_down"] == 0
    assert b26["migrations"] == {"1->2": 20, "2->3": 68, "3->4": 92,
                                 "4->5": 33} == census["moves_total"]
    per_group_flips = {}
    for g in GROUPS:
        dd, cg = data[g], census["peer_groups"][g]
        assert len(dd["pid"]) == cg["n"]
        shipc = {str(s): dd["ship"].count(s) for s in range(1, 6)}
        dpc = {str(s): dd["dp"].count(s) for s in range(1, 6)}
        assert shipc == cg["cms_star_counts"]
        assert dpc == cg["dp_star_counts"]
        moved = [i for i in range(len(dd["pid"])) if dd["dp"][i] != dd["ship"][i]]
        assert len(moved) == cg["hospitals_star_differs"]
        mv = {}
        for i in moved:
            mv[f"{dd['ship'][i]}->{dd['dp'][i]}"] = \
                mv.get(f"{dd['ship'][i]}->{dd['dp'][i]}", 0) + 1
        assert mv == cg["moves"]
        per_group_flips[g] = len(moved)
        dd["moved"] = set(moved)
    assert [per_group_flips[g] for g in GROUPS] == [55, 68, 90]
    # before/after star groups of the hospitals whose star changes, from the totals file
    assert b26["before_bins"] == {"1": 20, "2": 68, "3": 92, "4": 33}
    assert b26["after_bins"] == {"2": 20, "3": 68, "4": 92, "5": 33}

    # -- cut points: recorded intervals == derived extremes, full precision
    for g in GROUPS:
        dd = data[g]
        mg = mech["stage1_boundary_dissection"]["per_group"][g]
        assert mg["shipped_block_sizes"] == [dd["ship"].count(s) for s in range(1, 6)]
        assert mg["dp_block_sizes"] == [dd["dp"].count(s) for s in range(1, 6)]
        dd["cuts"] = []
        for k, b in enumerate(mg["boundaries"], start=1):
            assert b["cut"] == f"{k}|{k+1}"
            for key, lab in (("shipped_cut_interval", "ship"),
                             ("dp_cut_interval", "dp")):
                lo_bank, hi_bank = b[key]
                lo = max(dd["score"][i] for i in range(len(dd["pid"]))
                         if dd[lab][i] == k)
                hi = min(dd["score"][i] for i in range(len(dd["pid"]))
                         if dd[lab][i] == k + 1)
                # mechanism_allupward.json stores the cut intervals as
                # the 15-significant-digit decimals written by R's
                # write.csv; the CSV carries 17-digit round-trip doubles.
                # Compare at 15 digits, exactly; both readings give the
                # same census.
                assert float(f"{lo:.15g}") == lo_bank, (g, k, key)
                assert float(f"{hi:.15g}") == hi_bank, (g, k, key)
            ship_mid = 0.5 * (b["shipped_cut_interval"][0]
                              + b["shipped_cut_interval"][1])
            dp_mid = 0.5 * (b["dp_cut_interval"][0] + b["dp_cut_interval"][1])
            assert ship_mid > dp_mid  # all 12 published cut points lie above the optimal ones
            # every hospital whose star changes at this boundary lies
            # strictly between the two cut points
            movers_here = [i for i in dd["moved"] if dd["ship"][i] == k]
            assert len(movers_here) == b["hospitals_crossing"]
            for i in movers_here:
                assert dp_mid < dd["score"][i] < ship_mid
            dd["cuts"].append({"ship": ship_mid, "dp": dp_mid})
        # centers: optimal from mechanism_allupward.json (full precision,
        # re-derived); published = converged k-means centers = group
        # means, checked against the 6-decimal log of the re-run of CMS's
        # code in the same file.
        dp_centers = mg["dp_centers"]
        for s in range(1, 6):
            pts = [dd["score"][i] for i in range(len(dd["pid"]))
                   if dd["dp"][i] == s]
            mean = math.fsum(pts) / len(pts)
            assert abs(mean - dp_centers[s - 1]) <= 1e-10
        logline = [ln for ln in
                   mech["shipped_algorithm_characterization"]["replica_logs"][g]
                   if ln.startswith("stage1 centers:")][0]
        log_centers = [float(x) for x in
                       re.findall(r"-?\d+\.\d+", logline.split("|")[0])]
        ship_centers = []
        for s in range(1, 6):
            pts = [dd["score"][i] for i in range(len(dd["pid"]))
                   if dd["ship"][i] == s]
            mean = math.fsum(pts) / len(pts)
            assert abs(mean - log_centers[s - 1]) <= 5e-7
            ship_centers.append(mean)
        dd["ship_centers"] = ship_centers
        dd["dp_centers"] = dp_centers
    return data, census


def as_instance(data, census, g):
    """Package one peer group for binlib.draw_strip, with the
    sum-of-squares pair from the hospital census file (checked: optimal
    strictly below published, gap consistent, distinct at the printed
    precision)."""
    dd, cg = data[g], census["peer_groups"][g]
    deltas = [dd["dp"][i] - dd["ship"][i] for i in range(len(dd["pid"]))]
    moved = [i for i, d in enumerate(deltas) if d]
    assert set(moved) == dd["moved"] and all(deltas[i] == 1 for i in moved)
    ship_n = [dd["ship"].count(s) for s in range(1, 6)]
    dp_n = [dd["dp"].count(s) for s in range(1, 6)]
    s_ship, s_opt = cg["ssq_cms_float"], cg["ssq_optimal_float"]
    assert s_opt < s_ship
    assert abs((s_ship - s_opt) - cg["ssq_gap_float"]) < 1e-12
    return {
        "title": f"Peer group {g[-1]}", "n": len(dd["pid"]),
        "scores": [float(s) for s in dd["score"]], "deltas": deltas,
        "pub_cuts": [c["ship"] for c in dd["cuts"]],
        "opt_cuts": [c["dp"] for c in dd["cuts"]],
        "pub_counts": ship_n, "opt_counts": dp_n,
        "pub_means": dd["ship_centers"], "opt_means": dd["dp_centers"],
        "ssq_pub": s_ship, "ssq_opt": s_opt, "ssq_decimals": 4,
        "up": len(moved), "dn": 0, "optimal": False,
    }


def draw(data, census):
    plt.rcParams.update({
        "font.size": 8, "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
    })
    import binlib
    inst = as_instance(data, census, "peer3")
    assert inst["up"] == 55 and inst["n"] == 177
    fig = plt.figure(figsize=(6.3, 4.9) if not SINGLE_PANEL else (6.3, 2.75))
    rng = np.random.default_rng(20260802)
    # ---------------- the peer-3 score line --------------------------
    ax = fig.add_axes([0.030, 0.475, 0.835, 0.375] if not SINGLE_PANEL else [0.030, 0.16, 0.835, 0.64])
    lo, hi = min(inst["scores"]), max(inst["scores"])
    pad = 0.03 * (hi - lo)
    binlib.draw_strip(ax, inst, rng, (lo - pad, hi + pad), dot=4.0,
                      numeral_size=7.5, ssq_size=7.2, ruler_labels=True,
                      label_size=6.8)
    ax.set_xlabel("2026 summary score", fontsize=8)
    ax.set_title(f"One of the 18 hospital clusterings: 2026, peer group 3 ({inst['n']} hospitals)", loc="left",
                 fontsize=8, pad=4)
    fig.legend(handles=binlib.legend_handles(
        dot_label="one hospital's 2026 summary score"),
        loc="upper center", ncol=3, frameon=False, fontsize=6.4,
        bbox_to_anchor=(0.5, 1.002), handletextpad=0.5,
        columnspacing=1.2)
    # ---------------- the peer-3 group populations --------------------
    if not SINGLE_PANEL:
        axh = fig.add_axes([0.315, 0.085, 0.37, 0.235])
        binlib.draw_hist(axh, inst["pub_counts"], inst["opt_counts"],
                         label_size=7.5, tick_size=7.0, legend=True)
        axh.set_ylabel("hospitals", fontsize=7.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)


def main():
    data, census = load()
    draw(data, census)
    print(f"fig_migrations: all asserts passed; wrote {os.path.normpath(OUT)}"
          f" sha256-16 {sha256(os.path.normpath(OUT))[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
