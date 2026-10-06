#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""fig_allbins_medicare.py — hospital stars and Medicare Advantage in the
two-ruler format (fig:allbins-medicare). Peer group 3 is the main-text
figure (fig_migrations.py); peer groups 4 and 5 render here, in the same
format, with their star-group population histograms, followed by three
Medicare Advantage cut-point computations, the largest-gap one per star
year: Ward's method (the clustering step of CMS's method) and the exact
optimum on the same score list after outlier trimming, before resampling,
the guardrail and rounding. The published MA cut points are averages over
ten subsamples and are not drawn as clustering output (the caption states
this). The sum-of-squares pair beside every panel comes from the census
files: hospital pairs from the hospital census file, MA pairs from the
gap table's exact rationals.

DATA and checks:
  hospital  fig_migrations.load() (every check of that script re-runs:
            fullprec_2026apr.csv, exact_census_2026.json,
            mechanism_allupward.json and star_migrations.json)
  MA        runs/exact_gaps_fullsample_docsonly.json (sha256 pinned
            below); trimmed score lists rebuilt with the gap table's own
            job builder and outlier trim (src/); n_input, n_trimmed, the
            Ward cut points, the exact Ward and optimal sums of squares,
            and the recount of star changes all checked against the row.
Deterministic: fixed SOURCE_DATE_EPOCH and fixed jitter seed.
Output: ../../figures/fig_allbins_medicare.pdf (relative to this script)
"""
import json
import os
import sys
from fractions import Fraction as F

os.environ["SOURCE_DATE_EPOCH"] = "1754092800"  # fixed build date for a byte-identical PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.normpath(os.path.join(HERE, "..", "src")))
import binlib
from binlib import canon_sha16, draw_strip, legend_handles, sha16
import fig_migrations_ha as fm

OUT = os.path.join(HERE, "..", "..", "figures", "fig_allbins_medicare.pdf")
MA_BANK = os.path.normpath(os.path.join(HERE, "..", "runs",
                                        "exact_gaps_fullsample_docsonly.json"))
# first 16 hex digits of the sha256 of the gap table the figure reads
# (123 computations)
MA_BANK_SHA16 = "a56f8768c47d73b1"
MA_PICKS = [("2024", "C17", "ALL"), ("2025", "C14", "ALL"),
            ("2026", "C21", "ALL")]
GLABEL = {"peer3": "peer group 3", "peer4": "peer group 4",
          "peer5": "peer group 5"}


def hospital_instances():
    """Peer groups 4 and 5 (peer group 3 is the main-text figure);
    fm.load() re-runs every fig_migrations check, and the sum-of-squares
    pairs come from the hospital census file with the gap re-checked."""
    data, census = fm.load()
    out = []
    for g in ("peer4", "peer5"):
        inst = fm.as_instance(data, census, g)
        inst["title"] = GLABEL[g].capitalize()
        out.append(inst)
    # 213 star changes in total = 55 (peer3, the main-text figure) + 68 + 90 (here)
    assert [inst["up"] for inst in out] == [68, 90]
    assert 55 + sum(inst["up"] for inst in out) == 213
    return out


def ma_instances():
    from exact_gaps_docsonly import build_jobs
    from tukey import tukey_trim
    from ward_cluster_tie import ward_sas_cut
    from ward_cluster import cutpoints_from_clusters
    from dp_kmeans import partition_ssq
    assert canon_sha16(MA_BANK) == MA_BANK_SHA16, "MA gap table sha256 mismatch"
    bank = {(r["year"], r["mid"], r["org"]): r
            for r in json.load(open(MA_BANK))}
    jobs = {(j["year"], j["mid"], j["org"]): j for j in build_jobs()}
    out = []
    for key in MA_PICKS:
        j, b = jobs[key], bank[key]
        assert b["membership"] == "fence-exact" and b["k"] == 5
        assert j["higher"] is True
        scores = [s for _, s in j["kept"]]
        assert len(scores) == b["n_input"]
        mask, _, _ = tukey_trim(scores, cap_lo="0", cap_hi=j["cap_hi"])
        trimmed = [s for s, m in zip(scores, mask) if m]
        assert len(trimmed) == b["n_trimmed"]
        vals = [float(s) for s in trimmed]
        ward_cl = ward_sas_cut(vals, b["k"])
        labels = [0] * len(vals)
        for ci, c in enumerate(ward_cl):
            for i in c:
                labels[i] = ci
        assert str(partition_ssq(trimmed, labels)) == b["ward_ssq"]
        ward_cps = cutpoints_from_clusters(ward_cl, vals, True)
        assert [str(c) for c in ward_cps] == b["ward_cutpoints"]
        dp_cps = [F(c) for c in b["dp_cutpoints"]]

        def star_of(cps, x):
            xs = F(x)
            return 1 + sum(1 for t in cps if xs >= t)

        dp_labels = [star_of(dp_cps, s) - 1 for s in trimmed]
        assert str(partition_ssq(trimmed, dp_labels)) == b["dp_ssq"]
        ward_f = [F(str(c)) for c in ward_cps]
        deltas = [star_of(dp_cps, s) - star_of(ward_f, s) for s in trimmed]
        n_mv = sum(1 for d in deltas if d)
        assert n_mv == b["n_star_delta_fullsample"]
        # per-group membership under both cut-point sets (exact star
        # reads), for the ruler counts and the mean-score center ticks
        pub_bins = {s: [] for s in range(1, 6)}
        opt_bins = {s: [] for s in range(1, 6)}
        for s in trimmed:
            pub_bins[star_of(ward_f, s)].append(F(s))
            opt_bins[star_of(dp_cps, s)].append(F(s))
        pub_counts = [len(pub_bins[s]) for s in range(1, 6)]
        opt_counts = [len(opt_bins[s]) for s in range(1, 6)]
        assert sum(pub_counts) == sum(opt_counts) == b["n_trimmed"]
        pub_means = [float(sum(pub_bins[s]) / len(pub_bins[s]))
                     for s in range(1, 6)]
        opt_means = [float(sum(opt_bins[s]) / len(opt_bins[s]))
                     for s in range(1, 6)]
        # the exact sum-of-squares pair from the gap table; optimum strictly below
        ssq_pub, ssq_opt = F(b["ward_ssq"]), F(b["dp_ssq"])
        assert ssq_opt < ssq_pub
        # the 2026 measure's full name is too long for a panel title; the
        # short form renders and the caption carries the full name.
        short = {"C21": "Follow-up after Emergency Department Visit"}
        out.append({
            "title": f"MA {b['year']}, {short.get(b['mid'], b['name'])}",
            "n": len(trimmed),
            "scores": vals, "deltas": deltas,
            "pub_cuts": [float(c) for c in ward_cps],
            "opt_cuts": [float(c) for c in dp_cps],
            "pub_counts": pub_counts, "opt_counts": opt_counts,
            "pub_means": pub_means, "opt_means": opt_means,
            "ssq_pub": float(ssq_pub), "ssq_opt": float(ssq_opt),
            "ssq_decimals": 1,
            "up": sum(1 for d in deltas if d > 0),
            "dn": sum(1 for d in deltas if d < 0),
            "optimal": not b["off_optimum"],
        })
    return out


def main():
    hosp = hospital_instances()
    ma = ma_instances()
    plt.rcParams.update({
        "font.size": 7.5, "axes.linewidth": 0.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.3, 7.2))
    # right margin leaves room for the sum-of-squares pairs beside every
    # panel; the histogram row carries the peer-4/5 group populations
    gs = fig.add_gridspec(7, 1, hspace=0.60,
                          left=0.025, right=0.870, top=0.885, bottom=0.056,
                          height_ratios=[1, 1, 0.82, 0.24, 1, 1, 1])
    rng = np.random.default_rng(20260802)

    hxlim = (min(min(i["scores"]) for i in hosp) - 0.12,
             max(max(i["scores"]) for i in hosp) + 0.12)
    first_hosp = first_ma = None
    for r, inst in enumerate(hosp):
        ax = fig.add_subplot(gs[r])
        if r == 0:
            first_hosp = ax
        draw_strip(ax, inst, rng, hxlim, dot=1.9, numeral_size=5.6,
                   ssq_size=5.8)
        ax.set_title(binlib.title_of(inst), fontsize=7, loc="left", pad=2)
        if r < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("2026 summary score", fontsize=7)

    # the peer-4/5 star-group populations, published vs exact optimum
    # (leading pad column indents the row so the y tick labels fit)
    gs_h = gs[2].subgridspec(1, 3, width_ratios=[0.16, 1, 1],
                             wspace=0.42)
    for c, inst in enumerate(hosp):
        axh = fig.add_subplot(gs_h[c + 1])
        binlib.draw_hist(axh, inst["pub_counts"], inst["opt_counts"],
                         label_size=6.5, tick_size=6.0, legend=(c == 0))
        if c == 0:
            axh.set_ylim(0, axh.get_ylim()[1] * 1.22)  # legend headroom
        # the peer label rides in the xlabel (a title would collide
        # with the strip xlabel above at this row spacing)
        axh.set_xlabel(f"star level, {inst['title'].lower()}",
                       fontsize=6.5)
        if c == 0:
            axh.set_ylabel("hospitals", fontsize=6.5)

    for r, inst in enumerate(ma):
        ax = fig.add_subplot(gs[4 + r])
        if r == 0:
            first_ma = ax
        draw_strip(ax, inst, rng, (0, 101.5), dot=1.9, numeral_size=5.6,
                   ssq_size=5.8)
        ax.set_title(binlib.title_of(inst), fontsize=7, loc="left", pad=2)
        if r < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("measure score (percent)", fontsize=7)

    # block headers placed above each block's first panel title
    fig.text(0.025, first_hosp.get_position().y1 + 0.045,
             "Hospital overall stars 2026, peer groups 4 and 5 "
             "(CMS's k-means, rerun)",
             fontsize=8, fontweight="bold")
    fig.text(0.025, first_ma.get_position().y1 + 0.045,
             "Medicare Advantage, largest gap per star year "
             "(Ward's method on the full score list)",
             fontsize=8, fontweight="bold")
    fig.legend(handles=legend_handles(
        ship_label="CMS's clustering, recomputed",
        dot_label="one hospital or contract"),
        loc="upper center", ncol=3, frameon=False, fontsize=6.3,
        bbox_to_anchor=(0.5, 0.998), handletextpad=0.5, columnspacing=0.9)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)
    print(f"fig_allbins_medicare: all asserts passed; wrote "
          f"{os.path.normpath(OUT)} sha256-16 "
          f"{sha16(os.path.normpath(OUT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
