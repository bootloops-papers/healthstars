#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""fig_allbins_cahps.py — every CAHPS clustering with public entity-level
scores (fig:allbins-cahps): the 8 HCAHPS current-refresh starred measures
and the 6 ICH CAHPS measures, one strip each, small multiples, in the
format of fig_migrations.py.

DATA (census files and CMS score files; every printed count is checked in
the binlib loaders):
  HCAHPS  HCAHPS_CENSUS_ALLVINTAGES.json (8c7a4d233723cce6), current
          block, data HCAHPS-Hospital.csv (sha256 e9d7d413..., as
          recorded in the census file)
  ICH     ICH_CENSUS.json (6559b2be746ac3eb), data
          ICH_CAHPS_FACILITY.csv (sha256 cf67c06c..., as recorded in the
          census file)
Deterministic: fixed SOURCE_DATE_EPOCH and fixed jitter seed.
Output: ../../figures/fig_allbins_cahps.pdf (relative to this script)
"""
import os
import sys

os.environ["SOURCE_DATE_EPOCH"] = "1754092800"  # fixed build date for a byte-identical PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import binlib
from binlib import draw_strip, legend_handles, sha16

OUT = os.path.join(HERE, "..", "..", "figures", "fig_allbins_cahps.pdf")


def main():
    hc, cur = binlib.load_hcahps()
    ich, ibank = binlib.load_ich()
    # figure-level totals, checked before drawing
    assert cur["off_optimum"] == 7 and cur["total_migrating"] == 3363
    assert sum(v["up"] + v["dn"] for v in hc.values()) == 3363
    assert hc["H_RECMND"]["optimal"] and hc["H_RECMND"]["up"] == 0
    assert ibank["headline"]["off_optimum"] == 6
    assert ibank["headline"]["total_migrating"] == 4061
    assert sum(v["up"] + v["dn"] for v in ich.values()) == 4061
    assert all(v["n"] == 3176 for v in hc.values())
    assert all(v["n"] == 2594 for v in ich.values())

    plt.rcParams.update({
        "font.size": 7.5, "axes.linewidth": 0.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.3, 7.6))
    # wspace/right leave room for the sum-of-squares pair right of each
    # ICH panel (the HCAHPS census file carries no such pair, so the
    # upper block prints none; binlib omits the text when ssq_pub is None)
    gs = fig.add_gridspec(8, 2, hspace=0.72, wspace=0.44,
                          left=0.025, right=0.910, top=0.900, bottom=0.054,
                          height_ratios=[1, 1, 1, 1, 0.34, 1, 1, 1])
    rng = np.random.default_rng(20260802)

    # --- HCAHPS block: rows 0-3, common axis --------------------------
    hx = (53, 101.5)
    hc_keys = [m for m, _ in binlib.HC_MEASURES]
    for i, m in enumerate(hc_keys):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c])
        draw_strip(ax, hc[m], rng, hx, dot=1.6, numeral_size=5.4,
                   ssq_size=5.2)
        ax.set_title(binlib.title_of(hc[m], with_n=False), fontsize=6.8,
                     loc="left", pad=2)
        if r < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("HCAHPS linear mean score (0-100)", fontsize=7)

    # --- ICH block: rows 5-7, common axis -----------------------------
    ix = (44, 101.5)
    for i, (key, _, _, _) in enumerate(binlib.ICH_MEASURES):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[5 + r, c])
        draw_strip(ax, ich[key], rng, ix, dot=1.6, numeral_size=5.4,
                   ssq_size=5.2)
        ax.set_title(binlib.title_of(ich[key], with_n=False),
                     fontsize=6.8, loc="left", pad=2)
        if r < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("ICH CAHPS linearized score (0-100)", fontsize=7)

    fig.text(0.025, 0.925, "HCAHPS (hospitals), current refresh: "
             "n = 3,176 hospitals per measure",
             fontsize=8, fontweight="bold")
    fig.text(0.025, 0.394, "ICH CAHPS (dialysis facilities), current "
             "file: n = 2,594 facilities per measure",
             fontsize=8, fontweight="bold")
    fig.legend(handles=legend_handles(
        ship_label="CMS's published star groups", dot_label="one hospital or facility"),
        loc="upper center", ncol=3, frameon=False, fontsize=6.3,
        bbox_to_anchor=(0.5, 0.998), handletextpad=0.5,
        columnspacing=0.9)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)
    print(f"fig_allbins_cahps: all asserts passed; wrote "
          f"{os.path.normpath(OUT)} sha256-16 "
          f"{sha16(os.path.normpath(OUT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
