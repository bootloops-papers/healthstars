#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""fig_ich_forty.py — the two 40% dialysis-survey measures (fig:ich-forty).

The two ICH CAHPS measures named in the text, each moving 1,042 of 2,594
facilities (40%), one measure entirely upward and one entirely downward,
in the two-ruler format of binlib.draw_strip. The ICH census file carries
the exact sum-of-squares pair per measure, so both panels print it
(recomputed as exact rationals in binlib.load_ich).

DATA (census file and CMS facility file; every printed count is checked in
binlib.load_ich against stars_census/legc_ich/):
  census  ICH_CENSUS.json        sha256-16 6559b2be746ac3eb
  data    ICH_CAHPS_FACILITY.csv sha256-16 cf67c06c... (as recorded in
          the census file)
Deterministic: fixed SOURCE_DATE_EPOCH and fixed jitter seed; a rerun is
byte-identical. Exit 0 only if every assert passes.
Output: ../../figures/fig_ich_forty.pdf (relative to this script)
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

OUT = os.path.join(HERE, "..", "..", "figures", "fig_ich_forty.pdf")
PANELS = ["nephrologists' communication and caring",
          "rating of the nephrologist"]


def main():
    ich, _ = binlib.load_ich()
    a = ich[PANELS[0]]
    b = ich[PANELS[1]]
    # the paired headline facts, asserted before they render
    assert a["n"] == b["n"] == 2594
    assert a["up"] == 1042 and a["dn"] == 0
    assert b["dn"] == 1042 and b["up"] == 0

    plt.rcParams.update({
        "font.size": 8, "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.3, 3.5))
    gs = fig.add_gridspec(2, 1, hspace=0.52, left=0.02, right=0.865,
                          top=0.77, bottom=0.13)
    rng = np.random.default_rng(20260802)
    xlim = (44, 101.5)
    for ax, inst in zip((fig.add_subplot(gs[0]), fig.add_subplot(gs[1])),
                        (a, b)):
        draw_strip(ax, inst, rng, xlim, dot=2.2, numeral_size=6.2,
                   ssq_size=6.2)
        ax.set_title(binlib.title_of(inst), fontsize=7.5, loc="left", pad=2)
    axes = fig.axes
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("linearized score (0-100)", fontsize=8)
    fig.legend(handles=legend_handles(
        ship_label="CMS's published star groups", dot_label="one rated facility"),
        loc="upper center", ncol=3, frameon=False, fontsize=6.3,
        bbox_to_anchor=(0.5, 1.005), handletextpad=0.5,
        columnspacing=0.9)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="pdf")
    plt.close(fig)
    print(f"fig_ich_forty: all asserts passed; wrote "
          f"{os.path.normpath(OUT)} sha256-16 "
          f"{sha16(os.path.normpath(OUT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
