#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""Per-measure optimality gaps (sum of squares of CMS's Ward grouping minus
the exact minimum), star years 2024-2026, one panel per star year with
measures sorted by gap on a log axis; measures whose gap is exactly zero
are drawn as open markers at a display floor below an axis break.

Input: runs/exact_gaps_fullsample_docsonly.json (the optimality-gap
table: one row per cut-point computation, with the exact rational gap, its
float rendering, the off-optimum flag and the number of optimal groupings).
Every displayed value is checked against that file; any mismatch exits
nonzero.
Output: figures/fig_gaps.pdf.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (AXIS, GRID, INK, INK2, MUTED, SURFACE, YEAR_STEP,
                    Checker, despine, rc)

import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent.parent
SOURCE = PAPER / "runs" / "exact_gaps_fullsample_docsonly.json"
OUT = Path(__file__).resolve().parent.parent.parent / "figures" / "fig_gaps.pdf"

YEARS = ("2024", "2025", "2026")
ZFLOOR = 2e-5          # display floor for exact-zero gaps (below axis break)

check = Checker("fig_gaps")


def main():
    rows = json.loads(SOURCE.read_text())

    # ---- source integrity: the values behind every displayed point --------
    # the gap table carries all 123 published computations; three rows have
    # fewer than five distinct scores and a gap of exactly 0.
    check(len(rows) == 123, f"expected 123 rows, got {len(rows)}")
    for r in rows:
        g = Fraction(r["gap"])
        check(g >= 0, f"negative gap {r['year']} {r['mid']}")
        check(float(g) == r["gap_float"],
              f"gap_float is not the float of the exact gap: {r['year']} {r['mid']}")
        check(r["off_optimum"] == (g > 0),
              f"off_optimum flag inconsistent with exact gap: {r['year']} {r['mid']}")
        check(r["dp_n_optima"] == 1,
              f"DP optimum not unique: {r['year']} {r['mid']}")

    byyear = {y: sorted((r for r in rows if r["year"] == y),
                        key=lambda r: -r["gap_float"]) for y in YEARS}

    # headline counts stated in caption/annotations
    check(sum(len(v) for v in byyear.values()) == 123, "year partition != 123")
    n_off_total = sum(1 for r in rows if r["off_optimum"])
    check(n_off_total == 104, f"off-optimum total {n_off_total} != 104")
    exp_n = {"2024": 40, "2025": 40, "2026": 43}
    exp_off = {"2024": 36, "2025": 33, "2026": 35}
    for y in YEARS:
        check(len(byyear[y]) == exp_n[y],
              f"{y}: n={len(byyear[y])} != {exp_n[y]}")
        n_off = sum(1 for r in byyear[y] if r["off_optimum"])
        check(n_off == exp_off[y], f"{y}: off={n_off} != {exp_off[y]}")
    n_zero = sum(1 for r in rows if not r["off_optimum"])
    # 19 = 16 that attain the minimum + 3 with fewer than five distinct scores (k<5; gap 0)
    check(n_zero == 19, f"zero-gap cells {n_zero} != 19")

    # per-panel maximum (direct-labeled)
    exp_max = {"2024": ("C17", "Transitions of Care", 1813.0112261841475)}
    mx24 = byyear["2024"][0]
    check((mx24["mid"], mx24["name"], mx24["gap_float"]) == exp_max["2024"],
          f"2024 max row changed: {mx24['mid']} {mx24['gap_float']}")
    check(max(r["gap_float"] for r in rows) == mx24["gap_float"],
          "2024 C17 is not the global max")

    # ---- figure ------------------------------------------------------------
    rc()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.10, left=0.10, right=0.99, top=0.90,
                        bottom=0.17)

    for ax, y in zip(axes, YEARS):
        rows_y = byyear[y]
        pos = [r for r in rows_y if r["off_optimum"]]
        zer = [r for r in rows_y if not r["off_optimum"]]

        xs_p = list(range(1, len(pos) + 1))
        ys_p = [r["gap_float"] for r in pos]
        xs_z = list(range(len(pos) + 1, len(rows_y) + 1))

        # displayed values == source values, point by point
        for xv, yv, r in zip(xs_p, ys_p, pos):
            check(yv == r["gap_float"], f"{y} rank {xv} plotted value drifted")
        for r in zer:
            check(r["gap_float"] == 0.0, f"{y} zero-marker row has gap != 0")

        ax.set_yscale("log")
        ax.set_ylim(8e-6, 9e3)
        ax.set_xlim(0, 41.5)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        despine(ax)

        ax.scatter(xs_p, ys_p, s=11, color=YEAR_STEP[y], edgecolors=SURFACE,
                   linewidths=0.5, zorder=3)
        ax.scatter(xs_z, [ZFLOOR] * len(xs_z), s=11, facecolors=SURFACE,
                   edgecolors=MUTED, linewidths=0.7, zorder=3)

        # axis-break glyph on the left spine (between min nonzero and floor)
        for yb in (6.0e-5, 1.05e-4):
            ax.plot([-0.012, 0.012], [yb * 0.82, yb * 1.22],
                    transform=ax.get_yaxis_transform(), color=SURFACE,
                    solid_capstyle="butt",
                    linewidth=2.6, clip_on=False, zorder=4)
            ax.plot([-0.012, 0.012], [yb * 0.82, yb * 1.22],
                    transform=ax.get_yaxis_transform(), color=AXIS,
                    linewidth=0.8, clip_on=False, zorder=5)

        ax.set_title(y, color=INK, pad=4)
        n_off, n_all = len(pos), len(rows_y)
        check(n_off == exp_off[y] and n_all == exp_n[y],
              f"{y} annotation counts drifted")
        ax.text(0.97, 0.96, f"{n_off}/{n_all} off optimum",
                transform=ax.transAxes, ha="right", va="top", color=INK2,
                fontsize=8)

        # direct-label the panel maximum (the extreme, selectively)
        top = rows_y[0]
        ax.annotate(f"max {top['gap_float']:.1f}", (1, top["gap_float"]),
                    xytext=(3, 5.5), textcoords="offset points", ha="left",
                    va="baseline", color=INK2, fontsize=7.5)

        ax.set_xticks([1, 10, 20, 30, 40])
        ax.tick_params(length=2.5, width=0.6)

    # shared y ticks incl. the zero floor
    ax0 = axes[0]
    ax0.set_yticks([ZFLOOR, 1e-4, 1e-2, 1, 1e2])
    ax0.set_yticklabels(["0 (exact)", r"$10^{-4}$", r"$10^{-2}$", r"$1$",
                         r"$10^{2}$"])
    ax0.set_ylabel("sum of squares of CMS's grouping minus the true minimum\n"
                   "(squared score units; log scale)")
    axes[1].set_xlabel("measures within star year, sorted by gap")
    ax0.text(0.03, 0.05,
             "open markers: CMS's grouping attains the minimum,\n"
             "drawn at display floor below the break",
             transform=ax0.transAxes, ha="left", va="bottom", color=MUTED,
             fontsize=7.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    check.report()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
