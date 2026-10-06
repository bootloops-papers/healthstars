#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""Worked example: one measure through CMS's cut-point method, at measure
level, for CMS's method and the exact optimum side by side.

2026 star year, C28 "Complaints about the Health Plan" (Part C), the
sampled realization seq_quota_last, from the measure_results block of
runs/enclosure_census_docsonly.json (the resampled rerun). Panel (a) shows
both computations' raw resampled cut-point means against the prior-year
(2025) published cut points and the guardrail window; panel (b) shows the
cut points after the guardrail, where three boundaries are clamped
identically and the 5|4 boundary keeps the difference between the two
computations. Prior-year cut points are read from
data/parsed/cutpoints_2025_current.csv; because measure IDs shift between
star years (the 2025 ID is C25), the row is matched and checked by name.
No contract IDs appear.

Every displayed value is checked against the two named sources; any
mismatch exits nonzero. The guardrail half-width (0.04) is not read from
any field: it is recovered as prior minus final at the three clamped
boundaries and checked to be identical across all three.
Output: figures/fig_worked_example.pdf.
"""
import csv
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (BLUE, GREEN, GRID, INK, INK2, MUTED, SURFACE,
                    Checker, despine, rc)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PAPER = Path(__file__).resolve().parent.parent
SOURCE = PAPER / "runs" / "enclosure_census_docsonly.json"
PRIOR_CSV = PAPER / "data" / "parsed" / "cutpoints_2025_current.csv"
OUT = Path(__file__).resolve().parent.parent.parent / "figures" / "fig_worked_example.pdf"

NAME = "Complaints about the Health Plan"
BOUNDARY_LABELS = ["5 | 4", "4 | 3", "3 | 2", "2 | 1"]  # star boundaries,
# ascending complaint rate (lower rate = more stars)

check = Checker("fig_worked_example")


def round2(x):
    """Exact round-half-away to the 0.01 display grid, as Fraction."""
    return F(int(x * 100 + F(1, 2)) if x >= 0 else -int(-x * 100 + F(1, 2)),
             100)


def main():
    # ---- source row --------------------------------------------------------
    doc = json.loads(SOURCE.read_text())
    hits = [m for m in doc["measure_results"]
            if (m["year"], m["mid"], m["org"], m["realization"])
            == ("2026", "C28", "ALL", "seq_quota_last")]
    check(len(hits) == 1, f"expected 1 source row, got {len(hits)}")
    m = hits[0]
    check(m["name"] == NAME, f"source name {m['name']!r} != {NAME!r}")
    check(m["guardrail"] == "applied", "guardrail not applied in the source row")

    wm = [F(s) for s in m["ward_mean"]]
    dm = [F(s) for s in m["dp_mean"]]
    wf = [F(s) for s in m["ward_final"]]
    df = [F(s) for s in m["dp_final"]]
    check([str(x) for x in m["ward_mean"]] == ["27/250", "121/500",
                                               "201/500", "5/8"],
          "ward_mean rationals changed")
    check([str(x) for x in m["dp_mean"]] == ["119/1000", "49/200",
                                             "81/200", "78/125"],
          "dp_mean rationals changed")
    check(m["ward_final"] == ["0.11", "0.33", "0.72", "1.35"],
          "ward_final changed")
    check(m["dp_final"] == ["0.12", "0.33", "0.72", "1.35"],
          "dp_final changed")
    check(all(len(v) == 4 for v in (wm, dm, wf, df)), "expected 4 boundaries")
    check(wm == sorted(wm) and dm == sorted(dm), "means not ascending")

    # ---- prior-year published cut points, matched BY NAME (ID shift) ------
    with PRIOR_CSV.open() as fh:
        rows = [r for r in csv.DictReader(fh)]
    named = [r for r in rows if r["measure_name"] == NAME and r["part"] == "C"]
    check(len(named) == 5, f"expected 5 star rows for {NAME!r}, got {len(named)}")
    check({r["measure_id"] for r in named} == {"C25"},
          "2025 ID for the name is not C25 (ID-shift check)")
    check(not any(r["measure_id"] == "C28" and r["measure_name"] == NAME
                  for r in rows), "2025 file also lists the name under C28")
    by_star = {int(r["star_level"]): r for r in named}
    # boundaries ascending = hi of stars 5,4,3,2 (lower rate is better)
    prior = [F(by_star[s]["hi"]) for s in (5, 4, 3, 2)]
    check(prior == [F("0.12"), F("0.37"), F("0.76"), F("1.39")],
          f"2025 published cut points changed: {[str(p) for p in prior]}")
    for s in (2, 3, 4):  # cell edges consistent: lo(star s) == hi(star s+1)
        check(F(by_star[s]["lo"]) == F(by_star[s + 1]["hi"]),
              f"2025 cell edges inconsistent at star {s}")

    # ---- guardrail window: half-width recovered from the clamped finals ---
    caps = {prior[i] - wf[i] for i in (1, 2, 3)} | \
           {prior[i] - df[i] for i in (1, 2, 3)}
    check(caps == {F("0.04")},
          f"clamped boundaries do not share one cap: {sorted(map(str, caps))}")
    cap = F("0.04")
    for i in (1, 2, 3):   # clamp binds: both arms' means below the window
        check(wm[i] < prior[i] - cap and dm[i] < prior[i] - cap,
              f"boundary {i}: mean not below window, clamp story wrong")
        check(wf[i] == df[i] == prior[i] - cap,
              f"boundary {i}: finals not identically at window edge")
    # slack at boundary 0: both means inside the window, finals = display-
    # rounded means, and the arm difference survives
    check(abs(wm[0] - prior[0]) <= cap and abs(dm[0] - prior[0]) <= cap,
          "boundary 0 means not inside the window")
    check(wf[0] == round2(wm[0]) and df[0] == round2(dm[0]),
          "boundary 0 finals != display-rounded means")
    check(wf[0] != df[0], "boundary 0 arm difference did not survive")

    # caption/annotation facts
    check(m["pub_match_ward"] == 1, "pub_match_ward != 1 (source row changed)")
    check(m["n_star_changes"] == 18, "n_star_changes != 18 (annotation)")
    check(all(d[1] == 4 and d[2] == 5 for d in m["deltas"])
          and len(m["deltas"]) == 18,
          "deltas are not 18 contracts moving 4 -> 5")

    # ---- figure ------------------------------------------------------------
    rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99, top=0.80,
                        bottom=0.15)
    xs = [1, 2, 3, 4]
    DX = 0.14

    for ax, title, wvals, dvals in (
            (axes[0], "(a)  raw resampled cut-point means", wm, dm),
            (axes[1], "(b)  after guardrails", wf, df)):
        despine(ax)
        ax.set_xlim(0.45, 4.55)
        ax.set_ylim(0, 1.55)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(BOUNDARY_LABELS)
        ax.set_title(title, color=INK, pad=4, loc="left")

        for x, p in zip(xs, prior):
            lo, hi = float(p - cap), float(p + cap)
            ax.add_patch(Rectangle((x - 0.30, lo), 0.60, hi - lo,
                                   facecolor=GRID, edgecolor="none", zorder=1))
            ax.plot([x - 0.30, x + 0.30], [float(p)] * 2, color=INK2,
                    linewidth=1.2, solid_capstyle="butt", zorder=2)

        wx = [x - DX for x in xs]
        dx_ = [x + DX for x in xs]
        wy = [float(v) for v in wvals]
        dy = [float(v) for v in dvals]
        for arr, src in ((wy, wvals), (dy, dvals)):
            for v_plot, v_src in zip(arr, src):
                check(v_plot == float(v_src), "plotted value drifted")
        ax.scatter(wx, wy, s=26, marker="o", color=BLUE, edgecolors=SURFACE,
                   linewidths=0.8, zorder=3)
        ax.scatter(dx_, dy, s=30, marker="D", color=GREEN, edgecolors=SURFACE,
                   linewidths=0.8, zorder=3)

    # selective direct labels at the 5|4 boundary
    a, b = axes
    lab_a = f"{float(wm[0]):.3f} / {float(dm[0]):.3f}"
    check(lab_a == "0.108 / 0.119", "panel (a) boundary-1 label drifted")
    a.annotate(lab_a, (1, float(prior[0] + cap)), xytext=(0, 4),
               textcoords="offset points", ha="center", va="bottom",
               color=INK2, fontsize=6.5)
    a.text(2.35, 1.17, "both arms' means fall below the\nwindow at 4|3, 3|2, "
           "2|1 $\\rightarrow$ guardrail holds", ha="left", va="center",
           color=MUTED, fontsize=6.5, linespacing=1.4)

    lab_b = f"{m['ward_final'][0]} / {m['dp_final'][0]}"
    check(lab_b == "0.11 / 0.12", "panel (b) boundary-1 label drifted")
    b.annotate(lab_b + "\nfree: difference survives",
               (1, float(prior[0] + cap)), xytext=(0, 4),
               textcoords="offset points", ha="center", va="bottom",
               color=INK2, fontsize=6.5, linespacing=1.4)
    for i in (1, 2, 3):
        b.annotate("held", (xs[i], float(wf[i])), xytext=(0, -9),
                   textcoords="offset points", ha="center", va="top",
                   color=MUTED, fontsize=6.5)
    b.text(4.35, 0.28, "the 0.01 difference at 5|4 moves\n18 contracts up "
           "one measure star", ha="right", va="center", color=MUTED,
           fontsize=6.5, linespacing=1.4)

    a.set_ylabel("complaints per 1,000 members\n(lower is better)")
    fig.supxlabel("star boundary (cut point between adjacent star ratings)",
                  fontsize=8, color=INK, y=0.015)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor=BLUE, markeredgecolor=SURFACE,
               markeredgewidth=0.8,
               label="CMS method (Ward, documented tie rule)"),
        Line2D([], [], marker="D", linestyle="none", markersize=5,
               markerfacecolor=GREEN, markeredgecolor=SURFACE,
               markeredgewidth=0.8, label="exact optimum"),
        Line2D([], [], color=INK2, linewidth=1.2,
               label="2025 published cut point"),
        Patch(facecolor=GRID, edgecolor="none",
              label="guardrail window ($\\pm$0.04)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.00), columnspacing=1.8,
               handlelength=1.4, labelcolor=INK2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    check.report()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
