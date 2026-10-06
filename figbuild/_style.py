# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""Shared style tokens and a fail-loud assertion helper for the paper figures.

Palette: an ordinal blue ramp for star years 2024/2025/2026; blue and green
for the two computations (CMS's method and the exact optimum), with distinct
marker shapes and direct labels as a secondary encoding; neutral ink and
chrome tokens on a white surface. Marks are thin, the grid is a solid
hairline, overlapping markers carry a surface-colored ring, and text is set
in the ink tokens, never in a series color.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ink & chrome (reference palette, light mode, white paper surface)
INK = "#0b0b0b"       # primary
INK2 = "#52514e"      # secondary
MUTED = "#898781"     # axis/labels
GRID = "#e1e0d9"      # hairline gridline
AXIS = "#c3c2b7"      # baseline/axis
BAND = "#f0efec"      # neutral light band (diverging-midpoint gray)
SURFACE = "#ffffff"   # paper

# series
BLUE = "#2a78d6"      # categorical slot 1 - CMS's method / totals
GREEN = "#008300"     # categorical slot 2 - exact optimum
YEAR_STEP = {"2024": "#86b6ef", "2025": "#2a78d6", "2026": "#104281"}  # ordinal blue 250/450/650


def rc():
    """Publication rcParams: serif (journal-compatible), recessive chrome."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # solid hairline, never dashed
        "legend.frameon": False,
        "pdf.fonttype": 42,             # embed TrueType (journal-safe)
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


class Checker:
    """Fail-loud assertion: any mismatch exits nonzero immediately."""

    def __init__(self, name):
        self.name = name
        self.n = 0

    def __call__(self, cond, msg):
        if not cond:
            print(f"[{self.name}] ASSERT FAIL: {msg}", file=sys.stderr)
            sys.exit(1)
        self.n += 1

    def report(self):
        print(f"[{self.name}] {self.n} assertions passed")
