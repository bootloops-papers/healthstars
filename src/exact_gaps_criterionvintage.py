# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Optimality-gap table on the full score list at the criterion vintages.

Same machinery as exact_gaps_docsonly, run at one operative vintage per
star year on both sides of the bonus-status census: {2024 recalc, 2025
current, 2026 current}. CMS's published 2025 cut points are identical
original->current (CMS re-scored 144 D01 contracts 98->100 at the Dec-2024
update but did not re-cluster), so the replay stays on the vintages CMS
clustered while this file carries the exact optima on the operative data.
Carried across vintages: the stage-1 membership layer (removals/adds),
determined at the replay vintages; membership is contract-level and the
synthetic add scores retain their values.
Output: runs/exact_gaps_criterionvintage.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.2 (the gap table at
    the data vintages the bonus-status census uses; convention (5) of Section
    2.5).
Run:  cd src && python3 exact_gaps_criterionvintage.py [workers]
Requires: Python >= 3.10; numpy, scipy.
"""
import os
import sys

import exact_gaps_docsonly as G

G.SCORE_VINTAGE = {"2024": "recalc", "2025": "current", "2026": "current"}

if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    G.main(workers=w, out_name="exact_gaps_criterionvintage.json")
