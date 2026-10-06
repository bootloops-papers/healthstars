# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.
"""binlib.py — two-ruler renderer and loaders for the score-line figures of
the star-ratings paper.

One format: one horizontal score line per clustering; every rated entity a
black dot of uniform size, jittered uniformly in y (y carries no meaning);
the five published star groups as a light-gray ruler above the dots and
the five groups at the exact optimum as a light-blue ruler below; each cut
point attached to its own ruler and reaching two thirds of the way into the
dot band from that ruler's side (published solid, optimal dashed); a thick
tick on each ruler at the group's mean score; and, where the census file
records the exact pair, the published and optimal sums of squares at the
right in the ruler colors.

Every printed count, interval, cut point, center and sum of squares is
checked against the census file before drawing; the loaders re-derive the
published intervals and the full migration tables from the entity-level CMS
score files and require exact equality with the census files (ICH sums of
squares are recomputed as exact rationals). Every dot is one entity's score
from a public CMS file.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from fractions import Fraction

import numpy as np

CEN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "stars_census"))

# Two-ruler palette: published = light gray band with a near-black dark;
# optimum = Okabe-Ito light blue band with the standard blue.
SHIP_LIGHT, SHIP_DARK = "#d9d9d9", "#3d3d3d"
OPT_LIGHT, OPT_DARK = "#bcd9ec", "#0072B2"
C_DOT = "#000000"
OPT_DASH = (0, (4, 2))
# strip geometry: published ruler above the dots, optimal ruler below,
# cut points reach 2/3 of the way into the dot band
POS = {"ship": (0.86, 1.0), "opt": (0.0, 0.14), "dots": (0.20, 0.80)}


def canon_sha16(path):
    """sha256 prefix of a JSON file's content with keys sorted, independent of byte layout."""
    import json as _json
    return hashlib.sha256(_json.dumps(_json.load(open(path)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- loaders
ICH_BANK = f"{CEN}/legc_ich/ICH_CENSUS.json"
ICH_BANK_SHA16 = "f555372c67118b0b"
ICH_CSV = f"{CEN}/legc_ich/data/ICH_CAHPS_FACILITY.csv"
# (census key, panel title, linearized-score column, star column); the
# column names as in the CMS facility file and in
# stars_census/legc_ich/census_ich.py.
ICH_MEASURES = [
    ("nephrologists' communication and caring",
     "Nephrologists' communication and caring",
     "Linearized score of nephrologists' communication and caring",
     "Star rating of nephrologists' communication and caring"),
    ("quality of dialysis center care and operations",
     "Quality of dialysis center care and operations",
     "Linearized score of quality of dialysis center care and operations",
     "Star rating of quality of dialysis center care and operations"),
    ("providing information to patients",
     "Providing information to patients",
     "Linearized score of providing information to patients",
     "Star rating of providing information to patients"),
    ("rating of the nephrologist", "Rating of the nephrologist",
     "Linearized score of rating of the nephrologist",
     "Star rating of the nephrologist"),
    ("rating of the dialysis center staff",
     "Rating of the dialysis center staff",
     "Linearized score of rating of the dialysis center staff",
     "Star rating of the dialysis center staff"),
    ("rating of the dialysis facility", "Rating of the dialysis facility",
     "Linearized score of rating of the dialysis facility",
     "Star rating of the dialysis facility"),
]
MISSING = ("", "Not Available", "Not Applicable")


def load_ich():
    """ICH CAHPS: six measures, facility-level linearized scores (integers).

    Returns {measure_key: dict(title, scores, deltas, pub_cuts, opt_cuts,
    up, dn)} with every count checked against the ICH census file.
    """
    assert canon_sha16(ICH_BANK) == ICH_BANK_SHA16, "ICH_CENSUS.json sha256 mismatch"
    bank = json.load(open(ICH_BANK))
    assert bank["data_sha256_16"] == sha16(ICH_CSV) == "cf67c06cb36cfe15"
    rows = list(csv.DictReader(open(ICH_CSV, newline="",
                                    encoding="utf-8-sig")))
    out = {}
    for key, title, lc, sc in ICH_MEASURES:
        b = bank["measures"][key]
        pts = []
        for r in rows:
            v, s = r[lc].strip(), r[sc].strip()
            if v in MISSING or s in MISSING:
                continue
            vf, sf = Fraction(v), Fraction(s)
            assert vf.denominator == 1 and sf.denominator == 1
            pts.append((int(vf), int(sf)))
        assert len(pts) == b["n"], (key, len(pts))
        # the ICH census file records the exact sum-of-squares pair per
        # measure; recompute both as exact rationals from the (score,
        # star) pairs and require equality before the floats render.
        ssq = (Fraction(b["published_ssq"]), Fraction(b["optimal_ssq"]))
        out[key] = _interval_instance(pts, b, title, ssq=ssq)
    return out, bank


HC_BANK = f"{CEN}/legb_hcahps/HCAHPS_CENSUS_ALLVINTAGES.json"
HC_BANK_SHA16 = "00b8f90ad2d86126"
HC_CSV = f"{CEN}/legb_hcahps/data/HCAHPS-Hospital.csv"
HC_VINTAGE = "current-2026-05-13"
HC_MEASURES = [
    ("H_COMP_1", "Nurse communication"),
    ("H_COMP_2", "Doctor communication"),
    ("H_COMP_5", "Communication about medicines"),
    ("H_COMP_6", "Discharge information"),
    ("H_CLEAN", "Cleanliness"),
    ("H_QUIET", "Quietness"),
    ("H_HSP_RATING", "Overall hospital rating"),
    ("H_RECMND", "Recommend hospital"),
]


def load_hcahps():
    """HCAHPS current refresh: eight starred measures, hospital-level
    integer linear mean scores; checked against the current block of the
    all-vintages census file."""
    assert canon_sha16(HC_BANK) == HC_BANK_SHA16, "HCAHPS census file sha256 mismatch"
    cur = json.load(open(HC_BANK))["vintages"][HC_VINTAGE]
    assert cur["file_sha256_16"] == sha16(HC_CSV) == "e9d7d4131a9704fc"
    lin = {m: {} for m, _ in HC_MEASURES}
    star = {m: {} for m, _ in HC_MEASURES}
    for r in csv.DictReader(open(HC_CSV, newline="", encoding="utf-8-sig")):
        mid, ccn = r["HCAHPS Measure ID"], r["Facility ID"]
        for m, _ in HC_MEASURES:
            if mid == f"{m}_LINEAR_SCORE" and r["HCAHPS Linear Mean Value"] \
                    not in MISSING:
                lin[m][ccn] = int(r["HCAHPS Linear Mean Value"])
            elif mid == f"{m}_STAR_RATING" \
                    and r["Patient Survey Star Rating"] not in MISSING:
                star[m][ccn] = int(r["Patient Survey Star Rating"])
    out = {}
    for m, title in HC_MEASURES:
        b = cur["measures"][m]
        common = sorted(set(lin[m]) & set(star[m]))
        pts = [(lin[m][c], star[m][c]) for c in common]
        assert len(pts) == b["n"], (m, len(pts))
        out[m] = _interval_instance(pts, b, title)
    return out, cur


def _interval_instance(pts, b, title, ssq=None):
    """Integer-interval surveys (ICH, HCAHPS): re-derive the published
    intervals from the (score, star) pairs, recount the migration table
    against the census file, and return what the renderer needs (per-group
    mean centers and counts for both groupings; the sum-of-squares pair
    where the census file carries it, recomputed exactly)."""
    iv = {}
    for v, s in sorted(pts):
        lo, hi = iv.get(s, (v, v))
        iv[s] = (min(lo, v), max(hi, v))
    pub = [[iv[s][0], iv[s][1]] for s in sorted(iv)]
    bank_pub = [[int(a), int(bb)] for a, bb in b["published_intervals"]]
    bank_opt = [[int(a), int(bb)] for a, bb in b["optimal_intervals"]]
    assert pub == bank_pub, (title, pub, bank_pub)

    def star_of(v, ivs):
        for s, (a, bb) in enumerate(ivs, 1):
            if a <= v <= bb:
                return s
        raise ValueError((v, ivs))

    mig = Counter()
    deltas = []
    for v, s in pts:
        a, bb = star_of(v, bank_pub), star_of(v, bank_opt)
        assert a == s  # published star == published-interval read
        deltas.append(bb - a)
        if a != bb:
            mig[f"{a}->{bb}"] += 1
    assert dict(mig) == b["migration"], (title, dict(mig), b["migration"])
    assert sum(mig.values()) == b["n_migrating"]
    up = sum(n for k, n in mig.items() if int(k.split("->")[1]) >
             int(k.split("->")[0]))
    dn = b["n_migrating"] - up
    # cut between star s and s+1 at the midpoint of the integer gap
    pub_cuts = [(bank_pub[s][1] + bank_pub[s + 1][0]) / 2.0
                for s in range(len(bank_pub) - 1)]
    opt_cuts = [(bank_opt[s][1] + bank_opt[s + 1][0]) / 2.0
                for s in range(len(bank_opt) - 1)]
    # every entity whose star changes lies between the two cut points of
    # its boundary; direction = sign(pub_cut - opt_cut)
    for (v, s), d in zip(pts, deltas):
        if d:
            k = min(s, s + d) - 1  # boundary index crossed (|d| == 1 here)
            assert abs(d) == 1, (title, v, s, d)
            lo, hi = sorted((pub_cuts[k], opt_cuts[k]))
            assert lo < v < hi
            assert (d > 0) == (pub_cuts[k] > opt_cuts[k])
    # per-group membership, mean-score centers, and exact sums of
    # squares for both groupings
    pub_bins = {s: [] for s in range(1, 6)}
    opt_bins = {s: [] for s in range(1, 6)}
    for (v, s), d in zip(pts, deltas):
        pub_bins[s].append(v)
        opt_bins[s + d].append(v)
    pub_counts = [len(pub_bins[s]) for s in range(1, 6)]
    opt_counts = [len(opt_bins[s]) for s in range(1, 6)]
    assert sum(pub_counts) == sum(opt_counts) == len(pts)
    pub_means = [float(Fraction(sum(pub_bins[s]), len(pub_bins[s])))
                 for s in range(1, 6)]
    opt_means = [float(Fraction(sum(opt_bins[s]), len(opt_bins[s])))
                 for s in range(1, 6)]
    ssq_pub = ssq_opt = None
    if ssq is not None:
        for bins, want in ((pub_bins, ssq[0]), (opt_bins, ssq[1])):
            got = Fraction(0)
            for s in range(1, 6):
                mu = Fraction(sum(bins[s]), len(bins[s]))
                got += sum((Fraction(v) - mu) ** 2 for v in bins[s])
            assert got == want, (title, float(got), float(want))
        assert ssq[1] < ssq[0], title  # optimal strictly below published
        ssq_pub, ssq_opt = float(ssq[0]), float(ssq[1])
    return {"title": title, "n": len(pts),
            "scores": [float(v) for v, _ in pts], "deltas": deltas,
            "pub_cuts": pub_cuts, "opt_cuts": opt_cuts,
            "pub_counts": pub_counts, "opt_counts": opt_counts,
            "pub_means": pub_means, "opt_means": opt_means,
            "ssq_pub": ssq_pub, "ssq_opt": ssq_opt, "ssq_decimals": 1,
            "up": up, "dn": dn, "optimal": b["published_is_optimal"]}


# ---------------------------------------------------------------- drawing
def fmt_n(n):
    return f"{n:,}"


def mover_note(up, dn):
    if up and dn:
        return f"{fmt_n(up)} up, {fmt_n(dn)} down"
    if up:
        return f"{fmt_n(up)} up"
    if dn:
        return f"{fmt_n(dn)} down"
    return "attains the true minimum"


def _ruler(ax, edges, means, y0, y1, light, dark, numeral_size,
           label=None, label_size=6.0, numeral_min_gap=0.0):
    """One grouping as a ruler stripe: five light intervals delimited by
    dark edges, a thick half-height tick at each group's mean score, star
    numerals in the wider sub-gap beside the tick (skipped where the gap
    is too narrow for a clean label)."""
    from matplotlib.patches import Rectangle
    for i in range(5):
        ax.add_patch(Rectangle((edges[i], y0), edges[i + 1] - edges[i],
                               y1 - y0, facecolor=light, edgecolor=dark,
                               linewidth=0.6, zorder=4))
    h = y1 - y0
    for i, c in enumerate(means):
        ax.plot([c, c], [y0 + 0.22 * h, y1 - 0.22 * h], color=dark,
                lw=2.0, zorder=5, solid_capstyle="butt")
        if numeral_size:
            gaps = [(c - edges[i], 0.5 * (edges[i] + c)),
                    (edges[i + 1] - c, 0.5 * (c + edges[i + 1]))]
            w, x = max(gaps)
            if w < numeral_min_gap:
                continue
            ax.text(x, 0.5 * (y0 + y1), str(i + 1), ha="center",
                    va="center", fontsize=numeral_size, color=dark,
                    zorder=6)
    if label is not None:
        ax.text(edges[0], y1 + 0.02, label, ha="left", va="bottom",
                fontsize=label_size, color=dark, zorder=6)


def draw_strip(ax, inst, rng, xlim, dot=2.0, numeral_size=6.0,
               ssq_size=6.2, ruler_labels=False, label_size=6.0):
    """One clustering in the two-ruler format: black jittered dots (y
    meaningless), published ruler above, optimal ruler below, cut points
    attached to their own ruler and reaching 2/3 of the way into the dot
    band (published solid from above, optimal dashed from below),
    mean-score ticks on the rulers, and the sum-of-squares pair at the
    right in the ruler colors where the census file records it
    (inst['ssq_pub'] is None otherwise)."""
    scores = np.asarray(inst["scores"], dtype=float)
    deltas = np.asarray(inst["deltas"], dtype=int)
    n = len(scores)
    (s0, s1), (o0, o1) = POS["ship"], POS["opt"]
    d0, d1 = POS["dots"]
    pad = 0.012 * (scores.max() - scores.min())
    outer = (scores.min() - pad, scores.max() + pad)
    pub_edges = [outer[0]] + list(inst["pub_cuts"]) + [outer[1]]
    opt_edges = [outer[0]] + list(inst["opt_cuts"]) + [outer[1]]
    # the drawn group populations equal the census counts for both
    # groupings, and the drawn number of star changes equals the
    # census's
    for edges, want in ((pub_edges, inst["pub_counts"]),
                        (opt_edges, inst["opt_counts"])):
        got = np.histogram(scores, bins=np.array(edges))[0].tolist()
        assert got == list(want), (inst["title"], got, want)
    assert int((deltas != 0).sum()) == inst["up"] + inst["dn"]
    # cut points, attached to their ruler, 2/3 into the dot band
    pen = (2.0 / 3.0) * (d1 - d0)
    for x in inst["pub_cuts"]:
        ax.plot([x, x], [d1 - pen, s0 + 0.02], color=SHIP_DARK, lw=0.9,
                ls="-", zorder=2, solid_capstyle="butt")
    for x in inst["opt_cuts"]:
        ax.plot([x, x], [o1 - 0.02, d0 + pen], color=OPT_DARK, lw=0.9,
                ls=OPT_DASH, zorder=2)
    # all dots black, uniform small size
    jit = rng.uniform(0.0, 1.0, size=n)
    ax.scatter(scores, d0 + jit * (d1 - d0), s=dot, color=C_DOT, lw=0,
               zorder=3)
    min_gap = 0.045 * (xlim[1] - xlim[0])
    _ruler(ax, pub_edges, inst["pub_means"], s0, s1, SHIP_LIGHT,
           SHIP_DARK, numeral_size,
           label="CMS's published star groups" if ruler_labels else None,
           label_size=label_size, numeral_min_gap=min_gap)
    _ruler(ax, opt_edges, inst["opt_means"], o0, o1, OPT_LIGHT,
           OPT_DARK, numeral_size,
           label="star groups at the true minimum" if ruler_labels else None,
           label_size=label_size, numeral_min_gap=min_gap)
    if inst["ssq_pub"] is not None:
        dec = inst["ssq_decimals"]
        a, b = f"{inst['ssq_pub']:.{dec}f}", f"{inst['ssq_opt']:.{dec}f}"
        assert inst["ssq_opt"] < inst["ssq_pub"] and a != b, inst["title"]
        tr = ax.get_yaxis_transform()
        ax.text(1.010, 0.5 * (s0 + s1), f"sum of\nsquares\n{a}", transform=tr,
                ha="left", va="center", fontsize=ssq_size,
                color=SHIP_DARK)
        ax.text(1.010, 0.5 * (o0 + o1), f"sum of\nsquares\n{b}", transform=tr,
                ha="left", va="center", fontsize=ssq_size,
                color=OPT_DARK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.03, 1.12 if ruler_labels else 1.03)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2)


def draw_hist(ax, pub_counts, opt_counts, label_size=7.0, tick_size=6.5,
              legend=False):
    """Per-star-group populations, published vs exact optimum, in the
    ruler colors."""
    x = np.arange(1, 6)
    ax.bar(x - 0.19, pub_counts, width=0.36, color=SHIP_LIGHT,
           edgecolor=SHIP_DARK, lw=0.6, label="published")
    ax.bar(x + 0.19, opt_counts, width=0.36, color=OPT_LIGHT,
           edgecolor=OPT_DARK, lw=0.6, label="exact optimum")
    ax.set_xticks(x)
    ax.set_xlabel("star level", fontsize=label_size)
    ax.set_ylim(0, max(max(pub_counts), max(opt_counts)) * 1.16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, labelsize=tick_size)
    if legend:
        ax.legend(frameon=False, fontsize=tick_size - 0.3,
                  loc="upper left", handlelength=1.1, handletextpad=0.5,
                  borderaxespad=0.1)


def title_of(inst, with_n=True):
    """Panel title: measure name, n, and the counts of entities whose star
    changes. with_n=False when the block header already states the
    uniform n."""
    if with_n:
        return f"{inst['title']} (n = {fmt_n(inst['n'])}; " \
               f"{mover_note(inst['up'], inst['dn'])})"
    return f"{inst['title']} ({mover_note(inst['up'], inst['dn'])})"


def legend_handles(ship_label="CMS's published star groups",
                   opt_label="star groups at the true minimum",
                   dot_label="one rated entity"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [
        Patch(facecolor=SHIP_LIGHT, edgecolor=SHIP_DARK, lw=0.7,
              label=ship_label),
        Patch(facecolor=OPT_LIGHT, edgecolor=OPT_DARK, lw=0.7,
              label=opt_label),
        Line2D([], [], marker="o", color=C_DOT, ls="none", ms=2.2,
               label=dot_label),
        Line2D([], [], marker="|", color="#222222", ls="none", ms=7,
               mew=1.4, label="group center (mean score)"),
        Line2D([], [], color=SHIP_DARK, lw=0.9,
               label="published cut point"),
        Line2D([], [], color=OPT_DARK, lw=0.9, ls=OPT_DASH,
               label="cut point at the true minimum"),
    ]
    return h
