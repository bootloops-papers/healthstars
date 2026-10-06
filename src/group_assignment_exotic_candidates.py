# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Twenty-five further PROC SURVEYSELECT GROUPS=10 candidate readouts,
fingerprinted against the registered candidates and screened against the
published cut points.

Implements 25 additional candidate readouts of the group-assignment step,
fingerprints them against the 11 registered candidates
(groups_assign.CANDIDATES) and the 10-readout RANUNI grid arm, screens all 25 x
{cid_asc, score_asc} on the four sharpest fence-exact 2024 targets
(C01/C02/C04/C08, the order_grid_screen recipe), extends any candidate reaching
snap >= 3/4 on >= 2 targets to the full 77-target screen, and writes
runs/group_assignment_exotic_candidates.json.

Families (textual definitions also embedded in the output record):
  A. per-group MT substreams: group g in 1..10 (sizes remainder-last) draws
     from a FRESH SasMT stream seeded f(seed,g), f in {seed+g, (seed*g) mod
     2^31, sha256-hash}; selection per group from the remaining units either by
     Floyd ordered-hash SRS (groups_assign.floyd_sample) or by sort-SRS (one
     uniform per remaining unit, take the sz smallest). 6 candidates.
  B. serpentine-after-sort: one uniform per obs from a single stream (SasMT or
     SasRanuni, seed 8675309); sort obs by uniform ascending (stable); deal
     labels boustrophedon (1..G,G..1,... or G..1,1..G,...). 4 candidates.
  C. PROC PLAN-style per-block label permutation: blocks of G=10 consecutive
     observations in dataset order; each block gets a fresh permutation of
     labels 1..G (rank variant: label of block position i = rank of its uniform
     among G fresh uniforms; FY variant: labels a[0..]+1 with a =
     groups_assign.fisher_yates(rng, G, "down")); last partial block takes the
     first r labels of a full permutation; stream SasMT or SasRanuni. 4 cands.
  D. RAND('integer',1,j) conversion variants on the raw MT u32 stream:
     direct-mod (r = u32 mod j + 1) and rejection (redraw while u32 >=
     floor(2^32/j)*j, then mod) replacing floor(u*j)+1 inside Floyd
     (floyd_last structure) and Fisher-Yates up/down + contiguous blocks
     (fy_up_blocks / fy_down_blocks structure). 6 candidates.
  E. RANUNI-sort/MT-draw hybrids incl. legacy-seed reseed:
     hyb_sortRANUNI_quotaMT — sort obs by SasRanuni(seed) uniforms, then assign
       by sequential quota (remainder-last) drawing SasMT(seed) uniforms along
       the sorted order;
     hyb_sortMT_quotaRANUNI — the mirror image;
     hyb_reseedMT_sortblocks / hyb_reseedMT_floyd — legacy-seed reseed: seed' =
       first RANUNI state 397204094*seed mod (2^31-1) (= 959153317 for
       8675309), then registered sort_blocks_last / floyd_last under
       SasMT(seed');
     hyb_reseedRANUNI_sortblocks — seed'' = first SasMT(seed) u32 mod (2^31-1)
       (adjusted +1 if 0), then sort-blocks-last under SasRanuni(seed'').
     5 candidates.
Any substream/reseed seed that violates SasMT's constraints (must be in
(0, 2^31), not divisible by 8192) is incremented by 1 until valid; RANUNI
seeds must be in (0, 2^31-1). No adjustment fires for seed 8675309 (checked
and recorded in the output).

Run: cd src && python3 group_assignment_exotic_candidates.py [workers]
Writes ../runs/group_assignment_exotic_candidates.json.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Appendix C.3
    (the 22 hand-built constructions of the group-assignment step).
Requires: Python >= 3.10; numpy, scipy.
"""

import hashlib
import json
import multiprocessing as mp
import os
from fractions import Fraction

import groups_assign
import sys
from groups_assign import (CANDIDATES as REGISTERED, fisher_yates, floyd_sample,
                           group_sizes)
from falsify_groups import round_display
from order_grid_screen import ALGS, gen_gids, order_pairs
from pipeline import measure_cutpoints
from mt19937_stream import SasMT, SasRanuni

SEED = 8675309
G = 10
FP_N = 503
SHARP_MIDS = ("C01", "C02", "C04", "C08")
ORDERS = ("cid_asc", "score_asc")
EXTEND_SNAP = 3          # boundary-snap bar per target
EXTEND_TARGETS = 2       # number of sharp targets at or above the bar

_REJ_COUNT = [0]         # rejection events, single-process fingerprinting only


# ---------- family A: per-group MT substreams ----------

def _valid_mt_seed(s):
    s %= 2**31
    adjusted = 0
    while s <= 0 or s % 8192 == 0:
        s += 1
        adjusted += 1
    return s, adjusted


def _sub_seed(seed, gid, rule):
    if rule == "add":
        raw = seed + gid
    elif rule == "mul":
        raw = seed * gid
    else:  # "hash": low 31 bits of the first 4 bytes of sha256("{seed},{gid}")
        raw = int.from_bytes(
            hashlib.sha256(f"{seed},{gid}".encode()).digest()[:4], "big") & 0x7FFFFFFF
    s, _ = _valid_mt_seed(raw)
    return s


def sub_floyd(n, g, seed, rule):
    sizes = group_sizes(n, g, "last")
    remaining = list(range(n))
    out = [0] * n
    for gid, sz in enumerate(sizes, 1):
        rng = SasMT(_sub_seed(seed, gid, rule))
        pick = floyd_sample(rng, len(remaining), sz)
        for i in [remaining[i - 1] for i in sorted(pick)]:
            out[i] = gid
        remaining = [i for i in remaining if out[i] == 0]
    return out


def sub_sortsrs(n, g, seed, rule):
    sizes = group_sizes(n, g, "last")
    remaining = list(range(n))
    out = [0] * n
    for gid, sz in enumerate(sizes, 1):
        rng = SasMT(_sub_seed(seed, gid, rule))
        us = [rng.uniform() for _ in remaining]
        order = sorted(range(len(remaining)), key=us.__getitem__)
        for i in sorted(remaining[k] for k in order[:sz]):
            out[i] = gid
        remaining = [i for i in remaining if out[i] == 0]
    return out


# ---------- family B: serpentine-after-sort ----------

def serpentine(n, g, seed, stream, start):
    rng = (SasMT if stream == "MT" else SasRanuni)(seed)
    us = [rng.uniform() for _ in range(n)]
    order = sorted(range(n), key=us.__getitem__)
    out = [0] * n
    for r, i in enumerate(order):
        c, p = divmod(r, g)
        forward = (c % 2 == 0) == (start == "asc")
        out[i] = p + 1 if forward else g - p
    return out


# ---------- family C: PROC PLAN-style per-block label permutation ----------

def plan_blocks(n, g, seed, stream, variant):
    rng = (SasMT if stream == "MT" else SasRanuni)(seed)
    out = [0] * n
    pos = 0
    while pos < n:
        blk = min(g, n - pos)
        if variant == "rank":
            us = [rng.uniform() for _ in range(g)]
            order = sorted(range(g), key=us.__getitem__)
            perm = [0] * g
            for rank0, idx in enumerate(order):
                perm[idx] = rank0 + 1
        else:  # "fy": FY-down shuffle of [0..g-1], labels a[i]+1
            a = fisher_yates(rng, g, "down")
            perm = [a[i] + 1 for i in range(g)]
        for i in range(blk):
            out[pos + i] = perm[i]
        pos += blk
    return out


# ---------- family D: RAND('integer',1,j) u32 conversion variants ----------

def _ri_mod(rng, j):
    return (rng.next_u32() % j) + 1


def _ri_rej(rng, j):
    lim = (2**32 // j) * j
    while True:
        y = rng.next_u32()
        if y < lim:
            return (y % j) + 1
        _REJ_COUNT[0] += 1


def floyd_ri(n, g, seed, ri):
    rng = SasMT(seed)
    sizes = group_sizes(n, g, "last")
    remaining = list(range(n))
    out = [0] * n
    for gid, sz in enumerate(sizes, 1):
        pool = len(remaining)
        T = set()
        for j in range(pool - sz + 1, pool + 1):
            r = ri(rng, j)
            T.add(j if r in T else r)
        for i in [remaining[i - 1] for i in sorted(T)]:
            out[i] = gid
        remaining = [i for i in remaining if out[i] == 0]
    return out


def fy_ri(n, g, seed, ri, direction):
    rng = SasMT(seed)
    a = list(range(n))
    if direction == "up":
        for i in range(2, n + 1):
            j = ri(rng, i)
            a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
    else:
        for i in range(n, 1, -1):
            j = ri(rng, i)
            a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
    out = [0] * n
    sizes = group_sizes(n, g, "last")
    pos = 0
    for gid, sz in enumerate(sizes, 1):
        for p in range(pos, pos + sz):
            out[a[p]] = gid
        pos += sz
    return out


# ---------- family E: hybrids ----------

def hyb_sort_quota(n, g, seed, sort_stream):
    rng_sort = (SasRanuni if sort_stream == "RANUNI" else SasMT)(seed)
    rng_draw = (SasMT if sort_stream == "RANUNI" else SasRanuni)(seed)
    us = [rng_sort.uniform() for _ in range(n)]
    order = sorted(range(n), key=us.__getitem__)
    remaining = group_sizes(n, g, "last")[:]
    out = [0] * n
    for i in order:
        u = rng_draw.uniform()
        total = sum(remaining)
        x = u * total
        acc = 0
        gid = None
        for j in range(g):
            nxt = acc + remaining[j]
            if x < nxt or (j == g - 1 and gid is None):
                gid = j
                break
            acc = nxt
        remaining[gid] -= 1
        out[i] = gid + 1
    return out


def _ranuni_first_state(seed):
    return (397204094 * seed) % (2**31 - 1)


def _sort_blocks_rng(rng, n, g):
    us = [rng.uniform() for _ in range(n)]
    order = sorted(range(n), key=us.__getitem__)
    out = [0] * n
    pos = 0
    for gid, sz in enumerate(group_sizes(n, g, "last"), 1):
        for i in order[pos:pos + sz]:
            out[i] = gid
        pos += sz
    return out


def hyb_reseed_mt(n, g, seed, mode):
    s, _ = _valid_mt_seed(_ranuni_first_state(seed))
    if mode == "sortblocks":
        return _sort_blocks_rng(SasMT(s), n, g)
    return groups_assign.floyd_groups(n, g, s, "last")


def hyb_reseed_ranuni(n, g, seed):
    s = SasMT(seed).next_u32() % (2**31 - 1)
    if s == 0:
        s = 1
    return _sort_blocks_rng(SasRanuni(s), n, g)


EXOTIC = {
    # A
    "sub_add_floyd":    lambda n, g, s: sub_floyd(n, g, s, "add"),
    "sub_add_sortsrs":  lambda n, g, s: sub_sortsrs(n, g, s, "add"),
    "sub_mul_floyd":    lambda n, g, s: sub_floyd(n, g, s, "mul"),
    "sub_mul_sortsrs":  lambda n, g, s: sub_sortsrs(n, g, s, "mul"),
    "sub_hash_floyd":   lambda n, g, s: sub_floyd(n, g, s, "hash"),
    "sub_hash_sortsrs": lambda n, g, s: sub_sortsrs(n, g, s, "hash"),
    # B
    "serp_MT_asc":      lambda n, g, s: serpentine(n, g, s, "MT", "asc"),
    "serp_MT_desc":     lambda n, g, s: serpentine(n, g, s, "MT", "desc"),
    "serp_RANUNI_asc":  lambda n, g, s: serpentine(n, g, s, "RANUNI", "asc"),
    "serp_RANUNI_desc": lambda n, g, s: serpentine(n, g, s, "RANUNI", "desc"),
    # C
    "plan_rank_MT":     lambda n, g, s: plan_blocks(n, g, s, "MT", "rank"),
    "plan_rank_RANUNI": lambda n, g, s: plan_blocks(n, g, s, "RANUNI", "rank"),
    "plan_fy_MT":       lambda n, g, s: plan_blocks(n, g, s, "MT", "fy"),
    "plan_fy_RANUNI":   lambda n, g, s: plan_blocks(n, g, s, "RANUNI", "fy"),
    # D
    "floyd_u32mod":     lambda n, g, s: floyd_ri(n, g, s, _ri_mod),
    "floyd_u32rej":     lambda n, g, s: floyd_ri(n, g, s, _ri_rej),
    "fyup_u32mod":      lambda n, g, s: fy_ri(n, g, s, _ri_mod, "up"),
    "fyup_u32rej":      lambda n, g, s: fy_ri(n, g, s, _ri_rej, "up"),
    "fydown_u32mod":    lambda n, g, s: fy_ri(n, g, s, _ri_mod, "down"),
    "fydown_u32rej":    lambda n, g, s: fy_ri(n, g, s, _ri_rej, "down"),
    # E
    "hyb_sortRANUNI_quotaMT":    lambda n, g, s: hyb_sort_quota(n, g, s, "RANUNI"),
    "hyb_sortMT_quotaRANUNI":    lambda n, g, s: hyb_sort_quota(n, g, s, "MT"),
    "hyb_reseedMT_sortblocks":   lambda n, g, s: hyb_reseed_mt(n, g, s, "sortblocks"),
    "hyb_reseedMT_floyd":        lambda n, g, s: hyb_reseed_mt(n, g, s, "floyd"),
    "hyb_reseedRANUNI_sortblocks": lambda n, g, s: hyb_reseed_ranuni(n, g, s),
}

DEFINITIONS = {
    "sub_add_floyd": "per-group MT substream seeded seed+g (g=1..10); group sizes remainder-last; per group, Floyd ordered-hash SRS (groups_assign.floyd_sample, randint=floor(u*j)+1) from the remaining units in original order",
    "sub_add_sortsrs": "per-group MT substream seeded seed+g; per group, one uniform per remaining unit (current remaining order), the sz smallest join the group",
    "sub_mul_floyd": "as sub_add_floyd with substream seed (seed*g) mod 2^31",
    "sub_mul_sortsrs": "as sub_add_sortsrs with substream seed (seed*g) mod 2^31",
    "sub_hash_floyd": "as sub_add_floyd with substream seed = low 31 bits of first 4 bytes of sha256('{seed},{g}')",
    "sub_hash_sortsrs": "as sub_add_sortsrs with the sha256 substream seed",
    "serp_MT_asc": "one SasMT(seed) uniform per obs; sort obs by uniform ascending (stable); deal labels boustrophedon starting 1..10 then 10..1 alternating",
    "serp_MT_desc": "as serp_MT_asc starting 10..1 then 1..10",
    "serp_RANUNI_asc": "as serp_MT_asc with the SasRanuni(seed) stream",
    "serp_RANUNI_desc": "as serp_MT_desc with the SasRanuni(seed) stream",
    "plan_rank_MT": "blocks of 10 consecutive obs in dataset order; per block draw 10 SasMT uniforms, label of block position i = rank of its uniform (1-based); last partial block takes the first r labels",
    "plan_rank_RANUNI": "as plan_rank_MT with the SasRanuni(seed) stream",
    "plan_fy_MT": "blocks of 10 consecutive obs; per block labels a[i]+1 with a = fisher_yates(rng,10,'down') (randint=floor(u*j)+1) on the SasMT(seed) stream; partial block takes first r labels",
    "plan_fy_RANUNI": "as plan_fy_MT with the SasRanuni(seed) stream",
    "floyd_u32mod": "floyd_last structure (single SasMT(seed) stream, sizes remainder-last) with randint(1..j) = (raw u32 mod j)+1",
    "floyd_u32rej": "floyd_last structure with rejection randint: redraw while u32 >= floor(2^32/j)*j, then (u32 mod j)+1",
    "fyup_u32mod": "Fisher-Yates up (i=2..n, swap(i, randint(1..i))) + contiguous blocks remainder-last, randint = (u32 mod j)+1",
    "fyup_u32rej": "as fyup_u32mod with the rejection randint",
    "fydown_u32mod": "Fisher-Yates down (i=n..2) + contiguous blocks remainder-last, randint = (u32 mod j)+1",
    "fydown_u32rej": "as fydown_u32mod with the rejection randint",
    "hyb_sortRANUNI_quotaMT": "sort obs by SasRanuni(seed) uniforms (asc, stable); walk sorted order assigning by sequential quota (remainder-last) with SasMT(seed) uniforms",
    "hyb_sortMT_quotaRANUNI": "sort obs by SasMT(seed) uniforms; sequential quota draws from SasRanuni(seed)",
    "hyb_reseedMT_sortblocks": "legacy-seed reseed: seed' = 397204094*seed mod (2^31-1) (first RANUNI state; 959153317 for 8675309); sort-blocks remainder-last under SasMT(seed')",
    "hyb_reseedMT_floyd": "same reseed seed'; registered floyd_last (groups_assign.floyd_groups) under SasMT(seed')",
    "hyb_reseedRANUNI_sortblocks": "seed'' = first SasMT(seed) u32 mod (2^31-1), +1 if 0; sort-blocks remainder-last under SasRanuni(seed'')",
}

assert len(EXOTIC) == 25 and set(DEFINITIONS) == set(EXOTIC)


def run_exotic(args):
    w, cname, order = args
    pairs = order_pairs(w["kept"], order)
    scores = [s for _, s in pairs]
    key = f"__exotic_{cname}"
    groups_assign.CANDIDATES[key] = EXOTIC[cname]
    try:
        r = measure_cutpoints(scores, higher_is_better=w["higher"], tukey=True,
                              cap_lo="0", cap_hi=w["cap_hi"],
                              groups_candidate=key, arm="ward_tie")
    finally:
        del groups_assign.CANDIDATES[key]
    row = {"year": w["year"], "mid": w["mid"], "org": w["org"],
           "candidate": cname, "order": order, "target": w["target"]}
    if "error" in r or len(r.get("mean_cutpoints", [])) != 4:
        row["error"] = True
        return row
    means = [float(c) for c in r["mean_cutpoints"]]
    dd = max(len(v.split(".")[1]) if "." in v else 0 for v in w["target"])
    row["means"] = means
    row["snap"] = sum(1 for m, t in zip(means, w["target"])
                      if abs(m - float(t)) < 0.5)
    row["n_exact"] = sum(
        1 for m, t in zip(means, w["target"])
        if Fraction(round_display(Fraction(str(m)).limit_denominator(10**9), dd))
        == Fraction(t))
    return row


def fingerprints():
    fps, gids_at_fp = {}, {}
    _REJ_COUNT[0] = 0
    for name, fn in EXOTIC.items():
        gids = fn(FP_N, G, SEED)
        counts = [gids.count(i) for i in range(1, G + 1)]
        assert sum(counts) == FP_N and max(counts) - min(counts) <= 1, (name, counts)
        gids_at_fp["exotic:" + name] = gids
    rej_events_at_fp = _REJ_COUNT[0]
    for name, fn in REGISTERED.items():
        gids_at_fp["registered:" + name] = fn(FP_N, G, SEED)
    for alg in ALGS:
        gids_at_fp["grid_ranuni:" + alg] = gen_gids(alg, SasRanuni, FP_N, SEED, G)
    for k, v in gids_at_fp.items():
        fps[k] = hashlib.sha256(",".join(map(str, v)).encode()).hexdigest()
    groups = {}
    for k, h in fps.items():
        groups.setdefault(h, []).append(k)
    collisions = {h: sorted(ks) for h, ks in groups.items() if len(ks) > 1}
    return fps, collisions, rej_events_at_fp


def main(workers=None):
    workers = workers or os.cpu_count() or 1
    from stage2_battery import build_worklist

    fps, collisions, rej_fp = fingerprints()
    # distinct additional-candidate count: one representative per fingerprint, and the
    # fingerprint must not be shared with any registered/grid generator
    exotic_by_fp = {}
    dup_of = {}
    for name in EXOTIC:
        h = fps["exotic:" + name]
        exotic_by_fp.setdefault(h, []).append(name)
    shared_with_registered = []
    for h, names in exotic_by_fp.items():
        others = [k for k, v in fps.items()
                  if v == h and not k.startswith("exotic:")]
        if others:
            shared_with_registered.extend(names)
        for extra in names[1:]:
            dup_of[extra] = names[0]
    n_distinct = sum(1 for h, names in exotic_by_fp.items()
                     if not any(fps[k] == h and not k.startswith("exotic:")
                                for k in fps))
    print(f"fingerprints: 25 additional candidates -> {len(exotic_by_fp)} distinct vectors; "
          f"{len(shared_with_registered)} shared with registered/grid; "
          f"rejection events at n={FP_N}: {rej_fp}", flush=True)

    work24 = [w for w in build_worklist(("2024",)) if w["mid"] in SHARP_MIDS]
    assert len(work24) == 4, [(w["mid"], w["org"]) for w in work24]
    jobs = [(w, name, order) for name in EXOTIC for order in ORDERS
            for w in work24]
    print(f"sharp screen: {len(jobs)} jobs on "
          f"{[(w['mid'], w['org'], len(w['kept'])) for w in work24]}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        screen_rows = pool.map(run_exotic, jobs)
    n_err = sum(1 for r in screen_rows if r.get("error"))

    # extension decision: candidate reaches snap >= 3 on >= 2 sharp targets
    # under at least one order
    from collections import Counter, defaultdict
    hits = defaultdict(set)
    for r in screen_rows:
        if not r.get("error") and r["snap"] >= EXTEND_SNAP:
            hits[(r["candidate"], r["order"])].add(r["mid"])
    extend = sorted({c for (c, o), mids in hits.items()
                     if len(mids) >= EXTEND_TARGETS})
    print(f"screen errors: {n_err}; extension candidates: {extend}", flush=True)

    ext_rows, ext_summary = [], {}
    if extend:
        work_full = build_worklist(("2024", "2025", "2026"))
        jobs2 = [(w, name, order) for name in extend for order in ORDERS
                 for w in work_full]
        print(f"full-screen extension: {len(jobs2)} jobs over "
              f"{len(work_full)} targets", flush=True)
        with ctx.Pool(workers) as pool:
            ext_rows = pool.map(run_exotic, jobs2)
        tally, denom, perfect = Counter(), Counter(), Counter()
        for r in ext_rows:
            k = f"{r['candidate']}|{r['order']}"
            if r.get("error"):
                continue
            tally[k] += r["n_exact"]
            denom[k] += 4
            if r["n_exact"] == 4:
                perfect[k] += 1
        ext_summary = {k: {"boundaries": f"{tally[k]}/{denom[k]}",
                           "targets_4of4": perfect[k]}
                       for k in sorted(tally, key=lambda kk: -tally[kk])}

    # verdicts
    best = defaultdict(lambda: {"snap": -1})
    per_cand_hi = defaultdict(int)
    for r in screen_rows:
        if r.get("error"):
            continue
        c = r["candidate"]
        if r["snap"] > best[c]["snap"]:
            best[c] = {"snap": r["snap"], "mid": r["mid"], "order": r["order"]}
        per_cand_hi[c] = max(per_cand_hi[c],
                             len(hits.get((c, "cid_asc"), set())),
                             len(hits.get((c, "score_asc"), set())))
    verdicts = {}
    for name in EXOTIC:
        v = {"distinct": name not in dup_of and name not in shared_with_registered}
        if name in dup_of:
            v["duplicate_of"] = dup_of[name]
        if name in shared_with_registered:
            v["shares_fingerprint_with_registered"] = True
        if name in extend:
            keys = [f"{name}|{o}" for o in ORDERS if f"{name}|{o}" in ext_summary]
            v["verdict"] = ("falsified on the full 77-target screen: " +
                            "; ".join(f"{k}: {ext_summary[k]['boundaries']} boundaries, "
                                      f"{ext_summary[k]['targets_4of4']} targets 4/4"
                                      for k in keys) +
                            " (a surviving candidate requires across-targets agreement)")
        else:
            b = best[name]
            v["verdict"] = (f"falsified on the sharp 4-target screen: best snap "
                            f"{b['snap']}/4 ({b.get('mid')}, {b.get('order')}); "
                            f"targets at snap>=3: {per_cand_hi[name]} of 4 "
                            f"(< {EXTEND_TARGETS} required to extend)")
        verdicts[name] = v

    record = {
        "register": ("25 additional GROUPS= candidate readouts, fingerprinted against "
                     "the 11 registered candidates and the 10-readout RANUNI grid arm "
                     "on (n=503,g=10,seed=8675309), screened x {cid_asc,score_asc} on "
                     "the 4 sharpest fence-exact 2024 targets (C01/C02/C04/C08, "
                     "ward_tie arm, reconciled membership); "
                     f"extension bar: snap>={EXTEND_SNAP}/4 on >={EXTEND_TARGETS} targets; "
                     "snap = |group-mean - published| < 0.5 per boundary; n_exact = "
                     "round_display equality at published display precision"),
        "seed": SEED,
        "definitions": DEFINITIONS,
        "fingerprints_sha256": fps,
        "fingerprint_collision_groups": collisions,
        "rejection_events_at_fingerprint_n": rej_fp,
        "n_exotic": 25,
        "n_exotic_distinct": n_distinct,
        "screen_targets": [{"year": w["year"], "mid": w["mid"], "org": w["org"],
                            "n_input": len(w["kept"]), "target": w["target"]}
                           for w in work24],
        "screen_rows": screen_rows,
        "n_screen_errors": n_err,
        "extension_candidates": extend,
        "extension_summary": ext_summary,
        "extension_rows": ext_rows,
        "verdicts": verdicts,
        "total_falsified": {
            "grid": 80,
            "exotic_distinct": n_distinct,
            "total": 80 + n_distinct,
            "note": ("grid record runs/stage2_full_grid.json (80 combinations x 77 "
                     "targets, exact); additional candidates falsified on the sharp "
                     "4-target screen (or the full 77 where extended); duplicates by "
                     "fingerprint are not counted"),
        },
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "group_assignment_exotic_candidates.json")
    json.dump(record, open(out, "w"), indent=1)
    print(f"saved {out}; n_exotic_distinct={n_distinct}; "
          f"total falsified = 80 + {n_distinct} = {80 + n_distinct}", flush=True)


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
