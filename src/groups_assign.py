# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Candidate algorithms for PROC SURVEYSELECT GROUPS=G SEED=s random group assignment.

The SURVEYSELECT chapter of the SAS/STAT 14.2 User's Guide (GROUPS= option,
pp. 9521-22) states WHAT GROUPS=n does: it assigns the observations at random
to n groups of equal size, or as nearly equal as possible (its worked example
splits 100 observations under GROUPS=3 into groups of 33, 33 and 34). It does
not state the assignment rule. Candidates below are tested end-to-end against CMS's published
cut points (improvement measures publish 6-decimal cut points and are guardrail-
exempt, so a candidate that fails to reproduce them is excluded); the readout
that the record's resampled replay ASSUMES is frontswap() below.

Group sizes: doc example (100,3) -> (33,33,34): base=floor(N/G), remainder r goes
to the LAST r groups. The first-r-groups variant is also provided for falsification.

Candidates (each returns groupid list, 1-based, parallel to input order):
  seq_quota    : one uniform per obs in order; obs assigned to group g with
                 probability (remaining quota of g)/(total remaining), by
                 partitioning [0,1) into group-order subintervals (the natural
                 multigroup generalization of Fan-Muller-Rezucha sequential SRS,
                 which SAS uses as METHOD=SRS2 and as SRS fallback).
  sort_blocks  : one uniform per obs; sort obs by uniform (ascending, stable);
                 first n_1 sorted obs -> group 1, next n_2 -> group 2, ...
  sort_deal    : one uniform per obs; sort obs by uniform; deal groups cyclically
                 1,2,...,G,1,2,... through the sorted order.
Each with sizes_last (doc-implied) or sizes_first remainder placement where
applicable (seq_quota, sort_blocks; sort_deal fixes sizes implicitly).

The uniform-consumption order is dataset order (one draw per observation) in all
candidates.

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Appendix C.3 (the group-
    assignment step; the assumed front-swap readout is the function frontswap).
Run:  cd src && python3 groups_assign.py   (self-tests; imported by the resampling scripts)
Requires: Python >= 3.10, standard library only.
"""

from mt19937_stream import SasMT, SasRanuni
import sys


def group_sizes(n, g, remainder="last"):
    base, r = divmod(n, g)
    if remainder == "last":
        return [base] * (g - r) + [base + 1] * r
    if remainder == "first":
        return [base + 1] * r + [base] * (g - r)
    raise ValueError(remainder)


def seq_quota(n, g, seed, remainder="last", conversion="y/2^32", boundary_eps=1e-9):
    """Sequential quota allocation. Returns (groupids, n_boundary_events).

    boundary events: a uniform within boundary_eps of a quota subinterval edge —
    where the unresolved u32->double conversion could flip the assignment. These
    are counted so the caller can report conversion-sensitivity.
    """
    rng = SasMT(seed, conversion)
    remaining = group_sizes(n, g, remainder)[:]
    out = []
    boundary = 0
    for _ in range(n):
        u = rng.uniform()
        total = sum(remaining)
        x = u * total
        acc = 0
        gid = None
        for j in range(g):
            nxt = acc + remaining[j]
            if x < nxt or j == g - 1 and gid is None:
                if gid is None:
                    gid = j
                break
            acc = nxt
        # boundary sensitivity check
        acc = 0
        for j in range(g):
            nxt = acc + remaining[j]
            if abs(x - nxt) < boundary_eps * total:
                boundary += 1
                break
            acc = nxt
        remaining[gid] -= 1
        out.append(gid + 1)
    assert all(r == 0 for r in remaining)
    return out, boundary


def _ranks(n, seed, conversion="y/2^32"):
    rng = SasMT(seed, conversion)
    us = [rng.uniform() for _ in range(n)]
    order = sorted(range(n), key=us.__getitem__)  # stable; double ties ~impossible
    return order, us


def sort_blocks(n, g, seed, remainder="last", conversion="y/2^32"):
    order, _ = _ranks(n, seed, conversion)
    sizes = group_sizes(n, g, remainder)
    out = [0] * n
    pos = 0
    for gid, sz in enumerate(sizes, 1):
        for i in order[pos:pos + sz]:
            out[i] = gid
        pos += sz
    return out


def sort_deal(n, g, seed, conversion="y/2^32"):
    order, _ = _ranks(n, seed, conversion)
    out = [0] * n
    for rank, i in enumerate(order):
        out[i] = (rank % g) + 1
    return out


def _rand_int(rng, j, conversion="floor"):
    """uniform -> integer in [1..j]. floor(u*j)+1 (ties at exact integers are
    measure-zero for MT doubles)."""
    u = rng.uniform()
    r = int(u * j) + 1
    return min(r, j)


def floyd_sample(rng, n_pool, k):
    """Floyd's ordered-hash sampling of k from [1..n_pool] (Bentley-Floyd F2),
    one uniform per selected unit. Returns SET of 1-based indices."""
    T = set()
    for j in range(n_pool - k + 1, n_pool + 1):
        r = _rand_int(rng, j)
        if r in T:
            T.add(j)
        else:
            T.add(r)
    return T


def floyd_groups(n, g, seed, remainder="last"):
    """Sequential per-group Floyd SRS from the REMAINING units (in original
    order); group sizes per group_sizes(). One plausible GROUPS= realization."""
    from mt19937_stream import SasMT
    rng = SasMT(seed)
    sizes = group_sizes(n, g, remainder)
    remaining = list(range(n))  # original 0-based indices, original order
    out = [0] * n
    for gid, sz in enumerate(sizes, 1):
        pick = floyd_sample(rng, len(remaining), sz)
        chosen = [remaining[i - 1] for i in sorted(pick)]
        for i in chosen:
            out[i] = gid
        remaining = [i for i in remaining if out[i] == 0]
    return out


def fisher_yates(rng, n, direction="up"):
    """SAS-style FY shuffle of [0..n-1]. 'up': for i=1..n swap(i, randint(1..i));
    'down': for i=n..2 swap(i, randint(1..i))."""
    a = list(range(n))
    if direction == "up":
        for i in range(2, n + 1):
            j = _rand_int(rng, i)
            a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
    else:
        for i in range(n, 1, -1):
            j = _rand_int(rng, i)
            a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
    return a


def fy_groups(n, g, seed, direction="down", mode="blocks", remainder="last"):
    """Fisher-Yates permutation then contiguous blocks or cyclic deal."""
    from mt19937_stream import SasMT
    rng = SasMT(seed)
    perm = fisher_yates(rng, n, direction)  # perm[pos] = original index
    out = [0] * n
    if mode == "blocks":
        sizes = group_sizes(n, g, remainder)
        pos = 0
        for gid, sz in enumerate(sizes, 1):
            for p in range(pos, pos + sz):
                out[perm[p]] = gid
            pos += sz
    else:
        for p, idx in enumerate(perm):
            out[idx] = (p % g) + 1
    return out


def frontswap(n, g, seed, remainder="last"):
    """THE ASSUMED READOUT.  The Technical Notes print the SURVEYSELECT call,
    the seed and the group count but not the rule that turns the uniform
    stream into group labels; this front-swap Fisher-Yates readout over the
    slot array is the ASSUMPTION under which every resampled-replay result in
    this record is stated (results that depend on it are conditional on it):

        S = [1]*n_1 + [2]*n_2 + ... + [G]*n_G   (sizes remainder-to-LAST, doc)
        for w = 0..n-1:
            u    = nextafter(y_w / 2^32, 0)     # assumed conversion (paper, App. C)
            j    = w + int(u * (n - w))
            out += [S[j]];  S[j] = S[w]

    One uniform per observation, dataset order; u -> integer by floor; the
    uint32 -> double conversion (mt19937_stream.py) is part of the stated
    assumption."""
    import math
    rng = SasMT(seed)
    S = []
    for gid, sz in enumerate(group_sizes(n, g, remainder), 1):
        S.extend([gid] * sz)
    out = []
    for w in range(n):
        u = math.nextafter(rng.next_u32() * 2.0**-32, 0.0)
        j = w + int(u * (n - w))
        if j >= n:
            j = n - 1
        out.append(S[j])
        S[j] = S[w]
    return out


def frontswap_ranuni(n, g, seed, remainder="last"):
    """The ASSUMED consumption pattern (front-swap readout) fed by the
    LEGACY RANUNI stream instead of MT19937 — the OTHER branch of the
    2024/2025 generator-family two-way (SURVEYSELECT default changed to MT at
    SAS/STAT 12.1 per its documentation; the RANUNI option uses the
    Fishman-Moore MCG).  The readout is the same assumption as frontswap();
    generator-agnosticism of the readout is a further assumption."""
    rng = SasRanuni(seed)
    S = []
    for gid, sz in enumerate(group_sizes(n, g, remainder), 1):
        S.extend([gid] * sz)
    out = []
    for w in range(n):
        u = rng.uniform()
        j = w + int(u * (n - w))
        if j >= n:
            j = n - 1
        out.append(S[j])
        S[j] = S[w]
    return out


CANDIDATES = {
    "frontswap": lambda n, g, s: frontswap(n, g, s),
    "frontswap_ranuni": lambda n, g, s: frontswap_ranuni(n, g, s),
    "seq_quota_last": lambda n, g, s: seq_quota(n, g, s, "last")[0],
    "seq_quota_first": lambda n, g, s: seq_quota(n, g, s, "first")[0],
    "sort_blocks_last": lambda n, g, s: sort_blocks(n, g, s, "last"),
    "sort_blocks_first": lambda n, g, s: sort_blocks(n, g, s, "first"),
    "sort_deal": lambda n, g, s: sort_deal(n, g, s),
    "floyd_last": lambda n, g, s: floyd_groups(n, g, s, "last"),
    "floyd_first": lambda n, g, s: floyd_groups(n, g, s, "first"),
    "fy_down_blocks": lambda n, g, s: fy_groups(n, g, s, "down", "blocks"),
    "fy_up_blocks": lambda n, g, s: fy_groups(n, g, s, "up", "blocks"),
    "fy_down_deal": lambda n, g, s: fy_groups(n, g, s, "down", "deal"),
    "fy_up_deal": lambda n, g, s: fy_groups(n, g, s, "up", "deal"),
}


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    # structural self-tests: sizes correct, deterministic, all groups hit
    for name, fn in CANDIDATES.items():
        for n, g in ((100, 3), (503, 10), (17, 10)):
            a = fn(n, g, 8675309)
            b = fn(n, g, 8675309)
            assert a == b, name
            counts = [a.count(i) for i in range(1, g + 1)]
            assert sum(counts) == n and max(counts) - min(counts) <= 1, (name, counts)
    s = group_sizes(100, 3, "last")
    assert s == [33, 33, 34], s  # SAS doc example
    # conversion insensitivity check at CMS-ish sizes: assignments identical
    # across all four conversion candidates for seq_quota and sort variants
    for n in (137, 503, 811):
        base = None
        for conv in ("y/2^32", "(y+0.5)/2^32", "y/(2^32-1)", "(y+1)/(2^32+1)"):
            gids, nb = seq_quota(n, 10, 8675309, "last", conv)
            sb = sort_blocks(n, 10, 8675309, "last", conv)
            cur = (gids, sb)
            if base is None:
                base = cur
            assert cur == base, (n, conv, "conversion-sensitive")
            assert nb == 0, (n, conv, nb)
    print("groups_assign: self-tests passed; conversion-insensitive at test sizes")
