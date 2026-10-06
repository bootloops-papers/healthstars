# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Random-order sweep under the assumed group-assignment rule (see
collapse_census.py).

The 23-contract flip set of collapse_census.py holds over the four
input-order conventions (cid/score x asc/desc) under the assumed front-swap
mapping. The four conventions are not the full order space; this driver
samples 24 uniform-random permutations under the same assumed rule.

Scope: a sampled extension, not a proof; a breaker remains possible at the
next realization. Each realization is a point statement: the assumed
front-swap mapping (seed 8675309) applied to a uniform-random permutation of
each measure-split's reconciled kept list, rperm_i = random.Random(6000+i)
shuffle (the same convention and seeds as the stored k72 census axis).
Everything else (jobs, membership, vintages, guardrails, arms, aggregation)
is identical to the stored docsonly census and the four-convention rerun.
The rperm order token is registered at runtime (spawn-safe via a Pool
initializer), following the census module's own urand convention.

Output: runs/enclosure_collapse_frontswap_rperm.json
Run:    python3 collapse_rperm.py pilot [workers]   # 12 jobs x rperm_1,2
        python3 collapse_rperm.py full  [workers]   # all jobs x rperm_1..24

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (the 28 sampled
    input orders: four conventional orderings and 24 random permutations);
    Appendix C.3.
Requires: Python >= 3.10; numpy, scipy.
"""
import json
import multiprocessing as mp
import os
import random
import sys

import enclosure_census_docsonly as C
import order_grid_screen as OG

N_RPERM = 24
OUT = os.path.join(C.BASE, "runs", "enclosure_collapse_frontswap_rperm.json")
STORED = os.path.join(C.BASE, "runs", "enclosure_collapse_frontswap.json")

_ORIG_ORDER_PAIRS = OG.order_pairs


def order_pairs_rperm(kept, order):
    """rperm_i: uniform-random permutation of the reconciled kept list,
    random.Random(6000+i), fresh rng per measure-split (k72 convention).
    Every other token defers to the module's order_pairs."""
    if order.startswith("rperm_"):
        i = int(order.split("_", 1)[1])
        rng = random.Random(6000 + i)
        k = list(kept)
        rng.shuffle(k)
        return k
    return _ORIG_ORDER_PAIRS(kept, order)


def _init_worker():
    # spawn children re-import modules: rebind the order hook where the
    # census _worker resolves it (its module globals) + the source module.
    C.order_pairs = order_pairs_rperm
    OG.order_pairs = order_pairs_rperm


def run_measure_phase_patched(jobs, realizations, workers):
    """C.run_measure_phase with a Pool initializer (it has no such param)."""
    tasks = [(j, rn) for rn in realizations for j in jobs]
    print(f"measure-splits: {len(jobs)}  realizations: {len(realizations)}  "
          f"tasks: {len(tasks)}", flush=True)
    ctx = mp.get_context("spawn")            # fork can deadlock with BLAS thread pools
    rows = []
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(C._worker, tasks,
                                                  chunksize=1)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} measure tasks", flush=True)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"measure phase: errors: {n_err}", flush=True)
    return rows


def main(mode, workers):
    _init_worker()                            # parent-side too
    jobs, skips = C.build_jobs()
    n_rperm = 2 if mode == "pilot" else N_RPERM
    if mode == "pilot":
        jobs = jobs[:12]
    realizations = [f"frontswap@rperm_{i}" for i in range(1, n_rperm + 1)]

    rows = run_measure_phase_patched(jobs, realizations, workers)
    result = {
        "register": (
            "Random-order sweep under the assumed group-assignment rule "
            "(front-swap Fisher-Yates readout, seed 8675309): frontswap x "
            f"rperm_1..{n_rperm}, rperm_i = random.Random(6000+i) uniform "
            "permutation of each measure-split's reconciled kept list (the "
            "stored k72 census's rperm convention and seeds, paired with the "
            "assumed readout). A sampled extension, not a proof: a breaker "
            "remains possible at the next realization. Each realization is a "
            "point statement conditional on its order. Everything else is "
            "identical to the stored four-convention rerun (jobs, membership, "
            "vintages, guardrails, arms, aggregation)."),
        "mode": mode,
        "realizations": realizations,
        "n_jobs": len(jobs),
        "skipped_splits": skips,
        "measure_results": rows,
    }

    if mode == "full":
        agg = C.enclosure_aggregate(rows, realizations)
        point_sets = {}
        for rn in realizations:
            flips = [dict(year=c["year"], contract_id=c["contract_id"],
                          org_class=c["org_class"], direction=c["direction"],
                          baseline=c["baseline"],
                          rating=c["ratings_by_realization"][rn])
                     for c in agg["contracts"]
                     if c["ratings_by_realization"].get(rn) is not None
                     and c["ratings_by_realization"][rn] != c["baseline"]
                     and ((C.F(c["baseline"]) >= 4)
                          != (C.F(c["ratings_by_realization"][rn]) >= 4))]
            point_sets[rn] = flips

        # survival of the stored 23-contract set across the rperm realizations
        stored = json.load(open(STORED))
        cert23 = [e for e in stored["four_order_envelope"] if e["n_flip"] == 4]
        flip_keys = {rn: {(f["year"], f["contract_id"]) for f in fl}
                     for rn, fl in point_sets.items()}
        survival = []
        for e in sorted(cert23, key=lambda x: (x["year"], x["contract_id"])):
            key = (e["year"], e["contract_id"])
            hits = [rn for rn in realizations if key in flip_keys[rn]]
            survival.append(dict(
                year=e["year"], contract_id=e["contract_id"],
                direction=e["direction"],
                n_flip_rperm=len(hits), n_rperm=n_rperm,
                survives_all=len(hits) == n_rperm,
                breaker_realizations=[rn for rn in realizations
                                      if key not in flip_keys[rn]]))
        n_surv = sum(1 for s in survival if s["survives_all"])
        # new flips appearing under random orders but not in the 4-conv envelope
        env_keys = {(e["year"], e["contract_id"])
                    for e in stored["four_order_envelope"]}
        new_flips = sorted({k for rn in realizations for k in flip_keys[rn]}
                           - env_keys)
        result.update(
            aggregation={k: agg[k] for k in ("per_realization",
                                             "per_realization_year",
                                             "errored_splits", "n_rated")},
            contracts=agg["contracts"],
            point_flip_sets=point_sets,
            certified23_survival=survival,
            summary=dict(
                n_certified23=len(cert23),
                n_survive_all_rperm=n_surv,
                n_broken=len(cert23) - n_surv,
                broken=[dict(year=s["year"], contract_id=s["contract_id"],
                             n_flip_rperm=s["n_flip_rperm"],
                             breakers=s["breaker_realizations"])
                        for s in survival if not s["survives_all"]],
                per_rperm_flip_counts={rn: len(point_sets[rn])
                                       for rn in realizations},
                new_flips_outside_4conv_envelope=[
                    dict(year=y, contract_id=c) for y, c in new_flips]),
        )
        print(json.dumps(result["summary"], indent=1))

    out = OUT if mode == "full" else OUT.replace(".json", "_pilot.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1, default=str)
        f.write("\n")
    print(f"-> {out}")


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    main(mode, workers)
