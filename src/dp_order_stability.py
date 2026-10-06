# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Order stability of the exact optimum and Ward's method inside the ten
resampling groups, derived from the stored 28-order verification files.
The exact optimizer inside CMS's ten resampling groups, averaged, is the
stored dp arm; this record measures how much the unpublished input order
moves each arm's final boundaries.
Output: runs/dp_order_stability.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 (65 of 114
    computations fully invariant to the input order with the exact optimum,
    against 20 with Ward's method).
Run:  cd src && python3 dp_order_stability.py
Requires: Python >= 3.10, standard library only.
"""
import json, os
import sys
from collections import defaultdict
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

b = json.load(open(os.path.join(BASE, 'runs/enclosure_collapse_frontswap_refinal.json')))
r = json.load(open(os.path.join(BASE, 'runs/enclosure_collapse_frontswap_rperm_refinal.json')))
allrows = b['measure_results'] + r['measure_results']
err = {(x['year'],x['mid'],x['org']) for x in allrows if 'error' in x}
by_split = defaultdict(list)
for x in allrows:
    if 'error' in x: continue
    k = (x['year'],x['mid'],x['org'])
    if k in err: continue
    by_split[k].append(x)
full = {k: rs for k, rs in by_split.items() if len(rs) == 28}
res = dict(n_realizations=28, n_splits=len(full))
for arm, fld in (('dp','dp_final'), ('ward','ward_final')):
    vi = sum(1 for rs in full.values() if len({tuple(x[fld]) for x in rs}) == 1)
    ci = sum(1 for rs in full.values() for i in range(4)
             if len({x[fld][i] for x in rs}) == 1)
    res[f'{arm}_vector_invariant_splits'] = vi
    res[f'{arm}_cells_invariant'] = ci
    res[f'{arm}_cells_total'] = 4 * len(full)
out = {
 'register': ("Order stability of the two arms within the resampling groups "
   "across the 28 sampled input orders (4 conventions + 24 uniform-random "
   "permutations, assumed mapping; splits erroring anywhere excluded, "
   "n=114). dp = exact optimizer within each of CMS's ten resampling "
   "groups, mean-averaged; ward = the published-method replay. The exact "
   "optimizer inside the resampling groups is about twice as order-stable "
   "as Ward's method (65 vs 20 of 114 splits fully invariant; 73.7% vs "
   "41.4% of boundary cells) but not order-free: the unpublished order "
   "still matters to this construction, whose sampled answer is the "
   "ambiguity census (11 stable / 118 order-dependent). Derived entirely "
   "from stored records; no new computation."),
 'generator': 'src/dp_order_stability.py',
 'inputs': ['runs/enclosure_collapse_frontswap_refinal.json',
            'runs/enclosure_collapse_frontswap_rperm_refinal.json'],
 'results': res,
}
p = os.path.join(BASE, 'runs/dp_order_stability.json')
json.dump(out, open(p, 'w'), indent=1); open(p,'a').write('\n')
print(json.dumps(res, indent=1)); print('->', p)
