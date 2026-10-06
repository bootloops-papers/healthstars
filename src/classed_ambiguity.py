# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""The classed ambiguity census.
129 contract-years touch the bonus line over the 28 sampled input orders of
the exact arm within the resampling groups vs published, classed by
organization type and published-rating status.
Output: runs/classed_ambiguity.json

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 4.3 and Table 2 block
    c (129 contract-years change bonus status under at least one sampled order,
    98 with no other caveat).
Run:  cd src && python3 classed_ambiguity.py
Requires: Python >= 3.10; numpy, scipy.
"""
import csv, json, os
from collections import defaultdict
import enclosure_census_docsonly as C
import sys
BASE = C.BASE
if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

b = json.load(open(os.path.join(BASE, 'runs/enclosure_collapse_frontswap_refinal.json')))
r = json.load(open(os.path.join(BASE, 'runs/enclosure_collapse_frontswap_rperm_refinal.json')))
count = defaultdict(int); dirs = {}
for src_ in (b['point_flip_sets'], r['point_flip_sets']):
    for fl in src_.values():
        for x in fl:
            k = (x['year'], x['contract_id'])
            count[k] += 1; dirs[k] = x['direction']
summ = {}
for y, v in (('2024','recalc'), ('2025','current'), ('2026','current')):
    with open(os.path.join(BASE, f'data/parsed/summary_{y}_{v}.csv'),
              newline='', encoding='utf-8-sig') as fh:
        summ[y] = {row['contract_id']: row for row in csv.DictReader(fh)}
def cls(k):
    y, c = k
    row = summ[y].get(c, {})
    if c.startswith('S'): return 'pdp'
    if 'Cost' in row.get('org_type', ''): return 'cost'
    if not row.get('overall_raw', '').strip().replace('.', '').isdigit():
        return 'no_published_overall'
    return 'clean'
def tally(keys):
    t = defaultdict(int)
    for k in keys: t[cls(k)] += 1
    return dict(t)
touch = sorted(count)
amb = sorted(k for k, n in count.items() if 0 < n < 28)
stable = sorted(k for k, n in count.items() if n == 28)
out = {
 'register': ("Classed ambiguity census: the exact arm within the resampling "
           "groups vs published ratings over the 28 sampled input orders; "
           "classes by organization type and published-rating status. The "
           "orders are sampled, so the sets can only grow or reclassify "
           "under further sampling."),
 'generator': 'src/classed_ambiguity.py',
 'inputs': ['runs/enclosure_collapse_frontswap_refinal.json',
            'runs/enclosure_collapse_frontswap_rperm_refinal.json',
            'data/parsed/summary_*.csv'],
 'touching_the_line': dict(n=len(touch), by_class=tally(touch)),
 'ambiguous': dict(n=len(amb), by_class=tally(amb)),
 'stable_all_28': dict(n=len(stable), by_class=tally(stable),
                       rows=[dict(year=k[0], contract_id=k[1],
                                  direction=dirs[k], cls=cls(k))
                             for k in stable]),
}
p = os.path.join(BASE, 'runs/classed_ambiguity.json')
json.dump(out, open(p, 'w'), indent=1); open(p, 'a').write('\n')
print(json.dumps({k: out[k] for k in ('touching_the_line','ambiguous')}, indent=1))
print('stable:', out['stable_all_28']['n'], out['stable_all_28']['by_class'])
