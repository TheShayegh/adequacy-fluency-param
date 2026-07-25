"""One more direct test of the composition-conflict hypothesis: split the
six en_de years into two groups by their OWN natural (uniform-weight)
balance alpha_0 -- below 0.6 vs above 0.6 -- then, at the single reweighted
target alpha=0.6, compute pooled tau-a agreement (lib.consistency's
tau-a-convention pooling, same machinery as everywhere else) within each
group and between the two groups. If the mid dip is composition conflict
(years approaching 0.6 from opposite sides of their own alpha_0), within-
group agreement should stay positive while between-group agreement goes
negative.

Usage: python codes/scripts/mid_alpha_group_split_test.py
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.consistency import load_scorer_inputs
from lib.dataset_dirs import datasets_for_pair
from lib.metametrics import METAMETRICS_ORDER, NEEDS_SEGMENT_SCORES
from lib.reweighted_consistency import pooled, rankings_at, dataset_alpha_0s, solve_w_for_datasets

# Reuse the cross-regime script's already-written, already-checked helpers
# (spa_pvalue_cache, bootstrap_ci) instead of duplicating them.
sys.path.insert(0, os.path.dirname(__file__))
from cross_regime_sign_test import spa_pvalue_cache, bootstrap_ci

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ALPHA_MID = 0.6
SPLIT_THRESHOLD = 0.6


if __name__ == '__main__':
  datasets = datasets_for_pair('en-de')

  a0 = dataset_alpha_0s(datasets, root=ROOT)

  group_lo = [d for d in datasets if a0[d] < SPLIT_THRESHOLD]
  group_hi = [d for d in datasets if a0[d] >= SPLIT_THRESHOLD]
  print('alpha_0 per dataset:', file=sys.stderr)
  for d in datasets:
    print(f'  {d:10} alpha_0={a0[d]:.4f}  -> {"lo" if d in group_lo else "hi"}', file=sys.stderr)
  print(f'group_lo (alpha_0 < {SPLIT_THRESHOLD}): {group_lo}', file=sys.stderr)
  print(f'group_hi (alpha_0 >= {SPLIT_THRESHOLD}): {group_hi}\n', file=sys.stderr)

  print(f'Solving w(alpha={ALPHA_MID}) for every dataset...', file=sys.stderr)
  w_cache = solve_w_for_datasets(datasets, [ALPHA_MID], root=ROOT)
  for (d, _alpha), r in w_cache.items():
    print(f'  {d:10} success={r.success}  achieved={r.alpha_achieved:.6f}  ess={r.ess:.3f}',
          file=sys.stderr)

  inputs_cache = {m: {d: load_scorer_inputs(d, m, root=ROOT) for d in datasets}
                   for m in METAMETRICS_ORDER if m not in NEEDS_SEGMENT_SCORES}
  spa_cache = {d: spa_pvalue_cache(d, root=ROOT) for d in datasets} if 'spa' in NEEDS_SEGMENT_SCORES else {}

  within_lo_pairs = list(itertools.combinations(group_lo, 2))
  within_hi_pairs = list(itertools.combinations(group_hi, 2))
  between_pairs = [(x, y) for x in group_lo for y in group_hi]
  print(f'\nwithin_lo pairs ({len(within_lo_pairs)}): {within_lo_pairs}', file=sys.stderr)
  print(f'within_hi pairs ({len(within_hi_pairs)}): {within_hi_pairs}', file=sys.stderr)
  print(f'between pairs ({len(between_pairs)}): {between_pairs}\n', file=sys.stderr)

  combined_outcomes = {'within_lo': [], 'within_hi': [], 'between': []}
  print(f'{"metametric":10} {"within_lo":>28} {"within_hi":>28} {"between":>28}', file=sys.stderr)
  for m in METAMETRICS_ORDER:
    rankings = rankings_at(m, datasets, ALPHA_MID, w_cache, inputs_cache.get(m, {}), spa_cache)
    results = {}
    for key, pairs in (('within_lo', within_lo_pairs), ('within_hi', within_hi_pairs),
                        ('between', between_pairs)):
      tau, oc, detail = pooled(pairs, rankings, rankings)
      combined_outcomes[key].extend(oc)
      results[key] = (tau, oc, detail)
    parts = []
    for key in ('within_lo', 'within_hi', 'between'):
      tau, oc, _ = results[key]
      if oc:
        lo, hi, _, _ = bootstrap_ci(oc)
        parts.append(f'{tau:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(oc)}')
      else:
        parts.append('n/a (no pairs)')
    print(f'{m:10} {parts[0]:>28} {parts[1]:>28} {parts[2]:>28}', file=sys.stderr)

  print(f'\n=== Combined across all 5 metametrics (alpha={ALPHA_MID}) ===', file=sys.stderr)
  for key in ('within_lo', 'within_hi', 'between'):
    oc = combined_outcomes[key]
    if not oc:
      print(f'{key:10} n/a (no pairs)', file=sys.stderr)
      continue
    tau_bar = float(np.mean(oc))
    lo, hi, p_ge_0, p_le_0 = bootstrap_ci(oc)
    print(f'{key:10} tau_bar={tau_bar:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={len(oc)}  '
          f'P(>=0)={p_ge_0:.4f}  P(<=0)={p_le_0:.4f}', file=sys.stderr)
