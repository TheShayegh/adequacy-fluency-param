"""Leave-one-dataset-out "shadow" curves for plot_consistency_curves.py:
for a tag's existing w_cache[_TAG].pkl (already solved w_d(alpha) for every
dataset independently, action_plan.md 6.1), recomputes ONLY the pooling
step of the M(alpha) x consistency curve (6.5) with each dataset dropped in
turn -- no new solver calls, since w_d(alpha) never depended on which other
datasets shared the run. Same alpha grid as the main curve, so shadows
overlay directly onto the same x-axis.

Usage: python codes/scripts/compute_loo_shadow_curves.py [--tag TAG]
(run compute_reweight_cache.py [--tag-matching-flags] first.)
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.consistency import real_systems, weighted_consistency
from lib.metametrics import METAMETRICS
from lib.reweighted_consistency import weighted_consistency_curve, _spa_pvalue_cache

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, default=None,
                  help='must match the tag used with compute_reweight_cache.py, if any')
  args = p.parse_args()
  suffix = f'_{args.tag}' if args.tag else ''

  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], cached['alphas'], cached['w_cache']
  print(f'loaded w_cache: {len(datasets)} datasets x {len(alphas)} alphas', file=sys.stderr)

  # Built once per dataset (weighting-independent, action_plan.md 6.5's SPA
  # pairwise p-values), reused across all 6 leave-one-out groups instead of
  # rebuilding per group.
  spa_cache = {d: _spa_pvalue_cache(d, root=ROOT) for d in datasets}

  for excluded in datasets:
    remaining = [d for d in datasets if d != excluded]
    shadow_tag = f'{suffix}_shadow_excl_{excluded}' if suffix else f'_shadow_excl_{excluded}'
    print(f'\n-- excluding {excluded} ({len(remaining)} datasets remain) --', file=sys.stderr)

    for metametric in sorted(METAMETRICS):
      t0 = time.time()
      points = weighted_consistency_curve(
          metametric, remaining, alphas, root=ROOT, w_cache=w_cache, spa_cache=spa_cache)
      baseline = weighted_consistency(metametric, remaining, root=ROOT)
      mean_K = sum(len(real_systems(d, root=ROOT)) for d in remaining) / len(remaining)
      out = {
          'metametric': metametric,
          'datasets': remaining,
          'excluded': excluded,
          'alphas': [pt.alpha for pt in points],
          'tau_bar': [pt.tau_bar for pt in points],
          'mean_ess': [pt.mean_ess for pt in points],
          'ess_by_dataset': [pt.ess_by_dataset for pt in points],
          'baseline_tau_bar': baseline['tau_bar'],
          'mean_K': mean_K,
      }
      out_path = os.path.join(OUT_DIR, f'curve_{metametric}{shadow_tag}.json')
      with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
      print(f'  {metametric:10} ({time.time() - t0:.1f}s) -> {os.path.basename(out_path)}',
            file=sys.stderr)
