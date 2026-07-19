"""Computes ONE meta-metric's M(alpha) x consistency curve (action_plan.md
6.1 x 6.5) using the w(alpha) cache from compute_reweight_cache.py, and
saves the result to artifacts/data/curve_<metametric>.json for the separate
plotting script (plot_consistency_curves.py) to pick up. Run once per
meta-metric (pearson/spearman/kendall/pa/spa) so progress is visible one at
a time, rather than computing all five before anything is saved.

Usage: python codes/scripts/compute_consistency_curve.py <metametric> [--tag TAG]
--tag must match the --years tag used with compute_reweight_cache.py (if
any); loads w_cache[_TAG].pkl and writes curve_<metametric>[_TAG].json.
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
from lib.reweighted_consistency import weighted_consistency_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('metametric', choices=sorted(METAMETRICS))
  p.add_argument('--tag', type=str, default=None,
                  help='must match the --years tag used with compute_reweight_cache.py, if any')
  args = p.parse_args()
  metametric = args.metametric
  suffix = f'_{args.tag}' if args.tag else ''

  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], cached['alphas'], cached['w_cache']
  print(f'{metametric}: loaded w_cache ({len(datasets)} datasets x {len(alphas)} alphas)',
        file=sys.stderr)

  t0 = time.time()
  points = weighted_consistency_curve(metametric, datasets, alphas, root=ROOT, w_cache=w_cache)
  print(f'{metametric}: curve computed in {time.time() - t0:.1f}s', file=sys.stderr)
  for p in points:
    print(f'  alpha={p.alpha:.4f}  tau_bar={p.tau_bar:.4f}  mean_ess={p.mean_ess:.2f}', file=sys.stderr)

  baseline = weighted_consistency(metametric, datasets, root=ROOT)
  print(f'{metametric}: baseline (unweighted) tau_bar={baseline["tau_bar"]:.4f}', file=sys.stderr)

  mean_K = sum(len(real_systems(d, root=ROOT)) for d in datasets) / len(datasets)

  out = {
      'metametric': metametric,
      'datasets': datasets,
      'alphas': [p.alpha for p in points],
      'tau_bar': [p.tau_bar for p in points],
      'mean_ess': [p.mean_ess for p in points],
      'ess_by_dataset': [p.ess_by_dataset for p in points],
      'baseline_tau_bar': baseline['tau_bar'],
      'mean_K': mean_K,
  }
  os.makedirs(OUT_DIR, exist_ok=True)
  out_path = os.path.join(OUT_DIR, f'curve_{metametric}{suffix}.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'Wrote {out_path}', file=sys.stderr)
