"""Type-1 (pooled tau_bar over ALL dataset pairs at once) consistency
curve for a group of real datasets that each had one system excluded --
companion to compute_reweight_cache_excl_system.py. Can't reuse plain
compute_consistency_curve.py: that calls scorer_scores(d, ...)/real_
systems(d) with no systems override, which would silently pull the
excluded system straight back in via each dataset's full roster.

Usage: python codes/scripts/compute_consistency_curve_excl_system.py <metametric> --tag 2024_excl_MSLC
(run compute_reweight_cache_excl_system.py first, same --tag.)
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from lib.consistency import weighted_consistency
from lib.metametrics import METAMETRICS
from lib.reweighted_consistency import dataset_alpha_0s, weighted_consistency_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('metametric', choices=sorted(METAMETRICS))
  p.add_argument('--tag', type=str, required=True)
  args = p.parse_args()
  m = args.metametric
  suffix = f'_{args.tag}'

  cache_path = os.path.join(DATA_DIR, f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  systems_by_dataset = cached['systems_by_dataset']
  print(f'{m}: loaded w_cache ({len(datasets)} datasets x {len(alphas)} alphas)', file=sys.stderr)

  points = weighted_consistency_curve(
      m, datasets, alphas, root=ROOT, w_cache=w_cache, systems_by_dataset=systems_by_dataset)
  for pt in points:
    print(f'  alpha={pt.alpha:.4f}  tau_bar={pt.tau_bar:.4f}  mean_ess={pt.mean_ess:.2f}', file=sys.stderr)

  baseline_tau_bar = weighted_consistency(
      m, datasets, root=ROOT, systems_by_dataset=systems_by_dataset)['tau_bar']
  print(f'{m}: baseline (unweighted) tau_bar={baseline_tau_bar:.4f}', file=sys.stderr)

  mean_K = float(np.mean([len(systems_by_dataset[d]) for d in datasets]))
  alpha_0 = dataset_alpha_0s(datasets, root=ROOT, systems_by_dataset=systems_by_dataset)

  out = {
      'metametric': m,
      'datasets': datasets,
      'alphas': [pt.alpha for pt in points],
      'tau_bar': [pt.tau_bar for pt in points],
      'mean_ess': [pt.mean_ess for pt in points],
      'ess_by_dataset': [pt.ess_by_dataset for pt in points],
      'baseline_tau_bar': baseline_tau_bar,
      'mean_K': mean_K,
      'alpha_0': alpha_0,
  }
  out_path = os.path.join(DATA_DIR, f'curve_{m}{suffix}.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'Wrote {out_path}', file=sys.stderr)
