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

from lib.consistency import load_scorer_inputs, scorer_scores, pool_weighted_tau
from lib.metametrics import METAMETRICS, NEEDS_SEGMENT_SCORES
from lib.reweighted_consistency import dataset_alpha_0s, rankings_at, spa_pvalue_cache

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

  # SPA isn't needed for rankings_at unless it's actually requested, but it
  # takes an explicit spa_cache dict either way.
  spa_cache = ({d: spa_pvalue_cache(d, systems_by_dataset[d], root=ROOT) for d in datasets}
               if m in NEEDS_SEGMENT_SCORES else {})

  inputs_cache = ({d: load_scorer_inputs(d, m, root=ROOT, systems=systems_by_dataset[d])
                    for d in datasets} if m not in NEEDS_SEGMENT_SCORES else {})

  points_alpha, points_tau, points_ess, points_ess_by_dataset = [], [], [], []
  for a in alphas:
    rankings = rankings_at(m, datasets, a, w_cache, inputs_cache, spa_cache)
    tau_bar, _, _ = pool_weighted_tau(rankings, datasets)
    ess_by = {d: w_cache[(d, a)].ess for d in datasets}
    points_alpha.append(a)
    points_tau.append(tau_bar)
    points_ess.append(float(np.mean(list(ess_by.values()))))
    points_ess_by_dataset.append(ess_by)
    print(f'  alpha={a:.4f}  tau_bar={tau_bar:.4f}  mean_ess={points_ess[-1]:.2f}', file=sys.stderr)

  natural_rankings = {d: scorer_scores(d, m, root=ROOT, systems=systems_by_dataset[d]) for d in datasets}
  baseline_tau_bar, _, _ = pool_weighted_tau(natural_rankings, datasets)
  print(f'{m}: baseline (unweighted) tau_bar={baseline_tau_bar:.4f}', file=sys.stderr)

  mean_K = float(np.mean([len(systems_by_dataset[d]) for d in datasets]))
  alpha_0 = dataset_alpha_0s(datasets, root=ROOT, systems_by_dataset=systems_by_dataset)

  out = {
      'metametric': m,
      'datasets': datasets,
      'alphas': points_alpha,
      'tau_bar': points_tau,
      'mean_ess': points_ess,
      'ess_by_dataset': points_ess_by_dataset,
      'baseline_tau_bar': baseline_tau_bar,
      'mean_K': mean_K,
      'alpha_0': alpha_0,
  }
  out_path = os.path.join(DATA_DIR, f'curve_{m}{suffix}.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'Wrote {out_path}', file=sys.stderr)
