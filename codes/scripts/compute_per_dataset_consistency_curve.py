"""Per-dataset variant of compute_consistency_curve.py: instead of pooling
tau-a over every (scorer-pair, dataset-pair) comparison at once (one number
per alpha), pools it separately for each dataset d against every OTHER
dataset ("d vs the rest", a star centered at d) -- one tau_bar(d, alpha)
curve per dataset. Reuses the existing w_cache[_TAG].pkl (no new solving:
w_d(alpha) never depended on which other datasets were in the pool) and
lib.consistency.scorer_scores' natural (unweighted) ranking for each
dataset's own baseline point (its natural alpha_0, and its natural,
unreweighted star-pooled tau against the others).

Usage: python codes/scripts/compute_per_dataset_consistency_curve.py [--tag TAG]
"""

import argparse
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, load_scorer_inputs, scorer_scores
from lib.metametrics import METAMETRICS, NEEDS_SEGMENT_SCORES
from mwb.mqm_scoring import load_system_scores
from cross_regime_sign_test import rankings_at, spa_pvalue_cache, pooled

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
  datasets, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  print(f'loaded w_cache: {len(datasets)} datasets x {len(alphas)} alphas', file=sys.stderr)

  K = {d: len(real_systems(d, root=ROOT)) for d in datasets}
  alpha_0 = {}
  for d in datasets:
    systems = real_systems(d, root=ROOT)
    df = load_system_scores(d, root=ROOT).loc[systems]
    alpha_0[d] = alpha0_of(df['a'].values, df['b'].values)

  spa_cache = {d: spa_pvalue_cache(d) for d in datasets}

  for metametric in sorted(METAMETRICS):
    t0 = time.time()
    inputs_cache = ({d: load_scorer_inputs(d, metametric, root=ROOT) for d in datasets}
                     if metametric not in NEEDS_SEGMENT_SCORES else {})
    rankings_by_alpha = {a: rankings_at(metametric, datasets, a, w_cache, inputs_cache, spa_cache)
                          for a in alphas}
    natural_rankings = {d: scorer_scores(d, metametric, root=ROOT) for d in datasets}

    tau_bar_by_dataset = {}
    ess_by_dataset = {}
    baseline_tau_by_dataset = {}
    for d in datasets:
      others = [dp for dp in datasets if dp != d]
      tau_bar_by_dataset[d] = [
          pooled([(d, dp) for dp in others], rankings_by_alpha[a], rankings_by_alpha[a])[0]
          for a in alphas
      ]
      ess_by_dataset[d] = [w_cache[(d, a)].ess for a in alphas]
      baseline_tau_by_dataset[d] = pooled([(d, dp) for dp in others],
                                           natural_rankings, natural_rankings)[0]

    out = {
        'metametric': metametric,
        'datasets': datasets,
        'alphas': alphas,
        'K': K,
        'alpha_0': alpha_0,
        'tau_bar_by_dataset': tau_bar_by_dataset,
        'ess_by_dataset': ess_by_dataset,
        'baseline_tau_by_dataset': baseline_tau_by_dataset,
    }
    out_path = os.path.join(OUT_DIR, f'curve_perdataset_{metametric}{suffix}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{metametric:10} ({time.time() - t0:.1f}s) -> {os.path.basename(out_path)}',
          file=sys.stderr)
