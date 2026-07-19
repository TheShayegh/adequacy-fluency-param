"""Type-2 (star-pooled tau_bar per dataset) consistency curve for a group
of real datasets that each had one system excluded -- companion to
compute_reweight_cache_excl_system.py, and the excl-system analog of
compute_per_dataset_consistency_curve.py (which has no systems override
and would silently pull the excluded system back in via each dataset's
full roster).

Usage: python codes/scripts/compute_per_dataset_consistency_curve_excl_system.py --tag 2024_excl_MSLC
(run compute_reweight_cache_excl_system.py first, same --tag.)
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import load_scorer_inputs, scorer_scores
from lib.metametrics import METAMETRICS, NEEDS_SEGMENT_SCORES
from mwb.mqm_scoring import load_system_scores
from cross_regime_sign_test import METAMETRICS_ORDER, rankings_at, pooled, spa_pvalue_cache

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')


def spa_pvalue_cache_for_systems(dataset, systems):
  from lib.metric_scores import discover_metrics, load_human_seg_scores, load_metric_seg_scores, jointly_valid_columns
  from lib.metametrics import pairwise_p_values
  _MIN_SPA_SEGMENTS = 10
  human_seg = load_human_seg_scores(dataset, systems, root=ROOT)
  if human_seg is None:
    return {}
  out = {}
  for name in discover_metrics(dataset, ROOT):
    metric_seg = load_metric_seg_scores(dataset, name, systems, root=ROOT)
    if metric_seg is None:
      continue
    mask = jointly_valid_columns(human_seg, metric_seg)
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = (pairwise_p_values(human_seg[:, mask]), pairwise_p_values(metric_seg[:, mask]))
  return out


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, required=True)
  args = p.parse_args()
  suffix = f'_{args.tag}'

  cache_path = os.path.join(DATA_DIR, f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  systems_by_dataset = cached['systems_by_dataset']
  print(f'loaded w_cache: {len(datasets)} datasets x {len(alphas)} alphas', file=sys.stderr)

  K = {d: len(systems_by_dataset[d]) for d in datasets}
  alpha_0 = {}
  for d in datasets:
    sub = load_system_scores(d, root=ROOT).loc[systems_by_dataset[d]]
    alpha_0[d] = alpha0_of(sub['a'].values, sub['b'].values)

  spa_cache = {d: spa_pvalue_cache_for_systems(d, systems_by_dataset[d]) for d in datasets}

  for m in METAMETRICS_ORDER:
    t0 = time.time()
    inputs_cache = ({d: load_scorer_inputs(d, m, root=ROOT, systems=systems_by_dataset[d])
                      for d in datasets} if m not in NEEDS_SEGMENT_SCORES else {})
    rankings_by_alpha = {a: rankings_at(m, datasets, a, w_cache, inputs_cache, spa_cache) for a in alphas}
    natural_rankings = {d: scorer_scores(d, m, root=ROOT, systems=systems_by_dataset[d]) for d in datasets}

    tau_bar_by_dataset = {}
    ess_by_dataset = {}
    baseline_tau_by_dataset = {}
    for d in datasets:
      others = [o for o in datasets if o != d]
      tau_bar_by_dataset[d] = [
          pooled([(d, o) for o in others], rankings_by_alpha[a], rankings_by_alpha[a])[0]
          for a in alphas
      ]
      ess_by_dataset[d] = [w_cache[(d, a)].ess for a in alphas]
      baseline_tau_by_dataset[d] = pooled([(d, o) for o in others], natural_rankings, natural_rankings)[0]

    out = {
        'metametric': m,
        'datasets': datasets,
        'K': K,
        'alpha_0': alpha_0,
        'alphas': list(alphas),
        'tau_bar_by_dataset': tau_bar_by_dataset,
        'ess_by_dataset': ess_by_dataset,
        'baseline_tau_by_dataset': baseline_tau_by_dataset,
    }
    out_path = os.path.join(DATA_DIR, f'curve_perdataset_{m}{suffix}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{m:10} ({time.time() - t0:.1f}s) -> {os.path.basename(out_path)}', file=sys.stderr)
