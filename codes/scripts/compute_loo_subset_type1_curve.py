"""Type-1 (pooled tau_bar over ALL pairs at once) consistency analysis for
leave-one-system-out subsets of a single dataset -- the pooled counterpart
to compute_loo_subset_consistency_curve.py's star-pooled (type-2) version.
Reuses that script's already-persisted w_cache_<tag>.pkl (no re-solving);
writes curve_<metametric>[_TAG].json in the exact schema compute_
consistency_curve.py produces, so plot_consistency_curves.py renders it
unchanged.

Can't just call compute_consistency_curve.py directly on the LOO tag: its
whole pipeline (weighted_consistency, scorer_scores(d, ...)) treats `d` as
a literal WMT dataset key it can hand to mwb.mqm_scoring.load_system_scores
-- the repeat_ids here ("excl_<system>") aren't real datasets, only valid
via the base-dataset + systems-override calling convention already used in
compute_loo_subset_consistency_curve.py.

Usage: python codes/scripts/compute_loo_subset_type1_curve.py [--dataset ende24] [--tag loo_ende24]
(run compute_loo_subset_consistency_curve.py first, same --dataset/--tag.)
"""

import argparse
import itertools
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, load_scorer_inputs, scorer_scores, pool_weighted_tau
from lib.metametrics import NEEDS_SEGMENT_SCORES
from mwb.mqm_scoring import load_system_scores
from cross_regime_sign_test import METAMETRICS_ORDER, rankings_at
from compute_loo_subset_consistency_curve import spa_pvalue_cache_for_systems

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende24')
  p.add_argument('--tag', type=str, default=None, help='defaults to loo_<dataset>[_excl_<system>]')
  p.add_argument('--exclude-system', type=str, default=None,
                  help='must match what was passed to compute_loo_subset_consistency_curve.py, if anything')
  args = p.parse_args()
  base = args.dataset
  tag = args.tag or (f'loo_{base}_excl_{args.exclude_system}' if args.exclude_system else f'loo_{base}')
  suffix = f'_{tag}'

  cache_path = os.path.join(DATA_DIR, f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  repeat_ids, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  print(f'loaded w_cache: {len(repeat_ids)} subsets x {len(alphas)} alphas', file=sys.stderr)

  full_systems = real_systems(base, root=ROOT)
  if args.exclude_system:
    full_systems = [s for s in full_systems if s != args.exclude_system]
  # Same repeat_id -> subset convention as compute_loo_subset_consistency_curve.py.
  systems_by_repeat = {f'excl_{s}': [x for x in full_systems if x != s] for s in full_systems}

  spa_cache = {rid: spa_pvalue_cache_for_systems(base, systems_by_repeat[rid]) for rid in repeat_ids}
  mean_K = float(np.mean([len(systems_by_repeat[rid]) for rid in repeat_ids]))

  # Each repeat's own natural balance computed directly on its |S|=K-1
  # subset (uniform weight over the remaining systems, i.e. the excluded
  # system at weight 0) -- real_systems/load_system_scores can't resolve
  # "excl_<system>" as a dataset name, so plot_consistency_curves.py's
  # usual dataset_alpha0s() lookup silently finds nothing for these; this
  # is read directly from the JSON instead when present (see that script).
  df_all = load_system_scores(base, root=ROOT)
  alpha_0 = {}
  for rid in repeat_ids:
    sub = df_all.loc[systems_by_repeat[rid]]
    alpha_0[rid] = alpha0_of(sub['a'].values, sub['b'].values)

  for m in METAMETRICS_ORDER:
    inputs_cache = ({rid: load_scorer_inputs(base, m, root=ROOT, systems=systems_by_repeat[rid])
                      for rid in repeat_ids} if m not in NEEDS_SEGMENT_SCORES else {})

    points_alpha, points_tau, points_ess, points_ess_by_dataset = [], [], [], []
    for a in alphas:
      rankings = rankings_at(m, repeat_ids, a, w_cache, inputs_cache, spa_cache)
      tau_bar, _, _ = pool_weighted_tau(rankings, repeat_ids)
      ess_by = {rid: w_cache[(rid, a)].ess for rid in repeat_ids}
      points_alpha.append(a)
      points_tau.append(tau_bar)
      points_ess.append(float(np.mean(list(ess_by.values()))))
      points_ess_by_dataset.append(ess_by)

    natural_rankings = {rid: scorer_scores(base, m, root=ROOT, systems=systems_by_repeat[rid])
                          for rid in repeat_ids}
    baseline_tau_bar, _, _ = pool_weighted_tau(natural_rankings, repeat_ids)

    out = {
        'metametric': m,
        'datasets': repeat_ids,
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
    print(f'{m:10} baseline_tau_bar={baseline_tau_bar:.4f} -> {os.path.basename(out_path)}',
          file=sys.stderr)
