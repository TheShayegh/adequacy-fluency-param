"""Computes ONE meta-metric's M(alpha) x consistency curve over bootstrap
resamples of a single dataset's systems (compute_bootstrap_reweight_
cache.py's cache) -- the literal action_plan.md 6.5 reading ("draw many
system subsets from the real systems only"), pooling weighted Kendall's tau
across PAIRS OF BOOTSTRAP REPEATS instead of pairs of real datasets. Saves
artifacts/data/curve_<metametric>_bootstrap_<dataset>.json in the same
schema plot_consistency_curves.py already reads, so no plotting changes are
needed -- just pass --tag bootstrap_<dataset>.

Usage: python codes/scripts/compute_bootstrap_consistency_curve.py <metametric> <dataset>
(run compute_bootstrap_reweight_cache.py <dataset> first.)
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from lib.consistency import load_scorer_inputs, evaluate_scorer_scores, scorer_scores, pool_weighted_tau
from lib.metametrics import (
    METAMETRICS, NEEDS_SEGMENT_SCORES, pairwise_p_values, soft_pairwise_accuracy_from_pvalues,
)
from lib.metric_scores import (
    discover_metrics, load_human_seg_scores, load_metric_seg_scores, jointly_valid_columns,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_MIN_SPA_SEGMENTS = 10


def spa_pvalue_cache(dataset: str, systems: list[str], root: str) -> dict:
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  out = {}
  if human_seg is None:
    return out
  for name in discover_metrics(dataset, root):
    metric_seg = load_metric_seg_scores(dataset, name, systems, root=root)
    if metric_seg is None:
      continue
    mask = jointly_valid_columns(human_seg, metric_seg)
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = (pairwise_p_values(human_seg[:, mask]), pairwise_p_values(metric_seg[:, mask]))
  return out


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('metametric', choices=sorted(METAMETRICS))
  p.add_argument('dataset')
  args = p.parse_args()
  metametric, dataset = args.metametric, args.dataset

  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache_bootstrap_{dataset}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  repeat_ids = cached['repeat_ids']
  systems_by_repeat = cached['systems_by_repeat']
  alphas = cached['alphas']
  w_cache = cached['w_cache']
  print(f'{metametric}: loaded bootstrap w_cache ({len(repeat_ids)} repeats x {len(alphas)} alphas)',
        file=sys.stderr)

  needs_seg = metametric in NEEDS_SEGMENT_SCORES
  t0 = time.time()
  if needs_seg:
    spa_cache = {rid: spa_pvalue_cache(dataset, systems_by_repeat[rid], ROOT) for rid in repeat_ids}
  else:
    inputs_cache = {
        rid: load_scorer_inputs(dataset, metametric, root=ROOT, systems=systems_by_repeat[rid])
        for rid in repeat_ids
    }
  print(f'{metametric}: setup in {time.time() - t0:.1f}s', file=sys.stderr)

  mean_ess_list, tau_bar_list, ess_by_repeat_list = [], [], []
  for alpha in alphas:
    ess_by_repeat = {rid: w_cache[(rid, alpha)].ess for rid in repeat_ids}
    if needs_seg:
      rankings = {}
      for rid in repeat_ids:
        w = w_cache[(rid, alpha)].w
        vals = {}
        for name, (hp, mp) in spa_cache[rid].items():
          v = soft_pairwise_accuracy_from_pvalues(hp, mp, w)
          if v == v:
            vals[name] = v
        rankings[rid] = pd.Series(vals, dtype=float)
    else:
      rankings = {
          rid: evaluate_scorer_scores(inputs_cache[rid], metametric, w_cache[(rid, alpha)].w)
          for rid in repeat_ids
      }
    tau_bar, _, _ = pool_weighted_tau(rankings, repeat_ids)
    tau_bar_list.append(tau_bar)
    mean_ess_list.append(float(np.mean(list(ess_by_repeat.values()))))
    ess_by_repeat_list.append(ess_by_repeat)
    print(f'  alpha={alpha:.4f}  tau_bar={tau_bar:.4f}  mean_ess={mean_ess_list[-1]:.2f}', file=sys.stderr)

  # Baseline: unweighted (natural) consistency across the SAME bootstrap repeats.
  baseline_rankings = {
      rid: scorer_scores(dataset, metametric, root=ROOT, systems=systems_by_repeat[rid])
      for rid in repeat_ids
  }
  baseline_tau, baseline_weight, _ = pool_weighted_tau(baseline_rankings, repeat_ids)
  print(f'{metametric}: baseline (unweighted, same repeats) tau_bar={baseline_tau:.4f}, '
        f'weight={baseline_weight}', file=sys.stderr)

  mean_K = float(np.mean([len(systems_by_repeat[rid]) for rid in repeat_ids]))

  out = {
      'metametric': metametric,
      'datasets': repeat_ids,
      'alphas': [float(a) for a in alphas],
      'tau_bar': tau_bar_list,
      'mean_ess': mean_ess_list,
      'ess_by_dataset': ess_by_repeat_list,
      'baseline_tau_bar': baseline_tau,
      'mean_K': mean_K,
  }
  out_dir = os.path.join(ROOT, 'artifacts', 'data')
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'curve_{metametric}_bootstrap_{dataset}.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'Wrote {out_path}', file=sys.stderr)
