"""Within-dataset analog of plot_delta_alpha0_vs_consistency.py: instead of
pairs of different datasets in a year-cluster, takes ONE dataset and pairs
of its own SUBSETS (every subset with |S| >= K-3 systems, i.e. dropping at
most 3 systems), then checks the same thing -- does a bigger gap between
two subsets' own natural balances (delta alpha_0) predict worse natural
scorer-ranking consistency between them?

All full-K-coverage scorers cover every subset of K automatically (a
scorer missing even one system wouldn't have made the "full coverage"
list in the first place), so unlike the cross-dataset version the shared-
scorer set is fixed and doesn't shrink pair to pair -- SPA is dropped here
purely for speed (segment-level permutation tests at this many subset
pairs would dominate runtime), not because of missing coverage.

Exhaustive over all pairs below N_PAIRS_MAX; otherwise draws a random
sample of that many unique pairs instead.

Usage: python codes/scripts/plot_delta_alpha0_within_dataset.py
"""

import itertools
import os
import random
import sys
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems
from lib.metametrics import MetaEvalInput, pearson, spearman, kendall, pairwise_accuracy
from lib.metric_scores import load_metric_sys_scores
from mwb.mqm_scoring import load_system_scores
from cross_regime_sign_test import pairwise_outcomes

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATASETS = ['ende22', 'ende23', 'ende24']
MIN_SIZE_DROP = 3  # |S| >= K - MIN_SIZE_DROP
N_PAIRS_MAX = 20000
SEED = 0
METAMETRIC_FNS = {'pearson': pearson, 'spearman': spearman, 'kendall': kendall, 'pa': pairwise_accuracy}


def prepare(dataset):
  systems = real_systems(dataset, root=ROOT)
  df = load_system_scores(dataset, root=ROOT).loc[systems]
  scorer_series = load_metric_sys_scores(dataset, systems, root=ROOT)
  scorers = {name: s.reindex(systems).values for name, s in scorer_series.items()}
  return dict(K=len(systems), a=df['a'].values, b=df['b'].values, t=df['t'].values, scorers=scorers)


def ranking_for_subset(data, idx, metametric_fn, cache):
  key = (idx, metametric_fn)
  if key in cache:
    return cache[key]
  idx_arr = np.array(idx)
  human = data['t'][idx_arr]
  out = {}
  for name, vals in data['scorers'].items():
    val = metametric_fn(MetaEvalInput(human, vals[idx_arr]))
    if val == val:
      out[name] = val
  cache[key] = out
  return out


if __name__ == '__main__':
  rng = random.Random(SEED)
  fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5.5))

  for ax, dataset in zip(axes, DATASETS):
    data = prepare(dataset)
    K = data['K']
    min_size = K - MIN_SIZE_DROP
    idxs = list(range(K))
    subsets = [c for size in range(min_size, K + 1) for c in itertools.combinations(idxs, size)]
    n_subsets = len(subsets)
    total_pairs = comb(n_subsets, 2)

    if total_pairs <= N_PAIRS_MAX:
      pairs = list(itertools.combinations(range(n_subsets), 2))
      exhaustive = True
    else:
      seen = set()
      while len(seen) < N_PAIRS_MAX:
        i, j = rng.randrange(n_subsets), rng.randrange(n_subsets)
        if i != j:
          seen.add((min(i, j), max(i, j)))
      pairs = list(seen)
      exhaustive = False

    print(f'{dataset}: K={K}, |S|>={min_size}, {n_subsets} subsets, {total_pairs:,} possible pairs, '
          f'using {len(pairs)} ({"exhaustive" if exhaustive else "sampled"})', file=sys.stderr)

    alpha0_cache = {}
    ranking_cache = {}
    delta_a0s, mean_taus = [], []
    for pi, pj in pairs:
      si, sj = subsets[pi], subsets[pj]
      a0_i = alpha0_cache.setdefault(si, alpha0_of(data['a'][list(si)], data['b'][list(si)]))
      a0_j = alpha0_cache.setdefault(sj, alpha0_of(data['a'][list(sj)], data['b'][list(sj)]))
      delta_a0 = abs(a0_i - a0_j)

      taus = []
      for m, fn in METAMETRIC_FNS.items():
        r_i = ranking_for_subset(data, si, fn, ranking_cache)
        r_j = ranking_for_subset(data, sj, fn, ranking_cache)
        common = sorted(set(r_i) & set(r_j))
        outcomes = pairwise_outcomes(pd.Series(r_i)[common], pd.Series(r_j)[common])
        if outcomes:
          taus.append(float(np.mean(outcomes)))
      if not taus:
        continue
      delta_a0s.append(delta_a0)
      mean_taus.append(float(np.mean(taus)))

    r, p = stats.pearsonr(delta_a0s, mean_taus)
    ax.scatter(delta_a0s, mean_taus, s=6, alpha=0.12, color='#1f77b4', linewidths=0)
    xs = np.linspace(min(delta_a0s), max(delta_a0s), 100)
    slope, intercept = np.polyfit(delta_a0s, mean_taus, 1)
    ax.plot(xs, slope * xs + intercept, color='red', linewidth=1.5)
    n_label = f'n={len(pairs)}' + ('' if exhaustive else ' (sampled)')
    ax.set_title(f'{dataset}\nr={r:+.3f}, p={p:.2e}, {n_label}', fontsize=11)
    ax.set_xlabel(r'$\Delta\alpha_0 = |\alpha_{0,i} - \alpha_{0,j}|$')
    if ax is axes[0]:
      ax.set_ylabel('Natural consistency\n(mean over pearson/spearman/kendall/pa)')

  fig.suptitle(r'Within-dataset: $\Delta\alpha_0$ between two leave-few-out subsets '
               'vs. their natural scorer-ranking consistency\n'
               '(every subset with $|S|\\geq K-3$; SPA excluded for speed)',
               fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.88])
  out_path = os.path.join(ROOT, 'artifacts', 'delta_alpha0_within_dataset.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
