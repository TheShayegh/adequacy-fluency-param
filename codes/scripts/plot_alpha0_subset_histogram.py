"""Histogram of alpha_0 over every size-k subset of one dataset's real
systems -- action_plan.md 3.2's natural (uniform-weight) balance, computed
exhaustively (not sampled) for every C(K,k) subset via lib.alpha_table.
alpha_0_values_for_subset_size. Companion to compute_alpha_table.py's
per-k mean/std summary (across every k); this shows the full distribution
at one fixed k.

Usage: python codes/scripts/plot_alpha0_subset_histogram.py [dataset] [k]
(defaults to ende24, k=10)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.alpha import alpha_0 as alpha0
from lib.alpha_table import alpha_0_values_for_subset_size
from lib.consistency import real_systems
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
STANDARD_BLUE = (0.121, 0.466, 0.705)  # matplotlib "tab:blue", matches plot_consistency_curves.py


if __name__ == '__main__':
  dataset = sys.argv[1] if len(sys.argv) > 1 else 'ende24'
  k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

  systems = real_systems(dataset, root=ROOT)
  sys_df = load_system_scores(dataset, root=ROOT).loc[systems]
  a, b = sys_df['a'].values, sys_df['b'].values
  K = len(a)
  if not (2 <= k < K):
    raise ValueError(f'k={k} must be in [2, {K - 1}] for {dataset} (K={K})')

  alphas = alpha_0_values_for_subset_size(a, b, k)
  full_alpha0 = alpha0(a, b)
  print(f'{dataset}: K={K}, k={k}, {len(alphas):,} subsets (C({K},{k}))', file=sys.stderr)
  print(f'  alpha_0 over subsets: mean={alphas.mean():.4f}  std={alphas.std():.4f}  '
        f'min={alphas.min():.4f}  max={alphas.max():.4f}', file=sys.stderr)
  print(f'  full-K (K={K}) alpha_0 = {full_alpha0:.4f}', file=sys.stderr)

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.hist(alphas, bins=60, color=STANDARD_BLUE, edgecolor='white', linewidth=0.3)
  ax.axvline(full_alpha0, color='red', linestyle='--', linewidth=1.5,
             label=f'full K={K} alpha_0 = {full_alpha0:.4f}')
  ax.set_xlabel(r'alpha_0 (natural, uniform-weight balance) of the size-$k$ subset')
  ax.set_ylabel(f'count (of {len(alphas):,} size-{k} subsets)')
  ax.set_title(f'{dataset}: alpha_0 over all $C({K},{k})$ = {len(alphas):,} size-{k} subsets')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(frameon=False)

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  out_path = os.path.join(ROOT, 'artifacts', f'alpha0_subset_histogram_{dataset}_k{k}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
