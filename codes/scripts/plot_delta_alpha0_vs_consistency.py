"""One-off check: within each year-cluster, is a bigger gap between two
datasets' own natural balances (delta alpha_0 = |alpha_0_i - alpha_0_j|)
associated with WORSE natural scorer-ranking consistency between them?
Reuses curve_pair_<metametric>_<d1>_<d2>.json's already-computed alpha_0
and natural_tau -- no new solving. One subplot per cluster (year), each
metametric plotted as its own point per pair (5 per pair), plus the
per-pair mean (bold) used for the reported Pearson r (avoids treating 5
highly-correlated metametric readings of the same pair as independent
samples).

Usage: python codes/scripts/plot_delta_alpha0_vs_consistency.py
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lib.metametrics import METAMETRICS_ORDER

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')

COLORS = {'pearson': '#1f77b4', 'spearman': '#ff7f0e', 'kendall': '#2ca02c',
          'pa': '#d62728', 'spa': '#9467bd'}

CLUSTERS = {
    '2020': ['ende20', 'zhen20'],
    '2021': ['ende21', 'zhen21', 'ted_ende', 'ted_zhen'],
    '2022': ['ende22', 'zhen22', 'enru22'],
    '2023': ['ende23', 'zhen23', 'heen23'],
    '2024': ['ende24', 'enes24', 'jazh24'],
}


def load_pair(metametric, d1, d2):
  for a, b in [(d1, d2), (d2, d1)]:
    path = os.path.join(DATA_DIR, f'curve_pair_{metametric}_{a}_{b}.json')
    if os.path.exists(path):
      with open(path) as f:
        return json.load(f)
  raise FileNotFoundError(f'no curve_pair file for {metametric} {d1}/{d2}')


if __name__ == '__main__':
  fig, axes = plt.subplots(1, 5, figsize=(22, 5))

  for ax, (year, datasets) in zip(axes, CLUSTERS.items()):
    pairs = list(itertools.combinations(datasets, 2))
    per_pair_means = []
    delta_a0s = []

    for d1, d2 in pairs:
      natural_taus = []
      for m in METAMETRICS_ORDER:
        d = load_pair(m, d1, d2)
        a0_1, a0_2 = d['alpha_0'][d1], d['alpha_0'][d2]
        delta_a0 = abs(a0_1 - a0_2)
        tau = d['natural_tau']
        natural_taus.append(tau)
        ax.scatter([delta_a0], [tau], color=COLORS[m], s=35, alpha=0.7,
                    label=m if (d1, d2) == pairs[0] else None, zorder=3)
      mean_tau = float(np.mean(natural_taus))
      ax.scatter([delta_a0], [mean_tau], facecolor='none', edgecolor='black',
                  s=110, linewidths=1.3, zorder=4)
      per_pair_means.append(mean_tau)
      delta_a0s.append(delta_a0)

    n = len(pairs)
    if n >= 2:
      r, p = stats.pearsonr(delta_a0s, per_pair_means)
      r_text = f'r={r:+.2f}, p={p:.2f}, n={n}'
    else:
      r_text = f'n={n} (no r)'

    ax.set_title(f'{year}\n{r_text}', fontsize=11)
    ax.set_xlabel(r'$\Delta\alpha_0 = |\alpha_{0,i} - \alpha_{0,j}|$')
    if ax is axes[0]:
      ax.set_ylabel('Natural (unreweighted) consistency')

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05),
              fontsize=10, frameon=False)
  fig.suptitle(r'$\Delta\alpha_0$ vs. natural scorer-ranking consistency, within each year-cluster'
               '\n(small dots: per-metametric; black-ringed dot: per-pair mean, used for r)',
               fontsize=13)
  fig.tight_layout(rect=[0, 0.05, 1, 0.92])
  out_path = os.path.join(ROOT, 'artifacts', 'delta_alpha0_vs_consistency.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
