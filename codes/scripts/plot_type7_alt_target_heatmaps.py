"""Four heatmaps testing whether Delta_alpha adds anything once an
alt-target's own gap is already accounted for (the follow-up to
analyze_type7_alt_target_vs_alpha.py's Check 2, in a form that doesn't
require pre-committing to a linear model): for each of the three alt
targets, mean Delta_tau_t colored over the JOINT (Delta_alpha,
Delta_stat_t) plane -- if fixing adequacy already partially fixes alpha
(the two are correlated within a dataset/level, per Check 1a), the
regression-based "R^2 gain" can understate a real but nonlinear or
threshold effect that a 2D binned-mean picture would still show.

A fourth heatmap looks at the ORIGINAL alpha-based (forward) experiment's
own Delta_tau_alpha over the joint (Delta_mean_fluency, Delta_mean_adequacy)
plane -- since alpha_0 is itself built from the adequacy and fluency
variances, this asks whether the alpha experiment's gain decomposes along
its own two raw components, without alpha as an axis at all.

Pools over all 6 datasets x all 3 levels, matching plot_type7_delta_tau_
heatmap_joint's pooling convention (analyze_type7_ess_delta_alpha_tau.py).
Rows are matched across the relevant per-target detail tables (and, for
heatmap 4, the main alpha-forward table) by (dataset, level, d).

Usage: python codes/scripts/plot_type7_alt_target_heatmaps.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ARTIFACTS_DIR = 'artifacts'
DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']
LEVELS = ['1-1', '2-2', '3-3']
TARGETS = ['adequacy', 'fluency', 'mqm']


def parse_alt_detail(path):
  """d -> (delta_tau, delta_stat, delta_alpha0)"""
  with open(path) as f:
    lines = f.readlines()
  out = {}
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or line.startswith('| context') or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      d = cells[1]
      delta_tau = float(cells[4].replace('+', ''))
      delta_stat = float(cells[5])
      delta_alpha0 = float(cells[6])
      out[d] = (delta_tau, delta_stat, delta_alpha0)
    except (ValueError, IndexError):
      pass
  return out


def parse_alpha_forward(path):
  """d -> delta_tau"""
  with open(path) as f:
    lines = f.readlines()
  out = {}
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or 'context' in line or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      d = cells[1]
      out[d] = float(cells[4].replace('+', ''))
    except (ValueError, IndexError):
      pass
  return out


def binned_mean_heatmap(ax, x, y, c, n_bins, xlabel, ylabel, title, cmap_name='RdYlGn'):
  x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
  y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
  stat, _, _, _ = stats.binned_statistic_2d(x, y, c, statistic='mean', bins=[x_edges, y_edges])
  counts, _, _, _ = stats.binned_statistic_2d(x, y, c, statistic='count', bins=[x_edges, y_edges])
  masked = np.ma.masked_where(counts.T < 3, stat.T)
  vmax = np.nanmax(np.abs(stat))
  cmap = plt.get_cmap(cmap_name).copy()
  cmap.set_bad(color='#eeeeee')
  im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, vmin=-vmax, vmax=vmax, shading='flat')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  return im


if __name__ == '__main__':
  # alt[target][(ds,lvl)] = dict d -> (delta_tau, delta_stat, delta_alpha0)
  alt = {t: {} for t in TARGETS}
  alpha_fwd = {}
  for ds in DATASETS:
    for lvl in LEVELS:
      tag = f'{ds}_level{lvl}_spa'
      alpha_fwd[(ds, lvl)] = parse_alpha_forward(f'{ARTIFACTS_DIR}/type7_level{lvl}_robustness_{tag}.md')
      for t in TARGETS:
        alt[t][(ds, lvl)] = parse_alt_detail(
            f'{ARTIFACTS_DIR}/type7_level{lvl}_alt_target_{t}_detail_{tag}.md')

  # --- heatmaps 1-3: Delta_tau_t / (Delta_alpha, Delta_stat_t) ---
  fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
  for ax, t in zip(axes, TARGETS):
    dtau, dstat, dalpha = [], [], []
    for ds in DATASETS:
      for lvl in LEVELS:
        for tau_, stat_, alpha_ in alt[t][(ds, lvl)].values():
          dtau.append(tau_); dstat.append(stat_); dalpha.append(alpha_)
    dtau, dstat, dalpha = np.array(dtau), np.array(dstat), np.array(dalpha)
    im = binned_mean_heatmap(
        ax, dalpha, dstat, dtau, n_bins=12,
        xlabel=r'$\Delta\alpha_0$', ylabel=rf'$\Delta$ mean {t}',
        title=rf'$\Delta\tau_{{\mathrm{{{t}}}}}$  (n={len(dtau)})')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'mean $\Delta\tau$')
  fig.suptitle(r'Pooled $\Delta\tau_t$ over joint ($\Delta\alpha_0$, $\Delta$ mean stat$_t$), '
               'all datasets $\\times$ all levels', fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.94])
  out1 = f'{ARTIFACTS_DIR}/type7_alt_target_heatmap_vs_alpha.png'
  fig.savefig(out1, dpi=150, bbox_inches='tight')
  print(f'Wrote {out1}')

  # --- heatmap 4: Delta_tau_alpha / (Delta_mean_fluency, Delta_mean_adequacy) ---
  dtau4, dflu4, dade4 = [], [], []
  for ds in DATASETS:
    for lvl in LEVELS:
      ade_rows = alt['adequacy'][(ds, lvl)]
      flu_rows = alt['fluency'][(ds, lvl)]
      fwd_rows = alpha_fwd[(ds, lvl)]
      common = set(ade_rows) & set(flu_rows) & set(fwd_rows)
      for d in common:
        dtau4.append(fwd_rows[d])
        dflu4.append(flu_rows[d][1])   # delta_stat from the fluency table
        dade4.append(ade_rows[d][1])   # delta_stat from the adequacy table
  dtau4, dflu4, dade4 = np.array(dtau4), np.array(dflu4), np.array(dade4)

  fig4, ax4 = plt.subplots(figsize=(7, 6))
  im4 = binned_mean_heatmap(
      ax4, dflu4, dade4, dtau4, n_bins=14,
      xlabel=r'$\Delta$ mean fluency', ylabel=r'$\Delta$ mean adequacy',
      title=rf'$\Delta\tau_{{\mathrm{{alpha}}}}$ (forward experiment)  (n={len(dtau4)})')
  cbar4 = fig4.colorbar(im4, ax=ax4, shrink=0.85)
  cbar4.set_label(r'mean $\Delta\tau_{\mathrm{alpha}}$')
  fig4.suptitle('Pooled, all datasets $\\times$ all levels', fontsize=11)
  fig4.tight_layout()
  out2 = f'{ARTIFACTS_DIR}/type7_alt_target_heatmap_fluency_vs_adequacy.png'
  fig4.savefig(out2, dpi=150, bbox_inches='tight')
  print(f'Wrote {out2}')
