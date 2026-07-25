"""tau_0 (natural robustness_before, no solving) version of plot_type7_alt_
target_heatmaps.py's four heatmaps, computed over the FULL A-B grid (up to
level 5, compute_type7_natural_robustness_grid.py) instead of just the
context-free L-L diagonal -- much denser coverage of the (Delta_alpha_0,
Delta_stat) plane, since staged (context>0) removals produce far more
varied (D, D') pairs than the L-L design alone.

Produces BOTH an aggregated (pooled across all 6 datasets) version and a
per-dataset-split version (one 2x3 dataset grid per heatmap type), per the
request: "once as aggregated ... and once split by datasets."

Usage: python codes/scripts/plot_type7_natural_robustness_heatmaps.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ARTIFACTS_DIR = 'artifacts'
DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']

# (x_key, y_key, x_label, y_label, out_suffix, title)
HEATMAP_SPECS = [
    ('delta_alpha0', 'delta_adequacy', r'$\Delta\alpha_0$', r'$\Delta$ mean adequacy', 'alpha_adequacy'),
    ('delta_alpha0', 'delta_fluency', r'$\Delta\alpha_0$', r'$\Delta$ mean fluency', 'alpha_fluency'),
    ('delta_alpha0', 'delta_mqm', r'$\Delta\alpha_0$', r'$\Delta$ mean mqm', 'alpha_mqm'),
    ('delta_fluency', 'delta_adequacy', r'$\Delta$ mean fluency', r'$\Delta$ mean adequacy', 'fluency_adequacy'),
]


def load_grid(dataset):
  """context/d can themselves contain comma-separated system lists (B>1 or
  A-B>1), which collide with the CSV's own delimiter and defeat naive
  comma-splitting/csv.DictReader (columns misalign). The last 5 fields of
  every row are always the 5 floats (tau_0, delta_alpha0, delta_adequacy,
  delta_fluency, delta_mqm), unambiguously, regardless of how many tokens
  the context/d fields expand into -- so grab those from the right instead
  of trying to parse context/d (not needed for this plot anyway)."""
  path = f'{ARTIFACTS_DIR}/type7_natural_robustness_grid_{dataset}_spa.csv'
  tau_0, da, dade, dflu, dmqm = [], [], [], [], []
  with open(path) as f:
    next(f)  # header
    for line in f:
      fields = line.rstrip('\n').split(',')[-5:]
      t, a, ade, flu, mqm = (float(x) for x in fields)
      tau_0.append(t); da.append(a); dade.append(ade); dflu.append(flu); dmqm.append(mqm)
  return {
      'tau_0': np.array(tau_0),
      'delta_alpha0': np.array(da),
      'delta_adequacy': np.array(dade),
      'delta_fluency': np.array(dflu),
      'delta_mqm': np.array(dmqm),
  }


def binned_mean_heatmap(ax, x, y, c, n_bins, xlabel, ylabel, title, vmin, vmax):
  x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
  y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
  stat, _, _, _ = stats.binned_statistic_2d(x, y, c, statistic='mean', bins=[x_edges, y_edges])
  counts, _, _, _ = stats.binned_statistic_2d(x, y, c, statistic='count', bins=[x_edges, y_edges])
  masked = np.ma.masked_where(counts.T < 3, stat.T)
  cmap = plt.get_cmap('viridis').copy()
  cmap.set_bad(color='#eeeeee')
  im = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=10)
  return im


if __name__ == '__main__':
  data = {ds: load_grid(ds) for ds in DATASETS}
  tau0_all = np.concatenate([data[ds]['tau_0'] for ds in DATASETS])
  vmin, vmax = float(tau0_all.min()), float(tau0_all.max())

  # --- aggregated: pooled across all 6 datasets, 4-panel figure ---
  fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
  for ax, (xk, yk, xl, yl, suf) in zip(axes, HEATMAP_SPECS):
    x = np.concatenate([data[ds][xk] for ds in DATASETS])
    y = np.concatenate([data[ds][yk] for ds in DATASETS])
    c = np.concatenate([data[ds]['tau_0'] for ds in DATASETS])
    im = binned_mean_heatmap(ax, x, y, c, n_bins=16, xlabel=xl, ylabel=yl,
                              title=f'n={len(c):,}', vmin=vmin, vmax=vmax)
  cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
  cbar.set_label(r'mean $\tau_0$ (natural robustness)')
  fig.suptitle(r'$\tau_0$ over the full A-B grid (levels 1..5, all context sizes), '
               'pooled across all 6 datasets', fontsize=13)
  out_path = f'{ARTIFACTS_DIR}/type7_natural_robustness_heatmap_aggregated.png'
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
  plt.close(fig)

  # --- per-dataset: one 2x3 grid per heatmap type ---
  for xk, yk, xl, yl, suf in HEATMAP_SPECS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, ds in zip(axes, DATASETS):
      x, y, c = data[ds][xk], data[ds][yk], data[ds]['tau_0']
      im = binned_mean_heatmap(ax, x, y, c, n_bins=14, xlabel=xl, ylabel=yl,
                                title=f'{ds}  (n={len(c):,})', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cbar.set_label(r'mean $\tau_0$')
    fig.suptitle(rf'$\tau_0$ over ({xl}, {yl}), full A-B grid (levels 1..5), split by dataset', fontsize=13)
    out_path = f'{ARTIFACTS_DIR}/type7_natural_robustness_heatmap_by_dataset_{suf}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Wrote {out_path}')
    plt.close(fig)
