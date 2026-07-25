"""Three-variable analysis of the type-7 removal-robustness tables: does
delta_robustness (Delta_tau) depend on ESS and |delta_alpha_0| (Delta_alpha)
JOINTLY, not just on ESS alone (the earlier, two-variable analysis)?

Hypothesis under test (user-supplied): Delta_alpha and ESS are themselves
correlated (by construction, roughly -- not linear or trivial), so a simple
bivariate Delta_tau-vs-ESS correlation conflates two channels:
  (a) higher Delta_alpha -> more room for improvement -> higher Delta_tau
  (b) higher ESS -> [some instability mechanism] -> lower Delta_tau
Partial correlations (controlling each predictor for the other) and a 2D
binned-mean heatmap over the (Delta_alpha, ESS/K) plane are the tools used
to separate these two channels from the tangled simple correlation.

Data: every individual (context, d) row already saved in
artifacts/type7_level<L>-<L>_robustness_<dataset>_level<L>-<L>_spa[_reversed|
_mean].md -- reads abs(delta_alpha_0) and ESS directly from those tables
(both already computed columns), no new solving.

Outputs:
  artifacts/type7_delta_tau_heatmap_<direction>.png -- one 2D heatmap per
    direction (forward/reversed/mean), binned-mean Delta_tau over
    (Delta_alpha, ESS/K), pooled across all 6 datasets x 3 levels.
  Prints a partial-correlation table per (dataset, level) cell plus one
  pooled aggregate row, per direction, to stdout.
"""

from __future__ import annotations

import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ARTIFACTS_DIR = 'artifacts'
DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']
LEVELS = ['1-1', '2-2', '3-3']
DIRECTIONS = {'forward': '', 'reversed': '_reversed', 'mean': '_mean'}


def parse_all_rows(path):
  with open(path) as f:
    lines = f.readlines()
  m = re.search(r'K=(\d+)', lines[2])
  K = int(m.group(1))
  rows = []  # (delta_alpha, ess_over_k, delta_tau)
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or 'context' in line or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      delta_tau = float(cells[4].replace('+', ''))
      delta_alpha = float(cells[7])
      ess = float(cells[8])
      rows.append((delta_alpha, ess / K, delta_tau))
    except (ValueError, IndexError):
      pass
  return K, rows


def partial_corr(x, y, z):
  """Partial correlation of x,y controlling for z (standard 3-variable
  formula: residualize the pairwise correlations)."""
  x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
  rxy = np.corrcoef(x, y)[0, 1]
  rxz = np.corrcoef(x, z)[0, 1]
  ryz = np.corrcoef(y, z)[0, 1]
  denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
  return (rxy - rxz * ryz) / denom if denom > 0 else float('nan')


def partial_p(r, n):
  """t-test p-value for a partial correlation with one variable
  conditioned out: df = n - 3 (vs. n - 2 for a simple correlation)."""
  if n <= 3 or abs(r) >= 1:
    return float('nan')
  df = n - 3
  t = r * np.sqrt(df / (1 - r ** 2))
  return 2 * (1 - stats.t.cdf(abs(t), df=df))


if __name__ == '__main__':
  # data[direction] = list of (dataset, level, delta_alpha, ess_over_k, delta_tau)
  data = {d: [] for d in DIRECTIONS}
  for dirname, suffix in DIRECTIONS.items():
    for ds in DATASETS:
      for lvl in LEVELS:
        tag = f'{ds}_level{lvl}_spa{suffix}'
        path = f'{ARTIFACTS_DIR}/type7_level{lvl}_robustness_{tag}.md'
        K, rows = parse_all_rows(path)
        for da, ek, dt in rows:
          data[dirname].append((ds, lvl, da, ek, dt))

  # ---- correlation tables ----
  for dirname in DIRECTIONS:
    print(f'\n=== direction={dirname} ===')
    print(f"{'dataset':8} {'level':6} {'n':6} {'r(dt,ess)':>11} {'r(dt,da)':>11} "
          f"{'r(ess,da)':>11} {'pcorr(dt,ess|da)':>18} {'pcorr(dt,da|ess)':>18}")
    all_da, all_ek, all_dt = [], [], []
    for ds in DATASETS:
      for lvl in LEVELS:
        rows = [r for r in data[dirname] if r[0] == ds and r[1] == lvl]
        da = np.array([r[2] for r in rows])
        ek = np.array([r[3] for r in rows])
        dt = np.array([r[4] for r in rows])
        all_da.append(da); all_ek.append(ek); all_dt.append(dt)
        r_dt_ess = np.corrcoef(dt, ek)[0, 1]
        r_dt_da = np.corrcoef(dt, da)[0, 1]
        r_ess_da = np.corrcoef(ek, da)[0, 1]
        pc_ess = partial_corr(dt, ek, da)
        pc_da = partial_corr(dt, da, ek)
        print(f"{ds:8} {lvl:6} {len(rows):6} {r_dt_ess:+11.3f} {r_dt_da:+11.3f} "
              f"{r_ess_da:+11.3f} {pc_ess:+11.3f}(p={partial_p(pc_ess, len(rows)):8.1e}) "
              f"{pc_da:+11.3f}(p={partial_p(pc_da, len(rows)):8.1e})")
    da_all = np.concatenate(all_da); ek_all = np.concatenate(all_ek); dt_all = np.concatenate(all_dt)
    n_all = len(dt_all)
    r_dt_ess = np.corrcoef(dt_all, ek_all)[0, 1]
    r_dt_da = np.corrcoef(dt_all, da_all)[0, 1]
    r_ess_da = np.corrcoef(ek_all, da_all)[0, 1]
    pc_ess = partial_corr(dt_all, ek_all, da_all)
    pc_da = partial_corr(dt_all, da_all, ek_all)
    print(f"{'POOLED':8} {'all':6} {n_all:6} {r_dt_ess:+11.3f} {r_dt_da:+11.3f} "
          f"{r_ess_da:+11.3f} {pc_ess:+11.3f}(p={partial_p(pc_ess, n_all):8.1e}) "
          f"{pc_da:+11.3f}(p={partial_p(pc_da, n_all):8.1e})")

  # ---- heatmaps: pooled over all datasets x all levels, one per direction ----
  fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
  for ax, dirname in zip(axes, DIRECTIONS):
    rows = data[dirname]
    da = np.array([r[2] for r in rows])
    ek = np.array([r[3] for r in rows])
    dt = np.array([r[4] for r in rows])

    n_bins = 12
    da_edges = np.linspace(da.min(), da.max(), n_bins + 1)
    ek_edges = np.linspace(ek.min(), ek.max(), n_bins + 1)
    stat, _, _, _ = stats.binned_statistic_2d(da, ek, dt, statistic='mean', bins=[da_edges, ek_edges])
    counts, _, _, _ = stats.binned_statistic_2d(da, ek, dt, statistic='count', bins=[da_edges, ek_edges])

    # Mask sparse bins (too few points for a trustworthy mean) instead of
    # showing a misleadingly solid-colored cell.
    masked = np.ma.masked_where(counts.T < 3, stat.T)
    vmax = np.nanmax(np.abs(stat))
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad(color='#eeeeee')
    im = ax.pcolormesh(da_edges, ek_edges, masked, cmap=cmap, vmin=-vmax, vmax=vmax, shading='flat')
    ax.set_xlabel(r'$\Delta\alpha_0 = |\alpha_0(D)-\alpha_0(D_{-d})|$')
    ax.set_ylabel('Normalized ESS (ESS/K)')
    ax.set_title(f'{dirname}  (n={len(rows)})')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'mean $\Delta\tau$')

  fig.suptitle(r'Pooled $\Delta\tau$ over joint ($\Delta\alpha_0$, ESS/K), all datasets $\times$ all levels',
               fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.94])
  out_path = f'{ARTIFACTS_DIR}/type7_delta_tau_heatmap_joint.png'
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'\nWrote {out_path}', file=sys.stderr)
