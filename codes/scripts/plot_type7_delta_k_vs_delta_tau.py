"""Delta_K vs. Delta_tau bar-chart grid: for each dataset, one bar per
level A-A (dropping A systems together at once, |D'| = K-A -- the "L-L"
diagonal of compute_type7_sensitivity.py's level notation), showing the
POOLED delta_robustness (robustness_after - robustness_before, i.e. tau
after calibration minus tau before) at that level. One panel per dataset,
2023+2024 datasets by default (lib.dataset_dirs.datasets_for_years).

Reads directly from the type7_level<A>-<A>_robustness_<dataset>_level<A>-<A>_
<metametric>.md files compute_type7_sensitivity.py already writes -- does
no computing of its own. Missing levels (not yet run) are simply skipped
for that dataset's bars, and a dataset with NO levels computed yet shows
an empty "no data yet" panel instead of erroring -- safe to rerun any time
as more type7_level<A>-<A>_robustness_* files land on disk; it always
reflects whatever's currently there.

Usage: python codes/scripts/plot_type7_delta_k_vs_delta_tau.py
           [--datasets ende23,ende24,...] [--metametric spa] [--max-level 6]
--datasets (optional, comma-separated): defaults to all 2023+2024 datasets
(ende23, zhen23, heen23, ende24, enes24, jazh24).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lib.dataset_dirs import datasets_for_years

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


def load_pooled_stats(dataset: str, level: int, metametric: str, tag_suffix: str = '') -> tuple[float, float] | None:
  """(tau_before, delta) from the pooled row, or None if that level hasn't been run yet."""
  tag = f'{dataset}_level{level}-{level}_{metametric}{tag_suffix}'
  path = os.path.join(ARTIFACTS_DIR, f'type7_level{level}-{level}_robustness_{tag}.md')
  if not os.path.exists(path):
    return None
  with open(path) as f:
    for line in f:
      if line.strip().startswith('| **pooled**'):
        cells = [c.strip().replace('*', '') for c in line.strip().strip('|').split('|')]
        # cells: ['pooled', '', tau_before, tau_after, delta, '', '', '', '']
        return float(cells[2]), float(cells[4])
  return None


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--datasets', type=str, default=None,
                  help='comma-separated; defaults to all 2023+2024 datasets')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--max-level', type=int, default=6)
  p.add_argument('--tag-suffix', type=str, default='',
                  help='appended to the per-(dataset,level) tag used to find the underlying '
                       'type7_level*_robustness_*.md files (e.g. "_incl_outliers"), and to the output '
                       'filename, so this never collides with the default-tag run\'s files')
  args = p.parse_args()
  datasets = [d.strip() for d in args.datasets.split(',')] if args.datasets else datasets_for_years([2023, 2024])
  m = args.metametric
  tag_suffix = args.tag_suffix
  levels = list(range(1, args.max_level + 1))

  fig, axes = plt.subplots(2, 3, figsize=(15, 8))
  axes = axes.flatten()

  panel_data = []  # (ax, dataset, levels_have)  -- have: list of (L, tau_before, delta)
  all_deltas = []
  all_values = []  # deltas + room heights, for the positive extent
  for ax, ds in zip(axes, datasets):
    stats = [load_pooled_stats(ds, L, m, tag_suffix) for L in levels]
    have = [(L, tb, d) for L, s in zip(levels, stats) if s is not None for tb, d in [s]]
    panel_data.append((ax, ds, have))
    all_deltas.extend(d for _, _, d in have)
    all_values.extend(d for _, _, d in have)
    all_values.extend(1 - tb for _, tb, _ in have)  # room-for-improvement bar heights
    print(f'{ds}: {len(have)}/{len(levels)} levels available '
          + (f'({[L for L, _, _ in have]})' if have else '(none yet)'), file=sys.stderr)

  # No forced symmetry around 0: the negative half is dead space whenever
  # every delta is >=0 (as it is here), so the axis only extends below 0 as
  # far as the data actually goes.
  ymin = min(0.0, min(all_deltas, default=0.0)) * 1.15
  ymax = max(all_values, default=1.0) * 1.15

  for ax, ds, have in panel_data:
    if not have:
      ax.text(0.5, 0.5, 'no data yet', ha='center', va='center', transform=ax.transAxes,
              color='gray', fontsize=11)
      ax.set_title(ds)
      ax.set_xticks(levels)
      ax.set_ylim(ymin, ymax)
      continue
    Ls, tbs, vals = zip(*have)
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vals]
    room = [1 - tb for tb in tbs]
    # Room for improvement (1 - pooled tau_before, the most delta_tau could
    # ever be since tau tops out at 1): a light, low-alpha bar drawn BEHIND
    # each level's actual bar, at the same x, so the actual gain reads
    # visually against its own ceiling.
    ax.bar(Ls, room, color='#2ca02c', alpha=0.2, width=0.7, zorder=1)
    ax.bar(Ls, vals, color=colors, width=0.7, zorder=2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(levels)
    ax.set_ylim(ymin, ymax)
    ax.set_title(ds)

  for ax in axes[len(datasets):]:
    ax.axis('off')

  for i, (ax, ds, have) in enumerate(panel_data):
    ax.set_xlabel(r'$\Delta K$ (level $L$-$L$)')
    if i % 3 == 0:
      ax.set_ylabel(r'pooled $\Delta\tau$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  room_proxy = Patch(facecolor='#2ca02c', alpha=0.2, label=r'room for improvement ($1-\tau_{\mathrm{orig}}$)')
  fig.legend(handles=[room_proxy], loc='upper center', bbox_to_anchor=(0.5, 1.0),
             ncol=1, frameon=False, fontsize=9)

  fig.suptitle(r'Pooled $\Delta\tau$ (robustness after $-$ before) vs. $\Delta K$'
               f' (systems dropped together), metametric={m}{tag_suffix}', fontsize=13, y=1.06)
  fig.tight_layout(rect=[0, 0, 1, 0.94])

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type7_delta_k_vs_delta_tau_{m}{tag_suffix}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
