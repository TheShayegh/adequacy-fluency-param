"""Delta_K vs. Delta_tau bar-chart grid for the alt-target ablation
(compute_type7_alt_target_ablation.py) -- same layout and conventions as
plot_type7_delta_k_vs_delta_tau.py (room-for-improvement bar behind each
level's actual bar, non-symmetric y-axis, one panel per dataset), but
reading pooled tau_before/delta for a chosen alt-target (adequacy /
fluency / mqm) instead of the alpha-based forward result.

Reads directly from the type7_level<L>-<L>_alt_target_ablation_<dataset>_
level<L>-<L>_<metametric>.md files -- does no computing of its own.

Usage: python codes/scripts/plot_type7_alt_target_delta_k.py
           [--datasets ende23,ende24,...] [--metametric spa] [--max-level 3]
           [--targets adequacy,fluency,mqm]
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

ALL_TARGETS = ['adequacy', 'fluency', 'mqm']


def load_target_stats(dataset: str, level: int, metametric: str, target: str) -> tuple[float, float] | None:
  """(tau_before, delta) for `target`'s row, or None if that level hasn't been run yet."""
  tag = f'{dataset}_level{level}-{level}_{metametric}'
  path = os.path.join(ARTIFACTS_DIR, f'type7_level{level}-{level}_alt_target_ablation_{tag}.md')
  if not os.path.exists(path):
    return None
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line.startswith(f'| {target} '):
        cells = [c.strip() for c in line.strip('|').split('|')]
        # cells: [target, tau_before, tau_after, delta, infeasible]
        return float(cells[1]), float(cells[3].replace('+', ''))
  return None


def make_plot(target: str, datasets: list[str], m: str, max_level: int) -> None:
  levels = list(range(1, max_level + 1))

  fig, axes = plt.subplots(2, 3, figsize=(15, 8))
  axes = axes.flatten()

  panel_data = []
  all_deltas = []
  all_values = []
  for ax, ds in zip(axes, datasets):
    stats = [load_target_stats(ds, L, m, target) for L in levels]
    have = [(L, tb, d) for L, s in zip(levels, stats) if s is not None for tb, d in [s]]
    panel_data.append((ax, ds, have))
    all_deltas.extend(d for _, _, d in have)
    all_values.extend(d for _, _, d in have)
    all_values.extend(1 - tb for _, tb, _ in have)
    print(f'{ds}: {len(have)}/{len(levels)} levels available '
          + (f'({[L for L, _, _ in have]})' if have else '(none yet)'), file=sys.stderr)

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

  target_label = {'adequacy': 'mean adequacy', 'fluency': 'mean fluency', 'mqm': 'mean all-MQM'}[target]
  fig.suptitle(rf'Pooled $\Delta\tau$ (D reweighted to D\'s own {target_label}, alpha-irrelevant) vs. $\Delta K$'
               f', metametric={m}', fontsize=13, y=1.06)
  fig.tight_layout(rect=[0, 0, 1, 0.94])

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type7_delta_k_vs_delta_tau_{m}_alt_{target}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--datasets', type=str, default=None,
                  help='comma-separated; defaults to all 2023+2024 datasets')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--max-level', type=int, default=3)
  p.add_argument('--targets', type=str, default=None,
                  help='comma-separated subset of adequacy,fluency,mqm; defaults to all three')
  args = p.parse_args()
  datasets = [d.strip() for d in args.datasets.split(',')] if args.datasets else datasets_for_years([2023, 2024])
  targets = [t.strip() for t in args.targets.split(',')] if args.targets else ALL_TARGETS

  for target in targets:
    make_plot(target, datasets, args.metametric, args.max_level)
