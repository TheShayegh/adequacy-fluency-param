"""Per-(system, segment) scatter of one scorer's raw segment score against
Adequacy MQM or Fluency MQM, one plot per (dataset, scorer, aspect).

For a fixed dataset, both Adequacy MQM and Fluency MQM segment-level values
are highly discrete (weighted sums of a handful of errors drawn from a small
weight table -- see mwb.mqm_scoring's module docstring), so many distinct
(system, segment) cells share the exact same `a` value. Each plot overlays,
on top of the raw blue scatter, a red mean-curve: for every unique `a` value
observed, the mean of the scorer's y-values across all cells sharing it,
connected in x-order -- the empirical calibration curve of "what does this
scorer's score look like, on average, at this MQM value."

Uses lib.spa_plane.positional_af_matrices for the (a_pos, b_pos) matrices:
line-position-indexed (not mwb.mqm_scoring's own (doc,doc_id,seg_id) index),
the same index space lib.metric_scores.load_metric_seg_scores's per-scorer
segment files use, with its own alignment self-validation (see that module's
docstring) -- so this script only works on datasets where that validation
passes (excl_missing_seg_granular_mqm=True below).

Usage: python codes/scripts/plot_scorer_af_scatter.py [--dataset ende22]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.consistency import real_systems
from lib.metric_scores import load_metric_seg_scores, load_metric_sys_scores
from lib.spa_plane import positional_af_matrices

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_scorer_af_scatter')

_SCATTER_COLOR = 'blue'
_SCATTER_ALPHA = 0.3
_MEAN_COLOR = 'red'

_ASPECTS = (
    ('adequacy_mqm', 'Adequacy MQM'),
    ('fluency_mqm', 'Fluency MQM'),
)


def _mean_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Per-unique-x mean of y, sorted by x -- the (x, y) points of the red
  mean-curve, grouped by exact float equality (matches np.unique; the vast
  majority of a/b values are exact sums of the {0, 0.1, 1, 5, 25} weight
  table, so collisions across (system, segment) cells are genuine, not
  float noise)."""
  means = pd.Series(y).groupby(x).mean()
  return means.index.values, means.values


def plot_one(ax, x: np.ndarray, y: np.ndarray, aspect_label: str, title: str):
  ax.scatter(x, y, s=18, color=_SCATTER_COLOR, alpha=_SCATTER_ALPHA,
             linewidth=0, zorder=2)
  mx, my = _mean_curve(x, y)
  ax.plot(mx, my, color=_MEAN_COLOR, linewidth=1.4, zorder=4)
  ax.scatter(mx, my, s=22, color=_MEAN_COLOR, zorder=5)
  ax.set_xlabel(f'{aspect_label} (segment-level)')
  ax.set_ylabel('Scorer segment score')
  ax.set_title(title)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende22')
  args = parser.parse_args()
  dataset = args.dataset

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  systems = real_systems(dataset, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{dataset}: 0 real systems (unsupported/excluded/missing segment-granular MQM)')

  pos = positional_af_matrices(dataset, systems, root=ROOT)
  if pos is None:
    sys.exit(f'{dataset}: positional adequacy/fluency alignment could not be validated')
  a_pos, b_pos = pos
  aspect_matrices = {'adequacy_mqm': a_pos, 'fluency_mqm': b_pos}

  candidates = load_metric_sys_scores(dataset, systems, root=ROOT)
  scorer_names = sorted(candidates)
  print(f'{dataset}: K={len(systems)} systems, {len(scorer_names)} candidate scorers',
        file=sys.stderr)

  t0 = time.time()
  n_written = 0
  for i, scorer in enumerate(scorer_names, 1):
    metric_seg = load_metric_seg_scores(dataset, scorer, systems, root=ROOT)
    if metric_seg is None or metric_seg.shape != a_pos.shape:
      print(f'SKIPPED {dataset}/{scorer} (no/mismatched segment scores)', file=sys.stderr)
      continue

    for aspect_name, aspect_label in _ASPECTS:
      a_matrix = aspect_matrices[aspect_name]
      mask = ~(np.isnan(a_matrix) | np.isnan(metric_seg))
      if not mask.any():
        continue
      x, y = a_matrix[mask], metric_seg[mask]

      fig, ax = plt.subplots(figsize=(7, 5.5))
      plot_one(ax, x, y, aspect_label,
               f'{dataset} / {scorer}: score vs. {aspect_label} '
               f'(K={len(systems)}, n={x.size})')
      fig.tight_layout()

      plot_path = os.path.join(
          ARTIFACTS_DIR, f'scorer_af_scatter_{dataset}_{scorer}_{aspect_name}.png')
      fig.savefig(plot_path, dpi=150, bbox_inches='tight')
      plt.close(fig)
      n_written += 1

    elapsed = time.time() - t0
    eta = elapsed / i * (len(scorer_names) - i)
    print(f'[{i}/{len(scorer_names)}] {dataset}/{scorer} done '
          f'(elapsed {elapsed:.1f}s, eta {eta:.1f}s, {n_written} plots written)',
          file=sys.stderr)

  print(f'Done: {n_written} plots written to {ARTIFACTS_DIR}', file=sys.stderr)
