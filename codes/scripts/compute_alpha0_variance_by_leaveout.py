"""Var(alpha_0) over exhaustive leave-N-systems-out subsets of each dataset,
N=1..4 (loo/l2o/l3o/l4o) -- 15 datasets x 4 levels = 60 variances -- computed
both with the full real-system roster and with outliers screened out first.

Complements compute_alpha_table.py's own per-k mean/std section (which
already sweeps every subset size k in [2, K-1]) by pulling out just the four
leave-N-out sizes (k = K-N) across every dataset into one compact table,
and reporting variance directly rather than std. Uses
lib.alpha_table.alpha_0_values_for_subset_size (exhaustive, not sampled)
directly at each of the four targeted sizes, rather than
alpha_0_by_subset_size's full k=2..K-1 sweep, since only these four sizes
are wanted here.

Outlier screen: the same default used throughout the type7 scripts
(compute_type7_sensitivity.py, compute_type7_natural_robustness_grid.py,
compute_type7_alt_target_ablation.py) and action_plan.md 6.5's own
"outlier screen" -- lib.outlier_detection.iterative_single_aspect_outliers on the
adequacy scores only (repeated modified-z-score passes, |z|>3.5, until a
pass flags nothing), with every flagged system dropped from the pool
entirely before anything else is computed. Datasets with zero flagged
outliers have identical with/without rows (nothing to drop).

Usage: python codes/scripts/compute_alpha0_variance_by_leaveout.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from lib.alpha import alpha_0 as alpha0_of
from lib.alpha_table import alpha_0_values_for_subset_size
from lib.consistency import real_systems
from lib.outlier_detection import DEFAULT_THRESHOLD, iterative_single_aspect_outliers
from mwb.mqm_scoring import SETS, load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATASETS = list(SETS.keys())  # all 15 well-formed sets (generalMT2022/enzh excluded, see mqm_scoring.py)
DROP_LEVELS = [1, 2, 3, 4]
LEVEL_NAME = {1: 'loo', 2: 'l2o', 3: 'l3o', 4: 'l4o'}


def variance_row(a: np.ndarray, b: np.ndarray) -> dict:
  """K, alpha_0_full, and var_<level> for every leave-N-out level whose
  remaining size (K-N) is still >= 2; NaN for a level that would need
  fewer than 2 systems (only possible for a badly outlier-shrunk pool)."""
  K = len(a)
  row = {'K': K, 'alpha_0_full': alpha0_of(a, b)}
  for drop in DROP_LEVELS:
    k = K - drop
    if k < 2:
      row[f'var_{LEVEL_NAME[drop]}'] = float('nan')
      continue
    alphas = alpha_0_values_for_subset_size(a, b, k)
    row[f'var_{LEVEL_NAME[drop]}'] = float(np.var(alphas))
  return row


if __name__ == '__main__':
  rows_with, rows_without = [], []
  for name in DATASETS:
    systems = real_systems(name, root=ROOT)
    if not systems:
      # real_systems() now applies lib.outlier_detection.wmt_official_outliers
      # by default, which treats some datasets (e.g. ende20/zhen20) as
      # UNSUPPORTED and drops their entire roster -- skip rather than crash
      # on alpha_0's 1/K (ZeroDivisionError at K=0).
      print(f'{name}: SKIPPED (0 systems -- unsupported under wmt_official_outliers)', file=sys.stderr)
      continue
    sys_df = load_system_scores(name, root=ROOT).loc[systems]
    a, b = sys_df['a'].values, sys_df['b'].values

    row_with = {'dataset': name, **variance_row(a, b)}
    rows_with.append(row_with)

    outliers = [s for s, _z, _it in iterative_single_aspect_outliers(systems, a)]
    keep_idx = [i for i, s in enumerate(systems) if s not in set(outliers)]
    a_no, b_no = a[keep_idx], b[keep_idx]
    row_without = {'dataset': name, 'n_outliers': len(outliers), 'outliers': ', '.join(outliers),
                   **variance_row(a_no, b_no)}
    rows_without.append(row_without)

    print(f'{name}: K={row_with["K"]}, outliers={outliers or "(none)"}, K_noout={row_without["K"]}',
          file=sys.stderr)

  out_with = pd.DataFrame(rows_with).set_index('dataset')
  out_without = pd.DataFrame(rows_without).set_index('dataset')
  print('\nWith outliers:\n' + out_with.round(6).to_string())
  print('\nWithout outliers:\n' + out_without.round(6).to_string())

  out_dir = os.path.join(ROOT, 'artifacts', 'inventory_alpha_table')
  os.makedirs(out_dir, exist_ok=True)
  md_path = os.path.join(out_dir, 'alpha0_variance_by_leaveout.md')
  with open(md_path, 'w') as f:
    f.write('# Var(alpha_0) over leave-N-systems-out subsets, N=1..4\n\n')
    f.write(
        'For each of the 15 well-formed datasets, the natural balance alpha_0 '
        '(action_plan.md 3.2, uniform-weight var_a/(var_a+var_b)) is recomputed '
        'on every exhaustive size-(K-N) subset of that dataset\'s own real systems '
        '(N=1: leave-one-out/"loo", N=2: "l2o", N=3: "l3o", N=4: "l4o" -- C(K,N) '
        'subsets each, all enumerated exhaustively, not sampled -- '
        'lib.alpha_table.alpha_0_values_for_subset_size), and the population '
        'variance (ddof=0) of that distribution is reported. 15 datasets x 4 '
        'levels = 60 variances per table below. alpha_0_full is the dataset\'s '
        'own full-roster alpha_0 (K = all systems in that table\'s pool, N=0), '
        'included for reference.\n\n'
        'Reported twice: once over each dataset\'s full real-system roster '
        '("with outliers"), and once after screening outliers out first '
        '("without outliers") -- the same iterative modified-z-score screen '
        f'(Iglewicz & Hoya 1993, |z| > {DEFAULT_THRESHOLD}, adequacy scores only, repeated '
        'until a pass flags nothing) used by default throughout the type7 '
        'consistency scripts (compute_type7_sensitivity.py, '
        'compute_type7_natural_robustness_grid.py, '
        'compute_type7_alt_target_ablation.py) and action_plan.md 6.5\'s own '
        'outlier screen -- lib.outlier_detection.iterative_single_aspect_outliers. A '
        'dataset with zero flagged outliers has identical rows in both tables.\n\n'
    )

    f.write('## With outliers (full real-system roster)\n\n')
    header = ['dataset', 'K', 'alpha_0_full'] + [f'var_{LEVEL_NAME[d]}' for d in DROP_LEVELS]
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name, r in out_with.iterrows():
      cells = [name, str(int(r['K'])), f"{r['alpha_0_full']:.4f}"]
      cells += [f"{r[f'var_{LEVEL_NAME[d]}']:.6f}" for d in DROP_LEVELS]
      f.write('| ' + ' | '.join(cells) + ' |\n')

    f.write('\n## Without outliers (iterative MAD screen on adequacy, |z| > '
             f'{DEFAULT_THRESHOLD})\n\n')
    header = ['dataset', 'K', 'n_outliers', 'outliers dropped', 'alpha_0_full'] + \
        [f'var_{LEVEL_NAME[d]}' for d in DROP_LEVELS]
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name, r in out_without.iterrows():
      cells = [name, str(int(r['K'])), str(int(r['n_outliers'])), r['outliers'] or '(none)',
                f"{r['alpha_0_full']:.4f}"]
      cells += [f"{r[f'var_{LEVEL_NAME[d]}']:.6f}" if r[f'var_{LEVEL_NAME[d]}'] == r[f'var_{LEVEL_NAME[d]}']
                else 'NaN' for d in DROP_LEVELS]
      f.write('| ' + ' | '.join(cells) + ' |\n')

  print(f'\nWrote {md_path}')