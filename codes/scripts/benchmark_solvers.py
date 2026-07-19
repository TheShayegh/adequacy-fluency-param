"""One-off runtime comparison between lib.reweight_numeric.solve_w_numeric
(local multi-start search, no optimality guarantee) and lib.reweight_exact.
solve_w_exact (support enumeration, sound+complete) across datasets of
increasing K, at a grid of alpha targets spanning each one's own reachable
range.

Usage: python codes/scripts/benchmark_solvers.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_min_max
from lib.consistency import real_systems
from lib.reweight_exact import solve_w_exact
from lib.reweight_numeric import solve_w_numeric
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
NUMERIC_RESTARTS = 15  # matches compute_reweight_cache.py's own default


def bench(dataset, n_grid):
  systems = real_systems(dataset, root=ROOT)
  df = load_system_scores(dataset, root=ROOT).loc[systems]
  a, b = df['a'].values, df['b'].values
  K = len(a)
  lo, hi = alpha_min_max(a, b)
  eps = (hi - lo) * 1e-3
  grid = list(np.linspace(lo + eps, hi - eps, n_grid))

  print(f'\n=== {dataset} (K={K}, range=[{lo:.4f},{hi:.4f}]) ===', file=sys.stderr)
  print(f'{"alpha":>8} {"numeric(s)":>11} {"exact(s)":>10} {"exact/numeric":>14} '
        f'{"exact_certified":>16} {"exact_supports":>15}', file=sys.stderr)

  num_times, exact_times = [], []
  for alpha in grid:
    t0 = time.time()
    rn = solve_w_numeric(a, b, alpha, n_restarts=NUMERIC_RESTARTS, seed=42)
    t_num = time.time() - t0

    t0 = time.time()
    re = solve_w_exact(a, b, alpha)
    t_exact = time.time() - t0

    num_times.append(t_num)
    exact_times.append(t_exact)
    ratio = t_exact / t_num if t_num > 0 else float('inf')
    print(f'{alpha:8.4f} {t_num:11.3f} {t_exact:10.3f} {ratio:14.1f}x '
          f'{str(re.certified):>16} {re.n_supports_visited:15d}', file=sys.stderr)

  print(f'  TOTAL: numeric={sum(num_times):.2f}s  exact={sum(exact_times):.2f}s  '
        f'mean_numeric={np.mean(num_times):.3f}s  mean_exact={np.mean(exact_times):.3f}s  '
        f'max_exact={max(exact_times):.3f}s', file=sys.stderr)
  return K, sum(num_times), sum(exact_times), np.mean(num_times), np.mean(exact_times), max(exact_times)


if __name__ == '__main__':
  results = []
  results.append(('ende20', *bench('ende20', 11)))
  results.append(('heen23', *bench('heen23', 11)))
  results.append(('ende24', *bench('ende24', 9)))

  print('\n=== Summary ===', file=sys.stderr)
  print(f'{"dataset":10} {"K":>4} {"total_numeric":>14} {"total_exact":>12} '
        f'{"mean_numeric":>13} {"mean_exact":>11} {"max_exact":>10}', file=sys.stderr)
  for name, K, tn, te, mn, me, mx in results:
    print(f'{name:10} {K:4d} {tn:14.2f} {te:12.2f} {mn:13.4f} {me:11.4f} {mx:10.3f}', file=sys.stderr)
