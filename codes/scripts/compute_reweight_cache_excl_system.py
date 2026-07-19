"""Like compute_reweight_cache.py, but drops one named system from EVERY
dataset in the group before solving -- e.g. re-running the 2024 cluster
(ende24/enes24/jazh24) with MSLC excluded from each, to see how much of
that cluster's behavior the single adequacy outlier was responsible for.
Datasets that don't contain the named system are used as-is (not an
error): a per-dataset skip, not a hard requirement that every dataset in
the group has to contain it.

Usage: python codes/scripts/compute_reweight_cache_excl_system.py --datasets ende24,enes24,jazh24 --exclude-system MSLC [--tag 2024_excl_MSLC]
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import alpha_min_max, alpha_0 as alpha0_of, build_alpha_grid
from lib.consistency import real_systems
from lib.reweight_numeric import solve_w_numeric
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--datasets', type=str, required=True, help='comma-separated dataset names')
  p.add_argument('--exclude-system', type=str, required=True)
  p.add_argument('--tag', type=str, default=None)
  p.add_argument('--grid', type=int, default=21)
  p.add_argument('--restarts', type=int, default=30)
  args = p.parse_args()
  datasets = [d.strip() for d in args.datasets.split(',')]
  excl = args.exclude_system
  tag = args.tag or f'{"_".join(datasets)}_excl_{excl}'
  suffix = f'_{tag}'

  systems_by_dataset = {}
  ab = {}
  for d in datasets:
    systems = real_systems(d, root=ROOT)
    if excl in systems:
      systems = [s for s in systems if s != excl]
      print(f'{d}: excluded {excl!r}, {len(systems)} systems remain', file=sys.stderr)
    else:
      print(f'{d}: {excl!r} not present, using all {len(systems)} systems unchanged', file=sys.stderr)
    systems_by_dataset[d] = systems
    df = load_system_scores(d, root=ROOT).loc[systems]
    ab[d] = (df['a'].values, df['b'].values)

  ranges = {d: alpha_min_max(*ab[d]) for d in datasets}
  alpha_0 = {d: alpha0_of(*ab[d]) for d in datasets}
  lo = max(r[0] for r in ranges.values())
  hi = min(r[1] for r in ranges.values())
  alphas, dropped_a0 = build_alpha_grid(lo, hi, args.grid, alpha_0)
  print(f'common alpha range: [{lo:.4f}, {hi:.4f}], grid of {args.grid} '
        f'+ {len(alphas) - args.grid} alpha_0 points = {len(alphas)}, '
        f'n_restarts={args.restarts}', file=sys.stderr)
  if dropped_a0:
    print(f'  ({len(dropped_a0)} alpha_0 outside the common range, not added: {dropped_a0})',
          file=sys.stderr)

  w_cache = {}
  for i, d in enumerate(datasets):
    t0 = time.time()
    a, b = ab[d]
    n_fail = 0
    for alpha in alphas:
      r = solve_w_numeric(a, b, alpha, n_restarts=args.restarts, seed=42)
      if not r.success:
        r = solve_w_numeric(a, b, alpha, n_restarts=150, seed=1)
        if not r.success:
          n_fail += 1
      w_cache[(d, alpha)] = r
    print(f'[{i + 1}/{len(datasets)}] {d}: {len(alphas)} solves in {time.time() - t0:.1f}s'
          f'{f", {n_fail} NOT feasible" if n_fail else ""}', file=sys.stderr)

  cache_path = os.path.join(DATA_DIR, f'w_cache{suffix}.pkl')
  with open(cache_path, 'wb') as f:
    pickle.dump({'datasets': datasets, 'alphas': alphas, 'w_cache': w_cache,
                 'systems_by_dataset': systems_by_dataset}, f)
  print(f'Wrote {cache_path}', file=sys.stderr)
