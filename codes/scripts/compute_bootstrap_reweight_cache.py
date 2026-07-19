"""Solves w_r(alpha) (NUMERIC solver) for every bootstrap repeat r of ONE
dataset's systems -- action_plan.md section 6.5's literal "draw many system
subsets from the real systems only" reading, as opposed to compute_reweight_
cache.py's cross-year/cross-pair generalization of "subset". Draws the same
n_sample-without-replacement x n_repeats samples as compute_alpha_table_
bootstrap.py (identical seed/order, lib.bootstrap.sample_subsets), so this
experiment is directly about the same 30 subsamples already reported in
artifacts/alpha_table_bootstrap_<dataset>.md.

Usage: python codes/scripts/compute_bootstrap_reweight_cache.py <dataset>
                                                                  [--n-sample 10] [--n-repeats 30]
                                                                  [--seed 42] [--grid N] [--restarts N]
Writes artifacts/data/w_cache_bootstrap_<dataset>.pkl.
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import alpha_min_max, alpha_0 as alpha0_of, build_alpha_grid
from lib.bootstrap import sample_subsets
from lib.consistency import real_systems
from lib.reweight_numeric import solve_w_numeric
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('dataset')
  p.add_argument('--n-sample', type=int, default=10)
  p.add_argument('--n-repeats', type=int, default=30)
  p.add_argument('--seed', type=int, default=42)
  p.add_argument('--grid', type=int, default=21)
  p.add_argument('--restarts', type=int, default=15)
  args = p.parse_args()

  systems = real_systems(args.dataset, root=ROOT)
  samples = sample_subsets(systems, args.n_sample, args.n_repeats, args.seed)
  repeat_ids = [f'{args.dataset}_boot{i}' for i in range(1, len(samples) + 1)]
  systems_by_repeat = dict(zip(repeat_ids, samples))

  sys_df = load_system_scores(args.dataset, root=ROOT).loc[systems]
  ab_by_repeat = {
      rid: (sys_df.loc[s, 'a'].values, sys_df.loc[s, 'b'].values)
      for rid, s in systems_by_repeat.items()
  }

  los, his = zip(*(alpha_min_max(a, b) for a, b in ab_by_repeat.values()))
  lo, hi = max(los), min(his)
  alpha_0 = {rid: alpha0_of(a, b) for rid, (a, b) in ab_by_repeat.items()}
  alphas, dropped_a0 = build_alpha_grid(lo, hi, args.grid, alpha_0)
  print(f'{args.dataset}: {len(samples)} bootstrap repeats of {args.n_sample} systems each '
        f'(full set K={len(systems)}), common alpha range: [{lo:.4f}, {hi:.4f}], '
        f'grid of {args.grid} + {len(alphas) - args.grid} alpha_0 points = {len(alphas)}, '
        f'n_restarts={args.restarts}', file=sys.stderr)
  if dropped_a0:
    print(f'  ({len(dropped_a0)} alpha_0 outside the common range, not added: {dropped_a0})',
          file=sys.stderr)

  w_cache = {}
  for i, rid in enumerate(repeat_ids):
    a, b = ab_by_repeat[rid]
    t0 = time.time()
    for alpha in alphas:
      w_cache[(rid, alpha)] = solve_w_numeric(a, b, alpha, n_restarts=args.restarts, seed=args.seed)
    n_fail = sum(1 for al in alphas if not w_cache[(rid, al)].success)
    print(f'[{i + 1}/{len(repeat_ids)}] {rid}: {len(alphas)} solves in {time.time() - t0:.1f}s'
          f'{f", {n_fail} NOT feasible" if n_fail else ""}', file=sys.stderr)

  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache_bootstrap_{args.dataset}.pkl')
  os.makedirs(os.path.dirname(cache_path), exist_ok=True)
  with open(cache_path, 'wb') as f:
    pickle.dump({
        'dataset': args.dataset,
        'repeat_ids': repeat_ids,
        'systems_by_repeat': systems_by_repeat,
        'alphas': alphas,
        'w_cache': w_cache,
    }, f)
  print(f'Wrote {cache_path}', file=sys.stderr)
