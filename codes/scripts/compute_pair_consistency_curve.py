"""Third kind of consistency analysis, restricted to a single pair of
datasets (rather than the whole-pool of compute_consistency_curve.py, or
the star-vs-the-rest of compute_per_dataset_consistency_curve.py): for two
named datasets, sweeps alpha over their OWN intersected reachable range
(action_plan.md 3.8's Corollary 2 range for each, intersected -- not
whichever wider/narrower shared range some other group of datasets happened
to produce), and at each alpha computes the single (not pooled, not
averaged) pairwise tau-a between the two datasets' scorer-rankings under
w(alpha). The natural (unweighted) version of that same single pairwise tau
is also computed once, as a fixed reference value (not swept) -- plotted as
a horizontal line, not a dot, since with only one pair there's no "this
dataset vs the rest" framing left to mark a landing point for.

Usage: python codes/scripts/compute_pair_consistency_curve.py DATASET1 DATASET2 [--grid N] [--restarts N]
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from lib.alpha import alpha_min_max, alpha_0 as alpha0_of, build_alpha_grid
from lib.consistency import real_systems, load_scorer_inputs, scorer_scores
from lib.metametrics import METAMETRICS, METAMETRICS_ORDER, NEEDS_SEGMENT_SCORES
from lib.reweight_numeric import solve_w_numeric
from lib.reweighted_consistency import pairwise_outcomes, rankings_at, spa_pvalue_cache
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('dataset1')
  p.add_argument('dataset2')
  p.add_argument('--grid', type=int, default=21)
  p.add_argument('--restarts', type=int, default=30)
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()
  d1, d2 = args.dataset1, args.dataset2
  datasets = [d1, d2]

  ab = {}
  K = {}
  for d in datasets:
    systems = real_systems(d, root=ROOT)
    df = load_system_scores(d, root=ROOT).loc[systems]
    ab[d] = (df['a'].values, df['b'].values)
    K[d] = len(systems)

  ranges = {d: alpha_min_max(*ab[d]) for d in datasets}
  alpha_0 = {d: alpha0_of(*ab[d]) for d in datasets}
  lo = max(r[0] for r in ranges.values())
  hi = min(r[1] for r in ranges.values())
  alphas, dropped_a0 = build_alpha_grid(lo, hi, args.grid, alpha_0)
  print(f'{d1} range: {ranges[d1]}', file=sys.stderr)
  print(f'{d2} range: {ranges[d2]}', file=sys.stderr)
  print(f'intersected range: [{lo:.4f}, {hi:.4f}], grid of {args.grid} '
        f'+ {len(alphas) - args.grid} alpha_0 points = {len(alphas)}', file=sys.stderr)
  if dropped_a0:
    print(f'  ({len(dropped_a0)} alpha_0 outside the intersected range, not added: {dropped_a0})',
          file=sys.stderr)

  w_cache = {}
  for d in datasets:
    a, b = ab[d]
    for alpha in alphas:
      r = solve_w_numeric(a, b, alpha, n_restarts=args.restarts, seed=args.seed)
      if not r.success:
        r = solve_w_numeric(a, b, alpha, n_restarts=150, seed=args.seed + 1)
      w_cache[(d, alpha)] = r
  n_fail = sum(1 for r in w_cache.values() if not r.success)
  print(f'solved {len(w_cache)} points, {n_fail} still infeasible after retry', file=sys.stderr)

  spa_cache = {d: spa_pvalue_cache(d, root=ROOT) for d in datasets}

  for m in METAMETRICS_ORDER:
    inputs_cache = ({d: load_scorer_inputs(d, m, root=ROOT) for d in datasets}
                     if m not in NEEDS_SEGMENT_SCORES else {})

    natural_rankings = {d: scorer_scores(d, m, root=ROOT) for d in datasets}
    natural_outcomes = pairwise_outcomes(natural_rankings[d1], natural_rankings[d2])
    natural_tau = float(np.mean(natural_outcomes)) if natural_outcomes else float('nan')

    tau_curve = []
    ess_over_k_curve = []
    for alpha in alphas:
      rankings = rankings_at(m, datasets, alpha, w_cache, inputs_cache, spa_cache)
      outcomes = pairwise_outcomes(rankings[d1], rankings[d2])
      tau_curve.append(float(np.mean(outcomes)) if outcomes else float('nan'))
      ess_over_k_curve.append(float(np.mean([
          w_cache[(d1, alpha)].ess / K[d1], w_cache[(d2, alpha)].ess / K[d2],
      ])))

    out = {
        'metametric': m,
        'dataset1': d1, 'dataset2': d2,
        'K': K,
        'alpha_0': alpha_0,
        'alpha_range': {'lo': lo, 'hi': hi},
        'alphas': list(alphas),
        'tau': tau_curve,
        'ess_over_k': ess_over_k_curve,
        'natural_tau': natural_tau,
        'natural_n_scorer_pairs': len(natural_outcomes),
    }
    out_path = os.path.join(OUT_DIR, f'curve_pair_{m}_{d1}_{d2}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{m:10} natural_tau={natural_tau:+.4f} n_shared_scorer_pairs={len(natural_outcomes)} '
          f'-> {os.path.basename(out_path)}', file=sys.stderr)
