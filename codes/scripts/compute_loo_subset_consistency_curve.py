"""Type-2 (star-pooled tau_bar) consistency analysis applied within a
SINGLE dataset's own systems, via leave-one-system-out: for a dataset with
K real systems, builds K pseudo-datasets (repeat_ids), each = all systems
but one, then runs the exact same star-pooled machinery as
compute_per_dataset_consistency_curve.py (tau_bar(repeat, alpha) = pooled
tau between repeat's ranking and every OTHER repeat's ranking) treating the
K leave-one-out subsets as if they were K different "datasets". Writes
output in the identical curve_perdataset_<metametric>[_TAG].json schema, so
the existing plot_per_dataset_consistency_curves.py renders it unchanged.

Usage: python codes/scripts/compute_loo_subset_consistency_curve.py [--dataset ende24] [--tag loo_ende24]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lib.alpha import alpha_min_max, alpha_0 as alpha0_of, build_alpha_grid
from lib.consistency import real_systems, load_scorer_inputs, scorer_scores
from lib.metametrics import METAMETRICS, METAMETRICS_ORDER, NEEDS_SEGMENT_SCORES
from lib.reweight_exact import solve_w_exact
from lib.reweight_numeric import solve_w_numeric
from lib.reweighted_consistency import pooled, rankings_at, spa_pvalue_cache
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende24')
  p.add_argument('--tag', type=str, default=None, help='defaults to loo_<dataset>[_excl_<system>]')
  p.add_argument('--exclude-system', type=str, default=None,
                  help='drop this one system from the base pool before taking leave-one-out '
                       'subsets of what remains (e.g. a known alpha_0 outlier)')
  p.add_argument('--grid', type=int, default=21)
  p.add_argument('--restarts', type=int, default=30)
  p.add_argument('--solver', choices=['numeric', 'exact'], default='numeric',
                  help='numeric: local multi-start search (default). '
                       'exact: support-enumeration solver, sound+complete, no restarts needed.')
  args = p.parse_args()
  base = args.dataset
  tag = args.tag or (f'loo_{base}_excl_{args.exclude_system}' if args.exclude_system else f'loo_{base}')
  suffix = f'_{tag}'

  full_systems = real_systems(base, root=ROOT)
  if args.exclude_system:
    if args.exclude_system not in full_systems:
      raise ValueError(f'{args.exclude_system!r} not in {base}\'s systems: {full_systems}')
    full_systems = [s for s in full_systems if s != args.exclude_system]
    print(f'excluding {args.exclude_system!r} from the base pool first -> {len(full_systems)} systems remain',
          file=sys.stderr)
  df_all = load_system_scores(base, root=ROOT)
  repeat_ids = [f'excl_{s}' for s in full_systems]
  systems_by_repeat = {f'excl_{s}': [x for x in full_systems if x != s] for s in full_systems}
  print(f'{base}: K={len(full_systems)} -> {len(repeat_ids)} leave-one-system-out subsets '
        f'(each K-1={len(full_systems) - 1})', file=sys.stderr)

  ab = {}
  K = {}
  alpha_0 = {}
  for rid in repeat_ids:
    systems = systems_by_repeat[rid]
    sub = df_all.loc[systems]
    ab[rid] = (sub['a'].values, sub['b'].values)
    K[rid] = len(systems)
    alpha_0[rid] = alpha0_of(*ab[rid])

  ranges = {rid: alpha_min_max(*ab[rid]) for rid in repeat_ids}
  lo = max(r[0] for r in ranges.values())
  hi = min(r[1] for r in ranges.values())
  alphas, dropped_a0 = build_alpha_grid(lo, hi, args.grid, alpha_0)
  print(f'common alpha range across all {len(repeat_ids)} subsets: [{lo:.4f}, {hi:.4f}], '
        f'grid of {args.grid} + {len(alphas) - args.grid} alpha_0 points = {len(alphas)}',
        file=sys.stderr)
  if dropped_a0:
    print(f'  ({len(dropped_a0)} alpha_0 outside the common range, not added: {dropped_a0})',
          file=sys.stderr)

  print(f'solving w(alpha) for every subset (solver={args.solver})...', file=sys.stderr)
  w_cache = {}
  n_fail = 0
  for rid in repeat_ids:
    a, b = ab[rid]
    for alpha in alphas:
      if args.solver == 'exact':
        r = solve_w_exact(a, b, alpha)
        if not r.success:
          n_fail += 1
      else:
        r = solve_w_numeric(a, b, alpha, n_restarts=args.restarts, seed=42)
        if not r.success:
          r = solve_w_numeric(a, b, alpha, n_restarts=150, seed=1)
          if not r.success:
            n_fail += 1
      w_cache[(rid, alpha)] = r
  print(f'solved {len(w_cache)} points, {n_fail} still infeasible after retry', file=sys.stderr)

  # Persisted in the same schema compute_reweight_cache.py uses, so
  # compute_consistency_curve.py (the type-1 pooled analysis) can reuse
  # these same solved weights via --tag, exactly like a normal dataset
  # group -- solving is the expensive step and otherwise wouldn't survive
  # past this process.
  import pickle
  w_cache_path = os.path.join(OUT_DIR, f'w_cache{suffix}.pkl')
  with open(w_cache_path, 'wb') as f:
    pickle.dump({'datasets': repeat_ids, 'alphas': alphas, 'w_cache': w_cache}, f)
  print(f'wrote {w_cache_path}', file=sys.stderr)

  spa_cache = {rid: spa_pvalue_cache(base, systems_by_repeat[rid], root=ROOT) for rid in repeat_ids}

  for m in METAMETRICS_ORDER:
    inputs_cache = ({rid: load_scorer_inputs(base, m, root=ROOT, systems=systems_by_repeat[rid])
                      for rid in repeat_ids} if m not in NEEDS_SEGMENT_SCORES else {})
    rankings_by_alpha = {a: rankings_at(m, repeat_ids, a, w_cache, inputs_cache, spa_cache) for a in alphas}
    natural_rankings = {rid: scorer_scores(base, m, root=ROOT, systems=systems_by_repeat[rid])
                          for rid in repeat_ids}

    tau_bar_by_dataset = {}
    ess_by_dataset = {}
    baseline_tau_by_dataset = {}
    for rid in repeat_ids:
      others = [r for r in repeat_ids if r != rid]
      tau_bar_by_dataset[rid] = [
          pooled([(rid, o) for o in others], rankings_by_alpha[a], rankings_by_alpha[a])[0]
          for a in alphas
      ]
      ess_by_dataset[rid] = [w_cache[(rid, a)].ess for a in alphas]
      baseline_tau_by_dataset[rid] = pooled([(rid, o) for o in others],
                                              natural_rankings, natural_rankings)[0]

    out = {
        'metametric': m,
        'datasets': repeat_ids,
        'K': K,
        'alpha_0': alpha_0,
        'alphas': list(alphas),
        'tau_bar_by_dataset': tau_bar_by_dataset,
        'ess_by_dataset': ess_by_dataset,
        'baseline_tau_by_dataset': baseline_tau_by_dataset,
    }
    out_path = os.path.join(OUT_DIR, f'curve_perdataset_{m}{suffix}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{m:10} -> {os.path.basename(out_path)}', file=sys.stderr)

  print(f'\nRun: python codes/scripts/plot_per_dataset_consistency_curves.py --tag {tag}', file=sys.stderr)
