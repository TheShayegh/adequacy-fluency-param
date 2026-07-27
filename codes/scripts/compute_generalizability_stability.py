"""CLI for lib.subsampling_stability.generalizability_stability: computes
ICC(C,1) (lib.icc.icc_c1) treating scorers as objects and B subsamples as
raters, under BOTH the natural (uniform-weight) condition and the
reweighted-to-target_alpha condition (same B subsamples, same seed, for a
fair paired comparison -- see draw_subsets), and reports the improvement --

  ICC(reweighted, target_alpha) - ICC(natural)

Replaces compute_subsampling_pairwise_stability.py's pairwise flip-rate
`robustness` with a proper generalizability-theory reliability coefficient
-- see lib/subsampling_stability.py's generalizability_stability docstring
and lib/icc.py's icc_c1 docstring for why.

target_alpha defaults to alpha_0(D) (the full pool's own natural balance)
but is an explicit, sweepable parameter.

Usage: python codes/scripts/compute_generalizability_stability.py
           --dataset ende23 --k-prime 8 [--b 200] [--metametric spa]
           [--target-alpha 0.75] [--seed 0]
           [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import alpha_0 as alpha0_of
from lib.subsampling_stability import generalizability_stability
from lib.consistency import real_systems
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende23')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, required=True, help="K': subsample size")
  p.add_argument('--b', type=int, default=200, help='number of subsamples')
  p.add_argument('--target-alpha', type=float, default=None,
                  help='reweight target for the "reweighted" condition (default: alpha_0(D))')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True,
                  help='passed through to real_systems (project default: wmt_official_outliers)')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  K_prime = args.k_prime
  B = args.b

  full_systems = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  outliers = ([s for s in real_systems(base, root=ROOT, exclude_outliers=False)
               if s not in set(full_systems)] if args.exclude_outliers else [])
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'b'].values
  a0_D = alpha0_of(a_D, b_D)
  target_alpha = args.target_alpha if args.target_alpha is not None else a0_D

  tag = args.tag or f'{base}_kprime{K_prime}_b{B}_{m}'
  print(f'D = {base}' + (f' minus {outliers}' if outliers else ' (full)')
        + f': K={len(full_systems)}, alpha_0(D)={a0_D:.4f}, '
        f"K'={K_prime}, B={B}, target_alpha={target_alpha:.4f}"
        + (' (default: alpha_0(D))' if args.target_alpha is None else ' (explicit)'),
        file=sys.stderr)

  t0 = time.time()
  print('\n-- natural (uniform-weight) condition --', file=sys.stderr)
  natural = generalizability_stability(base, m, K_prime, B, target_alpha=None, root=ROOT,
                                        seed=args.seed, exclude_outliers=args.exclude_outliers,
                                        progress_every=max(1, B // 10))
  print(f'  done in {time.time() - t0:.1f}s', file=sys.stderr)

  t1 = time.time()
  print(f'\n-- reweighted condition (target_alpha={target_alpha:.4f}) --', file=sys.stderr)
  reweighted = generalizability_stability(base, m, K_prime, B, target_alpha=target_alpha, root=ROOT,
                                           seed=args.seed, exclude_outliers=args.exclude_outliers,
                                           progress_every=max(1, B // 10))
  print(f'  done in {time.time() - t1:.1f}s', file=sys.stderr)

  improvement = reweighted['icc'] - natural['icc']

  lines = []
  lines.append(f'# Generalizability (ICC(C,1)) subsampling stability ({base}, metametric={m})')
  lines.append('')
  lines.append(f"K={natural['K']}" + (f' (outliers dropped: {outliers})' if outliers else '')
               + f", K'={K_prime}, B={B}, seed={args.seed}, alpha_0(D)={a0_D:.4f}, "
               f'target_alpha={target_alpha:.4f}')
  lines.append('')
  lines.append('| condition | ICC(C,1) | mean ESS | draws used | infeasible | scorers used | scorers dropped |')
  lines.append('|---|---:|---:|---:|---:|---:|---:|')
  lines.append(f"| natural | {natural['icc']:.4f} | {natural['mean_ess']:.2f} | "
               f"{natural['n_draws_used']}/{B} | {natural['n_infeasible']} | "
               f"{natural['n_scorers_used']} | {natural['n_scorers_dropped']} |")
  lines.append(f"| reweighted (a={target_alpha:.4f}) | {reweighted['icc']:.4f} | "
               f"{reweighted['mean_ess']:.2f} | {reweighted['n_draws_used']}/{B} | "
               f"{reweighted['n_infeasible']} | {reweighted['n_scorers_used']} | "
               f"{reweighted['n_scorers_dropped']} |")
  lines.append('')
  lines.append(f'**improvement (reweighted - natural) = {improvement:+.4f}**')
  if natural['scorers_dropped'] or reweighted['scorers_dropped']:
    lines.append('')
    lines.append(f"natural scorers dropped (missing on some draw): {natural['scorers_dropped']}")
    lines.append(f"reweighted scorers dropped (missing on some draw): {reweighted['scorers_dropped']}")
  table_md = '\n'.join(lines)
  print('\n' + table_md)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'generalizability_stability_{tag}.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)
