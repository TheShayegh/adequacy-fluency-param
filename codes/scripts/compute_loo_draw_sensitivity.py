"""Leave-one-draw-out sensitivity: which of the B subsampling draws is
responsible for the reweighted-vs-natural ICC(C,1) gap?

Scores every draw ONCE, under both the natural and reweighted (target_alpha,
default alpha_0(D)) conditions, in parallel (lib.subsampling_stability.
_score_one_draw via a single shared ProcessPoolExecutor -- see
paired_stability_icc's docstring for why this dispatch shape lets both
conditions run concurrently). Caches every draw's score dict, computes the
full-B baseline ICC for both conditions, then for each draw individually
recomputes both conditions' ICC with just that ONE draw excluded (cheap:
icc_c1 is pure vectorized numpy, no rescoring needed -- only the matrix
assembly changes).

Columns:
  ICC natural (LOO-out)     -- natural ICC with this draw excluded
  ICC reweighted (LOO-out)  -- reweighted ICC with this draw excluded
  Delta ICC (LOO-out)       -- reweighted(LOO-out) - natural(LOO-out): the
                                reweighting gap that REMAINS once this one
                                draw is excluded
  delta vs full baseline    -- reweighted(LOO-out) - reweighted(all B
                                draws): how much excluding this draw alone
                                changes reweighted ICC. Large positive =
                                this draw was dragging reweighted ICC down.

Sorted by "delta vs full baseline" descending -- the most harmful draws
first.

Usage: python codes/scripts/compute_loo_draw_sensitivity.py
           --dataset ende22 --k-prime 11 [--b 78] [--metametric spa]
           [--target-alpha 0.9338] [--seed 0]
           [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.icc import icc_c1
from lib.subsampling_stability import _score_one_draw, draw_subsets
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, required=True)
  p.add_argument('--b', type=int, default=None, help='default: C(K,K_prime), exhaustive')
  p.add_argument('--target-alpha', type=float, default=None, help='default: alpha_0(D)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  K_prime = args.k_prime

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full, 'a'].values, df.loc[full, 'b'].values
  target_alpha = args.target_alpha if args.target_alpha is not None else alpha0_of(a_D, b_D)
  import math
  B = args.b if args.b is not None else math.comb(K, K_prime)
  tag = args.tag or f'{base}_kprime{K_prime}_b{B}_{m}'

  draws = draw_subsets(K, K_prime, B, args.seed)
  print(f'D={base}: K={K}, target_alpha={target_alpha:.4f}, '
        f"K'={K_prime}, B={B}", file=sys.stderr)

  jobs = []
  for di, idx in enumerate(draws):
    subset = [full[k] for k in idx]
    jobs.append(('natural', di, (base, m, ROOT, subset, None)))
    jobs.append(('reweighted', di, (base, m, ROOT, subset, target_alpha)))

  scores_by = {'natural': [None] * len(draws), 'reweighted': [None] * len(draws)}
  ess_by = [None] * len(draws)
  t0 = time.time()
  with cf.ProcessPoolExecutor() as ex:
    futs = {ex.submit(_score_one_draw, job): (cond, di) for cond, di, job in jobs}
    for fut in cf.as_completed(futs):
      cond, di = futs[fut]
      sd, ess = fut.result()
      scores_by[cond][di] = sd
      if cond == 'reweighted':
        ess_by[di] = ess
  print(f'scoring pass ({2 * len(draws)} jobs): {time.time() - t0:.1f}s', file=sys.stderr)

  natural_full = scorer_scores(base, m, root=ROOT, systems=full)
  scorers = sorted(natural_full.index)

  def matrix_for(cond, exclude_di=None):
    valid_di = [di for di in range(len(draws)) if scores_by[cond][di] is not None and di != exclude_di]
    rows = {s: [scores_by[cond][di].get(s, float('nan')) for di in valid_di] for s in scorers}
    kept = [s for s in scorers if not any(v != v for v in rows[s])]
    return np.array([rows[s] for s in kept], dtype=float)

  icc_full_nat = icc_c1(matrix_for('natural'))
  icc_full_rw = icc_c1(matrix_for('reweighted'))
  print(f'baseline (all {B} draws): natural={icc_full_nat:.4f}, reweighted={icc_full_rw:.4f}, '
        f'diff={icc_full_rw - icc_full_nat:+.4f}', file=sys.stderr)

  rows_out = []
  for di in range(len(draws)):
    icc_loo_nat = icc_c1(matrix_for('natural', exclude_di=di))
    icc_loo_rw = icc_c1(matrix_for('reweighted', exclude_di=di))
    delta_icc = icc_loo_rw - icc_loo_nat
    vs_baseline = icc_loo_rw - icc_full_rw
    idx = draws[di]
    subset = [full[k] for k in idx]
    dropped_systems = sorted(s for s in full if s not in set(subset))
    a_sub, b_sub = df.loc[subset, 'a'].values, df.loc[subset, 'b'].values
    a0_Dp = alpha0_of(a_sub, b_sub)
    ess = ess_by[di]
    essk = ess / K_prime if ess is not None else float('nan')
    rows_out.append((di + 1, ','.join(dropped_systems), a0_Dp, ess, essk,
                      icc_loo_nat, icc_loo_rw, delta_icc, vs_baseline))

  rows_out.sort(key=lambda r: -r[8])

  lines = []
  lines.append(f'# Leave-one-draw-out ICC(C,1) sensitivity ({base}, metametric={m})')
  lines.append('')
  lines.append(f"D: K={K}, K'={K_prime}, B={B}, seed={args.seed}, target_alpha={target_alpha:.4f}")
  lines.append('')
  lines.append(f'baseline (all {B} draws): ICC natural={icc_full_nat:.4f}, '
               f'ICC reweighted={icc_full_rw:.4f}, diff={icc_full_rw - icc_full_nat:+.4f}')
  lines.append('')
  lines.append('Sorted by "delta vs full baseline" descending -- most harmful draws first '
               '(large positive = excluding this draw alone raises reweighted ICC the most).')
  lines.append('')
  lines.append('| draw# | dropped systems | alpha_0(D\\{dropped}) | ESS | ESS/K\' | '
               'ICC natural (LOO-out) | ICC reweighted (LOO-out) | Delta ICC (LOO-out) | '
               'delta vs full baseline |')
  lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
  for r in rows_out:
    di, dnames, a0_Dp, ess, essk, icc_n, icc_r, dicc, vb = r
    lines.append(f'| {di} | {dnames} | {a0_Dp:.4f} | {ess:.4f} | {essk:.4f} | {icc_n:.4f} | '
                 f'{icc_r:.4f} | {dicc:+.4f} | {vb:+.4f} |')

  table_md = '\n'.join(lines)
  print('\n' + table_md)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'loo_draw_sensitivity_{tag}.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)
