"""Computes SPA(scorer; beta) vs. target balance beta, for every real
scorer in a dataset whose name contains one of --substrings (case-
insensitive) -- by default the MetricX and xCOMET variants compared in the
paper's Figure 5 (NOT every COMET-family scorer: 'comet' alone would also
match CometKiwi, Calibri-COMET22, cometoid22-*, etc., which the paper does
not include here) -- and caches the result to disk (output/spa_vs_beta_<tag>_
n<n_beta>.npz) -- the expensive step (src.lib.reweight_exact.solve_w_exact per
beta, a permutation test per scorer, src.lib.spa_beta_sweep) separated from
plotting (plot_spa_vs_beta.py, plot_spa_vs_beta_combo.py) so
replotting/restyling never requires rerunning it.

For each beta on the sweep, w*(beta) = solve_w_exact(a_D, f_D, beta).w is
the reweighting of the dataset's systems to that target balance;
SPA(scorer; beta) is src.lib.metametrics.weighted_soft_pairwise_accuracy under
that w (via the precomputed-pairwise-p-values path,
soft_pairwise_accuracy_from_pvalues, so the 1000-permutation test is run once
per scorer and reused across the whole beta grid rather than once per
(scorer, beta) point).

Beta sweep: the sub-range of the dataset's reachable [beta_min, beta_max]
where ESS(w*(beta)) >= --min-ess (K = number of real systems) -- found by
bracketing beta_0(D) (where ESS is exactly K, its maximum) on each side and
solving ESS(beta) = --min-ess with Brent's method (src.lib.spa_beta_sweep.
beta_sweep_range) -- then --n-beta (default 10) points evenly spaced
across that sub-range.

--min-ess (default -1): a NEGATIVE value is interpreted relative to K, i.e.
target_ess = K + min_ess (so the default -1 means K-1). A value >= 0 is used
as an absolute ESS floor instead (target_ess = min_ess directly).

The cache filename ALWAYS includes --n-beta (spa_vs_beta_<tag>_n<n_beta>.
npz), even when --tag is given explicitly -- two runs that differ only in
--n-beta (e.g. a coarse 10-point pass and a finer 20-point pass over the
same sweep, for a side-by-side comparison via plot_spa_vs_beta_combo.py)
would otherwise silently overwrite each other's cache under a shared --tag.

Usage: python -m src.scripts.compute_spa_vs_beta
           [--dataset ende24] [--substrings metricx,xcomet] [--min-ess -1] [--n-beta 10] [--tag TAG]
"""

import argparse
import os
import sys
import time

import numpy as np

from src.lib.consistency import load_scorer_inputs, real_scorers, real_systems
from src.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from src.lib.reweight_exact import solve_w_exact
from src.lib.spa_beta_sweep import beta_sweep_range, resolve_target_ess, save_spa_vs_beta
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende24')
  p.add_argument('--substrings', type=str, default='metricx,xcomet',
                  help='comma-separated, case-insensitive substrings a scorer name must contain '
                       'to be included')
  p.add_argument('--min-ess', type=float, default=-1,
                  help='ESS threshold bounding the beta sweep. Negative: interpreted as K + '
                       'min_ess (default -1 means K-1). Non-negative: used directly as an '
                       'absolute ESS floor.')
  p.add_argument('--n-beta', type=int, default=10, help='number of beta points to sweep')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  substrings = [s.strip().lower() for s in args.substrings.split(',') if s.strip()]
  tag = args.tag or base

  systems = real_systems(base, root=ROOT)
  K = len(systems)
  if K < 3:
    sys.exit(f'{base}: only {K} real systems, nothing to sweep')
  df = load_system_scores(base, root=ROOT)
  a_D, f_D = df.loc[systems, 'a'].values, df.loc[systems, 'f'].values

  target_ess = resolve_target_ess(K, args.min_ess)
  beta_0_D, beta_lo, beta_hi, left, right = beta_sweep_range(a_D, f_D, target_ess)
  print(f'{base}: K={K} systems, beta_0(D)={beta_0_D:.4f}, reachable range [{beta_lo:.4f}, {beta_hi:.4f}]',
        file=sys.stderr)
  betas = np.linspace(left, right, args.n_beta)
  print(f'ESS >= {target_ess:.4g} (--min-ess={args.min_ess}) sweep range: [{left:.5f}, {right:.5f}], '
        f'{args.n_beta} points: {[round(b, 5) for b in betas]}', file=sys.stderr)

  candidates = real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True)
  scorers = sorted(s for s in candidates if any(sub in s.lower() for sub in substrings))
  print(f'{base}: {len(candidates)} real scorers, {len(scorers)} matching {substrings}: {scorers}',
        file=sys.stderr)
  if not scorers:
    sys.exit(f'{base}: no scorer matched {substrings}')

  inputs = load_scorer_inputs(base, 'spa', root=ROOT, systems=systems)

  # w*(beta) solved once per beta, reused across every scorer.
  t0 = time.time()
  w_by_beta = {}
  for i, beta in enumerate(betas, 1):
    w_by_beta[beta] = solve_w_exact(a_D, f_D, beta).w
    print(f'  [solve {i}/{len(betas)}] beta={beta:.5f} elapsed={time.time() - t0:.1f}s',
          file=sys.stderr, flush=True)

  # Pairwise p-values solved once per scorer (the expensive, beta-
  # independent permutation test), then SPA reweighted per beta via
  # soft_pairwise_accuracy_from_pvalues -- see module docstring.
  t1 = time.time()
  spa = np.full((len(scorers), len(betas)), np.nan)
  for i, scorer in enumerate(scorers, 1):
    x = inputs[scorer]
    human_p = pairwise_p_values(x.human_seg)
    scorer_p = pairwise_p_values(x.scorer_seg)
    ys = [soft_pairwise_accuracy_from_pvalues(human_p, scorer_p, w_by_beta[beta]) for beta in betas]
    spa[i - 1] = ys
    elapsed = time.time() - t1
    eta = elapsed / i * (len(scorers) - i)
    print(f'[{i}/{len(scorers)}] {scorer}: SPA range [{min(ys):.4f}, {max(ys):.4f}] '
          f'(elapsed={elapsed:.1f}s, eta={eta:.1f}s)', file=sys.stderr, flush=True)

  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'spa_vs_beta_{tag}_n{args.n_beta}.npz')
  save_spa_vs_beta(
      out_path, dataset=base, substrings=substrings, min_ess=args.min_ess, target_ess=target_ess, K=K,
      beta_0_D=beta_0_D, beta_lo=beta_lo, beta_hi=beta_hi, betas=betas, scorers=scorers, spa=spa,
  )
  print(f'Wrote {out_path}', file=sys.stderr)
