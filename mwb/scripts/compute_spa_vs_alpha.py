"""Computes SPA(scorer; alpha) vs. target balance alpha, for every real
scorer in a dataset whose name contains one of --substrings (case-
insensitive) -- by default the MetricX and xCOMET variants compared in the
paper's Figure 5 (NOT every COMET-family scorer: 'comet' alone would also
match CometKiwi, Calibri-COMET22, cometoid22-*, etc., which the paper does
not include here) -- and caches the result to disk (output/data/spa_vs_alpha_<tag>_
n<n_alpha>.npz) -- the expensive step (mwb.lib.reweight_exact.solve_w_exact per
alpha, a permutation test per scorer, mwb.lib.spa_alpha_sweep) separated from
plotting (plot_spa_vs_alpha.py, plot_spa_vs_alpha_combo.py) so
replotting/restyling never requires rerunning it.

For each alpha on the sweep, w*(alpha) = solve_w_exact(a_D, b_D, alpha).w is
the reweighting of the dataset's systems to that target balance;
SPA(scorer; alpha) is mwb.lib.metametrics.weighted_soft_pairwise_accuracy under
that w (via the precomputed-pairwise-p-values path,
soft_pairwise_accuracy_from_pvalues, so the 1000-permutation test is run once
per scorer and reused across the whole alpha grid rather than once per
(scorer, alpha) point).

Alpha sweep: the sub-range of the dataset's reachable [alpha_min, alpha_max]
where ESS(w*(alpha)) >= --min-ess (K = number of real systems) -- found by
bracketing alpha_0(D) (where ESS is exactly K, its maximum) on each side and
solving ESS(alpha) = --min-ess with Brent's method (mwb.lib.spa_alpha_sweep.
alpha_sweep_range) -- then --n-alpha (default 10) points evenly spaced
across that sub-range.

--min-ess (default -1): a NEGATIVE value is interpreted relative to K, i.e.
target_ess = K + min_ess (so the default -1 means K-1). A value >= 0 is used
as an absolute ESS floor instead (target_ess = min_ess directly).

The cache filename ALWAYS includes --n-alpha (spa_vs_alpha_<tag>_n<n_alpha>.
npz), even when --tag is given explicitly -- two runs that differ only in
--n-alpha (e.g. a coarse 10-point pass and a finer 20-point pass over the
same sweep, for a side-by-side comparison via plot_spa_vs_alpha_combo.py)
would otherwise silently overwrite each other's cache under a shared --tag.

Usage: python -m mwb.scripts.compute_spa_vs_alpha
           [--dataset ende24] [--substrings metricx,xcomet] [--min-ess -1] [--n-alpha 10] [--tag TAG]
"""

import argparse
import os
import sys
import time

import numpy as np

from mwb.lib.consistency import load_scorer_inputs, real_scorers, real_systems
from mwb.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from mwb.lib.reweight_exact import solve_w_exact
from mwb.lib.spa_alpha_sweep import alpha_sweep_range, resolve_target_ess, save_spa_vs_alpha
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende24')
  p.add_argument('--substrings', type=str, default='metricx,xcomet',
                  help='comma-separated, case-insensitive substrings a scorer name must contain '
                       'to be included')
  p.add_argument('--min-ess', type=float, default=-1,
                  help='ESS threshold bounding the alpha sweep. Negative: interpreted as K + '
                       'min_ess (default -1 means K-1). Non-negative: used directly as an '
                       'absolute ESS floor.')
  p.add_argument('--n-alpha', type=int, default=10, help='number of alpha points to sweep')
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
  a_D, b_D = df.loc[systems, 'a'].values, df.loc[systems, 'b'].values

  target_ess = resolve_target_ess(K, args.min_ess)
  a0_D, alpha_lo, alpha_hi, left, right = alpha_sweep_range(a_D, b_D, target_ess)
  print(f'{base}: K={K} systems, alpha_0(D)={a0_D:.4f}, reachable range [{alpha_lo:.4f}, {alpha_hi:.4f}]',
        file=sys.stderr)
  alphas = np.linspace(left, right, args.n_alpha)
  print(f'ESS >= {target_ess:.4g} (--min-ess={args.min_ess}) sweep range: [{left:.5f}, {right:.5f}], '
        f'{args.n_alpha} points: {[round(a, 5) for a in alphas]}', file=sys.stderr)

  candidates = real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True)
  scorers = sorted(s for s in candidates if any(sub in s.lower() for sub in substrings))
  print(f'{base}: {len(candidates)} real scorers, {len(scorers)} matching {substrings}: {scorers}',
        file=sys.stderr)
  if not scorers:
    sys.exit(f'{base}: no scorer matched {substrings}')

  inputs = load_scorer_inputs(base, 'spa', root=ROOT, systems=systems)

  # w*(alpha) solved once per alpha, reused across every scorer.
  t0 = time.time()
  w_by_alpha = {}
  for i, alpha in enumerate(alphas, 1):
    w_by_alpha[alpha] = solve_w_exact(a_D, b_D, alpha).w
    print(f'  [solve {i}/{len(alphas)}] alpha={alpha:.5f} elapsed={time.time() - t0:.1f}s',
          file=sys.stderr, flush=True)

  # Pairwise p-values solved once per scorer (the expensive, alpha-
  # independent permutation test), then SPA reweighted per alpha via
  # soft_pairwise_accuracy_from_pvalues -- see module docstring.
  t1 = time.time()
  spa = np.full((len(scorers), len(alphas)), np.nan)
  for i, scorer in enumerate(scorers, 1):
    x = inputs[scorer]
    human_p = pairwise_p_values(x.human_seg)
    metric_p = pairwise_p_values(x.metric_seg)
    ys = [soft_pairwise_accuracy_from_pvalues(human_p, metric_p, w_by_alpha[alpha]) for alpha in alphas]
    spa[i - 1] = ys
    elapsed = time.time() - t1
    eta = elapsed / i * (len(scorers) - i)
    print(f'[{i}/{len(scorers)}] {scorer}: SPA range [{min(ys):.4f}, {max(ys):.4f}] '
          f'(elapsed={elapsed:.1f}s, eta={eta:.1f}s)', file=sys.stderr, flush=True)

  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'spa_vs_alpha_{tag}_n{args.n_alpha}.npz')
  save_spa_vs_alpha(
      out_path, dataset=base, substrings=substrings, min_ess=args.min_ess, target_ess=target_ess, K=K,
      a0_D=a0_D, alpha_lo=alpha_lo, alpha_hi=alpha_hi, alphas=alphas, scorers=scorers, spa=spa,
  )
  print(f'Wrote {out_path}', file=sys.stderr)
