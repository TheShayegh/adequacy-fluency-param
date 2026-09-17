"""Computes Kendall's tau-a (src.lib.consistency.pool_weighted_tau, via
src.lib.subsampling_stability.leave_p_out_tau) pooled across a dataset's K
leave-one-out (LOO) raters, as a function of target balance beta -- this
produces the paper's Figure 6 (Appendix, Sec. "Does Controlling
Adequacy--Fluency Balance Stabilize Meta-Evaluation?"): a consistency
statistic swept over reweighted beta, ESS-colored, pooled across one
dataset's K LOO subsets, SPA-only.

For each beta on the sweep, every one of the dataset's K LOO subsets
D\\{i} is reweighted (src.lib.reweight_exact.solve_w_exact) to that SAME
shared target beta (identical "one target across every rater" convention
as leave_p_out_tau's own reweighted condition), scored under SPA, and the
resulting K scorer-rankings (fewer if beta was infeasible for some D\\{i},
Theorem "Reachable range") are pooled into one tau via pool_weighted_tau --
exactly compute_loo_tau.py's --reweighted tau computation, but swept over
many beta values instead of just beta_0(D)/median beta_ij(D).

Beta sweep: --n-beta (default 21) points evenly spaced across the FULL
dataset D's reachable [beta_min(D), beta_max(D)] (src.lib.beta.
beta_min_max), inset by --eps-frac (default 1e-3) of the range at each
end -- src.lib.synthetic_scorer_preference.full_range_betas's own
convention (the exact boundary admits only a single degenerate 2-system
support, best kept off it). Deliberately NOT an ESS-threshold-bounded
sub-range (contrast compute_spa_vs_beta.py's beta_sweep_range): pairs
with the plot script's moderated (opacity-only, not range-truncating)
reliability coloring, which fades low-ESS points out smoothly rather than
excluding them, so there's no need to truncate the domain to begin with --
and an ESS floor near the true minimum (2) sits right at beta_sweep_
range's own numerically-unstable edge (its Brent's-method root search
needs a strict sign change, which an exact-boundary target can fail to
bracket).

The cache filename ALWAYS includes --n-beta (loo_tau_vs_beta_<tag>_
n<n_beta>.npz), same reasoning as compute_spa_vs_beta.py.

STREAMED progress (project convention): per-beta elapsed/ETA logged to
stderr as each point finishes; the .npz itself is written once at the end
(matches compute_spa_vs_beta.py's own convention -- a handful of beta
points, not a long per-dataset loop, so there's nothing meaningful to
stream incrementally to disk).

Usage: python -m src.scripts.compute_loo_tau_vs_beta
           [--dataset zhen23] [--n-beta 21] [--eps-frac 1e-3] [--tag TAG]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from src.lib.beta import beta_0 as beta0_of, beta_min_max
from src.lib.consistency import real_systems
from src.lib.subsampling_stability import leave_p_out_tau
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output')

METAMETRIC = 'spa'
LEAVE_OUT = 1  # plain LOO


def save_loo_tau_vs_beta(
    path: str, *, dataset: str, K: int, beta_0_D: float, beta_lo: float, beta_hi: float,
    betas: np.ndarray, tau: np.ndarray, mean_ess: np.ndarray, n_raters_used: np.ndarray,
    baseline_tau: float,
) -> None:
  np.savez(
      path, dataset=dataset, metametric=METAMETRIC, leave_out=LEAVE_OUT, K=K, beta_0_D=beta_0_D,
      beta_lo=beta_lo, beta_hi=beta_hi, betas=np.asarray(betas, dtype=float),
      tau=np.asarray(tau, dtype=float), mean_ess=np.asarray(mean_ess, dtype=float),
      n_raters_used=np.asarray(n_raters_used, dtype=int), baseline_tau=baseline_tau,
  )


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='zhen23')
  p.add_argument('--n-beta', type=int, default=21, help='number of beta points to sweep')
  p.add_argument('--eps-frac', type=float, default=1e-3,
                  help='inset from [beta_min(D), beta_max(D)] at each end, as a fraction of '
                       'the range -- keeps the sweep off the exact boundary (a degenerate '
                       '2-system support), matching full_range_betas\' own convention')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  tag = args.tag or base

  full_systems = real_systems(base, root=ROOT)
  K = len(full_systems)
  if K < 3:
    sys.exit(f'{base}: only {K} real systems, nothing to sweep')
  df = load_system_scores(base, root=ROOT)
  a_D, f_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'f'].values

  beta_0_D = beta0_of(a_D, f_D)
  beta_lo, beta_hi = beta_min_max(a_D, f_D)
  eps = (beta_hi - beta_lo) * args.eps_frac
  left, right = beta_lo + eps, beta_hi - eps
  print(f'{base}: K={K} systems, beta_0(D)={beta_0_D:.4f}, reachable range '
        f'[{beta_lo:.4f}, {beta_hi:.4f}]', file=sys.stderr)
  betas = np.linspace(left, right, args.n_beta)
  print(f'full-range sweep (eps_frac={args.eps_frac}): [{left:.5f}, {right:.5f}], '
        f'{args.n_beta} points: {[round(b, 5) for b in betas]}', file=sys.stderr)

  t0 = time.time()
  baseline = leave_p_out_tau(base, METAMETRIC, p=LEAVE_OUT, root=ROOT, target_beta=None)
  print(f'baseline (natural) LOO tau = {baseline["tau"]:.4f} ({time.time() - t0:.1f}s)',
        file=sys.stderr)

  tau = np.full(args.n_beta, np.nan)
  mean_ess = np.full(args.n_beta, np.nan)
  n_raters_used = np.zeros(args.n_beta, dtype=int)
  t1 = time.time()
  for i, beta in enumerate(betas, 1):
    r = leave_p_out_tau(base, METAMETRIC, p=LEAVE_OUT, root=ROOT, target_beta=float(beta))
    tau[i - 1] = r['tau']
    mean_ess[i - 1] = r['mean_ess']
    n_raters_used[i - 1] = r['n_raters_used']
    elapsed = time.time() - t1
    eta = elapsed / i * (args.n_beta - i)
    print(f'[{i}/{args.n_beta}] beta={beta:.5f}: tau={r["tau"]:.4f}, mean_ess={r["mean_ess"]:.2f}, '
          f'raters={r["n_raters_used"]}/{r["n_raters_total"]} (elapsed={elapsed:.1f}s, eta={eta:.1f}s)',
          file=sys.stderr, flush=True)

  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'loo_tau_vs_beta_{tag}_n{args.n_beta}.npz')
  save_loo_tau_vs_beta(
      out_path, dataset=base, K=K, beta_0_D=beta_0_D, beta_lo=beta_lo, beta_hi=beta_hi,
      betas=betas, tau=tau, mean_ess=mean_ess, n_raters_used=n_raters_used,
      baseline_tau=baseline['tau'],
  )
  print(f'Wrote {out_path}', file=sys.stderr)
