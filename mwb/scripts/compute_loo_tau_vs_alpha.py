"""Computes Kendall's tau-a (mwb.lib.consistency.pool_weighted_tau, via
mwb.lib.subsampling_stability.leave_p_out_tau) pooled across a dataset's K
leave-one-out (LOO) raters, as a function of target balance alpha -- this
produces the paper's Figure 6 (Appendix, Sec. "Does Controlling
Adequacy--Fluency Balance Stabilize Meta-Evaluation?"): a consistency
statistic swept over reweighted alpha, ESS-colored, pooled across one
dataset's K LOO subsets, SPA-only.

For each alpha on the sweep, every one of the dataset's K LOO subsets
D\\{i} is reweighted (mwb.lib.reweight_exact.solve_w_exact) to that SAME
shared target alpha (identical "one target across every rater" convention
as leave_p_out_tau's own reweighted condition), scored under SPA, and the
resulting K scorer-rankings (fewer if alpha was infeasible for some D\\{i},
Theorem "Reachable range") are pooled into one tau via pool_weighted_tau --
exactly compute_loo_tau.py's --reweighted tau computation, but swept over
many alpha values instead of just alpha_0(D)/median alpha_ij(D).

Alpha sweep: --n-alpha (default 21) points evenly spaced across the FULL
dataset D's reachable [alpha_min(D), alpha_max(D)] (mwb.lib.alpha.
alpha_min_max), inset by --eps-frac (default 1e-3) of the range at each
end -- mwb.lib.synthetic_scorer_orientation.full_range_alphas's own
convention (the exact boundary admits only a single degenerate 2-system
support, best kept off it). Deliberately NOT an ESS-threshold-bounded
sub-range (contrast compute_spa_vs_alpha.py's alpha_sweep_range): pairs
with the plot script's moderated (opacity-only, not range-truncating)
reliability coloring, which fades low-ESS points out smoothly rather than
excluding them, so there's no need to truncate the domain to begin with --
and an ESS floor near the true minimum (2) sits right at alpha_sweep_
range's own numerically-unstable edge (its Brent's-method root search
needs a strict sign change, which an exact-boundary target can fail to
bracket).

The cache filename ALWAYS includes --n-alpha (loo_tau_vs_alpha_<tag>_
n<n_alpha>.npz), same reasoning as compute_spa_vs_alpha.py.

STREAMED progress (project convention): per-alpha elapsed/ETA logged to
stderr as each point finishes; the .npz itself is written once at the end
(matches compute_spa_vs_alpha.py's own convention -- a handful of alpha
points, not a long per-dataset loop, so there's nothing meaningful to
stream incrementally to disk).

Usage: python -m mwb.scripts.compute_loo_tau_vs_alpha
           [--dataset zhen23] [--n-alpha 21] [--eps-frac 1e-3] [--tag TAG]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from mwb.lib.alpha import alpha_0 as alpha0_of, alpha_min_max
from mwb.lib.consistency import real_systems
from mwb.lib.subsampling_stability import leave_p_out_tau
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output', 'data')

METAMETRIC = 'spa'
LEAVE_OUT = 1  # plain LOO


def save_loo_tau_vs_alpha(
    path: str, *, dataset: str, K: int, a0_D: float, alpha_lo: float, alpha_hi: float,
    alphas: np.ndarray, tau: np.ndarray, mean_ess: np.ndarray, n_raters_used: np.ndarray,
    baseline_tau: float,
) -> None:
  np.savez(
      path, dataset=dataset, metametric=METAMETRIC, leave_out=LEAVE_OUT, K=K, a0_D=a0_D,
      alpha_lo=alpha_lo, alpha_hi=alpha_hi, alphas=np.asarray(alphas, dtype=float),
      tau=np.asarray(tau, dtype=float), mean_ess=np.asarray(mean_ess, dtype=float),
      n_raters_used=np.asarray(n_raters_used, dtype=int), baseline_tau=baseline_tau,
  )


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='zhen23')
  p.add_argument('--n-alpha', type=int, default=21, help='number of alpha points to sweep')
  p.add_argument('--eps-frac', type=float, default=1e-3,
                  help='inset from [alpha_min(D), alpha_max(D)] at each end, as a fraction of '
                       'the range -- keeps the sweep off the exact boundary (a degenerate '
                       '2-system support), matching full_range_alphas\' own convention')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  tag = args.tag or base

  full_systems = real_systems(base, root=ROOT)
  K = len(full_systems)
  if K < 3:
    sys.exit(f'{base}: only {K} real systems, nothing to sweep')
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'b'].values

  a0_D = alpha0_of(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  eps = (alpha_hi - alpha_lo) * args.eps_frac
  left, right = alpha_lo + eps, alpha_hi - eps
  print(f'{base}: K={K} systems, alpha_0(D)={a0_D:.4f}, reachable range '
        f'[{alpha_lo:.4f}, {alpha_hi:.4f}]', file=sys.stderr)
  alphas = np.linspace(left, right, args.n_alpha)
  print(f'full-range sweep (eps_frac={args.eps_frac}): [{left:.5f}, {right:.5f}], '
        f'{args.n_alpha} points: {[round(a, 5) for a in alphas]}', file=sys.stderr)

  t0 = time.time()
  baseline = leave_p_out_tau(base, METAMETRIC, p=LEAVE_OUT, root=ROOT, target_alpha=None)
  print(f'baseline (natural) LOO tau = {baseline["tau"]:.4f} ({time.time() - t0:.1f}s)',
        file=sys.stderr)

  tau = np.full(args.n_alpha, np.nan)
  mean_ess = np.full(args.n_alpha, np.nan)
  n_raters_used = np.zeros(args.n_alpha, dtype=int)
  t1 = time.time()
  for i, alpha in enumerate(alphas, 1):
    r = leave_p_out_tau(base, METAMETRIC, p=LEAVE_OUT, root=ROOT, target_alpha=float(alpha))
    tau[i - 1] = r['tau']
    mean_ess[i - 1] = r['mean_ess']
    n_raters_used[i - 1] = r['n_raters_used']
    elapsed = time.time() - t1
    eta = elapsed / i * (args.n_alpha - i)
    print(f'[{i}/{args.n_alpha}] alpha={alpha:.5f}: tau={r["tau"]:.4f}, mean_ess={r["mean_ess"]:.2f}, '
          f'raters={r["n_raters_used"]}/{r["n_raters_total"]} (elapsed={elapsed:.1f}s, eta={eta:.1f}s)',
          file=sys.stderr, flush=True)

  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'loo_tau_vs_alpha_{tag}_n{args.n_alpha}.npz')
  save_loo_tau_vs_alpha(
      out_path, dataset=base, K=K, a0_D=a0_D, alpha_lo=alpha_lo, alpha_hi=alpha_hi,
      alphas=alphas, tau=tau, mean_ess=mean_ess, n_raters_used=n_raters_used,
      baseline_tau=baseline['tau'],
  )
  print(f'Wrote {out_path}', file=sys.stderr)
