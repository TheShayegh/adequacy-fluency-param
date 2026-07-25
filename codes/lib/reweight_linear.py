"""Closed-form solver for a LINEAR target statistic: min_w (1/2)sum(w_i^2)
s.t. sum(w)=1, w@stat=target, w>=0. Built for the alt-target ablation
(mean adequacy / mean fluency / mean all-MQM matching), where lib.
reweight_exact's machinery doesn't apply -- that module is built around a
QUADRATIC balance constraint (alpha), and this one is linear.

Unlike the alpha case, a linear equality constraint keeps the feasible set
(intersection of two affine hyperplanes with the simplex) convex, so the
whole problem is a standard convex QP: no secular polynomial, no curvature
certificate, no completeness-by-enumeration needed -- global optimality
follows from convexity alone once a KKT point is found.

Full-support closed form (Lagrangian stationarity w_i = -mu - lambda*stat_i,
solved against the two linear constraints):

  w_i = 1/|S| + (target - mean_S(stat)) / SS_S(stat) * (stat_i - mean_S(stat))

-- the same formula as regression/GREG calibration in survey statistics.
If every w_i >= 0 this is the (globally optimal) answer; else the most
negative coordinate is dropped and the support re-solved, repeating until
every remaining weight is non-negative -- an active-set loop, standard for
equality+non-negativity least-norm problems, guaranteed to terminate since
each iteration strictly shrinks a finite support.

Cross-validated against lib.reweight_exhaustive_linear's brute-force grid
at small K (where a fine-enough grid_n is actually tractable) -- see
codes/scripts/compute_type7_alt_target_ablation.py's own validation step.
"""

from __future__ import annotations

import dataclasses

import numpy as np

_DEFAULT_TOL = 1e-9


@dataclasses.dataclass
class LinearWResult:
  """Result of one call to solve_w_linear."""
  w: np.ndarray
  target: float
  achieved: float
  ess: float
  support_size: int
  message: str


def solve_w_linear(stat: np.ndarray, target: float, tol: float = _DEFAULT_TOL) -> LinearWResult:
  """Global optimum of min_w (1/2)sum(w^2) s.t. sum(w)=1, w@stat=target,
  w>=0. Raises ValueError if `target` is outside [min(stat), max(stat)] (a
  weighted mean always stays within that range -- Corollary-2-style
  feasibility check) or if the active-set loop collapses below 2 systems
  (only reachable by a near-degenerate point)."""
  stat = np.asarray(stat, dtype=float)
  K = len(stat)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  smin, smax = float(stat.min()), float(stat.max())
  if not (smin - 1e-6 <= target <= smax + 1e-6):
    raise ValueError(f'target={target} outside reachable range [{smin}, {smax}]')

  support = list(range(K))
  while True:
    s = stat[support]
    n = len(support)
    if n < 2:
      raise ValueError(f'active-set collapsed to {n} system(s) for target={target} -- '
                        f'only reachable by a near-degenerate weighting')
    mean_s = s.mean()
    ss = float(((s - mean_s) ** 2).sum())
    if ss < 1e-12:
      # Every remaining system has (numerically) the same stat value: any
      # split among them achieves the same w@stat, so uniform is as good
      # as any (min-norm) -- this only fires if target already equals
      # mean_s here, since the support only ever shrinks from a feasible
      # superset and feasibility (mean within [min,max]) was checked above.
      w_support = np.full(n, 1.0 / n)
    else:
      lam = (target - mean_s) / ss
      w_support = 1.0 / n + lam * (s - mean_s)
    if np.all(w_support >= -tol):
      w_support = np.clip(w_support, 0.0, None)
      w_support = w_support / w_support.sum()
      w = np.zeros(K)
      w[support] = w_support
      achieved = float(w @ stat)
      ess = 1.0 / float((w ** 2).sum())
      return LinearWResult(w=w, target=target, achieved=achieved, ess=ess,
                            support_size=n, message=f'support size {n}/{K}')
    drop_local = int(np.argmin(w_support))
    support.pop(drop_local)
