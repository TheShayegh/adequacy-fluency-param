"""NUMERIC solver for (P) (action_plan.md section 3.4): given real systems'
score vectors a, b and a target balance alpha, find the weights w closest
to uniform (min 1/2 sum w_i^2) subject to sum(w) = 1, w >= 0, and
alpha(w) = alpha exactly.

This is the *numeric* solver: multi-start scipy.optimize (trust-constr) run
directly on the nonconvex QCQP, using the closed-form gradient/Hessian of
the balance constraint that action_plan.md 3.5-3.6 already derives, but
with NO certificate of global optimality. action_plan.md 3.5-3.11 derives a
separate procedure (support enumeration + Theorem 2's sufficient KKT/
curvature certificate) that IS proven globally optimal; that becomes a
second module (the "exact"/"analytic" solver, not yet implemented) used to
cross-validate this one's output, not to replace it. Every name in this
file says "numeric" for exactly that reason -- don't let the two collide.

(P) is nonconvex (the balance constraint is an indefinite quadratic for
alpha in (0,1), action_plan.md 3.6), so a single local solve is not
guaranteed to find the global optimum -- action_plan.md 3.7 gives a
concrete K=4 instance where a KKT point is provably NOT globally optimal.
We mitigate (not eliminate) that risk with multi-start: several restarts
from the uniform weighting plus randomized Dirichlet draws, keeping the
best feasible result.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import scipy.optimize

from lib.alpha import alpha as alpha_of, alpha_min_max

_FEASIBILITY_TOL = 1e-6
_NONNEG_TOL = 1e-8
# A w concentrated on a single system has sigma(a;w)^2 = sigma(b;w)^2 = 0
# exactly, so g(w,alpha) = (1-alpha)*0 - alpha*0 = 0 *trivially, for every
# alpha* -- it satisfies the balance constraint vacuously regardless of the
# target, while alpha(w) itself is an ill-defined 0/0. Near alpha_min/
# alpha_max the true feasible region pinches to almost a point, so trust-
# constr sometimes fails to hit it precisely from a generic start while
# this spurious corner (trivially "feasible" from any start) gets found
# instead. Corollary 2 guarantees every genuine feasible point has ESS>=2
# (the two-system boundary case), so ESS<2 is a certificate the candidate
# is this spurious single-point trap, not a real solution -- reject it
# during selection rather than silently returning a meaningless alpha.
_MIN_ESS_TOL = 1e-4


@dataclasses.dataclass
class NumericWResult:
  """Result of one call to solve_w_numeric."""
  w: np.ndarray
  alpha_target: float
  alpha_achieved: float
  success: bool             # a restart converged to a feasible point
  message: str               # scipy's status message for the winning restart
  objective: float            # 1/2 * sum(w^2), the (P) objective
  ess: float                  # 1 / sum(w^2) (== 1 / (2*objective))
  n_restarts: int
  winning_restart: int
  budget_violation: float     # |sum(w) - 1|
  balance_violation: float    # |g(w, alpha_target)|
  min_weight: float           # min(w); should be >= -tol at a feasible point


def _M_and_r(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
  """M(alpha), r(alpha) of action_plan.md 3.6, s.t. g(w,alpha) = w^T M w + r^T w."""
  M = alpha * np.outer(b, b) - (1 - alpha) * np.outer(a, a)
  r = (1 - alpha) * a ** 2 - alpha * b ** 2
  return M, r


def _g(w: np.ndarray, M: np.ndarray, r: np.ndarray) -> float:
  return float(w @ M @ w + r @ w)


def _g_grad(w: np.ndarray, M: np.ndarray, r: np.ndarray) -> np.ndarray:
  """d(w,alpha) of action_plan.md 3.5: the gradient of g w.r.t. w."""
  return 2.0 * M @ w + r


def _random_starts(K: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
  """Uniform weighting plus n-1 Dirichlet draws at varied concentration
  (some near-uniform, some sparse/near-a-vertex) so restarts probe both the
  interior (good for near-alpha_0 targets) and the boundary (good for
  targets near alpha_min/alpha_max, where the optimum is nearly a
  two-system point mass, action_plan.md 3.8's edge case)."""
  starts = [np.full(K, 1.0 / K)]
  for _ in range(n - 1):
    concentration = rng.uniform(0.2, 4.0)
    starts.append(rng.dirichlet(np.full(K, concentration)))
  return starts


def solve_w_numeric(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    n_restarts: int = 25,
    seed: int = 0,
    range_tol: float = 1e-9,
) -> NumericWResult:
  """Numerically solves (P) at a single target alpha via multi-start
  scipy.optimize.minimize(method='trust-constr'), supplying the exact
  gradient/Hessian of the objective and of the balance constraint (both
  closed-form, action_plan.md 3.5-3.6) rather than relying on finite
  differences. Returns the best feasible restart by objective value; if no
  restart reaches feasibility, returns the restart with the smallest
  combined constraint violation and success=False.
  """
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  K = len(a)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= alpha <= 1.0):
    raise ValueError(f'alpha={alpha} outside [0,1]')
  amin, amax = alpha_min_max(a, b)
  if not (amin - range_tol <= alpha <= amax + range_tol):
    raise ValueError(f'alpha={alpha} outside reachable range [{amin}, {amax}] (Corollary 2)')

  M, r = _M_and_r(a, b, alpha)

  budget = scipy.optimize.LinearConstraint(np.ones((1, K)), 1.0, 1.0)
  balance = scipy.optimize.NonlinearConstraint(
      fun=lambda w: _g(w, M, r),
      lb=0.0, ub=0.0,
      jac=lambda w: _g_grad(w, M, r),
      hess=lambda w, v: v[0] * 2.0 * M,
  )
  bounds = scipy.optimize.Bounds(0.0, 1.0)

  rng = np.random.default_rng(seed)
  starts = _random_starts(K, n_restarts, rng)

  best = None  # (feasible: bool, objective: float, idx, w, message)
  for idx, w0 in enumerate(starts):
    # trust-constr's SQP step falls back to an SVD when the constraint
    # Jacobian is near-singular -- expected and already handled internally
    # when a restart lands near a target close to alpha_min/alpha_max
    # (there the feasible set is nearly a single two-system point, action_
    # plan.md 3.8's edge case), not a sign of a bad solve.
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', message='Singular Jacobian matrix')
      res = scipy.optimize.minimize(
          fun=lambda w: 0.5 * float(np.sum(w ** 2)),
          x0=w0,
          jac=lambda w: w,
          hess=lambda w: np.eye(K),
          constraints=[budget, balance],
          bounds=bounds,
          method='trust-constr',
          options={'gtol': 1e-12, 'xtol': 1e-13, 'maxiter': 2000},
      )
    w = np.clip(res.x, 0.0, None)
    budget_violation = abs(float(w.sum()) - 1.0)
    balance_violation = abs(_g(w, M, r))
    w_sum_sq = float(np.sum(w ** 2))
    ess = 1.0 / w_sum_sq if w_sum_sq > 0 else 0.0
    feasible = (
        budget_violation < _FEASIBILITY_TOL
        and balance_violation < _FEASIBILITY_TOL
        and float(w.min()) > -_NONNEG_TOL
        and ess >= 2.0 - _MIN_ESS_TOL
    )
    objective = 0.5 * float(np.sum(w ** 2))
    candidate = (feasible, objective, idx, w, res.message)
    if best is None or (candidate[0], -candidate[1]) > (best[0], -best[1]):
      best = candidate

  feasible, _, idx, w, message = best
  if w.sum() > 0:
    w = w / w.sum()  # correct tiny numerical drift in the budget constraint
  return NumericWResult(
      w=w,
      alpha_target=alpha,
      alpha_achieved=alpha_of(a, b, w),
      success=bool(feasible),
      message=message,
      objective=0.5 * float(np.sum(w ** 2)),
      ess=1.0 / float(np.sum(w ** 2)),
      n_restarts=len(starts),
      winning_restart=idx,
      budget_violation=abs(float(w.sum()) - 1.0),
      balance_violation=abs(_g(w, M, r)),
      min_weight=float(w.min()),
  )


def solve_w_curve_numeric(
    a: np.ndarray, b: np.ndarray, alphas, **kwargs
) -> list[NumericWResult]:
  """solve_w_numeric at each alpha in `alphas`, e.g. for tracing the M(alpha)
  curve of action_plan.md 6.1."""
  return [solve_w_numeric(a, b, alpha, **kwargs) for alpha in alphas]
