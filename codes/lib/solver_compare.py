"""Cross-checks two (P)-solver results (any of lib.reweight_numeric.
solve_w_numeric, lib.reweight_exact.solve_w_exact, lib.reweight_exhaustive.
solve_w_exhaustive, lib.reweight_exhaustive_gpu.solve_w_exhaustive_gpu --
anything returning an object with `.w` and `.success`) against each other
on the same problem.

The naive check (are the two `w` arrays equal?) produces false mismatches:
whenever the optimal objective has ties -- common on a discretized grid,
where many different sparse supports can reach the identical ESS -- two
solvers can each return a different, independently-optimal-and-feasible
`w` for the same (a, b, alpha, tol). That's agreement, not a bug (observed
directly: lib.reweight_exhaustive_gpu with device='cpu' vs device='mps'
on a K=15 synthetic instance at grid_n=10 returned two different supports,
both ESS=10.0 exactly, both independently verified within tol of the
target alpha -- a real tie, not a discrepancy).

So this module compares at the level of "did both solvers reach an
equally-good, independently-VALID answer", not "is the vector identical",
by recomputing everything directly from each solver's raw `w` in a single
fixed precision (float64, via lib.alpha) -- deliberately not trusting
either result's self-reported alpha_achieved/objective/budget_violation/
min_weight fields, since those may have been computed in a different
precision (e.g. float32 on lib.reweight_exhaustive_gpu's MPS path) or
under a different internal convention. "Valid" means a genuine point of
(P)'s feasible region: w sums to 1, w >= 0 (both checked here explicitly,
not assumed just because a solver reports success=True), AND
alpha(w) is within tol of the target.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lib.alpha import alpha as alpha_of


@dataclasses.dataclass
class SolverComparison:
  """Result of comparing two solver outputs on the same (a, b, alpha, tol)."""
  match: bool
  match_type: str    # 'exact' | 'tied_optimum' | 'both_infeasible' | 'mismatch'
  reason: str
  # Canonical (float64, recomputed from w -- not the solvers' own self-reported
  # fields) diagnostics, so a caller can see exactly what was compared:
  objective_1: float
  objective_2: float
  alpha_achieved_1: float
  alpha_achieved_2: float
  budget_violation_1: float   # |sum(w) - 1|
  budget_violation_2: float
  min_weight_1: float          # min(w); should be >= -nonneg_tol at a valid point
  min_weight_2: float
  valid_1: bool                 # simplex constraints AND alpha(w)~=target, all independently checked
  valid_2: bool
  w_max_abs_diff: float          # np.nan if either w is None (e.g. both_infeasible)


def compare_solvers(
    a: np.ndarray,
    b: np.ndarray,
    alpha_target: float,
    tol: float,
    result1,
    result2,
    w_tol: float = 1e-6,
    obj_tol: float = 1e-6,
    budget_tol: float = 1e-6,
    nonneg_tol: float = 1e-8,
) -> SolverComparison:
  """Compares two solver results on the same problem (a, b, alpha_target,
  tol). `result1`/`result2` need only duck-type `.w` (np.ndarray or None)
  and `.success` (bool) -- works across any of the reweight_* solver
  families' result types.

  Returns a SolverComparison whose `match_type` is:
    'mismatch'        -- success flags disagree, OR a claimed-feasible w
                         fails ANY of (P)'s own constraints under
                         independent recomputation (budget, non-negativity,
                         or the alpha-target equality), OR the two
                         objectives differ by more than obj_tol (one
                         solver actually found a better optimum).
    'both_infeasible' -- both solvers report success=False (weak agreement:
                         neither found a feasible point, but this doesn't
                         confirm they'd agree on one if it existed).
    'exact'           -- both succeeded, both independently verified valid
                         and equally optimal, AND w1 ~= w2.
    'tied_optimum'    -- both succeeded, both independently verified valid
                         and equally optimal, but w1 != w2 -- a genuine tie
                         in the optimal objective, not a bug.
  `match` is True for every case except 'mismatch'.
  """
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)

  if result1.success != result2.success:
    return SolverComparison(
        match=False, match_type='mismatch',
        reason=f'success flags disagree: result1.success={result1.success}, '
               f'result2.success={result2.success}',
        objective_1=float('nan'), objective_2=float('nan'),
        alpha_achieved_1=float('nan'), alpha_achieved_2=float('nan'),
        budget_violation_1=float('nan'), budget_violation_2=float('nan'),
        min_weight_1=float('nan'), min_weight_2=float('nan'),
        valid_1=False, valid_2=False, w_max_abs_diff=float('nan'),
    )

  if not result1.success:  # both False
    return SolverComparison(
        match=True, match_type='both_infeasible',
        reason='both solvers report no feasible point found (weak agreement: '
               'this does not confirm they would agree on a solution if one existed)',
        objective_1=float('nan'), objective_2=float('nan'),
        alpha_achieved_1=float('nan'), alpha_achieved_2=float('nan'),
        budget_violation_1=float('nan'), budget_violation_2=float('nan'),
        min_weight_1=float('nan'), min_weight_2=float('nan'),
        valid_1=False, valid_2=False, w_max_abs_diff=float('nan'),
    )

  w1 = np.asarray(result1.w, dtype=float)
  w2 = np.asarray(result2.w, dtype=float)
  if w1.shape != w2.shape:
    raise ValueError(f'w shapes differ ({w1.shape} vs {w2.shape}) -- not the same problem instance')

  achieved1, achieved2 = alpha_of(a, b, w1), alpha_of(a, b, w2)
  budget1, budget2 = abs(float(w1.sum()) - 1.0), abs(float(w2.sum()) - 1.0)
  min_w1, min_w2 = float(w1.min()), float(w2.min())
  objective1, objective2 = 0.5 * float(np.sum(w1 ** 2)), 0.5 * float(np.sum(w2 ** 2))
  w_max_abs_diff = float(np.max(np.abs(w1 - w2)))

  alpha_ok1 = abs(achieved1 - alpha_target) <= tol
  alpha_ok2 = abs(achieved2 - alpha_target) <= tol
  valid1 = alpha_ok1 and budget1 <= budget_tol and min_w1 >= -nonneg_tol
  valid2 = alpha_ok2 and budget2 <= budget_tol and min_w2 >= -nonneg_tol

  if not (valid1 and valid2):
    bad, achieved, budget, min_w = ('result1', achieved1, budget1, min_w1) if not valid1 \
        else ('result2', achieved2, budget2, min_w2)
    return SolverComparison(
        match=False, match_type='mismatch',
        reason=f'{bad} claims success but its w fails (P)\'s own constraints under independent '
               f'recomputation: alpha(w)={achieved:.6g} (target={alpha_target:.6g}, tol={tol:.3g}), '
               f'budget_violation={budget:.3g}, min_weight={min_w:.3g}',
        objective_1=objective1, objective_2=objective2,
        alpha_achieved_1=achieved1, alpha_achieved_2=achieved2,
        budget_violation_1=budget1, budget_violation_2=budget2,
        min_weight_1=min_w1, min_weight_2=min_w2,
        valid_1=valid1, valid_2=valid2, w_max_abs_diff=w_max_abs_diff,
    )

  if abs(objective1 - objective2) > obj_tol:
    better = 'result1' if objective1 < objective2 else 'result2'
    return SolverComparison(
        match=False, match_type='mismatch',
        reason=f'objectives differ beyond obj_tol ({objective1:.6g} vs {objective2:.6g}) -- '
               f'{better} found a strictly better optimum, the other did not find it '
               f'(not a tie -- one solver actually underperformed)',
        objective_1=objective1, objective_2=objective2,
        alpha_achieved_1=achieved1, alpha_achieved_2=achieved2,
        budget_violation_1=budget1, budget_violation_2=budget2,
        min_weight_1=min_w1, min_weight_2=min_w2,
        valid_1=valid1, valid_2=valid2, w_max_abs_diff=w_max_abs_diff,
    )

  match_type = 'exact' if w_max_abs_diff <= w_tol else 'tied_optimum'
  reason = (f'both valid and equally optimal (objective={objective1:.6g} vs {objective2:.6g}); '
            + ('w matches within w_tol' if match_type == 'exact' else
               f'landed on different w within the tied optimal set (max abs diff {w_max_abs_diff:.3g}) '
               f'-- a genuine tie, not a discrepancy'))
  return SolverComparison(
      match=True, match_type=match_type, reason=reason,
      objective_1=objective1, objective_2=objective2,
      alpha_achieved_1=achieved1, alpha_achieved_2=achieved2,
      budget_violation_1=budget1, budget_violation_2=budget2,
      min_weight_1=min_w1, min_weight_2=min_w2,
      valid_1=valid1, valid_2=valid2, w_max_abs_diff=w_max_abs_diff,
  )
