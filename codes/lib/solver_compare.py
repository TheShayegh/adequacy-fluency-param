"""Cross-checks two (P)-solver results (any of lib.reweight_exact.
solve_w_exact, lib.reweight_exhaustive.solve_w_exhaustive -- anything
returning an object with `.w` and `.success`) against each other on the
same problem.

The naive check (are the two `w` arrays equal?) produces false mismatches:
whenever the optimal objective has ties -- common on a discretized grid,
where many different sparse supports can reach the identical ESS -- two
solvers can each return a different, independently-optimal-and-feasible
`w` for the same (a, b, alpha, tol). That's agreement, not a bug (observed
directly: lib.reweight_exhaustive's solve_w_exhaustive with device='cpu'
vs device='mps' on a K=15 synthetic instance at grid_n=10 returned two
different supports, both ESS=10.0 exactly, both independently verified
within tol of the target alpha -- a real tie, not a discrepancy).

So this module compares at the level of "did both solvers reach an
equally-good, independently-VALID answer", not "is the vector identical",
by recomputing everything directly from each solver's raw `w` in a single
fixed precision (float64, via lib.alpha) -- deliberately not trusting
either result's self-reported alpha_achieved/objective/budget_violation/
min_weight fields, since those may have been computed in a different
precision (e.g. float32 on lib.reweight_exhaustive's MPS path) or under a
different internal convention. "Valid" means a genuine point of (P)'s
feasible region: w sums to 1, w >= 0 (both checked here explicitly, not
assumed just because a solver reports success=True), AND alpha(w) is
within tol of the target.

Two entry points:
  compare_solvers            -- two solvers, SAME nominal target alpha.
                                 Handles tied optima (different w, equally
                                 valid/optimal) as a match, not a mismatch.
  cross_validate_via_exhaustive -- a continuous solver vs. a trusted
                                 exhaustive-grid result, re-anchored at
                                 exhaustive's ACTUAL achieved alpha (not the
                                 nominal target) before comparing -- removes
                                 the grid-tolerance-gap confound that makes
                                 a naive same-target comparison produce
                                 false "violations" near steep regions of
                                 alpha(w) (see its own docstring for the
                                 concrete incident that motivated it). Use
                                 this one when validating a continuous
                                 solver against lib.reweight_exhaustive
                                 specifically -- it catches real bugs that
                                 compare_solvers' same-target check cannot
                                 distinguish from tolerance artifacts.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

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


@dataclasses.dataclass
class ExhaustiveCrossCheck:
  """Result of cross_validate_via_exhaustive."""
  match: bool
  match_type: str    # 'match' | 'mismatch' | 'exhaustive_inconclusive'
  reason: str
  exhaustive_ess: float
  exhaustive_achieved_alpha: float    # recomputed from exhaustive_result.w, not self-reported
  resolved_ess: float                  # `solver`'s ESS when re-targeted at exhaustive_achieved_alpha
  resolved_achieved_alpha: float        # recomputed from the re-solve's w
  resolved_valid: bool                   # re-solve's w independently verified as a genuine (P) point
  ess_gap: float                          # resolved_ess - exhaustive_ess (>= -obj_tol expected)


def cross_validate_via_exhaustive(
    a: np.ndarray,
    b: np.ndarray,
    exhaustive_result,
    solver: Callable = None,
    solver_kwargs: dict | None = None,
    obj_tol: float = 1e-6,
    alpha_tol: float = 1e-6,
    budget_tol: float = 1e-6,
    nonneg_tol: float = 1e-8,
) -> ExhaustiveCrossCheck:
  """Cross-validates any continuous solver (default lib.reweight_exact.
  solve_w_exact) against a trusted lib.reweight_exhaustive.solve_w_exhaustive
  result, WITHOUT the false-positive trap of comparing them at the nominal
  target alpha directly.

  Why not just compare at the shared target alpha: solve_w_exhaustive is
  grid-quantized, so its accepted point almost never sits at EXACTLY the
  target -- it's merely within `tol` of it (see that module's `.gap` field).
  Comparing ESS at the nominal target then confounds two different things:
  "did the continuous solver find a worse point at the true target" (a real
  bug) vs. "exhaustive's accepted point actually sits at a nearby, easier
  alpha" (not a bug at all -- see lib.reweight_exhaustive's own docstring on
  why `.gap` must be checked before trusting `.ess`). Concretely: comparing
  lib.reweight_exact.solve_w_exact against solve_w_exhaustive at nominal
  target alphas across K=4/6/7/8 real-data sweeps initially showed 84/100
  "exhaustive beats exact" cases; every single one had exhaustive's `.gap`
  nonzero (its accepted point sat at a measurably different alpha, easiest
  to see near alpha_min/alpha_max where alpha(w) is steepest -- exactly
  where this module's own module docstring already warns ESS is most
  sensitive to a small alpha error).

  This function instead: (1) recomputes exhaustive's ACTUAL achieved alpha
  directly from its raw w (never trusting self-reported fields), (2)
  re-solves `solver` at THAT alpha (not the original nominal target) so both
  sides are now unambiguously targeting the identical constraint, then (3)
  compares ESS there. Since solve_w_exact's feasible region strictly
  contains any exhaustive grid at the same target, a correct continuous
  solver must match or beat exhaustive once this confound is removed --
  applying this to the same 84 apparent violations left zero real ones,
  every re-solved case matching or exceeding exhaustive's ESS at its own
  actual alpha (e.g. one case: exact 2.02604 @ nominal target vs exhaustive
  2.06718 @ its actual achieved alpha 0.938057 -- looks like a violation;
  re-solving exact AT 0.938057 gives 2.06765, matching/exceeding exhaustive
  -- confirming the earlier "violation" was purely the target mismatch).

  `exhaustive_result` needs `.w` and `.success` (duck-typed, works with any
  lib.reweight_exhaustive.ExhaustiveWResult). `solver` defaults to lib.
  reweight_exact.solve_w_exact if not given (deferred import to avoid a
  hard dependency for callers cross-validating some other solver instead).
  """
  if solver is None:
    from lib.reweight_exact import solve_w_exact
    solver = solve_w_exact
  solver_kwargs = solver_kwargs or {}

  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)

  if not exhaustive_result.success:
    return ExhaustiveCrossCheck(
        match=True, match_type='exhaustive_inconclusive',
        reason='exhaustive_result.success=False (no grid point found within its tol) -- '
               'nothing to cross-validate against',
        exhaustive_ess=float('nan'), exhaustive_achieved_alpha=float('nan'),
        resolved_ess=float('nan'), resolved_achieved_alpha=float('nan'),
        resolved_valid=False, ess_gap=float('nan'),
    )

  w_exh = np.asarray(exhaustive_result.w, dtype=float)
  exhaustive_alpha = alpha_of(a, b, w_exh)
  exhaustive_ess = 1.0 / float(np.sum(w_exh ** 2))

  resolved = solver(a, b, exhaustive_alpha, **solver_kwargs)
  w_res = np.asarray(resolved.w, dtype=float)
  resolved_alpha = alpha_of(a, b, w_res)
  resolved_ess = 1.0 / float(np.sum(w_res ** 2))
  budget_v = abs(float(w_res.sum()) - 1.0)
  min_w = float(w_res.min())
  resolved_valid = (abs(resolved_alpha - exhaustive_alpha) <= alpha_tol
                     and budget_v <= budget_tol and min_w >= -nonneg_tol)

  ess_gap = resolved_ess - exhaustive_ess

  if not resolved_valid:
    return ExhaustiveCrossCheck(
        match=False, match_type='mismatch',
        reason=f're-solve at exhaustive\'s actual achieved alpha ({exhaustive_alpha:.6g}) produced '
               f'an invalid w under independent recomputation: alpha(w)={resolved_alpha:.6g}, '
               f'budget_violation={budget_v:.3g}, min_weight={min_w:.3g}',
        exhaustive_ess=exhaustive_ess, exhaustive_achieved_alpha=exhaustive_alpha,
        resolved_ess=resolved_ess, resolved_achieved_alpha=resolved_alpha,
        resolved_valid=resolved_valid, ess_gap=ess_gap,
    )

  if ess_gap < -obj_tol:
    return ExhaustiveCrossCheck(
        match=False, match_type='mismatch',
        reason=f'at the SAME alpha ({exhaustive_alpha:.6g}, exhaustive\'s own actual achieved value), '
               f'`solver` found ess={resolved_ess:.6g} but exhaustive found ess={exhaustive_ess:.6g} '
               f'-- a coarser discretization of solver\'s own feasible region beat it, which is only '
               f'possible if solver missed the true optimum (a real bug, not a tolerance artifact, '
               f'since the target-mismatch confound has been eliminated)',
        exhaustive_ess=exhaustive_ess, exhaustive_achieved_alpha=exhaustive_alpha,
        resolved_ess=resolved_ess, resolved_achieved_alpha=resolved_alpha,
        resolved_valid=resolved_valid, ess_gap=ess_gap,
    )

  return ExhaustiveCrossCheck(
      match=True, match_type='match',
      reason=f'at exhaustive\'s actual achieved alpha ({exhaustive_alpha:.6g}), solver matched or '
             f'exceeded it (resolved ess={resolved_ess:.6g} vs exhaustive ess={exhaustive_ess:.6g})',
      exhaustive_ess=exhaustive_ess, exhaustive_achieved_alpha=exhaustive_alpha,
      resolved_ess=resolved_ess, resolved_achieved_alpha=resolved_alpha,
      resolved_valid=resolved_valid, ess_gap=ess_gap,
  )
