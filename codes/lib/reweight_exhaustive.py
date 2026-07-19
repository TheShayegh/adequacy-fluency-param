"""EXHAUSTIVE (brute-force grid) solver for (P) -- see lib.reweight_numeric's
module docstring for what (P) is. Unlike that module (multi-start local
optimization) or lib.reweight_exact (closed-form KKT/support-enumeration
derivation), this one uses NO structure at all: it lays down every point of
a uniform lattice on the whole weight simplex, checks each one directly
against alpha(w) == target (within `tol`, since a discrete lattice will
essentially never land on an equality constraint exactly), and keeps the
minimum-objective (= max-ESS) survivor by plain comparison.

Why this exists despite lib.reweight_exact already being sound+complete:
that module's correctness rests on a long closed-form derivation (secular
equation, curvature certificate, live-pole handling) that is easy to get
subtly wrong in a way its own outputs wouldn't reveal. This module is the
opposite trade: no derivation to get wrong, at the cost of being useless at
real problem sizes. It's a trust anchor for validating the other two
solvers against small, hand-picked or synthetic (a, b, alpha) instances --
e.g. reproducing action_plan.md 3.7's K=4 non-global-KKT-point example, or
a shrunk-down version of a real dataset -- not a drop-in for them. (One
concrete motivation: lib.reweight_numeric.solve_w_numeric, run with only
30 restarts, was found to land on a local optimum -- ESS 4.28 instead of
the reachable 7.37 -- for one of jazh24's leave-one-system-out subsets near
alpha~1; a solver with no restarts to get unlucky with is the natural way
to confirm what the true optimum there actually is, on a small enough
instance.)

Cost is C(grid_n + K - 1, K - 1) lattice points -- combinatorial, and
intentionally not optimized away. That's already too many to finish at
real dataset sizes (K in the teens) for any grid fine enough to mean
anything; this is only ever meant to run on small K (roughly K <= 7-8).
solve_w_exhaustive refuses outright (rather than hang) once the lattice
would exceed `max_candidates`.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from lib.alpha import alpha_min_max

_DEFAULT_GRID_N = 60
_DEFAULT_TOL = 1e-3
_DEFAULT_MAX_CANDIDATES = 20_000_000


@dataclasses.dataclass
class ExhaustiveWResult:
  """Result of one call to solve_w_exhaustive."""
  w: np.ndarray
  alpha_target: float
  alpha_achieved: float
  gap: float                # |alpha_achieved - alpha_target| -- check this before trusting
                              # `ess`: near alpha~0 or alpha~1 (see module docstring), alpha(w)
                              # can be steep enough that a gap near `tol` already reaches a much
                              # easier, higher-ESS point than one actually at alpha_target would.
  success: bool          # some lattice point satisfied |alpha(w) - target| <= tol
  objective: float        # 1/2 * sum(w^2), the (P) objective
  ess: float               # 1 / sum(w^2)
  grid_n: int
  tol: float
  n_candidates: int        # total lattice points checked (the whole simplex lattice)
  n_feasible: int          # how many of those satisfied the tolerance
  message: str


def _simplex_lattice_counts(K: int, grid_n: int, chunk: int = 200_000):
  """Yields (n_in_chunk, K)-shaped integer arrays of nonnegative counts
  summing to grid_n -- every point of the K-dim simplex lattice at
  resolution grid_n (the standard stars-and-bars enumeration), batched so
  the caller can vectorize alpha(w)/objective over each chunk instead of
  evaluating one w at a time."""
  combos = itertools.combinations_with_replacement(range(K), grid_n)
  while True:
    batch = list(itertools.islice(combos, chunk))
    if not batch:
      return
    counts = np.zeros((len(batch), K), dtype=np.int64)
    for row, combo in zip(counts, batch):
      # combo is a length-grid_n multiset of labels in [0,K); tallying it
      # per label gives "how many of grid_n units this label got". combo
      # must be an array/list here, not a bare tuple -- np.add.at treats a
      # plain tuple index as a *multi-dimensional* index (like row[(0,1)]
      # meaning row[0][1]) rather than a sequence of 1-D indices.
      np.add.at(row, list(combo), 1)
    yield counts


def solve_w_exhaustive(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    grid_n: int = _DEFAULT_GRID_N,
    tol: float = _DEFAULT_TOL,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> ExhaustiveWResult:
  """Checks literally every w on the K-dim simplex lattice of resolution
  grid_n (all w_i in {0, 1/grid_n, 2/grid_n, ..., 1} with sum(w)=1) against
  the balance constraint alpha(w) == alpha (within `tol`), and returns the
  one with the smallest objective (largest ESS) among those that pass --
  the brute-force answer to (P), no algorithmic insight applied anywhere.

  Raises ValueError up front (rather than hang) if the lattice would exceed
  `max_candidates` -- this method's cost is combinatorial in K and grid_n
  and that is never optimized away, so refusing is the honest behavior once
  the requested (K, grid_n) makes it infeasible to actually finish.
  """
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  K = len(a)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= alpha <= 1.0):
    raise ValueError(f'alpha={alpha} outside [0,1]')
  amin, amax = alpha_min_max(a, b)
  if not (amin - tol <= alpha <= amax + tol):
    raise ValueError(f'alpha={alpha} outside reachable range [{amin}, {amax}] (Corollary 2)')

  n_lattice = math.comb(grid_n + K - 1, K - 1)
  if n_lattice > max_candidates:
    raise ValueError(
        f'simplex lattice for K={K}, grid_n={grid_n} has {n_lattice:,} points, over '
        f'max_candidates={max_candidates:,} -- this solver is brute-force by design and '
        f'that cost is never optimized away; lower grid_n, lower K (a smaller test '
        f'instance), or raise max_candidates and be prepared to wait.')

  best_obj = None
  best_w = None
  best_achieved = None
  best_gap = None  # |alpha(w) - target| of the closest miss, tracked even off-tolerance
  n_checked = 0
  n_feasible = 0

  for counts in _simplex_lattice_counts(K, grid_n):
    n_checked += counts.shape[0]
    w = counts.astype(float) / grid_n  # (chunk, K), rows are valid simplex points by construction

    # Two-pass (mean-centered) weighted variance per row, vectorized -- the
    # single-pass identity (sum w*x^2 - mu^2) suffers the same catastrophic
    # cancellation lib.alpha.weighted_var's docstring warns about, and nothing
    # here should ever trade robustness for speed (that would defeat the
    # point of a "trusted" solver).
    #
    # errstate suppresses spurious divide/overflow/invalid FloatingPointWarnings
    # that numpy's Accelerate BLAS backend raises on well-formed, finite,
    # contiguous float64 matmuls on Apple Silicon (confirmed here: w has no
    # NaN/Inf and the results agree with solve_w_numeric/solve_w_exact) --
    # not a sign of an actual numerical problem in this computation.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
      mu_a = w @ a
      mu_b = w @ b
    var_a = np.sum(w * (a[None, :] - mu_a[:, None]) ** 2, axis=1)
    var_b = np.sum(w * (b[None, :] - mu_b[:, None]) ** 2, axis=1)
    denom = var_a + var_b
    with np.errstate(invalid='ignore', divide='ignore'):
      achieved = np.where(denom > 0, var_a / denom, np.nan)

    gap = np.abs(achieved - alpha)
    feasible = gap <= tol
    n_feasible += int(np.count_nonzero(feasible))

    objective = 0.5 * np.sum(w ** 2, axis=1)

    if np.any(feasible):
      idx = np.flatnonzero(feasible)
      local_best = idx[np.argmin(objective[idx])]
      if best_obj is None or objective[local_best] < best_obj:
        best_obj = float(objective[local_best])
        best_w = w[local_best].copy()
        best_achieved = float(achieved[local_best])

    if best_obj is None:
      # No feasible point yet this chunk (or ever) -- keep the closest miss
      # around so a failed search still reports something informative.
      valid = ~np.isnan(gap)
      if np.any(valid):
        local_closest = np.flatnonzero(valid)[np.argmin(gap[valid])]
        if best_gap is None or gap[local_closest] < best_gap:
          best_gap = float(gap[local_closest])
          if best_w is None:  # only used for the failure message/fallback w
            best_w = w[local_closest].copy()
            best_achieved = float(achieved[local_closest])

  if best_obj is not None:
    gap = abs(best_achieved - alpha)
    return ExhaustiveWResult(
        w=best_w, alpha_target=alpha, alpha_achieved=best_achieved, gap=gap,
        success=True, objective=best_obj, ess=1.0 / (2.0 * best_obj),
        grid_n=grid_n, tol=tol, n_candidates=n_checked, n_feasible=n_feasible,
        message=f'best of {n_feasible} lattice points within tol={tol} of alpha={alpha} '
                f'(actual gap {gap:.3g})',
    )
  gap = abs(best_achieved - alpha) if best_achieved is not None else float('nan')
  return ExhaustiveWResult(
      w=best_w, alpha_target=alpha, alpha_achieved=best_achieved, gap=gap,
      success=False, objective=float('nan'), ess=float('nan'),
      grid_n=grid_n, tol=tol, n_candidates=n_checked, n_feasible=0,
      message=(f'no lattice point within tol={tol} of alpha={alpha} '
               f'(closest miss off by {best_gap:.3g}) -- raise grid_n or tol'),
  )
