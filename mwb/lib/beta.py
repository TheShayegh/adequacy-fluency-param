"""Core adequacy-fluency balance math (paper section 3: "The Balance
Parameter" and "Reweighting as an Optimization Problem").

Operates on plain numpy arrays of system-level scores -- no knowledge of
mqm_scoring's loading/parsing machinery -- so it's reusable by both
descriptive analysis and the reweighting solver: beta_min/beta_max bound
the feasible range of the reweighting problem (Theorem "Reachable range"),
and beta_ij's argmin/argmax pairs are exactly the two-system supports that
attain them.

Also holds beta_to_beta_std/beta_std_to_beta and friends: the paper's
footnote reparameterization beta_std = 1/(1+sqrt(1/beta-1)) -- a
*different* quantity from beta itself, not used in any paper table.
"""

from __future__ import annotations

import numpy as np


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
  return float(np.dot(w, x))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
  """sigma(x; w)^2 = sum_i w_i (x_i - mu)^2, computed in this mean-centered,
  two-pass form for numerical stability: the one-pass form (sum_i w_i x_i^2
  - (sum_i w_i x_i)^2) suffers catastrophic cancellation whenever the
  variance is small relative to mu^2, which happens routinely at low ESS
  (near-degenerate w concentrates on a couple of systems) -- seen in
  practice as spurious tiny-negative variances feeding sqrt() downstream.
  Clipped to >=0 as a final guard against residual floating-point noise."""
  mu = weighted_mean(x, w)
  return float(max(0.0, np.dot(w, (x - mu) ** 2)))


def beta(a: np.ndarray, f: np.ndarray, w: np.ndarray) -> float:
  """beta(w) = sigma(a;w)^2 / (sigma(a;w)^2 + sigma(f;w)^2), Eq. (*)."""
  va = weighted_var(a, w)
  vf = weighted_var(f, w)
  return va / (va + vf)


def beta_0(a: np.ndarray, f: np.ndarray) -> float:
  """The natural (uniform-weight) balance: beta(1/K, ..., 1/K) -- the
  paper's beta_0."""
  K = len(a)
  return beta(a, f, np.full(K, 1.0 / K))


def variances(a: np.ndarray, f: np.ndarray) -> tuple[float, float]:
  """(sigma(a)^2, sigma(f)^2) at uniform weight -- the two quantities whose
  ratio beta_0 reports."""
  K = len(a)
  w = np.full(K, 1.0 / K)
  return weighted_var(a, w), weighted_var(f, w)


def beta_ij(a_i: float, f_i: float, a_j: float, f_j: float) -> float | None:
  """Pairwise balance of systems i,j (Eq. beta_ij): the balance of the
  two-system pool {i,j}, independent of how weight is split between them
  (a two-system support's beta(w) is constant on its simplex). None for a
  coinciding pair (no defined pairwise balance -- the paper assumes no two
  systems share the same (a_k,f_k))."""
  da2 = (a_i - a_j) ** 2
  df2 = (f_i - f_j) ** 2
  denom = da2 + df2
  if denom == 0:
    return None
  return da2 / denom


def pairwise_betas(a: np.ndarray, f: np.ndarray) -> list[tuple[int, int, float]]:
  """All (i, j, beta_ij) for i < j, skipping coinciding pairs."""
  K = len(a)
  out = []
  for i in range(K):
    for j in range(i + 1, K):
      v = beta_ij(a[i], f[i], a[j], f[j])
      if v is not None:
        out.append((i, j, v))
  return out


def beta_min_max(a: np.ndarray, f: np.ndarray) -> tuple[float, float]:
  """Reachable range [beta_min, beta_max] of the reweighting problem
  (Theorem "Reachable range"): the smallest/largest pairwise balance among
  all system pairs."""
  vals = [v for _, _, v in pairwise_betas(a, f)]
  return min(vals), max(vals)


def union_beta_grid(a: np.ndarray, f: np.ndarray, step: float = 0.01) -> list[float]:
  """The beta grid used by the scorer-preference experiment: the union of
  every pairwise beta_ij of the dataset (`pairwise_betas`) with a plain
  uniform sweep of [beta_min, beta_max] at `step` -- no adaptive
  gap-filling, just two grids merged and deduplicated (rounded to avoid
  float-noise near-duplicates). The beta_ij's themselves are exactly the
  beta values where some system pair's relative contribution to the
  pooled variance ratio changes -- i.e. potential kinks in the optimal
  w*(beta) -- so this grid samples those exactly, on top of the uniform
  sweep's even coverage everywhere else."""
  beta_lo, beta_hi = beta_min_max(a, f)
  n = int(round((beta_hi - beta_lo) / step))
  uniform = np.linspace(beta_lo, beta_hi, n + 1)
  beta_ijs = [v for _, _, v in pairwise_betas(a, f)]
  return sorted(set(round(v, 10) for v in list(uniform) + beta_ijs))


def beta_to_beta_std(beta):
  """beta_std = 1 / (1 + sqrt(1/beta - 1)) -- the paper's footnote
  reparameterization of beta that's monotonically increasing over (0, 1]
  with a fixed point at 0.5 (beta_std(0.5)=0.5, beta_std(1)=1, beta_std->0
  as beta->0+), used by --beta-std-axis plotting options. Vectorized
  (accepts a scalar or an array); beta<=0 divides by zero and returns
  inf/nan via ordinary numpy float semantics (RuntimeWarning suppressed,
  not an error) rather than raising, matching beta_std_to_beta's own
  handling of its symmetric edge case at beta_std=0."""
  beta = np.asarray(beta, dtype=float)
  with np.errstate(divide='ignore', invalid='ignore'):
    return 1.0 / (1.0 + np.sqrt(1.0 / beta - 1.0))


def beta_std_to_beta(beta_std):
  """Inverse of beta_to_beta_std: beta = 1 / (1 + ((1-beta_std)/beta_std)^2)
  -- solved directly from beta_std = 1/(1+sqrt(1/beta-1)) by isolating
  beta. beta_std=0 divides by zero; ordinary numpy float semantics send
  that to beta=0 (the correct limit: (1-0)/0 = inf, inf^2 = inf,
  1/(1+inf) = 0), with the RuntimeWarning suppressed rather than raised."""
  beta_std = np.asarray(beta_std, dtype=float)
  with np.errstate(divide='ignore', invalid='ignore'):
    ratio = (1.0 - beta_std) / beta_std
    return 1.0 / (1.0 + ratio ** 2)


def beta_grid_for_beta_std(
    beta_std_stepsize: float = 0.05, beta_std_lo: float = 0.0, beta_std_hi: float = 1.0,
) -> list[float]:
  """List of beta values corresponding to a beta_std grid evenly spaced at
  `beta_std_stepsize` over [beta_std_lo, beta_std_hi]. Each grid point is
  mapped back to its beta via beta_std_to_beta; since that map is
  monotonically increasing, the returned list is already ascending in
  beta, matching every other beta-grid builder's own convention -- no
  re-sort needed. See union_beta_grid_beta_std below for a caller that
  passes in a dataset's own reachable range translated into beta_std-space
  via beta_to_beta_std."""
  n = int(round((beta_std_hi - beta_std_lo) / beta_std_stepsize))
  beta_stds = np.array([beta_std_lo + k * beta_std_stepsize for k in range(n + 1)])
  return [round(float(v), 10) for v in beta_std_to_beta(beta_stds)]


def union_beta_grid_beta_std(a: np.ndarray, f: np.ndarray, beta_std_stepsize: float = 0.05) -> list[float]:
  """union_beta_grid's exact structure (the union of every pairwise
  beta_ij with a sweep of [beta_min, beta_max], deduplicated) EXCEPT the
  sweep half is evenly spaced in beta_std space (beta_grid_for_beta_std)
  instead of beta space -- the beta_ij union stays, only the
  uniform-sweep component changes. [beta_lo, beta_hi] (this dataset's own
  reachable range, beta_min_max) is translated to beta_std-space via
  beta_to_beta_std before building the beta_std sweep, so the resulting
  beta values land inside the dataset's own reachable range exactly like
  union_beta_grid's np.linspace(beta_lo, beta_hi, ...) does (including
  the exact endpoints -- union_beta_grid doesn't eps-inset either)."""
  beta_lo, beta_hi = beta_min_max(a, f)
  beta_std_lo, beta_std_hi = float(beta_to_beta_std(beta_lo)), float(beta_to_beta_std(beta_hi))
  beta_std_sweep = beta_grid_for_beta_std(
      beta_std_stepsize=beta_std_stepsize, beta_std_lo=beta_std_lo, beta_std_hi=beta_std_hi)
  beta_ijs = [v for _, _, v in pairwise_betas(a, f)]
  return sorted(set(round(v, 10) for v in beta_std_sweep + beta_ijs))


def build_beta_grid(
    lo: float, hi: float, n_grid: int, beta_0s: dict[str, float] | None = None,
    eps_frac: float = 1e-3,
) -> tuple[np.ndarray, list[str]]:
  """The n_grid-point evenly-spaced sweep over (lo, hi) used by the compute_*
  cache scripts (inset by eps_frac*(hi-lo) at each end, so the endpoints
  themselves -- exactly beta_min/beta_max of the tightest member of the
  group -- are never targeted, since only a single degenerate 2-system
  support is admissible right at them), UNIONED with each entry's own
  natural (uniform-weight) beta_0 (name -> value) that lands strictly
  inside that same inset window -- so a point where some dataset/subset's
  uniform weighting is already exactly optimal (w=1/K) gets solved exactly
  rather than only approximated by its nearest linspace neighbor.

  Returns (sorted unique betas, names of any beta_0s dropped for falling
  outside the inset window -- can't be added since some OTHER member of
  the group isn't jointly solvable there)."""
  eps = (hi - lo) * eps_frac
  grid_lo, grid_hi = lo + eps, hi - eps
  base = np.linspace(grid_lo, grid_hi, n_grid)
  extra, dropped = [], []
  for name, b0 in (beta_0s or {}).items():
    if grid_lo <= b0 <= grid_hi:
      extra.append(b0)
    else:
      dropped.append(name)
  betas = np.unique(np.concatenate([base, extra])) if extra else base
  return betas, dropped
