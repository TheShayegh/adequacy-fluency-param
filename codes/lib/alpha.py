"""Core adequacy-fluency balance math (action_plan.md section 3.2, 3.8).

Operates on plain numpy arrays of system-level scores -- no knowledge of
mqm_scoring's loading/parsing machinery -- so it's reusable by both
descriptive analysis (compute_alpha_table.py) and the reweighting solver
(section 3.4 onward: alpha_min/alpha_max bound the feasible range of (P),
and alpha_ij's argmin/argmax pairs are exactly the two-system supports that
attain them, Proposition 2's edge case).
"""

from __future__ import annotations

import numpy as np


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
  return float(np.dot(w, x))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
  """sigma(x; w)^2 = sum_i w_i (x_i - mu)^2 (action_plan.md 3.2's algebraic
  identity sum_i w_i x_i^2 - (sum_i w_i x_i)^2, but computed in this
  mean-centered, two-pass form for numerical stability: the one-pass form
  suffers catastrophic cancellation whenever the variance is small relative
  to mu^2, which happens routinely at low ESS (near-degenerate w
  concentrates on a couple of systems) -- seen in practice as spurious
  tiny-negative variances feeding sqrt() in weighted_pearson. Clipped to
  >=0 as a final guard against residual floating-point noise."""
  mu = weighted_mean(x, w)
  return float(max(0.0, np.dot(w, (x - mu) ** 2)))


def alpha(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
  """alpha(w) = sigma(a;w)^2 / (sigma(a;w)^2 + sigma(b;w)^2) (action_plan.md 3.2)."""
  va = weighted_var(a, w)
  vb = weighted_var(b, w)
  return va / (va + vb)


def alpha_0(a: np.ndarray, b: np.ndarray) -> float:
  """The natural (uniform-weight) balance: alpha(1/K, ..., 1/K)."""
  K = len(a)
  return alpha(a, b, np.full(K, 1.0 / K))


def variances(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
  """(sigma(a)^2, sigma(b)^2) at uniform weight -- the two quantities whose
  ratio alpha_0 reports (action_plan.md 3.2)."""
  K = len(a)
  w = np.full(K, 1.0 / K)
  return weighted_var(a, w), weighted_var(b, w)


def alpha_ij(a_i: float, b_i: float, a_j: float, b_j: float) -> float | None:
  """Pairwise balance of systems i,j (action_plan.md 3.8): the balance of
  the two-system pool {i,j}, independent of how weight is split between
  them (Proposition 2's two-system edge case). None for a coinciding pair
  (no defined pairwise balance, action_plan.md 3.2's non-degeneracy note)."""
  da2 = (a_i - a_j) ** 2
  db2 = (b_i - b_j) ** 2
  denom = da2 + db2
  if denom == 0:
    return None
  return da2 / denom


def pairwise_alphas(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int, float]]:
  """All (i, j, alpha_ij) for i < j, skipping coinciding pairs."""
  K = len(a)
  out = []
  for i in range(K):
    for j in range(i + 1, K):
      v = alpha_ij(a[i], b[i], a[j], b[j])
      if v is not None:
        out.append((i, j, v))
  return out


def alpha_min_max(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
  """Reachable range [alpha_min, alpha_max] of (P) (action_plan.md 3.8,
  Corollary 2): the smallest/largest pairwise balance among all system
  pairs."""
  vals = [v for _, _, v in pairwise_alphas(a, b)]
  return min(vals), max(vals)


def union_alpha_grid(a: np.ndarray, b: np.ndarray, step: float = 0.01) -> list[float]:
  """The alpha grid for the scorer-orientation experiment (lib.synthetic_
  scorer_orientation): the union of every pairwise alpha_ij of D
  (`pairwise_alphas`) with a plain uniform sweep of [alpha_min, alpha_max]
  at `step` -- no adaptive gap-filling, just two grids merged and
  deduplicated (rounded to avoid float-noise near-duplicates). The
  alpha_ij's themselves are exactly the alpha values where some system
  pair's relative contribution to the pooled variance ratio changes
  (action_plan.md 3.8) -- i.e. potential kinks in w*(alpha) (Corollary 2)
  -- so this grid samples those exactly, on top of the uniform sweep's even
  coverage everywhere else. An earlier version filled only the gaps between
  adjacent alpha_ij wider than a threshold, instead of a plain uniform
  sweep; dropped as needless complexity once the simpler union looked just
  as good."""
  alpha_lo, alpha_hi = alpha_min_max(a, b)
  n = int(round((alpha_hi - alpha_lo) / step))
  uniform = np.linspace(alpha_lo, alpha_hi, n + 1)
  alpha_ijs = [v for _, _, v in pairwise_alphas(a, b)]
  return sorted(set(round(v, 10) for v in list(uniform) + alpha_ijs))


def build_alpha_grid(
    lo: float, hi: float, n_grid: int, alpha_0s: dict[str, float] | None = None,
    eps_frac: float = 1e-3,
) -> tuple[np.ndarray, list[str]]:
  """The n_grid-point evenly-spaced sweep over (lo, hi) every compute_*_
  cache.py script builds (inset by eps_frac*(hi-lo) at each end, so the
  endpoints themselves -- exactly alpha_min/alpha_max of the tightest
  member of the group -- are never targeted, since only a single
  degenerate 2-system support is admissible right at them), UNIONED with
  each entry's own natural (uniform-weight) alpha_0 (name -> value) that
  lands strictly inside that same inset window -- so a point where some
  dataset/subset's uniform weighting is already exactly optimal (Lemma 3's
  anchor case, w=1/K) gets solved exactly rather than only approximated by
  its nearest linspace neighbor.

  Returns (sorted unique alphas, names of any alpha_0s dropped for falling
  outside the inset window -- can't be added since some OTHER member of
  the group isn't jointly solvable there)."""
  eps = (hi - lo) * eps_frac
  grid_lo, grid_hi = lo + eps, hi - eps
  base = np.linspace(grid_lo, grid_hi, n_grid)
  extra, dropped = [], []
  for name, a0 in (alpha_0s or {}).items():
    if grid_lo <= a0 <= grid_hi:
      extra.append(a0)
    else:
      dropped.append(name)
  alphas = np.unique(np.concatenate([base, extra])) if extra else base
  return alphas, dropped


def compute_alpha_ij_grid(
    a_pool: np.ndarray, b_pool: np.ndarray, subsets: list[tuple[int, ...]], eps_frac: float = 1e-3,
    trim_frac: float = 0.0,
) -> tuple[list[float], float, float]:
  """The shared alpha grid: every pairwise alpha_ij of the FULL pool
  (a_pool, b_pool), restricted to the range every one of `subsets` can
  jointly reach (their own Corollary 2 ranges, intersected -- same
  eps-inset convention as build_alpha_grid, so the exact boundary itself --
  where only a single degenerate 2-system support is admissible -- is
  excluded). A subset's own pairwise alphas are always a SUBSET of the full
  pool's C(K,2) (any pair inside a sampled subset is also a pair of the
  full pool), so this is well-defined and, by construction, every returned
  alpha is within every sampled subset's own reachable range too.

  trim_frac additionally shrinks the common range by trim_frac on EACH
  side (e.g. 1/6 leaves the middle 2/3) before the eps-inset and filter --
  a cheaper preview knob, independent of eps_frac's boundary-degeneracy
  exclusion."""
  los, his = [], []
  for s in subsets:
    lo, hi = alpha_min_max(a_pool[list(s)], b_pool[list(s)])
    los.append(lo)
    his.append(hi)
  lo_common, hi_common = max(los), min(his)
  span = hi_common - lo_common
  lo_trim, hi_trim = lo_common + trim_frac * span, hi_common - trim_frac * span
  eps = (hi_trim - lo_trim) * eps_frac
  alpha_ijs = sorted(set(v for _, _, v in pairwise_alphas(a_pool, b_pool)))
  grid = [v for v in alpha_ijs if lo_trim + eps <= v <= hi_trim - eps]
  return grid, lo_trim, hi_trim
