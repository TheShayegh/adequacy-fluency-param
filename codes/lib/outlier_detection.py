"""Iterative wrapper around lib.outliers.mad_outliers: even the MAD-based
modified z-score, robust as it is, can still mask a second outlier when
one is extreme enough to distort the pool's own median/MAD before it's
removed (observed directly this session -- IKUN only reads as an outlier
in ende24 once MSLC is already gone; IKUN-C's z-score in jazh24 jumps from
-2.88 to -4.23, crossing the flag threshold, only after MSLC is removed
first). A single MAD pass over the full pool doesn't see that, since
MSLC's own extremeness is still baked into the median/MAD it's computed
against.

This module is the fix: repeat the MAD pass on the shrinking pool,
dropping every system flagged in one pass before recomputing on what's
left, until a pass flags nothing.

Also home to studentized_outlier_test: a generic (not MAD-specific)
regression-outlier test for a DIFFERENT failure mode found in type 7 --
some systems (e.g. zhen23's ONLINE-Y) are outliers in the robustness-vs-
delta_alpha_0 relationship itself (unusually disruptive to the ranking
given how little they shift alpha_0), which the adequacy-score-based MAD
test has no way to see, since it never looks at robustness/tau at all.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from lib.outliers import DEFAULT_THRESHOLD, mad_outliers


def iterative_mad_outliers(
    systems: list[str], x: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float, int]]:
  """(system, modified_z_score, iteration) for every system removed across
  repeated MAD passes -- iteration 1 is systems flagged on the full pool,
  iteration 2 on the pool with iteration 1's systems already removed, etc.
  Each pass flags (and removes) every system over `threshold` at once, not
  just the single worst one; stops the first pass that flags nothing.
  Order: iteration ascending, then |z| descending within an iteration
  (matching mad_outliers' own most-extreme-first convention)."""
  x = np.asarray(x, dtype=float)
  remaining_systems = list(systems)
  remaining_x = x.copy()
  removed = []
  iteration = 1
  while True:
    flagged = mad_outliers(remaining_systems, remaining_x, threshold=threshold)
    if not flagged:
      return removed
    removed.extend((s, z, iteration) for s, z in flagged)
    flagged_names = {s for s, _ in flagged}
    keep = [i for i, s in enumerate(remaining_systems) if s not in flagged_names]
    remaining_systems = [remaining_systems[i] for i in keep]
    remaining_x = remaining_x[keep]
    iteration += 1
    if len(remaining_systems) < 2:
      return removed


def studentized_outlier_test(
    labels: list[str], xs: np.ndarray, ys: np.ndarray, alpha: float = 0.05,
) -> list[tuple[str, float, float, float]]:
  """Externally studentized (jackknife) residuals from an OLS fit of
  ys ~ xs, with a Bonferroni-corrected outlier test (the same method as
  R's car::outlierTest) -- flags points that are outliers in y GIVEN
  their x, as opposed to leverage points (outliers in x). Needs n >= 5 to
  be meaningful; returns [] below that.

  Returns (label, residual, t_stat, bonferroni_p) for every point with
  bonferroni_p < alpha, sorted by |t_stat| descending."""
  xs = np.asarray(xs, dtype=float)
  ys = np.asarray(ys, dtype=float)
  n = len(xs)
  if n < 5:
    return []
  slope, intercept = np.polyfit(xs, ys, 1)
  resid = ys - (slope * xs + intercept)
  xbar = xs.mean()
  Sxx = np.sum((xs - xbar) ** 2)
  h = 1.0 / n + (xs - xbar) ** 2 / Sxx
  dof = n - 2
  s = np.sqrt(np.sum(resid ** 2) / dof)
  internal = resid / (s * np.sqrt(1 - h) + 1e-300)
  ext_dof = n - 3
  ext = internal * np.sqrt(np.maximum(ext_dof / (ext_dof + 1 - internal ** 2 + 1e-12), 0))
  pvals = 2 * (1 - _stats.t.cdf(np.abs(ext), df=ext_dof))
  bonf_p = pvals * n
  flagged = [
      (lab, float(r), float(t), float(bp))
      for lab, r, t, bp in zip(labels, resid, ext, bonf_p) if bp < alpha
  ]
  return sorted(flagged, key=lambda row: -abs(row[2]))
