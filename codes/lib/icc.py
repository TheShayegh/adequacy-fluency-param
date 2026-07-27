"""Intraclass correlation for generalizability-theory reliability -- pure
math, no I/O (parallel to lib/alpha.py). Used as the replacement
robustness/consistency statistic in lib/subsampling_stability.py:
scorers are the objects, B subsamples are the raters.
"""

from __future__ import annotations

import numpy as np


def icc_c1(X: np.ndarray) -> float:
  """ICC(C,1) / ICC(3,1) (Shrout & Fleiss 1979; "C" for consistency,
  McGraw & Wong 1996) -- two-way ANOVA without replication. X[s, b]:
  object s (row) rated once by rater b (column), no missing cells.

  Standard two-way sums of squares, with an EXPLICIT grand-mean
  subtraction (not a running/nested-loop variance -- centering first keeps
  the sums well-conditioned):
    SS_total = sum((X - grand_mean)^2)
    SS_rows  = k * sum((row_mean - grand_mean)^2)   -- between-objects
    SS_cols  = n * sum((col_mean - grand_mean)^2)   -- between-raters
    SS_error = SS_total - SS_rows - SS_cols         -- object x rater interaction

  Mean squares divide each SS by its degrees of freedom:
    BMS = SS_rows  / (n - 1)
    EMS = SS_error / ((n - 1) * (k - 1))

  ICC(C,1) deliberately excludes JMS (the rater/column mean square): a
  systematic level shift across raters (e.g. reweighting changing every
  subsample's overall score scale) does not count against "consistency"
  reliability -- only "agreement"-type ICC(A,1) would penalize that, and
  that's not what was asked for here.

    ICC(C,1) = (BMS - EMS) / (BMS + (k - 1) * EMS)

  Algebraically identical to the variance-component form S / (S + E), with
  S = (BMS - EMS) / k the between-objects true-score variance component
  and E = EMS the error variance component -- see _icc_c1_variance_
  components below and the self-test at the bottom of this file, which
  assert the two independently-derived forms agree to floating tolerance
  on random input (catches a df or divisor slip immediately)."""
  X = np.asarray(X, dtype=float)
  if X.ndim != 2:
    raise ValueError(f'icc_c1 expects a 2-D (objects x raters) array, got shape {X.shape}')
  n, k = X.shape
  if n < 2 or k < 2:
    raise ValueError(f'icc_c1 needs at least 2 objects and 2 raters, got shape {X.shape}')

  grand_mean = X.mean()
  row_means = X.mean(axis=1)
  col_means = X.mean(axis=0)
  ss_total = np.sum((X - grand_mean) ** 2)
  ss_rows = k * np.sum((row_means - grand_mean) ** 2)
  ss_cols = n * np.sum((col_means - grand_mean) ** 2)
  ss_error = ss_total - ss_rows - ss_cols

  bms = ss_rows / (n - 1)
  ems = ss_error / ((n - 1) * (k - 1))
  denom = bms + (k - 1) * ems
  if denom == 0:
    return float('nan')
  return float((bms - ems) / denom)


def _icc_c1_variance_components(X: np.ndarray) -> float:
  """Independently-derived cross-check for icc_c1, via explicit variance
  components S / (S + E): S = (BMS - EMS) / k (between-objects true-score
  variance), E = EMS (error variance). Not a second public entry point --
  exists solely so the self-test below can catch a df/divisor slip in
  icc_c1 by comparing two algebraically-equivalent but separately-written
  computations."""
  X = np.asarray(X, dtype=float)
  n, k = X.shape
  grand_mean = X.mean()
  row_means = X.mean(axis=1)
  col_means = X.mean(axis=0)
  ss_total = np.sum((X - grand_mean) ** 2)
  ss_rows = k * np.sum((row_means - grand_mean) ** 2)
  ss_cols = n * np.sum((col_means - grand_mean) ** 2)
  ss_error = ss_total - ss_rows - ss_cols

  bms = ss_rows / (n - 1)
  ems = ss_error / ((n - 1) * (k - 1))
  S = (bms - ems) / k
  E = ems
  if S + E == 0:
    return float('nan')
  return float(S / (S + E))


def _self_test(n_trials: int = 200, seed: int = 0) -> None:
  """Asserts the mean-square form (icc_c1) and the variance-component form
  (_icc_c1_variance_components) agree to floating tolerance on random
  (n, k) matrices -- a cheap invariant that would immediately fail on a df
  or divisor slip in either implementation."""
  rng = np.random.default_rng(seed)
  n_checked = 0
  for trial in range(n_trials):
    n = int(rng.integers(2, 30))
    k = int(rng.integers(2, 30))
    X = rng.normal(size=(n, k)) * rng.uniform(0.1, 5.0) + rng.uniform(-10, 10)
    a = icc_c1(X)
    b = _icc_c1_variance_components(X)
    if a == a and b == b:  # skip NaN-degenerate draws (denom == 0), both forms agree it's degenerate
      assert abs(a - b) < 1e-9, (
          f'trial {trial}: mean-square form {a!r} != variance-component form {b!r} (n={n}, k={k})')
      n_checked += 1
  print(f'icc_c1 self-test: mean-square and variance-component forms agree '
        f'on {n_checked}/{n_trials} random trials (n,k up to 30).')


if __name__ == '__main__':
  _self_test()
