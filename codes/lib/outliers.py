"""Robust (median/MAD-based) outlier detection for a dataset's system-level
adequacy scores -- a companion to lib.alpha_table's leave-one-out
variance-drop check (mean/variance-based, and prone to masking: several
moderate outliers together can each look mild under one-at-a-time removal,
since removing just one still leaves the others inflating variance and
diluting any single system's own % contribution). The modified z-score
(Iglewicz & Hoya 1993) uses the median and MAD (median absolute deviation)
instead of mean/std, so it isn't dragged around by the very outliers it's
trying to find, and flags every outlier in a single pass over the full
sample rather than needing iterative removal.
"""

from __future__ import annotations

import numpy as np

# Iglewicz & Hoya's own recommended threshold and normalizing constant
# (0.6745 = the standard normal distribution's 0.75 quantile, so a
# modified z-score is on roughly the same scale as an ordinary z-score
# for approximately-normal data).
_MZ_CONSTANT = 0.6745
DEFAULT_THRESHOLD = 3.5


def modified_z_scores(x: np.ndarray) -> np.ndarray:
  """0.6745 * (x_i - median(x)) / MAD(x) for every entry of x. All-NaN
  (not a spurious 0) wherever MAD == 0 (every value identical) -- there is
  no meaningful notion of "outlier" in a constant sample."""
  x = np.asarray(x, dtype=float)
  med = np.median(x)
  mad = np.median(np.abs(x - med))
  if mad == 0:
    return np.full(x.shape, np.nan)
  return _MZ_CONSTANT * (x - med) / mad


def mad_outliers(
    systems: list[str], x: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float]]:
  """(system, modified_z_score) for every system whose |modified z-score|
  exceeds `threshold`, sorted by |z| descending (most extreme first)."""
  z = modified_z_scores(x)
  flagged = [(s, float(zi)) for s, zi in zip(systems, z) if not np.isnan(zi) and abs(zi) > threshold]
  return sorted(flagged, key=lambda t: -abs(t[1]))
