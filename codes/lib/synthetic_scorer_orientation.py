"""Adequacy/fluency ORIENTATION of a weighted meta-metric (material/
synthetic_scorer_construction.md's dial families, material/action_plan.md's
alpha reweighting): for one base donor scorer s_0, build its synthetic
A-family and B-family (lib.synthetic_scorer_alpha_grid.donor_alpha_dial_grid)
over the full dial grid ([0, 1] by 0.1, 11 points -- the complete section-4
dial sweep, real donor at dial=0 through the pure aspect-conditional-mean
scorer at dial=1), and ask, at a fixed weighted meta-metric (e.g. weighted
SPA at one alpha): across every pair of same-family dial scorers, how often
does the meta-metric prefer the MORE EXTREME one -- the dial further from
the real donor toward the pure aspect-conditional-mean scorer?

  adequacy_orientation(metametric, s_0) = orientation_score of the A-family's
  scores under `metametric` -- averaged over all C(11,2)=55 same-family pairs.
  fluency_orientation(metametric, s_0) is the B-family's mirror.

1.0 means the metametric always rewards pushing further toward pure
adequacy (fluency) response; 0.0 means it always penalizes it; 0.5 means no
systematic preference either way. Averaging these over every donor in a
dataset gives the dataset-level <adequacy|fluency>_orientation(metametric; D).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

DIAL_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def full_range_alphas(alpha_lo: float, alpha_hi: float, step: float, eps_frac: float = 1e-3) -> list[float]:
  """Every clean multiple of `step` strictly inside [alpha_lo, alpha_hi],
  inset by eps_frac*(hi-lo) at each end (lib.alpha.build_alpha_grid's own
  convention -- the exact boundary admits only a single degenerate
  2-system support, so solve_w_exact is best kept off it). Snapped to
  clean multiples of `step` (e.g. 0.01, 0.02, ...) rather than starting
  exactly at the inset lo, for readable axis ticks."""
  eps = (alpha_hi - alpha_lo) * eps_frac
  lo, hi = alpha_lo + eps, alpha_hi - eps
  start = math.ceil(lo / step) * step
  n = int(math.floor((hi - start) / step + 1e-9))
  return [round(start + k * step, 10) for k in range(n + 1)]


def orientation_tag(dataset: str, n_steps: int, step: float, full_range: bool, metametric: str = 'spa') -> str:
  """Filename tag shared by compute_scorer_orientation_vs_alpha.py (writer)
  and plot_scorer_orientation_vs_alpha.py (reader), so the same CLI flags
  resolve to the same cache file on both sides. metametric='spa' adds no
  suffix (keeps existing spa cache filenames from before this parameter
  existed valid); any other metametric (e.g. 'pa') gets its own suffix so
  it never collides with an spa cache for the same dataset/grid."""
  suffix = '' if metametric == 'spa' else f'_{metametric}'
  if full_range:
    return f'{dataset}_full_s{step:g}{suffix}'
  return f'{dataset}_n{n_steps}_s{step:g}{suffix}'


def save_orientation_data(
    path: str, *, dataset: str, alphas, center_alpha: float, K: int, donors: list[str],
    A: np.ndarray, B: np.ndarray, ess: np.ndarray, dial_grid, metametric: str = 'spa',
) -> None:
  """Persists everything plot_scorer_orientation_vs_alpha.py needs: A/B are
  (n_donors, n_alpha) matrices of per-donor orientation curves, row order ==
  `donors`; ess is (n_alpha,) ESS(w*(alpha)) (action_plan.md section 5) --
  depends only on alpha, not on donor, since w*(alpha) is shared across
  every donor (lib.reweight_exact.solve_w_exact solved once per alpha).
  metametric records which weighted meta-metric ('spa' or 'pa') A/B were
  scored with (lib.synthetic_scorer_alpha_grid.donor_alpha_dial_grid)."""
  np.savez(
      path, dataset=np.asarray(dataset), alphas=np.asarray(alphas, dtype=float),
      center_alpha=np.asarray(float(center_alpha)), K=np.asarray(int(K)),
      donors=np.asarray(donors, dtype='<U128'), A=np.asarray(A, dtype=float),
      B=np.asarray(B, dtype=float), ess=np.asarray(ess, dtype=float),
      dial_grid=np.asarray(dial_grid, dtype=float), metametric=np.asarray(metametric),
  )


def load_orientation_data(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']),
      'alphas': npz['alphas'],
      'center_alpha': float(npz['center_alpha']),
      'K': int(npz['K']),
      'donors': [str(d) for d in npz['donors']],
      'metametric': str(npz['metametric']) if 'metametric' in npz else 'spa',
      'A': npz['A'],
      'B': npz['B'],
      'ess': npz['ess'],
      'dial_grid': npz['dial_grid'],
  }


def orientation_score(scores_by_dial: np.ndarray) -> float:
  """Fraction of pairs (i, j) with dial_i < dial_j (i.e. i, j are indices
  into an array already sorted by ascending dial) where scores_by_dial[j]
  (the more extreme dial) exceeds scores_by_dial[i] -- 1.0 if the
  metametric always prefers the more extreme dial, 0.0 if always the less
  extreme one, 0.5 per tied pair. NaN if fewer than 2 finite values."""
  vals = np.asarray(scores_by_dial, dtype=float)
  finite = np.flatnonzero(~np.isnan(vals))
  if len(finite) < 2:
    return float('nan')
  wins = 0.0
  n_pairs = 0
  for i, j in itertools.combinations(finite, 2):  # i < j since `finite` is sorted ascending
    n_pairs += 1
    if vals[j] > vals[i]:
      wins += 1.0
    elif vals[j] == vals[i]:
      wins += 0.5
  return wins / n_pairs


def donor_orientation_by_alpha(grids: dict) -> dict[str, np.ndarray]:
  """{'A': (n_alpha,) adequacy_orientation per alpha, 'B': (n_alpha,)
  fluency_orientation per alpha} for one donor, from lib.
  synthetic_scorer_alpha_grid.donor_alpha_dial_grid's output -- one
  orientation_score per column (alpha), read off that column's dial-ordered
  SPA values."""
  out = {}
  for label in ('A', 'B'):
    g = grids[label]  # (n_dial, n_alpha), rows already in ascending dial order
    out[label] = np.array([orientation_score(g[:, ai]) for ai in range(g.shape[1])])
  return out
