"""SPA(scorer; beta) vs. target balance beta (see
mwb/scripts/compute_spa_vs_beta.py): shared math (the ESS>=target_ess
sweep-range solve) and the .npz cache format, so the compute script
(expensive: solve_w_exact per beta, a permutation test per scorer) and
any number of plot scripts (cheap: just matplotlib) can be iterated on
independently.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mwb.lib.beta import beta_0 as beta0_of, beta_min_max
from mwb.lib.reweight_exact import solve_w_exact

# tab20/tab20b/tab20c concatenated: 60 visually-distinct qualitative colors
# (plain tab10, used by the original single-cache script, collided once an
# 11-scorer cache -- zhen23's metricx+xcomet screen -- exceeded its 10-color
# cycle, silently drawing two different scorers in the same color). Shared
# by plot_spa_vs_beta.py and plot_spa_vs_beta_combo.py so a scorer gets
# the same color in both, and across every panel of a combo plot.
_PALETTE = [c for cmap_name in ('tab20', 'tab20b', 'tab20c') for c in plt.get_cmap(cmap_name).colors]


def scorer_color_map(scorer_names: list[str]) -> dict[str, tuple]:
  """scorer name -> RGB color, one per name in `scorer_names` (deduplicated,
  sorted for a stable, input-order-independent assignment), cycling through
  _PALETTE's 60 colors if there are more scorers than that."""
  names = sorted(set(scorer_names))
  return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(names)}


# 20 visually-distinct marker shapes -- assigned the same way as
# scorer_color_map (sorted name -> cycled index), so color and marker both
# stay fixed per scorer across panels/scripts; a color collision beyond 60
# scorers is unlikely to matter, but a shared color+marker pair repeating
# every 20 scorers is still enough entropy to disambiguate two same-colored
# lines in practice.
_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', '8', 'p', 'd', '+', 'x', '1', '2', '3', '4']


def scorer_marker_map(scorer_names: list[str]) -> dict[str, str]:
  """scorer name -> matplotlib marker character, same cycling convention as
  scorer_color_map (see there)."""
  names = sorted(set(scorer_names))
  return {s: _MARKERS[i % len(_MARKERS)] for i, s in enumerate(names)}


def resolve_target_ess(K: int, min_ess: float) -> float:
  """--min-ess's negative-means-relative-to-K convention: a negative value
  is K + min_ess (e.g. -1 -> K-1, the default), a non-negative value is used
  directly as an absolute ESS floor. Raises ValueError outside (0, K), since
  ESS(w*(beta)) never exceeds K (attained only at beta_0(D)) and a target
  <=0 would make every beta in [beta_min, beta_max] admissible."""
  target_ess = K + min_ess if min_ess < 0 else min_ess
  if not (0.0 < target_ess < K):
    raise ValueError(f'min_ess={min_ess} resolves to target ESS={target_ess}, must be in (0, K={K})')
  return target_ess


def _find_ess_boundary(a_D: np.ndarray, f_D: np.ndarray, lo: float, hi: float, target_ess: float) -> float:
  """brentq root of ESS(w*(beta)) - target_ess on [lo, hi] (assumes the
  target is bracketed, i.e. ESS(lo) and ESS(hi) sit on opposite sides of it --
  true for lo/hi = a bound of [beta_min, beta_max] and beta_0(D), since
  ESS(beta_0(D)) = K exactly and ESS -> 2 at either reachable-range
  boundary -- the optimum there is the unique two-system support
  (0.5, 0.5), see mwb.lib.reweight_exact's boundary shortcut)."""
  def ess_gap(beta):
    return solve_w_exact(a_D, f_D, beta).ess - target_ess
  return brentq(ess_gap, lo, hi, xtol=1e-8)


def beta_sweep_range(a_D: np.ndarray, f_D: np.ndarray, target_ess: float) -> tuple[float, float, float, float, float]:
  """(beta_0_D, beta_lo, beta_hi, left, right): beta_0_D and the dataset's
  full reachable range, plus [left, right] -- the sub-range of that
  reachable range where ESS(w*(beta)) >= target_ess, found by bracketing
  beta_0_D (where ESS is exactly K, its maximum) on each side and solving
  ESS(beta) = target_ess with Brent's method."""
  beta_0_D = beta0_of(a_D, f_D)
  beta_lo, beta_hi = beta_min_max(a_D, f_D)
  eps = (beta_hi - beta_lo) * 1e-6
  left = _find_ess_boundary(a_D, f_D, beta_lo + eps, beta_0_D, target_ess)
  right = _find_ess_boundary(a_D, f_D, beta_0_D, beta_hi - eps, target_ess)
  return beta_0_D, beta_lo, beta_hi, left, right


def save_spa_vs_beta(
    path: str, *, dataset: str, substrings: list[str], min_ess: float, target_ess: float, K: int,
    beta_0_D: float, beta_lo: float, beta_hi: float, betas: np.ndarray, scorers: list[str], spa: np.ndarray,
) -> None:
  """spa is (n_scorers, n_beta), row order == `scorers`, matching betas'
  order -- the two axes plot_spa_vs_beta*.py needs, plus enough metadata
  (dataset, the resolved min_ess/target_ess, K, beta_0_D, the dataset's full
  reachable range) to reproduce/relabel a plot without recomputing anything."""
  np.savez(
      path, dataset=dataset, substrings=np.array(substrings), min_ess=min_ess, target_ess=target_ess, K=K,
      beta_0_D=beta_0_D, beta_lo=beta_lo, beta_hi=beta_hi, betas=np.asarray(betas, dtype=float),
      scorers=np.array(scorers), spa=np.asarray(spa, dtype=float),
  )


def load_spa_vs_beta(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']),
      'substrings': [str(s) for s in npz['substrings']],
      'min_ess': float(npz['min_ess']),
      'target_ess': float(npz['target_ess']),
      'K': int(npz['K']),
      'beta_0_D': float(npz['beta_0_D']),
      'beta_lo': float(npz['beta_lo']),
      'beta_hi': float(npz['beta_hi']),
      'betas': npz['betas'],
      'scorers': [str(s) for s in npz['scorers']],
      'spa': npz['spa'],
  }
