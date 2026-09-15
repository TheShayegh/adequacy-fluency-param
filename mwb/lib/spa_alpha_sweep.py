"""SPA(scorer; alpha) vs. target balance alpha (see
mwb/scripts/compute_spa_vs_alpha.py): shared math (the ESS>=target_ess
sweep-range solve) and the .npz cache format, so the compute script
(expensive: solve_w_exact per alpha, a permutation test per scorer) and
any number of plot scripts (cheap: just matplotlib) can be iterated on
independently.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mwb.lib.alpha import alpha_0 as alpha0_of, alpha_min_max
from mwb.lib.reweight_exact import solve_w_exact

# tab20/tab20b/tab20c concatenated: 60 visually-distinct qualitative colors
# (plain tab10, used by the original single-cache script, collided once an
# 11-scorer cache -- zhen23's metricx+xcomet screen -- exceeded its 10-color
# cycle, silently drawing two different scorers in the same color). Shared
# by plot_spa_vs_alpha.py and plot_spa_vs_alpha_combo.py so a scorer gets
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
  ESS(w*(alpha)) never exceeds K (attained only at alpha_0(D)) and a target
  <=0 would make every alpha in [alpha_min, alpha_max] admissible."""
  target_ess = K + min_ess if min_ess < 0 else min_ess
  if not (0.0 < target_ess < K):
    raise ValueError(f'min_ess={min_ess} resolves to target ESS={target_ess}, must be in (0, K={K})')
  return target_ess


def _find_ess_boundary(a_D: np.ndarray, b_D: np.ndarray, lo: float, hi: float, target_ess: float) -> float:
  """brentq root of ESS(w*(alpha)) - target_ess on [lo, hi] (assumes the
  target is bracketed, i.e. ESS(lo) and ESS(hi) sit on opposite sides of it --
  true for lo/hi = a bound of [alpha_min, alpha_max] and alpha_0(D), since
  ESS(alpha_0(D)) = K exactly and ESS -> 2 at either reachable-range
  boundary -- the optimum there is the unique two-system support
  (0.5, 0.5), see mwb.lib.reweight_exact's boundary shortcut)."""
  def f(alpha):
    return solve_w_exact(a_D, b_D, alpha).ess - target_ess
  return brentq(f, lo, hi, xtol=1e-8)


def alpha_sweep_range(a_D: np.ndarray, b_D: np.ndarray, target_ess: float) -> tuple[float, float, float, float, float]:
  """(a0_D, alpha_lo, alpha_hi, left, right): a0_D and the dataset's full
  reachable range, plus [left, right] -- the sub-range of that reachable
  range where ESS(w*(alpha)) >= target_ess, found by bracketing a0_D (where
  ESS is exactly K, its maximum) on each side and solving ESS(alpha) =
  target_ess with Brent's method."""
  a0_D = alpha0_of(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  eps = (alpha_hi - alpha_lo) * 1e-6
  left = _find_ess_boundary(a_D, b_D, alpha_lo + eps, a0_D, target_ess)
  right = _find_ess_boundary(a_D, b_D, a0_D, alpha_hi - eps, target_ess)
  return a0_D, alpha_lo, alpha_hi, left, right


def save_spa_vs_alpha(
    path: str, *, dataset: str, substrings: list[str], min_ess: float, target_ess: float, K: int,
    a0_D: float, alpha_lo: float, alpha_hi: float, alphas: np.ndarray, scorers: list[str], spa: np.ndarray,
) -> None:
  """spa is (n_scorers, n_alpha), row order == `scorers`, matching alphas'
  order -- the two axes plot_spa_vs_alpha*.py needs, plus enough metadata
  (dataset, the resolved min_ess/target_ess, K, a0_D, the dataset's full
  reachable range) to reproduce/relabel a plot without recomputing anything."""
  np.savez(
      path, dataset=dataset, substrings=np.array(substrings), min_ess=min_ess, target_ess=target_ess, K=K,
      a0_D=a0_D, alpha_lo=alpha_lo, alpha_hi=alpha_hi, alphas=np.asarray(alphas, dtype=float),
      scorers=np.array(scorers), spa=np.asarray(spa, dtype=float),
  )


def load_spa_vs_alpha(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']),
      'substrings': [str(s) for s in npz['substrings']],
      'min_ess': float(npz['min_ess']),
      'target_ess': float(npz['target_ess']),
      'K': int(npz['K']),
      'a0_D': float(npz['a0_D']),
      'alpha_lo': float(npz['alpha_lo']),
      'alpha_hi': float(npz['alpha_hi']),
      'alphas': npz['alphas'],
      'scorers': [str(s) for s in npz['scorers']],
      'spa': npz['spa'],
  }
