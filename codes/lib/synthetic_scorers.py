"""Synthetic scorer construction (material/synthetic_scorer_construction.md):
given a real donor scorer's segment scores, builds two families of synthetic
scorers per donor -- one dialed on Adequacy MQM (dial A), one on Fluency MQM
(dial B) -- each a straight-line sweep between the donor itself (dial=0) and
the scorer whose system-level ranking is determined by that aspect's
group-mean response alone (dial=1). See the instruction file for the full
derivation; this module implements its section 4 (the generator) and section
5 (the procedure) directly.

Reuses lib.spa_plane's positional alignment (a_pos, b_pos, and a donor's own
line-position-indexed segment matrix) -- the same (a, b, donor) triple
scorer_spa_points needs -- so a family's points land on the exact same SPA
plane axes (x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM) as the real
scorers.
"""

from __future__ import annotations

import numpy as np

from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from lib.metric_scores import load_metric_seg_scores, load_metric_sys_scores
from lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices

# Same floor lib.consistency/lib.spa_plane use before trusting SPA's
# permutation p-values (kept as a local copy rather than importing the
# other modules' underscore-prefixed constants -- same convention lib.
# consistency already follows).
_MIN_SPA_SEGMENTS = 10

DIAL_GRID = tuple(round(float(x), 1) for x in np.arange(0.0, 1.0001, 0.1))


def aspect_lookup_table(y: np.ndarray, aspect: np.ndarray) -> np.ndarray:
  """m(a(k,i)), broadcast back to y's shape (section 3.1's group mean m): for
  every distinct value occurring in `aspect`, the mean of `y` over every
  (k,i) cell sharing that value. A lookup table, not a fitted function --
  whatever bends, plateaus, or jumps the donor's response to `aspect` has
  are carried exactly. Exact-float grouping is safe here because Adequacy/
  Fluency MQM segment values are highly discrete (weighted sums from a
  small error-weight table, mwb.mqm_scoring), the same discreteness plot_
  scorer_af_scatter.py's own mean-curve already relies on."""
  flat_aspect = aspect.ravel()
  flat_y = y.ravel()
  uniq, inv = np.unique(flat_aspect, return_inverse=True)
  sums = np.bincount(inv, weights=flat_y, minlength=len(uniq))
  counts = np.bincount(inv, minlength=len(uniq))
  means = sums / counts
  return means[inv].reshape(y.shape)


def donor_family(y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the generator of section 4:

      y_dial(k,i) = y(k,i) - dial * ebar_k

  with m (aspect_lookup_table), e = y - m, and ebar_k = mean_i e(k,i) all
  computed once on the full (y, aspect) pool passed in -- section 5.1's
  invariance requirement: callers must pass the entire jointly-valid pool
  for this donor, never a resampled subset, since these quantities are held
  fixed for the whole sweep. dial=0 reproduces y exactly (the real donor);
  dial=1 is y(k,i) - ebar_k, i.e. the donor with each system's constant
  offset from its own aspect-conditional mean removed."""
  m = aspect_lookup_table(y, aspect)
  e = y - m
  ebar_k = e.mean(axis=1)  # constant across segments within a system (section 3.3)
  return {dial: y - dial * ebar_k[:, None] for dial in dial_grid}


def donor_family_spa_points(
    donor_seg: np.ndarray, a_pos: np.ndarray, b_pos: np.ndarray, dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> dict[str, dict[float, tuple[float, float]]]:
  """{'A': {dial -> (x, y)}, 'B': {dial -> (x, y)}} for one donor's two
  synthetic families -- 'A' dialed on Adequacy (a_pos), 'B' on Fluency
  (b_pos) -- with x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM, the same
  two axes lib.spa_plane.scorer_spa_points uses. donor_seg/a_pos/b_pos must
  already be restricted to the same jointly-valid segment columns (the
  pool). p_a/p_b are computed once here and reused, at the same `seed`, for
  every dial and both families -- section 5.2's requirement that the same
  set of permutation sign vectors be reused across the whole sweep, which
  pairwise_p_values already satisfies by construction (same seed + same
  segment count -> identical signs drawn)."""
  p_a = pairwise_p_values(a_pos, num_permutations, seed)
  p_b = pairwise_p_values(b_pos, num_permutations, seed)
  out: dict[str, dict[float, tuple[float, float]]] = {}
  for label, aspect in (('A', a_pos), ('B', b_pos)):
    family = donor_family(donor_seg, aspect, dial_grid)
    pts = {}
    for dial, y_dial in family.items():
      p_m = pairwise_p_values(y_dial, num_permutations, seed)
      x = soft_pairwise_accuracy_from_pvalues(p_b, p_m)
      yv = soft_pairwise_accuracy_from_pvalues(p_a, p_m)
      if x == x and yv == yv:  # excludes NaN
        pts[dial] = (x, yv)
    out[label] = pts
  return out


def all_synthetic_family_points(
    dataset: str, systems: list[str], root: str = '.', dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, dict[float, tuple[float, float]]]]:
  """donor name -> donor_family_spa_points(...), for every donor with a
  usable segment-level file and enough jointly-valid positions with a_pos/
  b_pos -- the synthetic-scorer analogue of lib.spa_plane.scorer_spa_points
  (same candidate gate and same positional alignment), one donor expanding
  into 2*len(dial_grid) points instead of a single one. {} if this
  dataset's positional alignment can't be validated at all."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return {}
  a_pos, b_pos = pos

  candidates = load_metric_sys_scores(dataset, systems, root=root)
  out = {}
  for name in candidates:
    donor_seg = load_metric_seg_scores(dataset, name, systems, root=root)
    if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
      continue
    mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0) | np.isnan(donor_seg).any(axis=0))
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = donor_family_spa_points(
        donor_seg[:, mask], a_pos[:, mask], b_pos[:, mask],
        dial_grid=dial_grid, num_permutations=num_permutations, seed=seed)
  return out
