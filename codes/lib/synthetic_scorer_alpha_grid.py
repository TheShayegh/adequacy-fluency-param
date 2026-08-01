"""Meta-evaluation of synthetic scorer families (lib.synthetic_scorers,
material/synthetic_scorer_construction.md) across the adequacy-fluency
balance dial alpha (material/action_plan.md section 3.2/6.1): for one donor
scorer, builds the (dial, alpha) grid of weighted SPA scores against All MQM
-- dial is the synthesis dial (A or B, section 4 of the construction doc),
alpha is the meta-evaluation reweighting target realized by lib.
reweight_exact.solve_w_exact.

w*(alpha) is solved once per alpha and shared across every donor and every
dial (it depends only on the system pool D, section 3.11), so callers
should build it once via lib.reweighted_consistency.solve_w_for_datasets (or
equivalent) and pass the per-alpha weight vectors in; this module only does
the per-donor segment work (building the dial families and the weighted-SPA
sweep).
"""

from __future__ import annotations

import numpy as np

from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from lib.metric_scores import load_human_seg_scores, load_metric_seg_scores
from lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices
from lib.synthetic_scorers import DIAL_GRID, donor_family

# Same floor lib.consistency/lib.spa_plane/lib.synthetic_scorers use before
# trusting SPA's permutation p-values.
_MIN_SPA_SEGMENTS = 10


def alpha_grid_around(
    center: float, alpha_lo: float, alpha_hi: float, n_steps: int = 5, step: float = 0.02,
) -> list[float]:
  """alpha_0(D) (`center`) in the middle, n_steps steps of `step` in both
  directions (11 points at the defaults: center +/- 0.10 by 0.02). Points
  landing outside the reachable range [alpha_lo, alpha_hi] (action_plan.md
  Corollary 2) are dropped rather than clipped -- solve_w_exact raises on
  an out-of-range target, and a dropped column reads as a missing (blank)
  cell rather than a duplicated boundary one."""
  raw = [center + k * step for k in range(-n_steps, n_steps + 1)]
  return [a for a in raw if alpha_lo <= a <= alpha_hi]


def donor_alpha_dial_grid(
    dataset: str, donor_name: str, systems: list[str], w_by_alpha: dict[float, np.ndarray],
    root: str = '.', dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray] | None:
  """{'A': (n_dial, n_alpha) weighted-SPA grid, 'B': same for the fluency
  family, 'dials': the dial_grid as an array, 'alphas': sorted(w_by_alpha)
  as an array} for one donor. None if this donor lacks the segment coverage
  lib.synthetic_scorers.all_synthetic_family_points already requires (no
  usable segment file, dataset alignment unvalidated, or too few
  jointly-valid positions -- now also intersected with All MQM's own
  segment coverage, needed here for the human side of weighted SPA but not
  by the plain SPA-plane family points).

  Column ai's ESS/weighting come from w_by_alpha[alphas[ai]]; row di's dial
  value is dial_grid[di]. p_human is computed once per donor (fixed mask)
  and reused across every dial and alpha; p_metric is computed once per
  dial and reused across every alpha -- only the weighted aggregation
  (cheap) varies with alpha, matching the project's existing spa_pvalue_
  cache/solve_w_for_datasets split between expensive I/O+permutation work
  and cheap per-alpha aggregation."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None

  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]
  p_human = pairwise_p_values(human_seg, num_permutations, seed)

  alphas = sorted(w_by_alpha)
  out: dict[str, np.ndarray] = {}
  for label, aspect in (('A', a_pos), ('B', b_pos)):
    family = donor_family(donor_seg, aspect, dial_grid)
    grid = np.full((len(dial_grid), len(alphas)), np.nan)
    for di, dial in enumerate(dial_grid):
      p_metric = pairwise_p_values(family[dial], num_permutations, seed)
      for ai, a in enumerate(alphas):
        grid[di, ai] = soft_pairwise_accuracy_from_pvalues(p_human, p_metric, w_by_alpha[a])
    out[label] = grid
  out['dials'] = np.asarray(dial_grid, dtype=float)
  out['alphas'] = np.asarray(alphas, dtype=float)
  return out
