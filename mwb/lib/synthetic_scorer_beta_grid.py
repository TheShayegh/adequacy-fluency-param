"""Meta-evaluation of synthetic scorer families (mwb.lib.synthetic_scorers)
across the adequacy-fluency balance beta: for one base scorer, builds the
(dial, beta) grid of a weighted meta-metric (SPA or PA) against All MQM --
dial is the synthesis dial (A or F), beta is the meta-evaluation
reweighting target realized by mwb.lib.reweight_exact.solve_w_exact.

w*(beta) is solved once per beta and shared across every base scorer and every
dial (it depends only on the system pool D), so callers should build it
once via mwb.lib.reweighted_consistency.solve_w_for_datasets (or
equivalent) and pass the per-beta weight vectors in; this module only does
the per-base scorer segment work (building the dial families and the weighted
meta-metric sweep).

SPA needs segment-level permutation p-values (pairwise_p_values); PA reads
only system-level scores -- u_k(A) = mean_i y_A(k,i) -- against
system-level All MQM ('t' = a+f, mwb.mqm_scoring.load_system_scores), so it
skips the permutation-test machinery entirely and is correspondingly
cheaper.
"""

from __future__ import annotations

import numpy as np

from mwb.lib.metametrics import (
    pairwise_p_values, soft_pairwise_accuracy_from_pvalues, WEIGHTED_METAMETRICS, MetaEvalInput,
)
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from mwb.lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices
from mwb.lib.synthetic_scorers import (
    AF_ADDITIVE_MEAN_DIAL_GRID, DIAL_GRID, SYNTHESIS_GENERATORS,
    base_scorer_family_additive_mean_af, base_scorer_family_joint,
)
from mwb.mqm_scoring import load_system_scores

# Same floor mwb.lib.consistency/mwb.lib.spa_plane/mwb.lib.synthetic_scorers use before
# trusting SPA's permutation p-values. PA doesn't need segment-level
# p-values at all, but the SAME mask/coverage gate is kept for it too, so
# a base scorer's inclusion and its dial families are identical between an
# spa run and a pa run on the same dataset -- only the aggregation differs.
_MIN_SPA_SEGMENTS = 10

# "Fake base scorer" names: instead of a real scorer's segment scores, base_scorer_seg
# is one of the gold aspect signals itself (a_pos/f_pos/human_seg, already
# loaded for every base scorer anyway) -- i.e. what happens if the base scorer IS
# a perfect oracle for one aspect (or their sum), dialed through the exact
# same family/beta/metametric machinery as any real base scorer. AllMQM uses
# human_seg (the official All-MQM file, mwb.mqm_scoring's 't'/all_mqm
# column) rather than a_pos+f_pos, since that's the literal ground-truth
# total rather than a derived sum of the two sub-scores.
ASPECT_BASE_SCORERS = ('AdequacyMQM', 'FluencyMQM', 'AllMQM')


def _aspect_base_scorer_seg(base_scorer_name: str, a_pos: np.ndarray, f_pos: np.ndarray, human_seg: np.ndarray) -> np.ndarray:
  if base_scorer_name == 'AdequacyMQM':
    return a_pos
  if base_scorer_name == 'FluencyMQM':
    return f_pos
  if base_scorer_name == 'AllMQM':
    return human_seg
  raise ValueError(f'not an aspect base scorer: {base_scorer_name!r}')


def beta_grid_around(
    center: float, beta_lo: float, beta_hi: float, n_steps: int = 5, step: float = 0.02,
) -> list[float]:
  """beta_0(D) (`center`) in the middle, n_steps steps of `step` in both
  directions (11 points at the defaults: center +/- 0.10 by 0.02). Points
  landing outside the reachable range [beta_lo, beta_hi] (Theorem
  "Reachable range") are dropped rather than clipped -- solve_w_exact
  raises on an out-of-range target, and a dropped column reads as a
  missing (blank) cell rather than a duplicated boundary one."""
  raw = [center + k * step for k in range(-n_steps, n_steps + 1)]
  return [v for v in raw if beta_lo <= v <= beta_hi]


def base_scorer_beta_dial_grid(
    dataset: str, base_scorer_name: str, systems: list[str], w_by_beta: dict[float, np.ndarray],
    root: str = '.', dial_grid=DIAL_GRID, metametric: str = 'spa', synthesis: str = 'offset',
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
    use_af_family: bool = False, af_dial_grid=None,
) -> dict[str, np.ndarray] | None:
  """{'A': (n_dial, n_beta) weighted meta-metric grid, 'F': same for the
  fluency family (or, with use_af_family=True, 'AF' replaces
  both -- see below), 'J' or 'T': same for the third family (see below),
  'dials': the dial_grid as an array, 'betas': sorted(w_by_beta) as an
  array} for one base scorer. None if this base scorer lacks
  the segment coverage mwb.lib.synthetic_scorers.all_synthetic_family_points
  already requires (no usable segment file, dataset alignment unvalidated,
  or too few jointly-valid positions -- now also intersected with All MQM's
  own segment coverage, needed for SPA's human side and, for consistency,
  applied to PA too even though PA itself only reads system-level scores).

  base_scorer_name may also be one of ASPECT_BASE_SCORERS ('AdequacyMQM', 'FluencyMQM',
  'AllMQM'): a "fake base scorer" whose base_scorer_seg is the gold aspect signal itself
  (a_pos/f_pos/human_seg) rather than a real scorer's segment scores --
  everything downstream (family building, beta/metametric sweep) is
  identical.

  Two dial families are always built: 'A' (dialed on Adequacy MQM) and 'F'
  (dialed on Fluency MQM). A third, synthesis-dependent family is also
  built:

    - synthesis='offset': 'J' (mwb.lib.synthetic_scorers.base_scorer_family_joint),
      dialed on the JOINT (Adequacy, Fluency) pair -- the generalization of
      offset's own m/e/ebar_k decomposition to a joint condition instead of
      either aspect alone.
    - synthesis='additive'/'additive_mean': 'T' (dialed on All MQM, i.e.
      human_seg -- the official total, not a_pos+f_pos, matching
      ASPECT_BASE_SCORERS' own 'AllMQM' convention above), preference-neutral by
      construction (t=a+f favors neither aspect) -- a natural-truth family
      alongside the two aspect-only ones. Restricted to the additive
      methods because that neutrality argument relies on injecting the
      aspect directly rather than going through m/e/ebar_k -- it does not
      hold for synthesis='offset' (see mwb.lib.synthetic_scorers module
      docstring), which is why 'offset' gets 'J' here instead.

  metametric: 'spa' (default), 'pa', or 'pearson'. For spa, p_human is
  computed once per base scorer (fixed mask) and reused across every dial and
  beta; p_scorer is computed once per dial and reused across every beta
  -- only the weighted aggregation (cheap) varies with beta, keeping the
  expensive I/O+permutation work separate from the cheap per-beta
  aggregation. For pa/pearson, there is no permutation work at all:
  u_k(A) = mean_i y_A(k,i) is computed once per dial and reused across
  every beta, weighed (mwb.lib.metametrics.WEIGHTED_METAMETRICS) against
  system-level All MQM ('t').

  synthesis: which mwb.lib.synthetic_scorers family generator builds the A/F
  (and, for additive/additive_mean, T) dial sweep -- 'offset' (default,
  the original generator), 'additive' (y + dial*aspect), or
  'additive_mean' (y + dial*abar_k). J (offset only) always goes through
  base_scorer_family_joint directly, not build_family, since it needs both
  aspects jointly rather than one aspect vector.

  Column bi's ESS/weighting come from w_by_beta[betas[bi]]; row di's dial
  value is dial_grid[di]."""
  if metametric not in ('spa', 'pa', 'pearson'):
    raise ValueError(f"metametric must be 'spa', 'pa', or 'pearson', got {metametric!r}")
  if synthesis not in SYNTHESIS_GENERATORS:
    raise ValueError(f'synthesis must be one of {sorted(SYNTHESIS_GENERATORS)}, got {synthesis!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = _aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None

  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  base_scorer_seg, human_seg = base_scorer_seg[:, mask], human_seg[:, mask]

  betas = sorted(w_by_beta)
  out: dict[str, np.ndarray] = {}

  if use_af_family:
    # The paper's actual "Adequacy-fluency family" (Sec. "Scorer
    # Augmentation"): ONE dial pushing toward more-adequate-AND-less-
    # fluent simultaneously, not two separately-dialed single-aspect
    # families. Its own dial grid (AF_ADDITIVE_MEAN_DIAL_GRID,
    # [-0.5, 0.5] by 0.02, matching the paper's dial-grid paragraph in
    # Sec. "Evaluation Setup") is unrelated to `dial_grid` (which stays
    # in effect for T/J below) -- pass af_dial_grid explicitly to
    # override it.
    built = {'AF': base_scorer_family_additive_mean_af(
        base_scorer_seg, a_pos, f_pos, af_dial_grid or AF_ADDITIVE_MEAN_DIAL_GRID)}
  else:
    built = {'A': build_family(base_scorer_seg, a_pos, dial_grid), 'F': build_family(base_scorer_seg, f_pos, dial_grid)}
  if synthesis == 'offset':
    built['J'] = base_scorer_family_joint(base_scorer_seg, a_pos, f_pos, dial_grid)
  else:
    built['T'] = build_family(base_scorer_seg, human_seg, dial_grid)

  if metametric == 'spa':
    p_human = pairwise_p_values(human_seg, num_permutations, seed)
    for label, family in built.items():
      # Each family's OWN dial keys, not the shared `dial_grid` parameter:
      # AF (use_af_family=True) is built on its own
      # AF_ADDITIVE_MEAN_DIAL_GRID (a different length/values than
      # A/F/T/J's dial_grid), so indexing `family[dial]` with `dial_grid`'s
      # values would KeyError for it.
      family_dials = list(family)
      grid = np.full((len(family_dials), len(betas)), np.nan)
      for di, dial in enumerate(family_dials):
        p_scorer = pairwise_p_values(family[dial], num_permutations, seed)
        for bi, beta_val in enumerate(betas):
          grid[di, bi] = soft_pairwise_accuracy_from_pvalues(p_human, p_scorer, w_by_beta[beta_val])
      out[label] = grid
  else:  # 'pa' or 'pearson': both read only system-level scores
    t = load_system_scores(dataset, root=root).loc[systems, 't'].values
    weighted_fn = WEIGHTED_METAMETRICS[metametric]
    for label, family in built.items():
      family_dials = list(family)  # see the spa branch above for why not `dial_grid`
      grid = np.full((len(family_dials), len(betas)), np.nan)
      for di, dial in enumerate(family_dials):
        u_k = family[dial].mean(axis=1)  # u_k(A) = mean_i y_A(k,i)
        for bi, beta_val in enumerate(betas):
          x = MetaEvalInput(human_sys=t, scorer_sys=u_k)
          grid[di, bi] = weighted_fn(x, w_by_beta[beta_val])
      out[label] = grid

  out['dials'] = np.asarray(dial_grid, dtype=float)
  out['betas'] = np.asarray(betas, dtype=float)
  return out
