"""Meta-evaluation of synthetic scorer families (mwb.lib.synthetic_scorers)
across the adequacy-fluency balance alpha (the paper's beta): for one donor
scorer, builds the (dial, alpha) grid of a weighted meta-metric (SPA or PA)
against All MQM -- dial is the synthesis dial (A or B), alpha is the
meta-evaluation reweighting target realized by
mwb.lib.reweight_exact.solve_w_exact.

w*(alpha) is solved once per alpha and shared across every donor and every
dial (it depends only on the system pool D), so callers should build it
once via mwb.lib.reweighted_consistency.solve_w_for_datasets (or
equivalent) and pass the per-alpha weight vectors in; this module only does
the per-donor segment work (building the dial families and the weighted
meta-metric sweep).

SPA needs segment-level permutation p-values (pairwise_p_values); PA reads
only system-level scores -- u_k(A) = mean_i y_A(k,i) -- against
system-level All MQM ('t' = a+b, mwb.mqm_scoring.load_system_scores), so it
skips the permutation-test machinery entirely and is correspondingly
cheaper.
"""

from __future__ import annotations

import numpy as np

from mwb.lib.metametrics import (
    pairwise_p_values, soft_pairwise_accuracy_from_pvalues, WEIGHTED_METAMETRICS, MetaEvalInput,
)
from mwb.lib.metric_scores import load_human_seg_scores, load_metric_seg_scores
from mwb.lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices
from mwb.lib.synthetic_scorers import (
    DIAGONAL_ADDITIVE_MEAN_DIAL_GRID, DIAL_GRID, SYNTHESIS_GENERATORS,
    donor_family_additive_mean_diagonal, donor_family_joint,
)
from mwb.mqm_scoring import load_system_scores

# Same floor mwb.lib.consistency/mwb.lib.spa_plane/mwb.lib.synthetic_scorers use before
# trusting SPA's permutation p-values. PA doesn't need segment-level
# p-values at all, but the SAME mask/coverage gate is kept for it too, so
# a donor's inclusion and its dial families are identical between an
# spa run and a pa run on the same dataset -- only the aggregation differs.
_MIN_SPA_SEGMENTS = 10

# "Fake donor" names: instead of a real metric's segment scores, donor_seg
# is one of the gold aspect signals itself (a_pos/b_pos/human_seg, already
# loaded for every donor anyway) -- i.e. what happens if the base scorer IS
# a perfect oracle for one aspect (or their sum), dialed through the exact
# same family/alpha/metametric machinery as any real donor. AllMQM uses
# human_seg (the official All-MQM file, mwb.mqm_scoring's 't'/all_mqm
# column) rather than a_pos+b_pos, since that's the literal ground-truth
# total rather than a derived sum of the two sub-scores.
ASPECT_DONORS = ('AdequacyMQM', 'FluencyMQM', 'AllMQM')


def _aspect_donor_seg(donor_name: str, a_pos: np.ndarray, b_pos: np.ndarray, human_seg: np.ndarray) -> np.ndarray:
  if donor_name == 'AdequacyMQM':
    return a_pos
  if donor_name == 'FluencyMQM':
    return b_pos
  if donor_name == 'AllMQM':
    return human_seg
  raise ValueError(f'not an aspect donor: {donor_name!r}')


def alpha_grid_around(
    center: float, alpha_lo: float, alpha_hi: float, n_steps: int = 5, step: float = 0.02,
) -> list[float]:
  """alpha_0(D) (`center`) in the middle, n_steps steps of `step` in both
  directions (11 points at the defaults: center +/- 0.10 by 0.02). Points
  landing outside the reachable range [alpha_lo, alpha_hi] (Theorem
  "Reachable range") are dropped rather than clipped -- solve_w_exact
  raises on an out-of-range target, and a dropped column reads as a
  missing (blank) cell rather than a duplicated boundary one."""
  raw = [center + k * step for k in range(-n_steps, n_steps + 1)]
  return [a for a in raw if alpha_lo <= a <= alpha_hi]


def donor_alpha_dial_grid(
    dataset: str, donor_name: str, systems: list[str], w_by_alpha: dict[float, np.ndarray],
    root: str = '.', dial_grid=DIAL_GRID, metametric: str = 'spa', synthesis: str = 'offset',
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
    adequacy_fluency_diagonal: bool = False, diagonal_dial_grid=None,
) -> dict[str, np.ndarray] | None:
  """{'A': (n_dial, n_alpha) weighted meta-metric grid, 'B': same for the
  fluency family (or, with adequacy_fluency_diagonal=True, 'AF' replaces
  both -- see below), 'J' or 'T': same for the third family (see below),
  'dials': the dial_grid as an array, 'alphas': sorted(w_by_alpha) as an
  array} for one donor. None if this donor lacks
  the segment coverage mwb.lib.synthetic_scorers.all_synthetic_family_points
  already requires (no usable segment file, dataset alignment unvalidated,
  or too few jointly-valid positions -- now also intersected with All MQM's
  own segment coverage, needed for SPA's human side and, for consistency,
  applied to PA too even though PA itself only reads system-level scores).

  donor_name may also be one of ASPECT_DONORS ('AdequacyMQM', 'FluencyMQM',
  'AllMQM'): a "fake donor" whose donor_seg is the gold aspect signal itself
  (a_pos/b_pos/human_seg) rather than a real metric's segment scores --
  everything downstream (family building, alpha/metametric sweep) is
  identical.

  Two dial families are always built: 'A' (dialed on Adequacy MQM) and 'B'
  (dialed on Fluency MQM). A third, synthesis-dependent family is also
  built:

    - synthesis='offset': 'J' (mwb.lib.synthetic_scorers.donor_family_joint),
      dialed on the JOINT (Adequacy, Fluency) pair -- the generalization of
      offset's own m/e/ebar_k decomposition to a joint condition instead of
      either aspect alone.
    - synthesis='additive'/'additive_mean': 'T' (dialed on All MQM, i.e.
      human_seg -- the official total, not a_pos+b_pos, matching
      ASPECT_DONORS' own 'AllMQM' convention above), orientation-neutral by
      construction (t=a+b favors neither aspect) -- a natural-truth family
      alongside the two aspect-only ones. Restricted to the additive
      methods because that neutrality argument relies on injecting the
      aspect directly rather than going through m/e/ebar_k -- it does not
      hold for synthesis='offset' (see mwb.lib.synthetic_scorers module
      docstring), which is why 'offset' gets 'J' here instead.

  metametric: 'spa' (default), 'pa', or 'pearson'. For spa, p_human is
  computed once per donor (fixed mask) and reused across every dial and
  alpha; p_metric is computed once per dial and reused across every alpha
  -- only the weighted aggregation (cheap) varies with alpha, keeping the
  expensive I/O+permutation work separate from the cheap per-alpha
  aggregation. For pa/pearson, there is no permutation work at all:
  u_k(A) = mean_i y_A(k,i) is computed once per dial and reused across
  every alpha, weighed (mwb.lib.metametrics.WEIGHTED_METAMETRICS) against
  system-level All MQM ('t').

  synthesis: which mwb.lib.synthetic_scorers family generator builds the A/B
  (and, for additive/additive_mean, T) dial sweep -- 'offset' (default,
  the original generator), 'additive' (y + dial*aspect), or
  'additive_mean' (y + dial*abar_k). J (offset only) always goes through
  donor_family_joint directly, not build_family, since it needs both
  aspects jointly rather than one aspect vector.

  Column ai's ESS/weighting come from w_by_alpha[alphas[ai]]; row di's dial
  value is dial_grid[di]."""
  if metametric not in ('spa', 'pa', 'pearson'):
    raise ValueError(f"metametric must be 'spa', 'pa', or 'pearson', got {metametric!r}")
  if synthesis not in SYNTHESIS_GENERATORS:
    raise ValueError(f'synthesis must be one of {sorted(SYNTHESIS_GENERATORS)}, got {synthesis!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if donor_name in ASPECT_DONORS:
    donor_seg = _aspect_donor_seg(donor_name, a_pos, b_pos, human_seg)
  else:
    donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None

  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]

  alphas = sorted(w_by_alpha)
  out: dict[str, np.ndarray] = {}

  if adequacy_fluency_diagonal:
    # The paper's actual "Adequacy-fluency family" (Sec. "Scorer
    # Augmentation"): ONE dial pushing toward more-adequate-AND-less-
    # fluent simultaneously, not two separately-dialed single-aspect
    # families. Its own dial grid (DIAGONAL_ADDITIVE_MEAN_DIAL_GRID,
    # [-0.5, 0.5] by 0.02, matching the paper's dial-grid paragraph in
    # Sec. "Evaluation Setup") is unrelated to `dial_grid` (which stays
    # in effect for T/J below) -- pass diagonal_dial_grid explicitly to
    # override it.
    built = {'AF': donor_family_additive_mean_diagonal(
        donor_seg, a_pos, b_pos, diagonal_dial_grid or DIAGONAL_ADDITIVE_MEAN_DIAL_GRID)}
  else:
    built = {'A': build_family(donor_seg, a_pos, dial_grid), 'B': build_family(donor_seg, b_pos, dial_grid)}
  if synthesis == 'offset':
    built['J'] = donor_family_joint(donor_seg, a_pos, b_pos, dial_grid)
  else:
    built['T'] = build_family(donor_seg, human_seg, dial_grid)

  if metametric == 'spa':
    p_human = pairwise_p_values(human_seg, num_permutations, seed)
    for label, family in built.items():
      # Each family's OWN dial keys, not the shared `dial_grid` parameter:
      # AF (adequacy_fluency_diagonal=True) is built on its own
      # DIAGONAL_ADDITIVE_MEAN_DIAL_GRID (a different length/values than
      # A/B/T/J's dial_grid), so indexing `family[dial]` with `dial_grid`'s
      # values would KeyError for it.
      family_dials = list(family)
      grid = np.full((len(family_dials), len(alphas)), np.nan)
      for di, dial in enumerate(family_dials):
        p_metric = pairwise_p_values(family[dial], num_permutations, seed)
        for ai, a in enumerate(alphas):
          grid[di, ai] = soft_pairwise_accuracy_from_pvalues(p_human, p_metric, w_by_alpha[a])
      out[label] = grid
  else:  # 'pa' or 'pearson': both read only system-level scores
    t = load_system_scores(dataset, root=root).loc[systems, 't'].values
    weighted_fn = WEIGHTED_METAMETRICS[metametric]
    for label, family in built.items():
      family_dials = list(family)  # see the spa branch above for why not `dial_grid`
      grid = np.full((len(family_dials), len(alphas)), np.nan)
      for di, dial in enumerate(family_dials):
        u_k = family[dial].mean(axis=1)  # u_k(A) = mean_i y_A(k,i)
        for ai, a in enumerate(alphas):
          x = MetaEvalInput(human_sys=t, metric_sys=u_k)
          grid[di, ai] = weighted_fn(x, w_by_alpha[a])
      out[label] = grid

  out['dials'] = np.asarray(dial_grid, dtype=float)
  out['alphas'] = np.asarray(alphas, dtype=float)
  return out
