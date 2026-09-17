"""SPA_synth25 / PA_synth25: a metametric BASELINE that reproduces Shayegh
et al. (2025)'s system-synthesis meta-evaluation setup -- the baseline the
paper's Sec. "Main Results" compares its own reweighting method against.
It computes PA/SPA of a scorer against All MQM, not over a dataset's real
systems alone but over the UNION of the real systems and TWO sets of K
synthesized pseudo-systems, one exhibiting extreme Adequacy-MQM variation
and one extreme Fluency-MQM variation.

Their synthesis rule: given K real systems, for each segment rank their
outputs by Adequacy MQM (ties broken uniformly at random) and let synthetic
adequacy-system k receive, at that segment, whichever real system's score
ranks k-th there -- for EVERY quantity at that segment (its own Adequacy
MQM, but also its All MQM and any scorer's own segment score), so synthetic
system k is a per-segment patchwork of real systems' outputs, internally
consistent segment-by-segment. Likewise for a fluency ranking. The pool
kept here retains the real systems too: original UNION
synthesized-by-adequacy UNION synthesized-by-fluency, 3K systems in total.
This module reproduces the fuller, un-averaged per-segment version needed
to feed a real scorer's own segment scores through PA/SPA -- not the
coarser scalar-per-system version described next.

Isolation (by design, per this project's convention that SYSTEM synthesis
and SCORER synthesis must never be confused):
  - This module is entirely self-contained. It is NOT wired into
    src.lib.metametrics.METAMETRICS/WEIGHTED_METAMETRICS or
    src.lib.consistency -- no existing caller (scorer_scores, weighted_
    consistency, the beta-grid/preference scripts, real_scorers, etc.)
    can reach a synthesized system by accident. The only way to get a
    PA_synth25/SPA_synth25 number is to call this module directly.
  - "Synthesis" here means Shayegh et al.'s TRANSLATION-SYSTEM synthesis
    (pseudo translation systems, patched together from real systems'
    per-segment outputs). It is unrelated to this project's SCORER
    synthesis (synthetic_scorers.py, synthetic_scorer_beta_grid.py,
    synthetic_scorer_preference.py), which fabricates pseudo automatic
    METRICS (linear mixes of Adequacy/Fluency MQM plus noise) to test
    metametric recovery of a known ground-truth beta -- a different axis
    of the same rectangle (systems vs. scorers), and this module touches
    only the system axis.
  - A coarser implementation of the same synthesis rule once existed
    elsewhere in this project, collapsing each synthesized system to a
    single scalar (the segment-mean, per rank) for use as a mechanical
    beta-range extender; it has since been removed as unused. This module
    keeps every segment (needed to run an actual scorer's segment scores
    through PA/SPA) and is used for nothing beyond the two baselines
    defined here.

The reachable dataset roster is a real subset: only datasets where
src.lib.spa_plane.positional_af_matrices can validate that Adequacy/Fluency
MQM aligns positionally with a scorer's own .seg.score file (its own
module docstring) produce any baseline numbers here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lib.beta import beta_0 as _beta_0
from src.lib.consistency import real_systems
from src.lib.metametrics import (
    MetaEvalInput, pairwise_accuracy, pairwise_p_values,
    soft_pairwise_accuracy_from_pvalues,
)
from src.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores, load_scorer_sys_scores
from src.lib.spa_plane import positional_af_matrices

# Same floor used throughout this project (src.lib.consistency,
# src.lib.spa_plane) before trusting SPA's permutation p-values / a
# rank-based synthesis -- below this a (dataset, scorer) cell is skipped
# rather than reported on noise.
_MIN_SPA_SEGMENTS = 10

DEFAULT_NUM_PERMUTATIONS = 1000
DEFAULT_SEED = 4          # SPA's permutation-test null-sampling seed --
                           # matches src.lib.spa_plane.DEFAULT_SEED so a
                           # SPA_synth25 number is comparable to this
                           # project's other SPA numbers on the same
                           # (dataset, scorer): same null-sampling draws,
                           # the only difference is the enlarged pool.
DEFAULT_TIE_SEED = 25     # separate RNG stream for synthesis's own random
                           # tie-break (Shayegh et al. 2025 footnote 7),
                           # distinct from the permutation test's seed --
                           # named for the "25" in "synth25".

SYNTH25_METAMETRICS = ('pa_synth25', 'spa_synth25')

# The 6 pools this module can build from a dataset's K real systems, each a
# subset of the 3 available blocks ('real' = the K real systems themselves,
# 'adeq'/'flu' = their K adequacy-/fluency-synthesized counterparts,
# S2.4/module docstring). 'real+adeq+flu' (Row 7 of Table
# tab:synthesized_systems) is the branded SPA_synth25/PA_synth25 baseline
# and every other function in this module defaults to it; the other 5 are
# for direct comparison (e.g. plot_scorer_preference_vs_beta.py's
# --overlay-synth25 pool markers) -- an ordered dict so callers that
# iterate it (building one marker per pool) get a stable, meaningful order:
# the branded baseline first, then progressively fewer blocks.
POOL_BLOCKS = {
    'real+adeq+flu': ('real', 'adeq', 'flu'),
    'real+adeq': ('real', 'adeq'),
    'real+flu': ('real', 'flu'),
    'adeq+flu': ('adeq', 'flu'),
    'flu': ('flu',),
    'adeq': ('adeq',),
}


def _segment_rank_order(rank_seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """(K, n_pos) int array; order[k, j] = row-index into rank_seg of the
  system whose rank_seg value is k-th largest at position j (0 = best),
  i.e. descending per column. Ties broken uniformly at random via a fresh
  random tie-break value per cell (Shayegh et al. 2025 footnote 7), not
  numpy argsort's arbitrary-but-deterministic tie order."""
  tie_break = rng.random(rank_seg.shape)
  return np.lexsort((tie_break, -rank_seg), axis=0)


def _apply_rank_order(order: np.ndarray, seg: np.ndarray) -> np.ndarray:
  """seg (K, n_pos) reindexed per column by `order`: row k of the result
  holds, at every position, the value from whichever real system `order`
  says ranks k-th there -- Shayegh et al.'s "carry" rule applied to an
  arbitrary segment-level quantity (All MQM, or a scorer's own segment
  score), not collapsed to a per-system scalar."""
  return np.take_along_axis(seg, order, axis=0)


def _union_seg_matrices(
    real_seg: np.ndarray, order_adequacy: np.ndarray, order_fluency: np.ndarray,
    blocks: tuple[str, ...] = ('real', 'adeq', 'flu'),
) -> np.ndarray:
  """(len(blocks)*K, n_pos): the requested subset of {real_seg itself,
  its adequacy-synthesized counterpart, its fluency-synthesized
  counterpart} stacked together, same quantity (e.g. All MQM, or one
  scorer's own segment score) reindexed by each order. blocks=('real',
  'adeq', 'flu') (the default) is Row 7 of Table tab:synthesized_systems;
  see POOL_BLOCKS for the other 5 combinations this module supports."""
  parts = []
  if 'real' in blocks:
    parts.append(real_seg)
  if 'adeq' in blocks:
    parts.append(_apply_rank_order(order_adequacy, real_seg))
  if 'flu' in blocks:
    parts.append(_apply_rank_order(order_fluency, real_seg))
  return np.vstack(parts)


def _pa_synth25_from_seg(human_seg: np.ndarray, scorer_seg: np.ndarray) -> float:
  human_sys = human_seg.mean(axis=1)
  scorer_sys = scorer_seg.mean(axis=1)
  return pairwise_accuracy(MetaEvalInput(human_sys, scorer_sys))


def _spa_synth25_from_seg(
    human_seg: np.ndarray, scorer_seg: np.ndarray, num_permutations: int, seed: int,
) -> float:
  human_p = pairwise_p_values(human_seg, num_permutations, seed)
  scorer_p = pairwise_p_values(scorer_seg, num_permutations, seed)
  return soft_pairwise_accuracy_from_pvalues(human_p, scorer_p)


def synth25_value(
    metametric_name: str,
    human_seg: np.ndarray,
    scorer_seg: np.ndarray,
    a_pos: np.ndarray,
    f_pos: np.ndarray,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    blocks: tuple[str, ...] = ('real', 'adeq', 'flu'),
) -> float:
  """Core PA_synth25/SPA_synth25 computation, decoupled from file loading
  and from what "the scorer" is: human_seg/scorer_seg/a_pos/f_pos are
  already (K, n_pos) row-aligned to the SAME K real systems and already
  restricted to jointly-valid positions (the caller's job -- see
  synth25_scorer_scores for the file-backed, "real automatic scorer" case).
  `scorer_seg` need not come from a real scorer's own segment file at
  all -- any (K, n_pos) segment-score matrix works, e.g. one dial point of
  a src.lib.synthetic_scorers dial family, which is how this function is
  reused to compare SYNTHETIC scorers against the synth25 baseline
  (kept out of THIS module, per its isolation policy -- see
  src.lib.synth25_preference, the one place that bridges the two).

  blocks: which of the 3 blocks (POOL_BLOCKS) to union -- default
  ('real', 'adeq', 'flu') is Row 7, the branded SPA_synth25/PA_synth25
  baseline every other caller in this project should use; pass one of
  POOL_BLOCKS' other values to score against a different pool instead
  (e.g. for a direct pool-to-pool comparison, not a branded baseline)."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  rng = np.random.default_rng(tie_seed)
  order_adequacy = _segment_rank_order(a_pos, rng)
  order_fluency = _segment_rank_order(f_pos, rng)
  human_union = _union_seg_matrices(human_seg, order_adequacy, order_fluency, blocks)
  scorer_union = _union_seg_matrices(scorer_seg, order_adequacy, order_fluency, blocks)
  if metametric_name == 'pa_synth25':
    return _pa_synth25_from_seg(human_union, scorer_union)
  return _spa_synth25_from_seg(human_union, scorer_union, num_permutations, seed)


def pool_beta_0(
    a_pos: np.ndarray, f_pos: np.ndarray, blocks: tuple[str, ...], tie_seed: int = DEFAULT_TIE_SEED,
) -> float:
  """beta_0 (src.lib.beta.beta_0: Var(a)/(Var(a)+Var(f)) at uniform
  weight) of one pool's OWN system-level Adequacy/Fluency MQM -- a property
  of the pool's composition alone, independent of any scorer/base scorer/
  metametric. a_pos/f_pos: already (K, n_pos) row-aligned to the K real
  systems and restricted to positions where BOTH are non-NaN for every
  system (this function's own mask requirement -- callers should NOT
  further restrict to any one base scorer's coverage, since this is a
  dataset-level constant, not a per-base scorer one)."""
  rng = np.random.default_rng(tie_seed)
  order_adequacy = _segment_rank_order(a_pos, rng)
  order_fluency = _segment_rank_order(f_pos, rng)
  a_pool = _union_seg_matrices(a_pos, order_adequacy, order_fluency, blocks).mean(axis=1)
  f_pool = _union_seg_matrices(f_pos, order_adequacy, order_fluency, blocks).mean(axis=1)
  return float(_beta_0(a_pool, f_pool))


def synth25_scorer_scores(
    dataset: str,
    metametric_name: str,
    root: str = '.',
    systems: list[str] | None = None,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
  """Scorer name -> PA_synth25 or SPA_synth25 value against the All-MQM
  human ranking, on the Row-7 union pool (real systems UNION synthesized-
  by-adequacy UNION synthesized-by-fluency, 3K systems). `metametric_name`
  must be 'pa_synth25' or 'spa_synth25'.

  systems: defaults to real_systems(dataset) (this project's standard
  screened roster, src.lib.consistency.real_systems); these K real systems are
  also the ones the two synthesized sets are built from.

  Empty Series if positional_af_matrices can't validate this dataset's
  Adequacy/Fluency-MQM-to-scorer-segment-score alignment at all (module
  docstring), or if `systems` is empty. Per scorer, silently skipped (not
  included in the result) if its .seg.score file is missing, doesn't cover
  every requested system, or leaves fewer than _MIN_SPA_SEGMENTS jointly-
  valid positions once Adequacy MQM, Fluency MQM, All MQM, and the
  scorer's own segment scores are all masked together."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  if systems is None:
    systems = real_systems(dataset, root=root)
  if not systems:
    return pd.Series(dtype=float)

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return pd.Series(dtype=float)
  a_pos, f_pos = pos

  human_seg_all = load_human_seg_scores(dataset, systems, root=root)
  if human_seg_all is None or human_seg_all.shape[1] != a_pos.shape[1]:
    return pd.Series(dtype=float)

  candidates = load_scorer_sys_scores(dataset, systems, root=root)
  values = {}
  for name in candidates:
    scorer_seg = load_scorer_seg_scores(dataset, name, systems, root=root)
    if scorer_seg is None or scorer_seg.shape[1] != a_pos.shape[1]:
      continue
    mask = ~(
        np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
        | np.isnan(human_seg_all).any(axis=0) | np.isnan(scorer_seg).any(axis=0)
    )
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue

    val = synth25_value(
        metametric_name, human_seg_all[:, mask], scorer_seg[:, mask], a_pos[:, mask], f_pos[:, mask],
        tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
    )
    if val == val:  # excludes NaN
      values[name] = val
  return pd.Series(values, dtype=float)
