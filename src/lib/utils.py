"""Shared helpers factored out of near-identical code that had accumulated
across src.lib.synthetic_scorer_beta_grid and src.lib.synth25_preference --
both build dial families for one base scorer against the same coverage
gate, just scoring the result differently downstream (a weighted-metametric
grid vs. a synth25 preference reduction). See each function's own
docstring for exactly which call sites it replaces.
"""

from __future__ import annotations

import numpy as np

from src.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from src.lib.spa_plane import positional_af_matrices
from src.lib.synthetic_scorers import AF_ADDITIVE_MEAN_DIAL_GRID, base_scorer_family_additive_mean_af


def coverage_mask(*arrays: np.ndarray) -> np.ndarray:
  """Boolean mask over the shared segment axis (axis=1): True wherever
  every one of `arrays` (each (K, N), same N) is non-NaN in every row --
  the project's shared segment-coverage gate, applied before trusting a
  downstream SPA permutation test or per-segment family construction."""
  return ~np.any([np.isnan(a).any(axis=0) for a in arrays], axis=0)


# "Fake base scorer" names: instead of a real scorer's segment scores,
# base_scorer_seg is one of the gold aspect signals itself (a_pos/f_pos/
# human_seg, already loaded for every base scorer anyway) -- i.e. what
# happens if the base scorer IS a perfect oracle for one aspect (or their
# sum), dialed through the exact same family/beta/metametric machinery as
# any real base scorer. AllMQM uses human_seg (the official All-MQM file,
# src.lib.mqm_scoring's 't'/all_mqm column) rather than a_pos+f_pos, since
# that's the literal ground-truth total rather than a derived sum of the
# two sub-scores.
ASPECT_BASE_SCORERS = ('AdequacyMQM', 'FluencyMQM', 'AllMQM')


def aspect_base_scorer_seg(base_scorer_name: str, a_pos: np.ndarray, f_pos: np.ndarray, human_seg: np.ndarray) -> np.ndarray:
  if base_scorer_name == 'AdequacyMQM':
    return a_pos
  if base_scorer_name == 'FluencyMQM':
    return f_pos
  if base_scorer_name == 'AllMQM':
    return human_seg
  raise ValueError(f'not an aspect base scorer: {base_scorer_name!r}')


def load_base_scorer_coverage(
    dataset: str, base_scorer_name: str, systems: list[str], root: str = '.', min_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
  """(a_pos, f_pos, base_scorer_seg, human_seg), each already restricted to
  their jointly-valid segment columns -- the shared load-and-gate pipeline
  every family-building entry point in this project applies before
  building dial families for one base scorer (a real scorer's segment
  file, or one of ASPECT_BASE_SCORERS). None if the base scorer lacks a
  usable segment file, dataset alignment can't be validated against
  `systems`, or (when min_segments is given) fewer than that many segment
  columns survive the joint mask -- callers pass their own local
  _MIN_SPA_SEGMENTS explicitly rather than relying on a shared default
  here, so this stays decoupled from any one module's own threshold."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = coverage_mask(a_pos, f_pos, base_scorer_seg, human_seg)
  if min_segments is not None and mask.sum() < min_segments:
    return None
  return a_pos[:, mask], f_pos[:, mask], base_scorer_seg[:, mask], human_seg[:, mask]


def build_af_or_split_families(
    build_family, base_scorer_seg: np.ndarray, a_pos: np.ndarray, f_pos: np.ndarray, dial_grid,
    use_af_family: bool, af_dial_grid=None,
) -> dict[str, dict]:
  """{'AF': ...} if use_af_family else {'A': ..., 'F': ...} -- the shared
  adequacy/fluency half of every dial-family dispatch in this project
  (src.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid,
  src.lib.synth25_preference.base_scorer_synth25_preference*). `build_family`
  is the synthesis generator (SYNTHESIS_GENERATORS[synthesis]) used for the
  non-AF branch. AF's own dial grid (default AF_ADDITIVE_MEAN_DIAL_GRID) is
  unrelated to `dial_grid` -- pass af_dial_grid explicitly to override it.
  Callers add their own third family ('J' or 'T') to the returned dict on
  top of this, since that choice varies by call site/synthesis mode."""
  if use_af_family:
    return {'AF': base_scorer_family_additive_mean_af(
        base_scorer_seg, a_pos, f_pos, af_dial_grid or AF_ADDITIVE_MEAN_DIAL_GRID)}
  return {'A': build_family(base_scorer_seg, a_pos, dial_grid), 'F': build_family(base_scorer_seg, f_pos, dial_grid)}
