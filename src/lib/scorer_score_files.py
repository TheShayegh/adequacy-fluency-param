"""Loading automatic-scorer scores (external/mt-metrics-eval-data/
mt-metrics-eval-v2/<year>/metric-scores/<pair>/*.{sys,seg}.score) aligned to
our canonical MQM system set, for the meta-evaluation / consistency
experiments.

System names in metric-scores files mostly match wmt-mqm-human-evaluation's
exactly; wmt20 is the one confirmed exception (trailing ".NNNN" submission-id
suffix, and different case -- "UEDIN.1136" vs "UEdin"), handled via
src.lib.wmt_metadata.join_key. Metric-scores rosters are also wider than our
MQM-rated roster (WMT restricts human evaluation to a curated top-scoring
subset, not every submission); we keep only the intersection.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from src.lib.wmt_metadata import DATASET_DIRS, is_reference_or_human, join_key, split_variant

_ROOT_DATA = 'external/mt-metrics-eval-data/mt-metrics-eval-v2'

# Preference order when a dataset scores one scorer family against multiple
# references (e.g. ende24's BLEU-refA and BLEU-refB): pick the first
# variant present. "refA" is WMT's primary reference from 2021 on; "ref" is
# wmt20's primary reference (Human-A). Remaining/QE-only variants fall back
# to alphabetical order.
_VARIANT_PRIORITY = ('refA', 'ref', 'all', 'src')


def _scorer_dir(dataset: str, root: str) -> str:
  year, pair = DATASET_DIRS[dataset]
  return os.path.join(root, _ROOT_DATA, year, 'metric-scores', pair)


def discover_scorers(dataset: str, root: str = '.') -> dict[str, str]:
  """Stripped scorer name -> chosen '<scorer>-<variant>' file stem, for
  every scorer with a .sys.score file in this dataset (src.lib.wmt_metadata.
  split_variant defines "stripped"; see module docstring there for why
  version-tagged families are deliberately left unmerged)."""
  d = _scorer_dir(dataset, root)
  by_name = defaultdict(list)
  for fh in os.listdir(d):
    if not fh.endswith('.sys.score'):
      continue
    stem = fh[: -len('.sys.score')]
    name, variant = split_variant(stem)
    by_name[name].append((variant, stem))

  def sort_key(v):
    variant, stem = v
    rank = _VARIANT_PRIORITY.index(variant) if variant in _VARIANT_PRIORITY else len(_VARIANT_PRIORITY)
    return (rank, stem)

  return {name: sorted(variants, key=sort_key)[0][1] for name, variants in by_name.items()}


def _load_score_file(path: str) -> dict[str, list[float]]:
  """Parses a "<system>\\t<score>" score file. wmt20's human-scores files
  are the one exception -- space-delimited, not tab -- so fall back to
  splitting on the last space when no tab is present."""
  scores = defaultdict(list)
  with open(path) as fh:
    for line in fh:
      line = line.rstrip('\n')
      if not line.strip():
        continue
      parts = line.split('\t') if '\t' in line else line.rsplit(' ', 1)
      if len(parts) != 2:
        continue
      sysname, val = parts
      scores[sysname].append(float(val) if val not in ('None', '') else float('nan'))
  return scores


def _canonical_map(mqm_systems) -> dict[str, str]:
  return {join_key(s): s for s in mqm_systems if not is_reference_or_human(s)}


def load_scorer_sys_scores(dataset: str, mqm_systems, root: str = '.') -> dict[str, pd.Series]:
  """Stripped scorer name -> system-level Series indexed by the canonical
  MQM system names in `mqm_systems` (real systems only). A scorer is
  included only if it covers every one of those systems with a numeric
  score -- partial coverage can't produce a fair scorer-ranking entry."""
  key_to_canonical = _canonical_map(mqm_systems)
  wanted = set(key_to_canonical.values())
  d = _scorer_dir(dataset, root)
  out = {}
  for name, stem in discover_scorers(dataset, root).items():
    raw = _load_score_file(os.path.join(d, stem + '.sys.score'))
    vals = {}
    for raw_name, xs in raw.items():
      canon = key_to_canonical.get(join_key(raw_name))
      if canon is not None:
        vals[canon] = xs[0]
    if set(vals) >= wanted and all(v == v for v in vals.values()):  # v==v excludes NaN
      out[name] = pd.Series({s: vals[s] for s in sorted(wanted)})
  return out


def _seg_matrix(raw: dict[str, list[float]], key_to_canonical: dict[str, str],
                 systems_order: list[str]) -> np.ndarray | None:
  by_canonical = {}
  for raw_name, xs in raw.items():
    canon = key_to_canonical.get(join_key(raw_name))
    if canon is not None:
      by_canonical[canon] = xs
  if not set(systems_order) <= set(by_canonical):
    return None
  lengths = {len(by_canonical[s]) for s in systems_order}
  if len(lengths) != 1:
    return None
  return np.array([by_canonical[s] for s in systems_order])


def load_human_seg_scores(dataset: str, systems_order: list[str], root: str = '.') -> np.ndarray | None:
  """(K, n_positions) All-MQM segment-score matrix from the official
  <pair>.mqm.seg.score file (positionally aligned to sources/<pair>.txt),
  row order == systems_order. NaN where the official file has "None"."""
  year, pair = DATASET_DIRS[dataset]
  path = os.path.join(root, _ROOT_DATA, year, 'human-scores', f'{pair}.mqm.seg.score')
  if not os.path.exists(path):
    return None
  raw = _load_score_file(path)
  key_to_canonical = _canonical_map(systems_order)
  return _seg_matrix(raw, key_to_canonical, systems_order)


def load_scorer_seg_scores(dataset: str, scorer_name: str, systems_order: list[str],
                            root: str = '.') -> np.ndarray | None:
  """(K, n_positions) segment-score matrix for one (already-discovered)
  scorer, row order == systems_order, or None if its .seg.score file is
  missing or doesn't cover every requested system."""
  stem = discover_scorers(dataset, root).get(scorer_name)
  if stem is None:
    return None
  path = os.path.join(_scorer_dir(dataset, root), stem + '.seg.score')
  if not os.path.exists(path):
    return None
  raw = _load_score_file(path)
  key_to_canonical = _canonical_map(systems_order)
  return _seg_matrix(raw, key_to_canonical, systems_order)


def jointly_valid_columns(human_seg: np.ndarray, scorer_seg: np.ndarray) -> np.ndarray:
  """Boolean mask over segment positions valid (non-NaN) for every system
  in both matrices -- SPA's permutation test needs a fully numeric matrix,
  and coverage can be ragged in either source independently."""
  return ~(np.isnan(human_seg).any(axis=0) | np.isnan(scorer_seg).any(axis=0))
