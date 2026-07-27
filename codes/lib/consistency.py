"""Cross-dataset scorer-ranking consistency (action_plan.md section 6.5):
for a chosen meta-metric, score every available automatic metric (scorer)
against the human ranking within each dataset, then measure how consistent
those per-dataset scorer-rankings are with each other.

Implemented as a single Kendall's tau pooled over every (scorer-pair,
dataset-pair) comparison at once, using the tau-a convention (fixed
denominator C(n_ij, 2) per dataset pair, ties contributing 0 to the
numerator but still counted in the denominator) -- this is mathematically
identical to the scorer-pair-count-weighted average of per-pair Kendall's
taus derived in action_plan.md section 6.5, just computed directly rather
than as a weighted sum of separately-computed tau_ij's.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.metametrics import METAMETRICS, WEIGHTED_METAMETRICS, NEEDS_SEGMENT_SCORES, MetaEvalInput
from lib.metric_scores import (
    load_metric_sys_scores, load_human_seg_scores, load_metric_seg_scores,
    jointly_valid_columns,
)
from lib.outlier_detection import wmt_official_outliers
from lib.systems import is_reference_or_human
from mwb.mqm_scoring import load_system_scores

# SPA's permutation test is unreliable with too few shared, fully-covered
# segments; below this we skip the (dataset, metric) cell rather than trust
# a noisy p-value (mirrors action_plan.md 6.5's own ESS-floor caution for
# SPA in the subset-consistency experiment).
_MIN_SPA_SEGMENTS = 10

# Datasets missing GRANULAR (line-position-indexed) segment-level Adequacy/
# Fluency MQM: mwb.mqm_scoring's raw-TSV seg_id for these is a per-document
# index, not a global line position in the source file, so it cannot be
# positionally reconciled with automatic-metric segment-score files
# (lib.spa_plane.positional_af_matrices's self-check found 0.35 correlation
# for enru22 vs. 0.99+ for every other raw-TSV/official-ratings dataset --
# i.e. genuinely misaligned, not just noisy). This ONLY breaks analyses that
# need that positional alignment (currently just the SPA plane's scorer
# points -- system-level scores and mqm_scoring's own (doc,doc_id,seg_id)-
# keyed segment matrices are both unaffected), so it is opt-in via
# real_systems(..., excl_missing_seg_granular_mqm=True) rather than dropped
# from the plain default the way WMT_OFFICIAL_UNSUPPORTED's 2020 sets are.
MISSING_SEG_GRANULAR_MQM = frozenset({'enru22'})


def real_systems(
    dataset: str, root: str = '.', exclude_outliers: bool = True,
    excl_missing_seg_granular_mqm: bool = False,
) -> list[str]:
  """Sorted real (non-reference/human) system names for a dataset -- the
  canonical ordering scorer_scores() uses internally, and the ordering any
  caller passing an explicit weight vector `w` to scorer_scores must use to
  build it (e.g. lib.reweight_exact.solve_w_exact(a, b, alpha) with a,b
  built in this same order).

  exclude_outliers (default True): THE PROJECT STANDARD as of this
  session -- lib.outlier_detection.wmt_official_outliers is now the
  project's default outlier-removal method, applied here so every caller
  of real_systems() automatically works on the screened pool without
  applying any screen itself. This INCLUDES wmt_official's own
  "unsupported" stance on the 2020 datasets: real_systems('ende20'/
  'zhen20') now returns an EMPTY list by default, not their raw 7/8-system
  roster, since none of WMT20's officially-listed outliers exist in this
  project's smaller raw-TSV 2020 subset to be verified against, so
  wmt_official cannot certify them clean and drops them entirely rather
  than silently claiming otherwise. Pass exclude_outliers=False to get the
  untouched raw roster instead -- needed by, among others, wmt_official_
  outliers/wmt_non_competative_outliers's own validation (which checks
  their hardcoded names against a roster that must still contain them)
  and any "no removal" baseline that needs the genuinely raw pool.

  excl_missing_seg_granular_mqm (default False): pass True to ALSO drop
  MISSING_SEG_GRANULAR_MQM datasets (currently just enru22) entirely -- for
  callers that specifically need line-position-aligned segment MQM (e.g.
  lib.spa_plane's scorer points) and would otherwise silently get a roster
  they can't use for that purpose. False (the default) leaves enru22 in the
  pool, since every other analysis in this project has no such need.
  Bypassed (like exclude_outliers's own drops) by exclude_outliers=False."""
  sys_df = load_system_scores(dataset, root=root)
  raw = sorted(s for s in sys_df.index if not is_reference_or_human(s))
  if not exclude_outliers:
    return raw
  if excl_missing_seg_granular_mqm and dataset in MISSING_SEG_GRANULAR_MQM:
    return []
  outliers = wmt_official_outliers(dataset, raw)
  return [s for s in raw if s not in outliers]


def load_scorer_inputs(
    dataset: str, metametric_name: str, root: str = '.', systems: list[str] | None = None,
) -> dict[str, MetaEvalInput]:
  """The expensive, weighting-independent I/O step of scorer_scores
  (parsing human/metric score files off disk), factored out so callers
  evaluating many different weightings at fixed (dataset, metametric) --
  e.g. an alpha grid, lib.reweighted_consistency -- can do it once and
  reuse it rather than re-reading every score file per alpha. Returns
  scorer name -> MetaEvalInput (human_sys/metric_sys, and for SPA the
  already-jointly-masked human_seg/metric_seg); no weight is applied here.

  systems: defaults to all of `dataset`'s real systems (real_systems); pass
  an explicit subset to restrict to it instead -- e.g. one bootstrap
  resample of the dataset's systems (lib.bootstrap.sample_subsets,
  lib.bootstrap_consistency), rather than every real system in it.
  """
  if systems is None:
    systems = real_systems(dataset, root=root)
  sys_df = load_system_scores(dataset, root=root)
  human_sys = sys_df.loc[systems, 't'].values
  metric_series = load_metric_sys_scores(dataset, systems, root=root)

  needs_seg = metametric_name in NEEDS_SEGMENT_SCORES
  human_seg = load_human_seg_scores(dataset, systems, root=root) if needs_seg else None

  inputs = {}
  for name, series in metric_series.items():
    m = series.reindex(systems).values
    if needs_seg:
      if human_seg is None:
        continue
      metric_seg = load_metric_seg_scores(dataset, name, systems, root=root)
      if metric_seg is None:
        continue
      mask = jointly_valid_columns(human_seg, metric_seg)
      if mask.sum() < _MIN_SPA_SEGMENTS:
        continue
      inputs[name] = MetaEvalInput(human_sys, m, human_seg[:, mask], metric_seg[:, mask])
    else:
      inputs[name] = MetaEvalInput(human_sys, m)
  return inputs


def evaluate_scorer_scores(
    inputs: dict[str, MetaEvalInput], metametric_name: str, w: np.ndarray | None = None,
) -> pd.Series:
  """Cheap step of scorer_scores: apply the (weighted or unweighted)
  meta-metric to already-loaded inputs (load_scorer_inputs)."""
  fn = WEIGHTED_METAMETRICS[metametric_name] if w is not None else METAMETRICS[metametric_name]
  values = {}
  for name, x in inputs.items():
    val = fn(x, w) if w is not None else fn(x)
    if val == val:  # excludes NaN
      values[name] = val
  return pd.Series(values, dtype=float)


def scorer_scores(
    dataset: str, metametric_name: str, root: str = '.', w: np.ndarray | None = None,
    systems: list[str] | None = None,
) -> pd.Series:
  """Metric name -> meta-metric value against the human All-MQM ranking,
  for every scorer with full, numeric coverage of `systems` (default: all
  of this dataset's real, non-reference systems -- real_systems).

  w: optional system-weight vector, aligned to `systems` order (or
  real_systems(dataset, root) order if systems is None). None (default)
  uses the unweighted meta-metric (METAMETRICS, equivalent to uniform
  weight); otherwise uses the weighted counterpart (WEIGHTED_METAMETRICS)
  under that weighting -- e.g. w = lib.reweight_exact.solve_w_exact(a,
  b, alpha).w for an M(alpha)-curve point.

  One-shot convenience wrapping load_scorer_inputs + evaluate_scorer_
  scores; callers evaluating many weightings at fixed (dataset,
  metametric, systems) should call those directly to avoid redundant I/O.
  """
  resolved_systems = systems if systems is not None else real_systems(dataset, root=root)
  if w is not None and len(w) != len(resolved_systems):
    raise ValueError(f'{dataset}: w has length {len(w)}, expected {len(resolved_systems)}')
  inputs = load_scorer_inputs(dataset, metametric_name, root=root, systems=resolved_systems)
  return evaluate_scorer_scores(inputs, metametric_name, w)


@dataclasses.dataclass
class DatasetPairResult:
  dataset_i: str
  dataset_j: str
  n_common: int          # shared scorers
  weight: int             # C(n_common, 2), scorer-pairs tau_ij is computed over
  tau: float


def pool_weighted_tau(
    rankings: dict[str, pd.Series], datasets: list[str] | None = None,
) -> tuple[float, int, list[DatasetPairResult]]:
  """The action_plan.md 6.5 pooling: given each dataset's scorer-ranking
  (as produced by scorer_scores, weighted or not), returns (tau_bar,
  total_weight, per-dataset-pair detail) -- one Kendall's tau pooled over
  every (scorer-pair, dataset-pair) comparison, tau-a convention (see
  module docstring). Shared by weighted_consistency (unweighted/baseline
  rankings) and lib.reweighted_consistency (rankings under w(alpha))."""
  datasets = list(rankings) if datasets is None else datasets
  pair_results = []
  total_num = 0
  total_den = 0
  for i, j in itertools.combinations(datasets, 2):
    common = sorted(set(rankings[i].index) & set(rankings[j].index))
    n = len(common)
    if n < 2:
      pair_results.append(DatasetPairResult(i, j, n, 0, float('nan')))
      continue
    w = comb(n, 2)
    # Vectorized replacement of the old itertools.combinations(common, 2)
    # Python loop with per-pair pandas Series.__getitem__ lookups (profiled:
    # that pattern was 83% pandas-indexing overhead, not the actual
    # comparison) -- reindex once to plain numpy arrays in `common`'s order,
    # then compare every pair at once via triu_indices broadcasting. Same
    # tau-a convention: a tie in either ranking drops the pair from the
    # numerator but it still counts toward the denominator w.
    vi = rankings[i].reindex(common).values
    vj = rankings[j].reindex(common).values
    iu, ju = np.triu_indices(n, 1)
    si = np.sign(vi[iu] - vi[ju])
    sj = np.sign(vj[iu] - vj[ju])
    untied = (si != 0) & (sj != 0)
    concordant = int(np.count_nonzero(untied & (si == sj)))
    discordant = int(np.count_nonzero(untied & (si != sj)))
    tau_ij = (concordant - discordant) / w
    pair_results.append(DatasetPairResult(i, j, n, w, tau_ij))
    total_num += concordant - discordant
    total_den += w

  tau_bar = total_num / total_den if total_den else float('nan')
  return tau_bar, total_den, pair_results


def weighted_consistency(
    metametric_name: str, datasets: list[str], root: str = '.',
    systems_by_dataset: dict[str, list[str]] | None = None,
) -> dict:
  """systems_by_dataset: restrict dataset d's systems to
  systems_by_dataset[d] instead of real_systems(d) -- e.g. a LOO/
  bootstrap/exclude-system subset of one base dataset, id'd by a synthetic
  `d` (see lib.reweighted_consistency.solve_w_for_datasets)."""
  rankings = {d: scorer_scores(d, metametric_name, root=root, systems=(systems_by_dataset or {}).get(d))
              for d in datasets}
  tau_bar, total_den, pair_results = pool_weighted_tau(rankings, datasets)
  return {
      'metametric': metametric_name,
      'rankings': rankings,
      'pairs': pair_results,
      'tau_bar': tau_bar,
      'inconsistency': (1 - tau_bar) if tau_bar == tau_bar else float('nan'),
      'total_weight': total_den,
  }
