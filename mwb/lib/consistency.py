"""Cross-dataset scorer-ranking consistency: for a chosen meta-metric,
score every available automatic metric (scorer) against the human ranking
within each dataset, then measure how consistent those per-dataset
scorer-rankings are with each other.

Implemented as a single Kendall's tau pooled over every (scorer-pair,
dataset-pair) comparison at once, using the tau-a convention (fixed
denominator C(n_ij, 2) per dataset pair, ties contributing 0 to the
numerator but still counted in the denominator) -- this is mathematically
identical to the scorer-pair-count-weighted average of per-pair Kendall's
taus, just computed directly rather than as a weighted sum of
separately-computed tau_ij's.
"""

from __future__ import annotations

import dataclasses
import itertools
from math import comb

import numpy as np
import pandas as pd

from mwb.lib.metametrics import METAMETRICS, WEIGHTED_METAMETRICS, NEEDS_SEGMENT_SCORES, MetaEvalInput
from mwb.lib.metric_scores import (
    load_metric_sys_scores, load_human_seg_scores, load_metric_seg_scores,
    jointly_valid_columns,
)
from mwb.lib.outlier_detection import wmt_official_outliers, wmt_non_competative_outliers
from mwb.lib.systems import is_reference_or_human
from mwb.mqm_scoring import load_system_scores

# SPA's permutation test is unreliable with too few shared, fully-covered
# segments; below this we skip the (dataset, metric) cell rather than trust
# a noisy p-value.
_MIN_SPA_SEGMENTS = 10

# Datasets missing GRANULAR (line-position-indexed) segment-level Adequacy/
# Fluency MQM: mwb.mqm_scoring's raw-TSV seg_id for these is a per-document
# index, not a global line position in the source file, so it cannot be
# positionally reconciled with automatic-metric segment-score files
# (mwb.lib.spa_plane.positional_af_matrices's self-check found 0.35 correlation
# for enru22 vs. 0.99+ for every other raw-TSV/official-ratings dataset --
# i.e. genuinely misaligned, not just noisy). This ONLY breaks analyses that
# need that positional alignment (currently just the SPA plane's scorer
# points -- system-level scores and mqm_scoring's own (doc,doc_id,seg_id)-
# keyed segment matrices are both unaffected), so it is opt-in via
# real_systems(..., excl_missing_seg_granular_mqm=True) rather than dropped
# from the plain default the way WMT_OFFICIAL_UNSUPPORTED's 2020 sets are.
MISSING_SEG_GRANULAR_MQM = frozenset({'enru22'})


# Datasets manually excluded from every analysis in this project, decided
# outside of any statistical/wmt_official detector -- see each entry's own
# reason. Only takes effect when exclude_outliers=True (same bypass rule as
# MISSING_SEG_GRANULAR_MQM below), so wmt_official_outliers/wmt_non_
# competative_outliers's own hardcoded-name validation and any genuinely-
# raw-pool caller (exclude_outliers=False) still see the true roster.
MANUALLY_EXCLUDED_DATASETS = frozenset({
    # ende23: by far the largest "uncategorized MQM" component of any
    # dataset (segment-level std 1.50 vs. <=0.34 everywhere else), driven
    # almost entirely by MQM's vague 'Other' catch-all category (407 of
    # 766 uncategorized errors; 'Source issue' contributes zero weight) --
    # too many of its annotations aren't cleanly assignable to adequacy or
    # fluency to trust the split (matches the paper's own stated reason
    # for excluding it, Sec. "Evaluation Setup"). Excluded project-wide.
    'ende23',
})

# real_scorers' zero-variance test for "this scorer cannot rank the roster
# at all". Exact rather than statistical: the scorers it catches assign
# literally the same float to every system, so any positive tolerance well
# under float noise on the score scale works.
_DEGENERATE_TOL = 1e-12


def real_systems(
    dataset: str, root: str = '.', exclude_outliers: bool = True,
    excl_missing_seg_granular_mqm: bool = False, exclude_non_competative: bool = False,
) -> list[str]:
  """Sorted real (non-reference/human) system names for a dataset -- the
  canonical ordering scorer_scores() uses internally, and the ordering any
  caller passing an explicit weight vector `w` to scorer_scores must use to
  build it (e.g. mwb.lib.reweight_exact.solve_w_exact(a, b, alpha) with a,b
  built in this same order).

  exclude_outliers (default True): THE PROJECT STANDARD as of this
  session -- mwb.lib.outlier_detection.wmt_official_outliers is now the
  project's default outlier-removal method, applied here so every caller
  of real_systems() automatically works on the screened pool without
  applying any screen itself. This INCLUDES wmt_official's own
  "unsupported" stance on the 2020 datasets: real_systems('ende20'/
  'zhen20') now returns an EMPTY list by default, not their raw 7/8-system
  roster, since none of WMT20's officially-listed outliers exist in this
  project's smaller raw-TSV 2020 subset to be verified against, so
  wmt_official cannot certify them clean and drops them entirely rather
  than silently claiming otherwise. It ALSO includes MANUALLY_EXCLUDED_
  DATASETS (currently just ende23, see its own comment above): real_
  systems('ende23') now returns an EMPTY list too, project-wide, by
  explicit decision rather than any statistical detector.

  exclude_non_competative (default False): pass True to ALSO drop
  wmt_non_competative_outliers (WMT_NON_COMPETATIVE_SYSTEMS) -- hand-curated
  WMT organizer-inserted systems (calibration placeholders, MBR-reranking
  baselines, MSLC) that are not independent MT submissions, which wmt_
  official's own outlier metadata doesn't flag (a different concern --
  wmt_official is about ranking-quality outliers, this is about roster
  membership). Tried as a project-wide default in this session -- the
  scorer-orientation investigation that surfaced it found solve_w_exact's
  optimal support pivoting onto zhen21's 5 metricsystemN placeholders
  (K=13, so >1/3 of the whole roster) around alpha~0.71-0.85 -- but the
  result of turning it on by default project-wide wasn't good, so it's
  back to opt-in, off by default, rather than folded into exclude_outliers.

  Pass exclude_outliers=False to get the untouched raw roster instead --
  needed by, among others, wmt_official_outliers/wmt_non_competative_
  outliers's own validation (which checks their hardcoded names against a
  roster that must still contain them) and any "no removal" baseline that
  needs the genuinely raw pool.

  excl_missing_seg_granular_mqm (default False): pass True to ALSO drop
  MISSING_SEG_GRANULAR_MQM datasets (currently just enru22) entirely -- for
  callers that specifically need line-position-aligned segment MQM (e.g.
  mwb.lib.spa_plane's scorer points) and would otherwise silently get a roster
  they can't use for that purpose. False (the default) leaves enru22 in the
  pool, since every other analysis in this project has no such need.
  Bypassed (like exclude_outliers's own drops) by exclude_outliers=False."""
  sys_df = load_system_scores(dataset, root=root)
  raw = sorted(s for s in sys_df.index if not is_reference_or_human(s))
  if not exclude_outliers:
    return raw
  if dataset in MANUALLY_EXCLUDED_DATASETS:
    return []
  if excl_missing_seg_granular_mqm and dataset in MISSING_SEG_GRANULAR_MQM:
    return []
  outliers = wmt_official_outliers(dataset, raw)
  if exclude_non_competative:
    outliers = outliers | wmt_non_competative_outliers(dataset, raw)
  return [s for s in raw if s not in outliers]


def real_scorers(
    dataset: str, root: str = '.', systems: list[str] | None = None,
    require_seg_scores: bool = False, exclude_degenerate: bool = True,
    deduplicate: bool = True,
) -> list[str]:
  """Sorted usable scorer (automatic metric) names for a dataset -- the
  scorer-side analogue of real_systems(), which screens the OTHER axis of
  the same rectangle. Neither implies the other: real_systems drops
  translation systems (outliers, excluded datasets), this drops metrics.

  systems: the roster to evaluate the scorers against; defaults to
  real_systems(dataset) with excl_missing_seg_granular_mqm tied to
  require_seg_scores, since a caller needing segment scores needs the
  segment-aligned roster too. Returns [] when the roster is empty.

  exclude_degenerate (default True): drop scorers that assign every system
  the identical score, i.e. that cannot rank the roster at all. Detected,
  not hardcoded -- it is an exact zero-variance test with no threshold to
  tune. Currently this is exactly WMT24's `sentinel-ref-mqm` and
  `sentinel-src-mqm` in ende24/enes24/jazh24, which score the reference or
  the source alone and ignore the candidate translation entirely (they
  exist to expose metric artifacts, and are not metrics of translation
  quality). `sentinel-cand-mqm` does score the candidate and is NOT
  degenerate, so it survives -- which is why this is a data test rather
  than a name blacklist. Such scorers make every ratio-to-baseline
  quantity 0/0 and every ranking metametric undefined.

  deduplicate (default True): several WMT24 submissions are byte-identical
  to each other (ende24/enes24/jazh24 XLsimDA == XLsimMqm, plus three
  metametrics_mt_mqm_* pairs). Keeping both double-counts one scorer in
  any "distribution across scorers" statistic, so only the alphabetically
  first of each identical group is kept. Deduplication happens at the
  level the caller will actually read -- segment matrices when
  require_seg_scores, system score vectors otherwise -- because the two
  disagree: a .sys.score file is supplied by the submitter, not recomputed
  as the mean of that metric's own .seg.score file, so enes24's
  metametrics_mt_mqm_kendall matches metametrics_mt_mqm_hybrid_kendall
  segment-for-segment while their system vectors differ.

  require_seg_scores (default False, mirroring real_systems'
  excl_missing_seg_granular_mqm opt-in): also drop scorers with no usable
  segment-level file, for callers that read segment scores and would
  otherwise get names they cannot use. Costs a segment-file read per
  scorer. Currently drops 20 (scorer, dataset) pairs, including BLEU on
  every wmt21 set and the six ende22/zhen22 HuaweiTSC_EE_BERTScore
  variants. Left in by default, since system-level-only analyses can use
  them."""
  if systems is None:
    systems = real_systems(dataset, root=root,
                           excl_missing_seg_granular_mqm=require_seg_scores)
  if not systems:
    return []

  sys_scores = load_metric_sys_scores(dataset, systems, root=root)
  names = sorted(sys_scores)

  if exclude_degenerate:
    names = [
        s for s in names
        if np.nanstd(sys_scores[s].to_numpy(dtype=float)) > _DEGENERATE_TOL
    ]

  seg = {}
  if require_seg_scores:
    for s in names:
      m = load_metric_seg_scores(dataset, s, systems, root=root)
      if m is not None and m.shape[0] == len(systems) and not np.isnan(m).all():
        seg[s] = m
    names = [s for s in names if s in seg]

  if deduplicate:
    kept, seen = [], set()
    for s in names:
      key = (seg[s] if require_seg_scores
             else np.round(sys_scores[s].to_numpy(dtype=float), 12)).tobytes()
      if key not in seen:
        seen.add(key)
        kept.append(s)
    names = kept

  return names


def load_scorer_inputs(
    dataset: str, metametric_name: str, root: str = '.', systems: list[str] | None = None,
) -> dict[str, MetaEvalInput]:
  """The expensive, weighting-independent I/O step of scorer_scores
  (parsing human/metric score files off disk), factored out so callers
  evaluating many different weightings at fixed (dataset, metametric) --
  e.g. an alpha grid, mwb.lib.reweighted_consistency -- can do it once and
  reuse it rather than re-reading every score file per alpha. Returns
  scorer name -> MetaEvalInput (human_sys/metric_sys, and for SPA the
  already-jointly-masked human_seg/metric_seg); no weight is applied here.

  systems: defaults to all of `dataset`'s real systems (real_systems); pass
  an explicit subset to restrict to it instead -- e.g. a leave-some-out
  subset, rather than every real system in it.
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
  under that weighting -- e.g. w = mwb.lib.reweight_exact.solve_w_exact(a,
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
  """Pools each dataset's scorer-ranking (as produced by scorer_scores,
  weighted or not) into (tau_bar, total_weight, per-dataset-pair detail) --
  one Kendall's tau pooled over every (scorer-pair, dataset-pair)
  comparison, tau-a convention (see module docstring). Shared by
  weighted_consistency (unweighted/baseline rankings) and
  mwb.lib.reweighted_consistency (rankings under w(alpha))."""
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
  `d` (see mwb.lib.reweighted_consistency.solve_w_for_datasets)."""
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
