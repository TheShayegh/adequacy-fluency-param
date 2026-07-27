"""Meta-evaluation of the adequacy/fluency components themselves as
"scorers" (action_plan.md's a, b vectors), reusing lib.metametrics' five
meta-metrics (PA, SPA, Pearson, Spearman, Kendall's tau) exactly the way
lib.consistency does for external automatic metrics -- except here the
"scorer" being judged is Adequacy MQM alone or Fluency MQM alone, judged
against the All-MQM (t = a+b) human ranking, per dataset. Missing from
action_plan.md's evaluation plan (section 6): the plan reports each real
external metric's meta-eval score, but never scores the two MQM components
against each other this way.

This underlies compute_af_ratio_vs_alpha0.py's cross-dataset check: does a
dataset's natural balance alpha_0 (lib.alpha.alpha_0 -- the extent to which
the system pool is adequacy-heavy) predict the ratio
metametric(adequacy_mqm; dataset) / metametric(fluency_mqm; dataset) on that
same dataset, i.e. does the reported balance actually show up as an
asymmetry in how well each raw component alone recovers the human ranking?
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from lib.consistency import real_systems
from lib.metametrics import (
    METAMETRICS, NEEDS_SEGMENT_SCORES, MetaEvalInput,
    pairwise_p_values, soft_pairwise_accuracy_from_pvalues,
)
from mwb.mqm_scoring import load_system_scores, load_segment_scores

SCORERS = ('adequacy_mqm', 'fluency_mqm')

# Same floor lib.consistency uses before trusting SPA's permutation p-values
# (segment counts below this make the permutation test's p-values noisy).
_MIN_SPA_SEGMENTS = 10

_SEGMENT_KEYS = ['doc', 'doc_id', 'seg_id']


def _af_seg_matrices(dataset: str, systems: list[str], root: str = '.'):
  """(human_seg, adequacy_seg, fluency_seg), each a (K, n_full_segments)
  array row-ordered as `systems`, restricted to segments every one of
  `systems` rated -- mirrors lib.synthesis.synthesize_pool's
  pivot_table + dropna(how='any') pattern, and lib.metric_scores.
  _seg_matrix's row-order/shape contract for MetaEvalInput. Returns
  (None, None, None) if fewer than _MIN_SPA_SEGMENTS segments are fully
  covered."""
  seg_df = load_segment_scores(dataset, root=root)
  seg_df = seg_df[seg_df['system'].isin(systems)]
  wide_all = seg_df.pivot_table(index=_SEGMENT_KEYS, columns='system', values='all_mqm')
  full_index = wide_all.dropna(how='any').index
  if len(full_index) < _MIN_SPA_SEGMENTS:
    return None, None, None
  wide_a = seg_df.pivot_table(index=_SEGMENT_KEYS, columns='system', values='adequacy').loc[full_index]
  wide_b = seg_df.pivot_table(index=_SEGMENT_KEYS, columns='system', values='fluency').loc[full_index]
  wide_all = wide_all.loc[full_index]
  return (
      wide_all.reindex(columns=systems).values.T,
      wide_a.reindex(columns=systems).values.T,
      wide_b.reindex(columns=systems).values.T,
  )


def load_af_inputs(
    dataset: str, metametric_name: str, root: str = '.', systems: list[str] | None = None,
) -> dict[str, MetaEvalInput]:
  """scorer name ('adequacy_mqm'/'fluency_mqm') -> MetaEvalInput, mirroring
  lib.consistency.load_scorer_inputs but for the two MQM components
  themselves rather than external automatic metrics. Empty dict if
  metametric_name needs segment scores (SPA) and too few segments are
  fully covered (_af_seg_matrices)."""
  if systems is None:
    systems = real_systems(dataset, root=root)
  sys_df = load_system_scores(dataset, root=root)
  human_sys = sys_df.loc[systems, 't'].values
  a = sys_df.loc[systems, 'a'].values
  b = sys_df.loc[systems, 'b'].values

  if metametric_name in NEEDS_SEGMENT_SCORES:
    human_seg, a_seg, b_seg = _af_seg_matrices(dataset, systems, root=root)
    if human_seg is None:
      return {}
    return {
        'adequacy_mqm': MetaEvalInput(human_sys, a, human_seg, a_seg),
        'fluency_mqm': MetaEvalInput(human_sys, b, human_seg, b_seg),
    }
  return {
      'adequacy_mqm': MetaEvalInput(human_sys, a),
      'fluency_mqm': MetaEvalInput(human_sys, b),
  }


def af_metametric_scores(
    dataset: str, metametric_name: str, root: str = '.', systems: list[str] | None = None,
) -> pd.Series:
  """scorer -> meta-metric value, for 'adequacy_mqm' and 'fluency_mqm'
  (Series indexed by scorer name; empty if SPA can't be computed at all,
  per load_af_inputs)."""
  inputs = load_af_inputs(dataset, metametric_name, root=root, systems=systems)
  fn = METAMETRICS[metametric_name]
  return pd.Series({name: fn(x) for name, x in inputs.items()}, dtype=float)


def af_ratio(
    dataset: str, metametric_name: str, root: str = '.', systems: list[str] | None = None,
) -> float:
  """metametric(adequacy_mqm; dataset) / metametric(fluency_mqm; dataset).
  NaN if either side is missing/NaN or the fluency side is exactly 0 (an
  undefined ratio)."""
  scores = af_metametric_scores(dataset, metametric_name, root=root, systems=systems)
  if 'adequacy_mqm' not in scores.index or 'fluency_mqm' not in scores.index:
    return float('nan')
  num, den = scores['adequacy_mqm'], scores['fluency_mqm']
  if den != den or num != num or den == 0:
    return float('nan')
  return float(num / den)


# --- System-level leave-k-out subsets within ONE dataset --------------------
#
# compute_af_ratio_vs_alpha0.py's K-1/K-2 check operates the same way
# compute_alpha_table_loo.py does: within a single dataset with K real
# systems, drop `drop` of them at a time (every exhaustive size-(K-drop)
# subset), recompute alpha_0 and the adequacy/fluency ratio on the smaller
# system set, and treat each subset as its OWN scatter point (colored by
# which dataset it came from) -- not a subset of which DATASETS are
# considered, which is a different, coarser question.
#
# A dataset's SPA ratio needs two permutation-test-based p-value matrices
# (adequacy vs. all_mqm, fluency vs. all_mqm); a pair (i,j)'s permutation
# test only ever looks at those two systems' own segment scores (see
# lib.metametrics module docstring), so it does not depend on which other
# systems are in the pool. DatasetContext therefore computes each pairwise
# p-value matrix ONCE per dataset, at full-K, and every subset just reads
# off the submatrix for its own system positions (np.ix_) instead of
# rerunning the permutation test -- the same one-cache-many-alphas pattern
# lib.metametrics.pairwise_p_values's docstring recommends for M(alpha)
# curves, applied here across system subsets instead.


@dataclasses.dataclass
class DatasetContext:
  dataset: str
  systems: list[str]  # canonical order; all subset indices are positions into this list
  a: np.ndarray
  b: np.ndarray
  t: np.ndarray
  human_p: np.ndarray | None = None      # pairwise p-values, human_seg=all_mqm
  adequacy_p: np.ndarray | None = None   # pairwise p-values, metric_seg=adequacy
  fluency_p: np.ndarray | None = None    # pairwise p-values, metric_seg=fluency


def build_dataset_context(
    dataset: str, root: str = '.', systems: list[str] | None = None,
    num_permutations: int = 1000, seed: int = 4,
) -> DatasetContext:
  """Loads and caches everything a dataset's subset-level SPA needs exactly
  once (the expensive step: two permutation tests over the full K x K
  system pairs), so every later system-subset can be evaluated by cheap
  numpy indexing alone. `human_p`/`adequacy_p`/`fluency_p` stay None (SPA
  unavailable for this dataset) if too few segments are fully covered by
  every real system (_af_seg_matrices)."""
  if systems is None:
    systems = real_systems(dataset, root=root)
  sys_df = load_system_scores(dataset, root=root)
  a = sys_df.loc[systems, 'a'].values
  b = sys_df.loc[systems, 'b'].values
  t = sys_df.loc[systems, 't'].values

  human_seg, a_seg, b_seg = _af_seg_matrices(dataset, systems, root=root)
  if human_seg is None:
    return DatasetContext(dataset, systems, a, b, t)
  human_p = pairwise_p_values(human_seg, num_permutations, seed)
  adequacy_p = pairwise_p_values(a_seg, num_permutations, seed)
  fluency_p = pairwise_p_values(b_seg, num_permutations, seed)
  return DatasetContext(dataset, systems, a, b, t, human_p, adequacy_p, fluency_p)


def subset_af_ratio(ctx: DatasetContext, metametric_name: str, idx: np.ndarray) -> float:
  """metametric(adequacy_mqm)/metametric(fluency_mqm) restricted to the
  system positions in `idx` (indices into ctx.systems/a/b/t, ascending) --
  the per-subset counterpart of af_ratio, but reusing ctx's precomputed
  pairwise SPA p-values (submatrix lookup) instead of rerunning the
  permutation test for every subset. NaN if SPA is requested but
  unavailable for this dataset, or if either side is missing/NaN/zero."""
  human_sys = ctx.t[idx]
  if metametric_name == 'spa':
    if ctx.human_p is None:
      return float('nan')
    sub = np.ix_(idx, idx)
    num = soft_pairwise_accuracy_from_pvalues(ctx.human_p[sub], ctx.adequacy_p[sub])
    den = soft_pairwise_accuracy_from_pvalues(ctx.human_p[sub], ctx.fluency_p[sub])
  else:
    fn = METAMETRICS[metametric_name]
    num = fn(MetaEvalInput(human_sys, ctx.a[idx]))
    den = fn(MetaEvalInput(human_sys, ctx.b[idx]))
  if den != den or num != num or den == 0:
    return float('nan')
  return float(num / den)
