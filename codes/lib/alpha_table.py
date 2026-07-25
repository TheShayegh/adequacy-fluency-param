"""Shared row-computation for the descriptive alpha table (K, var_a, var_b,
alpha_0, alpha_min, alpha_max, alpha_synth_adequacy, alpha_synth_fluency,
alpha_orig_synthAdeq_synthFlu, n_segments) -- factored out of
compute_alpha_table.py so it can be reused per (sub)set, not just per full
dataset: compute_alpha_table.py calls it once per real dataset,
compute_alpha_table_loo.py calls it once per leave-one-system-out subset of
a single dataset. Column semantics documented in compute_alpha_table.py's
module docstring; kept here rather than duplicated so the two scripts can't
silently drift apart on what a column means.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lib.alpha import alpha_0 as alpha0, alpha_min_max, variances
from lib.synthesis import synthesize_pool


def alpha_table_row(a: np.ndarray, b: np.ndarray, seg_df: pd.DataFrame) -> dict:
  """a, b: real system-level Adequacy/Fluency vectors for the (sub)set in
  question. seg_df: the matching system x segment table (mwb.mqm_scoring.
  load_segment_scores output), already restricted to exactly this
  (sub)set's systems -- synthesize_pool infers its system universe from
  seg_df's own 'system' column, so caller and seg_df must agree on which
  systems are in play."""
  K = len(a)
  var_a, var_b = variances(a, b)
  a0 = alpha0(a, b)
  amin, amax = alpha_min_max(a, b)

  a_ad, b_ad, _n_full_ad, n_segments = synthesize_pool(seg_df, rank_by='adequacy', carry='fluency')
  a_fl, b_fl, _n_full_fl, _ = synthesize_pool(seg_df, rank_by='fluency', carry='adequacy')
  alpha_synth_ad = alpha0(a_ad, b_ad)
  alpha_synth_fl = alpha0(a_fl, b_fl)

  a_pool = np.concatenate([a, a_ad, a_fl])
  b_pool = np.concatenate([b, b_ad, b_fl])
  alpha_pool = alpha0(a_pool, b_pool)

  return {
      'K': K, 'var_a': var_a, 'var_b': var_b,
      'alpha_0': a0, 'alpha_min': amin, 'alpha_max': amax,
      'alpha_synth_adequacy': alpha_synth_ad, 'alpha_synth_fluency': alpha_synth_fl,
      'alpha_orig_synthAdeq_synthFlu': alpha_pool,
      'n_segments': n_segments,
  }


def _subset_combos(K: int, k: int) -> np.ndarray:
  """(n_combos, k) int64 array of every size-k index-subset of range(K), in
  itertools.combinations order -- the shared enumeration step for every
  exhaustive-subset function in this module."""
  return np.fromiter(
      itertools.chain.from_iterable(itertools.combinations(range(K), k)),
      dtype=np.int64,
  ).reshape(-1, k)


def alpha_0_values_for_subset_size(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
  """alpha_0 (natural, uniform-weight balance -- action_plan.md 3.2,
  var_a/(var_a+var_b) at uniform weight) for EVERY C(K,k) subset of size k
  of the K systems in a/b, exhaustively enumerated (not sampled), in
  itertools.combinations order. One array entry per subset -- e.g. for a
  histogram of how alpha_0 is distributed at one fixed subset size, or as
  the building block for alpha_0_by_subset_size's per-k summary."""
  K = len(a)
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  combos = _subset_combos(K, k)
  var_a = a[combos].var(axis=1)
  var_b = b[combos].var(axis=1)
  return var_a / (var_a + var_b)


def alpha_0_values_with_drop_mask(
    a: np.ndarray, b: np.ndarray, k: int, marked_indices,
) -> tuple[np.ndarray, np.ndarray]:
  """Like alpha_0_values_for_subset_size, but also returns, per subset (same
  order), a bitmask over `marked_indices` (0-based positions into a/b --
  e.g. lib.outliers-flagged outlier systems, taken in sorted order) marking
  exactly WHICH of them are EXCLUDED from that subset: bit i set means
  sorted(marked_indices)[i] is dropped. 0 = drops none of them, 2**len(
  marked_indices)-1 = drops all of them -- the popcount of the mask is the
  drop COUNT, but the mask also distinguishes different SUBSETS at the same
  count (e.g. "drop only system A" vs "drop only system B"), needed to
  color a histogram by exactly which outliers a subset happens to drop
  rather than only how many."""
  K = len(a)
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  combos = _subset_combos(K, k)
  var_a = a[combos].var(axis=1)
  var_b = b[combos].var(axis=1)
  alphas = var_a / (var_a + var_b)

  marked = np.array(sorted(marked_indices))
  included = np.zeros((combos.shape[0], K), dtype=bool)
  np.put_along_axis(included, combos, True, axis=1)
  marked_dropped = ~included[:, marked]  # True where sorted(marked_indices)[i] is EXCLUDED
  weights = (1 << np.arange(len(marked))).astype(np.int64)
  mask = (marked_dropped.astype(np.int64) * weights).sum(axis=1)
  return alphas, mask


def alpha_0_by_subset_size(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
  """For every subset size k in [2, K-1] (K = len(a)), summarizes the
  exhaustive alpha_0_values_for_subset_size distribution as its mean and
  std. A sensitivity view of how the natural balance itself moves as the
  system count shrinks -- directly relevant to why small supports
  (Corollary 3: ESS(w) <= |S|, action_plan.md 3.8) can behave erratically:
  a small, non-representative subset can land at a very different alpha_0
  than the full K-system set.

  Exhaustive, not sampled: total work across k=2..K-1 is 2^K - K - 2
  subsets, e.g. ~131K for K=17 (the largest real dataset here) -- fast
  enough in practice not to need random subsampling.

  Returns one row per k: k, n_subsets, mean_alpha_0, std_alpha_0."""
  K = len(a)
  rows = []
  for k in range(2, K):
    alphas = alpha_0_values_for_subset_size(a, b, k)
    rows.append({
        'k': k, 'n_subsets': len(alphas),
        'mean_alpha_0': float(np.nanmean(alphas)), 'std_alpha_0': float(np.nanstd(alphas)),
    })
  return pd.DataFrame(rows)
