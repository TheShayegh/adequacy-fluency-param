"""Synthesized pseudo-system pool construction (action_plan.md section 2.1,
Shayegh et al. 2025's method): per fully-covered segment, rank real systems
by one component and assign the k-th best's value to synthetic system k,
carrying the other component along under that same ranking.

Kept separate from mwb.mqm_scoring (which only knows how to parse/aggregate
raw ratings into a system x segment table) and from lib.alpha (component-
agnostic balance math): this module is the one place that encodes Shayegh
et al.'s specific synthesis rule, used both as a critique target (section
2.1) and, later, as a range-extender (sections 6.2, 6.6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthesize_pool(
    seg_df: pd.DataFrame,
    rank_by: str,
    carry: str,
    segment_keys: list[str] = ('doc', 'doc_id', 'seg_id'),
) -> tuple[np.ndarray, np.ndarray, int, int]:
  """Builds the synthesized-by-`rank_by` pool from a system x segment table
  (as returned by mwb.mqm_scoring.load_segment_scores /
  per_segment_scores): columns must include 'system', the segment_keys, and
  the two score columns rank_by/carry.

  Per fully-covered segment (rated by all K systems present in seg_df),
  sorts systems by `rank_by` descending (best/least-negative MQM first) and
  assigns the k-th best's `rank_by` value to synthetic system k, carrying
  along the *same system's* `carry` value at that rank. Segments not rated
  by all K systems are dropped (a partial ranking can't fill all K slots).

  Returns (pool_adequacy, pool_fluency, n_full_segments, n_segments_total)
  -- the two arrays always in (adequacy, fluency) order regardless of which
  was rank_by, each of length K (one synthetic system per rank).
  """
  segment_keys = list(segment_keys)
  wide_rank = seg_df.pivot_table(index=segment_keys, columns='system', values=rank_by)
  wide_carry = seg_df.pivot_table(index=segment_keys, columns='system', values=carry)
  n_total = len(wide_rank)
  full = wide_rank.dropna(how='any')
  wide_carry = wide_carry.loc[full.index]

  order = np.argsort(-full.values, axis=1)  # descending: best (least-negative) first
  ranked_rank = np.take_along_axis(full.values, order, axis=1)
  ranked_carry = np.take_along_axis(wide_carry.values, order, axis=1)
  pool_rank = ranked_rank.mean(axis=0)
  pool_carry = ranked_carry.mean(axis=0)

  n_full = len(full)
  if rank_by == 'adequacy':
    return pool_rank, pool_carry, n_full, n_total
  return pool_carry, pool_rank, n_full, n_total
