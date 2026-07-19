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
