"""Leave-p-out (LpO) scorer-ranking stability: exhaustively drop every
size-p subset of a dataset's systems, score the remaining K-p systems as
one "rater" (natural, uniform weight, or reweighted to a shared fixed
target_beta via mwb.lib.reweight_exact.solve_w_exact), and pool the
resulting C(K,p) scorer rankings into a single Kendall's tau-a via
mwb.lib.consistency.pool_weighted_tau -- the same pooling convention this
project uses for cross-dataset scorer-ranking consistency: every pair of
raters' scorer rankings is compared, concordant/discordant counts (tau-a
convention: ties drop out of the numerator but still count in the
denominator) are summed across ALL C(n_raters_used,2) rater-pairs x
scorer-pairs, and tau is computed ONCE from those pooled totals -- not a
plain average of per-pair taus.

p=1 (leave-ONE-out) is this project's primary case, used both to measure a
dataset's raw system-selection sensitivity (target_beta=None) and to test
whether fixing the adequacy-fluency balance across LOO subsets stabilizes
scorer rankings (target_beta=<float>, the same target for every subset).
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import math
import os

import numpy as np
import pandas as pd

from mwb.lib.consistency import pool_weighted_tau, real_scorers, real_systems, scorer_scores
from mwb.lib.metametrics import NEEDS_SEGMENT_SCORES
from mwb.lib.reweight_exact import solve_w_exact
from mwb.mqm_scoring import load_system_scores


def _score_one_draw(job: tuple) -> tuple[dict | None, float | None]:
  """Top-level (module-level, so it's picklable for ProcessPoolExecutor)
  worker: scores ONE dropped-combo rater under ONE condition. job =
  (dataset, metametric, root, subset, target_beta) -- target_beta=None is
  the natural condition (ESS is exactly len(subset) by definition, no solve
  needed); a float reweights to that target via solve_w_exact first.
  Returns (scores_dict, ess), or (None, None) if target_beta was given but
  infeasible for this subset (outside its reachable range) -- the caller
  drops it."""
  dataset, metametric, root, subset, target_beta = job
  if target_beta is None:
    scores = scorer_scores(dataset, metametric, root=root, systems=subset)
    return scores.to_dict(), float(len(subset))
  df = load_system_scores(dataset, root=root)
  a_sub, f_sub = df.loc[subset, 'a'].values, df.loc[subset, 'f'].values
  try:
    r = solve_w_exact(a_sub, f_sub, target_beta)
  except ValueError:
    return None, None
  scores = scorer_scores(dataset, metametric, root=root, w=r.w, systems=subset)
  return scores.to_dict(), float(r.ess)


def _score_lpo_combos(
    dataset: str, metametric: str, p: int, root: str, exclude_outliers: bool,
    systems: list[str] | None, max_workers: int | None, target_beta: float | None,
    caller_name: str,
) -> tuple[list[str], list[str], list[tuple], dict[tuple, dict], dict[tuple, float],
           list[tuple], int]:
  """Builds the C(K,p) dropped-combo jobs, scores each once (parallel, via
  _score_one_draw) and returns everything leave_p_out_tau needs: (
  full_systems, outliers, combos, scores_by_combo, ess_by_combo,
  used_combos, n_infeasible)."""
  if systems is not None:
    full_systems = list(systems)
    outliers = []
  else:
    full_systems = real_systems(dataset, root=root, exclude_outliers=exclude_outliers)
    outliers = ([s for s in real_systems(dataset, root=root, exclude_outliers=False)
                 if s not in set(full_systems)] if exclude_outliers else [])
  K = len(full_systems)
  n_combos = math.comb(K, p) if K >= p >= 0 else 0
  if K - p < 2 or n_combos < 2:
    raise ValueError(f'{caller_name} needs K-p>=2 and C(K,p)>=2, '
                      f'got K={K}, p={p} (C(K,p)={n_combos}) for {dataset!r}')

  combos = list(itertools.combinations(full_systems, p))
  jobs = [(dataset, metametric, root, [s for s in full_systems if s not in set(combo)],
           target_beta) for combo in combos]

  scores_by_combo: dict[tuple, dict] = {}
  ess_by_combo: dict[tuple, float] = {}
  with cf.ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
    future_to_combo = {ex.submit(_score_one_draw, job): combo
                        for job, combo in zip(jobs, combos)}
    for fut in cf.as_completed(future_to_combo):
      combo = future_to_combo[fut]
      scores_dict, ess = fut.result()
      if scores_dict is not None:  # None only when target_beta was infeasible for this combo
        scores_by_combo[combo] = scores_dict
        ess_by_combo[combo] = ess
  if target_beta is None:
    assert len(scores_by_combo) == len(combos)  # natural condition is never infeasible
  used_combos = [c for c in combos if c in scores_by_combo]  # preserve combinations() order
  n_infeasible = len(combos) - len(used_combos)

  return full_systems, outliers, combos, scores_by_combo, ess_by_combo, used_combos, n_infeasible


def leave_p_out_tau(
    dataset: str,
    metametric: str,
    p: int = 1,
    root: str = '.',
    exclude_outliers: bool = True,
    systems: list[str] | None = None,
    max_workers: int | None = None,
    target_beta: float | None = None,
) -> dict:
  """Pooled Kendall's tau-a over the C(K,p) exhaustive leave-p-out (LpO)
  raters (each scored once, natural or reweighted to a shared fixed
  target_beta -- see module docstring). Objects are restricted up front to
  mwb.lib.consistency.real_scorers(dataset, systems=full_systems,
  require_seg_scores=...), the project's canonical scorer screen; unlike a
  fully-crossed design, pool_weighted_tau does NOT need every rater to
  share every scorer -- each rater-pair only needs the scorers THAT PAIR
  shares, so a scorer missing from one rater's LpO subset still contributes
  to every other pair it IS present in."""
  full_systems, outliers, combos, scores_by_combo, ess_by_combo, used_combos, n_infeasible = (
      _score_lpo_combos(dataset, metametric, p, root, exclude_outliers, systems,
                         max_workers, target_beta, 'leave_p_out_tau'))
  K = len(full_systems)

  natural_full = scorer_scores(dataset, metametric, root=root, systems=full_systems)
  canonical_scorers = set(real_scorers(
      dataset, root=root, systems=full_systems,
      require_seg_scores=(metametric in NEEDS_SEGMENT_SCORES))) & set(natural_full.index)

  rankings = {}
  for combo in used_combos:
    s = pd.Series(scores_by_combo[combo], dtype=float)
    rankings[combo] = s[s.index.isin(canonical_scorers)]
  mean_ess = float(np.mean(list(ess_by_combo.values()))) if ess_by_combo else float('nan')

  if len(used_combos) < 2:
    tau, total_weight, pair_results = float('nan'), 0, []
  else:
    tau, total_weight, pair_results = pool_weighted_tau(rankings, list(rankings.keys()))

  return {
      'dataset': dataset, 'metametric': metametric, 'K': K, 'p': p,
      'target_beta': target_beta, 'tau': tau, 'total_weight': total_weight,
      'mean_ess': mean_ess, 'n_scorers_canonical': len(canonical_scorers),
      'outliers_dropped': outliers, 'n_raters_total': len(combos),
      'n_raters_used': len(used_combos), 'n_infeasible': n_infeasible,
  }
