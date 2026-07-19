"""M(alpha)-style curves (action_plan.md section 6.1) through the
cross-dataset consistency metric (section 6.5): re-run lib.consistency's
pooled weighted-Kendall-tau consistency measure, but with every dataset's
systems reweighted to a common target alpha via the NUMERIC solver
(lib.reweight_numeric.solve_w_numeric) instead of their natural (uniform)
weighting -- so this module's name says "reweighted", not "numeric": the
numeric-vs-exact distinction lives one level down, inside solve_w_numeric,
and this module is agnostic to which solver produced its w(alpha) (pass a
different solve function via `solver` to cross-validate later against the
exact/support-enumeration solver once it exists).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from lib.alpha import alpha_min_max
from lib.consistency import real_systems, load_scorer_inputs, evaluate_scorer_scores, pool_weighted_tau
from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues, NEEDS_SEGMENT_SCORES
from lib.metric_scores import (
    discover_metrics, load_human_seg_scores, load_metric_seg_scores, jointly_valid_columns,
)
from lib.reweight_numeric import solve_w_numeric
from mwb.mqm_scoring import load_system_scores

_MIN_SPA_SEGMENTS = 10  # matches lib.consistency's floor


def common_alpha_range(datasets: list[str], root: str = '.') -> tuple[float, float]:
  """[max_d alpha_min(d), min_d alpha_max(d)] -- the alpha range every
  dataset can be reweighted into *simultaneously* (each dataset's own
  Corollary 2 range intersected across all of them)."""
  los, his = [], []
  for d in datasets:
    systems = real_systems(d, root=root)
    sys_df = load_system_scores(d, root=root).loc[systems]
    lo, hi = alpha_min_max(sys_df['a'].values, sys_df['b'].values)
    los.append(lo)
    his.append(hi)
  return max(los), min(his)


def solve_w_for_datasets(
    datasets: list[str], alphas, root: str = '.', solver=solve_w_numeric, **solver_kwargs,
) -> dict[tuple[str, float], object]:
  """Solves w_d(alpha) once for every (dataset, alpha) pair -- shared
  across all 5 meta-metrics' curves, since the weighting itself doesn't
  depend on which meta-metric will later use it. Returns {(dataset, alpha):
  NumericWResult}."""
  cache = {}
  for d in datasets:
    systems = real_systems(d, root=root)
    sys_df = load_system_scores(d, root=root).loc[systems]
    a, b = sys_df['a'].values, sys_df['b'].values
    for alpha in alphas:
      cache[(d, alpha)] = solver(a, b, alpha, **solver_kwargs)
  return cache


def _spa_pvalue_cache(dataset: str, root: str = '.') -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """scorer name -> (human_p, metric_p) pairwise p-value matrices for this
  dataset, computed once (they don't depend on any system weighting) so
  the alpha loop only pays for the cheap weighted-average step."""
  systems = real_systems(dataset, root=root)
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None:
    return {}
  out = {}
  for name in discover_metrics(dataset, root):
    metric_seg = load_metric_seg_scores(dataset, name, systems, root=root)
    if metric_seg is None:
      continue
    mask = jointly_valid_columns(human_seg, metric_seg)
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = (pairwise_p_values(human_seg[:, mask]), pairwise_p_values(metric_seg[:, mask]))
  return out


@dataclasses.dataclass
class CurvePoint:
  alpha: float
  tau_bar: float
  mean_ess: float
  ess_by_dataset: dict[str, float]


def weighted_consistency_curve(
    metametric_name: str,
    datasets: list[str],
    alphas,
    root: str = '.',
    w_cache: dict | None = None,
    spa_cache: dict | None = None,
) -> list[CurvePoint]:
  """The M(alpha) x consistency curve for one meta-metric: at each alpha,
  reweights every dataset's real systems to that target (via w_cache, or
  solving fresh if not supplied), scores every scorer under that
  weighting, and pools the weighted mean Kendall's tau across dataset
  pairs exactly as lib.consistency.weighted_consistency does for the
  natural baseline.

  w_cache/spa_cache: pass the outputs of solve_w_for_datasets /
  _spa_pvalue_cache (per dataset) to reuse across multiple metametric
  calls instead of recomputing the (expensive) solver / permutation tests
  for each of the 5 meta-metrics separately. Built automatically if
  omitted (costs more if called once per metametric).
  """
  if w_cache is None:
    w_cache = solve_w_for_datasets(datasets, alphas, root=root)
  needs_seg = metametric_name in NEEDS_SEGMENT_SCORES
  if needs_seg:
    if spa_cache is None:
      spa_cache = {d: _spa_pvalue_cache(d, root=root) for d in datasets}
  else:
    # Load each dataset's (weighting-independent) scorer inputs once,
    # outside the alpha loop -- avoids re-reading every score file off
    # disk at every grid point (the dominant cost before this caching).
    inputs_cache = {d: load_scorer_inputs(d, metametric_name, root=root) for d in datasets}

  points = []
  for alpha in alphas:
    ess_by_dataset = {d: w_cache[(d, alpha)].ess for d in datasets}
    if needs_seg:
      rankings = {}
      for d in datasets:
        w = w_cache[(d, alpha)].w
        values = {}
        for name, (hp, mp) in spa_cache[d].items():
          val = soft_pairwise_accuracy_from_pvalues(hp, mp, w)
          if val == val:
            values[name] = val
        rankings[d] = pd.Series(values, dtype=float)
    else:
      rankings = {d: evaluate_scorer_scores(inputs_cache[d], metametric_name, w_cache[(d, alpha)].w)
                  for d in datasets}
    tau_bar, _, _ = pool_weighted_tau(rankings, datasets)
    points.append(CurvePoint(
        alpha=alpha, tau_bar=tau_bar,
        mean_ess=float(np.mean(list(ess_by_dataset.values()))),
        ess_by_dataset=ess_by_dataset,
    ))
  return points
