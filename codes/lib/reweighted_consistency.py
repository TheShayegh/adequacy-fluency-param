"""M(alpha)-style curves (action_plan.md section 6.1) through the
cross-dataset consistency metric (section 6.5): re-run lib.consistency's
pooled weighted-Kendall-tau consistency measure, but with every dataset's
systems reweighted to a common target alpha via the EXACT solver
(lib.reweight_exact.solve_w_exact) instead of their natural (uniform)
weighting -- so this module's name says "reweighted", not "exact": which
solver produced w(alpha) lives one level down, and this module is agnostic
to it (pass a different solve function via `solver` if needed).

This used to default to lib.reweight_numeric.solve_w_numeric (a multi-start
local optimizer); that module was retired after cross-validation against
lib.reweight_exhaustive showed it missing the true optimum in 13/15 tested
cases at its default restart count, sometimes by a large margin (e.g. ESS
1.07 vs the achievable 4.0) -- lib.reweight_exact has no such risk (a
certified global optimum every time) and no restart/seed parameters to
tune, so `solver_kwargs` downstream of this function no longer needs to
carry n_restarts/seed through to the solver.

Also home to the shared pooling machinery (rankings_at/pairwise_outcomes/
pooled) originally written for cross_regime_sign_test.py's ad hoc analysis
-- rankings_at/pooled are still used there and by mid_alpha_group_split_
test.py; pairwise_outcomes is also used directly by the type7 removal-
robustness family (compute_type7_sensitivity.py and friends) and by
plot_delta_alpha0_within_dataset.py, without needing rankings_at/pooled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.alpha import alpha_min_max, alpha_0
from lib.consistency import (
    real_systems, load_scorer_inputs, evaluate_scorer_scores, scorer_scores, pool_weighted_tau,
)
from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues, NEEDS_SEGMENT_SCORES
from lib.metric_scores import (
    discover_metrics, load_human_seg_scores, load_metric_seg_scores, jointly_valid_columns,
)
from lib.reweight_exact import solve_w_exact
from mwb.mqm_scoring import load_system_scores

_MIN_SPA_SEGMENTS = 10  # matches lib.consistency's floor


def dataset_score_arrays(
    datasets: list[str], root: str = '.', systems_by_dataset: dict[str, list[str]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """(a, b) system-score arrays per dataset, restricted to
  systems_by_dataset[d] if given (default: real_systems(d)) -- the one I/O
  step alpha_0/alpha_min_max/grid-building all need."""
  out = {}
  for d in datasets:
    systems = (systems_by_dataset or {}).get(d) or real_systems(d, root=root)
    df = load_system_scores(d, root=root).loc[systems]
    out[d] = (df['a'].values, df['b'].values)
  return out


def dataset_alpha_0s(
    datasets: list[str], root: str = '.', systems_by_dataset: dict[str, list[str]] | None = None,
) -> dict[str, float]:
  """Natural (uniform-weight) alpha_0 per dataset, built on
  dataset_score_arrays."""
  return {d: alpha_0(a, b) for d, (a, b) in dataset_score_arrays(datasets, root, systems_by_dataset).items()}


def common_alpha_range(
    datasets: list[str], root: str = '.', systems_by_dataset: dict[str, list[str]] | None = None,
) -> tuple[float, float]:
  """[max_d alpha_min(d), min_d alpha_max(d)] -- the alpha range every
  dataset can be reweighted into *simultaneously* (each dataset's own
  Corollary 2 range intersected across all of them)."""
  los, his = [], []
  for a, b in dataset_score_arrays(datasets, root, systems_by_dataset).values():
    lo, hi = alpha_min_max(a, b)
    los.append(lo)
    his.append(hi)
  return max(los), min(his)


def solve_w_for_datasets(
    datasets: list[str], alphas, root: str = '.', solver=solve_w_exact,
    systems_by_dataset: dict[str, list[str]] | None = None, **solver_kwargs,
) -> dict[tuple[str, float], object]:
  """Solves w_d(alpha) once for every (dataset, alpha) pair -- shared
  across all 5 meta-metrics' curves, since the weighting itself doesn't
  depend on which meta-metric will later use it. Returns {(dataset, alpha):
  ExactWResult}.

  systems_by_dataset: restrict dataset d's systems to systems_by_dataset[d]
  instead of real_systems(d) -- e.g. a LOO/bootstrap/exclude-system subset
  of one base dataset, id'd by a synthetic `d` that isn't itself a real
  WMT dataset key (see dataset_score_arrays)."""
  cache = {}
  for d, (a, b) in dataset_score_arrays(datasets, root, systems_by_dataset).items():
    for alpha in alphas:
      cache[(d, alpha)] = solver(a, b, alpha, **solver_kwargs)
  return cache


def spa_pvalue_cache(
    dataset: str, systems: list[str] | None = None, root: str = '.',
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """scorer name -> (human_p, metric_p) pairwise p-value matrices for this
  dataset (or an explicit systems subset, default: real_systems(dataset)),
  computed once (they don't depend on any system weighting) so the alpha
  loop only pays for the cheap weighted-average step."""
  if systems is None:
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


def rankings_at(metametric, datasets, alpha, w_cache, inputs_cache, spa_cache):
  """Every dataset's scorer-ranking (pd.Series) at a single alpha, under
  w_cache[(d, alpha)].w -- the shared per-alpha step both the type-1
  (pooled) and type-2 (star-pooled) curve computations build on."""
  if metametric in NEEDS_SEGMENT_SCORES:
    rankings = {}
    for d in datasets:
      w = w_cache[(d, alpha)].w
      values = {}
      for name, (hp, mp) in spa_cache[d].items():
        val = soft_pairwise_accuracy_from_pvalues(hp, mp, w)
        if val == val:
          values[name] = val
      rankings[d] = pd.Series(values, dtype=float)
    return rankings
  return {d: evaluate_scorer_scores(inputs_cache[d], metametric, w_cache[(d, alpha)].w) for d in datasets}


def pairwise_outcomes(ranking_i: pd.Series, ranking_j: pd.Series) -> list[int]:
  """+1 (concordant) / -1 (discordant) / 0 (tie) per shared scorer pair --
  the same tau-a-convention counting as lib.consistency.pool_weighted_tau's
  inner loop, just returning the raw outcome list instead of a summary.

  Vectorized the same way as that function: reindex once to plain numpy
  arrays in `common`'s order, then compare every pair at once via
  triu_indices broadcasting instead of a per-pair pandas Series.__getitem__
  loop (which profiled at 83% pandas-indexing overhead, not the actual
  comparison, for pool_weighted_tau's identical pattern)."""
  common = sorted(set(ranking_i.index) & set(ranking_j.index))
  n = len(common)
  vi = ranking_i.reindex(common).values
  vj = ranking_j.reindex(common).values
  iu, ju = np.triu_indices(n, 1)
  si = np.sign(vi[iu] - vi[ju])
  sj = np.sign(vj[iu] - vj[ju])
  untied = (si != 0) & (sj != 0)
  outcomes = np.where(untied, np.where(si == sj, 1, -1), 0)
  return outcomes.tolist()


def pooled(pairs, left, right):
  """pairs: iterable of (key_left, key_right). left/right: dataset -> ranking
  lookup dicts (may be the same dict for a same-regime/single-pool case, or
  two different ones -- e.g. lo-side, hi-side -- for a cross-regime pool).
  Returns (tau_bar, flat_outcomes, per_pair_detail)."""
  all_outcomes = []
  per_pair = []
  for y, yp in pairs:
    oc = pairwise_outcomes(left[y], right[yp])
    all_outcomes.extend(oc)
    per_pair.append((y, yp, len(oc), float(np.mean(oc)) if oc else float('nan')))
  tau_bar = float(np.mean(all_outcomes)) if all_outcomes else float('nan')
  return tau_bar, all_outcomes, per_pair
