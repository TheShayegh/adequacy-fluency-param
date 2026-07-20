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
but reused, as-is, by every subset-consistency script in this family (LOO,
bootstrap, exclude-system) -- moved here since they're genuinely shared
infrastructure, not specific to that one experiment.
"""

from __future__ import annotations

import dataclasses
import itertools

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
    systems_by_dataset: dict[str, list[str]] | None = None,
) -> list[CurvePoint]:
  """The M(alpha) x consistency curve for one meta-metric: at each alpha,
  reweights every dataset's real systems to that target (via w_cache, or
  solving fresh if not supplied), scores every scorer under that
  weighting, and pools the weighted mean Kendall's tau across dataset
  pairs exactly as lib.consistency.weighted_consistency does for the
  natural baseline.

  w_cache/spa_cache: pass the outputs of solve_w_for_datasets /
  spa_pvalue_cache (per dataset) to reuse across multiple metametric
  calls instead of recomputing the (expensive) solver / permutation tests
  for each of the 5 meta-metrics separately. Built automatically if
  omitted (costs more if called once per metametric).

  systems_by_dataset: restrict dataset d's systems to systems_by_dataset[d]
  instead of real_systems(d) -- see solve_w_for_datasets. Only consulted
  when w_cache/spa_cache/inputs aren't already supplied (systems only
  matter at the point of solving/scoring, not once results are cached).
  """
  if w_cache is None:
    w_cache = solve_w_for_datasets(datasets, alphas, root=root, systems_by_dataset=systems_by_dataset)
  needs_seg = metametric_name in NEEDS_SEGMENT_SCORES
  if needs_seg:
    if spa_cache is None:
      spa_cache = {d: spa_pvalue_cache(d, (systems_by_dataset or {}).get(d), root=root) for d in datasets}
  else:
    # Load each dataset's (weighting-independent) scorer inputs once,
    # outside the alpha loop -- avoids re-reading every score file off
    # disk at every grid point (the dominant cost before this caching).
    inputs_cache = {d: load_scorer_inputs(d, metametric_name, root=root,
                                            systems=(systems_by_dataset or {}).get(d))
                     for d in datasets}

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
  inner loop, just returning the raw outcome list instead of a summary."""
  common = sorted(set(ranking_i.index) & set(ranking_j.index))
  outcomes = []
  for x, y in itertools.combinations(common, 2):
    si = np.sign(ranking_i[x] - ranking_i[y])
    sj = np.sign(ranking_j[x] - ranking_j[y])
    outcomes.append(1 if (si != 0 and sj != 0 and si == sj)
                     else (-1 if (si != 0 and sj != 0) else 0))
  return outcomes


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


def star_pooled_consistency_curve(
    metametric_name: str,
    datasets: list[str],
    alphas,
    w_cache: dict,
    root: str = '.',
    spa_cache: dict | None = None,
    inputs_cache: dict | None = None,
    systems_by_dataset: dict[str, list[str]] | None = None,
) -> dict:
  """Per-dataset star-pooled ("d vs the rest") curve: for each dataset d,
  pools tau-a between d's ranking and every OTHER dataset's at each alpha,
  plus d's own natural (unweighted) star-pooled baseline against the same
  others. Returns {'tau_bar_by_dataset', 'ess_by_dataset',
  'baseline_tau_by_dataset'}, each keyed by dataset.

  w_cache is required (every caller of this already has one on hand, unlike
  weighted_consistency_curve which can solve fresh); spa_cache/inputs_cache
  reuse spa_pvalue_cache/load_scorer_inputs-style caches, built
  automatically if omitted. systems_by_dataset: see solve_w_for_datasets --
  only valid when every `d` in `datasets` is itself directly resolvable via
  load_system_scores(d) (a real dataset name, optionally with a systems
  subset of ITS OWN roster) -- not for synthetic subset ids of some other
  base dataset (e.g. LOO/bootstrap repeat ids), which need the caller's own
  base-dataset + systems-override plumbing instead.
  """
  needs_seg = metametric_name in NEEDS_SEGMENT_SCORES
  if needs_seg:
    if spa_cache is None:
      spa_cache = {d: spa_pvalue_cache(d, (systems_by_dataset or {}).get(d), root=root) for d in datasets}
  elif inputs_cache is None:
    inputs_cache = {d: load_scorer_inputs(d, metametric_name, root=root,
                                            systems=(systems_by_dataset or {}).get(d))
                     for d in datasets}
  rankings_by_alpha = {a: rankings_at(metametric_name, datasets, a, w_cache, inputs_cache, spa_cache)
                        for a in alphas}
  natural_rankings = {d: scorer_scores(d, metametric_name, root=root, systems=(systems_by_dataset or {}).get(d))
                       for d in datasets}

  tau_bar_by_dataset = {}
  ess_by_dataset = {}
  baseline_tau_by_dataset = {}
  for d in datasets:
    others = [o for o in datasets if o != d]
    tau_bar_by_dataset[d] = [pooled([(d, o) for o in others], rankings_by_alpha[a], rankings_by_alpha[a])[0]
                              for a in alphas]
    ess_by_dataset[d] = [w_cache[(d, a)].ess for a in alphas]
    baseline_tau_by_dataset[d] = pooled([(d, o) for o in others], natural_rankings, natural_rankings)[0]

  return {
      'tau_bar_by_dataset': tau_bar_by_dataset,
      'ess_by_dataset': ess_by_dataset,
      'baseline_tau_by_dataset': baseline_tau_by_dataset,
  }
