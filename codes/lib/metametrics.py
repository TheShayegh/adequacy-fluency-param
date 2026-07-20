"""The five meta-evaluation metrics of action_plan.md section 1.1/4: PA, SPA,
Pearson, Spearman, Kendall's tau. Each judges a scorer (an automatic MT
metric) by how well its system-level ranking/scores agree with the human
(All MQM) ranking/scores. Higher = better agreement, for all five (in
particular, mt-metrics-eval-v2's metric-score files are already oriented so
"higher = better translation" for every metric, error metrics included --
e.g. TER is stored negated -- so no per-metric sign-flipping is needed; our
own All MQM scores follow the same convention, see mqm_scoring.py).

SPA (PairwiseConfidenceError -> "1 - PCE", Thompson 2024) is a clean-room
reimplementation of the algorithm in external/mt-metrics-eval/mt_metrics_eval/
pce.py -- not imported, since that package isn't installed here (Python
version gate, see mqm_scoring.py's docstring for the same situation with
score_mqm.py). It needs segment-level scores (both human and metric), unlike
the other four which only need the system-level vectors.

WEIGHTED variants (weighted_pearson, ..., WEIGHTED_METAMETRICS) apply a
system-weight vector w (e.g. from lib.reweight_exact.solve_w_exact)
instead of treating every system as equally important -- used to trace
M(alpha)-style curves (action_plan.md 6.1) through the consistency
experiment (6.5). Each is the natural weighted generalization of its
unweighted counterpart:

  - pearson/spearman: weighted covariance/variance (lib.alpha's
    weighted_mean/weighted_var operators, action_plan.md 3.2's Sigma(w)),
    which by the section 3.3 identity is exactly equivalent to reweighting
    every *pair* (i,j)'s contribution by w_i*w_j -- so this is the same
    notion of "weighted" as everywhere else in the project, not a
    different one reused loosely. Spearman uses *weighted* ranks (a
    weighted-midrank transform) rather than plain ranks, so a
    near-zero-weight system can't anchor a rank position it barely
    occupies.
  - kendall/PA/SPA: inherently pairwise already (a sign or a p-value per
    system pair), so the natural generalization reweights the aggregation
    over pairs directly by w_i*w_j, with the same tau-a-style convention
    used in lib.consistency (ties contribute 0 to the numerator but are
    still counted in the denominator).

At uniform weight (w_i = 1/K), every weighted_* function reduces to its
unweighted counterpart (verified in the module's own sanity checks) except
for Kendall's minor tie-convention difference (tau-a vs scipy's tau-b),
immaterial here since our scores are continuous and exact ties are rare.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import scipy.stats

from lib.alpha import weighted_mean, weighted_var


@dataclasses.dataclass
class MetaEvalInput:
  """Bundles what a meta-metric needs. human_seg/metric_seg (shape (K,
  n_segments), system-major, same row order as human_sys/metric_sys) are
  only required by SPA; leave None for the other four."""
  human_sys: np.ndarray
  metric_sys: np.ndarray
  human_seg: np.ndarray | None = None
  metric_seg: np.ndarray | None = None


def pearson(x: MetaEvalInput) -> float:
  # A constant metric_sys (seen for at least one WMT23 control metric,
  # "Random-sysname") makes Pearson/Spearman undefined; scipy warns and
  # returns NaN, which callers already filter out (scorer_scores), so the
  # warning itself is silenced rather than left to print on expected data.
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', scipy.stats.ConstantInputWarning)
    return float(scipy.stats.pearsonr(x.human_sys, x.metric_sys)[0])


def spearman(x: MetaEvalInput) -> float:
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', scipy.stats.ConstantInputWarning)
    return float(scipy.stats.spearmanr(x.human_sys, x.metric_sys)[0])


def kendall(x: MetaEvalInput) -> float:
  return float(scipy.stats.kendalltau(x.human_sys, x.metric_sys, variant='b')[0])


def pairwise_accuracy(x: MetaEvalInput) -> float:
  """Fraction of system pairs where the metric agrees with the human sign,
  among pairs the human scores actually distinguish (human tie -> excluded,
  matching the standard WMT PA definition); a metric tie on a
  human-distinguished pair counts as discordant."""
  h, m = np.asarray(x.human_sys), np.asarray(x.metric_sys)
  K = len(h)
  concordant = 0
  total = 0
  for i in range(K):
    for j in range(i + 1, K):
      if h[i] == h[j]:
        continue
      total += 1
      if (h[i] - h[j]) * (m[i] - m[j]) > 0:
        concordant += 1
  return concordant / total if total else float('nan')


def pairwise_p_values(seg_scores: np.ndarray, num_permutations: int = 1000, seed: int = 4) -> np.ndarray:
  """For each system pair (i,j), the paired-permutation-test p-value for
  "system i's segment-score sum > system j's" (Thompson 2024, ported from
  pce.compute_pairwise_p_values). seg_scores: (num_systems, num_segments).
  Returns an upper-triangular (num_systems, num_systems) matrix, NaN
  elsewhere. Public (not the module's SPA internals only) because it does
  not depend on any system weighting -- callers tracing an M(alpha) curve
  should compute it once per (dataset, scorer) and reuse it across every
  alpha via soft_pairwise_accuracy_from_pvalues, rather than recomputing
  the permutation test at every point on the curve."""
  num_systems, num_segments = seg_scores.shape
  rng = np.random.default_rng(seed)
  signs = rng.integers(0, 2, size=(num_permutations, num_segments)).astype(np.float64) * 2 - 1

  seg_scores = seg_scores.astype(np.float64)
  sys_scores = seg_scores.sum(axis=1)
  # macOS/Accelerate's BLAS backend emits spurious "divide by zero"/
  # "overflow"/"invalid value" RuntimeWarnings on float64 @ for some shapes
  # regardless of the actual values (reproduces on a plain random matmul of
  # this shape with no extreme values at all) -- numpy 2.0.2 on arm64 here.
  # Not a real numeric fault; silenced locally rather than filtered globally.
  with np.errstate(all='ignore'):
    partial = signs @ seg_scores.T  # (num_permutations, num_systems)

  p_vals = np.full((num_systems, num_systems), np.nan)
  for i in range(num_systems):
    for j in range(i + 1, num_systems):
      null_delta = partial[:, i] - partial[:, j]
      test_delta = sys_scores[i] - sys_scores[j]
      p_vals[i, j] = np.mean(null_delta >= test_delta)
  return p_vals


def soft_pairwise_accuracy_from_pvalues(
    human_p: np.ndarray, metric_p: np.ndarray, w: np.ndarray | None = None,
) -> float:
  """1 - (weighted) Pairwise Confidence Error, given precomputed pairwise
  p-value matrices (pairwise_p_values). Unweighted: 1 minus the mean
  |p_human - p_metric| over all system pairs. Weighted: same, but each
  pair (i,j) is weighted by w_i*w_j -- the same pairwise-reweighting
  convention as weighted_kendall/weighted_pairwise_accuracy, and the
  natural way to fold in a system weighting since the two pairwise
  p-values themselves don't depend on any other system's weight."""
  K = human_p.shape[0]
  iu = np.triu_indices(K, 1)
  diffs = np.abs(human_p[iu] - metric_p[iu])
  if w is None:
    return float(1.0 - np.nanmean(diffs))
  wij = np.asarray(w)[iu[0]] * np.asarray(w)[iu[1]]
  valid = ~np.isnan(diffs)
  if not valid.any() or wij[valid].sum() == 0:
    return float('nan')
  return float(1.0 - np.average(diffs[valid], weights=wij[valid]))


def soft_pairwise_accuracy(x: MetaEvalInput, num_permutations: int = 1000, seed: int = 4) -> float:
  """1 - Pairwise Confidence Error: for every system pair, compare the
  human-judged and metric-judged p-value for "system i is better", and
  average |difference| over all pairs; report 1 minus that (so higher is
  still better, matching the other four). Requires segment-level scores for
  both human and metric, position-aligned per system (action_plan.md
  section 1.1's note on Freitag et al. 2024/Kocmi et al.)."""
  if x.human_seg is None or x.metric_seg is None:
    raise ValueError('soft_pairwise_accuracy needs segment-level scores')
  human_p = pairwise_p_values(x.human_seg, num_permutations, seed)
  metric_p = pairwise_p_values(x.metric_seg, num_permutations, seed)
  return soft_pairwise_accuracy_from_pvalues(human_p, metric_p)


def weighted_soft_pairwise_accuracy(
    x: MetaEvalInput, w: np.ndarray, num_permutations: int = 1000, seed: int = 4,
) -> float:
  """Weighted counterpart of soft_pairwise_accuracy: same per-pair p-values
  (unaffected by w -- each pair's permutation test only looks at those two
  systems' segment scores), aggregated with pair-weight w_i*w_j instead of
  a plain mean. For repeated calls across an alpha grid at fixed (dataset,
  scorer), prefer computing pairwise_p_values once and calling
  soft_pairwise_accuracy_from_pvalues(..., w) directly -- this wrapper
  recomputes the permutation test every call, matching soft_pairwise_
  accuracy's (unweighted) one-shot convenience."""
  if x.human_seg is None or x.metric_seg is None:
    raise ValueError('weighted_soft_pairwise_accuracy needs segment-level scores')
  human_p = pairwise_p_values(x.human_seg, num_permutations, seed)
  metric_p = pairwise_p_values(x.metric_seg, num_permutations, seed)
  return soft_pairwise_accuracy_from_pvalues(human_p, metric_p, w)


METAMETRICS = {
    'pearson': pearson,
    'spearman': spearman,
    'kendall': kendall,
    'pa': pairwise_accuracy,
    'spa': soft_pairwise_accuracy,
}

# Canonical presentation/iteration order (plotting, curve computation) --
# distinct from METAMETRICS' dict (insertion) order only in that this is
# the one every caller should use rather than each redefining its own copy.
METAMETRICS_ORDER = ['pearson', 'spearman', 'kendall', 'pa', 'spa']

# Which meta-metrics need segment-level (not just system-level) score data.
NEEDS_SEGMENT_SCORES = {'spa'}


# --- Weighted variants (module docstring explains the convention) ----------

def weighted_pearson(x: MetaEvalInput, w: np.ndarray) -> float:
  h, m, w = np.asarray(x.human_sys), np.asarray(x.metric_sys), np.asarray(w)
  mu_h, mu_m = weighted_mean(h, w), weighted_mean(m, w)
  var_h, var_m = weighted_var(h, w), weighted_var(m, w)
  cov = float(np.dot(w, h * m) - mu_h * mu_m)
  denom = np.sqrt(var_h * var_m)
  return cov / denom if denom > 0 else float('nan')


def _weighted_midranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
  """Weighted analogue of scipy.stats.rankdata's average-tie convention:
  rank_i = (weight strictly below x_i) + half the weight tied with x_i.
  Reduces to ordinary average ranks at uniform weight."""
  x, w = np.asarray(x), np.asarray(w)
  ranks = np.empty(len(x))
  for i in range(len(x)):
    below = w[x < x[i]].sum()
    tied = w[x == x[i]].sum()
    ranks[i] = below + 0.5 * tied
  return ranks


def weighted_spearman(x: MetaEvalInput, w: np.ndarray) -> float:
  w = np.asarray(w)
  rh = _weighted_midranks(np.asarray(x.human_sys), w)
  rm = _weighted_midranks(np.asarray(x.metric_sys), w)
  return weighted_pearson(MetaEvalInput(rh, rm), w)


def weighted_kendall(x: MetaEvalInput, w: np.ndarray) -> float:
  """Pair-weighted Kendall's tau-a: sum_{i<j} w_i w_j * sign(h_i-h_j) *
  sign(m_i-m_j), normalized by sum_{i<j} w_i w_j; a tie in either vector
  contributes 0 to the numerator but the pair still counts in the
  denominator (see lib.consistency's identical convention, chosen there so
  the scorer-pair-weighted mean equals a single pooled tau)."""
  h, m, w = np.asarray(x.human_sys), np.asarray(x.metric_sys), np.asarray(w)
  K = len(h)
  num = den = 0.0
  for i in range(K):
    for j in range(i + 1, K):
      wij = w[i] * w[j]
      den += wij
      sh, sm = np.sign(h[i] - h[j]), np.sign(m[i] - m[j])
      if sh != 0 and sm != 0:
        num += wij if sh == sm else -wij
  return num / den if den > 0 else float('nan')


def weighted_pairwise_accuracy(x: MetaEvalInput, w: np.ndarray) -> float:
  """Pair-weighted PA: same human-tie exclusion and metric-tie-counts-as-
  discordant convention as pairwise_accuracy, aggregated by w_i*w_j."""
  h, m, w = np.asarray(x.human_sys), np.asarray(x.metric_sys), np.asarray(w)
  K = len(h)
  num = den = 0.0
  for i in range(K):
    for j in range(i + 1, K):
      if h[i] == h[j]:
        continue
      wij = w[i] * w[j]
      den += wij
      if (h[i] - h[j]) * (m[i] - m[j]) > 0:
        num += wij
  return num / den if den > 0 else float('nan')


WEIGHTED_METAMETRICS = {
    'pearson': weighted_pearson,
    'spearman': weighted_spearman,
    'kendall': weighted_kendall,
    'pa': weighted_pairwise_accuracy,
    'spa': weighted_soft_pairwise_accuracy,
}
