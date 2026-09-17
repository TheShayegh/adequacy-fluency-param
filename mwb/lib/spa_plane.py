"""SPA Plane (Shayegh et al. 2025's construction, reused by our paper's
Figure 2): for a dataset, a 2-D plot where every scorer (external automatic
MT scorer, e.g. BLEU/Comet/MetricX) is one point at

    x = SPA(scorer; Fluency MQM)     y = SPA(scorer; Adequacy MQM)

i.e. mwb.lib.metametrics.soft_pairwise_accuracy computed TWICE per scorer, once
treating Fluency MQM's segment scores as the "human" ground truth and once
treating Adequacy MQM's as the "human" ground truth (rather than the usual
All MQM). Augmented with three sentinel curves:

  - tradeoff_line: score_seg(lam) = lam*a_seg + (1-lam)*f_seg, lam in [0,1]
    -- the segment-level linear interpolation of Adequacy and Fluency MQM.
    lam=1 -> the Adequacy MQM vertex (y=1 exactly); lam=0 -> the Fluency MQM
    vertex (x=1 exactly). "No system can surpass" this curve.
  - adequacy_knowledge_line: score_seg(lam) = lam*a_seg + (1-lam)*r_seg,
    r_seg segment-and-system-wise uniform noise. Traces what's reachable by
    mixing Adequacy MQM with pure noise -- any fluency-axis position it
    reaches is due only to the correlation between the two aspects.
  - fluency_knowledge_line: the mirror, mixing Fluency MQM with noise.

Both knowledge lines are paper-specified as an average of 10 curves, each
redrawing r_seg from a fresh RNG instance (see knowledge_lines' `shadows`
returns) -- computed TOGETHER by knowledge_lines, not as two independent
sweeps, and sharing the SAME r_seg (one draw per instance k, uniform over
the per-segment range that COVERS BOTH a_seg and f_seg -- see
knowledge_lines) rather than each line rescaling its own independent draw
to its own aspect's range. This is what makes the two lines a genuine
CONNECTED pair at each instance k: at lam=0 both curves evaluate the exact
same r_seg, so shadow k of the adequacy-knowledge line and shadow k of the
fluency-knowledge line meet at the exact same (x, y) point -- the shared
noise-only vertex both curves fan out from -- rather than two
independently-scaled endpoints that merely happened to be seeded together.

Two distinct notions of "segment" are load-bearing here, kept in separate
functions:

  - build_af_seg_matrices: the sentinel lines only ever compare Adequacy MQM
    against Fluency MQM against synthetic mixtures of the two -- no external
    scorer is involved -- so they can use mwb.mqm_scoring's own
    (doc, doc_id, seg_id)-keyed segment index directly.
  - positional_af_matrices: a scorer's own segment scores
    (mwb.lib.scorer_score_files.load_scorer_seg_scores) are indexed by LINE POSITION
    in the dataset's official WMT source file, a different index space than
    mqm_scoring's (doc, doc_id, seg_id) keys for most datasets. Only for
    the sets where mwb.mqm_scoring.OFFICIAL_RATINGS applies (or, for the
    raw-TSV sets, wherever the file's own seg_id/globalSegId column
    happens to already BE that line position -- true for every set tested
    here except enru22, ende20, zhen20) does int(seg_id) recover the
    correct column of a scorer's (K, n_positions) matrix. Rather than
    hardcode that list, positional_af_matrices self-validates per dataset:
    it reconstructs the segment dataframe's 'full' column (the unweighted
    sum of EVERY error -- mwb.mqm_scoring's error_weight()/official-rating
    total, what actually reproduces the official score files exactly;
    'all_mqm' = adequacy+fluency is a strict subset that excludes errors
    classify_aspect() maps to neither aspect, so validating against it
    instead under-corrs even on perfectly-aligned datasets -- found
    empirically on ende23: 0.993 against 'all_mqm' vs. exact 1.0 against
    'full') at each int(seg_id) position, and requires near-exact
    correlation against mwb.lib.scorer_score_files.load_human_seg_scores's OFFICIAL
    positionally-indexed all_mqm at the same positions (a valid alignment
    should match almost exactly; a broken one is uncorrelated). Datasets
    that fail this check contribute sentinel lines but no scorer points
    (scorer_spa_points returns {}).
"""

from __future__ import annotations

import numpy as np

from mwb.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores, load_scorer_sys_scores
from mwb.mqm_scoring import load_segment_scores

# Same floor mwb.lib.consistency uses before trusting SPA's permutation
# p-values (segment counts below this make them noisy).
_MIN_SPA_SEGMENTS = 10

# Minimum jointly-valid (system, position) cells required before trusting
# positional_af_matrices' alignment-validation correlation at all -- below
# this, a spuriously high/low correlation is itself just noise.
_MIN_VALIDATION_CELLS = 50
# 0.99, not something closer to 1: even genuinely-aligned OFFICIAL_RATINGS
# datasets don't always hit an exact match, since mwb.mqm_scoring
# deliberately excludes each set's *.roundN.mqm.merged.seg.rating files (see
# its OFFICIAL_RATINGS comment) -- a targeted QC re-annotation on a small
# subset of segments/systems that would make the match WORSE if included.
# Empirically this leaves correlations of 1.0 (ende23, no round files ever
# existed) down to ~0.996 (zhen23: 269/17655 positions differ, all from one
# system's QC-revisited segment run) for datasets that ARE positionally
# aligned -- vs. 0.35 for enru22, whose raw-TSV seg_id resets per document
# and is not a global position at all. 0.99 cleanly separates the two.
_VALIDATION_CORR_FLOOR = 0.99

_SEGMENT_KEYS = ['doc', 'doc_id', 'seg_id']

LAMBDA_GRID = tuple(np.linspace(0.0, 1.0, 21))
N_RANDOM_INSTANCES = 10
DEFAULT_NUM_PERMUTATIONS = 1000
DEFAULT_SEED = 4          # the permutation test's own RNG -- fixed across
                           # every point on a line/scorer so points differ
                           # only in the scores being compared, not in
                           # which null permutations were drawn.
DEFAULT_NOISE_SEED = 1234 # base seed for the 10 random-noise instances.


# --- (doc, doc_id, seg_id)-keyed matrices, for the sentinel lines ----------

def build_af_seg_matrices(dataset: str, systems: list[str], root: str = '.'):
  """(a_seg, f_seg), each (K, n_common_segments) row-ordered as `systems`,
  restricted to segments every one of `systems` has both an adequacy and a
  fluency score for. (None, None) if fewer than _MIN_SPA_SEGMENTS such
  segments exist."""
  seg_df = load_segment_scores(dataset, root=root)
  seg_df = seg_df[seg_df['system'].isin(systems)]
  wide_a = seg_df.pivot_table(index=_SEGMENT_KEYS, columns='system', values='adequacy')
  wide_f = seg_df.pivot_table(index=_SEGMENT_KEYS, columns='system', values='fluency')
  valid = ~(wide_a.isna().any(axis=1) | wide_f.isna().any(axis=1))
  wide_a, wide_f = wide_a.loc[valid], wide_f.loc[valid]
  if len(wide_a) < _MIN_SPA_SEGMENTS:
    return None, None
  return wide_a.reindex(columns=systems).values.T, wide_f.reindex(columns=systems).values.T


# --- Line-position-keyed matrices, for scorer points ------------------------

def positional_af_matrices(dataset: str, systems: list[str], root: str = '.'):
  """(a_pos, f_pos), each (K, n_positions) row-ordered as `systems` and
  column-indexed by line position in the dataset's official WMT source file
  (module docstring) -- the same index space mwb.lib.scorer_score_files' segment
  matrices use. None if that alignment can't be validated for this dataset
  (module docstring)."""
  official_t = load_human_seg_scores(dataset, systems, root=root)
  if official_t is None:
    return None
  n_positions = official_t.shape[1]
  K = len(systems)
  sys_idx = {s: i for i, s in enumerate(systems)}

  seg_df = load_segment_scores(dataset, root=root)
  seg_df = seg_df[seg_df['system'].isin(systems)]

  a_pos = np.full((K, n_positions), np.nan)
  f_pos = np.full((K, n_positions), np.nan)
  full_pos = np.full((K, n_positions), np.nan)
  for row in seg_df.itertuples(index=False):
    try:
      pos = int(row.seg_id)
    except (TypeError, ValueError):
      continue
    if not (0 <= pos < n_positions):
      continue
    i = sys_idx[row.system]
    a_pos[i, pos] = row.adequacy
    f_pos[i, pos] = row.fluency
    full_pos[i, pos] = row.full

  mask = ~np.isnan(full_pos) & ~np.isnan(official_t)
  if mask.sum() < _MIN_VALIDATION_CELLS:
    return None
  corr = np.corrcoef(full_pos[mask], official_t[mask])[0, 1]
  if not (corr == corr) or corr < _VALIDATION_CORR_FLOOR:
    return None
  return a_pos, f_pos


def scorer_spa_points(
    dataset: str, systems: list[str], root: str = '.',
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> dict[str, tuple[float, float]]:
  """scorer name -> (x=SPA vs Fluency MQM, y=SPA vs Adequacy MQM), for every
  scorer with full system-level coverage (mwb.lib.scorer_score_files.
  load_scorer_sys_scores -- same candidate gate mwb.lib.consistency.
  load_scorer_inputs uses) AND enough jointly-valid positions with a_pos/
  f_pos. {} if positional_af_matrices can't validate this dataset's
  alignment at all."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return {}
  a_pos, f_pos = pos

  candidates = load_scorer_sys_scores(dataset, systems, root=root)
  out = {}
  for name in candidates:
    scorer_seg = load_scorer_seg_scores(dataset, name, systems, root=root)
    if scorer_seg is None or scorer_seg.shape[1] != a_pos.shape[1]:
      continue
    mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0) | np.isnan(scorer_seg).any(axis=0))
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    p_a = pairwise_p_values(a_pos[:, mask], num_permutations, seed)
    p_f = pairwise_p_values(f_pos[:, mask], num_permutations, seed)
    p_m = pairwise_p_values(scorer_seg[:, mask], num_permutations, seed)
    x = soft_pairwise_accuracy_from_pvalues(p_f, p_m)
    y = soft_pairwise_accuracy_from_pvalues(p_a, p_m)
    if x == x and y == y:  # excludes NaN
      out[name] = (x, y)
  return out


# --- Sentinel lines -----------------------------------------------------

def tradeoff_line(
    a_seg: np.ndarray, f_seg: np.ndarray, lambda_grid=LAMBDA_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> list[tuple[float, float]]:
  """[(x, y)] for lam in lambda_grid, score_seg(lam) = lam*a_seg +
  (1-lam)*f_seg -- exact Adequacy MQM vertex (x, 1.0) at lam=1, exact
  Fluency MQM vertex (1.0, y) at lam=0 (soft_pairwise_accuracy_from_pvalues
  of a p-value matrix against itself, at fixed seed, is exactly 1)."""
  p_a = pairwise_p_values(a_seg, num_permutations, seed)
  p_f = pairwise_p_values(f_seg, num_permutations, seed)
  pts = []
  for lam in lambda_grid:
    score = lam * a_seg + (1.0 - lam) * f_seg
    p_m = pairwise_p_values(score, num_permutations, seed)
    x = soft_pairwise_accuracy_from_pvalues(p_f, p_m)
    y = soft_pairwise_accuracy_from_pvalues(p_a, p_m)
    pts.append((x, y))
  return pts


def _combined_range(a_seg: np.ndarray, f_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Per-segment (column) (lo, hi), covering BOTH a_seg and f_seg at once
  (lo = min over both, hi = max over both) -- the range one SHARED noise
  draw is rescaled into, symmetric between the two aspects rather than
  favoring either one's own range."""
  lo = np.minimum(a_seg.min(axis=0, keepdims=True), f_seg.min(axis=0, keepdims=True))
  hi = np.maximum(a_seg.max(axis=0, keepdims=True), f_seg.max(axis=0, keepdims=True))
  return lo, hi


def knowledge_lines(
    a_seg: np.ndarray, f_seg: np.ndarray, lambda_grid=LAMBDA_GRID,
    n_random: int = N_RANDOM_INSTANCES, num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED, noise_seed: int = DEFAULT_NOISE_SEED,
) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float]],
           list[list[tuple[float, float]]], list[tuple[float, float]]]:
  """(a_shadows, a_mean, f_shadows, f_mean): the adequacy- and fluency-
  knowledge lines (paper Figure SPAplane) computed TOGETHER, so their
  shadow instances are genuinely CONNECTED pairs, not two independent
  sweeps. For shadow instance k, ONE noise segment r_seg (per-segment
  Uniform[lo, hi], (lo, hi) = _combined_range(a_seg, f_seg) -- covering
  both aspects' ranges rather than favoring either one) is drawn ONCE and
  used for BOTH curves: score_a(lam) = lam*a_seg + (1-lam)*r_seg (the k-th
  adequacy-knowledge shadow) and score_f(lam) = lam*f_seg + (1-lam)*r_seg
  (the k-th fluency-knowledge shadow). At lam=0 both curves evaluate the
  exact same r_seg, so pts_a[0] == pts_f[0] EXACTLY -- shadow k of one line
  and shadow k of the other visibly meet at that shared noise-only point,
  rather than two independently-scaled endpoints. The OLD design rescaled
  a shared raw draw separately per aspect (once to a_seg's own range, once
  to f_seg's), which kept the two curves correlated but never let them
  actually touch, since a rescaled-to-a's-range point and a rescaled-to-
  f's-range point are generally different points even from the same raw
  draw. p_a/p_f are computed once and reused across every instance and
  lam."""
  p_a = pairwise_p_values(a_seg, num_permutations, seed)
  p_f = pairwise_p_values(f_seg, num_permutations, seed)
  lo, hi = _combined_range(a_seg, f_seg)
  a_shadows, f_shadows = [], []
  for k in range(n_random):
    rng = np.random.default_rng(noise_seed + k)
    noise_seg = lo + rng.random(a_seg.shape) * (hi - lo)
    pts_a, pts_f = [], []
    for lam in lambda_grid:
      score_a = lam * a_seg + (1.0 - lam) * noise_seg
      p_m_a = pairwise_p_values(score_a, num_permutations, seed)
      pts_a.append((soft_pairwise_accuracy_from_pvalues(p_f, p_m_a),
                    soft_pairwise_accuracy_from_pvalues(p_a, p_m_a)))

      score_f = lam * f_seg + (1.0 - lam) * noise_seg
      p_m_f = pairwise_p_values(score_f, num_permutations, seed)
      pts_f.append((soft_pairwise_accuracy_from_pvalues(p_f, p_m_f),
                    soft_pairwise_accuracy_from_pvalues(p_a, p_m_f)))
    a_shadows.append(pts_a)
    f_shadows.append(pts_f)
  a_mean = np.mean(np.array(a_shadows), axis=0).tolist()
  f_mean = np.mean(np.array(f_shadows), axis=0).tolist()
  return a_shadows, [tuple(pt) for pt in a_mean], f_shadows, [tuple(pt) for pt in f_mean]