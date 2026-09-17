"""Adequacy-over-fluency preference over the (ESS, realized beta) plane --
the math for mwb/scripts/compute_beta_ess_preference_heatmap.py.

The question this answers is NOT the one the rest of the project asks. Every
other beta experiment here solves (P) -- given a target beta, find the
min-collision (max-ESS) weighting that realizes it exactly (mwb.lib.
reweight_exact). This module optimizes NOTHING. It covers the simplex
{w : sum(w)=1, w >= 0} and treats each w purely as an observation: it
realizes some beta(w) and some ESS(w), which places it in a cell of the
(ESS, beta) plane, and scores some adequacy-over-fluency preference, which
contributes to that cell's value. The optimum is one point among many; here
it gets no special status.

TWO WAYS TO COVER THE SIMPLEX
-----------------------------
lattice_beta_ess -- the uniform lattice w = c/grid_n, c integral. Exhaustive
and deterministic, but it forces ESS into the finite set
{grid_n^2/m : m integral}: near K the attainable values are spaced about
2/(grid_n/K)^2 apart, and halving that spacing costs a lattice roughly
C((q+1)K+K-1, K-1)/C(qK+K-1, K-1) times bigger -- ~300x at K=12 going from
grid_n=24 to 36. A narrow, finely-resolved ESS window is out of reach this
way.

sample_simplex_beta_ess -- importance sampling against uniform-on-the-
simplex. ESS is CONTINUOUS in w (the simplex is connected and ESS is
continuous on it, so its image is exactly [1, K] and every value in it is
attained), so bins can be placed anywhere at any resolution; the finite
attainable set above is an artifact of discretizing w, not a property of
ESS. The cost is that each cell becomes an estimate with a variance, which
weighted_mean_std's n_eff exists to report.

Both feed the same downstream code (preference_scores, bin_cells) and, where
their cells overlap, agree closely -- median |difference| in cell mean of
0.0009 over 31 shared cells on heen23, despite averaging over genuinely
different measures (counting measure on lattice compositions vs uniform on
the simplex conditioned on the cell).

WHY THIS IS AFFORDABLE (the whole design rests on this)
-------------------------------------------------------
Naively, each lattice point needs a full weighted-SPA sweep over every dial
of every base scorer -- hopeless at millions of points. But weighted SPA is a
RATIO OF TWO QUADRATIC FORMS in w. From mwb.lib.metametrics.
soft_pairwise_accuracy_from_pvalues:

    SPA(w) = 1 - sum_{k<k'} w_k w_k' |p_h[k,k'] - p_s[k,k']|
                 / sum_{k<k'} w_k w_k'

The pairwise p-values p_h, p_s do not depend on w at all (each pair's
permutation test looks only at those two systems' segment scores -- see
pairwise_p_values' own docstring). So per (base scorer, dial) the vector

    d_m = ( |p_h[k,k'] - p_s[k,k']| )_{k<k'}      (length P = K(K-1)/2)

is FIXED, and per lattice point the vector

    q_n = ( w_k w_k' )_{k<k'}                      (length P)

is fixed. Every SPA value in the entire experiment is then one entry of a
single dense matrix product Q @ D.T -- which BLAS does at full speed. The
permutation tests (the genuinely expensive part) run once per (base scorer, dial),
never per weight.

WHAT THE PREFERENCE SCORE IS
----------------------------
The paper's adequacy-over-fluency preference: the Adequacy-fluency dial family
(mwb.lib.synthetic_scorers.base_scorer_family_additive_mean_af -- A = +dial,
F = -dial, so increasing dial trades fluency signal for adequacy signal),
reduced by mwb.lib.synthetic_scorer_preference.preference_score -- the fraction
of dial-adjacent pairs where the weighted meta-metric prefers the
more-adequacy-leaning scorer. 1.0 = always prefers adequacy, 0.0 = always
fluency, 0.5 = no preference. Averaged over every base scorer of the
dataset, exactly as the main results are.

Note this family is deliberately NOT routed through mwb.lib.synthetic_scorer_
beta_grid.base_scorer_beta_dial_grid, which only builds A/F plus T-or-J and has
no Adequacy-fluency option; the generator is called directly instead.

VALIDITY MASK
-------------
pairwise_p_values returns NaN off the upper triangle by construction, and
can in principle return NaN within it. The mask is taken ONCE PER BASE SCORER (the
pairs finite in p_human and in every one of that base scorer's dials), not per
dial, so the SPA denominator is identical across a base scorer's dials and the
preference comparison between two dials is never contaminated by the two
being averaged over different pair sets.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from mwb.lib.metametrics import pairwise_p_values
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from mwb.lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices
from mwb.lib.synthetic_scorers import AF_ADDITIVE_MEAN_DIAL_GRID, base_scorer_family_additive_mean_af
from mwb.lib.reweight_exhaustive import _binom_table, _simplex_lattice_chunk

# _simplex_lattice_chunk/_binom_table are imported rather than reimplemented:
# they are the project's already-cross-validated combinadic simplex-lattice
# generator (see mwb/lib/reweight_exhaustive.py's docstring for the validation
# and the ~50-90x speedup over itertools). Reimplementing the same bijection
# here would be a second thing to get wrong for no benefit. They are private
# to that module only in the sense of "not part of the solver's public API".


@dataclasses.dataclass
class BaseScorerPValues:
  """One base scorer's fixed, w-independent SPA ingredients.

  diffs: (n_dials, P) -- |p_human - p_dial| over the P valid system pairs,
         rows in ascending dial order.
  valid_pairs: (P,) int index into the K(K-1)/2 upper-triangle pair list.
  """
  base_scorer: str
  diffs: np.ndarray
  valid_pairs: np.ndarray


def _upper_tri_pairs(K: int) -> tuple[np.ndarray, np.ndarray]:
  return np.triu_indices(K, 1)


def base_scorer_pvalue_table(
    dataset: str, base_scorer: str, systems: list[str], a_pos: np.ndarray, f_pos: np.ndarray,
    human_seg: np.ndarray, dial_grid=AF_ADDITIVE_MEAN_DIAL_GRID, root: str = '.',
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
    min_segments: int = 10,
) -> BaseScorerPValues | None:
  """Build one base scorer's Adequacy-fluency family and run its per-dial permutation
  tests -- the expensive step, done once per base scorer for the whole experiment.
  None if the base scorer lacks the joint segment coverage (same gate as
  base_scorer_beta_dial_grid)."""
  base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None
  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if int(mask.sum()) < min_segments:
    return None

  a_m, f_m = a_pos[:, mask], f_pos[:, mask]
  base_scorer_m, human_m = base_scorer_seg[:, mask], human_seg[:, mask]

  p_human = pairwise_p_values(human_m, num_permutations, seed)
  family = base_scorer_family_additive_mean_af(base_scorer_m, a_m, f_m, dial_grid)
  dials = sorted(family)
  p_dials = [pairwise_p_values(family[d], num_permutations, seed) for d in dials]

  iu = _upper_tri_pairs(a_pos.shape[0])
  finite = np.isfinite(p_human[iu])
  for pd in p_dials:
    finite &= np.isfinite(pd[iu])
  valid = np.flatnonzero(finite)
  if len(valid) < 1:
    return None

  diffs = np.stack([np.abs(p_human[iu][valid] - pd[iu][valid]) for pd in p_dials])
  return BaseScorerPValues(base_scorer=base_scorer, diffs=diffs, valid_pairs=valid)


def load_base_scorer_tables(
    dataset: str, systems: list[str], base_scorers: list[str], root: str = '.',
    dial_grid=AF_ADDITIVE_MEAN_DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED,
    progress=None,
) -> list[BaseScorerPValues]:
  """Every usable base scorer's BaseScorerPValues for one dataset. Raises if the
  dataset has no positional alignment or human segment scores at all (i.e.
  the whole experiment is impossible there -- enru22)."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    raise ValueError(f'{dataset}: positional alignment unavailable; no base_scorer can be built')
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    raise ValueError(f'{dataset}: human segment scores unavailable or misaligned')

  out = []
  for i, d in enumerate(base_scorers):
    tbl = base_scorer_pvalue_table(dataset, d, systems, a_pos, f_pos, human_seg,
                             dial_grid=dial_grid, root=root,
                             num_permutations=num_permutations, seed=seed)
    if tbl is not None:
      out.append(tbl)
    if progress is not None:
      progress(i + 1, len(base_scorers), d, tbl is not None)
  return out


def lattice_size(K: int, grid_n: int) -> int:
  """C(grid_n + K - 1, K - 1) -- the number of points on the K-dim simplex
  lattice at resolution grid_n."""
  return math.comb(grid_n + K - 1, K - 1)


def max_lattice_ess(K: int, grid_n: int) -> float:
  """The largest ESS any point of this lattice can reach: min(K, grid_n).

  ESS is maximized by spreading weight as evenly as possible, and a
  composition of grid_n units into K slots can occupy at most min(K, grid_n)
  slots -- so grid_n < K caps ESS at grid_n no matter how many systems
  there are, and the uniform point (ESS = K) is on the lattice only when K
  divides grid_n. Callers should check this before choosing an ESS window;
  a window above it is silently empty."""
  return float(min(K, grid_n))


def _comp_dtype(grid_n: int):
  """Smallest unsigned int dtype holding a composition entry (0..grid_n)."""
  if grid_n <= np.iinfo(np.uint8).max:
    return np.uint8
  if grid_n <= np.iinfo(np.uint16).max:
    return np.uint16
  return np.uint32


def lattice_beta_ess(
    a: np.ndarray, f: np.ndarray, grid_n: int, ess_lo: float | None = None,
    ess_hi: float | None = None, chunk: int = 1_000_000, progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
  """Stream the ENTIRE K-dim simplex lattice at resolution grid_n, keeping
  only points whose ESS falls in [ess_lo, ess_hi] (both optional).

  Returns (comps, beta, ess, n_total, ess_min_seen, ess_max_seen). `comps`
  holds the surviving points as the INTEGER COMPOSITIONS themselves, in the
  narrowest unsigned dtype that fits grid_n -- NOT as float64 weights.
  Recover the weights with comps[rows].astype(float) / grid_n, on the subset
  actually being scored.

  Two separate things make grid_n >= K affordable, and both matter:

  (1) Filtering inside the chunk loop, so memory scales with the survivor
      count rather than with C(grid_n+K-1, K-1), which is what explodes.
  (2) Storing survivors as compositions rather than weights. At K=12,
      grid_n=24 roughly 95M points survive a typical ESS window; as float64
      weights that is ~9.2GB (and ~18GB peak through the concatenate, which
      in practice thrashes and never returns), while as uint8 compositions
      it is ~1.1GB. The float conversion then happens per scoring chunk, on
      a few thousand rows at a time.

  beta(w) and ESS(w) are pure vectorized tensor ops -- no solver, no scorer
  -- so the discarded majority costs almost nothing.

  Weights are integer compositions divided by grid_n, so the lattice
  contains the uniform point exactly iff K divides grid_n."""
  K = len(a)
  M = grid_n + K - 1
  n_total = math.comb(M, K - 1)
  device = torch.device('cpu')
  binom = {j: _binom_table(M, j, device) for j in range(1, K)}
  cdt = _comp_dtype(grid_n)

  a_t = torch.as_tensor(np.asarray(a, dtype=np.float64))
  f_t = torch.as_tensor(np.asarray(f, dtype=np.float64))
  a2, f2 = a_t ** 2, f_t ** 2

  comp_parts, beta_parts, ess_parts = [], [], []
  n_kept = 0
  ess_min_seen, ess_max_seen = float('inf'), float('-inf')
  for start in range(0, n_total, chunk):
    end = min(start + chunk, n_total)
    c = _simplex_lattice_chunk(K, grid_n, start, end, device, binom)
    w = c.to(torch.float64) / float(grid_n)

    ess = 1.0 / (w ** 2).sum(1)
    ess_min_seen = min(ess_min_seen, float(ess.min()))
    ess_max_seen = max(ess_max_seen, float(ess.max()))

    sel = torch.ones_like(ess, dtype=torch.bool)
    if ess_lo is not None:
      sel &= ess >= ess_lo
    if ess_hi is not None:
      sel &= ess <= ess_hi
    if not bool(sel.any()):
      if progress is not None:
        progress(end, n_total, n_kept)
      continue

    c, w, ess = c[sel], w[sel], ess[sel]
    mu_a, mu_f = w @ a_t, w @ f_t
    var_a = ((w * a2).sum(1) - mu_a ** 2).clamp_min(0.0)
    var_f = ((w * f2).sum(1) - mu_f ** 2).clamp_min(0.0)
    denom = var_a + var_f
    beta = torch.where(denom > 0, var_a / denom, torch.full_like(denom, float('nan')))

    comp_parts.append(c.numpy().astype(cdt, copy=False))
    beta_parts.append(beta.numpy().astype(np.float32))
    ess_parts.append(ess.numpy().astype(np.float32))
    n_kept += int(sel.sum())
    if progress is not None:
      progress(end, n_total, n_kept)

  if not comp_parts:
    return (np.zeros((0, K), dtype=cdt), np.zeros(0, np.float32), np.zeros(0, np.float32),
            n_total, ess_min_seen, ess_max_seen)
  return (np.concatenate(comp_parts), np.concatenate(beta_parts), np.concatenate(ess_parts),
          n_total, ess_min_seen, ess_max_seen)


def preference_scores(
    w: np.ndarray, tables: list[BaseScorerPValues], K: int, chunk: int = 4096,
) -> np.ndarray:
  """Adequacy-over-fluency preference for each row of `w` (n, K): for every
  base scorer, weighted SPA at every dial via the Q @ D.T identity in this
  module's docstring, reduced by the adjacent-pair preference rule, then
  averaged across base scorers.

  Returns (n,) with NaN wherever no base scorer produced a finite score."""
  iu_i, iu_j = _upper_tri_pairs(K)
  n = w.shape[0]
  out = np.full(n, np.nan)

  for start in range(0, n, chunk):
    W = w[start:start + chunk]
    # q[n, p] = w_k * w_k' for the p-th upper-triangle pair.
    q_full = W[:, iu_i] * W[:, iu_j]

    per_base_scorer = np.full((W.shape[0], len(tables)), np.nan)
    for di, tbl in enumerate(tables):
      q = q_full[:, tbl.valid_pairs]
      den = q.sum(axis=1)
      # (n_w, n_dials): weighted mean |p_h - p_s|, i.e. 1 - SPA.
      # errstate wraps the matmul for the same reason mwb.lib.metametrics.
      # pairwise_p_values does: macOS/Accelerate's BLAS emits spurious
      # divide-by-zero/overflow/invalid warnings on float64 @ for some
      # shapes regardless of the actual values. Not a real numeric fault.
      with np.errstate(all='ignore'):
        num = q @ tbl.diffs.T
        spa = 1.0 - num / den[:, None]
      spa[den <= 0] = np.nan

      # preference_score, adjacent pairs, vectorized over weights: a win
      # where the higher dial scores higher, half a win on an exact tie.
      lo, hi = spa[:, :-1], spa[:, 1:]
      wins = np.where(hi > lo, 1.0, np.where(hi == lo, 0.5, 0.0))
      wins[~(np.isfinite(lo) & np.isfinite(hi))] = np.nan
      with np.errstate(invalid='ignore'):
        per_base_scorer[:, di] = np.nanmean(wins, axis=1)

    with np.errstate(invalid='ignore'):
      out[start:start + chunk] = np.nanmean(per_base_scorer, axis=1)
  return out


def alpha_for_target_ess(K: int, e: float, alpha_max: float = 1e5,
                         alpha_min: float = 0.02) -> float:
  """The Dirichlet concentration alpha whose typical draw has ESS ~= e.

  For x ~ Dir(alpha*1_K), Var(x_k) = (1/K)(1-1/K)/(K*alpha+1), and
  E[sum x^2] ~= 1/K + (1-1/K)/(K*alpha+1), so ESS ~= 1/that. Inverting:

      alpha = ( e(K-1)/(K-e) - 1 ) / K

  Exact only to first order (it ignores the variance of sum x^2 itself),
  which is all that is needed: this only has to place proposal components in
  roughly the right part of the ESS range, and the importance weights
  correct whatever it actually produces. alpha -> infinity as e -> K, hence
  the cap.

  alpha_min is deliberately BELOW 1. The formula returns alpha < 1 for any
  target below about (K+1)/2 -- the typical ESS of a uniform draw on the
  simplex -- and those sparse Dirichlets, which pile mass near the simplex
  boundary, are the only way to reach the low-ESS end at all. Clipping at 1
  would leave every target below ~K/2 served solely by the alpha=1
  component, i.e. by rare tail draws, and the resulting cells would have
  n_eff of a handful however many millions were drawn. This floor only
  engages for windows reaching well below K/2; the committed
  [0.8K, K] window asks for alpha >= 3.6 at K=13 and never sees it."""
  if e >= K:
    return alpha_max
  alpha = (e * (K - 1) / (K - e) - 1.0) / K if e > 1.0 else alpha_min
  return float(np.clip(alpha, alpha_min, alpha_max))


def build_proposal_alphas(K: int, ess_lo: float, ess_hi: float, n_alpha: int = 8) -> np.ndarray:
  """Concentrations for the defensive mixture proposal, placed so the
  mixture's typical ESS spans [ess_lo, ess_hi].

  alpha = 1 (uniform on the simplex, i.e. the TARGET measure itself) is
  always included, both to cover the low-ESS end and so that every point of
  the simplex has strictly positive proposal density -- without it, a draw
  on the simplex boundary (some w_k = 0) would have zero density under every
  alpha > 1 component and an undefined importance weight."""
  targets = np.linspace(ess_lo, min(ess_hi, K - (K - ess_lo) * 1e-3), max(n_alpha - 1, 1))
  alphas = [1.0] + [alpha_for_target_ess(K, float(e)) for e in targets]
  return np.unique(np.round(np.asarray(alphas, dtype=float), 6))


def _log_dirichlet_mixture_density(x: np.ndarray, alphas: np.ndarray) -> np.ndarray:
  """log q(x) for the equal-weight mixture (1/M) sum_m Dir(x; alpha_m * 1).

  The additive -log(M) is dropped: every importance weight carries it, and
  they are only ever used after normalizing within a cell."""
  from scipy.special import gammaln, logsumexp
  K = x.shape[1]
  with np.errstate(divide='ignore'):
    log_x_sum = np.log(x).sum(axis=1)      # -inf on the simplex boundary
  comps = np.empty((x.shape[0], len(alphas)))
  for m, al in enumerate(alphas):
    const = gammaln(K * al) - K * gammaln(al)
    comps[:, m] = const + (al - 1.0) * log_x_sum if al != 1.0 else const
  return logsumexp(comps, axis=1)


def sample_simplex_beta_ess(
    a: np.ndarray, f: np.ndarray, n_samples: int, alphas: np.ndarray,
    ess_lo: float | None = None, ess_hi: float | None = None,
    seed: int = 0, chunk: int = 500_000, progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
  """Importance-sampling counterpart of lattice_beta_ess.

  Draws from the defensive Dirichlet mixture `alphas`, keeps points whose
  ESS is in [ess_lo, ess_hi], and returns (w, beta, ess, log_iw, n_drawn),
  where log_iw is the UNNORMALIZED log importance weight taking the mixture
  proposal back to the target measure -- uniform on the simplex, i.e.
  Dir(1,...,1):

      log_iw = log p(x) - log q(x) = const - log q(x)

  Callers must normalize log_iw within whatever subset they average over
  (see weighted_mean_std); only differences matter, so the dropped constants
  are irrelevant.

  Why any of this: ESS is CONTINUOUS in w (the simplex is connected and ESS
  is continuous on it, so its image is exactly [1, K] -- every value is
  attained). The lattice's finitely many ESS values are an artifact of
  discretizing w as c/grid_n with c integral, which forces ESS into
  {grid_n^2/m : m integral} and, near K, leaves gaps of about 2/(grid_n/K)^2
  that cost a ~K-fold larger lattice to halve. Sampling has no such
  structure: an ESS bin can be placed anywhere in [1, K], as narrow as
  wanted. The price is that the answer is now an estimate with a variance,
  which is what the importance weights and their own effective sample size
  (weighted_mean_std) exist to keep honest."""
  K = len(a)
  rng = np.random.default_rng(seed)
  a = np.asarray(a, dtype=float)
  f = np.asarray(f, dtype=float)

  w_parts, beta_parts, ess_parts, iw_parts, cov_parts = [], [], [], [], []
  drawn = 0
  while drawn < n_samples:
    m = min(chunk, n_samples - drawn)
    # Equal-weight mixture: assign each draw to one component.
    comp = rng.integers(0, len(alphas), size=m)
    x = np.empty((m, K))
    for mi, al in enumerate(alphas):
      sel = comp == mi
      if sel.any():
        x[sel] = rng.dirichlet(np.full(K, al), size=int(sel.sum()))
    drawn += m

    ess = 1.0 / np.sum(x ** 2, axis=1)
    keep = np.ones(m, dtype=bool)
    if ess_lo is not None:
      keep &= ess >= ess_lo
    if ess_hi is not None:
      keep &= ess <= ess_hi
    if keep.any():
      xk, essk = x[keep], ess[keep]
      # errstate: macOS/Accelerate BLAS emits spurious divide/overflow/
      # invalid warnings on float64 @ for some shapes -- see the same guard
      # in mwb.lib.metametrics.pairwise_p_values.
      with np.errstate(all='ignore'):
        mu_a, mu_f = xk @ a, xk @ f
        var_a = np.maximum((xk * a ** 2).sum(1) - mu_a ** 2, 0.0)
        var_f = np.maximum((xk * f ** 2).sum(1) - mu_f ** 2, 0.0)
        den = var_a + var_f
        beta = np.where(den > 0, var_a / den, np.nan)
      # Sigma(a,f;w), the weighted adequacy-fluency covariance -- another
      # quadratic form in w, carried alongside beta so that residual analyses
      # can ask whether it, rather than beta, drives the preference.
      cov = (xk * (a * f)).sum(1) - mu_a * mu_f
      w_parts.append(xk)
      beta_parts.append(beta)
      ess_parts.append(essk)
      cov_parts.append(cov)
      iw_parts.append(-_log_dirichlet_mixture_density(xk, alphas))
    if progress is not None:
      progress(drawn, n_samples, sum(len(p) for p in w_parts))

  if not w_parts:
    return (np.zeros((0, K)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), drawn)
  return (np.concatenate(w_parts), np.concatenate(beta_parts), np.concatenate(ess_parts),
          np.concatenate(iw_parts), np.concatenate(cov_parts), drawn)


def weighted_mean_std(v: np.ndarray, log_iw: np.ndarray | None) -> tuple[float, float, float]:
  """(mean, std, n_eff) of `v` under normalized importance weights.

  log_iw None -> plain unweighted mean/std with n_eff = len(v), which is the
  lattice path (every enumerated point counts once).

  n_eff = 1 / sum(p_i^2) on the normalized weights is Kish's effective
  sample size -- the same formula as ESS(w) itself, one level up. It is the
  honest precision of the cell: a cell with 10,000 draws but n_eff = 3 is
  reporting essentially three observations, and its mean/std should not be
  read as if it had 10,000."""
  v = np.asarray(v, dtype=float)
  ok = np.isfinite(v)
  if not ok.any():
    return float('nan'), float('nan'), 0.0
  v = v[ok]
  if log_iw is None:
    return float(v.mean()), float(v.std()), float(len(v))

  lw = np.asarray(log_iw, dtype=float)[ok]
  lw = lw - lw.max()  # stabilize before exponentiating
  p = np.exp(lw)
  total = p.sum()
  if not np.isfinite(total) or total <= 0:
    return float('nan'), float('nan'), 0.0
  p = p / total
  mean = float(np.dot(p, v))
  var = float(np.dot(p, (v - mean) ** 2))
  return mean, float(np.sqrt(max(var, 0.0))), float(1.0 / np.sum(p ** 2))


def bin_cells(
    beta: np.ndarray, ess: np.ndarray, beta_edges: np.ndarray, ess_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(row, col, inside) index of each point into the cell grid. `row` indexes
  beta_edges (y), `col` indexes ess_edges (x); both are -1 where the point
  falls outside the window, flagged by `inside`."""
  row = np.digitize(beta, beta_edges) - 1
  col = np.digitize(ess, ess_edges) - 1
  # np.digitize puts a value exactly on the top edge in a new bin past the
  # end; fold it back so the window is closed on both sides.
  row = np.where(beta == beta_edges[-1], len(beta_edges) - 2, row)
  col = np.where(ess == ess_edges[-1], len(ess_edges) - 2, col)
  inside = (
      (row >= 0) & (row < len(beta_edges) - 1)
      & (col >= 0) & (col < len(ess_edges) - 1)
      & np.isfinite(beta) & np.isfinite(ess)
  )
  return np.where(inside, row, -1), np.where(inside, col, -1), inside
