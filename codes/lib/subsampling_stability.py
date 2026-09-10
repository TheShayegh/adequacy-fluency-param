"""Subsampling pairwise-stability robustness metric -- a fresh replacement
for tau-based robustness (used throughout the type7 family), motivated by
a real gap in tau: it's threshold-blind, so a reweighting-induced flip of
two near-tied scorers counts exactly as much "inconsistency" as a flip of
two robustly-separated ones.

This is SUBSAMPLING (Politis & Romano, "Subsampling", 1999), not the
bootstrap: draws are size-K' subsets of D drawn WITHOUT replacement
(K' < K), never a with-replacement resample of size K. Deliberately not
the bootstrap -- a with-replacement draw could repeat a system, which has
no clean meaning in this project's weight-per-system framework (would a
repeated system get double weight? merge with itself?), and subsampling
sidesteps the question entirely by construction.

For every pair of scorers (i, j), draw B size-K' subsets D' of D (without
replacement, from D's outlier-screened pool) and measure the EMPIRICAL
RATE at which i and j's relative order flips across draws:

  p(i>j) = mean( M(i; D') > M(j; D') )  over the B draws
  p(j>i) = mean( M(j; D') > M(i; D') )  over the B draws
  stability(i,j) = |p(i>j) - p(j>i)|  in [0, 1]

0 means the pair's order is a coin flip across subsamples -- indistinguish-
able from noise, exactly the near-tie case tau can't see. 1 means the
order never flips across any draw -- a robust, real gap. No threshold, no
clustering, no distributional assumption: the "softness" comes entirely
from resampling variability, nonparametrically.

robustness(D; K', B, M) = mean(stability(i,j)) over every scorer pair.

M is the meta-metric under one of two conditions, both computed via the
SAME B draws (same seed passed to both calls -- draw_subsets rebuilds a
fresh np.random.default_rng(seed) every call, with no dependence on
anything else that ran in between, so two pairwise_stability calls with
matching (dataset, K_prime, B, seed, exclude_outliers) are GUARANTEED
subsample-identical regardless of target_alpha) for a fair paired
comparison:
  - natural (target_alpha=None): plain scorer_scores on D' with uniform
    weights -- "does this pair's order survive realistic system
    resampling AT ALL, with no correction."
  - reweighted (target_alpha=<float>): D' is reweighted (lib.
    reweight_exact.solve_w_exact) to hit target_alpha -- the SAME fixed
    target for every one of the B draws, so every draw is pinned to a
    common anchor instead of floating to its own natural balance, rather
    than type7's D-vs-D' recoverability question. target_alpha is a free
    parameter (no baked-in default here) specifically so a caller can
    sweep it; the natural default to reach for is alpha_0(D), the full
    pool's own natural balance.

The headline comparison a caller makes: robustness(D;K',B,M,target_alpha)
minus robustness(D;K',B,M,None) -- does pinning the balance make the
scorer ranking more reproducible across subsamples than leaving it
uncontrolled?
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import math
import os

import numpy as np
import pandas as pd

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import pool_weighted_tau, real_scorers, real_systems, scorer_scores
from lib.icc import icc_c1
from lib.metametrics import NEEDS_SEGMENT_SCORES
from lib.reweight_exact import solve_w_exact
from mwb.mqm_scoring import load_system_scores


def draw_subsets(K: int, K_prime: int, B: int, seed: int) -> list[tuple[int, ...]]:
  """B DISTINCT size-K_prime index-subsets of range(K) -- Politis-Romano-
  Wolf subsampling, NOT bootstrap: each draw is itself without replacement
  (no repeated system within a draw), and draws are deduplicated against
  each other too (a repeat subset can otherwise occur by chance for small
  C(K,K')). Raises if B exceeds the number of distinct subsets that exist.

  Deterministic in `seed` alone: builds a fresh np.random.default_rng(seed)
  every call, untouched by any other randomness elsewhere in the process --
  two calls with the same (K, K_prime, B, seed) always return the same
  list, in the same order, regardless of what else has run in between."""
  n_possible = math.comb(K, K_prime)
  if B > n_possible:
    raise ValueError(f'B={B} exceeds C({K},{K_prime})={n_possible} distinct subsets available')
  rng = np.random.default_rng(seed)
  seen = set()
  out = []
  while len(out) < B:
    idx = tuple(sorted(rng.choice(K, size=K_prime, replace=False).tolist()))
    if idx not in seen:
      seen.add(idx)
      out.append(idx)
  return out


def _alpha_0_over_subsets(dataset: str, K_prime: int, root: str = '.',
                           exclude_outliers: bool = True) -> list[float]:
  """alpha_0(D') for EVERY size-K_prime subset D' of D (exhaustive
  enumeration, exact -- not sampled: C(K,K') stays small at this project's
  K~11-17, e.g. C(13,8)=1287, cheap since alpha_0 is closed-form with no
  solving involved). Shared by median_alpha_0 and mean_alpha_0."""
  full_systems = real_systems(dataset, root=root, exclude_outliers=exclude_outliers)
  df = load_system_scores(dataset, root=root)
  K = len(full_systems)
  if not (2 <= K_prime < K):
    raise ValueError(f"K'={K_prime} must be in [2, {K - 1}] for this pool (K={K})")
  alphas = []
  for combo in itertools.combinations(full_systems, K_prime):
    a_sub = df.loc[list(combo), 'a'].values
    b_sub = df.loc[list(combo), 'b'].values
    alphas.append(alpha0_of(a_sub, b_sub))
  return alphas


def median_alpha_0(dataset: str, K_prime: int, root: str = '.',
                    exclude_outliers: bool = True) -> float:
  """Median of alpha_0(D') over EVERY size-K_prime subset D' of D.

  Motivation: the default reweighting target alpha_0(D) is D's OWN natural
  balance, but a K'-sized subsample D' is drawn from a structurally smaller,
  different population -- alpha_0(D) can be a poor, barely-reachable target
  for a typical D' (this is the "reversed reweighting" failure mode also
  seen in the type7 family: pinning a smaller pool toward a larger, external
  pool's balance forces low-ESS, concentrated reweighting). median_alpha_0
  instead asks what K'-sized subsets THEMSELVES naturally look like, so the
  target is drawn from the same population being reweighted. Robust to
  skew/outlier subsets among the C(K,K') alphas -- see mean_alpha_0 for the
  non-robust sibling."""
  alphas = _alpha_0_over_subsets(dataset, K_prime, root=root, exclude_outliers=exclude_outliers)
  return float(np.median(alphas))


def mean_alpha_0(dataset: str, K_prime: int, root: str = '.',
                  exclude_outliers: bool = True) -> float:
  """Mean of alpha_0(D') over EVERY size-K_prime subset D' of D -- same
  motivation and exhaustive-enumeration machinery as median_alpha_0, but
  the (non-robust) mean rather than the median. Comparing the two indicates
  whether the distribution of alpha_0 across K'-sized subsets is skewed:
  a large mean/median gap means a few atypical subsets pull the mean away
  from where most subsets actually sit."""
  alphas = _alpha_0_over_subsets(dataset, K_prime, root=root, exclude_outliers=exclude_outliers)
  return float(np.mean(alphas))


def target_alpha_ess_sweep(
    dataset: str, K_prime: int, B: int, target_alphas: list[float],
    seed: int = 0, root: str = '.', exclude_outliers: bool = True,
    progress_every: int | None = None,
) -> dict:
  """For each candidate in target_alphas, reweight the SAME B draws (drawn
  once, shared across every candidate via `seed` -- draw_subsets
  determinism, same convention as pairwise_stability) to that target via
  solve_w_exact, and record mean ESS across the draws feasible for it.

  Deliberately skips scorer_scores/metametric computation entirely -- this
  measures only how COSTLY a target is to reach (mean ESS), not its effect
  on scorer-ranking stability, which is orders of magnitude cheaper to sweep
  over many candidates (no SPA scoring in the loop). Pair with
  pairwise_stability at whichever target this sweep recommends (e.g. the
  ESS-argmax) to then check its actual effect on robustness."""
  full_systems = real_systems(dataset, root=root, exclude_outliers=exclude_outliers)
  df = load_system_scores(dataset, root=root)
  K = len(full_systems)
  if not (2 <= K_prime < K):
    raise ValueError(f"K'={K_prime} must be in [2, {K - 1}] for this pool (K={K})")

  draws = draw_subsets(K, K_prime, B, seed)
  ab_list = []
  for idx in draws:
    subset = [full_systems[k] for k in idx]
    ab_list.append((df.loc[subset, 'a'].values, df.loc[subset, 'b'].values))

  mean_ess, n_infeasible = [], []
  for ti, target in enumerate(target_alphas, 1):
    ess_vals = []
    infeasible = 0
    for a_sub, b_sub in ab_list:
      try:
        r = solve_w_exact(a_sub, b_sub, target)
      except ValueError:
        infeasible += 1
        continue
      ess_vals.append(float(r.ess))
    mean_ess.append(float(np.mean(ess_vals)) if ess_vals else float('nan'))
    n_infeasible.append(infeasible)
    if progress_every and (ti % progress_every == 0 or ti == len(target_alphas)):
      import sys
      print(f'  [{ti}/{len(target_alphas)} targets]', file=sys.stderr, flush=True)

  return {
      'dataset': dataset, 'K': K, 'K_prime': K_prime, 'B': B, 'seed': seed,
      'target_alphas': list(target_alphas), 'mean_ess': mean_ess,
      'n_infeasible': n_infeasible,
  }


def pairwise_stability(
    dataset: str,
    metametric: str,
    K_prime: int,
    B: int,
    target_alpha: float | None = None,
    root: str = '.',
    seed: int = 0,
    exclude_outliers: bool = True,
    progress_every: int | None = None,
) -> dict:
  """Computes robustness(D; K', B, metametric) at a given target_alpha
  (None = natural/unweighted condition; a float = reweight every draw to
  that fixed target). Returns a dict with the aggregate `robustness`
  scalar, the full per-pair `stability` map, and bookkeeping (how many
  draws were actually usable, how many were infeasible for the requested
  target -- only relevant when target_alpha is set, since a small K' can
  push the target outside that specific draw's own reachable range,
  Corollary 2).

  seed: pass the SAME seed (together with matching dataset/K_prime/B/
  exclude_outliers) across calls you want compared on identical subsamples
  -- e.g. one call with target_alpha=None and another with target_alpha=
  <a>, same seed, to get a fair paired natural-vs-reweighted comparison (see
  draw_subsets: same seed => bit-identical draws, unconditionally).

  exclude_outliers (default True): passed straight through to real_systems
  -- the project standard as of this session is lib.outlier_detection.
  wmt_official_outliers, baked into real_systems itself, so this function
  does no outlier screening of its own."""
  full_systems = real_systems(dataset, root=root, exclude_outliers=exclude_outliers)
  outliers = ([s for s in real_systems(dataset, root=root, exclude_outliers=False)
               if s not in set(full_systems)] if exclude_outliers else [])
  df = load_system_scores(dataset, root=root)

  K = len(full_systems)
  if not (2 <= K_prime < K):
    raise ValueError(f"K'={K_prime} must be in [2, {K - 1}] for this pool (K={K})")

  draws = draw_subsets(K, K_prime, B, seed)

  # Fixes the scorer-pair universe once, from the full pool -- a scorer
  # missing from some individual draw (insufficient coverage) is just
  # skipped for the pairs it's in on that draw, not dropped globally.
  natural_full = scorer_scores(dataset, metametric, root=root, systems=full_systems)
  scorers = sorted(natural_full.index)
  pairs = list(itertools.combinations(scorers, 2))
  # counts[pair] = [n_i_gt_j, n_j_gt_i, n_valid_draws]
  counts = {pair: [0, 0, 0] for pair in pairs}

  n_infeasible = 0
  n_used = 0
  ess_list = []
  for di, idx in enumerate(draws, 1):
    subset = [full_systems[k] for k in idx]
    if target_alpha is None:
      scores = scorer_scores(dataset, metametric, root=root, systems=subset)
      ess_list.append(float(K_prime))  # uniform weights -- ESS is K' by definition
    else:
      a_sub = df.loc[subset, 'a'].values
      b_sub = df.loc[subset, 'b'].values
      try:
        r = solve_w_exact(a_sub, b_sub, target_alpha)
      except ValueError:
        n_infeasible += 1
        continue
      scores = scorer_scores(dataset, metametric, root=root, w=r.w, systems=subset)
      ess_list.append(float(r.ess))
    n_used += 1

    for (i, j) in pairs:
      si, sj = scores.get(i, float('nan')), scores.get(j, float('nan'))
      if si != si or sj != sj:  # NaN check (NaN != NaN)
        continue
      c = counts[(i, j)]
      c[2] += 1
      if si > sj:
        c[0] += 1
      elif sj > si:
        c[1] += 1
      # exact tie: neither incremented -- vanishingly rare for continuous
      # scores, and correctly shows up as slightly lower stability either way.

    if progress_every and (di % progress_every == 0 or di == len(draws)):
      import sys
      print(f'  [{di}/{len(draws)} draws] target_alpha={target_alpha}', file=sys.stderr, flush=True)

  stability = {}
  for pair, (n_i, n_j, n_valid) in counts.items():
    if n_valid == 0:
      stability[pair] = float('nan')
      continue
    stability[pair] = abs(n_i / n_valid - n_j / n_valid)

  valid = [v for v in stability.values() if v == v]
  robustness = float(np.mean(valid)) if valid else float('nan')
  mean_ess = float(np.mean(ess_list)) if ess_list else float('nan')

  return {
      'dataset': dataset, 'metametric': metametric, 'K': K, 'K_prime': K_prime,
      'B': B, 'target_alpha': target_alpha, 'scorers': scorers,
      'stability': stability, 'robustness': robustness, 'mean_ess': mean_ess,
      'n_pairs': len(pairs), 'n_pairs_valid': len(valid),
      'n_draws_used': n_used, 'n_infeasible': n_infeasible,
      'outliers_dropped': outliers,
  }


def generalizability_stability(
    dataset: str,
    metametric: str,
    K_prime: int,
    B: int,
    target_alpha: float | None = None,
    root: str = '.',
    seed: int = 0,
    exclude_outliers: bool = True,
    progress_every: int | None = None,
    systems: list[str] | None = None,
) -> dict:
  """ICC(C,1) (lib.icc.icc_c1) treating scorers as the objects and the B
  subsamples as the raters: for each draw, every scorer gets one
  metametric score (natural, uniform weights, or reweighted to
  target_alpha -- same two conditions and same seed/draw-sharing contract
  as pairwise_stability, see that function's docstring). Replaces
  pairwise_stability's ad hoc pairwise flip-rate with a proper
  generalizability-theory reliability coefficient: BMS/EMS mean squares
  are unbiased regardless of B, unlike raw empirical proportions estimated
  from only B draws.

  systems: overrides the base pool D (default: real_systems(dataset)) with
  an explicit list -- e.g. one leave-one-out D' of some outer D, for a
  NESTED subsampling analysis (subsample within a subsample). When given,
  exclude_outliers is ignored and no outliers_dropped list is reported
  (the override already defines the pool; there's nothing left to screen).

  A scorer missing a valid score on ANY draw actually used is dropped
  entirely (icc_c1 needs a fully-crossed, no-missing-cells matrix) --
  reported as `n_scorers_dropped`. An infeasible reweighting draw
  (Corollary 2: target_alpha outside that draw's reachable range) is
  dropped from B for this call only, same as pairwise_stability."""
  if systems is not None:
    full_systems = list(systems)
    outliers = []
  else:
    full_systems = real_systems(dataset, root=root, exclude_outliers=exclude_outliers)
    outliers = ([s for s in real_systems(dataset, root=root, exclude_outliers=False)
                 if s not in set(full_systems)] if exclude_outliers else [])
  df = load_system_scores(dataset, root=root)

  K = len(full_systems)
  if not (2 <= K_prime < K):
    raise ValueError(f"K'={K_prime} must be in [2, {K - 1}] for this pool (K={K})")

  draws = draw_subsets(K, K_prime, B, seed)

  natural_full = scorer_scores(dataset, metametric, root=root, systems=full_systems)
  scorers = sorted(natural_full.index)

  # rows[scorer] accumulates one score per USED draw, in draw order --
  # a scorer with a NaN on any used draw is dropped after the loop.
  rows = {s: [] for s in scorers}
  n_infeasible = 0
  n_used = 0
  ess_list = []
  for di, idx in enumerate(draws, 1):
    subset = [full_systems[k] for k in idx]
    if target_alpha is None:
      scores = scorer_scores(dataset, metametric, root=root, systems=subset)
      ess_list.append(float(K_prime))
    else:
      a_sub = df.loc[subset, 'a'].values
      b_sub = df.loc[subset, 'b'].values
      try:
        r = solve_w_exact(a_sub, b_sub, target_alpha)
      except ValueError:
        n_infeasible += 1
        continue
      scores = scorer_scores(dataset, metametric, root=root, w=r.w, systems=subset)
      ess_list.append(float(r.ess))
    n_used += 1
    for s in scorers:
      rows[s].append(scores.get(s, float('nan')))

    if progress_every and (di % progress_every == 0 or di == len(draws)):
      import sys
      print(f'  [{di}/{len(draws)} draws] target_alpha={target_alpha}', file=sys.stderr, flush=True)

  kept = [s for s in scorers if n_used > 0 and not any(v != v for v in rows[s])]
  dropped = [s for s in scorers if s not in set(kept)]
  mean_ess = float(np.mean(ess_list)) if ess_list else float('nan')

  if len(kept) < 2 or n_used < 2:
    icc = float('nan')
  else:
    X = np.array([rows[s] for s in kept], dtype=float)
    icc = icc_c1(X)

  return {
      'dataset': dataset, 'metametric': metametric, 'K': K, 'K_prime': K_prime,
      'B': B, 'target_alpha': target_alpha, 'icc': icc, 'mean_ess': mean_ess,
      'ess_list': ess_list, 'draws': draws,
      'n_scorers_used': len(kept), 'n_scorers_dropped': len(dropped),
      'scorers_dropped': dropped,
      'n_draws_used': n_used, 'n_infeasible': n_infeasible,
      'outliers_dropped': outliers,
  }


def _score_one_draw(job: tuple) -> tuple[dict | None, float | None]:
  """Top-level (module-level, so it's picklable for ProcessPoolExecutor)
  worker: scores ONE draw under ONE condition. job = (dataset,
  metametric, root, subset, target_alpha) -- target_alpha=None is the
  natural condition (ESS is exactly len(subset) by definition, no solve
  needed); a float reweights to that target via solve_w_exact first.
  Returns (scores_dict, ess), or (None, None) if target_alpha was given
  but infeasible for this subset (Corollary 2) -- the caller drops it."""
  dataset, metametric, root, subset, target_alpha = job
  if target_alpha is None:
    scores = scorer_scores(dataset, metametric, root=root, systems=subset)
    return scores.to_dict(), float(len(subset))
  df = load_system_scores(dataset, root=root)
  a_sub, b_sub = df.loc[subset, 'a'].values, df.loc[subset, 'b'].values
  try:
    r = solve_w_exact(a_sub, b_sub, target_alpha)
  except ValueError:
    return None, None
  scores = scorer_scores(dataset, metametric, root=root, w=r.w, systems=subset)
  return scores.to_dict(), float(r.ess)


def _ess_one_draw(job: tuple) -> float | None:
  """Cheap top-level (picklable) worker: ESS ONLY for one draw at one
  target_alpha -- deliberately skips scorer_scores/SPA entirely (unlike
  _score_one_draw), since ESS only depends on solve_w_exact, not on the
  metametric. Use this (via ess_sweep_parallel below) whenever only the
  ESS/reachability of a target is needed, e.g. to build an ESS-based
  filter before running the (much more expensive) actual scoring pass --
  solve_w_exact is a closed-form/certified solver with no segment-level
  I/O, so this is dramatically cheaper per draw than _score_one_draw.
  job = (dataset, root, subset, target_alpha). Returns None if infeasible
  (Corollary 2: target_alpha outside this subset's reachable range)."""
  dataset, root, subset, target_alpha = job
  df = load_system_scores(dataset, root=root)
  a_sub, b_sub = df.loc[subset, 'a'].values, df.loc[subset, 'b'].values
  try:
    r = solve_w_exact(a_sub, b_sub, target_alpha)
  except ValueError:
    return None
  return float(r.ess)


def ess_sweep_parallel(
    dataset: str,
    draws: list[tuple[int, ...]],
    full_systems: list[str],
    target_alpha: float,
    root: str = '.',
    max_workers: int | None = None,
) -> list[float | None]:
  """ESS for every draw in `draws` at a single target_alpha, dispatched to
  a process pool via the cheap _ess_one_draw worker (no SPA/metametric
  scoring at all) -- the fast way to build an ESS-based filter over many
  draws before committing to the expensive full scoring pass
  (paired_stability_icc). Returns a list aligned 1:1 with `draws`; an
  entry is None where target_alpha was infeasible for that draw."""
  jobs = [(dataset, root, [full_systems[k] for k in idx], target_alpha) for idx in draws]
  ess_list: list[float | None] = [None] * len(draws)
  with cf.ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
    futures = {ex.submit(_ess_one_draw, job): i for i, job in enumerate(jobs)}
    for fut in cf.as_completed(futures):
      ess_list[futures[fut]] = fut.result()
  return ess_list


def paired_stability_icc(
    dataset: str,
    metametric: str,
    draws: list[tuple[int, ...]],
    full_systems: list[str],
    target_alpha: float,
    root: str = '.',
    max_workers: int | None = None,
) -> dict:
  """ICC(C,1) for the natural AND reweighted (target_alpha) conditions
  over an EXPLICIT `draws` list (index-tuples into full_systems) -- e.g. a
  filtered subset of a larger draw_subsets() call, not necessarily every
  draw -- with all 2*len(draws) per-draw scoring jobs (both conditions,
  every draw) dispatched to a SINGLE shared process pool at once: this is
  what makes the two conditions run in parallel with each other (not just
  each draw within a condition), since there's no separate pool per
  condition to serialize against.

  icc_c1 itself (lib.icc) is already fully vectorized (numpy sums, no
  Python-level loops over objects/raters) -- the actual bottleneck this
  parallelizes away is scorer_scores' per-draw SPA computation, which is
  irreducibly per-subset (data-dependent I/O + stats), not vectorizable
  across draws."""
  jobs = []  # (condition, draw_index, job_tuple)
  for di, idx in enumerate(draws):
    subset = [full_systems[k] for k in idx]
    jobs.append(('natural', di, (dataset, metametric, root, subset, None)))
    jobs.append(('reweighted', di, (dataset, metametric, root, subset, target_alpha)))

  scores_by = {'natural': [None] * len(draws), 'reweighted': [None] * len(draws)}
  ess_by = {'natural': [None] * len(draws), 'reweighted': [None] * len(draws)}
  with cf.ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
    future_to_job = {ex.submit(_score_one_draw, job): (cond, di) for cond, di, job in jobs}
    for fut in cf.as_completed(future_to_job):
      cond, di = future_to_job[fut]
      scores_dict, ess = fut.result()
      scores_by[cond][di] = scores_dict
      ess_by[cond][di] = ess

  natural_full = scorer_scores(dataset, metametric, root=root, systems=full_systems)
  scorers = sorted(natural_full.index)

  def build_matrix(cond):
    valid_di = [di for di in range(len(draws)) if scores_by[cond][di] is not None]
    rows = {s: [scores_by[cond][di].get(s, float('nan')) for di in valid_di] for s in scorers}
    kept = [s for s in scorers if not any(v != v for v in rows[s])]
    dropped = [s for s in scorers if s not in set(kept)]
    if len(kept) < 2 or len(valid_di) < 2:
      return float('nan'), len(kept), dropped, len(valid_di)
    X = np.array([rows[s] for s in kept], dtype=float)
    return icc_c1(X), len(kept), dropped, len(valid_di)

  icc_nat, n_nat, drop_nat, used_nat = build_matrix('natural')
  icc_rw, n_rw, drop_rw, used_rw = build_matrix('reweighted')
  ess_valid = [e for e in ess_by['reweighted'] if e is not None]
  mean_ess_rw = float(np.mean(ess_valid)) if ess_valid else float('nan')

  return {
      'dataset': dataset, 'metametric': metametric, 'target_alpha': target_alpha,
      'n_draws': len(draws),
      'icc_natural': icc_nat, 'icc_reweighted': icc_rw, 'improvement': icc_rw - icc_nat,
      'n_scorers_natural': n_nat, 'n_scorers_reweighted': n_rw,
      'scorers_dropped_natural': drop_nat, 'scorers_dropped_reweighted': drop_rw,
      'n_draws_used_natural': used_nat, 'n_draws_used_reweighted': used_rw,
      'mean_ess_reweighted': mean_ess_rw,
  }


def leave_p_out_generalizability(
    dataset: str,
    metametric: str,
    p: int = 1,
    root: str = '.',
    exclude_outliers: bool = True,
    systems: list[str] | None = None,
    max_workers: int | None = None,
    transpose: bool = False,
    target_alpha: float | None = None,
) -> dict:
  """ICC(C,1) across all C(K,p) exhaustive leave-p-out (LpO) subsets of a
  dataset's pool D (|D|=K): every size-p combination of systems dropped
  from D gives one subset D minus that combo, each scored ONCE and
  treated as one RATER; scorers are the objects. p=1 (the default) is
  plain leave-one-out (LOO): C(K,1)=K raters, one per excluded system.
  Every subset is used whenever feasible (see target_alpha below) -- this
  is generalizability_stability's design specialized to the single
  fully-exhaustive case K_prime=K-p, B=C(K,p) (every size-(K-p) subset of
  a size-K pool IS a leave-p-out subset, one per dropped combo), but
  implemented directly -- explicit combo exclusion (itertools.combinations,
  not draw_subsets' rejection sampler) via the same picklable
  _score_one_draw worker, dispatched to a shared process pool -- so raters
  are labeled by which combo they exclude rather than an opaque draw
  index, and there's no sampling variance to worry about. C(K,p) grows
  fast with p (e.g. C(16,2)=120 vs C(16,1)=16) -- still cheap at this
  project's K~11-17 for p=1 or 2, but a caller sweeping larger p should
  mind the cost.

  target_alpha (default None = natural/uniform weight): None scores every
  LpO subset at uniform weight, same as scorer_scores' default and never
  infeasible. A float instead reweights EVERY LpO subset to that SAME
  fixed target (lib.reweight_exact.solve_w_exact via _score_one_draw, same
  "one shared target across every rater" convention as
  generalizability_stability's own reweighted condition) -- typically a
  property of the FULL, un-LpO'd pool D (e.g. alpha_0(D) or the median of
  D's own pairwise alpha_ij's, lib.alpha.pairwise_alphas), passed in by
  the caller rather than recomputed here, since a dropped-combo subset has
  no privileged natural target of its own in this design (contrast
  type-8's upsampling, which targets alpha_0(D') of the SMALLER pool). An
  LpO subset for which target_alpha falls outside its own reachable
  [alpha_min, alpha_max] range (Corollary 2) is INFEASIBLE and dropped
  from the rater set for this call only -- tracked in
  `n_infeasible`/`n_raters_used`, same convention as pairwise_stability/
  generalizability_stability's own infeasible-draw handling. Needs >= 2
  raters to survive for icc_c1; fewer and `icc` is NaN.

  Needs K - p >= 2 (every LpO subset still has >= 2 systems to score) and
  C(K,p) >= 2 (>= 2 raters for icc_c1) -- raises ValueError below either
  bound, mirroring generalizability_stability's own K_prime bound check.

  Objects (the ICC matrix's rows, before any `transpose`) are restricted
  to lib.consistency.real_scorers(dataset, systems=full_systems,
  require_seg_scores=...) -- the project's canonical scorer screen (drops
  constant-score sentinels and deduplicates byte-identical WMT
  submissions), evaluated once on the FULL pool D, not re-screened per LpO
  subset -- so the scorer universe is fixed across all raters and only
  actual coverage gaps (see next paragraph) can still drop one.
  require_seg_scores is tied to whether `metametric` is in
  NEEDS_SEGMENT_SCORES (currently just spa), matching scorer_scores' own
  internal segment-coverage requirement for those metametrics.

  A real_scorers-screened scorer missing a valid score on ANY LpO subset
  used is ALSO dropped (icc_c1 needs a fully-crossed, no-missing-cells
  matrix), reported separately in `scorers_dropped` -- same convention as
  generalizability_stability.

  transpose (default False, MATCHES the caller's original request -- "each
  LpO meta-evaluation is a RATER" -- and is the orientation you almost
  certainly want): False keeps scorers as objects, the C(K,p) LpO subsets
  as raters. True swaps the matrix (LpO subsets become the objects,
  screened scorers become the raters) -- a DIFFERENT, NOT
  transpose-invariant question (icc_c1 excludes rater main effects, so
  scorers' hugely-varying absolute score scale is correctly ignored as
  raters but pollutes between-object variance as objects). Exists purely
  as a diagnostic to confirm orientation sensitivity, not as an
  alternative statistic to report as a headline result -- per project
  convention as of this session, DO NOT run transpose=True again (already
  used once to confirm an axis-swap bug elsewhere)."""
  full_systems, outliers, combos, scores_by_combo, ess_by_combo, used_combos, n_infeasible = (
      _score_lpo_combos(dataset, metametric, p, root, exclude_outliers, systems,
                         max_workers, target_alpha, 'leave_p_out_generalizability'))
  K = len(full_systems)

  natural_full = scorer_scores(dataset, metametric, root=root, systems=full_systems)
  canonical_scorers = real_scorers(dataset, root=root, systems=full_systems,
                                    require_seg_scores=(metametric in NEEDS_SEGMENT_SCORES))
  scorers = sorted(set(canonical_scorers) & set(natural_full.index))

  rows = {s: [scores_by_combo[combo].get(s, float('nan')) for combo in used_combos]
          for s in scorers}
  kept = [s for s in scorers if not any(v != v for v in rows[s])]
  dropped = [s for s in scorers if s not in set(kept)]
  mean_ess = float(np.mean(list(ess_by_combo.values()))) if ess_by_combo else float('nan')

  if len(kept) < 2 or len(used_combos) < 2:
    icc = float('nan')
  else:
    X = np.array([rows[s] for s in kept], dtype=float)  # [n_scorers, n_raters_used]
    icc = icc_c1(X.T if transpose else X)

  return {
      'dataset': dataset, 'metametric': metametric, 'K': K, 'p': p,
      'target_alpha': target_alpha, 'dropped_combos': combos, 'used_combos': used_combos,
      'icc': icc, 'transpose': transpose, 'mean_ess': mean_ess,
      'n_scorers_used': len(kept), 'n_scorers_dropped': len(dropped),
      'scorers_dropped': dropped, 'outliers_dropped': outliers,
      'n_raters_total': len(combos), 'n_raters_used': len(used_combos),
      'n_infeasible': n_infeasible,
  }


def _score_lpo_combos(
    dataset: str, metametric: str, p: int, root: str, exclude_outliers: bool,
    systems: list[str] | None, max_workers: int | None, target_alpha: float | None,
    caller_name: str,
) -> tuple[list[str], list[str], list[tuple], dict[tuple, dict], dict[tuple, float],
           list[tuple], int]:
  """Shared dispatch step of leave_p_out_generalizability and leave_p_out_tau:
  builds the C(K,p) dropped-combo jobs, scores each once (parallel, via
  _score_one_draw) and returns everything both callers need to build their
  own statistic (an ICC matrix vs. pool_weighted_tau's rankings dict) --
  see leave_p_out_generalizability's docstring for the full contract
  (target_alpha semantics, infeasibility handling, etc.), which applies
  identically here. Returns (full_systems, outliers, combos, scores_by_combo,
  ess_by_combo, used_combos, n_infeasible)."""
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
           target_alpha) for combo in combos]

  scores_by_combo: dict[tuple, dict] = {}
  ess_by_combo: dict[tuple, float] = {}
  with cf.ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
    future_to_combo = {ex.submit(_score_one_draw, job): combo
                        for job, combo in zip(jobs, combos)}
    for fut in cf.as_completed(future_to_combo):
      combo = future_to_combo[fut]
      scores_dict, ess = fut.result()
      if scores_dict is not None:  # None only when target_alpha was infeasible for this combo
        scores_by_combo[combo] = scores_dict
        ess_by_combo[combo] = ess
  if target_alpha is None:
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
    target_alpha: float | None = None,
) -> dict:
  """Kendall's tau-a analogue of leave_p_out_generalizability: same C(K,p)
  exhaustive leave-p-out (LpO) raters (each scored once, natural or
  reweighted to a shared fixed target_alpha -- identical semantics and
  infeasibility handling to leave_p_out_generalizability, via the same
  shared dispatch helper, _score_lpo_combos), but pooled via
  lib.consistency.pool_weighted_tau instead of icc_c1's two-way ANOVA --
  the SAME pooling convention this project already uses for cross-dataset
  scorer-ranking consistency (action_plan.md 6.5): every pair of raters'
  scorer rankings is compared, concordant/discordant counts (tau-a
  convention: ties drop out of the numerator but still count in the
  denominator) are summed across ALL C(n_raters_used,2) rater-pairs x
  scorer-pairs, and tau is computed ONCE from those pooled totals -- not a
  plain average of per-pair taus.

  Unlike icc_c1, pool_weighted_tau does NOT need a fully-crossed matrix:
  each rater-pair only needs the scorers THAT PAIR shares, so a scorer
  missing from one rater's LpO subset still contributes to every other
  pair it IS present in, rather than being dropped globally the way
  leave_p_out_generalizability's `scorers_dropped` works. Objects are
  still restricted up front to lib.consistency.real_scorers(dataset,
  systems=full_systems, require_seg_scores=...), the project's canonical
  scorer screen -- same as leave_p_out_generalizability."""
  full_systems, outliers, combos, scores_by_combo, ess_by_combo, used_combos, n_infeasible = (
      _score_lpo_combos(dataset, metametric, p, root, exclude_outliers, systems,
                         max_workers, target_alpha, 'leave_p_out_tau'))
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
      'target_alpha': target_alpha, 'tau': tau, 'total_weight': total_weight,
      'mean_ess': mean_ess, 'n_scorers_canonical': len(canonical_scorers),
      'outliers_dropped': outliers, 'n_raters_total': len(combos),
      'n_raters_used': len(used_combos), 'n_infeasible': n_infeasible,
  }


def loo_generalizability(
    dataset: str,
    metametric: str,
    root: str = '.',
    exclude_outliers: bool = True,
    systems: list[str] | None = None,
    max_workers: int | None = None,
    transpose: bool = False,
    target_alpha: float | None = None,
) -> dict:
  """Plain leave-ONE-out: leave_p_out_generalizability(..., p=1). Kept as a
  distinct name since p=1 is this project's primary/most-used case (K
  raters, one per excluded system) -- see that function's docstring for
  the full generalized (leave-p-out) contract, which this just forwards
  to."""
  return leave_p_out_generalizability(
      dataset, metametric, p=1, root=root, exclude_outliers=exclude_outliers,
      systems=systems, max_workers=max_workers, transpose=transpose, target_alpha=target_alpha)
