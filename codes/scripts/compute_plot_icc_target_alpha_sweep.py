"""Recreates alpha0_target_ess_sweep_ende22_kprime8_b20.png's sweep style
(target_alpha over every alpha_0(D') of the C(K,K') size-K' subsets), but
with our actual consistency metric -- ICC(C,1) -- as the y-axis instead of
ESS, and restricted to the "targeted filtered" design from recent turns:
at each candidate target_alpha, keep only the draws with ESS/K' >
--threshold (default 0.99), and compute natural/reweighted ICC(C,1) over
just those.

Unlike the ESS sweep, there's no ESS coloring here -- by construction
every draw actually used has ESS/K' > threshold, so color would be nearly
uniform and uninformative.

Unlike a plain reweighting-improvement plot, natural ICC is NOT a flat
baseline here: which draws pass the ESS filter (and how many, B(alpha))
changes with alpha, so BOTH curves -- natural(alpha) and reweighted(alpha)
-- are computed over the SAME alpha-dependent filtered draw set and both
vary with alpha.

The x-axis is restricted to the range where B(alpha) >= --min-b (default
30) -- outside that range there aren't enough surviving draws for a
meaningful ICC. Two-stage compute for tractability:
  Stage 1 (cheap): ESS(candidate, draw) for the full candidate x draw
    grid, via solve_w_exact only (no SPA) -- determines B(alpha) and the
    valid range essentially for free.
  Stage 2 (expensive): natural SPA scores computed ONCE per draw (reused
    across every candidate, since they don't depend on target_alpha);
    reweighted SPA scores computed per (candidate, its filtered draws)
    only for a downsampled set of candidates within the valid range
    (--max-points, default 40 -- plenty for a smooth curve, and keeps
    stage 2's job count tractable instead of scoring at up to C(K,K')
    candidates).

Usage: python codes/scripts/compute_plot_icc_target_alpha_sweep.py
           --dataset ende22 --k-prime 10 [--threshold 0.99] [--min-b 30]
           [--max-points 40] [--metametric spa] [--seed 0]
           [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.icc import icc_c1
from lib.reweight_exact import solve_w_exact
from lib.subsampling_stability import _score_one_draw, draw_subsets
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

STANDARD_BLUE = (0.121, 0.466, 0.705)
ORANGE = (0.90, 0.45, 0.10)


def _ess_from_ab(job: tuple) -> float | None:
  """Pure-numpy ESS worker for the stage-1 grid -- takes already-sliced
  (a_sub, b_sub) arrays directly (no dataset/root/systems lookup, no disk
  I/O per job), since the candidate x draw grid is large (up to
  C(K,K')^2 jobs) and re-loading load_system_scores per job would dominate
  the cost otherwise."""
  a_sub, b_sub, target_alpha = job
  try:
    r = solve_w_exact(a_sub, b_sub, target_alpha)
  except ValueError:
    return None
  return float(r.ess)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, required=True)
  p.add_argument('--threshold', type=float, default=0.99, help="ESS/K' filter threshold")
  p.add_argument('--min-b', type=int, default=30, help='minimum surviving draws to plot a point')
  p.add_argument('--max-points', type=int, default=40, help='candidates to actually score (stage 2)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  K_prime = args.k_prime
  tag = args.tag or f'{base}_kprime{K_prime}'

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full, 'a'].values, df.loc[full, 'b'].values
  a0_D = alpha0_of(a_D, b_D)
  B_full = math.comb(K, K_prime)
  draws = draw_subsets(K, K_prime, B_full, args.seed)
  print(f'D={base}: K={K}, alpha_0(D)={a0_D:.4f}, K\'={K_prime}, B_full={B_full} (exhaustive)',
        file=sys.stderr)

  # Candidates: alpha_0(D') of every one of the B_full subsets (same
  # population as alpha0_target_ess_sweep's sweep) -- here this population
  # IS `draws` itself, so no separate enumeration is needed.
  draw_ab = []
  draw_a0 = []
  for idx in draws:
    subset = [full[k] for k in idx]
    a_sub, b_sub = df.loc[subset, 'a'].values, df.loc[subset, 'b'].values
    draw_ab.append((a_sub, b_sub))
    draw_a0.append(alpha0_of(a_sub, b_sub))
  candidates = sorted(set(draw_a0))
  print(f'{len(candidates)} distinct candidate target_alphas', file=sys.stderr)

  # ---- Stage 1: cheap ESS(candidate, draw) grid, pure numpy, no SPA ----
  t0 = time.time()
  jobs = [(a_sub, b_sub, cand) for cand in candidates for (a_sub, b_sub) in draw_ab]
  ess_grid = np.full((len(candidates), len(draws)), np.nan)
  with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {}
    for ci, cand in enumerate(candidates):
      for di, (a_sub, b_sub) in enumerate(draw_ab):
        fut = ex.submit(_ess_from_ab, (a_sub, b_sub, cand))
        futures[fut] = (ci, di)
    done = 0
    log_every = max(1, len(futures) // 20)
    for fut in cf.as_completed(futures):
      ci, di = futures[fut]
      ess = fut.result()
      if ess is not None:
        ess_grid[ci, di] = ess
      done += 1
      if done % log_every == 0:
        elapsed = time.time() - t0
        print(f'  stage1 [{done}/{len(futures)}] elapsed={elapsed:.1f}s', file=sys.stderr, flush=True)
  print(f'stage 1 (ESS grid, {len(futures)} jobs): {time.time() - t0:.1f}s', file=sys.stderr)

  ess_frac_grid = ess_grid / K_prime
  B_of_alpha = np.sum(ess_frac_grid > args.threshold, axis=1)
  valid_mask = B_of_alpha >= args.min_b
  n_valid = int(valid_mask.sum())
  print(f'candidates with B(alpha) >= {args.min_b} at threshold {args.threshold}: '
        f'{n_valid} / {len(candidates)}', file=sys.stderr)
  if n_valid == 0:
    raise SystemExit('No candidate alpha has enough surviving draws -- lower --min-b or --threshold.')

  valid_candidates = [candidates[ci] for ci in range(len(candidates)) if valid_mask[ci]]
  lo, hi = min(valid_candidates), max(valid_candidates)
  print(f'valid alpha range: [{lo:.4f}, {hi:.4f}]', file=sys.stderr)

  # Downselect to --max-points, evenly spaced BY ALPHA VALUE across the
  # valid range (not by rank), so stage 2 stays tractable regardless of
  # how many raw candidates fall in-range.
  if n_valid > args.max_points:
    target_xs = np.linspace(lo, hi, args.max_points)
    chosen = sorted(set(min(valid_candidates, key=lambda c: abs(c - x)) for x in target_xs))
  else:
    chosen = valid_candidates
  print(f'stage 2 will score {len(chosen)} candidates', file=sys.stderr)

  # ---- Stage 2: real SPA scoring ----
  # Natural scores: computed ONCE per draw (target-independent), reused
  # for every candidate's natural(alpha) curve point.
  t1 = time.time()
  natural_scores = [None] * len(draws)
  with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {ex.submit(_score_one_draw, (base, m, ROOT, [full[k] for k in draws[di]], None)): di
               for di in range(len(draws))}
    for fut in cf.as_completed(futures):
      di = futures[fut]
      sd, _ess = fut.result()
      natural_scores[di] = sd
  print(f'stage 2a (natural scores, {len(draws)} draws): {time.time() - t1:.1f}s', file=sys.stderr)

  # Reweighted scores: per (chosen candidate, its own ESS-filtered draws).
  t2 = time.time()
  cand_filtered_draws = {}
  rw_jobs = []
  for cand in chosen:
    ci = candidates.index(cand)
    filtered_di = [di for di in range(len(draws)) if ess_frac_grid[ci, di] > args.threshold]
    cand_filtered_draws[cand] = filtered_di
    for di in filtered_di:
      rw_jobs.append((cand, di))
  print(f'stage 2b: {len(rw_jobs)} reweighted-scoring jobs across {len(chosen)} candidates',
        file=sys.stderr)

  rw_scores = {}  # (cand, di) -> scores dict
  with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {}
    for cand, di in rw_jobs:
      subset = [full[k] for k in draws[di]]
      fut = ex.submit(_score_one_draw, (base, m, ROOT, subset, cand))
      futures[fut] = (cand, di)
    done = 0
    log_every = max(1, len(futures) // 20)
    for fut in cf.as_completed(futures):
      cand, di = futures[fut]
      sd, _ess = fut.result()
      rw_scores[(cand, di)] = sd
      done += 1
      if done % log_every == 0:
        print(f'  stage2b [{done}/{len(futures)}] elapsed={time.time() - t2:.1f}s',
              file=sys.stderr, flush=True)
  print(f'stage 2b (reweighted scores): {time.time() - t2:.1f}s', file=sys.stderr)

  natural_full = scorer_scores(base, m, root=ROOT, systems=full)
  scorers = sorted(natural_full.index)

  def matrix_from(score_dicts):
    kept = [s for s in scorers if all(sd is not None and s in sd for sd in score_dicts)]
    X = np.array([[sd[s] for sd in score_dicts] for s in kept], dtype=float)
    return X, len(kept)

  xs, icc_nat_ys, icc_rw_ys, b_ys = [], [], [], []
  for cand in chosen:
    filtered_di = cand_filtered_draws[cand]
    if len(filtered_di) < args.min_b:
      continue
    nat_dicts = [natural_scores[di] for di in filtered_di]
    rw_dicts = [rw_scores.get((cand, di)) for di in filtered_di]
    X_nat, n_nat = matrix_from(nat_dicts)
    X_rw, n_rw = matrix_from(rw_dicts)
    if n_nat < 2 or n_rw < 2:
      continue
    xs.append(cand)
    icc_nat_ys.append(icc_c1(X_nat))
    icc_rw_ys.append(icc_c1(X_rw))
    b_ys.append(len(filtered_di))

  xs = np.array(xs)
  icc_nat_ys = np.array(icc_nat_ys)
  icc_rw_ys = np.array(icc_rw_ys)
  print(f'plotted points: {len(xs)}', file=sys.stderr)

  fig, ax = plt.subplots(figsize=(8, 6.5))
  ax.plot(xs, icc_nat_ys, color=STANDARD_BLUE, linewidth=2.0, marker='o', markersize=3,
          label='natural ICC(C,1)')
  ax.plot(xs, icc_rw_ys, color=ORANGE, linewidth=2.0, marker='o', markersize=3,
          label='reweighted ICC(C,1)')
  ax.axvline(a0_D, color='black', alpha=0.35, linewidth=1.2, zorder=1)
  ax.annotate(r'$\alpha_0(D)$', xy=(a0_D, ax.get_ylim()[1]), xytext=(0, 2),
              textcoords='offset points', ha='center', va='bottom', fontsize=8, color='black')

  ax.set_xlim(xs.min(), xs.max())
  ax.set_xlabel(r"Candidate Target Balance $\alpha$  (= $\alpha_0$(D') of some size-K' subset)")
  ax.set_ylabel("ICC(C,1)")
  ax.set_title(f"{base}: consistency (ICC) vs. target " + r'$\alpha$' + "\n"
               f"filtered to draws with ESS/K' > {args.threshold} (K={K}, K'={K_prime}, "
               f"min B={args.min_b})", fontsize=11)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(loc='best', frameon=False, fontsize=9)

  ax2 = ax.twinx()
  ax2.plot(xs, b_ys, color='gray', linewidth=1.0, linestyle=':', alpha=0.6)
  ax2.set_ylabel('B(alpha)  (surviving draws, dotted gray)', color='gray', fontsize=8)
  ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
  ax2.spines['top'].set_visible(False)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'icc_target_alpha_sweep_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
