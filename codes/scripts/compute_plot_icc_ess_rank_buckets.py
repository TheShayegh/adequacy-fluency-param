"""Rank-bucketed ICC(C,1) vs ESS: sorts every size-K' subsample D' of D by
its ESS(reweighted to target_alpha=alpha_0(D)) -- lib.subsampling_stability.
ess_sweep_parallel -- ascending (lowest-ESS first), splits the sorted list
into consecutive groups of --group-b (default 30, last group whatever
remains), and computes natural/reweighted ICC(C,1) WITHIN each group
(same paired, cached-score machinery as the ESS-band heatmap: every draw
is scored once, reused across every group's matrix).

Unlike a continuous curve, groups are RANK buckets, not a smooth function
of ESS -- consecutive groups can span very different ESS ranges (the
population isn't uniformly distributed in ESS, see the earlier alpha0-
target-ess-sweep gap finding), so each group is drawn as its own
disconnected horizontal segment (spanning that group's [ess_min, ess_max])
plus a scatter marker at its mean ESS, rather than one continuous line
connecting every group's point together.

Usage: python codes/scripts/compute_plot_icc_ess_rank_buckets.py
           --dataset ende22 [--k-prime K-3] [--group-b 30]
           [--metametric spa] [--seed 0]
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
from lib.subsampling_stability import _score_one_draw, draw_subsets, ess_sweep_parallel
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

STANDARD_BLUE = (0.121, 0.466, 0.705)
ORANGE = (0.90, 0.45, 0.10)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, default=None, help='default: K-3')
  p.add_argument('--group-b', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  K_prime = args.k_prime if args.k_prime is not None else K - 3
  tag = args.tag or f'{base}_kprime{K_prime}_groupb{args.group_b}'
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full, 'a'].values, df.loc[full, 'b'].values
  target_alpha = alpha0_of(a_D, b_D)
  B_full = math.comb(K, K_prime)
  draws = draw_subsets(K, K_prime, B_full, args.seed)
  print(f'D={base}: K={K}, target_alpha={target_alpha:.4f}, K\'={K_prime}, B_full={B_full}, '
        f'group_b={args.group_b}', file=sys.stderr)

  ess_list = ess_sweep_parallel(base, draws, full, target_alpha, root=ROOT)
  order = sorted(range(len(draws)), key=lambda i: (ess_list[i] is None, ess_list[i]))
  print(f'ESS range: [{min(e for e in ess_list if e is not None):.4f}, '
        f'{max(e for e in ess_list if e is not None):.4f}]', file=sys.stderr)

  groups = [order[i:i + args.group_b] for i in range(0, len(order), args.group_b)]
  print(f'{len(groups)} groups (sizes: {[len(g) for g in groups]})', file=sys.stderr)

  t0 = time.time()
  jobs = []
  for di, idx in enumerate(draws):
    subset = [full[k] for k in idx]
    jobs.append(('natural', di, (base, m, ROOT, subset, None)))
    jobs.append(('reweighted', di, (base, m, ROOT, subset, target_alpha)))
  scores_by = {'natural': [None] * len(draws), 'reweighted': [None] * len(draws)}
  with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    futures = {ex.submit(_score_one_draw, job): (cond, di) for cond, di, job in jobs}
    for fut in cf.as_completed(futures):
      cond, di = futures[fut]
      sd, _ess = fut.result()
      scores_by[cond][di] = sd
  print(f'scoring pass ({len(jobs)} jobs): {time.time() - t0:.1f}s', file=sys.stderr)

  natural_full = scorer_scores(base, m, root=ROOT, systems=full)
  scorers = sorted(natural_full.index)

  def icc_for(cond, draw_indices):
    dicts = [scores_by[cond][di] for di in draw_indices]
    kept = [s for s in scorers if all(sd is not None and s in sd for sd in dicts)]
    if len(kept) < 2 or len(dicts) < 2:
      return float('nan')
    X = np.array([[sd[s] for sd in dicts] for s in kept], dtype=float)
    return icc_c1(X)

  group_stats = []
  for gi, group in enumerate(groups):
    ess_vals = [ess_list[di] for di in group if ess_list[di] is not None]
    if not ess_vals:
      continue
    icc_nat = icc_for('natural', group)
    icc_rw = icc_for('reweighted', group)
    group_stats.append(dict(
        gi=gi, ess_min=min(ess_vals), ess_max=max(ess_vals), ess_mean=float(np.mean(ess_vals)),
        icc_nat=icc_nat, icc_rw=icc_rw, n=len(group)))
    print(f'  group {gi + 1}: n={len(group)}, ESS=[{min(ess_vals):.3f}, {max(ess_vals):.3f}], '
          f'ICC natural={icc_nat:.4f}, reweighted={icc_rw:.4f}', file=sys.stderr)

  fig, ax = plt.subplots(figsize=(9, 6.5))
  for cond_key, color, label in [('icc_nat', STANDARD_BLUE, 'natural ICC(C,1)'),
                                  ('icc_rw', ORANGE, 'reweighted ICC(C,1)')]:
    xs = [g['ess_mean'] for g in group_stats if g[cond_key] == g[cond_key]]
    ys = [g[cond_key] for g in group_stats if g[cond_key] == g[cond_key]]
    # A real connecting line across all groups, but each vertex gets a
    # white halo (larger, drawn on top) so the line visually breaks right
    # at each dot instead of running continuously through it -- "connected
    # but not joined."
    ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3, label=label)
    ax.scatter(xs, ys, color='white', edgecolor='white', s=90, zorder=4)
    ax.scatter(xs, ys, color=color, edgecolor='black', linewidths=0.5, s=35, zorder=5)

  # Anchored at K' instead of at 1 (the opposite of the earlier version):
  # forward(x) = -log(K' - x) has a vertical asymptote AT x=K', so ESS
  # values close to K' get stretched apart dramatically while everything
  # further below K' gets compressed together near the low end. Still
  # monotonically increasing in x (K'-x -> 0+ as x -> K'-, so -log(K'-x)
  # -> +inf), so left-to-right still reads as increasing ESS.
  def _forward(x):
    return -np.log(np.maximum(K_prime - x, 1e-6))

  def _inverse(y):
    return K_prime - np.exp(-y)

  ax.set_xscale('function', functions=(_forward, _inverse))
  ax.set_xlabel("ESS (reweighted to alpha_0(D)) -- rank-bucketed, ascending -- "
                 "-log(K'-ESS) scale")
  ax.set_ylabel('ICC(C,1)')
  ax.set_title(f"{base}: ICC by ESS rank-bucket (group size={args.group_b})\n"
               f"K'={K_prime}, B_full={B_full}, target_alpha=alpha_0(D)={target_alpha:.4f}",
               fontsize=11)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(loc='best', frameon=False, fontsize=9)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'icc_ess_rank_buckets_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
