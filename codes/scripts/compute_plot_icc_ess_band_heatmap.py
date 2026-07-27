"""Heatmap: for ende22, K'=10, target_alpha=alpha_0(D), B=max=C(K,K')
(exhaustive), sweeps EVERY (low_ess, high_ess) threshold BAND -- keep only
draws with low_ess <= ESS(reweighted) <= high_ess -- and computes
ICC(reweighted) - ICC(natural) over just those draws, the same paired,
same-seed comparison used throughout (lib.subsampling_stability.
paired_stability_icc's logic), just re-parameterized as a 2-D band instead
of a single lower threshold.

x-axis: low_ess (band floor). y-axis: high_ess (band ceiling). Only
low_ess <= high_ess is defined (upper triangle); the rest is left blank.

Efficiency: target_alpha is FIXED across every cell (unlike the earlier
alpha-sweep plot), so natural and reweighted SPA scores depend only on
which of the 286 draws is used, NOT on the band -- computed ONCE per draw
(572 scoring jobs total, one shared parallel pass) and reused/refiltered
for every cell. Only icc_c1 (cheap, vectorized) reruns per cell.

Color: ICC(reweighted) - ICC(natural), a 2-stop RED->GREEN colormap that
does NOT pass through white at zero (unlike a standard diverging colormap
such as RdYlGn) -- a straight linear RGB blend, so 0 renders as a muted
brown/olive, not a light/washed-out center.

Alpha (transparency) per cell: 0 (fully transparent) at B(band)=1,
1 (fully opaque) at B(band)=C(K,K')=286, linear in between. B(band)<=1
(including 0) is also fully transparent (ICC is undefined below 2 raters).

Usage: python codes/scripts/compute_plot_icc_ess_band_heatmap.py
           --dataset ende22 --k-prime 10 [--metametric spa]
           [--ess-step 0.25] [--seed 0]
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.icc import icc_c1
from lib.subsampling_stability import _score_one_draw, draw_subsets, ess_sweep_parallel
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

DARK_RED = (0.35, 0.0, 0.0)
RED = (0.90, 0.15, 0.15)
GREEN = (0.20, 0.75, 0.25)
DARK_GREEN = (0.0, 0.30, 0.05)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, default=None, help='default: K-3')
  p.add_argument('--ess-step', type=float, default=0.25)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  K_prime = args.k_prime if args.k_prime is not None else K - 3
  tag = args.tag or f'{base}_kprime{K_prime}'
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full, 'a'].values, df.loc[full, 'b'].values
  target_alpha = alpha0_of(a_D, b_D)
  B_full = math.comb(K, K_prime)
  draws = draw_subsets(K, K_prime, B_full, args.seed)
  print(f'D={base}: K={K}, target_alpha={target_alpha:.4f}, K\'={K_prime}, B_full={B_full}',
        file=sys.stderr)

  ess_list = ess_sweep_parallel(base, draws, full, target_alpha, root=ROOT)
  ess_arr = np.array([e if e is not None else np.nan for e in ess_list])
  valid_ess = ess_arr[~np.isnan(ess_arr)]
  print(f'ESS range: [{valid_ess.min():.4f}, {valid_ess.max():.4f}]', file=sys.stderr)

  # Combined natural + reweighted scoring pass, ONCE per draw -- reused
  # for every (low, high) band since target_alpha is fixed.
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

  # Grid: low_ess (x) and high_ess (y), snapped to a --ess-step grid
  # covering the observed ESS range with a small margin.
  lo_bound = math.floor(valid_ess.min() / args.ess_step) * args.ess_step
  hi_bound = math.ceil(valid_ess.max() / args.ess_step) * args.ess_step
  edges = np.round(np.arange(lo_bound, hi_bound + args.ess_step / 2, args.ess_step), 6)
  n = len(edges)
  print(f'grid: {n} x {n} ({edges[0]:.3f} to {edges[-1]:.3f}, step {args.ess_step})', file=sys.stderr)

  diff_grid = np.full((n, n), np.nan)  # [high_idx, low_idx] for imshow origin='lower'
  b_grid = np.zeros((n, n), dtype=int)

  for li, low in enumerate(edges):
    for hi_i in range(li, n):
      high = edges[hi_i]
      mask = (ess_arr >= low) & (ess_arr <= high)
      draw_indices = list(np.nonzero(mask)[0])
      b_cell = len(draw_indices)
      b_grid[hi_i, li] = b_cell
      if b_cell < 2:
        continue
      icc_nat = icc_for('natural', draw_indices)
      icc_rw = icc_for('reweighted', draw_indices)
      if icc_nat == icc_nat and icc_rw == icc_rw:
        diff_grid[hi_i, li] = icc_rw - icc_nat

  print(f'cells computed: {np.sum(~np.isnan(diff_grid))} / {n * (n + 1) // 2} valid (low<=high)',
        file=sys.stderr)

  # ---- Colors: two independent branches, hard jump at 0, no white ----
  # [-1, 0) normalized -> dark-red -> red (darkest at the most negative end,
  # brightening toward 0); [0, 1] normalized -> green -> dark-green
  # (brightest at 0, darkening toward the most positive end). Built as one
  # 256-step ListedColormap (128 dark_red->red, 128 green->dark_green) so
  # it plugs into the standard Normalize/colorbar machinery, but the jump
  # from red to green happens in a single step right at the center.
  neg_half = np.linspace(0, 1, 128)[:, None] * (np.array(RED) - np.array(DARK_RED)) + np.array(DARK_RED)
  pos_half = np.linspace(0, 1, 128)[:, None] * (np.array(DARK_GREEN) - np.array(GREEN)) + np.array(GREEN)
  stops = np.vstack([neg_half, pos_half])
  cmap = ListedColormap(stops, name='red_green_hardstop')
  finite = diff_grid[~np.isnan(diff_grid)]
  max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
  norm = Normalize(vmin=-max_abs, vmax=max_abs)

  rgba = np.zeros((n, n, 4))
  for hi_i in range(n):
    for li in range(n):
      d = diff_grid[hi_i, li]
      b_cell = b_grid[hi_i, li]
      if li > hi_i:  # low > high -- undefined, leave fully transparent
        continue
      # Nonlinear visibility boost: b in [0,1] anchored exactly as before
      # (0 at B=1, 1 at B=B_full), then alpha=2b-b^2 (concave -- matches b
      # at the endpoints, exceeds it in between) so mid-B cells show up
      # more clearly without changing the fully-transparent/fully-opaque
      # anchors.
      b_lin = 0.0 if B_full <= 1 else float(np.clip((b_cell - 1) / (B_full - 1), 0.0, 1.0))
      alpha = 2 * b_lin - b_lin ** 2
      if d != d:  # NaN (B<2) -- fully transparent regardless of the formula above
        alpha = 0.0
        rgb = (1, 1, 1)
      else:
        rgb = cmap(norm(d))[:3]
      rgba[hi_i, li] = (*rgb, alpha)

  fig, ax = plt.subplots(figsize=(8.5, 7.5))
  ax.imshow(rgba, origin='lower', extent=[edges[0], edges[-1], edges[0], edges[-1]],
            aspect='equal', interpolation='nearest')
  ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color='black', linewidth=0.6, alpha=0.3)
  ax.set_xlabel('low ESS (band floor)')
  ax.set_ylabel('high ESS (band ceiling)')
  ax.set_title(f"{base}: ICC(reweighted) - ICC(natural) by ESS band\n"
               f"K'={K_prime}, B_full={B_full} (exhaustive), target_alpha=alpha_0(D)={target_alpha:.4f}\n"
               "color = diff (red<0, green>0); alpha = 2b-b^2, b=B(band)/B_full", fontsize=10)

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03)
  cbar.set_label('ICC(reweighted) - ICC(natural)')

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'icc_ess_band_heatmap_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)

  # Cache the raw grid (not just the rendered PNG) so follow-up questions
  # about specific cells don't require a full rescoring pass.
  npz_path = os.path.join(ARTIFACTS_DIR, f'icc_ess_band_heatmap_{tag}.npz')
  np.savez(npz_path, edges=edges, diff_grid=diff_grid, b_grid=b_grid,
           target_alpha=target_alpha, K=K, K_prime=K_prime, B_full=B_full)
  print(f'Wrote {npz_path}', file=sys.stderr)
