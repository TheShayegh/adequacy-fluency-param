"""Heatmap #2: traditional (non-overlapping) 2-D histogram bucketing,
unlike compute_plot_icc_ess_band_heatmap.py's low/high BAND design.

y-axis: ESS(reweighted to target_alpha=alpha_0(D)), log-spaced and
anchored at K' (thinnest right at K', widest far below it).
x-axis: abs(delta_alpha_0) = |alpha_0(D') - alpha_0(D)|, log-spaced and
anchored at 0 (thinnest near 0, widest far from it) -- same logic as the
y-axis, mirrored onto the axis's own natural "good" anchor.

Both axes are built via build_log_edges(...): raw geomspace edges, then
DEDUPED (drop edges that collapse to float-identical due to precision
limits near the anchor -- these were silently degenerate zero-width
buckets in the previous version) and each surviving real bucket BISECTED
once (double the real bucket count) rather than just requesting more raw
geomspace points, which would only create more degenerate buckets at the
precision floor instead of adding real resolution.

Each cell's ICC is computed from a SMOOTHED/pooled sample: not just the
draws whose own (ESS, |delta_alpha_0|) land in that exact cell, but the
union of draws from that cell AND its up to 8 immediate neighbors (3x3
neighborhood, edge/corner cells use however many exist) -- finer buckets
mean fewer draws per cell, so pooling neighbors keeps each cell's B large
enough for a meaningful ICC estimate. The plotted cell BOUNDARIES are
still each cell's own (non-pooled) bin, only the statistic inside is
smoothed.

Color/alpha conventions unchanged from the band heatmap: red<0/green>0
hard-stop at zero (no white); alpha=2b-b^2 with
b=(B_cell-1)/(--alpha-full-b - 1), B_cell now being the POOLED count.

Rendered as explicit Rectangle patches (not imshow) since both axes are
non-uniform width.

Usage: python codes/scripts/compute_plot_icc_ess_deltaalpha_heatmap.py
           --dataset zhen22 [--k-prime K-3] [--n-ess-bins 20]
           [--n-delta-bins 16] [--alpha-full-b 30] [--metametric spa]
           [--seed 0] [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
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
from matplotlib.patches import Rectangle

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


def build_log_edges(data: np.ndarray, anchor: float, n_bins: int, anchor_is_high: bool) -> np.ndarray:
  """Log-spaced bin edges in u = |data - anchor| (thin near anchor, wide
  far), deduped against float-precision collapse and then each real
  bucket bisected once (geometric midpoint in u-space) to double the
  count. anchor_is_high=True means anchor >= all data (e.g. K' for ESS);
  False means anchor <= all data (e.g. 0 for |delta_alpha_0|)."""
  u = (anchor - data) if anchor_is_high else (data - anchor)
  u_max = float(u.max())
  u_min_pos = float(u[u > 0].min()) if np.any(u > 0) else u_max * 1e-3
  raw_u = np.concatenate([[0.0], np.geomspace(u_min_pos, u_max * 1.001, n_bins)])
  raw_edges = np.sort((anchor - raw_u) if anchor_is_high else (anchor + raw_u))

  real = [raw_edges[0]]
  for e in raw_edges[1:]:
    if e > real[-1]:
      real.append(e)
  real = np.array(real)

  out = [real[0]]
  for i in range(len(real) - 1):
    lo, hi = real[i], real[i + 1]
    u_lo = (anchor - lo) if anchor_is_high else (lo - anchor)
    u_hi = (anchor - hi) if anchor_is_high else (hi - anchor)
    u_mid = math.sqrt(u_lo * u_hi) if u_lo > 0 and u_hi > 0 else (u_lo + u_hi) / 2
    mid = (anchor - u_mid) if anchor_is_high else (anchor + u_mid)
    if mid > out[-1]:
      out.append(mid)
    out.append(hi)
  return np.array(out)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='zhen22')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--k-prime', type=int, default=None, help='default: K-3')
  p.add_argument('--n-ess-bins', type=int, default=20,
                  help="raw ESS candidate bins before dedupe+double, log-anchored at K'")
  p.add_argument('--n-delta-bins', type=int, default=16,
                  help='raw |delta_alpha_0| candidate bins before dedupe+double, log-anchored at 0')
  p.add_argument('--alpha-full-b', type=int, default=30,
                  help='pooled B(cell) at which plot alpha reaches 1 (fully opaque); still 0 at B=1')
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

  delta_alpha = np.empty(len(draws))
  for di, idx in enumerate(draws):
    subset = [full[k] for k in idx]
    a_sub, b_sub = df.loc[subset, 'a'].values, df.loc[subset, 'b'].values
    delta_alpha[di] = alpha0_of(a_sub, b_sub) - target_alpha
  abs_delta = np.abs(delta_alpha)

  valid_ess = ess_arr[~np.isnan(ess_arr)]
  print(f'ESS range: [{valid_ess.min():.4f}, {valid_ess.max():.4f}] (K\'={K_prime})',
        file=sys.stderr)
  print(f'abs(delta_alpha_0) range: [{abs_delta.min():.4f}, {abs_delta.max():.4f}]', file=sys.stderr)

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

  ess_edges = build_log_edges(valid_ess, K_prime, args.n_ess_bins, anchor_is_high=True)
  da_edges = build_log_edges(abs_delta, 0.0, args.n_delta_bins, anchor_is_high=False)
  n_y = len(ess_edges) - 1
  n_x = len(da_edges) - 1
  print(f'ESS: {n_y} real+doubled buckets (requested {args.n_ess_bins} raw)', file=sys.stderr)
  print(f'|delta_alpha_0|: {n_x} real+doubled buckets (requested {args.n_delta_bins} raw)',
        file=sys.stderr)
  print(f'grid: {n_x} x {n_y} = {n_x * n_y} cells', file=sys.stderr)

  ess_bin = np.clip(np.digitize(ess_arr, ess_edges) - 1, 0, n_y - 1)
  da_bin = np.clip(np.digitize(abs_delta, da_edges) - 1, 0, n_x - 1)

  # Own-cell draw indices first (needed both standalone and for pooling).
  own_indices = {}
  for yi in range(n_y):
    for xi in range(n_x):
      mask = (ess_bin == yi) & (da_bin == xi) & ~np.isnan(ess_arr)
      own_indices[(yi, xi)] = np.nonzero(mask)[0]

  diff_grid = np.full((n_y, n_x), np.nan)
  icc_nat_grid = np.full((n_y, n_x), np.nan)
  icc_rw_grid = np.full((n_y, n_x), np.nan)
  b_grid = np.zeros((n_y, n_x), dtype=int)

  # Smoothing: pool draws from the 3x3 neighborhood (cell + up to 8
  # neighbors, fewer at edges/corners) before computing each cell's ICC --
  # finer buckets mean fewer draws per cell on their own, so this keeps B
  # large enough to be meaningful. Cell BOUNDARIES stay at the cell's own
  # (unpooled) bin for plotting.
  for yi in range(n_y):
    for xi in range(n_x):
      pooled = []
      for dyi in (-1, 0, 1):
        for dxi in (-1, 0, 1):
          nyi, nxi = yi + dyi, xi + dxi
          if 0 <= nyi < n_y and 0 <= nxi < n_x:
            pooled.append(own_indices[(nyi, nxi)])
      draw_indices = np.concatenate(pooled).tolist() if pooled else []
      b_cell = len(draw_indices)
      b_grid[yi, xi] = b_cell
      if b_cell < 2:
        continue
      icc_nat = icc_for('natural', draw_indices)
      icc_rw = icc_for('reweighted', draw_indices)
      if icc_nat == icc_nat and icc_rw == icc_rw:
        diff_grid[yi, xi] = icc_rw - icc_nat
        icc_nat_grid[yi, xi] = icc_nat
        icc_rw_grid[yi, xi] = icc_rw

  print(f'cells computed: {np.sum(~np.isnan(diff_grid))} / {n_x * n_y}', file=sys.stderr)

  neg_half = np.linspace(0, 1, 128)[:, None] * (np.array(RED) - np.array(DARK_RED)) + np.array(DARK_RED)
  pos_half = np.linspace(0, 1, 128)[:, None] * (np.array(DARK_GREEN) - np.array(GREEN)) + np.array(GREEN)
  cmap = ListedColormap(np.vstack([neg_half, pos_half]), name='red_green_hardstop')
  finite = diff_grid[~np.isnan(diff_grid)]
  max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
  norm = Normalize(vmin=-max_abs, vmax=max_abs)

  fig, ax = plt.subplots(figsize=(9.5, 7.5))
  for yi in range(n_y):
    for xi in range(n_x):
      d = diff_grid[yi, xi]
      b_cell = b_grid[yi, xi]
      b_lin = (0.0 if args.alpha_full_b <= 1
               else float(np.clip((b_cell - 1) / (args.alpha_full_b - 1), 0.0, 1.0)))
      alpha = 2 * b_lin - b_lin ** 2
      if d != d:
        continue
      rgb = cmap(norm(d))[:3]
      ax.add_patch(Rectangle((da_edges[xi], ess_edges[yi]), da_edges[xi + 1] - da_edges[xi],
                              ess_edges[yi + 1] - ess_edges[yi], facecolor=rgb, alpha=alpha,
                              edgecolor='none'))

  ax.set_xlim(da_edges[0], da_edges[-1])
  ax.set_ylim(ess_edges[0], ess_edges[-1])

  # Same view-scale transform as compute_plot_icc_ess_rank_buckets.py's
  # K'-anchored axis: -log(K'-y) has a vertical asymptote AT y=K', so ESS
  # near K' is stretched apart and everything further below is compressed
  # -- stacked on top of the already K'-anchored bucket EDGES above, so
  # the rendering emphasizes the near-K' region twice over (real bucket
  # boundaries + a matching nonlinear view scale).
  def _yforward(y):
    return -np.log(np.maximum(K_prime - y, 1e-6))

  def _yinverse(y):
    return K_prime - np.exp(-y)

  ax.set_yscale('function', functions=(_yforward, _yinverse))

  # Mirror of the above for the x-axis: anchored at 0 (the axis's own
  # natural "good" point), so this is just log(x) with an epsilon floor at
  # x=0 to keep it finite -- expands |delta_alpha_0| values near 0,
  # compresses everything further out. Stacked on top of the already
  # log-anchored-at-0 bucket EDGES, same as the y-axis treatment.
  def _xforward(x):
    return np.log(np.maximum(x, 1e-6))

  def _xinverse(x):
    return np.exp(x)

  ax.set_xscale('function', functions=(_xforward, _xinverse))

  ax.set_xlabel("abs(delta alpha_0)  (=  |alpha_0(D') - alpha_0(D)| ) -- log-anchored at 0")
  ax.set_ylabel(f"ESS (reweighted to alpha_0(D)) -- log-anchored at K'={K_prime}")
  ax.set_title(f"{base}: ICC(reweighted) - ICC(natural) by ESS x |delta_alpha_0| bucket "
               "(3x3-pooled)\n"
               f"K'={K_prime}, B_full={B_full} (exhaustive), target_alpha=alpha_0(D)={target_alpha:.4f}\n"
               f"color = diff (red<0, green>0); alpha = 2b-b^2, "
               f"b=(pooled B-1)/({args.alpha_full_b}-1)", fontsize=10)

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03)
  cbar.set_label('ICC(reweighted) - ICC(natural)')

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'icc_ess_deltaalpha_heatmap_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)

  npz_path = os.path.join(ARTIFACTS_DIR, f'icc_ess_deltaalpha_heatmap_{tag}.npz')
  np.savez(npz_path, ess_edges=ess_edges, da_edges=da_edges, diff_grid=diff_grid, b_grid=b_grid,
           icc_nat_grid=icc_nat_grid, icc_rw_grid=icc_rw_grid,
           target_alpha=target_alpha, K=K, K_prime=K_prime, B_full=B_full)
  print(f'Wrote {npz_path}', file=sys.stderr)
