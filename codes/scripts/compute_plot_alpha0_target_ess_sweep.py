"""Sweeps target_alpha over EVERY alpha_0(D') from the C(K,K') size-K'
subsets of D (lib.subsampling_stability._alpha_0_over_subsets -- the same
population median_alpha_0/mean_alpha_0 summarize), and for each candidate
measures mean ESS/K' across the SAME B subsampling draws used by
compute_subsampling_pairwise_stability.py (lib.subsampling_stability.
target_alpha_ess_sweep -- solve_w_exact only, no metametric/SPA scoring, so
a full sweep is ~2 min instead of the ~4.6 hours a full pairwise_stability
sweep at this many points would take).

Purpose: mean_alpha_0/median_alpha_0 turned out to differ only slightly
from alpha_0(D) and didn't change the (negative) reweighting effect on
ende22 -- this sweep instead asks, directly, which target is CHEAPEST to
reach (highest mean ESS) for typical K'-sized subsamples, rather than which
one summarizes their central tendency. Follows the "type 1" plot style
(codes/scripts/_obsolete_type1_type2_plot_styles.py::render_type1_figure):
an ESS-driven white -> light blue -> standard blue -> black -> red
colormap, per-dataset alpha_0 vertical markers, dashed red reference line.

Usage: python codes/scripts/compute_plot_alpha0_target_ess_sweep.py
           --dataset ende22 --k-prime 8 [--b 20] [--seed 0]
           [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems
from lib.subsampling_stability import (
    _alpha_0_over_subsets, mean_alpha_0, median_alpha_0, target_alpha_ess_sweep,
)
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

LIGHT_BLUE = (0.55, 0.75, 0.92)
STANDARD_BLUE = (0.121, 0.466, 0.705)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--k-prime', type=int, required=True, help="K': subsample size")
  p.add_argument('--b', type=int, default=20, help='number of subsamples')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  K_prime = args.k_prime
  B = args.b
  tag = args.tag or f'{base}_kprime{K_prime}_b{B}'

  full_systems = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'b'].values
  K = len(full_systems)
  a0_D = alpha0_of(a_D, b_D)
  med = median_alpha_0(base, K_prime, root=ROOT, exclude_outliers=args.exclude_outliers)
  mean_a0 = mean_alpha_0(base, K_prime, root=ROOT, exclude_outliers=args.exclude_outliers)

  candidates = sorted(set(_alpha_0_over_subsets(base, K_prime, root=ROOT,
                                                  exclude_outliers=args.exclude_outliers)))
  print(f'D = {base}: K={K}, alpha_0(D)={a0_D:.4f}, median_alpha_0(D,{K_prime})={med:.4f}, '
        f'mean_alpha_0(D,{K_prime})={mean_a0:.4f}', file=sys.stderr)
  print(f"K'={K_prime}, B={B}, seed={args.seed}, {len(candidates)} distinct candidate targets "
        f'(alpha_0 of every size-{K_prime} subset of D)', file=sys.stderr)

  t0 = time.time()
  sweep = target_alpha_ess_sweep(base, K_prime, B, candidates, seed=args.seed, root=ROOT,
                                  exclude_outliers=args.exclude_outliers,
                                  progress_every=max(1, len(candidates) // 10))
  print(f'done in {time.time() - t0:.1f}s', file=sys.stderr)

  alphas = np.array(sweep['target_alphas'])
  ess_frac = np.array(sweep['mean_ess']) / K_prime
  best_i = int(np.nanargmax(ess_frac))
  best_alpha, best_ess_frac = float(alphas[best_i]), float(ess_frac[best_i])
  print(f'ESS-argmax target_alpha={best_alpha:.4f} (mean ESS/K\'={best_ess_frac:.4f})', file=sys.stderr)

  # Type-1-style ESS colormap, calibrated to what this sweep actually
  # achieves (not the theoretical [0,1] range): opaque white at/below the
  # achieved min, fading through light blue -> standard blue -> black at
  # the achieved peak, red beyond (mirrors render_type1_figure exactly,
  # substituting ESS/K' in place of that figure's raw ESS/mean_K scale).
  valid = ess_frac[~np.isnan(ess_frac)]
  frac_min, frac_peak = float(valid.min()), float(valid.max())
  frac_third = frac_min + (frac_peak - frac_min) / 3.0
  frac_twothirds = frac_min + (frac_peak - frac_min) * 2.0 / 3.0
  cmap = LinearSegmentedColormap.from_list('ess_fade', [
      (0.0, (1, 1, 1, 1.0)),
      (frac_min, (1, 1, 1, 1.0)),
      (frac_third, (*LIGHT_BLUE, 1.0)),
      (frac_twothirds, (*STANDARD_BLUE, 1.0)),
      (frac_peak, (0, 0, 0, 1.0)),
      (1.0, (1, 0, 0, 1.0)),
  ])
  norm = Normalize(vmin=frac_min, vmax=frac_peak)

  fig, ax = plt.subplots(figsize=(8, 6.5))

  points = np.array([alphas, ess_frac]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  seg_ess = (ess_frac[:-1] + ess_frac[1:]) / 2.0
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  lc.set_array(seg_ess)
  lc.set_linewidth(2.2)
  lc.set_joinstyle('round')
  lc.set_capstyle('round')
  ax.add_collection(lc)

  ax.axhline(1.0, color='red', linestyle='--', linewidth=1.3, alpha=0.8, zorder=2)

  for a0, color, tex_label, dy in [
      (a0_D, 'black', r'$\alpha_0(D)$', 10),
      (med, 'purple', 'median', 24),
      (mean_a0, 'darkgreen', 'mean', 38),
  ]:
    ax.axvline(a0, color=color, alpha=0.35, linewidth=1.2, zorder=1)
    y_at = float(np.interp(a0, alphas, ess_frac))
    ax.scatter([a0], [y_at], marker='o', s=40, color=color, edgecolor='black',
               linewidths=0.5, zorder=5)
    ax.annotate(tex_label, xy=(a0, y_at), xytext=(0, dy), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color=color, zorder=6)

  ax.scatter([best_alpha], [best_ess_frac], marker='*', s=220, color='gold', edgecolor='black',
             linewidths=0.8, zorder=7)
  ax.annotate(f'ESS-argmax\n$\\alpha$={best_alpha:.4f}', xy=(best_alpha, best_ess_frac),
              xytext=(0, -34), textcoords='offset points', ha='center', va='top',
              fontsize=8, color='black', zorder=8)

  ax.set_xlim(alphas.min(), alphas.max())
  ax.set_ylim(0, 1.08)
  ax.set_xlabel(r"Candidate Target Balance $\alpha$  (= $\alpha_0$(D') of some size-K' subset)")
  ax.set_ylabel(r"Reliability of reaching $\alpha$  (mean ESS/K' over B draws)")
  ax.set_title(f"{base}: reweighting cost vs. target " + r'$\alpha$' + "\n"
               r"sweep over $\alpha_0$(D') of every size-K' subset "
               f"(K={K}, K'={K_prime}, B={B})", fontsize=11)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03)
  cbar.set_label("Reliability (ESS/K')")

  proxy_red = Line2D([0], [0], color='red', ls='--', lw=1.3, label="Natural (ESS/K'=1, no reweighting)")
  ax.legend(handles=[proxy_red], loc='lower center', frameon=False, fontsize=8)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'alpha0_target_ess_sweep_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
