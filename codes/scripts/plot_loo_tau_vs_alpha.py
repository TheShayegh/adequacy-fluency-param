"""Single-panel plot of LOO cross-rater Kendall tau vs. target alpha, from
one compute_loo_tau_vs_alpha.py cache -- see that script's docstring for
how the data was produced. Purely a plotting script (no solving), so it's
cheap to rerun while iterating on styling.

Reliability coloring reuses plot_scorer_orientation_vs_alpha.py's
moderated style (SplineReliabilityNorm + _ess_over_k_cmap +
_add_ess_colored_line, same-directory import, no duplicated tuning) rather
than the retired "type 1" curve's harsher multi-stop white/blue/black/red
colormap: a SINGLE dark hue whose OPACITY follows a tuned nonlinear
(cubic-Hermite-spline) ramp from a faint, deliberately-hard-to-read smudge
at the absolute ESS floor (2.0) up to fully opaque at an LOO rater's own
ESS ceiling (K-1). Segments are pre-blended to solid RGB against white
before drawing (avoids translucent-overlap seam artifacts at line joints),
and the reliability legend is a manually-rendered strip (_draw_reliability_
strip) rather than fig.colorbar, since a plain colorbar can't render the
spline's nonlinear shape.

A red dashed horizontal line marks the natural (unweighted) baseline tau;
a black vertical line marks the dataset's own alpha_0(D). Y-axis is fixed
to [0.5, 1.0] (not data-driven), so multiple datasets' plots stay directly
comparable at a glance.

Usage: python codes/scripts/plot_loo_tau_vs_alpha.py
           --cache artifacts/data/loo_tau_vs_alpha_zhen23_n21.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_scorer_orientation_vs_alpha import (
    SplineReliabilityNorm, _add_ess_colored_line, _draw_reliability_strip, _ess_over_k_cmap,
    _ESS_CUTOFF_ABS,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

# --- Styling knobs -- edit these, then just rerun (no recompute needed) ---
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 9
ANNOTATION_FONTSIZE = 8
LINE_WIDTH = 2.5

DARK_COLOR = (0.03, 0.15, 0.35)  # same navy as orientation plots' 'A' (adequacy) family
Y_LIMITS = (0.5, 1.0)

XLABEL = r'Target Balance $\alpha$'
YLABEL = "Kendall's $\\tau$ (pooled across K LOO raters)"
ALPHA0_ANNOTATION_TEXT = r'$\alpha_0(D)$'


def load_loo_tau_vs_alpha(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']),
      'leave_out': int(npz['leave_out']), 'K': int(npz['K']), 'a0_D': float(npz['a0_D']),
      'alpha_lo': float(npz['alpha_lo']), 'alpha_hi': float(npz['alpha_hi']),
      'alphas': npz['alphas'], 'tau': npz['tau'], 'mean_ess': npz['mean_ess'],
      'n_raters_used': npz['n_raters_used'], 'baseline_tau': float(npz['baseline_tau']),
  }


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--cache', type=str, required=True,
                  help='path to a compute_loo_tau_vs_alpha.py .npz cache')
  p.add_argument('--out', type=str, default=None,
                  help='output path (default: derived from the cache filename)')
  args = p.parse_args()

  d = load_loo_tau_vs_alpha(args.cache)
  alphas, tau, ess = d['alphas'], d['tau'], d['mean_ess']
  K_minus_1 = d['K'] - d['leave_out']  # an LOO rater's own system count -- ESS's ceiling
  ess_over_k = ess / K_minus_1
  cutoff_frac = _ESS_CUTOFF_ABS / K_minus_1

  cmap = _ess_over_k_cmap(DARK_COLOR)
  norm = SplineReliabilityNorm(vmin=cutoff_frac, vmax=1.0)

  fig, ax = plt.subplots(figsize=(8, 6))
  _add_ess_colored_line(ax, alphas, tau, ess_over_k, cmap, norm, linewidth=LINE_WIDTH, zorder=3)

  ax.axhline(d['baseline_tau'], color='red', linestyle='--', linewidth=1.5, zorder=2)
  ax.axvline(d['a0_D'], color='black', alpha=0.3, linewidth=1.0, zorder=1)
  ax.annotate(ALPHA0_ANNOTATION_TEXT, xy=(d['a0_D'], Y_LIMITS[1]), xytext=(0, -4),
              textcoords='offset points', ha='center', va='top',
              fontsize=ANNOTATION_FONTSIZE, color='black')

  ax.set_xlim(alphas.min(), alphas.max())
  ax.set_ylim(*Y_LIMITS)
  ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
  ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
  ax.set_title(
      f"{d['dataset']}: LOO cross-rater Kendall $\\tau$ vs. target $\\alpha$ (SPA)\n"
      f"full range [{d['alpha_lo']:.3f}, {d['alpha_hi']:.3f}] (K={d['K']}), {len(alphas)} points",
      fontsize=TITLE_FONTSIZE)
  ax.tick_params(labelsize=TICK_FONTSIZE)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  _draw_reliability_strip(fig, ax, cmap, norm, K_minus_1, _ESS_CUTOFF_ABS)

  proxy_red = Line2D([0], [0], color='red', ls='--', lw=1.5, label='Natural (unweighted) LOO $\\tau$')
  proxy_black = Line2D([0], [0], color='black', alpha=0.3, lw=1.0, label=r'$\alpha_0(D)$')
  ax.legend(handles=[proxy_red, proxy_black], loc='best', frameon=False, fontsize=LEGEND_FONTSIZE)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  base_name = os.path.splitext(os.path.basename(args.cache))[0]
  out_path = args.out or os.path.join(ARTIFACTS_DIR, f'{base_name}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
