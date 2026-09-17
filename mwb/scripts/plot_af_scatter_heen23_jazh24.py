"""This is the paper's Figure 1: a two-panel adequacy-vs-fluency scatter for
(heen23, jazh24) side by side, shared x/y ranges and a single shared y-axis
(drawn once, on the left), each panel boxed on all four sides. Unlabeled
markers (no per-system names) -- just the (a,f) point cloud against the
GK-robust Mahalanobis-distance ellipses.

Usage: python -m mwb.scripts.plot_af_scatter_heen23_jazh24
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from mwb.lib.consistency import real_systems
from mwb.lib.outlier_detection import gk_robust_loc_cov
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'output')

DATASETS = ['heen23', 'jazh24']
PANEL_LETTERS = ['a', 'b']
POINT_COLOR = '#800020'  # burgundy
POINT_ALPHA = 0.6
POINT_SIZE = 320
ELLIPSE_ALPHA = 0.55
ELLIPSE_LINEWIDTH = 3.2
TITLE_FONTSIZE = 72
LABEL_FONTSIZE = 64
TICK_FONTSIZE = 54
LEGEND_FONTSIZE = 50
TICK_PAD = 25
LABEL_PAD = -50

# Single-hue sequential ramp (green, light->dark), anchored so Z=0 (never
# drawn -- there's no such contour, just the scale's own origin) would be
# white: each level's color sits at t = Z / Z_MAX along the ramp, not an
# evenly spaced index, so the ramp's lightest end is anchored at the
# meaningful zero rather than at Z_LEVELS[0].
Z_LEVELS = [1, 2, 3, 4, 5, 6, 7]
Z_MAX = max(Z_LEVELS)
_GK_CMAP = plt.get_cmap('Greens')
_GK_COLORS = [_GK_CMAP(z / Z_MAX) for z in Z_LEVELS]


def add_gk_ellipses(ax, f, a, alpha=1.0, linewidth=1.6):
  """Unfilled ellipses at each Z in Z_LEVELS: the boundary where the joint
  (a,f) Mahalanobis distance from the GK-ROBUST location, under the
  GK-ROBUST covariance (mwb.lib.outlier_detection.gk_robust_loc_cov), equals
  Z. Returns the legend handles, or [] if the GK fit is degenerate."""
  out = gk_robust_loc_cov(f, a)
  if out is None:
    return []
  mean, cov = out
  eigvals, eigvecs = np.linalg.eigh(cov)
  order = np.argsort(eigvals)[::-1]
  eigvals, eigvecs = eigvals[order], eigvecs[:, order]
  angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
  handles = []
  for z, color in zip(Z_LEVELS, _GK_COLORS):
    width, height = 2 * z * np.sqrt(eigvals[0]), 2 * z * np.sqrt(eigvals[1])
    patch = Ellipse(xy=mean, width=width, height=height, angle=angle,
                     fill=False, edgecolor=color, linewidth=linewidth, zorder=1,
                     linestyle='solid', label=f'Z={z:g}', alpha=alpha)
    ax.add_patch(patch)
    handles.append(patch)
  return handles

if __name__ == '__main__':
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  pool = {}
  for base in DATASETS:
    systems = real_systems(base, root=ROOT)
    if not systems:
      sys.exit(f'{base}: 0 systems under wmt_official_outliers -- cannot build the combined plot')
    df = load_system_scores(base, root=ROOT)
    a = df.loc[systems, 'a'].values
    f = df.loc[systems, 'f'].values
    pool[base] = (systems, f, a)

  # Shared x/y ranges: pad the combined extent of BOTH datasets by the same
  # fraction so a given (a,f) position means the same thing in either panel.
  all_f = np.concatenate([pool[base][1] for base in DATASETS])
  all_a = np.concatenate([pool[base][2] for base in DATASETS])
  f_pad = 0.25 * (all_f.max() - all_f.min() or 1)
  a_pad = 0.25 * (all_a.max() - all_a.min() or 1)
  xlim = (all_f.min() - f_pad, all_f.max() + f_pad)
  ylim = (all_a.min() - a_pad, all_a.max() + a_pad)

  fig, axes = plt.subplots(1, 2, figsize=(34, 17), sharex=True, sharey=True)

  for ax, base, letter in zip(axes, DATASETS, PANEL_LETTERS):
    systems, f, a = pool[base]

    gk_handles = add_gk_ellipses(ax, f, a, alpha=ELLIPSE_ALPHA,
                                  linewidth=ELLIPSE_LINEWIDTH)
    ax.scatter(f, a, s=POINT_SIZE, color=POINT_COLOR, alpha=POINT_ALPHA, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Only the min/max tick of each axis, taken from the default locator's
    # in-range ticks (rather than the raw padded xlim/ylim floats) so the
    # labels stay at the same "nice" values matplotlib would otherwise show.
    xticks_in = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    yticks_in = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
    ax.set_xticks([xticks_in[0], xticks_in[-1]])
    ax.set_yticks([yticks_in[0], yticks_in[-1]])
    ax.set_xlabel('Fluency MQM', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_title(f'({letter}) {base}', fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, pad=TICK_PAD)
    # Full box (all four spines) so each panel reads as its own frame.
    for side in ('top', 'right', 'bottom', 'left'):
      ax.spines[side].set_visible(True)

    # heen23's positive a/f correlation leaves its ellipses' major axis
    # running bottom-left to top-right, so the bottom-right corner of THIS
    # panel stays blank -- put the legend there instead of stealing figure
    # margin.
    if gk_handles and letter == PANEL_LETTERS[0]:
      ax.legend(handles=gk_handles, loc='lower right',
                frameon=True, framealpha=0.9, edgecolor='none',
                fontsize=LEGEND_FONTSIZE)

  axes[0].set_ylabel('Adequacy MQM', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
  # Shared y-axis drawn once, on the left panel only.
  axes[1].tick_params(labelleft=False)

  fig.tight_layout(pad=0.5, w_pad=1.0)

  plot_path = os.path.join(ARTIFACTS_DIR, 'af_scatter_heen23_jazh24.pdf')
  fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
  plt.close(fig)
  print(f'Wrote {plot_path}')
