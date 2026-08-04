"""Scatter of fluency (x) vs. adequacy (y) MQM, one point per system, one plot
per dataset -- the raw (a_i, b_i) points underlying the balance parameter
alpha (action_plan.md Sec 3.2) and the joint-outlier discussion (session
notes: between-system corr(a,b) is far from within-system corr(a_seg,b_seg),
so this plot is the direct picture of the former).

Each plot overlays Mahalanobis-distance iso-contours (closed ellipses) for
the JOINT (a,b) random vector at Z = 1, 2, 3, 4, 5, 6, 7, using
lib.outlier_detection.gk_robust_loc_cov's median + Gnanadesikan-Kettenring-
covariance fit -- the direct bivariate generalization of this project's
median/MAD modified z-score. A single static fit on the full pool, for
visual reference only -- not an active removal decision.

Usage: python codes/scripts/plot_af_scatter.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from lib.consistency import real_systems
from lib.dataset_dirs import DATASET_DIRS
from lib.outlier_detection import gk_robust_loc_cov
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_outliers')

Z_LEVELS = [1, 2, 3, 4, 5, 6, 7]
Z_MAX = max(Z_LEVELS)
# Single-hue sequential ramp (green, light->dark), anchored so Z=0 (never
# drawn -- there's no such contour, just the scale's own origin) would be
# white: each level's color sits at t = Z / Z_MAX along the ramp, not an
# evenly spaced index, so the ramp's lightest end is anchored at the
# meaningful zero rather than at Z_LEVELS[0].
_GK_CMAP = plt.get_cmap('Greens')
_GK_COLORS = [_GK_CMAP(z / Z_MAX) for z in Z_LEVELS]


def add_gk_ellipses(ax, b, a, alpha=1.0, linewidth=1.6):
  """Unfilled ellipses at each Z in Z_LEVELS: the boundary where the joint
  (a,b) Mahalanobis distance from the GK-ROBUST location, under the
  GK-ROBUST covariance (lib.outlier_detection.gk_robust_loc_cov), equals Z.
  Returns the legend handles, or [] if the GK fit is degenerate."""
  out = gk_robust_loc_cov(b, a)
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
  datasets = list(DATASET_DIRS)
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  for base in datasets:
    systems = real_systems(base, root=ROOT)
    if not systems:
      # real_systems() now applies lib.outlier_detection.wmt_official_outliers
      # by default, which treats some datasets (e.g. ende20/zhen20) as
      # UNSUPPORTED and drops their entire roster -- skip rather than crash
      # (gk_mahalanobis on an empty pool raises LinAlgError; .max()/.min() on
      # an empty array raises ValueError).
      print(f'SKIPPED {base} (0 systems -- unsupported under wmt_official_outliers)', file=sys.stderr)
      continue
    df = load_system_scores(base, root=ROOT)
    a = df.loc[systems, 'a'].values
    b = df.loc[systems, 'b'].values
    r = np.corrcoef(a, b)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    b_pad, a_pad = 0.25 * (b.max() - b.min() or 1), 0.25 * (a.max() - a.min() or 1)
    ax.set_xlim(b.min() - b_pad, b.max() + b_pad)
    ax.set_ylim(a.min() - a_pad, a.max() + a_pad)
    gk_handles = add_gk_ellipses(ax, b, a)
    ax.scatter(b, a, s=40, color='#1f77b4', zorder=3)
    for name, xx, yy in zip(systems, b, a):
      ax.annotate(name, xy=(xx, yy), fontsize=7, color='brown', xytext=(4, 4),
                  textcoords='offset points', zorder=4)
    ax.set_xlabel('Fluency MQM (b)')
    ax.set_ylabel('Adequacy MQM (a)')
    ax.set_title(f'{base}: system-level Adequacy vs. Fluency (K={len(systems)}, r={r:+.3f})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if gk_handles:
      ax.legend(handles=gk_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                frameon=False, fontsize=7, title='GK-robust joint Z', title_fontsize=8)
    fig.tight_layout()

    plot_path = os.path.join(ARTIFACTS_DIR, f'af_scatter_{base}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {plot_path} (r={r:+.3f})')