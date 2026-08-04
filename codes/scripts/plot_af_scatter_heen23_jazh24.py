"""Two-panel version of plot_af_scatter.py for a single pair of datasets
(heen23, jazh24), side by side in one figure for direct visual comparison:
shared x/y ranges and a single shared y-axis (drawn once, on the left), each
panel boxed on all four sides. Unlabeled markers (no per-system names) --
just the (a,b) point cloud against the GK-robust ellipses.

Usage: python codes/scripts/plot_af_scatter_heen23_jazh24.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.consistency import real_systems
from mwb.mqm_scoring import load_system_scores
from plot_af_scatter import add_gk_ellipses

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_outliers')

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

if __name__ == '__main__':
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  pool = {}
  for base in DATASETS:
    systems = real_systems(base, root=ROOT)
    if not systems:
      sys.exit(f'{base}: 0 systems under wmt_official_outliers -- cannot build the combined plot')
    df = load_system_scores(base, root=ROOT)
    a = df.loc[systems, 'a'].values
    b = df.loc[systems, 'b'].values
    pool[base] = (systems, b, a)

  # Shared x/y ranges: pad the combined extent of BOTH datasets by the same
  # fraction so a given (a,b) position means the same thing in either panel.
  all_b = np.concatenate([pool[base][1] for base in DATASETS])
  all_a = np.concatenate([pool[base][2] for base in DATASETS])
  b_pad = 0.25 * (all_b.max() - all_b.min() or 1)
  a_pad = 0.25 * (all_a.max() - all_a.min() or 1)
  xlim = (all_b.min() - b_pad, all_b.max() + b_pad)
  ylim = (all_a.min() - a_pad, all_a.max() + a_pad)

  fig, axes = plt.subplots(1, 2, figsize=(34, 17), sharex=True, sharey=True)

  for ax, base, letter in zip(axes, DATASETS, PANEL_LETTERS):
    systems, b, a = pool[base]

    gk_handles = add_gk_ellipses(ax, b, a, alpha=ELLIPSE_ALPHA,
                                  linewidth=ELLIPSE_LINEWIDTH)
    ax.scatter(b, a, s=POINT_SIZE, color=POINT_COLOR, alpha=POINT_ALPHA, zorder=3)

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

    # heen23's positive a/b correlation leaves its ellipses' major axis
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
