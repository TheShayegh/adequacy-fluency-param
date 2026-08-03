"""SPA Plane (material/prior_work/metrics-dp-tradeoff.tex Sec 4.2, Figure
SPAplane): one plot per dataset, x = SPA vs. Fluency MQM, y = SPA vs.
Adequacy MQM, one point per available scorer (every automatic MT metric
with segment-level score files for that dataset -- lib.spa_plane.
scorer_spa_points), plus the paper's three sentinel curves:

  - tradeoff line (black): segment-level linear interpolation of Adequacy
    and Fluency MQM. No scorer can lie above/right of it.
  - adequacy-knowledge line (blue): interpolation of Adequacy MQM with
    per-segment-uniform noise -- 10 shadow instances (light) + their
    mean (solid).
  - fluency-knowledge line (red): the mirror, interpolating Fluency MQM.

See lib.spa_plane's module docstring for the full construction and for why
scorer points are unavailable on some datasets (positional-alignment
validation) even though the sentinel lines never are.

Usage: python codes/scripts/plot_spa_plane.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.consistency import real_systems
from lib.dataset_dirs import DATASET_DIRS
from lib.spa_plane import build_af_seg_matrices, knowledge_lines, scorer_spa_points, tradeoff_line

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_spa_plane')

_TRADEOFF_COLOR = '#000000'
_ADEQUACY_COLOR = '#1f4fd8'   # blue -- matches the paper's adequacy-knowledge line
_FLUENCY_COLOR = '#d81f2f'    # red -- matches the paper's fluency-knowledge line
_SCORER_COLOR = '#555555'


def _plot_line(ax, pts, color, label, lw=2.2, zorder=3):
  xs, ys = zip(*pts)
  ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, label=label)


def _plot_shadows(ax, shadows, color, zorder=2):
  for pts in shadows:
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25, zorder=zorder)


if __name__ == '__main__':
  datasets = list(DATASET_DIRS)
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  t0 = time.time()

  for i, base in enumerate(datasets, 1):
    # excl_missing_seg_granular_mqm=True: this script's scorer points need
    # line-position-aligned segment MQM (lib.spa_plane.
    # positional_af_matrices), which enru22 doesn't have (lib.consistency.
    # MISSING_SEG_GRANULAR_MQM) -- everywhere else in the project keeps it.
    systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
    if not systems:
      print(f'SKIPPED {base} (0 real systems)', file=sys.stderr)
      continue

    a_seg, b_seg = build_af_seg_matrices(base, systems, root=ROOT)
    if a_seg is None:
      print(f'SKIPPED {base} (too few jointly-valid adequacy/fluency segments)', file=sys.stderr)
      continue

    tradeoff_pts = tradeoff_line(a_seg, b_seg)
    a_shadows, a_mean, b_shadows, b_mean = knowledge_lines(a_seg, b_seg)
    points = scorer_spa_points(base, systems, root=ROOT)

    elapsed = time.time() - t0
    eta = elapsed / i * (len(datasets) - i)
    print(f'{base}: K={len(systems)} scorers={len(points)} '
          f'(elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

    all_xy = tradeoff_pts + a_mean + b_mean + list(points.values())
    for shadows in (a_shadows, b_shadows):
      for pts in shadows:
        all_xy.extend(pts)
    xs_all, ys_all = zip(*all_xy)
    pad = 0.05
    xlo, xhi = max(0.0, min(xs_all) - pad), min(1.05, max(xs_all) + pad)
    ylo, yhi = max(0.0, min(ys_all) - pad), min(1.05, max(ys_all) + pad)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    _plot_shadows(ax, a_shadows, _ADEQUACY_COLOR)
    _plot_shadows(ax, b_shadows, _FLUENCY_COLOR)
    _plot_line(ax, tradeoff_pts, _TRADEOFF_COLOR, 'Tradeoff (Adequacy <-> Fluency MQM)')
    _plot_line(ax, a_mean, _ADEQUACY_COLOR, 'Adequacy-knowledge (Adequacy MQM <-> noise)')
    _plot_line(ax, b_mean, _FLUENCY_COLOR, 'Fluency-knowledge (Fluency MQM <-> noise)')

    if points:
      xs = [xy[0] for xy in points.values()]
      ys = [xy[1] for xy in points.values()]
      ax.scatter(xs, ys, s=26, color=_SCORER_COLOR, edgecolor='white', linewidth=0.5, zorder=4)
      for name, (x, y) in points.items():
        ax.annotate(name, xy=(x, y), fontsize=6, color=_SCORER_COLOR, xytext=(3, 3),
                    textcoords='offset points', zorder=5)
    else:
      ax.text(0.5, 0.5, 'No scorer segment-position alignment available for this dataset',
              transform=ax.transAxes, ha='center', va='center', fontsize=9, color='#888888')

    ax.set_xlabel('SPA vs. Fluency MQM')
    ax.set_ylabel('SPA vs. Adequacy MQM')
    ax.set_title(f'{base}: SPA plane (K={len(systems)} systems, {len(points)} scorers)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    plot_path = os.path.join(ARTIFACTS_DIR, f'spa_plane_{base}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {plot_path}', file=sys.stderr)