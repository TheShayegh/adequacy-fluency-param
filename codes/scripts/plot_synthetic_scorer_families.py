"""SPA Plane augmented with synthetic scorer families (material/synthetic_
scorer_construction.md, lib.synthetic_scorers): one plot per real donor
scorer (all 27+ donors overlaid on a single plane made the per-donor
families impossible to tell apart), each showing that donor's Adequacy-
dialed (A) and Fluency-dialed (B) synthetic families on the same SPA plane
as lib.spa_plane / plot_spa_plane.py -- x = SPA vs. Fluency MQM, y = SPA vs.
Adequacy MQM.

Dots carry no scorer-name label and are colored by dial value alone: family
A goes black (dial=0) -> blue (dial=1), family B goes black (dial=0) -> red
(dial=1). Concretely, RGB = (0, 0, dial) for family A and (dial, 0, 0) for
family B, so at dial=0 both families collapse onto the same black point --
the real donor scorer, exactly (section 4's endpoint). A thin gray line
connects each family's dials in order (0 -> 1) to trace the sweep.

Usage: python codes/scripts/plot_synthetic_scorer_families.py [--dataset ende21]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.consistency import real_systems
from lib.spa_plane import build_af_seg_matrices, knowledge_line, tradeoff_line
from lib.synthetic_scorers import DIAL_GRID, all_synthetic_family_points

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_synthetic_scorers')

_TRADEOFF_COLOR = '#000000'
_ADEQUACY_COLOR = '#1f4fd8'
_FLUENCY_COLOR = '#d81f2f'


def _plot_line(ax, pts, color, label, lw=2.2, zorder=3):
  xs, ys = zip(*pts)
  ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, label=label)


def _plot_shadows(ax, shadows, color, zorder=2):
  for pts in shadows:
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25, zorder=zorder)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  args = parser.parse_args()
  base = args.dataset

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to plot')

  a_seg, b_seg = build_af_seg_matrices(base, systems, root=ROOT)
  if a_seg is None:
    sys.exit(f'{base}: too few jointly-valid adequacy/fluency segments')

  print(f'{base}: K={len(systems)} systems -- building sentinel lines', file=sys.stderr)
  tradeoff_pts = tradeoff_line(a_seg, b_seg)
  a_shadows, a_mean = knowledge_line(a_seg, a_seg, b_seg)
  b_shadows, b_mean = knowledge_line(b_seg, a_seg, b_seg)

  print(f'{base}: building synthetic scorer families (donor scan)', file=sys.stderr)
  t0 = time.time()
  families = all_synthetic_family_points(base, systems, root=ROOT)
  print(f'{base}: {len(families)} donors, {sum(len(f["A"]) + len(f["B"]) for f in families.values())} '
        f'synthetic points ({time.time() - t0:.1f}s)', file=sys.stderr)

  # Shared axis limits across every donor's plot, so the per-donor figures
  # stay visually comparable -- derived from the sentinel lines plus every
  # donor's own points, not just the one being drawn.
  all_xy = list(tradeoff_pts) + list(a_mean) + list(b_mean)
  for fam in families.values():
    all_xy.extend(fam['A'].values())
    all_xy.extend(fam['B'].values())
  for shadows in (a_shadows, b_shadows):
    for pts in shadows:
      all_xy.extend(pts)
  xs_all, ys_all = zip(*all_xy)
  pad = 0.05
  xlo, xhi = max(0.0, min(xs_all) - pad), min(1.05, max(xs_all) + pad)
  ylo, yhi = max(0.0, min(ys_all) - pad), min(1.05, max(ys_all) + pad)

  out_dir = os.path.join(ARTIFACTS_DIR, base)
  os.makedirs(out_dir, exist_ok=True)
  n = len(families)
  t0 = time.time()
  for i, (donor, fam) in enumerate(sorted(families.items()), 1):
    a_pts = sorted(fam['A'].items())  # dial order, for the connecting line
    b_pts = sorted(fam['B'].items())
    a_x = [xy[0] for _, xy in a_pts]; a_y = [xy[1] for _, xy in a_pts]
    a_c = [(0.0, 0.0, d) for d, _ in a_pts]
    b_x = [xy[0] for _, xy in b_pts]; b_y = [xy[1] for _, xy in b_pts]
    b_c = [(d, 0.0, 0.0) for d, _ in b_pts]

    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    _plot_shadows(ax, a_shadows, _ADEQUACY_COLOR)
    _plot_shadows(ax, b_shadows, _FLUENCY_COLOR)
    _plot_line(ax, tradeoff_pts, _TRADEOFF_COLOR, 'Tradeoff (Adequacy <-> Fluency MQM)')
    _plot_line(ax, a_mean, _ADEQUACY_COLOR, 'Adequacy-knowledge (Adequacy MQM <-> noise)')
    _plot_line(ax, b_mean, _FLUENCY_COLOR, 'Fluency-knowledge (Fluency MQM <-> noise)')

    if a_x:
      ax.plot(a_x, a_y, color='#999999', linewidth=0.8, zorder=3)
      ax.scatter(a_x, a_y, s=30, c=a_c, edgecolor='none', zorder=4,
                 label='Adequacy-dialed family (A: black->blue)')
    if b_x:
      ax.plot(b_x, b_y, color='#999999', linewidth=0.8, zorder=3)
      ax.scatter(b_x, b_y, s=30, c=b_c, edgecolor='none', zorder=4,
                 label='Fluency-dialed family (B: black->red)')

    ax.set_xlabel('SPA vs. Fluency MQM')
    ax.set_ylabel('SPA vs. Adequacy MQM')
    ax.set_title(f'{base}: {donor} synthetic families (K={len(systems)} systems)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()

    safe_donor = donor.replace('/', '_')
    plot_path = os.path.join(out_dir, f'spa_plane_synthetic_{base}_{safe_donor}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    print(f'[{i}/{n}] wrote {plot_path} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
