"""Parallel plot to plot_consistency_curves.py: instead of one pooled
tau_bar(alpha) curve per meta-metric, plots one tau_bar(dataset, alpha)
curve PER DATASET per meta-metric (from compute_per_dataset_consistency_
curve.py's curve_perdataset_<metametric>[_TAG].json), each colored by its
own ESS(alpha)/K -- reliability normalized to that dataset's own system
count, so 1.0 always means "this dataset's own natural/uniform weighting"
regardless of K.

Each dataset gets one hollow red dot at (alpha_0, its natural/unreweighted
star-pooled consistency against the other datasets' OWN natural rankings).
That does NOT generally land on the dataset's own curve: the curve at
alpha=alpha_0(d) reweights every OTHER dataset to alpha_0(d) too (not their
own alpha_0), so the two numbers differ whenever the other datasets' own
alpha_0 differs from d's. A thin dotted vertical line plus a white/black-
edged dot make that gap -- and its direction -- visible directly.

Usage: python codes/scripts/plot_per_dataset_consistency_curves.py [--tag TAG]
(run compute_reweight_cache.py and compute_per_dataset_consistency_curve.py
first, with the same --tag if any.)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from plot_consistency_curves import METAMETRICS_ORDER, TITLES, LIGHT_BLUE, interpolate_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')
DARK_BLUE = (0.03, 0.15, 0.35)


def load_curve(metametric: str, suffix: str) -> dict:
  path = os.path.join(DATA_DIR, f'curve_perdataset_{metametric}{suffix}.json')
  with open(path) as f:
    return json.load(f)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, default=None,
                  help='must match the tag used with compute_reweight_cache.py, if any')
  p.add_argument('--title', type=str, default=None,
                  help='override the figure suptitle entirely (default: auto-generated)')
  args = p.parse_args()
  suffix = f'_{args.tag}' if args.tag else ''
  out_path = os.path.join(ROOT, 'artifacts', f'consistency_curves_per_dataset{suffix}.png')

  curves = {m: load_curve(m, suffix) for m in METAMETRICS_ORDER}
  datasets = curves[METAMETRICS_ORDER[0]]['datasets']
  K = curves[METAMETRICS_ORDER[0]]['K']
  alpha_0 = curves[METAMETRICS_ORDER[0]]['alpha_0']

  # Fixed [0, 1] normalization (ESS/K, per-dataset-normalized reliability) --
  # unlike the pooled plot, this doesn't depend on what's actually achieved,
  # so no data-driven calibration is needed. One hue (dark blue) throughout,
  # spectrum carried by alpha: fully transparent at and below K/2, then an
  # even linear alpha ramp 0 -> 1 from K/2 to K -- except right at ESS=3K/4
  # (x=0.75), which is called out in light blue at that same alpha (0.5) so
  # the 3-quarter point has a visible landmark on an otherwise monochrome
  # line.
  cmap = LinearSegmentedColormap.from_list('ess_over_k', [
      (0.0, (*DARK_BLUE, 0.0)),
      (0.5, (*DARK_BLUE, 0.0)),
      (0.75, (*LIGHT_BLUE, 0.5)),
      (1.0, (*DARK_BLUE, 1.0)),
  ])
  norm = Normalize(vmin=0, vmax=1)

  fig, axes = plt.subplots(2, 3, figsize=(18, 7.5),
                             gridspec_kw=dict(wspace=0.6, hspace=0.5, top=0.85, bottom=0.08))
  axes = axes.flatten()
  left_column = {0, 3}

  all_series = {}  # metametric -> {dataset -> series dict}, needed again in the annotation pass
  all_alphas = {}

  for i, (ax, m) in enumerate(zip(axes, METAMETRICS_ORDER)):
    c = curves[m]
    alphas = np.array(c['alphas'])
    all_alphas[m] = alphas
    x_lo, x_hi = alphas.min(), alphas.max()
    y_all = []

    # Per-dataset series (tau curve, natural-balance dot, this curve's own
    # value at alpha_0) -- kept around (all_series) for the annotation pass
    # below, which needs every OTHER dataset's curve value too.
    series = {}
    for d in datasets:
      tau = np.array(c['tau_bar_by_dataset'][d])
      ess = np.array(c['ess_by_dataset'][d])
      a0 = alpha_0[d]
      series[d] = dict(
          tau=tau, ess_over_k=ess / K[d], a0=a0,
          base_tau=c['baseline_tau_by_dataset'][d],
          on_curve_tau=float(np.interp(a0, alphas, tau)),
      )
    all_series[m] = series

    # Draw every line, the dotted connector, and the two dots.
    for d in datasets:
      s = series[d]
      y_all.append(s['tau'])
      y_all.append(np.array([s['base_tau'], s['on_curve_tau']]))

      fine_alphas, fine_tau, fine_eok = interpolate_curve(alphas, s['tau'], s['ess_over_k'])
      points = np.array([fine_alphas, fine_tau]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)
      seg_eok = (fine_eok[:-1] + fine_eok[1:]) / 2.0
      lc = LineCollection(segments, cmap=cmap, norm=norm)
      lc.set_array(seg_eok)
      lc.set_linewidth(2.5)
      lc.set_joinstyle('round')
      lc.set_capstyle('round')
      ax.add_collection(lc)

      # Natural-balance dot: this dataset's OWN natural consistency against
      # everyone else's OWN natural rankings -- generally NOT the same
      # point as this curve at alpha=alpha_0(d) (the curve reweights every
      # other dataset to alpha_0(d) too, not their own alpha_0). The solid
      # red connector plus white/black intersection dot make that gap
      # visible.
      ax.plot([s['a0'], s['a0']], [s['base_tau'], s['on_curve_tau']], linestyle='-',
               color='red', alpha=0.5, linewidth=1.2, zorder=4)
      ax.scatter([s['a0']], [s['on_curve_tau']], facecolor='white', edgecolor='black',
                  linewidths=0.8, s=28, zorder=5)
      ax.scatter([s['a0']], [s['base_tau']], facecolor='none', edgecolor='red',
                  linewidths=1.5, s=55, zorder=6)

    ax.set_xlim(x_lo, x_hi)
    y_all = np.concatenate(y_all)
    pad = 0.05 * (y_all.max() - y_all.min() + 1e-9)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_xlabel('Controlled Balance\n(Adequacy over Fluency)')
    if i in left_column:
      ax.set_ylabel('Consistency over Years', labelpad=45)
    ax.set_title(TITLES[m])
    ax.set_box_aspect(1)
    # box_aspect otherwise centers the resulting square within its taller
    # grid cell -- anchoring it to the top hugs it against the row above
    # (the title, for row 1) instead of floating with dead space over it.
    ax.set_anchor('N')

  for ax in axes[len(METAMETRICS_ORDER):]:
    ax.axis('off')

  # Every axes' box_aspect/wspace/hspace only actually resolves to real
  # pixel geometry once the whole figure has been drawn once -- do that
  # single draw here, THEN place annotations, so the "outside the axes"
  # test below is checked against final layout, not a stale mid-loop one.
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  fig_bbox = fig.get_window_extent(renderer=renderer)

  # Annotate each red dot OUTSIDE its own subplot's box (never over the
  # data area, so it can never sit on top of a line): for each of the 4
  # directions (left/right/up/down), computes how far the dot already is
  # from that edge of the axes' pixel bounding box, then takes whichever
  # direction requires the smallest push to clear it (plus a margin).
  # top/bottom get a large extra penalty -- that's where the title and the
  # x-axis label text live, both at the same shared height across every
  # subplot, so an exit there is much likelier to collide with THAT text
  # than a left/right exit is with a neighboring subplot.
  margin_px = 18
  min_label_gap_px = 24  # enforced next among labels that end up on the same side
  for i, (ax, m) in enumerate(zip(axes, METAMETRICS_ORDER)):
    series = all_series[m]
    dpi_scale = ax.figure.dpi / 72.0
    ax_bbox = ax.get_window_extent(renderer=renderer)
    placements = {}  # dataset -> [side, dx, dy, anchor_y_px]
    for d in datasets:
      s = series[d]
      px, py = ax.transData.transform((s['a0'], s['base_tau']))
      options = sorted([
          ('left', px - ax_bbox.x0 + margin_px, (-(px - ax_bbox.x0 + margin_px), 0.0)),
          ('right', ax_bbox.x1 - px + margin_px, (ax_bbox.x1 - px + margin_px, 0.0)),
          ('top', 2.5 * (ax_bbox.y1 - py + margin_px), (0.0, ax_bbox.y1 - py + margin_px)),
          ('bottom', 2.5 * (py - ax_bbox.y0 + margin_px), (0.0, -(py - ax_bbox.y0 + margin_px))),
      ], key=lambda t: t[1])
      side, _, (ddx, ddy) = options[0]
      for s2, _, (ddx2, ddy2) in options:
        cx, cy = px + ddx2, py + ddy2
        if fig_bbox.x0 + 5 <= cx <= fig_bbox.x1 - 5 and fig_bbox.y0 + 5 <= cy <= fig_bbox.y1 - 5:
          side, ddx, ddy = s2, ddx2, ddy2
          break
      placements[d] = [side, ddx, ddy, py + ddy]

    # Declutter: among labels exiting the same side, enforce a minimum
    # vertical gap (in display pixels) so two dots that are close in y
    # don't produce two overlapping boxes.
    for side in ('left', 'right', 'top', 'bottom'):
      group = sorted((d for d in datasets if placements[d][0] == side),
                      key=lambda d: placements[d][3])
      for k in range(1, len(group)):
        prev_y, cur = placements[group[k - 1]][3], placements[group[k]]
        if cur[3] - prev_y < min_label_gap_px:
          shift = min_label_gap_px - (cur[3] - prev_y)
          cur[2] += shift
          cur[3] += shift

    for d in datasets:
      s = series[d]
      _, ddx, ddy, _ = placements[d]
      dx_pts, dy_pts = ddx / dpi_scale, ddy / dpi_scale
      ax.annotate(d, xy=(s['a0'], s['base_tau']), fontsize=6.5, color='brown',
                  xytext=(dx_pts, dy_pts), textcoords='offset points',
                  ha=('left' if dx_pts >= 0 else ('right' if dx_pts < 0 else 'center')),
                  va=('bottom' if dy_pts > 0 else ('top' if dy_pts < 0 else 'center')),
                  zorder=7, annotation_clip=False,
                  bbox=dict(boxstyle='round,pad=0.15', fc='#FFF7CC', ec='brown', alpha=0.85, lw=0.5),
                  arrowprops=dict(arrowstyle='-', color='#D4B942', lw=0.7, shrinkA=2, shrinkB=4))

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.08, ticks=[0, 0.5, 1])
  cbar.set_ticklabels(['0', 'K/2', 'K'])
  cbar.set_label('Reliability (ESS)')

  slot = axes[len(METAMETRICS_ORDER)].get_position()
  slot_center = (slot.x0 + slot.width / 2, slot.y0 + slot.height / 2)
  fig.text(slot_center[0], slot_center[1],
            'Line: one dataset vs. every\n'
            'other, reweighted to $\\alpha$\n'
            '(color = ESS/K).\n\n'
            'Red circle: natural\n'
            'consistency at $\\alpha_0$.\n\n'
            'White/black dot: this\n'
            'curve\'s own value at $\\alpha_0$\n'
            '(others still reweighted).',
            fontsize=9, ha='center', va='center',
            transform=fig.transFigure)

  if args.title:
    fig.suptitle(args.title, fontsize=14)
  else:
    title_suffix = f' ({args.tag})' if args.tag else ''
    fig.suptitle(
        r'Cross-dataset scorer-ranking consistency vs. reweighted balance $\alpha$' + title_suffix
        + '\n(action_plan.md 6.1 x 6.5, numeric solver)',
        fontsize=14,
    )
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
