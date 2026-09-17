"""Side-by-side (1 x N) combo of SPA(scorer; beta) vs. target beta panels,
one panel per compute_spa_vs_beta.py .npz cache -- e.g. the same (dataset,
substrings, min_ess) sweep at two different --n-beta resolutions, for a
direct side-by-side comparison. Style follows plot_spa_plane_synthetic_
combo.py's two-panel layout (one panel boxed on all four sides), except
axes are independent (see below) and each panel keeps its OWN legend,
placed outside its frame -- left panel's legend to its left, right panel's
legend to its right (own cache's scorers, which need not match the other
panel's). Purely a plotting script (no solving), so it's cheap to rerun
while iterating on styling -- see compute_spa_vs_beta.py for the
(separately cached) expensive step.

Each panel gets its own independent x/y limits (no sharex/sharey -- axes are
NOT synced across panels). A scorer's color is still the SAME in every panel
it appears in (assigned from the union of scorer names across all caches,
not per-panel), so the panels stay visually comparable despite the
independent axes.

Usage: python -m src.scripts.plot_spa_vs_beta_combo
           --caches output/spa_vs_beta_e3a10_n20.npz,output/spa_vs_beta_e3a10_n10.npz
           [--out output/spa_vs_beta_combo.pdf]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.lib.spa_beta_sweep import load_spa_vs_beta, scorer_color_map, scorer_marker_map

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'output')

# --- Styling knobs -- edit these, then just rerun (no recompute needed) ---
TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 17
LEGEND_FONTSIZE = 20
ANNOTATION_FONTSIZE = 17

MARKER_SIZE = 7
LINE_WIDTH = 1.0

# Legend left-right position, in bbox_to_anchor's axes-fraction x (0 = the
# axes' own left edge, 1 = its right edge -- values outside [0, 1] land
# outside the frame, which is where both side legends live). More negative
# LEGEND_LEFT_MARGIN pushes the left-panel legend further left (needs to
# clear that panel's own y-tick labels + rotated ylabel, both also outside
# the frame at negative x); more positive LEGEND_RIGHT_MARGIN pushes the
# right-panel legend further right (same reason, mirrored -- that panel's
# y-axis was moved to its right side, see below).
LEGEND_LEFT_MARGIN = -0.18
LEGEND_RIGHT_MARGIN = 1.18
LEGEND_MIDDLE_Y_MARGIN = -0.15  # a middle panel (n > 2 only) has no free side; placed below instead

# Per-panel title; {dataset}/{substrings}/{target_ess}/{K}/{n_points} are
# filled in from that panel's own cache. X/Y labels and the beta_0(D)
# annotation are shared across every panel.
TITLE_TEMPLATE = '{dataset}: {substrings}\nESS >= {target_ess:.4g} (K={K}), {n_points} points'
XLABEL = r'$\beta$'
YLABEL = 'SPA'+r'$_\beta$'
BETA0_ANNOTATION_TEXT = r'$\beta_0$'


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--caches', type=str, required=True,
                  help='comma-separated paths to compute_spa_vs_beta.py .npz caches, one per panel')
  p.add_argument('--out', type=str, default=None, help='output path (default: derived from the cache tags)')
  args = p.parse_args()
  cache_paths = [c.strip() for c in args.caches.split(',') if c.strip()]
  if len(cache_paths) < 2:
    sys.exit('--caches needs at least 2 paths for a side-by-side comparison')

  panels = [load_spa_vs_beta(path) for path in cache_paths]

  # Color/marker assignment is shared across ALL panels (union of scorer
  # names, in sorted order) so the same scorer always gets the same color
  # and marker, even if one panel's cache dropped a scorer another one has.
  all_scorer_names = [s for d in panels for s in d['scorers']]
  color_by_scorer = scorer_color_map(all_scorer_names)
  marker_by_scorer = scorer_marker_map(all_scorer_names)

  n = len(panels)
  fig, axes = plt.subplots(1, n, figsize=(8 * n, 6.5))
  if n == 1:
    axes = [axes]

  for panel_i, (ax, d) in enumerate(zip(axes, panels)):
    x_pad = 0.02 * (max(d['betas']) - min(d['betas']) + 1e-9)
    y_pad = 0.02 * (float(d['spa'].max()) - float(d['spa'].min()) + 1e-9)
    xlim = (min(d['betas']) - x_pad, max(d['betas']) + x_pad)
    ylim = (float(d['spa'].min()) - y_pad, float(d['spa'].max()) + y_pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for i, scorer in enumerate(d['scorers']):
      ax.plot(d['betas'], d['spa'][i], color=color_by_scorer[scorer], linewidth=LINE_WIDTH,
              marker=marker_by_scorer[scorer], markersize=MARKER_SIZE, label=scorer)
    ax.axvline(d['beta_0_D'], color='black', alpha=0.3, linewidth=1.0, zorder=1)
    ax.annotate(BETA0_ANNOTATION_TEXT, xy=(d['beta_0_D'], ylim[1]), xytext=(0, -12),
                textcoords='offset points', ha='center', va='top', fontsize=ANNOTATION_FONTSIZE, color='black')
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
    # ax.set_title(TITLE_TEMPLATE.format(dataset=d['dataset'], substrings=d['substrings'],
    #                                     target_ess=d['target_ess'], K=d['K'], n_points=len(d['betas'])),
                #  fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    for side in ('top', 'right', 'bottom', 'left'):
      ax.spines[side].set_visible(True)
    ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
    # Rightmost panel: y-axis (ticks, tick labels, and the axis label
    # itself) moved to the right side, so it reads naturally as the last
    # panel's own axis rather than a duplicate of the left panel's.
    if panel_i == n - 1:
      ax.yaxis.tick_right()
      ax.yaxis.set_label_position('right')

    # Each panel gets its OWN legend (its own cache's scorers, which need
    # not match another panel's) placed OUTSIDE that panel's frame -- on
    # the left for the leftmost panel, the right for the rightmost, so
    # neither legend covers plotted data. A middle panel (n > 2) falls back
    # to below the axes, since there's no free outer side for it.
    handles, labels_ = ax.get_legend_handles_labels()
    if panel_i == 0:
      # markerfirst=False + alignment='right': label text then marker (marker
      # on the right, closest to the axes), text right-justified within the
      # legend box -- mirrors the right-panel legend's natural reading
      # direction (marker-then-text, flowing away from that panel's axis).
      ax.legend(handles, labels_, loc='center right', bbox_to_anchor=(LEGEND_LEFT_MARGIN, 0.5),
                frameon=False, fontsize=LEGEND_FONTSIZE, markerfirst=False, alignment='right')
    elif panel_i == n - 1:
      ax.legend(handles, labels_, loc='center left', bbox_to_anchor=(LEGEND_RIGHT_MARGIN, 0.5),
                frameon=False, fontsize=LEGEND_FONTSIZE)
    else:
      ax.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, LEGEND_MIDDLE_Y_MARGIN),
                frameon=False, fontsize=LEGEND_FONTSIZE, ncol=3)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  if args.out:
    out_path = args.out
  else:
    tags = '_'.join(os.path.splitext(os.path.basename(p))[0].replace('spa_vs_beta_', '') for p in cache_paths)
    out_path = os.path.join(ARTIFACTS_DIR, f'spa_vs_beta_combo_{tags}.pdf')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
