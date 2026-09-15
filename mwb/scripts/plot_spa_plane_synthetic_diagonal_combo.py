"""Two-panel SPA-plane figure, one donor scorer per panel -- a variant of
plot_spa_plane_synthetic_combo.py (same tradeoff triangle + Adequacy/
Fluency-knowledge curves + T/J families + panel layout) that replaces the
Adequacy family (A) and Fluency family (B) with the single Diagonal family
(D, mwb.lib.synthetic_scorers.donor_family_additive_mean_diagonal): A = dial,
B = -dial, so the two aspects' effects move in OPPOSITE directions on one
dial instead of being swept independently.

Dial grid: D itself uses DIAGONAL_DIAL_GRID_SPARSE, a plotting-only half-
precision rebuild of mwb.lib.synthetic_scorers.DIAGONAL_ADDITIVE_MEAN_DIAL_GRID
-- same [-0.5, 0.5] anchor/extent, but step 0.04 instead of the canonical
0.02 (26 points instead of 51). This is local to the script (the library
constant/default is untouched) since it's a plotting-density choice, not a
change to the family's own committed default. T keeps using the full-
precision DIAGONAL_ADDITIVE_MEAN_DIAL_GRID (T's own sweep wasn't asked to
change); J keeps plot_spa_plane_synthetic_combo.py's own sparse two-sided
grid unchanged.

D is colored by a fixed red->blue diverging ramp (linearly interpolating
this project's own _FLUENCY_COLOR -> _ADEQUACY_COLOR, which passes through
purple at the midpoint with no extra color stop needed), anchored at
dial=-1 (red) / dial=0 (purple) / dial=+1 (blue) -- NOT at this sweep's own
[-0.5, 0.5] extremes. Since the actual dial grid only reaches +-0.5, points
only span t in [0.25, 0.75] of that ramp: every dot comes out visibly more
purple and less saturated red/blue than the anchor colors themselves, by
design (a full-range sweep would reach pure red/blue; this one doesn't).
This also matches the project's existing convention (mwb.lib.spa_plane) of
blue=Adequacy, red=Fluency: D's positive dial pushes toward Adequacy
(adds abar_k, subtracts bbar_k), so blue-at-+1 and red-at-1 are the same
color/aspect pairing used everywhere else in this codebase.

Usage: python -m mwb.scripts.plot_spa_plane_synthetic_diagonal_combo
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mwb.lib.consistency import real_systems
from mwb.lib.spa_plane import build_af_seg_matrices, knowledge_lines, tradeoff_line
from mwb.lib.synthetic_scorers import DIAGONAL_ADDITIVE_MEAN_DIAL_GRID, all_synthetic_family_points

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'output', 'synthetic_scorers')

_TRADEOFF_COLOR = '#000000'
_ADEQUACY_COLOR = '#1f4fd8'
_FLUENCY_COLOR = '#d81f2f'

# Sparse (20-point) rebuild of mwb.lib.synthetic_scorers.ABTJ_J_DIAL_GRID's own
# (C, base) formula -- same rationale as plot_spa_plane_synthetic_combo.py:
# the 201-point constant is far denser than a single static plot needs.
N_DIAL_POINTS = 20
ABTJ_J_DIAL_GRID_SPARSE = tuple(round(-3.5 + 4.0 ** (k / (N_DIAL_POINTS - 1)), 10) for k in range(N_DIAL_POINTS))

# D's own dial grid, plotting-only: DIAGONAL_ADDITIVE_MEAN_DIAL_GRID's same
# [-0.5, 0.5] construction at half precision -- step 0.04 instead of the
# canonical 0.02 (26 points instead of 51). Local to this script; the
# library default is untouched.
DIAGONAL_DIAL_GRID_SPARSE = tuple(round(float(x), 2) for x in np.arange(-0.5, 0.5001, 0.04))

PANELS = [('a', 'hLEPOR', 'ende21'), ('b', 'MetricX-24-Hybrid', 'ende24')]

# D's connecting line is dotted, like every family in the parent script;
# its marker is a diamond so it isn't confused with T's '+' or J's circle.
_STYLE = {
    'J': {'marker': 'o', 'linestyle': ':'},
    'T': {'marker': '+', 'linestyle': ':', 'cmap': 'Greens'},
    'D': {'marker': 'v', 'linestyle': ':'},
}
MARKER_LINEWIDTH = {'T': 6.0}  # '+' is stroke-only -- default scatter linewidth reads as thin
_LABEL_DESC = {
    'D': 'Adequacy-fluency',
    'T': 'MQM adherence',
    'J': 'Explainability by MQM',
}
_LEGEND_COLOR = {
    'D': mcolors.LinearSegmentedColormap.from_list('diagonal_rpb', [_FLUENCY_COLOR, _ADEQUACY_COLOR])(0.75),
    'T': plt.get_cmap('Greens')(0.75),
    'J': (1.0, 0.65, 0.0),
}
TITLE_FONTSIZE = 60
LABEL_FONTSIZE = 64
TICK_FONTSIZE = 54
LEGEND_FONTSIZE = 48
TICK_PAD = 25
LABEL_PAD = -50
VERTEX_FONTSIZE = 44
VERTEX_MARKER_SIZE = 32
FAMILY_MARKER_SIZE = 640
LEGEND_MARKER_SIZE = 36
SENTINEL_LINEWIDTH = 5
Y_LOW_TICK = 0.5  # explicit, smaller than the default locator's 0.6 -- sits
                  # lower on the axis, clear of the "SPA" in the ylabel

# Fixed red(-1)->purple(0)->blue(+1) ramp for D -- a straight 2-stop
# interpolation between this project's own Fluency/Adequacy colors already
# passes through purple at the midpoint, no third stop needed.
_DIAGONAL_CMAP = mcolors.LinearSegmentedColormap.from_list('diagonal_rpb', [_FLUENCY_COLOR, _ADEQUACY_COLOR])


def _diagonal_color(dial: float):
  """D's color for one dial value: t = (dial + 1) / 2 against FIXED anchors
  dial=-1 (red) / dial=+1 (blue), not this sweep's own (smaller) actual
  range -- DIAGONAL_DIAL_GRID_SPARSE only reaches +-0.5, so t only spans
  [0.25, 0.75] here, deliberately muted/purple-heavy rather than reaching
  the ramp's own saturated red/blue ends."""
  t = (dial + 1.0) / 2.0
  return _DIAGONAL_CMAP(t)


def _family_colors_T(n: int):
  """Light->dark green ramp for T, by rank (n points total) -- unchanged
  from plot_spa_plane_synthetic_combo.py's own _family_colors."""
  if n <= 1:
    n = 2
  cmap = plt.get_cmap(_STYLE['T']['cmap'])
  return [cmap(0.25 + 0.75 * r / (n - 1)) for r in range(n)]


def _family_colors_J(n: int):
  """Bespoke gold ramp for J, by rank -- unchanged from
  plot_spa_plane_synthetic_combo.py's own _family_colors."""
  if n <= 1:
    n = 2
  return [(1.0, 1.0 - 0.35 * r / (n - 1), 0.0) for r in range(n)]


def _line_legend_handles():
  return [
      Line2D([0], [0], color=_TRADEOFF_COLOR, lw=SENTINEL_LINEWIDTH, label='Tradeoff'),
      Line2D([0], [0], color=_ADEQUACY_COLOR, lw=SENTINEL_LINEWIDTH, label='Adequacy-knowledge'),
      Line2D([0], [0], color=_FLUENCY_COLOR, lw=SENTINEL_LINEWIDTH, label='Fluency-knowledge'),
  ]


def _marker_legend_handles():
  return [
      Line2D([0], [0], color=_LEGEND_COLOR[label], marker=_STYLE[label]['marker'],
             linestyle=_STYLE[label]['linestyle'], markersize=LEGEND_MARKER_SIZE, label=_LABEL_DESC[label],
             markeredgewidth=MARKER_LINEWIDTH.get(label, 1.0))
      for label in ('D', 'T', 'J')
  ]


# Per-vertex label defaults; PANEL_VERTEX_OVERRIDES below patches individual
# (panel, vertex) entries where the default offset collides with that
# panel's own curve shape. Unchanged from plot_spa_plane_synthetic_combo.py.
_VERTEX_ANNOTATIONS = {
    'adequacy': dict(text='Adequacy MQM', dx=-50, dy=-60, ha='right', va='top'),
    'fluency': dict(text='Fluency MQM', dx=-40, dy=-80, ha='right', va='top'),
    'noise': dict(text='Random noise', dx=16, dy=-16, ha='left', va='top'),
}
PANEL_VERTEX_OVERRIDES = {
    'a': {
        'fluency': dict(text='Fluency MQM', dx=-15, dy=-130, ha='right', va='top'),
    },
    'b': {
        'noise': dict(text='Random noise', dx=+100, dy=-70, ha='right', va='top'),
    },
}
PANEL_VERTEX_SKIP = {'a': {'noise'}}


if __name__ == '__main__':
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  panel_data = {}
  for letter, donor, dataset in PANELS:
    systems = real_systems(dataset, root=ROOT, excl_missing_seg_granular_mqm=True)
    if not systems:
      sys.exit(f'{dataset}: 0 real systems, nothing to plot')
    a_seg, b_seg = build_af_seg_matrices(dataset, systems, root=ROOT)
    if a_seg is None:
      sys.exit(f'{dataset}: too few jointly-valid adequacy/fluency segments')

    print(f'{dataset}/{donor}: K={len(systems)} systems -- building sentinel lines', file=sys.stderr)
    tradeoff_pts = tradeoff_line(a_seg, b_seg)
    _, a_mean, _, b_mean = knowledge_lines(a_seg, b_seg)

    print(f'{dataset}/{donor}: building synthetic families (D + T + J)', file=sys.stderr)
    fam_am = all_synthetic_family_points(
        dataset, systems, root=ROOT, dial_grid=DIAGONAL_ADDITIVE_MEAN_DIAL_GRID, synthesis='additive_mean',
        diagonal_dial_grid=DIAGONAL_DIAL_GRID_SPARSE).get(donor)
    fam_off = all_synthetic_family_points(dataset, systems, root=ROOT, dial_grid=ABTJ_J_DIAL_GRID_SPARSE,
                                           synthesis='offset').get(donor)
    if fam_am is None or fam_off is None:
      sys.exit(f'{dataset}: donor {donor!r} has no usable synthetic family')
    families = {'D': fam_am['D'], 'T': fam_am['T'], 'J': fam_off['J']}

    panel_data[letter] = dict(donor=donor, dataset=dataset, tradeoff=tradeoff_pts, a_mean=a_mean, b_mean=b_mean,
                               families=families, adequacy_vertex=tradeoff_pts[-1], fluency_vertex=tradeoff_pts[0],
                               noise_vertex=a_mean[0], allmqm_point=tradeoff_pts[len(tradeoff_pts) // 2])

  all_xy = []
  for d in panel_data.values():
    all_xy += list(d['tradeoff']) + list(d['a_mean']) + list(d['b_mean'])
    for dial_pts in d['families'].values():
      all_xy += list(dial_pts.values())
  xs_all, ys_all = zip(*all_xy)
  pad = 0.015
  xlim = (max(0.0, min(xs_all) - pad), 1.0)
  ylim = (max(0.0, min(ys_all) - pad), 1.0)

  fig, axes = plt.subplots(1, 2, figsize=(34, 17), sharex=True, sharey=True)

  for ax, (letter, _, _) in zip(axes, PANELS):
    d = panel_data[letter]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xticks_in = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    x_low = xticks_in[0]
    ax.set_xticks([x_low, 1.0])
    ax.set_xticklabels([f'{x_low:g}', '1'])
    ax.set_yticks([Y_LOW_TICK, 1.0])
    ax.set_yticklabels([f'{Y_LOW_TICK:g}', '1'])

    for pts, color in ((d['tradeoff'], _TRADEOFF_COLOR), (d['a_mean'], _ADEQUACY_COLOR), (d['b_mean'], _FLUENCY_COLOR)):
      xs, ys = zip(*pts)
      ax.plot(xs, ys, color=color, linewidth=SENTINEL_LINEWIDTH, zorder=3)

    ax.plot(*d['allmqm_point'], marker='o', color='black', markersize=VERTEX_MARKER_SIZE, linestyle='none', zorder=5)

    vertex_specs = {**_VERTEX_ANNOTATIONS, **PANEL_VERTEX_OVERRIDES.get(letter, {})}
    skip = PANEL_VERTEX_SKIP.get(letter, set())
    for key, xy in (('adequacy', d['adequacy_vertex']), ('fluency', d['fluency_vertex']),
                     ('noise', d['noise_vertex'])):
      if key in skip:
        continue
      spec = vertex_specs[key]
      ax.annotate(spec['text'], xy=xy, xytext=(spec['dx'], spec['dy']), textcoords='offset points',
                  fontsize=VERTEX_FONTSIZE, ha=spec['ha'], va=spec['va'], zorder=5,
                  arrowprops=dict(arrowstyle='->', color='black', lw=2.5, shrinkA=0, shrinkB=20))

    for label in ('D', 'T', 'J'):
      dial_pts = sorted(d['families'][label].items())  # dial order, for the connecting line
      xs = [xy[0] for _, xy in dial_pts]
      ys = [xy[1] for _, xy in dial_pts]
      if label == 'D':
        colors = [_diagonal_color(dial) for dial, _ in dial_pts]
      elif label == 'T':
        colors = _family_colors_T(len(dial_pts))
      else:
        colors = _family_colors_J(len(dial_pts))
      style = _STYLE[label]
      ax.plot(xs, ys, color='#999999', linewidth=0.8, linestyle=style['linestyle'], zorder=3)
      ax.scatter(xs, ys, s=FAMILY_MARKER_SIZE, c=colors, marker=style['marker'], edgecolor='none', zorder=4,
                 linewidths=MARKER_LINEWIDTH.get(label, 1.0))

    ax.set_xlabel('SPA w.r.t. Fluency MQM', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_title(f'({letter}) {d["donor"]} @ {d["dataset"]}', fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, pad=TICK_PAD)
    for side in ('top', 'right', 'bottom', 'left'):
      ax.spines[side].set_visible(True)

  axes[0].set_ylabel('SPA w.r.t. Adequacy MQM', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
  axes[1].tick_params(labelleft=False)

  axes[0].legend(handles=_line_legend_handles(), loc='lower right', frameon=True, framealpha=0.9,
                 edgecolor='none', fontsize=LEGEND_FONTSIZE)
  axes[1].legend(handles=_marker_legend_handles(), loc='lower right', frameon=True, framealpha=0.9,
                 edgecolor='none', fontsize=LEGEND_FONTSIZE)

  fig.tight_layout(pad=0.5, w_pad=0.05)

  plot_path = os.path.join(ARTIFACTS_DIR, 'spa_plane_synthetic_diagonal_combo_hlepor-ende21_metricx24hybrid-ende24.pdf')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}')
