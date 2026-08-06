"""Two-panel SPA-plane figure, one donor scorer per panel, in the style of
plot_synthetic_scorer_families.py's per-donor plots (tradeoff triangle +
Adequacy/Fluency-knowledge curves + shadows, lib.spa_plane) overlaid with
that donor's synthetic A/B/T (additive_mean) and J (offset) dial families
(lib.synthetic_scorers).

Dial grids: the project's TWO-SIDED grids (lib.synthetic_scorers.
ABT_DIAL_GRID for A/B/T, ABTJ_J_DIAL_GRID for J -- the pair
compute_scorer_orientation_vs_alpha.py's --green-family ABTJ uses), each a
geometric ramp dial(i) = C + base**i:
  - A/B/T: C=-1.5, base=3 -- range [-0.5, 1.5].
  - J:     C=-3.5, base=4 -- range [-2.5, 0.5].
Both are defined at 201 points (i in [0, 0.005, ..., 1]) in lib.
synthetic_scorers -- far denser than a single static plot needs (most of
the visible movement is in the first handful of points anyway), so this
script rebuilds the SAME two formulas at N_DIAL_POINTS=20 instead of
reusing the 201-point constants directly.

Panels: (a) hLEPOR on ende21, (b) MetricX-24-Hybrid on ende24. Layout
mirrors plot_af_scatter_heen23_jazh24.py: shared x/y ranges, a single
shared y-axis on the left panel, each panel boxed on all four sides, one
legend (on the left panel only). No shadow lines (lib.spa_plane.
knowledge_lines' random-noise instances) -- only the tradeoff/knowledge
mean curves and the four synthetic families. The triangle's three vertices
are annotated directly (Adequacy MQM / Fluency MQM / Random noise) instead
of relying on the legend to convey what they are.

Usage: python codes/scripts/plot_spa_plane_synthetic_combo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.consistency import real_systems
from lib.spa_plane import build_af_seg_matrices, knowledge_lines, tradeoff_line
from lib.synthetic_scorers import all_synthetic_family_points

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_synthetic_scorers')

_TRADEOFF_COLOR = '#000000'
_ADEQUACY_COLOR = '#1f4fd8'
_FLUENCY_COLOR = '#d81f2f'

# Sparse (20-point) rebuild of lib.synthetic_scorers.ABT_DIAL_GRID /
# ABTJ_J_DIAL_GRID's own (C, base) formulas -- see module docstring.
N_DIAL_POINTS = 20
ABT_DIAL_GRID_SPARSE = tuple(round(-1.5 + 3.0 ** (k / (N_DIAL_POINTS - 1)), 10) for k in range(N_DIAL_POINTS))
ABTJ_J_DIAL_GRID_SPARSE = tuple(round(-3.5 + 4.0 ** (k / (N_DIAL_POINTS - 1)), 10) for k in range(N_DIAL_POINTS))

PANELS = [('a', 'hLEPOR', 'ende21'), ('b', 'MetricX-24-Hybrid', 'ende24')]

# Connecting line is dotted for all four families. T's marker is '+'
# (thicker stroke, MARKER_LINEWIDTH below); cmap is None for J, which gets
# its own bespoke gold ramp in _family_colors instead of a named
# Matplotlib colormap (none sits at that hue) -- T uses plain 'Greens'.
_STYLE = {
    'J': {'marker': 'o', 'linestyle': ':', 'cmap': None},
    'T': {'marker': '+', 'linestyle': ':', 'cmap': 'Greens'},
    'A': {'marker': '^', 'linestyle': ':', 'cmap': 'Blues'},
    'B': {'marker': '^', 'linestyle': ':', 'cmap': 'Reds'},
}
MARKER_LINEWIDTH = {'T': 6.0}  # '+' is stroke-only -- default scatter linewidth reads as thin
_LABEL_DESC = {
    'A': 'Adequacy family',
    'B': 'Fluency family',
    'T': 'Total family',
    'J': 'Orthogonal family',
}
# One representative color per family, for the legend proxy only -- the
# points/lines themselves still use _family_colors' light->dark ramp.
_LEGEND_COLOR = {
    'A': plt.get_cmap('Blues')(0.75),
    'B': plt.get_cmap('Reds')(0.75),
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

# Per-vertex label defaults; PANEL_VERTEX_OVERRIDES below patches individual
# (panel, vertex) entries where the default offset collides with that
# panel's own curve shape.
_VERTEX_ANNOTATIONS = {
    'adequacy': dict(text='Adequacy MQM', dx=-50, dy=-60, ha='right', va='top'),
    'fluency': dict(text='Fluency MQM', dx=-40, dy=-80, ha='right', va='top'),
    'noise': dict(text='Random noise', dx=16, dy=-16, ha='left', va='top'),
}
PANEL_VERTEX_OVERRIDES = {
    'a': {
        # Pushed further down (clears the red Fluency-knowledge curve,
        # which sits closer to this vertex for hLEPOR than for MetricX)
        # and closer to the vertex horizontally ("right" of the default).
        'fluency': dict(text='Fluency MQM', dx=-15, dy=-130, ha='right', va='top'),
    },
    'b': {
        # Toward the lower-left of the subplot (down and left of the
        # vertex) -- clear of the marker legend, which sits lower-right.
        'noise': dict(text='Random noise', dx=+100, dy=-70, ha='right', va='top'),
    },
}
# No good spot was found for this one in panel (a) that didn't cross a
# curve or collide with the lines legend -- dropped there entirely rather
# than force an awkward placement.
PANEL_VERTEX_SKIP = {'a': {'noise'}}


def _family_colors(label: str, n: int):
  """Light->dark color per point, by rank (n points total). J gets a
  bespoke gold ramp (no single-hue Colormap sits there); A/B/T use their
  named Matplotlib sequential colormap, starting at t=0.25 rather than 0
  so the lightest point stays visible against a white background."""
  if n <= 1:
    n = 2
  if label == 'J':
    return [(1.0, 1.0 - 0.35 * r / (n - 1), 0.0) for r in range(n)]
  cmap = plt.get_cmap(_STYLE[label]['cmap'])
  return [cmap(0.25 + 0.75 * r / (n - 1)) for r in range(n)]


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
      for label in ('A', 'B', 'T', 'J')
  ]


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
    # Shadows (the random-noise instance curves) are dropped from this
    # figure entirely -- only the tradeoff/knowledge MEAN curves are drawn.
    _, a_mean, _, b_mean = knowledge_lines(a_seg, b_seg)

    print(f'{dataset}/{donor}: building synthetic families on the sparse two-sided dial grids', file=sys.stderr)
    fam_am = all_synthetic_family_points(dataset, systems, root=ROOT, dial_grid=ABT_DIAL_GRID_SPARSE,
                                          synthesis='additive_mean').get(donor)
    fam_off = all_synthetic_family_points(dataset, systems, root=ROOT, dial_grid=ABTJ_J_DIAL_GRID_SPARSE,
                                           synthesis='offset').get(donor)
    if fam_am is None or fam_off is None:
      sys.exit(f'{dataset}: donor {donor!r} has no usable synthetic family')
    families = {'A': fam_am['A'], 'B': fam_am['B'], 'T': fam_am['T'], 'J': fam_off['J']}

    # Triangle vertices (lib.spa_plane.tradeoff_line/knowledge_lines'
    # docstrings): tradeoff_pts[-1] (lam=1, score=a_seg exactly) is the
    # exact Adequacy MQM vertex (*, 1.0); tradeoff_pts[0] (lam=0, score=
    # b_seg exactly) is the exact Fluency MQM vertex (1.0, *); a_mean[0]
    # and b_mean[0] (both lam=0, i.e. pure shared noise, no signal) are
    # EXACTLY the same point -- the Random-noise vertex. tradeoff_pts'
    # MIDPOINT (lam=0.5, score = 0.5*a_seg + 0.5*b_seg, proportional to
    # a_seg+b_seg -- SPA is invariant to a positive rescaling, so this is
    # exactly SPA(All MQM) = SPA(Adequacy MQM + Fluency MQM)) sits ON the
    # black tradeoff line, roughly in its middle.
    panel_data[letter] = dict(donor=donor, dataset=dataset, tradeoff=tradeoff_pts, a_mean=a_mean, b_mean=b_mean,
                               families=families, adequacy_vertex=tradeoff_pts[-1], fluency_vertex=tradeoff_pts[0],
                               noise_vertex=a_mean[0], allmqm_point=tradeoff_pts[len(tradeoff_pts) // 2])

  # Shared x/y ranges across BOTH panels, so the two donors stay visually
  # comparable on the same SPA plane.
  all_xy = []
  for d in panel_data.values():
    all_xy += list(d['tradeoff']) + list(d['a_mean']) + list(d['b_mean'])
    for dial_pts in d['families'].values():
      all_xy += list(dial_pts.values())
  xs_all, ys_all = zip(*all_xy)
  # Axis max is STRICTLY 1.0 -- the SPA plane's own natural ceiling, and
  # exactly where the Adequacy/Fluency MQM vertices sit -- with no pad
  # beyond it at all. Every vertex label is placed BELOW/BESIDE its vertex
  # (in the triangle's blank exterior wedge on that side) rather than
  # above/beyond it, so nothing needs to extend past that ceiling to stay
  # readable. The lower/left side keeps a hairline pad so vertex markers
  # sitting there aren't clipped by the spine.
  pad = 0.015
  xlim = (max(0.0, min(xs_all) - pad), 1.0)
  ylim = (max(0.0, min(ys_all) - pad), 1.0)

  fig, axes = plt.subplots(1, 2, figsize=(34, 17), sharex=True, sharey=True)

  for ax, (letter, _, _) in zip(axes, PANELS):
    d = panel_data[letter]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Only the min/max tick of each axis (plot_af_scatter_heen23_jazh24.py's
    # convention). x keeps the default locator's own smallest in-range
    # tick; y is pinned to Y_LOW_TICK explicitly (0.5, not the locator's
    # own 0.6) so it sits lower on the axis, clear of "SPA" in the ylabel.
    # Both top ticks are pinned to the exact 1.0 ceiling.
    xticks_in = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    x_low = xticks_in[0]
    ax.set_xticks([x_low, 1.0])
    ax.set_xticklabels([f'{x_low:g}', '1'])
    ax.set_yticks([Y_LOW_TICK, 1.0])
    ax.set_yticklabels([f'{Y_LOW_TICK:g}', '1'])

    for pts, color in ((d['tradeoff'], _TRADEOFF_COLOR), (d['a_mean'], _ADEQUACY_COLOR), (d['b_mean'], _FLUENCY_COLOR)):
      xs, ys = zip(*pts)
      ax.plot(xs, ys, color=color, linewidth=SENTINEL_LINEWIDTH, zorder=3)

    # Single black bullet marker at SPA(All MQM) -- the tradeoff line's own
    # midpoint (lam=0.5) -- NOT at the Adequacy/Fluency MQM vertices, which
    # are pure single-aspect extremes, not All MQM.
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

    for label in ('A', 'B', 'T', 'J'):
      dial_pts = sorted(d['families'][label].items())  # dial order, for the connecting line
      xs = [xy[0] for _, xy in dial_pts]
      ys = [xy[1] for _, xy in dial_pts]
      colors = _family_colors(label, len(dial_pts))
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

  # Two legends, split by kind, one inside each panel -- at this font/marker
  # scale a single combined legend no longer fits in either panel's blank
  # corner without covering data. Both sit in the lower-right blank wedge
  # (below the red Fluency-knowledge curve), which is where the vertex
  # labels leave the most room clear in both panels.
  axes[0].legend(handles=_line_legend_handles(), loc='lower right', frameon=True, framealpha=0.9,
                 edgecolor='none', fontsize=LEGEND_FONTSIZE)
  axes[1].legend(handles=_marker_legend_handles(), loc='lower right', frameon=True, framealpha=0.9,
                 edgecolor='none', fontsize=LEGEND_FONTSIZE)

  fig.tight_layout(pad=0.5, w_pad=0.05)

  plot_path = os.path.join(ARTIFACTS_DIR, 'spa_plane_synthetic_combo_hlepor-ende21_metricx24hybrid-ende24.pdf')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}')
