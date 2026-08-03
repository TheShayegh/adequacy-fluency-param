"""SPA Plane augmented with synthetic scorer families (material/synthetic_
scorer_construction.md, lib.synthetic_scorers): one plot per real donor
scorer (all 27+ donors overlaid on a single plane made the per-donor
families impossible to tell apart), each showing that donor's synthetic
dial sweep(s) -- under one or more synthesis mechanisms (lib.
synthetic_scorers.SYNTHESIS_GENERATORS: 'offset', 'additive',
'additive_mean') overlaid together -- on the same SPA plane as lib.
spa_plane / plot_spa_plane.py -- x = SPA vs. Fluency MQM, y = SPA vs.
Adequacy MQM.

Default behavior (no flags): only the THIRD family per synthesis kind is
plotted (the aspect-only A/B dialed families are omitted), all three
synthesis kinds active:

  - 'offset': J (Joint), dialed on the (Adequacy, Fluency) pair jointly
    (lib.synthetic_scorers.donor_family_joint) -- default dial grid is
    lib.synthetic_scorer_orientation.NEG_J_DIAL_GRID (--offset-dial-grid
    neg, the project-wide default as of this session): dial = 0.5 - 2**i
    for i in [-1, 2] by 0.1 (31 points, dial=0 falling out exactly at
    i=-1, the sweep's own least-negative end) -- ranges from 0.0 (i=-1)
    down to -3.5 (i=2), every point <=0, densely sampled near the donor
    and increasingly sparse toward the negative extreme. Meant to pair
    with --additive-mean-ab (below) for an overlay that shows
    additive_mean's own A/B next to offset's negative-leaning J instead of
    additive_mean's T (see compute_scorer_orientation_vs_alpha.py's
    --green-family Jneg for the alpha-curve analogue of this same
    pairing, sharing this exact grid). --offset-dial-grid extended
    switches to lib.synthetic_scorers.EXTENDED_OFFSET_DIAL_GRID instead:
    dial in [-2, 1] by 0.1 (31 points; donor_family_joint's formula is
    well-defined for any dial, and dial=0 is always the fixed black
    anchor regardless of range).
  - 'additive'/'additive_mean': T (AllMQM), dialed on All MQM instead of
    either aspect (lib.synthetic_scorers.donor_family_spa_points,
    restricted to these two synthesis kinds for the same reason lib.
    synthetic_scorer_alpha_grid's T-family is -- additive/additive_mean
    inject the aspect directly rather than going through offset's
    m/e/ebar_k decomposition, which is what makes T orientation-neutral).
    additive_mean's dial grid is lib.synthetic_scorers.EXTENDED_ADDITIVE_
    MEAN_DIAL_GRID (the project-wide default for that synthesis kind as of
    this session, lib.synthetic_scorers.default_dial_preset): dial = 2**i
    - 2**-10 for i in [-10, 3] by 0.5 (27 points, dial=0 falling out
    exactly at i=-10, up to 2**3-2**-10~=8 at i=3). Plain 'additive' keeps
    a smaller sweep, _ADDITIVE_DIAL_GRID_WITH_ZERO (0.0 prepended to lib.
    synthetic_scorers.GEOM_DIAL_GRID, dial in [0, 1]) -- both geometric
    (packed toward 0) since all three saturate almost immediately on a
    linear grid (most of the SPA-plane movement happens between dial=0 and
    dial=0.1); additive_mean's range is wider because its endpoint is now
    standardized to the donor's own between-system spread (lib.
    synthetic_scorers.donor_family_additive_mean's standardized=True
    default), so GEOM_DIAL_GRID's dial=1 ceiling is only ~1 SD's worth of
    perturbation, no longer genuinely extreme the way it was against the
    old raw-abar_k scale. Neither is a --flag here (there is exactly one
    dial grid per kind per run, not a family of dial-preset variants).

--kinds (default 'offset,additive,additive_mean') restricts which
synthesis kinds are drawn at all -- e.g. --kinds offset,additive_mean
drops plain 'additive' entirely.

--additive-mean-ab draws additive_mean's own A (adequacy-dialed) and B
(fluency-dialed) families INSTEAD of its T family -- still on
EXTENDED_ADDITIVE_MEAN_DIAL_GRID. Has no effect on 'additive' (plain),
which always draws T when active.

Dots carry no scorer-name label and are colored by dial:

  - J (offset, dial grid per --offset-dial-grid, always the same fixed
    grid, so raw dial value is used directly, not rank): dial>=0 ramps
    black->magenta, normalized by that grid's own max positive dial;
    dial<0 ramps black->orange instead, normalized by that grid's own max
    NEGATIVE magnitude -- each side normalized independently (by whatever
    _family_colors finds in the actual dial_pts passed in, not a
    hardcoded constant), so --offset-dial-grid extended (max +1/-2) and
    neg (max +0.5/-3) each reach full saturation at their own extreme
    rather than one grid choice under- or over-saturating relative to the
    other's fixed normalization.
  - T (additive/additive_mean, dial in [0, 8] via EXTENDED_ADDITIVE_MEAN_
    DIAL_GRID for additive_mean, [0, 1] for plain additive): black -> full
    green, colored by RANK in the sorted dial grid, not raw value --
    identical to raw-value coloring on a linear grid, but the only sane
    choice here since the grid's raw values cluster near 0.
  - A/B (--additive-mean-ab, same dial grid as T): black -> full blue (A)
    or black -> full red (B), also by rank, same reasoning as T -- matching
    this project's adequacy=blue/fluency=red convention (lib.spa_plane).

Synthesis kind is encoded orthogonally via marker shape and connecting-line
style (offset: circle/solid, additive: square/dashed, additive_mean:
triangle/dotted); when --additive-mean-ab makes additive_mean draw TWO
families (A and B) they share that one marker/linestyle, distinguished
from each other by color (blue vs. red) instead.

Usage: python codes/scripts/plot_synthetic_scorer_families.py [--dataset ende21]
           [--kinds offset,additive,additive_mean] [--offset-dial-grid neg|extended]
           [--additive-mean-ab]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.consistency import real_systems
from lib.spa_plane import build_af_seg_matrices, knowledge_lines, tradeoff_line
from lib.synthetic_scorer_orientation import NEG_J_DIAL_GRID
from lib.synthetic_scorers import (
    EXTENDED_ADDITIVE_MEAN_DIAL_GRID, EXTENDED_OFFSET_DIAL_GRID, GEOM_DIAL_GRID, all_synthetic_family_points,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_synthetic_scorers')

_TRADEOFF_COLOR = '#000000'
_ADEQUACY_COLOR = '#1f4fd8'
_FLUENCY_COLOR = '#d81f2f'

SYNTHESIS_KINDS = ('offset', 'additive', 'additive_mean')
# Shape + connecting-line style per synthesis kind, orthogonal to the
# family-color encoding -- so a single plot can overlay every active kind
# (and, under --additive-mean-ab, TWO families out of the same kind)
# without the family-color encoding becoming ambiguous.
_SYNTHESIS_STYLE = {
    'offset': {'marker': 'o', 'linestyle': '-'},
    'additive': {'marker': 's', 'linestyle': '--'},
    'additive_mean': {'marker': '^', 'linestyle': ':'},
}

# additive_mean dial grid: lib.synthetic_scorers.EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID (dial = 2**i - 2**-10 for i in [-10, 3] by 0.5, 27 points,
# dial=0 falling out of the formula itself at i=-10) -- the project-wide
# default for additive_mean as of this session (lib.synthetic_scorers.
# default_dial_preset), no longer a copy local to this script. Plain
# 'additive' keeps its own smaller GEOM_DIAL_GRID-based sweep (still
# project default for that kind), with an explicit 0.0 prepended since
# GEOM_DIAL_GRID's own smallest point (2**-10) is never the exact donor --
# EXTENDED_ADDITIVE_MEAN_DIAL_GRID needs no such prepend, since its "-
# 2**-10" shift already anchors dial=0 exactly.
_ADDITIVE_DIAL_GRID_WITH_ZERO = (0.0,) + GEOM_DIAL_GRID

# --offset-dial-grid choices -> the actual grid. 'neg' is lib.
# synthetic_scorer_orientation.NEG_J_DIAL_GRID (dial = 0.5 - 2**i for i in
# [-1, 2] by 0.1, 31 points, every value <=0, dial=0 at i=-1) -- no longer
# a copy local to this script, and now the DEFAULT (see --offset-dial-grid
# below), matching the range this project settled on for the offset/J
# side of the same overlay EXTENDED_ADDITIVE_MEAN_DIAL_GRID serves on the
# additive_mean/A-B side.
_OFFSET_DIAL_GRIDS = {'extended': EXTENDED_OFFSET_DIAL_GRID, 'neg': NEG_J_DIAL_GRID}


def _labels_for_kind(syn: str, additive_mean_ab: bool) -> list[str]:
  """Which family label(s) (lib.synthetic_scorers.donor_family_spa_points'
  keys) to draw for one active synthesis kind -- always a single label
  except additive_mean under --additive-mean-ab, which draws two (A and B
  instead of T)."""
  if syn == 'offset':
    return ['J']
  if syn == 'additive_mean' and additive_mean_ab:
    return ['A', 'B']
  return ['T']


def _plot_line(ax, pts, color, label, lw=2.2, zorder=3):
  xs, ys = zip(*pts)
  ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, label=label)


def _plot_shadows(ax, shadows, color, zorder=2):
  for pts in shadows:
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25, zorder=zorder)


def _family_colors(label: str, dial_pts: list[tuple[float, tuple[float, float]]]):
  """RGB color per point in `dial_pts` (dial-sorted [(dial, (x, y)), ...]),
  keyed by family `label` ('J', 'T', 'A', or 'B') -- see module docstring
  for the full color scheme. J uses the raw dial value directly, each side
  (dial>=0 / dial<0) normalized independently by the max magnitude actually
  present in `dial_pts` on that side -- not a hardcoded constant, since
  _OFFSET_DIAL_GRIDS' two choices no longer share a common range (extended
  maxes out at +1/-2, neg at +0.5/-3) -- so whichever grid is active always
  reaches full black->magenta / black->orange saturation at its own
  extreme. dial=0 is still exactly black either way. T/A/B use rank in
  `dial_pts` instead of raw dial value, since their geometric grid's raw
  values cluster near 0."""
  if label == 'J':
    dials = [d for d, _ in dial_pts]
    pos_dials = [d for d in dials if d >= 0]
    neg_dials = [d for d in dials if d < 0]
    max_pos = max(pos_dials) if pos_dials else 1.0
    max_neg = max(-d for d in neg_dials) if neg_dials else 1.0
    colors = []
    for dial, _ in dial_pts:
      if dial >= 0:
        t = dial / max_pos if max_pos > 0 else 0.0
        colors.append((t, 0.0, t))  # black -> magenta
      else:
        mag = (-dial) / max_neg if max_neg > 0 else 0.0
        colors.append((mag, mag * 0.55, 0.0))  # black -> orange
    return colors
  n = max(len(dial_pts) - 1, 1)
  if label == 'A':
    return [(0.0, 0.0, r / n) for r, _ in enumerate(dial_pts)]  # black -> blue, by rank
  if label == 'B':
    return [(r / n, 0.0, 0.0) for r, _ in enumerate(dial_pts)]  # black -> red, by rank
  return [(0.0, r / n, 0.0) for r, _ in enumerate(dial_pts)]  # T: black -> green, by rank


_LABEL_COLOR_DESC = {
    'J': 'Joint family J, offset (black->orange/magenta)',
    'T': 'AllMQM family T, additive* (black->green, by rank)',
    'A': 'Adequacy family A, additive_mean (black->blue, by rank)',
    'B': 'Fluency family B, additive_mean (black->red, by rank)',
}


def _legend_handles(kind_labels: list[tuple[str, str]]):
  """Sentinel-line handles are added via _plot_line's own `label=` (kept on
  the automatic legend); these are the extra proxies for the two encodings
  _SYNTHESIS_STYLE and the per-label colors add, which no single plotted
  artist represents on its own. kind_labels: [(synthesis, label), ...] for
  every trace actually drawn (order-preserving, duplicates in `synthesis`
  collapsed for the style legend)."""
  kinds_seen = []
  for syn, _ in kind_labels:
    if syn not in kinds_seen:
      kinds_seen.append(syn)
  synthesis_handles = [
      Line2D([0], [0], color='#666666', marker=_SYNTHESIS_STYLE[syn]['marker'],
             linestyle=_SYNTHESIS_STYLE[syn]['linestyle'], markersize=6, label=f'synthesis={syn}')
      for syn in kinds_seen
  ]
  color_by_label = {'J': '#8f1f9e', 'T': '#1f9d3a', 'A': _ADEQUACY_COLOR, 'B': _FLUENCY_COLOR}
  labels_seen = []
  for _, label in kind_labels:
    if label not in labels_seen:
      labels_seen.append(label)
  color_handles = [
      Line2D([0], [0], color=color_by_label[label], marker='o', linestyle='none', markersize=6,
             label=_LABEL_COLOR_DESC[label])
      for label in labels_seen
  ]
  return synthesis_handles + color_handles


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--kinds', type=str, default='offset,additive,additive_mean',
                       help="comma-separated subset of ('offset', 'additive', 'additive_mean') to draw -- "
                            "e.g. 'offset,additive_mean' to drop plain 'additive' entirely.")
  parser.add_argument('--offset-dial-grid', choices=['extended', 'neg'], default='neg',
                       help="'neg' (default, project-wide default as of this session): lib.synthetic_scorer_"
                            'orientation.NEG_J_DIAL_GRID, dial = 0.5 - 2**i for i in [-1, 2] by 0.1 (dial=0 at '
                            "i=-1, ranging down to -3.5 -- see module docstring). 'extended': lib."
                            'synthetic_scorers.EXTENDED_OFFSET_DIAL_GRID, dial in [-2, 1] by 0.1 instead.')
  parser.add_argument('--additive-mean-ab', action=argparse.BooleanOptionalAction, default=False,
                       help="draw additive_mean's own A/B families instead of its T family (see module "
                            'docstring). No effect on plain \'additive\', which always draws T when active.')
  args = parser.parse_args()
  base = args.dataset
  kinds = [k.strip() for k in args.kinds.split(',') if k.strip()]
  for k in kinds:
    if k not in SYNTHESIS_KINDS:
      sys.exit(f'--kinds: {k!r} is not one of {SYNTHESIS_KINDS}')
  dial_grid_for = {
      'offset': _OFFSET_DIAL_GRIDS[args.offset_dial_grid],
      'additive': _ADDITIVE_DIAL_GRID_WITH_ZERO,
      'additive_mean': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  }
  labels_for = {syn: _labels_for_kind(syn, args.additive_mean_ab) for syn in kinds}

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to plot')

  a_seg, b_seg = build_af_seg_matrices(base, systems, root=ROOT)
  if a_seg is None:
    sys.exit(f'{base}: too few jointly-valid adequacy/fluency segments')

  print(f'{base}: K={len(systems)} systems -- building sentinel lines', file=sys.stderr)
  tradeoff_pts = tradeoff_line(a_seg, b_seg)
  a_shadows, a_mean, b_shadows, b_mean = knowledge_lines(a_seg, b_seg)

  print(f'{base}: building synthetic scorer families ({len(kinds)} synthesis kinds x donor scan)',
        file=sys.stderr)
  t0 = time.time()
  families_by_synthesis = {}
  for syn in kinds:
    dial_grid = dial_grid_for[syn]
    families_by_synthesis[syn] = all_synthetic_family_points(base, systems, root=ROOT,
                                                               dial_grid=dial_grid, synthesis=syn)
    labels = labels_for[syn]
    n_pts = sum(len(f.get(label, {})) for f in families_by_synthesis[syn].values() for label in labels)
    print(f'{base}: synthesis={syn} ({"/".join(labels)}-family): {len(families_by_synthesis[syn])} donors, '
          f'{n_pts} points, dial grid ({len(dial_grid)} pts) {dial_grid} ({time.time() - t0:.1f}s elapsed)',
          file=sys.stderr)

  # Donors common to every synthesis kind's output -- in practice identical
  # sets, since candidate discovery and the segment-coverage mask don't
  # depend on synthesis, but intersect defensively anyway.
  donor_names = sorted(set.intersection(*(set(f) for f in families_by_synthesis.values())))
  if not donor_names:
    sys.exit(f'{base}: no donor has usable synthetic families under every synthesis kind')

  # (synthesis, label) pairs actually drawn, in a stable order -- shared by
  # the axis-limit scan, the legend, and the per-donor plotting loop.
  kind_labels = [(syn, label) for syn in kinds for label in labels_for[syn]]

  # Shared axis limits across every donor's plot AND every (synthesis,
  # label) pair, so the per-donor figures stay visually comparable --
  # derived from the sentinel lines plus every donor's own points, not
  # just the one being drawn.
  all_xy = list(tradeoff_pts) + list(a_mean) + list(b_mean)
  for syn, label in kind_labels:
    for fam in families_by_synthesis[syn].values():
      all_xy.extend(fam.get(label, {}).values())
  for shadows in (a_shadows, b_shadows):
    for pts in shadows:
      all_xy.extend(pts)
  xs_all, ys_all = zip(*all_xy)
  pad = 0.05
  xlo, xhi = max(0.0, min(xs_all) - pad), min(1.05, max(xs_all) + pad)
  ylo, yhi = max(0.0, min(ys_all) - pad), min(1.05, max(ys_all) + pad)

  out_dir = os.path.join(ARTIFACTS_DIR, base)
  os.makedirs(out_dir, exist_ok=True)
  n = len(donor_names)
  t0 = time.time()
  for i, donor in enumerate(donor_names, 1):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    _plot_shadows(ax, a_shadows, _ADEQUACY_COLOR)
    _plot_shadows(ax, b_shadows, _FLUENCY_COLOR)
    _plot_line(ax, tradeoff_pts, _TRADEOFF_COLOR, 'Tradeoff (Adequacy <-> Fluency MQM)')
    _plot_line(ax, a_mean, _ADEQUACY_COLOR, 'Adequacy-knowledge (Adequacy MQM <-> noise)')
    _plot_line(ax, b_mean, _FLUENCY_COLOR, 'Fluency-knowledge (Fluency MQM <-> noise)')

    for syn, label in kind_labels:
      fam = families_by_synthesis[syn][donor]
      style = _SYNTHESIS_STYLE[syn]
      dial_pts = sorted(fam.get(label, {}).items())  # dial order, for the connecting line
      if not dial_pts:
        continue
      xs = [xy[0] for _, xy in dial_pts]
      ys = [xy[1] for _, xy in dial_pts]
      colors = _family_colors(label, dial_pts)
      ax.plot(xs, ys, color='#999999', linewidth=0.8, linestyle=style['linestyle'], zorder=3)
      ax.scatter(xs, ys, s=30, c=colors, marker=style['marker'], edgecolor='none', zorder=4)

    ax.set_xlabel('SPA vs. Fluency MQM')
    ax.set_ylabel('SPA vs. Adequacy MQM')
    families_desc = '/'.join(sorted({label for _, label in kind_labels}))
    ax.set_title(f'{base}: {donor} synthetic families ({families_desc} only) (K={len(systems)} systems)',
                 fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels_ = ax.get_legend_handles_labels()  # the 3 sentinel lines
    ax.legend(handles=handles + _legend_handles(kind_labels), fontsize=6.5, loc='best')
    fig.tight_layout()

    safe_donor = donor.replace('/', '_')
    plot_path = os.path.join(out_dir, f'spa_plane_synthetic_{base}_{safe_donor}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    print(f'[{i}/{n}] wrote {plot_path} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
