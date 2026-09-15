"""Heatmap of adequacy-over-fluency preference over the (ESS, realized beta)
plane, over weights covering the simplex.

--sampler picks how the simplex is covered: 'lattice' (exhaustive, uniform,
deterministic, but ESS takes only finitely many values on it) or 'simplex'
(importance sampling against uniform-on-the-simplex, so ESS bins can go
anywhere at any resolution). See mwb/lib/beta_ess_preference_grid.py's docstring
for the tradeoff and for the measured agreement between the two.

See mwb/lib/beta_ess_preference_grid.py's module docstring for the method and,
in particular, for why this is affordable (weighted SPA is a ratio of two
quadratic forms in w, so every SPA value in the run is one entry of a single
dense matrix product; the permutation tests run once per (donor, dial), never
per weight).

Unlike every other beta experiment in this project, NOTHING here is
optimized. Each lattice point of {w : sum(w) = 1, w >= 0} is just an
observation: it realizes some beta(w) and some ESS(w), which places it in a
cell, and scores some preference, which contributes to that cell's mean and
standard deviation.

Figure
------
x = ESS(w), y = realized beta(w). Each cell is split along its anti-diagonal
into two triangles: the upper-right one is colored by mean + std, the
lower-left one by mean - std, so the cell's color spread IS its dispersion.
Text "mean+-std" is drawn once, centered, on top of both triangles.
Colormap runs 0 -> red, 0.5 -> purple, 1 -> blue (0.5 = no preference).

Window and resolution
---------------------
Deliberately NOT the full plane. Defaults:

  ESS   [--ess-lo-frac 0.5 of K, K], 8 equal bins. Every dataset therefore
        gets the identical normalized window ESS/K in [0.5, 1], so the x
        axis is comparable across figures up to the factor K. The floor is
        empirical: the full range [1, K] was tried and is not usable, since
        uniform-on-the-simplex has almost no mass below ESS ~4 and those
        cells come back with n_eff in the single digits. Under --sampler
        lattice, note additionally that ESS = grid_n^2 / sum(c_k^2) is
        DISCRETE, so 8 even bins there can leave empty columns unless
        --grid-n is a large enough multiple of K; the run reports the
        attainable-value count so the mismatch is visible.
  beta  auto: the --beta-coverage (default 0.98) central quantile range of
        the points that fall in the ESS window, rounded outward to
        --beta-round (default 0.01); override with --beta-lo/--beta-hi.
        8 equal bins. Unlike ESS this range is dataset-dependent, so beta
        bin WIDTHS are not comparable across figures.

The populated region is a wedge: near ESS=K the weights are nearly uniform
so beta is pinned near beta_0, and it fans out as ESS falls. Choosing the
beta window from the ESS-windowed points (not from the whole lattice) keeps
the figure on that wedge instead of on the near-empty extremes. The
diagnostic count table exists to drive exactly this adjustment -- expect to
re-run with explicit bounds once you have seen it.

Cost control
------------
--max-per-cell (default 0 = no cap) caps how many of a cell's points are
SCORED; the cap is applied by uniform random subsample (seeded), and the
diagnostic table always reports the TRUE population count alongside the
scored count, so a capped cell is visible as such. Pass 0 to score every
point. Scoring cost is linear in the number of scored points, so this is the
knob that decides the run's wall clock.

Usage:
  python -m mwb.scripts.compute_beta_ess_preference_heatmap \
      --dataset ende24 [--grid-n 32] [--ess-bins 8] [--beta-bins 8] \
      [--ess-lo-frac 0.5] [--beta-lo 0.5 --beta-hi 1.0] [--max-per-cell N]

Writes into output/beta_ess_preference/ (see its README):
<tag>.{pdf,png} for the figure, <tag>_cells.md for the cell diagnostic
table, and <tag>.npz caching the binned statistics. <tag> defaults to
<dataset>_simplex<n_samples>k, or <dataset>_n<grid_n> under --sampler
lattice.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon

from mwb.lib.alpha import alpha_0
from mwb.lib.beta_ess_preference_grid import (
    bin_cells, build_proposal_alphas, lattice_beta_ess, lattice_size, load_donor_tables,
    max_lattice_ess, preference_scores, sample_simplex_beta_ess, weighted_mean_std)
from mwb.lib.consistency import real_scorers, real_systems
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS = os.path.join(ROOT, 'output', 'beta_ess_preference')

# Colormap stops, low -> high, evenly spaced over [0, 1]:
#   0.00 yellow | 0.25 red | 0.50 purple | 0.75 blue | 1.00 black
#
# Five hues rather than three, because the scale must stay pinned to [0, 1]
# for cross-dataset comparability while the datasets themselves occupy very
# different, narrow bands -- ende22 lives at 0.83-0.91, ted_ende crosses
# 0.5, jazh24 sits at 0.33-0.65. A three-stop ramp leaves most of those
# bands in a single hue. More stops raise the colour change per unit of
# preference everywhere along the range at once, which a warp of the scale
# cannot do: any warp has to pick a region to magnify, and would flatten
# whichever datasets live elsewhere.
#
# Purple stays at 0.5, so the neutral point is still the point where the two
# aspect hues meet rather than an arbitrary colour.
# The committed grid resolution: 8 equal-width bins on BOTH axes, for every
# dataset and under both samplers, so the cell layout is identical across
# figures and only the tick labels change.
#
# Equal-width is a deliberate choice rather than something the data dictates.
# The lattice CAN instead be binned one-per-attainable-ESS-value
# (--ess-bins 0), which is its maximum honest resolution, but that yields
# irregular, dataset-dependent bin widths, so it is kept for the lattice
# cross-check only, not for the figures.
#
# Note this fixes the bin COUNT, not the axis RANGES. The ESS range is
# [--ess-lo-frac * K, K], i.e. ESS/K in [0.5, 1] for every dataset, so ESS
# bin widths are comparable across datasets up to the factor K. The beta
# range is auto-fitted per dataset around its own beta_0 (see
# auto_beta_window), so beta bin WIDTHS still differ substantially between
# datasets -- from ~0.001 on jazh24, whose reachable balances are tightly
# concentrated, to ~0.02 on ted_ende.
DEFAULT_N_BINS = 8

PREFERENCE_CMAP_STOPS = ['#ffcc00', '#d62728', '#7b3f99', '#1f5fbf', '#000000']
PREFERENCE_CMAP = LinearSegmentedColormap.from_list(
    'adequacy_fluency', PREFERENCE_CMAP_STOPS, N=512)

# ---------------------------------------------------------------------------
# FIGURE CONFIGURATION -- every visual choice lives here; draw() reads these
# and nothing below this block needs editing to restyle the figure.
#
# Font sizes and pads follow mwb/scripts/plot_af_scatter_heen23_jazh24.py
# (the paper's Figure 1, output/outliers/Figure1.pdf), and
# FIG_WIDTH matches that figure's own width so the two sit at the same scale
# in the paper. The colorbar label placement follows plot_loo_tau_vs_alpha_
# combo.py's reliability strip (rotation 270 + labelpad, label on the right),
# which puts the label tight against the bar instead of floating off it.
# ---------------------------------------------------------------------------
SHOW_TITLE = False          # Figure 1 carries its caption in LaTeX, not on the axes

FIG_WIDTH = 34.0            # inches -- plot_af_scatter_heen23_jazh24.py's figsize[0]
FIG_HEIGHT = 34.0           # generous; the square box below plus bbox_inches='tight' sets the real height

# Force the PLOT BOX (the cell grid alone) square, so every cell is square
# regardless of how much width the colorbar and labels take. Setting the
# figure's own aspect cannot do this -- the axes rectangle is whatever is
# left after the colorbar, ylabel and ticks claim their space, so a square
# FIGURE still yields a wider-than-tall grid. set_box_aspect constrains the
# axes rectangle itself; FIG_HEIGHT is then only an upper bound, and
# bbox_inches='tight' trims the slack. Same approach as
# plot_loo_tau_vs_alpha_combo.py's own square panels. None disables it.
BOX_ASPECT = 1.0

LABEL_FONTSIZE = 64         # axis titles
TICK_FONTSIZE = 54          # axis tick labels
TICK_PAD = 25               # gap between ticks and their labels
# Figure 1 uses LABEL_PAD = -50, but that is a compensation for ITS layout
# (two wide panels with shared axes); reused here it drags the axis titles
# straight onto the tick labels. Positive pad instead -- the font sizes are
# what follow Figure 1, not this.
LABEL_PAD = 30

# How each cell renders its mean +- std spread.
#   'triangles' -- split along the anti-diagonal: lower-left filled at
#                  mean - std, upper-right at mean + std. Two flat colors.
#   'gradient'  -- a continuous vertical ramp, mean - std at the cell's
#                  bottom edge to mean + std at its top. EXPLORATORY.
# Both encode the same two numbers; the gradient reads the spread as a
# smooth extent rather than as two discrete patches, at the cost of no
# longer showing the two endpoint colors as flat, directly comparable areas.
CELL_STYLE = 'gradient'
GRADIENT_STEPS = 256        # vertical samples in a 'gradient' cell

# Per-cell "mean +- std" text. CELL_FONTSIZE is sized to nearly fill a cell;
# it is a plain number rather than something derived from the grid, so it
# needs revisiting if FIG_WIDTH or the 8x8 bin counts change a lot.
CELL_FONTSIZE = 54
CELL_LINESPACING = 0.95     # <1 tightens the two lines toward each other
CELL_TEXT_COLOR = 'white'

# False: draw the colorbar (to reserve exactly its space), then hide it, and
# save against the bbox measured WHILE it was visible -- so the outer figure
# size and the plot box are byte-for-byte what the colorbar version has, with
# the bar and its label replaced by whitespace. Simply not creating the
# colorbar would instead let the plot box expand and bbox_inches='tight'
# crop the figure narrower, which is not the same figure.
SHOW_COLORBAR = True

CBAR_LABEL = 'Adequacy over fluency preference'
CBAR_LABEL_FONTSIZE = LABEL_FONTSIZE
CBAR_TICK_FONTSIZE = TICK_FONTSIZE
CBAR_LABEL_ROTATION = 270
CBAR_LABEL_PAD = 34         # distance from the bar's tick labels to its own label
CBAR_FRACTION = 0.035       # colorbar width as a fraction of the axes
CBAR_PAD = 0.02             # gap between the axes and the colorbar
CBAR_TICKS = (0.0, 1.0)     # endpoints only -- no intermediate ticks
CBAR_TICKLABELS = ('0', '1')  # bare integers, not 0.00/1.00

XLABEL = 'ESS$(w)$'
YLABEL = r'Realized $\beta(w)$'

SAVE_PAD_INCHES = 0.1       # matplotlib's own bbox_inches='tight' default

EMPTY_CELL_COLOR = '#f2f2f2'
GRIDLINE_COLOR = 'white'
GRIDLINE_WIDTH = 1.0


def build_edges(lo: float, hi: float, n: int) -> np.ndarray:
  return np.linspace(lo, hi, n + 1)


def tick_labels(edges: np.ndarray, max_decimals: int = 6) -> list[str]:
  """Format axis edges with the FEWEST decimals that still renders every
  label distinctly. A fixed precision breaks here: the beta window is auto-
  fitted and, at a high ESS floor, collapses to something like [0.970, 1.000]
  where two decimals prints '0.99, 0.99, 0.98, 0.98' -- adjacent ticks
  showing the same number."""
  for d in range(1, max_decimals + 1):
    labels = [f'{v:.{d}f}' for v in edges]
    if len(set(labels)) == len(labels):
      return labels
  return [f'{v:.{max_decimals}f}' for v in edges]


def ess_edges_from_values(values: np.ndarray) -> np.ndarray:
  """One bin per ATTAINABLE ESS value: edges are the midpoints between
  consecutive distinct values, with the two outer edges reflected out by
  half a gap.

  ESS(w) = grid_n^2 / sum(c_k^2) over integer compositions, so it is
  discrete, and near K it is *sparse* -- at K=12, grid_n=24 only seven
  values exist in [9.6, 12]. Evenly-spaced bins over that range therefore
  cannot help: extra bins land between attainable values and come out
  empty, while two nearby values can share a bin. Binning on the values
  themselves is the only way to raise ESS resolution without raising
  --grid-n, and it guarantees every column is populated."""
  v = np.unique(values)
  if len(v) == 1:
    pad = max(abs(v[0]) * 1e-3, 1e-6)
    return np.array([v[0] - pad, v[0] + pad])
  mids = 0.5 * (v[:-1] + v[1:])
  return np.concatenate([[v[0] - (mids[0] - v[0])], mids, [v[-1] + (v[-1] - mids[-1])]])


def auto_beta_window(beta: np.ndarray, coverage: float, round_to: float) -> tuple[float, float]:
  """Central `coverage` quantile range of `beta`, rounded outward to a
  multiple of `round_to` and clipped to [0, 1]."""
  tail = (1.0 - coverage) / 2.0
  lo, hi = np.nanquantile(beta, [tail, 1.0 - tail])
  lo = max(0.0, np.floor(lo / round_to) * round_to)
  hi = min(1.0, np.ceil(hi / round_to) * round_to)
  if hi - lo < round_to:  # degenerate (all points in one rounding cell)
    lo, hi = max(0.0, lo - round_to), min(1.0, hi + round_to)
  return float(lo), float(hi)


def draw(stats, beta_edges, ess_edges, title, out_base, formats=('pdf', 'png')):
  """The split-triangle heatmap. stats: dict of (n_beta, n_ess) arrays with
  'mean', 'std', 'count'. All styling comes from the FIGURE CONFIGURATION
  block above."""
  mean, std, count = stats['mean'], stats['std'], stats['count']
  norm = Normalize(vmin=0.0, vmax=1.0)

  fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

  for r in range(len(beta_edges) - 1):
    for c in range(len(ess_edges) - 1):
      x0, x1 = ess_edges[c], ess_edges[c + 1]
      y0, y1 = beta_edges[r], beta_edges[r + 1]
      if count[r, c] == 0 or not np.isfinite(mean[r, c]):
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True,
                             facecolor=EMPTY_CELL_COLOR, edgecolor=GRIDLINE_COLOR,
                             linewidth=GRIDLINE_WIDTH))
        continue

      s_ = 0.0 if not np.isfinite(std[r, c]) else std[r, c]
      # Both styles encode the same interval, clipped to the colormap's
      # [0, 1] domain -- which is also the preference score's own range.
      lo_v = float(np.clip(mean[r, c] - s_, 0.0, 1.0))
      hi_v = float(np.clip(mean[r, c] + s_, 0.0, 1.0))

      if CELL_STYLE == 'gradient':
        # A tall 1-column image spanning the cell. origin='lower' puts row 0
        # at the bottom, so the ramp runs mean-std (bottom) -> mean+std
        # (top). aspect='auto' is required: imshow otherwise locks the axes
        # to equal aspect and fights BOX_ASPECT.
        ramp = np.linspace(lo_v, hi_v, GRADIENT_STEPS).reshape(-1, 1)
        ax.imshow(ramp, extent=(x0, x1, y0, y1), origin='lower', aspect='auto',
                  cmap=PREFERENCE_CMAP, norm=norm, interpolation='bilinear', zorder=1)
      else:
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x0, y1)], closed=True,
                             facecolor=PREFERENCE_CMAP(norm(lo_v)), edgecolor='none'))
        ax.add_patch(Polygon([(x1, y0), (x1, y1), (x0, y1)], closed=True,
                             facecolor=PREFERENCE_CMAP(norm(hi_v)), edgecolor='none'))
      ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True,
                           facecolor='none', edgecolor=GRIDLINE_COLOR,
                           linewidth=GRIDLINE_WIDTH, zorder=3))

      ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1),
              f'{mean[r, c]:.2f}\n$\\pm${s_:.2f}',
              ha='center', va='center', fontsize=CELL_FONTSIZE, color=CELL_TEXT_COLOR,
              linespacing=CELL_LINESPACING, zorder=5)

  ax.set_xlim(ess_edges[0], ess_edges[-1])
  ax.set_ylim(beta_edges[0], beta_edges[-1])
  ax.set_xticks(ess_edges)
  ax.set_yticks(beta_edges)
  ax.set_xticklabels(tick_labels(ess_edges))
  ax.set_yticklabels(tick_labels(beta_edges))
  ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
  ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
  ax.tick_params(labelsize=TICK_FONTSIZE, pad=TICK_PAD)
  if BOX_ASPECT is not None:
    ax.set_box_aspect(BOX_ASPECT)
  if SHOW_TITLE:
    ax.set_title(title, fontsize=LABEL_FONTSIZE)
  for side in ('top', 'right', 'bottom', 'left'):
    ax.spines[side].set_visible(True)

  sm = plt.cm.ScalarMappable(cmap=PREFERENCE_CMAP, norm=norm)
  cb = fig.colorbar(sm, ax=ax, fraction=CBAR_FRACTION, pad=CBAR_PAD)
  cb.set_ticks(list(CBAR_TICKS))
  cb.ax.set_yticklabels(list(CBAR_TICKLABELS), fontsize=CBAR_TICK_FONTSIZE)
  # Label on the right of the bar, rotated and pulled in by labelpad -- the
  # plot_loo_tau_vs_alpha_combo.py reliability-strip convention.
  cb.ax.set_ylabel(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE,
                   rotation=CBAR_LABEL_ROTATION, labelpad=CBAR_LABEL_PAD)
  cb.ax.yaxis.set_label_position('right')

  fig.tight_layout()

  # Measure the tight bbox WITH the colorbar drawn, then optionally hide it
  # and save against that same bbox -- see SHOW_COLORBAR.
  fig.canvas.draw()
  bbox = fig.get_tightbbox(fig.canvas.get_renderer())
  if not SHOW_COLORBAR:
    cb.ax.set_visible(False)

  paths = []
  for fmt in formats:
    p = f'{out_base}.{fmt}'
    fig.savefig(p, dpi=200, bbox_inches=bbox, pad_inches=SAVE_PAD_INCHES)
    paths.append(p)
  plt.close(fig)
  return paths


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='ende24')
  ap.add_argument('--sampler', choices=['lattice', 'simplex'], default='simplex',
                  help="'simplex' (default): importance sampling against uniform-on-the-"
                       'simplex (Dirichlet(1)), from a defensive Dirichlet mixture '
                       'concentrated on the requested ESS window. ESS is continuous in w, so '
                       'bins can be placed anywhere in [1, K] at any resolution; the cost is '
                       'that each cell is an estimate, whose precision is reported as n_eff. '
                       "'lattice': exhaustive uniform simplex lattice at "
                       '--grid-n. Deterministic and reproducible, but ESS on a lattice takes '
                       'only the values grid_n^2/m for integer m, with gaps near K of about '
                       '2/(grid_n/K)^2 that cost a ~K-fold bigger lattice to halve -- so a '
                       'narrow, finely-resolved ESS window is unreachable that way. Kept as '
                       'the exhaustive, deterministic cross-check on the sampler.')
  ap.add_argument('--n-samples', type=int, default=4_000_000,
                  help='simplex sampler: total draws (before the ESS-window filter)')
  ap.add_argument('--n-alpha', type=int, default=8,
                  help='simplex sampler: Dirichlet mixture components spanning the ESS window')
  ap.add_argument('--grid-n', type=int, default=None,
                  help='simplex lattice resolution: weights are integer compositions of '
                       'this value divided by it. Default: K, the smallest resolution that '
                       'puts the uniform point (ESS=K) on the lattice at all -- max '
                       'reachable ESS is min(K, grid_n), so grid_n < K silently caps the '
                       'ESS axis. A multiple of K samples the high-ESS region more densely '
                       'but lattice size is C(grid_n+K-1, K-1), which explodes fast.')
  ap.add_argument('--max-lattice', type=int, default=2_000_000_000,
                  help='refuse to enumerate a lattice larger than this. Points are streamed '
                       'and ESS-filtered, so this bounds TIME, not memory: measured '
                       'throughput is ~11M lattice points/sec for generation+filtering and '
                       '~120K surviving points/sec for scoring, so a 400M-point lattice '
                       'with a few percent surviving runs in a couple of minutes.')
  ap.add_argument('--ess-bins', type=int, default=DEFAULT_N_BINS,
                  help=f'evenly-spaced ESS bins over the window (default '
                       f'{DEFAULT_N_BINS}, applied under BOTH samplers so the x '
                       'axis is the same however the simplex was covered). '
                       '0 = one bin per ATTAINABLE ESS value inside the window, '
                       'with edges at the midpoints between them. ESS is discrete on a '
                       'lattice and sparse near K, so this is the maximum honest resolution '
                       'and never leaves an empty column. Give a positive integer for '
                       'evenly-spaced bins instead; the run then reports how many attainable '
                       'values there actually are, and warns when you asked for more bins '
                       'than that.')
  ap.add_argument('--beta-bins', type=int, default=DEFAULT_N_BINS,
                  help=f'evenly-spaced beta bins over the auto-fitted (or --beta-lo/'
                       f'--beta-hi) beta window. Default {DEFAULT_N_BINS}, matching '
                       '--ess-bins so every figure is the same 8x8 grid.')
  ap.add_argument('--ess-lo-frac', type=float, default=0.5,
                  help='ESS window floor as a fraction of K (ceiling is always K). The '
                       'committed default is 0.5, i.e. ESS/K in [0.5, 1] for every dataset. '
                       'Chosen empirically: it is wide enough to show the preference '
                       'crossing 0.5 and the dispersion cone opening up as ESS falls, while '
                       'still excluding the region where the importance weights degenerate. '
                       'Going to the full [1, K] was tried and is NOT usable -- '
                       'uniform-on-the-simplex has almost no mass below ESS ~4, so those '
                       'cells came back with n_eff in the single digits and visibly '
                       'non-monotone means.')
  ap.add_argument('--ess-lo', type=float, default=None, help='absolute ESS floor, overrides --ess-lo-frac')
  ap.add_argument('--beta-lo', type=float, default=None)
  ap.add_argument('--beta-hi', type=float, default=None)
  ap.add_argument('--beta-coverage', type=float, default=0.98,
                  help='central quantile mass the auto beta window must cover')
  ap.add_argument('--beta-round', type=float, default=0.01,
                  help='the auto beta window is rounded outward to a multiple of this. '
                       'Small, because at a high ESS floor beta is pinned tightly around '
                       'beta_0 and coarse rounding leaves empty rows.')
  ap.add_argument('--max-per-cell', type=int, default=0,
                  help='cap on points SCORED per cell (0 = no cap, the default). Scoring '
                       'runs at ~100K points/sec (it is one dense matmul -- see '
                       'mwb/lib/beta_ess_preference_grid.py), so the cap is rarely needed; it '
                       'exists for windows holding millions of points. True population '
                       'counts are always reported in the diagnostic table alongside the '
                       'scored counts, so a capped cell is visible as such.')
  ap.add_argument('--save-points', type=int, default=0,
                  help='also save a random subsample of this many individual scored '
                       'weightings as <tag>_points.npz: their beta, ESS, preference and log '
                       'importance weight. Cell statistics aggregate these away, so any '
                       'analysis of the beta-preference relation that should see the '
                       'within-cell scatter (e.g. an ungrouped correlation) needs them. '
                       '0 = do not save.')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--tag', default=None)
  args = ap.parse_args()

  t0 = time.time()
  systems = real_systems(args.dataset, root=ROOT)
  if not systems:
    sys.exit(f'{args.dataset}: no systems under the project outlier screen')
  K = len(systems)
  sys_df = load_system_scores(args.dataset, root=ROOT).loc[systems]
  a, f = sys_df['a'].values, sys_df['b'].values
  b0 = alpha_0(a, f)
  print(f'{args.dataset}: K={K}, beta_0={b0:.4f}', file=sys.stderr)

  # --- 1. ESS window, then stream the lattice through it. ---------------
  grid_n = args.grid_n if args.grid_n is not None else K
  n_lattice = lattice_size(K, grid_n)
  # ESS on a lattice cannot exceed min(K, grid_n); under the sampler it is
  # continuous on [1, K], so neither the ceiling nor the size cap applies.
  ess_ceiling = max_lattice_ess(K, grid_n) if args.sampler == 'lattice' else float(K)
  if args.sampler == 'lattice':
    print(f'lattice: n={grid_n}, {n_lattice:,} points, max reachable ESS = {ess_ceiling:.2f}',
          file=sys.stderr)
  if args.sampler == 'lattice' and n_lattice > args.max_lattice:
    sys.exit(f'lattice of {n_lattice:,} points exceeds --max-lattice {args.max_lattice:,}; '
             f'lower --grid-n (max reachable ESS drops to min(K, grid_n)) or raise the cap')

  ess_lo = args.ess_lo if args.ess_lo is not None else args.ess_lo_frac * K
  ess_window = (ess_lo, float(K))
  ess_edges = (build_edges(*ess_window, args.ess_bins) if args.ess_bins > 0
               else np.array(ess_window))  # provisional; rebuilt from the data below
  if ess_edges[0] > ess_ceiling:
    sys.exit(f'ESS window starts at {ess_edges[0]:.2f} but this lattice cannot exceed '
             f'{ess_ceiling:.2f} -- raise --grid-n (>= K puts the uniform point on the '
             f'lattice) or lower --ess-lo-frac')

  log_iw = None
  cov_all = None
  if args.sampler == 'lattice':
    def lat_prog(done, total, kept):
      print(f'  lattice {done:,}/{total:,} kept {kept:,}', file=sys.stderr, end='\r')

    comps, beta_all, ess_all, n_total, ess_seen_lo, ess_seen_hi = lattice_beta_ess(
        a, f, grid_n, ess_lo=ess_window[0], ess_hi=ess_window[1], progress=lat_prog)
    print(f'\nstreamed {n_total:,} points (ESS seen [{ess_seen_lo:.2f}, {ess_seen_hi:.2f}]); '
          f'{len(comps):,} inside the ESS window, stored as {comps.dtype} compositions '
          f'({comps.nbytes / 2**30:.2f} GiB)  ({time.time() - t0:.1f}s)', file=sys.stderr)
    if len(comps) == 0:
      sys.exit(f'no lattice point has ESS in [{ess_window[0]:.2f}, {ess_window[1]:.2f}]')
    w_of = lambda rows: comps[rows].astype(np.float64) / grid_n
  else:
    alphas = build_proposal_alphas(K, ess_window[0], ess_window[1], args.n_alpha)
    print(f'proposal: {len(alphas)} Dirichlet components, alpha in '
          f'[{alphas.min():.3g}, {alphas.max():.3g}]', file=sys.stderr)

    def smp_prog(done, total, kept):
      print(f'  sampled {done:,}/{total:,} kept {kept:,}', file=sys.stderr, end='\r')

    xs, beta_all, ess_all, log_iw, cov_all, n_total = sample_simplex_beta_ess(
        a, f, args.n_samples, alphas, ess_lo=ess_window[0], ess_hi=ess_window[1],
        seed=args.seed, progress=smp_prog)
    print(f'\ndrew {n_total:,} points; {len(xs):,} inside the ESS window '
          f'({100.0 * len(xs) / n_total:.1f}% accepted)  ({time.time() - t0:.1f}s)',
          file=sys.stderr)
    if len(xs) == 0:
      sys.exit(f'no sample has ESS in [{ess_window[0]:.2f}, {ess_window[1]:.2f}]')
    w_of = lambda rows: xs[rows]

  # Auto ESS bins. On the lattice, one per ATTAINABLE value is the maximum
  # honest resolution (see ess_edges_from_values). Under the sampler ESS is
  # continuous, so there is no such constraint and evenly-spaced bins over
  # the window are the natural choice.
  if args.ess_bins <= 0:
    if args.sampler == 'lattice':
      ess_edges = ess_edges_from_values(np.unique(np.round(ess_all.astype(np.float64), 9)))
      print(f'ESS bins: auto -> {len(ess_edges) - 1} bins, one per attainable lattice value',
            file=sys.stderr)
    else:
      ess_edges = build_edges(ess_window[0], ess_window[1], DEFAULT_N_BINS)
      print(f'ESS bins: auto -> {DEFAULT_N_BINS} even bins '
            f'(ESS is continuous under the sampler)', file=sys.stderr)

  # --- 2. beta window (from the ESS-windowed points). -------------------
  if args.beta_lo is not None and args.beta_hi is not None:
    beta_lo, beta_hi = args.beta_lo, args.beta_hi
  else:
    beta_lo, beta_hi = auto_beta_window(beta_all, args.beta_coverage, args.beta_round)
  beta_edges = build_edges(beta_lo, beta_hi, args.beta_bins)
  print(f'window: ESS [{ess_edges[0]:.2f}, {ess_edges[-1]:.2f}] x '
        f'beta [{beta_edges[0]:.3f}, {beta_edges[-1]:.3f}]', file=sys.stderr)

  row, col, inside = bin_cells(beta_all, ess_all, beta_edges, ess_edges)
  n_in = int(inside.sum())
  n_window = len(beta_all)
  src = 'lattice' if args.sampler == 'lattice' else 'draws'
  print(f'{n_in:,} of the {n_window:,} ESS-windowed points fall inside the full window '
        f'({100.0 * n_in / n_total:.4f}% of the {n_total:,} {src})', file=sys.stderr)
  # How many DISTINCT ESS values the lattice actually offers here: ESS is
  # grid_n^2/sum(c^2) over integers, so it is discrete, and asking for more
  # ESS bins than there are attainable values guarantees empty columns.
  ess_vals = np.unique(np.round(ess_all[inside].astype(np.float64), 9))
  n_ess_bins = len(ess_edges) - 1
  if args.sampler == 'lattice':
    print(f'{len(ess_vals)} distinct ESS values attainable in the window for {n_ess_bins} bins'
          + ('  <-- fewer values than bins, so some columns must be empty; use --ess-bins 0 '
             '(one bin per value), raise --grid-n to a larger multiple of K, or switch to '
             '--sampler simplex (ESS is continuous there)'
             if len(ess_vals) < n_ess_bins else ''), file=sys.stderr)
  if n_in == 0:
    sys.exit('window is empty -- adjust --beta-lo/--beta-hi or the ESS floor')

  # --- 3. Per-cell true counts, then the scoring subsample. -------------
  n_b, n_e = args.beta_bins, len(ess_edges) - 1
  counts = np.zeros((n_b, n_e), dtype=np.int64)
  np.add.at(counts, (row[inside], col[inside]), 1)

  rng = np.random.default_rng(args.seed)
  idx_all = np.flatnonzero(inside)
  keep = []
  for r in range(n_b):
    for c in range(n_e):
      cell_idx = idx_all[(row[idx_all] == r) & (col[idx_all] == c)]
      if args.max_per_cell and len(cell_idx) > args.max_per_cell:
        cell_idx = rng.choice(cell_idx, size=args.max_per_cell, replace=False)
      keep.append(cell_idx)
  keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
  print(f'scoring {len(keep):,} points (cap {args.max_per_cell or "none"} per cell)', file=sys.stderr)

  # --- 4. Donor p-value tables (the expensive, w-independent step). -----
  donors = real_scorers(args.dataset, root=ROOT, systems=systems, require_seg_scores=True)
  t1 = time.time()

  def don_prog(i, n, name, ok):
    print(f'  donor {i}/{n} {name}{"" if ok else " (SKIPPED)"}      ', file=sys.stderr, end='\r')

  tables = load_donor_tables(args.dataset, systems, donors, root=ROOT, progress=don_prog)
  if not tables:
    sys.exit(f'{args.dataset}: no usable donors')
  print(f'\n{len(tables)} donors x {tables[0].diffs.shape[0]} dials '
        f'({time.time() - t1:.1f}s)', file=sys.stderr)

  # --- 5. Score, then aggregate per cell. -------------------------------
  t2 = time.time()
  pref = preference_scores(w_of(keep), tables, K)
  print(f'scored in {time.time() - t2:.1f}s', file=sys.stderr)

  mean = np.full((n_b, n_e), np.nan)
  std = np.full((n_b, n_e), np.nan)
  n_eff = np.zeros((n_b, n_e))
  scored = np.zeros((n_b, n_e), dtype=np.int64)
  kr, kc = row[keep], col[keep]
  kw = None if log_iw is None else log_iw[keep]
  for r in range(n_b):
    for c in range(n_e):
      sel = (kr == r) & (kc == c)
      scored[r, c] = int(sel.sum())
      if scored[r, c]:
        mean[r, c], std[r, c], n_eff[r, c] = weighted_mean_std(
            pref[sel], None if kw is None else kw[sel])

  # --- 6. Outputs. ------------------------------------------------------
  tag = args.tag or (f'{args.dataset}_n{grid_n}' if args.sampler == 'lattice'
                     else f'{args.dataset}_simplex{args.n_samples // 1000}k')
  os.makedirs(ARTIFACTS, exist_ok=True)
  base = os.path.join(ARTIFACTS, f'beta_ess_preference_{tag}')

  paths = draw({'mean': mean, 'std': std, 'count': counts}, beta_edges, ess_edges,
               f'{args.dataset}  (K={K}, $\\beta_0$={b0:.3f}, '
               + (f'lattice n={grid_n})' if args.sampler == 'lattice'
                  else f'simplex IS, {n_total / 1e6:.1f}M draws)'), base)

  np.savez(base + '.npz', mean=mean, std=std, counts=counts, scored=scored, n_eff=n_eff,
           beta_edges=beta_edges, ess_edges=ess_edges, K=K, beta_0=b0,
           dataset=args.dataset, grid_n=grid_n, n_donors=len(tables), sampler=args.sampler)

  if args.save_points:
    keep_pts = np.flatnonzero(np.isfinite(pref))
    if len(keep_pts) > args.save_points:
      keep_pts = rng.choice(keep_pts, size=args.save_points, replace=False)
    np.savez(base + '_points.npz',
             beta=beta_all[keep][keep_pts].astype(np.float32),
             ess=ess_all[keep][keep_pts].astype(np.float32),
             pref=pref[keep_pts].astype(np.float32),
             log_iw=(np.zeros(len(keep_pts), np.float32) if log_iw is None
                     else log_iw[keep][keep_pts].astype(np.float32)),
             cov=(np.full(len(keep_pts), np.nan, np.float32) if cov_all is None
                  else cov_all[keep][keep_pts].astype(np.float32)),
             wmax=xs[keep][keep_pts].max(axis=1).astype(np.float32),
             dataset=args.dataset, K=K, beta_0=b0, sampler=args.sampler)
    print(f'saved {len(keep_pts):,} individual weightings to {base}_points.npz', file=sys.stderr)

  md = base + '_cells.md'
  with open(md, 'w') as fh:
    fh.write(f'# Cell occupancy: {args.dataset} (K={K}, lattice n={grid_n})\n\n')
    fh.write(f'Distinct ESS values attainable inside the window: {len(ess_vals)} '
             f'(for {n_ess_bins} ESS bins'
             f'{", one per attainable value" if args.ess_bins <= 0 else ""}).\n\n')
    fh.write(f'{"Full simplex lattice" if args.sampler == "lattice" else "Draws"}: '
             f'{n_total:,} points, max reachable ESS '
             f'{ess_ceiling:.2f}. Inside the window: {n_in:,} '
             f'({100.0 * n_in / n_total:.4f}%). Scored: {len(keep):,} '
             f'(cap {args.max_per_cell or "none"} per cell).\n\n')
    if args.sampler == 'lattice':
      fh.write('`true` = lattice points falling in the cell. `scored` = how many of them '
               'were actually evaluated (equal to `true` unless the cap bound).\n\n')
    else:
      fh.write('Each cell shows its raw draw count and `n_eff`, the Kish effective sample '
               'size of that cell\'s normalized importance weights (1/sum p_i^2). n_eff, '
               'not the raw count, is the precision actually behind the cell: a cell with '
               '20,000 draws but n_eff 40 is reporting about forty independent '
               'observations.\n\n')
    fh.write('ESS on a lattice is DISCRETE: ESS(w) = grid_n^2 / sum(c_k^2) over integer '
             'compositions c summing to grid_n, so only certain values are attainable and '
             'they thin out sharply near the top. An entirely empty high-ESS column is '
             'therefore a lattice artifact, not missing data -- no weighting on this '
             'lattice realizes an ESS in that interval at all. Raising --grid-n to a '
             'larger multiple of K is the only fix; the maximum attainable ESS is '
             'min(K, grid_n), and it equals K only when K divides grid_n.\n\n')
    hdr = [f'ESS {ess_edges[c]:.2f}-{ess_edges[c + 1]:.2f}' for c in range(n_e)]
    fh.write('| beta bin | ' + ' | '.join(hdr) + ' | row total |\n')
    fh.write('|' + '|'.join(['---'] * (n_e + 2)) + '|\n')
    for r in range(n_b - 1, -1, -1):  # top row first, matching the figure
      cells = [(f'{counts[r, c]:,} ({scored[r, c]:,})' if args.sampler == 'lattice'
                else f'{counts[r, c]:,} / n_eff {n_eff[r, c]:.0f}') for c in range(n_e)]
      fh.write(f'| {beta_edges[r]:.3f}-{beta_edges[r + 1]:.3f} | '
               + ' | '.join(cells) + f' | {counts[r].sum():,} |\n')
    fh.write('| **col total** | ' + ' | '.join(f'**{counts[:, c].sum():,}**' for c in range(n_e))
             + f' | **{counts.sum():,}** |\n')

  print('\n=== cell occupancy: true (scored) ===', file=sys.stderr)
  for r in range(n_b - 1, -1, -1):
    line = ' '.join((f'{counts[r, c]:>7,}({scored[r, c]:>4,})' if args.sampler == 'lattice'
                     else f'{counts[r, c]:>7,}(eff{n_eff[r, c]:>6.0f})') for c in range(n_e))
    print(f'beta {beta_edges[r]:.3f}-{beta_edges[r + 1]:.3f} | {line}', file=sys.stderr)
  print('ESS bins: ' + '  '.join(f'{ess_edges[c]:.2f}-{ess_edges[c + 1]:.2f}' for c in range(n_e)),
        file=sys.stderr)

  print('\n' + '\n'.join(paths))
  print(md)
  print(base + '.npz')
  print(f'total {time.time() - t0:.1f}s', file=sys.stderr)
