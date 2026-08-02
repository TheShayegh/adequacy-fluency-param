"""Plots <adequacy|fluency>_orientation(metametric_alpha; D) vs. alpha, plus
ESS(w*(alpha)), from the cache written by compute_scorer_orientation_vs_alpha.py
(artifacts/data/scorer_orientation_<tag>.npz) -- pure plotting, no solver or
SPA-permutation work happens here, so restyling is cheap to iterate on.
metametric (spa or pa) is read from the cache itself (whichever
compute_scorer_orientation_vs_alpha.py --metametric produced it).

Panel 1/2: adequacy_orientation / fluency_orientation vs. alpha -- one
independent dot per alpha for the mean-across-donors value (no connecting
line -- see _add_ess_colored_dots for why), colored by ESS: fully
transparent at/below the absolute floor ESS=1 (action_plan.md section 5),
then a 2-segment cubic Hermite spline ramp (SplineReliabilityNorm, tuned via
codes/scripts/plot_ess_opacity_curve.py's --shape spline) up to opaque dark
blue/red at ESS=K -- plus a dashed reference at 0.5 (no systematic
preference) and a dotted vertical line at alpha_0(D). Per-donor shadow
curves are still computed and cached (compute_scorer_orientation_vs_
alpha.py's A/B matrices) but deliberately not drawn here for now -- reserved
for a later use.

Usage: python codes/scripts/plot_scorer_orientation_vs_alpha.py [--dataset ende21]
           [--n-steps 5] [--step 0.01] [--full-range] [--metametric spa|pa] [--tag TAG]
(same grid flags as the compute script -- used only to derive the matching
cache filename via lib.synthetic_scorer_orientation.orientation_tag, unless
--data points at a cache file directly)

A donor-capability screen (excluding donors whose dial=1 endpoint doesn't
correlate with the true aspect, lib.action_plan.md section 2.2) was tried
and dropped: it barely moved the dataset-level curves in practice (a
donor's orientation_score turned out to be close to uncorrelated with its
own capability), so this script always averages over every cached donor --
the simple version.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from lib.synthetic_scorer_orientation import load_orientation_data, orientation_tag

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_scorer_orientation')

_PANEL_TITLES = {'A': 'Adequacy orientation (A-family)', 'B': 'Fluency orientation (B-family)'}
# Light/dark pairs per family -- dark end matches the flat colors used
# elsewhere in this project (lib.spa_plane's adequacy=blue/fluency=red
# convention); light end is the "type 2" style's own LIGHT_BLUE, mirrored
# in hue for the B panel.
_DARK_COLOR = {'A': (0.03, 0.15, 0.35), 'B': (0.35, 0.03, 0.05)}


# Fully transparent at/below this ABSOLUTE ESS value, not a fraction of K:
# with K only known after loading the data, the actual color-fraction
# cutoff (cutoff_abs / K) is computed in __main__ once K is known, and
# _ESS_NORM built there too. 2, not action_plan.md section 5's theoretical
# floor of 1 -- the reported/colored range is [2, K].
_ESS_CUTOFF_ABS = 2.0

# 2-segment cubic Hermite spline (knot at t=0.5) mapping ESS/K in
# [cutoff, 1] to a color-fraction in [0, 1] (clipped below cutoff) --
# tuned interactively via plot_ess_opacity_curve.py's --shape spline.
# History of what didn't work, in order:
#  - plain linear (t): looked "solid" over most of its length, since the
#    reliable half of the range only got the same ordinary rate of color
#    change as everywhere else.
#  - power law (t**gamma, gamma>1): structurally has ZERO derivative at
#    t=0 for any gamma -- no amount of tuning gives visible variation
#    right at the cutoff, and pushing gamma up to get more spread near K
#    made the flat region at the bottom even wider.
#  - pure quadratic / quadratic-linear blend (zero 3rd derivative): fixes
#    the "solid" look and gives a nonzero start slope, but fixing
#    alpha(0)=0, alpha(1)=1, and zero cubic term caps the end slope at
#    2x the linear rate -- provably cannot be steeper than that.
#  - single cubic Hermite (independent start/end slopes m0, m1): breaks
#    that ceiling, but the two slopes trade off directly against each
#    other through the monotonicity constraint (m0+m1 bounded), so a
#    small m0 (visible near the cutoff) only bought m1 up to ~3.5.
# The spline adds a 3rd knob -- the midpoint value/slope -- so the middle
# can be pushed low and flat, "paying" for a much steeper m1 (~6.6 here,
# essentially double) at the same m0, each segment independently
# monotonicity-checked in plot_ess_opacity_curve.py.
_SPLINE_M0 = 0.02    # slope right at the cutoff (ESS=1) -- kept small so ESS~=2-3 stays
                     # near-invisible (see _SPLINE_MMID: that's now what does the "rise",
                     # not this)
_SPLINE_VMID = 0.15  # opacity value at the midpoint (ESS~=K/2), visible (~15% opacity)
_SPLINE_MMID = 1.0   # slope AT the midpoint -- raised well above m0 so segment 1 stays flat
                     # through ESS~=2-3 and only rises sharply approaching the ESS~=K/2 knot,
                     # instead of an even/gradual rise across the whole first segment (which,
                     # at the old mmid=m0=0.1, made ESS~=2-3 visibly too bright)
_SPLINE_M1 = 6.6     # slope at ESS=K -- the feasible max at the vmid/mmid settings here


class SplineReliabilityNorm(Normalize):
  """2-segment cubic Hermite spline from vmin (cutoff, alpha=0) to vmax
  (ESS=K, alpha=1), knotted at the midpoint (t=0.5) with independently
  chosen value/slope there -- see plot_ess_opacity_curve.py's --shape
  spline for the derivation and the monotonicity tradeoff it encodes."""

  def __init__(self, vmin, vmax, m0=_SPLINE_M0, m1=_SPLINE_M1, v_mid=_SPLINE_VMID, m_mid=_SPLINE_MMID):
    super().__init__(vmin=vmin, vmax=vmax, clip=True)
    self.m0, self.m1, self.v_mid, self.m_mid = m0, m1, v_mid, m_mid

  @staticmethod
  def _hermite01(y0, y1, s0, s1, s):
    c3 = 2 * (y0 - y1) + s0 + s1
    c2 = 3 * (y1 - y0) - 2 * s0 - s1
    c1 = s0
    return c3 * s ** 3 + c2 * s ** 2 + c1 * s + y0

  def _forward_t(self, t):
    """The spline evaluated on a plain t already in [0, 1] -- factored out
    of __call__ so inverse() can bisect against the same function."""
    w = 0.5
    out = np.empty_like(t)
    left = t <= 0.5
    out[left] = self._hermite01(0.0, self.v_mid, self.m0 * w, self.m_mid * w, t[left] / w)
    out[~left] = self._hermite01(self.v_mid, 1.0, self.m_mid * w, self.m1 * w, (t[~left] - 0.5) / w)
    return out

  def __call__(self, value, clip=None):
    x = np.ma.asarray(value, dtype=float)
    t = np.clip((x - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0)
    return self._forward_t(t)

  def inverse(self, value):
    """Numeric inverse via bisection: the spline is strictly increasing
    (by construction/monotonicity check in plot_ess_opacity_curve.py), so
    this is well-defined. matplotlib's Colorbar needs this for a
    continuous mappable -- WITHOUT it, it silently falls back to
    Normalize's own linear inverse(), which is wrong for a nonlinear norm
    and renders a garbled colorbar (confirmed empirically: a blank/
    all-transparent bar with overlapping tick labels)."""
    y = np.ma.asarray(value, dtype=float)
    lo = np.zeros_like(y)
    hi = np.ones_like(y)
    for _ in range(50):
      mid = (lo + hi) / 2.0
      lo = np.where(self._forward_t(mid) < y, mid, lo)
      hi = np.where(self._forward_t(mid) < y, hi, mid)
    t = (lo + hi) / 2.0
    return self.vmin + t * (self.vmax - self.vmin)


def _ess_over_k_cmap(dark, top_alpha=1.0):
  """Reliability colormap: a single ramp from fully transparent to opaque
  `dark`, with all of the cutoff/easing shape handled by
  SmoothReliabilityNorm rather than by extra color stops (type 2's
  original style, _obsolete_type1_type2_plot_styles.py's render_type2_
  figure, inserted a light-colored "landmark accent" stop partway up,
  which -- once the transparency cutoff moved to K/3 -- produced a
  false-looking local dip around ESS=2K/3; removed here). `top_alpha` lets
  shadow lines reuse the same reliability gradient at a lower opacity
  ceiling than the mean curve, rather than flattening it with a uniform
  Artist-level alpha (which would overwrite the colormap's own
  per-segment alpha instead of combining with it)."""
  return LinearSegmentedColormap.from_list('ess_over_k', [
      (0.0, (*dark, 0.0)),
      (1.0, (*dark, top_alpha)),
  ])


def _draw_reliability_strip(fig, ax, cmap, norm, K, cutoff_abs, x_offset=0.012, label=True):
  """A manually-drawn reliability legend strip, since matplotlib's
  fig.colorbar always renders a continuous mappable's gradient by
  sampling the colormap UNIFORMLY (norm only moves tick positions, never
  the swatch's own pixel content -- confirmed empirically: even after
  fixing SplineReliabilityNorm.inverse(), the colorbar swatch stayed a
  perfectly linear fade). Building the strip's image directly from
  cmap(norm(ess)) is the only way the legend actually shows the curve's
  real, sharply nonlinear shape (mostly faint, then a steep dark run
  right at the top near ESS=K) instead of a misleading plain fade.

  x_offset positions this strip relative to `ax`'s right edge (figure
  fraction), so two strips (one per family/hue) can sit side by side
  without overlapping; `label` suppresses the axis label/ticks on all but
  one of them (both would otherwise show the identical ESS scale twice)."""
  bbox = ax.get_position()
  strip_ax = fig.add_axes([bbox.x1 + x_offset, bbox.y0 + 0.08 * bbox.height,
                            0.012, 0.84 * bbox.height])
  ess_grid = np.linspace(cutoff_abs, K, 512)
  rgba = cmap(norm(ess_grid / K)).reshape(-1, 1, 4)
  strip_ax.imshow(rgba, aspect='auto', origin='lower', extent=[0, 1, cutoff_abs, K])
  strip_ax.set_xticks([])
  strip_ax.set_ylim(cutoff_abs, K)
  for spine in strip_ax.spines.values():
    spine.set_linewidth(0.6)
  if label:
    strip_ax.set_yticks([cutoff_abs, K])
    strip_ax.set_yticklabels([f'{cutoff_abs:g}', 'K'])
    strip_ax.yaxis.tick_right()
    strip_ax.set_ylabel('Reliability (ESS)', fontsize=8, rotation=270, labelpad=10)
    strip_ax.yaxis.set_label_position('right')
  else:
    strip_ax.set_yticks([])


def _add_ess_colored_dots(ax, alphas, ys, ess_over_k, cmap, norm, size, zorder):
  """One independent marker per (alpha, y) point, colored by cmap(norm(
  ess_over_k)) -- no connecting line at all. A LineCollection was tried
  first (segments between consecutive points): with one segment per alpha
  step (n_sub=1, needed to avoid a separate alpha-compositing bug where
  many overlapping subdivided segments stacked their alpha channels far
  darker than intended), each segment is rendered as its own independent
  stroke, and even edge-to-edge join styles (butt/miter) left visible
  seams/gaps between segments at low opacity -- plain per-point dots
  sidestep the whole segment-joining problem, at the honest cost of
  showing exactly what it is: discrete sampled points, not a curve."""
  ax.scatter(alphas, ys, c=ess_over_k, cmap=cmap, norm=norm, s=size,
             linewidths=0, zorder=zorder)


def _draw_orientation_axes(ax, alphas, mean_curves, ess_over_k, cmaps, norm, center_alpha):
  """Both families (A=adequacy, B=fluency) overlaid on one axes -- distinct
  by hue (cmaps['A']=blue, cmaps['B']=red), sharing the x-axis (alpha), the
  y-axis (orientation score), and the reliability norm (ESS depends only on
  alpha, not on family). Per-donor shadow lines are computed and stored
  (compute_scorer_orientation_vs_alpha.py's A/B matrices) but deliberately
  not drawn here for now -- reserved for a later use."""
  for label in ('A', 'B'):
    _add_ess_colored_dots(ax, alphas, mean_curves[label], ess_over_k, cmaps[label], norm, size=10, zorder=4)

  ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
  center_idx = int(np.argmin(np.abs(np.asarray(alphas) - center_alpha)))
  ax.axvline(alphas[center_idx], color='black', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)

  ax.set_xlim(min(alphas), max(alphas))
  ax.set_ylim(-0.02, 1.02)
  ax.set_xlabel('alpha (meta-evaluation balance)')
  ax.set_ylabel('orientation score')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  proxies = [
      Line2D([0], [0], color=cmaps['A'](1.0), marker='o', linestyle='none', markersize=5,
             label=_PANEL_TITLES['A']),
      Line2D([0], [0], color=cmaps['B'](1.0), marker='o', linestyle='none', markersize=5,
             label=_PANEL_TITLES['B']),
  ]
  ax.legend(handles=proxies, fontsize=8, loc='lower center', title='color = reliability, ESS',
            title_fontsize=7)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--n-steps', type=int, default=5)
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--full-range', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='which cached weighted meta-metric to load/plot')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--data', type=str, default=None, help='explicit path to a scorer_orientation_*.npz cache')
  args = parser.parse_args()
  base = args.dataset

  if args.data:
    data_path = args.data
  else:
    tag = args.tag or orientation_tag(base, args.n_steps, args.step, args.full_range, args.metametric)
    data_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  if not os.path.exists(data_path):
    sys.exit(f'{data_path} not found -- run compute_scorer_orientation_vs_alpha.py with matching flags first')

  data = load_orientation_data(data_path)
  print(f'Loaded {data_path}: dataset={data["dataset"]}, {len(data["donors"])} donors, '
        f'{len(data["alphas"])} alphas', file=sys.stderr)

  K = data['K']
  ess_over_k = data['ess'] / K
  cutoff_frac = _ESS_CUTOFF_ABS / K
  norm = SplineReliabilityNorm(vmin=cutoff_frac, vmax=1.0)

  cmaps = {label: _ess_over_k_cmap(_DARK_COLOR[label]) for label in ('A', 'B')}
  # data['A']/data['B'] are (n_donors, n_alpha) -- per-donor shadow curves,
  # kept for later use (see _draw_orientation_axes), not drawn here.
  mean_curves = {label: np.nanmean(data[label], axis=0) for label in ('A', 'B')}

  fig, ax = plt.subplots(figsize=(9, 6))
  ax.set_box_aspect(1)  # square PLOT box -- independent of the title/colorbar space around it
  _draw_orientation_axes(ax, data['alphas'], mean_curves, ess_over_k, cmaps, norm, data['center_alpha'])

  fig.suptitle(f'{data["dataset"]}: scorer orientation vs. alpha ({data["metametric"].upper()}, '
               f'K={K} systems, {len(data["donors"])} donors, '
               f'alpha_0(D)={data["center_alpha"]:.4f})', fontsize=11)
  fig.tight_layout(rect=[0, 0, 0.85, 0.95])

  # Strips are added AFTER tight_layout, positioned off the axes' FINAL
  # bbox -- adding them before would have tight_layout shove the panel
  # around without the strips following. Two strips, flush side by side
  # (second offset = first offset + strip width, no gap): A (blue) first,
  # B (red) right next to it.
  _STRIP_WIDTH = 0.012
  _draw_reliability_strip(fig, ax, cmaps['A'], norm, K, _ESS_CUTOFF_ABS, x_offset=0.02, label=False)
  _draw_reliability_strip(fig, ax, cmaps['B'], norm, K, _ESS_CUTOFF_ABS, x_offset=0.02 + _STRIP_WIDTH, label=True)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  metametric_suffix = '' if data['metametric'] == 'spa' else f'_{data["metametric"]}'
  plot_path = os.path.join(ARTIFACTS_DIR, f'orientation_vs_alpha_{base}{metametric_suffix}.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}', file=sys.stderr)
