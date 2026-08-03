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
           [--n-steps 5] [--step 0.01] [--full-range | --union-grid] [--metametric spa|pa]
           [--synthesis offset|additive|additive_mean] [--dial-preset linear|geometric]
           [--aspect-donors | --per-donor] [--tag TAG]
(same grid/metametric/synthesis/dial-preset flags as the compute script --
used only to derive the matching cache filename via lib.
synthetic_scorer_orientation.orientation_tag/union_grid_tag, unless --data
points at a cache file directly)

A donor-capability screen (excluding donors whose dial=1 endpoint doesn't
correlate with the true aspect, lib.action_plan.md section 2.2) was tried
and dropped: it barely moved the dataset-level curves in practice (a
donor's orientation_score turned out to be close to uncorrelated with its
own capability), so this script always averages over every cached donor --
the simple version.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from lib.synth25_metametrics import SYNTH25_METAMETRICS
from lib.synth25_orientation import load_synth25_pools, load_synth25_pools_J
from lib.synth25_orientation import pools_tag as synth25_pools_tag
from lib.synth25_orientation import pools_tag_J as synth25_pools_tag_J
from lib.synthetic_scorer_alpha_grid import ASPECT_DONORS
from lib.synthetic_scorer_orientation import load_orientation_data, orientation_tag, union_grid_tag
from lib.synthetic_scorers import default_dial_preset

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_scorer_orientation')

_PANEL_TITLES = {
    'A': 'Adequacy orientation (A-family)', 'B': 'Fluency orientation (B-family)',
    'T': 'AllMQM orientation (T-family)', 'J': 'Joint orientation (J-family)',
}
# Light/dark pairs per family -- dark end matches the flat colors used
# elsewhere in this project (lib.spa_plane's adequacy=blue/fluency=red
# convention); light end is the "type 2" style's own LIGHT_BLUE, mirrored
# in hue for the B panel. T and J are both orientation-neutral by
# construction (T because t=a+b favors neither aspect; J because it
# conditions on both aspects jointly rather than either alone), which used
# to get them the same dark green when a cache only ever had one of the
# two -- --green-family ABTJ caches now have BOTH T and J at once, so they
# need visually distinct colors: J stays dark green, T gets dark gold
# (darkgoldenrod), chosen to read clearly against green/blue/red at the
# same low marker alpha.
_DARK_COLOR = {'A': (0.03, 0.15, 0.35), 'B': (0.35, 0.03, 0.05), 'T': (0.05, 0.30, 0.05), 'J': (0.72, 0.53, 0.04)}
_MEAN_RED = '#c81e1e'
_ESS_FRAC_BIN_WIDTH = 0.02

# --overlay-synth25 pool markers (lib.synth25_metametrics.POOL_BLOCKS):
# distinct shape per pool, sharing color with the family (A/B/T) it's
# plotted in (_DARK_COLOR). 'real+adeq+flu' is Row 7 -- the branded
# SPA_synth25/PA_synth25 baseline this whole project reports elsewhere --
# so it gets both a distinct marker (circle, matching the plain dot-color
# legend's own marker) AND a visibly larger size; the other 5 are
# comparison points, same size as each other, shapes chosen to look
# distinct even when several land close together on the alpha axis.
_POOL_MARKERS = {
    'real+adeq+flu': 'o', 'real+adeq': '^', 'real+flu': 's', 'adeq+flu': 'D', 'flu': 'v', 'adeq': 'P',
}
_POOL_MARKER_SIZE_LARGE = 220 / 3
_POOL_MARKER_SIZE_DEFAULT = 90 / 3
_POOL_MARKER_SIZE = {'real+adeq+flu': _POOL_MARKER_SIZE_LARGE}
_FAMILY_FILE_PREFIX = {'A': 'adequacy', 'B': 'fluency', 'T': 'allmqm', 'J': 'joint'}


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
  showing exactly what it is: discrete sampled points, not a curve.

  Non-uniform alpha grids (e.g. lib.alpha.union_alpha_grid, which mixes
  alpha_ij's exact values into a uniform sweep) can place two dots close
  enough to visually overlap. Plain ax.scatter with a translucent facecolor
  would let matplotlib's normal painter's-algorithm blending COMPOSITE them
  -- the overlap region ends up darker than either dot alone, i.e.
  opacities summing -- the same
  fundamental trap as the LineCollection subdivision bug above, just via
  markers instead of segments. Avoided here by pre-blending each dot's
  color against the (white) axes background ourselves and plotting the
  result fully OPAQUE, in ascending order of opacity: every less-solid dot
  is drawn before every more-solid one regardless of position, so at any
  overlap the more-solid dot -- now an opaque patch -- simply paints over
  the less-solid one instead of blending with it. Only the single most
  solid color shows up anywhere two dots overlap."""
  rgba = np.asarray(cmap(norm(np.asarray(ess_over_k, dtype=float))))
  opacity = rgba[:, 3]
  white = np.ones((len(opacity), 3))
  solid_rgb = opacity[:, None] * rgba[:, :3] + (1 - opacity[:, None]) * white
  order = np.argsort(opacity, kind='stable')
  ax.scatter(np.asarray(alphas)[order], np.asarray(ys)[order], c=solid_rgb[order], s=size,
             linewidths=0, zorder=zorder)


def _draw_orientation_axes(ax, alphas, mean_curves, ess_over_k, cmaps, norm, center_alpha, families=('A', 'B'),
                            synth25_pools=None, synth25_metric_label=''):
  """`families` (A=adequacy, B=fluency, and/or the cache's third family --
  T=orientation-neutral All-MQM or J=orientation-neutral Joint, whichever
  is present) overlaid on one axes -- distinct by hue (cmaps['A']=blue,
  cmaps['B']=red, cmaps['T']/cmaps['J']=green), sharing the x-axis (alpha),
  the y-axis (orientation score), and the reliability norm (ESS depends
  only on alpha, not on family). Per-donor shadow lines are computed and
  stored (compute_scorer_orientation_vs_alpha.py's A/B/T-or-J matrices) but
  deliberately not drawn here for now -- reserved for a later use.

  synth25_pools: optional {pool_name: {'alpha0': .., 'A': .., 'B': ..,
  'T': .., 'J': ..}} from lib.synth25_orientation.load_synth25_pools (one
  entry per lib.synth25_metametrics.POOL_BLOCKS pool); 'J' is only present
  when the caller has merged it in from load_synth25_pools_J (a SEPARATE
  cache -- lib.synth25_orientation never computes J alongside A/B/T, see
  its "J (Joint) family" section), for overlaying e.g. additive_mean's A/B
  pool markers next to offset's negative-dial J instead of additive_mean's
  own T. Each pool has its OWN natural
  alpha_0 (a property of that pool's system-level Adequacy/Fluency MQM
  composition, independent of alpha reweighting) -- so rather than a flat
  horizontal reference line, each (pool, family) cell is drawn as ONE
  MARKER at (pool's alpha_0, that pool's orientation): color = family
  (cmaps[label](1.0), shared with the dot-color legend), shape = pool
  (_POOL_MARKERS), size = large for 'real+adeq+flu' (Row 7, the branded
  SPA_synth25/PA_synth25 baseline) and smaller for the other 5 -- own
  legend (marker shape -> pool name) kept separate from the dot-color
  legend so the two encodings (dot color = reliability, marker shape =
  which pool) don't get conflated."""
  for label in families:
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
      Line2D([0], [0], color=cmaps[label](1.0), marker='o', linestyle='none', markersize=5,
             label=_PANEL_TITLES[label])
      for label in families
  ]
  # Both legends live BELOW the axes (bbox_to_anchor y<0, in axes
  # fraction coordinates), side by side when both are present -- kept out
  # of the plot area entirely rather than overlapping the dots/markers.
  # fig.savefig's bbox_inches='tight' (set by every caller) expands the
  # saved image to include them, so nothing is cut off.
  dot_anchor = (0.1, -0.14) if synth25_pools else (0.5, -0.14)
  dot_legend = ax.legend(handles=proxies, fontsize=8, loc='upper center', bbox_to_anchor=dot_anchor,
                          title='color = reliability, ESS', title_fontsize=7)

  if synth25_pools:
    pool_proxies = []
    for pool_name, info in synth25_pools.items():
      a0 = info.get('alpha0')
      if a0 is None or a0 != a0:  # excludes NaN/absent
        continue
      marker = _POOL_MARKERS.get(pool_name, 'o')
      size = _POOL_MARKER_SIZE.get(pool_name, _POOL_MARKER_SIZE_DEFAULT)
      drawn = False
      for label in families:  # NOT ('A','B','T','J') -- info can hold both T and J (T from the main
                               # A/B/T pools cache, J merged in separately), but only `families` (whichever
                               # of T/J the curve itself is using) should ever be drawn, or T and J -- both
                               # green -- would draw two overlapping/duplicate-looking markers per pool.
        y = info.get(label)
        if y is None or y != y or label not in cmaps:
          continue
        ax.scatter([a0], [y], marker=marker, s=size, color=cmaps[label](1.0), edgecolors='black',
                   linewidths=0.8, alpha=0.5, zorder=6)
        drawn = True
      if drawn:
        pool_proxies.append(Line2D([0], [0], marker=marker, linestyle='none', color='0.5',
                                    markeredgecolor='black', alpha=0.5,
                                    markersize=(7.5 if size == _POOL_MARKER_SIZE_LARGE else 4.6),
                                    label=pool_name))
    if pool_proxies:
      ax.add_artist(dot_legend)
      ax.legend(handles=pool_proxies, fontsize=7, loc='upper center', bbox_to_anchor=(0.95, -0.14), ncol=2,
                title=f'pool (marker shape) @ its own alpha_0 ({synth25_metric_label})', title_fontsize=7)


def _plot_family_vs_ess(data, family, K, donor_desc, synthesis_title, dial_preset_title, out_dir,
                         base, grid_suffix, metametric_suffix, synthesis_suffix, dial_preset_suffix,
                         aspectdonors_suffix, green_family_suffix=''):
  """One family's (A, B, T, or J) orientation score directly against
  ESS(alpha)/K (reliability fraction, not raw ESS -- matches
  plot_allmqm_orientation_vs_ess_pooled.py's pooled version so bucket
  widths mean the same thing in both places) rather than alpha -- ESS is
  orientation-neutral by construction for T/J (not for A/B, but the same
  ESS-reliability question applies to them too: does the metametric's
  adequacy/fluency preference get noisier as ESS drops?). Light
  (family-colored): every (alpha, donor) pair -- data[family] is already
  (n_donors, n_alpha), so this is a reshape, not new computation. Red:
  0.02-wide ESS/K buckets, averaging the raw (alpha, donor) points
  directly within each bucket (one averaging pass -- see
  plot_allmqm_orientation_vs_ess_pooled.py's docstring for why this, not a
  per-alpha donor-mean, is the natural aggregate), connected in x-order.
  No-op if this cache has no `family` (e.g. a cache missing T, or an older
  cache predating J)."""
  if data.get(family) is None:
    return
  F = data[family]  # (n_donors, n_alpha)
  ess_frac = data['ess'] / K  # (n_alpha,)
  n_donors, n_alpha = F.shape
  ess_frac_all = np.tile(ess_frac, n_donors)  # repeats per donor, matching F.ravel()'s donor-major order
  f_all = F.ravel()
  finite = ~np.isnan(f_all)
  x_raw, y_raw = ess_frac_all[finite], f_all[finite]

  n_bins = int(round(1.0 / _ESS_FRAC_BIN_WIDTH))
  bin_idx = np.clip((x_raw / _ESS_FRAC_BIN_WIDTH).astype(int), 0, n_bins - 1)
  bin_sums = np.bincount(bin_idx, weights=y_raw, minlength=n_bins)
  bin_counts = np.bincount(bin_idx, minlength=n_bins)
  with np.errstate(invalid='ignore'):
    bin_means = bin_sums / bin_counts
  bin_centers = (np.arange(n_bins) + 0.5) * _ESS_FRAC_BIN_WIDTH
  has_data = bin_counts > 0
  x_bucket, y_bucket = bin_centers[has_data], bin_means[has_data]

  fig, ax = plt.subplots(figsize=(6, 5))
  ax.set_box_aspect(1)
  ax.scatter(x_raw, y_raw, color=_DARK_COLOR[family], s=6, alpha=0.15, linewidths=0,
             zorder=2, label=f'{n_donors} donors x {n_alpha} alphas')
  ax.plot(x_bucket, y_bucket, color=_MEAN_RED, linewidth=1.3, alpha=0.7, zorder=3)
  ax.scatter(x_bucket, y_bucket, color=_MEAN_RED, s=22, alpha=0.85, linewidths=0, zorder=3,
             label=f'{len(x_bucket)} buckets of {_ESS_FRAC_BIN_WIDTH:g}-wide ESS/K')
  ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.02, 1.02)
  ax.set_xlabel('ESS(alpha) / K  (reliability fraction)')
  ax.set_ylabel(_PANEL_TITLES[family])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(fontsize=7, loc='lower right')
  fig.suptitle(f'{data["dataset"]}: {_PANEL_TITLES[family]} vs. ESS ({data["metametric"].upper()}, '
               f'K={K} systems, {donor_desc}{synthesis_title}{dial_preset_title})', fontsize=10)
  fig.tight_layout()

  plot_path = os.path.join(
      out_dir,
      f'{_FAMILY_FILE_PREFIX[family]}_orientation_vs_ess_{base}{grid_suffix}{metametric_suffix}'
      f'{synthesis_suffix}{dial_preset_suffix}{green_family_suffix}{aspectdonors_suffix}.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}', file=sys.stderr)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--n-steps', type=int, default=5)
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--full-range', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--union-grid', action=argparse.BooleanOptionalAction, default=True,
                       help='load the union-alpha_ij-grid cache (compute_scorer_orientation_vs_alpha.py '
                            '--union-grid) instead of the other grid modes. Default: True (the committed '
                            'setup).')
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='which cached weighted meta-metric to load/plot')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help='which cached lib.synthetic_scorers dial family to load/plot -- must match '
                            'the compute run\'s --synthesis')
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended', 'symmetric'], default='symmetric',
                       help='which cached dial grid to load/plot -- must match the compute run\'s '
                            "--dial-preset. Default: 'symmetric' (the committed setup, matching "
                            'compute_scorer_orientation_vs_alpha.py\'s own default).')
  parser.add_argument('--green-family', choices=['default', 'Jneg', 'ABTJ'], default='ABTJ',
                       help='must match the compute run\'s --green-family -- only affects the auto-derived '
                            'cache tag (appends _Jneg/_ABTJ, see compute_scorer_orientation_vs_alpha.py) '
                            'when --data/--tag is not given. Default: ABTJ (the committed setup).')
  parser.add_argument('--aspect-donors', action=argparse.BooleanOptionalAction, default=False,
                       help='load the ASPECT_DONORS cache (compute_scorer_orientation_vs_alpha.py '
                            '--aspect-donors) and draw one panel per donor instead of the '
                            'mean-across-donors single-axes plot')
  parser.add_argument('--per-donor', action=argparse.BooleanOptionalAction, default=False,
                       help="draw one square panel per donor already in the cache (data['donors'], "
                            'sorted -- a real_scorers-screened cache typically has dozens), each showing '
                            "that donor's OWN curves, no averaging -- a grid (not --aspect-donors' single "
                            'row, which only ever has 3 panels). Same underlying cache as the default '
                            'mean-across-donors plot; this only changes how it is drawn.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--data', type=str, default=None, help='explicit path to a scorer_orientation_*.npz cache')
  parser.add_argument('--overlay-synth25', action=argparse.BooleanOptionalAction, default=True,
                       help='overlay one marker per (pool, family) -- adequacy/fluency/allmqm orientation of '
                            'the SPA_synth25/PA_synth25 system-synthesis metametric (lib.synth25_orientation, '
                            'material/prior_work/metrics-dp-tradeoff.tex S2.4), for EACH of lib.synth25_'
                            'metametrics.POOL_BLOCKS\' 6 pools (real+adeq+flu = Row 7, the branded baseline, '
                            'plus 5 comparison pools), each plotted at that POOL\'S OWN natural alpha_0 (not '
                            'at every alpha -- these are fixed points, not curves), from the cache written by '
                            'compute_synth25_orientation.py --all-pools --metametric matching --metametric '
                            'above.')
  parser.add_argument('--vs-ess', action=argparse.BooleanOptionalAction, default=False,
                       help='ALSO generate the second family of figures -- each of A/B/T/J plotted directly '
                            'against ESS(alpha)/K (_plot_family_vs_ess) -- off by default: these used to be '
                            'generated unconditionally on every run as a mandatory side effect of the main '
                            'orientation-vs-alpha plot; now opt-in only, requested explicitly via this flag.')
  args = parser.parse_args()
  base = args.dataset
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)

  if args.data:
    data_path = args.data
  else:
    if args.tag:
      tag = args.tag
    elif args.union_grid:
      tag = union_grid_tag(base, args.step, args.metametric, args.synthesis, args.dial_preset)
    else:
      tag = orientation_tag(base, args.n_steps, args.step, args.full_range, args.metametric, args.synthesis,
                             args.dial_preset)
    if args.aspect_donors and not args.tag:
      tag += '_aspectdonors'
    if args.green_family in ('Jneg', 'ABTJ') and not args.tag:
      tag += f'_{args.green_family}'
    data_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  if not os.path.exists(data_path):
    sys.exit(f'{data_path} not found -- run compute_scorer_orientation_vs_alpha.py with matching flags first')

  data = load_orientation_data(data_path)
  print(f'Loaded {data_path}: dataset={data["dataset"]}, {len(data["donors"])} donors, '
        f'{len(data["alphas"])} alphas', file=sys.stderr)

  synth25_pools, synth25_metric_label = None, ''
  if args.overlay_synth25:
    synth25_metametric_name = f'{args.metametric}_synth25'
    assert synth25_metametric_name in SYNTH25_METAMETRICS
    synth25_tag = synth25_pools_tag(data['dataset'], synth25_metametric_name, args.synthesis, args.dial_preset)
    synth25_path = os.path.join(DATA_DIR, f'synth25_orientation_{synth25_tag}.npz')
    if not os.path.exists(synth25_path):
      sys.exit(f'{synth25_path} not found -- run compute_synth25_orientation.py --dataset {data["dataset"]} '
                f'--metametric {args.metametric} --synthesis {args.synthesis} --dial-preset {args.dial_preset} '
                f'--all-pools first')
    synth25_data = load_synth25_pools(synth25_path)
    synth25_pools = {
        pool_name: {
            'alpha0': synth25_data['pool_alpha0'][pool_name],
            'A': synth25_data['pool_A'][pool_name],
            'B': synth25_data['pool_B'][pool_name],
            'T': synth25_data['pool_T'][pool_name],
        }
        for pool_name in synth25_data['pool_names']
    }
    synth25_metric_label = synth25_metametric_name.upper()
    print(f'Loaded {synth25_path}: {synth25_metric_label} pools ({synth25_data["n_donors"]} donors):', file=sys.stderr)
    for pool_name, info in synth25_pools.items():
      print(f'  {pool_name}: alpha_0={info["alpha0"]:.4f} A={info["A"]:.4f} B={info["B"]:.4f} T={info["T"]:.4f}',
            file=sys.stderr)

    # This cache's own third family is J (not T) -- e.g. compute_scorer_
    # orientation_vs_alpha.py's --green-family Jneg/ABTJ, which keeps A/B
    # (and, for ABTJ, T) from some other --synthesis but ALSO builds J from
    # an offset call. lib.synth25_orientation never computes J alongside
    # A/B/T in the SAME call (see its module docstring), so it lives in a
    # SEPARATE cache (pools_tag_J, tagged 'neg' or 'symmetric' depending on
    # which dial grid J itself used -- 'symmetric' when this cache's OWN
    # dial_preset is 'symmetric' i.e. an ABTJ cache, 'neg'/historical
    # NEG_J_DIAL_GRID otherwise, e.g. plain Jneg) -- merge its 'J' value
    # into each pool's dict alongside the A/B/T already loaded above,
    # rather than trying to load ONE cache with everything.
    if data.get('J') is not None:
      j_dial_preset = 'symmetric' if data['dial_preset'] == 'symmetric' else 'neg'
      synth25_j_tag = synth25_pools_tag_J(data['dataset'], synth25_metametric_name, dial_preset=j_dial_preset)
      synth25_j_path = os.path.join(DATA_DIR, f'synth25_orientation_{synth25_j_tag}.npz')
      if not os.path.exists(synth25_j_path):
        dial_preset_hint = ' --dial-preset symmetric' if j_dial_preset == 'symmetric' else ''
        sys.exit(f'{synth25_j_path} not found -- run compute_synth25_orientation.py --dataset '
                  f'{data["dataset"]} --metametric {args.metametric} --J{dial_preset_hint} --all-pools first')
      synth25_j_data = load_synth25_pools_J(synth25_j_path)
      for pool_name in synth25_j_data['pool_names']:
        if pool_name in synth25_pools:
          synth25_pools[pool_name]['J'] = synth25_j_data['pool_J'][pool_name]
      print(f'Loaded {synth25_j_path}: J pools ({synth25_j_data["n_donors"]} donors):', file=sys.stderr)
      for pool_name in synth25_j_data['pool_names']:
        print(f'  {pool_name}: J={synth25_j_data["pool_J"][pool_name]:.4f}', file=sys.stderr)

  K = data['K']
  ess_over_k = data['ess'] / K
  cutoff_frac = _ESS_CUTOFF_ABS / K
  norm = SplineReliabilityNorm(vmin=cutoff_frac, vmax=1.0)

  # Third/fourth families: 'T' (orientation-neutral All-MQM) and/or 'J'
  # (orientation-neutral Joint) -- whichever are present. Most caches have
  # at most one (additive/additive_mean's own T, or offset's own J /
  # --green-family Jneg's swapped-in J); --green-family ABTJ caches have
  # BOTH at once, so this is not an either/or.
  families = ['A', 'B']
  if data.get('T') is not None:
    families.append('T')
  if data.get('J') is not None:
    families.append('J')
  families = tuple(families)
  # Built for every possible family (not just `families`), not just the
  # ones this cache's curves happen to use: the --overlay-synth25 pool
  # markers always report A/B/T (lib.synth25_orientation never computes
  # J), so an 'offset' cache (whose OWN third family is J) still needs a
  # 'T' cmap available to draw its T-family pool markers alongside J's dots.
  cmaps = {label: _ess_over_k_cmap(_DARK_COLOR[label]) for label in ('A', 'B', 'T', 'J')}

  if args.aspect_donors:
    # One square panel per ASPECT_DONORS entry (fixed order, not
    # data['donors']'s alphabetical one), each showing that donor's OWN
    # family curves -- unlike the mean-across-donors default, there's only
    # one "donor" per panel here, so no averaging.
    donor_row = {d: i for i, d in enumerate(data['donors'])}
    fig, axes = plt.subplots(1, len(ASPECT_DONORS), figsize=(6 * len(ASPECT_DONORS), 6.5))
    for ax, donor in zip(axes, ASPECT_DONORS):
      ax.set_box_aspect(1)
      curves = {label: data[label][donor_row[donor]] for label in families}
      _draw_orientation_axes(ax, data['alphas'], curves, ess_over_k, cmaps, norm, data['center_alpha'],
                              families=families)
      ax.set_title(donor, fontsize=10)
    strip_ax = axes[-1]
    donor_desc = '/'.join(ASPECT_DONORS)
  elif args.per_donor:
    # One square panel per donor already in this cache -- a grid, since a
    # real_scorers-screened cache typically has dozens of donors (unlike
    # --aspect-donors' fixed 3), sized to roughly a square layout with any
    # leftover trailing slots hidden.
    donor_order = data['donors']
    n = len(donor_order)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.8 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, donor in enumerate(donor_order):
      ax = axes_flat[idx]
      ax.set_box_aspect(1)
      curves = {label: data[label][idx] for label in families}
      _draw_orientation_axes(ax, data['alphas'], curves, ess_over_k, cmaps, norm, data['center_alpha'],
                              families=families)
      ax.set_title(donor, fontsize=7)
    for ax in axes_flat[n:]:
      ax.axis('off')
    strip_ax = axes_flat[n - 1]
    donor_desc = f'{n} donors (no averaging)'
  else:
    # data['A']/data['B']/data['T'] are (n_donors, n_alpha) -- per-donor
    # shadow curves, kept for later use (see _draw_orientation_axes), not
    # drawn here.
    mean_curves = {label: np.nanmean(data[label], axis=0) for label in families}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_box_aspect(1)  # square PLOT box -- independent of the title/colorbar space around it
    _draw_orientation_axes(ax, data['alphas'], mean_curves, ess_over_k, cmaps, norm, data['center_alpha'],
                            families=families, synth25_pools=synth25_pools,
                            synth25_metric_label=synth25_metric_label)
    strip_ax = ax
    donor_desc = f'{len(data["donors"])} donors'

  synthesis_title = '' if data['synthesis'] == 'offset' else f', synthesis={data["synthesis"]}'
  dial_preset_title = '' if data['dial_preset'] == 'linear' else f', dial_preset={data["dial_preset"]}'
  fig.suptitle(f'{data["dataset"]}: scorer orientation vs. alpha ({data["metametric"].upper()}, '
               f'K={K} systems, {donor_desc}, '
               f'alpha_0(D)={data["center_alpha"]:.4f}{synthesis_title}{dial_preset_title})', fontsize=11)
  fig.tight_layout(rect=[0, 0, 0.85, 0.95])

  # Strips are added AFTER tight_layout, positioned off strip_ax's FINAL
  # bbox (the single axes, or the rightmost panel in the --aspect-donors
  # case, since ESS(alpha) is shared across every panel) -- adding them
  # before would have tight_layout shove the panel around without the
  # strips following. One strip per active family, flush side by side
  # (each offset = previous + strip width, no gap): A (blue), B (red),
  # then T or J (green) if present -- only the last one gets the axis label.
  _STRIP_WIDTH = 0.012
  for i, label in enumerate(families):
    _draw_reliability_strip(fig, strip_ax, cmaps[label], norm, K, _ESS_CUTOFF_ABS,
                             x_offset=0.02 + i * _STRIP_WIDTH, label=(i == len(families) - 1))

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  metametric_suffix = '' if data['metametric'] == 'spa' else f'_{data["metametric"]}'
  synthesis_suffix = '' if data['synthesis'] == 'offset' else f'_{data["synthesis"]}'
  dial_preset_suffix = {'linear': '', 'geometric': '_geom', 'extended': '_ext', 'symmetric': '_sym'}[data['dial_preset']]
  # This cache's third family is J even though data['synthesis'] != 'offset'
  # (compute_scorer_orientation_vs_alpha.py --green-family Jneg) -- WITHOUT
  # this, the filename would be identical to the plain synthesis-only
  # (T-family) cache's own plot and silently overwrite it (confirmed
  # empirically -- caught and fixed here).
  green_family_suffix = '_Jneg' if (data.get('J') is not None and data['synthesis'] != 'offset') else ''
  grid_suffix = '_union' if args.union_grid else ''
  aspectdonors_suffix = '_aspectdonors' if args.aspect_donors else ('_perdonor' if args.per_donor else '')
  synth25_suffix = '_synth25' if args.overlay_synth25 else ''
  if data.get('T') is not None and data.get('J') is not None:
    # --green-family ABTJ cache (both T and J at once) -- short, distinct
    # name instead of the long auto-chain above (which was designed around
    # exactly one of T/J ever being present at a time, and is already long
    # enough without a 5th thing to disambiguate).
    plot_path = os.path.join(ARTIFACTS_DIR, f'orientation_{base}_ABTJ{aspectdonors_suffix}{synth25_suffix}.png')
  else:
    plot_path = os.path.join(
        ARTIFACTS_DIR,
        f'orientation_vs_alpha_{base}{grid_suffix}{metametric_suffix}{synthesis_suffix}{dial_preset_suffix}'
        f'{green_family_suffix}{aspectdonors_suffix}{synth25_suffix}.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}', file=sys.stderr)

  # Second family of figures: each of A (adequacy), B (fluency), and the
  # cache's third family (T=AllMQM or J=Joint) plotted directly against
  # ESS(alpha)/K -- see _plot_family_vs_ess's docstring for the full
  # rationale. _plot_family_vs_ess no-ops for whichever of T/J is absent.
  # Opt-in only (--vs-ess) -- no longer a mandatory side effect of every run.
  if args.vs_ess:
    for family in ('A', 'B', 'T', 'J'):
      _plot_family_vs_ess(data, family, K, donor_desc, synthesis_title, dial_preset_title, ARTIFACTS_DIR,
                           base, grid_suffix, metametric_suffix, synthesis_suffix, dial_preset_suffix,
                           aspectdonors_suffix, green_family_suffix)
