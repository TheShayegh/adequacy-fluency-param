"""ARCHIVED / OBSOLETE -- reference only. Not imported by anything, not
meant to be run.

Preserves the "type 1" and "type 2" plot styles (colors, ESS colormap
conventions, curve interpolation, anchor-dot patterns) from the retired
type1-6 consistency-curve experiment family, after the active scripts
(plot_consistency_curves.py, plot_per_dataset_consistency_curves.py) and
the compute pipeline that fed them were deleted from codes/scripts/ during
cleanup (their input JSON schema -- artifacts/data/curve_<metametric>
[_TAG].json and curve_perdataset_<metametric>[_TAG].json -- has no
producer left in this repo). Kept because the visual conventions took real
design iteration and are worth copying from again later, even though the
pipeline that fed them is gone for good.

Two sections below, each a near-verbatim copy of one retired script's
__main__ body, wrapped as a function so this file stays import-safe (does
nothing on import, defines nothing that runs automatically):

  - render_type1_figure(): pooled M(alpha) x consistency curve, one line
    per meta-metric. ESS-driven white -> light blue -> standard blue ->
    black -> red colormap (fully transparent... no, OPAQUE white below the
    actually-achieved ESS minimum, fading through blue to black at the
    peak, red beyond it toward the theoretical ceiling).
    Originally: codes/scripts/plot_consistency_curves.py

  - render_type2_figure(): star-pooled per-dataset curves, one line per
    dataset. ESS/K-normalized colormap, transparent at/below K/2 then a
    linear fade-in through light blue to dark blue by K, with a red-circle
    (natural/unreweighted anchor) + white-dot (this curve's own value at
    alpha_0) + connecting-line convention that makes the gap between "this
    dataset's own natural balance" and "the curve's value there" visible.
    Originally: codes/scripts/plot_per_dataset_consistency_curves.py

Neither function is called anywhere in this file. To actually render
something you'd need to reconstruct the curve_<metametric>.json /
curve_perdataset_<metametric>.json schema (see each function's `curves`
parameter for the expected shape, inferred from usage) with a new producer
-- these functions only encode HOW to draw it, not how to compute it.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')

# ---- shared style constants (originally defined in plot_consistency_
# curves.py / "type 1", imported from there by "type 2") ----
TITLES = {
    'pearson': 'Pearson', 'spearman': 'Spearman', 'kendall': "Kendall's $\\tau$",
    'pa': 'Pairwise Accuracy', 'spa': 'Soft Pairwise Accuracy',
}
LIGHT_BLUE = (0.55, 0.75, 0.92)
STANDARD_BLUE = (0.121, 0.466, 0.705)  # matplotlib "tab:blue"
DARK_BLUE = (0.03, 0.15, 0.35)  # type-2 only


def interpolate_curve(alphas, ys, ess, n_sub=50):
  """Linearly densifies (alphas, ys, ess) by n_sub points per original
  interval. LineCollection colors each segment by the average of its two
  endpoints, which dilutes a single extreme vertex (e.g. the true ESS
  minimum) into its neighbor's color when segments span a whole original
  grid interval. Densifying first shrinks each segment's span, so the
  color right at any given vertex converges to that vertex's own exact
  value instead of being averaged away -- what makes the true minimum
  actually render as its own (e.g. pure white) color rather than a blend."""
  fine_a, fine_y, fine_e = [], [], []
  for i in range(len(alphas) - 1):
    t = np.linspace(0.0, 1.0, n_sub, endpoint=False)
    fine_a.append(alphas[i] + t * (alphas[i + 1] - alphas[i]))
    fine_y.append(ys[i] + t * (ys[i + 1] - ys[i]))
    fine_e.append(ess[i] + t * (ess[i + 1] - ess[i]))
  fine_a.append([alphas[-1]])
  fine_y.append([ys[-1]])
  fine_e.append([ess[-1]])
  return np.concatenate(fine_a), np.concatenate(fine_y), np.concatenate(fine_e)


# =========================================================================
# TYPE 1 STYLE -- pooled M(alpha) x consistency curve
# originally codes/scripts/plot_consistency_curves.py
# =========================================================================

def load_curve_type1(metametric: str, suffix: str) -> dict:
  path = os.path.join(DATA_DIR, f'curve_{metametric}{suffix}.json')
  with open(path) as f:
    return json.load(f)


def load_loo_shadow_curves(metametric: str, suffix: str, datasets: list[str]) -> list[dict]:
  """One leave-one-dataset-out curve per dataset in `datasets` (written by
  the now-deleted compute_loo_shadow_curves.py), skipping any not yet
  computed rather than erroring -- shadows are supplementary, never
  required for the main plot."""
  out = []
  for excluded in datasets:
    path = os.path.join(DATA_DIR, f'curve_{metametric}{suffix}_shadow_excl_{excluded}.json')
    if not os.path.exists(path):
      continue
    with open(path) as f:
      out.append(json.load(f))
  return out


def dataset_alpha0s(datasets: list[str], root: str) -> dict[str, float]:
  """Each entry's own natural (uniform-weight) balance -- for marking where
  a real dataset's unreweighted alpha_0 falls on the x-axis. Entries that
  aren't loadable as a real dataset (e.g. bootstrap repeat ids like
  'ende24_boot3') are silently skipped rather than erroring."""
  from lib.alpha import alpha_0 as alpha0_of
  from lib.consistency import real_systems
  from mwb.mqm_scoring import load_system_scores
  out = {}
  for d in datasets:
    try:
      systems = real_systems(d, root=root)
      sys_df = load_system_scores(d, root=root).loc[systems]
      out[d] = alpha0_of(sys_df['a'].values, sys_df['b'].values)
    except (FileNotFoundError, KeyError):
      continue
  return out


def render_type1_figure(metametrics_order, tag=None, title=None, loo_shadows=False):
  """Reference copy of plot_consistency_curves.py's __main__ body, wrapped
  as a function -- NOT wired to run anywhere in this file; preserved for
  its exact style logic (colormap construction, LineCollection usage,
  legend/colorbar layout). `curves[m]` is expected to have the shape
  written by the retired compute_consistency_curve.py: {'alphas',
  'tau_bar', 'mean_ess', 'mean_K', 'baseline_tau_bar', 'datasets',
  optionally 'alpha_0'}."""
  suffix = f'_{tag}' if tag else ''
  out_path = os.path.join(ROOT, 'artifacts', f'consistency_curves{suffix}.png')

  curves = {m: load_curve_type1(m, suffix) for m in metametrics_order}
  mean_K = curves[metametrics_order[0]]['mean_K']
  if 'alpha_0' in curves[metametrics_order[0]]:
    per_dataset_alpha0 = curves[metametrics_order[0]]['alpha_0']
  else:
    per_dataset_alpha0 = dataset_alpha0s(curves[metametrics_order[0]]['datasets'], ROOT)

  # ESS is identical across metametrics (it only depends on w(alpha), shared
  # from the same w_cache), so there's one actually-achieved min/peak for
  # the whole figure. Opaque white exactly at the actually-achieved
  # minimum -- flat white below it too (that span, [2, ess_min), is never
  # visited by the curve, so its color is arbitrary, but keeping it flat
  # white avoids implying a fade that doesn't correspond to real data);
  # then white -> light blue -> standard blue -> pure black at the
  # achieved peak; beyond it, up to mean(K) (the theoretical ceiling the
  # dashed baseline is informally pinned to, practically unreachable by
  # the curve) -> red.
  ess_min = min(min(c['mean_ess']) for c in curves.values())
  ess_peak = max(max(c['mean_ess']) for c in curves.values())
  frac_min = min(max((ess_min - 2) / (mean_K - 2), 0.0), 1 - 1e-6)
  frac_peak = min(max((ess_peak - 2) / (mean_K - 2), frac_min + 1e-6), 1 - 1e-6)
  frac_third = frac_min + (frac_peak - frac_min) / 3.0
  frac_twothirds = frac_min + (frac_peak - frac_min) * 2.0 / 3.0
  cmap = LinearSegmentedColormap.from_list('ess_fade', [
      (0.0, (1, 1, 1, 1.0)),
      (frac_min, (1, 1, 1, 1.0)),
      (frac_third, (*LIGHT_BLUE, 1.0)),
      (frac_twothirds, (*STANDARD_BLUE, 1.0)),
      (frac_peak, (0, 0, 0, 1.0)),
      (1.0, (1, 0, 0, 1.0)),
  ])
  norm = Normalize(vmin=2, vmax=mean_K)

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  axes = axes.flatten()
  left_column = {0, 3}

  for i, (ax, m) in enumerate(zip(axes, metametrics_order)):
    c = curves[m]
    alphas = np.array(c['alphas'])
    tau = np.array(c['tau_bar'])
    ess = np.array(c['mean_ess'])

    shadow_taus = []
    if loo_shadows:
      for shadow in load_loo_shadow_curves(m, suffix, c['datasets']):
        s_alphas = np.array(shadow['alphas'])
        s_tau = np.array(shadow['tau_bar'])
        s_ess = np.array(shadow['mean_ess'])
        shadow_taus.append(s_tau)
        fine_sa, fine_st, fine_se = interpolate_curve(s_alphas, s_tau, s_ess)
        s_points = np.array([fine_sa, fine_st]).T.reshape(-1, 1, 2)
        s_segments = np.concatenate([s_points[:-1], s_points[1:]], axis=1)
        s_seg_ess = (fine_se[:-1] + fine_se[1:]) / 2.0
        lc_shadow = LineCollection(s_segments, cmap=cmap, norm=norm)
        lc_shadow.set_array(s_seg_ess)
        lc_shadow.set_linewidth(2.5)
        lc_shadow.set_joinstyle('round')
        lc_shadow.set_capstyle('round')
        lc_shadow.set_alpha(0.01)
        ax.add_collection(lc_shadow)

    fine_alphas, fine_tau, fine_ess = interpolate_curve(alphas, tau, ess)
    points = np.array([fine_alphas, fine_tau]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_ess = (fine_ess[:-1] + fine_ess[1:]) / 2.0
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_ess)
    lc.set_linewidth(2.5)
    lc.set_joinstyle('round')
    lc.set_capstyle('round')
    ax.add_collection(lc)

    ax.axhline(c['baseline_tau_bar'], color='red', linestyle='--', linewidth=1.5)
    for a0 in per_dataset_alpha0.values():
      ax.axvline(a0, color='black', alpha=0.15, linewidth=1)

    ax.set_xlim(alphas.min(), alphas.max())
    y_all = np.concatenate([tau, [c['baseline_tau_bar']]] + shadow_taus)
    pad = 0.05 * (y_all.max() - y_all.min() + 1e-9)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_xlabel('Controlled Balance\n(Adequacy over Fluency)')
    if i in left_column:
      ax.set_ylabel('Consistency over Years')
    ax.set_title(TITLES[m])
    ax.set_box_aspect(1)

  for ax in axes[len(metametrics_order):]:
    ax.axis('off')

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.02)
  cbar.set_label('Reliability (ESS)')

  proxy_blue = Line2D([0], [0], color=LIGHT_BLUE, lw=2.5, label='*Reweighted by Controlled Balance')
  proxy_red = Line2D([0], [0], color='red', ls='--', lw=1.5, label='WMT (Uncontrolled Balance)')
  proxy_black = Line2D([0], [0], color='black', alpha=0.15, lw=1, label='Per-Year Balance')
  legend_handles = [proxy_blue, proxy_red, proxy_black]
  if loo_shadows:
    proxy_shadow = Line2D([0], [0], color=STANDARD_BLUE, alpha=0.3, lw=2.5,
                           label='Leave-One-Year-Out (shadow)')
    legend_handles.append(proxy_shadow)
  slot = axes[len(metametrics_order)].get_position()
  slot_center = (slot.x0 + slot.width / 2, slot.y0 + slot.height / 2)
  fig.legend(
      handles=legend_handles, loc='center',
      bbox_to_anchor=slot_center, fontsize=11, frameon=False)

  if title:
    fig.suptitle(title, fontsize=14)
  else:
    title_suffix = f' ({tag})' if tag else ''
    fig.suptitle(
        r'Cross-dataset scorer-ranking consistency vs. reweighted balance $\alpha$' + title_suffix
        + '\n(action_plan.md 6.1 x 6.5, numeric solver)',
        fontsize=14,
    )
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')


# =========================================================================
# TYPE 2 STYLE -- star-pooled per-dataset consistency curves
# originally codes/scripts/plot_per_dataset_consistency_curves.py
# =========================================================================

def load_curve_type2(metametric: str, suffix: str) -> dict:
  path = os.path.join(DATA_DIR, f'curve_perdataset_{metametric}{suffix}.json')
  with open(path) as f:
    return json.load(f)


def render_type2_figure(metametrics_order, tag=None, title=None):
  """Reference copy of plot_per_dataset_consistency_curves.py's __main__
  body, wrapped as a function -- NOT wired to run anywhere in this file;
  preserved for its exact style logic (ESS/K colormap, the red-circle +
  white-dot + connector anchor convention, the outside-the-axes label
  placement heuristic). `curves[m]` is expected to have the shape written
  by the retired compute_per_dataset_consistency_curve.py: {'datasets',
  'K', 'alpha_0', 'alphas', 'tau_bar_by_dataset', 'ess_by_dataset',
  'baseline_tau_by_dataset'}."""
  suffix = f'_{tag}' if tag else ''
  out_path = os.path.join(ROOT, 'artifacts', f'consistency_curves_per_dataset{suffix}.png')

  curves = {m: load_curve_type2(m, suffix) for m in metametrics_order}
  datasets = curves[metametrics_order[0]]['datasets']
  K = curves[metametrics_order[0]]['K']
  alpha_0 = curves[metametrics_order[0]]['alpha_0']

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

  all_series = {}
  all_alphas = {}

  for i, (ax, m) in enumerate(zip(axes, metametrics_order)):
    c = curves[m]
    alphas = np.array(c['alphas'])
    all_alphas[m] = alphas
    x_lo, x_hi = alphas.min(), alphas.max()
    y_all = []

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
    ax.set_anchor('N')

  for ax in axes[len(metametrics_order):]:
    ax.axis('off')

  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  fig_bbox = fig.get_window_extent(renderer=renderer)

  # Annotate each red dot OUTSIDE its own subplot's box (never over the
  # data area, so it can never sit on top of a line): for each of the 4
  # directions (left/right/up/down), computes how far the dot already is
  # from that edge of the axes' pixel bounding box, then takes whichever
  # direction requires the smallest push to clear it (plus a margin).
  margin_px = 18
  min_label_gap_px = 24
  for i, (ax, m) in enumerate(zip(axes, metametrics_order)):
    series = all_series[m]
    dpi_scale = ax.figure.dpi / 72.0
    ax_bbox = ax.get_window_extent(renderer=renderer)
    placements = {}
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

  slot = axes[len(metametrics_order)].get_position()
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

  if title:
    fig.suptitle(title, fontsize=14)
  else:
    title_suffix = f' ({tag})' if tag else ''
    fig.suptitle(
        r'Cross-dataset scorer-ranking consistency vs. reweighted balance $\alpha$' + title_suffix
        + '\n(action_plan.md 6.1 x 6.5, numeric solver)',
        fontsize=14,
    )
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
