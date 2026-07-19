"""Plots the M(alpha) x consistency curves (action_plan.md 6.1 x 6.5)
computed by compute_consistency_curve.py -- one artifacts/data/
curve_<metametric>.json per meta-metric -- into a single 5-subplot figure,
saved as artifacts/consistency_curves.png. Pure plotting: does no solving
or scoring itself, so restyling the figure never re-pays the reweighting
solver's cost.

Usage: python codes/scripts/plot_consistency_curves.py [--tag TAG]
(run compute_reweight_cache.py and compute_consistency_curve.py <m> for
each of the 5 meta-metrics first, with the same --tag if any.)
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
from matplotlib.lines import Line2D

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')

METAMETRICS_ORDER = ['pearson', 'spearman', 'kendall', 'pa', 'spa']
TITLES = {
    'pearson': 'Pearson', 'spearman': 'Spearman', 'kendall': "Kendall's $\\tau$",
    'pa': 'Pairwise Accuracy', 'spa': 'Soft Pairwise Accuracy',
}
LIGHT_BLUE = (0.55, 0.75, 0.92)
STANDARD_BLUE = (0.121, 0.466, 0.705)  # matplotlib "tab:blue"


def load_curve(metametric: str, suffix: str) -> dict:
  path = os.path.join(DATA_DIR, f'curve_{metametric}{suffix}.json')
  with open(path) as f:
    return json.load(f)


def load_loo_shadow_curves(metametric: str, suffix: str, datasets: list[str]) -> list[dict]:
  """One leave-one-dataset-out curve per dataset in `datasets` (written by
  compute_loo_shadow_curves.py), skipping any not yet computed rather than
  erroring -- shadows are supplementary, never required for the main plot."""
  out = []
  for excluded in datasets:
    path = os.path.join(DATA_DIR, f'curve_{metametric}{suffix}_shadow_excl_{excluded}.json')
    if not os.path.exists(path):
      continue
    with open(path) as f:
      out.append(json.load(f))
  return out


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


def dataset_alpha0s(datasets: list[str], root: str) -> dict[str, float]:
  """Each entry's own natural (uniform-weight) balance -- for marking where
  a real dataset's unreweighted alpha_0 falls on the x-axis. Entries that
  aren't loadable as a real dataset (e.g. bootstrap repeat ids like
  'ende24_boot3') are silently skipped rather than erroring."""
  out = {}
  for d in datasets:
    try:
      systems = real_systems(d, root=root)
      sys_df = load_system_scores(d, root=root).loc[systems]
      out[d] = alpha0_of(sys_df['a'].values, sys_df['b'].values)
    except (FileNotFoundError, KeyError):
      continue
  return out


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, default=None,
                  help='must match the tag used with compute_reweight_cache.py/compute_consistency_curve.py, if any')
  p.add_argument('--title', type=str, default=None,
                  help='override the figure suptitle entirely (default: auto-generated)')
  p.add_argument('--loo-shadows', action='store_true',
                  help='overlay faint leave-one-dataset-out curves (needs compute_loo_shadow_curves.py run first)')
  args = p.parse_args()
  suffix = f'_{args.tag}' if args.tag else ''
  out_path = os.path.join(ROOT, 'artifacts', f'consistency_curves{suffix}.png')

  curves = {m: load_curve(m, suffix) for m in METAMETRICS_ORDER}
  mean_K = curves[METAMETRICS_ORDER[0]]['mean_K']
  # Prefer alpha_0 embedded directly in the curve JSON (written by
  # producers whose "datasets" aren't real WMT dataset keys resolvable via
  # real_systems/load_system_scores, e.g. compute_loo_subset_type1_curve.py's
  # "excl_<system>" subset ids); fall back to the usual lookup-by-name path
  # for ordinary dataset groups, whose curve JSON has no such field.
  if 'alpha_0' in curves[METAMETRICS_ORDER[0]]:
    per_dataset_alpha0 = curves[METAMETRICS_ORDER[0]]['alpha_0']
  else:
    per_dataset_alpha0 = dataset_alpha0s(curves[METAMETRICS_ORDER[0]]['datasets'], ROOT)

  # ESS is identical across metametrics (it only depends on w(alpha), shared
  # from the same w_cache), so there's one actually-achieved min/peak for
  # the whole figure. Opaque (not transparent) pure white exactly at the
  # actually-achieved minimum -- flat white below it too (that span, [2,
  # ess_min), is never visited by the curve, so its color is arbitrary, but
  # keeping it flat white avoids implying a fade that doesn't correspond to
  # real data); then white -> light blue -> standard blue -> pure black at
  # the achieved peak; beyond it, up to mean(K) (the theoretical ceiling the
  # dashed baseline is informally pinned to, practically unreachable by the
  # curve) -> red.
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

  for i, (ax, m) in enumerate(zip(axes, METAMETRICS_ORDER)):
    c = curves[m]
    alphas = np.array(c['alphas'])
    tau = np.array(c['tau_bar'])
    ess = np.array(c['mean_ess'])

    # Shadows first, main curve last -- z-order = draw order, so the fully
    # opaque main line ends up on top of the translucent leave-one-out
    # cloud instead of being partly buried under it.
    shadow_taus = []
    if args.loo_shadows:
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
    seg_ess = (fine_ess[:-1] + fine_ess[1:]) / 2.0  # avg over a now-tiny sub-interval
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_ess)
    lc.set_linewidth(2.5)
    # Many short, near-solid-colored segments (from interpolate_curve) show
    # visible seams/hatching at their joints with the default miter/butt
    # style, especially on steep stretches -- round joins fix it cleanly.
    lc.set_joinstyle('round')
    lc.set_capstyle('round')
    ax.add_collection(lc)

    ax.axhline(c['baseline_tau_bar'], color='red', linestyle='--', linewidth=1.5)
    for a0 in per_dataset_alpha0.values():
      ax.axvline(a0, color='black', alpha=0.3, linewidth=1)

    ax.set_xlim(alphas.min(), alphas.max())
    y_all = np.concatenate([tau, [c['baseline_tau_bar']]] + shadow_taus)
    pad = 0.05 * (y_all.max() - y_all.min() + 1e-9)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_xlabel('Controlled Balance\n(Adequacy over Fluency)')
    if i in left_column:
      ax.set_ylabel('Consistency over Years')
    ax.set_title(TITLES[m])
    ax.set_box_aspect(1)  # square subplot box, independent of the alpha/tau data ranges

  for ax in axes[len(METAMETRICS_ORDER):]:
    ax.axis('off')

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  # Include the legend-slot axis (axes[5]) in the colorbar's layout group,
  # not just the 5 plotted ones -- otherwise its position isn't adjusted
  # for the colorbar's reserved space and the colorbar ends up drawn on
  # top of the legend (both sit in the grid's rightmost column).
  cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.02)
  cbar.set_label('Reliability (ESS)')

  proxy_blue = Line2D([0], [0], color=LIGHT_BLUE, lw=2.5, label='*Reweighted by Controlled Balance')
  proxy_red = Line2D([0], [0], color='red', ls='--', lw=1.5, label='WMT (Uncontrolled Balance)')
  proxy_black = Line2D([0], [0], color='black', alpha=0.3, lw=1, label='Per-Year Balance')
  legend_handles = [proxy_blue, proxy_red, proxy_black]
  if args.loo_shadows:
    # Legend swatch kept more visible than the actual shadow alpha (0.03,
    # imperceptible at legend size) so the entry stays legible.
    proxy_shadow = Line2D([0], [0], color=STANDARD_BLUE, alpha=0.3, lw=2.5,
                           label='Leave-One-Year-Out (shadow)')
    legend_handles.append(proxy_shadow)
  # A figure-level legend (not axes[5].legend()) isn't clipped to that one
  # subplot's narrow box, so longer labels have room. Anchor it at axes[5]'s
  # own center, read off *after* the colorbar call above (which resizes
  # every axes' position) so it lands in the actual empty grid slot rather
  # than a hand-guessed figure coordinate.
  slot = axes[len(METAMETRICS_ORDER)].get_position()
  slot_center = (slot.x0 + slot.width / 2, slot.y0 + slot.height / 2)
  fig.legend(
      handles=legend_handles, loc='center',
      bbox_to_anchor=slot_center, fontsize=11, frameon=False)

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
