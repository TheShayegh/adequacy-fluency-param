"""Integrates the synthetic scorer families (material/synthetic_scorer_
construction.md, lib.synthetic_scorers) into the reweighting meta-evaluation
machinery (material/action_plan.md sections 3, 5, 6.1): one figure per real
donor scorer, two heatmap panels each (Adequacy family / Fluency family).

Panel axes: y = synthesis dial (A or B, section 4 of the construction doc,
[0, 0.5] by 0.05 by default -- --dial-lo/--dial-hi/--dial-step), x =
meta-evaluation balance alpha (action_plan.md section 3.2), centered on the
pool's own natural balance alpha_0(D) with 5 steps of 0.02 in each direction
(lib.synthetic_scorer_alpha_grid.alpha_grid_around).

Each cell (dial, alpha) encodes up to TWO quantities as a 2-D colorbar:
  - ESS(w*(alpha)) (action_plan.md section 5) -> hue, red (lowest observed)
    to blue (highest observed) -- constant down a column, since w*(alpha)
    does not depend on the dial. Toggled via --show-ess/--no-show-ess,
    OFF by default -- when off, every cell is flat blue (no hue channel)
    and the ESS colorbar is omitted.
  - weighted SPA of the dialed synthetic scorer against All MQM at that
    (dial, alpha) -> opacity, transparent (lowest observed) to solid
    (highest observed). Always on.

Both channels are rescaled to their OBSERVED range WITHIN EACH DONOR'S OWN
PLOT (that donor's A and B grids pooled, not the theoretical [1,K]/[0,1]
bounds and not pooled across donors): in the default 5-step-by-0.02 window
around alpha_0(D), ESS stays close to K and SPA stays in a fairly tight
band for most donors, so a fixed/shared scale left nearly every cell
looking identically saturated. Ranges are therefore independent per plot --
the same color or opacity in two different donors' figures does NOT mean
the same underlying SPA/ESS value; read each figure's own legend/colorbar.

Usage: python codes/scripts/plot_synthetic_scorer_alpha_heatmaps.py [--dataset ende21]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from lib.alpha import alpha_0, alpha_min_max
from lib.consistency import real_systems
from lib.metric_scores import load_metric_sys_scores
from lib.reweighted_consistency import solve_w_for_datasets
from lib.synthetic_scorer_alpha_grid import alpha_grid_around, donor_alpha_dial_grid
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_synthetic_alpha_heatmaps')

ESS_CMAP = LinearSegmentedColormap.from_list('ess_red_blue', ['#c62828', '#1565c0'])
_FLAT_BLUE = np.array([0.086, 0.396, 0.753])  # ESS_CMAP's high (blue) end, used when the ESS toggle is off
_PANEL_TITLES = {'A': 'Adequacy family (dial A)', 'B': 'Fluency family (dial B)'}


def _panel_rgba(
    grid: np.ndarray, ess_arr: np.ndarray, ess_norm: Normalize, spa_norm: Normalize, show_ess: bool,
) -> np.ndarray:
  n_d, n_a = grid.shape
  rgba = np.zeros((n_d, n_a, 4))
  for ai in range(n_a):
    rgb = np.array(ESS_CMAP(ess_norm(ess_arr[ai]))[:3]) if show_ess else _FLAT_BLUE
    rgba[:, ai, :3] = rgb
    spa = grid[:, ai]
    valid = spa == spa  # excludes NaN
    rgba[valid, ai, 3] = np.clip(spa_norm(spa[valid]), 0.0, 1.0)
  return rgba


def _draw_panel(ax, grid, dials, alphas, ess_arr, ess_norm, spa_norm, title, center_alpha, show_ess):
  rgba = _panel_rgba(grid, ess_arr, ess_norm, spa_norm, show_ess)
  n_d, n_a = grid.shape
  ax.imshow(rgba, origin='lower', aspect='auto', extent=[-0.5, n_a - 0.5, -0.5, n_d - 0.5])
  ax.set_xticks(range(n_a))
  ax.set_xticklabels([f'{a:.3f}' for a in alphas], rotation=60, fontsize=6)
  ax.set_yticks(range(n_d))
  ax.set_yticklabels([f'{d:.2f}' for d in dials], fontsize=7)
  center_idx = int(np.argmin(np.abs(alphas - center_alpha)))
  ax.axvline(center_idx, color='black', linewidth=0.8, linestyle=':', alpha=0.6)
  ax.set_xlabel('alpha (meta-evaluation balance)')
  ax.set_ylabel('dial')
  ax.set_title(title, fontsize=10)


def _spa_opacity_legend(ax, spa_norm: Normalize):
  """Small standalone legend: black fading from transparent (lowest
  observed SPA) to solid (highest observed) over a white ground, since
  opacity isn't a matplotlib-native colorbar."""
  n = 256
  ramp = np.linspace(0, 1, n)
  rgba = np.zeros((1, n, 4))
  rgba[0, :, 3] = ramp
  ax.imshow(rgba, aspect='auto', extent=[0, 1, 0, 1])
  ax.set_yticks([])
  ax.set_xticks([0, 1])
  ax.set_xticklabels([f'{spa_norm.vmin:.3f}', f'{spa_norm.vmax:.3f}'], fontsize=7)
  ax.set_xlabel('weighted SPA (opacity, observed range)', fontsize=7)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--n-steps', type=int, default=5)
  parser.add_argument('--step', type=float, default=0.02)
  parser.add_argument('--dial-lo', type=float, default=0.0)
  parser.add_argument('--dial-hi', type=float, default=0.5)
  parser.add_argument('--dial-step', type=float, default=0.05)
  parser.add_argument('--show-ess', action=argparse.BooleanOptionalAction, default=False,
                       help='color cells by ESS (red-blue hue); off by default, all cells flat blue')
  args = parser.parse_args()
  base = args.dataset

  n_dial_steps = round((args.dial_hi - args.dial_lo) / args.dial_step)
  dial_grid = tuple(round(args.dial_lo + k * args.dial_step, 10) for k in range(n_dial_steps + 1))
  print(f'{base}: dial grid ({len(dial_grid)} pts): {list(dial_grid)}', file=sys.stderr)

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to plot')
  K = len(systems)

  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[systems, 'a'].values, df.loc[systems, 'b'].values
  center_alpha = alpha_0(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  alphas = alpha_grid_around(center_alpha, alpha_lo, alpha_hi, args.n_steps, args.step)
  print(f'{base}: K={K} systems, alpha_0(D)={center_alpha:.4f}, reachable=[{alpha_lo:.4f}, '
        f'{alpha_hi:.4f}], alpha grid ({len(alphas)} pts): {[round(a, 4) for a in alphas]}',
        file=sys.stderr)

  w_cache = solve_w_for_datasets([base], alphas, root=ROOT, systems_by_dataset={base: systems})
  w_by_alpha = {a: w_cache[(base, a)].w for a in alphas}
  ess_by_alpha = np.array([w_cache[(base, a)].ess for a in alphas])
  print(f'{base}: ESS(alpha) = {[round(e, 2) for e in ess_by_alpha]}', file=sys.stderr)

  candidates = sorted(load_metric_sys_scores(base, systems, root=ROOT))
  print(f'{base}: {len(candidates)} candidate donors', file=sys.stderr)

  out_dir = os.path.join(ARTIFACTS_DIR, base)
  os.makedirs(out_dir, exist_ok=True)

  # ESS depends only on alpha (not on donor or family), so its per-plot
  # range is the same number every time -- computed once here rather than
  # inside the loop.
  ess_norm = Normalize(vmin=float(ess_by_alpha.min()), vmax=float(ess_by_alpha.max()))

  n = len(candidates)
  t0 = time.time()
  n_written = 0
  for i, donor in enumerate(candidates, 1):
    grids = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=dial_grid)
    if grids is None:
      print(f'[{i}/{n}] SKIPPED {donor} (insufficient segment coverage)', file=sys.stderr)
      continue

    donor_spa = np.concatenate([grids['A'].ravel(), grids['B'].ravel()])
    donor_spa = donor_spa[~np.isnan(donor_spa)]
    spa_norm = Normalize(vmin=float(donor_spa.min()), vmax=float(donor_spa.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, label in zip(axes, ('A', 'B')):
      _draw_panel(ax, grids[label], grids['dials'], grids['alphas'], ess_by_alpha, ess_norm, spa_norm,
                  _PANEL_TITLES[label], center_alpha, args.show_ess)

    fig.suptitle(f'{base}: {donor} -- synthetic family x alpha (K={K}, alpha_0(D)={center_alpha:.4f})',
                 fontsize=11)

    right = 0.91 if args.show_ess else 0.98
    fig.tight_layout(rect=[0, 0.07, right, 0.95])

    if args.show_ess:
      cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
      sm = ScalarMappable(norm=ess_norm, cmap=ESS_CMAP)
      sm.set_array([])
      fig.colorbar(sm, cax=cbar_ax, label=f'ESS (observed [{ess_norm.vmin:.2f}, {ess_norm.vmax:.2f}] of [1, {K}])')

    legend_ax = fig.add_axes([0.40, 0.015, 0.2, 0.03])
    _spa_opacity_legend(legend_ax, spa_norm)

    safe_donor = donor.replace('/', '_')
    plot_path = os.path.join(out_dir, f'alpha_heatmap_{base}_{safe_donor}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    n_written += 1

    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    print(f'[{i}/{n}] wrote {plot_path} (SPA range [{spa_norm.vmin:.4f}, {spa_norm.vmax:.4f}]) '
          f'(elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  print(f'{base}: {n_written}/{n} donors plotted', file=sys.stderr)
