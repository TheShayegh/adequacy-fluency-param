"""Plots compute_pair_consistency_curve.py's output: one subplot per
meta-metric, each with a single pairwise-tau curve (colored by the pair's
own averaged, per-dataset-normalized ESS, same dark-blue/alpha scheme as
plot_per_dataset_consistency_curves.py) and one horizontal dashed line for
the natural (unreweighted) pairwise tau between the same two datasets.

Usage: python codes/scripts/plot_pair_consistency_curve.py DATASET1 DATASET2
(run compute_pair_consistency_curve.py first.)
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

from plot_consistency_curves import METAMETRICS_ORDER, TITLES, LIGHT_BLUE, interpolate_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')
DARK_BLUE = (0.03, 0.15, 0.35)


def load_curve(metametric: str, d1: str, d2: str) -> dict:
  path = os.path.join(DATA_DIR, f'curve_pair_{metametric}_{d1}_{d2}.json')
  with open(path) as f:
    return json.load(f)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('dataset1')
  p.add_argument('dataset2')
  p.add_argument('--title', type=str, default=None)
  args = p.parse_args()
  d1, d2 = args.dataset1, args.dataset2
  out_path = os.path.join(ROOT, 'artifacts', f'consistency_curve_pair_{d1}_{d2}.png')

  curves = {m: load_curve(m, d1, d2) for m in METAMETRICS_ORDER}

  # Same monochrome, alpha-carried spectrum as plot_per_dataset_consistency_
  # curves.py, but with both landmarks pulled down a third: fully
  # transparent at/below K/3, even ramp to K, with a light blue landmark at
  # the 2K/3 mark.
  cmap = LinearSegmentedColormap.from_list('ess_over_k', [
      (0.0, (*DARK_BLUE, 0.0)),
      (1 / 3, (*DARK_BLUE, 0.0)),
      (2 / 3, (*LIGHT_BLUE, 0.5)),
      (1.0, (*DARK_BLUE, 1.0)),
  ])
  norm = Normalize(vmin=0, vmax=1)

  fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw=dict(wspace=0.35, hspace=0.45, top=0.86, bottom=0.08))
  axes = axes.flatten()
  left_column = {0, 3}

  for i, (ax, m) in enumerate(zip(axes, METAMETRICS_ORDER)):
    c = curves[m]
    alphas = np.array(c['alphas'])
    tau = np.array(c['tau'])
    eok = np.array(c['ess_over_k'])
    x_lo, x_hi = alphas.min(), alphas.max()

    fine_a, fine_tau, fine_eok = interpolate_curve(alphas, tau, eok)
    points = np.array([fine_a, fine_tau]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_eok = (fine_eok[:-1] + fine_eok[1:]) / 2.0
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_eok)
    lc.set_linewidth(2.5)
    lc.set_joinstyle('round')
    lc.set_capstyle('round')
    ax.add_collection(lc)

    ax.axhline(c['natural_tau'], color='red', linestyle='--', linewidth=1.5, zorder=3)
    # Marks each dataset's own natural alpha_0 on the natural-tau line --
    # where the reweighted curve would touch it if only that one dataset,
    # not both, were held at its own natural balance.
    a0_1, a0_2 = c['alpha_0'][d1], c['alpha_0'][d2]
    ax.scatter([a0_1, a0_2], [c['natural_tau'], c['natural_tau']], marker='x',
                color='red', s=70, linewidths=2, zorder=6)

    ax.set_xlim(x_lo, x_hi)
    y_all = np.concatenate([tau, [c['natural_tau']]])
    pad = 0.05 * (y_all.max() - y_all.min() + 1e-9)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_xlabel('Controlled Balance\n(Adequacy over Fluency)')
    if i in left_column:
      ax.set_ylabel('Consistency')
    ax.set_title(TITLES[m])
    ax.set_box_aspect(1)

  for ax in axes[len(METAMETRICS_ORDER):]:
    ax.axis('off')

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.04, ticks=[0, 1 / 3, 2 / 3, 1])
  cbar.set_ticklabels(['0', 'K/3', '2K/3', 'K'])
  cbar.set_label('Reliability (ESS)')

  proxy_curve = Line2D([0], [0], color=DARK_BLUE, lw=2.5, label=f'Reweighted {d1} vs {d2}')
  proxy_natural = Line2D([0], [0], color='red', ls='--', lw=1.5, label='Natural (Unreweighted)')
  proxy_alpha0 = Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8,
                          markeredgewidth=2, label=r"Each dataset's own $\alpha_0$")
  slot = axes[len(METAMETRICS_ORDER)].get_position()
  slot_center = (slot.x0 + slot.width / 2, slot.y0 + slot.height / 2)
  fig.legend(handles=[proxy_curve, proxy_natural, proxy_alpha0], loc='center',
              bbox_to_anchor=slot_center, fontsize=11, frameon=False)

  if args.title:
    fig.suptitle(args.title, fontsize=14)
  else:
    fig.suptitle(
        f'{d1} vs {d2}: pairwise scorer-ranking consistency vs. reweighted balance '
        r'$\alpha$' + '\n(action_plan.md 6.1 x 6.5, single-pair variant, numeric solver)',
        fontsize=14,
    )
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')
