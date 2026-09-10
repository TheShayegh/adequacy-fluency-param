"""Single-panel plot of SPA(scorer; alpha) vs. target alpha from one
compute_spa_vs_alpha.py cache -- see that script's docstring for how the
data was produced. Purely a plotting script (no solving), so it's cheap to
rerun while iterating on styling.

Usage: python codes/scripts/plot_spa_vs_alpha.py --cache artifacts/data/spa_vs_alpha_ende24_n10.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.spa_alpha_sweep import load_spa_vs_alpha, scorer_color_map, scorer_marker_map

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

# --- Styling knobs -- edit these, then just rerun (no recompute needed) ---
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8

MARKER_SIZE = 4
LINE_WIDTH = 2.0

# {dataset}/{substrings}/{target_ess}/{K}/{n_points} are filled in from the
# cache's own metadata.
TITLE_TEMPLATE = ('{dataset}: SPA vs. target $\\alpha$ for scorers matching {substrings}\n'
                   'sweep = ESS(w*($\\alpha$)) >= {target_ess:.4g} (K={K}), {n_points} points')
XLABEL = r'Target Balance $\alpha$'
YLABEL = 'SPA(scorer; ' + r'$\alpha$' + ')'
ALPHA0_ANNOTATION_TEXT = r'$\alpha_0(D)$'


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--cache', type=str, required=True, help='path to a compute_spa_vs_alpha.py .npz cache')
  p.add_argument('--out', type=str, default=None, help='output path (default: derived from the cache filename)')
  args = p.parse_args()

  d = load_spa_vs_alpha(args.cache)
  color_by_scorer = scorer_color_map(d['scorers'])
  marker_by_scorer = scorer_marker_map(d['scorers'])

  fig, ax = plt.subplots(figsize=(8, 6))
  for i, scorer in enumerate(d['scorers']):
    ax.plot(d['alphas'], d['spa'][i], color=color_by_scorer[scorer], linewidth=LINE_WIDTH,
            marker=marker_by_scorer[scorer], markersize=MARKER_SIZE, label=scorer)
  y_max = float(d['spa'].max())
  ax.axvline(d['a0_D'], color='black', alpha=0.3, linewidth=1.0, zorder=1)
  ax.annotate(ALPHA0_ANNOTATION_TEXT, xy=(d['a0_D'], y_max), xytext=(0, 8),
              textcoords='offset points', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE, color='black')
  ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
  ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
  ax.set_title(TITLE_TEMPLATE.format(dataset=d['dataset'], substrings=d['substrings'],
                                      target_ess=d['target_ess'], K=d['K'], n_points=len(d['alphas'])),
               fontsize=TITLE_FONTSIZE)
  ax.tick_params(labelsize=TICK_FONTSIZE)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(loc='best', frameon=False, fontsize=LEGEND_FONTSIZE)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  base_name = os.path.splitext(os.path.basename(args.cache))[0]
  out_path = args.out or os.path.join(ARTIFACTS_DIR, f'{base_name}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}', file=sys.stderr)
