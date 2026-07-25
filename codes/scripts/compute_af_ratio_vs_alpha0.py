"""Cross-dataset check, missing from action_plan.md's evaluation plan
(section 6): does a dataset's natural balance alpha_0 (lib.alpha.alpha_0,
section 3.2/3.10 -- the extent to which the real system pool is
adequacy-heavy) predict the ratio

    metametric(adequacy_mqm; dataset) / metametric(fluency_mqm; dataset)

on that same dataset, where metametric ranges over the five of section 1.1/4
(Pearson, Spearman, Kendall's tau, PA, SPA) and "adequacy_mqm"/"fluency_mqm"
are the raw a/b component vectors themselves, scored as if they were
automatic-metric "scorers" being meta-evaluated against the All-MQM (t=a+b)
human ranking (lib.scorer_metametrics -- the same MetaEvalInput machinery
lib.consistency uses for real external metrics). Each scatter point is one
dataset.

This is a cross-dataset analogue of the paper's within-dataset removal-
robustness idea (section 6.5): instead of asking whether the balance itself
is robust to dropping systems, it asks whether the alpha_0-vs-ratio
RELATIONSHIP is robust to dropping datasets. Run at three population sizes,
all exhaustive (not sampled, matching every other subset enumeration in this
project -- lib.alpha_table, section 6.5's L-L design):

  - full K: all 15 well-formed sets (mwb.mqm_scoring.SETS).
  - every size-(K-1) subset (15 of them: each drops exactly one dataset).
  - every size-(K-2) subset (C(15,2)=105 of them: each drops exactly two).

Per-dataset (alpha_0, ratio) values are computed ONCE per dataset (the
expensive step, since SPA needs a segment-level permutation test per
scorer); every K-1/K-2 subset then just re-runs a correlation over a subset
of those 15 already-computed points, which is why this stays cheap even at
1+15+105 = 121 correlations per meta-metric.

The correlation used for the ratio-vs-alpha_0 relationship itself (--corr)
is configurable and distinct from the five Pearson/Spearman/Kendall/PA/SPA
meta-metrics that define the ratio's numerator/denominator in the first
place: 'spearman' (default) only assumes a monotone relationship, not a
linear one, and is robust to the ratio's occasional large values (e.g.
zhen23); 'pearson' assumes linearity, matching the OLS line drawn on the
full-K scatter plot.

Usage: python codes/scripts/compute_af_ratio_vs_alpha0.py [--corr pearson|spearman]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems
from lib.metametrics import METAMETRICS_ORDER
from lib.scorer_metametrics import af_ratio
from mwb.mqm_scoring import SETS, load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATASETS = list(SETS.keys())  # all 15 well-formed sets (generalMT2022/enzh excluded, see mqm_scoring.py)
METAMETRIC_ORDER = METAMETRICS_ORDER

METAMETRIC_LABEL = {'pearson': 'Pearson', 'spearman': 'Spearman', 'kendall': "Kendall's tau", 'pa': 'PA', 'spa': 'SPA'}

STANDARD_BLUE = (0.121, 0.466, 0.705)   # matplotlib tab:blue, matches plot_alpha0_subset_histogram.py
STANDARD_ORANGE = (1.0, 0.498, 0.055)   # matplotlib tab:orange -- CVD-distinguishable from blue


def compute_per_dataset_table() -> pd.DataFrame:
  """One row per dataset: alpha_0 and, for every meta-metric, the
  adequacy/fluency ratio (af_ratio) -- computed once, reused by every
  subsequent K/K-1/K-2 correlation."""
  rows = []
  t0 = time.time()
  for i, name in enumerate(DATASETS, 1):
    systems = real_systems(name, root=ROOT)
    sys_df = load_system_scores(name, root=ROOT).loc[systems]
    a0 = alpha0_of(sys_df['a'].values, sys_df['b'].values)
    row = {'dataset': name, 'K': len(systems), 'alpha_0': a0}
    for m in METAMETRIC_ORDER:
      row[f'ratio_{m}'] = af_ratio(name, m, root=ROOT, systems=systems)
    rows.append(row)
    elapsed = time.time() - t0
    eta = elapsed / i * (len(DATASETS) - i)
    print(f'[{i}/{len(DATASETS)}] {name}: alpha_0={a0:.4f}  '
          f'(elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
  return pd.DataFrame(rows).set_index('dataset')


def correlate(table: pd.DataFrame, metametric: str, dataset_subset: list[str] | None = None):
  """Pearson r, p, n for one meta-metric's ratio vs. alpha_0, over
  `dataset_subset` (default: every dataset in `table`), dropping any
  dataset where the ratio is NaN (e.g. too few SPA segments)."""
  sub = table if dataset_subset is None else table.loc[list(dataset_subset)]
  col = f'ratio_{metametric}'
  valid = sub[col].notna()
  x = sub.loc[valid, 'alpha_0'].values
  y = sub.loc[valid, col].values
  if len(x) < 3:
    return float('nan'), float('nan'), len(x)
  r, p = stats.pearsonr(x, y)
  return float(r), float(p), len(x)


def leave_k_out_correlations(table: pd.DataFrame, metametric: str, drop: int) -> list[float]:
  """r for every exhaustive size-(K-drop) subset of table's datasets (drop
  `drop` datasets at a time) -- the K-1 (drop=1) / K-2 (drop=2) robustness
  check."""
  datasets = list(table.index)
  rs = []
  for dropped in itertools.combinations(datasets, drop):
    subset = [d for d in datasets if d not in dropped]
    r, _, n = correlate(table, metametric, subset)
    if n >= 3:
      rs.append(r)
  return rs


def plot_full_scatter(table: pd.DataFrame, out_path: str):
  fig, axes = plt.subplots(2, 3, figsize=(15, 9))
  axes = axes.flatten()
  for ax, m in zip(axes, METAMETRIC_ORDER):
    col = f'ratio_{m}'
    sub = table[table[col].notna()]
    x, y = sub['alpha_0'].values, sub[col].values
    r, p, n = correlate(table, m)
    ax.scatter(x, y, s=28, color=STANDARD_BLUE, zorder=3)
    for name, xi, yi in zip(sub.index, x, y):
      ax.annotate(name, (xi, yi), fontsize=6, color='#555555',
                  xytext=(3, 3), textcoords='offset points')
    if n >= 2:
      xs = np.linspace(x.min(), x.max(), 100)
      slope, intercept = np.polyfit(x, y, 1)
      ax.plot(xs, slope * xs + intercept, color='red', linewidth=1.5, zorder=2)
    ax.set_title(f'{METAMETRIC_LABEL[m]}: r={r:+.3f}, p={p:.3f}, n={n}', fontsize=11)
    ax.set_xlabel(r'$\alpha_0$(dataset)')
    ax.set_ylabel('metametric(adequacy_mqm) / metametric(fluency_mqm)')
  axes[-1].axis('off')
  fig.suptitle('Adequacy/fluency meta-metric ratio vs. natural balance '
               r'$\alpha_0$, one point per dataset ($K$=' + str(len(table)) + ' datasets)',
               fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.94])
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {out_path}', file=sys.stderr)


def plot_robustness(table: pd.DataFrame, loo: dict, l2o: dict, out_path: str):
  fig, axes = plt.subplots(2, 3, figsize=(15, 9))
  axes = axes.flatten()
  for ax, m in zip(axes, METAMETRIC_ORDER):
    full_r, _, _ = correlate(table, m)
    data = [loo[m], l2o[m]]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                     medianprops=dict(color='black'))
    for patch, color in zip(bp['boxes'], (STANDARD_BLUE, STANDARD_ORANGE)):
      patch.set_facecolor((*color, 0.5))
      patch.set_edgecolor(color)
    ax.axhline(full_r, color='black', linestyle='--', linewidth=1.2, zorder=0)
    ax.text(1.35, full_r, f'full-K r={full_r:+.3f}', fontsize=8, va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'K-1 subsets\n(n={len(loo[m])})', f'K-2 subsets\n(n={len(l2o[m])})'])
    ax.set_ylabel('Pearson r (ratio vs. ' r'$\alpha_0$)')
    ax.set_title(METAMETRIC_LABEL[m], fontsize=11)
    ax.set_xlim(-0.6, 1.9)
  axes[-1].axis('off')
  fig.suptitle('Robustness of the ratio-vs-$\\alpha_0$ correlation to dropping datasets:\n'
               'distribution of r across every exhaustive leave-one-out / leave-two-out dataset subset',
               fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.92])
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {out_path}', file=sys.stderr)


def summarize_subset_rs(rs: list[float], full_r: float) -> dict:
  arr = np.array(rs, dtype=float)
  same_sign = np.sign(arr) == np.sign(full_r) if full_r == full_r else np.full(arr.shape, False)
  return {
      'n': len(arr), 'mean_r': float(arr.mean()), 'std_r': float(arr.std()),
      'min_r': float(arr.min()), 'max_r': float(arr.max()),
      'frac_same_sign_as_full': float(same_sign.mean()) if len(arr) else float('nan'),
  }


if __name__ == '__main__':
  print(f'Computing per-dataset alpha_0 and adequacy/fluency meta-metric ratios '
        f'for {len(DATASETS)} datasets...', file=sys.stderr)
  table = compute_per_dataset_table()
  print('\nPer-dataset table:', file=sys.stderr)
  print(table.round(4).to_string(), file=sys.stderr)

  full_results = {}
  loo_rs = {}
  l2o_rs = {}
  n_loo_subsets = comb(len(DATASETS), 1)
  n_l2o_subsets = comb(len(DATASETS), 2)
  for m in METAMETRIC_ORDER:
    r, p, n = correlate(table, m)
    full_results[m] = (r, p, n)
    print(f'\n{METAMETRIC_LABEL[m]}: full-K r={r:+.4f}, p={p:.4g}, n={n}', file=sys.stderr)

    t0 = time.time()
    loo_rs[m] = leave_k_out_correlations(table, m, drop=1)
    l2o_rs[m] = leave_k_out_correlations(table, m, drop=2)
    print(f'  K-1 subsets: {len(loo_rs[m])}/{n_loo_subsets} usable, '
          f'K-2 subsets: {len(l2o_rs[m])}/{n_l2o_subsets} usable '
          f'({time.time() - t0:.2f}s)', file=sys.stderr)

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  scatter_path = os.path.join(ROOT, 'artifacts', 'af_ratio_vs_alpha0_full.png')
  robustness_path = os.path.join(ROOT, 'artifacts', 'af_ratio_vs_alpha0_robustness.png')
  plot_full_scatter(table, scatter_path)
  plot_robustness(table, loo_rs, l2o_rs, robustness_path)

  md_path = os.path.join(ROOT, 'artifacts', 'af_ratio_vs_alpha0.md')
  with open(md_path, 'w') as f:
    f.write('# Adequacy/fluency meta-metric ratio vs. natural balance alpha_0\n\n')
    f.write(
        'For each dataset and each of the five meta-metrics (Pearson, Spearman, '
        "Kendall's tau, PA, SPA), scores the raw Adequacy MQM vector and the raw "
        'Fluency MQM vector each as if it were an automatic-metric "scorer" being '
        'meta-evaluated against the All-MQM (t=a+b) human ranking '
        '(lib.scorer_metametrics), then takes the ratio '
        'metametric(adequacy_mqm)/metametric(fluency_mqm). Correlates that ratio '
        'against the dataset\'s natural balance alpha_0 (lib.alpha.alpha_0, '
        'action_plan.md 3.2/3.10) across datasets -- one scatter point per dataset. '
        'Run at K=15 (all well-formed sets), and exhaustively at every leave-one-out '
        '(K-1) and leave-two-out (K-2) dataset subset, to check whether the '
        'correlation survives dropping datasets.\n\n'
    )

    f.write('## Per-dataset values\n\n')
    header = ['dataset', 'K', 'alpha_0'] + [f'ratio_{m}' for m in METAMETRIC_ORDER]
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name, r in table.iterrows():
      cells = [name, str(int(r['K'])), f"{r['alpha_0']:.4f}"]
      cells += [f"{r[f'ratio_{m}']:.4f}" if r[f'ratio_{m}'] == r[f'ratio_{m}'] else 'NaN' for m in METAMETRIC_ORDER]
      f.write('| ' + ' | '.join(cells) + ' |\n')

    f.write('\n## Full-K correlation (n=15 datasets)\n\n')
    f.write('| metametric | r | p | n |\n|---|---|---|---|\n')
    for m in METAMETRIC_ORDER:
      r, p, n = full_results[m]
      f.write(f'| {METAMETRIC_LABEL[m]} | {r:+.4f} | {p:.4g} | {n} |\n')

    for label, rs_dict, n_subsets in (('K-1 (leave-one-dataset-out)', loo_rs, n_loo_subsets),
                                       ('K-2 (leave-two-dataset-out)', l2o_rs, n_l2o_subsets)):
      f.write(f'\n## {label} robustness, exhaustive over {n_subsets} subsets\n\n')
      f.write('| metametric | n_usable | mean r | std r | min r | max r | frac same sign as full-K |\n')
      f.write('|---|---|---|---|---|---|---|\n')
      for m in METAMETRIC_ORDER:
        full_r = full_results[m][0]
        s = summarize_subset_rs(rs_dict[m], full_r)
        f.write(f"| {METAMETRIC_LABEL[m]} | {s['n']} | {s['mean_r']:+.4f} | {s['std_r']:.4f} | "
                f"{s['min_r']:+.4f} | {s['max_r']:+.4f} | {s['frac_same_sign_as_full']:.3f} |\n")

    f.write(f'\n## Figures\n\n')
    f.write(f'- Full-K scatter + fitted regressor: `{os.path.basename(scatter_path)}`\n')
    f.write(f'- K-1/K-2 robustness (distribution of r): `{os.path.basename(robustness_path)}`\n')

  print(f'\nWrote {md_path}', file=sys.stderr)
