"""Per-dataset outlier report: flags adequacy-score outliers via lib.
outliers.mad_outliers (median/MAD-based modified z-score, robust to
masking -- see that module's docstring), then, for every dataset with
O >= 1 flagged outliers, draws a histogram of alpha_0 over every C(K,K-O)
subset of size K-O of that dataset's real systems (exhaustive, via lib.
alpha_table.alpha_0_values_with_drop_mask), with one distinct, easily-
distinguishable color per SPECIFIC combination of outliers a subset happens
to drop (2^O of them, from "drops none" to "drops all O") -- not just how
MANY it drops, so e.g. "drop only system A" and "drop only system B" are
visually distinct even when they land at a similar alpha_0. Bars use a
unit-stacked style (unit_stacked_hist): each individual sample is its own
unit-height rectangle with a thin white border, so a bar's actual sample
count reads as a brick-wall texture rather than only its total height, and
two different combinations landing in the same bin both stay visible
instead of one overplotting the other. Datasets with O == 0 get no
histogram (there is nothing to drop).

Usage: python codes/scripts/compute_outlier_report.py [--threshold 3.5]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.alpha import alpha_0 as alpha0
from lib.alpha_table import alpha_0_values_with_drop_mask
from lib.consistency import real_systems
from lib.outliers import DEFAULT_THRESHOLD, mad_outliers
from mwb.mqm_scoring import SETS, load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')
DATASETS = list(SETS.keys())

FULL_K_BLACK = (0.0, 0.0, 0.0)  # the full-K alpha_0 reference line
BAR_ALPHA = 0.5  # histogram bars are translucent; the two dashed reference lines stay solid
N_BINS = 30  # coarser than a typical default so each bar's brick texture (unit_stacked_hist) reads clearly
# Qualitative, maximally-distinguishable palette for every OTHER group (not
# "drops all O", which is always tab:red, reserved) -- tab10 with tab:red
# (index 3) removed, so no accidental color clash with that reserved
# meaning. Plenty for every O seen in practice here (max O=3 -> 7 non-"all"
# combinations).
_TAB10 = plt.get_cmap('tab10').colors
ALL_DROPPED_RED = _TAB10[3]
_OTHER_COLORS = [c for i, c in enumerate(_TAB10) if i != 3]


def _drop_masks_by_popcount(O: int) -> list[int]:
  """Every mask in [0, 2**O), ordered by popcount (drops-none first,
  drops-all last) then by value -- a natural legend reading order."""
  return sorted(range(2 ** O), key=lambda m: (bin(m).count('1'), m))


def unit_stacked_hist(ax, groups: list[np.ndarray], bin_edges: np.ndarray,
                        colors: list, labels: list[str] | None = None, bar_alpha: float = 1.0) -> None:
  """Stacked histogram where each individual SAMPLE is its own unit-height
  rectangle with a thin white border, rather than one solid rectangle per
  (bin, group) -- makes the actual sample count inside a bar visible as a
  brick-wall texture instead of just its total height. groups/colors/
  labels: parallel lists, drawn in that stacking order (first = bottom).
  bin_edges: shared, evenly-spaced bin edges across every group."""
  n_bins = len(bin_edges) - 1
  bin_width = bin_edges[1] - bin_edges[0]
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  cum_bottom = np.zeros(n_bins)
  for i, vals in enumerate(groups):
    counts, _ = np.histogram(vals, bins=bin_edges)
    xs, bottoms = [], []
    for bi, c in enumerate(counts):
      if c == 0:
        continue
      xs.extend([bin_centers[bi]] * c)
      bottoms.extend(cum_bottom[bi] + np.arange(c))
    label = labels[i] if labels else None
    if xs:
      ax.bar(xs, height=1, bottom=bottoms, width=bin_width * 0.98, color=colors[i], alpha=bar_alpha,
             edgecolor='white', linewidth=0.4, label=label, align='center')
    elif label:
      # Still register an (empty) legend entry for a zero-count group.
      ax.bar([], [], color=colors[i], alpha=bar_alpha, label=label)
    cum_bottom += counts


def plot_dropoutlier_histogram(
    dataset: str, systems: list[str], a: np.ndarray, b: np.ndarray, outlier_names: list[str],
) -> str:
  """Histogram of alpha_0 over every size-(K-O) subset (O = len(outlier_
  names)) of `systems`, one color per exact combination of outliers a
  subset happens to drop (2^O of them -- a subset need not drop ALL O just
  because it's size K-O). Explains why O > 1 datasets show more than 2
  modes: each combination clusters at its own alpha_0, and combinations at
  the same drop COUNT can still land at different alpha_0 depending on
  which outlier(s) specifically. Stacked, so overlapping bins stay
  distinguishable. Returns the written PNG's path (relative to
  ARTIFACTS_DIR, for markdown embedding)."""
  K = len(a)
  O = len(outlier_names)
  k = K - O
  outlier_idx = [i for i, s in enumerate(systems) if s in outlier_names]
  sorted_outlier_names = [systems[i] for i in sorted(outlier_idx)]

  alphas, mask = alpha_0_values_with_drop_mask(a, b, k, outlier_idx)
  full_alpha0 = alpha0(a, b)
  drop_idx = [i for i in range(K) if i not in outlier_idx]
  drop_alpha0 = alpha0(a[drop_idx], b[drop_idx])

  masks = _drop_masks_by_popcount(O)

  def mask_label(m: int) -> str:
    dropped = [sorted_outlier_names[i] for i in range(O) if (m >> i) & 1]
    return 'drops none' if not dropped else f'drops {"+".join(dropped)}'

  all_dropped_mask = 2 ** O - 1
  groups = [alphas[mask == m] for m in masks]
  colors = []
  other_i = 0
  for m in masks:
    if m == all_dropped_mask:
      colors.append(ALL_DROPPED_RED)
    else:
      colors.append(_OTHER_COLORS[other_i % len(_OTHER_COLORS)])
      other_i += 1
  labels = [f'{mask_label(m)} (n={len(g):,})' for m, g in zip(masks, groups)]

  fig, ax = plt.subplots(figsize=(9, 5.5))
  bin_edges = np.linspace(alphas.min(), alphas.max(), N_BINS + 1)
  unit_stacked_hist(ax, groups, bin_edges, colors, labels, bar_alpha=BAR_ALPHA)

  ax.axvline(drop_alpha0, color=ALL_DROPPED_RED, linestyle='--', linewidth=1.5, zorder=5)
  ax.axvline(full_alpha0, color=FULL_K_BLACK, linestyle='--', linewidth=1.5, zorder=4,
             label=f'full K={K} alpha_0 = {full_alpha0:.4f}')

  ax.set_xlabel(r'alpha_0 (natural, uniform-weight balance) of the size-$k$ subset')
  ax.set_ylabel(f'count (of {len(alphas):,} size-{k} subsets)')
  ax.set_title(f'{dataset}: alpha_0 over all $C({K},{k})$ = {len(alphas):,} size-{k} subsets '
               f'(K-O, O={O} detected outlier{"s" if O != 1 else ""})')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(frameon=False, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  fname = f'alpha0_subset_histogram_{dataset}_dropoutliers.png'
  fig.savefig(os.path.join(ARTIFACTS_DIR, fname), dpi=150, bbox_inches='tight')
  plt.close(fig)
  return fname


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
  args = p.parse_args()

  rows = []
  for name in DATASETS:
    systems = real_systems(name, root=ROOT)
    df = load_system_scores(name, root=ROOT).loc[systems]
    a, b = df['a'].values, df['b'].values
    K = len(systems)

    flagged = mad_outliers(systems, a, threshold=args.threshold)
    O = len(flagged)
    print(f'{name:10} K={K:3d}  O={O}  ' +
          (', '.join(f'{s}(mz={z:.2f})' for s, z in flagged) if flagged else '(none)'),
          file=sys.stderr)

    png = None
    if O >= 1 and O < K - 1:
      outlier_names = [s for s, _ in flagged]
      png = plot_dropoutlier_histogram(name, systems, a, b, outlier_names)
      print(f'  -> {png}', file=sys.stderr)
    elif O >= K - 1:
      print(f'  (O={O} too close to K={K}, skipping histogram: subset size K-O < 2)', file=sys.stderr)

    rows.append({'dataset': name, 'K': K, 'O': O, 'flagged': flagged, 'png': png})

  md_path = os.path.join(ARTIFACTS_DIR, 'outlier_report.md')
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  with open(md_path, 'w') as f:
    f.write('# Per-dataset adequacy-score outlier report\n\n')
    f.write(
        f'Outliers are flagged via the modified z-score (Iglewicz & Hoya 1993): '
        f'`0.6745 * (a_i - median(a)) / MAD(a)`, threshold |z| > {args.threshold}. Median/MAD '
        'are robust to the outliers themselves (unlike a plain mean/std z-score or a leave-'
        'one-out variance-drop check), so this catches masked cases where several moderate '
        'outliers together dilute each other\'s individual variance contribution -- see '
        'lib.outliers.mad_outliers.\n\n'
        'For every dataset with O >= 1 flagged outliers, the histogram below shows alpha_0 '
        '(natural, uniform-weight balance) over every C(K,K-O) subset of size K-O of that '
        'dataset\'s real systems -- exhaustive, not sampled (lib.alpha_table.alpha_0_values_'
        'with_drop_mask) -- with one distinct color per exact COMBINATION of outliers a '
        'subset happens to drop (2^O of them, not just how many), which is why O > 1 datasets '
        'show more than 2 modes and why two combinations at the same drop count can still '
        'land at different alpha_0 (stacked where they overlap). The "drops all O outliers" '
        'group is always red (dashed red line marks that one specific subset); the full-K '
        'alpha_0 is always the dashed black line; every other combination gets its own color '
        'from the rest of the palette. Bars are drawn at alpha=0.5 (the two reference lines '
        'stay solid).\n\n'
    )
    f.write('| dataset | K | O | outliers (system: modified z-score) |\n')
    f.write('|---|---|---|---|\n')
    for r in rows:
      flagged_str = ', '.join(f'{s}: {z:+.2f}' for s, z in r['flagged']) if r['flagged'] else '(none)'
      f.write(f"| {r['dataset']} | {r['K']} | {r['O']} | {flagged_str} |\n")

    f.write('\n')
    for r in rows:
      if r['png'] is None:
        continue
      f.write(f"## {r['dataset']} (K={r['K']}, O={r['O']})\n\n")
      f.write(f"Outliers: {', '.join(f'{s} (mz={z:+.2f})' for s, z in r['flagged'])}\n\n")
      f.write(f"![{r['dataset']} alpha_0 histogram]({r['png']})\n\n")
  print(f'\nWrote {md_path}', file=sys.stderr)
