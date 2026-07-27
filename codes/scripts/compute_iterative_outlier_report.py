"""Runs lib.outlier_detection.iterative_single_aspect_outliers over every 2023 and
2024 dataset (ende23, zhen23, heen23, ende24, enes24, jazh24 --
lib.dataset_dirs.datasets_for_years([2023, 2024])), reporting the full
drop sequence per dataset: which systems get flagged, at which iteration,
and with what z-score, until a pass finds nothing left to flag.

SIMPLIFIED, NON-STANDARD: this is the single-aspect (one RV at a time)
screen -- see artifacts/inventory_outliers/README.md for the project
STANDARD (joint (a,b), lib.outlier_detection.iterative_gk_outliers).

This is the masking check for the single-pass MAD detector used
elsewhere in this project (lib.outliers.mad_outliers) -- e.g. IKUN-C in
jazh24 only crosses the |z|>3.5 threshold once MSLC is already removed;
a single pass over the full pool misses it.

--aspect {adequacy,fluency,all} (default adequacy) selects which single RV
the modified z-score is computed on: adequacy scores (a), fluency scores
(b), or all_mqm (t = a+b -- still a SINGLE RV, not the joint (a,b) pair;
see plot_af_scatter_allmqm_test.py for why its outlier boundary is a line
of constant a+b, not a region). Non-default aspects write to a
aspect-tagged filename so they never clobber the adequacy default.

Usage: python codes/scripts/compute_iterative_outlier_report.py [--aspect adequacy|fluency|all]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.consistency import real_systems
from lib.dataset_dirs import datasets_for_years
from lib.outlier_detection import DEFAULT_THRESHOLD, iterative_single_aspect_outliers
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_outliers')

ASPECT_COLUMN = {'adequacy': 'a', 'fluency': 'b', 'all': 't'}
ASPECT_LABEL = {'adequacy': 'adequacy scores (a)', 'fluency': 'fluency scores (b)', 'all': 'all_mqm (t = a+b)'}


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--aspect', choices=list(ASPECT_COLUMN), default='adequacy',
                  help='which single RV to modified-z-score: adequacy, fluency, or all (all_mqm = a+b)')
  args = p.parse_args()
  col = ASPECT_COLUMN[args.aspect]

  datasets = datasets_for_years([2023, 2024])
  print(f'datasets: {datasets}', file=sys.stderr)

  lines = []
  lines.append(f'# Iterative MAD outlier detection: 2023 + 2024 datasets ({args.aspect})')
  lines.append('')
  lines.append(f'threshold = {DEFAULT_THRESHOLD}. Each dataset\'s systems are repeatedly z-scored on '
               f'{ASPECT_LABEL[args.aspect]} (median/MAD-based modified z-score); every system over '
               'threshold in a pass is dropped, then the pass repeats on the shrunk pool, until a pass '
               'flags nothing.')
  lines.append('')

  summary_rows = []
  for base in datasets:
    systems = real_systems(base, root=ROOT)
    df = load_system_scores(base, root=ROOT)
    x = df.loc[systems, col].values
    removed = iterative_single_aspect_outliers(systems, x)

    print(f'{base} (K={len(systems)}): {removed}', file=sys.stderr)
    lines.append(f'## {base} (K={len(systems)})')
    lines.append('')
    if not removed:
      lines.append('No outliers detected (single pass, iteration 1, already clean).')
    else:
      lines.append('| Iteration | System | z-score |')
      lines.append('|---:|---|---:|')
      for s, z, it in removed:
        lines.append(f'| {it} | {s} | {z:+.3f} |')
    lines.append('')
    summary_rows.append((base, len(systems), [s for s, _, _ in removed]))

  lines.append('## Summary')
  lines.append('')
  lines.append('| Dataset | K | Dropped (in removal order) | n dropped |')
  lines.append('|---|---:|---|---:|')
  for base, K, dropped in summary_rows:
    lines.append(f'| {base} | {K} | {", ".join(dropped) if dropped else "(none)"} | {len(dropped)} |')
  table_md = '\n'.join(lines)

  print('\n' + table_md)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  suffix = '' if args.aspect == 'adequacy' else f'_{args.aspect}'
  out_path = os.path.join(ARTIFACTS_DIR, f'iterative_outlier_report_2023_2024{suffix}.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)