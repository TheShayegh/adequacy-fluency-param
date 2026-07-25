"""Runs lib.outlier_detection.iterative_mad_outliers over every 2023 and
2024 dataset (ende23, zhen23, heen23, ende24, enes24, jazh24 --
lib.dataset_dirs.datasets_for_years([2023, 2024])), reporting the full
drop sequence per dataset: which systems get flagged, at which iteration,
and with what z-score, until a pass finds nothing left to flag.

This is the masking check for the single-pass MAD detector used
elsewhere in this project (lib.outliers.mad_outliers) -- e.g. IKUN-C in
jazh24 only crosses the |z|>3.5 threshold once MSLC is already removed;
a single pass over the full pool misses it.

Usage: python codes/scripts/compute_iterative_outlier_report.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.consistency import real_systems
from lib.dataset_dirs import datasets_for_years
from lib.outlier_detection import DEFAULT_THRESHOLD, iterative_mad_outliers
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


if __name__ == '__main__':
  datasets = datasets_for_years([2023, 2024])
  print(f'datasets: {datasets}', file=sys.stderr)

  lines = []
  lines.append('# Iterative MAD outlier detection: 2023 + 2024 datasets')
  lines.append('')
  lines.append(f'threshold = {DEFAULT_THRESHOLD}. Each dataset\'s systems are repeatedly z-scored '
               '(median/MAD-based modified z-score); every system over threshold in a pass is '
               'dropped, then the pass repeats on the shrunk pool, until a pass flags nothing.')
  lines.append('')

  summary_rows = []
  for base in datasets:
    systems = real_systems(base, root=ROOT)
    df = load_system_scores(base, root=ROOT)
    a = df.loc[systems, 'a'].values
    removed = iterative_mad_outliers(systems, a)

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
  out_path = os.path.join(ARTIFACTS_DIR, 'iterative_outlier_report_2023_2024.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)
