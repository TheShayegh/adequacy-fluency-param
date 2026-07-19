"""Bootstrap alpha table: for a dataset with K real systems, repeatedly
samples n_sample systems *without replacement* and recomputes
alpha_table_row (lib/alpha_table.py) on each sample -- a system-selection
sensitivity check via random subsampling, complementing the leave-one-out
version (compute_alpha_table_loo.py, which drops exactly one system at a
time instead of resampling a fixed-size subset).

Usage: python codes/scripts/compute_alpha_table_bootstrap.py [dataset ...]
(defaults to all 2023/2024 datasets: ende23 zhen23 heen23 ende24 enes24 jazh24)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from mwb.mqm_scoring import load_system_scores, load_segment_scores
from lib.alpha_table import alpha_table_row
from lib.bootstrap import sample_subsets
from lib.systems import is_reference_or_human

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

DEFAULT_DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']
N_SAMPLE = 10
N_REPEATS = 30
SEED = 42


def run(dataset: str) -> tuple[pd.DataFrame, int]:
  sys_df = load_system_scores(dataset, root=ROOT)
  sys_df = sys_df[~sys_df.index.map(is_reference_or_human)]
  seg_df = load_segment_scores(dataset, root=ROOT)
  seg_df = seg_df[~seg_df['system'].map(is_reference_or_human)]

  systems = sorted(sys_df.index)
  K = len(systems)
  if K < N_SAMPLE:
    raise ValueError(f'{dataset}: only {K} real systems, need >= {N_SAMPLE}')
  print(f'{dataset}: K={K} real systems, sampling {N_SAMPLE} without replacement x {N_REPEATS}',
        file=sys.stderr)

  samples = sample_subsets(systems, N_SAMPLE, N_REPEATS, SEED)
  rows = []
  full_row = alpha_table_row(sys_df['a'].values, sys_df['b'].values, seg_df)
  rows.append({'repeat': '(full set)', 'sampled_systems': '; '.join(systems), **full_row})

  for rep, sample in enumerate(samples, start=1):
    sub_sys_df = sys_df.loc[sample]
    sub_seg_df = seg_df[seg_df['system'].isin(sample)]
    row = alpha_table_row(sub_sys_df['a'].values, sub_sys_df['b'].values, sub_seg_df)
    rows.append({'repeat': rep, 'sampled_systems': '; '.join(sample), **row})

  return pd.DataFrame(rows), K


if __name__ == '__main__':
  datasets = sys.argv[1:] or DEFAULT_DATASETS
  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)

  for dataset in datasets:
    out, K = run(dataset)
    print(out.drop(columns='sampled_systems').round(4).to_string(index=False))

    md_path = os.path.join(ROOT, 'artifacts', f'alpha_table_bootstrap_{dataset}.md')
    with open(md_path, 'w') as f:
      f.write(f'# Bootstrap alpha table: {dataset}\n\n')
      f.write(
          f'{dataset} has K={K} real (non-reference) systems. Each of the {N_REPEATS} '
          f'numbered rows samples {N_SAMPLE} systems without replacement (seed={SEED}) and '
          'recomputes the alpha_table_row (see compute_alpha_table.py / lib/alpha_table.py '
          'for column definitions); the first row is the full K-system set for reference. '
          'This is a random-subsampling sensitivity check, complementing the leave-one-out '
          'version (compute_alpha_table_loo.py, which drops exactly one system instead of '
          'resampling a fixed-size subset).\n\n'
      )
      header = ['repeat', 'K', 'var_a', 'var_b', 'alpha_0', 'alpha_min', 'alpha_max',
                'alpha_synth_adequacy', 'alpha_synth_fluency', 'alpha_orig_synthAdeq_synthFlu',
                'n_segments', 'sampled_systems']
      f.write('| ' + ' | '.join(header) + ' |\n')
      f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
      for _, r in out.iterrows():
        f.write(
            f"| {r['repeat']} | {int(r['K'])} | {r['var_a']:.4f} | {r['var_b']:.4f} | "
            f"{r['alpha_0']:.4f} | {r['alpha_min']:.4f} | {r['alpha_max']:.4f} | "
            f"{r['alpha_synth_adequacy']:.4f} | {r['alpha_synth_fluency']:.4f} | "
            f"{r['alpha_orig_synthAdeq_synthFlu']:.4f} | {int(r['n_segments'])} | "
            f"{r['sampled_systems']} |\n"
        )
    print(f'Wrote {md_path}\n')
