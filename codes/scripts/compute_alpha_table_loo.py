"""Leave-one-system-out alpha table: for a single dataset with K real
systems, computes the alpha_table_row (see lib/alpha_table.py) K times,
each time excluding one system -- a cheap systems-selection-sensitivity
check ("how much does dropping any one submission move alpha_0, the
reachable range, etc.?"), distinct from the reweighting machinery (which
keeps all K systems and changes their weights instead of dropping one).

Usage: python codes/scripts/compute_alpha_table_loo.py [dataset]
(defaults to ende23)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from mwb.mqm_scoring import load_system_scores, load_segment_scores
from lib.alpha_table import alpha_table_row
from lib.systems import is_reference_or_human

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


if __name__ == '__main__':
  dataset = sys.argv[1] if len(sys.argv) > 1 else 'ende23'

  sys_df = load_system_scores(dataset, root=ROOT)
  sys_df = sys_df[~sys_df.index.map(is_reference_or_human)]
  seg_df = load_segment_scores(dataset, root=ROOT)
  seg_df = seg_df[~seg_df['system'].map(is_reference_or_human)]

  systems = sorted(sys_df.index)
  K = len(systems)
  print(f'{dataset}: K={K} real systems: {systems}', file=sys.stderr)

  # Full-set row for reference (excluded_system = None).
  full_row = alpha_table_row(sys_df['a'].values, sys_df['b'].values, seg_df)
  rows = [{'excluded_system': '(none, full set)', **full_row}]

  for excluded in systems:
    sub_sys_df = sys_df.drop(index=excluded)
    sub_seg_df = seg_df[seg_df['system'] != excluded]
    row = alpha_table_row(sub_sys_df['a'].values, sub_sys_df['b'].values, sub_seg_df)
    rows.append({'excluded_system': excluded, **row})

  out = pd.DataFrame(rows).set_index('excluded_system')
  print(out.round(4).to_string())

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  md_path = os.path.join(ROOT, 'artifacts', f'alpha_table_loo_{dataset}.md')
  with open(md_path, 'w') as f:
    f.write(f'# Leave-one-system-out alpha table: {dataset}\n\n')
    f.write(
        f'{dataset} has K={K} real (non-reference) systems. Each row excludes '
        'one of them and recomputes the alpha_table_row (see compute_alpha_'
        'table.py / lib/alpha_table.py for column definitions) on the '
        'remaining K-1; the first row is the full K-system set for reference. '
        'This is a systems-selection sensitivity check, not the reweighting '
        'method (which keeps all K systems and moves their weights instead '
        'of dropping one).\n\n'
    )
    header = ['excluded_system', 'K', 'var_a', 'var_b', 'alpha_0', 'alpha_min', 'alpha_max',
              'alpha_synth_adequacy', 'alpha_synth_fluency', 'alpha_orig_synthAdeq_synthFlu',
              'n_segments']
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name, r in out.iterrows():
      f.write(
          f"| {name} | {int(r['K'])} | {r['var_a']:.4f} | {r['var_b']:.4f} | "
          f"{r['alpha_0']:.4f} | {r['alpha_min']:.4f} | {r['alpha_max']:.4f} | "
          f"{r['alpha_synth_adequacy']:.4f} | {r['alpha_synth_fluency']:.4f} | "
          f"{r['alpha_orig_synthAdeq_synthFlu']:.4f} | "
          f"{int(r['n_segments'])} |\n"
      )
  print(f'\nWrote {md_path}')
