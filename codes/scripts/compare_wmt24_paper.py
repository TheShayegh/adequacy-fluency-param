"""Compare our reconstructed system-level MQM scores against Table 5 of the
WMT24 Metrics Shared Task findings paper (Freitag et al. 2024,
"Are LLMs Breaking MT Metrics? Results of the WMT24 Metrics Shared Task",
aclanthology.org/2024.wmt-1.2, Table 5, "all" column -- macro-average over
domains, lower = better), alongside the official mt-metrics-eval score files
(both should trace to the same underlying pipeline).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from mwb.mqm_scoring import system_level_scores, parse_mqm_tsv, SETS, _normalize_system_name
from lib.dataset_dirs import DATASET_DIRS

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

# Table 5, "all" column (positive = error magnitude, lower is better)
PAPER_TABLE5 = {
    'ende24': {
        'Dubformer': 1.58, 'GPT-4': 1.58, 'Unbabel-Tower70B': 1.65, 'ONLINE-B': 1.81,
        'TranssionMT': 1.81, 'refB': 1.84, 'Mistral-Large': 1.93, 'CommandR-plus': 2.01,
        'refA': 2.12, 'Gemini-1.5-Pro': 2.20, 'ONLINE-W': 2.22, 'Claude-3': 2.28,
        'IOL_Research': 2.39, 'Aya23': 3.09, 'ONLINE-A': 3.30, 'Llama3-70B': 3.62,
        'IKUN': 3.86, 'IKUN-C': 5.07, 'MSLC': 13.46,
    },
    'enes24': {
        'GPT-4': 0.12, 'Unbabel-Tower70B': 0.20, 'Claude-3': 0.26, 'Mistral-Large': 0.26,
        'Gemini-1.5-Pro': 0.39, 'Dubformer': 0.43, 'Llama3-70B': 0.52, 'refA': 0.55,
        'IOL_Research': 0.57, 'CommandR-plus': 0.62, 'ONLINE-W': 0.64, 'IKUN': 0.94,
        'ONLINE-B': 1.08, 'Aya23': 1.52, 'MSLC': 6.80,
    },
    'jazh24': {
        'Claude-3': 1.22, 'refA': 1.32, 'GPT-4': 1.45, 'DLUT_GTCOM': 1.52,
        'Unbabel-Tower70B': 1.69, 'Gemini-1.5-Pro': 1.78, 'CommandR-plus': 1.91,
        'IOL_Research': 2.10, 'Aya23': 3.03, 'Llama3-70B': 3.07, 'Team-J': 3.91,
        'NTTSU': 4.34, 'ONLINE-B': 5.27, 'IKUN-C': 6.60, 'MSLC': 9.19,
    },
}

# Derived from DATASET_DIRS (the shared (year_dir, pair) source of truth)
# rather than hand-listed; only the 3 datasets in PAPER_TABLE5 are ever used.
OFFICIAL_PATHS = {
    name: f'external/mt-metrics-eval-data/mt-metrics-eval-v2/{year_dir}/human-scores/{pair}.mqm.sys.score'
    for name, (year_dir, pair) in DATASET_DIRS.items()
}


def load_official(name):
  """Raw official score file names aren't normalized (e.g. "Claude-3.5" where
  every other name in this project canonically resolves to "Claude-3", see
  mqm_scoring._normalize_system_name) -- apply the same normalization here so
  a lookup by this project's canonical system name (as PAPER_TABLE5/ab_df both
  use) finds the right row instead of silently missing it."""
  vals = {}
  with open(os.path.join(ROOT, OFFICIAL_PATHS[name])) as f:
    for line in f:
      s, v = line.strip().split('\t')
      vals[_normalize_system_name(s)] = float(v) if v != 'None' else float('nan')
  return vals


if __name__ == '__main__':
  for name in PAPER_TABLE5:
    ab_df = system_level_scores(parse_mqm_tsv(os.path.join(ROOT, SETS[name])))
    # 'full' (all-category score, not gated by adequacy/fluency classification)
    # is already a column of ab_df -- system_level_scores/parse_mqm_tsv compute
    # it via the same normalized-system-name, error_weight-accumulation path as
    # everything else in this project, so a separate hand-rolled recomputation
    # here would only be able to duplicate it, not independently verify it.
    full = ab_df['full']
    official = load_official(name)
    paper = PAPER_TABLE5[name]

    rows = []
    for sysname in paper:
      off_val = official.get(sysname, float('nan'))
      rows.append({
          'system': sysname,
          'paper_table5': paper[sysname],
          'official_sys.score': -off_val if off_val == off_val else float('nan'),
          'ours_full': -full.get(sysname, float('nan')),
          'ours_a+b_only': -ab_df['t'].get(sysname, float('nan')),
      })
    out = pd.DataFrame(rows).set_index('system').sort_values('paper_table5')
    print(f"=== {name} ===")
    print(out.round(3).to_string())
    print()
