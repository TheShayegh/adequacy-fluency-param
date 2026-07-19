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

import csv
from collections import defaultdict

import pandas as pd
from mwb.mqm_scoring import (
    error_weight, filter_modal_coverage, system_level_scores, parse_mqm_tsv, SETS,
    _doc_id_col, _seg_id_col,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

# Table 5, "all" column (positive = error magnitude, lower is better)
PAPER_TABLE5 = {
    'ende24': {
        'Dubformer': 1.58, 'GPT-4': 1.58, 'Unbabel-Tower70B': 1.65, 'ONLINE-B': 1.81,
        'TranssionMT': 1.81, 'refB': 1.84, 'Mistral-Large': 1.93, 'CommandR-plus': 2.01,
        'refA': 2.12, 'Gemini-1.5-Pro': 2.20, 'ONLINE-W': 2.22, 'Claude-3.5': 2.28,
        'IOL_Research': 2.39, 'Aya23': 3.09, 'ONLINE-A': 3.30, 'Llama3-70B': 3.62,
        'IKUN': 3.86, 'IKUN-C': 5.07, 'MSLC': 13.46,
    },
    'enes24': {
        'GPT-4': 0.12, 'Unbabel-Tower70B': 0.20, 'Claude-3.5': 0.26, 'Mistral-Large': 0.26,
        'Gemini-1.5-Pro': 0.39, 'Dubformer': 0.43, 'Llama3-70B': 0.52, 'refA': 0.55,
        'IOL_Research': 0.57, 'CommandR-plus': 0.62, 'ONLINE-W': 0.64, 'IKUN': 0.94,
        'ONLINE-B': 1.08, 'Aya23': 1.52, 'MSLC': 6.80,
    },
    'jazh24': {
        'Claude-3.5': 1.22, 'refA': 1.32, 'GPT-4': 1.45, 'DLUT_GTCOM': 1.52,
        'Unbabel-Tower70B': 1.69, 'Gemini-1.5-Pro': 1.78, 'CommandR-plus': 1.91,
        'IOL_Research': 2.10, 'Aya23': 3.03, 'Llama3-70B': 3.07, 'Team-J': 3.91,
        'NTTSU': 4.34, 'ONLINE-B': 5.27, 'IKUN-C': 6.60, 'MSLC': 9.19,
    },
}

OFFICIAL_PATHS = {
    'ende24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/en-de.mqm.sys.score',
    'enes24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/en-es.mqm.sys.score',
    'jazh24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/ja-zh.mqm.sys.score',
}


def load_official(name):
  vals = {}
  with open(os.path.join(ROOT, OFFICIAL_PATHS[name])) as f:
    for line in f:
      s, v = line.strip().split('\t')
      vals[s] = float(v) if v != 'None' else float('nan')
  return vals


def compute_full(name):
  """All-category score (matches what the official pipeline/paper report),
  not gated by adequacy/fluency classification."""
  path = os.path.join(ROOT, SETS[name])
  acc = defaultdict(float)
  all_keys = set()
  with open(path, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    fn = reader.fieldnames
    dcol, scol = _doc_id_col(fn), _seg_id_col(fn)
    for row in reader:
      if not row['rater']:
        continue
      if not row.get('target', '').strip() or not row.get('source', '').strip():
        continue
      key = (row['system'], row['doc'], row[dcol], row[scol], row['rater'])
      all_keys.add(key)
      severity, category = row['severity'], row['category']
      if not severity or not category:
        continue
      acc[key] += error_weight(severity, category)
  records = [
      {'system': k[0], 'doc': k[1], 'doc_id': k[2], 'seg_id': k[3], 'full': -acc.get(k, 0.0)}
      for k in all_keys
  ]
  df = pd.DataFrame.from_records(records)
  per_seg = df.groupby(['system', 'doc', 'doc_id', 'seg_id'])['full'].mean().reset_index()
  per_seg = filter_modal_coverage(per_seg)
  return per_seg.groupby('system')['full'].mean()


if __name__ == '__main__':
  for name in PAPER_TABLE5:
    ab_df = system_level_scores(parse_mqm_tsv(os.path.join(ROOT, SETS[name])))
    full = compute_full(name)
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
