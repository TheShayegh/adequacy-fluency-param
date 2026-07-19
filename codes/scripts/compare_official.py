"""Compare our reconstructed system-level MQM scores against the actual
OFFICIAL WMT scores (external/mt-metrics-eval-data/mt-metrics-eval-v2/*/
human-scores/*.mqm.sys.score) for every system-set where an official file
exists -- real ground truth. codes/mwb/mqm_scoring.load_system_scores()
prefers official per-error rating data when available (see its module
docstring); this script validates the 'full' (all-category) column it
produces against the official total directly.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

# set name -> official .mqm.sys.score path
OFFICIAL_PATHS = {
    'ende20': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt20/human-scores/en-de.mqm.sys.score',
    'zhen20': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt20/human-scores/zh-en.mqm.sys.score',
    'ende21': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt21.news/human-scores/en-de.mqm.sys.score',
    'zhen21': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt21.news/human-scores/zh-en.mqm.sys.score',
    'ted_ende': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt21.tedtalks/human-scores/en-de.mqm.sys.score',
    'ted_zhen': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt21.tedtalks/human-scores/zh-en.mqm.sys.score',
    'ende22': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt22/human-scores/en-de.mqm.sys.score',
    'zhen22': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.sys.score',
    'enru22': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt22/human-scores/en-ru.mqm.sys.score',
    'ende23': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt23/human-scores/en-de.mqm.sys.score',
    'zhen23': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt23/human-scores/zh-en.mqm.sys.score',
    'heen23': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt23/human-scores/he-en.mqm.sys.score',
    'ende24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/en-de.mqm.sys.score',
    'enes24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/en-es.mqm.sys.score',
    'jazh24': 'external/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24/human-scores/ja-zh.mqm.sys.score',
}
# Not available: generalMT2022/enzh (malformed raw TSV, no official data
# either -- see extract_labels.py for a sanity check on that set instead).

# wmt20's sys.score file space-delimits and appends a numeric submission-id
# suffix to system names (e.g. "Huoshan_Translate.832"); also renames the
# human/reference translations (confirmed by value against the published
# ranking in external/wmt-mqm-human-evaluation's README: Human-B 0.75 <
# Human-A 0.91 < Human-P 1.41, matching refb < ref < refp).
_WMT20_ALIASES = {'ref': 'Human-A', 'refb': 'Human-B', 'refp': 'Human-P'}


def load_official(path):
  vals = {}
  with open(os.path.join(ROOT, path)) as f:
    for line in f:
      line = line.rstrip('\n')
      if not line.strip():
        continue
      parts = line.split('\t') if '\t' in line else line.rsplit(' ', 1)
      s, v = parts
      s = re.sub(r'\.\d+$', '', s)
      s = _WMT20_ALIASES.get(s, s)
      vals[s] = float(v) if v != 'None' else float('nan')
  return pd.Series(vals, name='official')


if __name__ == '__main__':
  print(f"{'set':10} {'K(ours)':>7} {'K(official)':>11} {'max|diff|':>10} {'mean|diff|':>10} {'var(off)':>9} {'var(ours)':>10}")
  for name, official_path in OFFICIAL_PATHS.items():
    ours = load_system_scores(name, root=ROOT)['full']
    official = load_official(official_path)
    joined = pd.DataFrame({'ours': ours, 'official': official}).dropna()
    diff = (joined['ours'] - joined['official']).abs()
    print(f"{name:10} {len(ours):7d} {len(official):11d} {diff.max():10.4f} {diff.mean():10.4f} "
          f"{official.var(ddof=0):9.4f} {ours.var(ddof=0):10.4f}")
