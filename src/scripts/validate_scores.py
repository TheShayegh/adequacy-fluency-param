"""Compares our reconstructed system-level MQM scores against the actual
OFFICIAL WMT scores (external/mt-metrics-eval-data/mt-metrics-eval-v2/*/
human-scores/*.mqm.sys.score) for every system-set where an official file
exists -- real ground truth. src.lib.mqm_scoring.load_system_scores() prefers
official per-error rating data when available (see its module docstring);
this script validates the 'full' (all-category) column it produces against
the official total directly.

Usage: python -m src.scripts.validate_scores
"""

import os
import re

import pandas as pd

from src.lib.wmt_metadata import DATASET_DIRS
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

# set name -> official .mqm.sys.score path, derived from DATASET_DIRS (the
# shared (year_dir, pair) source of truth) rather than hand-listed -- every
# one of the 15 sets there has an official file at this same path shape.
# Not available: generalMT2022/enzh (malformed raw TSV, no official data
# either -- excluded from DATASET_DIRS for the same reason).
OFFICIAL_PATHS = {
    name: f'external/mt-metrics-eval-data/mt-metrics-eval-v2/{year_dir}/human-scores/{pair}.mqm.sys.score'
    for name, (year_dir, pair) in DATASET_DIRS.items()
}

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


def main(argv=None) -> None:
  print(f"{'set':10} {'K(ours)':>7} {'K(official)':>11} {'max|diff|':>10} {'mean|diff|':>10} {'var(off)':>9} {'var(ours)':>10}")
  for name, official_path in OFFICIAL_PATHS.items():
    ours = load_system_scores(name, root=ROOT)['full']
    official = load_official(official_path)
    joined = pd.DataFrame({'ours': ours, 'official': official}).dropna()
    diff = (joined['ours'] - joined['official']).abs()
    print(f"{name:10} {len(ours):7d} {len(official):11d} {diff.max():10.4f} {diff.mean():10.4f} "
          f"{official.var(ddof=0):9.4f} {ours.var(ddof=0):10.4f}")


if __name__ == '__main__':
  main()
