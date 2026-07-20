"""Type-2 (star-pooled tau_bar per dataset) consistency curve for a group
of real datasets that each had one system excluded -- companion to
compute_reweight_cache_excl_system.py, and the excl-system analog of
compute_per_dataset_consistency_curve.py (which has no systems override
and would silently pull the excluded system back in via each dataset's
full roster).

Usage: python codes/scripts/compute_per_dataset_consistency_curve_excl_system.py --tag 2024_excl_MSLC
(run compute_reweight_cache_excl_system.py first, same --tag.)
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lib.metametrics import METAMETRICS_ORDER
from lib.reweighted_consistency import dataset_alpha_0s, star_pooled_consistency_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, required=True)
  args = p.parse_args()
  suffix = f'_{args.tag}'

  cache_path = os.path.join(DATA_DIR, f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  systems_by_dataset = cached['systems_by_dataset']
  print(f'loaded w_cache: {len(datasets)} datasets x {len(alphas)} alphas', file=sys.stderr)

  K = {d: len(systems_by_dataset[d]) for d in datasets}
  alpha_0 = dataset_alpha_0s(datasets, root=ROOT, systems_by_dataset=systems_by_dataset)

  for m in METAMETRICS_ORDER:
    t0 = time.time()
    star = star_pooled_consistency_curve(
        m, datasets, alphas, w_cache, root=ROOT, systems_by_dataset=systems_by_dataset)

    out = {
        'metametric': m,
        'datasets': datasets,
        'K': K,
        'alpha_0': alpha_0,
        'alphas': list(alphas),
        **star,
    }
    out_path = os.path.join(DATA_DIR, f'curve_perdataset_{m}{suffix}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{m:10} ({time.time() - t0:.1f}s) -> {os.path.basename(out_path)}', file=sys.stderr)
