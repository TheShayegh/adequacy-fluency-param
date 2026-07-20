"""Per-dataset variant of compute_consistency_curve.py: instead of pooling
tau-a over every (scorer-pair, dataset-pair) comparison at once (one number
per alpha), pools it separately for each dataset d against every OTHER
dataset ("d vs the rest", a star centered at d) -- one tau_bar(d, alpha)
curve per dataset. Reuses the existing w_cache[_TAG].pkl (no new solving:
w_d(alpha) never depended on which other datasets were in the pool) and
lib.consistency.scorer_scores' natural (unweighted) ranking for each
dataset's own baseline point (its natural alpha_0, and its natural,
unreweighted star-pooled tau against the others).

Usage: python codes/scripts/compute_per_dataset_consistency_curve.py [--tag TAG]
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from lib.consistency import real_systems
from lib.metametrics import METAMETRICS
from lib.reweighted_consistency import dataset_alpha_0s, star_pooled_consistency_curve

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--tag', type=str, default=None,
                  help='must match the tag used with compute_reweight_cache.py, if any')
  args = p.parse_args()
  suffix = f'_{args.tag}' if args.tag else ''

  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache{suffix}.pkl')
  with open(cache_path, 'rb') as f:
    cached = pickle.load(f)
  datasets, alphas, w_cache = cached['datasets'], list(cached['alphas']), cached['w_cache']
  print(f'loaded w_cache: {len(datasets)} datasets x {len(alphas)} alphas', file=sys.stderr)

  K = {d: len(real_systems(d, root=ROOT)) for d in datasets}
  alpha_0 = dataset_alpha_0s(datasets, root=ROOT)

  for metametric in sorted(METAMETRICS):
    t0 = time.time()
    star = star_pooled_consistency_curve(metametric, datasets, alphas, w_cache, root=ROOT)

    out = {
        'metametric': metametric,
        'datasets': datasets,
        'alphas': alphas,
        'K': K,
        'alpha_0': alpha_0,
        **star,
    }
    out_path = os.path.join(OUT_DIR, f'curve_perdataset_{metametric}{suffix}.json')
    with open(out_path, 'w') as f:
      json.dump(out, f, indent=2)
    print(f'{metametric:10} ({time.time() - t0:.1f}s) -> {os.path.basename(out_path)}',
          file=sys.stderr)
