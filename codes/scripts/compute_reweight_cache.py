"""Solves w_d(alpha) (EXACT solver, lib.reweight_exact -- sound and
complete, no restarts to tune) for every real dataset d across a common
alpha grid, and caches the result to disk (artifacts/data/
w_cache[_TAG].pkl). This is the expensive, metametric-independent step
shared by every curve in compute_consistency_curve.py -- run once here,
reused by that script once per meta-metric so the solver never has to run
twice for the same (dataset, alpha).

Usage: python codes/scripts/compute_reweight_cache.py [--grid N]
                                                        [--years 2022,2023,2024]
                                                        [--pair en-de]
                                                        [--exclude ted_ende,ted_zhen]
--years restricts to those years' datasets (lib.dataset_dirs.
datasets_for_years); --pair restricts to one language pair across all years
(lib.dataset_dirs.datasets_for_pair, e.g. 'en-de' pulls in ende20/21/22/23/24
plus ted_ende); --exclude removes specific dataset names after those filters
(harmless to name one not present, e.g. passing both ted_ende and ted_zhen
to every run regardless of which one that run actually contains). Every
output file is tagged with whichever filter(s) were used, so a restricted-
dataset run never clobbers the full-15-dataset one (or another restricted
run) -- --exclude only contributes to the tag the datasets it actually
removed, so passing an inapplicable name doesn't add noise to the tag.
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import build_alpha_grid
from lib.dataset_dirs import DATASET_DIRS, datasets_for_years, datasets_for_pair
from lib.reweighted_consistency import common_alpha_range, dataset_alpha_0s, solve_w_for_datasets

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--grid', type=int, default=21, help='number of alpha grid points')
  p.add_argument('--years', type=str, default=None,
                  help='comma-separated years, e.g. 2022,2023,2024')
  p.add_argument('--pair', type=str, default=None,
                  help="language pair, e.g. 'en-de' (all years/domains, default: all 15 datasets)")
  p.add_argument('--exclude', type=str, default=None,
                  help='comma-separated dataset names to remove after --years/--pair filters')
  args = p.parse_args()

  DATASETS = set(DATASET_DIRS.keys())
  tag_parts = []
  if args.years:
    years = [y.strip() for y in args.years.split(',')]
    DATASETS &= set(datasets_for_years(years))
    tag_parts.extend(years)
  if args.pair:
    DATASETS &= set(datasets_for_pair(args.pair))
    tag_parts.append(args.pair.replace('-', '_'))
  if args.exclude:
    requested = [e.strip() for e in args.exclude.split(',')]
    actually_excluded = sorted(DATASETS & set(requested))
    DATASETS -= set(requested)
    if actually_excluded:
      tag_parts.append('excl_' + '_'.join(actually_excluded))
  DATASETS = [d for d in DATASET_DIRS if d in DATASETS]  # stable, canonical order
  tag = '_'.join(tag_parts) or None
  cache_path = os.path.join(ROOT, 'artifacts', 'data', f'w_cache{f"_{tag}" if tag else ""}.pkl')

  print(f'datasets ({len(DATASETS)}): {DATASETS}', file=sys.stderr)
  lo, hi = common_alpha_range(DATASETS, root=ROOT)
  alpha_0 = dataset_alpha_0s(DATASETS, root=ROOT)
  alphas, dropped_a0 = build_alpha_grid(lo, hi, args.grid, alpha_0)
  print(f'common alpha range: [{lo:.4f}, {hi:.4f}], grid of {args.grid} '
        f'+ {len(alphas) - args.grid} alpha_0 points = {len(alphas)}', file=sys.stderr)
  if dropped_a0:
    print(f'  ({len(dropped_a0)} alpha_0 outside the common range, not added: {dropped_a0})',
          file=sys.stderr)

  w_cache = {}
  for i, d in enumerate(DATASETS):
    t0 = time.time()
    partial = solve_w_for_datasets([d], alphas, root=ROOT)
    w_cache.update(partial)
    n_fail = sum(1 for r in partial.values() if not r.success)
    print(f'[{i + 1}/{len(DATASETS)}] {d}: {len(partial)} solves in {time.time() - t0:.1f}s'
          f'{f", {n_fail} NOT feasible" if n_fail else ""}', file=sys.stderr)

  os.makedirs(os.path.dirname(cache_path), exist_ok=True)
  with open(cache_path, 'wb') as f:
    pickle.dump({'datasets': DATASETS, 'alphas': alphas, 'w_cache': w_cache}, f)
  print(f'Wrote {cache_path}', file=sys.stderr)
