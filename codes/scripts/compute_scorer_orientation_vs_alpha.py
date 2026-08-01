"""Computes <adequacy|fluency>_orientation(SPA_alpha, s_0) (lib.
synthetic_scorer_orientation, material/synthetic_scorer_construction.md +
material/action_plan.md sections 3, 6.1) for every real donor scorer s_0 in
a dataset, across an alpha grid, and caches the result to disk
(artifacts/data/scorer_orientation_<tag>.npz) -- the expensive step (lib.
reweight_exact solves plus SPA permutation tests, one donor_alpha_dial_grid
call per donor) separated from plotting (plot_scorer_orientation_vs_alpha.py)
so replotting/restyling never requires rerunning it.

For every donor s_0, builds its A-family and B-family over dial in [0, 0.5]
by 0.1 (6 scorers each), then at every alpha, scores every same-family dial
pair with the weighted meta-metric SPA_alpha (lib.synthetic_scorer_alpha_grid.
donor_alpha_dial_grid) and reduces to orientation_score -- the fraction of
pairs where SPA_alpha prefers the more extreme (further-from-the-real-donor)
dial. Also stores ESS(w*(alpha)) (action_plan.md section 5) alongside, for
plotting next to the orientation curves.

Usage: python codes/scripts/compute_scorer_orientation_vs_alpha.py [--dataset ende21]
           [--n-steps 5] [--step 0.01] [--full-range] [--tag TAG]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0, alpha_min_max
from lib.consistency import real_systems
from lib.metric_scores import load_metric_sys_scores
from lib.reweighted_consistency import solve_w_for_datasets
from lib.synthetic_scorer_alpha_grid import alpha_grid_around, donor_alpha_dial_grid
from lib.synthetic_scorer_orientation import (
    DIAL_GRID, donor_orientation_by_alpha, full_range_alphas, orientation_tag, save_orientation_data,
)
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--n-steps', type=int, default=5)
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--full-range', action=argparse.BooleanOptionalAction, default=False,
                       help='sweep the entire reachable [alpha_min, alpha_max] at --step instead of '
                            'the alpha_0(D)-centered +/- n_steps*step window')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  args = parser.parse_args()
  base = args.dataset

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  K = len(systems)

  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[systems, 'a'].values, df.loc[systems, 'b'].values
  center_alpha = alpha_0(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  if args.full_range:
    alphas = full_range_alphas(alpha_lo, alpha_hi, args.step)
  else:
    alphas = alpha_grid_around(center_alpha, alpha_lo, alpha_hi, args.n_steps, args.step)
  print(f'{base}: K={K} systems, alpha_0(D)={center_alpha:.4f}, alpha grid ({len(alphas)} pts): '
        f'{[round(a, 4) for a in alphas]}', file=sys.stderr)

  w_cache = solve_w_for_datasets([base], alphas, root=ROOT, systems_by_dataset={base: systems})
  w_by_alpha = {a: w_cache[(base, a)].w for a in alphas}
  ess_by_alpha = np.array([w_cache[(base, a)].ess for a in alphas])
  print(f'{base}: ESS(alpha) range [{ess_by_alpha.min():.2f}, {ess_by_alpha.max():.2f}] of [1, {K}]',
        file=sys.stderr)

  candidates = sorted(load_metric_sys_scores(base, systems, root=ROOT))
  print(f'{base}: {len(candidates)} candidate donors, dial grid {DIAL_GRID}', file=sys.stderr)

  n = len(candidates)
  t0 = time.time()
  orient_by_donor = {}
  for i, donor in enumerate(candidates, 1):
    grids = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=DIAL_GRID)
    if grids is None:
      print(f'[{i}/{n}] SKIPPED {donor} (insufficient segment coverage)', file=sys.stderr)
      continue
    orient_by_donor[donor] = donor_orientation_by_alpha(grids)
    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    print(f'[{i}/{n}] computed {donor} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  print(f'{base}: {len(orient_by_donor)}/{n} donors used', file=sys.stderr)

  donors = sorted(orient_by_donor)
  A = np.array([orient_by_donor[d]['A'] for d in donors])
  B = np.array([orient_by_donor[d]['B'] for d in donors])

  tag = args.tag or orientation_tag(base, args.n_steps, args.step, args.full_range)
  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  save_orientation_data(
      out_path, dataset=base, alphas=alphas, center_alpha=center_alpha, K=K, donors=donors,
      A=A, B=B, ess=ess_by_alpha, dial_grid=DIAL_GRID,
  )
  print(f'Wrote {out_path}', file=sys.stderr)