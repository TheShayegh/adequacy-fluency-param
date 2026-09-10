"""Cross-validates lib.beta_ess_preference_grid.preference_scores -- the
vectorized, quadratic-form SPA path -- against the library's own
lib.metametrics.soft_pairwise_accuracy_from_pvalues + lib.
synthetic_scorer_orientation.orientation_score, evaluated one weight vector
and one dial at a time.

Why this exists: preference_scores does not call soft_pairwise_accuracy_
from_pvalues at all. It rewrites weighted SPA as a ratio of two quadratic
forms in w so that every SPA value in a run becomes one entry of a single
dense matrix product (see that module's docstring). That rewrite is exact in
principle but is precisely the kind of thing that can be subtly wrong -- an
off-by-one in the upper-triangle pair ordering, the validity mask taken per
dial instead of per donor, a transposed dial axis -- in a way the heatmap's
own output would look entirely plausible with. This script is the trust
anchor, in the same spirit as lib.reweight_exhaustive standing behind
lib.reweight_exact.

The slow path is deliberately written straight through the public library
functions, re-deriving everything (including recomputing the permutation
tests) rather than reusing any of the fast path's cached tables, so the two
share as little machinery as possible.

Both paths average over THE SAME donor subset, so their agreement is exact;
--n-donors trims that subset only for speed, and a trimmed run's absolute
preference values are not comparable to a full heatmap run's.

Usage: python codes/scripts/validate_beta_ess_preference.py
           [--dataset heen23] [--n-donors 4] [--n-weights 3] [--seed 7]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.beta_ess_preference_grid import load_donor_tables, preference_scores
from lib.consistency import real_scorers, real_systems
from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from lib.metric_scores import load_human_seg_scores, load_metric_seg_scores
from lib.spa_plane import positional_af_matrices
from lib.synthetic_scorer_orientation import orientation_score
from lib.synthetic_scorers import DIAGONAL_ADDITIVE_MEAN_DIAL_GRID, donor_family_additive_mean_diagonal

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def slow_preference(dataset, systems, donors, w, root=ROOT):
  """One weight vector's preference, entirely through the public library."""
  a_pos, b_pos = positional_af_matrices(dataset, systems, root=root)
  human = load_human_seg_scores(dataset, systems, root=root)
  per_donor = []
  for d in donors:
    seg = load_metric_seg_scores(dataset, d, systems, root=root)
    m = ~(np.isnan(a_pos).any(0) | np.isnan(b_pos).any(0)
          | np.isnan(seg).any(0) | np.isnan(human).any(0))
    p_human = pairwise_p_values(human[:, m])
    fam = donor_family_additive_mean_diagonal(
        seg[:, m], a_pos[:, m], b_pos[:, m], DIAGONAL_ADDITIVE_MEAN_DIAL_GRID)
    vals = [soft_pairwise_accuracy_from_pvalues(p_human, pairwise_p_values(fam[dl]), w)
            for dl in sorted(fam)]
    per_donor.append(orientation_score(np.asarray(vals)))
  return float(np.nanmean(per_donor))


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='heen23')
  ap.add_argument('--n-donors', type=int, default=4)
  ap.add_argument('--n-weights', type=int, default=3,
                  help='random Dirichlet weight vectors; the uniform weighting is always '
                       'appended on top of these')
  ap.add_argument('--seed', type=int, default=7)
  ap.add_argument('--atol', type=float, default=1e-12)
  args = ap.parse_args()

  systems = real_systems(args.dataset, root=ROOT)
  if not systems:
    sys.exit(f'{args.dataset}: no systems under the project outlier screen')
  K = len(systems)
  donors = real_scorers(args.dataset, root=ROOT, systems=systems, require_seg_scores=True)
  donors = donors[:args.n_donors]
  print(f'{args.dataset}: K={K}, {len(donors)} donors: {", ".join(donors)}')

  rng = np.random.default_rng(args.seed)
  W = rng.dirichlet(np.ones(K), size=args.n_weights)
  W = np.vstack([W, np.full((1, K), 1.0 / K)])  # the uniform weighting

  tables = load_donor_tables(args.dataset, systems, donors, root=ROOT)
  fast = preference_scores(W, tables, K)
  slow = np.array([slow_preference(args.dataset, systems, donors, w) for w in W])

  print(f'\n{"":>6} {"fast":>18} {"slow":>18} {"|diff|":>12}')
  for i, (a, b) in enumerate(zip(fast, slow)):
    label = 'uniform' if i == len(W) - 1 else f'w[{i}]'
    print(f'{label:>6} {a:>18.15f} {b:>18.15f} {abs(a - b):>12.2e}')

  worst = float(np.max(np.abs(fast - slow)))
  ok = bool(np.allclose(fast, slow, atol=args.atol, rtol=0.0))
  print(f'\nmax abs diff: {worst:.2e}   ->   {"MATCH" if ok else "MISMATCH"}')
  sys.exit(0 if ok else 1)
