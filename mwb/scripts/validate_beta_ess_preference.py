"""Cross-validates mwb.lib.beta_ess_preference_grid.preference_scores -- the
vectorized, quadratic-form SPA path -- against the library's own
mwb.lib.metametrics.soft_pairwise_accuracy_from_pvalues + mwb.lib.
synthetic_scorer_preference.preference_score, evaluated one weight vector
and one dial at a time.

Why this exists: preference_scores does not call soft_pairwise_accuracy_
from_pvalues at all. It rewrites weighted SPA as a ratio of two quadratic
forms in w so that every SPA value in a run becomes one entry of a single
dense matrix product (see that module's docstring). That rewrite is exact in
principle but is precisely the kind of thing that can be subtly wrong -- an
off-by-one in the upper-triangle pair ordering, the validity mask taken per
dial instead of per base scorer, a transposed dial axis -- in a way the heatmap's
own output would look entirely plausible with. This script is the trust
anchor, in the same spirit as mwb.lib.reweight_exhaustive standing behind
mwb.lib.reweight_exact.

The slow path is deliberately written straight through the public library
functions, re-deriving everything (including recomputing the permutation
tests) rather than reusing any of the fast path's cached tables, so the two
share as little machinery as possible.

Both paths average over THE SAME base scorer subset, so their agreement is exact;
--n-base scorers trims that subset only for speed, and a trimmed run's absolute
preference values are not comparable to a full heatmap run's.

Usage: python -m mwb.scripts.validate_beta_ess_preference
           [--dataset heen23] [--n-base scorers 4] [--n-weights 3] [--seed 7]
"""

import argparse
import os
import sys

import numpy as np

from mwb.lib.beta_ess_preference_grid import load_base_scorer_tables, preference_scores
from mwb.lib.consistency import real_scorers, real_systems
from mwb.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from mwb.lib.spa_plane import positional_af_matrices
from mwb.lib.synthetic_scorer_preference import preference_score
from mwb.lib.synthetic_scorers import AF_ADDITIVE_MEAN_DIAL_GRID, base_scorer_family_additive_mean_af

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def slow_preference(dataset, systems, base_scorers, w, root=ROOT):
  """One weight vector's preference, entirely through the public library."""
  a_pos, f_pos = positional_af_matrices(dataset, systems, root=root)
  human = load_human_seg_scores(dataset, systems, root=root)
  per_base_scorer = []
  for d in base_scorers:
    seg = load_scorer_seg_scores(dataset, d, systems, root=root)
    m = ~(np.isnan(a_pos).any(0) | np.isnan(f_pos).any(0)
          | np.isnan(seg).any(0) | np.isnan(human).any(0))
    p_human = pairwise_p_values(human[:, m])
    fam = base_scorer_family_additive_mean_af(
        seg[:, m], a_pos[:, m], f_pos[:, m], AF_ADDITIVE_MEAN_DIAL_GRID)
    vals = [soft_pairwise_accuracy_from_pvalues(p_human, pairwise_p_values(fam[dl]), w)
            for dl in sorted(fam)]
    per_base_scorer.append(preference_score(np.asarray(vals)))
  return float(np.nanmean(per_base_scorer))


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='heen23')
  ap.add_argument('--n-base_scorers', type=int, default=4)
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
  base_scorers = real_scorers(args.dataset, root=ROOT, systems=systems, require_seg_scores=True)
  base_scorers = base_scorers[:args.n_base_scorers]
  print(f'{args.dataset}: K={K}, {len(base_scorers)} base_scorers: {", ".join(base_scorers)}')

  rng = np.random.default_rng(args.seed)
  W = rng.dirichlet(np.ones(K), size=args.n_weights)
  W = np.vstack([W, np.full((1, K), 1.0 / K)])  # the uniform weighting

  tables = load_base_scorer_tables(args.dataset, systems, base_scorers, root=ROOT)
  fast = preference_scores(W, tables, K)
  slow = np.array([slow_preference(args.dataset, systems, base_scorers, w) for w in W])

  print(f'\n{"":>6} {"fast":>18} {"slow":>18} {"|diff|":>12}')
  for i, (a, b) in enumerate(zip(fast, slow)):
    label = 'uniform' if i == len(W) - 1 else f'w[{i}]'
    print(f'{label:>6} {a:>18.15f} {b:>18.15f} {abs(a - b):>12.2e}')

  worst = float(np.max(np.abs(fast - slow)))
  ok = bool(np.allclose(fast, slow, atol=args.atol, rtol=0.0))
  print(f'\nmax abs diff: {worst:.2e}   ->   {"MATCH" if ok else "MISMATCH"}')
  sys.exit(0 if ok else 1)
