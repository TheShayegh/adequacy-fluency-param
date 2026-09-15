"""Computes adequacy/fluency/allmqm orientation of the SPA_synth25/
PA_synth25 baseline (mwb.lib.synth25_orientation.dataset_synth25_orientation)
for one dataset and caches it (output/data/synth25_orientation_<tag>.npz)
-- the expensive step (one SPA/PA permutation-test sweep per dial per
family per donor, mwb.lib.synth25_orientation.donor_synth25_orientation)
separated from plotting (plot_scorer_orientation_vs_alpha.py's
--overlay-synth25), matching this project's usual compute/plot split.

Unlike compute_scorer_orientation_vs_alpha.py, there is no alpha grid here:
SPA_synth25/PA_synth25 (Shayegh et al. 2025's Row-7 pool -- real systems
union adequacy-synthesized union fluency-synthesized) take no alpha
argument, so each family's orientation is a single dataset-level number,
not a curve.

--all-pools additionally (or instead, see below) computes the SAME
orientation for the other 5 pools mwb.lib.synth25_metametrics.POOL_BLOCKS
defines (real+adeq, real+flu, adeq+flu, flu, adeq -- Row 7 itself is
always included as 'real+adeq+flu'), plus each pool's OWN alpha_0 (a
property of its system-level Adequacy/Fluency MQM composition alone,
mwb.lib.synth25_metametrics.pool_alpha_0) -- written to a separate
output/data/synth25_orientation_<tag>_pools.npz cache, for
plot_scorer_orientation_vs_alpha.py's --overlay-synth25 to plot one
marker per (pool, family) at (pool's own alpha_0, that pool's
orientation), instead of a flat horizontal line at the Row-7 value alone.

Usage: python -m mwb.scripts.compute_synth25_orientation [--dataset ende24]
           [--metametric spa|pa] [--synthesis offset|additive|additive_mean]
           [--dial-preset linear|geometric] [--all-pools] [--tag TAG]

--J: switch to the J (Joint) family instead of A/B/T -- always offset's own
donor_family_joint. --synthesis is ignored in this mode (J's construction
is fixed, not a synthesis choice). --dial-preset picks which dial grid J
itself uses: default (not passed, or anything other than 'symmetric')
keeps the historical mwb.lib.synthetic_scorer_orientation.NEG_J_DIAL_GRID
([0, -2] by -0.1) -- the synth25-baseline counterpart of
compute_scorer_orientation_vs_alpha.py's --green-family Jneg, paired with
some OTHER synthesis's A/B (e.g. additive_mean's) instead of that
synthesis's own T. --dial-preset symmetric instead uses mwb.lib.
synthetic_scorers.ABTJ_J_DIAL_GRID -- the counterpart of --green-family
ABTJ (whose A/B/T side uses the DIFFERENT mwb.lib.synthetic_scorers.
ABT_DIAL_GRID, not this one -- pass --dial-preset symmetric to BOTH this
script's plain --synthesis run for A/B/T and its --J run; each resolves to
its own matching grid, both labeled 'symmetric' for tag purposes). Only
--all-pools is supported for --J (the single-Row-7-value mode
plot_scorer_orientation_vs_alpha.py never reads is skipped).
"""

import argparse
import os
import sys

from mwb.lib.consistency import real_scorers, real_systems
from mwb.lib.synth25_orientation import (
    dataset_synth25_orientation, dataset_synth25_orientation_all_pools, dataset_synth25_orientation_J_all_pools,
    orientation_tag, pool_alpha_0s, pools_tag, pools_tag_J, save_synth25_orientation, save_synth25_pools,
    save_synth25_pools_J,
)
from mwb.lib.synth25_metametrics import POOL_BLOCKS, SYNTH25_METAMETRICS
from mwb.lib.synthetic_scorer_orientation import NEG_J_DIAL_GRID
from mwb.lib.synthetic_scorers import (
    ABT_DIAL_GRID, ABTJ_J_DIAL_GRID, DIAL_GRID, EXTENDED_ADDITIVE_MEAN_DIAL_GRID, GEOM_DIAL_GRID,
    default_dial_preset,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output', 'data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende24')
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='which synth25 metametric to score every dial with')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help='which mwb.lib.synthetic_scorers dial family to build (see '
                            'compute_scorer_orientation_vs_alpha.py --synthesis for the same choice). '
                            'Ignored under --J.')
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended', 'symmetric'], default='symmetric',
                       help="'linear': DIAL_GRID, 0/0.1/.../1. 'geometric': GEOM_DIAL_GRID, {2^-i} packed "
                            "toward dial=0. 'extended': mwb.lib.synthetic_scorers.EXTENDED_ADDITIVE_MEAN_DIAL_GRID "
                            "(additive_mean only, paired with donor_family_additive_mean's standardized=True "
                            "default). 'symmetric' (default, the committed setup): mwb.lib.synthetic_scorers."
                            "ABT_DIAL_GRID for A/B/T, or ABTJ_J_DIAL_GRID under --J (paired with "
                            "--green-family ABTJ) -- matching compute_scorer_orientation_vs_alpha.py's own "
                            "default so the alpha-curve cache and this cache use the same dial grid. Under "
                            "--J, only 'symmetric' has any effect (switches J from NEG_J_DIAL_GRID to "
                            'ABTJ_J_DIAL_GRID) -- see module docstring.')
  parser.add_argument('--all-pools', action=argparse.BooleanOptionalAction, default=True,
                       help='ALSO compute and cache the other 5 POOL_BLOCKS pools plus each pool\'s own '
                            'alpha_0 (see module docstring) -- roughly 6x the runtime of the single Row-7 '
                            'value alone, since each pool needs its own SPA/PA sweep per donor per dial, '
                            'though the dial families themselves are built only once per donor. Default: '
                            'True (the committed setup). Required (assumed) under --J.')
  parser.add_argument('--J', action=argparse.BooleanOptionalAction, default=False,
                       help='compute the J (Joint) family instead of A/B/T -- see module docstring.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--workers', type=int, default=1,
                       help='parallel worker processes (ProcessPoolExecutor) for the per-donor loop -- each '
                            'donor\'s permutation-test sweep is independent, no alpha sweep to amortize the '
                            'cost against unlike compute_scorer_orientation_vs_alpha.py, so this is usually '
                            'the slower of the two computations. Default: 1 (sequential, original behavior).')
  args = parser.parse_args()
  base = args.dataset
  metametric_name = f'{args.metametric}_synth25'
  assert metametric_name in SYNTH25_METAMETRICS
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)
  dial_grid = {
      'linear': DIAL_GRID, 'geometric': GEOM_DIAL_GRID, 'extended': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
      'symmetric': ABT_DIAL_GRID,
  }[args.dial_preset]

  # Under --J, only 'symmetric' changes anything (NEG_J_DIAL_GRID
  # otherwise, regardless of what --dial-preset resolved to above for the
  # unused A/B/T dial_grid) -- see module docstring.
  j_dial_preset = 'symmetric' if (args.J and args.dial_preset == 'symmetric') else 'neg'
  j_dial_grid = ABTJ_J_DIAL_GRID if j_dial_preset == 'symmetric' else NEG_J_DIAL_GRID

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  print(f'{base}: K={len(systems)} real systems, metametric={metametric_name}, '
        f'{f"J family (offset, dial_preset={j_dial_preset})" if args.J else f"synthesis={args.synthesis}, dial_preset={args.dial_preset}"}, '
        f'all_pools={args.all_pools}', file=sys.stderr)

  n_donors = len(real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True))
  os.makedirs(DATA_DIR, exist_ok=True)

  if args.J:
    pool_orientation = dataset_synth25_orientation_J_all_pools(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=j_dial_grid, progress=True,
        workers=args.workers,
    )
    for pool_name in POOL_BLOCKS:
      print(f'{base}: pool {pool_name} joint={pool_orientation[pool_name]:.4f}', file=sys.stderr)

    tag = args.tag or pools_tag_J(base, metametric_name, dial_preset=j_dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_orientation_{tag}.npz')
    save_synth25_pools_J(
        out_path, dataset=base, metametric_name=metametric_name, n_donors=n_donors,
        pool_orientation=pool_orientation, dial_preset=j_dial_preset,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
  elif args.all_pools:
    pool_alpha0 = pool_alpha_0s(base, systems, root=ROOT)
    for pool_name in POOL_BLOCKS:
      print(f'{base}: pool {pool_name} alpha_0={pool_alpha0[pool_name]:.4f}', file=sys.stderr)

    pool_orientation = dataset_synth25_orientation_all_pools(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=args.synthesis, progress=True, workers=args.workers,
    )
    for pool_name in POOL_BLOCKS:
      o = pool_orientation[pool_name]
      print(f'{base}: pool {pool_name} adequacy={o["A"]:.4f} fluency={o["B"]:.4f} allmqm={o["T"]:.4f}',
            file=sys.stderr)

    tag = args.tag or pools_tag(base, metametric_name, args.synthesis, args.dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_orientation_{tag}.npz')
    save_synth25_pools(
        out_path, dataset=base, metametric_name=metametric_name, synthesis=args.synthesis,
        dial_preset=args.dial_preset, n_donors=n_donors, pool_alpha0=pool_alpha0, pool_orientation=pool_orientation,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
  else:
    orientation = dataset_synth25_orientation(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=args.synthesis, progress=True,
    )
    print(f'{base}: adequacy={orientation["A"]:.4f} fluency={orientation["B"]:.4f} '
          f'allmqm={orientation["T"]:.4f}', file=sys.stderr)

    tag = args.tag or orientation_tag(base, metametric_name, args.synthesis, args.dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_orientation_{tag}.npz')
    save_synth25_orientation(
        out_path, dataset=base, metametric_name=metametric_name, synthesis=args.synthesis,
        dial_preset=args.dial_preset, n_donors=n_donors, orientation=orientation,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
