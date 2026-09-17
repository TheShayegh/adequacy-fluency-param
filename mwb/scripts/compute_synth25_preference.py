"""Computes adequacy/fluency/allmqm preference of the SPA_synth25/
PA_synth25 baseline (mwb.lib.synth25_preference.dataset_synth25_preference)
for one dataset and caches it (output/synth25_preference_<tag>.npz)
-- the expensive step (one SPA/PA permutation-test sweep per dial per
family per base scorer, mwb.lib.synth25_preference.base_scorer_synth25_preference)
separated from plotting (plot_scorer_preference_vs_beta.py's
--overlay-synth25), matching this project's usual compute/plot split.

Unlike compute_scorer_preference_vs_beta.py, there is no beta grid here:
SPA_synth25/PA_synth25 (Shayegh et al. 2025's Row-7 pool -- real systems
union adequacy-synthesized union fluency-synthesized) take no beta
argument, so each family's preference is a single dataset-level number,
not a curve.

--all-pools additionally (or instead, see below) computes the SAME
preference for the other 5 pools mwb.lib.synth25_metametrics.POOL_BLOCKS
defines (real+adeq, real+flu, adeq+flu, flu, adeq -- Row 7 itself is
always included as 'real+adeq+flu'), plus each pool's OWN beta_0 (a
property of its system-level Adequacy/Fluency MQM composition alone,
mwb.lib.synth25_metametrics.pool_beta_0) -- written to a separate
output/synth25_preference_<tag>_pools.npz cache, for
plot_scorer_preference_vs_beta.py's --overlay-synth25 to plot one
marker per (pool, family) at (pool's own beta_0, that pool's
preference), instead of a flat horizontal line at the Row-7 value alone.

Usage: python -m mwb.scripts.compute_synth25_preference [--dataset ende24]
           [--metametric spa|pa] [--synthesis offset|additive|additive_mean]
           [--dial-preset linear|geometric] [--all-pools] [--tag TAG]

--J: switch to the J (Joint) family instead of AF/T (or A/F/T) -- always
offset's own base_scorer_family_joint. --synthesis is ignored in this mode (J's
construction is fixed, not a synthesis choice). --dial-preset picks which
dial grid J itself uses: default (not passed, or anything other than
'symmetric') keeps the historical mwb.lib.synthetic_scorer_preference.
NEG_J_DIAL_GRID ([0, -2] by -0.1) -- the synth25-baseline counterpart of
compute_scorer_preference_vs_beta.py's --family-set Jneg, paired with
some OTHER synthesis's A/F (e.g. additive_mean's) instead of that
synthesis's own T. --dial-preset symmetric instead uses mwb.lib.
synthetic_scorers.AFTJ_J_DIAL_GRID -- the counterpart of --family-set
AFTJ (whose AF/T side uses the DIFFERENT mwb.lib.synthetic_scorers.
AF_ADDITIVE_MEAN_DIAL_GRID/AFT_DIAL_GRID, not this one -- pass
--dial-preset symmetric to BOTH this script's plain --synthesis run for
AF/T and its --J run; each resolves to its own matching grid, both labeled
'symmetric' for tag purposes). Only --all-pools is supported for --J (the
single-Row-7-value mode plot_scorer_preference_vs_beta.py never reads is
skipped).
"""

import argparse
import os
import sys

from mwb.lib.consistency import real_scorers, real_systems
from mwb.lib.synth25_preference import (
    dataset_synth25_preference, dataset_synth25_preference_all_pools, dataset_synth25_preference_J_all_pools,
    preference_tag, pool_beta_0s, pools_tag, pools_tag_J, save_synth25_preference, save_synth25_pools,
    save_synth25_pools_J,
)
from mwb.lib.synth25_metametrics import POOL_BLOCKS, SYNTH25_METAMETRICS
from mwb.lib.synthetic_scorer_preference import NEG_J_DIAL_GRID
from mwb.lib.synthetic_scorers import (
    AFT_DIAL_GRID, AFTJ_J_DIAL_GRID, DIAL_GRID, EXTENDED_ADDITIVE_MEAN_DIAL_GRID, GEOM_DIAL_GRID,
    default_dial_preset,
)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende24')
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='which synth25 metametric to score every dial with')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help='which mwb.lib.synthetic_scorers dial family to build (see '
                            'compute_scorer_preference_vs_beta.py --synthesis for the same choice). '
                            'Ignored under --J.')
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended', 'symmetric'], default='symmetric',
                       help="'linear': DIAL_GRID, 0/0.1/.../1. 'geometric': GEOM_DIAL_GRID, {2^-i} packed "
                            "toward dial=0. 'extended': mwb.lib.synthetic_scorers.EXTENDED_ADDITIVE_MEAN_DIAL_GRID "
                            "(additive_mean only, paired with base_scorer_family_additive_mean's standardized=True "
                            "default). 'symmetric' (default, the paper's committed setup): builds AF (the "
                            "single-dial Adequacy-fluency family, mwb.lib.synthetic_scorers.AF_ADDITIVE_"
                            "MEAN_DIAL_GRID) instead of separate A/F, plus T on mwb.lib.synthetic_scorers."
                            "AFT_DIAL_GRID, or AFTJ_J_DIAL_GRID under --J (paired with --family-set AFTJ) -- "
                            "matching compute_scorer_preference_vs_beta.py's own default so the beta-curve "
                            "cache and this cache use the same families/grids. Under --J, only 'symmetric' has "
                            "any effect (switches J from NEG_J_DIAL_GRID to AFTJ_J_DIAL_GRID) -- see module "
                            'docstring.')
  parser.add_argument('--all-pools', action=argparse.BooleanOptionalAction, default=True,
                       help='ALSO compute and cache the other 5 POOL_BLOCKS pools plus each pool\'s own '
                            'beta_0 (see module docstring) -- roughly 6x the runtime of the single Row-7 '
                            'value alone, since each pool needs its own SPA/PA sweep per base scorer per dial, '
                            'though the dial families themselves are built only once per base scorer. Default: '
                            'True (the committed setup). Required (assumed) under --J.')
  parser.add_argument('--J', action=argparse.BooleanOptionalAction, default=False,
                       help='compute the J (Joint) family instead of A/F/T -- see module docstring.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--workers', type=int, default=1,
                       help='parallel worker processes (ProcessPoolExecutor) for the per-base_scorer loop -- each '
                            'base scorer\'s permutation-test sweep is independent, no beta sweep to amortize the '
                            'cost against unlike compute_scorer_preference_vs_beta.py, so this is usually '
                            'the slower of the two computations. Default: 1 (sequential, original behavior).')
  args = parser.parse_args()
  base = args.dataset
  metametric_name = f'{args.metametric}_synth25'
  assert metametric_name in SYNTH25_METAMETRICS
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)
  dial_grid = {
      'linear': DIAL_GRID, 'geometric': GEOM_DIAL_GRID, 'extended': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
      'symmetric': AFT_DIAL_GRID,
  }[args.dial_preset]

  # Under --J, only 'symmetric' changes anything (NEG_J_DIAL_GRID
  # otherwise, regardless of what --dial-preset resolved to above for the
  # unused A/F/T dial_grid) -- see module docstring.
  j_dial_preset = 'symmetric' if (args.J and args.dial_preset == 'symmetric') else 'neg'
  j_dial_grid = AFTJ_J_DIAL_GRID if j_dial_preset == 'symmetric' else NEG_J_DIAL_GRID
  # 'symmetric' is the paper's committed setup (paired with --family-set
  # AFTJ over in compute_scorer_preference_vs_beta.py): AF (the
  # single-dial Adequacy-fluency family) replaces the separate A/F pair.
  use_af_family = args.dial_preset == 'symmetric'

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  print(f'{base}: K={len(systems)} real systems, metametric={metametric_name}, '
        f'{f"J family (offset, dial_preset={j_dial_preset})" if args.J else f"synthesis={args.synthesis}, dial_preset={args.dial_preset}"}, '
        f'all_pools={args.all_pools}', file=sys.stderr)

  n_base_scorers = len(real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True))
  os.makedirs(DATA_DIR, exist_ok=True)

  if args.J:
    pool_preference = dataset_synth25_preference_J_all_pools(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=j_dial_grid, progress=True,
        workers=args.workers,
    )
    for pool_name in POOL_BLOCKS:
      print(f'{base}: pool {pool_name} joint={pool_preference[pool_name]:.4f}', file=sys.stderr)

    tag = args.tag or pools_tag_J(base, metametric_name, dial_preset=j_dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_preference_{tag}.npz')
    save_synth25_pools_J(
        out_path, dataset=base, metametric_name=metametric_name, n_base_scorers=n_base_scorers,
        pool_preference=pool_preference, dial_preset=j_dial_preset,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
  elif args.all_pools:
    pool_beta0 = pool_beta_0s(base, systems, root=ROOT)
    for pool_name in POOL_BLOCKS:
      print(f'{base}: pool {pool_name} beta_0={pool_beta0[pool_name]:.4f}', file=sys.stderr)

    pool_preference = dataset_synth25_preference_all_pools(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=args.synthesis, progress=True, workers=args.workers,
        use_af_family=use_af_family,
    )
    for pool_name in POOL_BLOCKS:
      o = pool_preference[pool_name]
      if use_af_family:
        print(f'{base}: pool {pool_name} adequacy_over_fluency={o["AF"]:.4f} allmqm={o["T"]:.4f}',
              file=sys.stderr)
      else:
        print(f'{base}: pool {pool_name} adequacy={o["A"]:.4f} fluency={o["F"]:.4f} allmqm={o["T"]:.4f}',
              file=sys.stderr)

    tag = args.tag or pools_tag(base, metametric_name, args.synthesis, args.dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_preference_{tag}.npz')
    save_synth25_pools(
        out_path, dataset=base, metametric_name=metametric_name, synthesis=args.synthesis,
        dial_preset=args.dial_preset, n_base_scorers=n_base_scorers, pool_beta_0=pool_beta0, pool_preference=pool_preference,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
  else:
    preference = dataset_synth25_preference(
        base, systems, root=ROOT, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=args.synthesis, progress=True,
        use_af_family=use_af_family,
    )
    if use_af_family:
      print(f'{base}: adequacy_over_fluency={preference["AF"]:.4f} allmqm={preference["T"]:.4f}',
            file=sys.stderr)
    else:
      print(f'{base}: adequacy={preference["A"]:.4f} fluency={preference["F"]:.4f} '
            f'allmqm={preference["T"]:.4f}', file=sys.stderr)

    tag = args.tag or preference_tag(base, metametric_name, args.synthesis, args.dial_preset)
    out_path = os.path.join(DATA_DIR, f'synth25_preference_{tag}.npz')
    save_synth25_preference(
        out_path, dataset=base, metametric_name=metametric_name, synthesis=args.synthesis,
        dial_preset=args.dial_preset, n_base_scorers=n_base_scorers, preference=preference,
    )
    print(f'Wrote {out_path}', file=sys.stderr)
