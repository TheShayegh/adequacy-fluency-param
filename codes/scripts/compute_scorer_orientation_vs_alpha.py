"""Computes <adequacy|fluency>_orientation(metametric_alpha, s_0) (lib.
synthetic_scorer_orientation, material/synthetic_scorer_construction.md +
material/action_plan.md sections 3, 6.1, 7) for every real donor scorer s_0
in a dataset, across an alpha grid, and caches the result to disk
(artifacts/data/scorer_orientation_<tag>.npz) -- the expensive step (lib.
reweight_exact solves, plus SPA permutation tests when metametric=spa, one
donor_alpha_dial_grid call per donor) separated from plotting (plot_scorer_
orientation_vs_alpha.py) so replotting/restyling never requires rerunning it.

For every donor s_0, builds its A-family and B-family over dial in [0, 1]
by 0.1 (11 scorers each, the full section-4 sweep by default -- see
--synthesis for the two additive alternatives, lib.synthetic_scorers), then
at every alpha, scores every same-family dial pair (C(11,2)=55 of them)
with the weighted meta-metric (--metametric spa|pa, action_plan.md section
7 for PA; lib.synthetic_scorer_alpha_grid.donor_alpha_dial_grid) and
reduces to orientation_score -- the fraction of pairs where the metametric
prefers the more extreme (further-from-the-real-donor) dial. Also stores
ESS(w*(alpha)) (action_plan.md section 5) alongside, for plotting next to
the orientation curves.

Usage: python codes/scripts/compute_scorer_orientation_vs_alpha.py [--dataset ende21]
           [--n-steps 5] [--step 0.01] [--full-range | --union-grid] [--metametric spa|pa]
           [--synthesis offset|additive|additive_mean] [--dial-preset linear|geometric]
           [--aspect-donors] [--green-family default|Jneg] [--tag TAG]

--green-family Jneg (only meaningful with --synthesis additive/additive_mean,
whose own third family is T): overrides the cached third family with J
instead, but a DIFFERENT J than offset's own default -- built from an
EXTRA donor_alpha_dial_grid(..., synthesis='offset', dial_grid=lib.
synthetic_scorer_orientation.NEG_J_DIAL_GRID) call per donor (A/B still
come from the main --synthesis run; only J is swapped in from this second
call, and only J is kept from it -- its own A/B are discarded). A/B stay
exactly what --synthesis alone would have produced -- this is for overlay
plots that want e.g. additive_mean's A/B next to offset's negative-dial J
instead of additive_mean's own T. This extra J call always uses
NEG_J_DIAL_GRID regardless of --dial-preset (matches offset's own
EXTENDED_OFFSET_DIAL_GRID convention of ignoring --dial-preset entirely).

--green-family ABTJ (--synthesis must be additive/additive_mean): caches
ALL FOUR families at once -- A, B, T from the main --synthesis run (T is
that run's own third family, kept this time instead of being discarded),
PLUS J from an EXTRA donor_alpha_dial_grid(..., synthesis='offset') call,
only J kept from it. Both directions are forced regardless of
--dial-preset, but onto DIFFERENT grids: the main run uses lib.
synthetic_scorers.ABT_DIAL_GRID, the extra J call uses ABTJ_J_DIAL_GRID --
see those constants' own docstring for why ascending raw dial value is
ascending "goodness" for both (so, unlike NEG_J_DIAL_GRID, neither needs
reordering).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0, alpha_min_max, union_alpha_grid
from lib.consistency import real_scorers, real_systems
from lib.reweighted_consistency import solve_w_for_datasets
from lib.synthetic_scorer_alpha_grid import ASPECT_DONORS, alpha_grid_around, donor_alpha_dial_grid
from lib.synthetic_scorer_orientation import (
    DIAL_GRID, NEG_J_DIAL_GRID, donor_orientation_by_alpha, full_range_alphas, orientation_tag, save_orientation_data,
    union_grid_tag,
)
from lib.synthetic_scorers import (
    ABT_DIAL_GRID, ABTJ_J_DIAL_GRID, EXTENDED_ADDITIVE_MEAN_DIAL_GRID, GEOM_DIAL_GRID, default_dial_preset,
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
  parser.add_argument('--union-grid', action=argparse.BooleanOptionalAction, default=True,
                       help='use the union of every pairwise alpha_ij of D with a plain uniform '
                            '[alpha_min, alpha_max] sweep at --step (lib.alpha.union_alpha_grid) instead '
                            'of --full-range/the alpha_0(D)-centered window. Default: True (the committed '
                            'setup, matching --synthesis additive_mean/--green-family ABTJ below).')
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='weighted meta-metric to score each dial/alpha cell with')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help="which lib.synthetic_scorers dial family to build: 'offset' (default, the "
                            "original section-4 generator), 'additive' (y + dial*aspect), or "
                            "'additive_mean' (y + dial*abar_k)")
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended'], default=None,
                       help="'linear': DIAL_GRID, 0/0.1/.../1. 'geometric': GEOM_DIAL_GRID, {2^-i : "
                            'i=0..10} packed toward dial=0 -- additive/additive_mean saturate almost '
                            'immediately on the linear grid, so this resolves the early part of the '
                            "sweep instead (lib.synthetic_scorers). 'extended': lib.synthetic_scorers."
                            'EXTENDED_ADDITIVE_MEAN_DIAL_GRID, dial 0..~8 -- additive_mean only, paired '
                            "with donor_family_additive_mean's standardized=True default (injected term "
                            "rescaled to the donor's own between-system spread), where dial=1 is now only "
                            "~1 SD and a much wider reach is needed for a genuinely extreme endpoint. "
                            "Default: lib.synthetic_scorers.default_dial_preset(synthesis) -- 'geometric' "
                            "for additive/additive_mean, 'linear' for offset -- 'extended' is never a "
                            'default, always explicit.')
  parser.add_argument('--aspect-donors', action=argparse.BooleanOptionalAction, default=False,
                       help="compute only lib.synthetic_scorer_alpha_grid.ASPECT_DONORS ('AdequacyMQM', "
                            "'FluencyMQM', 'AllMQM' -- the gold aspect signals themselves faking to be a "
                            "donor scorer) instead of the real_scorers-screened candidate list")
  parser.add_argument('--green-family', choices=['default', 'Jneg', 'ABTJ'], default='ABTJ',
                       help="'default': third family is whatever --synthesis produces (T for additive/"
                            "additive_mean, J for offset). 'Jneg': override the third family with offset's "
                            "J built on NEG_J_DIAL_GRID ([0, -2] by -0.1) instead. 'ABTJ' (default, the "
                            'committed setup): cache A, B, T, and J all at once -- A/B/T on lib.'
                            'synthetic_scorers.ABT_DIAL_GRID, J on ABTJ_J_DIAL_GRID -- see module docstring '
                            'for both.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  args = parser.parse_args()
  base = args.dataset
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)
  dial_grid = {
      'linear': DIAL_GRID, 'geometric': GEOM_DIAL_GRID, 'extended': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  }[args.dial_preset]
  if args.green_family == 'ABTJ':
    dial_grid = ABT_DIAL_GRID
    args.dial_preset = 'symmetric'  # so the auto tag/cache metadata reflect the actual grid, not --dial-preset

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  K = len(systems)

  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[systems, 'a'].values, df.loc[systems, 'b'].values
  center_alpha = alpha_0(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  if args.union_grid:
    alphas = union_alpha_grid(a_D, b_D, step=args.step)
  elif args.full_range:
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

  if args.aspect_donors:
    candidates = list(ASPECT_DONORS)
  else:
    # real_scorers, not the raw load_metric_sys_scores candidate list: the
    # project's canonical scorer screen (drops zero-variance/sentinel
    # scorers and deduplicates byte-identical WMT submissions), so degenerate
    # or double-counted donors don't quietly bias the dataset-level
    # orientation average or the capability screen downstream.
    candidates = real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True)
  print(f'{base}: {len(candidates)} candidate donors, dial grid ({args.dial_preset}) {dial_grid}', file=sys.stderr)

  n = len(candidates)
  t0 = time.time()
  orient_by_donor = {}
  for i, donor in enumerate(candidates, 1):
    grids = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=dial_grid,
                                   metametric=args.metametric, synthesis=args.synthesis)
    if grids is None:
      print(f'[{i}/{n}] SKIPPED {donor} (insufficient segment coverage)', file=sys.stderr)
      continue
    orient = donor_orientation_by_alpha(grids)
    if args.green_family in ('Jneg', 'ABTJ'):
      j_dial_grid = ABTJ_J_DIAL_GRID if args.green_family == 'ABTJ' else NEG_J_DIAL_GRID
      grids_j = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=j_dial_grid,
                                       metametric=args.metametric, synthesis='offset')
      if grids_j is None:
        print(f'[{i}/{n}] SKIPPED {donor} (insufficient coverage for the offset-J overlay)', file=sys.stderr)
        continue
      orient['J'] = donor_orientation_by_alpha(grids_j)['J']
    orient_by_donor[donor] = orient
    elapsed = time.time() - t0
    eta = elapsed / i * (n - i)
    print(f'[{i}/{n}] computed {donor} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  print(f'{base}: {len(orient_by_donor)}/{n} donors used', file=sys.stderr)

  donors = sorted(orient_by_donor)
  A = np.array([orient_by_donor[d]['A'] for d in donors])
  B = np.array([orient_by_donor[d]['B'] for d in donors])
  if args.green_family == 'ABTJ':
    # Both T (the main --synthesis run's own third family) and J (the
    # extra offset call) are kept this time, not just one -- see module
    # docstring.
    extra = {'T': np.array([orient_by_donor[d]['T'] for d in donors]),
             'J': np.array([orient_by_donor[d]['J'] for d in donors])}
  else:
    # 'J' (Joint) for synthesis='offset' OR --green-family Jneg (which
    # overrides whatever --synthesis's own third family would have been),
    # 'T' (AllMQM) otherwise (additive/additive_mean's own third family) --
    # matches whichever third family ended up in orient_by_donor, never both.
    third_label = 'J' if (args.synthesis == 'offset' or args.green_family == 'Jneg') else 'T'
    extra = {third_label: np.array([orient_by_donor[d][third_label] for d in donors])}

  if args.tag:
    tag = args.tag
  elif args.union_grid:
    tag = union_grid_tag(base, args.step, args.metametric, args.synthesis, args.dial_preset)
  else:
    tag = orientation_tag(base, args.n_steps, args.step, args.full_range, args.metametric, args.synthesis,
                           args.dial_preset)
  if args.aspect_donors and not args.tag:
    tag += '_aspectdonors'
  if args.green_family in ('Jneg', 'ABTJ') and not args.tag:
    tag += f'_{args.green_family}'
  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  save_orientation_data(
      out_path, dataset=base, alphas=alphas, center_alpha=center_alpha, K=K, donors=donors,
      A=A, B=B, ess=ess_by_alpha, dial_grid=dial_grid, metametric=args.metametric, synthesis=args.synthesis,
      dial_preset=args.dial_preset, **extra,
  )
  print(f'Wrote {out_path}', file=sys.stderr)