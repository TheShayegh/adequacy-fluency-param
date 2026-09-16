"""Computes <adequacy|fluency>_orientation(metametric_alpha, s_0)
(mwb.lib.synthetic_scorer_orientation) for every real donor scorer s_0 in a
dataset, across an alpha grid, and caches the result to disk
(output/data/scorer_orientation_<tag>.npz) -- the expensive step
(mwb.lib.reweight_exact solves, plus SPA permutation tests when
metametric=spa, one donor_alpha_dial_grid call per donor) separated from
plotting (plot_scorer_orientation_vs_alpha.py) so replotting/restyling
never requires rerunning it. This produces the paper's Figure 3 (and its
appendix grid): the main-results adequacy-over-fluency-preference,
MQM-adherence, and explainability-by-MQM curves.

In the paper's committed setup (--green-family DiagTJ, the default), every
donor s_0 gets exactly THREE families: AF (the single-dial Adequacy-fluency
preference family, mwb.lib.synthetic_scorers.donor_family_additive_mean_diagonal,
on its own DIAGONAL_ADDITIVE_MEAN_DIAL_GRID), T (MQM-adherence, dialed on
AllMQM), and J (Explainability-by-MQM, the Joint family) -- matching the
paper's Figure 3 exactly (three curves: "Adequacy over fluency", "MQM
adherence", "Explainability by MQM"). At every alpha, scores every
same-family dial pair with the weighted meta-metric (--metametric
spa|pa|pearson; mwb.lib.synthetic_scorer_alpha_grid.donor_alpha_dial_grid)
and reduces to orientation_score -- the fraction of pairs where the
metametric prefers the more extreme (further-from-the-real-donor) dial.
Also stores ESS(w*(alpha)) alongside, for plotting next to the orientation
curves. Other --green-family modes (see below) instead build the separate
single-aspect A/B families (the paper's footnoted, excluded construction --
see mwb.lib.synthetic_scorers.donor_family_adequacy_adherence).

Usage: python -m mwb.scripts.compute_scorer_orientation_vs_alpha [--dataset ende21]
           [--n-steps 5] [--step 0.01] [--full-range | --union-grid] [--metametric spa|pa|pearson]
           [--synthesis offset|additive|additive_mean] [--dial-preset linear|geometric]
           [--aspect-donors] [--green-family default|Jneg|DiagTJ] [--dial-step 0.005] [--dial-n 201] [--tag TAG]

--dial-step/--dial-n: resolution of the --green-family DiagTJ T/J grids
(mwb.lib.synthetic_scorers.make_abt_dial_grid/make_abtj_j_dial_grid) --
step*k for k in range(n) sweeps a fixed exponent range, so raising n while
lowering step by the same factor (e.g. step=0.001, n=1001 instead of the
default step=0.005, n=201) only adds resolution within that same range,
useful for a higher-density poster-style render. AF's own dial grid
(DIAGONAL_ADDITIVE_MEAN_DIAL_GRID) is unaffected -- it matches the paper's
Sec. "Evaluation Setup" dial-grid paragraph exactly and isn't a tunable
here. Ignored unless --green-family DiagTJ.

--green-family Jneg (only meaningful with --synthesis additive/additive_mean,
whose own third family is T): overrides the cached third family with J
instead, but a DIFFERENT J than offset's own default -- built from an
EXTRA donor_alpha_dial_grid(..., synthesis='offset', dial_grid=mwb.lib.
synthetic_scorer_orientation.NEG_J_DIAL_GRID) call per donor (A/B still
come from the main --synthesis run; only J is swapped in from this second
call, and only J is kept from it -- its own A/B are discarded). A/B stay
exactly what --synthesis alone would have produced -- this is for overlay
plots that want e.g. additive_mean's A/B next to offset's negative-dial J
instead of additive_mean's own T. This extra J call always uses
NEG_J_DIAL_GRID regardless of --dial-preset (matches offset's own
EXTENDED_OFFSET_DIAL_GRID convention of ignoring --dial-preset entirely).

--green-family DiagTJ (--synthesis must be additive/additive_mean, the
paper's committed/default setup): caches THREE families at once -- AF (the
single-dial Adequacy-fluency family, replacing the separate A/B pair) and
T from the main --synthesis run (T is that run's own third family, kept
this time instead of being discarded), PLUS J from an EXTRA
donor_alpha_dial_grid(..., synthesis='offset') call, only J kept from it.
T and J are forced regardless of --dial-preset, but onto DIFFERENT grids:
the main run uses make_abt_dial_grid(--dial-step, --dial-n) for T, the
extra J call uses make_abtj_j_dial_grid(--dial-step, --dial-n) -- see
those functions' own docstring for why ascending raw dial value is
ascending "goodness" for both (so, unlike NEG_J_DIAL_GRID, neither needs
reordering). AF always uses its own DIAGONAL_ADDITIVE_MEAN_DIAL_GRID,
independent of --dial-step/--dial-n.
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

from mwb.lib.alpha import alpha_0, alpha_min_max, union_alpha_grid, union_alpha_grid_beta
from mwb.lib.consistency import real_scorers, real_systems
from mwb.lib.reweighted_consistency import solve_w_for_datasets
from mwb.lib.synthetic_scorer_alpha_grid import ASPECT_DONORS, alpha_grid_around, donor_alpha_dial_grid
from mwb.lib.synthetic_scorer_orientation import (
    DIAL_GRID, NEG_J_DIAL_GRID, donor_orientation_by_alpha, full_range_alphas, orientation_tag, save_orientation_data,
    union_grid_tag,
)
from mwb.lib.synthetic_scorers import (
    EXTENDED_ADDITIVE_MEAN_DIAL_GRID, GEOM_DIAL_GRID, default_dial_preset,
    make_abt_dial_grid, make_abtj_j_dial_grid,
)
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output', 'data')


def _compute_one_donor(job):
  """One donor's worker-process body for --workers > 1 (ProcessPoolExecutor):
  the same two donor_alpha_dial_grid calls the sequential loop makes below,
  factored out to a module-level function so it's importable/picklable under
  macOS's spawn start method. Donors are independent given the already-
  solved w_by_alpha (read-only, shared by value across processes), so this
  parallelizes with no cross-donor coordination needed. Returns (donor,
  orient_dict, None) on success or (donor, None, skip_reason) on skip --
  mirrors the two SKIPPED cases the sequential loop prints inline."""
  base, donor, systems, w_by_alpha, dial_grid, metametric, synthesis, green_family, j_dial_grid = job
  grids = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=dial_grid,
                                 metametric=metametric, synthesis=synthesis,
                                 adequacy_fluency_diagonal=(green_family == 'DiagTJ'))
  if grids is None:
    return donor, None, 'insufficient segment coverage'
  orient = donor_orientation_by_alpha(grids)
  if green_family in ('Jneg', 'DiagTJ'):
    grids_j = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=j_dial_grid,
                                     metametric=metametric, synthesis='offset')
    if grids_j is None:
      return donor, None, 'insufficient coverage for the offset-J overlay'
    orient['J'] = donor_orientation_by_alpha(grids_j)['J']
  return donor, orient, None


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
                            '[alpha_min, alpha_max] sweep at --step (mwb.lib.alpha.union_alpha_grid) instead '
                            'of --full-range/the alpha_0(D)-centered window. Default: True (the committed '
                            'setup, matching --synthesis additive_mean/--green-family DiagTJ below).')
  parser.add_argument('--beta-grid', action=argparse.BooleanOptionalAction, default=False,
                       help='like --union-grid (the alpha_ij union stays), but the uniform-sweep half is '
                            'evenly spaced in BETA space instead of alpha space (mwb.lib.alpha.'
                            'union_alpha_grid_beta/alpha_grid_for_beta, beta = 1/(1+sqrt(1/alpha-1)) -- see '
                            'plot_scorer_orientation_vs_alpha.py --beta-axis). --step is read as the beta '
                            'stepsize under this mode. Takes priority over --union-grid/--full-range when '
                            'set. Default: False.')
  parser.add_argument('--metametric', choices=['spa', 'pa', 'pearson'], default='spa',
                       help='weighted meta-metric to score each dial/alpha cell with')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help="which mwb.lib.synthetic_scorers dial family to build: 'offset' (the "
                            "original m/e/ebar_k generator), 'additive' (y + dial*aspect), or "
                            "'additive_mean' (default, y + dial*abar_k)")
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended'], default=None,
                       help="'linear': DIAL_GRID, 0/0.1/.../1. 'geometric': GEOM_DIAL_GRID, {2^-i : "
                            'i=0..10} packed toward dial=0 -- additive/additive_mean saturate almost '
                            'immediately on the linear grid, so this resolves the early part of the '
                            "sweep instead (mwb.lib.synthetic_scorers). 'extended': mwb.lib.synthetic_scorers."
                            'EXTENDED_ADDITIVE_MEAN_DIAL_GRID, dial 0..~8 -- additive_mean only, paired '
                            "with donor_family_additive_mean's standardized=True default (injected term "
                            "rescaled to the donor's own between-system spread), where dial=1 is now only "
                            "~1 SD and a much wider reach is needed for a genuinely extreme endpoint. "
                            "Default: mwb.lib.synthetic_scorers.default_dial_preset(synthesis) -- 'geometric' "
                            "for additive/additive_mean, 'linear' for offset -- 'extended' is never a "
                            'default, always explicit.')
  parser.add_argument('--aspect-donors', action=argparse.BooleanOptionalAction, default=False,
                       help="compute only mwb.lib.synthetic_scorer_alpha_grid.ASPECT_DONORS ('AdequacyMQM', "
                            "'FluencyMQM', 'AllMQM' -- the gold aspect signals themselves faking to be a "
                            "donor scorer) instead of the real_scorers-screened candidate list")
  parser.add_argument('--green-family', choices=['default', 'Jneg', 'DiagTJ'], default='DiagTJ',
                       help="'default': third family is whatever --synthesis produces (T for additive/"
                            "additive_mean, J for offset), A/B are the separate single-aspect families. "
                            "'Jneg': override the third family with offset's J built on NEG_J_DIAL_GRID "
                            "([0, -2] by -0.1) instead, A/B still separate. 'DiagTJ' (default, the paper's "
                            "committed setup): cache the single-dial Adequacy-fluency family (AF, replacing "
                            "A/B) together with T and J -- AF on DIAGONAL_ADDITIVE_MEAN_DIAL_GRID, T on "
                            'make_abt_dial_grid(--dial-step, --dial-n), J on '
                            'make_abtj_j_dial_grid(--dial-step, --dial-n) -- see module docstring for both.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--workers', type=int, default=1,
                       help='parallel worker processes (ProcessPoolExecutor), applied to BOTH expensive '
                            'steps: the solve_w_for_datasets alpha-solving pass (mwb.lib.reweighted_consistency, '
                            'one process per (dataset, alpha) -- often the larger cost, since solve_w_exact\'s '
                            'exhaustive-enumeration fallback can dominate on dense/near-breakpoint alpha '
                            'grids) and the per-donor loop below (_compute_one_donor, one process per donor). '
                            'Both are independent-job loops with no cross-job coordination needed. Default: 1 '
                            '(sequential, the original behavior -- opt-in only, since ProcessPoolExecutor '
                            'changes stderr interleaving/ordering).')
  parser.add_argument('--dial-step', type=float, default=0.005,
                       help='resolution of the --green-family DiagTJ T/J grids -- see module docstring. '
                            'Default: 0.005 (201 points over the committed [0,1] exponent range).')
  parser.add_argument('--dial-n', type=int, default=201,
                       help='number of points in the --green-family DiagTJ T/J grids -- see --dial-step.')
  args = parser.parse_args()
  base = args.dataset
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)
  dial_grid = {
      'linear': DIAL_GRID, 'geometric': GEOM_DIAL_GRID, 'extended': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  }[args.dial_preset]
  if args.green_family == 'DiagTJ':
    dial_grid = make_abt_dial_grid(args.dial_step, args.dial_n)
    args.dial_preset = 'symmetric'  # so the auto tag/cache metadata reflect the actual grid, not --dial-preset

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  K = len(systems)

  df = load_system_scores(base, root=ROOT)
  a_D, b_D = df.loc[systems, 'a'].values, df.loc[systems, 'b'].values
  center_alpha = alpha_0(a_D, b_D)
  alpha_lo, alpha_hi = alpha_min_max(a_D, b_D)
  if args.beta_grid:
    alphas = union_alpha_grid_beta(a_D, b_D, beta_stepsize=args.step)
  elif args.union_grid:
    alphas = union_alpha_grid(a_D, b_D, step=args.step)
  elif args.full_range:
    alphas = full_range_alphas(alpha_lo, alpha_hi, args.step)
  else:
    alphas = alpha_grid_around(center_alpha, alpha_lo, alpha_hi, args.n_steps, args.step)
  print(f'{base}: K={K} systems, alpha_0(D)={center_alpha:.4f}, alpha grid ({len(alphas)} pts): '
        f'{[round(a, 4) for a in alphas]}', file=sys.stderr)

  w_cache = solve_w_for_datasets([base], alphas, root=ROOT, systems_by_dataset={base: systems},
                                  workers=args.workers, progress=True)
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
  j_dial_grid = (make_abtj_j_dial_grid(args.dial_step, args.dial_n) if args.green_family == 'DiagTJ' else NEG_J_DIAL_GRID) \
      if args.green_family in ('Jneg', 'DiagTJ') else None
  if args.workers > 1:
    # Parallel path: donors are independent given the already-solved
    # w_by_alpha, so _compute_one_donor's two donor_alpha_dial_grid calls
    # (main + the DiagTJ/Jneg offset-J extra) run one donor per worker
    # process. Only 10s-100s of donors here (real_scorers-screened), so --
    # unlike a bulk many-thousands-of-jobs pipeline that needs bounded-
    # in-flight submission -- submitting every job at once is safe.
    jobs = [(base, donor, systems, w_by_alpha, dial_grid, args.metametric, args.synthesis, args.green_family,
             j_dial_grid) for donor in candidates]
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
      for donor, orient, skip_reason in ex.map(_compute_one_donor, jobs):
        done += 1
        if orient is None:
          print(f'[{done}/{n}] SKIPPED {donor} ({skip_reason})', file=sys.stderr)
          continue
        orient_by_donor[donor] = orient
        elapsed = time.time() - t0
        eta = elapsed / done * (n - done)
        print(f'[{done}/{n}] computed {donor} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
  else:
    for i, donor in enumerate(candidates, 1):
      grids = donor_alpha_dial_grid(base, donor, systems, w_by_alpha, root=ROOT, dial_grid=dial_grid,
                                     metametric=args.metametric, synthesis=args.synthesis,
                                     adequacy_fluency_diagonal=(args.green_family == 'DiagTJ'))
      if grids is None:
        print(f'[{i}/{n}] SKIPPED {donor} (insufficient segment coverage)', file=sys.stderr)
        continue
      orient = donor_orientation_by_alpha(grids)
      if args.green_family in ('Jneg', 'DiagTJ'):
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
  if args.green_family == 'DiagTJ':
    # The paper's actual committed setup: AF (the single-dial Adequacy-
    # fluency preference family) replaces the separate A/B pair. Both T
    # (the main --synthesis run's own third family) and J (the extra
    # offset call) are kept -- see module docstring.
    extra = {'AF': np.array([orient_by_donor[d]['AF'] for d in donors]),
             'T': np.array([orient_by_donor[d]['T'] for d in donors]),
             'J': np.array([orient_by_donor[d]['J'] for d in donors])}
  else:
    # 'J' (Joint) for synthesis='offset' OR --green-family Jneg (which
    # overrides whatever --synthesis's own third family would have been),
    # 'T' (AllMQM) otherwise (additive/additive_mean's own third family) --
    # matches whichever third family ended up in orient_by_donor, never both.
    third_label = 'J' if (args.synthesis == 'offset' or args.green_family == 'Jneg') else 'T'
    extra = {'A': np.array([orient_by_donor[d]['A'] for d in donors]),
             'B': np.array([orient_by_donor[d]['B'] for d in donors]),
             third_label: np.array([orient_by_donor[d][third_label] for d in donors])}

  if args.tag:
    tag = args.tag
  elif args.union_grid:
    tag = union_grid_tag(base, args.step, args.metametric, args.synthesis, args.dial_preset)
  else:
    tag = orientation_tag(base, args.n_steps, args.step, args.full_range, args.metametric, args.synthesis,
                           args.dial_preset)
  if args.aspect_donors and not args.tag:
    tag += '_aspectdonors'
  if args.green_family in ('Jneg', 'DiagTJ') and not args.tag:
    tag += f'_{args.green_family}'
  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  save_orientation_data(
      out_path, dataset=base, alphas=alphas, center_alpha=center_alpha, K=K, donors=donors,
      ess=ess_by_alpha, dial_grid=dial_grid, metametric=args.metametric, synthesis=args.synthesis,
      dial_preset=args.dial_preset, **extra,
  )
  print(f'Wrote {out_path}', file=sys.stderr)