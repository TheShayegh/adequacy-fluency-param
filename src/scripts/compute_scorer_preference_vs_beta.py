"""Computes <adequacy|fluency>_preference(metametric_beta, s_0)
(src.lib.synthetic_scorer_preference) for every real base scorer s_0 in a
dataset, across a beta grid, and caches the result to disk
(output/preference_vs_beta_<tag>.npz) -- the expensive step
(src.lib.reweight_exact solves, plus SPA permutation tests when
metametric=spa, one base_scorer_beta_dial_grid call per base scorer) separated from
plotting (plot_scorer_preference_vs_beta.py) so replotting/restyling
never requires rerunning it. This produces the paper's Figure 3 (and its
appendix grid): the main-results adequacy-over-fluency-preference,
MQM-adherence, and explainability-by-MQM curves.

In the paper's committed setup (--family-set AFTJ, the default), every
base scorer s_0 gets exactly THREE families: AF (the single-dial Adequacy-fluency
preference family, src.lib.synthetic_scorers.base_scorer_family_additive_mean_af,
on its own AF_ADDITIVE_MEAN_DIAL_GRID), T (MQM-adherence, dialed on
AllMQM), and J (Explainability-by-MQM, the Joint family) -- matching the
paper's Figure 3 exactly (three curves: "Adequacy over fluency", "MQM
adherence", "Explainability by MQM"). At every beta, scores every
same-family dial pair with the weighted meta-metric (--metametric
spa|pa|pearson; src.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid)
and reduces to preference_score -- the fraction of pairs where the
metametric prefers the more extreme (further-from-the-real-base scorer) dial.
Also stores ESS(w*(beta)) alongside, for plotting next to the preference
curves. Other --family-set modes (see below) instead build the separate
single-aspect A/F families (the paper's footnoted, excluded construction --
see src.lib.synthetic_scorers.base_scorer_family_adequacy_adherence).

Usage: python -m src.scripts.compute_scorer_preference_vs_beta [--dataset ende21]
           [--n-steps 5] [--step 0.01] [--full-range | --union-grid] [--metametric spa|pa|pearson]
           [--synthesis offset|additive|additive_mean] [--dial-preset linear|geometric]
           [--aspect-base-scorers] [--family-set default|Jneg|AFTJ] [--dial-step 0.005] [--dial-n 201] [--tag TAG]

--dial-step/--dial-n: resolution of the --family-set AFTJ T/J grids
(src.lib.synthetic_scorers.make_aft_dial_grid/make_aftj_j_dial_grid) --
step*k for k in range(n) sweeps a fixed exponent range, so raising n while
lowering step by the same factor (e.g. step=0.001, n=1001 instead of the
default step=0.005, n=201) only adds resolution within that same range,
useful for a higher-density poster-style render. AF's own dial grid
(AF_ADDITIVE_MEAN_DIAL_GRID) is unaffected -- it matches the paper's
Sec. "Evaluation Setup" dial-grid paragraph exactly and isn't a tunable
here. Ignored unless --family-set AFTJ.

--family-set Jneg (only meaningful with --synthesis additive/additive_mean,
whose own third family is T): overrides the cached third family with J
instead, but a DIFFERENT J than offset's own default -- built from an
EXTRA base_scorer_beta_dial_grid(..., synthesis='offset', dial_grid=src.lib.
synthetic_scorer_preference.NEG_J_DIAL_GRID) call per base scorer (A/F still
come from the main --synthesis run; only J is swapped in from this second
call, and only J is kept from it -- its own A/F are discarded). A/F stay
exactly what --synthesis alone would have produced -- this is for overlay
plots that want e.g. additive_mean's A/F next to offset's negative-dial J
instead of additive_mean's own T. This extra J call always uses
NEG_J_DIAL_GRID regardless of --dial-preset (matches offset's own
EXTENDED_OFFSET_DIAL_GRID convention of ignoring --dial-preset entirely).

--family-set AFTJ (--synthesis must be additive/additive_mean, the
paper's committed/default setup): caches THREE families at once -- AF (the
single-dial Adequacy-fluency family, replacing the separate A/F pair) and
T from the main --synthesis run (T is that run's own third family, kept
this time instead of being discarded), PLUS J from an EXTRA
base_scorer_beta_dial_grid(..., synthesis='offset') call, only J kept from it.
T and J are forced regardless of --dial-preset, but onto DIFFERENT grids:
the main run uses make_aft_dial_grid(--dial-step, --dial-n) for T, the
extra J call uses make_aftj_j_dial_grid(--dial-step, --dial-n) -- see
those functions' own docstring for why ascending raw dial value is
ascending "goodness" for both (so, unlike NEG_J_DIAL_GRID, neither needs
reordering). AF always uses its own AF_ADDITIVE_MEAN_DIAL_GRID,
independent of --dial-step/--dial-n.
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

from src.lib.beta import beta_0, beta_min_max, union_beta_grid, union_beta_grid_beta_std
from src.lib.consistency import real_scorers, real_systems
from src.lib.reweighted_consistency import solve_w_for_datasets
from src.lib.synthetic_scorer_beta_grid import ASPECT_BASE_SCORERS, beta_grid_around, base_scorer_beta_dial_grid
from src.lib.synthetic_scorer_preference import (
    DIAL_GRID, NEG_J_DIAL_GRID, base_scorer_preference_by_beta, full_range_betas, resolve_tag, save_preference_data,
)
from src.lib.synthetic_scorers import (
    EXTENDED_ADDITIVE_MEAN_DIAL_GRID, GEOM_DIAL_GRID, default_dial_preset,
    make_aft_dial_grid, make_aftj_j_dial_grid,
)
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'output')


def _compute_one_base_scorer(job):
  """One base scorer's worker-process body for --workers > 1 (ProcessPoolExecutor):
  the same two base_scorer_beta_dial_grid calls the sequential loop makes below,
  factored out to a module-level function so it's importable/picklable under
  macOS's spawn start method. Base scorers are independent given the already-
  solved w_by_beta (read-only, shared by value across processes), so this
  parallelizes with no cross-base scorer coordination needed. Returns (base scorer,
  pref_dict, None) on success or (base scorer, None, skip_reason) on skip --
  mirrors the two SKIPPED cases the sequential loop prints inline."""
  base, base_scorer, systems, w_by_beta, dial_grid, metametric, synthesis, family_set, j_dial_grid = job
  grids = base_scorer_beta_dial_grid(base, base_scorer, systems, w_by_beta, root=ROOT, dial_grid=dial_grid,
                                metametric=metametric, synthesis=synthesis,
                                use_af_family=(family_set == 'AFTJ'))
  if grids is None:
    return base_scorer, None, 'insufficient segment coverage'
  pref = base_scorer_preference_by_beta(grids)
  if family_set in ('Jneg', 'AFTJ'):
    grids_j = base_scorer_beta_dial_grid(base, base_scorer, systems, w_by_beta, root=ROOT, dial_grid=j_dial_grid,
                                    metametric=metametric, synthesis='offset')
    if grids_j is None:
      return base_scorer, None, 'insufficient coverage for the offset-J overlay'
    pref['J'] = base_scorer_preference_by_beta(grids_j)['J']
  return base_scorer, pref, None


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='ende21')
  parser.add_argument('--n-steps', type=int, default=5)
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--full-range', action=argparse.BooleanOptionalAction, default=False,
                       help='sweep the entire reachable [beta_min, beta_max] at --step instead of '
                            'the beta_0(D)-centered +/- n_steps*step window')
  parser.add_argument('--union-grid', action=argparse.BooleanOptionalAction, default=True,
                       help='use the union of every pairwise beta_ij of D with a plain uniform '
                            '[beta_min, beta_max] sweep at --step (src.lib.beta.union_beta_grid) instead '
                            'of --full-range/the beta_0(D)-centered window. Default: True (the committed '
                            'setup, matching --synthesis additive_mean/--family-set AFTJ below).')
  parser.add_argument('--beta-std-grid', action=argparse.BooleanOptionalAction, default=False,
                       help='like --union-grid (the beta_ij union stays), but the uniform-sweep half is '
                            'evenly spaced in BETA_STD space instead of beta space (src.lib.beta.'
                            'union_beta_grid_beta_std/beta_grid_for_beta_std, beta_std = 1/(1+sqrt(1/beta-1)) '
                            '-- see plot_scorer_preference_vs_beta.py --beta-std-axis). --step is read as the '
                            'beta_std stepsize under this mode. Takes priority over --union-grid/--full-range '
                            'when set. Default: False.')
  parser.add_argument('--metametric', choices=['spa', 'pa', 'pearson'], default='spa',
                       help='weighted meta-metric to score each dial/beta cell with')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='additive_mean',
                       help="which src.lib.synthetic_scorers dial family to build: 'offset' (the "
                            "original m/e/ebar_k generator), 'additive' (y + dial*aspect), or "
                            "'additive_mean' (default, y + dial*abar_k)")
  parser.add_argument('--dial-preset', choices=['linear', 'geometric', 'extended'], default=None,
                       help="'linear': DIAL_GRID, 0/0.1/.../1. 'geometric': GEOM_DIAL_GRID, {2^-i : "
                            'i=0..10} packed toward dial=0 -- additive/additive_mean saturate almost '
                            'immediately on the linear grid, so this resolves the early part of the '
                            "sweep instead (src.lib.synthetic_scorers). 'extended': src.lib.synthetic_scorers."
                            'EXTENDED_ADDITIVE_MEAN_DIAL_GRID, dial 0..~8 -- additive_mean only, paired '
                            "with base_scorer_family_additive_mean's standardized=True default (injected term "
                            "rescaled to the base scorer's own between-system spread), where dial=1 is now only "
                            "~1 SD and a much wider reach is needed for a genuinely extreme endpoint. "
                            "Default: src.lib.synthetic_scorers.default_dial_preset(synthesis) -- 'geometric' "
                            "for additive/additive_mean, 'linear' for offset -- 'extended' is never a "
                            'default, always explicit.')
  parser.add_argument('--aspect-base-scorers', action=argparse.BooleanOptionalAction, default=False,
                       help="compute only src.lib.synthetic_scorer_beta_grid.ASPECT_BASE_SCORERS ('AdequacyMQM', "
                            "'FluencyMQM', 'AllMQM' -- the gold aspect signals themselves faking to be a "
                            "base scorer) instead of the real_scorers-screened candidate list")
  parser.add_argument('--family-set', choices=['default', 'Jneg', 'AFTJ'], default='AFTJ',
                       help="'default': third family is whatever --synthesis produces (T for additive/"
                            "additive_mean, J for offset), A/F are the separate single-aspect families. "
                            "'Jneg': override the third family with offset's J built on NEG_J_DIAL_GRID "
                            "([0, -2] by -0.1) instead, A/F still separate. 'AFTJ' (default, the paper's "
                            "committed setup): cache the single-dial Adequacy-fluency family (AF, replacing "
                            "A/F) together with T and J -- AF on AF_ADDITIVE_MEAN_DIAL_GRID, T on "
                            'make_aft_dial_grid(--dial-step, --dial-n), J on '
                            'make_aftj_j_dial_grid(--dial-step, --dial-n) -- see module docstring for both.')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--workers', type=int, default=1,
                       help='parallel worker processes (ProcessPoolExecutor), applied to BOTH expensive '
                            'steps: the solve_w_for_datasets beta-solving pass (src.lib.reweighted_consistency, '
                            'one process per (dataset, beta) -- often the larger cost, since solve_w_exact\'s '
                            'exhaustive-enumeration fallback can dominate on dense/near-breakpoint beta '
                            'grids) and the per-base scorer loop below (_compute_one_base_scorer, one process per base scorer). '
                            'Both are independent-job loops with no cross-job coordination needed. Default: 1 '
                            '(sequential, the original behavior -- opt-in only, since ProcessPoolExecutor '
                            'changes stderr interleaving/ordering).')
  parser.add_argument('--dial-step', type=float, default=0.005,
                       help='resolution of the --family-set AFTJ T/J grids -- see module docstring. '
                            'Default: 0.005 (201 points over the committed [0,1] exponent range).')
  parser.add_argument('--dial-n', type=int, default=201,
                       help='number of points in the --family-set AFTJ T/J grids -- see --dial-step.')
  args = parser.parse_args()
  base = args.dataset
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)
  dial_grid = {
      'linear': DIAL_GRID, 'geometric': GEOM_DIAL_GRID, 'extended': EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  }[args.dial_preset]
  if args.family_set == 'AFTJ':
    dial_grid = make_aft_dial_grid(args.dial_step, args.dial_n)
    args.dial_preset = 'symmetric'  # so the auto tag/cache metadata reflect the actual grid, not --dial-preset

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if not systems:
    sys.exit(f'{base}: 0 real systems, nothing to compute')
  K = len(systems)

  df = load_system_scores(base, root=ROOT)
  a_D, f_D = df.loc[systems, 'a'].values, df.loc[systems, 'f'].values
  center_beta = beta_0(a_D, f_D)
  beta_lo, beta_hi = beta_min_max(a_D, f_D)
  if args.beta_std_grid:
    betas = union_beta_grid_beta_std(a_D, f_D, beta_std_stepsize=args.step)
  elif args.union_grid:
    betas = union_beta_grid(a_D, f_D, step=args.step)
  elif args.full_range:
    betas = full_range_betas(beta_lo, beta_hi, args.step)
  else:
    betas = beta_grid_around(center_beta, beta_lo, beta_hi, args.n_steps, args.step)
  print(f'{base}: K={K} systems, beta_0(D)={center_beta:.4f}, beta grid ({len(betas)} pts): '
        f'{[round(b, 4) for b in betas]}', file=sys.stderr)

  w_cache = solve_w_for_datasets([base], betas, root=ROOT, systems_by_dataset={base: systems},
                                  workers=args.workers, progress=True)
  w_by_beta = {b: w_cache[(base, b)].w for b in betas}
  ess_by_beta = np.array([w_cache[(base, b)].ess for b in betas])
  print(f'{base}: ESS(beta) range [{ess_by_beta.min():.2f}, {ess_by_beta.max():.2f}] of [1, {K}]',
        file=sys.stderr)

  if args.aspect_base_scorers:
    candidates = list(ASPECT_BASE_SCORERS)
  else:
    # real_scorers, not the raw load_scorer_sys_scores candidate list: the
    # project's canonical scorer screen (drops zero-variance/sentinel
    # scorers and deduplicates byte-identical WMT submissions), so degenerate
    # or double-counted base scorers don't quietly bias the dataset-level
    # preference average or the capability screen downstream.
    candidates = real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True)
  print(f'{base}: {len(candidates)} candidate base_scorers, dial grid ({args.dial_preset}) {dial_grid}', file=sys.stderr)

  n = len(candidates)
  t0 = time.time()
  pref_by_base_scorer = {}
  j_dial_grid = (make_aftj_j_dial_grid(args.dial_step, args.dial_n) if args.family_set == 'AFTJ' else NEG_J_DIAL_GRID) \
      if args.family_set in ('Jneg', 'AFTJ') else None
  if args.workers > 1:
    # Parallel path: base scorers are independent given the already-solved
    # w_by_beta, so _compute_one_base_scorer's two base_scorer_beta_dial_grid calls
    # (main + the AFTJ/Jneg offset-J extra) run one base scorer per worker
    # process. Only 10s-100s of base scorers here (real_scorers-screened), so --
    # unlike a bulk many-thousands-of-jobs pipeline that needs bounded-
    # in-flight submission -- submitting every job at once is safe.
    jobs = [(base, base_scorer, systems, w_by_beta, dial_grid, args.metametric, args.synthesis, args.family_set,
             j_dial_grid) for base_scorer in candidates]
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
      for base_scorer, pref, skip_reason in ex.map(_compute_one_base_scorer, jobs):
        done += 1
        if pref is None:
          print(f'[{done}/{n}] SKIPPED {base_scorer} ({skip_reason})', file=sys.stderr)
          continue
        pref_by_base_scorer[base_scorer] = pref
        elapsed = time.time() - t0
        eta = elapsed / done * (n - done)
        print(f'[{done}/{n}] computed {base_scorer} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
  else:
    for i, base_scorer in enumerate(candidates, 1):
      grids = base_scorer_beta_dial_grid(base, base_scorer, systems, w_by_beta, root=ROOT, dial_grid=dial_grid,
                                    metametric=args.metametric, synthesis=args.synthesis,
                                    use_af_family=(args.family_set == 'AFTJ'))
      if grids is None:
        print(f'[{i}/{n}] SKIPPED {base_scorer} (insufficient segment coverage)', file=sys.stderr)
        continue
      pref = base_scorer_preference_by_beta(grids)
      if args.family_set in ('Jneg', 'AFTJ'):
        grids_j = base_scorer_beta_dial_grid(base, base_scorer, systems, w_by_beta, root=ROOT, dial_grid=j_dial_grid,
                                        metametric=args.metametric, synthesis='offset')
        if grids_j is None:
          print(f'[{i}/{n}] SKIPPED {base_scorer} (insufficient coverage for the offset-J overlay)', file=sys.stderr)
          continue
        pref['J'] = base_scorer_preference_by_beta(grids_j)['J']
      pref_by_base_scorer[base_scorer] = pref
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      print(f'[{i}/{n}] computed {base_scorer} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  print(f'{base}: {len(pref_by_base_scorer)}/{n} base_scorers used', file=sys.stderr)

  base_scorers = sorted(pref_by_base_scorer)
  if args.family_set == 'AFTJ':
    # The paper's actual committed setup: AF (the single-dial Adequacy-
    # fluency preference family) replaces the separate A/F pair. Both T
    # (the main --synthesis run's own third family) and J (the extra
    # offset call) are kept -- see module docstring.
    extra = {'AF': np.array([pref_by_base_scorer[d]['AF'] for d in base_scorers]),
             'T': np.array([pref_by_base_scorer[d]['T'] for d in base_scorers]),
             'J': np.array([pref_by_base_scorer[d]['J'] for d in base_scorers])}
  else:
    # 'J' (Joint) for synthesis='offset' OR --family-set Jneg (which
    # overrides whatever --synthesis's own third family would have been),
    # 'T' (AllMQM) otherwise (additive/additive_mean's own third family) --
    # matches whichever third family ended up in pref_by_base_scorer, never both.
    third_label = 'J' if (args.synthesis == 'offset' or args.family_set == 'Jneg') else 'T'
    extra = {'A': np.array([pref_by_base_scorer[d]['A'] for d in base_scorers]),
             'F': np.array([pref_by_base_scorer[d]['F'] for d in base_scorers]),
             third_label: np.array([pref_by_base_scorer[d][third_label] for d in base_scorers])}

  tag = resolve_tag(args, base)
  os.makedirs(DATA_DIR, exist_ok=True)
  out_path = os.path.join(DATA_DIR, f'preference_vs_beta_{tag}.npz')
  save_preference_data(
      out_path, dataset=base, betas=betas, center_beta=center_beta, K=K, base_scorers=base_scorers,
      ess=ess_by_beta, dial_grid=dial_grid, metametric=args.metametric, synthesis=args.synthesis,
      dial_preset=args.dial_preset, **extra,
  )
  print(f'Wrote {out_path}', file=sys.stderr)
