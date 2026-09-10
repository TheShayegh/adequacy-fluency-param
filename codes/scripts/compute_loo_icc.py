"""For each dataset, report ICC(C,1) (Shrout & Fleiss 1979) across scorer
rankings produced by all leave-p-out (LpO) meta-evaluations of that
dataset: for a pool D of K real systems (lib.consistency.real_systems),
every size-p combination of dropped systems gives one LpO subset D minus
that combo (C(K,p) of them total), each scored ONCE and treated as one
RATER; scorers -- restricted to lib.consistency.real_scorers(D), the
project's canonical scorer screen -- are the objects. ICC(C,1) is then the
generalizability-theory reliability coefficient (lib.icc.icc_c1) of these
C(K,p) raters' scorer rankings agreeing with each other. p=1 (--leave-out
1, the default) is plain leave-one-out.

Three conditions per dataset, all reusing the SAME C(K,p) LpO subsets:
  natural            -- every subset scored at uniform weight (no reweighting).
  reweighted_alpha0   -- every subset reweighted (lib.reweight_exact.solve_w_exact)
                         to the SAME fixed target alpha_0(D): the FULL,
                         un-LpO'd pool's own natural adequacy/fluency balance
                         (lib.alpha.alpha_0).
  reweighted_med_aij  -- every subset reweighted to the SAME fixed target
                         median(alpha_ij(D)): the median, over every pair of
                         systems i<j in the FULL pool D, of that pair's
                         pairwise balance (lib.alpha.pairwise_alphas).
Both reweighted targets are computed ONCE from the full pool D (not
per-subset -- an LpO subset has no privileged natural target of its own
here, contrast type-8's upsampling design, which targets alpha_0(D') of
the smaller pool; see notes/type7_type8_robustness_metrics.md) and then
held fixed across all raters, same "one shared target for every rater"
convention as lib.subsampling_stability.generalizability_stability's own
reweighted condition. An LpO subset for which a target is infeasible
(Corollary 2: outside that subset's own reachable alpha range) is dropped
from that condition's rater set only (n_infeasible), not from the other
two conditions.

All the actual math/IO lives in lib.subsampling_stability.
leave_p_out_generalizability -- this script is just the per-dataset loop
plus reporting. See that function's docstring for exactly how it differs
from generalizability_stability's random-subsample raters (exhaustive, no
seed, explicit per-dropped-combo labeling) and from type-8's upsampling
design -- here the LpO subset itself IS the rater's meta-evaluation,
nothing is added back.

Datasets with K - p < 2 (too few systems left in a subset) or C(K,p) < 2
(too few raters) are skipped and reported as N/A rather than raising --
this always includes ende20/zhen20 (unsupported by wmt_official) and
ende23 (manually excluded, lib.consistency.MANUALLY_EXCLUDED_DATASETS),
which have K=0.

STREAMED (project convention, see notes on progress logging): the header
is written immediately, and each dataset's row is appended to the output
file as soon as that dataset finishes -- not batched to the end -- with
per-dataset elapsed/ETA progress on stderr. C(K,p) grows fast with p
(e.g. C(16,2)=120 vs C(16,1)=16), so larger p costs noticeably more wall
time per dataset -- mind this before sweeping p much past 2.

Usage: python codes/scripts/compute_loo_icc.py
           [--datasets ende22,zhen22,...] [--metametric spa]
           [--leave-out 1] [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
(default datasets: every dataset in lib.dataset_dirs.DATASET_DIRS)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0 as alpha0_of, pairwise_alphas
from lib.consistency import real_systems
from lib.dataset_dirs import DATASET_DIRS
from lib.subsampling_stability import leave_p_out_generalizability
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

HEADER = ('| dataset | K | raters (C(K,p)) | ICC natural | alpha_0(D) | '
          'ICC reweighted (alpha_0(D)) | median alpha_ij(D) | '
          'ICC reweighted (median alpha_ij(D)) |\n'
          '|---|---:|---:|---:|---:|---:|---:|---:|\n')


def icc_cell(result: dict) -> str:
  icc = result['icc']
  icc_str = f"{icc:.4f}" if icc == icc else 'N/A'
  if result['n_infeasible'] > 0:
    icc_str += f" ({result['n_raters_used']}/{result['n_raters_total']} raters)"
  return icc_str


def format_row(dataset: str, K_full: int, n_raters: int | None, natural: dict | None,
                a0_D: float | None, rw_a0: dict | None,
                med_aij_D: float | None, rw_med: dict | None) -> str:
  if natural is None:
    return f'| {dataset} | {K_full} | -- | N/A | -- | -- | -- | -- |\n'
  return (f"| {dataset} | {natural['K']} | {n_raters} | {icc_cell(natural)} | {a0_D:.4f} | "
          f"{icc_cell(rw_a0)} | {med_aij_D:.4f} | {icc_cell(rw_med)} |\n")


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--datasets', type=str, default=None,
                  help='comma-separated dataset names (default: every dataset in DATASET_DIRS)')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--leave-out', type=int, default=1,
                  help='p: number of systems dropped per LpO rater (default 1 = plain LOO)')
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True,
                  help='passed through to real_systems/real_scorers (project default: on)')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  m = args.metametric
  leave_out = args.leave_out
  datasets = ([d.strip() for d in args.datasets.split(',')] if args.datasets
              else list(DATASET_DIRS.keys()))
  tag = args.tag or m
  p_suffix = '' if leave_out == 1 else f'_p{leave_out}'

  out_path = os.path.join(ARTIFACTS_DIR, f'loo_icc_reweighted{p_suffix}_{tag}.md')
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  with open(out_path, 'w') as f:
    f.write(f'# Leave-{leave_out}-out ICC(C,1) across scorer rankings: natural vs. reweighted '
            f'(metametric={m})\n\n')
    f.write(f'Each row: C(K,{leave_out}) LpO subsets of the dataset (one per dropped '
            f'{leave_out}-system combo) are each scored once under three conditions -- natural '
            "(uniform weight), reweighted to the FULL dataset's own alpha_0(D), and reweighted "
            "to the median of the full dataset's pairwise alpha_ij -- and treated as raters; "
            'ICC(C,1) measures how consistently they rank the real_scorers-screened scorers. '
            '"N/total raters" is shown when a target was infeasible (outside a subset\'s own '
            'reachable alpha range) for some subset.\n\n')
    f.write(HEADER)

  footnotes = []
  print(f'{len(datasets)} datasets, metametric={m}, leave-{leave_out}-out', file=sys.stderr)
  t_start = time.time()
  for i, dataset in enumerate(datasets, 1):
    t0 = time.time()
    full_systems = real_systems(dataset, root=ROOT, exclude_outliers=args.exclude_outliers)
    K_full = len(full_systems)
    if K_full - leave_out < 2:
      natural = a0_D = rw_a0 = med_aij_D = rw_med = n_raters = None
      print(f'[{i}/{len(datasets)}] {dataset}: K={K_full}, skipped (N/A)', file=sys.stderr)
    else:
      df = load_system_scores(dataset, root=ROOT)
      a_D, b_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'b'].values
      a0_D = alpha0_of(a_D, b_D)
      med_aij_D = float(np.median([v for _, _, v in pairwise_alphas(a_D, b_D)]))

      natural = leave_p_out_generalizability(dataset, m, p=leave_out, root=ROOT,
                                              exclude_outliers=args.exclude_outliers)
      rw_a0 = leave_p_out_generalizability(dataset, m, p=leave_out, root=ROOT,
                                            exclude_outliers=args.exclude_outliers,
                                            target_alpha=a0_D)
      rw_med = leave_p_out_generalizability(dataset, m, p=leave_out, root=ROOT,
                                             exclude_outliers=args.exclude_outliers,
                                             target_alpha=med_aij_D)
      n_raters = natural['n_raters_total']
      elapsed = time.time() - t0
      mean_elapsed = (time.time() - t_start) / i
      eta = mean_elapsed * (len(datasets) - i)
      print(f"[{i}/{len(datasets)}] {dataset}: K={natural['K']}, raters={n_raters}, "
            f"ICC natural={natural['icc']:.4f}, "
            f"ICC reweighted(a0={a0_D:.4f})={rw_a0['icc']:.4f} "
            f"({rw_a0['n_raters_used']}/{n_raters} raters), "
            f"ICC reweighted(med_aij={med_aij_D:.4f})={rw_med['icc']:.4f} "
            f"({rw_med['n_raters_used']}/{n_raters} raters) "
            f'({elapsed:.1f}s, ETA {eta:.0f}s)', file=sys.stderr)
      for label, r in (('natural', natural), (f'reweighted(a0={a0_D:.4f})', rw_a0),
                        (f'reweighted(med_aij={med_aij_D:.4f})', rw_med)):
        if r['scorers_dropped']:
          footnotes.append(f"{dataset} [{label}]: scorers dropped (missing on some used LpO "
                            f"subset) = {r['scorers_dropped']}")

    with open(out_path, 'a') as f:
      f.write(format_row(dataset, K_full, n_raters, natural, a0_D, rw_a0, med_aij_D, rw_med))

  if footnotes:
    with open(out_path, 'a') as f:
      f.write('\n')
      for line in footnotes:
        f.write(f'- {line}\n')

  print(f'\nWrote {out_path}', file=sys.stderr)
