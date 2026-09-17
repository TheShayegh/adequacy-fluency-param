"""For each dataset, report Kendall's tau-a (src.lib.consistency.pool_weighted_tau)
pooled across all leave-p-out (LpO) meta-evaluations of that dataset:
for a pool D of K real systems (src.lib.consistency.real_systems), every
size-p combination of dropped systems gives one LpO subset D minus that
combo (C(K,p) of them total), each scored ONCE and treated as one RATER;
scorers -- restricted to src.lib.consistency.real_scorers(D), the project's
canonical scorer screen -- are the objects. Every pair of raters' scorer
rankings is compared (concordant/discordant counts, tau-a convention: ties
drop from the numerator but still count in the denominator), summed across
ALL C(n_raters,2) rater-pairs x scorer-pairs, and tau is computed ONCE from
those pooled totals, not a plain average of per-pair taus. p=1
(--leave-out 1, the default) is plain leave-one-out (LOO). This produces
the paper's Table "dataset_tau" (Appendix, Sec. "Does Controlling
Adequacy--Fluency Balance Stabilize Meta-Evaluation?").

NATURAL (uniform weight) by default -- this does NOT also compute
reweighted-to-beta_0(D)/reweighted-to-median-beta_ij(D) conditions unless
--reweighted is passed: "weighted" in pool_weighted_tau refers to weighting
each rater-pair's contribution to the pooled tau by its own scorer-pair
count (C(n_common,2)) -- an intrinsic part of the tau-a pooling itself,
nothing to do with src.lib.reweight_exact's system-level beta reweighting.
--reweighted tests whether fixing every LpO subset's balance to the full
dataset's own natural beta_0(D) (or its median pairwise beta_ij(D))
stabilizes scorer rankings -- the question behind Figure 6.

All the actual math/IO lives in src.lib.subsampling_stability.leave_p_out_tau
-- this script is just the per-dataset loop plus reporting. pool_weighted_tau
does NOT need a fully-crossed matrix -- a scorer missing from one LpO
subset still contributes to every rater-pair it IS present in, rather than
being dropped globally.

Datasets with K - p < 2 or C(K,p) < 2 are skipped and reported as N/A --
this always includes ende20/zhen20 (unsupported by wmt_official) and
ende23 (manually excluded, src.lib.consistency.MANUALLY_EXCLUDED_DATASETS),
which have K=0.

STREAMED: header written immediately, each dataset's row appended as soon
as that dataset finishes, with per-dataset elapsed/ETA progress on stderr.

Usage: python -m src.scripts.compute_loo_tau
           [--datasets ende22,zhen22,...] [--metametric spa]
           [--leave-out 1] [--reweighted]
           [--exclude-outliers | --no-exclude-outliers] [--tag TAG]
(default datasets: every dataset in src.lib.wmt_metadata.DATASET_DIRS)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from src.lib.beta import beta_0 as beta0_of, pairwise_betas
from src.lib.consistency import real_systems
from src.lib.wmt_metadata import DATASET_DIRS
from src.lib.subsampling_stability import leave_p_out_tau
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'output')

NATURAL_HEADER = '| dataset | K | raters (C(K,p)) | tau (natural) |\n|---|---:|---:|---:|\n'
REWEIGHTED_HEADER = ('| dataset | K | raters (C(K,p)) | tau natural | beta_0(D) | '
                      'tau reweighted (beta_0(D)) | median beta_ij(D) | '
                      'tau reweighted (median beta_ij(D)) |\n'
                      '|---|---:|---:|---:|---:|---:|---:|---:|\n')


def tau_cell(result: dict) -> str:
  tau = result['tau']
  tau_str = f"{tau:.4f}" if tau == tau else 'N/A'
  if result['n_infeasible'] > 0:
    tau_str += f" ({result['n_raters_used']}/{result['n_raters_total']} raters)"
  return tau_str


def format_row_natural(dataset: str, K_full: int, natural: dict | None) -> str:
  if natural is None:
    return f'| {dataset} | {K_full} | -- | N/A |\n'
  return f"| {dataset} | {natural['K']} | {natural['n_raters_total']} | {tau_cell(natural)} |\n"


def format_row_reweighted(dataset: str, K_full: int, natural: dict | None,
                           beta_0_D: float | None, rw_a0: dict | None,
                           med_beta_ij_D: float | None, rw_med: dict | None) -> str:
  if natural is None:
    return f'| {dataset} | {K_full} | -- | N/A | -- | -- | -- | -- |\n'
  return (f"| {dataset} | {natural['K']} | {natural['n_raters_total']} | {tau_cell(natural)} | "
          f"{beta_0_D:.4f} | {tau_cell(rw_a0)} | {med_beta_ij_D:.4f} | {tau_cell(rw_med)} |\n")


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--datasets', type=str, default=None,
                  help='comma-separated dataset names (default: every dataset in DATASET_DIRS)')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--leave-out', type=int, default=1,
                  help='p: number of systems dropped per LpO rater (default 1 = plain LOO)')
  p.add_argument('--reweighted', action='store_true',
                  help='opt-in: also compute reweighted-to-beta_0(D)/reweighted-to-median-'
                       'beta_ij(D) columns (system-level reweighting) alongside natural. '
                       'Off by default -- see module docstring for why natural-only is the '
                       'recommended default for this scorer-ranking statistic.')
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
  rw_prefix = 'loo_tau_reweighted' if args.reweighted else 'loo_tau'

  out_path = os.path.join(ARTIFACTS_DIR, f'{rw_prefix}{p_suffix}_{tag}.md')
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  with open(out_path, 'w') as f:
    if args.reweighted:
      f.write(f'# Leave-{leave_out}-out Kendall tau-a across scorer rankings: natural vs. '
              f'reweighted (metametric={m})\n\n')
      f.write(f'Each row: C(K,{leave_out}) LpO subsets of the dataset (one per dropped '
              f'{leave_out}-system combo) are each scored once under three conditions -- '
              "natural (uniform weight), reweighted to the FULL dataset's own beta_0(D), and "
              "reweighted to the median of the full dataset's pairwise beta_ij -- and treated "
              "as raters; Kendall's tau-a (src.lib.consistency.pool_weighted_tau) measures how "
              'consistently they rank the real_scorers-screened scorers, pooled over every '
              'rater-pair x scorer-pair comparison at once. "N/total raters" is shown when a '
              "target was infeasible (outside a subset's own reachable beta range) for some "
              'subset. NOTE: --reweighted applies system-level reweighting on top of a '
              "scorer-ranking statistic -- opt-in comparison only, not this script's "
              'recommended default (see module docstring).\n\n')
      f.write(REWEIGHTED_HEADER)
    else:
      f.write(f'# Leave-{leave_out}-out Kendall tau-a across scorer rankings (natural/uniform '
              f'weight only, metametric={m})\n\n')
      f.write(f'Each row: C(K,{leave_out}) LpO subsets of the dataset (one per dropped '
              f'{leave_out}-system combo), each scored once at natural/uniform weight and '
              "treated as one rater; Kendall's tau-a (src.lib.consistency.pool_weighted_tau) "
              'measures how consistently they rank the real_scorers-screened scorers, pooled '
              'over every rater-pair x scorer-pair comparison at once. No system-level '
              'reweighting -- this statistic is about scorer rankings, not system rankings.\n\n')
      f.write(NATURAL_HEADER)

  print(f'{len(datasets)} datasets, metametric={m}, leave-{leave_out}-out'
        + (', reweighted' if args.reweighted else ' (natural only)'), file=sys.stderr)
  t_start = time.time()
  for i, dataset in enumerate(datasets, 1):
    t0 = time.time()
    full_systems = real_systems(dataset, root=ROOT, exclude_outliers=args.exclude_outliers)
    K_full = len(full_systems)
    if K_full - leave_out < 2:
      natural = beta_0_D = rw_a0 = med_beta_ij_D = rw_med = None
      print(f'[{i}/{len(datasets)}] {dataset}: K={K_full}, skipped (N/A)', file=sys.stderr)
      row = (format_row_reweighted(dataset, K_full, natural, beta_0_D, rw_a0, med_beta_ij_D, rw_med)
             if args.reweighted else format_row_natural(dataset, K_full, natural))
    else:
      natural = leave_p_out_tau(dataset, m, p=leave_out, root=ROOT,
                                 exclude_outliers=args.exclude_outliers)
      if args.reweighted:
        df = load_system_scores(dataset, root=ROOT)
        a_D, f_D = df.loc[full_systems, 'a'].values, df.loc[full_systems, 'f'].values
        beta_0_D = beta0_of(a_D, f_D)
        med_beta_ij_D = float(np.median([v for _, _, v in pairwise_betas(a_D, f_D)]))
        rw_a0 = leave_p_out_tau(dataset, m, p=leave_out, root=ROOT,
                                 exclude_outliers=args.exclude_outliers, target_beta=beta_0_D)
        rw_med = leave_p_out_tau(dataset, m, p=leave_out, root=ROOT,
                                  exclude_outliers=args.exclude_outliers,
                                  target_beta=med_beta_ij_D)
        row = format_row_reweighted(dataset, K_full, natural, beta_0_D, rw_a0, med_beta_ij_D, rw_med)
        msg = (f"[{i}/{len(datasets)}] {dataset}: K={natural['K']}, "
               f"raters={natural['n_raters_total']}, tau natural={natural['tau']:.4f}, "
               f"tau reweighted(a0={beta_0_D:.4f})={rw_a0['tau']:.4f}, "
               f"tau reweighted(med_aij={med_beta_ij_D:.4f})={rw_med['tau']:.4f}")
      else:
        row = format_row_natural(dataset, K_full, natural)
        msg = (f"[{i}/{len(datasets)}] {dataset}: K={natural['K']}, "
               f"raters={natural['n_raters_total']}, tau={natural['tau']:.4f}")
      elapsed = time.time() - t0
      mean_elapsed = (time.time() - t_start) / i
      eta = mean_elapsed * (len(datasets) - i)
      print(f'{msg} ({elapsed:.1f}s, ETA {eta:.0f}s)', file=sys.stderr)

    with open(out_path, 'a') as f:
      f.write(row)

  print(f'\nWrote {out_path}', file=sys.stderr)
