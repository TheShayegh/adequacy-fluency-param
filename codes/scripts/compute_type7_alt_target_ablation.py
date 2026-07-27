"""Control ablation for the type-7 removal-robustness experiment
(action_plan.md 6.5): instead of reweighting D to alpha_0(D') (the
adequacy-fluency BALANCE), reweight D to match D''s own mean ADEQUACY,
mean FLUENCY, or mean ALL-MQM (a+b) -- three alpha-IRRELEVANT targets.

Why: if matching any of these also produces a positive pooled Delta-tau,
the earlier "fixing alpha helps" result would be confounded -- a generic
"move D toward anything reachable near D'" effect, unrelated to controlling
the adequacy-fluency balance specifically, could explain the whole
forward-direction finding. This ablation is the missing control.

Solved via lib.reweight_linear.solve_w_linear -- a closed-form convex QP
(this constraint is linear, not the quadratic alpha-balance one, so no
secular-polynomial/curvature-certificate machinery is needed; global
optimality follows from convexity alone). An initial version of this
script used lib.reweight_exhaustive_linear's brute-force grid instead, at
grid_n=10 -- that turned out to be a genuine bug, not just an approximation:
with only `grid_n` quanta of weight to spread over K>grid_n systems, the
lattice's OWN max-ESS ceiling is achieved uniquely by dropping exactly
K-grid_n systems, regardless of target, so every one of the three "alt"
targets landed on the identical drop-one-system point by construction, not
because any of them meaningfully matched. Fixed by using the closed-form
solver as the primary result (cross-validated against a fine exhaustive
grid at small K, see compute_type7_alt_target_ablation.py's git history /
the session that built it) rather than pushing grid_n high enough to avoid
the artifact directly, which would need grid_n several times K -- the same
combinatorial wall lib.reweight_exhaustive's own docstring already flags.

Same "L-L" design and pooling convention as compute_type7_sensitivity.py:
D is always the full (outlier-screened) pool; D' = D minus an L-subset
dropped together; robustness_before/after are pooled by summing raw
concordant/discordant/tie counts across every drop-set BEFORE recomputing
tau, not by averaging per-row tau values.

Usage: python codes/scripts/compute_type7_alt_target_ablation.py
           --dataset ende23 --level 1 [--metametric spa]
           [--drop-outliers | --no-drop-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.outlier_detection import iterative_single_aspect_outliers
from lib.reweight_linear import solve_w_linear
from lib.reweighted_consistency import pairwise_outcomes
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

TARGETS = ['adequacy', 'fluency', 'mqm']


def tau_of(n, s):
  return (s / n) if n else float('nan')


def stat_vector(a, b, target_name):
  if target_name == 'adequacy':
    return a
  if target_name == 'fluency':
    return b
  if target_name == 'mqm':
    return a + b
  raise ValueError(target_name)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende23')
  p.add_argument('--level', type=int, default=1, help='L: systems dropped together (the "L-L" design)')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--drop-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  L = args.level
  tag = args.tag or f'{base}_level{L}-{L}_{m}'

  raw_systems = real_systems(base, root=ROOT)
  df = load_system_scores(base, root=ROOT)

  outliers = []
  if args.drop_outliers:
    a_full = df.loc[raw_systems, 'a'].values
    outliers = [s for s, _, _ in iterative_single_aspect_outliers(raw_systems, a_full)]
  D_systems = [s for s in raw_systems if s not in set(outliers)]
  K_D = len(D_systems)
  if not (1 <= L <= K_D - 2):
    raise ValueError(f'--level {L} must be in [1, {K_D - 2}] for this pool (K={K_D})')

  a_D, b_D = df.loc[D_systems, 'a'].values, df.loc[D_systems, 'b'].values
  natural_D = scorer_scores(base, m, root=ROOT, systems=D_systems)
  print(f'D = {base}' + (f' minus {outliers}' if outliers else ' (full)')
        + f': K={K_D}', file=sys.stderr)

  drop_sets = list(itertools.combinations(D_systems, L))
  n_total = len(drop_sets) * len(TARGETS)
  print(f'{len(drop_sets)} drop-sets x {len(TARGETS)} alt-targets = {n_total} solves',
        file=sys.stderr)

  a0_D = alpha0_of(a_D, b_D)

  # pooled[target] = [n_before, sum_before, n_after, sum_after]
  pooled = {t: [0, 0, 0, 0] for t in TARGETS}
  n_infeasible = {t: 0 for t in TARGETS}
  # per_row[target] = list of (context_str, drop_str, tau_b, tau_a, delta_tau,
  #                             delta_stat, delta_alpha0, ess)
  per_row = {t: [] for t in TARGETS}

  t0 = time.time()
  n_done = 0
  log_every = max(1, n_total // 30)

  for drop_set in drop_sets:
    drop_set_set = set(drop_set)
    drop_str = ','.join(sorted(drop_set))
    Dp_systems = [s for s in D_systems if s not in drop_set_set]
    a_Dp, b_Dp = df.loc[Dp_systems, 'a'].values, df.loc[Dp_systems, 'b'].values
    natural_Dp = scorer_scores(base, m, root=ROOT, systems=Dp_systems)
    a0_Dp = alpha0_of(a_Dp, b_Dp)
    delta_alpha0 = abs(a0_D - a0_Dp)

    before_outcomes = pairwise_outcomes(natural_D, natural_Dp)
    n_before, sum_before = len(before_outcomes), sum(before_outcomes)
    tau_b = tau_of(n_before, sum_before)

    for t in TARGETS:
      stat_D = stat_vector(a_D, b_D, t)
      stat_Dp = stat_vector(a_Dp, b_Dp, t)
      target_val = float(stat_Dp.mean())
      delta_stat = abs(float(stat_D.mean()) - target_val)
      try:
        r = solve_w_linear(stat_D, target_val)
        reweighted_D = scorer_scores(base, m, root=ROOT, w=r.w, systems=D_systems)
        after_outcomes = pairwise_outcomes(reweighted_D, natural_Dp)
        n_after, sum_after = len(after_outcomes), sum(after_outcomes)
      except ValueError:
        n_infeasible[t] += 1
        n_done += 1
        continue

      pooled[t][0] += n_before
      pooled[t][1] += sum_before
      pooled[t][2] += n_after
      pooled[t][3] += sum_after
      tau_a = tau_of(n_after, sum_after)
      per_row[t].append(('-', drop_str, tau_b, tau_a, tau_a - tau_b, delta_stat, delta_alpha0, r.ess))
      n_done += 1

      if n_done % log_every == 0 or n_done == n_total:
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        remaining_s = (n_total - n_done) / rate if rate > 0 else float('nan')
        print(f'  [{n_done}/{n_total}] elapsed={elapsed / 60:.1f}min est.remaining={remaining_s / 60:.1f}min',
              file=sys.stderr, flush=True)

  lines = []
  lines.append(f'# Type-7 level-{L}-{L} alt-target ablation ({base}, K={K_D})')
  lines.append('')
  lines.append('Reweight D toward D\'s own mean adequacy / mean fluency / mean all-MQM '
               '(alpha-irrelevant targets), instead of alpha_0(D\'). Same pooling convention '
               'as the main experiment.')
  lines.append('')
  lines.append('| target | pooled tau_before | pooled tau_after | pooled delta | infeasible |')
  lines.append('|---|---:|---:|---:|---:|')
  for t in TARGETS:
    n_b, s_b, n_a, s_a = pooled[t]
    tau_b_pooled = tau_of(n_b, s_b)
    tau_a_pooled = tau_of(n_a, s_a)
    delta = tau_a_pooled - tau_b_pooled
    lines.append(f'| {t} | {tau_b_pooled:.4f} | {tau_a_pooled:.4f} | {delta:+.4f} | '
                 f'{n_infeasible[t]}/{len(drop_sets)} |')
  table_md = '\n'.join(lines)
  print('\n' + table_md)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type7_level{L}-{L}_alt_target_ablation_{tag}.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)

  # Per-row detail tables, one per target: same 8-column layout regardless
  # of target, so a single parser (delta_stat is always column index 5,
  # abs(delta_alpha_0) index 6, ESS index 7) covers all three.
  for t in TARGETS:
    rows = per_row[t]
    n_b, s_b, n_a, s_a = pooled[t]
    tau_b_pooled = tau_of(n_b, s_b)
    tau_a_pooled = tau_of(n_a, s_a)
    delta_pooled = tau_a_pooled - tau_b_pooled

    row_lines = []
    row_lines.append(f'# Type-7 level-{L}-{L} alt-target ablation detail, target={t} ({base}, K={K_D})')
    row_lines.append('')
    row_lines.append(f'robustness_before/after/delta as in the main experiment; delta_stat = '
                     f'|mean_{t}(D) - mean_{t}(D_-d)| (the target-specific analog of abs(delta_alpha_0), '
                     f'which is also included here unchanged for direct comparison).')
    row_lines.append('')
    row_lines.append('| context | d | robustness_before | robustness_after | delta_robustness | '
                     f'delta_{t} | abs(delta_alpha_0) | ESS |')
    row_lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for context_str, drop_str, tb, ta, dt, ds_, da_, ess in rows:
      row_lines.append(f'| {context_str} | {drop_str} | {tb:.4f} | {ta:.4f} | {dt:+.4f} | '
                       f'{ds_:.4f} | {da_:.4f} | {ess:.2f} |')
    row_lines.append(f'| **pooled** | | **{tau_b_pooled:.4f}** | **{tau_a_pooled:.4f}** | '
                     f'**{delta_pooled:+.4f}** | | | |')
    detail_md = '\n'.join(row_lines)

    detail_path = os.path.join(ARTIFACTS_DIR, f'type7_level{L}-{L}_alt_target_{t}_detail_{tag}.md')
    with open(detail_path, 'w') as f:
      f.write(detail_md + '\n')
    print(f'Wrote {detail_path}', file=sys.stderr)
