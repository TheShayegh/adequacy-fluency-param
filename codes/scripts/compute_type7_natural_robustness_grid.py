"""Cheap variety-booster for the alt-target heatmaps: instead of Delta_tau
(which needs a solve per row -- expensive to densify), computes tau_0 =
robustness_before (the NATURAL, unweighted pairwise-concordance tau between
D and D', no reweighting at all) over the FULL "A-B" level grid up to A=5
-- both the "L-L" diagonal (B=A, context-free, already covered by the main
experiment) and every staged i-j case (B<A, D defined by an already-removed
context of size A-B). tau_0 needs no solve_w_exact/solve_w_linear call at
all, so densifying the grid this way is cheap where densifying Delta_tau
would not be.

Purpose: the earlier alt-target heatmaps (Delta_tau_t over (Delta_alpha_0,
Delta_stat_t)) were built only from the context-free L-L design (levels
1-1/2-2/3-3), whose data occupies a narrow wedge in that plane -- not
enough independent horizontal (Delta_alpha_0) spread at any fixed
Delta_stat_t to see whether Delta_alpha_0 has its own effect once
Delta_stat_t is controlled. The full A-B grid (staged removals produce far
more varied (D, D') pairs) fills that plane in much more, at zero solving
cost, before committing to the expensive fix of densifying Delta_tau itself.

Caching: a system-pool's natural scorer_scores/alpha_0/mean-adequacy/mean-
fluency/mean-mqm depend ONLY on which systems are excluded, not on whether
that exclusion arose as a "context" (defining D) or a "context+drop-set"
(defining D') -- so both roles share ONE cache keyed by frozenset(excluded
systems). This is what keeps the true cost down to ~1-5k distinct
scorer_scores calls even though the row count (every (context, drop_set)
pair) is 20k-120k.

Usage: python codes/scripts/compute_type7_natural_robustness_grid.py
           --dataset ende23 [--max-level 5] [--metametric spa]
           [--drop-outliers | --no-drop-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.outlier_detection import iterative_single_aspect_outliers
from lib.reweighted_consistency import pairwise_outcomes
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


def tau_of(n, s):
  return (s / n) if n else float('nan')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende23')
  p.add_argument('--max-level', type=int, default=5, help='max A (total systems removed)')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--drop-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  max_A = args.max_level
  tag = args.tag or f'{base}_{m}'

  raw_systems = real_systems(base, root=ROOT)
  df = load_system_scores(base, root=ROOT)

  outliers = []
  if args.drop_outliers:
    a_full = df.loc[raw_systems, 'a'].values
    outliers = [s for s, _, _ in iterative_single_aspect_outliers(raw_systems, a_full)]
  full_systems = [s for s in raw_systems if s not in set(outliers)]
  K = len(full_systems)
  print(f'D = {base}' + (f' minus {outliers}' if outliers else ' (full)') + f': K={K}', file=sys.stderr)

  # (A, B) pairs: 1<=B<=A<=max_A, and A<=K-2 (D' needs >=2 systems).
  ab_pairs = [(A, B) for A in range(1, max_A + 1) for B in range(1, A + 1) if A <= K - 2]

  # Shared cache: frozenset(excluded systems) -> (natural_scores, alpha_0,
  # mean_adequacy, mean_fluency, mean_mqm). Used for BOTH D (excluded =
  # context) and D' (excluded = context|drop_set) -- the whole point.
  cache = {}

  def pool_stats(excluded_key, pool_systems):
    if excluded_key not in cache:
      a_p = df.loc[pool_systems, 'a'].values
      b_p = df.loc[pool_systems, 'b'].values
      cache[excluded_key] = (
          scorer_scores(base, m, root=ROOT, systems=pool_systems),
          alpha0_of(a_p, b_p),
          float(a_p.mean()), float(b_p.mean()), float((a_p + b_p).mean()),
      )
    return cache[excluded_key]

  # Precompute row plan (cheap, no solving) so total row count is known
  # up front for the ETA.
  plan = []
  for A, B in ab_pairs:
    ctx_size = A - B
    for context in itertools.combinations(full_systems, ctx_size):
      context_set = set(context)
      D_systems = [s for s in full_systems if s not in context_set]
      for drop_set in itertools.combinations(D_systems, B):
        plan.append((A, B, context, drop_set, D_systems))
  n_total = len(plan)
  print(f'{len(ab_pairs)} (A,B) pairs, {n_total:,} total rows', file=sys.stderr)

  rows = []
  t0 = time.time()
  log_every = max(1, n_total // 40)
  for i, (A, B, context, drop_set, D_systems) in enumerate(plan, 1):
    context_key = frozenset(context)
    natural_D, a0_D, ade_D, flu_D, mqm_D = pool_stats(context_key, D_systems)

    drop_set_set = set(drop_set)
    Dp_systems = [s for s in D_systems if s not in drop_set_set]
    Dp_key = frozenset(context) | drop_set_set
    natural_Dp, a0_Dp, ade_Dp, flu_Dp, mqm_Dp = pool_stats(Dp_key, Dp_systems)

    outcomes = pairwise_outcomes(natural_D, natural_Dp)
    tau_0 = tau_of(len(outcomes), sum(outcomes))
    delta_alpha0 = abs(a0_D - a0_Dp)
    delta_adequacy = abs(ade_D - ade_Dp)
    delta_fluency = abs(flu_D - flu_Dp)
    delta_mqm = abs(mqm_D - mqm_Dp)

    rows.append((A, B, ','.join(sorted(context)) or '-', ','.join(sorted(drop_set)),
                 tau_0, delta_alpha0, delta_adequacy, delta_fluency, delta_mqm))

    if i % log_every == 0 or i == n_total:
      elapsed = time.time() - t0
      rate = i / elapsed if elapsed > 0 else 0.0
      remaining_s = (n_total - i) / rate if rate > 0 else float('nan')
      print(f'  [{i:,}/{n_total:,} rows, {len(cache):,} distinct pools cached] '
            f'elapsed={elapsed / 60:.1f}min est.remaining={remaining_s / 60:.1f}min',
            file=sys.stderr, flush=True)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type7_natural_robustness_grid_{tag}.csv')
  with open(out_path, 'w', newline='') as f:
    # context/d are comma-joined system lists (B>1 or A-B>1 produce
    # multiple names), which collide with the CSV's own delimiter unless
    # quoted -- csv.writer quotes them automatically where needed.
    writer = csv.writer(f)
    writer.writerow(['A', 'B', 'context', 'd', 'tau_0', 'delta_alpha0',
                      'delta_adequacy', 'delta_fluency', 'delta_mqm'])
    for A, B, context_str, drop_str, tau_0, da, dade, dflu, dmqm in rows:
      writer.writerow([A, B, context_str, drop_str, f'{tau_0:.6f}', f'{da:.6f}',
                        f'{dade:.6f}', f'{dflu:.6f}', f'{dmqm:.6f}'])
  print(f'Wrote {out_path} ({len(rows):,} rows, {len(cache):,} distinct pools computed)', file=sys.stderr)
