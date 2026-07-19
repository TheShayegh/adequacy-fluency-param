"""Sanity check: does lib.reweight_numeric.solve_w_numeric's REALIZED alpha
(alpha(w) recomputed from the returned weights) match the QUERIED/target
alpha it was asked to solve for? By construction (the solver enforces
g(w,alpha)=0 as an equality constraint, action_plan.md 3.4), these should
agree to numerical tolerance for every target the solver reports success=
True on -- that is the one invariant that must never break. This checks it
across all 15 real datasets, a production-realistic grid plus deliberately
harder near-boundary probes (where a spurious single-system w can trivially
satisfy the balance constraint for ANY target, action_plan.md 3.8's edge
case -- see lib/reweight_numeric.py's _MIN_ESS_TOL guard), and re-attempts
persistent failures with substantially more restarts.

Usage: python codes/scripts/sanity_check_realized_alpha.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_min_max
from lib.consistency import real_systems
from lib.reweight_numeric import solve_w_numeric
from mwb.mqm_scoring import SETS

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

DATASETS = list(SETS.keys())  # all 15
N_GRID = 21          # matches the production grid size used throughout
N_RESTARTS = 30       # more thorough than the quick first pass (15)
RETRY_RESTARTS = 150  # for cases that still fail, before calling them genuinely hard
TOL = 1e-4            # tolerance for the achieved==target invariant on success=True cases


def load_ab(dataset):
  from mwb.mqm_scoring import load_system_scores
  systems = real_systems(dataset, root=ROOT)
  sys_df = load_system_scores(dataset, root=ROOT).loc[systems]
  return sys_df['a'].values, sys_df['b'].values


if __name__ == '__main__':
  t_start = time.time()
  n_checked = 0
  n_success = 0
  n_fail_flagged = 0          # success=False -- expected/honest, not a bug
  invariant_violations = []   # success=True but achieved != target -- would be a REAL bug

  unresolved = []  # (dataset, target) that fail even at N_RESTARTS, to retry harder

  for d in DATASETS:
    a, b = load_ab(d)
    K = len(a)
    lo, hi = alpha_min_max(a, b)
    eps = (hi - lo) * 1e-3
    grid = list(np.linspace(lo + eps, hi - eps, N_GRID))
    # Deliberately harder near-boundary probes, well inside the eps margin
    # normal production grids use -- this is where the spurious single-
    # system trap lives.
    near_boundary = [lo + eps * 0.5, lo + eps * 0.1, hi - eps * 0.5, hi - eps * 0.1]
    targets = grid + [t for t in near_boundary if lo < t < hi]

    t0 = time.time()
    d_fail = 0
    for target in targets:
      r = solve_w_numeric(a, b, target, n_restarts=N_RESTARTS, seed=0)
      n_checked += 1
      diff = abs(r.alpha_achieved - target)
      if r.success:
        n_success += 1
        if diff >= TOL:
          invariant_violations.append((d, target, r.alpha_achieved, diff, r.ess, r.success))
      else:
        n_fail_flagged += 1
        d_fail += 1
        unresolved.append((d, target, a, b))
    print(f'{d:10} K={K:3d}  {len(targets)} targets, {d_fail} flagged failed, '
          f'{time.time() - t0:.1f}s', file=sys.stderr)

  print(f'\nFirst pass ({N_RESTARTS} restarts): {n_checked} checks, '
        f'{n_success} success, {n_fail_flagged} honestly flagged failed, '
        f'{time.time() - t_start:.1f}s total', file=sys.stderr)

  # Retry every flagged failure with far more restarts.
  print(f'\nRetrying {len(unresolved)} flagged failures with {RETRY_RESTARTS} restarts...',
        file=sys.stderr)
  still_unresolved = []
  recovered = 0
  for d, target, a, b in unresolved:
    r = solve_w_numeric(a, b, target, n_restarts=RETRY_RESTARTS, seed=1)
    diff = abs(r.alpha_achieved - target)
    if r.success:
      recovered += 1
      if diff >= TOL:
        invariant_violations.append((d, target, r.alpha_achieved, diff, r.ess, r.success))
      print(f'  RECOVERED  {d:10} target={target:.6f}  achieved={r.alpha_achieved:.6f}  '
            f'diff={diff:.2e}  ess={r.ess:.4f}', file=sys.stderr)
    else:
      still_unresolved.append((d, target, r.alpha_achieved, r.ess))
      print(f'  STILL FAIL {d:10} target={target:.6f}  achieved={r.alpha_achieved:.6f}  '
            f'diff={diff:.2e}  ess={r.ess:.4f}', file=sys.stderr)

  print(f'\n=== SUMMARY ===')
  print(f'Total checks (first pass): {n_checked}')
  print(f'success=True: {n_success}')
  print(f'success=False (first pass): {n_fail_flagged}')
  print(f'  of those, recovered with {RETRY_RESTARTS} restarts: {recovered}')
  print(f'  still unresolved even at {RETRY_RESTARTS} restarts: {len(still_unresolved)}')
  print(f'\nCRITICAL invariant (success=True implies achieved==target within {TOL:.0e}): '
        f'{"HOLDS -- 0 violations" if not invariant_violations else f"VIOLATED {len(invariant_violations)} times"}')
  for v in invariant_violations:
    print(f'  VIOLATION: dataset={v[0]} target={v[1]:.6f} achieved={v[2]:.6f} '
          f'diff={v[3]:.2e} ess={v[4]:.4f} success={v[5]}')
  if still_unresolved:
    print(f'\nGenuinely hard cases (need the exact/analytic solver, not more restarts):')
    for d, target, achieved, ess in still_unresolved:
      print(f'  {d:10} target={target:.6f}  best achieved={achieved:.6f}  ess={ess:.4f}')
