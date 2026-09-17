"""Instrumented ablation of Algorithm 1's three pruning mechanisms
(Theorems "Admissible supports", "Support size bound", "Exit certificate"),
reproducing the paper's efficiency tables (Appendix, Sec. "Efficiency and
Ablation Analysis"): Table "solve-exact-operation-counts" (# supports
visited, # candidates evaluated, success rate, and an approximate FLOP
count, for every combination of the three pruning theorems switched on/off,
swept over 50 beta targets in each of two regimes) and Table
"solve-exact-theorem-invocations" (how many times each theorem actually
fires over a 50-point sweep, with all three enabled -- the production
configuration).

This is a THIN WRAPPER around src.lib.reweight_exact's private search
primitives (_support_candidates, _admissible, _curvature_ok), reimplementing
only the outer support-enumeration loop with toggles for each pruning
mechanism -- the expensive inner math (the secular-equation scan, the
curvature test) is untouched, imported directly from the production module,
not duplicated. The production solve_w_exact itself is never called here;
the loop below is a superset of it with the three prunes made optional.

FLOP accounting is an approximate proxy, not a instruction-level profile:
each non-degenerate support's dominant cost is its secular-equation scan
(reweight_exact._support_candidates' n_window/n_outer segments, ~1000
evaluation points for a 2-pole support, ~600 for 1-pole, ~400 for a
degenerate/0-pole one), each point costing roughly OPS_PER_SCAN_POINT
arithmetic operations per pole. This reproduces the qualitative trends the
paper reports (near-beta_0 targets are cheap, the full reachable range is
far more expensive; pruning saves roughly an order of magnitude) but should
not be expected to match the paper's published FLOP figures to the digit --
those came from a since-lost ad hoc instrumented copy of an earlier solver
revision; this script is a from-scratch reconstruction of the same
methodology, not a byte-for-byte replay.

Usage: python -m src.scripts.solver_efficiency --dataset ende24 [--n-sweep 50]
"""

from __future__ import annotations

import argparse
import itertools
import math
import os

import numpy as np

from src.lib.beta import beta_0, beta_min_max, pairwise_betas
from src.lib.consistency import real_systems
from src.lib.reweight_exact import _admissible, _curvature_ok, _d_all, _prop1_2x2, _support_candidates, beta_of
from src.lib.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_TOL = 1e-7

# Secular-equation scan resolution, mirrored from src.lib.reweight_exact's
# _support_candidates (n_window, n_outer) -- see that function if these
# ever change there.
_N_WINDOW, _N_OUTER = 400, 300
_OPS_PER_SCAN_POINT = 15  # a few multiply-adds per pole per evaluation point


def _scan_points(n_poles: int) -> int:
  if n_poles == 2:
    return _N_WINDOW + 2 * _N_OUTER
  if n_poles == 1:
    return 2 * _N_OUTER
  return _N_WINDOW


def _support_flops(a_S: np.ndarray, f_S: np.ndarray, beta: float, tol: float = _TOL) -> int:
  """Approximate FLOP cost of _support_candidates on one support, without
  re-running the scan itself: _prop1_2x2 (a fixed small cost) plus the
  secular-equation scan cost, which only depends on how many of the two
  poles are non-degenerate (see module docstring)."""
  mu_a, mu_f = float(a_S.mean()), float(f_S.mean())
  va = float(np.mean((a_S - mu_a) ** 2))
  vf = float(np.mean((f_S - mu_f) ** 2))
  beta0_S = va / (va + vf) if (va + vf) > 0 else 0.5
  if abs(beta - beta0_S) < tol:
    return 20  # anchor case: closed-form, no scan
  theta1, theta2, rho, *_ = _prop1_2x2(a_S, f_S, beta)
  n_poles = 1 if abs(1.0 - rho ** 2) < tol else (2 if abs(theta2) > tol else 1)
  return 30 + _scan_points(n_poles) * _OPS_PER_SCAN_POINT


def solve_instrumented(
    a: np.ndarray, f: np.ndarray, beta: float,
    use_admissible: bool = True, use_size_pruning: bool = True, use_certificate: bool = True,
    tol: float = _TOL,
) -> dict:
  """Superset of src.lib.reweight_exact.solve_w_exact's search loop with
  each of its three pruning mechanisms independently toggleable. Always
  sound (every returned w is checked feasible); completeness (finding the
  TRUE global optimum) is only guaranteed with all three enabled, matching
  the production solver -- with pruning disabled this still explores a
  superset of supports, so it never finds a WORSE optimum, only a slower
  path to the same or a better one.

  Returns a dict: w, ess, success, n_supports_visited, n_candidates_evaluated,
  flops, certified, admissible_fires (# supports skipped by Theorem
  "Admissible supports"), size_pruning_fired (bool), certificate_fired
  (bool)."""
  a = np.asarray(a, dtype=float)
  f = np.asarray(f, dtype=float)
  K = len(a)
  pairs = pairwise_betas(a, f)
  beta_min, beta_max = beta_min_max(a, f)
  if not (beta_min - tol <= beta <= beta_max + tol):
    raise ValueError(f'beta={beta} outside reachable range [{beta_min}, {beta_max}]')

  best = None  # (ess, w, support, certified)
  n_supports_visited = 0
  n_candidates_evaluated = 0
  flops = 0
  admissible_fires = 0
  size_pruning_fired = False
  certificate_fired = False

  for size in range(K, 1, -1):
    if use_size_pruning and best is not None and size <= best[0]:
      size_pruning_fired = True
      break
    for S in itertools.combinations(range(K), size):
      lo_S = min((v for i, j, v in pairs if i in S and j in S), default=None)
      hi_S = max((v for i, j, v in pairs if i in S and j in S), default=None)
      if lo_S is None:
        continue
      if use_admissible:
        if not _admissible(beta, lo_S, hi_S, tol):
          admissible_fires += 1
          continue
      n_supports_visited += 1
      flops += _support_flops(a[list(S)], f[list(S)], beta, tol)

      # Silenced: with pruning disabled this deliberately runs
      # _support_candidates outside its guaranteed-valid domain (see the
      # comment below), which routinely produces harmless inf/nan
      # intermediate values -- caught and skipped explicitly below, so the
      # numpy-level warning is noise, not a signal.
      with np.errstate(all='ignore'):
        candidates = _support_candidates(a, f, S, beta, tol)
      for w, eta in candidates:
        n_candidates_evaluated += 1
        # With Theorem "Admissible supports" pruning disabled, this loop
        # calls _support_candidates on supports where the target beta is
        # genuinely unreachable -- outside that function's guaranteed-valid
        # domain (the production solver never does this, since it always
        # prunes first). A candidate there can come back numerically
        # degenerate (e.g. zero variance); such a candidate is simply not
        # valid for this support, same conclusion as any other feasibility
        # check below, not a real error.
        try:
          support_mask = np.zeros(K, dtype=bool)
          support_mask[list(S)] = True
          if np.any(w[support_mask] <= -tol) or np.any(w[~support_mask] > tol):
            continue
          if abs(float(w.sum()) - 1.0) > 1e-3:
            continue
          w = np.clip(w, 0.0, None)
          w_sum = float(w.sum())
          if w_sum <= 0:
            continue
          w = w / w_sum
          if not np.isfinite(w).all() or abs(beta_of(a, f, w) - beta) > 1e-6:
            continue
          actual_support = tuple(i for i in range(K) if w[i] > tol)
          if actual_support != S:
            continue
        except (ZeroDivisionError, FloatingPointError):
          continue

        ess = 1.0 / float(np.sum(w ** 2))
        certified = False
        if use_certificate:
          d_all = _d_all(w, a, f, beta)
          i0 = S[0]
          lam = -(w[i0] + eta * d_all[i0])
          outside = [i for i in range(K) if i not in S]
          cond2 = all(lam + eta * d_all[i] >= -tol for i in outside)
          cond3 = _curvature_ok(a, f, beta, eta, tol)
          certified = cond2 and cond3

        if best is None or ess > best[0]:
          best = (ess, w, S, certified)
        if certified:
          certificate_fired = True
          return {
              'w': w, 'ess': ess, 'success': True, 'certified': True,
              'n_supports_visited': n_supports_visited, 'n_candidates_evaluated': n_candidates_evaluated,
              'flops': flops, 'admissible_fires': admissible_fires,
              'size_pruning_fired': size_pruning_fired, 'certificate_fired': True,
          }

  if best is None:
    return {
        'w': None, 'ess': float('nan'), 'success': False, 'certified': False,
        'n_supports_visited': n_supports_visited, 'n_candidates_evaluated': n_candidates_evaluated,
        'flops': flops, 'admissible_fires': admissible_fires,
        'size_pruning_fired': size_pruning_fired, 'certificate_fired': False,
    }
  ess, w, S, certified = best
  return {
      'w': w, 'ess': ess, 'success': True, 'certified': False,
      'n_supports_visited': n_supports_visited, 'n_candidates_evaluated': n_candidates_evaluated,
      'flops': flops, 'admissible_fires': admissible_fires,
      'size_pruning_fired': size_pruning_fired, 'certificate_fired': False,
  }


_CONFIGS = [
    (True, True, True), (True, True, False), (False, True, True), (True, False, True),
    (False, True, False), (False, False, True), (True, False, False), (False, False, False),
]


def run_regime(a: np.ndarray, f: np.ndarray, targets: list[float]) -> dict:
  """Table "solve-exact-operation-counts": one row per (admissible,
  size_pruning, certificate) config, summed over `targets`."""
  rows = {}
  for cfg in _CONFIGS:
    n_sup = n_cand = flops = successes = 0
    for beta in targets:
      r = solve_instrumented(a, f, beta, *cfg)
      n_sup += r['n_supports_visited']
      n_cand += r['n_candidates_evaluated']
      flops += r['flops']
      successes += int(r['success'])
    rows[cfg] = {
        'n_supports': n_sup, 'n_candidates': n_cand, 'flops': flops,
        'success_pct': 100.0 * successes / len(targets),
    }
  return rows


def count_invocations(a: np.ndarray, f: np.ndarray, targets: list[float]) -> dict:
  """Table "solve-exact-theorem-invocations": totals over `targets`, all
  three pruning mechanisms enabled (the production configuration)."""
  admissible_total = 0
  size_pruning_total = 0
  certificate_total = 0
  for beta in targets:
    r = solve_instrumented(a, f, beta, True, True, True)
    admissible_total += r['admissible_fires']
    size_pruning_total += int(r['size_pruning_fired'])
    certificate_total += int(r['certificate_fired'])
  return {
      'admissible': admissible_total, 'size_pruning': size_pruning_total,
      'certificate': certificate_total,
  }


def _fmt(n: float) -> str:
  if n >= 1e6:
    return f'{n / 1e6:.1f}M'
  if n >= 1e3:
    return f'{n / 1e3:.1f}K'
  return f'{n:.0f}'


def main(argv=None) -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', default='ende24', help="paper's ablation dataset (largest K)")
  p.add_argument('--n-sweep', type=int, default=50, help='number of beta targets per regime')
  p.add_argument('--narrow-window', type=float, default=0.05, help='beta_0 +/- this, for the narrow regime')
  args = p.parse_args(argv)

  systems = real_systems(args.dataset, root=ROOT)
  if not systems:
    raise SystemExit(f'{args.dataset}: 0 real systems, nothing to compute')
  df = load_system_scores(args.dataset, root=ROOT)
  a, f = df.loc[systems, 'a'].values, df.loc[systems, 'f'].values
  K = len(systems)
  beta0 = beta_0(a, f)
  beta_lo, beta_hi = beta_min_max(a, f)
  print(f'{args.dataset}: K={K}, beta_0={beta0:.4f}, reachable range [{beta_lo:.4f}, {beta_hi:.4f}]')

  narrow_lo = max(beta_lo, beta0 - args.narrow_window)
  narrow_hi = min(beta_hi, beta0 + args.narrow_window)
  narrow_targets = list(np.linspace(narrow_lo, narrow_hi, args.n_sweep))
  eps = (beta_hi - beta_lo) * 1e-3
  full_targets = list(np.linspace(beta_lo + eps, beta_hi - eps, args.n_sweep))

  for regime, targets in (('narrow (beta_0 +/- %.2f)' % args.narrow_window, narrow_targets),
                           ('full range', full_targets)):
    print(f'\n=== {regime}, {args.n_sweep}-point sweep ===')
    rows = run_regime(a, f, targets)
    print(f'{"admissible":>11} {"size":>5} {"cert":>5} {"#supports":>10} {"#candidates":>12} '
          f'{"success%":>9} {"#flops":>8}')
    for cfg in _CONFIGS:
      r = rows[cfg]
      marks = tuple('Y' if c else '-' for c in cfg)
      print(f'{marks[0]:>11} {marks[1]:>5} {marks[2]:>5} {_fmt(r["n_supports"]):>10} '
            f'{_fmt(r["n_candidates"]):>12} {r["success_pct"]:>8.0f}% {_fmt(r["flops"]):>8}')

    inv = count_invocations(a, f, targets)
    print(f'Theorem invocations (all pruning on): admissible={_fmt(inv["admissible"])}, '
          f'size_pruning={inv["size_pruning"]}, certificate={inv["certificate"]}')


if __name__ == '__main__':
  main()
