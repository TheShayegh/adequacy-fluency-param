"""Weighted, M(beta)-style curves through the cross-dataset consistency
metric: re-run mwb.lib.consistency's pooled weighted-Kendall-tau
consistency measure, but with every dataset's systems reweighted to a
common target beta via the EXACT solver (mwb.lib.reweight_exact.
solve_w_exact) instead of their natural (uniform) weighting -- so this
module's name says "reweighted", not "exact": which solver produced
w(beta) lives one level down, and this module is agnostic to it (pass a
different solve function via `solver` if needed).

This used to default to a multi-start local optimizer; that solver was
retired after cross-validation against mwb.lib.reweight_exhaustive showed
it missing the true optimum in 13/15 tested cases at its default restart
count, sometimes by a large margin (e.g. ESS 1.07 vs the achievable 4.0) --
mwb.lib.reweight_exact has no such risk (a certified global optimum every
time) and no restart/seed parameters to tune.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time

import numpy as np

from mwb.lib.consistency import real_systems
from mwb.lib.reweight_exact import solve_w_exact
from mwb.mqm_scoring import load_system_scores


def dataset_score_arrays(
    datasets: list[str], root: str = '.', systems_by_dataset: dict[str, list[str]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """(a, f) system-score arrays per dataset, restricted to
  systems_by_dataset[d] if given (default: real_systems(d))."""
  out = {}
  for d in datasets:
    systems = (systems_by_dataset or {}).get(d) or real_systems(d, root=root)
    df = load_system_scores(d, root=root).loc[systems]
    out[d] = (df['a'].values, df['f'].values)
  return out


def _solve_w_job(job):
  """One (dataset, beta) worker-process body for solve_w_for_datasets's
  --workers > 1 path -- module-level so it's importable/picklable under
  macOS's spawn start method. Every (d, beta) pair is independent (each
  call only reads its own dataset's (a, f) arrays, no shared mutable
  state), so this parallelizes with no cross-job coordination needed."""
  d, a, f, beta, solver, solver_kwargs = job
  return d, beta, solver(a, f, beta, **solver_kwargs)


def solve_w_for_datasets(
    datasets: list[str], betas, root: str = '.', solver=solve_w_exact,
    systems_by_dataset: dict[str, list[str]] | None = None, workers: int = 1, progress: bool = False,
    **solver_kwargs,
) -> dict[tuple[str, float], object]:
  """Solves w_d(beta) once for every (dataset, beta) pair -- shared
  across every meta-metric's curve, since the weighting itself doesn't
  depend on which meta-metric will later use it. Returns {(dataset, beta):
  ExactWResult}.

  systems_by_dataset: restrict dataset d's systems to systems_by_dataset[d]
  instead of real_systems(d) -- e.g. a leave-one-out/exclude-system subset
  of one base dataset, id'd by a synthetic `d` that isn't itself a real
  WMT dataset key (see dataset_score_arrays).

  workers: >1 dispatches every (d, beta) job to a ProcessPoolExecutor
  (_solve_w_job) instead of solving sequentially -- solve_w_exact's cost
  is highly non-uniform across beta (a fast certified KKT pass vs. a slow
  exhaustive-enumeration fallback, see mwb.lib.reweight_exact's module
  docstring), so this is the actual bottleneck in dense-grid callers like
  compute_scorer_preference_vs_beta.py, not just the per-base scorer loop.
  Default 1 (sequential); every other caller of this shared function is
  unaffected unless it opts in.

  progress: log '[done/n] elapsed, eta' to stderr every ~5% of jobs (both
  the sequential and parallel paths) -- solve_w_exact's per-beta cost
  swings from milliseconds to 15+ seconds depending on whether that
  fallback fires, so a flat elapsed/n_total estimate isn't available up
  front; this reports the same running elapsed/done*(n-done) estimate the
  base scorer loop below already uses. Default False -- opt-in, since some
  callers solve as few as one beta and would get a log line per call for
  no benefit."""
  arrays = dataset_score_arrays(datasets, root, systems_by_dataset)
  jobs = [(d, a, f, beta, solver, solver_kwargs) for d, (a, f) in arrays.items() for beta in betas]
  n = len(jobs)
  log_every = max(1, n // 20)
  t0 = time.time()
  cache = {}
  done = 0

  def _log():
    if progress and (done % log_every == 0 or done == n):
      elapsed = time.time() - t0
      eta = elapsed / done * (n - done)
      print(f'solve_w_for_datasets: [{done}/{n}] elapsed {elapsed:.1f}s, eta {eta:.1f}s', file=sys.stderr)

  if workers <= 1:
    for d, a, f, beta, solver_, solver_kwargs_ in jobs:
      cache[(d, beta)] = solver_(a, f, beta, **solver_kwargs_)
      done += 1
      _log()
    return cache

  with cf.ProcessPoolExecutor(max_workers=workers) as ex:
    for d, beta, result in ex.map(_solve_w_job, jobs):
      cache[(d, beta)] = result
      done += 1
      _log()
  return cache
