"""GPU-accelerated version of lib.reweight_exhaustive: same brute-force
simplex-lattice search over (P) (see that module's docstring for what (P)
is and why a dumb, structure-free solver is worth having at all), but
built to actually use an accelerator instead of only vectorizing the math.

Profiling lib.reweight_exhaustive showed its Python `itertools.
combinations_with_replacement` + per-row `np.add.at` candidate generation,
not the vectorized numpy evaluation math, dominates its wall-clock cost
(~180K points/sec, all on one CPU core) -- so porting only the evaluation
step to a GPU would leave the actual bottleneck untouched. This module
instead replaces candidate generation itself with a combinadic
(combinatorial-number-system) unranking: every lattice point is produced
directly from its integer rank via closed-form arithmetic (a handful of
torch.searchsorted calls against small precomputed binomial-coefficient
tables, one per composition slot), with no per-point Python-level loop at
all. Validated against lib.reweight_exhaustive's itertools-based
enumeration (identical point sets) before being trusted here; see this
module's `_selftest_matches_cpu_enumeration`.

Generation runs on CPU UNCONDITIONALLY, regardless of `device` --
measured directly (rerun the comparison yourself to reproduce): torch's
own vectorized CPU tensor ops already turn
that ~180K pts/sec Python loop into ~15M+ pts/sec with no accelerator
involved at all, while torch.searchsorted specifically was ~90x SLOWER on
this machine's MPS backend than on CPU for the same batch (8.2s vs
0.09s for one call over 17M values against a length-200 table) --
apparently a poorly-optimized MPS kernel path in this torch version, not
a subtlety of this problem. Sending generation through MPS would have
made the whole pipeline slower than the plain CPU port, not faster. Only
the evaluation step (weighted variance / alpha / objective -- large
matmuls and reductions, an actually GPU-friendly shape) is dispatched to
`device`; measured ~1.75x faster on MPS than CPU for that step alone on
this machine. Net effect: expect this module to beat lib.
reweight_exhaustive substantially (generation dominates, and CPU tensor
ops alone fix that), but expect only a modest further win from `device`
pointing at an actual accelerator rather than 'cpu' -- verify on your own
hardware/torch version before assuming otherwise, since the MPS
searchsorted slowness in particular may be version-specific.

Still fundamentally combinatorial (C(grid_n+K-1, K-1) points -- faster
generation raises the practical ceiling, it does not remove the
combinatorial wall), so this remains a small-K cross-check tool, not a
production solver -- see lib.reweight_exhaustive's docstring for that
same caveat, which applies here unchanged.

Device (evaluation only): default is `device='smart'`, which picks 'cpu'
below _SMART_CANDIDATE_THRESHOLD lattice points and the best available
accelerator (CUDA > MPS) at or above it -- below that threshold, a GPU's
fixed dispatch/transfer overhead outweighs its per-point speed edge for
this evaluation step. Threshold (2,000,000) comes from isolated,
per-process measurement (no shared-process warm-cache contamination, which
earlier inflated apparent GPU wins) across K=5/10/15: cpu vs mps crossed
over right around 2M points for K=10 and K=15 alike (near-ties at exactly
that size, cpu clearly ahead below ~1.3-1.5M for either); K=5 is a known
outlier still favoring cpu by 35% at 1.5M points, so this one constant is
NOT well-tuned for small K -- small-K problems will default to mps
somewhat earlier than is actually optimal for them. Pass 'cpu'/'mps'/
'cuda' explicitly to force one regardless of problem size. MPS (Apple
Silicon) does not support float64
-- evaluation runs in float32 there, float64 elsewhere (generation is
integer arithmetic throughout, unaffected). float32 is still far more
precise than any `tol` this brute-force method is used with (>= 1e-4),
but a result computed on MPS should not be expected to match a CPU/exact
solver's answer to more than ~7 significant digits.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lib.alpha import alpha_min_max
from lib.reweight_exhaustive import ExhaustiveWResult

_DEFAULT_GRID_N = 60
_DEFAULT_TOL = 1e-3
# Higher than lib.reweight_exhaustive's CPU default (20M): a GPU chews
# through this workload roughly an order of magnitude faster, but the cost
# is still combinatorial and this is still just a raised ceiling, not a
# removed wall -- see module docstring.
_DEFAULT_MAX_CANDIDATES = 300_000_000
_DEFAULT_CHUNK = 20_000_000
# Below this many lattice points, plain CPU beat every accelerator tried in
# direct measurement (see module docstring); at or above it, the best
# available accelerator won. One round-number threshold, not tuned per K.
# Chosen from isolated (subprocess-per-measurement, no shared-process
# warm-cache contamination) benchmarking across K=5/10/15: at K=10 and
# K=15, cpu-vs-mps crossed over right around 2M points (K=10, grid_n=16,
# 2.04M: cpu 0.2226s vs mps 0.2275s; K=15, grid_n=10, 1.96M: cpu 0.3186s
# vs mps 0.3108s) -- both near-ties there, cpu still clearly ahead below
# ~1.3-1.5M for either K. K=5 is a known outlier needing a much higher
# threshold before mps catches up (still cpu-favored by 35% at 1.5M
# points, untested how much higher its true crossover sits) -- this single
# constant does not correct for that; small-K problems will still default
# to mps somewhat earlier than truly optimal for them.
_SMART_CANDIDATE_THRESHOLD = 2_000_000


def pick_device(prefer: str | None = 'smart', n_candidates: int | None = None) -> torch.device:
  """'cpu'/'mps'/'cuda' force that device outright. 'smart' (the default)
  picks 'cpu' when n_candidates < _SMART_CANDIDATE_THRESHOLD, else the best
  available accelerator (CUDA > MPS) -- requires n_candidates. None (or any
  other falsy value) skips the size check entirely and always picks the
  best available accelerator, matching this function's pre-'smart' default
  behavior (useful if you want CUDA/MPS regardless of problem size)."""
  if prefer == 'smart':
    if n_candidates is None:
      raise ValueError("pick_device(prefer='smart') requires n_candidates")
    if n_candidates < _SMART_CANDIDATE_THRESHOLD:
      return torch.device('cpu')
    prefer = None  # at/above threshold: fall through to best-available, same as prefer=None
  if prefer:
    return torch.device(prefer)
  if torch.cuda.is_available():
    return torch.device('cuda')
  if torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


def _float_dtype(device: torch.device) -> torch.dtype:
  return torch.float32 if device.type == 'mps' else torch.float64


def _binom_table(M: int, j: int, device: torch.device) -> torch.Tensor:
  """table[v] = C(v, j) for v = 0..M-1, as an int64 tensor on `device` --
  small (length M) and built once per (M, j), reused across every chunk."""
  vals = [math.comb(v, j) for v in range(M)]
  return torch.tensor(vals, dtype=torch.int64, device=device)


def _simplex_lattice_chunk(
    K: int, grid_n: int, r_start: int, r_end: int,
    device: torch.device, binom_tables: dict[int, torch.Tensor],
) -> torch.Tensor:
  """Combinadic-unranks integer ranks [r_start, r_end) directly into an
  (n, K) int64 tensor of nonnegative composition counts summing to
  grid_n -- every point of the K-dim simplex lattice at resolution
  grid_n, batched with no per-point Python loop (contrast lib.
  reweight_exhaustive's _simplex_lattice_counts, which has one).

  Bijection (stars and bars): a composition (c_0..c_{K-1}) summing to
  grid_n corresponds to choosing K-1 "divider" positions d_1<...<d_{K-1}
  from M=grid_n+K-1 total slots (0-indexed), via
    c_0 = d_1, c_i = d_{i+1}-d_i-1 (0<i<K-1), c_{K-1} = (M-1)-d_{K-1}.
  The d's are found by standard combinadic unranking: for j=K-1 down to
  1, d_j is the largest v with C(v,j) <= remaining-rank (a vectorized
  torch.searchsorted against the precomputed C(*,j) table), then
  remaining -= C(d_j,j). This is the same bijection lib.reweight_
  exhaustive's itertools enumeration realizes implicitly; see this
  module's docstring for the cross-check that confirms they agree.
  """
  M = grid_n + K - 1
  r = torch.arange(r_start, r_end, dtype=torch.int64, device=device)
  remaining = r.clone()
  d_desc = []  # d_{K-1}, d_{K-2}, ..., d_1 (found in this order)
  for j in range(K - 1, 0, -1):
    table_j = binom_tables[j]
    idx = torch.searchsorted(table_j, remaining, right=True) - 1
    d_desc.append(idx)
    remaining = remaining - table_j[idx]
  d = list(reversed(d_desc))  # d_1, d_2, ..., d_{K-1} (ascending)

  n = r.shape[0]
  c = torch.empty((n, K), dtype=torch.int64, device=device)
  c[:, 0] = d[0]
  for i in range(1, K - 1):
    c[:, i] = d[i] - d[i - 1] - 1
  c[:, K - 1] = (M - 1) - d[-1]
  return c


def solve_w_exhaustive_gpu(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    grid_n: int = _DEFAULT_GRID_N,
    tol: float = _DEFAULT_TOL,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    chunk: int = _DEFAULT_CHUNK,
    device: str | None = 'smart',
) -> ExhaustiveWResult:
  """Same contract as lib.reweight_exhaustive.solve_w_exhaustive (checks
  every simplex-lattice point at resolution grid_n against |alpha(w) -
  alpha| <= tol, returns the min-objective/max-ESS survivor), computed as
  batched tensor ops instead of a Python loop. Returns the same
  ExhaustiveWResult type, so callers can compare this against the CPU
  version's output directly.

  `device` (see pick_device) only controls where the EVALUATION step
  runs; candidate generation is always CPU (see module docstring for why:
  torch.searchsorted was found much slower on this machine's MPS backend
  than plain CPU, for the exact op this generator relies on). Default is
  'smart': CPU below _SMART_CANDIDATE_THRESHOLD lattice points, best
  available accelerator at or above it -- pass 'cpu'/'mps'/'cuda' to force
  one regardless of problem size."""
  a_np = np.asarray(a, dtype=float)
  b_np = np.asarray(b, dtype=float)
  K = len(a_np)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= alpha <= 1.0):
    raise ValueError(f'alpha={alpha} outside [0,1]')
  amin, amax = alpha_min_max(a_np, b_np)
  if not (amin - tol <= alpha <= amax + tol):
    raise ValueError(f'alpha={alpha} outside reachable range [{amin}, {amax}] (Corollary 2)')

  M = grid_n + K - 1
  n_lattice = math.comb(M, K - 1)
  if n_lattice > max_candidates:
    raise ValueError(
        f'simplex lattice for K={K}, grid_n={grid_n} has {n_lattice:,} points, over '
        f'max_candidates={max_candidates:,} -- this solver is brute-force by design and '
        f'that cost is never optimized away (a GPU raises the practical ceiling, it does '
        f'not remove the combinatorial wall); lower grid_n, lower K, or raise '
        f'max_candidates and be prepared to wait.')

  dev = pick_device(device, n_candidates=n_lattice)
  dtype = _float_dtype(dev)

  # Generation always on CPU (see module docstring: torch.searchsorted was
  # found ~90x slower on this machine's MPS backend than CPU for this exact
  # shape, so routing generation through `dev` would only make things
  # worse there) -- only the counts tensor crosses to `dev`, for evaluation.
  cpu = torch.device('cpu')
  binom_tables = {j: _binom_table(M, j, cpu) for j in range(1, K)}
  a_t = torch.tensor(a_np, dtype=dtype, device=dev)
  b_t = torch.tensor(b_np, dtype=dtype, device=dev)

  best_obj = None
  best_w = None
  best_achieved = None
  best_gap = None
  n_feasible = 0

  for r0 in range(0, n_lattice, chunk):
    r1 = min(r0 + chunk, n_lattice)
    counts = _simplex_lattice_chunk(K, grid_n, r0, r1, cpu, binom_tables)
    w = counts.to(device=dev, dtype=dtype) / grid_n  # (n, K), simplex points by construction

    # Two-pass (mean-centered) weighted variance, vectorized -- same
    # robustness rationale as lib.reweight_exhaustive: the single-pass
    # identity (sum w*x^2 - mu^2) suffers catastrophic cancellation
    # exactly where it matters most (near-degenerate, low-ESS w), and a
    # "trusted" solver should never trade that away for speed.
    mu_a = w @ a_t
    mu_b = w @ b_t
    var_a = (w * (a_t.unsqueeze(0) - mu_a.unsqueeze(1)) ** 2).sum(dim=1)
    var_b = (w * (b_t.unsqueeze(0) - mu_b.unsqueeze(1)) ** 2).sum(dim=1)
    denom = var_a + var_b
    achieved = torch.where(denom > 0, var_a / denom, torch.full_like(denom, float('nan')))

    gap = (achieved - alpha).abs()
    feasible = gap <= tol
    n_feasible += int(feasible.sum().item())
    objective = 0.5 * (w ** 2).sum(dim=1)

    if bool(feasible.any()):
      masked_obj = torch.where(feasible, objective, torch.full_like(objective, float('inf')))
      local_idx = int(torch.argmin(masked_obj).item())
      local_obj = float(masked_obj[local_idx].item())
      if best_obj is None or local_obj < best_obj:
        best_obj = local_obj
        best_w = w[local_idx].to('cpu').to(torch.float64).numpy()
        best_achieved = float(achieved[local_idx].item())

    if best_obj is None:
      # No feasible point found anywhere yet -- track the closest miss
      # (mirrors lib.reweight_exhaustive's failure-reporting behavior).
      valid = ~torch.isnan(gap)
      if bool(valid.any()):
        masked_gap = torch.where(valid, gap, torch.full_like(gap, float('inf')))
        local_idx = int(torch.argmin(masked_gap).item())
        local_gap = float(masked_gap[local_idx].item())
        if best_gap is None or local_gap < best_gap:
          best_gap = local_gap
          if best_w is None:
            best_w = w[local_idx].to('cpu').to(torch.float64).numpy()
            best_achieved = float(achieved[local_idx].item())

  if best_obj is not None:
    gap = abs(best_achieved - alpha)
    return ExhaustiveWResult(
        w=best_w, alpha_target=alpha, alpha_achieved=best_achieved, gap=gap,
        success=True, objective=best_obj, ess=1.0 / (2.0 * best_obj),
        grid_n=grid_n, tol=tol, n_candidates=n_lattice, n_feasible=n_feasible,
        message=f'best of {n_feasible} lattice points within tol={tol} of alpha={alpha} '
                f'(actual gap {gap:.3g}, device={dev.type})',
    )
  gap = abs(best_achieved - alpha) if best_achieved is not None else float('nan')
  return ExhaustiveWResult(
      w=best_w, alpha_target=alpha, alpha_achieved=best_achieved, gap=gap,
      success=False, objective=float('nan'), ess=float('nan'),
      grid_n=grid_n, tol=tol, n_candidates=n_lattice, n_feasible=0,
      message=(f'no lattice point within tol={tol} of alpha={alpha} '
               f'(closest miss off by {best_gap:.3g}, device={dev.type}) -- raise grid_n or tol'),
  )


def _selftest_matches_cpu_enumeration(max_K: int = 6, max_grid_n: int = 6) -> None:
  """Confirms _simplex_lattice_chunk produces exactly the same point SET
  as lib.reweight_exhaustive's itertools-based enumeration (order doesn't
  matter, coverage must be identical) -- the trust basis for this whole
  module, since a wrong combinadic bijection would silently under- or
  over-cover the simplex rather than crash. Run on cpu (float64 unused
  here, this only checks the integer counts) so it works with no
  accelerator present."""
  from lib.reweight_exhaustive import _simplex_lattice_counts
  dev = torch.device('cpu')
  for K in range(2, max_K + 1):
    for grid_n in range(1, max_grid_n + 1):
      M = grid_n + K - 1
      n = math.comb(M, K - 1)
      tables = {j: _binom_table(M, j, dev) for j in range(1, K)}
      got = set(tuple(row) for row in _simplex_lattice_chunk(K, grid_n, 0, n, dev, tables).tolist())
      want = set(tuple(row.tolist()) for chunk in _simplex_lattice_counts(K, grid_n) for row in chunk)
      assert got == want, f'K={K}, grid_n={grid_n}: mismatch ({len(got)} vs {len(want)} points)'
  print(f'OK: combinadic lattice matches itertools enumeration for K in [2,{max_K}], '
        f'grid_n in [1,{max_grid_n}]')


if __name__ == '__main__':
  _selftest_matches_cpu_enumeration()
