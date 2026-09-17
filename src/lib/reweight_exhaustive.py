"""EXHAUSTIVE (brute-force grid) solver for (P) -- see src.lib.reweight_exact's
module docstring for what (P) is. Unlike that module (closed-form
KKT/support-enumeration derivation), this one uses NO structure at all: it
lays down every point of a uniform lattice on the whole weight simplex,
checks each one directly against beta(w) == target (within `tol`, since a
discrete lattice will essentially never land on an equality constraint
exactly), and keeps the minimum-objective (= max-ESS) survivor by plain
comparison.

Why this exists despite src.lib.reweight_exact already being sound+complete:
that module's correctness rests on a long closed-form derivation (secular
equation, curvature certificate, live-pole handling) that is easy to get
subtly wrong in a way its own outputs wouldn't reveal. This module is the
opposite trade: no derivation to get wrong, at the cost of being useless at
real problem sizes. It's a trust anchor for validating src.lib.reweight_exact
against small, hand-picked or synthetic (a, f, beta) instances -- e.g.
tests/test_reweight_exact.py's K=4 non-global-KKT-point example, or a
shrunk-down version of a real dataset -- not a drop-in for it. (Historical
motivation: this project used to also have a NUMERIC solver, a multi-start
local optimizer; run with only 30 restarts, it was found to land on a local
optimum -- ESS 4.28 instead of the reachable 7.37 -- for one of jazh24's
leave-one-system-out subsets near beta~1. That module was retired after
broader cross-validation confirmed this wasn't an isolated case (13/15
real-data test cases underperformed src.lib.reweight_exact's certified
optimum at default settings); src.lib.reweight_exact replaced it as the
production solver, with this module as its trust anchor instead.)

Candidate generation is a combinadic (combinatorial-number-system)
unranking: every lattice point is produced directly from its integer rank
via closed-form arithmetic (a handful of torch.searchsorted calls against
small precomputed binomial-coefficient tables, one per composition slot),
batched as tensor ops with no per-point Python-level loop. An earlier
version of this module generated candidates via plain Python `itertools.
combinations_with_replacement`; profiling showed THAT loop, not the
vectorized evaluation math, dominated its wall-clock cost (~180K
points/sec on one CPU core) -- replacing it with the combinadic approach
turned that into ~15M+ points/sec with no accelerator involved at all, a
~50-90x win before any GPU is even in the picture (exact multiplier
depends on K and grid_n -- itertools' per-candidate cost scales with
grid_n itself, since each candidate is a grid_n-length tuple, not just
with the final candidate count). This was cross-validated extensively
against that itertools-based version before it was retired: identical
point-set coverage for small (K, grid_n) by direct enumeration, identical
answers (ESS, feasibility) across real WMT datasets spanning K=7-17 at a
variety of beta targets and grid_n, and a fine-grained 21-point beta
sweep at K=4 -- 0 mismatches across every comparison run.

Generation runs on CPU UNCONDITIONALLY, regardless of `device` -- measured
directly: torch.searchsorted (which the generator depends on) was found
~90x SLOWER on this machine's MPS backend than CPU for the same batch (8.2s
vs 0.09s for one call over 17M values against a length-200 table) --
apparently a poorly-optimized MPS kernel path in this torch version, not a
subtlety of this problem. Sending generation through MPS would make the
whole pipeline slower, not faster. Only the evaluation step (weighted
variance / beta / objective -- large matmuls and reductions, an actually
GPU-friendly shape) is dispatched to `device`.

Still fundamentally combinatorial (C(grid_n+K-1, K-1) points -- faster
generation raises the practical ceiling, it does not remove the
combinatorial wall), so this remains a small-K cross-check tool, not a
production solver.

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
Silicon) does not support float64 -- evaluation runs in float32 there,
float64 elsewhere (generation is integer arithmetic throughout,
unaffected). float32 is still far more precise than any `tol` this
brute-force method is used with (>= 1e-4), but a result computed on MPS
should not be expected to match a CPU/exact solver's answer to more than
~7 significant digits.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from src.lib.beta import beta_min_max

_DEFAULT_GRID_N = 60
_DEFAULT_TOL = 1e-3
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


@dataclasses.dataclass
class ExhaustiveWResult:
  """Result of one call to solve_w_exhaustive."""
  w: np.ndarray
  beta_target: float
  beta_achieved: float
  gap: float                # |beta_achieved - beta_target| -- check this before trusting
                              # `ess`: near beta~0 or beta~1 (see module docstring), beta(w)
                              # can be steep enough that a gap near `tol` already reaches a much
                              # easier, higher-ESS point than one actually at beta_target would.
  success: bool          # some lattice point satisfied |beta(w) - target| <= tol
  objective: float        # 1/2 * sum(w^2), the (P) objective
  ess: float               # 1 / sum(w^2)
  grid_n: int
  tol: float
  n_candidates: int        # total lattice points checked (the whole simplex lattice)
  n_feasible: int          # how many of those satisfied the tolerance
  message: str


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
  grid_n, batched with no per-point Python loop.

  Bijection (stars and bars): a composition (c_0..c_{K-1}) summing to
  grid_n corresponds to choosing K-1 "divider" positions d_1<...<d_{K-1}
  from M=grid_n+K-1 total slots (0-indexed), via
    c_0 = d_1, c_i = d_{i+1}-d_i-1 (0<i<K-1), c_{K-1} = (M-1)-d_{K-1}.
  The d's are found by standard combinadic unranking: for j=K-1 down to
  1, d_j is the largest v with C(v,j) <= remaining-rank (a vectorized
  torch.searchsorted against the precomputed C(*,j) table), then
  remaining -= C(d_j,j).
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


def solve_w_exhaustive(
    a: np.ndarray,
    f: np.ndarray,
    beta: float,
    grid_n: int = _DEFAULT_GRID_N,
    tol: float = _DEFAULT_TOL,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    chunk: int = _DEFAULT_CHUNK,
    device: str | None = 'smart',
) -> ExhaustiveWResult:
  """Checks literally every w on the K-dim simplex lattice of resolution
  grid_n (all w_i in {0, 1/grid_n, 2/grid_n, ..., 1} with sum(w)=1) against
  the balance constraint beta(w) == beta (within `tol`), and returns the
  one with the smallest objective (largest ESS) among those that pass --
  the brute-force answer to (P), no algorithmic insight applied anywhere.

  Raises ValueError up front (rather than hang) if the lattice would exceed
  `max_candidates` -- this method's cost is combinatorial in K and grid_n
  and that is never optimized away, so refusing is the honest behavior once
  the requested (K, grid_n) makes it infeasible to actually finish.

  `device` (see pick_device) only controls where the EVALUATION step runs;
  candidate generation is always CPU (see module docstring for why).
  Default is 'smart': CPU below _SMART_CANDIDATE_THRESHOLD lattice points,
  best available accelerator at or above it -- pass 'cpu'/'mps'/'cuda' to
  force one regardless of problem size.
  """
  a_np = np.asarray(a, dtype=float)
  f_np = np.asarray(f, dtype=float)
  K = len(a_np)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= beta <= 1.0):
    raise ValueError(f'beta={beta} outside [0,1]')
  beta_lo, beta_hi = beta_min_max(a_np, f_np)
  if not (beta_lo - tol <= beta <= beta_hi + tol):
    raise ValueError(f'beta={beta} outside reachable range [{beta_lo}, {beta_hi}] (Theorem "Reachable range")')

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
  f_t = torch.tensor(f_np, dtype=dtype, device=dev)

  best_obj = None
  best_w = None
  best_achieved = None
  best_gap = None
  n_feasible = 0

  for r0 in range(0, n_lattice, chunk):
    r1 = min(r0 + chunk, n_lattice)
    counts = _simplex_lattice_chunk(K, grid_n, r0, r1, cpu, binom_tables)
    w = counts.to(device=dev, dtype=dtype) / grid_n  # (n, K), simplex points by construction

    # Two-pass (mean-centered) weighted variance, vectorized -- the
    # single-pass identity (sum w*x^2 - mu^2) suffers catastrophic
    # cancellation exactly where it matters most (near-degenerate, low-ESS
    # w), and a "trusted" solver should never trade that away for speed.
    mu_a = w @ a_t
    mu_f = w @ f_t
    var_a = (w * (a_t.unsqueeze(0) - mu_a.unsqueeze(1)) ** 2).sum(dim=1)
    var_f = (w * (f_t.unsqueeze(0) - mu_f.unsqueeze(1)) ** 2).sum(dim=1)
    denom = var_a + var_f
    achieved = torch.where(denom > 0, var_a / denom, torch.full_like(denom, float('nan')))

    gap = (achieved - beta).abs()
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
      # so a failed search still reports something informative.
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
    gap = abs(best_achieved - beta)
    return ExhaustiveWResult(
        w=best_w, beta_target=beta, beta_achieved=best_achieved, gap=gap,
        success=True, objective=best_obj, ess=1.0 / (2.0 * best_obj),
        grid_n=grid_n, tol=tol, n_candidates=n_lattice, n_feasible=n_feasible,
        message=f'best of {n_feasible} lattice points within tol={tol} of beta={beta} '
                f'(actual gap {gap:.3g}, device={dev.type})',
    )
  gap = abs(best_achieved - beta) if best_achieved is not None else float('nan')
  return ExhaustiveWResult(
      w=best_w, beta_target=beta, beta_achieved=best_achieved, gap=gap,
      success=False, objective=float('nan'), ess=float('nan'),
      grid_n=grid_n, tol=tol, n_candidates=n_lattice, n_feasible=0,
      message=(f'no lattice point within tol={tol} of beta={beta} '
               f'(closest miss off by {best_gap:.3g}, device={dev.type}) -- raise grid_n or tol'),
  )
