"""EXACT (support-enumeration) solver for Problem (P), the paper's
reweighting optimization (Sec. "Reweighting as an Optimization Problem",
Algorithm 1).

(P): given real systems' score vectors a, f and a target balance beta,
find the weights w closest to uniform (min 1/2 sum w_i^2) subject to
sum(w) = 1, w >= 0, and beta(w) = beta exactly. (P) is nonconvex: the
balance constraint is an indefinite quadratic in w for beta in (0,1) (the
matrix H in Theorem "Exit certificate"'s proof has one positive and one
negative eigenvalue), so a naive local solve is not guaranteed to find the
true optimum -- there exist instances where a KKT point is provably not
globally optimal (see tests/test_reweight_exact.py's adversarial K=4 case).

This solver is proven both SOUND and COMPLETE -- every value it returns is
a genuine global optimum of (P), certified either directly (Theorem
"Exit certificate"'s KKT + curvature sufficiency condition) or, when no
support's candidate passes that local certificate, by exhaustively
enumerating every admissible support's stationary points (Theorem "Bounded
candidate count") and returning the min-norm (max-ESS) one among them --
itself the global optimum by completeness, just without a local
Theorem-"Exit certificate" witness.

This is the project's sole production solver for (P). An earlier
multi-start local-search solver (scipy.optimize-based) filled this role;
it was retired after cross-validation against mwb.lib.reweight_exhaustive
(the brute-force trust anchor) showed it missing the true optimum in
13/15 tested real-data cases at its default restart count (sometimes by a
large margin, e.g. ESS 1.07 vs the achievable 4.0) -- this solver has no
such risk (a certified global optimum every time) and no restart/seed
parameters to tune.

This module intentionally does not implement a "descent heuristic"
priority-ordering optimization for visiting size-|S|-1 supports in an order
informed by which coordinate a failed candidate went most negative on --
that heuristic would only affect typical-case speed, never correctness, so
the plain largest-first enumeration here is still sound and complete, just
not necessarily as fast as it could be with a tuned search order.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from scipy.optimize import brentq

from mwb.lib.beta import beta as beta_of, beta_min_max, pairwise_betas, weighted_mean

_TOL = 1e-7
_POLE_TOL = 1e-6


@dataclasses.dataclass
class ExactWResult:
  """Result of one call to solve_w_exact. Every returned `w` is a genuine
  global optimum (soundness+completeness); `certified` records HOW that
  was established, not whether it's correct."""
  w: np.ndarray
  beta_target: float
  beta_achieved: float
  success: bool               # solver ran to completion and found a global optimum
  certified: bool              # True: an explicit Theorem "Exit certificate" (KKT+curvature) pass fired.
                                 # False: found via exhaustive-enumeration completeness instead -- still
                                 # provably the global optimum.
  support: tuple[int, ...]
  phase: str                    # 'anchor' | 'certified' | 'fallback'
  n_supports_visited: int
  n_candidates_evaluated: int
  objective: float
  ess: float
  message: str


def _M_r(a: np.ndarray, f: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
  """M(beta), r(beta): the quadratic form and linear term of the balance
  constraint b(w,beta) = w^T M w + w^T r (Theorem "Bounded candidate
  count"'s proof)."""
  M = beta * np.outer(f, f) - (1 - beta) * np.outer(a, a)
  r = (1 - beta) * a ** 2 - beta * f ** 2
  return M, r


def _g(w: np.ndarray, M: np.ndarray, r: np.ndarray) -> float:
  return float(w @ M @ w + r @ w)


def _d_all(w: np.ndarray, a: np.ndarray, f: np.ndarray, beta: float) -> np.ndarray:
  """d_i(w,beta) = the per-system gradient term of the balance constraint,
  for EVERY system i (including ones outside w's support) -- needed for the
  Theorem "Exit certificate" KKT condition (ii), which must hold at every
  i not in the support."""
  mu_a = weighted_mean(a, w)
  mu_f = weighted_mean(f, w)
  return (1 - beta) * (a ** 2 - 2 * mu_a * a) - beta * (f ** 2 - 2 * mu_f * f)


def _prop1_2x2(a: np.ndarray, f: np.ndarray, beta: float):
  """Mean-centers a,f, returns rho, the norms, and the closed-form
  eigenstructure (theta1>0>theta2) of the 2x2 curvature block spanned by
  the centered a,f directions -- shared by both the full-problem curvature
  test (Theorem "Exit certificate" condition (iii)) and each support's
  candidate construction below."""
  mu_a, mu_f = a.mean(), f.mean()
  ta, tf = a - mu_a, f - mu_f
  na, nf = float(np.linalg.norm(ta)), float(np.linalg.norm(tf))
  rho = float(np.clip((ta @ tf) / (na * nf), -1.0, 1.0)) if na > 0 and nf > 0 else 0.0
  T = beta * nf ** 2 - (1 - beta) * na ** 2
  D = -beta * (1 - beta) * (1 - rho ** 2) * na ** 2 * nf ** 2
  disc = float(np.sqrt(max(T ** 2 - 4 * D, 0.0)))
  theta1 = (T + disc) / 2.0
  theta2 = (T - disc) / 2.0
  return theta1, theta2, rho, ta, tf, na, nf


def _curvature_ok(a: np.ndarray, f: np.ndarray, beta: float, eta: float, tol: float = _TOL) -> bool:
  """Theorem "Exit certificate" condition (iii): v^T(I+2*eta*M(beta))v >= 0
  for every v with sum(v)=0, on the FULL K systems. Reduces to a 2x2
  positive-semidefinite test (tr H >= 0, det H >= 0) on the curvature
  block from _prop1_2x2, which degenerates correctly to the scalar |rho|=1
  case since theta2 -> 0 there (verified: det H >= 0 then implies
  tr H >= 0 automatically, so testing both uniformly is equivalent)."""
  theta1, theta2, *_ = _prop1_2x2(a, f, beta)
  tr_h = 2.0 + 2.0 * eta * (theta1 + theta2)
  det_h = (1.0 + 2.0 * eta * theta1) * (1.0 + 2.0 * eta * theta2)
  return tr_h >= -tol and det_h >= -tol


def _support_candidates(a: np.ndarray, f: np.ndarray, S: tuple[int, ...], beta: float, tol: float = _TOL):
  """Every stationary candidate on support S per Theorem "Bounded candidate
  count": the uniform anchor if beta == beta_0(S), else the points built
  from the secular equation and the live-pole formulas (that theorem's
  proof, Case 2). Returns a list of (w_full_Kvec, eta) pairs -- not yet
  filtered for feasibility or exact-support-S (the caller does that, since
  infeasible candidates are still useful diagnostically)."""
  K = len(a)
  idx = np.array(S)
  n = len(idx)
  a_S, f_S = a[idx], f[idx]
  u0_S = np.full(n, 1.0 / n)

  def embed(w_S: np.ndarray) -> np.ndarray:
    w = np.zeros(K)
    w[idx] = w_S
    return w

  mu_a, mu_f = float(a_S.mean()), float(f_S.mean())
  va = float(np.mean((a_S - mu_a) ** 2))
  vf = float(np.mean((f_S - mu_f) ** 2))
  beta0_S = va / (va + vf) if (va + vf) > 0 else 0.5

  if abs(beta - beta0_S) < tol:
    return [(embed(u0_S), 0.0)]

  # --- Case 2-4 setup: beta != beta_0(S) ---
  theta1, theta2, rho, ta, tf, na, nf = _prop1_2x2(a_S, f_S, beta)
  if na < tol or nf < tol:
    return []  # degenerate support (shouldn't occur for an admissible S)

  degenerate_rho = abs(1.0 - rho ** 2) < tol  # |rho| = 1: span is 1-D
  e1 = tf / nf
  if not degenerate_rho:
    e2 = (ta - rho * na * e1) / (na * np.sqrt(max(1.0 - rho ** 2, 0.0)))
  else:
    e2 = None

  p = beta * nf ** 2 - (1 - beta) * rho ** 2 * na ** 2
  q = -(1 - beta) * rho * np.sqrt(max(1.0 - rho ** 2, 0.0)) * na ** 2
  s = -(1 - beta) * (1 - rho ** 2) * na ** 2

  def eigvec(theta_k: float) -> np.ndarray:
    if e2 is None:
      return e1.copy()
    coord = np.array([q, theta_k - p])
    nrm = float(np.linalg.norm(coord))
    if nrm < tol:
      coord = np.array([1.0, 0.0])
      nrm = 1.0
    coord = coord / nrm
    return coord[0] * e1 + coord[1] * e2

  if degenerate_rho:
    v1, v2 = e1.copy(), None
    theta2 = 0.0  # matches the degenerate scalar test (span is 1-D)
  else:
    v1, v2 = eigvec(theta1), eigvec(theta2)

  d_S = (1 - beta) * (a_S ** 2 - 2 * mu_a * a_S) - beta * (f_S ** 2 - 2 * mu_f * f_S)
  r1 = float(v1 @ d_S)
  r2 = float(v2 @ d_S) if v2 is not None else 0.0
  d_S_centered = d_S - d_S.mean()
  r0_sq = max(float(d_S_centered @ d_S_centered) - r1 ** 2 - r2 ** 2, 0.0)
  c_tilde = _g(u0_S, *_M_r(a_S, f_S, beta))  # = g(u0, beta) on S

  def w_from_eta(eta: float) -> np.ndarray:
    """Explicit candidate formula (Theorem "Bounded candidate count"'s
    proof, Case 2.3), rewritten with v1,v2 embedded directly in R^n (no
    explicit null-space basis matrix ever constructed)."""
    w_S = u0_S - eta * d_S_centered
    for theta_k, r_k, v_k in ((theta1, r1, v1), (theta2, r2, v2)):
      if v_k is None:
        continue
      denom = 1.0 + 2.0 * eta * theta_k
      if abs(denom) < _POLE_TOL:
        continue
      w_S = w_S + eta * (2.0 * eta * theta_k * r_k / denom) * v_k
    return w_S

  candidates: list[tuple[np.ndarray, float]] = []
  poles = [(-1.0 / (2.0 * theta1), theta1, r1, v1)]
  if v2 is not None and abs(theta2) > tol:
    poles.append((-1.0 / (2.0 * theta2), theta2, r2, v2))

  live_pole_etas = []
  for lam_k, theta_k, r_k, v_k in poles:
    if abs(r_k) > tol * max(1.0, abs(theta_k)):
      continue  # dead pole: no candidates
    # Live pole: u_circ from the OTHER curved direction + flat part, at eta=lam_k.
    w_circ = u0_S - lam_k * d_S_centered
    for lam_o, theta_o, r_o, v_o in poles:
      if v_o is v_k:
        continue
      denom = 1.0 + 2.0 * lam_k * theta_o
      if abs(denom) > _POLE_TOL:
        w_circ = w_circ + lam_k * (2.0 * lam_k * theta_o * r_o / denom) * v_o
    g_circ = _g(w_circ, *_M_r(a_S, f_S, beta))
    val = -g_circ / theta_k
    if val >= -tol:
      uk = float(np.sqrt(max(val, 0.0)))
      for sign in (1.0, -1.0):
        candidates.append((embed(w_circ + sign * uk * v_k), lam_k))
    live_pole_etas.append(lam_k)

  # --- Generic case: secular equation (Theorem "Bounded candidate count"'s
  # proof, degree-<=5 polynomial Phi_{S,beta}) ---
  # Clearing denominators into that polynomial is exact in principle, but
  # rescales the equation by det(H)^2, which spans dozens of orders of
  # magnitude whenever theta1 or theta2 is small -- routine near
  # beta_min/beta_max, where the discriminant D -> 0. np.roots's
  # companion-matrix eigensolve on such a badly-scaled polynomial can
  # return roots that are simply wrong (not just imprecise), which
  # polishing can't fix if the starting point itself is garbage --
  # confirmed empirically: a support with theta2 ~ -1e-6 produced a 5-term
  # Phi spanning coefficients from 1e-37 to 1e-5, and every np.roots root
  # was numerically meaningless.
  #
  # So this case is solved directly on the RATIONAL secular equation
  # instead (never cleared to a polynomial): it is smooth except at the
  # (at most 2) known poles, so bracketing + Brent's method finds every
  # real root robustly regardless of how extreme theta1/theta2 are, since
  # it only ever evaluates the well-behaved original function.
  def secular_rational(eta: float) -> float:
    total = c_tilde - eta * r0_sq
    for theta_k, r_k in poles_thetas_rs:
      denom = 1.0 + 2.0 * eta * theta_k
      if abs(denom) < 1e-300:
        return float('nan')
      total -= eta * (r_k ** 2 * (1.0 + theta_k * eta)) / denom ** 2
    return total

  def secular_rational_vec(etas: np.ndarray) -> np.ndarray:
    total = c_tilde - etas * r0_sq
    for theta_k, r_k in poles_thetas_rs:
      denom = 1.0 + 2.0 * etas * theta_k
      total = total - etas * (r_k ** 2 * (1.0 + theta_k * etas)) / denom ** 2
    return total

  poles_thetas_rs = [(theta1, r1)] + ([(theta2, r2)] if v2 is not None else [])
  pole_locs = sorted(lam for lam, *_ in poles)

  # Scan eta segment by segment, each anchored at the REAL pole location(s)
  # bounding it, rather than one global compactification over the whole
  # real line. A single global arctan grid (eta = tan(phi)) puts almost all
  # its resolution beyond the outermost pole and almost none near a pole
  # that's merely large-but-finite (e.g. a root at eta=528 with the
  # bounding pole at eta=713: arctan(528) and arctan(713) differ by under
  # 0.001 rad out of a total range of pi, so an 800-point uniform-in-phi
  # grid can step clean over the entire window without landing a single
  # sample inside it -- confirmed empirically as the cause of missed
  # candidates). Per-segment anchoring fixes this: the window between two
  # finite poles is scanned on a plain uniform grid (it's already bounded,
  # no compactification needed), and each unbounded outer segment gets its
  # own compactification anchored at ITS pole, so resolution concentrates
  # correctly regardless of that pole's absolute scale.
  n_window, n_outer = 400, 300
  segments = []
  if len(pole_locs) == 2:
    lo_p, hi_p = pole_locs
    eps_w = (hi_p - lo_p) * 1e-9
    segments.append(np.linspace(lo_p + eps_w, hi_p - eps_w, n_window))
    phi = np.linspace(1e-7, 0.5 * np.pi * (1 - 1e-9), n_outer)
    segments.append((lo_p - np.tan(phi))[::-1])
    segments.append(hi_p + np.tan(phi))
  elif len(pole_locs) == 1:
    p = pole_locs[0]
    phi = np.linspace(1e-7, 0.5 * np.pi * (1 - 1e-9), n_outer)
    segments.append((p - np.tan(phi))[::-1])
    segments.append(p + np.tan(phi))
  else:
    phi = np.linspace(-0.5 * np.pi * (1 - 1e-9), 0.5 * np.pi * (1 - 1e-9), n_window)
    segments.append(np.tan(phi))

  for etas_scan in segments:
    vals_scan = secular_rational_vec(etas_scan)
    prev_eta = prev_val = None
    for eta, val in zip(etas_scan, vals_scan):
      if val != val:
        prev_eta = prev_val = None
        continue
      if prev_val is not None:
        if prev_val == 0.0:
          candidates.append((embed(w_from_eta(prev_eta)), prev_eta))
        elif prev_val * val < 0.0:
          try:
            eta_root = brentq(secular_rational, prev_eta, eta, xtol=1e-15, rtol=1e-14, maxiter=200)
            candidates.append((embed(w_from_eta(eta_root)), eta_root))
          except (ValueError, RuntimeError):
            pass
      prev_eta, prev_val = eta, val

  return candidates


def _admissible(beta: float, lo: float, hi: float, tol: float = _TOL) -> bool:
  """Theorem "Admissible supports": a support S can carry a feasible
  weighting for target beta only if beta is between S's own smallest and
  largest pairwise balance."""
  if lo < beta < hi:
    return True
  return abs(lo - hi) < tol and abs(beta - lo) < tol  # all-equal-balance edge case


def solve_w_exact(a: np.ndarray, f: np.ndarray, beta: float, tol: float = _TOL) -> ExactWResult:
  """Global optimum of (P) via support enumeration (Algorithm 1). Sound and
  complete: unconditionally returns a genuine global optimum, at the cost
  of visiting admissible supports largest-first (Theorem "Support size
  bound" pruning) until either an explicit Theorem "Exit certificate"
  fires or every admissible support of size > the best ESS found has been
  exhausted."""
  a = np.asarray(a, dtype=float)
  f = np.asarray(f, dtype=float)
  K = len(a)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= beta <= 1.0):
    raise ValueError(f'beta={beta} outside [0,1]')

  pairs = pairwise_betas(a, f)
  beta_min, beta_max = beta_min_max(a, f)
  if not (beta_min - tol <= beta <= beta_max + tol):
    raise ValueError(f'beta={beta} outside reachable range [{beta_min}, {beta_max}] (Theorem "Reachable range")')

  # Boundary shortcut: at beta == beta_min or beta_max exactly, no
  # support of size >= 3 can EVER be admissible -- a non-degenerate support
  # of that size only reaches the OPEN interval (beta_min(S), beta_max(S)),
  # never its own endpoint, let alone the global one. The search below
  # would still visit (and correctly reject) every subset of every size
  # from K down to 3 to discover this, which is pure overhead: _admissible
  # rescans all pairs per subset, and there are exponentially many subsets
  # in the middle sizes. Skip straight to the answer when it's unambiguous:
  # on the segment {w: w_i+w_j=1, w>=0} of a two-system support, beta(w)
  # is CONSTANT and 1/2||w||^2 is minimized at the uniform split, so
  # w=(0.5,0.5) is the unique global optimum whenever a SINGLE pair attains
  # the extremal value. A tie among >= 2 pairs could in principle be beaten
  # by a larger all-equal-balance support (3+ systems with proportional
  # centered score vectors), so ties fall through to the general search,
  # which already handles that case correctly via largest-first order.
  if beta_max - beta_min > tol:
    for target in (beta_min, beta_max):
      if abs(beta - target) > tol:
        continue
      tied = [(i, j) for i, j, v in pairs if abs(v - target) <= tol]
      if len(tied) != 1:
        break  # ambiguous -- fall through to the general search
      i, j = tied[0]
      w = np.zeros(K)
      w[i] = w[j] = 0.5
      which = 'beta_min' if target == beta_min else 'beta_max'
      return ExactWResult(
          w=w, beta_target=beta, beta_achieved=beta_of(a, f, w),
          success=True, certified=True, support=(i, j), phase='anchor',
          n_supports_visited=1, n_candidates_evaluated=1,
          objective=0.5 * float(np.sum(w ** 2)), ess=2.0,
          message=f'boundary shortcut: beta == {which}, attained uniquely by pair {(i, j)}',
      )

  best = None  # (ess, w, support, eta, phase)
  n_supports_visited = 0
  n_candidates_evaluated = 0

  for size in range(K, 1, -1):
    if best is not None and size <= best[0]:
      break  # Theorem "Support size bound": no remaining (smaller) support can beat the incumbent's ESS
    for S in itertools.combinations(range(K), size):
      lo_S = min((v for i, j, v in pairs if i in S and j in S), default=None)
      hi_S = max((v for i, j, v in pairs if i in S and j in S), default=None)
      if lo_S is None or not _admissible(beta, lo_S, hi_S, tol):
        continue
      n_supports_visited += 1

      for w, eta in _support_candidates(a, f, S, beta, tol):
        n_candidates_evaluated += 1
        support_mask = np.zeros(K, dtype=bool)
        support_mask[list(S)] = True
        if np.any(w[support_mask] <= -tol):
          continue  # left the simplex: not a valid candidate for this support
        if np.any(w[~support_mask] > tol):
          continue  # shouldn't happen (embed() zeros outside S), safety net
        if abs(float(w.sum()) - 1.0) > 1e-3:
          continue  # budget grossly violated -- not a valid candidate
        w = np.clip(w, 0.0, None)
        w_sum = float(w.sum())
        if w_sum <= 0:
          continue
        w = w / w_sum
        # g(w,beta)'s ABSOLUTE residual isn't a meaningful feasibility
        # test near the beta_min/beta_max boundary: g = total_variance *
        # (beta_achieved - beta), and total_variance can be tiny there
        # (near-collinear support), so a small g-residual can still hide a
        # large beta error. Testing beta_achieved directly is scale-
        # invariant and catches exactly that case.
        if abs(beta_of(a, f, w) - beta) > 1e-6:
          continue
        actual_support = tuple(i for i in range(K) if w[i] > tol)
        if actual_support != S:
          continue  # belongs to a smaller support, handled when THAT support is visited

        ess = 1.0 / float(np.sum(w ** 2))
        phase = 'anchor' if eta == 0.0 else 'certified'
        if best is None or ess > best[0]:
          best = (ess, w, S, eta, phase)

        # Theorem "Exit certificate" conditions (i) (checked above), (ii), (iii):
        d_all = _d_all(w, a, f, beta)
        i0 = S[0]
        lam = -(w[i0] + eta * d_all[i0])
        outside = [i for i in range(K) if i not in S]
        cond2 = all(lam + eta * d_all[i] >= -tol for i in outside)
        cond3 = _curvature_ok(a, f, beta, eta, tol)
        if cond2 and cond3:
          return ExactWResult(
              w=w, beta_target=beta, beta_achieved=beta_of(a, f, w),
              success=True, certified=True, support=S, phase=phase,
              n_supports_visited=n_supports_visited, n_candidates_evaluated=n_candidates_evaluated,
              objective=0.5 * float(np.sum(w ** 2)), ess=ess,
              message=f'certified on support of size {size} ({phase})',
          )

  if best is None:
    raise RuntimeError(
        f'solve_w_exact found no feasible candidate for beta={beta} -- should be unreachable '
        f'for an in-range beta (Theorem "Reachable range"); likely a bug in _support_candidates.')
  ess, w, S, eta, _ = best
  return ExactWResult(
      w=w, beta_target=beta, beta_achieved=beta_of(a, f, w),
      success=True, certified=False, support=S, phase='fallback',
      n_supports_visited=n_supports_visited, n_candidates_evaluated=n_candidates_evaluated,
      objective=0.5 * float(np.sum(w ** 2)), ess=ess,
      message='no support certified directly; returned via exhaustive-enumeration completeness',
  )
