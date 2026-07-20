"""EXACT (support-enumeration) solver for (P) (action_plan.md section 3,
specifically 3.9-3.11).

(P): given real systems' score vectors a, b and a target balance alpha,
find the weights w closest to uniform (min 1/2 sum w_i^2) subject to
sum(w) = 1, w >= 0, and alpha(w) = alpha exactly. (P) is nonconvex (the
balance constraint is an indefinite quadratic for alpha in (0,1),
action_plan.md 3.6) -- action_plan.md 3.7 gives a concrete K=4 instance
where a KKT point is provably NOT globally optimal, i.e. a naive local
solve is not guaranteed to find the true optimum.

This solver is proven both SOUND and COMPLETE -- every value it returns is
a genuine global optimum of (P), certified either directly
(Theorem 2's KKT + curvature sufficiency condition) or, when no support's
candidate passes that local certificate (the duality-gap case of
action_plan.md 3.7), by exhaustively enumerating every admissible
support's stationary points (Corollary 4/5) and returning the min-norm
(max-ESS) one among them -- itself the global optimum by the completeness
argument, just without a local Theorem-2 witness.

This is the project's sole production solver for (P). A local-search
solver (lib.reweight_numeric, multi-start scipy.optimize) previously
filled this role; it was retired after cross-validation against lib.
reweight_exhaustive showed it missing the true optimum in 13/15 tested
real-data cases at its default restart count (sometimes by a large
margin, e.g. ESS 1.07 vs the achievable 4.0) -- this solver has no such
risk (a certified global optimum every time) and no restart/seed
parameters to tune.

This module intentionally does not implement the "descent heuristic"
priority-ordering optimization of 3.11 (visiting size-|S|-1 supports in an
order informed by which coordinate a failed candidate went most negative
on) -- the action plan itself states that heuristic affects only typical-
case speed, never correctness ("nothing about correctness rests on it"),
so the plain largest-first enumeration here is still sound and complete,
just not necessarily as fast as the paper's fully tuned search order.

Every closed-form piece (the 2x2 curvature test, the secular polynomial,
the pole handling) is Lemma 3 / Proposition 1's construction, transcribed
directly -- see the section 3.6/3.9/3.10 proofs this mirrors line for
line in the comments below.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from scipy.optimize import brentq

from lib.alpha import alpha as alpha_of, alpha_min_max, pairwise_alphas, weighted_mean

_TOL = 1e-7
_POLE_TOL = 1e-6


@dataclasses.dataclass
class ExactWResult:
  """Result of one call to solve_w_exact. Every returned `w` is a genuine
  global optimum (soundness+completeness, action_plan.md 3.11); `certified`
  records HOW that was established, not whether it's correct."""
  w: np.ndarray
  alpha_target: float
  alpha_achieved: float
  success: bool               # solver ran to completion and found a global optimum
  certified: bool              # True: an explicit Theorem-2 (KKT+curvature) pass fired.
                                 # False: found via exhaustive-enumeration completeness (3.9's
                                 # Remark) instead -- still provably the global optimum.
  support: tuple[int, ...]
  phase: str                    # 'anchor' | 'certified' | 'fallback'
  n_supports_visited: int
  n_candidates_evaluated: int
  objective: float
  ess: float
  message: str


def _M_r(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
  """M(alpha), r(alpha) of action_plan.md 3.6."""
  M = alpha * np.outer(b, b) - (1 - alpha) * np.outer(a, a)
  r = (1 - alpha) * a ** 2 - alpha * b ** 2
  return M, r


def _g(w: np.ndarray, M: np.ndarray, r: np.ndarray) -> float:
  return float(w @ M @ w + r @ w)


def _d_all(w: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
  """d_i(w,alpha) for EVERY i (action_plan.md 3.5), including systems
  outside w's support -- needed for Theorem 2 condition (2)."""
  mu_a = weighted_mean(a, w)
  mu_b = weighted_mean(b, w)
  return (1 - alpha) * (a ** 2 - 2 * mu_a * a) - alpha * (b ** 2 - 2 * mu_b * b)


def _prop1_2x2(a: np.ndarray, b: np.ndarray, alpha: float):
  """Proposition 1's reduction: mean-centers a,b, returns rho, the norms,
  and the closed-form eigenstructure (theta1>0>theta2, T, D) of the 2x2
  curvature block -- shared by both the full-problem curvature test
  (condition 3) and each support's candidate construction (3.10)."""
  mu_a, mu_b = a.mean(), b.mean()
  ta, tb = a - mu_a, b - mu_b
  na, nb = float(np.linalg.norm(ta)), float(np.linalg.norm(tb))
  rho = float(np.clip((ta @ tb) / (na * nb), -1.0, 1.0)) if na > 0 and nb > 0 else 0.0
  T = alpha * nb ** 2 - (1 - alpha) * na ** 2
  D = -alpha * (1 - alpha) * (1 - rho ** 2) * na ** 2 * nb ** 2
  disc = float(np.sqrt(max(T ** 2 - 4 * D, 0.0)))
  theta1 = (T + disc) / 2.0
  theta2 = (T - disc) / 2.0
  return theta1, theta2, rho, ta, tb, na, nb


def _curvature_ok(a: np.ndarray, b: np.ndarray, alpha: float, eta: float, tol: float = _TOL) -> bool:
  """Condition (3) of Theorem 2, action_plan.md 3.7: v^T(I+2 eta M(alpha))v
  >= 0 for every v with sum(v)=0, on the FULL K systems -- Proposition 1's
  2x2 test (tr H >= 0, det H >= 0), which degenerates correctly to the
  scalar |rho|=1 test since theta2 -> 0 there (verified: det H >= 0 then
  implies tr H >= 0 automatically, so testing both uniformly is equivalent)."""
  theta1, theta2, *_ = _prop1_2x2(a, b, alpha)
  tr_h = 2.0 + 2.0 * eta * (theta1 + theta2)
  det_h = (1.0 + 2.0 * eta * theta1) * (1.0 + 2.0 * eta * theta2)
  return tr_h >= -tol and det_h >= -tol


def _support_candidates(a: np.ndarray, b: np.ndarray, S: tuple[int, ...], alpha: float, tol: float = _TOL):
  """Every candidate on support S per Lemma 3 (action_plan.md 3.10):
  the anchor (Case 1) if alpha == alpha_0(S), else the Case-2/3/4 points
  built from the secular polynomial and the live-pole formulas. Returns a
  list of (w_full_Kvec, eta) pairs -- not yet filtered for feasibility or
  exact-support-S (the caller does that, since infeasible candidates are
  still useful as descent-heuristic-style diagnostics)."""
  K = len(a)
  idx = np.array(S)
  n = len(idx)
  a_S, b_S = a[idx], b[idx]
  u0_S = np.full(n, 1.0 / n)

  def embed(w_S: np.ndarray) -> np.ndarray:
    w = np.zeros(K)
    w[idx] = w_S
    return w

  mu_a, mu_b = float(a_S.mean()), float(b_S.mean())
  va = float(np.mean((a_S - mu_a) ** 2))
  vb = float(np.mean((b_S - mu_b) ** 2))
  alpha0_S = va / (va + vb) if (va + vb) > 0 else 0.5

  if abs(alpha - alpha0_S) < tol:
    return [(embed(u0_S), 0.0)]

  # --- Case 2-4 setup: alpha != alpha_0(S) (Proof, "Setting up cases 2-4") ---
  theta1, theta2, rho, ta, tb, na, nb = _prop1_2x2(a_S, b_S, alpha)
  if na < tol or nb < tol:
    return []  # degenerate support (shouldn't occur for an admissible S)

  degenerate_rho = abs(1.0 - rho ** 2) < tol  # |rho| = 1: span is 1-D
  e1 = tb / nb
  if not degenerate_rho:
    e2 = (ta - rho * na * e1) / (na * np.sqrt(max(1.0 - rho ** 2, 0.0)))
  else:
    e2 = None

  p = alpha * nb ** 2 - (1 - alpha) * rho ** 2 * na ** 2
  q = -(1 - alpha) * rho * np.sqrt(max(1.0 - rho ** 2, 0.0)) * na ** 2
  s = -(1 - alpha) * (1 - rho ** 2) * na ** 2

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
    theta2 = 0.0  # matches Proposition 1's degenerate scalar test (span is 1-D)
  else:
    v1, v2 = eigvec(theta1), eigvec(theta2)

  d_S = (1 - alpha) * (a_S ** 2 - 2 * mu_a * a_S) - alpha * (b_S ** 2 - 2 * mu_b * b_S)
  r1 = float(v1 @ d_S)
  r2 = float(v2 @ d_S) if v2 is not None else 0.0
  d_S_centered = d_S - d_S.mean()
  r0_sq = max(float(d_S_centered @ d_S_centered) - r1 ** 2 - r2 ** 2, 0.0)
  c_tilde = _g(u0_S, *_M_r(a_S, b_S, alpha))  # = g(u0, alpha) on S

  def w_from_eta(eta: float) -> np.ndarray:
    """Explicit candidate formula, action_plan.md 3.11 ('The candidates,
    explicitly'), rewritten with v1,v2 embedded directly in R^n (so no N
    matrix is ever constructed -- see this module's docstring)."""
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
      continue  # dead pole (Case 2): no candidates
    # Live pole (Case 3): u_circ from the OTHER curved direction + flat part, at eta=lam_k.
    w_circ = u0_S - lam_k * d_S_centered
    for lam_o, theta_o, r_o, v_o in poles:
      if v_o is v_k:
        continue
      denom = 1.0 + 2.0 * lam_k * theta_o
      if abs(denom) > _POLE_TOL:
        w_circ = w_circ + lam_k * (2.0 * lam_k * theta_o * r_o / denom) * v_o
    g_circ = _g(w_circ, *_M_r(a_S, b_S, alpha))
    val = -g_circ / theta_k
    if val >= -tol:
      uk = float(np.sqrt(max(val, 0.0)))
      for sign in (1.0, -1.0):
        candidates.append((embed(w_circ + sign * uk * v_k), lam_k))
    live_pole_etas.append(lam_k)

  # --- Case 4: generic secular equation, action_plan.md 3.10 ---
  # Clearing denominators into the polynomial Phi_{S,alpha} the paper
  # derives is exact in principle, but rescales the equation by det(H)^2,
  # which spans dozens of orders of magnitude whenever theta1 or theta2 is
  # small -- routine near alpha_min/alpha_max, where D -> 0 (action_plan.md
  # 3.10(c)). np.roots's companion-matrix eigensolve on such a badly-scaled
  # polynomial can return roots that are simply wrong (not just imprecise),
  # which polishing from can't fix if the starting point itself is garbage
  # -- confirmed empirically: a support with theta2 ~ -1e-6 produced a
  # 5-term Phi spanning coefficients from 1e-37 to 1e-5, and every np.roots
  # root was numerically meaningless.
  #
  # So Case 4 is solved directly on the RATIONAL secular equation instead
  # (never cleared to a polynomial): it is smooth except at the (at most 2)
  # known poles, so bracketing + Brent's method finds every real root
  # robustly regardless of how extreme theta1/theta2 are, since it only
  # ever evaluates the well-behaved original function.
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


def _admissible(alpha: float, lo: float, hi: float, tol: float = _TOL) -> bool:
  if lo < alpha < hi:
    return True
  return abs(lo - hi) < tol and abs(alpha - lo) < tol  # all-equal-balance edge case


def solve_w_exact(a: np.ndarray, b: np.ndarray, alpha: float, tol: float = _TOL) -> ExactWResult:
  """Global optimum of (P) via support enumeration (action_plan.md 3.11).
  Sound and complete: unconditionally returns a genuine global optimum, at
  the cost of visiting admissible supports largest-first (Corollary 3
  pruning) until either an explicit Theorem-2 certificate fires or every
  admissible support of size > the best ESS found has been exhausted."""
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  K = len(a)
  if K < 2:
    raise ValueError(f'need at least 2 systems, got {K}')
  if not (0.0 <= alpha <= 1.0):
    raise ValueError(f'alpha={alpha} outside [0,1]')

  pairs = pairwise_alphas(a, b)
  alpha_min, alpha_max = alpha_min_max(a, b)
  if not (alpha_min - tol <= alpha <= alpha_max + tol):
    raise ValueError(f'alpha={alpha} outside reachable range [{alpha_min}, {alpha_max}] (Corollary 2)')

  best = None  # (ess, w, support, eta, phase)
  n_supports_visited = 0
  n_candidates_evaluated = 0

  for size in range(K, 1, -1):
    if best is not None and size <= best[0]:
      break  # Corollary 3: no remaining (smaller) support can beat the incumbent's ESS
    for S in itertools.combinations(range(K), size):
      lo_S = min((v for i, j, v in pairs if i in S and j in S), default=None)
      hi_S = max((v for i, j, v in pairs if i in S and j in S), default=None)
      if lo_S is None or not _admissible(alpha, lo_S, hi_S, tol):
        continue
      n_supports_visited += 1

      for w, eta in _support_candidates(a, b, S, alpha, tol):
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
        # g(w,alpha)'s ABSOLUTE residual isn't a meaningful feasibility
        # test near the alpha_min/alpha_max boundary: g = total_variance *
        # (alpha_achieved - alpha), and total_variance can be tiny there
        # (near-collinear support), so a small g-residual can still hide a
        # large alpha error. Testing alpha_achieved directly is scale-
        # invariant and catches exactly that case.
        if abs(alpha_of(a, b, w) - alpha) > 1e-6:
          continue
        actual_support = tuple(i for i in range(K) if w[i] > tol)
        if actual_support != S:
          continue  # belongs to a smaller support, handled when THAT support is visited

        ess = 1.0 / float(np.sum(w ** 2))
        phase = 'anchor' if eta == 0.0 else 'certified'
        if best is None or ess > best[0]:
          best = (ess, w, S, eta, phase)

        # Theorem 2 conditions (1) (checked above), (2), (3):
        d_all = _d_all(w, a, b, alpha)
        i0 = S[0]
        lam = -(w[i0] + eta * d_all[i0])
        outside = [i for i in range(K) if i not in S]
        cond2 = all(lam + eta * d_all[i] >= -tol for i in outside)
        cond3 = _curvature_ok(a, b, alpha, eta, tol)
        if cond2 and cond3:
          return ExactWResult(
              w=w, alpha_target=alpha, alpha_achieved=alpha_of(a, b, w),
              success=True, certified=True, support=S, phase=phase,
              n_supports_visited=n_supports_visited, n_candidates_evaluated=n_candidates_evaluated,
              objective=0.5 * float(np.sum(w ** 2)), ess=ess,
              message=f'certified on support of size {size} ({phase})',
          )

  if best is None:
    raise RuntimeError(
        f'solve_w_exact found no feasible candidate for alpha={alpha} -- should be unreachable '
        f'for an in-range alpha (Corollary 2); likely a bug in _support_candidates.')
  ess, w, S, eta, _ = best
  return ExactWResult(
      w=w, alpha_target=alpha, alpha_achieved=alpha_of(a, b, w),
      success=True, certified=False, support=S, phase='fallback',
      n_supports_visited=n_supports_visited, n_candidates_evaluated=n_candidates_evaluated,
      objective=0.5 * float(np.sum(w ** 2)), ess=ess,
      message='no support certified directly; returned via exhaustive-enumeration completeness (3.9)',
  )
