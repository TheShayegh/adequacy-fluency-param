"""Diagnostic-only plot for tuning the reliability (ESS) opacity ramp used
by plot_scorer_orientation_vs_alpha.py's mean/shadow curves: x = ESS,
y = the opacity (alpha channel) that ramp actually assigns -- isolated from
the real orientation data, so the ramp's shape can be inspected and tuned
directly instead of squinting at its subtle effect on a busy data plot.

Two ramp families, --shape power|quad:

  power (the original): opacity = t**gamma, t = (ess/K - cutoff)/(1 - cutoff).
  Structurally has ZERO derivative at t=0 for any gamma>1 -- no choice of
  gamma can give a nonzero slope right at the cutoff, since d/dt(t^gamma)
  = gamma*t^(gamma-1) -> 0 as t->0 whenever gamma>1.

  quad: opacity = A*t**2 + (1-A)*t, the general quadratic through (0,0)
  and (1,1) with no cubic term -- i.e. EXACTLY zero third derivative
  (the lowest possible) for any A, and a constant second derivative 2*A.
  A=1 is a pure parabola (max curvature 2, but flat start, matching
  "power"'s gamma=2 case); A=0 is a plain linear ramp (zero curvature,
  full unit slope at both ends). A in (0,1) trades the two off: there is
  no A that simultaneously maximizes curvature AND the starting slope --
  moving A closer to 1 sharpens the curve overall (more spread near K)
  at the cost of flattening the start (less spread near the cutoff), and
  vice versa. This is a HARD ceiling, not a tuning limitation: fixing
  alpha(0)=0, alpha(1)=1 and forcing the cubic term to exactly zero pins
  the end slope to 1+A, which never exceeds 2 (at A=1) -- "quad" cannot
  ever be steeper at ESS=K than 2x the plain-linear rate.

  cubic: the unique cubic Hermite spline through (0,0)->(1,1) with
  INDEPENDENTLY chosen slopes m0 (at the cutoff) and m1 (at K) -- lets you
  have both a nonzero start slope AND an end slope steeper than "quad"
  can ever reach, at the cost of a nonzero (but constant -- still simpler
  than "power"'s t-dependent one) third derivative = 6*(m0+m1-2). "quad"
  is the special case m0=0, m1=2*A. Must keep 0 <= m0, m1 <= 3 for the
  cubic to stay monotonic (checked numerically below; a warning prints if
  it isn't).

  spline: a 2-segment cubic Hermite spline, knotted at t=0.5, with two
  extra free parameters beyond "cubic" -- the midpoint VALUE v_mid and
  the midpoint SLOPE m_mid (default: m0, i.e. keep the low start-slope
  going straight through the knot, rather than -- e.g. -- the average of
  m0/m1, which scales up with m1 and defeats the point). Pushing v_mid
  DOWN keeps the curve low and flat through the middle, which relaxes the
  per-segment monotonicity ceiling on the second half enough to support a
  much steeper m1 than a single cubic can reach at the same m0 --
  literally "paying" for the extra end-slope out of the middle, each
  segment checked (and warned) for monotonicity independently.

Usage: python codes/scripts/plot_ess_opacity_curve.py [--K 13] [--cutoff-frac 0.0769]
           [--shape spline --m0 0.1 --m1 6.0 --vmid 0.15]
           | [--shape cubic --m0 0.3 --m1 5.0] | [--shape quad --quad-weight 0.85]
           | [--shape power --gamma 4.0]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_scorer_orientation')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--K', type=float, default=13.0, help='system count (ESS ceiling)')
  parser.add_argument('--cutoff-frac', type=float, default=1.0 / 13,
                       help='fully-transparent cutoff as a fraction of K')
  parser.add_argument('--shape', choices=['power', 'quad', 'cubic', 'spline'], default='quad')
  parser.add_argument('--gamma', type=float, default=4.0, help='[power] exponent')
  parser.add_argument('--quad-weight', type=float, default=0.85,
                       help='[quad] A in A*t^2+(1-A)*t; 1=pure parabola, 0=linear')
  parser.add_argument('--m0', type=float, default=0.3, help='[cubic/spline] slope at the cutoff (ESS=1)')
  parser.add_argument('--m1', type=float, default=2.5, help='[cubic/spline] slope at K')
  parser.add_argument('--vmid', type=float, default=0.5,
                       help='[spline] opacity value at the midpoint (t=0.5); push below 0.5 to keep '
                            'the middle low/flat and buy headroom for a steeper m1')
  parser.add_argument('--mmid', type=float, default=None,
                       help='[spline] slope at the midpoint; default = m0 (keep the low slope going '
                            'through the knot, rather than the average of m0/m1 which scales up with '
                            'm1 and defeats the point)')
  args = parser.parse_args()

  K = args.K
  cutoff = args.cutoff_frac

  if args.shape == 'power':
    norm = PowerNorm(gamma=args.gamma, vmin=cutoff, vmax=1.0, clip=True)
    shape_desc = f'power: t^{args.gamma}'
  elif args.shape == 'quad':
    A = args.quad_weight

    def norm(x):
      t = np.clip((np.asarray(x, dtype=float) - cutoff) / (1.0 - cutoff), 0.0, 1.0)
      return A * t ** 2 + (1 - A) * t

    shape_desc = f'quad: A={A}*t^2 + {1 - A:.2f}*t (2nd deriv={2 * A:.2f} const, 3rd deriv=0)'
  elif args.shape == 'cubic':
    m0, m1 = args.m0, args.m1
    # Cubic Hermite basis: alpha(t) = 0*h00 + m0*h10 + 1*h01 + m1*h11.
    c3, c2, c1 = (m0 + m1 - 2.0), (3.0 - 2.0 * m0 - m1), m0

    def norm(x):
      t = np.clip((np.asarray(x, dtype=float) - cutoff) / (1.0 - cutoff), 0.0, 1.0)
      return c3 * t ** 3 + c2 * t ** 2 + c1 * t

    t_check = np.linspace(0, 1, 2000)
    deriv = 3 * c3 * t_check ** 2 + 2 * c2 * t_check + c1
    if deriv.min() < -1e-9:
      print(f'WARNING: non-monotonic for m0={m0}, m1={m1} (min slope {deriv.min():.4f} < 0) '
            f'-- opacity will dip and rise, keep both in [0,3] for a safe margin', file=sys.stderr)
    shape_desc = f'cubic: m0={m0} (slope at cutoff), m1={m1} (slope at K), 3rd deriv={6 * c3:.2f} const'

  else:  # spline
    m0, m1, v_mid = args.m0, args.m1, args.vmid
    m_mid = args.mmid if args.mmid is not None else m0

    def _hermite01(y0, y1, s0, s1, s):
      # s in [0,1], s0/s1 = dy/ds (LOCAL slope over this unit segment).
      c3 = 2 * (y0 - y1) + s0 + s1
      c2 = 3 * (y1 - y0) - 2 * s0 - s1
      c1 = s0
      return c3 * s ** 3 + c2 * s ** 2 + c1 * s + y0, (3 * c3, 2 * c2, c1)

    w = 0.5  # both segments span half the [0,1] range in t

    def norm(x):
      t = np.clip((np.asarray(x, dtype=float) - cutoff) / (1.0 - cutoff), 0.0, 1.0)
      out = np.empty_like(t)
      left = t <= 0.5
      s1_ = t[left] / w
      out[left] = _hermite01(0.0, v_mid, m0 * w, m_mid * w, s1_)[0]
      s2_ = (t[~left] - 0.5) / w
      out[~left] = _hermite01(v_mid, 1.0, m_mid * w, m1 * w, s2_)[0]
      return out

    s_check = np.linspace(0, 1, 2000)
    _, (a3, a2, a1) = _hermite01(0.0, v_mid, m0 * w, m_mid * w, s_check)
    deriv1 = (a3 * s_check ** 2 + a2 * s_check + a1) / w
    _, (b3, b2, b1) = _hermite01(v_mid, 1.0, m_mid * w, m1 * w, s_check)
    deriv2 = (b3 * s_check ** 2 + b2 * s_check + b1) / w
    if min(deriv1.min(), deriv2.min()) < -1e-9:
      print(f'WARNING: non-monotonic spline for m0={m0}, m1={m1}, vmid={v_mid} '
            f'(min slope {min(deriv1.min(), deriv2.min()):.4f} < 0)', file=sys.stderr)
    shape_desc = f'spline: m0={m0}, m1={m1}, vmid={v_mid} (mmid={m_mid:.2f})'

  ess = np.linspace(0.0, K, 1000)
  opacity = norm(ess / K)

  fig, ax = plt.subplots(figsize=(7.5, 5.5))
  ax.plot(ess, opacity, color='#1f4fd8', linewidth=2.5, zorder=3)
  ax.axvline(cutoff * K, color='black', linestyle=':', linewidth=1, alpha=0.7,
             label=f'cutoff = {cutoff:.3f}*K = {cutoff * K:.2f}')
  ax.axvline(K, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'K = {K:.2f}')
  ax.axhline(0, color='gray', linewidth=0.5, zorder=1)
  ax.axhline(1, color='gray', linewidth=0.5, zorder=1)

  ax.set_xlabel('ESS')
  ax.set_ylabel('opacity (alpha channel)')
  ax.set_title(f'Reliability ramp: opacity vs. ESS\n({shape_desc}, cutoff={cutoff:.4f}*K, K={K:.2f})',
               fontsize=11)
  ax.set_xlim(0, K)
  ax.set_ylim(-0.02, 1.02)
  ax.legend(fontsize=8, loc='upper left')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  fig.tight_layout()

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  plot_path = os.path.join(ARTIFACTS_DIR, 'ess_opacity_curve.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {plot_path}', file=sys.stderr)
