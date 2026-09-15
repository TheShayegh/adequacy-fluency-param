"""Regression tests for mwb.lib.reweight_exact.solve_w_exact.

Run: python -m unittest tests.test_reweight_exact
"""

import unittest

import numpy as np

from mwb.lib.alpha import alpha as alpha_of
from mwb.lib.reweight_exact import solve_w_exact


class TestAdversarialInstance(unittest.TestCase):
  """A hand-picked K=4 instance where the target alpha is NOT attained by
  the full-support KKT point -- the true global optimum sits on a smaller
  support instead, so a solver that stops at the first KKT point it finds
  (rather than exhaustively checking completeness, as solve_w_exact does)
  would silently return the wrong answer here. Found during solver
  cross-validation against the brute-force exhaustive solver."""

  A = np.array([0.5574, 0.9883, 1.4687, 3.0894])
  B = np.array([0.4012, 0.5353, 1.4986, 1.2467])
  TARGET_ALPHA = 0.9095
  EXPECTED_W = np.array([0.0088, 0.0861, 0.4245, 0.4806])

  def test_recovers_known_global_optimum(self):
    r = solve_w_exact(self.A, self.B, self.TARGET_ALPHA)
    np.testing.assert_allclose(r.w, self.EXPECTED_W, atol=1e-3)

  def test_not_certified_via_full_support_kkt(self):
    # The whole point of this instance: no support's stationary point
    # passes the Theorem "Exit certificate" test directly, so the solver
    # must fall back to exhaustive-enumeration completeness to find it.
    r = solve_w_exact(self.A, self.B, self.TARGET_ALPHA)
    self.assertFalse(r.certified)
    self.assertEqual(r.phase, 'fallback')

  def test_achieves_target_alpha(self):
    r = solve_w_exact(self.A, self.B, self.TARGET_ALPHA)
    self.assertAlmostEqual(alpha_of(self.A, self.B, r.w), self.TARGET_ALPHA, places=5)


class TestBasicProperties(unittest.TestCase):
  """Sanity checks that should hold for any solve_w_exact call: the
  returned weights are a valid probability vector and actually achieve
  the requested balance."""

  A = np.array([0.5574, 0.9883, 1.4687, 3.0894])
  B = np.array([0.4012, 0.5353, 1.4986, 1.2467])

  def test_weights_are_a_simplex_point(self):
    for target in (0.2, 0.5, 0.8):
      r = solve_w_exact(self.A, self.B, target)
      self.assertTrue(np.all(r.w >= -1e-9))
      self.assertAlmostEqual(float(r.w.sum()), 1.0, places=6)

  def test_uniform_weight_is_optimal_at_natural_alpha(self):
    K = len(self.A)
    uniform = np.full(K, 1.0 / K)
    alpha_0 = alpha_of(self.A, self.B, uniform)
    r = solve_w_exact(self.A, self.B, alpha_0)
    np.testing.assert_allclose(r.w, uniform, atol=1e-6)
    self.assertAlmostEqual(r.ess, float(K), places=4)

  def test_out_of_range_alpha_raises(self):
    with self.assertRaises(ValueError):
      solve_w_exact(self.A, self.B, 2.0)


if __name__ == '__main__':
  unittest.main()
