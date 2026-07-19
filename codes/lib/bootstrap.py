"""Bootstrap resampling of systems within a single dataset -- the literal
reading of action_plan.md section 6.5's "draw many system subsets from the
real systems only" (as opposed to lib.reweighted_consistency's cross-year/
cross-pair generalization of "subset", used earlier for practical reasons).
Shared by compute_alpha_table_bootstrap.py and compute_bootstrap_reweight_
cache.py so both draw identical subsamples for a given (systems, n_sample,
n_repeats, seed) -- letting the consistency-curve experiment reuse exactly
the 30 samples an alpha_table_bootstrap_<dataset>.md run already produced.
"""

from __future__ import annotations

import numpy as np


def sample_subsets(systems: list[str], n_sample: int, n_repeats: int, seed: int) -> list[list[str]]:
  """n_repeats samples of n_sample systems each, drawn without replacement
  from `systems`, each sorted for determinism. One rng, drawn from
  sequentially across all repeats (matching compute_alpha_table_
  bootstrap.py's original loop) -- reusing the same (systems, n_sample,
  n_repeats, seed) always reproduces the identical sample sequence."""
  rng = np.random.default_rng(seed)
  return [sorted(rng.choice(systems, size=n_sample, replace=False).tolist()) for _ in range(n_repeats)]
