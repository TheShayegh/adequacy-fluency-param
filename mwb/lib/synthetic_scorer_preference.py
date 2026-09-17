"""Adequacy/fluency PREFERENCE of a weighted meta-metric: for one base
base scorer s_0, build its synthetic A-family and F-family
(mwb.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid) over the full
dial grid ([0, 1] by 0.1, 11 points -- the complete dial sweep, real base scorer
at dial=0 through the pure aspect-conditional-mean scorer at dial=1), and
ask, at a fixed weighted meta-metric (e.g. weighted SPA at one beta):
across every pair of same-family dial scorers, how often does the
meta-metric prefer the MORE EXTREME one -- the dial further from the real
base scorer toward the pure aspect-conditional-mean scorer?

This preference score is what the paper's success-rate measures (Sec.
"Evaluation Setup": adequacy-over-fluency preference, MQM adherence,
explainability by MQM) reduce to -- see this module's third family below
for which preference maps to which paper measure.

  adequacy_preference(metametric, s_0) = preference_score of the A-family's
  scores under `metametric` -- averaged over all C(11,2)=55 same-family pairs.
  fluency_preference(metametric, s_0) is the F-family's mirror.

1.0 means the metametric always rewards pushing further toward pure
adequacy (fluency) response; 0.0 means it always penalizes it; 0.5 means no
systematic preference either way. Averaging these over every base scorer in a
dataset gives the dataset-level <adequacy|fluency>_preference(metametric; D).

A third family gives a third preference score the same way, but which
family depends on synthesis (mwb.lib.synthetic_scorer_beta_grid.
base_scorer_beta_dial_grid):

  - synthesis='additive'/'additive_mean': T-family (dialed on All MQM,
    t=a+f), preference-neutral by construction -- since t favors neither
    aspect, a well-behaved metametric should prefer the more-t-aligned
    dial at EVERY beta, not just cross over near beta_0(D) the way A/F
    do; this makes T a direction-unambiguous probe of whether ESS (not
    beta's own adequacy/fluency lean) is what governs whether the
    metametric recovers the known dial ordering. Restricted to the
    additive methods -- see mwb.lib.synthetic_scorers module docstring for why
    T isn't a meaningful construction under synthesis='offset'.
  - synthesis='offset': J-family (dialed on the JOINT (Adequacy, Fluency)
    pair, mwb.lib.synthetic_scorers.base_scorer_family_joint) gives
    joint_preference(metametric, s_0) instead.

Only one of T/J is ever present in a given base_scorer_beta_dial_grid result
(and hence a given cache), matching whichever synthesis built it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

DIAL_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# J restricted to its NEGATIVE half only (base_scorer_family_joint's dial<=0
# side -- 0.0 the real base scorer, more negative = more artificially-corrupted,
# since y - dial*ebar_k ADDS ebar_k for dial<0 instead of subtracting it,
# amplifying the base scorer's own deviation rather than erasing it), ordered by
# ASCENDING SIGNED DIAL VALUE: the most negative (most corrupted) point
# first, 0.0 (the real base scorer, unmodified) last. preference_score treats
# "later array index" as "the one a well-behaved metametric SHOULD prefer"
# (its own docstring: "1.0 if the metametric always prefers the more
# extreme dial"), which for the usual [0, 1] families coincides with "more
# extreme" because dial=1 there is the IMPROVED endpoint -- but J's
# negative range is the opposite: more negative is WORSE, 0.0 is the point
# a sound metametric should favor. Ranking by ascending EXTREMITY (0.0
# first) is backwards here -- confirmed the hard way: it made
# preference_score measure "prefers the more corrupted point," which came
# out near 0 for exactly the datasets/base scorers where the metametric was
# behaving CORRECTLY. Ranking by ascending SIGNED value instead (this
# ordering) fixes it: last index is 0.0, the point that should be
# preferred, matching every other family's "later index = the one worth
# preferring" convention. (Reversing an already-computed preference_score
# array algebraically gives EXACTLY 1 - the old score -- proven and
# confirmed empirically when this was first fixed.)
#
# dial(i) = 0.5 - 2^i for i in [-1, -0.9, ..., 1.9, 2] (31 points, step
# 0.1): geometric, packed toward dial=0 same as GEOM_DIAL_GRID/
# EXTENDED_ADDITIVE_MEAN_DIAL_GRID are for their own families, anchoring
# dial=0 EXACTLY at i=-1 (2^-1 == 0.5 exactly, so 0.5 - 0.5 == 0.0 exactly)
# and reaching dial=-3.5 at i=2 -- wider than the original [-2, 0] sweep,
# to reach a comparably extreme endpoint now that base_scorer_family_additive_
# mean's own endpoint (EXTENDED_ADDITIVE_MEAN_DIAL_GRID) is standardized
# to the base scorer's own scale rather than the aspect's raw, often much larger
# scale. `sorted(...)` applies the ascending-signed-value fix above --
# the raw generator (i ascending) produces DESCENDING dial values (0.0
# first, since dial(i) is a decreasing function of i), so it must be
# resorted, unlike EXTENDED_ADDITIVE_MEAN_DIAL_GRID, which is already
# ascending by construction.
NEG_J_DIAL_GRID = tuple(sorted(round(0.5 - 2.0 ** i, 10) for i in (-1.0 + 0.1 * k for k in range(31))))


def full_range_betas(beta_lo: float, beta_hi: float, step: float, eps_frac: float = 1e-3) -> list[float]:
  """Every clean multiple of `step` strictly inside [beta_lo, beta_hi],
  inset by eps_frac*(hi-lo) at each end (mwb.lib.beta.build_beta_grid's own
  convention -- the exact boundary admits only a single degenerate
  2-system support, so solve_w_exact is best kept off it). Snapped to
  clean multiples of `step` (e.g. 0.01, 0.02, ...) rather than starting
  exactly at the inset lo, for readable axis ticks."""
  eps = (beta_hi - beta_lo) * eps_frac
  lo, hi = beta_lo + eps, beta_hi - eps
  start = math.ceil(lo / step) * step
  n = int(math.floor((hi - start) / step + 1e-9))
  return [round(start + k * step, 10) for k in range(n + 1)]


def _synthesis_suffix(synthesis: str) -> str:
  """Shared by union_grid_tag/preference_tag: synthesis='offset' (the
  original generator) adds no suffix, keeping existing cache filenames from
  before this parameter existed valid; 'additive'/'additive_mean' (mwb.lib.
  synthetic_scorers) get their own suffix so they never collide with an
  offset-synthesis cache for the same dataset/grid/metametric."""
  return '' if synthesis == 'offset' else f'_{synthesis}'


def _dial_preset_suffix(dial_preset: str) -> str:
  """Shared by union_grid_tag/preference_tag: dial_preset='linear' (mwb.lib.
  synthetic_scorers.DIAL_GRID, the original 0/0.1/.../1 sweep) adds no
  suffix, keeping existing cache filenames valid; 'geometric' (mwb.lib.
  synthetic_scorers.GEOM_DIAL_GRID, {2^-i} packed toward dial=0) gets its
  own suffix -- '_geom'; 'extended' (mwb.lib.
  synthetic_scorers.EXTENDED_ADDITIVE_MEAN_DIAL_GRID, dial 0..~8, paired
  with base_scorer_family_additive_mean's standardized=True default) gets
  '_ext'; 'symmetric' (mwb.lib.synthetic_scorers.AFT_DIAL_GRID/AFTJ_J_DIAL_GRID,
  compute_scorer_preference_vs_beta.py's --family-set AFTJ) gets '_sym'
  -- so none of the four ever collides with another for the same
  dataset/synthesis/metametric."""
  return {'linear': '', 'geometric': '_geom', 'extended': '_ext', 'symmetric': '_sym'}[dial_preset]


def union_grid_tag(
    dataset: str, step: float, metametric: str = 'spa', synthesis: str = 'offset', dial_preset: str = 'linear',
) -> str:
  """Filename tag for mwb.lib.beta.union_beta_grid (the union of every
  pairwise beta_ij with a plain uniform [beta_min, beta_max] sweep at
  `step`) -- mirrors preference_tag's compute/plot filename contract, but
  never collides with it (different literal infix) for the same
  dataset/step/metametric/synthesis/dial_preset."""
  suffix = '' if metametric == 'spa' else f'_{metametric}'
  return f'{dataset}_union_s{step:g}{suffix}{_synthesis_suffix(synthesis)}{_dial_preset_suffix(dial_preset)}'


def preference_tag(
    dataset: str, n_steps: int, step: float, full_range: bool, metametric: str = 'spa', synthesis: str = 'offset',
    dial_preset: str = 'linear',
) -> str:
  """Filename tag shared by compute_scorer_preference_vs_beta.py (writer)
  and plot_scorer_preference_vs_beta.py (reader), so the same CLI flags
  resolve to the same cache file on both sides. metametric='spa' adds no
  suffix (keeps existing spa cache filenames from before this parameter
  existed valid); any other metametric (e.g. 'pa') gets its own suffix so
  it never collides with an spa cache for the same dataset/grid. Likewise
  synthesis='offset' (the original generator) and dial_preset='linear'
  (the original dial grid) add no suffix; other values get their own (see
  _synthesis_suffix/_dial_preset_suffix)."""
  suffix = '' if metametric == 'spa' else f'_{metametric}'
  suffix += _synthesis_suffix(synthesis) + _dial_preset_suffix(dial_preset)
  if full_range:
    return f'{dataset}_full_s{step:g}{suffix}'
  return f'{dataset}_n{n_steps}_s{step:g}{suffix}'


def save_preference_data(
    path: str, *, dataset: str, betas, center_beta: float, K: int, base_scorers: list[str],
    ess: np.ndarray, dial_grid, metametric: str = 'spa',
    synthesis: str = 'offset', A: np.ndarray | None = None, F: np.ndarray | None = None,
    AF: np.ndarray | None = None, T: np.ndarray | None = None, J: np.ndarray | None = None,
    dial_preset: str = 'linear',
) -> None:
  """Persists everything plot_scorer_preference_vs_beta.py needs. Each of
  A/F/AF/T/J is an optional (n_base_scorers, n_beta) matrix of per-base scorer
  preference curves, row order == `base scorers`. Either (A and F) or AF is
  expected (AF -- the paper's actual "adequacy over fluency" preference,
  mwb.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid's
  use_af_family=True family -- replaces the separate single-
  aspect A/F pair in the paper's committed setup; A/F alone are kept for
  the footnoted single-aspect-adherence use case and older caches).
  Exactly one of T (the preference-neutral All-MQM family,
  synthesis='additive'/'additive_mean') or J (the Joint family,
  synthesis='offset') is expected, matching base_scorer_beta_dial_grid's output
  for that synthesis -- both are optional, for backward compatibility with
  caches written before either existed (or before J replaced T for offset
  caches). ess is (n_beta,) ESS(w*(beta)) -- depends only on beta, not
  on base scorer, since w*(beta) is shared across every base scorer
  (mwb.lib.reweight_exact.solve_w_exact solved once per beta). metametric
  records which weighted meta-metric ('spa' or 'pa') the families were
  scored with. synthesis records which mwb.lib.synthetic_scorers dial
  family ('offset', 'additive', or 'additive_mean') built them (moot for
  AF, which is always additive_mean-style regardless of synthesis).
  dial_preset records which named dial grid ('linear' or 'geometric')
  `dial_grid`'s values came from, purely for display -- dial_grid itself
  already has the actual numbers."""
  extra = {}
  for label, arr in (('A', A), ('F', F), ('AF', AF), ('T', T), ('J', J)):
    if arr is not None:
      extra[label] = np.asarray(arr, dtype=float)
  np.savez(
      path, dataset=np.asarray(dataset), betas=np.asarray(betas, dtype=float),
      center_beta=np.asarray(float(center_beta)), K=np.asarray(int(K)),
      base_scorers=np.asarray(base_scorers, dtype='<U128'), ess=np.asarray(ess, dtype=float),
      dial_grid=np.asarray(dial_grid, dtype=float), metametric=np.asarray(metametric),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), **extra,
  )


def load_preference_data(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']),
      'betas': npz['betas'],
      'center_beta': float(npz['center_beta']),
      'K': int(npz['K']),
      'base_scorers': [str(d) for d in npz['base_scorers']],
      'metametric': str(npz['metametric']) if 'metametric' in npz else 'spa',
      'synthesis': str(npz['synthesis']) if 'synthesis' in npz else 'offset',
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'linear',
      'A': npz['A'] if 'A' in npz else None,
      'F': npz['F'] if 'F' in npz else None,
      'AF': npz['AF'] if 'AF' in npz else None,
      'T': npz['T'] if 'T' in npz else None,
      'J': npz['J'] if 'J' in npz else None,
      'ess': npz['ess'],
      'dial_grid': npz['dial_grid'],
  }


# Flip this to revert every caller to the all-pairs behavior at once.
#
# Why default True: preference_score originally averaged over EVERY
# C(n,2) pair. That's fine on a narrow, evenly-spaced grid, but once a
# grid's range gets wide (mwb.lib.synthetic_scorers.EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID, mwb.lib.synthetic_scorer_preference.NEG_J_DIAL_GRID -- both
# introduced to fix a DIFFERENT problem, additive_mean's raw-abar_k scale
# mismatch), most pairs end up comparing two dial points that are
# separated by a large gap (empirically: ~38% of EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID's 351 pairs differ by >1 SD, ~49% of NEG_J_DIAL_GRID's 465
# pairs). A large gap is a trivially easy "which one is more extreme"
# comparison for almost any metametric, regardless of whether it's
# actually good at fine discrimination -- so the average got dragged
# toward 1.0 by these easy pairs, diluting whatever the metametric does on
# the genuinely hard, CLOSE pairs (confirmed empirically: A/F/J all jumped
# toward ~1.0 once the grids were widened to fix the scale issue). Only
# scoring ADJACENT pairs (consecutive dial values, in dial order) removes
# every easy long-range comparison and measures specifically the
# metametric's ability to resolve neighboring, fine-grained differences --
# a harder, more informative question, and one that stays meaningful
# regardless of how wide the overall grid is.
PREFERENCE_ADJACENT_ONLY_DEFAULT = True


def preference_score(scores_by_dial: np.ndarray, adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT) -> float:
  """Fraction of pairs (i, j) with dial_i < dial_j (i.e. i, j are indices
  into an array already sorted by ascending dial) where scores_by_dial[j]
  (the more extreme dial) exceeds scores_by_dial[i] -- 1.0 if the
  metametric always prefers the more extreme dial, 0.0 if always the less
  extreme one, 0.5 per tied pair. NaN if fewer than 2 finite values.

  adjacent_only (default True, PREFERENCE_ADJACENT_ONLY_DEFAULT -- a
  toggle, not a permanent fork): if True, only ADJACENT pairs -- consecutive
  entries among the finite, dial-ordered values, i.e. (finite[0],
  finite[1]), (finite[1], finite[2]), ... -- are scored, not every C(n,2)
  pair (see PREFERENCE_ADJACENT_ONLY_DEFAULT's comment for why). If
  False, every pair counts (the original all-pairs reduction)."""
  vals = np.asarray(scores_by_dial, dtype=float)
  finite = np.flatnonzero(~np.isnan(vals))
  if len(finite) < 2:
    return float('nan')
  wins = 0.0
  n_pairs = 0
  pairs = zip(finite[:-1], finite[1:]) if adjacent_only else itertools.combinations(finite, 2)
  for i, j in pairs:  # i < j (adjacent_only: by construction; else since `finite` is sorted ascending)
    n_pairs += 1
    if vals[j] > vals[i]:
      wins += 1.0
    elif vals[j] == vals[i]:
      wins += 0.5
  return wins / n_pairs


def base_scorer_preference_by_beta(
    grids: dict, adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
) -> dict[str, np.ndarray]:
  """{'A': (n_beta,) adequacy_preference per beta, 'F': (n_beta,)
  fluency_preference per beta, 'AF': (n_beta,) the paper's actual
  "adequacy over fluency" preference (base_scorer_beta_dial_grid's
  use_af_family=True family, replacing A/F when present),
  'T' or 'J': (n_beta,) allmqm_preference or joint_preference per beta
  (whichever base_scorer_beta_dial_grid built, depending on synthesis)} for one
  base scorer, from mwb.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid's
  output -- one preference_score per column (beta), read off that
  column's dial-ordered SPA values. adjacent_only passed straight through
  to preference_score."""
  out = {}
  for label in ('A', 'F', 'AF', 'T', 'J'):
    if label not in grids:
      continue
    g = grids[label]  # (n_dial, n_beta), rows already in ascending dial order
    out[label] = np.array([preference_score(g[:, bi], adjacent_only=adjacent_only) for bi in range(g.shape[1])])
  return out
