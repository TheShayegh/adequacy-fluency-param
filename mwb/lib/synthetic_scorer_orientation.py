"""Adequacy/fluency ORIENTATION of a weighted meta-metric: for one base
donor scorer s_0, build its synthetic A-family and B-family
(mwb.lib.synthetic_scorer_alpha_grid.donor_alpha_dial_grid) over the full
dial grid ([0, 1] by 0.1, 11 points -- the complete dial sweep, real donor
at dial=0 through the pure aspect-conditional-mean scorer at dial=1), and
ask, at a fixed weighted meta-metric (e.g. weighted SPA at one alpha):
across every pair of same-family dial scorers, how often does the
meta-metric prefer the MORE EXTREME one -- the dial further from the real
donor toward the pure aspect-conditional-mean scorer?

This orientation score is what the paper's success-rate measures (Sec.
"Evaluation Setup": adequacy-over-fluency preference, MQM adherence,
explainability by MQM) reduce to -- see this module's third family below
for which orientation maps to which paper measure.

  adequacy_orientation(metametric, s_0) = orientation_score of the A-family's
  scores under `metametric` -- averaged over all C(11,2)=55 same-family pairs.
  fluency_orientation(metametric, s_0) is the B-family's mirror.

1.0 means the metametric always rewards pushing further toward pure
adequacy (fluency) response; 0.0 means it always penalizes it; 0.5 means no
systematic preference either way. Averaging these over every donor in a
dataset gives the dataset-level <adequacy|fluency>_orientation(metametric; D).

A third family gives a third orientation score the same way, but which
family depends on synthesis (mwb.lib.synthetic_scorer_alpha_grid.
donor_alpha_dial_grid):

  - synthesis='additive'/'additive_mean': T-family (dialed on All MQM,
    t=a+b), orientation-neutral by construction -- since t favors neither
    aspect, a well-behaved metametric should prefer the more-t-aligned
    dial at EVERY alpha, not just cross over near alpha_0(D) the way A/B
    do; this makes T a direction-unambiguous probe of whether ESS (not
    alpha's own adequacy/fluency lean) is what governs whether the
    metametric recovers the known dial ordering. Restricted to the
    additive methods -- see mwb.lib.synthetic_scorers module docstring for why
    T isn't a meaningful construction under synthesis='offset'.
  - synthesis='offset': J-family (dialed on the JOINT (Adequacy, Fluency)
    pair, mwb.lib.synthetic_scorers.donor_family_joint) gives
    joint_orientation(metametric, s_0) instead.

Only one of T/J is ever present in a given donor_alpha_dial_grid result
(and hence a given cache), matching whichever synthesis built it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

DIAL_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# J restricted to its NEGATIVE half only (donor_family_joint's dial<=0
# side -- 0.0 the real donor, more negative = more artificially-corrupted,
# since y - dial*ebar_k ADDS ebar_k for dial<0 instead of subtracting it,
# amplifying the donor's own deviation rather than erasing it), ordered by
# ASCENDING SIGNED DIAL VALUE: the most negative (most corrupted) point
# first, 0.0 (the real donor, unmodified) last. orientation_score treats
# "later array index" as "the one a well-behaved metametric SHOULD prefer"
# (its own docstring: "1.0 if the metametric always prefers the more
# extreme dial"), which for the usual [0, 1] families coincides with "more
# extreme" because dial=1 there is the IMPROVED endpoint -- but J's
# negative range is the opposite: more negative is WORSE, 0.0 is the point
# a sound metametric should favor. Ranking by ascending EXTREMITY (0.0
# first) is backwards here -- confirmed the hard way: it made
# orientation_score measure "prefers the more corrupted point," which came
# out near 0 for exactly the datasets/donors where the metametric was
# behaving CORRECTLY. Ranking by ascending SIGNED value instead (this
# ordering) fixes it: last index is 0.0, the point that should be
# preferred, matching every other family's "later index = the one worth
# preferring" convention. (Reversing an already-computed orientation_score
# array algebraically gives EXACTLY 1 - the old score -- proven and
# confirmed empirically when this was first fixed.)
#
# dial(i) = 0.5 - 2^i for i in [-1, -0.9, ..., 1.9, 2] (31 points, step
# 0.1): geometric, packed toward dial=0 same as GEOM_DIAL_GRID/
# EXTENDED_ADDITIVE_MEAN_DIAL_GRID are for their own families, anchoring
# dial=0 EXACTLY at i=-1 (2^-1 == 0.5 exactly, so 0.5 - 0.5 == 0.0 exactly)
# and reaching dial=-3.5 at i=2 -- wider than the original [-2, 0] sweep,
# to reach a comparably extreme endpoint now that donor_family_additive_
# mean's own endpoint (EXTENDED_ADDITIVE_MEAN_DIAL_GRID) is standardized
# to the donor's own scale rather than the aspect's raw, often much larger
# scale. `sorted(...)` applies the ascending-signed-value fix above --
# the raw generator (i ascending) produces DESCENDING dial values (0.0
# first, since dial(i) is a decreasing function of i), so it must be
# resorted, unlike EXTENDED_ADDITIVE_MEAN_DIAL_GRID, which is already
# ascending by construction.
NEG_J_DIAL_GRID = tuple(sorted(round(0.5 - 2.0 ** i, 10) for i in (-1.0 + 0.1 * k for k in range(31))))


def full_range_alphas(alpha_lo: float, alpha_hi: float, step: float, eps_frac: float = 1e-3) -> list[float]:
  """Every clean multiple of `step` strictly inside [alpha_lo, alpha_hi],
  inset by eps_frac*(hi-lo) at each end (mwb.lib.alpha.build_alpha_grid's own
  convention -- the exact boundary admits only a single degenerate
  2-system support, so solve_w_exact is best kept off it). Snapped to
  clean multiples of `step` (e.g. 0.01, 0.02, ...) rather than starting
  exactly at the inset lo, for readable axis ticks."""
  eps = (alpha_hi - alpha_lo) * eps_frac
  lo, hi = alpha_lo + eps, alpha_hi - eps
  start = math.ceil(lo / step) * step
  n = int(math.floor((hi - start) / step + 1e-9))
  return [round(start + k * step, 10) for k in range(n + 1)]


def _synthesis_suffix(synthesis: str) -> str:
  """Shared by union_grid_tag/orientation_tag: synthesis='offset' (the
  original generator) adds no suffix, keeping existing cache filenames from
  before this parameter existed valid; 'additive'/'additive_mean' (mwb.lib.
  synthetic_scorers) get their own suffix so they never collide with an
  offset-synthesis cache for the same dataset/grid/metametric."""
  return '' if synthesis == 'offset' else f'_{synthesis}'


def _dial_preset_suffix(dial_preset: str) -> str:
  """Shared by union_grid_tag/orientation_tag: dial_preset='linear' (mwb.lib.
  synthetic_scorers.DIAL_GRID, the original 0/0.1/.../1 sweep) adds no
  suffix, keeping existing cache filenames valid; 'geometric' (mwb.lib.
  synthetic_scorers.GEOM_DIAL_GRID, {2^-i} packed toward dial=0) gets its
  own suffix -- '_geom', matching plot_synthetic_scorer_families.py's own
  filename convention for the same preset; 'extended' (mwb.lib.
  synthetic_scorers.EXTENDED_ADDITIVE_MEAN_DIAL_GRID, dial 0..~8, paired
  with donor_family_additive_mean's standardized=True default) gets
  '_ext'; 'symmetric' (mwb.lib.synthetic_scorers.ABT_DIAL_GRID/ABTJ_J_DIAL_GRID,
  compute_scorer_orientation_vs_alpha.py's --green-family ABTJ) gets '_sym'
  -- so none of the four ever collides with another for the same
  dataset/synthesis/metametric."""
  return {'linear': '', 'geometric': '_geom', 'extended': '_ext', 'symmetric': '_sym'}[dial_preset]


def union_grid_tag(
    dataset: str, step: float, metametric: str = 'spa', synthesis: str = 'offset', dial_preset: str = 'linear',
) -> str:
  """Filename tag for mwb.lib.alpha.union_alpha_grid (the union of every
  pairwise alpha_ij with a plain uniform [alpha_min, alpha_max] sweep at
  `step`) -- mirrors orientation_tag's compute/plot filename contract, but
  never collides with it (different literal infix) for the same
  dataset/step/metametric/synthesis/dial_preset."""
  suffix = '' if metametric == 'spa' else f'_{metametric}'
  return f'{dataset}_union_s{step:g}{suffix}{_synthesis_suffix(synthesis)}{_dial_preset_suffix(dial_preset)}'


def orientation_tag(
    dataset: str, n_steps: int, step: float, full_range: bool, metametric: str = 'spa', synthesis: str = 'offset',
    dial_preset: str = 'linear',
) -> str:
  """Filename tag shared by compute_scorer_orientation_vs_alpha.py (writer)
  and plot_scorer_orientation_vs_alpha.py (reader), so the same CLI flags
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


def save_orientation_data(
    path: str, *, dataset: str, alphas, center_alpha: float, K: int, donors: list[str],
    A: np.ndarray, B: np.ndarray, ess: np.ndarray, dial_grid, metametric: str = 'spa',
    synthesis: str = 'offset', T: np.ndarray | None = None, J: np.ndarray | None = None,
    dial_preset: str = 'linear',
) -> None:
  """Persists everything plot_scorer_orientation_vs_alpha.py needs: A/B are
  (n_donors, n_alpha) matrices of per-donor orientation curves, row order ==
  `donors`. Exactly one of T (the orientation-neutral All-MQM family,
  synthesis='additive'/'additive_mean') or J (the Joint family,
  synthesis='offset') is expected, matching mwb.lib.synthetic_scorer_alpha_grid.
  donor_alpha_dial_grid's output for that synthesis -- both are optional,
  for backward compatibility with caches written before either existed (or
  before J replaced T for offset caches). ess is (n_alpha,) ESS(w*(alpha))
  -- depends only on alpha, not on donor, since w*(alpha) is shared across
  every donor (mwb.lib.reweight_exact.solve_w_exact
  solved once per alpha). metametric records which weighted meta-metric
  ('spa' or 'pa') A/B/T/J were scored with. synthesis records which mwb.lib.
  synthetic_scorers dial family ('offset', 'additive', or 'additive_mean')
  built them. dial_preset records which named dial grid ('linear' or
  'geometric') `dial_grid`'s values came from, purely for display --
  dial_grid itself already has the actual numbers."""
  extra = {}
  if T is not None:
    extra['T'] = np.asarray(T, dtype=float)
  if J is not None:
    extra['J'] = np.asarray(J, dtype=float)
  np.savez(
      path, dataset=np.asarray(dataset), alphas=np.asarray(alphas, dtype=float),
      center_alpha=np.asarray(float(center_alpha)), K=np.asarray(int(K)),
      donors=np.asarray(donors, dtype='<U128'), A=np.asarray(A, dtype=float),
      B=np.asarray(B, dtype=float), ess=np.asarray(ess, dtype=float),
      dial_grid=np.asarray(dial_grid, dtype=float), metametric=np.asarray(metametric),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), **extra,
  )


def load_orientation_data(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']),
      'alphas': npz['alphas'],
      'center_alpha': float(npz['center_alpha']),
      'K': int(npz['K']),
      'donors': [str(d) for d in npz['donors']],
      'metametric': str(npz['metametric']) if 'metametric' in npz else 'spa',
      'synthesis': str(npz['synthesis']) if 'synthesis' in npz else 'offset',
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'linear',
      'A': npz['A'],
      'B': npz['B'],
      'T': npz['T'] if 'T' in npz else None,
      'J': npz['J'] if 'J' in npz else None,
      'ess': npz['ess'],
      'dial_grid': npz['dial_grid'],
  }


# Flip this to revert every caller to the all-pairs behavior at once.
#
# Why default True: orientation_score originally averaged over EVERY
# C(n,2) pair. That's fine on a narrow, evenly-spaced grid, but once a
# grid's range gets wide (mwb.lib.synthetic_scorers.EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID, mwb.lib.synthetic_scorer_orientation.NEG_J_DIAL_GRID -- both
# introduced to fix a DIFFERENT problem, additive_mean's raw-abar_k scale
# mismatch), most pairs end up comparing two dial points that are
# separated by a large gap (empirically: ~38% of EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID's 351 pairs differ by >1 SD, ~49% of NEG_J_DIAL_GRID's 465
# pairs). A large gap is a trivially easy "which one is more extreme"
# comparison for almost any metametric, regardless of whether it's
# actually good at fine discrimination -- so the average got dragged
# toward 1.0 by these easy pairs, diluting whatever the metametric does on
# the genuinely hard, CLOSE pairs (confirmed empirically: A/B/J all jumped
# toward ~1.0 once the grids were widened to fix the scale issue). Only
# scoring ADJACENT pairs (consecutive dial values, in dial order) removes
# every easy long-range comparison and measures specifically the
# metametric's ability to resolve neighboring, fine-grained differences --
# a harder, more informative question, and one that stays meaningful
# regardless of how wide the overall grid is.
ORIENTATION_ADJACENT_ONLY_DEFAULT = True


def orientation_score(scores_by_dial: np.ndarray, adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT) -> float:
  """Fraction of pairs (i, j) with dial_i < dial_j (i.e. i, j are indices
  into an array already sorted by ascending dial) where scores_by_dial[j]
  (the more extreme dial) exceeds scores_by_dial[i] -- 1.0 if the
  metametric always prefers the more extreme dial, 0.0 if always the less
  extreme one, 0.5 per tied pair. NaN if fewer than 2 finite values.

  adjacent_only (default True, ORIENTATION_ADJACENT_ONLY_DEFAULT -- a
  toggle, not a permanent fork): if True, only ADJACENT pairs -- consecutive
  entries among the finite, dial-ordered values, i.e. (finite[0],
  finite[1]), (finite[1], finite[2]), ... -- are scored, not every C(n,2)
  pair (see ORIENTATION_ADJACENT_ONLY_DEFAULT's comment for why). If
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


def donor_orientation_by_alpha(
    grids: dict, adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> dict[str, np.ndarray]:
  """{'A': (n_alpha,) adequacy_orientation per alpha, 'B': (n_alpha,)
  fluency_orientation per alpha, 'T' or 'J': (n_alpha,) allmqm_orientation
  or joint_orientation per alpha (whichever donor_alpha_dial_grid built,
  depending on synthesis)} for one donor, from mwb.lib.synthetic_scorer_alpha_grid.
  donor_alpha_dial_grid's output -- one orientation_score per column
  (alpha), read off that column's dial-ordered SPA values. adjacent_only
  passed straight through to orientation_score."""
  out = {}
  for label in ('A', 'B', 'T', 'J'):
    if label not in grids:
      continue
    g = grids[label]  # (n_dial, n_alpha), rows already in ascending dial order
    out[label] = np.array([orientation_score(g[:, ai], adjacent_only=adjacent_only) for ai in range(g.shape[1])])
  return out
