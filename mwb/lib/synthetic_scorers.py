"""Synthetic scorer construction: given a real base scorer's segment
scores, builds two families of synthetic scorers per base scorer -- one dialed on
Adequacy MQM (dial A), one on Fluency MQM (dial F) -- each a straight-line
sweep between the base scorer itself (dial=0) and some aspect-determined endpoint
(dial=1). Three of these dial constructions are the ones the paper reports
(Sec. "Scorer Augmentation"): the MQM-adherence family is
base_scorer_family_additive_mean's 'T' (AllMQM) aspect; the adequacy-fluency
family is base_scorer_family_additive_mean_af (A=dial, F=-dial); the
explainability-by-MQM family is base_scorer_family_joint (the 'offset' method's
m/e/ebar_k decomposition conditioned on the joint (Adequacy, Fluency)
pair). Two more, footnoted but excluded from the paper's main analysis,
are named wrappers around the same MQM-adherence construction retargeted
at a single aspect: base_scorer_family_adequacy_adherence and
base_scorer_family_fluency_adherence (see their own docstrings for why the
paper leaves them out). This module implements the generator and the
per-base scorer sweep directly, plus two sibling generators sharing the same
dial=0/family-per-aspect shape but a different endpoint:

  - base_scorer_family (synthesis='offset', the original/default): dial=1 is the
    scorer whose system-level ranking is determined by the aspect's
    group-mean response alone, via section 3's m/e/ebar_k decomposition.
  - base_scorer_family_additive (synthesis='additive'): dial=1 is the base scorer plus
    the aspect's raw per-segment value, y(k,i) + aspect(k,i).
  - base_scorer_family_additive_mean (synthesis='additive_mean'): dial=1 is the
    base scorer plus the aspect's per-system mean, y(k,i) + abar_k.

The 'offset' method also has a third family with no A/F counterpart:
base_scorer_family_joint conditions section 3's m/e/ebar_k decomposition on the
JOINT (Adequacy, Fluency) pair instead of either aspect alone, so m retains
whatever the base scorer's response to both aspects together looks like and only
the joint-independent residual is dialed away. There is a single dial here
(not an A/F split), since the joint conditioning already uses both aspects
at once. This construction is specific to 'offset' -- additive/
additive_mean's endpoints inject the aspect directly rather than going
through m/e/ebar_k, so they have no joint analogue.

'additive_mean' also has a fourth family with no A/F/offset counterpart:
base_scorer_family_additive_mean_af_cartesian takes BOTH aspects at once, one dial each,
y_dial(k,i) = y(k,i) + A * abar_k + F * fbar_k -- the two-dial generalization
of base_scorer_family_additive_mean's single-dial y(k,i) + dial * abar_k. Its
sweep is 2-D (every (A, F) pair from a dial grid's own Cartesian square, so
len(dial_grid)**2 points, not len(dial_grid)), which is also why it's the
only family that's opt-in rather than built automatically wherever A/F/T
are (see base_scorer_family_spa_points' `af_cartesian_dial_grid`, labeled
'AF_CARTESIAN' in its output dict). Not to be confused with
base_scorer_family_additive_mean_af (labeled 'AF' -- the paper's actual
Adequacy-fluency family, A=dial, F=-dial, moving A and F in OPPOSITE
directions along a single shared dial, not independently over a 2-D
square).

Reuses mwb.lib.spa_plane's positional alignment (a_pos, f_pos, and a base scorer's own
line-position-indexed segment matrix) -- the same (a, f, base scorer) triple
scorer_spa_points needs -- so a family's points land on the exact same SPA
plane axes (x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM) as the real
scorers.
"""

from __future__ import annotations

import numpy as np

from mwb.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores, load_scorer_sys_scores
from mwb.lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices

# Same floor mwb.lib.consistency/mwb.lib.spa_plane use before trusting SPA's
# permutation p-values (kept as a local copy rather than importing the
# other modules' underscore-prefixed constants -- same convention mwb.lib.
# consistency already follows).
_MIN_SPA_SEGMENTS = 10

DIAL_GRID = tuple(round(float(x), 1) for x in np.arange(0.0, 1.0001, 0.1))

# Extends DIAL_GRID's [0, 1] sweep backward through dial=-2 -- 'offset'
# only (base_scorer_family's y - dial*ebar_k formula is well-defined for any
# real dial; nothing in its derivation is bounded to [0, 1], that's just
# the endpoint the construction doc names 'dial=1'). Negative dial moves
# AWAY from the aspect-conditional-mean endpoint, amplifying the base scorer's
# own deviation from it instead of erasing it. additive/additive_mean's
# endpoints are direct injections of the aspect signal, not a subtracted
# offset, so there's no analogous "away" direction worth naming here --
# this grid is offset-specific.
EXTENDED_OFFSET_DIAL_GRID = tuple(round(float(x), 1) for x in np.arange(-2.0, 1.0001, 0.1))

# {2^-i : i = 0..10}, sorted ascending -- an alternative dial preset packing
# points toward dial=0 instead of spacing them evenly. additive/additive_mean
# saturate almost immediately (most of their movement happens between
# dial=0 and dial=0.1 on DIAL_GRID), so this is the grid that actually
# resolves that early part of the sweep instead of spending 10 of 11
# points on an already-flat tail. offset doesn't share this behavior, so
# GEOM_DIAL_GRID is mainly useful paired with the other two generators.
GEOM_DIAL_GRID = tuple(sorted(2.0 ** -i for i in range(11)))

# additive_mean, standardized=True ONLY (base_scorer_family_additive_mean's
# injected_k is now rescaled to the base scorer's own between-system spread, so
# dial=1 is only ~1 SD's worth of perturbation -- GEOM_DIAL_GRID's dial=1
# ceiling no longer reaches a genuinely extreme endpoint the way it did
# against the old raw-abar_k scale). dial(i) = 2^i - 2^-10 for i in
# [-10, -9.5, ..., 2.5, 3] (27 points, step 0.5): geometric like
# GEOM_DIAL_GRID (same rationale -- still saturates fast near 0), but the
# "- 2^-10" offset anchors dial=0 EXACTLY at i=-10 (2^-10 - 2^-10 == 0.0
# exactly, not merely close), and the range now reaches dial ~= 8 (~8 SDs)
# at i=3 instead of stopping at dial=1 (~1 SD). Already in ascending dial
# order by construction (2^i is increasing in i), matching every other
# family's "ascending value = ascending extremity" convention -- no
# reordering needed, unlike mwb.lib.synthetic_scorer_preference.
# NEG_J_DIAL_GRID (whose extreme end sits at the NEGATIVE side).
EXTENDED_ADDITIVE_MEAN_DIAL_GRID = tuple(
    round(2.0 ** i - 2.0 ** -10, 10) for i in (-10.0 + 0.5 * k for k in range(27))
)

# Two-sided dial grids for --family-set AFTJ (compute_scorer_
# preference_vs_beta.py, when A/F are built as two separate families
# instead of the combined AF family) -- AFT_DIAL_GRID for A/F/T (base_scorer_family_
# additive_mean, standardized=True), AFTJ_J_DIAL_GRID for J (base_scorer_family_
# joint), each its OWN plain geometric ramp (not the earlier "positive half
# + its negation" construction): dial(i) = C + base^i for i in
# [0, 0.005, ..., 1.0] (201 points), monotonically increasing in i since
# base^i is increasing -- so both are already in ascending dial order
# (ascending "goodness", same convention as every other two-sided grid
# here: positive dial = more of the standardized aspect/AllMQM signal for
# A/F/T, or removes the base scorer's own joint-conditional deviation for J;
# negative dial is each family's own "worse" mirror -- see the retired
# SYMMETRIC_DIAL_GRID's own comment, superseded by this pair, for the full
# per-family derivation) with no reordering needed.
#
# Different (C, base) per family, hence different endpoints/density:
#   - AFT_DIAL_GRID: C=-1.5, base=3 -- range [-0.5, 1.5].
#   - AFTJ_J_DIAL_GRID: C=-3.5, base=4 -- range [-2.5, 0.5], the steeper
#     base=4 packing points far more densely near the negative end (i=0)
#     than the positive one (i=1).
# Neither grid lands on dial=0 exactly (unlike EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID/the old SYMMETRIC_DIAL_GRID, both deliberately anchored there)
# -- an explicit choice this time, trading the exact-base scorer anchor point for
# these specific endpoints.
#
# Built via a resolution parameter (step, n) rather than hardcoded to 201
# points: step*k sweeps [0, step*(n-1)] as k ranges over range(n), so
# (step=0.005, n=201) sweeps exponent [0, 1] at the paper's committed
# resolution, and a smaller step with a correspondingly larger n (e.g.
# step=0.001, n=1001) sweeps the SAME [0, 1] exponent range at finer
# resolution -- used for the poster-style renders in compute_scorer_
# preference_vs_beta.py's --dial-step/--dial-n.
def make_aft_dial_grid(step: float = 0.005, n: int = 201) -> tuple[float, ...]:
  return tuple(round(-1.5 + 3.0 ** (step * k), 10) for k in range(n))


def make_aftj_j_dial_grid(step: float = 0.005, n: int = 201) -> tuple[float, ...]:
  return tuple(round(-3.5 + 4.0 ** (step * k), 10) for k in range(n))


AFT_DIAL_GRID = make_aft_dial_grid()
AFTJ_J_DIAL_GRID = make_aftj_j_dial_grid()

# base_scorer_family_additive_mean_af's own default -- linear, unlike every
# other additive_mean grid above (which all pack points toward dial=0
# geometrically): [-0.5, 0.5] in steps of 0.02, 51 points, evenly spaced.
AF_ADDITIVE_MEAN_DIAL_GRID = tuple(round(float(x), 2) for x in np.arange(-0.5, 0.5001, 0.02))


def aspect_lookup_table(y: np.ndarray, aspect: np.ndarray) -> np.ndarray:
  """m(a(k,i)), broadcast back to y's shape (section 3.1's group mean m): for
  every distinct value occurring in `aspect`, the mean of `y` over every
  (k,i) cell sharing that value. A lookup table, not a fitted function --
  whatever bends, plateaus, or jumps the base scorer's response to `aspect` has
  are carried exactly. Exact-float grouping is safe here because Adequacy/
  Fluency MQM segment values are highly discrete (weighted sums from a
  small error-weight table, mwb.mqm_scoring), the same discreteness plot_
  scorer_af_scatter.py's own mean-curve already relies on."""
  flat_aspect = aspect.ravel()
  flat_y = y.ravel()
  uniq, inv = np.unique(flat_aspect, return_inverse=True)
  sums = np.bincount(inv, weights=flat_y, minlength=len(uniq))
  counts = np.bincount(inv, minlength=len(uniq))
  means = sums / counts
  return means[inv].reshape(y.shape)


def joint_aspect_lookup_table(y: np.ndarray, aspect_a: np.ndarray, aspect_f: np.ndarray) -> np.ndarray:
  """m(a(k,i), f(k,i)), broadcast back to y's shape: aspect_lookup_table's
  joint generalization -- instead of grouping by one aspect's distinct
  values, groups by the distinct (Adequacy, Fluency) PAIR, so a cell is only
  pooled with others sharing BOTH aspect values exactly, not each aspect
  separately. Same exact-float grouping rationale as aspect_lookup_table
  (Adequacy/Fluency MQM segment values are discrete), extended to pairs via
  np.unique(..., axis=0)."""
  flat_a = aspect_a.ravel()
  flat_f = aspect_f.ravel()
  flat_y = y.ravel()
  pairs = np.stack([flat_a, flat_f], axis=1)
  uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
  inv = inv.ravel()
  sums = np.bincount(inv, weights=flat_y, minlength=len(uniq))
  counts = np.bincount(inv, minlength=len(uniq))
  means = sums / counts
  return means[inv].reshape(y.shape)


def base_scorer_family(y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the generator of section 4 ('offset'
  synthesis, the original/default one):

      y_dial(k,i) = y(k,i) - dial * ebar_k

  with m (aspect_lookup_table), e = y - m, and ebar_k = mean_i e(k,i) all
  computed once on the full (y, aspect) pool passed in -- section 5.1's
  invariance requirement: callers must pass the entire jointly-valid pool
  for this base scorer, never a resampled subset, since these quantities are held
  fixed for the whole sweep. dial=0 reproduces y exactly (the real base scorer);
  dial=1 is y(k,i) - ebar_k, i.e. the base scorer with each system's constant
  offset from its own aspect-conditional mean removed."""
  m = aspect_lookup_table(y, aspect)
  e = y - m
  ebar_k = e.mean(axis=1)  # constant across segments within a system (section 3.3)
  return {dial: y - dial * ebar_k[:, None] for dial in dial_grid}


def base_scorer_family_joint(
    y: np.ndarray, aspect_a: np.ndarray, aspect_f: np.ndarray, dial_grid=DIAL_GRID,
) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'offset' method's Joint family:
  same section-4 construction as base_scorer_family, but m is conditioned on the
  JOINT (Adequacy, Fluency) pair (joint_aspect_lookup_table) rather than
  either aspect alone:

      y_dial(k,i) = y(k,i) - dial * ebar_k,   ebar_k = mean_i (y - m(a,f))(k,i)

  so the joint-aspect-conditional response is what m retains, and dial
  removes only the part of the base scorer's system-level offset that the JOINT
  doesn't already explain. Unlike the A/F families there is a single dial
  family here, not an A-side/F-side split, since the joint conditioning
  already uses both aspects at once. dial=0 reproduces y exactly; dial=1 is
  the base scorer with each system's constant offset from its own joint-aspect-
  conditional mean removed."""
  m = joint_aspect_lookup_table(y, aspect_a, aspect_f)
  e = y - m
  ebar_k = e.mean(axis=1)
  return {dial: y - dial * ebar_k[:, None] for dial in dial_grid}


def base_scorer_family_additive(y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'additive' synthesis: a linear
  sweep from the real base scorer (dial=0) to the base scorer plus the aspect's raw
  per-segment value (dial=1):

      y_dial(k,i) = y(k,i) + dial * aspect(k,i)

  Unlike base_scorer_family, this does not go through section 3's m/e/ebar_k
  decomposition at all -- the aspect is injected directly, segment by
  segment, with no shape-preservation machinery. dial=1 is exactly
  y(k,i) + adequacyMQM(k,i) (or fluencyMQM for the F family)."""
  return {dial: y + dial * aspect for dial in dial_grid}


ADDITIVE_MEAN_STANDARDIZED_DEFAULT = True  # flip this to revert every caller to the raw-abar_k behavior at once


def base_scorer_family_additive_mean(
    y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID, standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'additive_mean' synthesis: like
  base_scorer_family_additive, but the aspect is collapsed to its per-system mean
  before being added, so the injected term is a per-system constant rather
  than varying by segment:

      y_dial(k,i) = y(k,i) + dial * abar_k,   abar_k = mean_i aspect(k,i)

  dial=0 reproduces y exactly; dial=1 is exactly y(k,i) + abar_k.

  standardized (default True, ADDITIVE_MEAN_STANDARDIZED_DEFAULT -- a
  toggle, not a permanent fork, so the raw behavior above stays one flip
  away): abar_k lives on the aspect's own raw scale, which need not have
  anything to do with the base scorer y's scale (e.g. y in [0, 100] for BLEU vs.
  abar_k around [-6, 0] for MQM) -- at dial=1 the injected term can then be
  numerically negligible or completely dominant depending on that
  accident, rather than on anything about the base scorer or the aspect. When
  standardized=True, abar_k is z-scored across the K systems and rescaled
  to the base scorer's OWN natural between-system spread instead:

      y_dial(k,i) = y(k,i) + dial * SD_k'(ybar_k') * z_k
      ybar_k = (1/n) sum_i y(k,i)                        (base scorer's own per-system mean)
      z_k = (abar_k - mean_k'(abar_k')) / SD_k'(abar_k')  (abar_k, standardized across systems)
      SD_k(x_k) = sqrt( (1/K) sum_k (x_k - mean_k'(x_k'))^2 )   (population std, matching np.std's default ddof=0)

  so at dial=1 the injected per-system offset has, by construction, the
  same across-system standard deviation as the base scorer's own ybar_k --
  commensurate with the base scorer's natural spread regardless of either
  quantity's absolute scale."""
  abar_k = aspect.mean(axis=1)
  if standardized:
    ybar_k = y.mean(axis=1)
    z_k = (abar_k - abar_k.mean()) / abar_k.std()
    injected_k = ybar_k.std() * z_k
  else:
    injected_k = abar_k
  return {dial: y + dial * injected_k[:, None] for dial in dial_grid}


def base_scorer_family_adequacy_adherence(
    y: np.ndarray, aspect_a: np.ndarray, dial_grid=DIAL_GRID, standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[float, np.ndarray]:
  """The paper's footnoted Adequacy-MQM-adherence family (Sec. "Scorer
  Augmentation", footnote to the MQM-adherence family): "the same approach
  can produce Adequacy MQM-adherence or Fluency MQM-adherence families of
  synthetic scorers by considering z as the system-level z-score of a or f,
  respectively." This is exactly base_scorer_family_additive_mean with the
  AllMQM aspect swapped for Adequacy MQM alone -- a thin, explicitly named
  wrapper, not new math (it's also exactly what base_scorer_beta_dial_grid's 'A'
  family already builds under synthesis='additive_mean').

  The paper excludes this family (and its fluency counterpart) from its
  main analysis: "the partial correlation between the two aspects
  [...] confounds their interpretation," since a scorer dialed toward pure
  Adequacy MQM inevitably drifts toward Fluency MQM too whenever rho(a,f)
  is far from 0, making "adherence to adequacy specifically" hard to read
  off cleanly."""
  return base_scorer_family_additive_mean(y, aspect_a, dial_grid, standardized)


def base_scorer_family_fluency_adherence(
    y: np.ndarray, aspect_f: np.ndarray, dial_grid=DIAL_GRID, standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[float, np.ndarray]:
  """The Fluency-MQM-adherence mirror of base_scorer_family_adequacy_adherence --
  see that function's docstring for the full derivation and the paper's
  stated reason for excluding both from its main analysis."""
  return base_scorer_family_additive_mean(y, aspect_f, dial_grid, standardized)


def base_scorer_family_additive_mean_af_cartesian(
    y: np.ndarray, aspect_a: np.ndarray, aspect_f: np.ndarray, dial_grid=GEOM_DIAL_GRID,
    standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[tuple[float, float], np.ndarray]:
  """{(A, F) dial pair -> y_dial matrix}, the 'additive_mean' method's AF
  Cartesian family: base_scorer_family_additive_mean generalized to TWO
  simultaneous dials, one per aspect, injected as per-system means the same
  way:

      y_dial(k,i) = y(k,i) + A * abar_k + F * fbar_k
      abar_k = mean_i aspectA(k,i),   fbar_k = mean_i aspectF(k,i)

  (or, standardized=True -- the default, matching base_scorer_family_additive_
  mean's own default -- abar_k/fbar_k each independently z-scored across
  systems and rescaled to the base scorer's own between-system spread, exactly as
  base_scorer_family_additive_mean's `standardized` does; see that function's
  docstring for the full derivation. Here it's applied once per aspect
  rather than once overall, so A and F each land on the base scorer's own scale
  independently of each other.) dial=(0, 0) reproduces y exactly, the same
  shared anchor every other family/method starts from.

  Unlike every other family in this module, the sweep is 2-D: dial_grid is
  swept as its own Cartesian square (every (A, F) pair with A, F both in
  dial_grid), so the returned dict has len(dial_grid)**2 entries keyed by
  (A, F) tuples instead of a single float. Callers that expect a scalar
  dial key (e.g. sorting a 1-D sweep) need to be aware of this -- it's why
  this family isn't in SYNTHESIS_GENERATORS (whose signature is (y, aspect,
  dial_grid) -> {float: matrix}, one aspect in, one dial out) and is opt-in
  wherever it's wired in (base_scorer_family_spa_points' `af_cartesian_dial_grid`), rather
  than always built the way A/F/T/J are -- a naive Cartesian sweep is
  quadratic in dial_grid's length, and DIAL_GRID/GEOM_DIAL_GRID-sized grids
  weren't sized with that cost in mind.

  Not to be confused with base_scorer_family_additive_mean_af (moves A and
  F in OPPOSITE directions along a single shared dial, not independently
  over a 2-D square)."""
  abar_k = aspect_a.mean(axis=1)
  fbar_k = aspect_f.mean(axis=1)
  if standardized:
    ybar_k = y.mean(axis=1)
    za_k = (abar_k - abar_k.mean()) / abar_k.std()
    zf_k = (fbar_k - fbar_k.mean()) / fbar_k.std()
    injected_a_k = ybar_k.std() * za_k
    injected_f_k = ybar_k.std() * zf_k
  else:
    injected_a_k = abar_k
    injected_f_k = fbar_k
  return {
      (A, F): y + A * injected_a_k[:, None] + F * injected_f_k[:, None]
      for A in dial_grid for F in dial_grid
  }


def base_scorer_family_additive_mean_af(
    y: np.ndarray, aspect_a: np.ndarray, aspect_f: np.ndarray, dial_grid=AF_ADDITIVE_MEAN_DIAL_GRID,
    standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'additive_mean' method's Adequacy-
  fluency family (the paper's actual reported family -- Sec. "Scorer
  Augmentation"): base_scorer_family_additive_mean_af_cartesian restricted to a single dial moving
  A and F in OPPOSITE directions at once (A = dial, F = -dial) instead of
  independently over the full (A, F) square:

      y_dial(k,i) = y(k,i) + dial * abar_k - dial * fbar_k

  abar_k/fbar_k (and standardized's rescaling to the base scorer's own between-
  system spread) are computed exactly as in base_scorer_family_additive_mean_af_cartesian --
  see that function's docstring for the full derivation, including the
  standardized=True (default) z-scoring. The only difference here is the
  sign on the F term: where the AF Cartesian family's diagonal A=F=dial
  moves both aspects the SAME way (both toward or both away from the base scorer
  together), this family's A=dial, F=-dial moves them in OPPOSITE
  directions -- one aspect's effect grows while the other's shrinks as dial
  increases. dial=0 reproduces y exactly, the same shared anchor every
  other family uses.

  Unlike base_scorer_family_additive_mean_af_cartesian, the sweep is 1-D again (one dial,
  not a Cartesian (A, F) square), so this returns a plain {float: matrix}
  dict like every other single-dial family in this module.

  Default dial_grid is AF_ADDITIVE_MEAN_DIAL_GRID: [-0.5, 0.5] in
  linear steps of 0.02 (51 points), unlike every other additive_mean grid
  in this module, which packs points geometrically toward dial=0 -- since
  this family combines two aspects' worth of injected signal into one
  dial, its saturation profile isn't the single-aspect one those grids
  were sized for, so it gets its own evenly-spaced grid instead."""
  abar_k = aspect_a.mean(axis=1)
  fbar_k = aspect_f.mean(axis=1)
  if standardized:
    ybar_k = y.mean(axis=1)
    za_k = (abar_k - abar_k.mean()) / abar_k.std()
    zf_k = (fbar_k - fbar_k.mean()) / fbar_k.std()
    injected_a_k = ybar_k.std() * za_k
    injected_f_k = ybar_k.std() * zf_k
  else:
    injected_a_k = abar_k
    injected_f_k = fbar_k
  return {dial: y + dial * injected_a_k[:, None] - dial * injected_f_k[:, None] for dial in dial_grid}


# synthesis name -> family generator, all sharing the same signature
# (y, aspect, dial_grid) -> {dial -> y_dial matrix} and the same
# dial=0-reproduces-the-base scorer convention. Shared with mwb.lib.
# synthetic_scorer_beta_grid (imported from there) so both the beta-grid
# meta-eval pipeline and the SPA-plane family plots pick synthesis kinds
# the same way.
SYNTHESIS_GENERATORS = {
    'offset': base_scorer_family,
    'additive': base_scorer_family_additive,
    'additive_mean': base_scorer_family_additive_mean,
}


def default_dial_preset(synthesis: str) -> str:
  """The committed-to default dial preset for a given synthesis kind:
  'linear' (DIAL_GRID) for 'offset', which doesn't saturate fast near
  dial=0 the way the other two do; 'geometric' (GEOM_DIAL_GRID) for plain
  'additive', which does; and 'extended' (EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  dial 0..~8) for 'additive_mean' specifically -- since base_scorer_family_
  additive_mean's standardized=True default (the project default as of
  this session) rescales its injected term to the base scorer's own between-
  system spread, GEOM_DIAL_GRID's dial=1 ceiling is now only ~1 SD's worth
  of perturbation, no longer a genuinely extreme endpoint the way it was
  against the old raw-abar_k scale -- EXTENDED_ADDITIVE_MEAN_DIAL_GRID's
  wider range (~8 SDs) is what actually reaches a comparably extreme
  endpoint again. Callers (compute_scorer_preference_vs_beta.py,
  compute_synth25_preference.py, plot_scorer_preference_vs_beta.py) use
  this only when the user hasn't explicitly passed --dial-preset; an
  explicit choice always wins."""
  if synthesis == 'offset':
    return 'linear'
  if synthesis == 'additive_mean':
    return 'extended'
  return 'geometric'


def base_scorer_family_spa_points(
    base_scorer_seg: np.ndarray, a_pos: np.ndarray, f_pos: np.ndarray, dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED, synthesis: str = 'offset',
    human_seg: np.ndarray | None = None, af_cartesian_dial_grid=None, af_dial_grid=None,
) -> dict[str, dict[float, tuple[float, float]]]:
  """{'A': {dial -> (x, y)}, 'F': {dial -> (x, y)}} for one base scorer's two
  synthetic families -- 'A' dialed on Adequacy (a_pos), 'F' on Fluency
  (f_pos) -- with x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM, the same
  two axes mwb.lib.spa_plane.scorer_spa_points uses. base_scorer_seg/a_pos/f_pos must
  already be restricted to the same jointly-valid segment columns (the
  pool). p_a/p_f are computed once here and reused, at the same `seed`, for
  every dial and both families -- section 5.2's requirement that the same
  set of permutation sign vectors be reused across the whole sweep, which
  pairwise_p_values already satisfies by construction (same seed + same
  segment count -> identical signs drawn).

  When synthesis='offset', a third entry 'J' is also included: the Joint
  family (base_scorer_family_joint), dialed on the (Adequacy, Fluency) pair
  jointly rather than either aspect alone. It has no additive/
  additive_mean analogue (see module docstring), so it's omitted for those
  two synthesis kinds rather than raising.

  When synthesis is 'additive' or 'additive_mean' AND `human_seg` is given
  (the official All-MQM total, mwb.lib.scorer_score_files.load_human_seg_scores,
  restricted to the same jointly-valid columns as base_scorer_seg/a_pos/f_pos), a
  third entry 'T' is included: the same synthesis kind's family dialed on
  All MQM instead of either aspect (mirrors mwb.lib.synthetic_scorer_beta_grid.
  base_scorer_beta_dial_grid's T-family, restricted to the additive methods for
  the same reason -- see module docstring). Omitted (not raised) when
  synthesis='offset' or human_seg is None, since offset's third family is J
  instead.

  When synthesis='additive_mean' AND `af_cartesian_dial_grid` is given (not None), a
  fourth entry 'AF_CARTESIAN' is included: base_scorer_family_additive_mean_af_cartesian's two-dial
  Cartesian family, keyed by (A, F) tuples rather than a single float dial
  (see that function's docstring; not to be confused with the paper's
  actual Adequacy-fluency family's own 'AF' entry below). Unlike A/F/T,
  this is opt-in rather than automatic -- af_cartesian_dial_grid defaults
  to None ('AF_CARTESIAN' omitted) since its Cartesian sweep is quadratic
  in len(af_cartesian_dial_grid) and existing callers weren't sized with
  that cost in mind; pass e.g. GEOM_DIAL_GRID explicitly to build it.
  Ignored (no 'AF_CARTESIAN' entry, even if given) for every other
  synthesis, since this family has no offset/additive analogue.

  When synthesis='additive_mean' AND `af_dial_grid` is given (not
  None), a fifth entry 'AF' is included: base_scorer_family_additive_mean_af's
  single-dial Adequacy-fluency family (A = dial, F = -dial -- the paper's
  actual reported family, see that function's docstring), keyed by a plain
  float dial like A/F/T rather than the Cartesian family's (A, F) tuples.
  Opt-in the same way that family is (af_dial_grid defaults to None, AF
  omitted) even though the sweep here is 1-D, not quadratic -- kept opt-in
  for consistency rather than for cost reasons. Ignored (no 'AF' entry,
  even if given) for every other synthesis, since it has no offset/additive
  analogue.

  synthesis: which SYNTHESIS_GENERATORS entry builds the dial sweep --
  'offset' (default), 'additive', or 'additive_mean'. All three share the
  same dial=0 point (the real base scorer) by construction, so overlaying more
  than one synthesis's family for the same base scorer traces multiple paths out
  of that same shared point."""
  if synthesis not in SYNTHESIS_GENERATORS:
    raise ValueError(f'synthesis must be one of {sorted(SYNTHESIS_GENERATORS)}, got {synthesis!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]
  p_a = pairwise_p_values(a_pos, num_permutations, seed)
  p_f = pairwise_p_values(f_pos, num_permutations, seed)

  def _points(family: dict[float, np.ndarray]) -> dict[float, tuple[float, float]]:
    pts = {}
    for dial, y_dial in family.items():
      p_m = pairwise_p_values(y_dial, num_permutations, seed)
      x = soft_pairwise_accuracy_from_pvalues(p_f, p_m)
      yv = soft_pairwise_accuracy_from_pvalues(p_a, p_m)
      if x == x and yv == yv:  # excludes NaN
        pts[dial] = (x, yv)
    return pts

  out: dict[str, dict[float, tuple[float, float]]] = {}
  for label, aspect in (('A', a_pos), ('F', f_pos)):
    out[label] = _points(build_family(base_scorer_seg, aspect, dial_grid))
  if synthesis == 'offset':
    out['J'] = _points(base_scorer_family_joint(base_scorer_seg, a_pos, f_pos, dial_grid))
  elif human_seg is not None:
    out['T'] = _points(build_family(base_scorer_seg, human_seg, dial_grid))
  if synthesis == 'additive_mean' and af_cartesian_dial_grid is not None:
    out['AF_CARTESIAN'] = _points(base_scorer_family_additive_mean_af_cartesian(base_scorer_seg, a_pos, f_pos, af_cartesian_dial_grid))
  if synthesis == 'additive_mean' and af_dial_grid is not None:
    out['AF'] = _points(base_scorer_family_additive_mean_af(base_scorer_seg, a_pos, f_pos, af_dial_grid))
  return out


def all_synthetic_family_points(
    dataset: str, systems: list[str], root: str = '.', dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED, synthesis: str = 'offset',
    af_cartesian_dial_grid=None, af_dial_grid=None,
) -> dict[str, dict[str, dict[float, tuple[float, float]]]]:
  """base scorer name -> base_scorer_family_spa_points(..., synthesis=synthesis), for
  every base scorer with a usable segment-level file and enough jointly-valid
  positions with a_pos/f_pos -- the synthetic-scorer analogue of mwb.lib.
  spa_plane.scorer_spa_points (same candidate gate and same positional
  alignment), one base scorer expanding into 2*len(dial_grid) points (3*len(
  dial_grid) for synthesis='offset', which also gets the Joint family, or
  for 'additive'/'additive_mean' when this dataset's All-MQM segment file
  loads, which also get the T family) instead of a single one. {} if this
  dataset's positional alignment can't be validated at all.

  For synthesis in ('additive', 'additive_mean'), human_seg (All-MQM) is
  loaded once here and intersected into each base scorer's own coverage mask
  (on top of a_pos/f_pos/base_scorer_seg) so the T-family's points are computed
  on exactly the segments the rest of that base scorer's families use -- offset's
  mask is left untouched (no human_seg load at all) since it has no T
  family and this must not change offset's existing point set.

  af_cartesian_dial_grid: passed straight through to base_scorer_family_spa_points -- None
  (default) omits the 'AF_CARTESIAN' family entirely (matching every
  existing caller's cost); give it (e.g. GEOM_DIAL_GRID) to opt a
  synthesis='additive_mean' run into also building it. No effect for any
  other synthesis.

  af_dial_grid: same pass-through, for the 'AF' family (base_scorer_family_
  additive_mean_af, the paper's actual Adequacy-fluency family) instead --
  None (default) omits it; give it (e.g. AF_ADDITIVE_MEAN_DIAL_GRID) to opt
  in. No effect for any other synthesis."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return {}
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root) if synthesis != 'offset' else None
  if human_seg is not None and human_seg.shape[1] != a_pos.shape[1]:
    human_seg = None

  candidates = load_scorer_sys_scores(dataset, systems, root=root)
  out = {}
  for name in candidates:
    base_scorer_seg = load_scorer_seg_scores(dataset, name, systems, root=root)
    if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
      continue
    nan_mats = [a_pos, f_pos, base_scorer_seg] if human_seg is None else [a_pos, f_pos, base_scorer_seg, human_seg]
    mask = ~np.any([np.isnan(m).any(axis=0) for m in nan_mats], axis=0)
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = base_scorer_family_spa_points(
        base_scorer_seg[:, mask], a_pos[:, mask], f_pos[:, mask],
        dial_grid=dial_grid, num_permutations=num_permutations, seed=seed, synthesis=synthesis,
        human_seg=None if human_seg is None else human_seg[:, mask], af_cartesian_dial_grid=af_cartesian_dial_grid,
        af_dial_grid=af_dial_grid)
  return out
