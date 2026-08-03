"""Synthetic scorer construction (material/synthetic_scorer_construction.md):
given a real donor scorer's segment scores, builds two families of synthetic
scorers per donor -- one dialed on Adequacy MQM (dial A), one on Fluency MQM
(dial B) -- each a straight-line sweep between the donor itself (dial=0) and
some aspect-determined endpoint (dial=1). See the instruction file for the
full derivation of the original ('offset') generator; this module implements
its section 4 (the generator) and section 5 (the procedure) directly, plus
two sibling generators sharing the same dial=0/family-per-aspect shape but a
different endpoint:

  - donor_family (synthesis='offset', the original/default): dial=1 is the
    scorer whose system-level ranking is determined by the aspect's
    group-mean response alone, via section 3's m/e/ebar_k decomposition.
  - donor_family_additive (synthesis='additive'): dial=1 is the donor plus
    the aspect's raw per-segment value, y(k,i) + aspect(k,i).
  - donor_family_additive_mean (synthesis='additive_mean'): dial=1 is the
    donor plus the aspect's per-system mean, y(k,i) + abar_k.

The 'offset' method also has a third family with no A/B counterpart:
donor_family_joint conditions section 3's m/e/ebar_k decomposition on the
JOINT (Adequacy, Fluency) pair instead of either aspect alone, so m retains
whatever the donor's response to both aspects together looks like and only
the joint-independent residual is dialed away. There is a single dial here
(not an A/B split), since the joint conditioning already uses both aspects
at once. This construction is specific to 'offset' -- additive/
additive_mean's endpoints inject the aspect directly rather than going
through m/e/ebar_k, so they have no joint analogue.

'additive_mean' also has a fourth family with no A/B/offset counterpart:
donor_family_additive_mean_ab takes BOTH aspects at once, one dial each,
y_dial(k,i) = y(k,i) + A * abar_k + B * bbar_k -- the two-dial generalization
of donor_family_additive_mean's single-dial y(k,i) + dial * abar_k. Its
sweep is 2-D (every (A, B) pair from a dial grid's own Cartesian square, so
len(dial_grid)**2 points, not len(dial_grid)), which is also why it's the
only family that's opt-in rather than built automatically wherever A/B/T
are (see donor_family_spa_points' `ab_dial_grid`).

Reuses lib.spa_plane's positional alignment (a_pos, b_pos, and a donor's own
line-position-indexed segment matrix) -- the same (a, b, donor) triple
scorer_spa_points needs -- so a family's points land on the exact same SPA
plane axes (x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM) as the real
scorers.
"""

from __future__ import annotations

import numpy as np

from lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from lib.metric_scores import load_human_seg_scores, load_metric_seg_scores, load_metric_sys_scores
from lib.spa_plane import DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, positional_af_matrices

# Same floor lib.consistency/lib.spa_plane use before trusting SPA's
# permutation p-values (kept as a local copy rather than importing the
# other modules' underscore-prefixed constants -- same convention lib.
# consistency already follows).
_MIN_SPA_SEGMENTS = 10

DIAL_GRID = tuple(round(float(x), 1) for x in np.arange(0.0, 1.0001, 0.1))

# Extends DIAL_GRID's [0, 1] sweep backward through dial=-2 -- 'offset'
# only (donor_family's y - dial*ebar_k formula is well-defined for any
# real dial; nothing in its derivation is bounded to [0, 1], that's just
# the endpoint the construction doc names 'dial=1'). Negative dial moves
# AWAY from the aspect-conditional-mean endpoint, amplifying the donor's
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

# additive_mean, standardized=True ONLY (donor_family_additive_mean's
# injected_k is now rescaled to the donor's own between-system spread, so
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
# reordering needed, unlike lib.synthetic_scorer_orientation.
# NEG_J_DIAL_GRID (whose extreme end sits at the NEGATIVE side).
EXTENDED_ADDITIVE_MEAN_DIAL_GRID = tuple(
    round(2.0 ** i - 2.0 ** -10, 10) for i in (-10.0 + 0.5 * k for k in range(27))
)

# Two-sided dial grids for --green-family ABTJ (compute_scorer_
# orientation_vs_alpha.py) -- ABT_DIAL_GRID for A/B/T (donor_family_
# additive_mean, standardized=True), ABTJ_J_DIAL_GRID for J (donor_family_
# joint), each its OWN plain geometric ramp (not the earlier "positive half
# + its negation" construction): dial(i) = C + base^i for i in
# [0, 0.005, ..., 1.0] (201 points), monotonically increasing in i since
# base^i is increasing -- so both are already in ascending dial order
# (ascending "goodness", same convention as every other two-sided grid
# here: positive dial = more of the standardized aspect/AllMQM signal for
# A/B/T, or removes the donor's own joint-conditional deviation for J;
# negative dial is each family's own "worse" mirror -- see the retired
# SYMMETRIC_DIAL_GRID's own comment, superseded by this pair, for the full
# per-family derivation) with no reordering needed.
#
# Different (C, base) per family, hence different endpoints/density:
#   - ABT_DIAL_GRID: C=-1.5, base=3 -- range [-0.5, 1.5].
#   - ABTJ_J_DIAL_GRID: C=-3.5, base=4 -- range [-2.5, 0.5], the steeper
#     base=4 packing points far more densely near the negative end (i=0)
#     than the positive one (i=1).
# Neither grid lands on dial=0 exactly (unlike EXTENDED_ADDITIVE_MEAN_
# DIAL_GRID/the old SYMMETRIC_DIAL_GRID, both deliberately anchored there)
# -- an explicit choice this time, trading the exact-donor anchor point for
# these specific endpoints.
ABT_DIAL_GRID = tuple(round(-1.5 + 3.0 ** (0.005 * k), 10) for k in range(201))
ABTJ_J_DIAL_GRID = tuple(round(-3.5 + 4.0 ** (0.005 * k), 10) for k in range(201))


def aspect_lookup_table(y: np.ndarray, aspect: np.ndarray) -> np.ndarray:
  """m(a(k,i)), broadcast back to y's shape (section 3.1's group mean m): for
  every distinct value occurring in `aspect`, the mean of `y` over every
  (k,i) cell sharing that value. A lookup table, not a fitted function --
  whatever bends, plateaus, or jumps the donor's response to `aspect` has
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


def joint_aspect_lookup_table(y: np.ndarray, aspect_a: np.ndarray, aspect_b: np.ndarray) -> np.ndarray:
  """m(a(k,i), b(k,i)), broadcast back to y's shape: aspect_lookup_table's
  joint generalization -- instead of grouping by one aspect's distinct
  values, groups by the distinct (Adequacy, Fluency) PAIR, so a cell is only
  pooled with others sharing BOTH aspect values exactly, not each aspect
  separately. Same exact-float grouping rationale as aspect_lookup_table
  (Adequacy/Fluency MQM segment values are discrete), extended to pairs via
  np.unique(..., axis=0)."""
  flat_a = aspect_a.ravel()
  flat_b = aspect_b.ravel()
  flat_y = y.ravel()
  pairs = np.stack([flat_a, flat_b], axis=1)
  uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
  inv = inv.ravel()
  sums = np.bincount(inv, weights=flat_y, minlength=len(uniq))
  counts = np.bincount(inv, minlength=len(uniq))
  means = sums / counts
  return means[inv].reshape(y.shape)


def donor_family(y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the generator of section 4 ('offset'
  synthesis, the original/default one):

      y_dial(k,i) = y(k,i) - dial * ebar_k

  with m (aspect_lookup_table), e = y - m, and ebar_k = mean_i e(k,i) all
  computed once on the full (y, aspect) pool passed in -- section 5.1's
  invariance requirement: callers must pass the entire jointly-valid pool
  for this donor, never a resampled subset, since these quantities are held
  fixed for the whole sweep. dial=0 reproduces y exactly (the real donor);
  dial=1 is y(k,i) - ebar_k, i.e. the donor with each system's constant
  offset from its own aspect-conditional mean removed."""
  m = aspect_lookup_table(y, aspect)
  e = y - m
  ebar_k = e.mean(axis=1)  # constant across segments within a system (section 3.3)
  return {dial: y - dial * ebar_k[:, None] for dial in dial_grid}


def donor_family_joint(
    y: np.ndarray, aspect_a: np.ndarray, aspect_b: np.ndarray, dial_grid=DIAL_GRID,
) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'offset' method's Joint family:
  same section-4 construction as donor_family, but m is conditioned on the
  JOINT (Adequacy, Fluency) pair (joint_aspect_lookup_table) rather than
  either aspect alone:

      y_dial(k,i) = y(k,i) - dial * ebar_k,   ebar_k = mean_i (y - m(a,b))(k,i)

  so the joint-aspect-conditional response is what m retains, and dial
  removes only the part of the donor's system-level offset that the JOINT
  doesn't already explain. Unlike the A/B families there is a single dial
  family here, not an A-side/B-side split, since the joint conditioning
  already uses both aspects at once. dial=0 reproduces y exactly; dial=1 is
  the donor with each system's constant offset from its own joint-aspect-
  conditional mean removed."""
  m = joint_aspect_lookup_table(y, aspect_a, aspect_b)
  e = y - m
  ebar_k = e.mean(axis=1)
  return {dial: y - dial * ebar_k[:, None] for dial in dial_grid}


def donor_family_additive(y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'additive' synthesis: a linear
  sweep from the real donor (dial=0) to the donor plus the aspect's raw
  per-segment value (dial=1):

      y_dial(k,i) = y(k,i) + dial * aspect(k,i)

  Unlike donor_family, this does not go through section 3's m/e/ebar_k
  decomposition at all -- the aspect is injected directly, segment by
  segment, with no shape-preservation machinery. dial=1 is exactly
  y(k,i) + adequacyMQM(k,i) (or fluencyMQM for the B family)."""
  return {dial: y + dial * aspect for dial in dial_grid}


ADDITIVE_MEAN_STANDARDIZED_DEFAULT = True  # flip this to revert every caller to the raw-abar_k behavior at once


def donor_family_additive_mean(
    y: np.ndarray, aspect: np.ndarray, dial_grid=DIAL_GRID, standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[float, np.ndarray]:
  """{dial value -> y_dial matrix}, the 'additive_mean' synthesis: like
  donor_family_additive, but the aspect is collapsed to its per-system mean
  before being added, so the injected term is a per-system constant rather
  than varying by segment:

      y_dial(k,i) = y(k,i) + dial * abar_k,   abar_k = mean_i aspect(k,i)

  dial=0 reproduces y exactly; dial=1 is exactly y(k,i) + abar_k.

  standardized (default True, ADDITIVE_MEAN_STANDARDIZED_DEFAULT -- a
  toggle, not a permanent fork, so the raw behavior above stays one flip
  away): abar_k lives on the aspect's own raw scale, which need not have
  anything to do with the donor y's scale (e.g. y in [0, 100] for BLEU vs.
  abar_k around [-6, 0] for MQM) -- at dial=1 the injected term can then be
  numerically negligible or completely dominant depending on that
  accident, rather than on anything about the donor or the aspect. When
  standardized=True, abar_k is z-scored across the K systems and rescaled
  to the donor's OWN natural between-system spread instead:

      y_dial(k,i) = y(k,i) + dial * SD_k'(ybar_k') * z_k
      ybar_k = (1/n) sum_i y(k,i)                        (donor's own per-system mean)
      z_k = (abar_k - mean_k'(abar_k')) / SD_k'(abar_k')  (abar_k, standardized across systems)
      SD_k(x_k) = sqrt( (1/K) sum_k (x_k - mean_k'(x_k'))^2 )   (population std, matching np.std's default ddof=0)

  so at dial=1 the injected per-system offset has, by construction, the
  same across-system standard deviation as the donor's own ybar_k --
  commensurate with the donor's natural spread regardless of either
  quantity's absolute scale."""
  abar_k = aspect.mean(axis=1)
  if standardized:
    ybar_k = y.mean(axis=1)
    z_k = (abar_k - abar_k.mean()) / abar_k.std()
    injected_k = ybar_k.std() * z_k
  else:
    injected_k = abar_k
  return {dial: y + dial * injected_k[:, None] for dial in dial_grid}


def donor_family_additive_mean_ab(
    y: np.ndarray, aspect_a: np.ndarray, aspect_b: np.ndarray, dial_grid=GEOM_DIAL_GRID,
    standardized: bool = ADDITIVE_MEAN_STANDARDIZED_DEFAULT,
) -> dict[tuple[float, float], np.ndarray]:
  """{(A, B) dial pair -> y_dial matrix}, the 'additive_mean' method's AB
  family: donor_family_additive_mean generalized to TWO simultaneous dials,
  one per aspect, injected as per-system means the same way:

      y_dial(k,i) = y(k,i) + A * abar_k + B * bbar_k
      abar_k = mean_i aspectA(k,i),   bbar_k = mean_i aspectB(k,i)

  (or, standardized=True -- the default, matching donor_family_additive_
  mean's own default -- abar_k/bbar_k each independently z-scored across
  systems and rescaled to the donor's own between-system spread, exactly as
  donor_family_additive_mean's `standardized` does; see that function's
  docstring for the full derivation. Here it's applied once per aspect
  rather than once overall, so A and B each land on the donor's own scale
  independently of each other.) dial=(0, 0) reproduces y exactly, the same
  shared anchor every other family/method starts from.

  Unlike every other family in this module, the sweep is 2-D: dial_grid is
  swept as its own Cartesian square (every (A, B) pair with A, B both in
  dial_grid), so the returned dict has len(dial_grid)**2 entries keyed by
  (A, B) tuples instead of a single float. Callers that expect a scalar
  dial key (e.g. sorting a 1-D sweep) need to be aware of this -- it's why
  this family isn't in SYNTHESIS_GENERATORS (whose signature is (y, aspect,
  dial_grid) -> {float: matrix}, one aspect in, one dial out) and is opt-in
  wherever it's wired in (donor_family_spa_points' `ab_dial_grid`), rather
  than always built the way A/B/T/J are -- a naive Cartesian sweep is
  quadratic in dial_grid's length, and DIAL_GRID/GEOM_DIAL_GRID-sized grids
  weren't sized with that cost in mind."""
  abar_k = aspect_a.mean(axis=1)
  bbar_k = aspect_b.mean(axis=1)
  if standardized:
    ybar_k = y.mean(axis=1)
    za_k = (abar_k - abar_k.mean()) / abar_k.std()
    zb_k = (bbar_k - bbar_k.mean()) / bbar_k.std()
    injected_a_k = ybar_k.std() * za_k
    injected_b_k = ybar_k.std() * zb_k
  else:
    injected_a_k = abar_k
    injected_b_k = bbar_k
  return {
      (A, B): y + A * injected_a_k[:, None] + B * injected_b_k[:, None]
      for A in dial_grid for B in dial_grid
  }


# synthesis name -> family generator, all sharing the same signature
# (y, aspect, dial_grid) -> {dial -> y_dial matrix} and the same
# dial=0-reproduces-the-donor convention. Shared with lib.
# synthetic_scorer_alpha_grid (imported from there) so both the alpha-grid
# meta-eval pipeline and the SPA-plane family plots pick synthesis kinds
# the same way.
SYNTHESIS_GENERATORS = {
    'offset': donor_family,
    'additive': donor_family_additive,
    'additive_mean': donor_family_additive_mean,
}


def default_dial_preset(synthesis: str) -> str:
  """The committed-to default dial preset for a given synthesis kind:
  'linear' (DIAL_GRID) for 'offset', which doesn't saturate fast near
  dial=0 the way the other two do; 'geometric' (GEOM_DIAL_GRID) for plain
  'additive', which does; and 'extended' (EXTENDED_ADDITIVE_MEAN_DIAL_GRID,
  dial 0..~8) for 'additive_mean' specifically -- since donor_family_
  additive_mean's standardized=True default (the project default as of
  this session) rescales its injected term to the donor's own between-
  system spread, GEOM_DIAL_GRID's dial=1 ceiling is now only ~1 SD's worth
  of perturbation, no longer a genuinely extreme endpoint the way it was
  against the old raw-abar_k scale -- EXTENDED_ADDITIVE_MEAN_DIAL_GRID's
  wider range (~8 SDs) is what actually reaches a comparably extreme
  endpoint again. Callers (compute_scorer_orientation_vs_alpha.py,
  compute_synth25_orientation.py, plot_scorer_orientation_vs_alpha.py,
  plot_synthetic_scorer_families.py) use this only when the user hasn't
  explicitly passed --dial-preset; an explicit choice always wins."""
  if synthesis == 'offset':
    return 'linear'
  if synthesis == 'additive_mean':
    return 'extended'
  return 'geometric'


def donor_family_spa_points(
    donor_seg: np.ndarray, a_pos: np.ndarray, b_pos: np.ndarray, dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED, synthesis: str = 'offset',
    human_seg: np.ndarray | None = None, ab_dial_grid=None,
) -> dict[str, dict[float, tuple[float, float]]]:
  """{'A': {dial -> (x, y)}, 'B': {dial -> (x, y)}} for one donor's two
  synthetic families -- 'A' dialed on Adequacy (a_pos), 'B' on Fluency
  (b_pos) -- with x = SPA vs. Fluency MQM, y = SPA vs. Adequacy MQM, the same
  two axes lib.spa_plane.scorer_spa_points uses. donor_seg/a_pos/b_pos must
  already be restricted to the same jointly-valid segment columns (the
  pool). p_a/p_b are computed once here and reused, at the same `seed`, for
  every dial and both families -- section 5.2's requirement that the same
  set of permutation sign vectors be reused across the whole sweep, which
  pairwise_p_values already satisfies by construction (same seed + same
  segment count -> identical signs drawn).

  When synthesis='offset', a third entry 'J' is also included: the Joint
  family (donor_family_joint), dialed on the (Adequacy, Fluency) pair
  jointly rather than either aspect alone. It has no additive/
  additive_mean analogue (see module docstring), so it's omitted for those
  two synthesis kinds rather than raising.

  When synthesis is 'additive' or 'additive_mean' AND `human_seg` is given
  (the official All-MQM total, lib.metric_scores.load_human_seg_scores,
  restricted to the same jointly-valid columns as donor_seg/a_pos/b_pos), a
  third entry 'T' is included: the same synthesis kind's family dialed on
  All MQM instead of either aspect (mirrors lib.synthetic_scorer_alpha_grid.
  donor_alpha_dial_grid's T-family, restricted to the additive methods for
  the same reason -- see module docstring). Omitted (not raised) when
  synthesis='offset' or human_seg is None, since offset's third family is J
  instead.

  When synthesis='additive_mean' AND `ab_dial_grid` is given (not None), a
  fourth entry 'AB' is included: donor_family_additive_mean_ab's two-dial
  family, keyed by (A, B) tuples rather than a single float dial (see that
  function's docstring). Unlike A/B/T, this is opt-in rather than
  automatic -- ab_dial_grid defaults to None (AB omitted) since its
  Cartesian sweep is quadratic in len(ab_dial_grid) and existing callers
  weren't sized with that cost in mind; pass e.g. GEOM_DIAL_GRID explicitly
  to build it. Ignored (no 'AB' entry, even if given) for every other
  synthesis, since AB has no offset/additive analogue.

  synthesis: which SYNTHESIS_GENERATORS entry builds the dial sweep --
  'offset' (default), 'additive', or 'additive_mean'. All three share the
  same dial=0 point (the real donor) by construction, so overlaying more
  than one synthesis's family for the same donor traces multiple paths out
  of that same shared point."""
  if synthesis not in SYNTHESIS_GENERATORS:
    raise ValueError(f'synthesis must be one of {sorted(SYNTHESIS_GENERATORS)}, got {synthesis!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]
  p_a = pairwise_p_values(a_pos, num_permutations, seed)
  p_b = pairwise_p_values(b_pos, num_permutations, seed)

  def _points(family: dict[float, np.ndarray]) -> dict[float, tuple[float, float]]:
    pts = {}
    for dial, y_dial in family.items():
      p_m = pairwise_p_values(y_dial, num_permutations, seed)
      x = soft_pairwise_accuracy_from_pvalues(p_b, p_m)
      yv = soft_pairwise_accuracy_from_pvalues(p_a, p_m)
      if x == x and yv == yv:  # excludes NaN
        pts[dial] = (x, yv)
    return pts

  out: dict[str, dict[float, tuple[float, float]]] = {}
  for label, aspect in (('A', a_pos), ('B', b_pos)):
    out[label] = _points(build_family(donor_seg, aspect, dial_grid))
  if synthesis == 'offset':
    out['J'] = _points(donor_family_joint(donor_seg, a_pos, b_pos, dial_grid))
  elif human_seg is not None:
    out['T'] = _points(build_family(donor_seg, human_seg, dial_grid))
  if synthesis == 'additive_mean' and ab_dial_grid is not None:
    out['AB'] = _points(donor_family_additive_mean_ab(donor_seg, a_pos, b_pos, ab_dial_grid))
  return out


def all_synthetic_family_points(
    dataset: str, systems: list[str], root: str = '.', dial_grid=DIAL_GRID,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS, seed: int = DEFAULT_SEED, synthesis: str = 'offset',
    ab_dial_grid=None,
) -> dict[str, dict[str, dict[float, tuple[float, float]]]]:
  """donor name -> donor_family_spa_points(..., synthesis=synthesis), for
  every donor with a usable segment-level file and enough jointly-valid
  positions with a_pos/b_pos -- the synthetic-scorer analogue of lib.
  spa_plane.scorer_spa_points (same candidate gate and same positional
  alignment), one donor expanding into 2*len(dial_grid) points (3*len(
  dial_grid) for synthesis='offset', which also gets the Joint family, or
  for 'additive'/'additive_mean' when this dataset's All-MQM segment file
  loads, which also get the T family) instead of a single one. {} if this
  dataset's positional alignment can't be validated at all.

  For synthesis in ('additive', 'additive_mean'), human_seg (All-MQM) is
  loaded once here and intersected into each donor's own coverage mask
  (on top of a_pos/b_pos/donor_seg) so the T-family's points are computed
  on exactly the segments the rest of that donor's families use -- offset's
  mask is left untouched (no human_seg load at all) since it has no T
  family and this must not change offset's existing point set.

  ab_dial_grid: passed straight through to donor_family_spa_points -- None
  (default) omits the 'AB' family entirely (matching every existing
  caller's cost); give it (e.g. GEOM_DIAL_GRID) to opt a synthesis=
  'additive_mean' run into also building it. No effect for any other
  synthesis."""
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return {}
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root) if synthesis != 'offset' else None
  if human_seg is not None and human_seg.shape[1] != a_pos.shape[1]:
    human_seg = None

  candidates = load_metric_sys_scores(dataset, systems, root=root)
  out = {}
  for name in candidates:
    donor_seg = load_metric_seg_scores(dataset, name, systems, root=root)
    if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
      continue
    nan_mats = [a_pos, b_pos, donor_seg] if human_seg is None else [a_pos, b_pos, donor_seg, human_seg]
    mask = ~np.any([np.isnan(m).any(axis=0) for m in nan_mats], axis=0)
    if mask.sum() < _MIN_SPA_SEGMENTS:
      continue
    out[name] = donor_family_spa_points(
        donor_seg[:, mask], a_pos[:, mask], b_pos[:, mask],
        dial_grid=dial_grid, num_permutations=num_permutations, seed=seed, synthesis=synthesis,
        human_seg=None if human_seg is None else human_seg[:, mask], ab_dial_grid=ab_dial_grid)
  return out
