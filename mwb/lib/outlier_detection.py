"""Outlier detection for a dataset's system pool, from simplest to standard.

THE PROJECT DEFAULT, as of this session, is wmt_official_outliers -- see
its own docstring below. It is baked directly into mwb.lib.consistency.
real_systems() (exclude_outliers=True, the default), so every caller that
gets a dataset's system list through real_systems() -- which is nearly
the whole project -- already works on the wmt_official-screened pool
without applying any screen itself. This INCLUDES real_systems() now
returning an EMPTY list by default for the two datasets wmt_official
treats as UNSUPPORTED (ende20, zhen20) -- callers that loop over all 15
datasets unconditionally must skip a dataset with 0 systems rather than
assume every one of the 15 is non-empty (see the `if not systems:
continue`-style guards in mwb/scripts/, and
mwb.lib.consistency.real_systems' own docstring for exactly what "default"
means and how to opt out with exclude_outliers=False).

iterative_gk_outliers, labeled "GK" in comparison scripts/CLIs, is THE
PROJECT STANDARD *STATISTICAL* detector -- the one to reach for when a
data-driven, score-based screen is wanted rather than the hand-curated
wmt_official default: a system is an outlier if its JOINT (adequacy,
fluency) point is too far, in Mahalanobis distance, from a robust
bivariate fit of the whole pool (Gnanadesikan & Kettenring 1972; Iglewicz
& Hoya 1993 for the underlying MAD scale -- see gk_robust_loc_cov's own
docstring for the exact math). Iterated the same masking-resistant way as
the single-aspect version below, at GK_THRESHOLD (Z > 4). It is NOT
applied automatically anywhere -- unlike wmt_official,
using it means calling it explicitly.

iterative_single_aspect_outliers and iterative_single_aspect_outliers_either
are the SIMPLIFIED, NON-STANDARD predecessors this module started from --
each looks at only ONE aspect (adequacy or fluency) at a time, via
mwb.lib.outliers.mad_outliers' modified z-score, never the joint relationship
between the two. Kept for the pipelines that already depend on their exact
(narrower) behavior; new work should use iterative_gk_outliers.

Even the single-aspect modified z-score, robust as it is, can still mask a
second outlier when one is extreme enough to distort the pool's own
median/MAD before it's removed (observed directly this session -- IKUN only
reads as an outlier in ende24 once MSLC is already gone; IKUN-C's z-score in
jazh24 jumps from -2.88 to -4.23, crossing the flag threshold, only after
MSLC is removed first). A single MAD pass over the full pool doesn't see
that, since MSLC's own extremeness is still baked into the median/MAD it's
computed against -- hence the iterative wrapper: repeat the pass on the
shrinking pool, dropping every system flagged in one pass before
recomputing on what's left, until a pass flags nothing. The same iteration
pattern is reused for the joint/GK version below.

Three more, all hand-curated (NOT statistical detectors, no (a,b) score
ever enters them):

wmt_non_competative_outliers -- a fixed per-dataset lookup table
(WMT_NON_COMPETATIVE_SYSTEMS) of WMT organizer-inserted systems --
calibration placeholders and MBR-reranking baselines that are not
independent MT submissions -- hand-curated from each dataset's own roster.

shayegh_et_al_outliers -- standing in for what the prior work this project
builds on (Shayegh et al. 2025) itself did: nothing dropped for
ende23/zhen23, MSLC dropped for the three 2024 datasets, and every other
dataset in this project (added later, beyond that paper's own coverage)
UNSUPPORTED -- represented by dropping the dataset's entire roster, not by
silently returning no outliers.

wmt_official_outliers -- the WMT organizers'/mt-metrics-eval toolkit's OWN
outlier_systems metadata (external/mt-metrics-eval/mt_metrics_eval/
meta_info.py), which that toolkit excludes by default when building a
system list for final ranking. 2020 datasets are treated as UNSUPPORTED
here (entire roster dropped) rather than "verified clean," since none of
their officially-listed outliers are even present in this project's
smaller raw-TSV-sourced 2020 subset -- see the function's own docstring.
"""

from __future__ import annotations

import numpy as np

from mwb.lib.outliers import _MZ_CONSTANT, DEFAULT_THRESHOLD, mad_outliers

GK_THRESHOLD = 4.0


def iterative_single_aspect_outliers(
    systems: list[str], x: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float, int]]:
  """SIMPLIFIED, NON-STANDARD: single aspect only (whichever `x` is passed
  in -- adequacy or fluency, never both together). Superseded by
  iterative_gk_outliers, the project standard, which looks at the joint
  (adequacy, fluency) relationship instead. Kept for pipelines that already
  depend on this exact (narrower) behavior.

  (system, modified_z_score, iteration) for every system removed across
  repeated MAD passes -- iteration 1 is systems flagged on the full pool,
  iteration 2 on the pool with iteration 1's systems already removed, etc.
  Each pass flags (and removes) every system over `threshold` at once, not
  just the single worst one; stops the first pass that flags nothing.
  Order: iteration ascending, then |z| descending within an iteration
  (matching mad_outliers' own most-extreme-first convention)."""
  x = np.asarray(x, dtype=float)
  remaining_systems = list(systems)
  remaining_x = x.copy()
  removed = []
  iteration = 1
  while True:
    flagged = mad_outliers(remaining_systems, remaining_x, threshold=threshold)
    if not flagged:
      return removed
    removed.extend((s, z, iteration) for s, z in flagged)
    flagged_names = {s for s, _ in flagged}
    keep = [i for i, s in enumerate(remaining_systems) if s not in flagged_names]
    remaining_systems = [remaining_systems[i] for i in keep]
    remaining_x = remaining_x[keep]
    iteration += 1
    if len(remaining_systems) < 2:
      return removed


def iterative_single_aspect_outliers_either(
    systems: list[str], a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, str, float, int]]:
  """SIMPLIFIED, NON-STANDARD, like iterative_single_aspect_outliers: looks
  at adequacy and fluency SEPARATELY (one aspect at a time) and unions the
  two flagged sets, never the joint (adequacy, fluency) relationship the
  two aspects actually form together. Superseded by iterative_gk_outliers,
  the project standard.

  Flags a system if EITHER aspect's
  modified z-score exceeds `threshold` on the CURRENT (shrinking) pool.
  iterative_single_aspect_outliers (this function's single-aspect sibling)
  applies this only to adequacy; this is the two-aspect version: each pass
  recomputes modified z-scores for BOTH
  a and b on the remaining pool, flags the UNION of systems over threshold
  on either one, removes them all at once, and repeats until a pass flags
  nothing on both. Masking-resistant across aspects as well as within one:
  a system whose adequacy z-score is only borderline can still get
  uncovered once a fluency-flagged system is removed, and vice versa.

  Returns (system, aspect, modified_z_score, iteration) for every flagged
  system -- aspect is 'adequacy' or 'fluency' (whichever crossed
  threshold; a system flagged on both aspects in the same pass appears as
  two rows, one per aspect). Order: iteration ascending, then |z|
  descending within an (iteration, aspect) group, adequacy before fluency
  at a tie -- deterministic, not semantically load-bearing."""
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  remaining_systems = list(systems)
  remaining_a = a.copy()
  remaining_b = b.copy()
  removed = []
  iteration = 1
  while True:
    flagged_a = mad_outliers(remaining_systems, remaining_a, threshold=threshold)
    flagged_b = mad_outliers(remaining_systems, remaining_b, threshold=threshold)
    if not flagged_a and not flagged_b:
      return removed
    removed.extend((s, 'adequacy', z, iteration) for s, z in flagged_a)
    removed.extend((s, 'fluency', z, iteration) for s, z in flagged_b)
    flagged_names = {s for s, _ in flagged_a} | {s for s, _ in flagged_b}
    keep = [i for i, s in enumerate(remaining_systems) if s not in flagged_names]
    remaining_systems = [remaining_systems[i] for i in keep]
    remaining_a = remaining_a[keep]
    remaining_b = remaining_b[keep]
    iteration += 1
    if len(remaining_systems) < 2:
      return removed


def gk_robust_loc_cov(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
  """Robust bivariate location (median vector) and covariance for the JOINT
  (x,y) random vector, via the Gnanadesikan-Kettenring (GK) estimator: the
  direct multivariate generalization of median/MAD that this project's
  univariate modified z-score (mwb.lib.outliers) already uses -- unlike a
  Minimum Covariance Determinant fit, it needs no subset search over
  C(n, h) candidate supports, so it stays well-defined and deterministic
  down to small K (session notes: MCD's subset search was the mechanism
  that made a K=17 joint fit unstable/over-flag mainstream systems).

  The off-diagonal entry comes from the classic MAD-based identity
  Var(X+Y) - Var(X-Y) = 4 Cov(X,Y), with each MAD standardized by the
  Iglewicz & Hoya constant (_MZ_CONSTANT) to be a consistent robust std
  estimate; the resulting correlation is bounded in [-1,1] by construction
  (the two MADs it's built from are both >=0), so the covariance matrix is
  always valid. Returns None if either marginal MAD is 0 (a constant
  sample -- no meaningful scale)."""
  medx, medy = np.median(x), np.median(y)
  madx = np.median(np.abs(x - medx))
  mady = np.median(np.abs(y - medy))
  if madx == 0 or mady == 0:
    return None
  u, v = (x - medx) / madx, (y - medy) / mady
  s, d = u + v, u - v
  mad_s = np.median(np.abs(s - np.median(s)))
  mad_d = np.median(np.abs(d - np.median(d)))
  denom = mad_s ** 2 + mad_d ** 2
  rho = (mad_s ** 2 - mad_d ** 2) / denom if denom > 0 else 0.0
  rho = np.clip(rho, -0.999, 0.999)
  sx, sy = madx / _MZ_CONSTANT, mady / _MZ_CONSTANT
  cov = np.array([[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]])
  return np.array([medx, medy]), cov


def gk_mahalanobis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """The joint-RV analogue of a modified z-score: sqrt of the Mahalanobis
  distance of every (x_i, y_i) from the GK-robust location/covariance
  (gk_robust_loc_cov) of the full sample. All-NaN if the GK fit is
  degenerate (constant x or y)."""
  n = len(x)
  out = gk_robust_loc_cov(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
  if out is None:
    return np.full(n, np.nan)
  loc, cov = out
  cov_inv = np.linalg.pinv(cov)
  pts = np.column_stack([x, y]) - loc
  d2 = np.einsum('ij,jk,ik->i', pts, cov_inv, pts)
  return np.sqrt(d2)


def iterative_gk_outliers(
    systems: list[str], x: np.ndarray, y: np.ndarray, threshold: float = GK_THRESHOLD,
) -> list[tuple[str, float, int]]:
  """THE PROJECT STANDARD outlier detector (labeled "GK" in comparison
  scripts/CLIs): flags a system by its JOINT (x,y) Mahalanobis distance
  from a GK-robust bivariate fit of the pool (gk_mahalanobis /
  gk_robust_loc_cov -- Gnanadesikan & Kettenring 1972, see that function's
  own docstring for the exact math), not by either aspect alone. Default
  threshold is GK_THRESHOLD (Z > 4), not the single-aspect
  DEFAULT_THRESHOLD (3.5).

  (system, gk_mahalanobis_z, iteration) for every system removed across
  repeated GK-Mahalanobis passes over the joint (x,y) -- the masking-
  resistant iteration of iterative_single_aspect_outliers, but for
  gk_mahalanobis instead of the univariate modified z-score. Stops
  (returns what it has) once fewer than 5 systems remain, since a 2x2
  correlation from fewer points is not meaningful.

  CAVEAT (session finding): unlike the univariate version, this iteration
  can cascade even without any single dominating outlier, because GK's own
  rho estimate is itself high-variance at K~15 and can swing sharply
  (observed 0.90 -> 0.52 -> 0.38 across passes, dataset ende24 at
  threshold=3.5) as each pass's removals reshape the remaining pool -- each
  reshaping can newly flag ordinary points. threshold=4 (the project
  standard, GK_THRESHOLD) resolves that specific cascade for ende24,
  but NOT universally -- at threshold=4, heen23 and zhen23 still flag more
  than half their pool (9/12 and 8/15) across a few iterations each. This
  isn't a bug to fix by raising the threshold further on a per-dataset
  basis (that would be tuning the screen to the answer, exactly what this
  detector is meant to avoid) -- see output/outliers/README.md
  for the full per-dataset numbers and how to read them."""
  systems = list(systems)
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  removed = []
  iteration = 1
  while True:
    n = len(systems)
    if n < 5:
      return removed
    z = gk_mahalanobis(x, y)
    flagged_idx = [i for i in range(n) if not np.isnan(z[i]) and z[i] > threshold]
    if not flagged_idx:
      return removed
    removed.extend((systems[i], float(z[i]), iteration) for i in flagged_idx)
    flagged_idx_set = set(flagged_idx)
    keep = [i for i in range(n) if i not in flagged_idx_set]
    systems = [systems[i] for i in keep]
    x, y = x[keep], y[keep]
    iteration += 1


# Hand-curated, not computed: WMT organizer-inserted systems per dataset --
# calibration placeholders (metricsystem1-5) and MBR-reranking baselines
# (M2M100_1.2B-B4, QUARTZ_TuneReranking, *_bestmbr, NLLB_Greedy/NLLB_MBR_BLEU)
# that are not independent MT submissions, plus MSLC (ende24/enes24/jazh24).
# Session origin: manual review of each dataset's real-system roster: a
# dataset absent here (e.g. ende20, zhen20) has none.
WMT_NON_COMPETATIVE_SYSTEMS: dict[str, frozenset[str]] = {
    'ende21': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'zhen21': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ted_ende': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ted_zhen': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ende22': frozenset({'M2M100_1.2B-B4', 'QUARTZ_TuneReranking', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr'}),
    'zhen22': frozenset({'M2M100_1.2B-B4', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr'}),
    'enru22': frozenset({'M2M100_1.2B-B4', 'QUARTZ_TuneReranking', 'bleu_bestmbr', 'comet_bestmbr'}),
    'ende23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'zhen23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'heen23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'ende24': frozenset({'MSLC'}),
    'enes24': frozenset({'MSLC'}),
    'jazh24': frozenset({'MSLC'}),
}


def wmt_non_competative_outliers(dataset: str, systems: list[str] | None = None) -> set[str]:
  """NOT a statistical detector -- looks up WMT_NON_COMPETATIVE_SYSTEMS[dataset]
  (empty set for a dataset not in the table, e.g. ende20/zhen20; not an
  error). If `systems` is given, validates every looked-up name is
  actually in it (raises ValueError listing any that aren't -- a
  mismatch here means the roster changed or the table has a typo, not
  something to silently paper over)."""
  ref = set(WMT_NON_COMPETATIVE_SYSTEMS.get(dataset, frozenset()))
  if systems is not None:
    missing = ref - set(systems)
    if missing:
      raise ValueError(f'{dataset}: wmt_non_competative_outliers names not found in `systems`: {sorted(missing)}')
  return ref


# Hand-curated: which datasets Shayegh et al. (2025) itself covers, and
# what it drops for each. Datasets this project added beyond that paper's
# own coverage are UNSUPPORTED -- represented below by dropping the whole
# roster, not by an empty (and misleadingly "nothing to drop") set.
SHAYEGH_ET_AL_NO_DROP = frozenset({'ende23', 'zhen23'})
SHAYEGH_ET_AL_DROP_MSLC = frozenset({'ende24', 'enes24', 'jazh24'})


def shayegh_et_al_outliers(dataset: str, systems: list[str] | None = None) -> set[str]:
  """NOT a statistical detector -- hand-curated, matching what the prior
  work this project builds on (Shayegh et al. 2025) itself did: drops
  nothing for ende23/zhen23 (SHAYEGH_ET_AL_NO_DROP), drops {'MSLC'} for
  the three 2024 datasets (SHAYEGH_ET_AL_DROP_MSLC, validated against
  `systems` the same way as wmt_non_competative_outliers if given), and
  for every OTHER dataset -- outside that paper's own coverage -- drops
  the dataset's ENTIRE roster, since `systems` is REQUIRED to know what
  "everything" is; raises ValueError if `systems` is None for such a
  dataset, rather than silently returning an empty (i.e. "nothing wrong
  here") set for a dataset this method has no actual basis to judge."""
  if dataset in SHAYEGH_ET_AL_NO_DROP:
    return set()
  if dataset in SHAYEGH_ET_AL_DROP_MSLC:
    ref = {'MSLC'}
    if systems is not None:
      missing = ref - set(systems)
      if missing:
        raise ValueError(f'{dataset}: shayegh_et_al_outliers names not found in `systems`: {sorted(missing)}')
    return ref
  if systems is None:
    raise ValueError(f'{dataset}: shayegh_et_al_outliers is unsupported for this dataset and drops '
                      'its entire roster -- pass `systems` explicitly to get that set.')
  return set(systems)


# Hand-curated: the WMT organizers'/mt-metrics-eval toolkit's OWN
# outlier_systems metadata (external/mt-metrics-eval/mt_metrics_eval/
# meta_info.py's DATA[key][pair].outlier_systems), restricted to entries
# that are actually present in this project's own real_systems() roster
# per dataset -- most official names aren't (verified empirically): wmt23
# en-de/zh-en's official outlier is 'synthetic_ref', which never surfaces
# as a system in this project's pipeline at all; wmt20's official outliers
# are numeric-ID systems (e.g. 'zlabs-nlp.179') absent from this project's
# smaller raw-TSV-sourced 2020 subset. A dataset absent here has either no
# official outlier or one not present in our data.
WMT_OFFICIAL_SYSTEMS: dict[str, frozenset[str]] = {
    'ende22': frozenset({'M2M100_1.2B-B4'}),
    'ende24': frozenset({'MSLC'}),
    'enes24': frozenset({'MSLC'}),
    'jazh24': frozenset({'MSLC'}),
}

# 2020 is UNSUPPORTED in wmt_official_outliers (see its docstring): none of
# the official outliers for wmt20 en-de/zh-en are present in our roster, so
# returning WMT_OFFICIAL_SYSTEMS.get(dataset, frozenset()) as-is would give
# an empty set indistinguishable from "verified clean" -- which it isn't.
WMT_OFFICIAL_UNSUPPORTED = frozenset({'ende20', 'zhen20'})


def wmt_official_outliers(dataset: str, systems: list[str] | None = None) -> set[str]:
  """THE PROJECT DEFAULT, as of this session -- baked directly into
  mwb.lib.consistency.real_systems() (exclude_outliers=True, its default), so
  this is applied automatically almost everywhere in the project without
  any caller having to call it explicitly. NOT a statistical detector --
  the WMT organizers'/mt-metrics-eval toolkit's OWN outlier_systems
  metadata (WMT_OFFICIAL_SYSTEMS), which that toolkit excludes by default
  when building a system list for final ranking. Cross-checked against
  this project's own real_systems(..., exclude_outliers=False) roster per
  dataset; WMT_OFFICIAL_SYSTEMS lists only entries verified present.

  2020 (WMT_OFFICIAL_UNSUPPORTED: ende20, zhen20) is UNSUPPORTED in this
  mode: the officially-flagged 2020 outliers never appear in this
  project's roster at all, so applying the usual rule would silently
  return an empty set indistinguishable from "verified clean" -- instead
  this drops the ENTIRE roster for 2020, the same "unsupported" convention
  shayegh_et_al_outliers uses elsewhere; `systems` is REQUIRED for these
  two, raises ValueError if not given.

  For every other dataset, validates any looked-up name against `systems`
  the same way as wmt_non_competative_outliers/shayegh_et_al_outliers if
  given."""
  if dataset in WMT_OFFICIAL_UNSUPPORTED:
    if systems is None:
      raise ValueError(f'{dataset}: wmt_official_outliers treats 2020 datasets as unsupported and '
                        'drops the entire roster -- pass `systems` explicitly to get that set.')
    return set(systems)
  ref = set(WMT_OFFICIAL_SYSTEMS.get(dataset, frozenset()))
  if systems is not None:
    missing = ref - set(systems)
    if missing:
      raise ValueError(f'{dataset}: wmt_official_outliers names not found in `systems`: {sorted(missing)}')
  return ref
