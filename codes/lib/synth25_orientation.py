"""adequacy_orientation(SPA_synth25) / fluency_orientation(SPA_synth25) /
allmqm_orientation(SPA_synth25): lib.synthetic_scorer_orientation's
dial-pair-preference method (material/synthetic_scorer_construction.md),
applied to the SPA_synth25/PA_synth25 baseline (lib.synth25_metametrics,
Shayegh et al. 2025's system-synthesis method) instead of the weighted-
SPA(alpha)/weighted-PA(alpha) family compute_scorer_orientation_vs_alpha.py
sweeps.

Same dial families (lib.synthetic_scorers' A/B/T sweep for one donor
scorer), same reduction (lib.synthetic_scorer_orientation.orientation_score:
fraction of same-family dial pairs where the metametric prefers the more
extreme dial) -- the only thing that changes is which metametric scores
each dial. Since SPA_synth25/PA_synth25 take no alpha argument (S2.4 of
material/prior_work/metrics-dp-tradeoff.tex: PA/SPA computed once, on the
fixed Row-7 union pool), each donor's A/B/T family collapses to a single
orientation_score per family rather than a curve over alpha. Averaged over
donors, this gives 3 dataset-level CONSTANTS -- meant to be drawn as
horizontal reference lines on an existing orientation-vs-alpha plot (they
do not depend on alpha at all), not as new points on the alpha axis.

This is deliberately the one place system synthesis (lib.synth25_
metametrics) and scorer synthesis (lib.synthetic_scorers /
lib.synthetic_scorer_alpha_grid, an unrelated axis -- see lib.synth25_
metametrics' own module docstring) are allowed to meet: lib.synth25_
metametrics stays entirely self-contained (nothing there imports scorer-
synthesis machinery), and nothing in the scorer-synthesis modules imports
this one back, so the dependency runs one way only.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time

import numpy as np

from lib.consistency import real_scorers
from lib.metric_scores import load_human_seg_scores, load_metric_seg_scores
from lib.spa_plane import positional_af_matrices
from lib.synth25_metametrics import (
    DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, DEFAULT_TIE_SEED, POOL_BLOCKS, SYNTH25_METAMETRICS,
    pool_alpha_0, synth25_value,
)
from lib.synthetic_scorer_alpha_grid import ASPECT_DONORS, _aspect_donor_seg
from lib.synthetic_scorer_orientation import (
    NEG_J_DIAL_GRID, ORIENTATION_ADJACENT_ONLY_DEFAULT, orientation_score,
)
from lib.synthetic_scorers import DIAL_GRID, SYNTHESIS_GENERATORS, donor_family_joint

# Same floor used throughout this project's SPA/synthesis machinery.
_MIN_SPA_SEGMENTS = 10

FAMILIES = ('A', 'B', 'T')


def donor_synth25_orientation(
    dataset: str,
    donor_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> dict[str, float] | None:
  """{'A': .., 'B': .., 'T': ..} adequacy/fluency/allmqm orientation of
  `metametric_name` for one donor -- mirrors lib.synthetic_scorer_
  alpha_grid.donor_alpha_dial_grid's coverage gate and family construction
  exactly (same mask, same build_family call), but scores each dial with
  synth25_value (fixed Row-7 union pool, no alpha) instead of a
  weighted-metametric-at-alpha sweep, then reduces straight to
  orientation_score (adjacent_only passed straight through -- see its own
  docstring) over the 11 dial values -- one number per family, no alpha
  column. None if this donor lacks sufficient segment coverage (same gate
  donor_alpha_dial_grid uses)."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if donor_name in ASPECT_DONORS:
    donor_seg = _aspect_donor_seg(donor_name, a_pos, b_pos, human_seg)
  else:
    donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]

  out = {}
  for label, aspect in (('A', a_pos), ('B', b_pos), ('T', human_seg)):
    family = build_family(donor_seg, aspect, dial_grid)
    vals = np.array([
        synth25_value(metametric_name, human_seg, family[dial], a_pos, b_pos,
                       tie_seed=tie_seed, num_permutations=num_permutations, seed=seed)
        for dial in dial_grid
    ])
    out[label] = orientation_score(vals, adjacent_only=adjacent_only)
  return out


def dataset_synth25_orientation(
    dataset: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    candidates: list[str] | None = None,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    progress: bool = False,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> dict[str, float]:
  """Dataset-level {'A': .., 'B': .., 'T': ..} -- mean over every usable
  donor (candidates, default real_scorers(dataset, systems=systems,
  require_seg_scores=True), this project's canonical scorer screen, same
  roster compute_scorer_orientation_vs_alpha.py uses) of donor_synth25_
  orientation. NaN for a family with zero usable donors. progress=True
  prints one '[i/n] donor (elapsed, eta)' line per donor to stderr (this
  loop runs a full SPA permutation test per dial per family per donor, so
  it is not cheap at dozens of donors)."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)

  t0 = time.time()
  per_donor = []
  n = len(candidates)
  for i, donor in enumerate(candidates, 1):
    r = donor_synth25_orientation(
        dataset, donor, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
        adjacent_only=adjacent_only,
    )
    if r is not None:
      per_donor.append(r)
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {donor} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  out = {}
  for label in FAMILIES:
    vals = [d[label] for d in per_donor if d[label] == d[label]]
    out[label] = float(np.mean(vals)) if vals else float('nan')
  return out


def donor_synth25_orientation_all_pools(
    dataset: str,
    donor_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> dict[str, dict[str, float]] | None:
  """{pool_name: {'A': .., 'B': .., 'T': ..}} for EVERY pool in POOL_BLOCKS,
  for one donor -- same coverage gate and dial-family construction as
  donor_synth25_orientation (in fact identical for pool_name='real+adeq+flu'),
  but each family's dial family is built ONCE and then scored against all 6
  pools (synth25_value's blocks argument) instead of rebuilding it per pool,
  since build_family does not depend on which pool the result is later
  compared against. None on insufficient coverage, matching donor_synth25_
  orientation's gate exactly."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if donor_name in ASPECT_DONORS:
    donor_seg = _aspect_donor_seg(donor_name, a_pos, b_pos, human_seg)
  else:
    donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]

  out = {pool_name: {} for pool_name in POOL_BLOCKS}
  for label, aspect in (('A', a_pos), ('B', b_pos), ('T', human_seg)):
    family = build_family(donor_seg, aspect, dial_grid)
    for pool_name, blocks in POOL_BLOCKS.items():
      vals = np.array([
          synth25_value(metametric_name, human_seg, family[dial], a_pos, b_pos, tie_seed=tie_seed,
                         num_permutations=num_permutations, seed=seed, blocks=blocks)
          for dial in dial_grid
      ])
      out[pool_name][label] = orientation_score(vals, adjacent_only=adjacent_only)
  return out


def _donor_synth25_job(job):
  """One donor's worker-process body for dataset_synth25_orientation_all_
  pools's --workers > 1 path -- module-level so it's importable/picklable
  under macOS's spawn start method. Donors are independent (each call only
  reads its own donor's segment file off disk), so this parallelizes with
  no cross-donor coordination needed. Returns (donor, result_or_None)."""
  (dataset, donor, systems, root, metametric_name, dial_grid, synthesis, tie_seed, num_permutations, seed,
   adjacent_only) = job
  r = donor_synth25_orientation_all_pools(
      dataset, donor, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
      synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
      adjacent_only=adjacent_only,
  )
  return donor, r


def dataset_synth25_orientation_all_pools(
    dataset: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    candidates: list[str] | None = None,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    progress: bool = False,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
    workers: int = 1,
) -> dict[str, dict[str, float]]:
  """{pool_name: {'A': .., 'B': .., 'T': ..}} for every pool in POOL_BLOCKS --
  the all-pools counterpart of dataset_synth25_orientation, mean over every
  usable donor of donor_synth25_orientation_all_pools. NaN for a
  (pool, family) cell with zero usable donors.

  workers: >1 dispatches every donor's donor_synth25_orientation_all_pools
  call to a ProcessPoolExecutor (_donor_synth25_job) instead of running
  sequentially -- this is the expensive step (one SPA/PA permutation-test
  sweep per dial per family per pool per donor, no alpha sweep to amortize
  it against, unlike compute_scorer_orientation_vs_alpha.py's donor loop),
  and donors are independent. Default 1 (sequential, original behavior)."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)

  t0 = time.time()
  per_donor = []
  n = len(candidates)

  def _log(i, donor, r):
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {donor} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  if workers > 1:
    jobs = [(dataset, donor, systems, root, metametric_name, dial_grid, synthesis, tie_seed, num_permutations,
              seed, adjacent_only) for donor in candidates]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
      for i, (donor, r) in enumerate(ex.map(_donor_synth25_job, jobs), 1):
        if r is not None:
          per_donor.append(r)
        _log(i, donor, r)
  else:
    for i, donor in enumerate(candidates, 1):
      r = donor_synth25_orientation_all_pools(
          dataset, donor, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
          synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
          adjacent_only=adjacent_only,
      )
      if r is not None:
        per_donor.append(r)
      _log(i, donor, r)

  out = {}
  for pool_name in POOL_BLOCKS:
    out[pool_name] = {}
    for label in FAMILIES:
      vals = [d[pool_name][label] for d in per_donor if d[pool_name][label] == d[pool_name][label]]
      out[pool_name][label] = float(np.mean(vals)) if vals else float('nan')
  return out


def pool_alpha_0s(dataset: str, systems: list[str], root: str = '.', tie_seed: int = DEFAULT_TIE_SEED) -> dict[str, float]:
  """{pool_name: alpha_0} for every pool in POOL_BLOCKS -- a dataset-level
  property of the pool's own system-level Adequacy/Fluency MQM (lib.
  synth25_metametrics.pool_alpha_0), independent of any donor/scorer/
  metametric. Restricted to positions where Adequacy AND Fluency MQM are
  both available for every one of `systems` (pool_alpha_0's own mask
  requirement) -- NOT intersected with any donor's own coverage, since
  this is meant to describe the pool once per dataset, not per donor."""
  a_pos, b_pos = positional_af_matrices(dataset, systems, root=root)
  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0))
  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  return {pool_name: pool_alpha_0(a_pos, b_pos, blocks, tie_seed=tie_seed) for pool_name, blocks in POOL_BLOCKS.items()}


_DIAL_PRESET_SUFFIX = {'linear': '', 'geometric': '_geom', 'extended': '_ext', 'symmetric': '_sym'}


def orientation_tag(
    dataset: str, metametric_name: str, synthesis: str = 'offset', dial_preset: str = 'linear',
) -> str:
  """Filename tag shared by compute_synth25_orientation.py (writer) and
  plot_scorer_orientation_vs_alpha.py's --overlay-synth25 (reader).
  metametric_name is included in full ('pa_synth25'/'spa_synth25', unlike
  lib.synthetic_scorer_orientation.orientation_tag's abbreviated suffix)
  since there is no default to omit here. synthesis='offset' (the default
  dial generator) and dial_preset='linear' (the original dial grid) add no
  suffix, mirroring the scorer-orientation cache's own convention
  (lib.synthetic_scorer_orientation's _synthesis_suffix/_dial_preset_suffix);
  'additive'/'additive_mean', 'geometric', and 'extended' each get their own
  suffix so they never collide with an offset/linear cache for the same
  dataset/metametric."""
  suffix = '' if synthesis == 'offset' else f'_{synthesis}'
  suffix += _DIAL_PRESET_SUFFIX[dial_preset]
  return f'{dataset}_{metametric_name}{suffix}'


def save_synth25_orientation(
    path: str, *, dataset: str, metametric_name: str, synthesis: str, dial_preset: str, n_donors: int,
    orientation: dict[str, float],
) -> None:
  """Persists what plot_scorer_orientation_vs_alpha.py's --overlay-synth25
  needs: just the 3 dataset-level constants plus enough metadata to label
  them -- no alpha/dial grid to store, unlike lib.synthetic_scorer_
  orientation's save_orientation_data, since this is 3 numbers, not a
  curve."""
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), n_donors=np.asarray(int(n_donors)),
      A=np.asarray(float(orientation['A'])), B=np.asarray(float(orientation['B'])),
      T=np.asarray(float(orientation['T'])),
  )


def load_synth25_orientation(path: str) -> dict:
  npz = np.load(path)
  return {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']),
      'synthesis': str(npz['synthesis']),
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'linear',
      'n_donors': int(npz['n_donors']),
      'A': float(npz['A']), 'B': float(npz['B']), 'T': float(npz['T']),
  }


def pools_tag(dataset: str, metametric_name: str, synthesis: str = 'offset', dial_preset: str = 'linear') -> str:
  """Filename tag for the all-pools cache (compute_synth25_orientation.py
  --all-pools writer, plot_scorer_orientation_vs_alpha.py's
  --overlay-synth25 reader) -- same convention as orientation_tag, kept as
  a separate function (rather than reusing orientation_tag) so the two
  cache kinds (single Row-7 constants vs. all-6-pools alpha_0+orientation)
  never collide on the same filename for the same dataset/metametric."""
  suffix = '' if synthesis == 'offset' else f'_{synthesis}'
  suffix += _DIAL_PRESET_SUFFIX[dial_preset]
  return f'{dataset}_{metametric_name}_pools{suffix}'


def save_synth25_pools(
    path: str, *, dataset: str, metametric_name: str, synthesis: str, dial_preset: str, n_donors: int,
    pool_alpha0: dict[str, float], pool_orientation: dict[str, dict[str, float]],
) -> None:
  """Persists what plot_scorer_orientation_vs_alpha.py's --overlay-synth25
  pool markers need: for every pool_name in POOL_BLOCKS, its own alpha_0
  (pool_alpha0[pool_name]) and its A/B/T orientation (pool_orientation
  [pool_name]). pool_names is saved explicitly (not just relying on
  POOL_BLOCKS' current definition/order) so a cache stays self-describing
  even if POOL_BLOCKS is later reordered or extended."""
  pool_names = list(POOL_BLOCKS)
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), n_donors=np.asarray(int(n_donors)),
      pool_names=np.asarray(pool_names, dtype='<U32'),
      pool_alpha0=np.asarray([pool_alpha0[p] for p in pool_names], dtype=float),
      pool_A=np.asarray([pool_orientation[p]['A'] for p in pool_names], dtype=float),
      pool_B=np.asarray([pool_orientation[p]['B'] for p in pool_names], dtype=float),
      pool_T=np.asarray([pool_orientation[p]['T'] for p in pool_names], dtype=float),
  )


def load_synth25_pools(path: str) -> dict:
  npz = np.load(path)
  pool_names = [str(p) for p in npz['pool_names']]
  return {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']), 'synthesis': str(npz['synthesis']),
      'dial_preset': str(npz['dial_preset']), 'n_donors': int(npz['n_donors']), 'pool_names': pool_names,
      'pool_alpha0': dict(zip(pool_names, npz['pool_alpha0'].tolist())),
      'pool_A': dict(zip(pool_names, npz['pool_A'].tolist())),
      'pool_B': dict(zip(pool_names, npz['pool_B'].tolist())),
      'pool_T': dict(zip(pool_names, npz['pool_T'].tolist())),
  }


# --- J (Joint) family, negative-dial-only overlay -------------------------
#
# Everything above builds A/B/T via SYNTHESIS_GENERATORS[synthesis] (offset/
# additive/additive_mean, dial in [0, 1]). J is different on purpose: it's
# ALWAYS offset's own donor_family_joint, ALWAYS scored on NEG_J_DIAL_GRID
# ([0, -2] by -0.1, ascending EXTREMITY -- see that constant's own
# docstring for why raw-value ascending would be backwards here), with no
# synthesis/dial_preset parameter at all -- unlike A/B/T, there is nothing
# to choose. This exists so an overlay plot can pair some OTHER synthesis's
# A/B (e.g. additive_mean's) with offset's negative-dial J instead of that
# synthesis's own T -- see plot_scorer_orientation_vs_alpha.py's
# --overlay-synth25 and compute_scorer_orientation_vs_alpha.py's
# --green-family Jneg, which this is the synth25-baseline counterpart of.

def donor_synth25_orientation_J(
    dataset: str,
    donor_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=NEG_J_DIAL_GRID,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> float | None:
  """orientation_score of the J family (donor_family_joint) scored against
  `metametric_name` -- same coverage gate and mask as donor_synth25_
  orientation, but a single number (not a dict), since there is only one
  family here. None on insufficient coverage."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if donor_name in ASPECT_DONORS:
    donor_seg = _aspect_donor_seg(donor_name, a_pos, b_pos, human_seg)
  else:
    donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]

  family = donor_family_joint(donor_seg, a_pos, b_pos, dial_grid)
  vals = np.array([
      synth25_value(metametric_name, human_seg, family[dial], a_pos, b_pos,
                     tie_seed=tie_seed, num_permutations=num_permutations, seed=seed)
      for dial in dial_grid
  ])
  return orientation_score(vals, adjacent_only=adjacent_only)


def donor_synth25_orientation_J_all_pools(
    dataset: str,
    donor_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=NEG_J_DIAL_GRID,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
) -> dict[str, float] | None:
  """{pool_name: orientation_score} for EVERY pool in POOL_BLOCKS, J family
  only -- the J counterpart of donor_synth25_orientation_all_pools, built
  once per donor and scored against every pool (synth25_value's blocks
  argument) rather than rebuilt per pool. None on insufficient coverage."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, b_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if donor_name in ASPECT_DONORS:
    donor_seg = _aspect_donor_seg(donor_name, a_pos, b_pos, human_seg)
  else:
    donor_seg = load_metric_seg_scores(dataset, donor_name, systems, root=root)
  if donor_seg is None or donor_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(b_pos).any(axis=0)
           | np.isnan(donor_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, b_pos = a_pos[:, mask], b_pos[:, mask]
  donor_seg, human_seg = donor_seg[:, mask], human_seg[:, mask]

  family = donor_family_joint(donor_seg, a_pos, b_pos, dial_grid)
  out = {}
  for pool_name, blocks in POOL_BLOCKS.items():
    vals = np.array([
        synth25_value(metametric_name, human_seg, family[dial], a_pos, b_pos, tie_seed=tie_seed,
                       num_permutations=num_permutations, seed=seed, blocks=blocks)
        for dial in dial_grid
    ])
    out[pool_name] = orientation_score(vals, adjacent_only=adjacent_only)
  return out


def _donor_synth25_J_job(job):
  """_donor_synth25_job's J counterpart -- see that function's own
  docstring."""
  dataset, donor, systems, root, metametric_name, dial_grid, tie_seed, num_permutations, seed, adjacent_only = job
  r = donor_synth25_orientation_J_all_pools(
      dataset, donor, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
      tie_seed=tie_seed, num_permutations=num_permutations, seed=seed, adjacent_only=adjacent_only,
  )
  return donor, r


def dataset_synth25_orientation_J_all_pools(
    dataset: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=NEG_J_DIAL_GRID,
    candidates: list[str] | None = None,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    progress: bool = False,
    adjacent_only: bool = ORIENTATION_ADJACENT_ONLY_DEFAULT,
    workers: int = 1,
) -> dict[str, float]:
  """{pool_name: orientation_score}, mean over every usable donor of
  donor_synth25_orientation_J_all_pools -- the J counterpart of
  dataset_synth25_orientation_all_pools. NaN for a pool with zero usable
  donors. workers: see dataset_synth25_orientation_all_pools's own
  docstring -- same independent-per-donor parallelization."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)

  t0 = time.time()
  per_donor = []
  n = len(candidates)

  def _log(i, donor, r):
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {donor} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  if workers > 1:
    jobs = [(dataset, donor, systems, root, metametric_name, dial_grid, tie_seed, num_permutations, seed,
              adjacent_only) for donor in candidates]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
      for i, (donor, r) in enumerate(ex.map(_donor_synth25_J_job, jobs), 1):
        if r is not None:
          per_donor.append(r)
        _log(i, donor, r)
  else:
    for i, donor in enumerate(candidates, 1):
      r = donor_synth25_orientation_J_all_pools(
          dataset, donor, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
          tie_seed=tie_seed, num_permutations=num_permutations, seed=seed, adjacent_only=adjacent_only,
      )
      if r is not None:
        per_donor.append(r)
      _log(i, donor, r)

  out = {}
  for pool_name in POOL_BLOCKS:
    vals = [d[pool_name] for d in per_donor if d[pool_name] == d[pool_name]]
    out[pool_name] = float(np.mean(vals)) if vals else float('nan')
  return out


def pools_tag_J(dataset: str, metametric_name: str, dial_preset: str = 'neg') -> str:
  """Filename tag for the J-family all-pools cache -- no synthesis
  component (unlike pools_tag), since J's construction here is always
  offset's donor_family_joint, never a synthesis choice. dial_preset:
  'neg' (default -- NEG_J_DIAL_GRID, the historical/only grid before
  --green-family ABTJ existed) adds no suffix, keeping existing cache
  filenames valid; 'symmetric' (lib.synthetic_scorers.ABTJ_J_DIAL_GRID,
  paired with ABTJ) gets '_sym' so it never collides with a 'neg' cache
  for the same dataset/metametric."""
  suffix = '' if dial_preset == 'neg' else '_sym'
  return f'{dataset}_{metametric_name}_pools_J{suffix}'


def save_synth25_pools_J(
    path: str, *, dataset: str, metametric_name: str, n_donors: int, pool_orientation: dict[str, float],
    dial_preset: str = 'neg',
) -> None:
  """Persists the J counterpart of save_synth25_pools: just pool_J (no
  alpha0 -- reuse pool_alpha_0s/the main A/B/T pools cache's own pool_alpha0
  for that, it's family-independent) plus enough metadata to label it,
  including which dial grid J itself used (dial_preset -- 'neg' or
  'symmetric', see pools_tag_J)."""
  pool_names = list(POOL_BLOCKS)
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      n_donors=np.asarray(int(n_donors)), dial_preset=np.asarray(dial_preset),
      pool_names=np.asarray(pool_names, dtype='<U32'),
      pool_J=np.asarray([pool_orientation[p] for p in pool_names], dtype=float),
  )


def load_synth25_pools_J(path: str) -> dict:
  npz = np.load(path)
  pool_names = [str(p) for p in npz['pool_names']]
  return {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']), 'n_donors': int(npz['n_donors']),
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'neg',
      'pool_names': pool_names, 'pool_J': dict(zip(pool_names, npz['pool_J'].tolist())),
  }
