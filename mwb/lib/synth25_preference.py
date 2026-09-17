"""adequacy_preference(SPA_synth25) / fluency_preference(SPA_synth25) /
allmqm_preference(SPA_synth25): mwb.lib.synthetic_scorer_preference's
dial-pair-preference method, applied to the SPA_synth25/PA_synth25
baseline (mwb.lib.synth25_metametrics, Shayegh et al. 2025's
system-synthesis method) instead of the weighted-SPA(beta)/weighted-
PA(beta) family compute_scorer_preference_vs_beta.py sweeps. This
produces the paper's markers showing Shayegh et al. (2025)'s
synthesis-based meta-evaluation setups on the main results figure.

Same dial families (mwb.lib.synthetic_scorers' A/B/T sweep for one base scorer
scorer), same reduction (mwb.lib.synthetic_scorer_preference.preference_score:
fraction of same-family dial pairs where the metametric prefers the more
extreme dial) -- the only thing that changes is which metametric scores
each dial. Since SPA_synth25/PA_synth25 take no beta argument (PA/SPA
computed once, on the fixed Row-7 union pool), each base scorer's A/B/T family
collapses to a single preference_score per family rather than a curve
over beta. Averaged over base scorers, this gives 3 dataset-level CONSTANTS --
meant to be drawn as horizontal reference lines on an existing
preference-vs-beta plot (they do not depend on beta at all), not as new
points on the beta axis.

This is deliberately the one place system synthesis (mwb.lib.synth25_
metametrics) and scorer synthesis (mwb.lib.synthetic_scorers /
mwb.lib.synthetic_scorer_beta_grid, an unrelated axis -- see mwb.lib.synth25_
metametrics' own module docstring) are allowed to meet: mwb.lib.synth25_
metametrics stays entirely self-contained (nothing there imports scorer-
synthesis machinery), and nothing in the scorer-synthesis modules imports
this one back, so the dependency runs one way only.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time

import numpy as np

from mwb.lib.consistency import real_scorers
from mwb.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from mwb.lib.spa_plane import positional_af_matrices
from mwb.lib.synth25_metametrics import (
    DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED, DEFAULT_TIE_SEED, POOL_BLOCKS, SYNTH25_METAMETRICS,
    pool_beta_0, synth25_value,
)
from mwb.lib.synthetic_scorer_beta_grid import ASPECT_BASE_SCORERS, _aspect_base_scorer_seg
from mwb.lib.synthetic_scorer_preference import (
    NEG_J_DIAL_GRID, PREFERENCE_ADJACENT_ONLY_DEFAULT, preference_score,
)
from mwb.lib.synthetic_scorers import (
    AF_ADDITIVE_MEAN_DIAL_GRID, DIAL_GRID, SYNTHESIS_GENERATORS,
    base_scorer_family_additive_mean_af, base_scorer_family_joint,
)

# Same floor used throughout this project's SPA/synthesis machinery.
_MIN_SPA_SEGMENTS = 10

FAMILIES = ('A', 'F', 'T')


def base_scorer_synth25_preference(
    dataset: str,
    base_scorer_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
    use_af_family: bool = False, af_dial_grid=None,
) -> dict[str, float] | None:
  """{'A': .., 'F': .., 'T': ..} adequacy/fluency/allmqm preference of
  `metametric_name` for one base scorer -- mirrors mwb.lib.synthetic_scorer_
  beta_grid.base_scorer_beta_dial_grid's coverage gate and family construction
  exactly (same mask, same build_family call), but scores each dial with
  synth25_value (fixed Row-7 union pool, no beta) instead of a
  weighted-metametric-at-beta sweep, then reduces straight to
  preference_score (adjacent_only passed straight through -- see its own
  docstring) over the 11 dial values -- one number per family, no beta
  column. None if this base scorer lacks sufficient segment coverage (same gate
  base_scorer_beta_dial_grid uses).

  use_af_family=True replaces 'A'/'F' with 'AF' -- the paper's
  actual Adequacy-fluency family (mwb.lib.synthetic_scorers.
  base_scorer_family_additive_mean_af), scored on its own
  af_dial_grid (default AF_ADDITIVE_MEAN_DIAL_GRID) instead of
  `dial_grid`. This is the Shayegh-et-al-2025-baseline counterpart of
  mwb.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid's identically-
  named parameter -- see that function for why the paper needs this
  instead of separate single-aspect families."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = _aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  base_scorer_seg, human_seg = base_scorer_seg[:, mask], human_seg[:, mask]

  out = {}
  if use_af_family:
    families = [('AF', base_scorer_family_additive_mean_af(
        base_scorer_seg, a_pos, f_pos, af_dial_grid or AF_ADDITIVE_MEAN_DIAL_GRID))]
  else:
    families = [('A', build_family(base_scorer_seg, a_pos, dial_grid)), ('F', build_family(base_scorer_seg, f_pos, dial_grid))]
  families.append(('T', build_family(base_scorer_seg, human_seg, dial_grid)))
  for label, family in families:
    vals = np.array([
        synth25_value(metametric_name, human_seg, family[dial], a_pos, f_pos,
                       tie_seed=tie_seed, num_permutations=num_permutations, seed=seed)
        for dial in family
    ])
    out[label] = preference_score(vals, adjacent_only=adjacent_only)
  return out


def dataset_synth25_preference(
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
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
    use_af_family: bool = False, af_dial_grid=None,
) -> dict[str, float]:
  """Dataset-level {'A': .., 'F': .., 'T': ..} (or {'AF': .., 'T': ..} with
  use_af_family=True -- see base_scorer_synth25_preference) -- mean
  over every usable base scorer (candidates, default real_scorers(dataset,
  systems=systems, require_seg_scores=True), this project's canonical
  scorer screen, same roster compute_scorer_preference_vs_beta.py uses)
  of base_scorer_synth25_preference. NaN for a family with zero usable base scorers.
  progress=True prints one '[i/n] base scorer (elapsed, eta)' line per base scorer to
  stderr (this loop runs a full SPA permutation test per dial per family
  per base scorer, so it is not cheap at dozens of base scorers)."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)
  families = ('AF', 'T') if use_af_family else ('A', 'F', 'T')

  t0 = time.time()
  per_base_scorer = []
  n = len(candidates)
  for i, base_scorer in enumerate(candidates, 1):
    r = base_scorer_synth25_preference(
        dataset, base_scorer, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
        synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
        adjacent_only=adjacent_only, use_af_family=use_af_family,
        af_dial_grid=af_dial_grid,
    )
    if r is not None:
      per_base_scorer.append(r)
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {base_scorer} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  out = {}
  for label in families:
    vals = [d[label] for d in per_base_scorer if d[label] == d[label]]
    out[label] = float(np.mean(vals)) if vals else float('nan')
  return out


def base_scorer_synth25_preference_all_pools(
    dataset: str,
    base_scorer_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=DIAL_GRID,
    synthesis: str = 'offset',
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
    use_af_family: bool = False, af_dial_grid=None,
) -> dict[str, dict[str, float]] | None:
  """{pool_name: {'A': .., 'F': .., 'T': ..}} for EVERY pool in POOL_BLOCKS,
  for one base scorer -- same coverage gate and dial-family construction as
  base_scorer_synth25_preference (in fact identical for pool_name='real+adeq+flu'),
  but each family's dial family is built ONCE and then scored against all 6
  pools (synth25_value's blocks argument) instead of rebuilding it per pool,
  since build_family does not depend on which pool the result is later
  compared against. None on insufficient coverage, matching base_scorer_synth25_
  preference's gate exactly. use_af_family/af_dial_grid:
  see base_scorer_synth25_preference -- 'AF' replaces 'A'/'F' when True."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')
  build_family = SYNTHESIS_GENERATORS[synthesis]

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = _aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  base_scorer_seg, human_seg = base_scorer_seg[:, mask], human_seg[:, mask]

  out = {pool_name: {} for pool_name in POOL_BLOCKS}
  if use_af_family:
    families = [('AF', base_scorer_family_additive_mean_af(
        base_scorer_seg, a_pos, f_pos, af_dial_grid or AF_ADDITIVE_MEAN_DIAL_GRID))]
  else:
    families = [('A', build_family(base_scorer_seg, a_pos, dial_grid)), ('F', build_family(base_scorer_seg, f_pos, dial_grid))]
  families.append(('T', build_family(base_scorer_seg, human_seg, dial_grid)))
  for label, family in families:
    for pool_name, blocks in POOL_BLOCKS.items():
      vals = np.array([
          synth25_value(metametric_name, human_seg, family[dial], a_pos, f_pos, tie_seed=tie_seed,
                         num_permutations=num_permutations, seed=seed, blocks=blocks)
          for dial in family
      ])
      out[pool_name][label] = preference_score(vals, adjacent_only=adjacent_only)
  return out


def _base_scorer_synth25_job(job):
  """One base scorer's worker-process body for dataset_synth25_preference_all_
  pools's --workers > 1 path -- module-level so it's importable/picklable
  under macOS's spawn start method. Base scorers are independent (each call only
  reads its own base scorer's segment file off disk), so this parallelizes with
  no cross-base scorer coordination needed. Returns (base scorer, result_or_None)."""
  (dataset, base_scorer, systems, root, metametric_name, dial_grid, synthesis, tie_seed, num_permutations, seed,
   adjacent_only, use_af_family, af_dial_grid) = job
  r = base_scorer_synth25_preference_all_pools(
      dataset, base_scorer, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
      synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
      adjacent_only=adjacent_only, use_af_family=use_af_family,
      af_dial_grid=af_dial_grid,
  )
  return base_scorer, r


def dataset_synth25_preference_all_pools(
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
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
    workers: int = 1,
    use_af_family: bool = False, af_dial_grid=None,
) -> dict[str, dict[str, float]]:
  """{pool_name: {'A': .., 'F': .., 'T': ..}} for every pool in POOL_BLOCKS --
  the all-pools counterpart of dataset_synth25_preference, mean over every
  usable base scorer of base_scorer_synth25_preference_all_pools. NaN for a
  (pool, family) cell with zero usable base scorers. use_af_family/
  af_dial_grid: see base_scorer_synth25_preference -- 'AF' replaces
  'A'/'F' when True (the paper's committed setup, matching Figure 3's
  markers).

  workers: >1 dispatches every base scorer's base_scorer_synth25_preference_all_pools
  call to a ProcessPoolExecutor (_base_scorer_synth25_job) instead of running
  sequentially -- this is the expensive step (one SPA/PA permutation-test
  sweep per dial per family per pool per base scorer, no beta sweep to amortize
  it against, unlike compute_scorer_preference_vs_beta.py's base scorer loop),
  and base scorers are independent. Default 1 (sequential, original behavior)."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)
  families = ('AF', 'T') if use_af_family else ('A', 'F', 'T')

  t0 = time.time()
  per_base_scorer = []
  n = len(candidates)

  def _log(i, base_scorer, r):
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {base_scorer} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  if workers > 1:
    jobs = [(dataset, base_scorer, systems, root, metametric_name, dial_grid, synthesis, tie_seed, num_permutations,
              seed, adjacent_only, use_af_family, af_dial_grid) for base_scorer in candidates]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
      for i, (base_scorer, r) in enumerate(ex.map(_base_scorer_synth25_job, jobs), 1):
        if r is not None:
          per_base_scorer.append(r)
        _log(i, base_scorer, r)
  else:
    for i, base_scorer in enumerate(candidates, 1):
      r = base_scorer_synth25_preference_all_pools(
          dataset, base_scorer, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
          synthesis=synthesis, tie_seed=tie_seed, num_permutations=num_permutations, seed=seed,
          adjacent_only=adjacent_only, use_af_family=use_af_family,
          af_dial_grid=af_dial_grid,
      )
      if r is not None:
        per_base_scorer.append(r)
      _log(i, base_scorer, r)

  out = {}
  for pool_name in POOL_BLOCKS:
    out[pool_name] = {}
    for label in families:
      vals = [d[pool_name][label] for d in per_base_scorer if d[pool_name][label] == d[pool_name][label]]
      out[pool_name][label] = float(np.mean(vals)) if vals else float('nan')
  return out


def pool_beta_0s(dataset: str, systems: list[str], root: str = '.', tie_seed: int = DEFAULT_TIE_SEED) -> dict[str, float]:
  """{pool_name: beta_0} for every pool in POOL_BLOCKS -- a dataset-level
  property of the pool's own system-level Adequacy/Fluency MQM (mwb.lib.
  synth25_metametrics.pool_beta_0), independent of any base scorer/scorer/
  metametric. Restricted to positions where Adequacy AND Fluency MQM are
  both available for every one of `systems` (pool_beta_0's own mask
  requirement) -- NOT intersected with any base scorer's own coverage, since
  this is meant to describe the pool once per dataset, not per base scorer."""
  a_pos, f_pos = positional_af_matrices(dataset, systems, root=root)
  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0))
  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  return {pool_name: pool_beta_0(a_pos, f_pos, blocks, tie_seed=tie_seed) for pool_name, blocks in POOL_BLOCKS.items()}


_DIAL_PRESET_SUFFIX = {'linear': '', 'geometric': '_geom', 'extended': '_ext', 'symmetric': '_sym'}


def preference_tag(
    dataset: str, metametric_name: str, synthesis: str = 'offset', dial_preset: str = 'linear',
) -> str:
  """Filename tag shared by compute_synth25_preference.py (writer) and
  plot_scorer_preference_vs_beta.py's --overlay-synth25 (reader).
  metametric_name is included in full ('pa_synth25'/'spa_synth25', unlike
  mwb.lib.synthetic_scorer_preference.preference_tag's abbreviated suffix)
  since there is no default to omit here. synthesis='offset' (the default
  dial generator) and dial_preset='linear' (the original dial grid) add no
  suffix, mirroring the scorer-preference cache's own convention
  (mwb.lib.synthetic_scorer_preference's _synthesis_suffix/_dial_preset_suffix);
  'additive'/'additive_mean', 'geometric', and 'extended' each get their own
  suffix so they never collide with an offset/linear cache for the same
  dataset/metametric."""
  suffix = '' if synthesis == 'offset' else f'_{synthesis}'
  suffix += _DIAL_PRESET_SUFFIX[dial_preset]
  return f'{dataset}_{metametric_name}{suffix}'


def save_synth25_preference(
    path: str, *, dataset: str, metametric_name: str, synthesis: str, dial_preset: str, n_base_scorers: int,
    preference: dict[str, float],
) -> None:
  """Persists what plot_scorer_preference_vs_beta.py's --overlay-synth25
  needs: just the dataset-level constants (A/B or AF, plus T -- whichever
  `preference` has, matching dataset_synth25_preference's own
  use_af_family-dependent output) plus enough metadata to label
  them -- no beta/dial grid to store, unlike mwb.lib.synthetic_scorer_
  preference's save_preference_data, since this is a handful of numbers,
  not a curve."""
  extra = {label: np.asarray(float(val)) for label, val in preference.items()}
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), n_base_scorers=np.asarray(int(n_base_scorers)),
      **extra,
  )


def load_synth25_preference(path: str) -> dict:
  npz = np.load(path)
  out = {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']),
      'synthesis': str(npz['synthesis']),
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'linear',
      'n_base_scorers': int(npz['n_base_scorers']),
  }
  for label in ('A', 'F', 'AF', 'T'):
    if label in npz:
      out[label] = float(npz[label])
  return out


def pools_tag(dataset: str, metametric_name: str, synthesis: str = 'offset', dial_preset: str = 'linear') -> str:
  """Filename tag for the all-pools cache (compute_synth25_preference.py
  --all-pools writer, plot_scorer_preference_vs_beta.py's
  --overlay-synth25 reader) -- same convention as preference_tag, kept as
  a separate function (rather than reusing preference_tag) so the two
  cache kinds (single Row-7 constants vs. all-6-pools beta_0+preference)
  never collide on the same filename for the same dataset/metametric."""
  suffix = '' if synthesis == 'offset' else f'_{synthesis}'
  suffix += _DIAL_PRESET_SUFFIX[dial_preset]
  return f'{dataset}_{metametric_name}_pools{suffix}'


def save_synth25_pools(
    path: str, *, dataset: str, metametric_name: str, synthesis: str, dial_preset: str, n_base_scorers: int,
    pool_beta_0: dict[str, float], pool_preference: dict[str, dict[str, float]],
) -> None:
  """Persists what plot_scorer_preference_vs_beta.py's --overlay-synth25
  pool markers need: for every pool_name in POOL_BLOCKS, its own beta_0
  (pool_beta_0[pool_name]) and its preference per family (pool_preference
  [pool_name], whichever of A/B/AF/T are present -- matching
  dataset_synth25_preference_all_pools's own use_af_family-
  dependent output). pool_names is saved explicitly (not just relying on
  POOL_BLOCKS' current definition/order) so a cache stays self-describing
  even if POOL_BLOCKS is later reordered or extended."""
  pool_names = list(POOL_BLOCKS)
  labels = list(next(iter(pool_preference.values())))
  extra = {
      f'pool_{label}': np.asarray([pool_preference[p][label] for p in pool_names], dtype=float)
      for label in labels
  }
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      synthesis=np.asarray(synthesis), dial_preset=np.asarray(dial_preset), n_base_scorers=np.asarray(int(n_base_scorers)),
      pool_names=np.asarray(pool_names, dtype='<U32'),
      pool_beta_0=np.asarray([pool_beta_0[p] for p in pool_names], dtype=float),
      **extra,
  )


def load_synth25_pools(path: str) -> dict:
  npz = np.load(path)
  pool_names = [str(p) for p in npz['pool_names']]
  out = {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']), 'synthesis': str(npz['synthesis']),
      'dial_preset': str(npz['dial_preset']), 'n_base_scorers': int(npz['n_base_scorers']), 'pool_names': pool_names,
      'pool_beta_0': dict(zip(pool_names, npz['pool_beta_0'].tolist())),
  }
  for label in ('A', 'F', 'AF', 'T'):
    key = f'pool_{label}'
    if key in npz:
      out[key] = dict(zip(pool_names, npz[key].tolist()))
  return out


# --- J (Joint) family, negative-dial-only overlay -------------------------
#
# Everything above builds A/B/T via SYNTHESIS_GENERATORS[synthesis] (offset/
# additive/additive_mean, dial in [0, 1]). J is different on purpose: it's
# ALWAYS offset's own base_scorer_family_joint, ALWAYS scored on NEG_J_DIAL_GRID
# ([0, -2] by -0.1, ascending EXTREMITY -- see that constant's own
# docstring for why raw-value ascending would be backwards here), with no
# synthesis/dial_preset parameter at all -- unlike A/B/T, there is nothing
# to choose. This exists so an overlay plot can pair some OTHER synthesis's
# A/B (e.g. additive_mean's) with offset's negative-dial J instead of that
# synthesis's own T -- see plot_scorer_preference_vs_beta.py's
# --overlay-synth25 and compute_scorer_preference_vs_beta.py's
# --family-set Jneg, which this is the synth25-baseline counterpart of.

def base_scorer_synth25_preference_J(
    dataset: str,
    base_scorer_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=NEG_J_DIAL_GRID,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
) -> float | None:
  """preference_score of the J family (base_scorer_family_joint) scored against
  `metametric_name` -- same coverage gate and mask as base_scorer_synth25_
  preference, but a single number (not a dict), since there is only one
  family here. None on insufficient coverage."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = _aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  base_scorer_seg, human_seg = base_scorer_seg[:, mask], human_seg[:, mask]

  family = base_scorer_family_joint(base_scorer_seg, a_pos, f_pos, dial_grid)
  vals = np.array([
      synth25_value(metametric_name, human_seg, family[dial], a_pos, f_pos,
                     tie_seed=tie_seed, num_permutations=num_permutations, seed=seed)
      for dial in dial_grid
  ])
  return preference_score(vals, adjacent_only=adjacent_only)


def base_scorer_synth25_preference_J_all_pools(
    dataset: str,
    base_scorer_name: str,
    systems: list[str],
    root: str = '.',
    metametric_name: str = 'spa_synth25',
    dial_grid=NEG_J_DIAL_GRID,
    tie_seed: int = DEFAULT_TIE_SEED,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
) -> dict[str, float] | None:
  """{pool_name: preference_score} for EVERY pool in POOL_BLOCKS, J family
  only -- the J counterpart of base_scorer_synth25_preference_all_pools, built
  once per base scorer and scored against every pool (synth25_value's blocks
  argument) rather than rebuilt per pool. None on insufficient coverage."""
  if metametric_name not in SYNTH25_METAMETRICS:
    raise ValueError(f'metametric_name must be one of {SYNTH25_METAMETRICS}, got {metametric_name!r}')

  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return None
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return None
  if base_scorer_name in ASPECT_BASE_SCORERS:
    base_scorer_seg = _aspect_base_scorer_seg(base_scorer_name, a_pos, f_pos, human_seg)
  else:
    base_scorer_seg = load_scorer_seg_scores(dataset, base_scorer_name, systems, root=root)
  if base_scorer_seg is None or base_scorer_seg.shape[1] != a_pos.shape[1]:
    return None

  mask = ~(np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0)
           | np.isnan(base_scorer_seg).any(axis=0) | np.isnan(human_seg).any(axis=0))
  if mask.sum() < _MIN_SPA_SEGMENTS:
    return None
  a_pos, f_pos = a_pos[:, mask], f_pos[:, mask]
  base_scorer_seg, human_seg = base_scorer_seg[:, mask], human_seg[:, mask]

  family = base_scorer_family_joint(base_scorer_seg, a_pos, f_pos, dial_grid)
  out = {}
  for pool_name, blocks in POOL_BLOCKS.items():
    vals = np.array([
        synth25_value(metametric_name, human_seg, family[dial], a_pos, f_pos, tie_seed=tie_seed,
                       num_permutations=num_permutations, seed=seed, blocks=blocks)
        for dial in dial_grid
    ])
    out[pool_name] = preference_score(vals, adjacent_only=adjacent_only)
  return out


def _base_scorer_synth25_J_job(job):
  """_base_scorer_synth25_job's J counterpart -- see that function's own
  docstring."""
  dataset, base_scorer, systems, root, metametric_name, dial_grid, tie_seed, num_permutations, seed, adjacent_only = job
  r = base_scorer_synth25_preference_J_all_pools(
      dataset, base_scorer, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
      tie_seed=tie_seed, num_permutations=num_permutations, seed=seed, adjacent_only=adjacent_only,
  )
  return base_scorer, r


def dataset_synth25_preference_J_all_pools(
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
    adjacent_only: bool = PREFERENCE_ADJACENT_ONLY_DEFAULT,
    workers: int = 1,
) -> dict[str, float]:
  """{pool_name: preference_score}, mean over every usable base scorer of
  base_scorer_synth25_preference_J_all_pools -- the J counterpart of
  dataset_synth25_preference_all_pools. NaN for a pool with zero usable
  base scorers. workers: see dataset_synth25_preference_all_pools's own
  docstring -- same independent-per-base scorer parallelization."""
  if candidates is None:
    candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)

  t0 = time.time()
  per_base_scorer = []
  n = len(candidates)

  def _log(i, base_scorer, r):
    if progress:
      elapsed = time.time() - t0
      eta = elapsed / i * (n - i)
      status = 'ok' if r is not None else 'skipped (insufficient coverage)'
      print(f'[{i}/{n}] {base_scorer} {status} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  if workers > 1:
    jobs = [(dataset, base_scorer, systems, root, metametric_name, dial_grid, tie_seed, num_permutations, seed,
              adjacent_only) for base_scorer in candidates]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
      for i, (base_scorer, r) in enumerate(ex.map(_base_scorer_synth25_J_job, jobs), 1):
        if r is not None:
          per_base_scorer.append(r)
        _log(i, base_scorer, r)
  else:
    for i, base_scorer in enumerate(candidates, 1):
      r = base_scorer_synth25_preference_J_all_pools(
          dataset, base_scorer, systems, root=root, metametric_name=metametric_name, dial_grid=dial_grid,
          tie_seed=tie_seed, num_permutations=num_permutations, seed=seed, adjacent_only=adjacent_only,
      )
      if r is not None:
        per_base_scorer.append(r)
      _log(i, base_scorer, r)

  out = {}
  for pool_name in POOL_BLOCKS:
    vals = [d[pool_name] for d in per_base_scorer if d[pool_name] == d[pool_name]]
    out[pool_name] = float(np.mean(vals)) if vals else float('nan')
  return out


def pools_tag_J(dataset: str, metametric_name: str, dial_preset: str = 'neg') -> str:
  """Filename tag for the J-family all-pools cache -- no synthesis
  component (unlike pools_tag), since J's construction here is always
  offset's base_scorer_family_joint, never a synthesis choice. dial_preset:
  'neg' (default -- NEG_J_DIAL_GRID, the historical/only grid before
  --family-set AFTJ existed) adds no suffix, keeping existing cache
  filenames valid; 'symmetric' (mwb.lib.synthetic_scorers.AFTJ_J_DIAL_GRID,
  paired with AFTJ) gets '_sym' so it never collides with a 'neg' cache
  for the same dataset/metametric."""
  suffix = '' if dial_preset == 'neg' else '_sym'
  return f'{dataset}_{metametric_name}_pools_J{suffix}'


def save_synth25_pools_J(
    path: str, *, dataset: str, metametric_name: str, n_base_scorers: int, pool_preference: dict[str, float],
    dial_preset: str = 'neg',
) -> None:
  """Persists the J counterpart of save_synth25_pools: just pool_J (no
  beta_0 -- reuse pool_beta_0s/the main A/B/T pools cache's own pool_beta_0
  for that, it's family-independent) plus enough metadata to label it,
  including which dial grid J itself used (dial_preset -- 'neg' or
  'symmetric', see pools_tag_J)."""
  pool_names = list(POOL_BLOCKS)
  np.savez(
      path, dataset=np.asarray(dataset), metametric=np.asarray(metametric_name),
      n_base_scorers=np.asarray(int(n_base_scorers)), dial_preset=np.asarray(dial_preset),
      pool_names=np.asarray(pool_names, dtype='<U32'),
      pool_J=np.asarray([pool_preference[p] for p in pool_names], dtype=float),
  )


def load_synth25_pools_J(path: str) -> dict:
  npz = np.load(path)
  pool_names = [str(p) for p in npz['pool_names']]
  return {
      'dataset': str(npz['dataset']), 'metametric': str(npz['metametric']), 'n_base_scorers': int(npz['n_base_scorers']),
      'dial_preset': str(npz['dial_preset']) if 'dial_preset' in npz else 'neg',
      'pool_names': pool_names, 'pool_J': dict(zip(pool_names, npz['pool_J'].tolist())),
  }
