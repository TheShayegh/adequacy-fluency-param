"""Cross-dataset check, missing from action_plan.md's evaluation plan
(section 6): does a dataset's natural balance alpha_0 (lib.alpha.alpha_0,
section 3.2/3.10 -- the extent to which a real system pool is
adequacy-heavy) predict the ratio

    metametric(adequacy_mqm; systems) / metametric(fluency_mqm; systems)

where metametric ranges over the five of section 1.1/4 (Pearson, Spearman,
Kendall's tau, PA, SPA) and "adequacy_mqm"/"fluency_mqm" are the raw a/b
component vectors themselves, scored as if they were automatic-metric
"scorers" being meta-evaluated against the All-MQM (t=a+b) human ranking
(lib.scorer_metametrics -- the same MetaEvalInput machinery lib.consistency
uses for real external metrics).

Each scatter point is a SYSTEM SET, not necessarily a whole dataset's full
roster. Run at five levels, all exhaustive (not sampled, matching every
other subset enumeration in this project -- lib.alpha_table,
compute_alpha_table_loo.py, section 6.5's L-L design):

  - full K: one point per dataset, using every one of its real systems
    (15 points total, across mwb.mqm_scoring.SETS).
  - K-1..K-4: for EVERY dataset, every exhaustive leave-N-systems-out
    subset of its own K_d real systems, N=1,2,3,4 (recomputing alpha_0 and
    the ratio on that smaller system set) -- e.g. ende24 (K=17)
    contributes 17 points at K-1, C(17,2)=136 at K-2, C(17,3)=680 at K-3,
    C(17,4)=2380 at K-4, all colored as ende24. Total per level = sum of
    C(K_d,N) over the 15 datasets (194 / 1203 / 4721 / 13069 for N=1..4).

This mirrors compute_alpha_table_loo.py's own within-dataset system-removal
sensitivity check, extended to the adequacy/fluency ratio and pooled across
every dataset to ask: does the alpha_0-vs-ratio relationship still hold when
each dataset is probed at many different system-subset "natural balances,"
not just its one full-roster value? Points are colored by origin dataset so
each dataset's own cloud of subset-points is visually distinguishable.

SPA is the expensive part (needs a segment-level permutation test), so
DatasetContext (lib.scorer_metametrics) runs that ONCE per dataset at full
K and every subset reuses the cached pairwise p-value matrices via cheap
numpy indexing (a pair's permutation test depends only on those two
systems' own segment scores, never on who else is in the pool) -- this is
what keeps even the K-4 pass (tens of thousands of subset-points) fast.

The correlation used for the ratio-vs-alpha_0 relationship itself (--corr)
is configurable and distinct from the five Pearson/Spearman/Kendall/PA/SPA
meta-metrics that define the ratio's numerator/denominator in the first
place: 'pearson' (default) assumes linearity; 'spearman' only assumes a
monotone relationship, not a linear one, and is more robust to the ratio's
occasional extreme values -- available via --corr spearman, but not run by
default (no spearman artifacts are generated/committed unless explicitly
requested).

--outlier-mode screens each dataset's own roster first, dropping flagged
systems entirely before building contexts (see build_contexts):
  - 'none' (default): no screening, full real-system roster.
  - 'adequacy': SIMPLIFIED, NON-STANDARD -- the type7 scripts' original
    screen (iterative_single_aspect_outliers) on adequacy scores only,
    |z|>3.5, repeated until a pass flags nothing.
  - 'either': SIMPLIFIED, NON-STANDARD two-aspect version
    (iterative_single_aspect_outliers_either) -- flags a system if EITHER
    adequacy or fluency crosses threshold on the current pool, unioned and
    removed together each pass.
  - 'GK': THE PROJECT STANDARD (iterative_gk_outliers) -- flags a
    system by its JOINT (adequacy, fluency) Mahalanobis distance from a
    Gnanadesikan-Kettenring-robust bivariate fit of the pool, at
    GK_THRESHOLD (4.0); reproduces artifacts/inventory_outliers/
    gk_outlier_report.md exactly. See artifacts/inventory_outliers/
    README.md for the math and why it superseded 'adequacy'/'either'.
  - 'wmt_non_competative': NOT a statistical detector -- a hand-curated
    lookup (WMT_NON_COMPETATIVE_SYSTEMS) of WMT organizer-inserted
    calibration/MBR-reranking placeholders per dataset.
  - 'shayegh': NOT a statistical detector -- matches Shayegh et al.
    (2025)'s own 5-dataset coverage (ende23/zhen23/ende24/enes24/jazh24);
    every other dataset is UNSUPPORTED and has its entire roster dropped,
    i.e. excluded from this analysis entirely.
  - 'wmt_official': NOT a statistical detector -- the WMT organizers'/
    mt-metrics-eval toolkit's own outlier_systems metadata
    (WMT_OFFICIAL_SYSTEMS); 2020 datasets (ende20/zhen20) are UNSUPPORTED
    (their officially-listed outliers never appear in this project's
    smaller raw-TSV 2020 subset) and have their entire roster dropped,
    i.e. excluded entirely, the same convention 'shayegh' uses.
Each mode writes to its own suffixed png/md files (none: unsuffixed,
adequacy: _dropoutliers, either: _dropoutliers_either, GK: _gk,
wmt_non_competative: _wmtnoncompetative, shayegh: _shayegh, wmt_official:
_wmtofficial) rather than overwriting the others, so all seven stay
available side by side.

Final preprocessing step: any dataset whose system count AFTER whatever
screening just ran is below MIN_SYSTEMS_AFTER_PREPROCESSING (10) is dropped
from the analysis entirely -- not just excluded from some tables, but
treated as though it were never one of the 15 well-formed sets to begin
with (no row anywhere, no legend entry, not counted in "n datasets"). Under
--outlier-mode none this only screens on raw roster size; under the other
modes it compounds with whatever those already removed. Toggle with
--min-systems-gate/--no-min-systems-gate (default: on); disabling it writes
to a separate _nogate-suffixed file rather than overwriting the gated run,
so both are available side by side.

Usage: python codes/scripts/compute_af_ratio_vs_alpha0.py [--corr pearson|spearman] [--outlier-mode none|adequacy|either|GK|wmt_non_competative|shayegh|wmt_official] [--min-systems-gate | --no-min-systems-gate]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems
from lib.metametrics import METAMETRICS_ORDER
from lib.outlier_detection import (
    DEFAULT_THRESHOLD, GK_THRESHOLD, iterative_single_aspect_outliers,
    iterative_single_aspect_outliers_either, iterative_gk_outliers, wmt_non_competative_outliers,
    shayegh_et_al_outliers, wmt_official_outliers,
)
from lib.scorer_metametrics import DatasetContext, build_dataset_context, subset_af_ratio
from mwb.mqm_scoring import SETS, load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATASETS = list(SETS.keys())  # all 15 well-formed sets (generalMT2022/enzh excluded, see mqm_scoring.py)
METAMETRIC_ORDER = METAMETRICS_ORDER

METAMETRIC_LABEL = {'pearson': 'Pearson', 'spearman': 'Spearman', 'kendall': "Kendall's tau", 'pa': 'PA', 'spa': 'SPA'}

CORR_FN = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr}
CORR_LABEL = {'pearson': 'Pearson', 'spearman': 'Spearman'}

STANDARD_BLUE = (0.121, 0.466, 0.705)   # matplotlib tab:blue, matches plot_alpha0_subset_histogram.py

# One fixed color per dataset (identity, not magnitude), in DATASETS' own
# order so it never depends on which points happen to be plotted -- tab20
# has 20 discrete swatches, enough for all 15 datasets with none repeated.
DATASET_COLORS = {name: plt.get_cmap('tab20').colors[i] for i, name in enumerate(DATASETS)}


OUTLIER_MODE_SUFFIX = {
    'none': '', 'adequacy': '_dropoutliers', 'either': '_dropoutliers_either', 'GK': '_gk',
    'wmt_non_competative': '_wmtnoncompetative', 'shayegh': '_shayegh', 'wmt_official': '_wmtofficial',
}
OUTLIER_MODE_DESC = {
    'none': None,
    'adequacy': 'adequacy scores only, SIMPLIFIED/NON-STANDARD (iterative_single_aspect_outliers)',
    'either': 'adequacy OR fluency scores, either aspect, SIMPLIFIED/NON-STANDARD '
              '(iterative_single_aspect_outliers_either)',
    'wmt_non_competative': 'NOT a statistical detector -- a hand-curated lookup '
               '(WMT_NON_COMPETATIVE_SYSTEMS) of WMT organizer-inserted calibration/MBR-reranking '
               'placeholders per dataset (wmt_non_competative_outliers)',
    'shayegh': 'NOT a statistical detector -- matches Shayegh et al. (2025)\'s own coverage '
               '(ende23/zhen23: drop nothing; ende24/enes24/jazh24: drop MSLC); every OTHER '
               'dataset is UNSUPPORTED and has its entire roster dropped, i.e. excluded from '
               'this analysis entirely (shayegh_et_al_outliers)',
    'wmt_official': 'NOT a statistical detector -- the WMT organizers\'/mt-metrics-eval toolkit\'s '
               'OWN outlier_systems metadata (WMT_OFFICIAL_SYSTEMS), cross-checked against this '
               'project\'s roster; 2020 datasets (ende20/zhen20) are UNSUPPORTED -- their officially '
               'listed outliers never appear in this project\'s smaller raw-TSV 2020 subset -- and '
               'have their entire roster dropped, i.e. excluded entirely (wmt_official_outliers)',
    'GK': f'the project-standard joint (adequacy, fluency) GK-Mahalanobis screen, '
                f'threshold={GK_THRESHOLD} (iterative_gk_outliers -- see '
                'artifacts/inventory_outliers/gk_outlier_report.md and README.md)',
}

# Final preprocessing gate (module docstring): a dataset whose system count
# after screening falls below this is dropped from the analysis entirely,
# as if it were never one of the 15 well-formed sets.
MIN_SYSTEMS_AFTER_PREPROCESSING = 10


def build_contexts(
    outlier_mode: str = 'none', min_systems_gate: bool = True, gk_threshold: float | None = None,
) -> tuple[dict[str, DatasetContext], list[tuple[str, int]]]:
  """One DatasetContext per dataset that survives preprocessing -- the
  expensive step (two full-K permutation tests per dataset for SPA), done
  ONCE and reused by every subsequent subset.

  outlier_mode: 'none' (full real-system roster); 'adequacy' (SIMPLIFIED,
  NON-STANDARD -- lib.outlier_detection.iterative_single_aspect_outliers on
  adequacy scores only, the type7 scripts' original default:
  compute_type7_sensitivity.py, compute_type7_natural_robustness_grid.py,
  compute_type7_alt_target_ablation.py, and compute_alpha0_variance_by_
  leaveout.py's "without outliers" table); 'either' (SIMPLIFIED,
  NON-STANDARD -- iterative_single_aspect_outliers_either, flags a system
  if EITHER adequacy or fluency crosses threshold); or 'GK' -- THE
  PROJECT STANDARD as of artifacts/inventory_outliers/README.md:
  iterative_gk_outliers, which flags a system by its JOINT (adequacy,
  fluency) Mahalanobis distance from a Gnanadesikan-Kettenring-robust
  bivariate fit of the pool, at GK_THRESHOLD -- reproduces
  artifacts/inventory_outliers/gk_outlier_report.md exactly (same
  function, same inputs). Whichever mode, every flagged system is dropped
  from the roster entirely (not just down-weighted) BEFORE building its
  context, so K-1..K-4 subsets below then operate on the screened, smaller
  roster.

  min_systems_gate: when True (default), a dataset whose resulting system
  count is below MIN_SYSTEMS_AFTER_PREPROCESSING is dropped entirely (not
  included in the returned dict at all) rather than run with a tiny
  roster; when False, every dataset with >=2 remaining systems (the bare
  minimum alpha_0/variance need) is kept regardless of how small. Either
  way, excluded/undersized datasets are returned separately as (dataset, K)
  pairs so callers can report what was left out even though it plays no
  further part in the analysis.

  gk_threshold: only used when outlier_mode == 'GK'; overrides
  iterative_gk_outliers' own default (GK_THRESHOLD, 4.0) so other
  operating points (e.g. 3.5, 3.0) can be compared side by side -- per-
  dataset flagged counts do not vary monotonically with this value (the
  GK correlation estimate itself shifts each iteration as points are
  removed, see iterative_gk_outliers' docstring), so this is an explicit
  comparison knob, not a tuning parameter to chase a "right-looking" count."""
  contexts = {}
  excluded = []
  t0 = time.time()
  for i, name in enumerate(DATASETS, 1):
    systems = real_systems(name, root=ROOT)
    if outlier_mode == 'wmt_non_competative':
      outliers = wmt_non_competative_outliers(name, systems)
      systems = [s for s in systems if s not in outliers]
    elif outlier_mode == 'shayegh':
      # Unsupported datasets get their ENTIRE roster back as "outliers" (see
      # shayegh_et_al_outliers' docstring), so `systems` becomes empty and
      # the min-systems floor below excludes them automatically -- no
      # special-casing needed here.
      outliers = shayegh_et_al_outliers(name, systems)
      systems = [s for s in systems if s not in outliers]
    elif outlier_mode == 'wmt_official':
      # Same "unsupported -> entire roster dropped" convention as shayegh,
      # here for the 2020 datasets specifically (see wmt_official_outliers'
      # docstring) -- the min-systems floor excludes them automatically.
      outliers = wmt_official_outliers(name, systems)
      systems = [s for s in systems if s not in outliers]
    elif outlier_mode != 'none':
      sys_df = load_system_scores(name, root=ROOT)
      a_full = sys_df.loc[systems, 'a'].values
      if outlier_mode == 'adequacy':
        outliers = {s for s, _z, _it in iterative_single_aspect_outliers(systems, a_full)}
      elif outlier_mode == 'either':
        b_full = sys_df.loc[systems, 'b'].values
        outliers = {s for s, _aspect, _z, _it in iterative_single_aspect_outliers_either(systems, a_full, b_full)}
      elif outlier_mode == 'GK':
        b_full = sys_df.loc[systems, 'b'].values
        gk_kwargs = {} if gk_threshold is None else {'threshold': gk_threshold}
        outliers = {s for s, _z, _it in iterative_gk_outliers(systems, a_full, b_full, **gk_kwargs)}
      else:
        raise ValueError(f'unknown outlier_mode {outlier_mode!r}')
      systems = [s for s in systems if s not in outliers]
    elapsed = time.time() - t0
    eta = elapsed / i * (len(DATASETS) - i)
    floor = MIN_SYSTEMS_AFTER_PREPROCESSING if min_systems_gate else 2
    if len(systems) < floor:
      excluded.append((name, len(systems)))
      print(f'[{i}/{len(DATASETS)}] {name}: K={len(systems)} < {floor} -- '
            f'EXCLUDED, treated as nonexistent (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
      continue
    contexts[name] = build_dataset_context(name, root=ROOT, systems=systems)
    print(f'[{i}/{len(DATASETS)}] {name}: K={len(contexts[name].systems)} '
          f'(elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)
  return contexts, excluded


def make_points_df(contexts: dict[str, DatasetContext], drop: int) -> pd.DataFrame:
  """One row per exhaustive size-(K_d - drop) system subset, pooled over
  EVERY dataset (drop=0 -> exactly one row per dataset, the full roster).
  Columns: origin_dataset, K_subset, alpha_0, rho, ratio_<metametric> for
  each of the five meta-metrics. rho is action_plan.md 3.2/3.6's
  adequacy-fluency Pearson correlation ACROSS that subset's own systems
  (one point per system in the subset -- "across systems a and b are
  positively correlated," section 3.2; always Pearson, since that's the
  paper's own fixed definition, not tied to --corr). rho is the natural
  confound to condition on: it's what drives the reachable alpha_0 range
  (section 5), so part of any ratio-vs-alpha_0 relationship could in
  principle just be riding along with rho rather than alpha_0 itself."""
  rows = []
  for name, ctx in contexts.items():
    K_d = len(ctx.systems)
    idxs = list(range(K_d))
    for dropped in itertools.combinations(idxs, drop):
      kept = np.array([i for i in idxs if i not in dropped])
      if len(kept) < 2:
        continue
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', stats.ConstantInputWarning)
        rho = float(stats.pearsonr(ctx.a[kept], ctx.b[kept])[0])
      row = {
          'origin_dataset': name, 'K_subset': len(kept),
          'alpha_0': alpha0_of(ctx.a[kept], ctx.b[kept]), 'rho': rho,
      }
      for m in METAMETRIC_ORDER:
        row[f'ratio_{m}'] = subset_af_ratio(ctx, m, kept)
      rows.append(row)
  return pd.DataFrame(rows)


def correlate(points_df: pd.DataFrame, metametric: str, corr: str = 'pearson'):
  """Pooled r, p, n (corr method `corr`) for one meta-metric's ratio vs.
  alpha_0, over every row of `points_df` (regardless of origin dataset),
  dropping rows where the ratio is NaN (e.g. SPA unavailable for that
  dataset)."""
  col = f'ratio_{metametric}'
  valid = points_df[col].notna()
  x = points_df.loc[valid, 'alpha_0'].values
  y = points_df.loc[valid, col].values
  if len(x) < 3:
    return float('nan'), float('nan'), len(x)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', stats.ConstantInputWarning)
    r, p = CORR_FN[corr](x, y)
  return float(r), float(p), len(x)


def partial_corr_stat(x: np.ndarray, y: np.ndarray, z: np.ndarray, corr: str = 'pearson'):
  """First-order partial correlation between x and y controlling for z,
  using `corr`'s pairwise r for all three pairs (the standard textbook
  formula partial_r = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2));
  using Spearman's rho for every pairwise term is the standard rank-based
  generalization, equivalent to applying the same formula to
  rank-transformed x, y, z). Returns (partial_r, p) via the usual
  t-distributed test on n-3 degrees of freedom (one controlling
  variable)."""
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', stats.ConstantInputWarning)
    r_xy = CORR_FN[corr](x, y)[0]
    r_xz = CORR_FN[corr](x, z)[0]
    r_yz = CORR_FN[corr](y, z)[0]
  denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
  if denom == 0 or denom != denom:
    return float('nan'), float('nan')
  r_partial = (r_xy - r_xz * r_yz) / denom
  n = len(x)
  df = n - 3
  if df <= 0 or abs(r_partial) >= 1:
    return float(r_partial), float('nan')
  t_stat = r_partial * np.sqrt(df / (1 - r_partial ** 2))
  p = float(2 * stats.t.sf(abs(t_stat), df))
  return float(r_partial), p


def correlate_partial(points_df: pd.DataFrame, metametric: str, corr: str = 'pearson'):
  """Partial r, p, n for one meta-metric's ratio vs. alpha_0, controlling
  for rho (the subset's own adequacy-fluency correlation) -- same row
  filtering as correlate()."""
  col = f'ratio_{metametric}'
  valid = points_df[col].notna() & points_df['rho'].notna()
  x = points_df.loc[valid, 'alpha_0'].values
  y = points_df.loc[valid, col].values
  z = points_df.loc[valid, 'rho'].values
  if len(x) < 4:
    return float('nan'), float('nan'), len(x)
  r, p = partial_corr_stat(x, y, z, corr=corr)
  return r, p, len(x)


def per_dataset_correlate(points_df: pd.DataFrame, metametric: str, corr: str = 'pearson') -> dict[str, tuple]:
  """dataset -> (r, p, n) computed using only that dataset's OWN
  subset-points (as opposed to correlate(), which pools every dataset's
  points together) -- distinguishes "the relationship holds pooled across
  datasets" from "the relationship holds within each dataset on its own"."""
  return {name: correlate(group, metametric, corr=corr) for name, group in points_df.groupby('origin_dataset')}


def per_dataset_correlate_partial(points_df: pd.DataFrame, metametric: str, corr: str = 'pearson') -> dict[str, tuple]:
  """dataset -> (partial_r, p, n), the per-dataset counterpart of
  correlate_partial: controls for rho using only that dataset's own
  subset-points, rather than rho's pooled relationship across every
  dataset."""
  return {name: correlate_partial(group, metametric, corr=corr) for name, group in points_df.groupby('origin_dataset')}


def plot_points(
    points_df: pd.DataFrame, out_path: str, corr: str, title: str, point_size: float, label_points: bool,
    active_datasets: list[str], show_regressor: bool = True, exclude_outliers: bool = False,
):
  """exclude_outliers: hide points whose ratio value is an outlier WITHIN
  that panel's own metametric (iterative modified z-score, |z|>3.5 --
  lib.outlier_detection.iterative_single_aspect_outliers, the same masking-resistant
  screen action_plan.md 6.5 uses for system outliers, applied here to the
  plotted ratio values instead). Display-only: r/p/n in the title still
  come from correlate() on the FULL data (matching the .md tables), so the
  title's n can exceed the number of dots actually drawn."""
  fig, axes = plt.subplots(2, 3, figsize=(15, 9))
  axes = axes.flatten()
  for ax, m in zip(axes, METAMETRIC_ORDER):
    col = f'ratio_{m}'
    sub = points_df[points_df[col].notna()]
    r, p, n = correlate(points_df, m, corr=corr)

    n_excluded = 0
    if exclude_outliers and len(sub) > 0:
      labels = [str(i) for i in sub.index]
      flagged = iterative_single_aspect_outliers(labels, sub[col].values)
      if flagged:
        flagged_idx = {int(lab) for lab, _z, _it in flagged}
        n_excluded = len(flagged_idx)
        sub = sub[~sub.index.isin(flagged_idx)]

    for name, group in sub.groupby('origin_dataset'):
      ax.scatter(group['alpha_0'], group[col], s=point_size, color=DATASET_COLORS[name],
                 edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
      if label_points:
        for xi, yi in zip(group['alpha_0'], group[col]):
          ax.annotate(name, (xi, yi), fontsize=6, color='#555555',
                      xytext=(3, 3), textcoords='offset points')

    if show_regressor and len(sub) >= 2:
      x_plot, y_plot = sub['alpha_0'].values, sub[col].values
      xs = np.linspace(x_plot.min(), x_plot.max(), 100)
      slope, intercept = np.polyfit(x_plot, y_plot, 1)
      ax.plot(xs, slope * xs + intercept, color='red', linewidth=1.5, zorder=2)
    outlier_note = f', {n_excluded} outliers hidden' if n_excluded else ''
    ax.set_title(f'{METAMETRIC_LABEL[m]}: {CORR_LABEL[corr]} r={r:+.3f}, p={p:.3g}, n={n}{outlier_note}', fontsize=11)
    ax.set_xlabel(r'$\alpha_0$(system subset)')
    ax.set_ylabel('metametric(adequacy_mqm) / metametric(fluency_mqm)')

  axes[-1].axis('off')
  handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=DATASET_COLORS[d],
                         markersize=7, label=d) for d in active_datasets]
  axes[-1].legend(handles=handles, loc='center', ncol=2, fontsize=8, frameon=False,
                  title='origin dataset (dot color)')
  fig.suptitle(title, fontsize=13)
  fig.tight_layout(rect=[0, 0, 1, 0.92])
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {out_path}', file=sys.stderr)


DROP_LEVELS = [1, 2, 3, 4]
_NUM_WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
POINT_SIZE = {1: 18, 2: 10, 3: 6, 4: 4}


def level_key(drop: int) -> str:
  return 'full-K' if drop == 0 else f'K-{drop}'


def level_label(drop: int) -> str:
  if drop == 0:
    return 'Full-K'
  plural = 'system' if drop == 1 else 'systems'
  return f'K-{drop} (pooled leave-{_NUM_WORD[drop]}-{plural}-out)'


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--corr', choices=['pearson', 'spearman'], default='pearson',
                  help="correlation method for the ratio-vs-alpha_0 relationship (default: pearson); "
                       "distinct from the five meta-metrics used to build the ratio itself")
  p.add_argument('--outlier-mode',
                  choices=['none', 'adequacy', 'either', 'GK', 'wmt_non_competative', 'shayegh', 'wmt_official'],
                  default='none',
                  help="screen each dataset's own roster before building contexts (default: none). "
                       "'adequacy'/'either': SIMPLIFIED, NON-STANDARD single-aspect screens (kept for "
                       "the runs that already depend on them). 'GK': THE PROJECT STANDARD "
                       "(iterative_gk_outliers) -- joint (adequacy, fluency) GK-Mahalanobis screen, "
                       "reproducing artifacts/inventory_outliers/gk_outlier_report.md. "
                       "'wmt_non_competative': NOT a statistical detector -- drops the hand-curated "
                       "WMT_NON_COMPETATIVE_SYSTEMS lookup (organizer-inserted calibration/MBR-reranking "
                       "placeholders) per dataset. "
                       "'shayegh': NOT a statistical detector -- matches Shayegh et al. (2025)'s own "
                       "5-dataset coverage (ende23/zhen23/ende24/enes24/jazh24); every other dataset is "
                       "unsupported and excluded entirely. "
                       "'wmt_official': NOT a statistical detector -- the WMT organizers'/mt-metrics-eval "
                       "toolkit's own outlier_systems metadata; 2020 datasets (ende20/zhen20) are "
                       "unsupported and excluded entirely. Each mode writes to its own suffixed files "
                       "rather than overwriting the others.")
  p.add_argument('--min-systems-gate', action=argparse.BooleanOptionalAction, default=True,
                  help=f'drop any dataset whose post-screening system count is below '
                       f'{MIN_SYSTEMS_AFTER_PREPROCESSING} (default: on). Pass --no-min-systems-gate to '
                       'keep every dataset with >=2 remaining systems instead, however small; writes to '
                       'a separate _nogate-suffixed file rather than overwriting the gated run.')
  p.add_argument('--gk-threshold', type=float, default=None,
                  help=f'only used with --outlier-mode GK: override iterative_gk_outliers\' own '
                       f'default (GK_THRESHOLD={GK_THRESHOLD}) to compare other operating '
                       'points (e.g. 3.5, 3.0); appends _z<threshold> to the output suffix so each '
                       'threshold tried stays available side by side rather than overwriting the others.')
  args = p.parse_args()
  corr = args.corr
  gk_suffix = ''
  if args.outlier_mode == 'GK' and args.gk_threshold is not None:
    gk_suffix = f'_z{args.gk_threshold:g}'
  suffix = OUTLIER_MODE_SUFFIX[args.outlier_mode] + gk_suffix + ('' if args.min_systems_gate else '_nogate')

  gate_floor = MIN_SYSTEMS_AFTER_PREPROCESSING if args.min_systems_gate else 2
  effective_gk_threshold = args.gk_threshold if args.gk_threshold is not None else GK_THRESHOLD
  print(f'Building per-dataset contexts (system-level a/b/t + cached SPA pairwise p-values) '
        f'for {len(DATASETS)} datasets (outlier_mode={args.outlier_mode}, '
        f'gk_threshold={effective_gk_threshold if args.outlier_mode == "GK" else "n/a"}, '
        f'min_systems_gate={args.min_systems_gate}, floor={gate_floor})...', file=sys.stderr)
  contexts, excluded_datasets = build_contexts(
      outlier_mode=args.outlier_mode, min_systems_gate=args.min_systems_gate, gk_threshold=args.gk_threshold)

  outlier_mode_desc = OUTLIER_MODE_DESC[args.outlier_mode]
  if args.outlier_mode == 'GK' and args.gk_threshold is not None:
    outlier_mode_desc = (
        f'the project-standard joint (adequacy, fluency) GK-Mahalanobis screen, threshold='
        f'{args.gk_threshold:g} (OVERRIDE -- project standard is {GK_THRESHOLD}; '
        'iterative_gk_outliers -- see artifacts/inventory_outliers/gk_outlier_report'
        f'_z{args.gk_threshold:g}.md and README.md)'
    )
  active_datasets = [d for d in DATASETS if d in contexts]
  if excluded_datasets:
    print(f'\nExcluded (K < {gate_floor}, treated as nonexistent): {excluded_datasets}', file=sys.stderr)
  print(f'Active datasets: {len(active_datasets)}/{len(DATASETS)}: {active_datasets}', file=sys.stderr)

  print('\nBuilding point sets (full-K, K-1..K-4)...', file=sys.stderr)
  points = {}
  for drop in [0] + DROP_LEVELS:
    t0 = time.time()
    points[drop] = make_points_df(contexts, drop)
    print(f'  {level_key(drop)}: {len(points[drop])} points ({time.time() - t0:.1f}s)', file=sys.stderr)

  results = {}
  partial_results = {}
  for drop in [0] + DROP_LEVELS:
    key, df = level_key(drop), points[drop]
    results[key] = {}
    partial_results[key] = {}
    for m in METAMETRIC_ORDER:
      r, pv, n = correlate(df, m, corr=corr)
      results[key][m] = (r, pv, n)
      print(f'{key} {METAMETRIC_LABEL[m]}: {CORR_LABEL[corr]} r={r:+.4f}, p={pv:.4g}, n={n}', file=sys.stderr)

      pr, ppv, pn = correlate_partial(df, m, corr=corr)
      partial_results[key][m] = (pr, ppv, pn)
      print(f'{key} {METAMETRIC_LABEL[m]}: {CORR_LABEL[corr]} partial r (controlling for rho)='
            f'{pr:+.4f}, p={ppv:.4g}, n={pn}', file=sys.stderr)

  # Per-dataset correlations (only meaningful for K-1..K-4 -- full-K has a
  # single point per dataset) plus the plain average of those per-dataset
  # r's, as a check against the pooled correlation above: pooling can be
  # driven by BETWEEN-dataset spread alone even if no single dataset's own
  # subsets show the relationship, so this is the complementary view.
  per_dataset_results = {}
  per_dataset_partial_results = {}
  for drop in DROP_LEVELS:
    key, df = level_key(drop), points[drop]
    per_dataset_results[key] = {}
    per_dataset_partial_results[key] = {}
    for m in METAMETRIC_ORDER:
      per_dataset_results[key][m] = per_dataset_correlate(df, m, corr=corr)
      per_dataset_partial_results[key][m] = per_dataset_correlate_partial(df, m, corr=corr)

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  fig_paths = {0: os.path.join(ROOT, 'artifacts', f'af_ratio_vs_alpha0_full_{corr}{suffix}.png')}
  fig_names = {1: 'loo', 2: 'l2o', 3: 'l3o', 4: 'l4o'}
  for drop in DROP_LEVELS:
    fig_paths[drop] = os.path.join(
        ROOT, 'artifacts', f'af_ratio_vs_alpha0_{fig_names[drop]}_scatter_{corr}{suffix}.png')

  plot_points(points[0], fig_paths[0], corr,
              f'Full-K: one point per dataset ({len(points[0])} datasets), dots colored by dataset',
              point_size=42, label_points=True, active_datasets=active_datasets)
  for drop in DROP_LEVELS:
    plural = 'system' if drop == 1 else 'systems'
    plot_points(points[drop], fig_paths[drop], corr,
                f'K-{drop} (leave-{_NUM_WORD[drop]}-{plural}-out): every dataset\'s own leave-{_NUM_WORD[drop]}-'
                f'{plural}-out subsets pooled ({len(points[drop])} points), dots colored by origin dataset; '
                'per-panel ratio outliers hidden, no fit line',
                point_size=POINT_SIZE[drop], label_points=False, active_datasets=active_datasets,
                show_regressor=False, exclude_outliers=True)

  md_path = os.path.join(ROOT, 'artifacts', f'af_ratio_vs_alpha0_{corr}{suffix}.md')
  with open(md_path, 'w') as f:
    mode_title = f', outliers dropped -- {outlier_mode_desc}' if args.outlier_mode != 'none' else ''
    f.write(f'# Adequacy/fluency meta-metric ratio vs. natural balance alpha_0 '
            f'({CORR_LABEL[corr]} correlation{mode_title})\n\n')
    f.write(
        'For each of the five meta-metrics (Pearson, Spearman, '
        "Kendall's tau, PA, SPA), scores the raw Adequacy MQM vector and the raw "
        'Fluency MQM vector each as if it were an automatic-metric "scorer" being '
        'meta-evaluated against the All-MQM (t=a+b) human ranking '
        '(lib.scorer_metametrics), then takes the ratio '
        'metametric(adequacy_mqm)/metametric(fluency_mqm). Correlates that ratio '
        f'(via {CORR_LABEL[corr]} correlation -- see --corr) '
        'against the natural balance alpha_0 (lib.alpha.alpha_0, action_plan.md '
        '3.2/3.10) of the SYSTEM SUBSET each point was computed on. Full-K uses one '
        "point per dataset (its complete roster); K-1..K-4 pool every dataset's own "
        'exhaustive leave-N-systems-out subsets (mirroring '
        'compute_alpha_table_loo.py), each point colored by which dataset it came '
        'from.\n\n'
    )
    if args.outlier_mode != 'none':
      other_modes = ', '.join(
          f'`af_ratio_vs_alpha0_{corr}{OUTLIER_MODE_SUFFIX[m]}.md` ({m})'
          for m in ('none', 'adequacy', 'either', 'GK', 'wmt_non_competative', 'shayegh', 'wmt_official')
          if m != args.outlier_mode
      )
      f.write(
          f'**Outliers dropped**: each dataset\'s own real-system roster was first screened '
          f'on {outlier_mode_desc}, repeated until a pass flags nothing, and '
          'every flagged system dropped entirely before anything below was computed -- so K here '
          'is each dataset\'s POST-screen system count, not its raw roster size. This is a '
          f'separate run from the other outlier-modes -- see {other_modes} for those.\n\n'
      )

    gate_note = (f'any dataset whose system count after the above screening (or its raw roster, under '
                 f'"none") falls below {MIN_SYSTEMS_AFTER_PREPROCESSING} is dropped from this analysis '
                 'entirely -- treated as if it were never one of the 15 well-formed sets, not merely '
                 'excluded from a table.'
                 if args.min_systems_gate else
                 'DISABLED for this run (--no-min-systems-gate): every dataset with at least 2 remaining '
                 'systems is kept, however small its post-screening roster.')
    f.write(
        f'**Minimum-system-count gate**: {gate_note} '
        + (f'Excluded here: {", ".join(f"{name} (K={K})" for name, K in excluded_datasets)}. '
           if excluded_datasets else 'Nothing was excluded at this gate for this run. ')
        + f'{len(active_datasets)} of {len(DATASETS)} datasets remain active.\n\n'
    )

    f.write('## Dataset system counts and subset totals\n\n')
    header = ['dataset', 'K'] + [f'n_K-{d}_subsets' for d in DROP_LEVELS]
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name in active_datasets:
      K_d = len(contexts[name].systems)
      cells = [name, str(K_d)] + [str(comb(K_d, d)) for d in DROP_LEVELS]
      f.write('| ' + ' | '.join(cells) + ' |\n')
    total_cells = [f'**{len(points[d])}**' for d in DROP_LEVELS]
    f.write('| **total** | | ' + ' | '.join(total_cells) + ' |\n')

    for drop in [0] + DROP_LEVELS:
      key, label, df = level_key(drop), level_label(drop), points[drop]
      rho_mean, rho_std = float(df['rho'].mean()), float(df['rho'].std())
      f.write(f'\n## {label} correlation\n\n')
      f.write(f'rho (adequacy-fluency correlation across each point\'s own systems, action_plan.md '
              f'3.2): mean {rho_mean:+.4f}, std {rho_std:.4f} across these {len(df)} points.\n\n')
      f.write('| metametric | r | p | n | partial r (controlling for rho) | partial p |\n'
              '|---|---|---|---|---|---|\n')
      for m in METAMETRIC_ORDER:
        r, pv, n = results[key][m]
        pr, ppv, pn = partial_results[key][m]
        f.write(f'| {METAMETRIC_LABEL[m]} | {r:+.4f} | {pv:.4g} | {n} | {pr:+.4f} | {ppv:.4g} |\n')

      if key in per_dataset_results:
        for sub_label, per_ds in ((f'{label}: per-dataset correlation (computed within each dataset\'s own subsets)',
                                    per_dataset_results[key]),
                                   (f'{label}: per-dataset partial correlation (controlling for rho, within '
                                    'each dataset\'s own subsets)', per_dataset_partial_results[key])):
          f.write(f'\n### {sub_label}\n\n')
          f.write('| dataset | ' + ' | '.join(METAMETRIC_LABEL[m] for m in METAMETRIC_ORDER) + ' |\n')
          f.write('|---|' + '|'.join(['---'] * len(METAMETRIC_ORDER)) + '|\n')
          for name in active_datasets:
            cells = []
            for m in METAMETRIC_ORDER:
              # A dataset can be entirely absent from per_ds[m] if its roster
              # shrank (e.g. --drop-outliers) enough that every subset at
              # this level had fewer than 2 systems remaining (make_points_df
              # skips those), leaving it with zero points at this level.
              r, _pv, _n = per_ds[m].get(name, (float('nan'), float('nan'), 0))
              cells.append(f'{r:+.4f}' if r == r else 'NaN')
            f.write(f'| {name} | ' + ' | '.join(cells) + ' |\n')
          avg_cells = []
          for m in METAMETRIC_ORDER:
            rs = [r for r, _pv, _n in per_ds[m].values() if r == r]
            avg_cells.append(f'{np.mean(rs):+.4f}' if rs else 'NaN')
          f.write('| **average** | ' + ' | '.join(f'**{c}**' for c in avg_cells) + ' |\n')

    f.write(
        '\nPartial r/p: first-order partial correlation of ratio vs. alpha_0 controlling for rho '
        '(partial_r = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)), all three pairwise terms using '
        'the same correlation method as the main analysis; t-test on n-3 df). Compares the raw '
        'ratio-vs-alpha_0 relationship against what remains once rho\'s own contribution to both '
        'variables is removed -- if partial r stays close to raw r, the relationship is not just '
        'riding along with rho.\n\n'
        'Per-dataset correlation: for K-1..K-4, in addition to the POOLED r above (every subset-point '
        'from every dataset combined into one correlation), each dataset\'s own subset-points are '
        'ALSO correlated on their own (per_dataset_correlate), with the plain average of those '
        'per-dataset r values reported as **average**. This distinguishes two different claims: the '
        'pooled r can be driven mostly by BETWEEN-dataset spread (datasets with high alpha_0 having a '
        'high ratio); the per-dataset r values test whether the relationship also holds WITHIN a '
        'single dataset\'s own system-removal subsets, on their own.\n'
    )

    f.write('\n## Figures\n\n')
    f.write(f'- Full-K scatter (one point per dataset), with OLS fit line: `{os.path.basename(fig_paths[0])}`\n')
    for drop in DROP_LEVELS:
      plural = 'system' if drop == 1 else 'systems'
      f.write(f'- K-{drop} scatter, pooled leave-{_NUM_WORD[drop]}-{plural}-out subsets, dots colored by origin '
              f'dataset, no fit line, per-panel ratio outliers (iterative modified z-score, |z|>3.5) hidden: '
              f'`{os.path.basename(fig_paths[drop])}`\n')

  print(f'\nWrote {md_path}', file=sys.stderr)