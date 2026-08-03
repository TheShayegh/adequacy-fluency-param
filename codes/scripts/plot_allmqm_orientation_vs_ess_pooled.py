"""Pooled orientation-vs-ESS plots (lib.synthetic_scorer_orientation,
plot_scorer_orientation_vs_alpha.py's per-dataset version): for each of A
(adequacy), B (fluency), and the cache's third family (T=AllMQM for
synthesis='additive'/'additive_mean', J=Joint for synthesis='offset' --
whichever the loaded caches actually have, see lib.synthetic_scorer_
alpha_grid.donor_alpha_dial_grid), every cached (alpha, donor) orientation
point -- from EVERY dataset with a scorer_orientation_<tag>.npz cache, not
just one -- pooled into a single scatter (family-colored), with a second
scatter (red) of 0.02-wide-ESS/K-bucketed means of THOSE SAME raw points on
top (one averaging pass, not two -- see below), connected by a red line. No
fitted curve of any kind. Pure reanalysis of existing caches; no new
computation. Three separate output files, one per active family.

ESS(alpha) alone isn't comparable across datasets (different K), so the
x-axis here is ESS(alpha)/K -- the reliability FRACTION already used
elsewhere in this project (material/action_plan.md section 5's "ESS ~= K
trustworthy, K/3 the usual line", plot_scorer_orientation_vs_alpha.py's own
ess_over_k) -- putting every dataset on the same [1/K, 1] scale regardless
of its own K, which is what makes pooling across datasets meaningful at
all.

The red points bucket the raw (alpha, donor) points directly by ESS/K and
average within each bucket -- ONE averaging pass. (An earlier version
averaged across donors per (dataset, alpha) first and then averaged THOSE
means again per bucket -- two passes -- but since every donor at a given
alpha shares the same ESS/K, that two-pass version is just an unweighted-
by-donor-count special case of this one-pass version; bucketing the raw
points directly is both simpler and the more natural weighting, since a
(dataset, alpha) with more donors then contributes proportionally more to
its bucket instead of counting as a single flattened mean regardless of
donor count.)

Usage: python codes/scripts/plot_allmqm_orientation_vs_ess_pooled.py
           [--datasets ende21 zhen21 ...] [--step 0.01] [--metametric spa|pa]
           [--synthesis offset|additive|additive_mean] [--dial-preset linear|geometric]
(same grid/metametric/synthesis/dial-preset contract as compute_scorer_
orientation_vs_alpha.py's --union-grid mode -- one cache per dataset in
--datasets must already exist under the matching tag)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.synthetic_scorer_orientation import load_orientation_data, union_grid_tag
from lib.synthetic_scorers import default_dial_preset

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts', 'inventory_scorer_orientation')

_PANEL_TITLES = {
    'A': 'Adequacy orientation (A-family)', 'B': 'Fluency orientation (B-family)',
    'T': 'AllMQM orientation (T-family)', 'J': 'Joint orientation (J-family)',
}
_DARK_COLOR = {'A': (0.03, 0.15, 0.35), 'B': (0.35, 0.03, 0.05), 'T': (0.05, 0.30, 0.05), 'J': (0.05, 0.30, 0.05)}
_MEAN_RED = '#c81e1e'
_FAMILY_FILE_PREFIX = {'A': 'adequacy', 'B': 'fluency', 'T': 'allmqm', 'J': 'joint'}
_FAMILIES = ('A', 'B', 'T', 'J')
_BIN_WIDTH = 0.02

# Every dataset with a usable real-system pool under the default outlier
# screen (lib.consistency.real_systems) as of this analysis -- ende20,
# zhen20, enru22, ende23 are excluded (empty rosters / manual exclusion,
# see MEMORY.md).
DEFAULT_DATASETS = (
    'ende21', 'zhen21', 'ende22', 'zhen22', 'zhen23', 'heen23',
    'ende24', 'enes24', 'jazh24', 'ted_ende', 'ted_zhen',
)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS),
                       help='which per-dataset scorer_orientation_*.npz caches to pool')
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa')
  parser.add_argument('--synthesis', choices=['offset', 'additive', 'additive_mean'], default='offset')
  parser.add_argument('--dial-preset', choices=['linear', 'geometric'], default=None,
                       help='default: lib.synthetic_scorers.default_dial_preset(synthesis)')
  args = parser.parse_args()
  if args.dial_preset is None:
    args.dial_preset = default_dial_preset(args.synthesis)

  chunks = {fam: {'x': [], 'y': []} for fam in _FAMILIES}
  n_by_dataset = {}
  for ds in args.datasets:
    tag = union_grid_tag(ds, args.step, args.metametric, args.synthesis, args.dial_preset)
    path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
    if not os.path.exists(path):
      print(f'{ds}: no cache at {path}, skipping', file=sys.stderr)
      continue
    data = load_orientation_data(path)
    ess = data['ess']  # (n_alpha,)
    K = data['K']
    counted = False
    for fam in _FAMILIES:
      if data.get(fam) is None:
        continue
      F = data[fam]  # (n_donors, n_alpha)
      n_donors, n_alpha = F.shape
      ess_frac = np.tile(ess / K, n_donors)  # matches F.ravel()'s donor-major order
      f_flat = F.ravel()
      finite = ~np.isnan(f_flat)
      chunks[fam]['x'].append(ess_frac[finite])
      chunks[fam]['y'].append(f_flat[finite])
      if not counted:
        n_by_dataset[ds] = int(finite.sum())
        counted = True
    print(f'{ds}: K={K}, donors/alphas from cache shape', file=sys.stderr)

  if not n_by_dataset:
    sys.exit('no usable caches found -- run compute_scorer_orientation_vs_alpha.py for --datasets first')
  n_datasets = len(n_by_dataset)

  synthesis_title = '' if args.synthesis == 'offset' else f', synthesis={args.synthesis}'
  metametric_suffix = '' if args.metametric == 'spa' else f'_{args.metametric}'
  synthesis_suffix = '' if args.synthesis == 'offset' else f'_{args.synthesis}'
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  for fam in _FAMILIES:
    if not chunks[fam]['x']:
      print(f'{fam}: no data across any dataset, skipping', file=sys.stderr)
      continue
    x = np.concatenate(chunks[fam]['x'])
    y = np.concatenate(chunks[fam]['y'])
    n_total = len(x)
    print(f'{fam}: pooled {n_total} points across {n_datasets} datasets', file=sys.stderr)

    # Bucket the raw (alpha, donor) points directly into 0.02-wide ESS/K
    # bins and average within each bucket -- ONE averaging pass (every
    # donor at a given alpha shares that alpha's ESS/K, so this is the
    # natural, donor-count-weighted version of averaging per dataset/alpha
    # first).
    n_bins = int(round(1.0 / _BIN_WIDTH))
    bin_idx = np.clip((x / _BIN_WIDTH).astype(int), 0, n_bins - 1)
    bin_sums = np.bincount(bin_idx, weights=y, minlength=n_bins)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(invalid='ignore'):
      bin_means = bin_sums / bin_counts
    bin_centers = (np.arange(n_bins) + 0.5) * _BIN_WIDTH
    has_data = bin_counts > 0
    x_bucket = bin_centers[has_data]
    y_bucket = bin_means[has_data]
    print(f'{fam}: bucketed into {has_data.sum()}/{n_bins} non-empty {_BIN_WIDTH:g}-wide ESS/K bins',
          file=sys.stderr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_box_aspect(1)
    ax.scatter(x, y, color=_DARK_COLOR[fam], s=4, alpha=0.06, linewidths=0, zorder=2,
               label=f'{n_total} (alpha, donor) points x {n_datasets} datasets')
    # x_bucket is already ascending (bin_centers built in increasing
    # order, has_data mask preserves that), so a plain line plot connects
    # the buckets in x-order with no extra sorting needed. Gaps at empty
    # bins just draw a longer straight segment across the gap.
    ax.plot(x_bucket, y_bucket, color=_MEAN_RED, linewidth=1.3, alpha=0.7, zorder=3)
    ax.scatter(x_bucket, y_bucket, color=_MEAN_RED, s=22, alpha=0.85, linewidths=0, zorder=3,
               label=f'{len(x_bucket)} buckets of {_BIN_WIDTH:g}-wide ESS/K (mean of raw points per bucket)')
    ax.axhline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('ESS(alpha) / K  (reliability fraction)')
    ax.set_ylabel(_PANEL_TITLES[fam])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='lower right')
    fig.suptitle(f'Pooled {_PANEL_TITLES[fam]} vs. ESS/K ({args.metametric.upper()}, '
                 f'{n_datasets} datasets{synthesis_title})', fontsize=11)
    fig.tight_layout()

    plot_path = os.path.join(
        ARTIFACTS_DIR, f'{_FAMILY_FILE_PREFIX[fam]}_orientation_vs_ess_pooled{metametric_suffix}{synthesis_suffix}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {plot_path}', file=sys.stderr)
