"""Cross-dataset scorer-ranking consistency (action_plan.md section 6.5):
for each of the five meta-metrics, score every available automatic metric
against the human All-MQM ranking within each of our 15 real system-sets,
then measure how consistent those per-dataset scorer-rankings are with
each other via the scorer-pair-weighted Kendall's tau defined in
action_plan.md 6.5 (equivalently: one Kendall's tau pooled over every
(scorer-pair, dataset-pair) comparison -- see codes/lib/consistency.py).

This treats our 15 real WMT system-sets themselves as the "subsets" being
compared (as opposed to bootstrap-resampled system subsets within one
set) -- the natural real-data instance of section 6.5's inconsistency
metric, and the one that actually needs the scorer-count weighting (metric
rosters differ sharply across years, see artifacts/shared_metrics.md).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.consistency import weighted_consistency
from lib.dataset_dirs import DATASET_DIRS
from lib.metametrics import METAMETRICS

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATASETS = list(DATASET_DIRS.keys())


if __name__ == '__main__':
  results = {}
  for mm in METAMETRICS:
    print(f'Computing {mm}...', file=sys.stderr)
    results[mm] = weighted_consistency(mm, DATASETS, root=ROOT)

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)

  print(f"{'metametric':10} {'tau_bar':>8} {'inconsistency':>13} {'total_weight':>13}")
  for mm, r in results.items():
    print(f"{mm:10} {r['tau_bar']:8.4f} {r['inconsistency']:13.4f} {r['total_weight']:13d}")

  summary_path = os.path.join(ROOT, 'artifacts', 'consistency.md')
  with open(summary_path, 'w') as f:
    f.write('# Cross-dataset scorer-ranking consistency (action_plan.md 6.5)\n\n')
    f.write(
        'Subsets = our 15 real system-sets. Within each, every automatic '
        'metric with full numeric coverage of the real (non-reference) '
        'systems is scored against the human All-MQM ranking by the given '
        'meta-metric, giving a per-dataset scorer-ranking. tau_bar is the '
        'scorer-pair-weighted mean pairwise Kendall\'s tau between every '
        'pair of datasets\' rankings, restricted each time to their shared '
        'scorers and weighted by C(n_common, 2) (action_plan.md 6.5) -- '
        'equivalently, one Kendall\'s tau pooled over every (scorer-pair, '
        'dataset-pair) comparison at once (codes/lib/consistency.py). '
        'inconsistency = 1 - tau_bar. total_weight = the pooled comparison '
        'count (sum of C(n_common, 2) over all 105 dataset pairs).\n\n'
    )
    f.write('| metametric | tau_bar | inconsistency | total_weight (scorer-pair comparisons) |\n')
    f.write('|---|---|---|---|\n')
    for mm, r in results.items():
      f.write(f"| {mm} | {r['tau_bar']:.4f} | {r['inconsistency']:.4f} | {r['total_weight']} |\n")

    f.write('\n## Scorers per dataset (metric roster available for meta-evaluation)\n\n')
    f.write('| dataset | ' + ' | '.join(METAMETRICS) + ' |\n')
    f.write('|---|' + '---|' * len(METAMETRICS) + '\n')
    for d in DATASETS:
      counts = [str(len(results[mm]['rankings'][d])) for mm in METAMETRICS]
      f.write(f'| {d} | ' + ' | '.join(counts) + ' |\n')
    f.write(
        '\n(SPA typically covers fewer scorers than the other four: it '
        'additionally requires a .seg.score file and >= 10 jointly-valid '
        'segments for both the metric and the human All-MQM scores, see '
        '_MIN_SPA_SEGMENTS in codes/lib/consistency.py.)\n'
    )

  # Per-metametric pairwise detail (105 dataset-pair rows each).
  for mm, r in results.items():
    path = os.path.join(ROOT, 'artifacts', f'consistency_{mm}_pairs.md')
    with open(path, 'w') as f:
      f.write(f'# Pairwise dataset-pair detail: {mm}\n\n')
      f.write(f"tau_bar = {r['tau_bar']:.4f}, inconsistency = {r['inconsistency']:.4f}, "
              f"total_weight = {r['total_weight']}\n\n")
      f.write('| dataset_i | dataset_j | n_common_scorers | weight (C(n,2)) | tau_ij |\n')
      f.write('|---|---|---|---|---|\n')
      for p in sorted(r['pairs'], key=lambda p: -p.weight):
        tau_str = f'{p.tau:.4f}' if p.tau == p.tau else 'n/a'
        f.write(f'| {p.dataset_i} | {p.dataset_j} | {p.n_common} | {p.weight} | {tau_str} |\n')
    print(f'Wrote {path}')
  print(f'Wrote {summary_path}')
