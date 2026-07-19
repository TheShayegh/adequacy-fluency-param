"""Which automatic MT metrics (scorers, action_plan.md section 1.1) have
official system-level score files available for our 15 system-sets, and
which of those are shared across all/most of them -- relevant for any
meta-evaluation experiment that needs the same scorer across sets (e.g. the
subset-consistency experiment, action_plan.md section 6.5).

Reads external/mt-metrics-eval-data/mt-metrics-eval-v2/<year>/metric-scores/
<lang-pair>/*.sys.score directly (metric name + reference variant is the
file's basename minus the .sys.score suffix, e.g. "COMET-DA_2021-refA",
"BLEU-refA", "MetricX-24-hybrid-src" for a QE/reference-free variant).
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.metric_names import split_variant
from lib.dataset_dirs import DATASET_DIRS

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_ROOT_DATA = 'external/mt-metrics-eval-data/mt-metrics-eval-v2'


def metrics_for(dataset: str) -> set[str]:
  year, pair = DATASET_DIRS[dataset]
  d = os.path.join(ROOT, _ROOT_DATA, year, 'metric-scores', pair)
  if not os.path.isdir(d):
    raise FileNotFoundError(d)
  return {
      f[: -len('.sys.score')]
      for f in os.listdir(d)
      if f.endswith('.sys.score')
  }


if __name__ == '__main__':
  per_dataset = {name: metrics_for(name) for name in DATASET_DIRS}
  n_datasets = len(per_dataset)

  # Exact filename match (metric + reference-variant together).
  counts = defaultdict(int)
  for metrics in per_dataset.values():
    for m in metrics:
      counts[m] += 1
  shared_all = sorted(m for m, c in counts.items() if c == n_datasets)

  # Reference-variant stripped: groups e.g. BLEU-refA/BLEU-refb/BLEU-ref
  # together as "BLEU", but leaves genuinely different scored variants of
  # an evolving family apart (COMET-DA_2021 vs COMET-22 vs bare COMET are
  # NOT merged -- see lib/metric_names.py docstring).
  per_dataset_names = {
      name: {split_variant(m)[0] for m in metrics}
      for name, metrics in per_dataset.items()
  }
  name_counts = defaultdict(int)
  for names in per_dataset_names.values():
    for nm in names:
      name_counts[nm] += 1
  names_shared_all = sorted(nm for nm, c in name_counts.items() if c == n_datasets)

  print(f'{n_datasets} datasets. {len(counts)} distinct metric+variant files total.')
  print(f'Exact filename match shared across ALL {n_datasets}: {len(shared_all)}')
  print(f'Reference-variant-stripped name shared across ALL {n_datasets}: {len(names_shared_all)}')
  for nm in names_shared_all:
    print(f'  {nm}')

  print()
  print(f"{'dataset':10} {'n_metrics':>9} {'n_names':>8}")
  for name, metrics in per_dataset.items():
    print(f'{name:10} {len(metrics):9d} {len(per_dataset_names[name]):8d}')

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  md_path = os.path.join(ROOT, 'artifacts', 'shared_metrics.md')
  with open(md_path, 'w') as f:
    f.write('# Metrics shared across our system-sets\n\n')
    f.write(
        f'{n_datasets} datasets (the same 15 used throughout, generalMT2022/'
        'enzh excluded as usual). Each WMT year runs a largely different '
        'roster of submitted automatic metrics, and even continuing metric '
        'families get re-versioned/re-labeled year to year (COMET-2R \'20, '
        'COMET-DA_2021/COMET-MQM_2021 \'21, COMET-20/COMET-22 \'22, bare '
        'COMET \'23, COMET-22 \'24 -- four distinct scored variants, not '
        'the same file). So exact filename overlap across all 15 sets is '
        '**empty** (checked below); the useful cut is the reference-variant-'
        'stripped metric name (lib/metric_names.split_variant -- drops only '
        'the trailing refA/refB/refC/refD/ref/refb/refp/all/src/'
        'synthetic_ref token, does not otherwise unify version-tagged '
        'names).\n\n'
    )
    f.write(f'## Exact filename match shared across all {n_datasets} ({len(shared_all)})\n\n')
    for m in shared_all:
      f.write(f'- {m}\n')
    if not shared_all:
      f.write('(none)\n')

    f.write(f'\n## Reference-variant-stripped name shared across all {n_datasets} ({len(names_shared_all)})\n\n')
    for nm in names_shared_all:
      f.write(f'- {nm}\n')

    f.write('\n## Per-dataset counts\n\n')
    f.write('| dataset | year | pair | n_metric_files | n_metric_names |\n|---|---|---|---|---|\n')
    for name, (year, pair) in DATASET_DIRS.items():
      f.write(f'| {name} | {year} | {pair} | {len(per_dataset[name])} | {len(per_dataset_names[name])} |\n')

    f.write(f'\n## Coverage of every reference-variant-stripped metric name (datasets out of {n_datasets})\n\n')
    f.write('| metric name | n_datasets | datasets |\n|---|---|---|\n')
    for nm, c in sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0])):
      present_in = sorted(name for name, names in per_dataset_names.items() if nm in names)
      f.write(f'| {nm} | {c} | {", ".join(present_in)} |\n')
  print(f'\nWrote {md_path}')
