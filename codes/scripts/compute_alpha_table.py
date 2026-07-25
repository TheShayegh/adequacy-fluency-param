"""Per-dataset descriptive table: K, alpha_0 (natural/uniform-weight balance),
the reachable range (alpha_min, alpha_max), and the balance of the two
synthesized pseudo-system pools (action_plan.md section 2.1: per-segment,
rank real systems by one component, assign the k-th best to synthetic
system k, carry the other component along under that same ranking).

Reference/human translations (refA/refB/refC/refD, Human-A/B/P, ...) are
excluded throughout: action_plan.md section 1.1 defines K as the count of
*translation systems being ranked*, and references are not competing
submissions -- including them would also inflate variance with a
qualitatively different kind of point (see notes/data_inventory.md).

alpha_0, alpha_min, alpha_max are section 3.2/3.8 quantities computed at
uniform weight / over all system pairs on the real system-level (a, b)
vectors. alpha_synth_adequacy / alpha_synth_fluency apply the same alpha_0
(uniform-weight) formula to the synthesized-by-adequacy / synthesized-by-
fluency pools instead of the real pool.

Synthesis needs a fully rectangular (system x segment) design to assign all
K ranks per segment; segments not rated by all K systems of the set (after
the modal-coverage filter mqm_scoring already applies) are dropped for this
step only -- they never contribute a complete rank-1..K assignment. This is
noted per-dataset via the n_full_segments / n_segments columns.

alpha_orig_synthAdeq_synthFlu is alpha_0 of the pooled 3K-system set (K real
+ K synthesized-by-adequacy + K synthesized-by-fluency), matching Shayegh et
al. (2025)'s own meta-evaluation pool (action_plan.md section 2.1: "a pool
of original + synthesized-by-adequacy + synthesized-by-fluency systems"),
before any of their Row 6/7 subsetting.

A second section (per dataset, one row per subset size k in [2, K-1])
reports alpha_0 exhaustively enumerated over every C(K,k) subset of that
dataset's real systems, summarized as mean_alpha_0/std_alpha_0 -- see
lib.alpha_table.alpha_0_by_subset_size for why this is exhaustive rather
than sampled, and its relevance to small-support behavior (action_plan.md
3.8, Corollary 3).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from mwb.mqm_scoring import SETS, OFFICIAL_RATINGS, load_system_scores, load_segment_scores
from lib.alpha_table import alpha_table_row, alpha_0_by_subset_size
from lib.consistency import real_systems

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

# All 15 well-formed sets (generalMT2022/enzh excluded -- see mqm_scoring.py
# SETS comment: different raw-TSV column layout, no official data either).
DATASETS = list(SETS.keys())


if __name__ == '__main__':
  rows = []
  std_by_k = {}
  for name in DATASETS:
    systems = real_systems(name, root=ROOT)
    sys_df = load_system_scores(name, root=ROOT).loc[systems]
    a = sys_df['a'].values
    b = sys_df['b'].values

    seg_df = load_segment_scores(name, root=ROOT)
    seg_df = seg_df[seg_df['system'].isin(systems)]

    row = alpha_table_row(a, b, seg_df)
    source = 'official-ratings' if name in OFFICIAL_RATINGS else 'raw-TSV'
    rows.append({'dataset': name, 'source': source, **row})
    std_by_k[name] = alpha_0_by_subset_size(a, b)
    print(f'{name}: computed alpha_0 over subsets for k=2..{row["K"] - 1}', file=sys.stderr)

  out = pd.DataFrame(rows).set_index('dataset')
  print(out.round(4).to_string())

  os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)
  md_path = os.path.join(ROOT, 'artifacts', 'alpha_table.md')
  with open(md_path, 'w') as f:
    f.write('# Per-dataset natural balance, reachable range, and synthesized-pool balance\n\n')
    f.write(
        'K = system count. var_a/var_b = uniform-weight system-level Adequacy/'
        'Fluency MQM variance (action_plan.md 3.2). alpha_0 = natural '
        '(uniform-weight) balance, var_a/(var_a+var_b). alpha_min/alpha_max = '
        'reachable range over all system pairs (3.8, Corollary 2). '
        'alpha_synth_adequacy/alpha_synth_fluency = alpha_0 of the synthesized-'
        'by-adequacy / synthesized-by-fluency pseudo-system pools (2.1). '
        'alpha_orig_synthAdeq_synthFlu = alpha_0 of the pooled 3K-system set '
        '(original + both synthesized pools), matching Shayegh et al. (2025)\'s '
        'own meta-evaluation pool. n_segments = segments used for the real '
        'system-level scores (post modal-coverage filter).\n\n'
    )
    header = ['dataset', 'source', 'K', 'var_a', 'var_b', 'alpha_0', 'alpha_min', 'alpha_max',
              'alpha_synth_adequacy', 'alpha_synth_fluency', 'alpha_orig_synthAdeq_synthFlu',
              'n_segments']
    f.write('| ' + ' | '.join(header) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(header)) + '|\n')
    for name, r in out.iterrows():
      f.write(
          f"| {name} | {r['source']} | {int(r['K'])} | {r['var_a']:.4f} | {r['var_b']:.4f} | "
          f"{r['alpha_0']:.4f} | {r['alpha_min']:.4f} | {r['alpha_max']:.4f} | "
          f"{r['alpha_synth_adequacy']:.4f} | {r['alpha_synth_fluency']:.4f} | "
          f"{r['alpha_orig_synthAdeq_synthFlu']:.4f} | "
          f"{int(r['n_segments'])} |\n"
      )

    f.write('\n## alpha_0 by subset size k, exhaustive over all C(K,k) subsets\n\n')
    f.write(
        'For each dataset and each subset size k from 2 to K-1, every C(K,k) '
        'subset of that dataset\'s real systems is enumerated exactly (not '
        'sampled) and alpha_0 (natural, uniform-weight balance, restricted to '
        'that subset) computed for each one; mean_alpha_0/std_alpha_0 '
        'summarize that size-k distribution. See '
        'lib.alpha_table.alpha_0_by_subset_size.\n\n'
    )
    for name in DATASETS:
      f.write(f'### {name}\n\n')
      f.write('| k | n_subsets | mean_alpha_0 | std_alpha_0 |\n')
      f.write('|---|---|---|---|\n')
      for _, r in std_by_k[name].iterrows():
        f.write(f"| {int(r['k'])} | {int(r['n_subsets'])} | {r['mean_alpha_0']:.4f} | {r['std_alpha_0']:.4f} |\n")
      f.write('\n')
  print(f'\nWrote {md_path}')
