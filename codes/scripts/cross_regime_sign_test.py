"""Cross-regime sign test (action_plan.md 6.1 x 6.5, ad hoc extension):
tests whether the mid-alpha dip seen in consistency_curves_en_de.png is
"composition conflict" (years pulled toward a common mid target from
opposite sides of their own natural alpha_0, so a mid-alpha comparison
mixes two anti-correlated rankings) or just "crossover noise" (vanishing
gaps between adjacent scorers at any single alpha, which would show up as
noise, not a sign flip).

For every ordered pair of en_de years (y, y') with y != y', compares y's
scorer-ranking at alpha_lo=0.2 against y''s scorer-ranking at alpha_hi=0.9,
using the exact same pooled tau-a machinery as lib.consistency.
pool_weighted_tau (one concordant/discordant/tie outcome per shared scorer
pair, pooled across all included year-pairs) -- only here the two sides of
each comparison sit at different targets instead of the same one. Reports
this pooled cross-regime tau_bar against two same-regime baselines (all six
years at alpha_lo; all six at alpha_hi), each with a bootstrap CI built by
resampling the flat list of per-scorer-pair outcomes.

Usage: python codes/scripts/cross_regime_sign_test.py
"""

import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.consistency import load_scorer_inputs
from lib.dataset_dirs import datasets_for_pair
from lib.metametrics import METAMETRICS, METAMETRICS_ORDER, NEEDS_SEGMENT_SCORES
from lib.reweighted_consistency import pooled, rankings_at, spa_pvalue_cache, solve_w_for_datasets

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT_DIR = os.path.join(ROOT, 'artifacts', 'data')

REGIMES = [('lo', 0.2), ('mid', 0.6), ('hi', 0.9)]
N_BOOT = 2000
BOOT_SEED = 0


def solve_w_cache(datasets, alphas):
  """w_d(alpha) for alpha in `alphas`, one solve per (dataset, alpha) --
  lib.reweight_exact is sound and complete (a certified global optimum
  every time for an in-range alpha), so no retry/fallback is needed here."""
  cache = solve_w_for_datasets(datasets, alphas, root=ROOT)
  for (d, alpha), r in cache.items():
    print(f'  {d:10} alpha={alpha:.1f}  success={r.success}  achieved={r.alpha_achieved:.6f}  '
          f'ess={r.ess:.3f}', file=sys.stderr)
  return cache


def bootstrap_ci(outcomes: list[int], n_boot=N_BOOT, seed=BOOT_SEED):
  """Percentile CI on the pooled tau_bar, resampling the flat list of
  per-shared-scorer-pair outcomes with replacement -- i.e. bootstrapping
  over shared scorer pairs, as requested, not over years/datasets."""
  arr = np.array(outcomes, dtype=float)
  rng = np.random.default_rng(seed)
  n = len(arr)
  boot_means = np.empty(n_boot)
  for i in range(n_boot):
    idx = rng.integers(0, n, size=n)
    boot_means[i] = arr[idx].mean()
  lo, hi = np.percentile(boot_means, [2.5, 97.5])
  p_ge_zero = float(np.mean(boot_means >= 0))  # one-sided evidence against "significantly negative"
  p_le_zero = float(np.mean(boot_means <= 0))  # one-sided evidence against "significantly positive"
  return float(lo), float(hi), p_ge_zero, p_le_zero


if __name__ == '__main__':
  datasets = datasets_for_pair('en-de')
  regime_alphas = [a for _, a in REGIMES]
  regime_name = {a: name for name, a in REGIMES}
  print(f'en_de datasets: {datasets}', file=sys.stderr)
  print(f'regimes: {REGIMES}\n', file=sys.stderr)

  t0 = time.time()
  print('Solving w(alpha) for every dataset x regime...', file=sys.stderr)
  w_cache = solve_w_cache(datasets, regime_alphas)
  print(f'Solved in {time.time() - t0:.1f}s\n', file=sys.stderr)

  inputs_cache = {m: {d: load_scorer_inputs(d, m, root=ROOT) for d in datasets}
                   for m in METAMETRICS_ORDER if m not in NEEDS_SEGMENT_SCORES}
  spa_cache = {d: spa_pvalue_cache(d, root=ROOT) for d in datasets} if 'spa' in NEEDS_SEGMENT_SCORES else {}

  same_pairs = list(itertools.combinations(datasets, 2))            # 15 unordered
  cross_pairs = [(y, yp) for y in datasets for yp in datasets if y != yp]  # 30 ordered

  # Every same-regime baseline (one per regime) plus every cross-regime test
  # between two DIFFERENT regimes, always oriented low-alpha-side first
  # (matches the original lo-vs-hi convention: left ranking is the lower
  # alpha, right ranking is the higher one) -- so "mid vs lo" is computed as
  # lo-vs-mid, and "mid vs hi" as mid-vs-hi.
  same_keys = [f'same_{name}' for name, _ in REGIMES]
  cross_specs = [(f'{n1}_{n2}', a1, a2) for (n1, a1), (n2, a2)
                 in itertools.combinations(REGIMES, 2)]  # (lo,mid), (lo,hi), (mid,hi)

  results = {}
  combined_outcomes = {k: [] for k in same_keys + [f'cross_{spec[0]}' for spec in cross_specs]}

  for m in METAMETRICS_ORDER:
    t1 = time.time()
    rankings = {a: rankings_at(m, datasets, a, w_cache, inputs_cache.get(m, {}), spa_cache)
                for a in regime_alphas}

    m_results = {}
    line = f'{m:10} ({0:.1f}s)  '
    for name, alpha in REGIMES:
      tau, oc, detail = pooled(same_pairs, rankings[alpha], rankings[alpha])
      ci = bootstrap_ci(oc)
      key = f'same_{name}'
      combined_outcomes[key].extend(oc)
      m_results[key] = {'tau_bar': tau, 'n': len(oc), 'ci95': ci[:2],
                         'p_ge_0': ci[2], 'p_le_0': ci[3], 'detail': detail}

    for spec_name, a1, a2 in cross_specs:
      tau, oc, detail = pooled(cross_pairs, rankings[a1], rankings[a2])
      ci = bootstrap_ci(oc)
      key = f'cross_{spec_name}'
      combined_outcomes[key].extend(oc)
      m_results[key] = {'tau_bar': tau, 'n': len(oc), 'ci95': ci[:2],
                         'p_ge_0': ci[2], 'p_le_0': ci[3], 'detail': detail}

    results[m] = m_results
    parts = [f'{k}={v["tau_bar"]:+.4f} [{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}]'
             for k, v in m_results.items()]
    print(f'{m:10} ({time.time() - t1:.1f}s)  ' + '  '.join(parts), file=sys.stderr)

  # Combined-across-metametrics headline verdict: pool every metametric's
  # flat outcome list together for a single number per regime/cross-pair.
  combined = {}
  for key, oc in combined_outcomes.items():
    tau_bar = float(np.mean(oc))
    lo, hi, p_ge_0, p_le_0 = bootstrap_ci(oc)
    combined[key] = {'tau_bar': tau_bar, 'n': len(oc), 'ci95': [lo, hi], 'p_ge_0': p_ge_0, 'p_le_0': p_le_0}

  print(f'\n=== Combined across all 5 metametrics ===', file=sys.stderr)
  for name, alpha in REGIMES:
    c = combined[f'same_{name}']
    print(f'same_{name:4} (alpha={alpha}-{alpha}): tau_bar={c["tau_bar"]:+.4f} '
          f'95% CI [{c["ci95"][0]:+.4f}, {c["ci95"][1]:+.4f}]  n={c["n"]}', file=sys.stderr)
  for spec_name, a1, a2 in cross_specs:
    c = combined[f'cross_{spec_name}']
    print(f'cross_{spec_name:8} (alpha={a1}->{a2}): tau_bar={c["tau_bar"]:+.4f} '
          f'95% CI [{c["ci95"][0]:+.4f}, {c["ci95"][1]:+.4f}]  n={c["n"]}  '
          f'P(>=0)={c["p_ge_0"]:.4f}  P(<=0)={c["p_le_0"]:.4f}', file=sys.stderr)

  def classify(same_a_key, same_b_key, cross_key):
    same_a_positive = combined[same_a_key]['ci95'][0] > 0
    same_b_positive = combined[same_b_key]['ci95'][0] > 0
    cross_negative = combined[cross_key]['ci95'][1] < 0
    cross_zero_ish = combined[cross_key]['ci95'][0] <= 0 <= combined[cross_key]['ci95'][1]
    if cross_negative and same_a_positive and same_b_positive:
      return 'COMPOSITION CONFLICT'
    elif cross_zero_ish:
      return 'CROSSOVER NOISE'
    else:
      return 'MIXED/INCONCLUSIVE (dilution-like: cross sig. positive but below both/either baseline, or other non-canonical pattern)'

  verdicts = {}
  for spec_name, a1, a2 in cross_specs:
    n1, n2 = regime_name[a1], regime_name[a2]
    v = classify(f'same_{n1}', f'same_{n2}', f'cross_{spec_name}')
    verdicts[spec_name] = v
    print(f'\nVERDICT [{n1}-vs-{n2}, alpha {a1}->{a2}]: {v}', file=sys.stderr)

  out = {
      'datasets': datasets,
      'regimes': REGIMES,
      'n_boot': N_BOOT,
      'w_diagnostics': {
          f'{d}@{alpha}': {'success': w_cache[(d, alpha)].success,
                            'alpha_achieved': w_cache[(d, alpha)].alpha_achieved,
                            'ess': w_cache[(d, alpha)].ess}
          for d in datasets for alpha in regime_alphas
      },
      'per_metametric': {
          m: {k: {kk: vv for kk, vv in v.items() if kk != 'detail'} for k, v in results[m].items()}
          for m in METAMETRICS_ORDER
      },
      'per_metametric_detail': {
          m: {k: v['detail'] for k, v in results[m].items()} for m in METAMETRICS_ORDER
      },
      'combined': combined,
      'verdicts': verdicts,
  }
  os.makedirs(OUT_DIR, exist_ok=True)
  out_path = os.path.join(OUT_DIR, 'cross_regime_sign_test_en_de.json')
  with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'\nWrote {out_path}', file=sys.stderr)
