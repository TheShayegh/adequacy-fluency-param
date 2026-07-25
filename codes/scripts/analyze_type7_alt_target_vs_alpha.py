"""Two follow-up checks on the alt-target ablation (compute_type7_alt_
target_ablation.py), asking how much of its result overlaps with the
alpha-based forward experiment (compute_type7_sensitivity.py):

Check 1 -- overlap:
  (a) correlation(delta_stat_t, delta_alpha_0) per row: does the
      alt-target's own "how far D is from D'" gap just track delta_alpha_0
      anyway (both driven by the same underlying D-vs-D' difference)?
  (b) correlation(Delta_tau_t, Delta_tau_alpha) per row (matched by drop-set
      d across the two experiments): do the two ablations succeed/fail on
      the SAME specific removals?

Check 2 -- incremental value of delta_alpha_0:
  Does delta_alpha_0 explain additional variance in Delta_tau_t beyond
  what delta_stat_t (the alt-target's own gap) already explains?
  Reported as (i) the partial correlation pcorr(Delta_tau_t, delta_alpha_0
  | delta_stat_t) and (ii) the R^2 gain of a 2-predictor OLS fit
  (delta_stat_t + delta_alpha_0) over the 1-predictor fit (delta_stat_t
  alone).

Data: per-row detail tables written by compute_type7_alt_target_ablation.py
(type7_level<L>-<L>_alt_target_<target>_detail_<dataset>_level<L>-<L>_spa.md,
columns: context, d, tau_before, tau_after, delta_tau, delta_stat,
abs(delta_alpha_0), ESS) and the main forward tables (type7_level<L>-<L>_
robustness_<dataset>_level<L>-<L>_spa.md) for Delta_tau_alpha, matched by
the `d` (drop-set) column.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import stats

ARTIFACTS_DIR = 'artifacts'
DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']
LEVELS = ['1-1', '2-2', '3-3']
TARGETS = ['adequacy', 'fluency', 'mqm']


def parse_alt_detail(path):
  """d -> (tau_before, tau_after, delta_tau, delta_stat, delta_alpha0, ess)"""
  with open(path) as f:
    lines = f.readlines()
  out = {}
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or line.startswith('| context') or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      d = cells[1]
      tau_b, tau_a = float(cells[2]), float(cells[3])
      delta_tau = float(cells[4].replace('+', ''))
      delta_stat = float(cells[5])
      delta_alpha0 = float(cells[6])
      ess = float(cells[7])
      out[d] = (tau_b, tau_a, delta_tau, delta_stat, delta_alpha0, ess)
    except (ValueError, IndexError):
      pass
  return out


def parse_alpha_forward(path):
  """d -> delta_tau (the alpha-based forward experiment's own gain)"""
  with open(path) as f:
    lines = f.readlines()
  out = {}
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or 'context' in line or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      d = cells[1]
      delta_tau = float(cells[4].replace('+', ''))
      out[d] = delta_tau
    except (ValueError, IndexError):
      pass
  return out


def partial_corr(x, y, z):
  x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
  rxy = np.corrcoef(x, y)[0, 1]
  rxz = np.corrcoef(x, z)[0, 1]
  ryz = np.corrcoef(y, z)[0, 1]
  denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
  return (rxy - rxz * ryz) / denom if denom > 0 else float('nan')


def partial_p(r, n):
  if n <= 3 or abs(r) >= 1:
    return float('nan')
  df = n - 3
  t = r * np.sqrt(df / (1 - r ** 2))
  return 2 * (1 - stats.t.cdf(abs(t), df=df))


def r2_ols(y, X):
  """R^2 of y ~ X (X: (n,k) design matrix, intercept added here)."""
  X1 = np.column_stack([np.ones(len(y)), X])
  beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
  yhat = X1 @ beta
  ss_res = np.sum((y - yhat) ** 2)
  ss_tot = np.sum((y - y.mean()) ** 2)
  return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


if __name__ == '__main__':
  # Load everything once: alt[target][(ds,lvl)] = dict d -> tuple
  #                        alpha_fwd[(ds,lvl)] = dict d -> delta_tau
  alt = {t: {} for t in TARGETS}
  alpha_fwd = {}
  for ds in DATASETS:
    for lvl in LEVELS:
      tag = f'{ds}_level{lvl}_spa'
      alpha_fwd[(ds, lvl)] = parse_alpha_forward(
          f'{ARTIFACTS_DIR}/type7_level{lvl}_robustness_{tag}.md')
      for t in TARGETS:
        alt[t][(ds, lvl)] = parse_alt_detail(
            f'{ARTIFACTS_DIR}/type7_level{lvl}_alt_target_{t}_detail_{tag}.md')

  print('=' * 100)
  print('CHECK 1a: correlation(delta_stat_t, delta_alpha_0) per row -- does the alt-target\'s own gap')
  print('          just track delta_alpha_0 anyway?')
  print('=' * 100)
  for t in TARGETS:
    print(f'\n--- target={t} ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'r(dstat,dalpha)':>16}")
    all_ds_, all_da_ = [], []
    for ds in DATASETS:
      for lvl in LEVELS:
        rows = list(alt[t][(ds, lvl)].values())
        dstat = np.array([r[3] for r in rows])
        dalpha = np.array([r[4] for r in rows])
        all_ds_.append(dstat); all_da_.append(dalpha)
        r_val = np.corrcoef(dstat, dalpha)[0, 1]
        print(f"{ds:8} {lvl:6} {len(rows):6} {r_val:+16.3f}")
    ds_all = np.concatenate(all_ds_); da_all = np.concatenate(all_da_)
    r_val, p_val = stats.pearsonr(ds_all, da_all)
    print(f"{'POOLED':8} {'all':6} {len(ds_all):6} {r_val:+16.3f} (p={p_val:.1e})")

  print()
  print('=' * 100)
  print('CHECK 1b: correlation(Delta_tau_t, Delta_tau_alpha) per row, matched by drop-set d --')
  print('          do the two ablations succeed/fail on the SAME removals?')
  print('=' * 100)
  for t in TARGETS:
    print(f'\n--- target={t} ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'r(dtau_t,dtau_a)':>17}")
    all_dtt, all_dta = [], []
    for ds in DATASETS:
      for lvl in LEVELS:
        alt_rows = alt[t][(ds, lvl)]
        fwd_rows = alpha_fwd[(ds, lvl)]
        common = sorted(set(alt_rows) & set(fwd_rows))
        dtt = np.array([alt_rows[d][2] for d in common])
        dta = np.array([fwd_rows[d] for d in common])
        all_dtt.append(dtt); all_dta.append(dta)
        r_val = np.corrcoef(dtt, dta)[0, 1] if len(common) > 1 else float('nan')
        print(f"{ds:8} {lvl:6} {len(common):6} {r_val:+17.3f}")
    dtt_all = np.concatenate(all_dtt); dta_all = np.concatenate(all_dta)
    r_val, p_val = stats.pearsonr(dtt_all, dta_all)
    print(f"{'POOLED':8} {'all':6} {len(dtt_all):6} {r_val:+17.3f} (p={p_val:.1e})")

  print()
  print('=' * 100)
  print('CHECK 2: does delta_alpha_0 add explanatory power for Delta_tau_t BEYOND delta_stat_t?')
  print('         pcorr = partial corr(Delta_tau_t, delta_alpha_0 | delta_stat_t)')
  print('         R2(stat) = R^2 of Delta_tau_t ~ delta_stat_t alone')
  print('         R2(stat+alpha) = R^2 of Delta_tau_t ~ delta_stat_t + delta_alpha_0')
  print('=' * 100)
  for t in TARGETS:
    print(f'\n--- target={t} ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'pcorr':>10} {'R2(stat)':>10} {'R2(stat+a)':>11} {'gain':>8}")
    all_dt, all_ds_, all_da_ = [], [], []
    for ds in DATASETS:
      for lvl in LEVELS:
        rows = list(alt[t][(ds, lvl)].values())
        dtau = np.array([r[2] for r in rows])
        dstat = np.array([r[3] for r in rows])
        dalpha = np.array([r[4] for r in rows])
        all_dt.append(dtau); all_ds_.append(dstat); all_da_.append(dalpha)
        pc = partial_corr(dtau, dalpha, dstat)
        r2_1 = r2_ols(dtau, dstat.reshape(-1, 1))
        r2_2 = r2_ols(dtau, np.column_stack([dstat, dalpha]))
        print(f"{ds:8} {lvl:6} {len(rows):6} {pc:+10.3f} {r2_1:10.3f} {r2_2:11.3f} {r2_2 - r2_1:+8.3f}")
    dt_all = np.concatenate(all_dt); ds_all = np.concatenate(all_ds_); da_all = np.concatenate(all_da_)
    n_all = len(dt_all)
    pc = partial_corr(dt_all, da_all, ds_all)
    r2_1 = r2_ols(dt_all, ds_all.reshape(-1, 1))
    r2_2 = r2_ols(dt_all, np.column_stack([ds_all, da_all]))
    print(f"{'POOLED':8} {'all':6} {n_all:6} {pc:+10.3f} {r2_1:10.3f} {r2_2:11.3f} {r2_2 - r2_1:+8.3f} "
          f"(pcorr p={partial_p(pc, n_all):.1e})")
