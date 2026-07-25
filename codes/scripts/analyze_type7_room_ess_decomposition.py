"""Clean test of the room/ESS/Delta_alpha causal decomposition:

    Delta_alpha_0 --> room (= 1 - tau_0)  --\\
         |                                   >--> Delta_tau
         \\--> ESS ------------------------- /

room is a headroom/ceiling effect (more natural disagreement -> more space
to recover); ESS is a distortion-harm effect (less distortion needed to
hit the target -> more of that headroom actually gets captured). The
hypothesis is that Delta_alpha has NO residual direct effect on Delta_tau
once room and ESS are both controlled for -- i.e. full mediation through
those two channels.

Data: every row already has tau_before (=tau_0), ESS, abs(delta_alpha_0),
and delta_robustness (=Delta_tau) in the existing type7_level*_robustness_
*[_reversed|_mean].md tables -- no new computation. room = 1 - tau_before
is identical across all three direction tables for the same row (it's
computed before any reweighting), so Check 1 runs once; Checks 2-4 are
per-direction since Delta_tau differs by direction.

Checks:
  1. r(room, delta_alpha_0) -- how much of room is explained by
     delta_alpha alone (direction-independent).
  2. pcorr(Delta_tau, room | ESS) and pcorr(Delta_tau, ESS | room) -- do
     BOTH channels survive controlling for the other?
  3. pcorr(Delta_tau, delta_alpha_0 | room, ESS) -- mediation closure:
     does delta_alpha have any residual direct effect once BOTH room and
     ESS are controlled for?
  4. capture_fraction = Delta_tau / room (rows with room above a floor
     only), correlated against ESS/K -- the most literal reading of
     "distortion harm prevents full recovery of headroom."
"""

from __future__ import annotations

import re

import numpy as np
from scipy import stats

ARTIFACTS_DIR = 'artifacts'
DATASETS = ['ende23', 'zhen23', 'heen23', 'ende24', 'enes24', 'jazh24']
LEVELS = ['1-1', '2-2', '3-3']
DIRECTIONS = {'forward': '', 'reversed': '_reversed', 'mean': '_mean'}
ROOM_FLOOR = 0.02  # capture_fraction excludes rows with room below this


def parse_rows(path):
  """Returns K, list of (tau_before, delta_tau, delta_alpha0, ess)."""
  with open(path) as f:
    lines = f.readlines()
  m = re.search(r'K=(\d+)', lines[2])
  K = int(m.group(1))
  rows = []
  for line in lines:
    line = line.strip()
    if (line.startswith('| **pooled**') or line.startswith('|---')
        or 'context' in line or not line.startswith('|')):
      continue
    cells = [c.strip() for c in line.strip('|').split('|')]
    try:
      tau_before = float(cells[2])
      delta_tau = float(cells[4].replace('+', ''))
      delta_alpha0 = float(cells[7])
      ess = float(cells[8])
      rows.append((tau_before, delta_tau, delta_alpha0, ess))
    except (ValueError, IndexError):
      pass
  return K, rows


def partial_corr_multi(x, y, controls):
  """Partial correlation of x,y controlling for an arbitrary list of
  control variables, via the precision-matrix (inverse covariance) method:
  partial_r = -P[x,y] / sqrt(P[x,x] * P[y,y])."""
  M = np.column_stack([x, y] + list(controls))
  cov = np.cov(M, rowvar=False)
  prec = np.linalg.inv(cov)
  return -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])


def partial_p(r, n, n_controls):
  df = n - 2 - n_controls
  if df <= 0 or abs(r) >= 1:
    return float('nan')
  t = r * np.sqrt(df / (1 - r ** 2))
  return 2 * (1 - stats.t.cdf(abs(t), df=df))


if __name__ == '__main__':
  # ---- Check 1: r(room, delta_alpha_0), direction-independent ----
  print('=' * 90)
  print('CHECK 1: r(room, delta_alpha_0)  [room = 1 - tau_before; direction-independent]')
  print('=' * 90)
  all_room, all_da = [], []
  print(f"{'dataset':8} {'level':6} {'n':6} {'r(room,dalpha)':>15}")
  for ds in DATASETS:
    for lvl in LEVELS:
      tag = f'{ds}_level{lvl}_spa'
      K, rows = parse_rows(f'{ARTIFACTS_DIR}/type7_level{lvl}_robustness_{tag}.md')
      room = np.array([1 - r[0] for r in rows])
      da = np.array([r[2] for r in rows])
      all_room.append(room); all_da.append(da)
      r_val = np.corrcoef(room, da)[0, 1]
      print(f"{ds:8} {lvl:6} {len(rows):6} {r_val:+15.3f}")
  room_all = np.concatenate(all_room); da_all = np.concatenate(all_da)
  r_val, p_val = stats.pearsonr(room_all, da_all)
  print(f"{'POOLED':8} {'all':6} {len(room_all):6} {r_val:+15.3f} (p={p_val:.1e})")

  # ---- Checks 2-4: per direction ----
  for dirname, suffix in DIRECTIONS.items():
    print()
    print('=' * 90)
    print(f'direction={dirname}')
    print('=' * 90)

    per_cell = []  # (ds, lvl, room, ess_k, da, dtau)
    for ds in DATASETS:
      for lvl in LEVELS:
        tag = f'{ds}_level{lvl}_spa{suffix}'
        K, rows = parse_rows(f'{ARTIFACTS_DIR}/type7_level{lvl}_robustness_{tag}.md')
        room = np.array([1 - r[0] for r in rows])
        dtau = np.array([r[1] for r in rows])
        da = np.array([r[2] for r in rows])
        ess_k = np.array([r[3] / K for r in rows])
        per_cell.append((ds, lvl, room, ess_k, da, dtau))

    print('\n--- CHECK 2: pcorr(Delta_tau, room | ESS) and pcorr(Delta_tau, ESS | room) ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'pcorr(dt,room|ess)':>19} {'pcorr(dt,ess|room)':>19}")
    for ds, lvl, room, ess_k, da, dtau in per_cell:
      pc_room = partial_corr_multi(dtau, room, [ess_k])
      pc_ess = partial_corr_multi(dtau, ess_k, [room])
      print(f"{ds:8} {lvl:6} {len(dtau):6} {pc_room:+19.3f} {pc_ess:+19.3f}")
    room_all = np.concatenate([c[2] for c in per_cell])
    ess_all = np.concatenate([c[3] for c in per_cell])
    da_all = np.concatenate([c[4] for c in per_cell])
    dtau_all = np.concatenate([c[5] for c in per_cell])
    n_all = len(dtau_all)
    pc_room = partial_corr_multi(dtau_all, room_all, [ess_all])
    pc_ess = partial_corr_multi(dtau_all, ess_all, [room_all])
    print(f"{'POOLED':8} {'all':6} {n_all:6} "
          f"{pc_room:+19.3f}(p={partial_p(pc_room, n_all, 1):.1e}) "
          f"{pc_ess:+19.3f}(p={partial_p(pc_ess, n_all, 1):.1e})")

    print('\n--- CHECK 3: pcorr(Delta_tau, delta_alpha_0 | room, ESS) -- mediation closure ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'pcorr(dt,da|room,ess)':>22}")
    for ds, lvl, room, ess_k, da, dtau in per_cell:
      pc_da = partial_corr_multi(dtau, da, [room, ess_k])
      print(f"{ds:8} {lvl:6} {len(dtau):6} {pc_da:+22.3f}")
    pc_da = partial_corr_multi(dtau_all, da_all, [room_all, ess_all])
    print(f"{'POOLED':8} {'all':6} {n_all:6} {pc_da:+22.3f}(p={partial_p(pc_da, n_all, 2):.1e})")

    print(f'\n--- CHECK 4: capture_fraction = Delta_tau / room (room > {ROOM_FLOOR}) vs. ESS/K ---')
    print(f"{'dataset':8} {'level':6} {'n':6} {'r(capfrac,ess)':>15}")
    all_cf, all_ess_f = [], []
    for ds, lvl, room, ess_k, da, dtau in per_cell:
      mask = room > ROOM_FLOOR
      if mask.sum() < 3:
        print(f"{ds:8} {lvl:6} {mask.sum():6}  (too few rows with room>{ROOM_FLOOR})")
        continue
      cap_frac = dtau[mask] / room[mask]
      ess_f = ess_k[mask]
      all_cf.append(cap_frac); all_ess_f.append(ess_f)
      r_val = np.corrcoef(cap_frac, ess_f)[0, 1]
      print(f"{ds:8} {lvl:6} {mask.sum():6} {r_val:+15.3f}")
    cf_all = np.concatenate(all_cf); ess_f_all = np.concatenate(all_ess_f)
    r_val, p_val = stats.pearsonr(cf_all, ess_f_all)
    print(f"{'POOLED':8} {'all':6} {len(cf_all):6} {r_val:+15.3f} (p={p_val:.1e})")
