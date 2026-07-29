"""Type-8: ICC(C,1) via UPSAMPLING, not subsampling -- the mirror image of
lib.subsampling_stability's design, structured like type7's level A-B.

Level A-B (same meaning as compute_type7_sensitivity.py): A systems are
removed from D to reach D' (|D'| = K-A); B <= A. Here, for EVERY possible
D' (every C(K,A) way to remove A systems), the "raters" for D''s ICC(C,1)
are built by ADDING BACK B of the A removed systems: D'' = D' U X for every
X in C(removed_set, B) -- C(A,B) raters per D', each of size K-A+B.

Two conditions per D'' (same pair as everywhere else this session):
  natural:    scorer_scores(D'', uniform weight)
  reweighted: scorer_scores(D'', w = solve_w_exact(D'', target_alpha))
              target_alpha defaults to alpha_0(D') -- D's OWN reduced pool's
              natural balance, i.e. every upsampled D'' is pulled back
              toward matching what the smaller D' looked like naturally.

Per D': ICC_natural, ICC_reweighted (each computed with scorers as objects
and the C(A,B) D'' variants as raters), and Delta = reweighted - natural.
ICC needs >= 2 raters -- whenever B=A, C(A,B)=1 and ICC is UNDEFINED for
every D' at that level (reported as such, not silently skipped).

Reports one row per D' plus the average of (ICC_natural, ICC_reweighted,
Delta) across every D' where both were defined. STREAMED: the header is
written immediately, and each D''s row is appended to the output file as
soon as all of its own raters finish scoring (not batched to the end) --
scoring jobs for many D' are still dispatched to one shared parallel pool
for throughput, but completion is tracked per-D' so a row can be flushed
the moment it's ready. The summary/average line is rewritten in place each
time a new row completes, so the file is a valid, readable table at any
point during a long run, not just at 100%.

Usage: python codes/scripts/compute_type8_icc_upsample.py
           --dataset ende22 --level 2-1 [--metametric spa]
           [--target-alpha-mode alpha0_Dp] [--exclude-outliers] [--tag TAG]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.icc import icc_c1
from lib.subsampling_stability import _score_one_draw
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

HEADER_COLS = ("| removed systems (D minus D') | alpha_0(D') | ICC natural | ICC reweighted | Delta | "
               "mean ESS (reweighted) |")
HEADER_SEP = '|---|---:|---:|---:|---:|---:|'


def row_line(name, a0, ic, rw, d, ess):
  ic_s = f'{ic:.4f}' if ic == ic else 'N/A'
  rw_s = f'{rw:.4f}' if rw == rw else 'N/A'
  d_s = f'{d:+.4f}' if d == d else 'N/A'
  ess_s = f'{ess:.4f}' if ess == ess else 'N/A'
  return f'| {name} | {a0:.4f} | {ic_s} | {rw_s} | {d_s} | {ess_s} |'


def summary_line(valid_deltas, valid_nats, valid_rws, valid_ess, n_total):
  if not valid_deltas:
    return '| **average** | | N/A | N/A | N/A | N/A |'
  return (f'| **average (n={len(valid_deltas)}/{n_total})** | | '
          f'**{np.mean(valid_nats):.4f}** | **{np.mean(valid_rws):.4f}** | '
          f'**{np.mean(valid_deltas):+.4f}** | **{np.mean(valid_ess):.4f}** |')


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--level', type=str, required=True, help='"A-B": A removed to reach D, B added back per rater')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  A_str, B_str = args.level.split('-')
  A, B = int(A_str), int(B_str)
  if not (1 <= B <= A):
    raise ValueError(f'--level {args.level!r} invalid: need 1 <= B <= A')

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  if not (1 <= A <= K - 2):
    raise ValueError(f'A={A} must be in [1, {K - 2}] for K={K}')
  df = load_system_scores(base, root=ROOT)
  tag = args.tag or f'{base}_level{A}-{B}_{m}'

  n_raters = math.comb(A, B)
  n_Dp = math.comb(K, A)
  print(f'D={base}: K={K}, level={A}-{B}: {n_Dp} D\' (each size {K - A}), '
        f'{n_raters} raters per D\' (each size {K - A + B})', file=sys.stderr)
  if n_raters < 2:
    print(f'WARNING: C({A},{B})={n_raters} < 2 -- ICC is UNDEFINED for every D\' at this level '
          '(needs >= 2 raters). Still computing scores/reporting alpha_0(D\'), ICC columns will be N/A.',
          file=sys.stderr)

  removed_sets = list(itertools.combinations(full, A))
  jobs = []  # (dp_idx, rater_idx, cond, job_tuple)
  dp_info = []  # (removed_set, Dp_systems, a0_Dp)
  for dpi, removed in enumerate(removed_sets):
    removed_set = set(removed)
    Dp_systems = [s for s in full if s not in removed_set]
    a_Dp, b_Dp = df.loc[Dp_systems, 'a'].values, df.loc[Dp_systems, 'b'].values
    a0_Dp = alpha0_of(a_Dp, b_Dp)
    dp_info.append((removed, Dp_systems, a0_Dp))
    for ri, X in enumerate(itertools.combinations(removed, B)):
      Dpp_systems = sorted(Dp_systems + list(X))
      jobs.append((dpi, ri, 'natural', (base, m, ROOT, Dpp_systems, None)))
      jobs.append((dpi, ri, 'reweighted', (base, m, ROOT, Dpp_systems, a0_Dp)))

  print(f'{len(jobs)} scoring jobs ({n_Dp} D\' x {n_raters} raters x 2 conditions)', file=sys.stderr)

  natural_full = scorer_scores(base, m, root=ROOT, systems=full)
  scorers = sorted(natural_full.index)

  def icc_for(dicts):
    kept = [s for s in scorers if all(sd is not None and s in sd for sd in dicts)]
    if len(kept) < 2 or n_raters < 2:
      return float('nan')
    X = np.array([[sd[s] for sd in dicts] for s in kept], dtype=float)
    return icc_c1(X)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type8_icc_upsample_level{A}-{B}_{tag}.md')

  header_lines = [
      f'# Type-8 ICC(C,1) via upsampling ({base}, level {A}-{B}, metametric={m})',
      '',
      f"D: K={K}. D' = D minus A={A} systems ({n_Dp} distinct D'). Raters per D': "
      f"D'' = D' + X, X subset of the A removed systems with |X|=B={B} "
      f"(C({A},{B})={n_raters} raters, each D'' has size {K - A + B}).",
      '',
      "target_alpha (reweighted condition) = alpha_0(D') -- D's own reduced pool's natural "
      "balance, default per this design.",
      '',
  ]
  if n_raters < 2:
    header_lines += [f"**ICC undefined at this level: C({A},{B})={n_raters} < 2 raters.**", '']
  header_lines += [HEADER_COLS, HEADER_SEP]

  # STREAMING: write the header now; each D' row gets appended (and the
  # summary line rewritten) as soon as that D' finishes, not at the end.
  with open(out_path, 'w') as f:
    f.write('\n'.join(header_lines) + '\n')
  print(f'Wrote header -> {out_path}', file=sys.stderr)

  t0 = time.time()
  pending = {dpi: {} for dpi in range(n_Dp)}  # dpi -> {(ri, cond): (scores_dict, ess)}
  row_text = [None] * n_Dp
  valid_nats, valid_rws, valid_deltas, valid_ess = [], [], [], []
  done = 0
  log_every = max(1, len(jobs) // 20)

  # Bounded in-flight submission: submitting all jobs to the pool at once
  # (one Future + dict entry per job, tens of thousands at the largest
  # levels) was observed to hang indefinitely on the biggest run of this
  # family (ende24, level 5-2, 87360 jobs) with zero CPU activity and zero
  # progress after 4+ hours -- looked like a multiprocessing queue/pipe
  # deadlock from the submission burst itself, not a slow computation (a
  # slow solve would still show CPU%, this showed none). Capping how many
  # futures are outstanding at once and topping up as they complete avoids
  # that burst entirely.
  max_inflight = max(64, (os.cpu_count() or 4) * 8)
  job_iter = iter(jobs)
  futures = {}
  with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    for _ in range(max_inflight):
      try:
        dpi, ri, cond, job = next(job_iter)
      except StopIteration:
        break
      futures[ex.submit(_score_one_draw, job)] = (dpi, ri, cond)

    while futures:
      done_set, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
      for fut in done_set:
        dpi, ri, cond = futures.pop(fut)
        sd, ess = fut.result()
        pending[dpi][(ri, cond)] = (sd, ess)
        done += 1

        try:
          ndpi, nri, ncond, njob = next(job_iter)
          futures[ex.submit(_score_one_draw, njob)] = (ndpi, nri, ncond)
        except StopIteration:
          pass

        if len(pending[dpi]) == 2 * n_raters:
          removed, Dp_systems, a0_Dp = dp_info[dpi]
          nat_dicts = [pending[dpi][(ri, 'natural')][0] for ri in range(n_raters)]
          rw_dicts = [pending[dpi][(ri, 'reweighted')][0] for ri in range(n_raters)]
          rw_ess = [pending[dpi][(ri, 'reweighted')][1] for ri in range(n_raters)]
          mean_ess_Dp = float(np.mean(rw_ess)) if rw_ess else float('nan')
          icc_nat = icc_for(nat_dicts)
          icc_rw = icc_for(rw_dicts)
          delta = icc_rw - icc_nat if (icc_nat == icc_nat and icc_rw == icc_rw) else float('nan')
          row_text[dpi] = row_line(','.join(sorted(removed)), a0_Dp, icc_nat, icc_rw, delta, mean_ess_Dp)
          if icc_nat == icc_nat and icc_rw == icc_rw:
            valid_nats.append(icc_nat)
            valid_rws.append(icc_rw)
            valid_deltas.append(delta)
            valid_ess.append(mean_ess_Dp)
          pending[dpi] = None  # free memory -- this D' is done

          # Rewrite: header + every completed row so far + current summary.
          with open(out_path, 'w') as f:
            f.write('\n'.join(header_lines) + '\n')
            for rt in row_text:
              if rt is not None:
                f.write(rt + '\n')
            f.write(summary_line(valid_deltas, valid_nats, valid_rws, valid_ess, n_Dp) + '\n')

        if done % log_every == 0:
          elapsed = time.time() - t0
          rate = done / elapsed if elapsed > 0 else 0.0
          remaining_s = (len(jobs) - done) / rate if rate > 0 else float('nan')
          n_rows_done = sum(1 for rt in row_text if rt is not None)
          print(f'  [{done}/{len(jobs)} jobs, {n_rows_done}/{n_Dp} D\' rows] '
                f'elapsed={elapsed:.1f}s est.remaining={remaining_s:.1f}s', file=sys.stderr, flush=True)

  print(f'scoring+aggregation pass: {time.time() - t0:.1f}s', file=sys.stderr)
  print(f'Wrote {out_path}', file=sys.stderr)
