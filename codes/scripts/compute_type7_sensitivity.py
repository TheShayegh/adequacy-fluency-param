"""Type-7 ranking-flip ROBUSTNESS, generalized to any level "A-B" (merges
the former per-level type-7 scripts into one, so new levels/tweaks never
need syncing across files again).

Level A-B: A total systems are ultimately removed from the pool to reach
D' (|D'| = K-A). Of those A, B are removed TOGETHER as the final
comparison step (the generalized "d", now possibly a SET of B systems,
not just one), while the remaining A-B form the "context" already removed
to define D (|D| = K-A+B). A-1 (the old single-number "level L") is the
special case B=1: one system at a time, same as before -- 1-1 = old
level 1, 2-1 = old level 2, 3-1 = old level 3. B=A (context size 0) means
D = the whole pool and D' = pool minus all A systems removed AT ONCE, no
sequential structure at all -- e.g. 2-2 = drop 2 systems together, 3-3 =
drop 3 together. A bare integer (e.g. --level 2) is shorthand for "2-1".

Iterates over every C(K, A-B) context (an unordered set of already-removed
systems) x every C(K-A+B, B) drop-set of B remaining systems, so every
D' = pool minus (context union drop_set), a set of size K-A, gets used
exactly C(A, B) times (once per way of splitting its A missing systems
into a size-(A-B) context and a size-B drop_set). B=1 recovers the old
"each D' used A times"; B=A gives C(A,A)=1, i.e. each D' appears exactly
once. Results are reported ONE ROW PER (context, drop_set) PAIR -- no
per-system averaging -- so every individual (D, D') comparison stays
visible.

  robustness_before(D, drop_set) = tau(D, D')
      pairwise-concordance consistency (this project's usual tau_a
      convention) between D's own natural ranking and D''s natural
      (unweighted) ranking -- how much of the ranking SURVIVES simply
      dropping drop_set, no correction. (= 1 - "sensitivity" in earlier
      versions of this analysis; robustness is the more natural-to-read
      complement, and it's exactly tau, the same quantity used
      everywhere else in this project.)

  robustness_after(D, drop_set)  = tau(D reweighted to alpha_0(D'), D')
      same, but D (the larger, more flexible side) is reweighted -- a
      fresh solve_w_exact(a_D, b_D, alpha_0(D')) call per row, since the
      target varies -- to match D''s OWN natural alpha; D' is left at its
      natural balance. (Reweighting the smaller D' to D's own alpha_0
      instead was an earlier attempt that misbehaves whenever drop_set
      itself dominates D's alpha_0 -- D' would have to fabricate that
      extremeness from scratch.)

  delta_robustness = robustness_after - robustness_before
      >0 means reweighting recovered some of the consistency lost by
      dropping drop_set; ~0 means no effect; <0 means reweighting made it
      worse (a real, observed outcome -- see session notes on AIRC/
      ONLINE-M: hitting an exact alpha_0 target doesn't uniquely
      determine *how* weight gets redistributed, and the solver's chosen
      admissible solution isn't guaranteed to resemble "just remove
      drop_set").

Reports every (context, d) row (alpha_0(D), alpha_0(D'), |delta_alpha_0|,
robustness_before/after, delta_robustness, ESS(D)) to artifacts/
type7_level<A>-<B>_robustness_<tag>.md. The table's own final row is NOT a
column-wise average -- it's the POOLED robustness: every row's raw
concordant/discordant/tie counts summed across the whole table first,
then tau recomputed from those totals (same pooling convention as types
1/5/6's tau_bar). Also draws a scatter of robustness_before vs.
|delta_alpha_0|, one point per (context, d) row, to artifacts/
type7_level<A>-<B>_robustness_vs_delta_alpha0_<tag>.png.

Usage: python codes/scripts/compute_type7_sensitivity.py
           --dataset ende24 --level 2-2 [--metametric spa] [--tag TAG]
           [--drop-outliers | --no-drop-outliers] [--drop-systems SYS1,SYS2,...]
           [--exclude-from-plot SYSTEM1,SYSTEM2,...]
--level: "A-B" (or a bare integer A, shorthand for "A-1"). Requires
1 <= B <= A <= K-2 (K = pool size after drops) -- D' (size K-A) needs at
least 2 systems for its scorer-ranking metametric to be meaningful.
--drop-outliers (default True): auto-detect and entirely remove this
dataset's MAD-detected outliers (lib.outlier_detection.iterative_mad_outliers) from the pool
before anything else. Pass --no-drop-outliers to keep them in.
--drop-leverage-points (default False): after --drop-outliers/--drop-systems,
run a cheap level-1-style diagnostic pass (natural tau only, no solving) on
what's left, and additionally drop any system that's a statistically
significant OUTLIER IN robustness_before GIVEN its own delta_alpha_0 --
lib.outlier_detection.studentized_outlier_test, externally studentized
residuals + Bonferroni-corrected outlier test (same method as R's car::
outlierTest). This catches a different failure mode than --drop-outliers:
systems unusually disruptive to the ranking in a way NOT explained by how
much they shift alpha_0 (e.g. zhen23's ONLINE-Y), which the adequacy-
score-based MAD test can't see since it never looks at robustness/tau.
Pass --no-drop-leverage-points to skip this. --leverage-alpha (default
0.05) sets the Bonferroni-corrected significance threshold.
--drop-systems (default none, comma-separated): additionally remove these
specific systems from the pool, on top of --drop-outliers/--drop-leverage-points.
--exclude-from-plot (optional, comma-separated): ADDITIONALLY leaves rows
whose d is one of these systems out of the scatter plot only (table still
includes them) -- e.g. one system still dominates the fit even after
--drop-outliers/--drop-systems.
"""

import argparse
import itertools
import os
import sys
import time
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lib.alpha import alpha_0 as alpha0_of
from lib.consistency import real_systems, scorer_scores
from lib.outlier_detection import iterative_mad_outliers, studentized_outlier_test
from lib.reweight_exact import solve_w_exact
from lib.reweighted_consistency import pairwise_outcomes
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')


def tau_of(n, s):
  return (s / n) if n else float('nan')


def detect_leverage_points(base, m, pool_systems, df, alpha=0.05):
  """Cheap level-1-style diagnostic (natural tau only, no solve_w_exact
  calls) over `pool_systems`: for each d, robustness_before = tau(D, D'_d)
  and delta_alpha_0 = |alpha_0(D) - alpha_0(D'_d)|, then flags systems via
  studentized_outlier_test -- ones whose robustness_before doesn't fit the
  pool's own robustness-vs-delta_alpha_0 relationship. Returns the flagged
  list (label, residual, t_stat, bonferroni_p), possibly empty."""
  if len(pool_systems) < 5:
    return []
  a_D, b_D = df.loc[pool_systems, 'a'].values, df.loc[pool_systems, 'b'].values
  a0_D = alpha0_of(a_D, b_D)
  natural_D = scorer_scores(base, m, root=ROOT, systems=pool_systems)
  labels, xs, ys = [], [], []
  for d in pool_systems:
    Dp_systems = [s for s in pool_systems if s != d]
    a_Dp, b_Dp = df.loc[Dp_systems, 'a'].values, df.loc[Dp_systems, 'b'].values
    a0_Dp = alpha0_of(a_Dp, b_Dp)
    natural_Dp = scorer_scores(base, m, root=ROOT, systems=Dp_systems)
    outcomes = pairwise_outcomes(natural_D, natural_Dp)
    labels.append(d)
    xs.append(abs(a0_D - a0_Dp))
    ys.append(tau_of(len(outcomes), sum(outcomes)))
  return studentized_outlier_test(labels, np.array(xs), np.array(ys), alpha=alpha)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende24')
  p.add_argument('--level', type=str, required=True,
                  help='"A-B" (total A removed, B removed together as the final step) or a bare '
                       'integer A, shorthand for "A-1"')
  p.add_argument('--metametric', type=str, default='spa')
  p.add_argument('--tag', type=str, default=None)
  p.add_argument('--drop-outliers', action=argparse.BooleanOptionalAction, default=True,
                  help='auto-detect and remove this dataset\'s MAD-detected outliers from the pool '
                       '(default True; pass --no-drop-outliers to keep them in)')
  p.add_argument('--drop-leverage-points', action=argparse.BooleanOptionalAction, default=False,
                  help='additionally drop systems that are statistically significant outliers in '
                       'robustness_before given their own delta_alpha_0 (default False; pass '
                       '--drop-leverage-points to enable)')
  p.add_argument('--leverage-alpha', type=float, default=0.05,
                  help='Bonferroni-corrected significance threshold for --drop-leverage-points')
  p.add_argument('--drop-systems', type=str, default='',
                  help='comma-separated systems to additionally remove from the pool')
  p.add_argument('--exclude-from-plot', type=str, default=None,
                  help='comma-separated systems (matched against d) to ADDITIONALLY leave out of the '
                       'scatter plot only (table still includes them)')
  p.add_argument('--direction', type=str, default='forward',
                  choices=['forward', 'reversed', 'mean'],
                  help='which side gets reweighted and to what target (default "forward"): "forward" '
                       'reweights D to alpha_0(D_prime) (the original design); "reversed" reweights '
                       'D_prime to alpha_0(D) instead -- often infeasible when D_prime is small, since '
                       'alpha_0(D) can fall outside D_prime\'s narrower reachable range (Corollary 2), '
                       'in which case that row is skipped and counted as infeasible; "mean" reweights '
                       'BOTH D and D_prime to the mean of their two natural alphas. Non-forward runs are '
                       'tagged with the direction so they never overwrite the forward-direction artifacts.')
  args = p.parse_args()
  base = args.dataset
  m = args.metametric
  if '-' in args.level:
    A_str, B_str = args.level.split('-')
    A, B = int(A_str), int(B_str)
  else:
    A, B = int(args.level), 1
  if not (1 <= B <= A):
    raise ValueError(f'--level {args.level!r} invalid: need 1 <= B <= A (got A={A}, B={B})')
  plot_excl = set(s.strip() for s in args.exclude_from_plot.split(',')) if args.exclude_from_plot else set()

  raw_systems = real_systems(base, root=ROOT)
  df = load_system_scores(base, root=ROOT)

  manual_drops = [s.strip() for s in args.drop_systems.split(',') if s.strip()]
  missing = [s for s in manual_drops if s not in raw_systems]
  if missing:
    raise ValueError(f'--drop-systems {missing} not in {base}\'s systems: {raw_systems}')

  outliers = []
  if args.drop_outliers:
    a_full = df.loc[raw_systems, 'a'].values
    outliers = [s for s, _, _ in iterative_mad_outliers(raw_systems, a_full)]

  pool_after_outliers = [s for s in raw_systems if s not in set(outliers) | set(manual_drops)]

  leverage_points = []
  if args.drop_leverage_points:
    flagged = detect_leverage_points(base, m, pool_after_outliers, df, alpha=args.leverage_alpha)
    leverage_points = [lab for lab, _, _, _ in flagged]
    if leverage_points:
      print('leverage-point systems flagged (robustness outlier given delta_alpha_0, '
            f'Bonferroni p<{args.leverage_alpha}): '
            + ', '.join(f'{lab} (t={t:+.2f}, p={bp:.4f})' for lab, _, t, bp in flagged), file=sys.stderr)

  dropped = sorted(set(outliers) | set(manual_drops) | set(leverage_points))
  full_systems = [s for s in raw_systems if s not in dropped]
  K = len(full_systems)

  if not (1 <= A <= K - 2):
    raise ValueError(f'--level {args.level!r} (A={A}) must have A in [1, {K - 2}] for this pool '
                      f'(K={K} after drops) -- D\' (size K-A) needs at least 2 systems for its '
                      f'metametric to be meaningful')

  level_tag = f'{A}-{B}'
  dir_suffix = '' if args.direction == 'forward' else f'_{args.direction}'
  tag = args.tag or f'{base}_level{level_tag}_{m}{dir_suffix}'
  context_size = A - B
  contexts = list(itertools.combinations(full_systems, context_size))
  n_drop_sets_per_context = comb(K - context_size, B)
  n_total = len(contexts) * n_drop_sets_per_context
  print(f'pool = {base}' + (f' minus {dropped}' if dropped else ' (full)')
        + f': K={K}, level={level_tag} -> {len(contexts)} contexts of size {context_size}, '
        f'{n_drop_sets_per_context} drop-sets of size {B} per context, '
        f'{n_total} rows total (each D\' used C({A},{B})={comb(A, B)} times)', file=sys.stderr)

  natural_Dp_cache = {}
  alpha0_Dp_cache = {}

  def get_Dp(triple_key, Dp_systems):
    if triple_key not in natural_Dp_cache:
      a_Dp, b_Dp = df.loc[Dp_systems, 'a'].values, df.loc[Dp_systems, 'b'].values
      alpha0_Dp_cache[triple_key] = alpha0_of(a_Dp, b_Dp)
      natural_Dp_cache[triple_key] = scorer_scores(base, m, root=ROOT, systems=Dp_systems)
    return natural_Dp_cache[triple_key], alpha0_Dp_cache[triple_key]

  rows = []  # (context_str, drop_str, n_before, sum_before, n_after, sum_after, ess_D, ess_Dp, a0_D, a0_Dp)
  t0 = time.time()
  n_done = 0
  n_infeasible = 0
  log_every = max(1, len(contexts) // 20)
  for ci, context in enumerate(contexts, 1):
    context_set = set(context)
    context_str = ','.join(sorted(context)) if context else '-'
    D_systems = [s for s in full_systems if s not in context_set]
    a_D, b_D = df.loc[D_systems, 'a'].values, df.loc[D_systems, 'b'].values
    a0_D = alpha0_of(a_D, b_D)
    natural_D = scorer_scores(base, m, root=ROOT, systems=D_systems)

    for drop_set in itertools.combinations(D_systems, B):
      drop_set_set = set(drop_set)
      drop_str = ','.join(sorted(drop_set))
      Dp_systems = [s for s in D_systems if s not in drop_set_set]
      triple_key = frozenset(context_set | drop_set_set)
      natural_Dp, a0_Dp = get_Dp(triple_key, Dp_systems)

      before_outcomes = pairwise_outcomes(natural_D, natural_Dp)
      n_before, sum_before = len(before_outcomes), sum(before_outcomes)

      try:
        if args.direction == 'forward':
          r_D = solve_w_exact(a_D, b_D, a0_Dp)
          reweighted_D = scorer_scores(base, m, root=ROOT, w=r_D.w, systems=D_systems)
          after_outcomes = pairwise_outcomes(reweighted_D, natural_Dp)
          ess_D, ess_Dp = r_D.ess, float('nan')
        else:
          a_Dp, b_Dp = df.loc[Dp_systems, 'a'].values, df.loc[Dp_systems, 'b'].values
          if args.direction == 'reversed':
            r_Dp = solve_w_exact(a_Dp, b_Dp, a0_D)
            reweighted_Dp = scorer_scores(base, m, root=ROOT, w=r_Dp.w, systems=Dp_systems)
            after_outcomes = pairwise_outcomes(natural_D, reweighted_Dp)
            ess_D, ess_Dp = float('nan'), r_Dp.ess
          else:  # mean
            alpha_mean = 0.5 * (a0_D + a0_Dp)
            r_D = solve_w_exact(a_D, b_D, alpha_mean)
            r_Dp = solve_w_exact(a_Dp, b_Dp, alpha_mean)
            reweighted_D = scorer_scores(base, m, root=ROOT, w=r_D.w, systems=D_systems)
            reweighted_Dp = scorer_scores(base, m, root=ROOT, w=r_Dp.w, systems=Dp_systems)
            after_outcomes = pairwise_outcomes(reweighted_D, reweighted_Dp)
            ess_D, ess_Dp = r_D.ess, r_Dp.ess
      except ValueError:
        n_infeasible += 1
        n_done += 1
        continue

      n_after, sum_after = len(after_outcomes), sum(after_outcomes)
      rows.append((context_str, drop_str, n_before, sum_before, n_after, sum_after, ess_D, ess_Dp, a0_D, a0_Dp))
      n_done += 1

    if ci % log_every == 0 or ci == len(contexts):
      elapsed = time.time() - t0
      rate = n_done / elapsed if elapsed > 0 else 0.0
      remaining_s = (n_total - n_done) / rate if rate > 0 else float('nan')
      infeasible_note = f', {n_infeasible} infeasible (skipped)' if n_infeasible else ''
      print(f'  [{ci}/{len(contexts)} contexts, {n_done}/{n_total} rows{infeasible_note}] '
            f'elapsed={elapsed / 60:.1f}min  est.remaining={remaining_s / 60:.1f}min',
            file=sys.stderr, flush=True)

  def combine_ess(ess_D, ess_Dp):
    vals = [v for v in (ess_D, ess_Dp) if not np.isnan(v)]
    return min(vals) if vals else float('nan')

  table_rows = []
  for context_str, drop_str, n_b, s_b, n_a, s_a, ess_D, ess_Dp, a0_D, a0_Dp in rows:
    tau_b = tau_of(n_b, s_b)
    tau_a = tau_of(n_a, s_a)
    delta_tau = tau_a - tau_b
    delta_a0 = abs(a0_D - a0_Dp)
    ess = combine_ess(ess_D, ess_Dp)
    table_rows.append((context_str, drop_str, tau_b, tau_a, delta_tau, a0_D, a0_Dp, delta_a0, ess))
  table_rows.sort(key=lambda r: r[2])  # ascending robustness_before -- least robust first

  pooled_tau_b = tau_of(sum(r[2] for r in rows), sum(r[3] for r in rows))
  pooled_tau_a = tau_of(sum(r[4] for r in rows), sum(r[5] for r in rows))
  pooled_delta = pooled_tau_a - pooled_tau_b

  d_col = 'd' if B == 1 else 'drop_set (d)'
  if args.direction == 'forward':
    calib_desc = f'D reweighted to alpha_0(D_-{d_col})'
    ess_note = 'ESS(D), the reweighted side'
  elif args.direction == 'reversed':
    calib_desc = f'D_-{d_col} reweighted to alpha_0(D)'
    ess_note = f'ESS(D_-{d_col}), the reweighted side'
  else:
    calib_desc = f'BOTH D and D_-{d_col} reweighted to mean(alpha_0(D), alpha_0(D_-{d_col}))'
    ess_note = f'min(ESS(D), ESS(D_-{d_col})), the worse of the two reweighted sides'
  lines = []
  lines.append(f'# Type-7 level-{level_tag} robustness, direction={args.direction} '
               f'(one row per (context, {d_col}))')
  lines.append('')
  lines.append(f'D = `{base}`' + (f' minus `{dropped}`' if dropped else '') + f' (K={K}), '
               f'level={level_tag} (A={A} total removed, B={B} removed together), '
               f'metametric = `{m}` -- {n_total} rows attempted ({len(contexts)} contexts of size '
               f'{context_size}, {n_drop_sets_per_context} drop-sets of size {B} per context, '
               f'{n_total // comb(A, B)} distinct D\' sets, each used C({A},{B})={comb(A, B)} times)'
               + (f'; {n_infeasible} infeasible and skipped (target outside the reweighted side\'s '
                  f'reachable range, Corollary 2)' if n_infeasible else ''))
  lines.append('')
  lines.append(f'robustness_before = tau(D, D_-{d_col}) (natural, unweighted on both sides); '
               f'robustness_after = tau after {calib_desc}; delta_robustness = robustness_after - '
               'robustness_before (>0 = reweighting improved consistency); ESS column = ' + ess_note)
  lines.append('')
  lines.append(f'| context (already removed) | {d_col} | robustness_before | robustness_after | '
               f'delta_robustness | alpha_0(D) | alpha_0(D_-{d_col}) | abs(delta_alpha_0) | ESS |')
  lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
  for context_str, drop_str, tau_b, tau_a, delta_tau, a0_D, a0_Dp, delta_a0, ess in table_rows:
    lines.append(f'| {context_str} | {drop_str} | {tau_b:.4f} | {tau_a:.4f} | {delta_tau:+.4f} | '
                 f'{a0_D:.4f} | {a0_Dp:.4f} | {delta_a0:.4f} | {ess:.2f} |')
  lines.append(f'| **pooled** | | **{pooled_tau_b:.4f}** | **{pooled_tau_a:.4f}** | '
               f'**{pooled_delta:+.4f}** | | | | |')
  table_md = '\n'.join(lines)

  print('\n' + table_md)

  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'type7_level{level_tag}_robustness_{tag}.md')
  with open(out_path, 'w') as f:
    f.write(table_md + '\n')
  print(f'\nWrote {out_path}', file=sys.stderr)

  # Scatter: robustness_before vs. |delta_alpha_0|, one point per (context, drop_set) row.
  # plot_excl matches against INDIVIDUAL system names, so split multi-system drop_str's.
  plot_table_rows = [r for r in table_rows if not (set(r[1].split(',')) & plot_excl)]
  xs = np.array([r[7] for r in plot_table_rows])  # delta_a0
  ys = np.array([r[2] for r in plot_table_rows])  # robustness_before
  names = [r[1] for r in plot_table_rows]
  r_val, p_val = stats.pearsonr(xs, ys)
  slope, intercept = np.polyfit(xs, ys, 1)

  dense = len(xs) > 30
  fig, ax = plt.subplots(figsize=(7, 6))
  ax.scatter(xs, ys, s=18 if dense else 40, color='#1f77b4', zorder=3, alpha=0.4 if dense else 1.0)
  if not dense:
    for name, xx, yy in zip(names, xs, ys):
      ax.annotate(name, xy=(xx, yy), fontsize=7, color='brown', xytext=(4, 4), textcoords='offset points')
  xline = np.linspace(xs.min(), xs.max(), 100)
  ax.plot(xline, slope * xline + intercept, color='red', linewidth=1.5,
          label=f'fit: y={slope:.3f}x+{intercept:.3f}')
  ax.set_xlabel(rf'$\Delta\alpha_0 = |\alpha_0(D) - \alpha_0(D_{{-{d_col}}})|$')
  ax.set_ylabel(f'Robustness (unweighted): tau(D, D_-{d_col})')
  excl_note = f' (excl. {", ".join(sorted(plot_excl))})' if plot_excl else ''
  ax.set_title(f'{base}: level-{level_tag} robustness vs. alpha_0 shift, '
               f'one point per (context, {d_col}){excl_note}\n'
               f'r={r_val:+.3f}, p={p_val:.2e}, n={len(xs)}')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(loc='best', frameon=False, fontsize=9)
  fig.tight_layout()

  plot_path = os.path.join(ARTIFACTS_DIR, f'type7_level{level_tag}_robustness_vs_delta_alpha0_{tag}.png')
  fig.savefig(plot_path, dpi=150, bbox_inches='tight')
  print(f'r={r_val:.4f}  p={p_val:.4e}', file=sys.stderr)
  print(f'Wrote {plot_path}', file=sys.stderr)
