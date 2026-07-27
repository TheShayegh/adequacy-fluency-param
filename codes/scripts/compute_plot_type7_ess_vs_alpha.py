"""Type-7 ESS(alpha) diagnostic: for level "L-L" (D = full screened pool,
D' = D minus an L-subset dropped together), plots normalized ESS
(ESS(alpha)/K) against target alpha for BOTH D and D', independently
solved at each alpha (lib.reweight_exact.solve_w_exact) -- one figure per
drop-set, two curves each (ESS(D)/K, ESS(D')/K). Purely a solver/
feasibility diagnostic: no scorer-ranking/metametric involved at all.

Alpha grid: reuses lib.alpha.compute_alpha_ij_grid exactly -- every
pairwise alpha_ij breakpoint of D's own full pool, restricted to the range
D and D' can JOINTLY reach (which is just D''s own range, since D' subset D
means D's own range always contains it -- action_plan.md 3.8, Corollary 2),
optionally trimmed by --alpha-trim on EACH side (default 0.0: the full
joint range, every alpha_ij).

D is the SAME pool for every drop-set in the L-L design (only D' changes),
so r_D(alpha) is solved once per alpha and CACHED, then reused across every
drop-set -- solving it fresh per (drop-set, alpha) pair, as an earlier
version of this script did, wastefully repeats identical solves once per
drop-set and was the main reason a K=15 run took 8+ hours: near-full-pool
solves can fall into solve_w_exact's expensive exhaustive-enumeration
fallback, and every one of those got redone up to K times for nothing.

Usage: python codes/scripts/compute_plot_type7_ess_vs_alpha.py
           --dataset ende23 [--level 1] [--alpha-trim 0.0]
           [--drop-outliers | --no-drop-outliers] [--tag TAG]
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.alpha import alpha_0 as alpha0_of, compute_alpha_ij_grid
from lib.consistency import real_systems
from lib.outlier_detection import iterative_single_aspect_outliers
from lib.reweight_exact import solve_w_exact
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')
STANDARD_BLUE = (0.121, 0.466, 0.705)
ORANGE = (0.90, 0.45, 0.10)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende23')
  p.add_argument('--level', type=int, default=1, help='L: systems dropped together (the "L-L" design)')
  p.add_argument('--alpha-trim', type=float, default=0.0,
                  help='trims this fraction off EACH side of the joint reachable range (default 0.0: '
                       'the full range, every alpha_ij)')
  p.add_argument('--drop-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  L = args.level
  tag = args.tag or f'{base}_level{L}-{L}'

  raw_systems = real_systems(base, root=ROOT)
  df = load_system_scores(base, root=ROOT)

  outliers = []
  if args.drop_outliers:
    a_full = df.loc[raw_systems, 'a'].values
    outliers = [s for s, _, _ in iterative_single_aspect_outliers(raw_systems, a_full)]
  D_systems = [s for s in raw_systems if s not in set(outliers)]
  K_D = len(D_systems)
  if not (1 <= L <= K_D - 2):
    raise ValueError(f'--level {L} must be in [1, {K_D - 2}] for this pool (K={K_D})')

  a_D, b_D = df.loc[D_systems, 'a'].values, df.loc[D_systems, 'b'].values
  a0_D = alpha0_of(a_D, b_D)
  print(f'D = {base}' + (f' minus {outliers}' if outliers else ' (full)')
        + f': K={K_D}, alpha_0(D)={a0_D:.4f}', file=sys.stderr)

  drop_sets = list(itertools.combinations(range(K_D), L))
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  # Each drop-set's own alpha grid first (cheap: no solving) -- gives an
  # accurate total solve count up front for the ETA below, and lets the
  # D-side cache be built once and shared correctly across every drop-set.
  per_drop = []
  for drop_idx in drop_sets:
    Dp_idx = [i for i in range(K_D) if i not in set(drop_idx)]
    grid, lo_trim, hi_trim = compute_alpha_ij_grid(
        a_D, b_D, [tuple(range(K_D)), tuple(Dp_idx)], trim_frac=args.alpha_trim)
    per_drop.append(dict(drop_idx=drop_idx, Dp_idx=Dp_idx, grid=grid, lo_trim=lo_trim, hi_trim=hi_trim))

  n_Dp_solves = sum(len(pd['grid']) for pd in per_drop)
  n_unique_alphas = len(set(v for pd in per_drop for v in pd['grid'])) + 1  # +1 for a0_D itself
  print(f'{len(drop_sets)} drop-sets, {n_Dp_solves} total D\' solves (D-side: only '
        f'{n_unique_alphas} unique alphas need solving, cached and reused across every drop-set)',
        file=sys.stderr)

  r_D_ess_cache = {}

  def get_D_ess(alpha):
    if alpha not in r_D_ess_cache:
      r_D_ess_cache[alpha] = solve_w_exact(a_D, b_D, alpha).ess / K_D
    return r_D_ess_cache[alpha]

  ess_D_anchor = get_D_ess(a0_D)  # D's own natural balance -- same for every drop-set, solved once

  t0 = time.time()
  n_done = 0
  log_every = max(1, n_Dp_solves // 50)

  for di, pd in enumerate(per_drop, 1):
    drop_idx, Dp_idx, grid, lo_trim, hi_trim = pd['drop_idx'], pd['Dp_idx'], pd['grid'], pd['lo_trim'], pd['hi_trim']
    drop_names = ','.join(sorted(D_systems[i] for i in drop_idx))
    a_Dp, b_Dp = a_D[Dp_idx], b_D[Dp_idx]
    K_Dp = len(Dp_idx)
    a0_Dp = alpha0_of(a_Dp, b_Dp)

    ess_D_over_k, ess_Dp_over_k = [], []
    for alpha in grid:
      ess_D_over_k.append(get_D_ess(alpha))
      r_Dp = solve_w_exact(a_Dp, b_Dp, alpha)
      ess_Dp_over_k.append(r_Dp.ess / K_Dp)
      n_done += 1
      if n_done % log_every == 0:
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        remaining_s = (n_Dp_solves - n_done) / rate if rate > 0 else float('nan')
        print(f'  [{n_done}/{n_Dp_solves} solves, drop-set {di}/{len(drop_sets)} ({drop_names})] '
              f'elapsed={elapsed / 60:.1f}min est.remaining={remaining_s / 60:.1f}min',
              file=sys.stderr, flush=True)

    # alpha_0(D)/alpha_0(D') are each curve's own natural (unweighted)
    # balance -- the anchor case of Lemma 3, where the solver returns w
    # exactly uniform, so ESS/K is exactly 1 there. Solved explicitly
    # (rather than assumed) so the marker is exact even if it falls outside
    # the grid above -- a real possibility for alpha_0(D), since it need
    # not lie inside D''s narrower reachable range at all.
    ess_Dp_anchor = solve_w_exact(a_Dp, b_Dp, a0_Dp).ess / K_Dp

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(grid, ess_D_over_k, color=STANDARD_BLUE, linewidth=2.0, marker='o', markersize=3,
            label=f'ESS(D)/K  (K={K_D})')
    ax.plot(grid, ess_Dp_over_k, color=ORANGE, linewidth=2.0, marker='o', markersize=3,
            label=f"ESS(D')/K  (K={K_Dp})")

    # Stagger the two labels' heights (rather than both at the same offset)
    # so they stay legible when alpha_0(D) and alpha_0(D') land close
    # together in x -- common when dropping just one system out of K.
    for a0, ess_anchor, color, tex_label, dy in [
        (a0_D, ess_D_anchor, STANDARD_BLUE, r'$\alpha_0(D)$', 8),
        (a0_Dp, ess_Dp_anchor, ORANGE, r"$\alpha_0(D')$", 22),
    ]:
      ax.axvline(a0, color=color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
      ax.scatter([a0], [ess_anchor], marker='*', s=150, color=color, edgecolor='black',
                 linewidths=0.6, zorder=6)
      ax.annotate(tex_label, xy=(a0, ess_anchor), xytext=(0, dy), textcoords='offset points',
                  ha='center', va='bottom', fontsize=8, color=color, zorder=7)

    ax.set_xlabel(r'Target Balance $\alpha$')
    ax.set_ylabel(r'Normalized ESS ($\mathrm{ESS}/K$)')
    ax.set_ylim(0, 1.1)
    x_all = grid + [a0_D, a0_Dp]
    x_lo, x_hi = min(x_all), max(x_all)
    x_pad = 0.04 * (x_hi - x_lo + 1e-9)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_title(f"{base}: ESS vs. $\\alpha$, D vs. D' = D-{{{drop_names}}}\n"
                 f'middle {100 * (1 - 2 * args.alpha_trim):.0f}% of joint reachable range '
                 f'[{lo_trim:.4f}, {hi_trim:.4f}]', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=False, fontsize=9)

    out_path = os.path.join(ARTIFACTS_DIR, f'type7_ess_vs_alpha_level{L}-{L}_{tag}_drop_{drop_names}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[{di}/{len(drop_sets)}] wrote {out_path}', file=sys.stderr, flush=True)
