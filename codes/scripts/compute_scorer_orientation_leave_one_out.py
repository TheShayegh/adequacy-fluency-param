"""Computes A/B/T/J orientation exactly like compute_scorer_orientation_vs_
alpha.py --green-family ABTJ, but at ONE alpha per leave-one-out subset D'
= D \\ {s}, for every system s in D -- a jackknife-style robustness overlay
for plot_scorer_orientation_vs_alpha.py --overlay-loo, sitting next to the
overlay-synth25 pool markers but built from the SAME plain weighted-SPA/PA
machinery as the main curves (lib.synthetic_scorer_alpha_grid.
donor_alpha_dial_grid), not the synth25 union metametric.

Each D' is scored UNWEIGHTED, i.e. with uniform system weight w'_i =
1/|D'| -- which by lib.alpha.alpha_0's own definition ("the natural
(uniform-weight) balance") realizes EXACTLY alpha_0(D'), so no
reweight_exact solve is needed: the uniform vector is passed straight into
donor_alpha_dial_grid's w_by_alpha as the single entry {alpha_0(D'): w'}.

Same two dial grids as --green-family ABTJ: lib.synthetic_scorers.
ABT_DIAL_GRID (synthesis='additive_mean') for A/B/T, ABTJ_J_DIAL_GRID
(synthesis='offset') for J -- so this overlay is directly comparable to
the ABTJ curves/synth25 markers already on the plot. Per-donor orientation
values are averaged over lib.consistency.real_scorers(dataset), matching
every other family-level number on these plots (synth25 pools, etc.).

Cost note: each leave-out subset requires its OWN full permutation-test
sweep across both 201-point dial grids for every donor (dropping even one
system changes every pairwise segment comparison), so this costs roughly
as much as ONE FULL --green-family ABTJ compute run PER leave-out system
-- i.e. about K times that cost for a K-system dataset, not K times
cheaper as the single-alpha framing might suggest.

Usage: python codes/scripts/compute_scorer_orientation_leave_one_out.py
           [--dataset zhen21] [--metametric spa|pa] [--tag TAG]
           [--only SYSTEM1,SYSTEM2,...] [--part-tag PART_TAG] [--merge PART1.npz PART2.npz ...]

--only: restrict this run to a comma-separated subset of `systems` (each
must be a real system name) instead of every leave-out subset -- lets
several invocations cover disjoint slices of the K leave-out subsets in
parallel (bounded concurrency instead of all K or a single sequential
run), each writing its own partial cache (--part-tag names it, default
derived from --only) instead of the final merged one.

--merge: instead of computing anything, combine a list of partial .npz
caches (written by separate --only runs, together covering every system)
into the final scorer_orientation_<tag>.npz plot_scorer_orientation_vs_
alpha.py --overlay-loo reads.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from lib.alpha import alpha_0
from lib.consistency import real_scorers, real_systems
from lib.synthetic_scorer_alpha_grid import donor_alpha_dial_grid
from lib.synthetic_scorer_orientation import donor_orientation_by_alpha
from lib.synthetic_scorers import ABT_DIAL_GRID, ABTJ_J_DIAL_GRID
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(ROOT, 'artifacts', 'data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='zhen21')
  parser.add_argument('--metametric', choices=['spa', 'pa'], default='spa',
                       help='weighted meta-metric to score each dial cell with')
  parser.add_argument('--tag', type=str, default=None, help='override the auto-derived cache filename tag')
  parser.add_argument('--only', type=str, default=None,
                       help='comma-separated subset of systems to process as leave-out subsets (default: '
                            'all of them) -- for splitting the K subsets across several bounded-concurrency '
                            'invocations, each writing its own partial cache; see --merge to combine them.')
  parser.add_argument('--part-tag', type=str, default=None,
                       help='filename tag for this --only run\'s partial cache (default: derived from --only)')
  parser.add_argument('--merge', type=str, nargs='+', default=None,
                       help='combine these partial .npz caches (written by separate --only runs) into the '
                            'final scorer_orientation_<tag>.npz -- no computation happens in this mode.')
  args = parser.parse_args()
  base = args.dataset

  if args.merge:
    parts = [np.load(p) for p in args.merge]
    dropped_systems = sorted({s for p in parts for s in [str(x) for x in p['dropped_systems']]})
    by_system = {}
    for p in parts:
      for i, s in enumerate([str(x) for x in p['dropped_systems']]):
        by_system[s] = {fam: p[fam][i] for fam in ('alpha0', 'A', 'B', 'T', 'J')}
    missing = [s for s in dropped_systems if s not in by_system]
    if missing:
      sys.exit(f'merge: {len(missing)} systems missing from the given parts: {missing}')
    os.makedirs(DATA_DIR, exist_ok=True)
    tag = args.tag or (f'{base}_loo' + ('' if args.metametric == 'spa' else f'_{args.metametric}'))
    out_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
    np.savez(
        out_path, dataset=base, dropped_systems=np.array(dropped_systems),
        alpha0=np.array([by_system[s]['alpha0'] for s in dropped_systems]),
        A=np.array([by_system[s]['A'] for s in dropped_systems]),
        B=np.array([by_system[s]['B'] for s in dropped_systems]),
        T=np.array([by_system[s]['T'] for s in dropped_systems]),
        J=np.array([by_system[s]['J'] for s in dropped_systems]),
    )
    print(f'Wrote {out_path} ({len(dropped_systems)} leave-out subsets merged from {len(args.merge)} parts)',
          file=sys.stderr)
    sys.exit(0)

  systems = real_systems(base, root=ROOT, excl_missing_seg_granular_mqm=True)
  if len(systems) < 3:
    sys.exit(f'{base}: only {len(systems)} real systems, leave-one-out needs at least 3')
  candidates = real_scorers(base, root=ROOT, systems=systems, require_seg_scores=True)
  df = load_system_scores(base, root=ROOT)

  if args.only:
    wanted = [s.strip() for s in args.only.split(',')]
    unknown = [s for s in wanted if s not in systems]
    if unknown:
      sys.exit(f'{base}: --only names not in real_systems: {unknown}')
    loo_systems = wanted
  else:
    loo_systems = systems

  n = len(loo_systems)
  print(f'{base}: K={len(systems)} real systems, {len(candidates)} candidate donors, leave-one-out over '
        f'{n} subsets (|D\'|={len(systems) - 1} each)', file=sys.stderr)

  results = {}
  t0 = time.time()
  for si, dropped in enumerate(loo_systems, 1):
    Dp = [s for s in systems if s != dropped]
    a_Dp, b_Dp = df.loc[Dp, 'a'].values, df.loc[Dp, 'b'].values
    alpha0_p = alpha_0(a_Dp, b_Dp)
    w_p = np.full(len(Dp), 1.0 / len(Dp))
    w_by_alpha = {alpha0_p: w_p}

    per_donor = {'A': [], 'B': [], 'T': [], 'J': []}
    for donor in candidates:
      grids_abt = donor_alpha_dial_grid(base, donor, Dp, w_by_alpha, root=ROOT, dial_grid=ABT_DIAL_GRID,
                                         metametric=args.metametric, synthesis='additive_mean')
      grids_j = donor_alpha_dial_grid(base, donor, Dp, w_by_alpha, root=ROOT, dial_grid=ABTJ_J_DIAL_GRID,
                                       metametric=args.metametric, synthesis='offset')
      if grids_abt is None or grids_j is None:
        continue
      orient_abt = donor_orientation_by_alpha(grids_abt)
      orient_j = donor_orientation_by_alpha(grids_j)
      per_donor['A'].append(orient_abt['A'][0])
      per_donor['B'].append(orient_abt['B'][0])
      per_donor['T'].append(orient_abt['T'][0])
      per_donor['J'].append(orient_j['J'][0])

    results[dropped] = {
        'alpha0': alpha0_p,
        **{fam: (float(np.mean(vals)) if vals else float('nan')) for fam, vals in per_donor.items()},
    }
    elapsed = time.time() - t0
    eta = elapsed / si * (n - si)
    r = results[dropped]
    print(f'[{si}/{n}] dropped={dropped} alpha0(D\')={alpha0_p:.4f} A={r["A"]:.4f} B={r["B"]:.4f} '
          f'T={r["T"]:.4f} J={r["J"]:.4f} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)', file=sys.stderr)

  os.makedirs(DATA_DIR, exist_ok=True)
  base_tag = args.tag or (f'{base}_loo' + ('' if args.metametric == 'spa' else f'_{args.metametric}'))
  if args.only:
    part_tag = args.part_tag or (base_tag + '_part_' + '_'.join(
        ''.join(c if c.isalnum() else '' for c in s) for s in loo_systems))
    tag = part_tag
  else:
    tag = base_tag
  out_path = os.path.join(DATA_DIR, f'scorer_orientation_{tag}.npz')
  dropped_systems = sorted(results)
  np.savez(
      out_path,
      dataset=base,
      dropped_systems=np.array(dropped_systems),
      alpha0=np.array([results[s]['alpha0'] for s in dropped_systems]),
      A=np.array([results[s]['A'] for s in dropped_systems]),
      B=np.array([results[s]['B'] for s in dropped_systems]),
      T=np.array([results[s]['T'] for s in dropped_systems]),
      J=np.array([results[s]['J'] for s in dropped_systems]),
  )
  print(f'Wrote {out_path}', file=sys.stderr)
