# How to cross-validate a (P) solver

Scope: practices for validating any solver of (P) (see `codes/lib/
reweight_exact.py`'s module docstring for what (P) is) against the trusted
ground truth, `codes/lib/reweight_exhaustive.py`. Written after cross-
validating `solve_w_exact` this way and finding it clean across K=4-17 and
the full alpha range, including boundary-adjacent targets.

## The tools

`codes/lib/solver_compare.py` has two entry points — pick based on what
you're comparing:

- **`compare_solvers(a, b, alpha_target, tol, result1, result2)`** — two
  solvers at the SAME nominal target alpha. Recomputes everything
  (alpha(w), budget, non-negativity, objective) directly from each
  result's raw `w`, in float64, never trusting either solver's self-
  reported fields. Correctly classifies a **tied optimum** (different `w`,
  same valid objective — common on a discretized grid, where many sparse
  supports reach the identical ESS) as a match, not a bug.

- **`cross_validate_via_exhaustive(a, b, exhaustive_result, solver=solve_w_exact)`**
  — a continuous solver vs. a trusted exhaustive-grid result. Use this one
  whenever the other side is `reweight_exhaustive`, not `compare_solvers`.
  It re-anchors: recomputes exhaustive's ACTUAL achieved alpha from its raw
  `w` (never the nominal target — a grid point is only ever within `tol`
  of the target, essentially never exactly on it), then re-solves the
  candidate solver AT that actual alpha before comparing ESS. See "The
  gotcha" below for why this matters — skipping this step produced 84
  false positives out of 100 test points in practice.

## The gotcha: same-target comparison against a grid solver is a trap

`reweight_exhaustive`'s accepted point is only ever within `tol` of the
target alpha — check its `.gap` field, never trust `.ess` without it. Near
alpha_min/alpha_max, `alpha(w)` gets steep (small weight changes swing
alpha a lot — see `reweight_exhaustive.py`'s own module docstring), so even
a small residual gap can correspond to a much easier point with inflated
ESS. Comparing a continuous solver against exhaustive at the literal
nominal target conflates two different things: "the continuous solver
missed the true optimum" (a real bug) vs. "exhaustive's accepted point
actually sits at a nearby, easier alpha" (not a bug). Concretely: comparing
`solve_w_exact` vs. `solve_w_exhaustive` at nominal target alphas across a
K=4/6/7/8 real-data sweep showed 84/100 "exhaustive wins" cases; every one
had nonzero `.gap`, and re-solving `solve_w_exact` at exhaustive's actual
achieved alpha instead matched or exceeded it in every case.
`cross_validate_via_exhaustive` automates this re-anchoring so this
confound can't recur.

## Which direction is the actual red flag

A continuous solver's feasible region strictly *contains* any exhaustive
grid at the same target alpha (the grid is a discretized subset of it). So:

- `solver_ess >= exhaustive_ess` at the same alpha → expected, healthy.
- `solver_ess < exhaustive_ess` at the same alpha → **the violation** — a
  coarser subset of the solver's own search space beat it, which is only
  possible if the solver missed the true optimum. This is the direction
  `cross_validate_via_exhaustive`'s `match_type == 'mismatch'` flags.

It's easy to get this backwards when skimming results — "exact does worse
than exhaustive" sounds alarming regardless of which one is actually
supposed to dominate; double check which solver's search space contains
the other's before calling something a violation.

## What a thorough sweep looks like

- **Emphasize the alpha boundaries.** Near alpha_min/alpha_max is where
  things get numerically hardest (steep `alpha(w)`, near-degenerate
  curvature, secular-equation pole handling in `reweight_exact.py`) — a
  sweep should be denser there, not uniform. (Also exactly where
  `reweight_exhaustive` is most likely to return `success=False` — no grid
  point within `tol` — at a given resolution; that's an honest
  "inconclusive," not a failure of either solver, and not a violation.)
- **Use real datasets across a range of K**, not only synthetic instances
  — `codes/lib/dataset_dirs.py`'s `DATASET_DIRS` spans K=7 to K=19; a
  smaller-K dataset can be truncated to an even smaller K by slicing
  `real_systems(dataset)[:K]` for finer exhaustive-grid resolution.
- **Size `grid_n` to what's tractable**: cost is
  `math.comb(grid_n + K - 1, K - 1)` — compute this before picking `grid_n`
  for a given K, not after. Small K (4-8) tolerates `grid_n` in the
  hundreds; K=13-17 is generally stuck around `grid_n` ~9-11 to stay under
  a few million candidates.
- **Also run the known adversarial instance**: action_plan.md §3.7 gives an
  explicit K=4 counterexample (`a=(0.5574,0.9883,1.4687,3.0894)`,
  `b=(0.4012,0.5353,1.4986,1.2467)`, target alpha=0.9095, global optimum
  `w*=(0.0088,0.0861,0.4245,0.4806)`) where a naive full-support stationary
  point is infeasible (negative weight) and no Lagrangian certificate
  exists — a solver should recover `w*` on this and report
  `certified=False, phase='fallback'`. Cheap, fast, and directly exercises
  the completeness machinery rather than the common case.
- **Always record execution time per call, not just aggregate.**
  `solve_w_exact`'s cost is highly non-uniform: sub-millisecond for K<=8,
  but observed spiking to 15-27 seconds per call at K=17 near the lower
  alpha boundary specifically — cases where no Theorem-2 certificate fires
  early and the solver falls back to enumerating supports of decreasing
  size (it does not implement the paper's speed-oriented visiting-order
  heuristic; see that module's docstring). A single aggregate time can
  hide this entirely — report per-(K, alpha-region) breakdowns.
