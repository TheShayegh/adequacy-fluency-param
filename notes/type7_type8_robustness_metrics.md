# Type-7 and type-8: the two subset-robustness measures

Scope: what "robustness" means in the type-7 and type-8 analysis families
and exactly how each is computed. Both measure the same underlying
question -- *how much does a scorer ranking change when the system pool
D changes?* -- but they use different statistics and opposite resampling
directions. `codes/scripts/compute_type7_sensitivity.py` (tau) and
`codes/scripts/compute_type8_icc_upsample.py` (ICC) are the canonical
entry points; `codes/lib/subsampling_stability.py` is type-8's
non-"type7-leveled" sibling (same ICC statistic, plain subsampling
instead of the level-A-B upsampling design).

## Shared setup

- **Pool**: both start from `lib.consistency.real_systems(dataset, root,
  exclude_outliers=...)`, this project's canonical outlier-screened system
  roster (see project memory on `wmt_official`/`ende23` exclusions).
- **Scorer ranking**: `lib.consistency.scorer_scores(dataset, metametric,
  root, systems=S, w=...)` returns one score per scorer (the automatic
  metric under evaluation, e.g. a COMET variant), computed over system
  pool `S` under the given metametric (default `spa`) and weight vector
  `w` (uniform/"natural" if omitted). Robustness is always about how much
  *this scorer ranking* is preserved when `S` changes -- not about system
  rankings.
- **alpha_0 and reweighting**: `alpha_0(a, b) = alpha(a, b, uniform w)` is
  a pool's natural adequacy/fluency balance (`lib.alpha.alpha_0`).
  `lib.reweight_exact.solve_w_exact(a, b, alpha_target)` finds the
  maximum-ESS weight vector `w` achieving a *given* target alpha on a
  pool (a certified global optimum, see `notes/solver_cross_validation.md`).
  Both type-7 and type-8 use this to ask: "if we recalibrate pool X's
  balance to match pool Y's natural balance, does X's scorer ranking move
  closer to Y's?"
- **Level A-B**: both scripts share this pool-perturbation vocabulary
  (`--level`). A systems are removed from the full pool to define the
  smaller pool; B (<= A) of those A are the ones varied as the "final
  step" (type-7: dropped together; type-8: added back together). A bare
  integer A is shorthand for A-1.

## Type-7: tau-based robustness (downsampling)

**Statistic**: Kendall's tau-a on scorer rankings, this project's usual
convention (`lib.reweighted_consistency.pairwise_outcomes`).

**Design**: for every context (a set of `A-B` systems already removed)
and every drop-set (`B` more systems removed from what's left), let
`D` = pool minus context, `D'` = D minus drop-set (`|D| = K-A+B`,
`|D'| = K-A`). Every distinct `D'` is visited `C(A,B)` times, once per way
of splitting its `A` missing systems into a size-`(A-B)` context and a
size-`B` drop-set (B=1 recovers the old "each D' used A times" design;
B=A means each D' is used exactly once, no sequential structure).

**Computation per (context, drop-set) row**:

1. `natural_D = scorer_scores(D, uniform w)`, `natural_Dp = scorer_scores(D', uniform w)`.
2. `outcomes_before = pairwise_outcomes(natural_D, natural_Dp)`: for every
   pair of scorers common to both rankings, +1 if the two rankings agree
   on which scorer is higher, -1 if they disagree, 0 on a tie in either
   ranking. `robustness_before = tau(D, D') = sum(outcomes) / n_pairs`.
3. Reweight (direction controlled by `--direction`, default `forward`):
   - `forward` (the primary design): solve `w_D = solve_w_exact(a_D, b_D,
     alpha_0(D'))` -- pull D's balance to match D''s own natural alpha --
     then `reweighted_D = scorer_scores(D, w=w_D)`, and
     `robustness_after = tau(reweighted_D, natural_Dp)`.
   - `reversed`: reweight D' to `alpha_0(D)` instead (often infeasible,
     since D' is smaller and has a narrower reachable alpha range,
     Corollary 2; skipped rows are counted as infeasible).
   - `mean`: reweight *both* sides to the mean of their two natural
     alphas.
4. `delta_robustness = robustness_after - robustness_before`. Positive
   means reweighting recovered ranking consistency that was lost purely
   from dropping the drop-set; near zero means no effect; negative means
   reweighting actively hurt (a real observed outcome -- hitting an exact
   alpha_0 target doesn't uniquely determine *how* weight redistributes,
   and the solver's chosen admissible solution needn't resemble "just
   remove the drop-set").
5. ESS of the reweighted side (`solve_w_exact`'s `.ess`) is reported
   alongside, as a diagnostic for how much weight concentration the
   reweighting required.

**Pooling**: the table's final row is not a column-wise average of
per-row taus -- it's computed by summing every row's raw
concordant/discordant/tie counts across the whole table first, then
recomputing tau from those totals (the same pooling convention as
`pool_weighted_tau`, used by types 1/5/6).

**Optional leverage-point screen**: `--drop-leverage-points` runs a cheap
level-1 diagnostic pass (natural tau only) and flags systems that are
statistically significant outliers in `robustness_before` given their own
`delta_alpha_0`, via `lib.outlier_detection.studentized_outlier_test`
(externally studentized residuals, Bonferroni-corrected, same method as
R's `car::outlierTest`). This catches disruptive systems that the
alpha-based MAD outlier test (`--drop-outliers`, on by default) can't see,
since that test never looks at robustness/tau at all.

**Output**: `artifacts/type7_level<A>-<B>_robustness_<tag>.md` (full
table) plus a scatter of `robustness_before` vs. `|delta_alpha_0|` with a
linear fit (Pearson r, p-value).

## Type-8: ICC(C,1)-based robustness (upsampling)

**Statistic**: ICC(C,1) / ICC(3,1) (Shrout & Fleiss 1979; "consistency"
form, McGraw & Wong 1996), from `lib.icc.icc_c1` -- explicitly the
statistic used in place of pairwise flip-rate in
`lib.subsampling_stability.generalizability_stability`, here restructured
into type-7's level-A-B vocabulary. Scorers are the *objects* being
rated; different resampled pools are the *raters*.

**Design (the mirror image of type-7)**: for every way of removing `A`
systems from the full pool to get `D'` (`|D'| = K-A`, `C(K,A)` of them),
build `C(A,B)` "raters" by adding back every possible size-`B` subset `X`
of the `A` removed systems: `D'' = D' ∪ X` (`|D''| = K-A+B`). So instead
of shrinking D, type-8 starts small and grows back up -- upsampling, not
subsampling.

**Computation per D'**:

1. For each of the `C(A,B)` raters `D''`, compute two conditions:
   - `natural`: `scorer_scores(D'', uniform w)`.
   - `reweighted`: `scorer_scores(D'', w=solve_w_exact(a_{D''}, b_{D''},
     target_alpha))`, with `target_alpha = alpha_0(D')` by default --
     every upsampled D'' is pulled back toward matching what the smaller
     D' looked like naturally.
2. Stack each condition's `C(A,B)` rater score-vectors into an
   `n_scorers x n_raters` matrix (only scorers present in every rater are
   kept), and compute `icc_c1` on each: `ICC_natural`, `ICC_reweighted`.
3. `Delta = ICC_reweighted - ICC_natural`. Positive means reweighting each
   rater toward D''s balance made the scorer ranking *more* consistent
   across the C(A,B) raters than leaving them at natural weight.
4. Mean ESS across the `C(A,B)` reweighted raters is reported alongside
   each D' row.

**icc_c1 itself** (`lib.icc.icc_c1`, `X[s, b]` = scorer s's score under
rater b, no missing cells): standard two-way ANOVA without replication,
explicit grand-mean centering for numerical stability --

```
SS_total = sum((X - grand_mean)^2)
SS_rows  = k * sum((row_mean_s - grand_mean)^2)      # between-scorers
SS_cols  = n * sum((col_mean_b - grand_mean)^2)      # between-raters
SS_error = SS_total - SS_rows - SS_cols              # scorer x rater interaction
BMS = SS_rows / (n - 1)
EMS = SS_error / ((n - 1) * (k - 1))
ICC(C,1) = (BMS - EMS) / (BMS + (k - 1) * EMS)
```

`ICC(C,1)` deliberately excludes `JMS` (the rater/column mean square): a
systematic level shift across raters (e.g. reweighting changing every
rater's overall score scale) doesn't count against "consistency"
reliability -- that's what the "C" (vs. "A", agreement) stands for.
Needs >= 2 objects (scorers) and >= 2 raters; `B=A` gives `C(A,B)=1`
rater, so ICC is undefined and reported as `N/A` rather than silently
skipped. The module ships a self-test asserting this mean-square formula
agrees with an independently-derived variance-component form
(`S/(S+E)`, `S=(BMS-EMS)/k`, `E=EMS`) to floating tolerance.

**Aggregation across D'**: one row per D' (`ICC_natural`, `ICC_reweighted`,
`Delta`, mean ESS), plus a final average row over every D' where both ICC
values were defined (a plain mean, not pooled the way type-7's tau is --
ICC isn't a ratio of summable counts the way tau-a is).

**Output**: `artifacts/type8_icc_upsample_level<A>-<B>_<tag>.md`, written
*streamed* -- header first, each D' row appended (and the summary line
rewritten) as soon as that D' finishes, so the file is a valid, readable
table at any point during a long run rather than only at completion (see
project convention: [[feedback_progress_logging]]).

## Type-7 vs. type-8 at a glance

| | Type-7 | Type-8 |
|---|---|---|
| Statistic | Kendall's tau-a (pairwise concordance) | ICC(C,1) (two-way ANOVA reliability) |
| Resampling direction | Downsampling: D (larger) -> D' (smaller, minus drop-set) | Upsampling: D' (smaller, fixed) -> D'' (larger, plus added-back systems) |
| What's compared | Exactly 2 rankings per row: D vs. D' | C(A,B) rankings at once per D': all D'' raters together |
| Reweighting target | `alpha_0(D')` (or reversed/mean) applied to D | `alpha_0(D')` applied to every D'' rater |
| Robustness = | tau(natural D, natural D') vs. tau(reweighted D, natural D') | ICC across raters, natural vs. reweighted |
| Needs | >= 2 systems in D' | >= 2 raters (C(A,B) >= 2, i.e. B < A) |
| Pooling across rows | Sum raw concordant/discordant/tie counts, recompute tau | Plain mean of per-D' ICC values |
