# Operation-count analysis of solve_w_exact's three pruning theorems

Scope: how often each of `solve_w_exact`'s three search-space-reduction
theorems (admissible-support skip, support-size pruning, exit certificate —
see `codes/lib/reweight_exact.py`'s module docstring for the proofs) actually
fires on real data, and the operation-count cost of running with each one
switched off (individually and in combination), compared against the
brute-force trust anchor `codes/lib/reweight_exhaustive.py`.

## Methodology

**Instrumentation, not modification.** `solve_w_exact`'s own `ExactWResult`
doesn't break out per-theorem hit counts, and there's no way to toggle a
single theorem off in production code without editing it. Rather than touch
`lib/reweight_exact.py`, all measurements below come from a separate
instrumented copy of just its outer support-enumeration loop, which imports
and calls the *same* private closed-form helpers (`_support_candidates`,
`_admissible`, `_curvature_ok`, `_d_all`, `_prop1_2x2`) directly — so the
actual math is byte-identical to production; only the control flow (which
supports get skipped/visited, when to break, when to return early) is
duplicated, gated behind three independent booleans. Validated against the
real `solve_w_exact` (all three booleans on) across a 50-point sweep: 49/50
points matched exactly on ESS, success, supports-visited, and
candidates-evaluated; the one "mismatch" was the alpha_min boundary point,
where production's separate Proposition-2 shortcut reports `(supports=1,
candidates=1)` as bookkeeping for the same answer this instrumented version
reports as `(0, 0)` — same result (ESS=2.0), different counting convention
for a path outside the three theorems anyway.

**Two genuine edge cases only reachable by disabling the admissible-support
theorem.** Production's admissibility pre-filter guarantees every support
`_support_candidates` actually processes is non-degenerate. Turning that
filter off (to measure its cost) exposes supports it was silently protecting
the rest of the algorithm from: (1) a degenerate support with zero variance
can make `lib.alpha.alpha`'s `va / (va + vb)` divide by zero; (2) a
degenerate 2x2 curvature block can make `theta1 == 0`, and
`_support_candidates` unconditionally computes `-1.0 / (2.0 * theta1)` for
its first pole location. Both are caught and treated as "no candidates on
this support" (mirroring `_support_candidates`'s own existing "shouldn't
occur for an admissible S" comment) — not correctness bugs in production,
just two divide-by-zero paths that are unreachable there and only surfaced
because this analysis deliberately removes the guard that prevents them.

**Operation-count metric: scan-points, not wall time.** Wall-clock time
turned out to be a bad metric mid-analysis — identical configs measured 2-10x
apart run to run purely from contention with other processes/GPU load on the
machine running this, including a spell where a coarse-grid
`solve_w_exhaustive` sweep degraded ~5x over the course of ~50 sequential MPS
calls despite each being fast in isolation (root cause not fully pinned down;
switching to CPU regardless made timings match single-call benchmarks
again). So the tables below report **scan-points**: the exact count of
secular-equation grid evaluations `_support_candidates` performs per support
— its own module docstring identifies this scan as the dominant cost
("profiling showed THAT loop... dominated its wall-clock cost"). It's
computed deterministically per support from the same closed-form
`theta1`/`theta2`/`rho` (via `_prop1_2x2`) and the same
`degenerate_rho`/`abs(theta2) > tol` branch conditions production uses to
decide the scan's grid: 1000 points if the support has two live poles, 600 if
one, 0 if the alpha0(S) anchor case short-circuits before any scan. This is
an exact count, not an estimate, and immune to the wall-clock noise above.

**Dataset choice.** The request was to sweep 50 alphas densely around
alpha_0 (`[alpha_0 - 0.05, alpha_0 + 0.05]`). The obvious small-K WMT 2023/24
candidates don't have room for that window: `jazh24`'s alpha_0 = 0.9954 sits
only 0.0046 below alpha_max, `enes24`/`heen23`/`zhen23` are similarly pinned
near their upper boundary. `ende24` (K=16) is the one with real margin:
alpha_0 = 0.782073, distance to either boundary = 0.22. All results below
use `ende24`.

## Part 1 — full range, alpha in [alpha_min, alpha_max]

`ende24`, K=16, alpha swept as 50 points over **[0.007356, 0.999999]**.

### Theorem-hit counts (production config — all three theorems on)

| Theorem | Hits over the 50-alpha sweep |
|---|---|
| Admissible support (skip inadmissible S) | **65,522** |
| Support-size pruning (early break) | **29** |
| Exit certificate (early return) | **19 / 50** |

2 of the 50 alphas (the alpha_min/alpha_max endpoints) resolve via the
separate Proposition-2 boundary shortcut, outside all three theorems. Of the
remaining 48, only 19 certify directly — the other 29 fall through to the
fallback (exhaustive-enumeration-over-admissible-supports) path, each of
which is exactly where the size-pruning break fires (`pruning_hits=29`
matches `48 - 19` exactly). This is the opposite picture from Part 2: away
from alpha_0, the natural full-support weighting usually *isn't* already
near-optimal, so the certificate can't just fire immediately on the first
try the way it did for every alpha near alpha_0 — admissibility and pruning
are doing real, substantial work across this range (65,522 supports
rejected as inadmissible, vs. 0 in the alpha_0 window).

### 8-way operation count

| admissible | pruning | certificate | supports visited | candidates evaluated | scan points | total flops |
|---|---|---|---|---|---|---|
| ✔ | ✔ | ✔ *(production default)* | 951,432 | 2,116,144 | 951,432,000 | 26,304,769,832 |
| ✔ | ✔ | ✘ | 951,748 | 2,116,460 | 951,748,000 | 26,313,219,040 |
| ✔ | ✘ | ✔ | 1,771,170 | 3,862,176 | 1,771,169,000 | 48,911,136,918 |
| ✔ | ✘ | ✘ | 2,999,023 | 5,759,754 | 2,999,022,000 | 82,239,076,107 |
| ✘ | ✔ | ✔ | 1,016,954 | 2,303,079 | 1,016,954,000 | 28,146,906,927 |
| ✘ | ✔ | ✘ | 1,017,270 | 2,303,395 | 1,017,270,000 | 28,155,356,135 |
| ✘ | ✘ | ✔ | 1,900,122 | 4,253,661 | 1,898,723,800 | 52,520,584,713 |
| ✘ | ✘ | ✘ *(brute enumeration)* | 3,144,912 | 6,206,212 | 3,142,588,200 | 86,307,431,756 |

(flops computed with the same formula as Part 4 below: `supports×1995 +
scan_points×24 + candidates×743`, n=K=16 upper bound on the setup term.)

Unlike Part 2 (where only the certificate theorem mattered because it always
fired instantly), all three theorems visibly earn their keep here:
- Certificate on→off (row 1→2, row 5→6): tiny cost (~0.03% more flops) —
  because most of the certificate's value already happened, it just means
  the 19 alphas that *did* certify now also finish the size-pruning walk
  down to their incumbent's ESS before returning, which is cheap.
- Pruning on→off (row 1→3, row 5→7): ~1.9x more flops — meaningful but not
  dominant, since pruning only shortens the *tail* of the fallback search.
- Admissibility on→off (row 1→5, row 3→7): ~1.07-1.08x more flops here (a
  much smaller effect than in the jazh24 full-range run, where disabling it
  roughly doubled the candidate count) — `ende24`'s larger K=16 means a
  higher fraction of arbitrary subsets already happen to be inadmissible for
  most targets, so admissibility is doing comparatively less *additional*
  filtering beyond what pruning+certificate already avoid.
- All three off (row 8, brute enumeration): **86.3 billion flops** vs. the
  default's 26.3 billion — a **~3.3x** reduction from having all three
  theorems on, smaller than Part 2's ~66,000x reduction, because full-range
  alphas frequently need the genuinely-expensive fallback search regardless
  of which optimizations are active; the theorems here are trimming a large,
  necessary search rather than replacing it with an O(1) certified shortcut.

## Part 2 — narrow window around alpha_0

`ende24`, K=16, alpha_0 = **0.782073**, alpha swept as 50 points over
**[0.732073, 0.832073]**.

### Theorem-hit counts (production config — all three theorems on)

| Theorem | Hits over the 50-alpha sweep |
|---|---|
| Admissible support (skip inadmissible S) | **0** |
| Support-size pruning (early break) | **0** |
| Exit certificate (early return) | **50 / 50** |

Every single alpha in this window resolves via the exit certificate on the
very first support tried (the full K=16 support) — admissibility and pruning
never get a chance to do anything, because the algorithm is already done
before either would matter. Sharply different from the full-range picture in
Part 1 below: near alpha_0 the natural (near-uniform) weighting is already
at or very near the optimum, so the closed-form candidate on the full
support satisfies the KKT+curvature certificate immediately.

### 8-way operation count

| admissible | pruning | certificate | supports visited | candidates evaluated | scan points |
|---|---|---|---|---|---|
| ✔ | ✔ | ✔ *(production default)* | 50 | 50 | 50,000 |
| ✔ | ✔ | ✘ | 50 | 50 | 50,000 |
| ✔ | ✘ | ✔ | 50 | 50 | 50,000 |
| ✔ | ✘ | ✘ | 3,225,830 | 4,752,434 | 3,225,828,000 |
| ✘ | ✔ | ✔ | 50 | 50 | 50,000 |
| ✘ | ✔ | ✘ | 50 | 50 | 50,000 |
| ✘ | ✘ | ✔ | 50 | 50 | 50,000 |
| ✘ | ✘ | ✘ | 3,275,950 | 4,919,336 | 3,273,508,000 |

Only the certificate theorem does any work in this regime: every row with
`certificate=True` costs exactly 1,000 scan-points per alpha (the one
full-support candidate), completely independent of the admissible/pruning
settings, because the algorithm returns before either could matter. Only
when the certificate is turned off does the search fall through to full
enumeration — and even then, `admissible=True` vs `admissible=False` barely
changes the cost (3.226B vs 3.274B scan-points total), confirming that
almost every support turns out to be admissible for a target this close to
alpha_0 anyway, so that filter has little left to prune here.

## Part 3 — solve_exhaustive operation-count / success-rate tradeoff

Same narrow alpha_0 window as Part 2. `tol` fixed at 1e-3, `grid_n` varied —
this is the one axis that actually changes `solve_exhaustive`'s operation
count (the lattice size `C(grid_n+K-1, K-1)`; `tol` only changes which
lattice points count as "close enough", not how many get evaluated).

| grid_n | lattice points / call | total ops (×50 alphas) | success rate |
|---|---|---|---|
| 2 | 136 | 6,800 | 18% (9/50) |
| 3 | 816 | 40,800 | 80% (40/50) |
| 4 | 3,876 | 193,800 | **100%** (50/50) |

Each ~5-6x increase in `grid_n` buys a large jump in success rate, at a
proportional operation-count cost — this is the genuine tradeoff curve (the
first version of this table mistakenly held `grid_n` fixed and varied `tol`
instead, which changes success rate at *zero* operation-count cost and
hides the real tradeoff entirely).

Compare to Part 2: `solve_exact`'s production default costs only 50,000
scan-points (total, all 50 alphas) for a **certified global optimum every
time**. Even `solve_exhaustive`'s cheapest 100%-success configuration here
(193,800 ops) costs ~4x more than that and still gives no optimality
guarantee — just "closest lattice point found," with no certificate that
it's actually the best point on the simplex.

## Part 3b — solve_exhaustive operation-count / success-rate tradeoff, full range

Same `tol=1e-3`, `grid_n` varied, but now alpha swept over the full
**[0.007356, 0.999999]** range (Part 1's range) instead of the alpha_0
window.

| grid_n | lattice points / call | total ops (×50 alphas) | success rate |
|---|---|---|---|
| 2 | 136 | 6,800 | 16% (8/50) |
| 3 | 816 | 40,800 | 62% (31/50) |
| 4 | 3,876 | 193,800 | 88% (44/50) |
| 5 | 15,504 | 775,200 | **100%** (50/50) |

Counter-intuitive result: success rates here are comparable to — and by
grid_n=5, *better than* — the alpha_0-window table above (which needed
grid_n=4 to reach 100%; this one gets there at grid_n=5 but is already at
88% by grid_n=4, vs. the alpha_0 window's 100% at that same grid_n). The
lattice size for a given `grid_n` is identical in both tables (it depends
only on `grid_n` and K, not on which alpha is targeted) — what's happening
is that a coarse simplex lattice's achievable alpha values turn out to be
spread reasonably evenly across the *entire* reachable range, not
specially concentrated near alpha_0, so hitting a target anywhere in
[alpha_min, alpha_max] is roughly as easy as hitting one in a narrow window
around alpha_0. This is the opposite of what `solve_exact` shows (Part 1 vs
Part 2): for `solve_exact`, the full range is dramatically more expensive
than the alpha_0 window (26.3B vs 1.3M flops, see Part 4), because *that*
solver's cost is driven by how many supports it takes to find/certify the
true optimum — a structural question — not by lattice-point density.

## Part 4 — making these tables actually comparable: FLOPs

Scan-points (Part 2) and lattice-points (Part 3) are **not** the same unit of
work, so the two tables above cannot be compared directly by raw count —
that was a real gap in the original version of this analysis. A `solve_exact`
scan-point is one evaluation of the secular equation at a single `eta` (a
handful of scalar multiply/add/divide ops, independent of K). A
`solve_exhaustive` lattice-point is one *full* weighted-mean/weighted-variance
pass over all K systems (cost scales linearly with K). Putting "50,000 scan
points" next to "193,800 lattice points" and calling the smaller number
cheaper was wrong on its face — the two "points" aren't the same size.

The actual common unit: **floating-point operations (flops)**, counted
op-by-op directly from each function's real arithmetic (standard convention:
a length-n dot product is `2n-1` flops, an elementwise op over length n is
`n` flops, comparisons/selects counted as 1 flop each so both sides are
charged consistently for their control-flow overhead too).

**`solve_exhaustive`**, per lattice point (from the vectorized per-chunk loop
in `solve_w_exhaustive`): two weighted means (`mu_a`,`mu_b`), two weighted
variances (`var_a`,`var_b`), plus `denom`/`achieved`/`gap`/`feasible`/
`objective`/argmin bookkeeping ≈ **14K + 4** flops/point. At K=16: **229
flops/point**.

**`solve_exact`**, three cost centers, each with its own K/n-dependence:
- **Per support visited** (setup — mean/var/eigenstructure via `_prop1_2x2`,
  eigenvectors, and critically the `O(n²)` `M = alpha*outer(b,b) -
  (1-alpha)*outer(a,a)` matrix build + `w@M@w` in `_g`), support size n:
  `5n² + 56n + 54` flops. Using n=K=16 as an upper bound (real supports are
  often smaller, so this overstates the true cost): **1,995 flops/support**.
- **Per scan-point** (the secular equation itself): confirmed empirically
  that essentially every visited support in this dataset has 2 live poles
  (`scan_points / supports_visited` = 1000.0006 ≈ exactly 1000, the 2-pole
  grid size) — so **24 flops/scan-point** uniformly.
- **Per candidate evaluated** (simplex/budget/alpha-match feasibility check +
  the KKT condition-2 check + the curvature certificate, which itself calls
  `_prop1_2x2` again on the full K-length arrays): **≈ 45K + 30** flops. At
  K=16: **743 flops/candidate**.

`total_flops_exact = supports_visited × 1995 + scan_points × 24 + candidates_evaluated × 743`

### The actual comparable numbers (Part 2 / Part 3, near alpha_0)

| Method | Config | Total flops (50-alpha sweep) | Success | Certified optimum? |
|---|---|---|---|---|
| `solve_exact` | production default (all 3 theorems on) | **1,336,900** | 100% | **yes, always** |
| `solve_exhaustive` | grid_n=2, tol=1e-3 | 1,557,200 | 18% | no |
| `solve_exhaustive` | grid_n=3, tol=1e-3 | 9,343,200 | 80% | no |
| `solve_exhaustive` | grid_n=4, tol=1e-3 | 44,380,200 | 100% | no |
| `solve_exact` | admissible=T, pruning=F, cert=F | 87,386,461,312 | 100% | yes (fallback path) |
| `solve_exact` | all 3 theorems off (worst case) | 88,754,778,898 | 100% | yes (fallback path) |

This is the real story the earlier scan-points-vs-lattice-points table
missed entirely: `solve_exact`'s production default is not just *fewer
points* than `solve_exhaustive`'s coarsest grid (grid_n=2) — it does **less
actual arithmetic** (1.34M vs 1.56M flops) while succeeding on 100% of
alphas with a certified global optimum, against grid_n=2's 18% success and
no optimality guarantee at all. To match `solve_exact`'s 100% success rate,
`solve_exhaustive` needs grid_n=4, which costs **~33x more flops** than
`solve_exact`'s default — and even then only returns "closest lattice point
found," never a certificate that it's the true optimum.

The comparison also puts the three-theorems-off worst case in perspective:
even with every optimization disabled, `solve_exact`'s fallback-enumeration
path (~89 billion flops) is doing legitimate, unavoidable work for a
genuinely global search — it isn't cheating by being coarse the way
`solve_exhaustive` is. The theorems' real value shown here is turning that
~89 billion flops into 1.3 million (a ~66,000x reduction) for exactly the
same certified-optimal answer.

### The actual comparable numbers (Part 1 / Part 3b, full range)

| Method | Config | Total flops (50-alpha sweep) | Success | Certified optimum? |
|---|---|---|---|---|
| `solve_exhaustive` | grid_n=2, tol=1e-3 | 1,557,200 | 16% | no |
| `solve_exhaustive` | grid_n=3, tol=1e-3 | 9,343,200 | 62% | no |
| `solve_exhaustive` | grid_n=4, tol=1e-3 | 44,380,200 | 88% | no |
| `solve_exhaustive` | grid_n=5, tol=1e-3 | 177,520,800 | 100% | no |
| `solve_exact` | production default (all 3 theorems on) | **26,304,769,832** | 100% | **yes, always** |
| `solve_exact` | all 3 theorems off (worst case) | 86,307,431,756 | 100% | yes (fallback path) |

A completely different picture from the alpha_0 window. Here
`solve_exhaustive` is the cheap option: even its 100%-success configuration
(grid_n=5, 177.5M flops) costs **~148x less** than `solve_exact`'s
production default (26.3B flops) for the same 100% success rate — a full
reversal from the alpha_0-window comparison, where `solve_exact` won on both
cost *and* correctness guarantee. The reason is structural, not a flaw in
either solver: `solve_exact`'s cost is driven by how hard it is to
find/certify the true optimum (Part 1 showed only 19/50 alphas certify
directly here, the rest need a real fallback search over up to a million
admissible supports), while `solve_exhaustive`'s cost is driven purely by
lattice density, which — per Part 3b — turns out to not care much whether
the target is near alpha_0 or anywhere else in range. So which solver is
"cheaper" depends entirely on the regime: near alpha_0, `solve_exact` wins
by ~30x on flops *and* gives a certificate; across the full range,
`solve_exhaustive` wins by ~150x on flops but gives no certificate at all —
it can't tell you whether a smoother, un-sampled point elsewhere on the
simplex would have scored even better. That correctness gap is exactly why
`solve_exact` is the production solver despite being more expensive here:
Part 1 already showed 29 of 50 full-range alphas need the fallback path
specifically because a naive/coarse search would have missed the true
optimum (`lib.reweight_exact`'s own module docstring: the retired numeric
solver missed it in 13/15 real-data test cases at default settings).

## Part 5 — everything together

All 12 configurations (8 `solve_exact` theorem combinations + 4
`solve_exhaustive` grid resolutions), both alpha regimes side by side.
"Supports visited" / "candidates evaluated" only apply to `solve_exact`
(`solve_exhaustive` doesn't have either concept — it evaluates every lattice
point unconditionally, dashed out below). Every `solve_exact` row succeeds
on 100% of alphas in both regimes (it's sound+complete by construction), so
that column is constant for those 8 rows and only actually varies for the
`solve_exhaustive` rows.

<table>
<thead>
<tr>
<th rowspan="2">Solver</th>
<th rowspan="2">Config</th>
<th colspan="4">[alpha_0 &minus; 0.05, alpha_0 + 0.05]</th>
<th colspan="4">Full range [alpha_min, alpha_max]</th>
</tr>
<tr>
<th>Supports visited</th><th>Candidates evaluated</th><th>Success</th><th>Total flops</th>
<th>Supports visited</th><th>Candidates evaluated</th><th>Success</th><th>Total flops</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="8"><code>solve_exact</code></td><td>admissible=T, pruning=T, certificate=T <em>(production default)</em></td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>951,432</td><td>2,116,144</td><td>100%</td><td>26,304,769,832</td></tr>
<tr><td>admissible=T, pruning=T, certificate=F</td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>951,748</td><td>2,116,460</td><td>100%</td><td>26,313,219,040</td></tr>
<tr><td>admissible=T, pruning=F, certificate=T</td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>1,771,170</td><td>3,862,176</td><td>100%</td><td>48,911,136,918</td></tr>
<tr><td>admissible=T, pruning=F, certificate=F</td>
  <td>3,225,830</td><td>4,752,434</td><td>100%</td><td>87,386,461,312</td>
  <td>2,999,023</td><td>5,759,754</td><td>100%</td><td>82,239,076,107</td></tr>
<tr><td>admissible=F, pruning=T, certificate=T</td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>1,016,954</td><td>2,303,079</td><td>100%</td><td>28,146,906,927</td></tr>
<tr><td>admissible=F, pruning=T, certificate=F</td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>1,017,270</td><td>2,303,395</td><td>100%</td><td>28,155,356,135</td></tr>
<tr><td>admissible=F, pruning=F, certificate=T</td>
  <td>50</td><td>50</td><td>100%</td><td>1,336,900</td>
  <td>1,900,122</td><td>4,253,661</td><td>100%</td><td>52,520,584,713</td></tr>
<tr><td>admissible=F, pruning=F, certificate=F <em>(brute enumeration)</em></td>
  <td>3,275,950</td><td>4,919,336</td><td>100%</td><td>88,754,778,898</td>
  <td>3,144,912</td><td>6,206,212</td><td>100%</td><td>86,307,431,756</td></tr>
<tr><td rowspan="4"><code>solve_exhaustive</code></td><td>grid_n=2, tol=1e-3</td>
  <td>&mdash;</td><td>&mdash;</td><td>18% (9/50)</td><td>1,557,200</td>
  <td>&mdash;</td><td>&mdash;</td><td>16% (8/50)</td><td>1,557,200</td></tr>
<tr><td>grid_n=3, tol=1e-3</td>
  <td>&mdash;</td><td>&mdash;</td><td>80% (40/50)</td><td>9,343,200</td>
  <td>&mdash;</td><td>&mdash;</td><td>62% (31/50)</td><td>9,343,200</td></tr>
<tr><td>grid_n=4, tol=1e-3</td>
  <td>&mdash;</td><td>&mdash;</td><td>100% (50/50)</td><td>44,380,200</td>
  <td>&mdash;</td><td>&mdash;</td><td>88% (44/50)</td><td>44,380,200</td></tr>
<tr><td>grid_n=5, tol=1e-3</td>
  <td>&mdash;</td><td>&mdash;</td><td>100% (50/50)</td><td>177,520,800</td>
  <td>&mdash;</td><td>&mdash;</td><td>100% (50/50)</td><td>177,520,800</td></tr>
</tbody>
</table>

Reading it as one picture: `solve_exhaustive`'s flops column is *identical*
between the two big regimes at every grid_n (1,557,200 / 9,343,200 /
44,380,200 / 177,520,800 either way) — lattice size depends only on grid_n
and K, never on which alpha is targeted, so operation cost is regime-blind;
only its success rate moves (worse in the full range at grid_n=3/4, tying by
grid_n=5). `solve_exact`'s flops column, by contrast, swings by 4-5 orders
of magnitude between regimes for the *same* theorem configuration (e.g. the
production-default row: 1.34M near alpha_0 vs. 26.3B full-range) — because
its cost is driven by how much of the support space actually has to be
searched to certify the optimum, which is exactly what's different between
"alpha is already close to the natural uniform weighting" and "alpha is
somewhere the natural weighting doesn't already solve."