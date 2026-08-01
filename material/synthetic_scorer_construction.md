# Construction of Synthetic Scorers

A guide to building synthetic scorers with a controlled dependence on a single aspect ($a$ or $b$), derived from a real scorer by arithmetic only.

---

## 1. Setting

Work within one **pool** (dataset = language pair / domain / year). Never pool across datasets. Every quantity below is defined inside a single pool.

A pool is a rectangle: a fixed system set $\mathsf{Syss}$ and a fixed segment set $\mathsf{Segs}$, with every system scoring every segment.

| symbol | meaning |
|---|---|
| $k \in \mathsf{Syss}$ | translation system, $K = \|\mathsf{Syss}\|$ |
| $i \in \mathsf{Segs}$ | segment, $n = \|\mathsf{Segs}\|$ |
| $s_0$ | a real scorer, the **donor** |
| $y(s_0,k,i)$ | the donor's segment-level score |
| $a(k,i)$, $b(k,i)$ | the two gold aspects (adequacyMQM, fluencyMQM). Independent of the scorer. |

The construction is written for the aspect $a$. For the $b$-family, swap $a \leftrightarrow b$ throughout and use the dial name $B$. **Never combine the two in one scorer.** The $a$-family and the $b$-family are separate sets of synthetic scorers.

---

## 2. Why the donor's shape must be preserved rather than modelled

A synthetic scorer could be written down directly, e.g. $y = w_a a + w_b b + \text{noise}$. That gives perfect control and no realism. Two measurements on the WMT donor set show why it is not acceptable here.

### 2.1 Real scorers are structurally non-monotone in each aspect

For each donor, let $m$ be the aspect-conditional mean (§3.1), $m^{\text{iso}}$ its weighted isotonic regression, and

$$T \;=\; \sum_{a_0} |G_{a_0}| \,\big(m(a_0) - m^{\text{iso}}(a_0)\big)^2$$

the monotonicity violation. Testing $T$ against a parametric bootstrap under the isotonic null — resample $e$ within each aspect group, rebuild $y^* = m^{\text{iso}} + e^*$, recompute $T^*$ — rejects monotonicity for a large share of donors, with effect sizes reaching $T/\text{TSS} > 0.9$ for surface-overlap metrics.

The jaggedness is therefore **structural, not sampling noise**. Any parametric or monotone form imposed on the aspect response would discard measured structure. This holds separately for $a$ and for $b$; it says nothing about joint shape across $(a,b)$.

Real non-monotonicity here is not necessarily scorer error. Two segments with the same $a$ differ in error *composition* — one major versus several minors — and a scorer may legitimately respond to that difference.

### 2.2 Aspect responsiveness varies sharply and inconsistently across donors

Kendall $\tau(\bar m_k, \bar a_k)$ across systems (§3.4) measures whether a donor's aspect-predicted system ranking agrees with the aspect itself. Across the donor set it is not concentrated: neural donors (COMET, XCOMET, MetricX, BLEURT) are close to $+1$; surface-overlap donors (chrF, BLEU, spBLEU, eBLEU, MEE, REUSE, BLCOM) are far from it, overwhelmingly on fluency rather than adequacy. Discordance survives isotonization in a residual set of donors, so its cause there is distributional — systems differ in the *spread* of their aspect values, not only the mean — and not a shape artifact.

A single generative form cannot cover this range. Donor-specific structure can, at no cost, because it is copied rather than fitted.

**Consequence for the construction.** The aspect response is stored as a lookup table, never fitted, and $\tau(\bar m_k, \bar a_k)$ is reported as a **donor property** in the results, not as a validity condition on the method.

---

## 3. Definitions

### 3.1 Group mean $m$

For each distinct value $a_0$ occurring in the pool:

$$m(a_0) \;=\; \frac{1}{|G_{a_0}|}\sum_{(k,i) \in G_{a_0}} y(s_0,k,i), \qquad G_{a_0} = \{(k,i) : a(k,i) = a_0\}$$

A **lookup table**, one number per distinct $a$ value. Not a fitted function. Whatever shape the donor's response to $a$ has — bends, plateaus, jumps — is carried exactly.

### 3.2 Residual $e$

$$e(k,i) \;=\; y(s_0,k,i) \;-\; m\big(a(k,i)\big)$$

Everything in the donor not explained by $a$. By construction $\sum_{(k,i)\in G_{a_0}} e(k,i) = 0$ for every $a_0$.

### 3.3 System offset

$$\bar e_k \;=\; \frac{1}{n}\sum_i e(k,i)$$

How much the donor runs high or low on system $k$, beyond what $a$ predicts. Not zero: $e$ is centred within $a$-groups, not within systems.

$\bar e_k$ is **constant across segments within a system**. The whole construction rests on this.

### 3.4 System-level aggregate of $m$

$$\bar m_k \;=\; \frac{1}{n}\sum_i m\big(a(k,i)\big)$$

System $k$'s mean score if the donor used $a$ alone. Determined entirely by which $a$ values system $k$ produces.

---

## 4. The generator

$$\boxed{\;y_A(k,i) \;=\; y(s_0,k,i) \;-\; A\,\bar e_k\;}$$

equivalently, written to show the parts:

$$y_A(k,i) \;=\; m\big(a(k,i)\big) \;+\; (1-A)\,\bar e_k \;+\; \big[e(k,i) - \bar e_k\big]$$

with dial $A \in [0,1]$.

| term | role |
|---|---|
| $m(a(k,i))$ | the donor's real, shape-free response to $a$. Fixed. |
| $(1-A)\,\bar e_k$ | the donor's system offset. **The only thing the dial touches.** |
| $e(k,i) - \bar e_k$ | the donor's within-system residual, untouched at every $A$ — real noise, real heteroscedasticity, real cross-system segment structure, real per-system tilt. |

The construction subtracts a per-system constant. Nothing else.

### Endpoints

- $A = 0$: $\;y_0 = y(s_0,\cdot,\cdot)$. **Exactly the real donor scorer.**
- $A = 1$: $\;y_1 = y(s_0,\cdot,\cdot) - \bar e_k$. A scorer whose system-level ranking is determined by $\bar m_k$ alone, carrying the donor's real segment-level noise.

### System-level consequence

$$\boxed{\;u_k(A) \;=\; \bar m_k \;+\; (1-A)\,\bar e_k\;}$$

where $u_k = \operatorname{mean}_i y_A(k,i)$ is what every metametric reads.

A straight line in $\mathbb{R}^K$ between two fixed vectors. Consequences:

- Pearson has a closed form in $A$.
- Spearman, Kendall, and PA change only when two systems swap order. Along a line each pair swaps **at most once**, so these are step functions with at most $\binom{K}{2}$ jumps, and every jump point can be solved for exactly. No grid, no simulation.
- SPA also has a closed form. Its p-value for a pair depends on $A$ only through $\Delta_{kk'}(A) = u_k(A) - u_{k'}(A)$; see conservation 5. Precompute the pair's permutation coefficients once and sweep. No regeneration.

---

## 5. Procedure

Per pool, per donor $s_0$:

1. Build the $a$-group index. Compute $m(a_0)$ for every distinct $a_0$. **Use the full pool** — see §5.1.
2. Compute $e$, then $\bar e_k$.
3. Compute $\bar m_k$.
4. For each $A$ on your grid, emit $y_A(k,i) = y(s_0,k,i) - A\,\bar e_k$.
5. Repeat for aspect $b$ with dial $B$, producing a separate family.

### 5.1 Invariance requirement

$m$, $e$, $\bar e_k$, and $\bar m_k$ must be computed **once, on the full pool**, and then held fixed.

When evaluating on a subsample $S \subset \mathsf{Syss}$, **select** rows — never recompute. Recomputing inside a subsample changes the scorer's identity with the evaluation set, which is exactly the confound the analysis is designed to measure.

### 5.2 SPA permutation requirement

If SPA's p-values come from sampled rather than exhaustive permutations, the **same set of sign vectors must be reused at every $A$**. A fresh sample per $A$ adds Monte Carlo jitter unrelated to the dial.

This is consistent with the reference implementation, which already <cite index="10-1">caches a batch of permutations and reuses it for each system pair within a test set</cite>; the requirement here extends that reuse across dial settings.

With the set fixed, each pair's permuted statistics are affine in $\Delta_{kk'}$ with coefficients independent of $A$ (§7), so the coefficients can be precomputed once and the whole sweep evaluated by lookup.

### 5.3 Segment-mean aggregation (inherited assumption)

Every identity in §4 treats the system score as the unweighted mean of segment scores, $u_k = \operatorname{mean}_i y_A(k,i)$.

This assumption is not introduced here. It is SPA's own: <cite index="10-1">when computing p-values, system-level metric scores are assumed to be the average of segment-level metric scores</cite>, and <cite index="10-1">this does not hold for some metrics, notably BLEU and chrF, which compute statistics at the segment level and combine them into document-level scores</cite>. The authors handle it by <cite index="10-1">averaging the segment-level scores for simplicity, noting the results are self-consistent but will not accurately reflect those metrics' true performance</cite>, and observe that <cite index="10-1">the same approximation is made in prior work computing statistical significance of metrics, including the WMT shared tasks</cite>.

Two consequences for this construction.

For a donor whose native system score is not the segment mean, $u_k(A) = \bar m_k + (1-A)\bar e_k$ describes the segment-mean aggregation rather than the donor's published system score. Since the meta-evaluation pipeline applies the same convention, the synthetic scorer and the real donor are compared on equal terms and the dial behaves as specified — but neither is the metric as its authors defined it.

Record which donors are affected (corpus-level BLEU, chrF, and their variants) and report them as a separate arm, on the same grounds the original work gives: self-consistent, but not a faithful measurement of those metrics.

---

## 6. What the construction conserves

| # | Conserved | Why it holds |
|---|---|---|
| 1 | **Shape of the response to $a$** | $m$ is a lookup table, not a fit. Curvature, plateaus, and jumps are inherited, never imposed. |
| 2 | **All segment-level structure** | Only a per-system constant is subtracted. Heteroscedasticity, per-system tilt, atoms, discreteness, and cross-system segment difficulty are carried untouched at every $A$. |
| 3 | **Realism at $A=0$** | $y_0 = y(s_0)$ identically. One endpoint is a genuine WMT-submitted scorer, not an approximation of one. |
| 4 | **The aspect-determined endpoint** | $u_k(1) = \bar m_k$, a function of system $k$'s $a$ values alone. Whether $\bar m_k$ ranks systems in $\bar a$ order is a property of the donor, not a claim of the construction. |
| 5 | **The entire centred score matrix** | $y_A(k,i) - u_k(A) = y_0(k,i) - u_k(0)$ for every $k,i,A$: the dial moves column means and nothing else. Consequently every metametric — including SPA, whose permutation p-values are the only segment-level quantity in play — depends on $A$ solely through the system-level mean gaps. Proved exactly in §7. |
| 6 | **Linearity of the system-level path** | $u_k(A) = \bar m_k + (1-A)\bar e_k$. Four of five metametrics become closed-form or exactly solvable. |
| 7 | **Shape diversity across donors** | Each real $s_0$ supplies its own $m$. Robustness to scorer shape is tested with real shapes, not synthetic variation. |

---

## 7. Proof that SPA depends on the dial only through the system-level gaps

### What has to be shown

SPA scores a pair of systems by comparing two p-values: one computed from the scorer's segment scores, one from the gold's. The gold side uses no scorer output, so it is the same at every dial setting. Only the scorer-side p-value can move, and that is what this section is about.

### Notation

Fix one pool, one donor $s_0$, and one pair of systems $(k,k')$. Let $n = |\mathsf{Segs}|$.

| symbol | meaning |
|---|---|
| $y_A(k,i)$ | the synthetic scorer's score for system $k$ on segment $i$, dial at $A$ (§4) |
| $d_i(A)$ | the two systems' **gap on segment $i$**: $\;d_i(A) := y_A(k,i) - y_A(k',i)$ |
| $\Delta(A)$ | the **average gap**: $\;\Delta(A) := \frac1n\sum_i d_i(A)$, which equals $u_k(A) - u_{k'}(A)$ |
| $\delta$ | the pair's offset difference: $\;\delta := \bar e_k - \bar e_{k'}$, a single number |

Setting $A=0$ in the first two rows gives $d_i(0)$ and $\Delta(0)$ — the real donor's own per-segment and average gaps.

### Step 1 — turning the dial slides every segment's gap by the same amount

The generator subtracts a per-system constant: $y_A(k,i) = y_0(k,i) - A\,\bar e_k$. Subtracting the two systems,

$$d_i(A) \;=\; \big[y_0(k,i) - A\,\bar e_k\big] - \big[y_0(k',i) - A\,\bar e_{k'}\big] \;=\; d_i(0) \;-\; A\,\delta$$

$\delta$ does not depend on $i$. So every segment's gap slides by the same $A\delta$. The gaps keep their positions relative to one another exactly.

### Step 2 — so only the average gap moves; the shape of the gap vector is frozen

Averaging Step 1 over segments gives $\;\Delta(A) = \Delta(0) - A\delta$. Subtracting that from Step 1,

$$d_i(A) - \Delta(A) \;=\; d_i(0) - \Delta(0)$$

The left side is a quantity at dial setting $A$; the right side has no $A$ in it. So this common value — call it $c_i$ — is the same at every dial setting:

$$d_i(A) \;=\; c_i + \Delta(A), \qquad c \text{ fixed}, \qquad \sum_i c_i = 0$$

**$c$ is the entire segment-level structure of the pair, and it never moves. $\Delta(A)$ is the only moving part.**

### Step 3 — what the permutation test computes

SPA's p-value is **one-sided and directional**, not two-sided. In the original formulation, <cite index="10-1">$p_{ij}$ is the p-value for the hypothesis that system $i$ is better than system $j$</cite>, and <cite index="10-1">a value near 0 means near-certainty that $i$ beats $j$, a value near 1 means near-certainty that $j$ beats $i$, and 0.5 means the two are indistinguishable</cite>. A two-sided p-value could not carry that information, since it would be small at both extremes.

<cite index="10-1">The p-value is the fraction of permutations whose mean difference is greater than or equal to the observed mean difference</cite> — a plain $\ge$, with no absolute values anywhere.

The test is paired: <cite index="10-1">each split contains exactly one translation of every test-set sentence</cite>. With two systems, that means reassigning which system a segment's two scores belong to, which simply flips the sign of that segment's gap. So one permutation is a vector $\epsilon = (\epsilon_1,\dots,\epsilon_n)$ of $\pm 1$, producing

$$T(\epsilon, A) \;=\; \frac1n\sum_i \epsilon_i\, d_i(A)$$

The observed statistic is the all-$+1$ case, $\;T(\mathbf 1, A) = \Delta(A)$.

Let $E$ be the set of permutations used: all $2^n$ for an exact test, or a fixed sample (<cite index="10-1">the reference implementation uses 1000</cite>).

### Step 4 — each permutation's statistic is a straight line in $\Delta$, with fixed slope and intercept

Substituting Step 2 into Step 3:

$$T(\epsilon, A) \;=\; \frac1n\sum_i \epsilon_i\big(c_i + \Delta(A)\big) \;=\; \alpha_\epsilon \;+\; \beta_\epsilon\,\Delta(A)$$

$$\alpha_\epsilon = \frac1n\sum_i \epsilon_i c_i, \qquad \beta_\epsilon = \frac1n\sum_i \epsilon_i$$

$\alpha_\epsilon$ is built from $\epsilon$ and $c$; $\beta_\epsilon$ from $\epsilon$ alone. Neither contains $A$, and $c$ is fixed by Step 2. **Both are constants of the sweep.**

### Step 5 — conclusion

The p-value counts the permutations reaching at least the observed statistic. Using Step 4 and the one-sided form from Step 3:

$$p(A) \;=\; \frac{1}{|E|}\,\#\Big\{\epsilon \in E \;:\; \alpha_\epsilon + \beta_\epsilon\,\Delta(A) \;\ge\; \Delta(A)\Big\} \;=\; \frac{1}{|E|}\,\#\Big\{\epsilon \in E \;:\; \alpha_\epsilon \;\ge\; (1-\beta_\epsilon)\,\Delta(A)\Big\}$$

Every quantity on the right is fixed except $\Delta(A)$. Therefore

$$\boxed{\;p_{kk'}(A) \;=\; g_{kk'}\big(\Delta_{kk'}(A)\big)\;}$$

where $g_{kk'}$ is determined entirely by $\{(\alpha_\epsilon, \beta_\epsilon)\}_{\epsilon \in E}$ and does not depend on $A$. $\blacksquare$

### Step 5′ — the one-sided form makes $g$ monotone

Since $\beta_\epsilon = \frac1n\sum_i\epsilon_i \in [-1,1]$, we have $1-\beta_\epsilon \ge 0$, with equality only for $\epsilon = \mathbf 1$. For that one permutation $\alpha_{\mathbf 1} = \frac1n\sum_i c_i = 0$, so the condition reads $0 \ge 0$ and is always satisfied — the observed statistic always counts itself.

For every other $\epsilon$, divide by $1-\beta_\epsilon > 0$: the condition becomes

$$\Delta(A) \;\le\; \theta_\epsilon, \qquad \theta_\epsilon := \frac{\alpha_\epsilon}{1-\beta_\epsilon}$$

a single fixed threshold per permutation. Hence

$$p(A) \;=\; \frac{1}{|E|}\,\#\big\{\epsilon \in E : \theta_\epsilon \ge \Delta(A)\big\}$$

which is a **non-increasing step function of $\Delta$** — the complementary empirical distribution function of the fixed threshold set $\{\theta_\epsilon\}$.

Combined with $\Delta(A) = \Delta(0) - A\delta$ being linear in $A$, this gives: $p_{kk'}(A)$ is monotone in $A$, increasing or decreasing according to the sign of $\delta = \bar e_k - \bar e_{k'}$, and constant when $\delta = 0$. No non-monotone artifact can arise anywhere in the sweep.

### What follows

**It is exact.** No normal approximation, no moment condition, no assumption about the shape of the score distribution. It holds for the exhaustive test and for any fixed sample of permutations.

**SPA as a whole.** The gold-side p-value is constant in $A$, and SPA averages <cite index="10-1">$1 - |p^{\text{gold}}_{kk'} - p^{\text{scorer}}_{kk'}|$ over all system pairs</cite>. So SPA depends on the dial only through the system-level average gaps $\{\Delta_{kk'}(A)\}$ — exactly the information the other four metametrics read.

**One requirement.** $E$ must be identical at every dial setting. Redrawing it per $A$ changes $g_{kk'}$ between settings and adds Monte Carlo noise unrelated to the dial (§5.2). This matches the reference implementation, which <cite index="10-1">caches a batch of permutations and reuses it across system pairs within a test set</cite>.

**One saving.** $(\alpha_\epsilon, \beta_\epsilon)$ — or directly the thresholds $\{\theta_\epsilon\}$ — can be computed once per pair, after which the whole sweep is a lookup. Segment scores never need regenerating.

**One clarification of wording.** The spread of the permutation null is $\frac1n\big(\frac1n\sum_i c_i^2 + \Delta(A)^2\big)$, which does change along the sweep — but only through $\Delta(A)$, as the theorem says. This coupling belongs to the sign-flip test itself and is present for any scorer. So "the certainty channel is frozen" is the wrong phrase; "certainty is determined by the average gap" is the right one.

---

## 8. Scope statement

1. The synthetic scorers span the shapes of scorers actually submitted to WMT, via the donor set. They are not draws from a scorer population, and no prevalence claim follows from them.
2. The dial raises $a$-influence and lowers $b$-influence simultaneously, because $b$'s influence orthogonal to $a$ sits in $\bar e_k$. Report $\operatorname{corr}_k(u_k(A), \bar b_k)$ alongside $\operatorname{corr}_k(u_k(A), \bar a_k)$ across the sweep. Results are read per axis, never diagonally.
3. The dial moves the mean channel only. Every metametric's view of the data — including SPA's permutation p-values — depends on $A$ solely through the system-level mean gaps, exactly and without approximation (conservation 5).
4. $A=0$ is a real scorer; $A=1$ is the scorer whose system ranking is determined by the aspect-conditional response alone. Whether that endpoint ranks systems in aspect order is a donor property, reported per donor.
