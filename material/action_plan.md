# Mind Which Bird You Favor: Parameterizing Adequacy–Fluency Balance in Meta-Evaluation of Machine Translation

*Design specification: problem, scope, critique of prior work, the reweighting machinery and its proofs, the evaluation plan, and the limitations.*

**Prior work.** This work builds directly on Shayegh et al. (2025), *Feeding Two Birds or Favoring One? Adequacy–Fluency Tradeoffs in Evaluation and Meta-Evaluation of MT* (Proceedings of WMT 2025).

---

## 1. The exact problem and its scope

### 1.1 The setting

- **All at the system level.** We work purely at the system level: each system is a single point. The system-level scores are themselves averages over many segments, but we do not model segment-level scores in this work, following the system-level meta-evaluation setup of Shayegh et al. (2025).
- **Translation systems** are the objects being ranked. There are $K$ of them. Their Adequacy MQM scores form a vector $a = (a_1,\dots,a_K)$ and their Fluency MQM scores a vector $b = (b_1,\dots,b_K)$; system $i$ is the point $(a_i, b_i)$. All MQM is $t = a + b$, i.e. $t_i = a_i + b_i$ (the rare uncategorized errors are omitted, following Shayegh et al., 2025). "Variance of $a$" means the variance of the $K$ values $\{a_i\}$.
- **Scorers (metrics)** are the things doing the ranking: a scorer assigns a score $m_i$ to each system $i$, and ranks the systems by those scores.
- **Meta-evaluation** judges a scorer by how well its ranking of systems agrees with the human (All MQM) ranking, measured by a **meta-evaluation metric** (meta-metric). Common meta-metrics include **Pairwise Accuracy (PA)** and **Soft Pairwise Accuracy (SPA)**, as well as Pearson, Spearman, and Kendall's $\tau$ correlations.

### 1.2 The phenomenon

Because $t = a + b$, the component with larger system-level variance dominates the All MQM ranking, hence dominates the meta-evaluation verdict. In WMT, Adequacy MQM has much higher system-level variance than Fluency MQM (Shayegh et al., 2025, Table 2), so the standard meta-evaluation **leans toward adequacy** and rewards adequacy-oriented metrics.

### 1.3 The exact problem we solve

The submitted WMT systems are **an uncontrolled sample** from a population of possible systems — not a fair or designed sample. The adequacy–fluency balance of the meta-evaluation is therefore partly an **artifact of system selection**. We provide a method to **control that balance** by reweighting the real systems, exposing the balance as a single tunable parameter $\alpha$.

**Why no one can name the ideal $\alpha$.** The ideal balance would require the *true* distribution of translation systems, which is unknown; the WMT submissions are very unlikely to represent it, since the submitted set **shifts from year to year, across language pairs, and across subsets** (visible directly in our consistency experiment, §6.5). A moving, filtered, tiny sample cannot define a stable "correct" balance. This is what the method is for: since the right point cannot be named, the useful object is a transparent dial plus standpoint-free summaries (§6.1), with the choice left to whoever has the domain knowledge. Everything downstream — the retirement of the intrinsic/extrinsic split (§2.2), the survey-reweighting vocabulary without a population (§3.1), the expert-declared target — is a consequence of this one fact.

This **follows Shayegh et al. (2025) rather than contradicting it**, and is one realization of the post-hoc-debiasing direction they flagged as Future Work (debiasing meta-evaluation outputs without altering the datasets) — one realization, not the only one: reweighting units is not the same operation as normalizing scores.

**What is inherited, and what is new.** The conceptual thesis — that an adequacy–fluency tradeoff exists at the *meta-evaluation* level and is driven by system selection — is due to Shayegh et al. (2025). Our deltas are: a **continuous, artifact-free instrument** for moving the balance (reweighting real systems, §3); the **$M(\alpha)$ curve** as the object of study (§6.1); and the **subset-consistency** result (§6.5), which carries the novelty weight.

### 1.4 Scope and standpoint (the boundaries we deliberately keep)

- **We do not name the correct balance** (§1.3): $\alpha$ is a control surface, exposed as a free parameter and not resolved.
- **We drop the F-statistic, p-values, $B(\Delta p)$, and the intrinsic/extrinsic decomposition** as headline indicators (§2.2), replacing them with a general balance control. The method is a **conditional**: given the expert's declared intrinsic share, here is the rebalanced view.
- **System-level only** ⇒ the variance we control is total observed balance (intrinsic scaling × selection effect combined); with no segments we cannot separate the two, and do not try.
- **The imbalance is not equally visible to every meta-metric.** We report **all five** meta-metrics (PA, SPA, Pearson, Spearman, Kendall $\tau$) on equal footing; §4 gives the reading key for what each one sees of $\alpha$, and §7 gives the PA-native alternative target, left to future work.
- **One axis only.** Adequacy–fluency is one named tradeoff among many (verbosity, length, register, …). We control this one axis; the rest are uncontrolled and left to future work. The machinery generalizes to any nameable axis with a defined component decomposition.
- **System selection is the right level; segment selection is deliberately *not* controlled.** Segments are numerous and effectively random, and their segment-level variance is a trustworthiness signal SPA uses; the full argument and the per-meta-metric scope are in §4.

---

## 2. The prior work, and where we depart from it

### 2.1 The synthesis method and its flaws

To rebalance, Shayegh et al. (2025) **synthesize pseudo-systems**: for each segment, rank the real candidates by Adequacy MQM and assign the $k$-th best to synthetic adequacy-system $k$ (and likewise for fluency). They then meta-evaluate on a pool of original + synthesized-by-adequacy + synthesized-by-fluency systems, choosing a "balanced" pool (their Rows 6/7) by the $B(\Delta p)$ indicator.

**A conservation law pins the side effect.** Fix one component (say adequacy) and consider the pool of its per-segment scores across all systems and segments (the segments are shared across systems). A system's system-level score is its mean over segments. Grouping the pooled scores by system, the **law of total variance** (Eve's law; e.g. Blitzstein & Hwang, *Introduction to Probability*, 2nd ed., 2019) splits the pooled variance into within-system and between-system parts:

$$\mathrm{Var}(\text{segment score}) \;=\; \underbrace{\mathbb{E}_{\text{sys}}\big[\mathrm{Var}_{\text{seg}}(\text{score}\mid\text{sys})\big]}_{\text{within-system } (W)} \;+\; \underbrace{\mathrm{Var}_{\text{sys}}\big(\mathbb{E}_{\text{seg}}[\text{score}\mid\text{sys}]\big)}_{\text{between-system } (B)}.$$

Synthesis merely reassigns, per segment, which synthetic system receives which score, by rank (synthetic system $k$ gets the segment's rank-$k$ score). This permutes scores *within* each segment, so the pooled total $T = W + B$ is **identical** for the original, adequacy-synthesized, and fluency-synthesized sets. (Fluency scores are carried along under the same per-segment adequacy ranking, so the statement holds per component.) With $T$ fixed, $W$ and $B$ trade off exactly: rank-sorting inflates the spread of system means ($B\uparrow$) precisely by compressing each synthetic system's segment distribution ($W\downarrow$).

**The flaws.** Synthesis is therefore **coarser and confounded, though not invalid**. Three weaknesses matter here.

1. **Distorted SPA standard errors.** The compression of $W$ tightens each synthesized system's within-system spread, hence its standard error; SPA reads that standard error to form its confidence, so synthesized systems carry artificially tight standard errors and distorted confidence — a confound landing squarely on a meta-metric we report.
2. **Artificiality / unrepresentativeness.** The same compression gives synthesized systems a segment structure unlike any real system's, and they are not real translation systems to begin with — Shayegh et al. (2025) themselves flag this and restrict their use.
3. **No clean balance knob.** Synthesis produces only *extreme* adequacy/fluency directions; intermediate balances come from coarse, integer-grained **mix ratios** of pool sizes. It offers no continuous, interpretable balance knob.

Our reweighting, acting only on the $K$ system-level points, leaves every real system's segment-level structure untouched, so it carries none of these artifacts.

**What synthesis is still good for.** None of this makes synthesis useless. Retargeted through our own balance definition (§3.2) rather than the $B$ indicator, it becomes a legitimate **comparison baseline** at matched realized $\alpha$ (§6.5), and — in the mechanical experiments only, with the SPA caveat owned — a **range-extender** that shows the knob keeps turning and the solver stays cheap outside the natural range (§6.2, §6.6).

### 2.2 The F-statistic and $B(\Delta p)$: the insight we keep, and our alternative

**The insight we keep.** The conceptual contribution of Shayegh et al. (2025) is what launched this work: that a meta-evaluation's adequacy–fluency balance is, in large part, driven by *system selection*. They make it visible through the between-system variance and formalize it with the F-statistic, its right-tail p-value $\Delta p$, and the $B(\Delta p)$ balance indicator, splitting the observed imbalance into an undesired part due to biased system selection, referred to as *extrinsic*, and the rest as *intrinsic*. We inherit the finding; what we retire is the attempt to *measure* the extrinsic part.

**Extrinsic bias is defined as a contrast with an unknown reference.** Take the definition seriously. System-selection bias is not "the imbalance produced by this pool of systems"; it is the imbalance of *this* pool *relative to the imbalance under the true distribution of systems* — the excess that selection introduced. The first term is measurable; the second is not, and never will be, because the true distribution is unknown and the submitted set is a moving, filtered, tiny sample of it (§1.3). A quantity that is one measurable term minus one unknowable term is not estimable, and no statistic can recover it. This is the same unknowability that prevents anyone from naming the ideal $\alpha$ (§1.3), one level up: the reason we cannot name the correct balance is the reason we cannot isolate the extrinsic share.

The F-statistic does not escape this — it only hides it. Dividing between-system variance by within-system variance silently installs within-system variance as the missing reference, i.e. it *assumes* the true distribution contributes equally to the adequacy and fluency between-system variances. There is no reason it should: the ideal distribution's adequacy and fluency spreads need not be equal. $F$ is thus not a clean measurement of extrinsic bias but a measurement under an unstated, unverifiable choice of the very reference that makes the quantity ill-posed.

**Our alternative: don't partition; expose a dial.** The decomposition has thus completed its role. Its job was *diagnostic* — establishing that system selection controls the adequacy–fluency balance, and severely — not *prescriptive*, and we keep the diagnosis while dropping the un-estimable split. In its place we control a single, scale-explicit quantity: the **total system-level variance** and its adequacy–fluency ratio, *not* partitioned into intrinsic and extrinsic parts. We do not name the "correct" balance; we expose it as a free parameter $\alpha$ and hand the choice to the expert, who sets $\alpha$ by how much of the imbalance they judge undesired — **declared, not discovered**. The machinery that realizes it is §3.

---

## 3. The reweighting machinery

### 3.1 Idea

Treat the submitted systems as a biased sample and **importance-weight them**. Assign a weight $w_i$ to each system $i$ (normalized $\sum_i w_i = 1$). A system pair $(i,j)$ inherits weight $w_i w_j$. This is the standard machinery of **calibration / importance weighting / survey reweighting** (Deville & Särndal 1992; Hainmueller 2012; Horvitz–Thompson), which also imports its known failure modes (extreme weights, low effective sample size).

**A sample of what?** We borrow the vocabulary, not the estimand. One could regard the submitted systems as a sample of the *imaginary true distribution* of translation systems — but that distribution does not exist (§1.3, §2.2), so there is no population quantity to estimate. Reweighting is therefore the minimum-distortion way to impose a chosen balance on the systems actually at hand, a **control surface** rather than a distributional parameter to be tuned.

### 3.2 The control target

Let $w = (w_1,\dots,w_K)$ be the system weights (§3.1). Using the **weighted-covariance operator** $\Sigma(w) := \mathrm{diag}(w) - ww^\top$, define the weighted mean, variance, and covariance

$$\mu(a; w) := w^\top a, \qquad \sigma(a; w)^2 := a^\top \Sigma(w)\, a = \textstyle\sum_i w_i a_i^2 - (w^\top a)^2, \qquad \sigma(a,b; w) := a^\top \Sigma(w)\, b,$$

and likewise for $b$. The plain (unweighted) mean is the uniform-weight case, $\mu(a) := \mu(a; \tfrac1K\mathbf 1) = \tfrac1K\mathbf 1^\top a$. We control the weighted system-level variances $\sigma(a;w)^2$ and $\sigma(b;w)^2$, and define the balance parameter

$$\alpha = \frac{\sigma(a;w)^2}{\sigma(a;w)^2 + \sigma(b;w)^2} \in [0,1].$$

**Non-degeneracy.** We assume throughout that no two systems coincide in $(a_i,b_i)$ — such a pair has no defined pairwise balance, both squared gaps vanishing, and resolves by merging the two systems. Nothing further is assumed globally: where a proof needs $\tilde a,\tilde b$ nonzero or linearly independent on a particular support, it either states that as a hypothesis (Proposition 1) or derives it (Lemma 3(a)).

$\alpha = \tfrac12$ is equalization; the raw WMT setup sits at some $\alpha_0 > \tfrac12$ (adequacy-heavy). Across systems $a$ and $b$ are positively correlated: better systems tend to be better at both. (The high concordance rates of Shayegh et al. (2025, Table 1) are a *rank-level symptom* of that positive association, not the second-moment covariance itself.) The All MQM variance decomposes as $\sigma(t;w)^2 = \sigma(a;w)^2 + \sigma(b;w)^2 + 2\,\sigma(a,b;w)$. Our $\alpha$ uses the two marginal variances and **ignores the cross term**, so it does *not* partition $\sigma(t;w)^2$. We **declare** this rather than patch it: $\alpha$ is a *marginal, magnitude-influence* estimand — the balance of each aspect's total marginal variance ("shake adequacy alone vs. fluency alone"), i.e. each component's magnitude contribution across all pairs (the SPA-appropriate notion of influence, §7). Covariance is genuinely irrelevant to a statement about marginals, and the contribution never rested on $\alpha$ being "the true bias," so declaring the estimand — not correcting for covariance — is the whole resolution. A covariance-aware *unique-influence* $\alpha$ is available as a robustness alternative (§8).

### 3.3 Why controlling row (system) variance controls pairwise-difference variance — the identity

For any single variable $X$ with normalized weights ($\sum_i w_i = 1$), summing over **unique** system pairs $i<j$:

$$\sum_{i<j} w_i w_j (X_i - X_j)^2 = \sigma(X;w)^2.$$

**Proof.** Using $\sum_i w_i = 1$, the sum over all ordered pairs is $\sum_{i,j} w_iw_j (X_i-X_j)^2 = 2\sum_i w_iX_i^2 - 2\big(\sum_i w_iX_i\big)^2 = 2\,\sigma(X;w)^2$. Splitting that same sum by the sign of $i-j$,

$$2\,\sigma(X;w)^2 = \sum_{i,j} w_iw_j(X_i-X_j)^2 = \sum_{i<j} w_iw_j(X_i-X_j)^2 + \sum_{j<i} w_iw_j(X_i-X_j)^2 + \sum_{i=j} w_iw_j(X_i-X_j)^2 = 2\sum_{i<j} w_iw_j(X_i-X_j)^2 + 0,$$

since the summand is symmetric under $i\leftrightarrow j$ (the first two sums are equal) and vanishes on the diagonal. Dividing by two gives the claim. $\quad\blacksquare$

**What this buys us.** The quantity that drives the meta-evaluation balance between the two components is the **pairwise-difference variance** — how much a component's system gaps $X_i - X_j$ spread across pairs, i.e. the pair-weighted mean squared gap

$$q(X;w) := \frac{\sum_{i<j} w_iw_j (X_i-X_j)^2}{\sum_{i<j} w_iw_j}.$$

By the identity its numerator is $\sigma(X;w)^2$, and its denominator does not depend on $X$, so $q(X;w) \propto \sigma(X;w)^2$ with the same constant for every $X$. Hence the pairwise-gap balance is exactly the row-variance balance,

$$\frac{q(a;w)}{q(a;w)+q(b;w)} = \frac{\sigma(a;w)^2}{\sigma(a;w)^2+\sigma(b;w)^2} = \alpha:$$

controlling the row variance controls the real driver. The identity is single-variable ($a$ and $b$ never multiply), so it is **covariance-proof**, and it collapses an $O(K^2)$ pairwise quantity into an $O(K)$ row computation.

### 3.4 Choosing the weights: the optimization problem

Among the infinitely many weightings that achieve a target balance, pick the one **closest to uniform** — this preserves the sample and maximizes effective sample size. We measure closeness to uniform by $\tfrac12\sum_i w_i^2$ (smallest at uniform $w_i = 1/K$, large when weight concentrates); minimizing it maximizes the **Tsallis-2 entropy** $1-\sum_i w_i^2$. This is exactly **chi-square-distance calibration**, the canonical survey-calibration objective — minimize $\tfrac12\sum_i w_i^2$ subject to moment constraints (Deville & Särndal, *JASA* 1992). Entropy balancing (Hainmueller 2012) is the same family under a KL divergence instead; we use the squared (chi-square) form because it is what yields the closed-form quadratic $g(w,\alpha)$ and hence the certificate machinery of §3.6–§3.7 — the choice is load-bearing. Because $\mathrm{ESS}(w) = 1/\sum_i w_i^2$ under $\sum_i w_i = 1$ (§5), **minimizing $\sum_i w_i^2$ is exactly maximizing ESS** — the objective is the reliability quantity we report.

Writing the balance target $\alpha$ of §3.2 as a constraint via $g(w,\alpha) := (1-\alpha)\sigma(a;w)^2 - \alpha \sigma(b;w)^2$ (so $g=0 \iff \alpha = \sigma(a;w)^2/(\sigma(a;w)^2+\sigma(b;w)^2)$), the problem is

$$(\mathrm P)\qquad \min_w\ \tfrac12\sum_i w_i^2 \quad\text{s.t.}\quad \sum_i w_i = 1,\ \ g(w,\alpha) = 0,\ \ w_i \ge 0.$$

### 3.5 Optimality conditions and the sign rule

**Lagrangian.** Assign a multiplier to each constraint of $(\mathrm P)$: $\lambda\in\mathbb R$ to the budget $\sum_i w_i = 1$, $\eta\in\mathbb R$ to the balance $g(w,\alpha)=0$, and $\gamma_i\ge0$ to each $w_i\ge0$. The Lagrangian is

$$\mathcal L(w,\eta,\lambda,\gamma) = \tfrac12\sum_i w_i^2 + \eta\,g(w,\alpha) + \lambda\Big(\sum_i w_i - 1\Big) - \sum_i \gamma_i\,w_i .$$

Differentiating $\mathcal L$ in $w_i$, the balance term contributes its partial derivative

$$d_i(w,\alpha) := \frac{\partial g}{\partial w_i} = (1-\alpha)\big(a_i^2 - 2\mu(a;w)\,a_i\big) - \alpha\big(b_i^2 - 2\mu(b;w)\,b_i\big),$$

which depends on $w$ only through $\mu(a;w),\mu(b;w)$; we keep that dependence throughout. The full gradient is therefore

$$\frac{\partial \mathcal L}{\partial w_i} = w_i + \eta\,d_i(w,\alpha) + \lambda - \gamma_i .$$

**KKT points.** Setting $\partial\mathcal L/\partial w_i = 0$ and pairing it with feasibility gives the **Karush–Kuhn–Tucker (KKT) conditions** (Boyd & Vandenberghe §5.5.3). A $w$ satisfying them is a **KKT point**, and these are the points worth isolating: a KKT point together with the curvature certificate (Corollary 1, Theorem 2) is *sufficient* to be a global optimum, with no constraint qualification, so any KKT point that passes the certificate is the answer. They also yield the sign rule of Lemma 1 and drive the support search of §3.11. Explicitly, a KKT point is a $w$ for which there exist multipliers $\eta,\lambda\in\mathbb R$ and $\gamma\in\mathbb R^K$ with

$$
\begin{aligned}
\text{(stationarity: }\partial\mathcal L/\partial w_i=0\text{)}\quad & w_i + \eta\,d_i(w,\alpha) + \lambda = \gamma_i && \text{for all } i,\\
\text{(primal feasibility)}\quad & \textstyle\sum_i w_i = 1,\ \ g(w,\alpha)=0,\ \ w_i\ge0,\\
\text{(dual feasibility)}\quad & \gamma_i \ge 0 && \text{for all } i,\\
\text{(complementary slackness)}\quad & \gamma_i\,w_i = 0 && \text{for all } i.
\end{aligned}
$$

The multiplier $\eta$ is the tilt amount: $\eta=0$ leaves the weights uniform, larger $|\eta|$ pushes harder to meet the balance target.

**Lemma 1 (sign rule).** At a KKT point of $(\mathrm P)$,

$$w_i = \max\!\big(0,\ -(\lambda + \eta\,d_i(w,\alpha))\big),$$

and a support is valid exactly when every dropped system $i$ (those with $w_i=0$) satisfies $\lambda + \eta\,d_i(w,\alpha)\ge0$.

**Proof.** Eliminate $\gamma_i$ using complementary slackness. If $w_i>0$, then $\gamma_i=0$, and stationarity gives $w_i = -(\lambda+\eta d_i(w,\alpha))>0$. If $w_i=0$, then stationarity gives $\gamma_i = \lambda+\eta d_i(w,\alpha)$, and dual feasibility $\gamma_i\ge0$ forces $\lambda+\eta d_i(w,\alpha)\ge0$. The two cases are exhaustive and mutually exclusive, giving the stated maximum and the dropped-system condition. $\quad\blacksquare$

So $\gamma_i$ is confined to this derivation; downstream only the conclusion is used. Once $\eta,\lambda$ (and the means they involve) are fixed, every weight is determined.

### 3.6 Each fixed-support solve is globally optimal

On a fixed support the weights solve the equality-only problem $\min_w \tfrac12\|w\|^2$ s.t. $\mathbf 1^\top w = 1$ and $g(w,\alpha)=0$; we need it to return the **global** minimum. Expanding the balance value gives a quadratic in $w$: with $\sigma(a;w)^2 = \sum_i w_i a_i^2 - w^\top(aa^\top)w$ (and likewise for $b$),

$$
\begin{aligned}
g(w,\alpha) &= (1-\alpha)\,\sigma(a;w)^2 - \alpha\,\sigma(b;w)^2\\
&= (1-\alpha)\Big(\textstyle\sum_i w_i a_i^2 - w^\top aa^\top w\Big) - \alpha\Big(\textstyle\sum_i w_i b_i^2 - w^\top bb^\top w\Big)\\
&= w^\top M(\alpha)\,w + r(\alpha)^\top w,
\end{aligned}
$$

where

$$M(\alpha) = \alpha\,bb^\top - (1-\alpha)\,aa^\top, \qquad r(\alpha)_i = (1-\alpha)a_i^2 - \alpha b_i^2.$$

$M(\alpha)$ is symmetric and a difference of two rank-one matrices (a positive direction along $b$, a negative one along $a$), hence **indefinite** when $\alpha \in (0,1)$ and $a, b$ are linearly independent — the regime Lemma 3(a) establishes wherever indefiniteness is used; otherwise, at $\alpha \in \{0,1\}$ or with $a \propto b$, it is rank $\le1$ and semidefinite of a single sign, and the certificate below covers these cases without change. In the indefinite regime the balance surface is curved but not convex, and a first-order stationary point need not be globally optimal. To certify the global optimum we will reduce this solve to the problem handled by a classical certificate.

**Theorem 1 (the equality-constrained analogue of the trust-region certificate; Moré & Sorensen 1983; Boyd & Vandenberghe §B.1).** Let $\mathfrak M$ be a symmetric matrix, $\mathfrak r$ a vector, and $\mathfrak c$ a scalar, and write the constraint as $q(\mathfrak u) = \mathfrak u^\top \mathfrak M\mathfrak u + \mathfrak r^\top\mathfrak u + \mathfrak c$. In

$$\min_{\mathfrak u}\ \tfrac12\,\mathfrak u^\top \mathfrak u \quad\text{s.t.}\quad q(\mathfrak u) = 0,$$

form the Lagrangian $\mathcal L(\mathfrak u) = \tfrac12\mathfrak u^\top\mathfrak u + \eta\,q(\mathfrak u)$. A feasible $\mathfrak u^\star$ is a **global minimizer** if there exists $\eta \in \mathbb R$ (of either sign) with

$$\nabla\mathcal L(\mathfrak u^\star) = \mathfrak u^\star + \eta\big(2\mathfrak M\,\mathfrak u^\star + \mathfrak r\big) = 0 \quad\text{(stationarity)} \qquad\text{and}\qquad \nabla^2\mathcal L = I + 2\eta\mathfrak M \succeq 0 \quad\text{(curvature)},$$

where $\mathfrak H\succeq0$ (positive semidefinite) means $v^\top \mathfrak H v \ge 0$ for every $v$. The two conditions are the first and second derivatives of the *same* Lagrangian — which is why the curvature carries the objective's $I$ next to the constraint's $2\eta\mathfrak M$. We use only this sufficiency direction; whether an optimum must admit such a $\eta$ is a separate question, settled in §3.9. When the curvature is strict, $I + 2\eta\mathfrak M \succ 0$, the minimizer is moreover **unique**.

The theorem governs a **single** quadratic constraint in a free variable; our solve has that quadratic plus the linear budget constraint. Removing the budget constraint lands us exactly in its setting.

**Reduction.** Parametrize the budget-feasible set by an affine map (standard null-space elimination; Nocedal & Wright, *Numerical Optimization* 2006, §16.1): fix $u_0$ as the uniform weighting (the unique budget-feasible point in $\operatorname{span}(\mathbf 1)$, so $N^\top u_0 = 0$), and any constant matrix $N \in \mathbb R^{K\times(K-1)}$ whose columns are orthonormal and span $\{v : \mathbf 1^\top v = 0\}$ (any valid $N$ gives the same optimum). Then $w = u_0 + Nu$ ranges bijectively over $\{w : \mathbf 1^\top w = 1\}$ as $u \in \mathbb R^{K-1}$ ranges freely, and since $N^\top N = I_{K-1}$ and $N^\top u_0 = 0$, the objective becomes $\tfrac12\|w\|^2 = \tfrac12\|u\|^2 + \text{const}$. Substituting the affine map into $g$ makes it a single quadratic in the free variable $u$,

$$g = u^\top \widetilde M(\alpha)\,u + \widetilde r(\alpha)^\top u + \widetilde c(\alpha), \qquad \widetilde M(\alpha) = N^\top M(\alpha)\,N,$$

with $\widetilde r(\alpha),\widetilde c(\alpha)$ likewise determined by $u_0$ and $\alpha$. This is exactly Theorem 1's problem, instantiated at $(\mathfrak u,\mathfrak M,\mathfrak r,\mathfrak c) = (u,\ \widetilde M(\alpha),\ \widetilde r(\alpha),\ \widetilde c(\alpha))$. Translating the certificate back through $w = u_0 + Nu$ gives the working result.

**Corollary 1.** For $\min_w \tfrac12 w^\top w$ s.t. $\mathbf 1^\top w = 1$ and $g(w,\alpha) = 0$: if there exist $\eta, \lambda$ with

$$w^\star + \eta\,d(w^\star,\alpha) + \lambda\mathbf 1 = 0 \quad\text{(stationarity)}, \qquad v^\top\big(I + 2\eta M(\alpha)\big)v \ge 0 \ \text{ for all } v \text{ with } \mathbf 1^\top v = 0 \quad\text{(restricted curvature)},$$

then the feasible $w^\star$ is a **global minimizer**, and the **unique** one when the curvature is strict ($v^\top(I + 2\eta M(\alpha))v > 0$ for every nonzero $v \perp \mathbf 1$). Here $d(w,\alpha) = 2M(\alpha)w + r(\alpha) = \nabla_w g$ is the gradient of $g$ — the vector whose entries are the $d_i(w,\alpha)$ of §3.5.

**Proof.** In the reduced problem $\min_u \tfrac12\|u\|^2$ s.t. the single quadratic of the Reduction, the objective has gradient $u$, and (by the chain rule, at $w = u_0 + Nu$) the constraint has gradient $N^\top d(w,\alpha)$. Theorem 1's stationarity at $u^\star$ is thus $u^\star + \eta\,N^\top d(w^\star,\alpha) = 0$. Applying $N^\top$ to $w^\star = u_0 + Nu^\star$ gives $N^\top w^\star = N^\top u_0 + N^\top N u^\star = u^\star$ (since $N^\top u_0 = 0$ and $N^\top N = I$), so the stationarity reads

$$N^\top\big[\,w^\star + \eta\,d(w^\star,\alpha)\big] = 0 .$$

The bracket is therefore orthogonal to $\operatorname{range}(N) = \{v : \mathbf 1^\top v = 0\}$, so it lies in the orthogonal complement $\operatorname{span}(\mathbf 1)$: $w^\star + \eta\,d(w^\star,\alpha) = -\lambda\mathbf 1$ for a scalar $\lambda$, the stated stationarity.

Theorem 1's curvature condition is $I + 2\eta\widetilde M(\alpha) \succeq 0$. Since $\widetilde M(\alpha) = N^\top M(\alpha)N$ and $\operatorname{range}(N) = \{v : \mathbf 1^\top v = 0\}$, this is the stated restricted form:

$$I + 2\eta\widetilde M(\alpha) \succeq 0 \iff v^\top\big(I + 2\eta M(\alpha)\big)v \ge 0 \ \text{ for all } v \text{ with } \mathbf 1^\top v = 0.$$

When the curvature is strict, the reduced Lagrangian $\tfrac12\|u\|^2 + \eta\,g$ is strictly convex, so the minimizer is unique (Theorem 1). $\quad\blacksquare$

The curvature condition of Corollary 1 is a quadratic form on the $(K{-}1)$-dimensional space $\{v : \mathbf 1^\top v = 0\}$; the following reduces it to a $2\times2$ test.

**Proposition 1 (the curvature condition is a $2\times2$ test).** Let $\tilde a = a - \mu(a)\,\mathbf 1$ and $\tilde b = b - \mu(b)\,\mathbf 1$ be the mean-centered score vectors (both nonzero), and let

$$\rho \;=\; \frac{\tilde a^\top \tilde b}{\|\tilde a\|\,\|\tilde b\|}$$

be the adequacy–fluency (Pearson) correlation across systems. If $|\rho| < 1$, the restricted-curvature condition $v^\top(I + 2\eta M(\alpha))v \ge 0$ on $\{v:\mathbf 1^\top v = 0\}$ (Corollary 1; equivalently Theorem 2's condition 3) holds **iff** $H(\eta,\alpha) \succeq 0$, where $H$ is the **symmetric** matrix

$$H(\eta,\alpha) = I_2 + 2\eta \begin{bmatrix} \alpha\|\tilde b\|^2 - (1-\alpha)\rho^2\|\tilde a\|^2 & -(1-\alpha)\,\rho\sqrt{1-\rho^2}\,\|\tilde a\|^2 \\[2pt] -(1-\alpha)\,\rho\sqrt{1-\rho^2}\,\|\tilde a\|^2 & -(1-\alpha)(1-\rho^2)\|\tilde a\|^2 \end{bmatrix},$$

so the test is the textbook one for a symmetric $2\times2$ matrix: $\operatorname{tr} H \ge 0$ and $\det H \ge 0$ (strict inequalities give $H\succ0$, i.e. strict curvature, hence a *unique* optimum). If $|\rho| = 1$ the span is one-dimensional and the test collapses to the scalar inequality $1 + 2\eta\big(\alpha\|\tilde b\|^2 - (1-\alpha)\|\tilde a\|^2\big) \ge 0$.

**Proof.** Restrict to $\{v : \mathbf 1^\top v = 0\}$. Since $a = \tilde a + \mu(a)\mathbf 1$ and $\mathbf 1^\top v = 0$, we have $a^\top v = \tilde a^\top v$, and likewise $b^\top v = \tilde b^\top v$. Hence, for such $v$,

$$v^\top\big(I + 2\eta M(\alpha)\big)v = \|v\|^2 + 2\eta\Big[\alpha\,(\tilde b^\top v)^2 - (1-\alpha)\,(\tilde a^\top v)^2\Big]. \qquad (\ast)$$

*Only $\operatorname{span}\{\tilde a,\tilde b\}$ matters.* Decompose $v = v_\parallel + v_\perp$, where $v_\parallel$ is the orthogonal projection of $v$ onto $\operatorname{span}\{\tilde a,\tilde b\}$ and $v_\perp \perp \tilde a,\tilde b$. In $(\ast)$ the inner products see only $v_\parallel$ (as $\tilde a^\top v_\perp = \tilde b^\top v_\perp = 0$), while $\|v\|^2 = \|v_\parallel\|^2 + \|v_\perp\|^2$, so $(\ast) = \|v_\perp\|^2 + Q(v_\parallel)$, where $Q$ is the same form restricted to the span. Hence $(\ast)\ge0$ for all budget-preserving $v$ iff $Q\ge0$ on $\operatorname{span}\{\tilde a,\tilde b\}$: sufficiency because $\|v_\perp\|^2\ge0$, necessity by taking $v_\perp = 0$ (the span sits inside $\mathbf 1^\perp$, since $\tilde a,\tilde b\perp\mathbf 1$, so such $v$ are admissible).

*An orthonormal basis of that span.* Gram–Schmidt, taking $\tilde b$ first:

$$e_1 = \frac{\tilde b}{\|\tilde b\|}, \qquad e_2 = \frac{\tilde a - \rho\,\|\tilde a\|\,e_1}{\|\tilde a\|\sqrt{1-\rho^2}},$$

well defined because $\tilde a^\top e_1 = \rho\|\tilde a\|$ and $\|\tilde a - \rho\|\tilde a\|e_1\| = \|\tilde a\|\sqrt{1-\rho^2} \neq 0$ when $|\rho|<1$.

*The matrix.* Write $v_\parallel = x\,e_1 + y\,e_2$. Then $\|v_\parallel\|^2 = x^2+y^2$ (the basis is orthonormal), $\tilde b^\top v_\parallel = \|\tilde b\|\,x$, and $\tilde a^\top v_\parallel = \rho\|\tilde a\|\,x + \|\tilde a\|\sqrt{1-\rho^2}\,y$. Substituting into $(\ast)$ and collecting terms in $x^2, xy, y^2$ gives exactly $[x\ y]\,H(\eta,\alpha)\,[x\ y]^\top$. Because $H$ is symmetric, "the form is nonnegative" *is* $H \succeq 0$, which for a symmetric $2\times2$ matrix holds iff $\operatorname{tr} H \ge 0$ and $\det H \ge 0$.

*Degenerate case.* If $|\rho| = 1$ then $\tilde a = \pm\|\tilde a\|\,e_1$, the span is the line $\mathbb R e_1$, and $(\ast)$ at $v_\parallel = x\,e_1$ reads $x^2\big[1 + 2\eta(\alpha\|\tilde b\|^2 - (1-\alpha)\|\tilde a\|^2)\big]$, giving the stated scalar test. $\quad\blacksquare$

Therefore, for each fixed support, we obtain the global optimum by finding a point that meets this certificate — stationarity (Corollary 1) together with the curvature test just given, with $M(\alpha), r(\alpha)$ restricted to the support.

### 3.7 The support certificate

Corollary 1 settles the equality-only problem on a fixed support. Adding $w \ge 0$:

**Theorem 2 (support certificate).** Fix a support $S$, and let $w^\star$ (supported on $S$) with multipliers $\eta^\star,\lambda^\star$ satisfy (0) *stationarity* $w^\star + \eta^\star d(w^\star,\alpha) + \lambda^\star\mathbf 1 = 0$ on $S$ together with the equality constraints $\mathbf 1^\top w^\star = 1,\ g(w^\star,\alpha)=0$; (1) $w^\star_i \ge 0$ for all $i \in S$; (2) $\lambda^\star + \eta^\star d_i(w^\star,\alpha) \ge 0$ for all $i \notin S$; and (3) $v^\top(I + 2\eta^\star M(\alpha))v \ge 0$ for every $v$ with $\mathbf 1^\top v = 0$. Then $w^\star$ (with $w^\star_i = 0$ for $i \notin S$) is a **global** optimum of $(\mathrm P)$ — the **unique** one when condition (3) holds strictly.

**Proof.** With the dropped zeros, conditions (0)–(2) make $w^\star$ a KKT point of the full $(\mathrm P)$: stationarity and feasibility hold, and by Lemma 1 each dropped $i\notin S$ carries $\gamma_i = \lambda^\star+\eta^\star d_i(w^\star,\alpha)\ge0$, so dual feasibility and complementary slackness hold. Condition (3) adds the restricted curvature. A KKT point plus this curvature condition are sufficient for global optimality of $(\mathrm P)$ (Corollary 1's certificate, via Theorem 1; Boyd & Vandenberghe §5.5.3); when the curvature is strict it is the unique optimum (Corollary 1). $\quad\blacksquare$

**Soundness, not completeness.** Theorem 2 is a *sufficient* condition and nothing more: a point passing (0)–(3) is a global optimum, unconditionally and with no constraint qualification. The converse is false — **an optimum need not pass**. Condition (3) is the S-procedure test for being the global minimizer of the *sign-free* subproblem on $S$; it is blind to $w\ge0$. When the sign-free minimizer leaves the simplex, the true optimum is a *secondary branch* of the same quadric, and (3) provably cannot hold there.

A concrete instance ($K=4$): $a = (0.5574,\ 0.9883,\ 1.4687,\ 3.0894)$, $b = (0.4012,\ 0.5353,\ 1.4986,\ 1.2467)$, reachable range $[0.199,\ 0.976]$, target $\alpha = 0.9095$. The global optimum is $w^\star = (0.0088,\ 0.0861,\ 0.4245,\ 0.4806)$ — full support, $\tfrac12\|w^\star\|^2 = 0.2093$ — while the sign-free minimizer on that same support is $\hat w = (0.3534,\ 0.3521,\ -0.0248,\ 0.3193)$ with $\tfrac12\|\hat w\|^2 = 0.1757$. Since every $w^\star_i > 0$, no sign constraint is active, so stationarity fixes the multiplier **uniquely** at $\eta = -0.878$; the curvature window $\{\eta : I + 2\eta M(\alpha)\succeq0\}$ is $[-0.752,\ 2.313]$. There is no freedom to choose a friendlier multiplier, so condition (3) fails and **no Lagrangian certificate can certify $w^\star$**. This is a genuine duality gap, not a gap in the argument.

We therefore claim only soundness for Theorem 2, and obtain completeness by a different route: enumeration (§3.9).

### 3.8 Which supports can carry the answer

Write

$$\alpha_{ij} = \frac{(a_i-a_j)^2}{(a_i-a_j)^2 + (b_i-b_j)^2}$$

for the **pairwise balance** of systems $i,j$: by the identity of §3.3 it is the balance of the two-system pool $\{i,j\}$, whatever the split of weight between them. For a support $S$, let $\alpha_{\min}(S)$ and $\alpha_{\max}(S)$ be the smallest and largest pairwise balance among pairs drawn from $S$, and write $\alpha_{\min} = \alpha_{\min}(\{1,\dots,K\})$, $\alpha_{\max} = \alpha_{\max}(\{1,\dots,K\})$.

**Proposition 2 (support admissibility).** Let $|S|\ge2$. The set of targets $\alpha$ attainable by a weighting with $\operatorname{supp}(w) = S$ *exactly* is

$$\begin{cases}\big(\alpha_{\min}(S),\ \alpha_{\max}(S)\big) \quad\text{(open)}, & \text{if the pairwise balances within } S \text{ are not all equal},\\[2pt] \{\alpha_{ij}\}, & \text{if they are all equal to a common } \alpha_{ij}.\end{cases}$$

**Proof.** *Necessity.* By the §3.3 identity, for any $w$ with at least two active systems,

$$\alpha(w) = \frac{\sum_{i<j} w_iw_j(a_i-a_j)^2}{\sum_{i<j} w_iw_j\big[(a_i-a_j)^2+(b_i-b_j)^2\big]} = \sum_{i<j} p_{ij}\,\alpha_{ij}, \qquad p_{ij} = \frac{w_iw_j\big[(a_i-a_j)^2+(b_i-b_j)^2\big]}{\sum_{k<l} w_kw_l\big[(a_k-a_l)^2+(b_k-b_l)^2\big]},$$

with $p_{ij}\ge0$ summing to $1$. If $\operatorname{supp}(w) = S$ then $w_iw_j>0$ for every pair within $S$, so every such $p_{ij}$ is **strictly** positive and $\alpha(w)$ is a *strict* convex combination of those pairwise balances — hence strictly between the smallest and the largest, unless they all coincide, in which case it equals their common value.

*Sufficiency.* $\alpha(\cdot)$ is continuous on the relative interior of $S$'s simplex face (the denominator is strictly positive there), and that face is connected, so the image is an interval, contained in $(\alpha_{\min}(S),\alpha_{\max}(S))$ by necessity. Placing almost all weight on a pair attaining $\alpha_{\min}(S)$ and a vanishing share on the rest of $S$ drives $\alpha(w) \to \alpha_{\min}(S)$; symmetrically for $\alpha_{\max}(S)$. So the image is an interval whose infimum and supremum are those two values and which omits both — exactly the open interval. $\quad\blacksquare$

Two edge cases read straight off. A **two-system support** has a single pair, so its balances are trivially all equal and it attains exactly $\{\alpha_{ij}\}$. And a support whose balances are all equal is one whose centered score vectors are proportional: there $\alpha(\cdot)$ is constant, every weighting on the face is feasible at that one target, and the min-norm feasible point is the uniform one.

**Corollary 2 (reachable range).** $(\mathrm P)$ is feasible exactly for $\alpha \in [\alpha_{\min}, \alpha_{\max}]$.

**Proof.** Apply Proposition 2 to the full support, whose pairs are all pairs: it attains $(\alpha_{\min},\alpha_{\max})$. Apply Proposition 2 to a two-system support $\{i,j\}$ with $\alpha_{ij} = \alpha_{\min}$: it attains $\{\alpha_{\min}\}$; likewise a maximizing pair attains $\{\alpha_{\max}\}$. So all of $[\alpha_{\min},\alpha_{\max}]$ is reachable. Conversely any $w$ with support $S$ attains a value in $[\alpha_{\min}(S),\alpha_{\max}(S)] \subseteq [\alpha_{\min},\alpha_{\max}]$, since $S$'s pairs are among all pairs. $\quad\blacksquare$

**Using admissibility to prune the search.** The Proposition says exactly which supports can carry the answer at a given target: $S$ is **admissible** at $\alpha$ iff $\alpha_{\min}(S) < \alpha < \alpha_{\max}(S)$, or all of $S$'s pairwise balances equal $\alpha$. A search need never visit an inadmissible support.

A second prune, independent of the first, comes from the objective rather than the constraint.

**Lemma 2 (support size caps ESS).** For any $w$ with $\operatorname{supp}(w) \subseteq S$, $\mathrm{ESS}(w) \le |S|$, with equality exactly when $w$ is uniform on $S$.

**Proof.** Apply Cauchy–Schwarz to the vectors $(w_i)_{i\in S}$ and $(1)_{i\in S}$:

$$\Big(\sum_{i\in S} w_i\cdot 1\Big)^2 \;\le\; \Big(\sum_{i\in S} w_i^2\Big)\Big(\sum_{i\in S} 1^2\Big) \;=\; |S|\sum_{i\in S} w_i^2 .$$

The left side is $1$ by the budget, and $\sum_{i\in S} w_i^2 = \sum_i w_i^2$ since $w$ vanishes off $S$. So $\sum_i w_i^2 \ge 1/|S|$, i.e. $\mathrm{ESS}(w) = 1/\sum_i w_i^2 \le |S|$. Equality in Cauchy–Schwarz holds iff the two vectors are proportional — $w_i$ constant on $S$ — which with the budget is the uniform weighting. $\quad\blacksquare$

**Corollary 3 (size pruning).** Suppose a search holds a feasible incumbent of effective sample size $e$. Then no support $S$ with $|S| \le e$ carries a feasible point of strictly larger ESS, so such $S$ may be skipped; visiting supports largest-first, the search never descends below level $\lfloor e \rfloor + 1$.

**Proof.** Any feasible $w'$ on $S$ has $\mathrm{ESS}(w') \le |S| \le e$ by Lemma 2, so it cannot improve on the incumbent. Nor is the optimum ever lost: writing $w^\star$ for a global optimum with support $S^\star$, Lemma 2 gives $|S^\star| \ge \mathrm{ESS}(w^\star) \ge e$, the incumbent's ESS bounding the best from below. So $S^\star$ is skipped only when $|S^\star| = e$ — and then $\mathrm{ESS}(w^\star) = e$, the incumbent already attains the optimum, and returning it is correct. (This is why the test reads $|S| \le e$ rather than $|S| < e$: a support of size exactly $e$ can only tie.) $\quad\blacksquare$

The two prunes are strongest in opposite places. Near the natural balance $\alpha_0$ (§3.2) the optimum stays close to uniform, so ESS runs high and Corollary 3 cuts hardest — precisely where admissibility is weakest, almost every support straddling a central target. Toward the ends of the range they swap: ESS falls toward $2$ and size pruning idles, while admissibility narrows to the supports carrying an extremal pair. We report what each removes in §6.6.

### 3.9 Completeness by enumeration

**Corollary 4 (the optimum is stationary on its own support).** Let $\alpha$ be in range, let $w^\star$ be a global optimum of $(\mathrm P)$, and let $S = \operatorname{supp}(w^\star)$. Then either

1. **(stationary)** there are $\eta,\lambda$ with $w^\star + \eta\,d(w^\star,\alpha) + \lambda\mathbf 1 = 0$ on $S$; or
2. **(uniform)** $w^\star$ is the uniform weighting on $S$.

**Proof.** Because $w^\star_i>0$ for every $i\in S$, the sign constraints on $S$ are inactive, so $w^\star$ locally minimizes $\tfrac12\|w\|^2$ subject to the two equalities $\mathbf 1^\top w = 1$ and $g(w,\alpha)=0$ on $S$'s face. If the constraint gradients $d(w^\star,\alpha)$ and $\mathbf 1$, restricted to $S$, are linearly independent, then LICQ holds and stationarity gives (1).

Otherwise $d(w^\star,\alpha)$ is a multiple of $\mathbf 1$ on $S$; in the reduced variable $u$ of §3.6 ($w = u_0 + Nu$), this says the gradient of $g$ vanishes at $u^\star$. Together with feasibility $g=0$, the expansion is then exact: $g(u^\star + t\delta) = t^2\,\delta^\top \widetilde M(\alpha)\,\delta$. So every direction $\delta$ with $\delta^\top \widetilde M(\alpha)\,\delta = 0$ generates a line through $u^\star$ lying entirely in the feasible set, and min-norm optimality forces $u^{\star\top}\delta = 0$ for all of them. If $S$'s pairwise balances are not all equal, Proposition 2 puts $\alpha$ strictly inside $(\alpha_{\min}(S),\alpha_{\max}(S)) \subseteq [0,1]$, so $0<\alpha<1$ and $\widetilde M(\alpha)$ is indefinite of rank 2; its null directions then span the whole reduced space, forcing $u^\star = 0$ — case (2). If they are all equal, Proposition 2 says the entire face is feasible, and its min-norm point is again the uniform one — case (2). $\quad\blacksquare$

**Remark (completeness).** Collect, for every admissible $S$, the points of $S$'s face satisfying (1) or (2), and keep those that are feasible with support exactly $S$. By the Corollary this set contains every global optimum, and since all its members are feasible, the min-norm member of all these candidates across all supports is the global optimum.

### 3.10 Counting the candidates

§3.9 reduces the optimum to the points of each admissible face satisfying Corollary 4's (1) or (2). For that to be an algorithm the collection must be **finite**, and its members **constructible**; one count delivers both.

**Setup.** Fix an admissible support $S$ and restrict $a, b, M(\alpha), r(\alpha), g$ to it. §3.6's reduction needs a particular point $u_0$ on the budget hyperplane; take it to be the **uniform weighting on $S$** — already the point named in Corollary 4's case (2). Write $\widetilde g(u)$ for $g$ in the reduced variable. One line of the reduction then does most of the work below:

$$\widetilde c(\alpha) \;=\; g(u_0,\alpha),$$

the constant of the reduced balance quadratic *is* the balance constraint read at the uniform weighting.

**The anchor of the support.** Read that constant as an equation in $\alpha$. It has exactly one root,

$$\alpha_0(S) \;=\; \frac{\sigma(a;u_0)^2}{\sigma(a;u_0)^2 + \sigma(b;u_0)^2},$$

$S$'s **natural balance** — the balance $S$ already realizes with no reweighting at all. This is §3.2's $\alpha_0$ read per support: the one target on $S$ needing no tilt ($\eta = 0$, §3.5), and the one where $\mathrm{ESS} = |S|$ exactly. So every support has exactly one distinguished target, and $\widetilde c(\alpha) \ne 0$ at all the others.

**The route.** The proof sorts the candidates by the multiplier each carries. The anchor $\alpha = \alpha_0(S)$ is settled on its own; at every other target $u_0$ is infeasible, Corollary 4's case (2) cannot occur, and every candidate is stationary — pinned by a system that is *linear* once $\eta$ is fixed, and singular only at the two **poles** where the rank-2 structure of $\widetilde M(\alpha)$ makes $I + 2\eta\widetilde M(\alpha)$ singular. A singular square system is inconsistent or underdetermined; off the poles it solves uniquely, and there $\eta$ must root a polynomial of degree $\le5$. The count then closes because a pole that turns out underdetermined costs that polynomial exactly the two degrees it buys.

**Lemma 3 (candidate count).** Let $\alpha$ be admissible for $S$.

1. **(the anchor)** If $\alpha = \alpha_0(S)$, then $u_0$ is the optimum of $(\mathrm P)$ on $S$'s face. $S$ carries the single candidate $u_0$.
2. **(every other target)** If $\alpha \ne \alpha_0(S)$, then Corollary 4's case (2) cannot occur on $S$, and the points of $S$'s face satisfying its case (1) number **at most five**. Each is closed-form in the multiplier it carries, and the multipliers are cut out by an explicit polynomial $\Phi_{S,\alpha}$ of degree $\le5$.

Either way $S$ carries **at most five** candidates, and they cost one rootfind in one variable.

**Proof.** Four cases, in the order below. Each may assume the earlier ones do not apply, and together they exhaust the candidates on $S$.

**Case 1: $\alpha = \alpha_0(S)$ — exactly one candidate.** $u_0$ uniquely minimizes $\tfrac12\sum_{i\in S}w_i^2$ over the whole budget hyperplane $\{w \text{ on } S : \mathbf 1^\top w = 1\}$ — a strictly convex objective on an affine set — and it lies in the simplex. At $\alpha = \alpha_0(S)$ it is also feasible, so it minimizes over the smaller feasible set, and no competitor ties it. At the anchor $u_0$ is proven optimal, so no enumeration is needed — other stationary points may exist, but the candidate list can be truncated to $\{u_0\}$ without loss.

*Setting up cases 2–4.* From here $\alpha \ne \alpha_0(S)$, so $\widetilde c(\alpha) = g(u_0,\alpha) \ne 0$: the uniform weighting is infeasible, Corollary 4's case (2) cannot occur, and **every candidate is a stationary point**. Four preliminaries.

**(a) The geometry of $S$ at this target.** Cases 2–4 need two facts — that $\alpha$ is strictly interior to $[0,1]$, and that $\tilde a,\tilde b$ are linearly independent. Both follow from $\alpha \ne \alpha_0(S)$. *First, $S$'s pairwise balances are not all equal:* were they all equal to some $\alpha_{ij}$, admissibility (§3.8) would force $\alpha = \alpha_{ij}$, and at that target Proposition 2 makes *every* weighting on $S$'s face feasible — $u_0$ among them — giving $g(u_0,\alpha) = 0$, i.e. $\alpha = \alpha_0(S)$, contrary to hypothesis. *Hence $0 < \alpha < 1$:* with the balances not all equal, Proposition 2 puts $\alpha$ strictly inside $(\alpha_{\min}(S),\alpha_{\max}(S)) \subseteq [0,1]$. *Hence $\tilde a,\tilde b$ are independent:* by §3.8's edge-case reading, a support whose balances are all equal is exactly one whose centered score vectors are proportional; ours are not all equal, so $\tilde a,\tilde b$ are not proportional — both nonzero, and $|\rho| < 1$ (equality in Cauchy–Schwarz *is* proportionality), which is §3.6's hypothesis and gives $\operatorname{span}\{\tilde a,\tilde b\}$ dimension 2.

**(b) Stationarity carries one scalar unknown.** By the proof of Corollary 1, condition (1) of Corollary 4 reads, in the reduced variable, $u + \eta\big(2\widetilde M(\alpha)u + \widetilde r(\alpha)\big) = 0$. So the candidates are exactly the $u$ admitting some $\eta\in\mathbb R$ with

$$\big(I + 2\eta\widetilde M(\alpha)\big)\,u \;=\; -\eta\,\widetilde r(\alpha) \qquad\text{and}\qquad \widetilde g(u) = 0. \tag{$\dagger$}$$

Each candidate carries **exactly one** such $\eta$: if $2\widetilde M(\alpha)u + \widetilde r(\alpha) = 0$ then $(\dagger)$ forces $u = 0$ and hence $\widetilde c(\alpha) = \widetilde g(0) = 0$, excluded — so $\eta$ is pinned by any coordinate on which $2\widetilde M(\alpha)u + \widetilde r(\alpha)$ is nonzero.

**(c) The spectrum of $\widetilde M(\alpha)$, in closed form.** This caps the degree in Case 4 and locates the poles. It is Proposition 1, reread: on $\{v : \mathbf 1^\top v = 0\}$ the form $v^\top M(\alpha)v$ sees only $\operatorname{span}\{\tilde a,\tilde b\}$, and in the orthonormal basis $e_1,e_2$ built there its matrix is the $2\times2$ block of $H(\eta,\alpha) = I_2 + 2\eta\cdot(\text{block})$. By (a) that span is 2-dimensional, so $\widetilde M(\alpha)$ has **rank exactly 2**, its remaining eigenvalues all $0$. Its two nonzero eigenvalues $\theta_1,\theta_2$ are the block's, and expanding the §3.6 display that block has trace and determinant

$$T \;=\; \alpha\|\tilde b\|^2 - (1-\alpha)\|\tilde a\|^2, \qquad D \;=\; -\,\alpha(1-\alpha)\,(1-\rho^2)\,\|\tilde a\|^2\|\tilde b\|^2 ,$$

so $\theta_{1,2} = \tfrac12\big(T \pm \sqrt{T^2-4D}\big)$. Every factor of $D$ is strictly positive by (a), so $D < 0$: the square root is real, and $\theta_1\theta_2 = D < 0$, giving

$$\theta_1 \;>\; 0 \;>\; \theta_2 ,$$

i.e. $\widetilde M(\alpha)$ is indefinite. We write $\theta_k$ for either one, $k = 1,2$, and $v_1,v_2$ for unit eigenvectors. These lie in $\operatorname{span}\{\tilde a,\tilde b\}$, and diagonalizing the $2\times2$ block gives them explicitly: writing it $\begin{pmatrix}p & q\\ q & s\end{pmatrix}$, an eigenvector for $\theta_k$ has $(e_1,e_2)$-coordinates $(q,\ \theta_k - p)$, normalized. (If $q = 0$ — exactly when $\rho = 0$ — the block is already diagonal and $v_k$ is $e_1$ or $e_2$.) Finally, since $H(\eta,\alpha) = I_2 + 2\eta\cdot(\text{block})$,

$$\det H(\eta,\alpha) \;=\; (1 + 2\eta\theta_1)(1 + 2\eta\theta_2),$$

which vanishes at exactly two values of $\eta$ — the **poles** $\lambda_1 = -\tfrac1{2\theta_1} < 0 < \lambda_2 = -\tfrac1{2\theta_2}$.

**(d) $(\dagger)$ in coordinates.** $\widetilde M(\alpha) = N^\top M(\alpha)N$ is symmetric, so the spectral theorem supplies an orthonormal **basis** $v_1,\dots,v_{|S|-1}$ of the reduced space made of its eigenvectors. Take $v_1,v_2$ from (c) and $v_3,\dots,v_{|S|-1}$ any orthonormal basis of $\ker\widetilde M(\alpha)$; then $\theta_1 > 0 > \theta_2$ and $\theta_j = 0$ for $j\ge3$. Write $u_j = v_j^\top u$ and $\widetilde r_j = v_j^\top\widetilde r(\alpha)$.

**Index convention**, used to the end of the proof: $j$ ranges over **all** eigendirections; $j\in\{1,2\}$ are the two **curved** directions and $j\ge3$ the **flat** ones (spanning $\ker\widetilde M(\alpha)$); $k$ always denotes a curved index, and $k'$ the other one, so $\{k,k'\} = \{1,2\}$.

*The linear system, diagonalized.* Apply $v_j^\top$ to $(\dagger)$'s system. Symmetry moves $\widetilde M(\alpha)$ onto $v_j$, giving $v_j^\top(I + 2\eta\widetilde M(\alpha))u = u_j + 2\eta\theta_j u_j$, while the right side is $-\eta\widetilde r_j$. Testing against a basis loses nothing, so $(\dagger)$'s system is equivalent to the **diagonal** system

$$(1 + 2\eta\theta_j)\,u_j \;=\; -\eta\,\widetilde r_j, \qquad j = 1,\dots,|S|-1. \tag{$\ddagger$}$$

*The quadratic, diagonalized.* Substituting $u = \sum_j u_jv_j$ into $\widetilde g$ and collapsing both sums by orthonormality,

$$\widetilde g(u) \;=\; \sum_j \theta_j u_j^2 \;+\; \sum_j \widetilde r_j u_j \;+\; \widetilde c(\alpha). \tag{$\S$}$$

Cases 2–4 are read off $(\ddagger)$ and $(\S)$ and nothing else. The $j$-th equation of $(\ddagger)$ can only be singular for $j = 1,2$: for $j \ge 3$ its coefficient is $1 + 2\eta\cdot0 = 1$. So $(\ddagger)$ is singular exactly at the two poles of (c).

*One scalar for the flat directions.* The same orthonormality computation run with $\widetilde r(\alpha)$ in both slots is Parseval's identity, $\|\widetilde r(\alpha)\|^2 = \sum_j \widetilde r_j^2$. Setting

$$\widetilde r_0^2 \;=\; \sum_{j\ge3}\widetilde r_j^2 \;=\; \|\widetilde r(\alpha)\|^2 - \widetilde r_1^2 - \widetilde r_2^2 ,$$

$\widetilde r_1,\widetilde r_2$ are $\widetilde r(\alpha)$'s coordinates where $\widetilde g$ curves and $\widetilde r_0^2$ its squared length where $\widetilde g$ is flat. The right-hand expression involves no $v_j$ with $j\ge3$, so $\widetilde r_0^2$ does not depend on which basis of the kernel we chose. (Each of $v_1,v_2$ is fixed only up to sign, the eigenspaces being one-dimensional as $\theta_1\ne\theta_2$. Nothing below notices: $v_k$ enters only through $\widetilde r_k^2$, through the product $\widetilde r_kv_k$, or as the direction of a line.)

*Sorting the candidates.* By (b) every candidate carries exactly one $\eta$, so sort them by it. The question is whether $(\ddagger)$ is singular — by (d), exactly whether $\eta$ is a pole.

**Case 2: $\eta = \lambda_k$ is a pole, and $\widetilde r_k \ne 0$ — no candidates.** The $k$-th equation of $(\ddagger)$ has coefficient $1 + 2\lambda_k\theta_k = 0$, so it reads $0\cdot u_k = -\lambda_k\widetilde r_k$. Here $\lambda_k \ne 0$ and $\widetilde r_k \ne 0$, so the right side is nonzero: the system is **inconsistent**. No stationary point carries this multiplier.

**Case 3: $\eta = \lambda_k$ is a pole, and $\widetilde r_k = 0$ — at most two candidates.** The $k$-th equation now reads $0 = 0$: that coordinate is **free**. Every other equation determines its own coordinate, its coefficient being nonzero — for $j\ge3$ it equals $1$, and for $j = k'$ substituting $\lambda_k = -1/(2\theta_k)$ gives $1 + 2\lambda_k\theta_{k'} = 1 - \theta_{k'}/\theta_k > 1$, since $\theta_1 > 0 > \theta_2$ makes the ratio negative. So the solutions form a line,

$$u \;=\; u^\circ + u_k\,v_k, \qquad u^\circ \;:=\; \sum_{j \ne k} \frac{-\lambda_k\,\widetilde r_j}{1 + 2\lambda_k\theta_j}\;v_j, \qquad u_k \in \mathbb R ,$$

with $u^\circ \perp v_k$, so $u^\circ_k = 0$. Splitting $(\S)$'s sums at $j = k$ and using $u_j = u^\circ_j$ for $j\ne k$, the terms over $j \ne k$ reassemble into $\widetilde g(u^\circ)$ (restoring the $j=k$ terms adds nothing, as $u^\circ_k = 0$), while the $j = k$ terms are $\theta_ku_k^2 + \widetilde r_ku_k = \theta_ku_k^2$ — the case hypothesis $\widetilde r_k = 0$ doing its work. Feasibility $\widetilde g = 0$ therefore reads

$$\theta_k\,u_k^2 \;=\; -\,\widetilde g(u^\circ), \qquad\text{i.e.}\qquad u_k^2 \;=\; -\,\widetilde g(u^\circ)\big/\theta_k ,$$

legitimate as $\theta_k \ne 0$. A quadratic in $u_k$ with no linear term: **at most two candidates at this pole** — $u_k = \pm\sqrt{-\widetilde g(u^\circ)/\theta_k}$ when that quantity is positive, the single point $u_k = 0$ when it vanishes, none when it is negative. Nothing above fixed *which* curved index is at issue, so if $\widetilde r_1 = \widetilde r_2 = 0$ this applies at both poles: at most $2+2 = 4$ between them.

**Case 4: $\eta$ is not a pole — at most five candidates in total.** Every coefficient in $(\ddagger)$ is now nonzero, so it has the unique solution $u_k(\eta) = -\eta\widetilde r_k/(1 + 2\eta\theta_k)$ for $k = 1,2$ and $u_j(\eta) = -\eta\widetilde r_j$ for $j\ge3$: **at most one candidate carries this $\eta$.** Which $\eta$ can occur? Substitute into $(\S)$. The flat directions contribute $0 + \sum_{j\ge3}\widetilde r_j(-\eta\widetilde r_j) = -\eta\widetilde r_0^2$. Each curved direction contributes

$$\theta_k u_k(\eta)^2 + \widetilde r_k u_k(\eta) \;=\; \widetilde r_k^2\,\frac{\theta_k\eta^2 - \eta(1+2\eta\theta_k)}{(1+2\eta\theta_k)^2} \;=\; -\,\eta\,\frac{\widetilde r_k^2\,(1+\theta_k\eta)}{(1+2\eta\theta_k)^2} .$$

Adding both groups and $\widetilde c(\alpha)$, feasibility $\widetilde g(u(\eta)) = 0$ reads

$$\widetilde c(\alpha) \;-\; \eta\left[\;\sum_{k=1,2} \frac{\widetilde r_k^2\,(1+\theta_k\eta)}{(1+2\eta\theta_k)^2} \;+\; \widetilde r_0^2\;\right] \;=\; 0 .$$

Its only denominators are $(1+2\eta\theta_1)^2$ and $(1+2\eta\theta_2)^2$, whose product is $\det H(\eta,\alpha)^2$. Multiplying through by it is legitimate off the poles and changes no solutions there, and the equation becomes polynomial, $\Phi_{S,\alpha}(\eta) = 0$, with

$$\Phi_{S,\alpha}(\eta) \;=\; \widetilde c(\alpha)\,\det H^2 \;-\; \eta\Big[\,\widetilde r_1^2(1+\theta_1\eta)(1+2\eta\theta_2)^2 \;+\; \widetilde r_2^2(1+\theta_2\eta)(1+2\eta\theta_1)^2 \;+\; \widetilde r_0^2\,\det H^2\,\Big]$$

the **secular polynomial** of $S$ at $\alpha$. Its four terms have degrees $4,4,4,5$, so $\deg\Phi_{S,\alpha} \le 5$. And $\Phi_{S,\alpha}(0) = \widetilde c(\alpha) \ne 0$, so $\Phi_{S,\alpha}\not\equiv0$ and it has at most $\deg\Phi_{S,\alpha}$ roots — *this is the whole force of Case 1's absence*: the uniform point being infeasible is exactly what keeps the polynomial from vanishing identically. Each Case-4 candidate therefore has a distinct non-pole root of $\Phi_{S,\alpha}$ as its multiplier: **at most $\deg\Phi_{S,\alpha}\le5$ of them.**

*Adding up.* Call pole $k$ **dead** when $\widetilde r_k \ne 0$ (Case 2: no candidates) and **live** when $\widetilde r_k = 0$ (Case 3: at most two). Every pole is one or the other.

*Step 1: a live pole is a double root of $\Phi_{S,\alpha}$.* Ask which terms of $\Phi_{S,\alpha}$ are divisible by $(1+2\eta\theta_k)^2$. The first and last are, since $\det H^2$ carries **both** poles' factors; so is the $\widetilde r_{k'}^2$ term, which carries that factor explicitly. Exactly one term is not: the $\widetilde r_k^2$ one, which carries only the *other* pole's factor — and pole $k$ being live means precisely $\widetilde r_k = 0$, deleting it. So every surviving term is divisible, hence so is $\Phi_{S,\alpha}$, and $\lambda_k$ is a root of multiplicity $\ge2$.

*Step 2: a dead pole is not a root at all.* If $\widetilde r_k \ne 0$, substituting $\lambda_k = -1/(2\theta_k)$ leaves only the $\widetilde r_k^2$ term, giving

$$\Phi_{S,\alpha}(\lambda_k) \;=\; \frac{\widetilde r_k^2\,(\theta_k - \theta_{k'})^2}{4\,\theta_k^3} \;\ne\; 0 ,$$

nonzero because $\theta_k \ne 0$ and $\theta_1 \ne \theta_2$, both from (c). So a dead pole consumes none of $\Phi_{S,\alpha}$'s roots.

*Step 3: count the roots.* $\Phi_{S,\alpha}\not\equiv0$, so it has at most $\deg\Phi_{S,\alpha}\le5$ roots with multiplicity. Case 4 spends one candidate per **non-pole** root; each live pole consumes two more (Step 1), at a $\eta$ distinct from every non-pole root and — since $\lambda_1\ne\lambda_2$ — from the other pole. So

$$\#\{\text{Case 4 candidates}\} \;\le\; \deg\Phi_{S,\alpha} - 2\cdot\#\{\text{live poles}\} \;\le\; 5 - 2\cdot\#\{\text{live poles}\}.$$

*Step 4: the ledger balances.* Each live pole costs Case 4 two roots and pays Case 3 at most two candidates. **Two off, two on** — so no configuration exceeds five:

| | Case 4 (off the poles) | at $\lambda_1$ | at $\lambda_2$ | total |
|---|:---:|:---:|:---:|:---:|
| $\widetilde r_1, \widetilde r_2 \ne 0$ | $\le 5 - 0$ | $0$ (Case 2) | $0$ (Case 2) | $\le5$ |
| $\widetilde r_1 = 0 \ne \widetilde r_2$ (or swapped) | $\le 5-2$ | $\le2$ (Case 3) | $0$ (Case 2) | $\le5$ |
| $\widetilde r_1 = \widetilde r_2 = 0$ | $\le 5-4$ | $\le2$ (Case 3) | $\le2$ (Case 3) | $\le5$ |

At most five throughout. $\quad\blacksquare$

**The candidates, explicitly.** The proof is constructive, and recovering a weighting from its multiplier needs no linear solve: $(I + 2\eta\widetilde M(\alpha))^{-1}$ acts as the identity off $\operatorname{span}\{v_1,v_2\}$ and as $1/(1+2\eta\theta_k)$ along $v_k$, so for each root $\eta$ of $\Phi_{S,\alpha}$ with $\det H(\eta,\alpha)\ne0$,

$$u(\eta) \;=\; -\eta\Big[\;\widetilde r(\alpha) \;-\; \sum_{k=1,2}\frac{2\eta\theta_k}{1+2\eta\theta_k}\,\widetilde r_k\,v_k\;\Big], \qquad w \;=\; u_0 + N\,u(\eta).$$

In the degenerate case $\widetilde r_k = 0$ the pole $\lambda_k$ contributes the points $u^\circ \pm \sqrt{-\widetilde g(u^\circ)/\theta_k}\,v_k$, at most two and only when that root is real. So a support's five candidates are closed-form in $\theta_1,\theta_2,v_1,v_2,\widetilde r(\alpha),\widetilde c(\alpha)$ — all of which §3.6 already builds — and the single step admitting no closed form is the rootfind itself: $\Phi_{S,\alpha}$ is a quintic (§7).

**The bound is attained.** A concrete instance, so no sharper constant is available: $a = (2.1041,\ 1.8783,\ 1.1827,\ 0.9551)$, $b = (2.7964,\ 2.7981,\ 0.8222,\ 3.8888)$, reachable range $[0.005,\ 1.000]$, target $\alpha = 0.6661$ — five stationary points.

**Corollary 5 (the enumeration is finite).** For any in-range $\alpha$, the candidate set of §3.9 has at most $5\cdot\#\{\text{admissible } S\} \le 5\,(2^K - K - 1)$ elements.

**Proof.** Lemma 3 bounds each support by five; only supports with $|S|\ge2$ are eligible, and of those only the admissible ones can host the optimum (§3.8). $\quad\blacksquare$

**Remark (the curvature window, and the hard case).** Since $\widetilde M(\alpha)$'s nonzero eigenvalues are $\theta_1 > 0 > \theta_2$, condition (3) of Theorem 2 reads $1 + 2\eta\theta_k \ge 0$, i.e.

$$\eta \in \Big[-\tfrac1{2\theta_1},\ -\tfrac1{2\theta_2}\Big] \ \ni 0 :$$

**the curvature window is exactly the interval between $\Phi_{S,\alpha}$'s two poles.** Two readings, which must be kept apart. *Inside:* at most one root lies in the **open** window — there $I + 2\eta\widetilde M(\alpha) \succ 0$ is strict, so by Theorem 1 the point is the *unique* global minimizer of the sign-free problem on $S$, while distinct stationary points carry distinct multipliers (b), so two interior roots would coincide. *On the boundary — the hard case:* the poles **are** the endpoints, so the Case 3 points sit exactly on the window, where the curvature is only semidefinite. Condition (3) is a $\succeq$, so they remain **certifiable but not unique** — the two Case-3 points $u^\circ \pm \sqrt{-\widetilde g(u^\circ)/\theta_k}\,v_k$ have equal norm — and this is precisely when the interior root vanishes and the sign-free minimum migrates to the boundary. The early exit therefore tests at most one interior candidate, plus the $\le2$ points of a populated pole. Since $\widetilde r_k = 0$ is one exact scalar equation in the data, real system sets do not meet it; the enumeration covers it regardless, which is why the proof sorts candidates by their multiplier rather than reading them off $\Phi_{S,\alpha}$'s roots. *This reads §3.7's numbers:* there $\theta_1 = 0.665$, $\theta_2 = -0.216$, giving the window $[-0.752,\ 2.313]$ quoted — and the forced $\eta = -0.878$ falls outside it, which is why no certificate exists.

### 3.11 Computing $w(\alpha)$

**What the algorithm does.** Given an in-range target $\alpha \in [\alpha_{\min},\alpha_{\max}]$ (Corollary 2), it returns the global optimum $w^\star(\alpha)$ of $(\mathrm P)$.

**Preprocessing.** Compute the $\binom{K}{2}$ pairwise balances once, in $O(K^2)$. They give the reachable range (Corollary 2) and, for each support, the two numbers $\alpha_{\min}(S), \alpha_{\max}(S)$ that decide admissibility at any target.

**Candidates on a support.** For an admissible $S$, first test $\alpha = \alpha_0(S)$; if so, $u_0$ is the optimum on $S$ and the only candidate (Lemma 3). Otherwise assemble $\Phi_{S,\alpha}$ from $\theta_1,\theta_2$ — the eigenvalues of the $2\times2$ matrix the curvature test already builds (§3.6) — together with $\widetilde c(\alpha)$ and the components of $\widetilde r(\alpha)$, and take its real roots by a companion-matrix eigensolve. Each root gives one $w$ by the closed form of §3.10, and a pole contributes at most two more in the degenerate case $\widetilde r_k = 0$. **At most five candidates result** (Lemma 3). Discard any that miss the constraints, leave the simplex, or whose support is not exactly $S$. *Numerically:* a live pole is a **double** root of $\Phi_{S,\alpha}$ (§3.10, Step 1), where rootfinders lose half their digits, so when $\widetilde r_k = 0$ divide the known factor $(1+2\eta\theta_k)^2$ out first and root the quotient — faster, and it keeps near-pole roots from being misread as ordinary ones.

**Search order.** Visit the admissible supports from the largest downwards, keeping the best ESS found so far; by Corollary 3 the search stops once no remaining support is larger than it.

1. **Early exit (Theorem 2).** Test the surviving candidates against conditions (0)–(3). If one passes, **return it**: it is a global optimum, so no other support can hold a better point. This is the common case and costs a single support.
2. **Descent heuristic.** If none passes, take a candidate whose failure is a negative coordinate and continue with $S\setminus\{k\}$, $k = \arg\min_i w_i$ — drop the system the solve pushed furthest below zero. This is an *ordering* heuristic: it raises the chance of an early exit and shortens the search; nothing about correctness rests on it.
3. **Fallback (§3.9).** If the search exhausts the admissible supports without a certificate firing, return the min-norm feasible candidate collected along the way.

**Soundness, unconditional.** Every returned $w$ is either certified by Theorem 2 — sufficiency, so no constraint qualification whatsoever — or is the min-norm element of a set proved to contain a global optimum.

**Completeness.** Every in-range target is feasible (Corollary 2), and only admissible supports can host its optimum (Proposition 2). Whenever the optimum's own support is visited, the optimum is among the at most five candidates enumerated there (Corollary 4 with Lemma 3); and when Corollary 3 skips that support, the incumbent already attains the optimal value. Either way the search returns a global optimum. The enumeration is finite (Corollary 5), and the certificate's incompleteness (§3.7) costs efficiency, never correctness.

**Cost.** The early exit touches $O(1)$ supports, and tests at most the single root inside the curvature window rather than all five (§3.10, Remark). The fallback tours the admissible supports at one quintic rootfind each, for a worst case of $5(2^K-K-1)$ candidates (Corollary 5) — still exponential in $K$ in the middle of the range, but now with an explicit constant. We report realized cost — early-exit rate, supports visited, fallback frequency, wall-clock — in §6.6.

---

## 4. Scope: levels of selection, and what each meta-metric sees

**The dependence lives at multiple sampling levels.** The selection reframing (§1.3) is general: a meta-evaluation's adequacy–fluency balance can be skewed by *any* uncontrolled sampling choice. Three sources, of which we control one:

- **System selection** — which systems entered the meta-evaluation. **Controlled by $\alpha$**; this is **extrinsic dependence (due to system selection)**, the *extrinsic* notion of Shayegh et al. (2025).
- **Segment selection** — which source segments were sampled to score the systems (it feeds SPA's confidence through each system's standard error). **Deliberately not controlled** (below).
- **Intrinsic measurement** — how the MQM framework and annotators penalize adequacy vs. fluency errors. **Left to the expert** (§2.2).

**Why we control system selection and not segment selection.** For three reasons. (i) *Numerosity:* thousands of segments per set (Shayegh et al., 2025, Table 10) vs. ~10–15 systems — the large segment sample's distribution is trustworthy, the tiny system sample's fragile. (ii) *Filtering:* systems are submitted and admitted to WMT, passing an opaque competition filter, so the set is non-randomly selected; segments face far less of one — and a filtered, tiny sample is exactly where selection bias bites. (iii) *Signal, not bias:* segment-level variance is information SPA legitimately uses — with system-level variances equalized, a component that is noisy segment-to-segment while the other is consistent is the less reliably judged one, so equalizing segment variance would erase a trust signal.

**What each meta-metric sees.** What $\alpha$ controls is a *system-level magnitude* balance; it reaches each meta-metric through two switches — does the metric read pairwise **magnitude** (so concordant pairs count too, §7) and does it read segment-level **confidence**?

| Meta-metric | Segment variance? | Concordant pairs? |
|---|:---:|:---:|
| Pearson | No | Yes |
| PA | No | No |
| Kendall $\tau$ | No | No |
| Spearman | No | No |
| SPA | Yes | Yes |

**Reading key (no hierarchy).** We report all five on equal footing; the table says how to *read* them, not how to rank them. The $\alpha$-target is a **magnitude** notion of influence (§3.3), so it is **exact for the magnitude-readers** (Pearson exactly, SPA up to its confidence weighting) and an **approximation for the sign/rank-readers** (PA, Kendall $\tau$, Spearman), which couple to $\alpha$ only through the discordant region; §7 gives the discordant-pair target as the faithful alternative there, left to future work. The one place the segment factor enters is SPA's confidence denominator — an out-of-scope modulation of an in-scope magnitude effect — so for the other four meta-metrics there is nothing about segments to ignore.

Agreement of all five under the $\alpha$-curves and the consistency result is therefore **cross-validation**, mirroring the PA/SPA consistency argument of Shayegh et al. (2025, Appendix C); disagreement is itself a result, and the reading key says why.

---

## 5. Effective Sample Size (ESS) and reachable range

When weights are unequal, the weighted estimate behaves like one from **fewer** equally-weighted systems:

$$\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2}.$$

Equal weights ⇒ ESS $= K$; all weight on one system ⇒ ESS $= 1$. Reference points for $K \approx 10\text{–}15$: ESS $\approx K$ trustworthy; $\gtrsim K/2$ comfortable; $\approx K/3$ the usual "getting uncomfortable" line; $\le 3\text{–}4$ effectively resting on a handful of systems (cratered for our purposes); $\to 1\text{–}2$ reinvents the synthetic-extreme-system problem with real systems.

**Practice:** do not commit to a magic cutoff. **Plot $\mathrm{ESS}(\alpha)$ beside $M(\alpha)$, shade where ESS $< K/3$, hard-stop reporting below ~3, and report the reachable range at a stated floor** (e.g. "$\alpha \in [0.71, 0.88]$ at ESS $\ge 5$") — narrower than the mathematical range $[\alpha_{\min},\alpha_{\max}]$ of Corollary 2, which asks only what is *attainable*, not what is *trustworthy*. ESS *is* the quantified version of "the artificiality just moved from fake systems to a degenerate distribution over real ones."

**Make-or-break empirical question:** how far can $\alpha$ travel before ESS crosses the floor, on the real ~10–15 systems? Wide ⇒ the strong "full control surface" claim. Narrow ⇒ the honest modest claim ("real system sets barely permit rebalancing — itself an indictment of WMT selection"). The first run decides which paper we write.

**Covariance and the range.** High positive $\rho$ compresses the reachable $\alpha$-range — few systems are loud-in-$b$/quiet-in-$a$ — so ESS craters faster and the attainable range narrows. Corollary 2 makes the mechanism visible: the endpoints *are* the extreme pairwise balances $\alpha_{ij}$ (§3.8), so as the systems' gaps become proportional the pairs' balances cluster and $\alpha_{\min},\alpha_{\max}$ close on each other. Report the reachable range alongside $\rho$, so that a narrow range is *explained* by correlated systems rather than read as a hidden failure of the method.

---

## 6. Evaluation plan

Division of labor: the **synthetic-scorer testbed** supplies ground truth for *internal validity*; the **subset-consistency experiment** supplies *external value* on real data with no ground truth; **synthesis** appears as a critique target (§2), as a **consistency baseline** (§6.5), and as a **range-extender** in the mechanical experiments only (§6.2, §6.6).

### 6.1 Meta-evaluation as a function of bias (the central object)

A scorer's meta-eval score is a **curve $M_{\text{metric}}(\alpha)$**, not a point. This is the headline framing. **$M$ is computed with all five meta-metrics** (PA, SPA, Pearson, Spearman, Kendall $\tau$), reported on equal footing; §4 is the reading key for what each sees of $\alpha$. We **do not impose a measure $\pi(\alpha)$** — that would smuggle back the "what is the right balance?" question the whole project refuses. Instead:

- **Lead with the curve $M(\alpha)$** itself.
- **Standpoint-free scalar summaries** for ranking tables: (i) **uniform average over the reachable $\alpha$** — the unique expression of *no preference among balances*; (ii) **worst-case** $\min_\alpha M(\alpha)$ — robustness, needs no measure; (iii) the **reachable range** $[\alpha_{\min}, \alpha_{\max}]$ — a property of the system set itself.
- Any *non-uniform* weighting is exactly the expert/application choice we deliberately leave open.

This unifies the SPA-plane triangle of Shayegh et al. (2025, §5.2): that triangle *is* the reachable adequacy–fluency frontier, and $M(\alpha)$ is a principled scalar traverse of it. **SPA is well-behaved under reweighting** — its per-pair confidence comes from segment-level permutation p-values, which system weights do not touch; reweighting only rescales each pair's contribution, so a sigmoid-saturation concern that would arise from rescaling the raw scores $a, b$ does not apply here.

### 6.2 Faithfulness of the knob (synthetic / mechanical)

Verify the instrument is what we claim, no ground truth needed. For reweighting the target balance is met **exactly** — the solver enforces $g(w,\alpha)=0$ — so realized $\alpha$ *equals* target $\alpha$ by construction and there is nothing to check about the two tracking. The faithfulness checks are instead **solver correctness** (the returned $w$ satisfies $g(w,\alpha)=0$ to numerical tolerance, $\sum_i w_i=1$, and $w\ge0$; and it agrees with a brute-force global solve on small system sets) and **ESS behavior** ($\mathrm{ESS}(\alpha)$ degrades gracefully toward the range endpoints). We also chart the **reachable range** as a function of $K$, the variance gap $\sigma(a;w)^2{:}\sigma(b;w)^2$, and $\rho$. Foundation, not payoff. (The realized-vs-target tracking check is non-vacuous only for *synthesis*, whose coarse mix ratios cannot hit an arbitrary target; it lives with the synthesis baseline, §6.5.)

**Range extension by synthesis (mechanical use only).** Adding synthesized systems widens the reachable $\alpha$-range, which lets us show the knob keeps turning and the solver stays cheap outside the natural range. This is used **here and in §6.6 only** — no trust claim is attached to it. **Owned caveat:** synthesized systems carry compressed within-system segment variance and therefore distorted SPA standard errors (§2.1), so any *extended-range* SPA number inherits that artifact. The clean claim is the natural (real-systems-only) range; the load-bearing verdicts — the $M(\alpha)$ curves (§6.1) and subset consistency (§6.5) — stay there.

### 6.3 Metameta-evaluation: synthetic scorers, real systems (internal validity)

**Generator.** Synthesize scorers at the **system level**:
$$m_i = w_a a_i + w_b b_i + \varepsilon_i.$$
Two independent ground-truth dials: the ratio $w_a{:}w_b$ is **known character** (construct-pure: $w_b=0$ is pure adequacy, no training, immune to the AdequacyX/FluencyX Table-7 anomaly); the noise scale $\sigma_\varepsilon$ is **known quality** (smaller = better, monotone by construction). **Noise on the composite, not on $a,b$ separately** (separate noises become a hidden second character knob). System-level scores are segment averages, so **CLT justifies Gaussian noise** even if segment errors aren't — a benefit of the system-level focus. Match $\sigma_\varepsilon$ to real regression residual scale; watch heteroscedasticity (systems with fewer segments have larger averaging noise).

**Two falsifiable predictions, each with its measurement:**

1. **Character recovery.** As $\alpha$ sweeps adequacy-heavy → fluency-heavy, pure-adequacy and pure-fluency scorers must reorder and **cross** — confirmed by observing that reorder/cross as $\alpha$ sweeps.
2. **Quality recovery.** At fixed $\alpha$, lower-noise scorers must score higher — measured by generating scorers over a grid of $\sigma_\varepsilon$, ranking by the meta-eval, and reporting **Kendall's $\tau$** between recovered and true $\sigma$-ranking across $\alpha$ (does it hold everywhere, or degrade where ESS is low?).

**Wider synthetic systems (load-bearing).** Rerun the testbed on a *wider* synthetic system set, explicitly separating **method capability** (does the knob work in a rich regime, with a broad reachable range and healthy ESS?) from **data limitation** (how much of any observed limit is the impoverished real system set, not the method?). This is the part of the internal-validity story that is **not circular** — unlike §6.4, it never fits an $a/b$-based model to real metrics and then uses it to license an $a/b$-based method — so it carries the internal-validity weight rather than sitting as an extra. Report capability and limitation as separate lines, not one blended verdict.

### 6.4 Validating the scorer model on real metrics

This step is **descriptive, not validation** — using an $a/b$-based fit to license an $a/b$-based method would be mildly circular — and it serves two purposes. It **licenses transfer** from testbed to real claims: the testbed assumes a scorer ≈ linear combination of $a,b$ plus noise, so a high $R^2$ says real metrics resemble that caricature and the internal-validity results carry over, while a low $R^2$ would mean they carry signal orthogonal to both aspects (itself a finding). And it **places** real metrics on the axis empirically, rescuing the "we can't hand-order the metrics" problem. Regress each real metric on the components across systems, pooled across sets/pairs to escape small $K$. The regressors are collinear, so coefficients are unstable — **do not quote coefficient ratios**; report collinearity-robust character instead (partial correlations, angle/projection in the $a$–$b$ plane, dominance shares, or sign). A one-sentence $R^2$ range answers the reviewer's inevitable "do real metrics even look like the synthetic ones?"

### 6.5 Subset consistency (real data, no ground truth — the strongest external experiment)

The only real-data experiment needing **no ground truth**: consistency is measurable without knowing the right answer, and it operationalizes the thesis of Shayegh et al. (2025) directly (if the bias is selection-driven, resampling systems should swing the *traditional* ranking, and fixing $\alpha$ should damp the swing).

**Inconsistency metric.** Each subset yields a ranking of the scorers, computed only over the scorers actually available for that subset's dataset — the metric-score roster is largely dataset-specific, since WMT's submitted-metrics pool changes, sometimes drastically, from year to year (confirmed empirically: of the automatic metrics scored for our system-sets, only three — BLEU, chrF, YiSi-1 — are common to all 15; most others cover a handful of years at most). For a pair of subsets $i,j$, restrict to their shared scorer set (size $n_{ij}$) and compute Kendall's $\tau_{ij}$ on that restriction. Aggregate as the **weighted mean pairwise Kendall's $\tau$**, weighted by $w_{ij} = \binom{n_{ij}}{2}$ — the number of scorer *pairs* $\tau_{ij}$ itself was computed over:
$$\bar\tau = \frac{\sum_{i<j} w_{ij}\,\tau_{ij}}{\sum_{i<j} w_{ij}},$$
over all subset pairs; report inconsistency as $1-\bar\tau$ (high = the verdict is at the mercy of which systems were drawn). This choice of weight is not arbitrary: since $\tau_{ij}$ is itself (concordant $-$ discordant)$/w_{ij}$ over its $w_{ij}$ scorer-pairs, weighting by $w_{ij}$ makes $\bar\tau$ exactly the **concordance rate pooled over every shared-scorer-pair across every subset-pair** — i.e. $\bar\tau$ is what you'd get by computing a single Kendall's $\tau$ over the one pooled collection of all (scorer-pair, subset-pair) comparisons, not merely a convenient reweighting. It also keeps a subset-pair with only a handful of shared scorers — a noisy, low-resolution $\tau_{ij}$ drawn from few comparisons — from counting as much as a well-supported pair with dozens in common; an unweighted mean would let the former dominate. Lead with this (standard cross-condition agreement, matches how meta-eval already thinks in rankings). Supporting per-scorer view: variance across subsets of each scorer's meta-eval rank/score, to localize *which* scorers are unstable. **Compute the underlying scorer scores with all five meta-metrics** (§4). SPA's permutation p-values are themselves unstable with very few systems, so the ESS floor applies to the SPA version too.

**Same-subset comparison — the claim is *improvement*, not invariance.** Adequacy–fluency is one axis, so fixing $\alpha$ removes only the $\alpha$-attributable instability and leaves a multi-axis remainder ($\rho$, composition, every other axis); claiming full consistency would claim A–F is the *whole* story, contradicting the premise. So for each drawn subset $\mathcal{S}_k$ we compute *both* the traditional and the fixed-$\alpha$ ranking **on that identical $\mathcal{S}_k$**, and the **gap is the improvement**: the uncontrolled remainder sits in both terms on the same subset and subtracts away. This needs no decomposition of the remainder — same-subset design holds it constant for free — whereas *solving* would require enumerating it, which we do not claim. (The wrong design — traditional on one collection, fixed-$\alpha$ on another — would not cancel the remainder and would be confounded.)

**The curve (headline figure).** x-axis = target $\alpha$; y-axis = inconsistency ($1-\bar\tau$ across subsets, at that fixed $\alpha$). The fixed-$\alpha$ method traces a **curve** — "how stable is the meta-eval if the balance is pinned *here*?" Traditional meta-eval is a **single point/line** — it pins nothing, so each subset floats to its own $\alpha$ and yields one inconsistency number (plotted at the mean natural $\alpha_0$, or as a horizontal reference). The claim is visual: **the fixed-$\alpha$ curve lies below the traditional point across the reachable, ESS-healthy range** — pinning the balance *anywhere* beats letting it float. Overlay: (i) shade where ESS craters (the curve rises at the extremes — instability returning there is weight concentration, not balance, and showing you know that is part of the honesty); (ii) the curve's *minimum* is the most-*reproducible* operating point — a reproducibility claim, not a correctness one (§1.3).

**Per-subset view (companion figure).** Plot each subset directly rather than only the aggregate. For each drawn subset, its rebalanced meta-evaluation (**mean Kendall $\tau$**) traces a curve over $\alpha$; the per-subset curves form a **shaded band**, their **mean a solid curve**. Each subset's *traditional* meta-evaluation is a **single point at that subset's own natural $\alpha_0$** — so the points scatter along $\alpha$ — and their **mean is a horizontal line**. Consistency then reads off as spread: pinning $\alpha$ **bunches** the per-subset curves into a tight band where the **traditional points scatter**, and the scatter-to-band reduction is the balance-attributable instability the method removes (the composition remainder sits in both, so it does not).

**Procedure.** Draw many system subsets from the **real** systems only — this verdict is load-bearing, so it stays on the natural range (§6.2). Build the curve above. **Get right:** (i) report ESS per subset and condition on a floor — re-achieving a fixed $\alpha$ on a lopsided subset craters ESS and adds ranking noise that fights the gain; (ii) **mechanism** — show traditional inconsistency *tracks how much $\alpha$ varies across subsets* (ties instability causally to the thing we fix); (iii) the curve already shows several fixed $\alpha$, so the gain is the *fixing* itself — stable meta-eval deliverable at *any* expert-chosen balance; (iv) optionally mention $\rho$ as one modulating factor.

**Synthesis as a baseline (matched-$\alpha$ head-to-head).** Synthesis is coarser and confounded but usable (§2.1), so it is the natural comparison instrument. For each target $\alpha$, pick the synthesis mix whose **realized $\alpha$** — computed with our own definition $\sigma(a;w)^2/(\sigma(a;w)^2+\sigma(b;w)^2)$, using **no $F$ and no $B(\Delta p)$** — lands nearest, and plot its inconsistency at that realized $\alpha$ on the same axes. Here the **realized-vs-target check is non-trivial** (unlike reweighting, whose realized $\alpha$ equals the target by construction): synthesis reaches only a **coarse, integer-grained** set of realized $\alpha$ fixed by pool-size mix ratios, so we verify that its realized $\alpha$ moves **monotonically** with the mix ratio and record the **granularity** of the values it can actually hit — which bounds how closely any target can be matched and is itself part of the baseline's handicap. We deliberately do **not** use the $B$-selected Rows 6/7 of Shayegh et al. (2025) as the baseline: that would build the comparison out of the very criterion we disqualified (§2.2). Confine the head-to-head to the **natural (real-systems-only) range**, where the two instruments are genuinely distinct and artifact-free.

**Pre-registered mechanism (a hypothesis, not an assumed win).** We *expect* reweighting to be more stable across subsets than synthesis, because reweighting adjusts continuously while synthesis's mix ratio is integer-grained and its pool composition jumps discontinuously as the drawn subset changes. We **test** this rather than assume it; a null or reversed result is reportable and would say the continuity advantage does not bite at these $K$.

### 6.6 Solver cost (empirical)

Soundness is unconditional (§3.11), but the search carries no worst-case bound: the early exit (Theorem 2) usually settles a target at a single support, while the fallback (§3.9) may in principle tour every admissible one. We therefore measure the split rather than claim it. **Setup:** solve $w(\alpha)$ on a grid of targets for each real WMT system set, and on sets augmented with synthesized systems to push $K$ well beyond the native $\approx 10\text{–}15$. **Report**, as functions of $K$ and of the target's position in the reachable range: (i) the **early-exit rate** — the fraction of targets settled by a passing certificate; (ii) the number of **supports visited** before it fires; (iii) the fraction of targets falling through to the **enumeration fallback**; (iv) how many supports **admissibility** and the **ESS bound** of Corollary 3 remove (§3.8), separately and together, against the target's position in the range; (v) the **candidates enumerated per support**, and the share of targets settled by the $\alpha_0(S)$ test alone; and (vi) wall-clock per target. **Expectation:** the early exit fires at the large majority of targets and within a few supports, so the exponential fallback stays rare — an expectation to be measured, not a claim. This is a systems-cost sanity check, not a headline result.

---

## 7. PA vs. SPA: different influence targets, and what a PA-native mechanism would cost

**How they differ.** PA (Kocmi et al. 2021) reads only the **sign** of each system pair's gap — magnitude-blind. SPA (Thompson et al. 2024) reads **confidence**: the scorer's vs. the human's permutation p-value for "$j$ beats $i$," so the **magnitude** of the gap relative to its uncertainty matters. Shayegh et al. (2025) adopt SPA as primary and note (their Appendix C) that the two can disagree — a metric can read adequacy-biased under PA yet fluency-biased under SPA — so the choice is not cosmetic.

**Different targets.** "Influence" is defined differently by each, and each definition is a different reweighting target. Writing $\Delta a_{ij} = a_i - a_j$ and $\Delta b_{ij} = b_i - b_j$ for a pair's component gaps: under PA only **discordant** pairs are contested (a concordant pair's sign is fixed regardless of balance), so PA-influence of $a$ is the weighted fraction of discordant pairs it wins, and the PA-faithful target is
$$\sum_{\text{discordant }(i,j)} w_iw_j\big[\mathbb 1(|\Delta a_{ij}|>|\Delta b_{ij}|)-\mathbb 1(|\Delta b_{ij}|>|\Delta a_{ij}|)\big]=0.$$
Under SPA the magnitude of each component's contribution matters on **every** pair, so the target is $\sigma(\Delta a;w)^2=\sigma(\Delta b;w)^2$ — **exactly our machinery** (§3.3). Our tool is magnitude-shaped; the earlier "discordant-only" reading was a PA notion smuggled onto it. We keep the magnitude target — natively served, and read exactly by the magnitude-readers (§4) — rather than amputate it to PA's.

**What a PA-native mechanism would cost.** The PA target is still a single quadratic, $w^\top C w = 0$ with $C$ the fixed matrix of per-pair discordance signs, so the §3 certificate (Theorem 1) still applies. But $C$ is generally **full-rank**, unlike the rank-2 $M(\alpha)=\alpha bb^\top-(1-\alpha)aa^\top$ built from the two mean directions — and full rank forfeits every efficiency the SPA target enjoys: the cheap $2\times2$ curvature check, and a $w(\alpha)$ that is **piecewise algebraic of degree $\le5$** — per support a closed-form weighting driven by one quintic rootfind in one variable (Lemma 3) — leaving a general $(K{-}1)$-dimensional solve at each $\alpha$ (still feasible at $K\approx10$–15, but one-dimensional rootfind against $(K{-}1)$-dimensional solve is the operative difference). It also loses the §3.3 magnitude-balance reading. That extra cost is why the mechanism is built on the magnitude target. We do not work the PA mechanism out further: the matrix $C$ and this lost-efficiency accounting are all that is needed.

**Two residual caveats.** (i) *Covariance is a PA issue.* Under PA, equal component variance need not mean equal sign-influence — sign-flip frequency depends on the joint shape and on $\sigma(a,b;w)$ (which sets the discordance rate), not on second moments alone; its separate effect on the reachable range is in §5. Under SPA this dissolves, since magnitude is what SPA reads. (ii) *SPA weights by confidence.* SPA folds uncertainty into its p-values while our target uses raw score variance; they agree in direction but not identically, so $\alpha$ targets the *score-magnitude* balance — SPA's dominant driver, not a perfect proxy. We keep raw variance (clean, closed-form) and state this rather than re-engineer.

**Recommendation.** Report all five meta-metrics on equal footing ($M(\alpha)$ curves, subset consistency, testbed), reading each through the key of §4. The discordant-pair target is the PA-faithful *robustness alternative*. Shayegh et al. (2025) already found PA and SPA broadly consistent, so we expect agreement here; their discordant fraction (Table 5: 43–71%) is descriptive, not load-bearing.

---

## 8. Limitations (stated up front)

- **Marginal $\alpha$.** The cross term $2\,\sigma(a,b;w)$ is excluded by declaration; $\alpha$ is not "the true bias." Robustness alternatives: a covariance-aware *unique-influence* $\alpha$ — a Shapley/LMG partition into $a$-unique, $b$-unique, and shared variance, which must then state where the shared mass is assigned — and, for PA, the discordant-pair target (§7).
- **One axis.** Only adequacy–fluency is controlled; other latent tradeoffs are left to future work (§1.4).
- **No named optimum.** The expert/application decides which $\alpha$ to use; the method does not (§1.3).
- **Variance = influence is metric-dependent.** The variance target is exact for the magnitude-readers (Pearson exactly, SPA up to its confidence weighting) and only an approximation for the sign/rank-readers, for which the discordant-pair target is the faithful alternative (§4, §7). The residual approximation is localized to those metrics, not the method as a whole.
- **Reachable range bounded by real system spread**; ESS craters at the extremes; high $\rho$ compresses the range. Reported honestly via ESS-floored ranges.
- **One selection level.** System-selection bias is controlled; segment-level variance is deliberately left intact (§4).
- **System-level only.** Intrinsic share and extrinsic dependence are not separated; $\alpha$ is total observed balance (§1.4).

---

## Notation

| Symbol | Meaning |
|---|---|
| $K$ | number of translation systems |
| $a = (a_1,\dots,a_K)$ | Adequacy MQM score vector; $a_i$ is system $i$'s adequacy score |
| $b = (b_1,\dots,b_K)$ | Fluency MQM score vector; $b_i$ is system $i$'s fluency score |
| $t = a + b$ | All MQM score vector; $t_i = a_i + b_i$ |
| $\Delta a_{ij} = a_i - a_j,\ \ \Delta b_{ij} = b_i - b_j$ | a system pair's component gaps (§7) |
| $w = (w_1,\dots,w_K)$ | system weights (importance weights), $\sum_i w_i = 1$, $w_i \ge 0$ |
| $\Sigma(w) = \mathrm{diag}(w) - ww^\top$ | weighted-covariance operator |
| $\mu(a;w) = w^\top a$ | weighted mean of $a$ |
| $\mu(a) = \tfrac1K\mathbf 1^\top a$ | plain (unweighted) mean of $a$ (uniform-weight case) |
| $\sigma(a;w)^2 = a^\top\Sigma(w)\,a$ | weighted variance of $a$ |
| $\sigma(a,b;w) = a^\top\Sigma(w)\,b$ | weighted covariance of $a$ and $b$ |
| $q(a;w) = \dfrac{\sum_{i<j}w_iw_j(a_i-a_j)^2}{\sum_{i<j}w_iw_j} \propto \sigma(a;w)^2$ | pairwise-difference variance (pair-weighted mean squared system gap; the meta-evaluation driver) |
| $\alpha = \dfrac{\sigma(a;w)^2}{\sigma(a;w)^2+\sigma(b;w)^2}$ | balance parameter (adequacy's share of weighted between-system variance) |
| $g(w,\alpha) = (1-\alpha)\sigma(a;w)^2 - \alpha\,\sigma(b;w)^2$ | balance constraint ($g=0$ realizes target $\alpha$) |
| $\eta,\ \lambda$ | multipliers for the **balance** and **budget** constraints, signed so that stationarity reads $w + \eta\,d(w,\alpha) + \lambda\mathbf 1 = 0$ on the support ($\eta$ = tilt amount; $\eta = 0$ leaves the weights uniform). Subscripted $\lambda_k$ is a different object — a **pole**, see below |
| $d_i(w,\alpha) = \partial g/\partial w_i$ | balance gradient in the direction of weight $i$ |
| $\alpha_{ij} = (a_i-a_j)^2/\big[(a_i-a_j)^2+(b_i-b_j)^2\big]$ | **pairwise balance**: the balance of the two-system pool $\{i,j\}$ (§3.8) |
| $\alpha_{\min}(S),\ \alpha_{\max}(S)$ | smallest / largest pairwise balance among pairs drawn from support $S$; unqualified $\alpha_{\min},\alpha_{\max}$ take $S=\{1,\dots,K\}$ and bound the reachable range |
| $\alpha_0(S) = \sigma(a;u_0)^2/\big[\sigma(a;u_0)^2+\sigma(b;u_0)^2\big]$ | **natural balance** of support $S$: the balance it realizes unweighted, and the unique target at which $u_0$ is feasible and $w(\alpha) = u_0$ (§3.10). Unindexed $\alpha_0$ (§3.2) is $\alpha_0(\{1,\dots,K\})$ |
| $\theta_1 > 0 > \theta_2$ | nonzero eigenvalues of $\widetilde M(\alpha)$, equivalently of the $2\times2$ matrix of Proposition 1: $\theta_{1,2} = \tfrac12(T \pm \sqrt{T^2-4D})$ with $T = \alpha\|\tilde b\|^2 - (1-\alpha)\|\tilde a\|^2$ and $D = -\alpha(1-\alpha)(1-\rho^2)\|\tilde a\|^2\|\tilde b\|^2 < 0$. Then $\det H(\eta,\alpha) = (1+2\eta\theta_1)(1+2\eta\theta_2)$, and its **poles** $\lambda_k = -1/(2\theta_k)$ are the endpoints of the curvature window $\ni 0$; $v_k$ is a unit eigenvector for $\theta_k$ |
| $\widetilde r_1,\widetilde r_2,\ \widetilde r_0^2$ | $\widetilde r(\alpha)$ resolved along $\widetilde M(\alpha)$'s eigenbasis: $\widetilde r_k = v_k^\top\widetilde r(\alpha)$ are its coordinates on the two curved directions, $\widetilde r_0^2 = \|\widetilde r(\alpha)\|^2 - \widetilde r_1^2 - \widetilde r_2^2$ its squared length on the flat ones ($\ker\widetilde M(\alpha)$) |
| $\Phi_{S,\alpha}$ | **secular polynomial** of support $S$ at target $\alpha$; $\deg\le5$, and its roots carry the stationary points (Lemma 3) |
| $M(\alpha) = \alpha\,bb^\top - (1-\alpha)\,aa^\top$ | constant matrix of the balance quadratic, $g = w^\top M(\alpha)w + r(\alpha)^\top w$ |
| $r(\alpha)_i = (1-\alpha)a_i^2 - \alpha b_i^2$ | linear part of the balance quadratic |
| $S$ | support (the set of kept systems, $w_i > 0$) |
| $u$ | reduced (mirror) variable after eliminating the budget constraint, $w = u_0 + Nu$, where $u_0$ is the **uniform weighting** — the unique budget-feasible point in $\operatorname{span}(\mathbf 1)$, so $N^\top u_0 = 0$ — and $N$ is any fixed matrix with orthonormal columns spanning $\{v:\mathbf 1^\top v=0\}$; any valid $N$ gives the same optimum (§3.6). On a support $S$, $u_0$ is the uniform weighting on $S$ (§3.10) |
| $\widetilde M(\alpha) = N^\top M(\alpha)\,N$ | balance matrix in the reduced variable (with $\widetilde r(\alpha),\widetilde c(\alpha)$ similarly reduced) |
| $\tilde a = a - \mu(a)\mathbf 1,\ \tilde b = b - \mu(b)\mathbf 1$ | mean-centered score vectors (budget-preserving). On a support $S$ (§3.6, §3.10) they are read **within $S$**: $a,b$ restricted to $S$ and centered at their means over $S$, so $\rho$ is then the correlation across $S$'s systems |
| $H(\eta,\alpha)$ | $2\times2$ **symmetric** reduced curvature matrix in an orthonormal basis of $\operatorname{span}\{\tilde a,\tilde b\}$; the certificate checks $H \succeq 0$ (i.e. $\operatorname{tr}H\ge0$, $\det H\ge0$) |
| $\mathrm{ESS}(w) = 1/\sum_i w_i^2$ | effective sample size (the objective $\tfrac12\sum_i w_i^2$ equals $1/(2\,\mathrm{ESS})$, so minimizing it maximizes ESS) |
| $m_i = w_a a_i + w_b b_i + \varepsilon_i$ | synthetic scorer (testbed): aspect weights $w_a,w_b$, noise s.d. $\sigma_\varepsilon$ |
| $\rho$ | adequacy–fluency **Pearson correlation** across systems, $\rho = \tilde a^\top\tilde b/(\|\tilde a\|\|\tilde b\|)$; drives both the reachable range (§5) and the curvature test (§3.6) |
| PA, SPA | Pairwise Accuracy, Soft Pairwise Accuracy (meta-metrics) |
| $M_{\text{metric}}(\alpha)$ | a scorer's meta-evaluation score as a curve over $\alpha$ (the central object) |
