# A Practitioner's Guide to Choosing β

This guide accompanies the WMT2026 paper [*Mind Which Bird You Favour*](https://arxiv.org/abs/2609.14795). It explains how to use the tool to specify the preferred balance between adequacy and fluency in a meta-evaluation.

## Why you need to choose a β

Meta-evaluation ranks MT scorers (metrics) by how well they rank a set of translation systems. This ranking is based on two types of errors: **adequacy** errors (incorrect meaning) and **fluency** errors (unnatural or ungrammatical text). The final ranking is driven by whichever error type varies more across the systems in the pool. Consequently, the winning metric's adequacy-fluency balance is dictated by the specific systems annotated. This balance is arbitrary and fluctuates significantly. When selecting a metric for your own use, this balance is crucial:
- Adequacy matters more in legal, medical, or technical translation, where a fluent but inaccurate sentence is dangerous.
- Fluency matters more in marketing, subtitles, or creative texts, where readers immediately notice awkward phrasing.

The parameter β adjusts this balance. A β near 1 drives the ranking by adequacy, while a β near 0 drives it by fluency. Each dataset has a natural balance, β₀, representing the traditional default. This tool enables the use of alternative β values, but the target β must first be determined.

## Why there is no recommended β

We would like to give you a table of the form "use β = 0.7 for legal text".
That is not possible. The effect of a given β value depends on things that
change from dataset to dataset:

- the language pair and domain,
- how strongly Adequacy MQM and Fluency MQM are correlated across systems,
- how spread out the systems are, and which systems they are,
- the biases of the scorers being compared,
- the MQM taxonomy and annotator noise.

The same β can therefore produce a strongly adequacy-leaning meta-evaluation
on one dataset and a neutral one on another. We also cannot work out your
ideal β from first principles yet. Instead, consider β as a **directional knob**: raising it always
moves the meta-evaluation towards adequacy. But you have to **tune** it on
the dataset you are using, against your own preference.

## Find your β

There are two common situations. Pick the one that fits you.
- Scenario 1: Push as far towards adequacy (or fluency) as is reliable
- Scenario 2: You have a specific balance in mind

---

### Scenario 1: Push as far towards adequacy (or fluency) as is reliable

Here you do not have a precise balance in mind. You want the most
adequacy-oriented meta-evaluation, or the most fluency-oriented one, that the
data can still support.

#### Find the reliable range of β

Deviating from β₀ distorts the meta-evaluation and reduces its quality. The effective sample size (ESS) measures this distortion. At β₀, ESS equals the total number of systems K (highest quality) but drops toward 2 at the edges of the reachable range (lowest quality).
Each dataset therefore allows only a range of β around its natural balance.
**We recommend a minimum ESS of 0.9 × K.** The two ends of that range are
your most fluency-oriented and most adequacy-oriented reliable settings.

Run this from the repository root, after `bash setup.sh` and
`source .venv/bin/activate`:

```python
from src.lib.consistency import real_systems
from src.lib.mqm_scoring import load_system_scores
from src.lib.reweight_exact import solve_w_exact
from src.lib.spa_beta_sweep import beta_sweep_range

dataset = 'ende24'
systems = real_systems(dataset, excl_missing_seg_granular_mqm=True)
scores = load_system_scores(dataset).loc[systems]
a, f = scores['a'].values, scores['f'].values   # system-level Adequacy / Fluency MQM
K = len(systems)

beta_0, beta_min, beta_max, beta_lo, beta_hi = beta_sweep_range(a, f, target_ess=0.9 * K)
print(f'natural beta_0 = {beta_0:.3f}, reachable = [{beta_min:.3f}, {beta_max:.3f}]')
print(f'reliable (ESS >= 0.9K) = [{beta_lo:.3f}, {beta_hi:.3f}]')

w_fluency = solve_w_exact(a, f, beta_lo).w    # most fluency-oriented reliable weighting
w_adequacy = solve_w_exact(a, f, beta_hi).w   # most adequacy-oriented reliable weighting
```

For `ende24` this prints:

```
natural beta_0 = 0.782, reachable = [0.007, 1.000]
reliable (ESS >= 0.9K) = [0.548, 0.862]
```

Almost any β can be reached in principle, but only about 0.55–0.86 keeps the
meta-evaluation trustworthy. Some datasets allow much less. In `jazh24`, for
example, most systems sit in a narrow band, so there is little room to move.

If you use your own MQM data, the solver only needs the two arrays `a` and
`f`, one entry per system. Use the weights `w` in any weighted meta-metric,
for example `soft_pairwise_accuracy_from_pvalues(human_p, scorer_p, w)` in
`src/lib/metametrics.py`.

#### Double-check the quality

ESS is a proxy. To check the reweighted meta-evaluation directly, use the
paper's two consistency measures, **MQM adherence** and **explainability by
MQM** (Section 4 of the paper explains what each one captures). Both should
stay close to their values at β₀. A clear drop means the reweighting has
distorted the meta-evaluation.

```python
import numpy as np
from src.lib.consistency import real_scorers
from src.lib.synthetic_scorer_beta_grid import base_scorer_beta_dial_grid
from src.lib.synthetic_scorer_preference import base_scorer_preference_by_beta
from src.lib.synthetic_scorers import make_aft_dial_grid, make_aftj_j_dial_grid

betas = [beta_lo, beta_0, beta_hi]
w_by_beta = {b: solve_w_exact(a, f, b).w for b in betas}

per_scorer = []
for scorer in real_scorers(dataset, systems=systems, require_seg_scores=True):
    g = base_scorer_beta_dial_grid(dataset, scorer, systems, w_by_beta, synthesis='additive_mean',
                                   dial_grid=make_aft_dial_grid(), use_af_family=True)
    gj = base_scorer_beta_dial_grid(dataset, scorer, systems, w_by_beta, synthesis='offset',
                                    dial_grid=make_aftj_j_dial_grid())
    if g is None or gj is None:   # scorer lacks segment-level coverage
        continue
    p = base_scorer_preference_by_beta(g)
    p['J'] = base_scorer_preference_by_beta(gj)['J']
    per_scorer.append(p)

for name, key in [('Adequacy over fluency', 'AF'), ('MQM adherence', 'T'), ('Explainability by MQM', 'J')]:
    print(f'{name:22s}', np.round(np.mean([p[key] for p in per_scorer], axis=0), 3))
```

Each row prints three values, for β = `beta_lo`, `beta_0` and `beta_hi`. This
takes a few minutes, because every scorer runs permutation tests.

#### How much did the balance move?

The first row above, **adequacy over fluency**, shows how far a β choice
shifts the meta-evaluation towards adequacy. It rises steadily with β, so
comparing it at your chosen β with its value at β₀ tells you how big the
shift is.

**Only changes in this score are meaningful. Its absolute value is not.** A
value of 0.5 does not mean "balanced", and the same value on two datasets
does not mean the same balance. The score is computed by augmenting the
dataset's own base scorers, and those base scorers differ from dataset to
dataset and are not controlled. Read it only as "this β moves the
meta-evaluation towards adequacy (or fluency) on this dataset".

---

### Scenario 2: You have a specific balance in mind

Here you know what trade-off your application wants, but you cannot state it
as a number. The idea is to write your preference down as examples and tune
β until the meta-evaluation agrees with them.

#### 1. Craft example pairs around your decision line

Picture every translation as a point placed by its adequacy errors and its
fluency errors. Your preference is a line through that plane. On one side
you would pick the more adequate translation, and on the other side the more
fluent one. The informative examples sit **close to that line**:

- Each pair translates the same source.
- One translation is *slightly* more adequate, and the other is *slightly*
  more fluent.
- Overall quality is similar, so the choice is a genuine judgment call.

For each pair, record which translation your use case prefers. Pairs where
one translation is clearly better on both aspects tell you nothing about the
balance, so leave them out. A dozen pairs from your own domain are a good
start.

#### 2. Score the pairs with your candidate scorers

Run a subset of the scorers involved in the meta-evaluation on both translations in every pair. Count how often each scorer's preference aligns with yours. This yields an agreement rate for each scorer, establishing a ranking among them.

#### 3. Tune β until the meta-evaluation agrees

Sweep β over the reliable range from Scenario 1. At each β, rank the scorers
by weighted SPA and compare that ranking with your agreement ranking. Choose
the β where the two agree best.

```python
import numpy as np
from scipy.stats import kendalltau
from src.lib.consistency import load_scorer_inputs, real_systems
from src.lib.metametrics import pairwise_p_values, soft_pairwise_accuracy_from_pvalues
from src.lib.mqm_scoring import load_system_scores
from src.lib.reweight_exact import solve_w_exact
from src.lib.spa_beta_sweep import beta_sweep_range

dataset = 'zhen23'
systems = real_systems(dataset)
scores = load_system_scores(dataset).loc[systems]
a, f = scores['a'].values, scores['f'].values
K = len(systems)

# From step 2: fraction of your crafted pairs where each scorer agreed with you
agreement = {
    'MetricX-23': 0.70,
    'MetricX-23-QE': 0.55,
    'XCOMET-Ensemble': 0.40,
    'XCOMET-QE-Ensemble': 0.35,
}

inputs = load_scorer_inputs(dataset, 'spa', systems=systems)
names = [s for s in agreement if s in inputs]
p_values = {s: (pairwise_p_values(inputs[s].human_seg), pairwise_p_values(inputs[s].scorer_seg))
            for s in names}

_, _, _, beta_lo, beta_hi = beta_sweep_range(a, f, target_ess=0.9 * K)
results = []
for beta in np.linspace(beta_lo, beta_hi, 21):
    w = solve_w_exact(a, f, beta).w
    spa = [soft_pairwise_accuracy_from_pvalues(*p_values[s], w) for s in names]
    tau = kendalltau(spa, [agreement[s] for s in names]).statistic
    results.append((beta, tau))
    print(f'beta={beta:.3f}  tau={tau:+.2f}')

best_beta, best_tau = max(results, key=lambda r: r[1])
```

Some practical notes:

- **Use more than a handful of scorers.** With only four, as above, the
  correlation can take only a few values, and several β values tie. Look at
  where the agreement plateaus rather than at a single maximum.
- **If the best β sits at the edge of the reliable range**, this dataset
  cannot reliably express your preference. Try another dataset, or lower the
  ESS threshold knowingly and check quality as in Scenario 1.
- **Retune for every dataset.** Because of the reasons in the section on why
  there is no recommended β, a β tuned on one dataset does not carry over to
  another. What carries over is your crafted pairs, so keep them.
