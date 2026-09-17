# Mind Which Bird You Favour

> This repository's code was refactored and reorganized with the help of
> [Claude](https://claude.com/claude-code) (Anthropic).

Code for the WMT2026 paper **["Mind Which Bird You Favour: Parameterizing
Adequacy–Fluency Balance in Meta-Evaluation of Machine Translation."](https://arxiv.org/abs/2609.14795)**

Machine translation meta-evaluation (evaluating automatic MT scorers against
human judgment) inherits a hidden adequacy–fluency balance from whichever
translation systems happen to be in the evaluation pool. This project:

1. Exposes that balance as an explicit parameter, β, and provides an exact
   algorithm that reweights the systems in a meta-evaluation to hit any
   target β while staying as close to uniform weighting as possible
   (`mwb/lib/reweight_exact.py`).
2. Introduces a scorer-augmentation framework — controlled synthetic scorer
   families with a known relative identity — to validate a meta-evaluation's
   internal consistency (`mwb/lib/synthetic_scorers.py` and friends).
3. Applies both to compare the paper's reweighting method against prior
   system-synthesis approaches, and to sweep popular scorers (MetricX,
   xCOMET, ...) across the reachable β range.

**Naming note:** the code calls the paper's β `beta` throughout
(`mwb/lib/beta.py`), matching the paper directly. The paper's `a`/`f`
(adequacy/fluency) are likewise `a`/`f` in code. A separate, unrelated
quantity — the paper's footnote reparameterization β_std — lives in the
same module as `beta_to_beta_std`/`beta_std_to_beta`; it does not appear in
any paper table.

## Setup

Requires Python ≥3.9.

```bash
bash setup.sh
```

This creates `.venv` and installs `requirements.txt` into it, then fetches
the two git submodules this project reads at runtime —
[`mt-metrics-eval`](https://github.com/google-research/mt-metrics-eval)
(the WMT Metrics Shared Task toolkit; only its `score_mqm.py` converter
logic is reproduced, not imported, since that package needs Python ≥3.10)
and [`wmt-mqm-human-evaluation`](https://github.com/google/wmt-mqm-human-evaluation)
(the raw MQM error annotations) — plus `external/mt-metrics-eval-data`, the
official system/metric score release (~870MB download, ~2.3GB unpacked; a
plain GCS file, not a git repo, so it isn't a submodule).

Everything a run produces (figures, tables, `.npz` caches) is written under
`output/`, gitignored.

## Reproducing the paper

Every figure and table maps to one `cli.py` command. Each also runs
standalone as `python -m mwb.scripts.<name>` with its own, more detailed
`--help`.

| Paper artifact | Command |
|---|---|
| Figure 1 (adequacy vs. fluency scatter) | `python cli.py af-scatter` |
| Figure 2 (SPA plane, synthetic families) | `python cli.py spa-plane-synthetic` |
| Figure 3 (main results) | `python cli.py preference-compute -- --dataset ende21` then `python cli.py preference-synth25 -- --dataset ende21` then `python cli.py preference-plot -- --dataset ende21` |
| Figure 4 (β–ESS preference heatmap) | `python cli.py beta-ess-heatmap -- --dataset ende21` |
| Figure 5 (weighted meta-evaluation of popular scorers) | `python cli.py spa-vs-beta-compute -- --dataset zhen23` then `python cli.py spa-vs-beta-plot` |
| Figure 6 (LOO stability vs. β) | `python cli.py loo-tau-vs-beta-compute -- --dataset zhen23` then `python cli.py loo-tau-vs-beta-plot` |
| Table `dataset_tau` (Appendix, LOO stability) | `python cli.py loo-tau` |
| Tables `solve-exact-operation-counts` / `solve-exact-theorem-invocations` (Appendix, solver efficiency) | `python cli.py solver-efficiency -- --dataset ende24` |
| Appendix "Dataset Statistics" (4 tables) | `python cli.py dataset-stats` |
| Appendix pearson-correlation results | add `--metametric pearson` to the `preference-*` commands above |

Run `python cli.py --help` for the full command list, including
`validate-scores` (checks the reconstructed MQM scores against the official
WMT score files) and `validate-beta-ess` (cross-checks the β–ESS heatmap
sampler against the exhaustive lattice).

The compute steps above are the expensive part (solver calls plus SPA
permutation tests); the corresponding `*-plot` commands are cheap and safe
to rerun repeatedly while iterating on figure styling.

## Data notes

Worth knowing before touching `mwb/mqm_scoring.py` or the data under
`external/`:

- **Two scoring paths.** For most datasets, official per-error rating files
  in `mt-metrics-eval-v2` are used directly as ground truth
  (`OFFICIAL_RATINGS`); a few datasets (`enzh22`, `ende20`/`zhen20`/`enru22`)
  fall back to re-deriving scores from the raw MQM TSVs (`parse_mqm_tsv` +
  `error_weight`), because that path was found to match the official totals
  *better* than the official rating files for those specific sets. See
  `mwb/mqm_scoring.py`'s module docstring for the full weighting table and
  the three confirmed overrides.
- **Segment indexing gotcha.** `doc_id`/`seg_id` are 1-indexed in every
  dataset except `generalMT2022/enru`, which is 0-indexed. This doesn't
  affect scoring (rows are grouped by whatever values appear, never assumed
  to start at a fixed index), but will silently break any code that
  reconstructs segment identity by counting from a fixed start.
- **Taxonomy variants.** The Adequacy/Fluency category taxonomy differs by
  dataset — hierarchical (`Accuracy/Mistranslation`), flat
  (`do_not_translate`), and a mixed variant unique to `generalMT2022/enzh`
  (underscores before the slash). `classify_aspect` normalizes underscores
  to spaces universally to handle all three; an earlier version that only
  normalized in the flat-taxonomy branch silently dropped ~640 rows of
  `enzh22` to "unclassified."
- **Disjoint reference sub-samples.** A few datasets contain extra
  reference/human systems rated on a segment sub-sample that barely
  overlaps the main pool (e.g. `newstest2021/en-de`'s `ref.C`/`ref.D`).
  Left in, this biases that system's average toward its private sample.
  `filter_modal_coverage` drops any segment rated by fewer than 50% of the
  dataset's max observed per-segment system count — a threshold, not exact
  equality to the mode, since coverage legitimately varies by a little for
  unrelated reasons in some sets.
- **`ende23` is excluded project-wide** (not just from the paper's headline
  datasets): far more of its MQM annotations fall into the catch-all
  "Other" category than any other dataset, so its adequacy/fluency split is
  less trustworthy (see `mwb/lib/consistency.py`'s `MANUALLY_EXCLUDED_DATASETS`).

## Scorer-augmentation families

The paper's three synthetic-scorer families (Sec. "Scorer Augmentation")
are specific cases of a more general dial-based construction in
`mwb/lib/synthetic_scorers.py`:

| Paper family | Implementation |
|---|---|
| MQM-adherence | `base_scorer_family_additive_mean` dialed on the AllMQM ("T") aspect |
| Adequacy–fluency | `base_scorer_family_additive_mean_af` (dial on adequacy, `-`dial on fluency, simultaneously) |
| Explainability-by-MQM | `base_scorer_family_joint` (the `'offset'` method's group-mean decomposition, conditioned on the *joint* (adequacy, fluency) pair) |

The paper's footnote to the MQM-adherence family also describes two more,
single-aspect variants it excludes from the main analysis ("the partial
correlation between the two aspects confounds their interpretation"):
`base_scorer_family_adequacy_adherence` and `base_scorer_family_fluency_adherence`,
the same construction retargeted at Adequacy MQM or Fluency MQM alone
instead of AllMQM. Supported as named functions for anyone who wants to
run that excluded comparison themselves; not wired into the `cli.py`
pipeline, since the paper reports no figure or table for them.

Every family shares a `dial=0` anchor (the real base scorer, unchanged);
larger `|dial|` moves further toward (or past) the family's target
endpoint. `mwb/lib/synthetic_scorer_preference.py`'s `preference_score`
is the shared reduction — the fraction of same-family dial pairs where a
given meta-metric prefers the more extreme dial — that all three of the
paper's family-specific measures (adequacy-over-fluency preference, MQM
adherence, explainability by MQM) reduce to.

## Solver validation

`mwb/lib/reweight_exact.py`'s solver is validated against
`mwb/lib/reweight_exhaustive.py`, a brute-force grid search with no
algorithmic structure to get wrong — a trust anchor, not a production
solver.

One subtlety when comparing the exact solver against the exhaustive one
directly: the exhaustive solver's accepted point is only ever within `tol`
of its *nominal* target β (a discrete lattice essentially never lands on
it exactly), and β(w) can be steep near the reachable-range boundary, so a
small residual gap there can correspond to a much easier point with
inflated ESS. A fair comparison re-solves the exact solver at the
exhaustive result's *actual achieved* β, not the nominal target, before
comparing.

## Citation

The paper has been accepted to WMT2026, but the proceedings aren't published
yet — this cites the arXiv preprint for now; once the ACL Anthology entry
exists, prefer that citation instead.

```bibtex
@misc{shayegh2026mindbirdfavourparameterizing,
      title={Mind Which Bird You Favour: Parameterizing Adequacy-Fluency Balance in Meta-Evaluation of Machine Translation}, 
      author={Behzad Shayegh and Niloofar Kazemi},
      year={2026},
      eprint={2609.14795},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.14795}, 
}
```

<a href="https://TheShayegh.github.io/"><img src="https://TheShayegh.github.io/img/favicon.png" style="background-color:red;"/></a>
