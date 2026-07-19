# Data inventory, formats, and scoring practice

Scope: no metric training, no heavy model downloads. Data lives under
`external/` (third-party checkouts); code lives under `codes/`.

## 1. Sources (`external/`)

- **`external/wmt-mqm-human-evaluation/`** — raw segment-level MQM error
  annotations (category + severity per error span) for 16 real system-sets,
  WMT2020-2024, K=10-19 systems each.
- **`external/mt-metrics-eval/`** — Google's WMT Metrics Shared Task
  toolkit. We use one script directly: `mt_metrics_eval/converters/
  score_mqm.py`, the documented converter from the raw TSVs above into
  official score files. We don't install the package itself (`stats.py`
  needs Python ≥3.10; our venv is 3.9).
- **`external/mt-metrics-eval-data/mt-metrics-eval-v2/`** — the official
  data bundle (`https://data.statmt.org/wmt26/mt-metrics-eval-v2.tgz`,
  ~870MB full archive). Extracted so far: `human-scores/`, `documents/`,
  `sources/`, `metric-scores/` for `wmt20, wmt21.news, wmt21.tedtalks,
  wmt22, wmt23, wmt24` (other years in the tgz not extracted). Real ground
  truth — official MQM scores — used to validate our own scoring, not a
  reconstruction of it.

## 2. Raw TSV formats (`wmt-mqm-human-evaluation`)

Three distinct formats appear across the 16 sets. `codes/mwb/mqm_scoring.
parse_mqm_tsv()` handles the first two (auto-detects columns via
`_doc_id_col`/`_seg_id_col`); the third is unsupported.

- **"Marot" 9/10-column** (most sets): `system doc doc_id|docSegId
  seg_id|globalSegId rater source target category severity [metadata]`.
  2020/2021/2022 sets use `doc_id`/`seg_id`; 2023/2024 sets use
  `docSegId`/`globalSegId` (+ a JSON `metadata` column, populated for 2023
  sets, empty `{}` for 2024 sets).
- **"WMT/Google" 12-column** (`generalMT2022/enzh` only — the one set
  excluded from `SETS`): `year source_lang target_lang document_id
  doc_segment_id system_id rater_id source reference candidate category
  severity`. Not a broken file, just a different, unsupported schema; no
  official rating data exists for it either, so it stays out.
- Row granularity: one row per marked error span; a clean segment still
  gets one `No-error`/`No-error` row, so segment presence can be detected
  without special-casing.

### `doc_id`/`docSegId` indexing — a real gotcha

1-indexed (starts at 1) in every set **except `generalMT2022/enru`, which
is 0-indexed** (starts at 0). Confirmed by inspecting raw values directly
(`external/wmt-mqm-human-evaluation/generalMT2022/enru/...`: the first
segment of doc `t1_hqh0ref` has `doc_id=0`, not `1`). Doesn't affect
`mqm_scoring.py`'s actual scoring (it groups by whatever `doc_id` values
appear, never assumes an indexing convention) — but it silently breaks any
ad hoc script that reconstructs segment identity by counting occurrences
starting from 1 (as several of our validation scripts did before this was
found). Check raw values before writing new alignment code for any set.

### Category taxonomies (three variants, auto-detected by `classify_aspect`)

- **Hierarchical**, slash-separated: `Accuracy/Mistranslation`,
  `Fluency/Punctuation`, `Style/Unnatural or awkward`, `Locale
  convention/Currency format`, `Terminology/Inconsistent`,
  `Non-translation!`. Used by en-de, zh-en, ja-zh, and all pre-2023 sets.
- **Flat**, lowercase underscore-separated, no hierarchy:
  `do_not_translate`, `mt_hallucination`. Used by en-es'24, en-ru'22,
  he-en'23.
- **Mixed** (`generalMT2022/enzh` only): hierarchical-looking but with
  underscores instead of spaces before the slash, e.g.
  `locale_convention/name`, `non_translation` (vs. the usual `Locale
  convention/Name format`, `Non-translation!`). This is why
  `classify_aspect` normalizes underscores to spaces *before* checking for
  `/`, universally — an earlier version only normalized in the flat-taxonomy
  branch and silently dropped ~640 rows of `enzh22` data to unclassified.
- Severities seen: `Major`, `Minor`, `Neutral`, `No-error`, `Critical` (rare;
  en-es'24 has ~99 rows), `HOTW-test` (QC attention-check probe, paired
  with category `Found`/`Missed`, only in 2022/2023/2024 sets). Case varies
  by set (`Major` vs `major`); `error_weight`/`classify_aspect` lowercase
  everything.

## 3. `mt-metrics-eval-v2` official file formats

All files below are **positionally aligned**: line *i* of every file for a
given test-set + language-pair refers to the same segment, in the same
order as `sources/<pair>.txt` (one source segment per line — this is the
canonical position reference; use its line count as `n_positions`).

- **`human-scores/<pair>.mqm.sys.score`** — `system\tscore` per line
  (or `system score`, space-delimited, for wmt20 only — see below).
  `score` is `None` for systems in the language pair's full submission
  roster that were never selected for MQM human evaluation (WMT restricts
  human eval to a curated top-scoring subset, not all submissions — stated
  explicitly in Freitag et al. 2024 §3). This is why `K(official)` (all
  lines) can be much larger than `K(ours)` (only ever the MQM-rated
  subset) for some sets (e.g. en-de'24: 28 vs 19) — not a bug, not missing
  data, just a wider roster in this one file.
- **`human-scores/<pair>.mqm.seg.score`** — `system\tscore` per line,
  `n_positions` rows per system in file order, `None` for unrated
  positions. `sys.score` = mean of a system's non-`None` `seg.score`
  values, exactly (confirmed to the 4th decimal) — the official
  segment→system aggregation is a plain arithmetic mean, nothing fancier.
- **`human-scores/<pair>.mqm.merged.seg.rating`** — `system\t<json>\trater`
  per line, `n_positions` rows per system, `<json>` is `None` or
  `{"errors": [{"category":..., "severity":..., "score":..., "start":...,
  "end":..., "is_source_error": bool}, ...]}` — i.e. the *actual* per-error
  annotations with the official pre-computed `score` already attached.
  `is_source_error` is informational only (checked: excluding those errors
  makes no difference to totals); do not treat it as a filter.
  `codes/mwb/mqm_scoring.load_official_ratings()` reads this directly:
  route each error's `category` through `classify_aspect()`, sum `score`.
- **`human-scores/<pair>.round2.mqm.merged.seg.rating`,
  `...round3...`** (en-de/zh-en, wmt22 and wmt23 only) — **do not use.**
  Tested directly: averaging these in with the primary `merged` file as
  extra independent ratings makes the match against official scores worse,
  not better. They're some kind of targeted QC re-annotation on a subset of
  segments, not a full independent additional rating round — the exact
  combination rule (if any) that produces the true official aggregate isn't
  recoverable from what's published.
- **`human-scores/<pair>.mqm.raterN.seg.rating`** — individual-rater
  version of the merged file, same per-line format. Only source available
  for wmt20 (no `merged` file exists there); used as N independent ratings
  to average, same role as multiple raters in the raw-TSV path. Filename
  quirk: wmt20's `.sys.score` files use a hyphen (`mqm-rater1.sys.score`)
  but its `.seg.rating` files use a dot (`mqm.rater1.seg.rating`) — easy to
  typo.
- **`documents/<pair>.docs`** — `domain\tdocname` per line, positionally
  aligned to `sources/`. `docname` repeats contiguously for all segments of
  one document (no interleaving, checked). Cross-referencing this against
  our raw TSV's `doc` field to hand-build a position index is fragile (see
  `enru22` note below) — prefer `load_official_ratings`'s pure stream-index
  approach (no `.docs` file needed at all) wherever a `.seg.rating` file
  exists.
- **System-name mismatches between files** (all handled by
  `_normalize_system_name`, which currently only strips a trailing
  `.\d+`): wmt20's `.sys.score`/`.seg.rating` files append a numeric
  submission-id suffix (`Huoshan_Translate.832`); wmt20's `.sys.score` also
  renames the three human/reference translations to `ref`/`refb`/`refp`
  (confirmed by value against the README's published ranking: Human-B
  0.75 < Human-A 0.91 < Human-P 1.41, matching refb < ref < refp — the raw
  TSV and the `.seg.rating` files use `Human-A`/`Human-B`/`Human-P`
  directly, no renaming needed there). newstest2021's *raw TSV* (not the
  official files) uses `hyp.<name>`/`ref.<letter>` prefixes not present
  anywhere in the official files.

## 4. Structural data issues (real, found by direct diff — not guesses)

- **Disjoint reference sub-samples.** Some sets contain extra reference
  systems rated on a segment set that barely overlaps the main pool — e.g.
  newstest2021/en-de's `ref.C`/`ref.D` are rated on 475 segments never
  touched by the main 17-system pool's 527 segments. Left in, this biases
  that system's average toward whatever its private sub-sample happens to
  contain. `filter_modal_coverage()` handles it, but needs a *threshold*
  (≥50% of the max observed per-segment system count), not exact equality
  to the mode: an exact-mode version was tried first and found (by exact
  diff against official scores) to incorrectly drop individual,
  legitimately-rated segments whenever coverage varies by a little for
  unrelated reasons (e.g. wmt21.tedtalks has 13 segments at 13/14 = 93%
  coverage, which must stay, not get dropped for looking "off-mode").
- **Small per-cell data gaps in the official ratings themselves.** After
  the above fixes, a handful of specific (system, segment) cells still
  don't match official — checked directly, and it's not a weighting or
  aggregation issue: the official score at that exact cell simply isn't
  reconstructable from any available rating file (merged, round2, round3,
  or individual rater files all checked). Traced and quantified per set
  (as of this check): ende21 8 cells, zhen21 4, ted_ende 1, ted_zhen 38,
  ende22 11, zhen22 98, zhen23 290 — all out of 500-1900+ segments per set,
  i.e. a few percent at most. Excluding these specific cells from *both*
  our computation and official's own average (a fair like-for-like
  comparison) makes 5 of 7 sets match exactly and the other 2
  (zhen22, zhen23) match to within 0.008. This is real evidence the gap is
  external data incompleteness, not our bug — but it's a diagnostic
  finding, not something baked into the production pipeline (the specific
  bad-cell lists aren't hardcoded anywhere; they'd need re-deriving if
  ever needed again, e.g. via the method in this section).
- **`generalMT2022/enru`'s `.docs`-file position reconstruction is
  fragile and currently unresolved.** Source-text spot-checks proved a
  segment-count mismatch somewhere between `documents/en-ru.docs` and our
  raw TSV (after fixing the 0-indexing issue above, alignment still drifts
  starting partway through, confirmed by comparing actual source sentence
  text at the computed vs. true position). This only affects *our own
  validation scripts* trying to align at the individual-segment level for
  cross-checking purposes — `enru22`'s actual score computation
  (`SETS['enru22']`, the raw-TSV path) is unaffected and produces sane,
  usable output (K=16, 1282 segments/system). The aggregate comparison
  (`compare_official.py`, which needs no position reconstruction) shows a
  real but modest ~5% variance gap, similar in size to other secondary
  sets. Unresolved; not blocking.

## 5. Scoring practice

Implemented in `codes/mwb/mqm_scoring.py`; see its module docstring for the
authoritative, current description. Summary:

- **Preferred path: official per-error rating data**
  (`OFFICIAL_RATINGS` registry + `load_official_ratings()`), used for 12 of
  the 16 sets. Each error's official `score` is used directly — no
  re-derivation of weights — only routed through `classify_aspect()` for
  the Adequacy/Fluency split.
- **Fallback: raw-TSV path** (`parse_mqm_tsv()` + `error_weight()`), used
  for `enzh22` (no official data exists) and for `ende20`/`zhen20`/`enru22`
  (official rating data exists but was tested and found to match *worse*
  than the raw-TSV reconstruction for these three specifically — see §4).
- **Weighting** (`error_weight`, used by the fallback path) reproduces
  `score_mqm.py`'s default (Major=5, Minor=1, Neutral=0,
  Major/Non-translation!=25, Minor/Fluency/Punctuation=0.1), plus three
  confirmed overrides not expressible by plain path-matching: Critical=5
  (score_mqm.py's bare default has no entry for it at all); "Source
  issue"/"Source error" and any category containing "reinterpretation"
  score 0 regardless of severity; the Minor-Punctuation discount applies to
  any category containing "punctuation", not just the exact hierarchical
  path (needed for the flat taxonomy's bare `punctuation` category).
- **Adequacy/Fluency split** (`classify_aspect`, used by both paths)
  follows Flamich et al. (2025), "You Cannot Feed Two Birds with One
  Score" (arXiv:2503.24013), Appendix D — the primary source both Shayegh
  et al. (2025) and we depend on for this classification.
- **Aggregation**: per-segment score = sum of matched/official error
  weights (averaged across raters/rounds when more than one covers a
  segment); system-level score = plain mean over segments, after the
  modal-coverage filter (§4). Matches the WMT Metrics Shared Task's own
  stated definition (Freitag et al. 2024 §3.3).
- **No system is ever dropped** by any part of this pipeline, by design.

## 6. Validation

- **`codes/scripts/compare_official.py`** — our reconstructed system-level
  score vs. the official `.mqm.sys.score` file, for all 15 sets with one
  available (`enzh22` is the only set with neither official data nor a
  place in this comparison). 8/15 exact or floating-point-exact
  (ende20, zhen20, ted_ende, ende23, heen23, ende24, enes24, jazh24); the
  rest within a few percent, root-caused in §4 rather than left
  unexplained.
- **`codes/scripts/compare_wmt24_paper.py`** — our score vs. Table 5 of
  Freitag et al. (2024), "Are LLMs Breaking MT Metrics? Results of the
  WMT24 Metrics Shared Task" (aclanthology.org/2024.wmt-1.2), for all
  systems in En-De'24, En-Es'24, Ja-Zh'24. Matches to the paper's own
  rounding (2 decimal places) throughout.
- **`codes/scripts/extract_labels.py`** — for `enzh22` (no official data
  to diff against at all), lists every unique (severity, category) label
  and flags any with nonzero weight that `classify_aspect` drops to
  unclassified (a real bug when it found one — see §2's mixed-taxonomy
  note). Currently clean.

Preprocessing is validated from three independent angles (official data
file, published paper table, our own code) that agree on the primary sets,
plus explained residuals on the secondary ones. Good enough to build on for
early experiments; the small remaining per-cell gaps (§4) are worth
revisiting for rigor when writing up final results, not before.

## 7. Divergence from Shayegh et al. (2025)

Shayegh et al.'s own Table 2 (system-level Adequacy/Fluency variance) could
not be reproduced from either their paper or Flamich et al. (2025), despite
extensive triangulation (three independent implementations agreeing with
each other and with official WMT ground truth, all disagreeing with Table 2
by set-specific, non-uniform margins). Not treated as our problem to solve:
our numbers are validated against the real official WMT pipeline and a
peer-reviewed published table (§6), a stronger standard than reproducing an
unreproducible table. Not a blocker for further work.
