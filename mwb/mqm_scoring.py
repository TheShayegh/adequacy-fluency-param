"""Turn raw wmt-mqm-human-evaluation TSVs into per-system Adequacy/Fluency/All MQM vectors.

Weighting and aggregation reproduce the actual standard: mt-metrics-eval's
`score_mqm.py` (external/mt-metrics-eval/mt_metrics_eval/converters/
score_mqm.py), the documented converter from these same raw
wmt-mqm-human-eval TSVs into the score files bundled in mt-metrics-eval-v2 --
the package behind the WMT Metrics Shared Task papers (e.g. Freitag et al.
2024, "Are LLMs Breaking MT Metrics? Results of the WMT24 Metrics Shared
Task", https://aclanthology.org/2024.wmt-1.2/, whose Table 3 states the same
weight table).

Validated by direct comparison against the official per-segment/per-system
score files (external/mt-metrics-eval-data/.../human-scores/*.mqm.{seg,sys}.
score) across every system-set with an official file available (see
mwb/scripts/validate_scores.py): exact to 4 decimal places for most sets,
small explained residuals (a document needing multi-rater data our
single-rater default TSV lacks; a couple of raw-data naming quirks) for the
rest.

Weighting algorithm (error_weight): for each error, build
[severity, category-path-parts...] (lowercased) and try progressively
shorter joins as a key into DEFAULT_WEIGHTS, most specific first, defaulting
to 0. Base weights: Major=5, Minor=1, Neutral=0, Critical=5,
Major/Non-translation!=25, Minor/Fluency/Punctuation=0.1 -- plus three
overrides not expressible by plain path-matching, all confirmed by exact
diff against official per-segment scores: "Source issue"/"Source error" and
any category containing "reinterpretation" score 0 regardless of severity;
the Minor-Punctuation discount applies to any category containing
"punctuation" (not just the exact hierarchical path), so the flat taxonomy's
bare "punctuation" category gets it too.

Adequacy/Fluency category classification (classify_aspect) is auto-detected
per row, since the raw data mixes two MQM taxonomies across years/pairs:

  - "Hierarchical" (en-de, zh-en, ja-zh, and all pre-2023 sets): categories
    like "Accuracy/Mistranslation", "Fluency/Punctuation". Prefix-matched:
    Accuracy/* (+ "Non-translation!") -> adequacy; Fluency/*, Style/*,
    Locale convention/*, Terminology/Inconsistent*, Terminology/
    Inappropriate* -> fluency. Source: Flamich et al. (2025) "You Cannot
    Feed Two Birds with One Score" (arXiv:2503.24013), Appendix D, Table 1
    -- this differs from Shayegh et al. (2025)'s own reformatted LaTeX
    table, which is not our problem to reconcile; Flamich et al. is the
    primary source both papers depend on for this classification.

  - "Flat" (en-es'24, en-ru'22, he-en'23, and presumably others): lowercase,
    underscore-separated categories with no hierarchy, e.g.
    "do_not_translate". Matched by set membership; source for en-es is
    Flamich et al. (2025), Appendix D, Table 2; other pairs' categories not
    covered by either paper are classified by analogy (documented inline
    at _FLAT_ADEQUACY / _FLAT_FLUENCY).

The "Other" bucket (excluded from both a and f, per the All MQM = Adequacy
MQM + Fluency MQM simplification) is: "Other", "No-error", and the
HOTW-test QC-probe categories ("Found"/"Missed"). Note "Source issue"/
"Source error" fall here too, but via error_weight's zero-weight override
(they never accumulate any weight to classify in the first place), not via
classify_aspect.

Two real gotchas worth knowing before touching this file:

  - doc_id/seg_id indexing is 1-indexed in every set EXCEPT
    generalMT2022/enru, which is 0-indexed. Doesn't affect scoring (rows are
    grouped by whatever doc_id/seg_id values appear, never assumed to start
    at a fixed index), but would silently break any code that reconstructs
    segment identity by counting occurrences from a fixed start.
  - generalMT2022/enzh mixes taxonomy conventions (hierarchical-looking
    categories but with underscores instead of spaces before the slash,
    e.g. "locale_convention/name", "non_translation") -- this is why
    classify_aspect normalizes underscores to spaces universally, not just
    for the flat-taxonomy branch (an earlier version that normalized only
    in the flat branch silently dropped ~640 rows of this set's data to
    unclassified).
"""

from __future__ import annotations

import csv
import functools
import json
import os
import re
from collections import defaultdict

import pandas as pd

# --- Weighting -------------------------------------------------------------
# score_mqm.py's exact default --weights string, parsed the same way.
DEFAULT_WEIGHTS = {
    'major': 5.0,
    'minor': 1.0,
    'neutral': 0.0,
    'critical': 5.0,
    'major/non-translation!': 25.0,
    'minor/fluency/punctuation': 0.1,
}


# Categories confirmed (against official per-segment scores, see module
# docstring) to score 0 regardless of severity, which score_mqm.py's plain
# path-matching wouldn't catch (they'd fall back to bare Major/Minor).
_ZERO_WEIGHT_CATEGORIES = {'source issue', 'source error'}


def error_weight(severity: str, category: str, weights=DEFAULT_WEIGHTS) -> float:
  """score_mqm.py's Score() (most-specific severity/category path match,
  defaulting to 0), plus the confirmed overrides."""
  lcat = (category or '').lower().strip()
  if lcat.replace('_', ' ') in _ZERO_WEIGHT_CATEGORIES or 'reinterpretation' in lcat:
    return 0.0
  if (severity or '').lower().strip() == 'minor' and 'punctuation' in lcat:
    return weights.get('minor/fluency/punctuation', 0.1)
  items = [(severity or '').lower()] + [c.lower() for c in (category or '').split('/')]
  while items:
    key = '/'.join(items)
    if key in weights:
      return weights[key]
    items = items[:-1]
  return 0.0


# --- Adequacy/Fluency category taxonomy -------------------------------------

_HIER_ADEQUACY_PREFIXES = ('accuracy/',)
_HIER_FLUENCY_PREFIXES = (
    'fluency/', 'style/', 'locale convention/',
    'terminology/inconsistent', 'terminology/inappropriate',
)
_NON_TRANSLATION = {'non-translation!', 'non-translation', 'non translation'}

# "Flat" taxonomy (en-es'24, en-ru'22, he-en'23, ...): matched after
# normalizing underscores to spaces.
_FLAT_ADEQUACY = {
    'addition', 'agreement', 'do not translate', 'mistranslation',
    'mt hallucination', 'omission', 'untranslated', 'wrong named entity',
    'wrong term', 'term not applied',
}
_FLAT_FLUENCY = {
    'capitalization', 'date time format', 'inconsistency', 'lacks creativity',
    'grammar', 'measurement format', 'number format', 'punctuation',
    'register', 'spelling', 'unnatural flow', 'whitespace', 'word order',
    'wrong language variety', 'markup tag', 'culture-specific reference',
    'currency format',
}


def classify_aspect(category: str) -> str | None:
  """Returns 'adequacy', 'fluency', or None (excluded / other / unrecognized).

  Underscores are normalized to spaces universally (not just for the flat
  taxonomy) before any check -- see module docstring's generalMT2022/enzh
  gotcha.
  """
  if not category:
    return None
  c = category.strip().lower().replace('_', ' ')
  if c in _NON_TRANSLATION:
    return 'adequacy'
  if '/' in c:
    if c.startswith(_HIER_ADEQUACY_PREFIXES):
      return 'adequacy'
    if c.startswith(_HIER_FLUENCY_PREFIXES):
      return 'fluency'
    return None
  if c in _FLAT_ADEQUACY:
    return 'adequacy'
  if c in _FLAT_FLUENCY:
    return 'fluency'
  return None


# --- Parsing -----------------------------------------------------------------

def _doc_id_col(fieldnames) -> str:
  for cand in ('doc_id', 'docSegId'):
    if cand in fieldnames:
      return cand
  raise ValueError(f'No doc-id column found among {fieldnames}')


def _seg_id_col(fieldnames) -> str:
  for cand in ('seg_id', 'globalSegId'):
    if cand in fieldnames:
      return cand
  raise ValueError(f'No seg-id column found among {fieldnames}')


def parse_mqm_tsv(path: str) -> pd.DataFrame:
  """Parses one raw MQM ratings TSV into per-(system,doc,seg,rater) scores.

  Returns a DataFrame with columns: system, doc, doc_id, seg_id, rater,
  adequacy, fluency, all_mqm (all_mqm = adequacy + fluency by construction,
  i.e. it already excludes the 'other' category errors -- see module
  docstring).
  """
  # accumulate weighted error sums per (system, doc, doc_id, seg_id, rater)
  acc = defaultdict(lambda: {'adequacy': 0.0, 'fluency': 0.0, 'full': 0.0})
  all_keys = set()
  with open(path, newline='') as fh:
    reader = csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)
    fieldnames = reader.fieldnames
    doc_id_col = _doc_id_col(fieldnames)
    seg_id_col = _seg_id_col(fieldnames)
    for row in reader:
      if not row['rater']:
        continue
      # Drop rows with missing source/target text outright, rather than
      # scoring them -- e.g. generalMT2024/ende has 8 rows (2 systems, 1
      # segment) where the system's output was recorded as an empty
      # string, and the exported target text is blank on every error row
      # tied to that (system, segment). We can't just zero-fill: the
      # annotator still logged errors (e.g. Accuracy/Omission) against a
      # target we can't verify, and any zero-fill or blind acceptance
      # would either fabricate or arbitrarily distort that (system,
      # segment) cell. Since the key is never added to all_keys, this
      # segment simply disappears for the affected system, and
      # filter_modal_coverage() below then drops the *segment* for every
      # system (not just the affected one) once coverage is uneven --
      # this excludes a data point, never a whole system.
      if not row.get('target', '').strip() or not row.get('source', '').strip():
        continue
      key = (_normalize_system_name(row['system']), row['doc'], row[doc_id_col], row[seg_id_col], row['rater'])
      all_keys.add(key)
      severity = row['severity']
      category = row['category']
      if not severity or not category:
        continue
      w = error_weight(severity, category)
      if w == 0.0:
        continue
      acc[key]['full'] += w
      aspect = classify_aspect(category)
      if aspect is None:
        continue
      acc[key][aspect] += w

  records = []
  for key, scores in acc.items():
    system, doc, doc_id, seg_id, rater = key
    records.append({
        'system': system, 'doc': doc, 'doc_id': doc_id, 'seg_id': seg_id,
        'rater': rater, 'adequacy': -scores['adequacy'], 'fluency': -scores['fluency'],
        'full': -scores['full'],
    })
  # also need zero-score rows for (system, doc, seg, rater) combos that had
  # *no* matching errors at all (they were never inserted into acc, e.g. an
  # all-No-error segment).
  for key in all_keys - set(acc.keys()):
    system, doc, doc_id, seg_id, rater = key
    records.append({
        'system': system, 'doc': doc, 'doc_id': doc_id, 'seg_id': seg_id,
        'rater': rater, 'adequacy': -0.0, 'fluency': -0.0, 'full': -0.0,
    })

  df = pd.DataFrame.from_records(records)
  df['all_mqm'] = df['adequacy'] + df['fluency']
  return df


def filter_modal_coverage(per_segment: pd.DataFrame, min_fraction: float = 0.5) -> pd.DataFrame:
  """Keeps only segments rated by at least min_fraction of the max observed
  per-segment system count.

  Some sets contain a stray sub-sample rated by only a handful of systems
  disjoint from the main pool -- e.g. newstest2021/ende has 475 segments
  rated only by two extra reference translations (ref.C, ref.D: coverage
  2/17 = 12%), never by the main 17-system pool, alongside 527 segments
  rated by all 17. Left in, these create a ragged (system x segment) design
  and bias whichever system's private sub-sample happens to be
  easier/harder. This filter drops them.

  Uses a fraction-of-max threshold rather than exact-match-to-the-mode: a
  strict "count == mode" version was tried first and found (by exact diff
  against official scores) to incorrectly drop individual, legitimately-
  rated segments in the official-ratings path whenever coverage varies by a
  little for reasons unrelated to the disjoint-sub-sample problem (e.g. one
  unrelated system's rating happens to be missing at a position:
  ted_ende/wmt21.tedtalks has 13 segments at 13/14 = 93% coverage, which
  must stay). 50% cleanly separates both observed cases; it's a no-op on
  the many sets that are already fully rectangular.
  """
  counts = per_segment.groupby(['doc', 'doc_id', 'seg_id'])['system'].transform('count')
  return per_segment[counts >= counts.max() * min_fraction]


def per_segment_scores(seg_df: pd.DataFrame) -> pd.DataFrame:
  """Collapses a per-(system,doc,seg,rater) DataFrame to per-(system,doc,seg),
  averaging across raters within a segment and applying the modal-coverage
  filter (see filter_modal_coverage). This is the system x segment level --
  the shared basis for both system_level_scores() (mean over segments) and
  per-segment cross-system analyses, so both operate on the identical,
  already-filtered set of (system, segment) cells.
  """
  cols = ['adequacy', 'fluency', 'all_mqm'] + (['full'] if 'full' in seg_df.columns else [])
  per_segment = (
      seg_df.groupby(['system', 'doc', 'doc_id', 'seg_id'])[cols]
      .mean()
      .reset_index()
  )
  return filter_modal_coverage(per_segment)


def system_level_scores(seg_df: pd.DataFrame) -> pd.DataFrame:
  """Collapses a per-(system,doc,seg,rater) DataFrame to per-system a,f,t.

  Averages across raters within a segment first, applies the modal-coverage
  filter, then averages across segments within a system (uniform weight per
  segment, matching the paper's system-level mean).
  """
  cols = ['adequacy', 'fluency', 'all_mqm'] + (['full'] if 'full' in seg_df.columns else [])
  per_segment = per_segment_scores(seg_df)
  per_system = (
      per_segment.groupby('system')[cols]
      .mean()
      .rename(columns={'adequacy': 'a', 'fluency': 'f', 'all_mqm': 't'})
  )
  n_segments = per_segment.groupby('system').size().rename('n_segments')
  return per_system.join(n_segments)


# --- Official-ratings loading path (preferred ground truth) -----------------
#
# Standard: wherever mt-metrics-eval-v2 has official per-error rating data
# (*.mqm.merged.seg.rating, or *.mqm.raterN.seg.rating when no merged file
# exists), we use it as ground truth directly, instead of recomputing scores
# from the raw wmt-mqm-human-eval TSV with our own error_weight(). Each
# error in these files already carries the official "score" field; we only
# need to route it through classify_aspect() to get the Adequacy/Fluency
# split. For wmt20 (no merged file, only per-rater files) each rater file is
# treated as one more independent rating to average over, same as the
# raw-TSV path already does for multi-rater segments.
#
# The *.roundN.mqm.merged.seg.rating files (en-de/zh-en, wmt22 and wmt23)
# are deliberately NOT used: tested and found to make the match against
# official scores worse, not better, when averaged in alongside the primary
# merged file -- they're some kind of targeted QC re-annotation on a subset
# of segments, not an independent full additional rating.
#
# The raw-TSV path (error_weight() + parse_mqm_tsv()) remains in use only
# for generalMT2022/enzh (SETS excludes it -- malformed raw TSV, and no
# official rating data exists to fall back on either).

_ROOT_DATA = 'external/mt-metrics-eval-data/mt-metrics-eval-v2'

OFFICIAL_RATINGS = {
    # ende20/zhen20 deliberately NOT here despite having official per-rater
    # rating files: tested, and the raw-TSV path already matches the
    # official sys.score exactly, while the rating-file path does not (some
    # positions' official scores don't trace to data present in *any*
    # available per-rater file for this set, checked directly -- a genuine
    # gap in what's available, not a parsing bug). Same story for enru22
    # (also has rater1-4 files plus merged): 100 positions where the
    # official seg.score is non-null but every available rating file
    # (merged and all 4 individual raters) shows None at that position.
    'ende21': {
        'sources': f'{_ROOT_DATA}/wmt21.news/sources/en-de.txt',
        'ratings': [f'{_ROOT_DATA}/wmt21.news/human-scores/en-de.mqm.merged.seg.rating'],
    },
    'zhen21': {
        'sources': f'{_ROOT_DATA}/wmt21.news/sources/zh-en.txt',
        'ratings': [f'{_ROOT_DATA}/wmt21.news/human-scores/zh-en.mqm.merged.seg.rating'],
    },
    'ted_ende': {
        'sources': f'{_ROOT_DATA}/wmt21.tedtalks/sources/en-de.txt',
        'ratings': [f'{_ROOT_DATA}/wmt21.tedtalks/human-scores/en-de.mqm.merged.seg.rating'],
    },
    'ted_zhen': {
        'sources': f'{_ROOT_DATA}/wmt21.tedtalks/sources/zh-en.txt',
        'ratings': [f'{_ROOT_DATA}/wmt21.tedtalks/human-scores/zh-en.mqm.merged.seg.rating'],
    },
    'ende22': {
        'sources': f'{_ROOT_DATA}/wmt22/sources/en-de.txt',
        'ratings': [f'{_ROOT_DATA}/wmt22/human-scores/en-de.mqm.merged.seg.rating'],
    },
    'zhen22': {
        'sources': f'{_ROOT_DATA}/wmt22/sources/zh-en.txt',
        'ratings': [f'{_ROOT_DATA}/wmt22/human-scores/zh-en.mqm.merged.seg.rating'],
    },
    'ende23': {
        'sources': f'{_ROOT_DATA}/wmt23/sources/en-de.txt',
        'ratings': [f'{_ROOT_DATA}/wmt23/human-scores/en-de.mqm.merged.seg.rating'],
    },
    'zhen23': {
        'sources': f'{_ROOT_DATA}/wmt23/sources/zh-en.txt',
        'ratings': [f'{_ROOT_DATA}/wmt23/human-scores/zh-en.mqm.merged.seg.rating'],
    },
    'ende24': {
        'sources': f'{_ROOT_DATA}/wmt24/sources/en-de.txt',
        'ratings': [f'{_ROOT_DATA}/wmt24/human-scores/en-de.mqm.merged.seg.rating'],
    },
    'enes24': {
        'sources': f'{_ROOT_DATA}/wmt24/sources/en-es.txt',
        'ratings': [f'{_ROOT_DATA}/wmt24/human-scores/en-es.mqm.merged.seg.rating'],
    },
    'jazh24': {
        'sources': f'{_ROOT_DATA}/wmt24/sources/ja-zh.txt',
        'ratings': [f'{_ROOT_DATA}/wmt24/human-scores/ja-zh.mqm.merged.seg.rating'],
    },
    # heen23 has an official sys/seg.score but no per-error .seg.rating file
    # to re-split into Adequacy/Fluency, so it stays on the raw-TSV path
    # (already an exact match against the official total -- the weighting
    # is validated regardless, see mwb/scripts/validate_scores.py).
}


def _normalize_system_name(name: str) -> str:
  """Strips raw-data naming quirks seen across years so system names line up
  with the official score files' canonical naming (e.g. wmt20's
  "AFRL.1069" -> "AFRL")."""
  return re.sub(r'\.\d+$', '', name)


def load_official_ratings(rating_paths: list[str], n_positions: int) -> pd.DataFrame:
  """Parses one or more positionally-aligned *.seg.rating files (each file
  is one "round" -- an additional independent rating to average over, same
  role as an additional rater), routing each error's own official "score"
  through classify_aspect(). Returns per-(system, position, round) rows with
  columns system, doc, doc_id, seg_id, adequacy, fluency, all_mqm, shaped to
  drop straight into system_level_scores() (doc/doc_id are dummies; seg_id
  is the stream position, unique within doc).
  """
  records = []
  for round_idx, path in enumerate(rating_paths):
    sys_lines = defaultdict(list)
    with open(path) as fh:
      for line in fh:
        parts = line.rstrip('\n').split('\t')
        sys_lines[_normalize_system_name(parts[0])].append(parts[1] if len(parts) > 1 else 'None')
    for sysname, lines in sys_lines.items():
      if len(lines) != n_positions:
        raise ValueError(f'{path}: system {sysname} has {len(lines)} rows, expected {n_positions}')
      for pos, errors_json in enumerate(lines):
        if errors_json in ('None', 'null', ''):
          continue
        a_sum = f_sum = full_sum = 0.0
        for e in json.loads(errors_json).get('errors', []):
          aspect = classify_aspect(e['category'])
          score = e.get('score') or 0.0
          full_sum += score
          if aspect == 'adequacy':
            a_sum += score
          elif aspect == 'fluency':
            f_sum += score
        records.append({
            'system': sysname, 'doc': 'pos', 'doc_id': '0', 'seg_id': str(pos),
            'rater': f'round{round_idx}', 'adequacy': -a_sum, 'fluency': -f_sum,
            'full': -full_sum,
        })
  df = pd.DataFrame.from_records(records)
  df['all_mqm'] = df['adequacy'] + df['fluency']
  return df


def _count_lines(path: str) -> int:
  with open(path) as fh:
    return sum(1 for _ in fh)


# All 15 system-sets this project covers. generalMT2022/enzh is excluded:
# its raw TSV has a different/broken column layout, and no official rating
# data exists for it either.
SETS = {
    'ende23': 'external/wmt-mqm-human-evaluation/generalMT2023/ende/mqm_generalMT2023_ende.tsv',
    'zhen23': 'external/wmt-mqm-human-evaluation/generalMT2023/zhen/mqm_generalMT2023_zhen.tsv',
    'ende24': 'external/wmt-mqm-human-evaluation/generalMT2024/mqm_generalMT2024_ende.tsv',
    'enes24': 'external/wmt-mqm-human-evaluation/generalMT2024/mqm_generalMT2024_enes.tsv',
    'jazh24': 'external/wmt-mqm-human-evaluation/generalMT2024/mqm_generalMT2024_jazh.tsv',
    'heen23': 'external/wmt-mqm-human-evaluation/generalMT2023/heen/mqm_generalMT2023_heen.tsv',
    'ende22': 'external/wmt-mqm-human-evaluation/generalMT2022/ende/mqm_generalMT2022_ende.tsv',
    'zhen22': 'external/wmt-mqm-human-evaluation/generalMT2022/zhen/mqm_generalMT2022_zhen.tsv',
    'enru22': 'external/wmt-mqm-human-evaluation/generalMT2022/enru/mqm_generalMT2022_enru.tsv',
    'ende20': 'external/wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.tsv',
    'zhen20': 'external/wmt-mqm-human-evaluation/newstest2020/zhen/mqm_newstest2020_zhen.tsv',
    'ende21': 'external/wmt-mqm-human-evaluation/newstest2021/ende/mqm_newstest2021_ende.tsv',
    'zhen21': 'external/wmt-mqm-human-evaluation/newstest2021/zhen/mqm_newstest2021_zhen.tsv',
    'ted_ende': 'external/wmt-mqm-human-evaluation/ted/ende/mqm_ted_ende.tsv',
    'ted_zhen': 'external/wmt-mqm-human-evaluation/ted/zhen/mqm_ted_zhen.tsv',
}


@functools.lru_cache(maxsize=None)
def _load_seg_df(set_name: str, root: str = '.') -> pd.DataFrame:
  """Shared first step of load_system_scores/load_segment_scores: parses the
  set's per-(system,doc,seg,rater) rows via whichever source (official
  ratings or raw TSV) load_system_scores() would use for it.

  Cached (this is the expensive parse -- raw-TSV sets go through parse_mqm_
  tsv's per-row weighting/classification, official-ratings sets through
  load_official_ratings's per-row rating merge): profiling a single 21-beta
  curve over just 2 datasets found this step re-invoked 8 times for the same
  (set_name, root) pairs (real_systems/load_system_scores have no caching of
  their own and are each called independently by multiple call sites --
  solve_w_for_datasets, load_scorer_inputs, etc.), accounting for 92% of
  that run's total wall time. Callers only ever read the returned frame
  (slice via .loc, groupby, etc., never assign into it in place), so caching
  the shared object is safe -- verified by checking every load_system_
  scores/load_segment_scores call site in the project."""
  if set_name in OFFICIAL_RATINGS:
    spec = OFFICIAL_RATINGS[set_name]
    sources_path = os.path.join(root, spec['sources'])
    rating_paths = [os.path.join(root, p) for p in spec['ratings'] if os.path.exists(os.path.join(root, p))]
    if not rating_paths:
      raise FileNotFoundError(f'{set_name}: none of {spec["ratings"]} exist under root={root!r}')
    n_positions = _count_lines(sources_path)
    return load_official_ratings(rating_paths, n_positions)
  path = os.path.join(root, SETS[set_name])
  return parse_mqm_tsv(path)


def load_system_scores(set_name: str, root: str = '.') -> pd.DataFrame:
  return system_level_scores(_load_seg_df(set_name, root))


def load_segment_scores(set_name: str, root: str = '.') -> pd.DataFrame:
  """Per-(system, doc, doc_id, seg_id) adequacy/fluency, rater-averaged and
  modal-coverage-filtered -- the system x segment level, before collapsing
  to per-system means."""
  return per_segment_scores(_load_seg_df(set_name, root))
