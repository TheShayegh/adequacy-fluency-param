"""Static, hand-curated metadata about the WMT/mt-metrics-eval datasets and
systems this project processes -- as opposed to logic that computes
anything from MQM scores. Four small, previously-separate modules merged
here since none of them ever depended on each other or held more than a
lookup table plus a couple of pure functions over it:

- Dataset directories: DATASET_DIRS.
- System-name classification: is_reference_or_human, join_key.
- Scorer-filename parsing: split_variant.
- Hand-curated system-pool exclusions: wmt_official_outliers,
  wmt_non_competative_outliers.
"""

from __future__ import annotations

import re

# =====================================================================
# Dataset directories
# =====================================================================
#
# Dataset name -> (year dir, language-pair dir) under
# external/mt-metrics-eval-data/mt-metrics-eval-v2/, for the system-sets
# used throughout (src.lib.mqm_scoring.SETS, minus generalMT2022/enzh -- no
# official data of any kind exists for it, see mqm_scoring.py).

DATASET_DIRS = {
    'ende20': ('wmt20', 'en-de'),
    'zhen20': ('wmt20', 'zh-en'),
    'ende21': ('wmt21.news', 'en-de'),
    'zhen21': ('wmt21.news', 'zh-en'),
    'ted_ende': ('wmt21.tedtalks', 'en-de'),
    'ted_zhen': ('wmt21.tedtalks', 'zh-en'),
    'ende22': ('wmt22', 'en-de'),
    'zhen22': ('wmt22', 'zh-en'),
    'enru22': ('wmt22', 'en-ru'),
    'ende23': ('wmt23', 'en-de'),
    'zhen23': ('wmt23', 'zh-en'),
    'heen23': ('wmt23', 'he-en'),
    'ende24': ('wmt24', 'en-de'),
    'enes24': ('wmt24', 'en-es'),
    'jazh24': ('wmt24', 'ja-zh'),
}

# =====================================================================
# System-name classification
# =====================================================================
#
# e.g. flagging reference/human translation entries so system-level
# analyses (variance, beta, synthesis) can restrict to real MT systems,
# which is what the paper's system count K and score vectors a, f range
# over (translation systems are the objects being ranked, not references
# or human translations).

# Matches refA, refB, refC, refD, ref, refb, refp (mqm_scoring's
# _normalize_system_name/wmt20-alias-normalized forms) and Human-A,
# Human-B, Human-P (wmt20's raw names for the same reference/human
# translations) across every dataset this project covers. Does not match
# real systems: none of the wmt-mqm-human-evaluation system rosters contain
# "ref"/"human" as a substring of a real system's name (confirmed by
# inspection).
_REF_HUMAN_RE = re.compile(r'^(ref[a-z]*|human-.+)$', re.IGNORECASE)


def is_reference_or_human(system: str) -> bool:
  """True for reference/human-translation entries, as opposed to real MT
  systems."""
  return bool(_REF_HUMAN_RE.match(system.strip()))


def join_key(system: str) -> str:
  """Case/suffix-insensitive key for matching a system name across
  wmt-mqm-human-evaluation and mt-metrics-eval-v2/metric-scores naming,
  which disagree in two confirmed ways for wmt20: a trailing submission-id
  suffix ("Huoshan_Translate.832", same quirk mqm_scoring._normalize_
  system_name handles) and case ("UEdin" in the MQM data vs "UEDIN.1136" in
  metric-scores). Strips both; other years' names already match exactly, so
  this is a no-op there."""
  return re.sub(r'\.\d+$', '', system.strip()).lower()

# =====================================================================
# Scorer-filename parsing
# =====================================================================
#
# Parsing for mt-metrics-eval-v2's metric-score filenames, e.g.
# "COMET-DA_2021-refA.sys.score", "MetricX-23-QE-b-src.seg.score".
#
# The basename (minus the .sys.score/.seg.score suffix) is
# "<scorer-name>-<reference-variant>", where the reference variant is one
# of a small closed vocabulary: which reference the scorer was run against
# (refA/refB/refC/refD for wmt21+, ref/refb/refp/all for wmt20), or "src"
# for a reference-free (QE) scorer, or "synthetic_ref" (wmt23's paraphrased
# reference). Stripping that trailing token is the only *safe*
# normalization: scorer names themselves are not otherwise standardized
# across years (e.g. COMET-2R'20 / COMET-DA_2021 / COMET-20 / bare
# COMET'23 / COMET-22'24 are four distinct scored variants of an evolving
# model family, not the same file under a different name) -- so this
# doesn't attempt to unify those; call sites that want family-level
# grouping should do it explicitly and document the heuristic.

_VARIANT_SUFFIXES = (
    'refA', 'refB', 'refC', 'refD', 'refb', 'refp', 'ref', 'all', 'src',
    'synthetic_ref',
)


def split_variant(basename: str) -> tuple[str, str | None]:
  """Splits "<scorer>-<variant>" into (scorer, variant). variant is None if
  the trailing '-'-delimited token isn't in the known vocabulary (rare;
  such names are returned whole as the scorer with variant=None)."""
  if '-' in basename:
    scorer, _, tail = basename.rpartition('-')
    if tail in _VARIANT_SUFFIXES:
      return scorer, tail
  return basename, None

# =====================================================================
# Hand-curated system-pool exclusions
# =====================================================================
#
# NOT statistical detectors -- no (a,f) score ever enters either function
# below.
#
# wmt_official_outliers -- THE PROJECT DEFAULT, as of this session. It is
# baked directly into src.lib.consistency.real_systems() (exclude_outliers=
# True, the default), so every caller that gets a dataset's system list
# through real_systems() -- which is nearly the whole project -- already
# works on the wmt_official-screened pool without applying any screen
# itself. This INCLUDES real_systems() now returning an EMPTY list by
# default for the two datasets wmt_official treats as UNSUPPORTED (ende20,
# zhen20) -- callers that loop over all 15 datasets unconditionally must
# skip a dataset with 0 systems rather than assume every one of the 15 is
# non-empty (see the `if not systems: continue`-style guards in
# src/scripts/, and src.lib.consistency.real_systems' own docstring for
# exactly what "default" means and how to opt out with
# exclude_outliers=False). It is the WMT organizers'/mt-metrics-eval
# toolkit's OWN outlier_systems metadata (external/mt-metrics-eval/
# mt_metrics_eval/meta_info.py), which that toolkit excludes by default
# when building a system list for final ranking. 2020 datasets are treated
# as UNSUPPORTED here (entire roster dropped) rather than "verified clean,"
# since none of their officially-listed outliers are even present in this
# project's smaller raw-TSV-sourced 2020 subset -- see the function's own
# docstring.
#
# wmt_non_competative_outliers -- a fixed per-dataset lookup table
# (WMT_NON_COMPETATIVE_SYSTEMS) of WMT organizer-inserted systems --
# calibration placeholders and MBR-reranking baselines that are not
# independent MT submissions -- hand-curated from each dataset's own
# roster.
#
# A GK-robust (Gnanadesikan-Kettenring 1972) statistical outlier screen was
# also explored this project (joint-Mahalanobis and single-aspect
# modified-z-score variants) but never adopted -- wmt_official_outliers
# above is what every dataset actually uses. The GK-robust location/
# covariance estimator itself survives only as the ellipse-drawing helper
# in src/scripts/plot_af_scatter_heen23_jazh24.py, its one remaining
# caller.

# Hand-curated, not computed: WMT organizer-inserted systems per dataset --
# calibration placeholders (metricsystem1-5) and MBR-reranking baselines
# (M2M100_1.2B-B4, QUARTZ_TuneReranking, *_bestmbr, NLLB_Greedy/NLLB_MBR_BLEU)
# that are not independent MT submissions, plus MSLC (ende24/enes24/jazh24).
# Session origin: manual review of each dataset's real-system roster: a
# dataset absent here (e.g. ende20, zhen20) has none.
WMT_NON_COMPETATIVE_SYSTEMS: dict[str, frozenset[str]] = {
    'ende21': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'zhen21': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ted_ende': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ted_zhen': frozenset({'metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'}),
    'ende22': frozenset({'M2M100_1.2B-B4', 'QUARTZ_TuneReranking', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr'}),
    'zhen22': frozenset({'M2M100_1.2B-B4', 'bleu_bestmbr', 'bleurt_bestmbr', 'comet_bestmbr'}),
    'enru22': frozenset({'M2M100_1.2B-B4', 'QUARTZ_TuneReranking', 'bleu_bestmbr', 'comet_bestmbr'}),
    'ende23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'zhen23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'heen23': frozenset({'NLLB_Greedy', 'NLLB_MBR_BLEU'}),
    'ende24': frozenset({'MSLC'}),
    'enes24': frozenset({'MSLC'}),
    'jazh24': frozenset({'MSLC'}),
}


def wmt_non_competative_outliers(dataset: str, systems: list[str] | None = None) -> set[str]:
  """NOT a statistical detector -- looks up WMT_NON_COMPETATIVE_SYSTEMS[dataset]
  (empty set for a dataset not in the table, e.g. ende20/zhen20; not an
  error). If `systems` is given, validates every looked-up name is
  actually in it (raises ValueError listing any that aren't -- a
  mismatch here means the roster changed or the table has a typo, not
  something to silently paper over)."""
  ref = set(WMT_NON_COMPETATIVE_SYSTEMS.get(dataset, frozenset()))
  if systems is not None:
    missing = ref - set(systems)
    if missing:
      raise ValueError(f'{dataset}: wmt_non_competative_outliers names not found in `systems`: {sorted(missing)}')
  return ref


# Hand-curated: the WMT organizers'/mt-metrics-eval toolkit's OWN
# outlier_systems metadata (external/mt-metrics-eval/mt_metrics_eval/
# meta_info.py's DATA[key][pair].outlier_systems), restricted to entries
# that are actually present in this project's own real_systems() roster
# per dataset -- most official names aren't (verified empirically): wmt23
# en-de/zh-en's official outlier is 'synthetic_ref', which never surfaces
# as a system in this project's pipeline at all; wmt20's official outliers
# are numeric-ID systems (e.g. 'zlabs-nlp.179') absent from this project's
# smaller raw-TSV-sourced 2020 subset. A dataset absent here has either no
# official outlier or one not present in our data.
WMT_OFFICIAL_SYSTEMS: dict[str, frozenset[str]] = {
    'ende22': frozenset({'M2M100_1.2B-B4'}),
    'ende24': frozenset({'MSLC'}),
    'enes24': frozenset({'MSLC'}),
    'jazh24': frozenset({'MSLC'}),
}

# 2020 is UNSUPPORTED in wmt_official_outliers (see its docstring): none of
# the official outliers for wmt20 en-de/zh-en are present in our roster, so
# returning WMT_OFFICIAL_SYSTEMS.get(dataset, frozenset()) as-is would give
# an empty set indistinguishable from "verified clean" -- which it isn't.
WMT_OFFICIAL_UNSUPPORTED = frozenset({'ende20', 'zhen20'})


def wmt_official_outliers(dataset: str, systems: list[str] | None = None) -> set[str]:
  """THE PROJECT DEFAULT, as of this session -- baked directly into
  src.lib.consistency.real_systems() (exclude_outliers=True, its default), so
  this is applied automatically almost everywhere in the project without
  any caller having to call it explicitly. NOT a statistical detector --
  the WMT organizers'/mt-metrics-eval toolkit's OWN outlier_systems
  metadata (WMT_OFFICIAL_SYSTEMS), which that toolkit excludes by default
  when building a system list for final ranking. Cross-checked against
  this project's own real_systems(..., exclude_outliers=False) roster per
  dataset; WMT_OFFICIAL_SYSTEMS lists only entries verified present.

  2020 (WMT_OFFICIAL_UNSUPPORTED: ende20, zhen20) is UNSUPPORTED in this
  mode: the officially-flagged 2020 outliers never appear in this
  project's roster at all, so applying the usual rule would silently
  return an empty set indistinguishable from "verified clean" -- instead
  this drops the ENTIRE roster for 2020 rather than silently returning
  "verified clean"; `systems` is REQUIRED for these two, raises ValueError
  if not given.

  For every other dataset, validates any looked-up name against `systems`
  the same way as wmt_non_competative_outliers if given."""
  if dataset in WMT_OFFICIAL_UNSUPPORTED:
    if systems is None:
      raise ValueError(f'{dataset}: wmt_official_outliers treats 2020 datasets as unsupported and '
                        'drops the entire roster -- pass `systems` explicitly to get that set.')
    return set(systems)
  ref = set(WMT_OFFICIAL_SYSTEMS.get(dataset, frozenset()))
  if systems is not None:
    missing = ref - set(systems)
    if missing:
      raise ValueError(f'{dataset}: wmt_official_outliers names not found in `systems`: {sorted(missing)}')
  return ref
