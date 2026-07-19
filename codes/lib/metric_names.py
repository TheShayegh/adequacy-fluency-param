"""Parsing for mt-metrics-eval-v2's metric-score filenames, e.g.
"COMET-DA_2021-refA.sys.score", "MetricX-23-QE-b-src.seg.score".

The basename (minus the .sys.score/.seg.score suffix) is
"<metric-name>-<reference-variant>", where the reference variant is one of
a small closed vocabulary: which reference the metric was run against
(refA/refB/refC/refD for wmt21+, ref/refb/refp/all for wmt20), or "src" for
a reference-free (QE) metric, or "synthetic_ref" (wmt23's paraphrased
reference). Stripping that trailing token is the only *safe* normalization:
metric names themselves are not otherwise standardized across years (e.g.
COMET-2R'20 / COMET-DA_2021 / COMET-20 / bare COMET'23 / COMET-22'24 are
four distinct scored variants of an evolving model family, not the same
file under a different name) -- so this module does not attempt to unify
those; call sites that want family-level grouping should do it explicitly
and document the heuristic.
"""

from __future__ import annotations

_VARIANT_SUFFIXES = (
    'refA', 'refB', 'refC', 'refD', 'refb', 'refp', 'ref', 'all', 'src',
    'synthetic_ref',
)


def split_variant(basename: str) -> tuple[str, str | None]:
  """Splits "<metric>-<variant>" into (metric, variant). variant is None if
  the trailing '-'-delimited token isn't in the known vocabulary (rare;
  such names are returned whole as the metric with variant=None)."""
  if '-' in basename:
    metric, _, tail = basename.rpartition('-')
    if tail in _VARIANT_SUFFIXES:
      return metric, tail
  return basename, None
