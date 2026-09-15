"""System-name classification helpers -- e.g. flagging reference/human
translation entries so system-level analyses (variance, alpha/beta,
synthesis) can restrict to real MT systems, which is what the paper's
system count K and score vectors a, b range over (translation systems are
the objects being ranked, not references or human translations)."""

import re

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
