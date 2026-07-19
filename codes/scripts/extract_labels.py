"""Sanity-check codes/mwb/mqm_scoring's weighting/classification logic against
every unique (severity, category) label actually present in a raw MQM TSV,
for sets where we have no official score to validate against directly
(currently just generalMT2022/enzh -- excluded from SETS because its raw
TSV uses a different 12-column "WMT/Google" format: year, source_lang,
target_lang, document_id, doc_segment_id, system_id, rater_id, source,
reference, candidate, category, severity -- not the 9/10-column Marot
format parse_mqm_tsv() expects).

For each label, reports the weight error_weight() assigns and what
classify_aspect() routes it to, flagging any label with nonzero weight that
classify_aspect() drops to None -- i.e. contributes to the "full" total but
silently vanishes from both a and b. That's the actual failure mode we can't
catch by diffing against an official score here, since none exists for this
set. (Nonzero-weight + None is not automatically wrong -- the bare "other"
category is legitimately excluded from the Adequacy/Fluency split while
still counting toward the total, confirmed against official data for other
sets -- but everything else in that state is worth a look.)
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mwb.mqm_scoring import error_weight, classify_aspect

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PATH = os.path.join(ROOT, 'external/wmt-mqm-human-evaluation/generalMT2022/enzh/mqm_generalMT2022_enzh.tsv')


if __name__ == '__main__':
  counts = Counter()
  with open(PATH, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    print('columns:', reader.fieldnames)
    for row in reader:
      counts[(row['severity'], row['category'])] += 1

  print(f"\n{len(counts)} unique (severity, category) labels, {sum(counts.values())} rows total\n")
  print(f"{'severity':12} {'category':45} {'count':>7} {'weight':>7}  {'aspect':10}")
  flagged = []
  for (severity, category), n in counts.most_common():
    w = error_weight(severity, category)
    aspect = classify_aspect(category) if w > 0 else '-'
    print(f"{severity:12} {category:45} {n:7d} {w:7.2f}  {str(aspect):10}")
    if w > 0 and aspect is None and category.strip().lower() != 'other':
      flagged.append((severity, category, n, w))

  print()
  if flagged:
    print(f"CHECK: {len(flagged)} label(s) have nonzero weight but no aspect (not 'other'):")
    for severity, category, n, w in flagged:
      print(f"  {severity} / {category}: {n} rows, weight {w}")
  else:
    print("OK: every nonzero-weight label routes to adequacy or fluency, except bare 'other' (expected).")
