"""dataset name -> (year dir, language-pair dir) under
external/mt-metrics-eval-data/mt-metrics-eval-v2/, for the same 15
system-sets used throughout (mwb.mqm_scoring.SETS, minus generalMT2022/
enzh -- no official data of any kind exists for it, see mqm_scoring.py)."""

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


def datasets_for_years(years) -> list[str]:
  """Dataset names whose year dir is exactly one of `years` (e.g. [2022,
  2023, 2024] or ['22','23','24'] or ['wmt22',...] all work) -- matches on
  the year dir's 'wmtNN' prefix, so 'wmt21.news'/'wmt21.tedtalks' both count
  as year 2021 (there's no bare 'wmt21' dataset)."""
  wanted = set()
  for y in years:
    y = str(y)
    if y.startswith('wmt'):
      wanted.add(y)
    else:
      wanted.add(f'wmt{y[-2:]}')
  return [
      name for name, (year_dir, _pair) in DATASET_DIRS.items()
      if year_dir.split('.')[0] in wanted
  ]


def datasets_for_pair(pair: str) -> list[str]:
  """Dataset names whose language-pair dir is exactly `pair` (e.g.
  'en-de'), across all years/domains -- e.g. for 'en-de' this includes
  ted_ende (the TED-talks domain, not just the news-task sets), since it's
  still the same source/target language pair."""
  return [name for name, (_year_dir, p) in DATASET_DIRS.items() if p == pair]
