"""Dataset name -> (year dir, language-pair dir) under
external/mt-metrics-eval-data/mt-metrics-eval-v2/, for the system-sets used
throughout (mwb.mqm_scoring.SETS, minus generalMT2022/enzh -- no official
data of any kind exists for it, see mqm_scoring.py)."""

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
