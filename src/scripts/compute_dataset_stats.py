"""Per-dataset descriptive statistics for the paper's Appendix "Dataset
Statistics", restricted to quantities the experiments actually use, emitted
as LaTeX tables (Tables dataset-core, dataset-betaij, dataset-mqm,
dataset-scorers).

Columns
-------
K          -- real systems, src.lib.consistency.real_systems (the project default
              wmt_official_outliers screen). Datasets this screen empties
              (ende20, zhen20, ende23) are skipped.
n_scorers  -- the BASE SCORER (base scorer) pool the augmentation experiments
              actually average over: src.lib.consistency.real_scorers with
              require_seg_scores=True, deduplicate=True,
              exclude_degenerate=True, further filtered by
              src.lib.synthetic_scorer_beta_grid.base_scorer_beta_dial_grid's own
              coverage gate (positional alignment must validate, human and
              base scorer segment matrices must load at the pool's width, and at
              least _MIN_SPA_SEGMENTS=10 segments must be jointly non-NaN).
              That gate is replicated here directly -- see base_scorer_pool() --
              so this count is what compute_scorer_preference_vs_beta.py
              would actually iterate over, not merely what real_scorers
              offers it.
n_segments -- segments behind the system-level scores (post modal-coverage
              filter, src.lib.mqm_scoring).

beta_0            -- natural (uniform-weight) balance.
beta_min/beta_max -- reachable range (Theorem "Reachable range"), i.e.
                     min/max pairwise beta_ij.
beta_range        -- beta_max - beta_min.

n_pairs, mean/median/std/iqr_beta_ij
           -- the C(K,2) pairwise-balance distribution (src.lib.beta.
              pairwise_betas). Theorem "Admissible supports" makes any
              feasible target a convex combination of these, so their
              concentration is directly the reweighting's room to
              manoeuvre.

mean_a/var_a, mean_f/var_f
           -- system-level Adequacy / Fluency MQM mean and uniform-weight
              (1/K, population) variance. Both means are NEGATIVE: MQM is
              stored higher-is-better, i.e. negated error weight.
rho_af     -- Pearson correlation of a and f across systems: the rho of
              Theorem "Exit certificate"'s condition (iii), and the
              quantity the adequacy-fluency dial family needs strictly
              below 1.
cov_af     -- Sigma(a,f) at uniform weight.

The base-scorer NAME lists go into a fourth table (tab:dataset-scorers), a
full-width table* with one row per dataset. Its second column is a
>{\\raggedright\\arraybackslash}p{...} column, which needs \\usepackage{array}
in the preamble -- the paper currently loads booktabs and multirow, neither
of which pulls array in.

Usage: python -m src.scripts.compute_dataset_stats
Writes output/dataset_stats.tex (all four LaTeX tables),
output/dataset_scorers.md (the same name lists in Markdown), and prints
both to stdout.
"""

import os
import sys

import numpy as np
import pandas as pd

from src.lib.beta import beta_0, beta_min_max, pairwise_betas, variances
from src.lib.consistency import real_scorers, real_systems
from src.lib.scorer_score_files import load_human_seg_scores, load_scorer_seg_scores
from src.lib.spa_plane import positional_af_matrices
from src.lib.synthetic_scorers import _MIN_SPA_SEGMENTS
from src.lib.mqm_scoring import SETS, load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

DATASETS = list(SETS.keys())

# enru22 is excluded from the paper entirely. It has no base-scorer pool at
# all (its segment-level positional alignment cannot be validated -- see
# base_scorer_pool()'s coverage gate below -- so base_scorer_beta_dial_grid rejects
# EVERY base scorer), which already ruled it out of every augmentation
# experiment; it is now also dropped from the system-level-only
# leave-one-out table, so no result in the paper covers it. Kept as a
# named constant rather than silently omitted from ORDER so the reason
# survives.
EXCLUDED = ('enru22',)

# Paper-facing display names, in the paper's own dataset ordering.
DISPLAY = {
    'ende24': 'ende 24', 'enes24': 'enes 24', 'jazh24': 'jazh 24',
    'heen23': 'heen 23', 'zhen23': 'zhen 23',
    'ende22': 'ende 22', 'zhen22': 'zhen 22',
    'ende21': 'ende 21', 'zhen21': 'zhen 21',
    'ted_ende': 'ted ende', 'ted_zhen': 'ted zhen',
}
ORDER = ['ende24', 'enes24', 'jazh24', 'heen23', 'zhen23',
         'ende22', 'zhen22', 'ende21', 'zhen21', 'ted_ende', 'ted_zhen']


def base_scorer_pool(dataset: str, systems: list[str], root: str = '.') -> list[str]:
  """The base scorers an augmentation experiment actually uses: real_scorers'
  segment-level screen, then base_scorer_beta_dial_grid's own coverage gate
  (src/lib/synthetic_scorer_beta_grid.py) replicated exactly, minus the
  expensive dial/SPA work. Returns [] when the dataset's positional
  alignment or human segment scores are unusable, i.e. when NO base scorer can be
  built at all."""
  candidates = real_scorers(dataset, root=root, systems=systems, require_seg_scores=True)
  pos = positional_af_matrices(dataset, systems, root=root)
  if pos is None:
    return []
  a_pos, f_pos = pos
  human_seg = load_human_seg_scores(dataset, systems, root=root)
  if human_seg is None or human_seg.shape[1] != a_pos.shape[1]:
    return []

  bad = np.isnan(a_pos).any(axis=0) | np.isnan(f_pos).any(axis=0) | np.isnan(human_seg).any(axis=0)
  base_scorers = []
  for name in candidates:
    seg = load_scorer_seg_scores(dataset, name, systems, root=root)
    if seg is None or seg.shape[1] != a_pos.shape[1]:
      continue
    if int((~(bad | np.isnan(seg).any(axis=0))).sum()) < _MIN_SPA_SEGMENTS:
      continue
    base_scorers.append(name)
  return base_scorers


def dataset_stats_row(a: np.ndarray, f: np.ndarray) -> dict:
  """Every (a, f)-derived column, from the two system-level vectors alone --
  no I/O, so this is reusable per-subset (e.g. a leave-one-out pool)."""
  var_a, var_f = variances(a, f)
  beta0 = beta_0(a, f)
  beta_lo, beta_hi = beta_min_max(a, f)

  # Uniform-weight (population, 1/K) covariance -- same convention as
  # variances(), so var_a + var_f + 2*cov == var(a+f) exactly.
  cov_af = float(np.mean((a - a.mean()) * (f - f.mean())))
  rho = float(cov_af / np.sqrt(var_a * var_f)) if var_a > 0 and var_f > 0 else float('nan')

  beta_ijs = np.array([v for _, _, v in pairwise_betas(a, f)])
  q1, q3 = np.percentile(beta_ijs, [25, 75])

  return {
      'K': len(a),
      'beta_0': beta0, 'beta_min': beta_lo, 'beta_max': beta_hi, 'beta_range': beta_hi - beta_lo,
      'n_pairs': len(beta_ijs),
      'mean_beta_ij': float(beta_ijs.mean()), 'median_beta_ij': float(np.median(beta_ijs)),
      'std_beta_ij': float(beta_ijs.std()), 'iqr_beta_ij': float(q3 - q1),
      'mean_a': float(a.mean()), 'var_a': var_a,
      'mean_f': float(f.mean()), 'var_f': var_f,
      'rho_af': rho, 'cov_af': cov_af,
  }


INT_COLS = {'K', 'n_scorers', 'n_segments', 'n_pairs'}


def latex_table(out: pd.DataFrame, cols, headers, caption, label, note=''):
  """One booktabs table. Integer columns print bare; everything else to 4dp."""
  align = 'l' + 'r' * len(cols)
  lines = [
      r'\begin{table}[t]', r'\centering',
      r'\resizebox{\columnwidth}{!}{%',
      rf'\begin{{tabular}}{{{align}}}', r'\toprule',
      'Dataset & ' + ' & '.join(headers) + r' \\', r'\midrule',
  ]
  for name in ORDER:
    if name not in out.index:
      continue
    r = out.loc[name]
    vals = [f'{int(r[c])}' if c in INT_COLS else f'{r[c]:.4f}' for c in cols]
    lines.append(f'{DISPLAY[name]} & ' + ' & '.join(vals) + r' \\')
  lines += [
      r'\bottomrule', r'\end{tabular}%', '}',
      rf'\caption{{{caption}{(" " + note) if note else ""}}}',
      rf'\label{{{label}}}', r'\end{table}',
  ]
  return '\n'.join(lines)


def tex_escape(s: str) -> str:
  """Escape a scorer name for LaTeX text mode. Only `_` actually occurs in
  the real name set (metricx_xl_DA_2019, metametrics_mt_mqm_*, ...), but `&`,
  `%`, `#` and `$` are escaped too so a future WMT roster can't silently
  break the build."""
  for ch in ('&', '%', '#', '$', '_'):
    s = s.replace(ch, '\\' + ch)
  return s


def latex_scorer_table(scorer_lists: dict[str, list[str]], caption, label):
  """One full-width table*, one row per dataset, listing that dataset's base
  scorers. Uses a >{\\raggedright\\arraybackslash}p{} column (needs
  \\usepackage{array}) -- justified text would overfull, since \\texttt names
  cannot hyphenate."""
  lines = [
      r'\begin{table*}[t]', r'\centering', r'\footnotesize',
      r'\begin{tabular}{l >{\raggedright\arraybackslash}p{0.84\textwidth}}',
      r'\toprule',
      r'Dataset & Base scorers \\', r'\midrule',
  ]
  for name in ORDER:
    names = scorer_lists.get(name, [])
    cell = ', '.join(rf'\texttt{{{tex_escape(s)}}}' for s in names)
    lines.append(f'{DISPLAY[name]} & {cell} \\\\')
    lines.append(r'\addlinespace')
  lines = lines[:-1]  # drop the trailing \addlinespace
  lines += [
      r'\bottomrule', r'\end{tabular}',
      rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\end{table*}',
  ]
  return '\n'.join(lines)


if __name__ == '__main__':
  rows, scorer_lists = [], {}
  for name in DATASETS:
    if name in EXCLUDED:
      print(f'{name}: SKIPPED (excluded from the paper -- see EXCLUDED)', file=sys.stderr)
      continue
    systems = real_systems(name, root=ROOT)
    if not systems:
      print(f'{name}: SKIPPED (0 systems under the wmt_official_outliers screen)', file=sys.stderr)
      continue

    sys_df = load_system_scores(name, root=ROOT).loc[systems]
    a, f = sys_df['a'].values, sys_df['f'].values
    base_scorers = base_scorer_pool(name, systems, root=ROOT)
    scorer_lists[name] = base_scorers

    rows.append({
        'dataset': name, **dataset_stats_row(a, f),
        'n_scorers': len(base_scorers),
        'n_segments': int(sys_df['n_segments'].iloc[0]),
    })
    print(f'{name}: K={len(systems)} base scorers={len(base_scorers)}', file=sys.stderr)

  out = pd.DataFrame(rows).set_index('dataset')

  tables = [
      latex_table(
          out,
          ['K', 'n_scorers', 'n_segments', 'beta_0', 'beta_min', 'beta_max', 'beta_range'],
          ['$K$', '\\#Scorers', '\\#Segments', '$\\beta_0$', '$\\beta_{\\min}$',
           '$\\beta_{\\max}$', '$\\beta_{\\max}-\\beta_{\\min}$'],
          'Dataset size and reachable balance range. '
          '$K$: number of translation systems in the pool. '
          '\\#Scorers: number of base scorers augmented for that dataset, listed by name in '
          'Table~\\ref{tab:dataset-scorers}. '
          '\\#Segments: number of segments each system is scored on. '
          '$\\beta_0$: the balance $\\beta(w)$ at the uniform weighting $w=\\tfrac1K\\mathbf 1$. '
          '$\\beta_{\\min}$ and $\\beta_{\\max}$: the smallest and largest pairwise balance '
          '$\\beta_{kk\'}$ taken over all $\\binom{K}{2}$ system pairs, which by '
          'Theorem~\\ref{thm:range} are the endpoints of the reachable range of targets. '
          '$\\beta_{\\max}-\\beta_{\\min}$: the width of that range.',
          'tab:dataset-core'),
      latex_table(
          out,
          ['n_pairs', 'mean_beta_ij', 'median_beta_ij', 'std_beta_ij', 'iqr_beta_ij'],
          ['\\#Pairs', 'Mean', 'Median', 'Std', 'IQR'],
          'Distribution of the pairwise balance $\\beta_{kk\'}$ across system pairs. Every '
          'statistic below is taken over the set of $\\beta_{kk\'}$ values of a single '
          'dataset, one value per unordered pair of distinct systems. '
          '\\#Pairs: the size of that set, $\\binom{K}{2}$. '
          'Mean, Median, Std: the arithmetic mean, median, and standard deviation of the '
          '$\\beta_{kk\'}$ values over those pairs. '
          'IQR: their interquartile range, i.e. the $75$th percentile minus the $25$th.',
          'tab:dataset-betaij'),
      latex_table(
          out,
          ['mean_a', 'var_a', 'mean_f', 'var_f', 'rho_af', 'cov_af'],
          ['$\\mu(a)$', '$\\sigma(a)^2$', '$\\mu(f)$', '$\\sigma(f)^2$',
           '$\\rho$', '$\\Sigma(a,f)$'],
          'System-level Adequacy and Fluency MQM. Every statistic below is taken over the '
          '$K$ systems of a single dataset, at the uniform weighting $w=\\tfrac1K\\mathbf 1$, '
          'on the system-level score vectors $a$ (Adequacy~MQM) and $f$ (Fluency~MQM). '
          '$\\mu(a)$ and $\\sigma(a)^2$: the mean and variance of $a$ across those systems; '
          '$\\mu(f)$ and $\\sigma(f)^2$: the same for $f$. '
          '$\\rho$: the Pearson correlation between $a$ and $f$ across systems, the '
          'adequacy--fluency correlation appearing in Theorem~\\ref{thm:certificate}. '
          '$\\Sigma(a,f)$: the covariance between $a$ and $f$ across those same systems. '
          'Means are negative because MQM is oriented higher-is-better, i.e.\\ as a negated '
          'error weight.',
          'tab:dataset-mqm'),
  ]
  tables.append(latex_scorer_table(
      scorer_lists,
      'The base scorers augmented for each dataset, by name. These are the $s_0$ of '
      'Section~\\ref{sec:experiments}: every scorer in the dataset with segment-level '
      'scores, after removing scorers that assign every system an identical score and '
      'deduplicating scorer pairs whose scores are identical. Each is augmented '
      'independently into all three families, and every reported measure is averaged over '
      'the scorers listed in that dataset\'s row. Counts appear as \\#Scorers in '
      'Table~\\ref{tab:dataset-core}. Names are reproduced exactly as released by WMT, '
      'including three that WMT itself truncates at 35 characters.',
      'tab:dataset-scorers'))
  section = r"""\section{Dataset Statistics}
\label{app:dataset_stats}

Tables~\ref{tab:dataset-core}--\ref{tab:dataset-scorers} report descriptive
statistics for the eleven datasets used throughout this work.
Table~\ref{tab:dataset-core} covers pool size and the balance parameter: the
number of systems, the number of base scorers, the number of segments, the
natural balance $\beta_0$, and the reachable range $[\beta_{\min},
\beta_{\max}]$ together with its width. Table~\ref{tab:dataset-betaij}
expands on the two endpoints of that range by summarizing the full
distribution of pairwise balances $\beta_{kk'}$ they are drawn from,
reporting the number of system pairs along with the mean, median, standard
deviation, and interquartile range over them.
Table~\ref{tab:dataset-mqm} reports the two MQM components underlying
$\beta$ separately, giving the system-level mean and variance of
Adequacy~MQM and of Fluency~MQM, together with their correlation and
covariance across systems. Table~\ref{tab:dataset-scorers} lists by name the
base scorers augmented for each dataset, whose counts appear in
Table~\ref{tab:dataset-core}. Each table's caption defines its columns in
full."""

  tex = section + '\n\n' + '\n\n'.join(tables) + '\n'
  print(tex)

  os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
  tex_path = os.path.join(ROOT, 'output', 'dataset_stats.tex')
  with open(tex_path, 'w') as fh:
    fh.write(tex)

  md_path = os.path.join(ROOT, 'output', 'dataset_scorers.md')
  with open(md_path, 'w') as fh:
    fh.write('# Base scorers (augmentation base_scorers) per dataset\n\n')
    fh.write(
        'The pool every scorer-augmentation experiment averages over: '
        '`src.lib.consistency.real_scorers(require_seg_scores=True)` after '
        "`base_scorer_beta_dial_grid`'s coverage gate. See "
        '`src/scripts/compute_dataset_stats.py`.\n\n')
    for name in ORDER:
      if name not in scorer_lists:
        continue
      names = scorer_lists[name]
      fh.write(f'## {DISPLAY[name]} ({len(names)})\n\n')
      fh.write('none -- positional alignment unavailable\n\n' if not names
               else ', '.join(f'`{s}`' for s in names) + '\n\n')
  print(f'Wrote {tex_path}\nWrote {md_path}')
