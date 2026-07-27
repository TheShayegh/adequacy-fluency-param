"""Heatmap of pairwise balance alpha_ij (lib.alpha.alpha_ij) for every
system pair in a dataset -- a symmetric K x K matrix (alpha_ij is
symmetric in (i,j) by construction: it depends only on squared
differences). Diagonal is mathematically undefined (a coinciding pair,
zero denominator) and rendered as a neutral gray, not a color on the
0-1 scale.

Colormap: a 3-stop diverging scale, green (alpha_ij=0, all-fluency-driven
separation) -> white (alpha_ij=0.5, evenly split) -> blue (alpha_ij=1,
all-adequacy-driven separation), matching this project's convention of
"blue" for the adequacy-heavy end used elsewhere (e.g. STANDARD_BLUE in
the type1 plot style).

Usage: python codes/scripts/plot_alpha_ij_heatmap.py
           --dataset ende22 [--exclude-outliers | --no-exclude-outliers]
           [--tag TAG]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


class LogAnchoredHalfNorm(Normalize):
  """Log-scaled norm anchored at 0.5: -log(0.5 - |v-0.5|) on each side of
  the center, so most of [0,1] (moderate deviation from 0.5) compresses
  toward the middle of the colormap (looks near-white) and only values
  very close to the true extremes 0/1 stretch out to full color -- the
  mirror image of this session's earlier K'-anchored ESS scales, just
  anchored at the CENTER of a bounded [0,1] variable instead of one edge
  of an unbounded one. `eps` sets how close to 0/1 you have to get before
  color saturates (smaller eps = more of the matrix reads as white)."""

  def __init__(self, eps: float = 1e-3, vmin: float = 0.0, vmax: float = 1.0):
    super().__init__(vmin=vmin, vmax=vmax, clip=True)
    self.eps = eps
    self._tmax = -np.log(eps)

  def __call__(self, value, clip=None):
    v = np.ma.asarray(value, dtype=float)
    d = v - 0.5
    mag = np.clip(0.5 - np.abs(d), self.eps, 0.5)
    t = np.sign(d) * (-np.log(mag))
    out = np.clip(t / self._tmax * 0.5 + 0.5, 0.0, 1.0)
    return np.ma.masked_array(out, mask=np.ma.getmaskarray(v))

from lib.alpha import alpha_ij
from lib.consistency import real_systems
from mwb.mqm_scoring import load_system_scores

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

GREEN = (0.0, 0.55, 0.15)
WHITE = (1.0, 1.0, 1.0)
BLUE = (0.121, 0.466, 0.705)
DIAG_GRAY = (0.65, 0.65, 0.65)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--dataset', type=str, default='ende22')
  p.add_argument('--exclude-outliers', action=argparse.BooleanOptionalAction, default=True)
  p.add_argument('--log-eps', type=float, default=1e-3,
                  help='how close to 0/1 alpha_ij must get before color saturates (smaller = whiter)')
  p.add_argument('--tag', type=str, default=None)
  args = p.parse_args()
  base = args.dataset
  tag = args.tag or base

  full = real_systems(base, root=ROOT, exclude_outliers=args.exclude_outliers)
  K = len(full)
  df = load_system_scores(base, root=ROOT)
  a = df.loc[full, 'a'].values
  b = df.loc[full, 'b'].values

  M = np.full((K, K), np.nan)
  for i in range(K):
    for j in range(K):
      if i == j:
        continue
      v = alpha_ij(a[i], b[i], a[j], b[j])
      M[i, j] = v if v is not None else np.nan

  cmap = LinearSegmentedColormap.from_list('green_white_blue', [(0.0, GREEN), (0.5, WHITE), (1.0, BLUE)])
  cmap.set_bad(DIAG_GRAY)
  norm = LogAnchoredHalfNorm(eps=args.log_eps)
  Mm = np.ma.masked_invalid(M)

  fig, ax = plt.subplots(figsize=(10, 9))
  im = ax.imshow(Mm, cmap=cmap, norm=norm, aspect='equal')

  ax.set_xticks(range(K))
  ax.set_yticks(range(K))
  ax.set_xticklabels(full, rotation=90, fontsize=8)
  ax.set_yticklabels(full, fontsize=8)

  for i in range(K):
    for j in range(K):
      if i == j:
        continue
      v = M[i, j]
      # Background is near-white for most of [0,1] under the log-anchored
      # scale now, so black text is legible far more broadly than before --
      # only switch to white text once the cell is genuinely saturated.
      txt_color = 'black' if 0.03 < v < 0.97 else 'white'
      ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color=txt_color)

  ax.set_title(f"{base}: pairwise balance " + r'$\alpha_{ij}$' + f' (K={K})\n'
               'green=0 (fluency-driven) · white=0.5 · blue=1 (adequacy-driven); '
               'gray diagonal = undefined\n'
               'log-anchored-at-0.5 color scale: most cells read near-white, '
               'only near-0/1 values saturate', fontsize=10)

  sm = ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
  cbar.set_label(r'$\alpha_{ij}$' + f'  (log scale anchored at 0.5, eps={args.log_eps:g})')
  tick_vals = [0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0]
  cbar.set_ticks(norm(np.array(tick_vals)))
  cbar.set_ticklabels([f'{v:g}' for v in tick_vals])

  fig.tight_layout()
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)
  out_path = os.path.join(ARTIFACTS_DIR, f'alpha_ij_heatmap_{tag}.png')
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  print(f'Wrote {out_path}')