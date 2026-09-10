"""Two-panel LOO cross-rater Kendall tau vs. target alpha figure, one panel
per dataset (ende24, zhen23) -- combines the two independent
compute_loo_tau_vs_alpha.py caches into a single figure with ONE SHARED
reliability colorbar, following the panel-combo style (figsize, font
sizes) of plot_spa_plane_synthetic_diagonal_combo.py's two-panel layout
(artifacts/inventory_synthetic_scorers/spa_plane_synthetic_diagonal_combo_
hlepor-ende21_metricx24hybrid-ende24.pdf). Purely a plotting script (reads
the two existing .npz caches, no recomputation) -- see
compute_loo_tau_vs_alpha.py / plot_loo_tau_vs_alpha.py for how the data
was produced and for the single-panel version of this same plot.

Panel titles are short (PANEL_TITLES: "(a) ende24" / "(b) zhen23"), not
the single-panel script's long descriptive one (dataset / sweep-range / K
/ n-points) -- matches plot_spa_plane_synthetic_diagonal_combo.py's own
"(letter) label" title convention.

Style follows plot_spa_plane_synthetic_diagonal_combo.py throughout: all
four spines visible (closed box, not just left/bottom), square subplots
(ax.set_box_aspect(1)), and only the y-axis's min/max value ticked (no
intermediate gridline ticks).

Shared colorbar: both panels' lines are colored by the SAME cmap + norm
object, and ONE reliability strip is drawn (attached to the right panel
only) via a LOCAL _draw_reliability_strip -- NOT plot_scorer_orientation_
vs_alpha.py's own (that one hardcodes its label's rotation/labelpad/
fontsize convention internally; this local copy takes them as module-level
CONSTANTS instead, per the same "everything position/text-related is a
constant" convention as the rest of this file). Getting a shared strip
right requires normalizing BOTH panels' ESS by the SAME denominator --
lib.subsampling_stability's ESS ceiling is an LOO rater's own system count
(K-1), which differs slightly between datasets (zhen23: 14, ende24: 15) --
so this uses K_REF = the LARGER of the two K-1 values for both panels'
ess_over_k and for the norm/strip. This is a real (if honest) tradeoff: it
means zhen23's own best-achievable point (ESS=14) shows at fraction 14/15
rather than 1.0 -- slightly short of the strip's fully-opaque top -- which
is correct, not a bug: zhen23's LOO raters structurally cannot reach as
high a raw ESS as ende24's (one fewer system), and a truly SHARED
absolute-ESS color scale should show that rather than silently rescale
each panel to its own private ceiling.

Legend is FIGURE-level (fig.legend, not axes[0].legend), anchored to the
bottom-left of the whole combo figure via LEGEND_X/LEGEND_Y, so it reads
as belonging to the figure as a whole rather than to one specific panel.

ALL font sizes, positions (figure width, y-label position, legend
position, strip/colorbar position, strip label position), and text
strings live in the CONSTANTS block right below the imports -- edit those,
then just rerun (cheap: no solving happens here).

Usage: python codes/scripts/plot_loo_tau_vs_alpha_combo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_loo_tau_vs_alpha import load_loo_tau_vs_alpha
from plot_scorer_orientation_vs_alpha import SplineReliabilityNorm, _add_ess_colored_line, _ess_over_k_cmap, _ESS_CUTOFF_ABS

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ARTIFACTS_DIR = os.path.join(ROOT, 'artifacts')

# =====================================================================
# CONSTANTS -- edit these, then just rerun (no recompute needed)
# =====================================================================

# (panel letter, dataset, cache path)
PANELS = [
    ('a', 'ende24', os.path.join(ARTIFACTS_DIR, 'data', 'loo_tau_vs_alpha_ende24_n21.npz')),
    ('b', 'zhen23', os.path.join(ARTIFACTS_DIR, 'data', 'loo_tau_vs_alpha_zhen23_n21.npz')),
]

# --- Figure geometry ---
FIG_WIDTH = 40   # wider than plot_spa_plane_synthetic_diagonal_combo.py's own 34 -- this
                 # figure's labels run longer than that script's, so it needs more room
FIG_HEIGHT = 17
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)

LEFT_MARGIN = 0.35    # figure-fraction: where the left panel's axes start (room for YLABEL)
RIGHT_MARGIN = 1.1   # figure-fraction: where the right panel's axes end (room for the strip)

YLABEL_X = 0.335   # figure-fraction position of the shared y-axis label (fig.text, rotated 90)
YLABEL_Y = 0.55

LEGEND_X = 0.37   # figure-fraction anchor for the shared, figure-level legend (bottom-left corner
LEGEND_Y = 0.1   # of the legend box itself, via bbox_to_anchor -- pulled in from the true (0, 0)
                   # figure corner, not matplotlib's own unadorned 'lower left' loc)

# --- Reliability strip (shared colorbar) geometry ---
STRIP_X_OFFSET = 0.025   # figure-fraction gap between the right panel's right edge and the strip
STRIP_WIDTH = 0.014      # figure-fraction width of the strip itself
STRIP_Y0_FRAC = 0.08     # strip's vertical start, as a fraction of the right panel's own height
STRIP_HEIGHT_FRAC = 0.84  # strip's vertical extent, as a fraction of the right panel's own height
STRIP_BORDER = True

STRIP_LABEL_TEXT = 'Reliability (ESS)'
STRIP_LABEL_FONTSIZE = 50
STRIP_LABEL_ROTATION = 270
STRIP_LABEL_PAD = 34   # distance between the strip's tick labels and its own axis label
STRIP_TICK_FONTSIZE = 50

# --- Font sizes -- same numeric values as plot_spa_plane_synthetic_diagonal_combo.py ---
TITLE_FONTSIZE = 54
LABEL_FONTSIZE = 54
TICK_FONTSIZE = 50
LEGEND_FONTSIZE = 54
ANNOTATION_FONTSIZE = 50   # the alpha_0(D) in-axes annotation

TICK_PAD = 25
LINE_WIDTH = 12.0

DARK_COLOR = (0.03, 0.15, 0.35)  # same navy as orientation plots' 'A' (adequacy) family
Y_LIMITS = (0.5, 1.0)

# --- Text ---
XLABEL = r'$\beta$'
YLABEL = r"Cross LOO Kendall's $\tau$"
ALPHA0_ANNOTATION_TEXT = r'$\beta_0$'
BASELINE_LEGEND_TEXT = r'Unweighted $\tau$'
ALPHA0_LEGEND_TEXT = r'$\tau$($\beta$)'
PANEL_TITLES = {'ende24': '(a) ende24', 'zhen23': '(b) zhen23'}  # per-panel title text


def _draw_reliability_strip(fig, ax, cmap, norm, K, cutoff_abs):
  """Local, fully-parameterized reliability-strip legend (see module
  docstring for why this isn't plot_scorer_orientation_vs_alpha.py's own
  _draw_reliability_strip): a manually-drawn image strip, since fig.
  colorbar always samples a continuous mappable's gradient UNIFORMLY (norm
  only moves tick positions, never the swatch's own pixel content), which
  can't render SplineReliabilityNorm's nonlinear shape correctly. Position
  and label styling come entirely from the STRIP_*/YLABEL_* CONSTANTS
  above."""
  bbox = ax.get_position()
  strip_ax = fig.add_axes([bbox.x1 + STRIP_X_OFFSET, bbox.y0 + STRIP_Y0_FRAC * bbox.height,
                            STRIP_WIDTH, STRIP_HEIGHT_FRAC * bbox.height])
  ess_grid = np.linspace(cutoff_abs, K, 512)
  rgba = cmap(norm(ess_grid / K)).reshape(-1, 1, 4)
  strip_ax.imshow(rgba, aspect='auto', origin='lower', extent=[0, 1, cutoff_abs, K])
  strip_ax.set_xticks([])
  strip_ax.set_ylim(cutoff_abs, K)
  for spine in strip_ax.spines.values():
    spine.set_linewidth(0.6 if STRIP_BORDER else 0.0)
    spine.set_visible(STRIP_BORDER)
  strip_ax.set_yticks([cutoff_abs, K])
  strip_ax.set_yticklabels([f'{cutoff_abs:g}', 'K'], fontsize=STRIP_TICK_FONTSIZE)
  strip_ax.yaxis.tick_right()
  strip_ax.set_ylabel(STRIP_LABEL_TEXT, fontsize=STRIP_LABEL_FONTSIZE, rotation=STRIP_LABEL_ROTATION,
                       labelpad=STRIP_LABEL_PAD)
  strip_ax.yaxis.set_label_position('right')


if __name__ == '__main__':
  os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  panel_data = {letter: load_loo_tau_vs_alpha(cache) for letter, _, cache in PANELS}
  K_ref = max(d['K'] - d['leave_out'] for d in panel_data.values())
  cutoff_frac = _ESS_CUTOFF_ABS / K_ref

  cmap = _ess_over_k_cmap(DARK_COLOR)
  norm = SplineReliabilityNorm(vmin=cutoff_frac, vmax=1.0)

  fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

  for ax, (letter, dataset, _) in zip(axes, PANELS):
    d = panel_data[letter]
    alphas, tau, ess = d['alphas'], d['tau'], d['mean_ess']
    ess_over_k = ess / K_ref  # SAME denominator for both panels -- see module docstring

    ax.set_box_aspect(1)  # square PLOT box -- independent of the title/colorbar space around it
    _add_ess_colored_line(ax, alphas, tau, ess_over_k, cmap, norm, linewidth=LINE_WIDTH, zorder=3)

    ax.axhline(d['baseline_tau'], color='red', linestyle='--', linewidth=1.5, zorder=2)
    ax.axvline(d['a0_D'], color='black', alpha=0.3, linewidth=1.0, zorder=1)
    ax.annotate(ALPHA0_ANNOTATION_TEXT, xy=(d['a0_D'], Y_LIMITS[1]), xytext=(0, -4),
                textcoords='offset points', ha='center', va='top',
                fontsize=ANNOTATION_FONTSIZE, color='black')

    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([Y_LIMITS[0], Y_LIMITS[1]])  # min/max only, no intermediate ticks
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
    ax.set_title(PANEL_TITLES[dataset], fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE, pad=TICK_PAD)
    for side in ('top', 'right', 'bottom', 'left'):
      ax.spines[side].set_visible(True)

  axes[1].tick_params(labelleft=False)

  # rect reserves LEFT_MARGIN/RIGHT_MARGIN for the ylabel/strip, drawn AFTER
  # tight_layout off axes[1]'s FINAL bbox -- same order/reasoning as
  # plot_scorer_orientation_vs_alpha.py's own strip placement (calling
  # tight_layout after the strip exists would shove the panels around
  # without the strip in mind).
  fig.tight_layout(rect=[LEFT_MARGIN, 0, RIGHT_MARGIN, 1.0])
  fig.subplots_adjust(left=LEFT_MARGIN)

  # fig.text is anchored to the FIGURE canvas directly (independent of any
  # axes' own position, unlike ax.set_ylabel -- see git history for why).
  fig.text(YLABEL_X, YLABEL_Y, YLABEL, fontsize=LABEL_FONTSIZE, rotation=90, ha='left', va='center')

  proxy_red = Line2D([0], [0], color='red', ls='--', lw=1.5, label=BASELINE_LEGEND_TEXT)
  proxy_black = Line2D([0], [0], color='black', alpha=0.3, lw=1.0, label=ALPHA0_LEGEND_TEXT)
  fig.legend(handles=[proxy_red, proxy_black], loc='lower left', bbox_to_anchor=(LEGEND_X, LEGEND_Y),
             frameon=False, fontsize=LEGEND_FONTSIZE)

  _draw_reliability_strip(fig, axes[1], cmap, norm, K_ref, _ESS_CUTOFF_ABS)

  # bbox_inches='tight' doesn't reliably auto-detect every legend on its
  # own (see plot_scorer_orientation_vs_alpha.py's own note on this) --
  # pass every Legend artist explicitly via bbox_extra_artists.
  legend_artists = fig.findobj(matplotlib.legend.Legend)
  out_path = os.path.join(ARTIFACTS_DIR, 'loo_tau_vs_alpha_combo_ende24_zhen23.pdf')
  fig.savefig(out_path, dpi=150, bbox_inches='tight', bbox_extra_artists=legend_artists)
  plt.close(fig)
  print(f'Wrote {out_path}', file=sys.stderr)
