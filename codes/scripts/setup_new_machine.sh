#!/usr/bin/env bash
# Bootstraps this project on a fresh machine: creates the venv, installs
# pinned dependencies, clones + extracts the external data repos the
# scorer-orientation pipeline reads, then computes ONE example dataset's
# full cache (main curves + all 4 families' synth25 pool markers) and
# renders its poster-style PDF end to end, as a smoke test that
# everything above actually worked.
#
# Usage (run from anywhere -- paths below are resolved relative to this
# script's own location, not the caller's cwd):
#   bash codes/scripts/setup_new_machine.sh [dataset]
# dataset defaults to ende22 (matching the exact command this was built
# to reproduce). Any of the 15 names in codes/lib/dataset_dirs.py works;
# see generate_all_remaining_datasets.sh for the other 9 usable ones.
#
# Disk: ~3.1GB of external data (mt-metrics-eval-data's system/metric
# scores + wmt-mqm-human-evaluation's MQM annotations). Time: a few
# minutes to clone+extract depending on network, then anywhere from ~15
# to ~90+ minutes for the example dataset's compute -- the synth25
# permutation-test sweep (compute_synth25_orientation.py, no alpha sweep
# to amortize its cost against, unlike the main curves) is the slow part
# and its wall time varies a lot with how busy the machine's performance
# cores are (see the session note this script's own commit message/PR
# links, if any -- observed 20min-90min for the same dataset on a lightly
# vs. heavily loaded 4P+6E-core Mac).
set -euo pipefail

EXAMPLE_DATASET="${1:-ende22}"
WORKERS=8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
echo "[setup] project root: $ROOT"
echo "[setup] example dataset: $EXAMPLE_DATASET"

# ---------------------------------------------------------------------
# 1. Python venv + pinned deps
# ---------------------------------------------------------------------
if [ ! -x .venv/bin/python3 ]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
PYVER="$(.venv/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "[setup] .venv python: $PYVER (developed/tested on 3.9)"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r codes/scripts/requirements.txt
echo "[setup] dependencies installed"

# ---------------------------------------------------------------------
# 2. External data (gitignored -- not part of this project's own git
#    history, so a fresh clone of the project itself has none of this).
#    Only what's actually READ at runtime by codes/ is fetched here:
#    mt-metrics-eval-data (system/metric scores -- a plain GCS download,
#    NOT a git repo, see the comment at its own step below) and the
#    wmt-mqm-human-evaluation git repo (human MQM annotations). This
#    project's other external/ checkouts on the original machine
#    (mt-metrics-eval itself, wmt25-general-mt, wmt25-mteval) are cited
#    in a few comments for algorithm provenance but never imported or
#    read at runtime by this pipeline, so they're skipped here (~2.7GB
#    saved). Add them the same way below if you want full parity with
#    the original external/ for unrelated work.
# ---------------------------------------------------------------------
mkdir -p external
mkdir -p external/mt-metrics-eval-data

# NOT a git repo -- this is a plain data release from the official
# google-research/mt-metrics-eval toolkit (see external/mt-metrics-eval/
# README.md's own download instructions), a straight file download off a
# GCS bucket. An earlier version of this script wrongly `git clone`d
# https://github.com/TheShayegh/wmt26.git here instead: that URL is
# actually THIS project's own origin remote (confirmed via `git remote
# -v` at the project root), not a data repo -- `git -C external/mt-
# metrics-eval-data remote -v` was silently walking UP to the enclosing
# MindWhichBird checkout because this directory has no `.git` of its own,
# and that misread got baked into the clone command. Fixed to a plain
# download, which is also just simpler for a non-git data drop.
#
# storage.googleapis.com is commonly blocked on corporate/restricted
# networks even when github.com itself is reachable (different domain,
# often not on the same allowlist) -- if `curl` below fails for that
# reason, the fallback that worked on one such machine already was:
#   1. On a machine WITH GCS access, extract just the (year_dir, language
#      pair) subtrees actually needed -- see codes/lib/dataset_dirs.py
#      for which pairs each dataset name maps to, and note several
#      datasets are excluded from this project entirely (real_systems()
#      returns 0 for ende20/zhen20/ende23/enru22 under the committed
#      screen, lib/consistency.py) so their data was never packaged.
#   2. Zip each needed subtree (preserving the
#      external/mt-metrics-eval-data/mt-metrics-eval-v2/... relative
#      path from the project root, so `unzip` on the other end drops it
#      in the right place with no manual directory surgery) and attach
#      them as assets on a GitHub Release (public repo: plain `curl -L`
#      works with no auth; private repo: needs a token, more setup).
#   3. On the restricted machine: `curl -L -o x.zip <release-asset-url>`
#      then `unzip x.zip` from the project root, for each zip -- this
#      populates external/mt-metrics-eval-data/mt-metrics-eval-v2/
#      piecemeal instead of via the single tgz, which is exactly what
#      the check below is written to accept as equally valid.
if [ -d external/mt-metrics-eval-data/mt-metrics-eval-v2 ] \
    && [ -n "$(ls -A external/mt-metrics-eval-data/mt-metrics-eval-v2 2>/dev/null)" ]; then
  echo "[setup] mt-metrics-eval-v2/ already present (full download or manually-transferred subset), skipping"
else
  echo "[setup] downloading mt-metrics-eval-v2.tgz (~870MB) from the official GCS bucket"
  curl -L --fail -o external/mt-metrics-eval-data/mt-metrics-eval-v2.tgz \
      https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz
  echo "[setup] extracting mt-metrics-eval-v2.tgz (~2.3GB unpacked)"
  tar xzf external/mt-metrics-eval-data/mt-metrics-eval-v2.tgz -C external/mt-metrics-eval-data
fi

if [ ! -d external/wmt-mqm-human-evaluation/.git ]; then
  echo "[setup] cloning wmt-mqm-human-evaluation"
  git clone https://github.com/google/wmt-mqm-human-evaluation external/wmt-mqm-human-evaluation
  git -C external/wmt-mqm-human-evaluation checkout 7fadea281aff8db61cc72fc3d469a4dca78b82a6
else
  echo "[setup] external/wmt-mqm-human-evaluation already present, skipping clone"
fi

# generalMT2023 ende/zhen ship as tar.gz'd TSVs with a ".3ratingsPerSegment"
# infix in their filename that codes/mwb/mqm_scoring.py's SETS dict does
# NOT expect in the final path -- extract, then rename to strip that
# infix. Every other dataset's MQM tsv is already a plain file in the
# repo, no extraction needed.
for pair in ende zhen; do
  dir="external/wmt-mqm-human-evaluation/generalMT2023/$pair"
  archive="$dir/mqm_generalMT2023_${pair}.3ratingsPerSegment.tsv.tar.gz"
  target="$dir/mqm_generalMT2023_${pair}.tsv"
  if [ ! -f "$target" ]; then
    echo "[setup] extracting $archive"
    tar xzf "$archive" -C "$dir"
    mv "$dir/mqm_generalMT2023_${pair}.3ratingsPerSegment.tsv" "$target"
  fi
done
echo "[setup] external/ data ready"

# ---------------------------------------------------------------------
# 3. Dial-grid widen -> compute -> revert, then the plot. Same recipe as
#    generate_all_remaining_datasets.sh -- see that script's own comments
#    for why this three-step dance (not just editing the file once) is
#    needed: ABT_DIAL_GRID/ABTJ_J_DIAL_GRID are the project's committed
#    201-point dial grid; this session's plots all use a widened
#    1001-point version instead, applied as a temporary source edit
#    around the compute calls (never left modified in the repo).
# ---------------------------------------------------------------------
DIAL_GRID_FILE="codes/lib/synthetic_scorers.py"
OLD_DIAL='ABT_DIAL_GRID = tuple(round(-1.5 + 3.0 ** (0.005 * k), 10) for k in range(201))
ABTJ_J_DIAL_GRID = tuple(round(-3.5 + 4.0 ** (0.005 * k), 10) for k in range(201))'
NEW_DIAL='ABT_DIAL_GRID = tuple(round(-1.5 + 3.0 ** (0.001 * k), 10) for k in range(1001))
ABTJ_J_DIAL_GRID = tuple(round(-3.5 + 4.0 ** (0.001 * k), 10) for k in range(1001))'

widen_dial_grid() {
  .venv/bin/python3 - "$DIAL_GRID_FILE" <<PY
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = """$OLD_DIAL"""
new = """$NEW_DIAL"""
if old not in src:
    sys.exit(f"{p}: committed dial-grid text not found -- it may have changed "
             "since this script was written; check codes/lib/synthetic_scorers.py "
             "by hand before patching")
p.write_text(src.replace(old, new))
PY
}

revert_dial_grid() {
  .venv/bin/python3 - "$DIAL_GRID_FILE" <<PY
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = """$OLD_DIAL"""
new = """$NEW_DIAL"""
if new in src:
    p.write_text(src.replace(new, old))
PY
}

# Always revert on exit -- success, error, or Ctrl-C -- so the repo never
# ends up with the widened grid committed by accident.
trap revert_dial_grid EXIT

generate_one_dataset() {
  local dataset="$1"
  local tag="${dataset}_union_s0.05_additive_mean_dial0.001_ABTJ"
  local main_cache="artifacts/data/scorer_orientation_${tag}.npz"
  local synth25_plain="artifacts/data/synth25_orientation_${dataset}_spa_synth25_pools_additive_mean_sym.npz"
  local synth25_j="artifacts/data/synth25_orientation_${dataset}_spa_synth25_pools_J_sym.npz"
  local plot_out="artifacts/inventory_scorer_orientation/orientation_${dataset}_ABTJ_synth25.pdf"

  if [ -f "$main_cache" ]; then
    echo "[$dataset] main curve cache exists, skipping compute"
  else
    echo "[$dataset] computing main curves (union grid step=0.05, dial=0.001, workers=$WORKERS)"
    .venv/bin/python3 codes/scripts/compute_scorer_orientation_vs_alpha.py \
        --dataset "$dataset" --step 0.05 --workers "$WORKERS" --tag "$tag"
  fi

  if [ -f "$synth25_plain" ]; then
    echo "[$dataset] synth25 A/B/T cache exists, skipping compute"
  else
    echo "[$dataset] computing synth25 A/B/T pools (workers=$WORKERS)"
    .venv/bin/python3 codes/scripts/compute_synth25_orientation.py --dataset "$dataset" --workers "$WORKERS"
  fi

  if [ -f "$synth25_j" ]; then
    echo "[$dataset] synth25 J cache exists, skipping compute"
  else
    echo "[$dataset] computing synth25 J pool (workers=$WORKERS)"
    .venv/bin/python3 codes/scripts/compute_synth25_orientation.py --dataset "$dataset" --J --workers "$WORKERS"
  fi

  echo "[$dataset] rendering plot"
  .venv/bin/python3 codes/scripts/plot_scorer_orientation_vs_alpha.py \
      --dataset "$dataset" --tag "$tag" --poster --format pdf
  echo "[$dataset] done -> $plot_out"
}

echo "[setup] widening dial grid to 1001 points for this run"
widen_dial_grid

generate_one_dataset "$EXAMPLE_DATASET"

echo "[setup] all done -- dial grid will be reverted automatically on exit"
