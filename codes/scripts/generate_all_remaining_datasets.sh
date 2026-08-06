#!/usr/bin/env bash
# Generates the full cache (main curves + all 4 families' synth25 pool
# markers) AND the final poster-style PDF for every usable dataset except
# ende22 and zhen21 (already done separately) -- same recipe as
# setup_new_machine.sh's example-dataset step, just looped over the rest.
#
# Usage (run from anywhere; paths resolve relative to this script's own
# location):
#   bash codes/scripts/generate_all_remaining_datasets.sh
# Assumes setup_new_machine.sh has already been run at least once on this
# machine (venv + external/ data in place) -- this script does not repeat
# that setup.
#
# Sequential, not parallel across datasets: each dataset's compute steps
# already use --workers 8, so running two datasets at once would
# oversubscribe an 8-core budget. Idempotent/resumable -- every step
# checks for its own output file first and skips if already present, so
# a killed/interrupted run can just be re-invoked rather than restarted
# from scratch.
#
# Time: this is a genuinely long batch. Per dataset, the synth25
# permutation-test sweep (no alpha sweep to amortize its cost against)
# is the dominant cost and its wall time varies a lot with machine load
# -- on the machine this was developed on, one dataset ranged from ~20
# minutes to ~2 hours depending on how busy the other (4 performance +
# 6 efficiency core) cores were. Budget for several hours across all 9
# datasets; the per-dataset skip logic means an interrupted run loses
# at most the one in-flight step, not earlier progress.
set -euo pipefail

WORKERS=8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
echo "[batch] project root: $ROOT"

if [ ! -d external/mt-metrics-eval-data/mt-metrics-eval-v2 ] || [ ! -x .venv/bin/python3 ]; then
  echo "[batch] external/ data or .venv not found -- run setup_new_machine.sh first" >&2
  exit 1
fi

# All 15 (year, language-pair) system-sets this project covers (codes/lib/
# dataset_dirs.py), minus:
#   - ende22, zhen21: already generated separately (this script's whole
#     point is "the rest").
#   - ende20, zhen20, ende23, enru22: real_systems() returns 0 systems for
#     all four under this project's committed screen (ende20/zhen20 via
#     wmt_official outlier removal, enru22 via MISSING_SEG_GRANULAR_MQM,
#     ende23 via the explicit manual exclusion, lib/consistency.py) --
#     nothing for compute_scorer_orientation_vs_alpha.py to compute, it
#     would just exit immediately with "0 real systems, nothing to
#     compute". Verified empirically (real_systems(d, excl_missing_seg_
#     granular_mqm=True) for all four) before writing this list, not
#     guessed -- if that screen ever changes, re-check before trusting
#     this list again.
DATASETS=(ende21 ted_ende ted_zhen zhen22 zhen23 heen23 ende24 enes24 jazh24)

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
# ends up with the widened grid committed by accident, even if a later
# dataset in the loop fails.
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

  if [ -f "$plot_out" ]; then
    echo "[$dataset] plot already exists, skipping render (rm it to force a rebuild)"
  else
    echo "[$dataset] rendering plot"
    .venv/bin/python3 codes/scripts/plot_scorer_orientation_vs_alpha.py \
        --dataset "$dataset" --tag "$tag" --poster --format pdf
  fi
  echo "[$dataset] done -> $plot_out"
}

echo "[batch] widening dial grid to 1001 points for this run"
widen_dial_grid

N=${#DATASETS[@]}
START_TS=$(date +%s)
for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  idx=$((i + 1))
  now=$(date +%s)
  elapsed=$((now - START_TS))
  echo "=== [$idx/$N] $dataset (elapsed ${elapsed}s since batch start) ==="
  generate_one_dataset "$dataset"
done

echo "[batch] all $N datasets done -- dial grid will be reverted automatically on exit"
