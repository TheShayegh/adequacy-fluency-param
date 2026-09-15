#!/usr/bin/env bash
# The one setup step this project needs: Python venv + pinned deps, the two
# git-submodule dependencies, and the external/mt-metrics-eval-data data
# release (a plain GCS download, not a git repo -- see the comment at that
# step). Safe to re-run -- every step skips work it's already done.
#
# Usage (run from anywhere -- paths below resolve relative to this script's
# own location, not the caller's cwd):
#   bash setup.sh
#
# Disk: ~3.1GB (mt-metrics-eval-data's system/metric scores + the two
# submodules' MQM annotations and converter code).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo "[setup] project root: $ROOT"

# ---------------------------------------------------------------------
# 1. Python venv + pinned deps
# ---------------------------------------------------------------------
if [ ! -x .venv/bin/python3 ]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
echo "[setup] .venv python: $(.venv/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "[setup] dependencies installed"

# ---------------------------------------------------------------------
# 2. Git submodules: mt-metrics-eval (the WMT toolkit's converter code)
#    and wmt-mqm-human-evaluation (the raw MQM annotations).
# ---------------------------------------------------------------------
echo "[setup] fetching git submodules"
git submodule update --init --recursive

# ---------------------------------------------------------------------
# 3. external/mt-metrics-eval-data: the official data bundle. NOT a git
#    repo -- a plain file download off a GCS bucket, hence not a
#    submodule. storage.googleapis.com is commonly blocked on corporate/
#    restricted networks even when github.com itself is reachable
#    (different domain, often not on the same allowlist); if curl below
#    fails for that reason, fetch mt-metrics-eval-v2.tgz from a machine
#    that does have access and copy it to
#    external/mt-metrics-eval-data/mt-metrics-eval-v2.tgz on this one,
#    then re-run this script (it skips the download when the extracted
#    directory already exists and is non-empty, so a partial manual
#    transfer of just the needed year/pair subtrees also works --
#    mwb/lib/dataset_dirs.py's DATASET_DIRS lists which (year, pair)
#    subtrees the datasets this project covers actually need).
# ---------------------------------------------------------------------
mkdir -p external/mt-metrics-eval-data
if [ -d external/mt-metrics-eval-data/mt-metrics-eval-v2 ] \
    && [ -n "$(ls -A external/mt-metrics-eval-data/mt-metrics-eval-v2 2>/dev/null)" ]; then
  echo "[setup] mt-metrics-eval-v2/ already present, skipping download"
else
  echo "[setup] downloading mt-metrics-eval-v2.tgz (~870MB) from the official GCS bucket"
  curl -L --fail -o external/mt-metrics-eval-data/mt-metrics-eval-v2.tgz \
      https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz
  echo "[setup] extracting mt-metrics-eval-v2.tgz (~2.3GB unpacked)"
  tar xzf external/mt-metrics-eval-data/mt-metrics-eval-v2.tgz -C external/mt-metrics-eval-data
fi

# generalMT2023 ende/zhen ship as tar.gz'd TSVs with a ".3ratingsPerSegment"
# infix in their filename that mwb/mqm_scoring.py's SETS dict does NOT
# expect in the final path -- extract, then rename to strip that infix.
# Every other dataset's MQM tsv is already a plain file, no extraction
# needed.
for pair in ende zhen; do
  dir="external/wmt-mqm-human-evaluation/generalMT2023/$pair"
  archive="$dir/mqm_generalMT2023_${pair}.3ratingsPerSegment.tsv.tar.gz"
  target="$dir/mqm_generalMT2023_${pair}.tsv"
  if [ -f "$archive" ] && [ ! -f "$target" ]; then
    echo "[setup] extracting $archive"
    tar xzf "$archive" -C "$dir"
    mv "$dir/mqm_generalMT2023_${pair}.3ratingsPerSegment.tsv" "$target"
  fi
done

echo "[setup] done. Try: python cli.py dataset-stats"
