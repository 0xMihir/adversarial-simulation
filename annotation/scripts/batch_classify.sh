#!/usr/bin/env bash
# Pre-classify all symbol names from a directory of .far files.
# Saves results to a pickle that the annotation server loads on startup.
#
# Usage:
#   ./annotation/scripts/batch_classify.sh [--data-dir PATH] [--output PATH] [--merge]
#
# Defaults:
#   --data-dir  data/nhtsa-ciss/data/output
#   --output    annotation/annotations/clf_cache.pkl
#   --merge     (not set — overwrites existing cache)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/nhtsa-ciss/data/output}"
OUTPUT="${OUTPUT:-annotation/annotations/clf_cache.pkl}"
MERGE=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output)   OUTPUT="$2";   shift 2 ;;
    --merge)    MERGE="--merge"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "=== Batch Classification ==="
echo "  Data dir : $DATA_DIR"
echo "  Output   : $OUTPUT"
echo "  Merge    : ${MERGE:-(no, overwrite)}"
echo

uv run python -m annotation.scripts.batch_classify \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT" \
  $MERGE

echo
echo "Done. Start the server and it will load the cache automatically."
