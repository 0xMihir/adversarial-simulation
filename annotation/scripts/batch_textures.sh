#!/usr/bin/env bash
# Extract all sign textures from a directory of .far files into a flat pickle cache.
# The annotation server loads this pickle on startup for cross-file sign resolution.
#
# Usage:
#   ./annotation/scripts/batch_textures.sh [--data-dir PATH] [--output PATH] [--merge]
#
# Defaults:
#   --data-dir  data/nhtsa-ciss/data/output
#   --output    annotation/annotations/tex_cache.pkl
#   --merge     (not set — overwrites existing cache)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/nhtsa-ciss/data/output}"
OUTPUT="${OUTPUT:-annotation/annotations/tex_cache.pkl}"
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

echo "=== Batch Texture Extraction ==="
echo "  Data dir : $DATA_DIR"
echo "  Output   : $OUTPUT"
echo "  Merge    : ${MERGE:-(no, overwrite)}"
echo

uv run python -m annotation.scripts.batch_textures \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT" \
  $MERGE

echo
echo "Done. Start the server and it will load the texture cache automatically."
