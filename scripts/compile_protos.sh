#!/usr/bin/env bash
# Compile WOMD proto stubs for WOMDScenarioLoader (no TensorFlow required).
# Run once from the project root after cloning:
#   bash scripts/compile_protos.sh
set -euo pipefail

PROTO_SRC="synthetic/loaders/proto"
PROTO_OUT="synthetic/loaders/proto"

echo "Compiling scenario.proto → ${PROTO_OUT}/scenario_pb2.py"
python -m grpc_tools.protoc \
    --proto_path="${PROTO_SRC}" \
    --python_out="${PROTO_OUT}" \
    "${PROTO_SRC}/scenario.proto"

# Remove the gRPC service stub (not needed — we only use message types)
rm -f "${PROTO_OUT}/scenario_pb2_grpc.py"

echo "Done. Generated files:"
ls -1 "${PROTO_OUT}"/*_pb2*.py 2>/dev/null || echo "  (none found — check for errors above)"
