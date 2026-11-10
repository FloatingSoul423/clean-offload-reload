#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
model="${MODEL_PATH:?Set MODEL_PATH to the same model used by the server}"
output="${OUTPUT_JSON:-results/replay.json}"
mkdir -p "$(dirname -- "$output")"

exec "$python_bin" controller/swe_replay.py \
  --service-curve-edf \
  --model "$model" \
  --port "${PORT:-30000}" \
  --sessions "${SESSIONS:-4}" \
  --duration "${DURATION:-120}" \
  --initial-context-tokens "${INITIAL_CONTEXT_TOKENS:-9000}" \
  --deadline-quantile "${DEADLINE_QUANTILE:-0.10}" \
  --shared-bandwidth-gbps "${SHARED_BANDWIDTH_GBPS:-8.0}" \
  --tool-mean "${TOOL_MEAN:-8.0}" \
  --tool-sigma "${TOOL_SIGMA:-0.6}" \
  --seed "${SEED:-20261104}" \
  --gen "${GEN_TOKENS:-8}" \
  --issues-dataset "${ISSUES_DATASET:-princeton-nlp/SWE-bench}" \
  --issues-split "${ISSUES_SPLIT:-test}" \
  --json "$output" \
  "$@"
