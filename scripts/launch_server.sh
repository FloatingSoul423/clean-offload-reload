#!/usr/bin/env bash
set -euo pipefail

workspace_arg="${WORKSPACE_ROOT:?Set WORKSPACE_ROOT to a writable workspace}"
allowed_arg="${ALLOWED_ROOT:-$workspace_arg}"
workspace_dir="$(realpath -e -- "$workspace_arg")"
allowed_root="$(realpath -e -- "$allowed_arg")"
case "$workspace_dir/" in
  "$allowed_root"/*) ;;
  *) echo "Workspace is outside ALLOWED_ROOT: $workspace_dir" >&2; exit 2 ;;
esac

gpu_id="${1:?Pass an idle GPU index after checking nvidia-smi}"
hicache_gb="${2:-16}"
storage_name="${3:?Pass a unique relative storage directory name}"
[[ "$storage_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "Storage must be one new direct child name" >&2
  exit 2
}

sglang_src="$(realpath -e -- "${SGLANG_SRC_DIR:?Set the patched SGLang runtime directory}")"
[[ -d "$sglang_src/python/sglang" ]] || { echo "Invalid runtime: $sglang_src" >&2; exit 2; }

cd "$workspace_dir"
python_bin="${PYTHON_BIN:-$workspace_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || { echo "Invalid Python: $python_bin" >&2; exit 2; }

results_dir="$workspace_dir/results"
if [[ -L "$results_dir" ]]; then
  echo "Refusing symlinked results directory" >&2
  exit 2
fi
if [[ -e "$results_dir" ]]; then
  results_real="$(realpath -e -- "$results_dir")"
  case "$results_real/" in
    "$allowed_root"/*) ;;
    *) echo "Results directory escaped ALLOWED_ROOT" >&2; exit 2 ;;
  esac
fi

storage_dir="$workspace_dir/$storage_name"
case "$storage_dir/" in
  "$allowed_root"/*) ;;
  *) echo "Storage target escaped ALLOWED_ROOT" >&2; exit 2 ;;
esac
[[ ! -e "$storage_dir" ]] || { echo "Storage target already exists" >&2; exit 2; }

mkdir -p "$results_dir" "$storage_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH="$sglang_src/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$storage_dir"
trace_arg="${TRACE_PATH:-$results_dir/reload}"
trace_parent="$(dirname -- "$trace_arg")"
mkdir -p "$trace_parent"
trace_parent_real="$(realpath -e -- "$trace_parent")"
case "$trace_parent_real/" in
  "$allowed_root"/*) ;;
  *) echo "Trace path escaped ALLOWED_ROOT" >&2; exit 2 ;;
esac
[[ ! -L "$trace_arg" ]] || { echo "Refusing symlinked trace path" >&2; exit 2; }
export SGLANG_RELOAD_TRACE_PATH="$trace_arg"

unset SGLANG_TWO_PHASE_PRESSURE
policy="${SGLANG_POLICY:-deadline-floor}"
case "$policy" in
  deadline-floor) export SGLANG_DEADLINE_FLOOR=1 ;;
  baseline) unset SGLANG_DEADLINE_FLOOR ;;
  *) echo "SGLANG_POLICY must be deadline-floor or baseline" >&2; exit 2 ;;
esac
export SGLANG_DEADLINE_FLOOR_SAFETY="${SGLANG_DEADLINE_FLOOR_SAFETY:-0.9}"
export SGLANG_DEADLINE_FLOOR_SHARED_TOKENS_S="${SGLANG_DEADLINE_FLOOR_SHARED_TOKENS_S:-80000}"
export SGLANG_DEADLINE_FLOOR_SESSION_TOKENS_S="${SGLANG_DEADLINE_FLOOR_SESSION_TOKENS_S:-10000}"
export SGLANG_DEADLINE_FLOOR_INTERVAL="${SGLANG_DEADLINE_FLOOR_INTERVAL:-0.05}"
export SGLANG_DEADLINE_FLOOR_RESERVE_LOW="${SGLANG_DEADLINE_FLOOR_RESERVE_LOW:-0.10}"
export SGLANG_DEADLINE_FLOOR_RESERVE_HIGH="${SGLANG_DEADLINE_FLOOR_RESERVE_HIGH:-0.15}"
export SGLANG_DEADLINE_FLOOR_RESERVE_STEP="${SGLANG_DEADLINE_FLOOR_RESERVE_STEP:-0.02}"
export SGLANG_DEADLINE_FLOOR_RESERVE_COOLDOWN="${SGLANG_DEADLINE_FLOOR_RESERVE_COOLDOWN:-0.5}"
export SGLANG_DEADLINE_FLOOR_IDLE_GRACE="${SGLANG_DEADLINE_FLOOR_IDLE_GRACE:-0.1}"
export SGLANG_PROBABILITY_RESIDENCY_CHECK=1
export SGLANG_PROBABILITY_RELOAD_BUDGET=1
export SGLANG_PROBABILITY_BACKUP_CHAIN=1
export SGLANG_PROBABILITY_PRODUCER_RESERVE=1
# 3.1 used 0; the server default of 100 ms would delay each proactive Reload.
export SGLANG_PROBABILITY_RELOAD_BATCH_WINDOW_MS="${SGLANG_PROBABILITY_RELOAD_BATCH_WINDOW_MS:-0}"

exec "$python_bin" -m sglang.launch_server \
  --model-path "${MODEL_PATH:?Set MODEL_PATH to a local model or Hugging Face ID}" \
  --served-model-name m \
  --host 127.0.0.1 --port 30000 \
  --enable-session-radix-cache \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-size "$hicache_gb" \
  --hicache-write-policy write_through \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.72}" \
  --context-length 32768 \
  --enable-metrics
