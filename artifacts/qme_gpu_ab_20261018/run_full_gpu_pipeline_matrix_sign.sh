#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/autodl-tmp/SurrogateNN_DSGE_qme_bench"
PYTHON_BIN="/root/miniconda3/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE="$ROOT/benchmarks/results/gpuhub_full_pipeline_matrix_sign_$STAMP"

mkdir -p "$BASE"
echo "$BASE" > /root/.last_surrogatenn_full_sequence_dir
exec > >(tee -a "$BASE/sequence.log") 2>&1

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PROFILE_SUPPRESS_JSON_STDOUT=1
export VERBOSE_PROGRESS=1
export PROGRESS_CHUNK_INTERVAL=1

monitor_loop() {
  while true; do
    date -u
    pgrep -af "gpuhub_hlt_resnn_runner.py|profile_surrogate_pipeline_gpu.py|run_hlt_gpu_estimation.sh" || true
    nvidia-smi \
      --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader || true
    sleep 30
  done
}

monitor_loop > "$BASE/gpu_monitor.log" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > "$BASE/gpu_monitor.pid"

finish() {
  local rc=$?
  trap - EXIT
  set +e
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  printf '%s\n' "$rc" > "$BASE/sequence.rc"
  printf '[sequence] finished rc=%s utc=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cd "$(dirname "$BASE")" || exit "$rc"
  tar -czf "$(basename "$BASE").tar.gz" "$(basename "$BASE")"
  printf '%s\n' "$(dirname "$BASE")/$(basename "$BASE").tar.gz" > "$BASE/archive_path.txt"
  sync
  exit "$rc"
}
trap finish EXIT

run_runner() {
  local mode="$1"
  local result_dir="$2"
  shift 2
  mkdir -p "$result_dir"
  printf '[sequence] starting mode=%s result_dir=%s utc=%s\n' \
    "$mode" "$result_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" scripts/gpuhub_hlt_resnn_runner.py \
    --mode "$mode" \
    --run-label "$mode" \
    --results-dir "$result_dir" \
    --skip-repo-sync \
    --skip-install \
    --skip-gpu-probe \
    --env PROFILE_SUPPRESS_JSON_STDOUT=1 \
    --env VERBOSE_PROGRESS=1 \
    --env PROGRESS_CHUNK_INTERVAL=1 \
    "$@"
  local rc=$?
  printf '[sequence] finished mode=%s rc=%s utc=%s\n' \
    "$mode" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$rc"
}

cd "$ROOT"
printf '[sequence] started utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '[sequence] base=%s\n' "$BASE"
"$PYTHON_BIN" - <<'PY'
import jax
import numpyro

print("jax", jax.__version__)
print("numpyro", numpyro.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("x64", jax.config.read("jax_enable_x64"))
PY

FINAL_DIR="$BASE/final_nonlinear_validation"
FULL_DIR="$BASE/full_hlt"

run_runner smoke_then_final_nonlinear "$FINAL_DIR" \
  --env HLT_SEP_BATCH_CHUNK_SIZE=2 \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
FINAL_RC=$?
printf '%s\n' "$FINAL_RC" > "$BASE/final_nonlinear_validation.rc"
if [[ "$FINAL_RC" != "0" ]]; then
  printf '[sequence] final_nonlinear did not pass; full_hlt will not run.\n'
  exit "$FINAL_RC"
fi

run_runner full_hlt "$FULL_DIR" \
  --env HLT_SEP_BATCH_CHUNK_SIZE=2 \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
FULL_RC=$?
printf '%s\n' "$FULL_RC" > "$BASE/full_hlt.rc"
exit "$FULL_RC"
