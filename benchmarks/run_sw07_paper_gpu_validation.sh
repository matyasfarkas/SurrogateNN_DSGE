#!/usr/bin/env bash
set -euo pipefail

# Paper-matched GPU validation for the linearized Smets-Wouters (2007) model.
# The empirical data are not bundled; point SW07_DATA_MAT at the archived
# usmodel_data.mat whose SHA256 is checked by posterior_sampling_speed.py.
#
# Examples:
#   SW07_DATA_MAT=/path/to/usmodel_data.mat MODE=preflight bash benchmarks/run_sw07_paper_gpu_validation.sh
#   SW07_DATA_MAT=/path/to/usmodel_data.mat MODE=pilot INIT=prior_mean bash benchmarks/run_sw07_paper_gpu_validation.sh
#   SW07_DATA_MAT=/path/to/usmodel_data.mat MODE=publication INIT=dynare bash benchmarks/run_sw07_paper_gpu_validation.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SW07_DATA_MAT:?Set SW07_DATA_MAT to the archived usmodel_data.mat}"

MODE="${MODE:-preflight}"
INIT="${INIT:-dynare}"
PYTHON="${PYTHON:-.venv/bin/python}"
PLATFORM="${PLATFORM:-gpu}"
QME_ALGORITHM="${QME_ALGORITHM:-schur_gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/results/sw07_paper_gpu_$(date -u +%Y%m%dT%H%M%SZ)}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

COMMON=(
  benchmarks/posterior_sampling_speed.py
  --preset sw07_paper
  --sw07-data-mat "$SW07_DATA_MAT"
  --sw07-sample-start-index 50
  --sw07-sample-periods 180
  --sw07-initialization "$INIT"
  --steady-state-mode solve
  --steady-state-max-iter 20
  --presample-periods 6
  --initial-covariance-strategy theoretical
  --parameters sw07_paper_36
  --dtype float64
  --platform "$PLATFORM"
  --force-gpu
  --qme-algorithm "$QME_ALGORITHM"
  --failure-value=-1e12
  --verbose
  --heartbeat-seconds 30
)

mkdir -p "$OUTPUT_ROOT"

case "$MODE" in
  preflight)
    "$PYTHON" "${COMMON[@]}" \
      --preflight-only \
      --preflight-reps 3 \
      --output "$OUTPUT_ROOT/preflight_${INIT}.json"
    ;;
  pilot)
    "$PYTHON" "${COMMON[@]}" \
      --preflight \
      --warmup 100 \
      --samples 100 \
      --chains 2 \
      --chain-method vectorized \
      --kernel nuts \
      --target-accept-prob 0.9 \
      --max-tree-depth 10 \
      --dense-mass \
      --store-posterior-draws \
      --output "$OUTPUT_ROOT/pilot_${INIT}.json"
    ;;
  publication)
    "$PYTHON" "${COMMON[@]}" \
      --preflight \
      --preflight-reps 3 \
      --warmup 500 \
      --samples 1000 \
      --chains 4 \
      --chain-method vectorized \
      --kernel nuts \
      --target-accept-prob 0.9 \
      --max-tree-depth 10 \
      --dense-mass \
      --store-posterior-draws \
      --output "$OUTPUT_ROOT/publication_${INIT}.json"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected preflight, pilot, or publication)" >&2
    exit 2
    ;;
esac

echo "SW07 validation output: $OUTPUT_ROOT"
