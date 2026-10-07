#!/usr/bin/env bash
set -euo pipefail

# Run the HLT nonlinear surrogate-estimation pipeline on a CUDA/JAX host.
#
# Usage:
#   MODE=smoke bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=pilot HLT_THETA_DRAWS=64 HMC_SAMPLES=64 bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=estimation_pilot bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=final_nonlinear bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=full_hlt bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=full bash benchmarks/run_hlt_gpu_estimation.sh
#
# The posterior stage can run either a fixed-reference smoke likelihood or the
# full-JAX likelihood that recomputes parameter-specific steady states and ROMs
# inside HMC. Use LIKELIHOOD_RUNTIME_MODE=full-jax for posterior comparisons.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
DEVICE="${DEVICE:-gpu}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
RESULT_ROOT="${RESULT_ROOT:-benchmarks/results/hlt_gpu_estimation_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-python3}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=''}"

if [[ "${JAX_LOG_DENSITY_GRADIENT:-1}" == "1" ]]; then
  JAX_LOG_DENSITY_GRADIENT_FLAG="--hlt-jax-log-density-gradient"
else
  JAX_LOG_DENSITY_GRADIENT_FLAG="--no-hlt-jax-log-density-gradient"
fi

if [[ "${DIFFERENTIATE_SHOCKS:-0}" == "1" ]]; then
  DIFFERENTIATE_SHOCKS_FLAG="--hlt-jax-differentiate-shocks"
else
  DIFFERENTIATE_SHOCKS_FLAG="--no-hlt-jax-differentiate-shocks"
fi

if [[ "${SEP_LINE_SEARCH:-1}" == "1" ]]; then
  SEP_LINE_SEARCH_FLAG="--sep-line-search"
else
  SEP_LINE_SEARCH_FLAG="--no-sep-line-search"
fi

if [[ "${VERBOSE_PROGRESS:-0}" == "1" ]]; then
  VERBOSE_PROGRESS_FLAG="--verbose-progress"
else
  VERBOSE_PROGRESS_FLAG="--no-verbose-progress"
fi

case "$MODE" in
  smoke)
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-batched-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-16}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-payload}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-1}"
    HLT_PERIODS="${HLT_PERIODS:-1}"
    SEP_PERIODS="${SEP_PERIODS:-1}"
    SEP_ORDER="${SEP_ORDER:-0}"
    SEP_NNODES="${SEP_NNODES:-1}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-4}"
    EPOCHS="${EPOCHS:-1}"
    HIDDEN="${HIDDEN:-8}"
    BLOCKS="${BLOCKS:-0}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float64}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-1}"
    HMC_WARMUP="${HMC_WARMUP:-0}"
    HMC_SAMPLES="${HMC_SAMPLES:-1}"
    HMC_CHAINS="${HMC_CHAINS:-1}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-1}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.001}"
    ;;
  calibration)
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-adaptive-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-0}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-payload}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-8}"
    HLT_PERIODS="${HLT_PERIODS:-2}"
    SEP_PERIODS="${SEP_PERIODS:-2}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-6}"
    EPOCHS="${EPOCHS:-10}"
    HIDDEN="${HIDDEN:-48}"
    BLOCKS="${BLOCKS:-1}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float64}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-4}"
    HMC_WARMUP="${HMC_WARMUP:-8}"
    HMC_SAMPLES="${HMC_SAMPLES:-16}"
    HMC_CHAINS="${HMC_CHAINS:-4}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-2}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.003}"
    ;;
  pilot)
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-adaptive-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-2}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-payload}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-32}"
    HLT_PERIODS="${HLT_PERIODS:-8}"
    SEP_PERIODS="${SEP_PERIODS:-8}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-8}"
    EPOCHS="${EPOCHS:-50}"
    HIDDEN="${HIDDEN:-96}"
    BLOCKS="${BLOCKS:-2}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-20}"
    HMC_WARMUP="${HMC_WARMUP:-32}"
    HMC_SAMPLES="${HMC_SAMPLES:-64}"
    HMC_CHAINS="${HMC_CHAINS:-16}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-4}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.005}"
    ;;
  estimation_pilot)
    # A larger, estimation-like HLT/SEP/ResNN run meant to expose GPU
    # parallelism without jumping directly to the very expensive all-parameter
    # full profile. Defaults target 32GB+ CUDA GPUs; lower
    # HLT_SEP_BATCH_CHUNK_SIZE to 4 or 2 on smaller devices.
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-batched-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-8}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-sw07_safe_27}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-128}"
    HLT_PERIODS="${HLT_PERIODS:-4}"
    HLT_PARAMETER_PERTURBATION="${HLT_PARAMETER_PERTURBATION:-0.0025}"
    HLT_TARGET_MIN_STABLE_PERIODS="${HLT_TARGET_MIN_STABLE_PERIODS:-1}"
    SEP_PERIODS="${SEP_PERIODS:-4}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-8}"
    EPOCHS="${EPOCHS:-200}"
    HIDDEN="${HIDDEN:-192}"
    BLOCKS="${BLOCKS:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-80}"
    JAX_LOG_DENSITY_REPEAT_EVALS="${JAX_LOG_DENSITY_REPEAT_EVALS:-10}"
    JAX_LOG_DENSITY_BATCH_SIZE="${JAX_LOG_DENSITY_BATCH_SIZE:-2048}"
    JAX_LOG_DENSITY_BATCH_REPEAT_EVALS="${JAX_LOG_DENSITY_BATCH_REPEAT_EVALS:-10}"
    HMC_WARMUP="${HMC_WARMUP:-256}"
    HMC_SAMPLES="${HMC_SAMPLES:-512}"
    HMC_CHAINS="${HMC_CHAINS:-64}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-6}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.0005}"
    HMC_MAX_RETRIES="${HMC_MAX_RETRIES:-4}"
    HMC_RETRY_STEP_SIZE_FACTOR="${HMC_RETRY_STEP_SIZE_FACTOR:-0.25}"
    ;;
  final_nonlinear)
    # Correctness-first HLT nonlinear estimation pipeline. This mode is meant
    # for the final run, not a throughput smoke test: it samples the Julia-
    # comparable narrow 18-parameter support, requires full SEP target paths,
    # trains only on fully successful theta draws, uses the GPU-batched HLT SEP
    # target builder, and checks the full-JAX likelihood/HMC path.
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-batched-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-16}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-phase1_18params_narrow}"
    HLT_THETA_DESIGN="${HLT_THETA_DESIGN:-prior}"
    HLT_THETA_DESIGN_SET="${HLT_THETA_DESIGN_SET:-phase1_18params_narrow}"
    HLT_THETA_INCLUDE_REFERENCE="${HLT_THETA_INCLUDE_REFERENCE:-1}"
    HLT_DROP_RUNTIME_FAILURES="${HLT_DROP_RUNTIME_FAILURES:-1}"
    HLT_MIN_RUNTIME_SUCCESSFUL_THETA="${HLT_MIN_RUNTIME_SUCCESSFUL_THETA:-64}"
    # Draw more candidates than the strict minimum. HLT prior support still
    # contains points where the solved steady-state/ROM preflight rejects the
    # draw, and the final stage must not silently relax that gate.
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-192}"
    HLT_PERIODS="${HLT_PERIODS:-4}"
    HLT_SHOCK_SCALE="${HLT_SHOCK_SCALE:-0.05}"
    HLT_PARAMETER_PERTURBATION="${HLT_PARAMETER_PERTURBATION:-0}"
    HLT_TARGET_MIN_STABLE_PERIODS="${HLT_TARGET_MIN_STABLE_PERIODS:--1}"
    HLT_REQUIRE_FULL_TARGET_SUCCESS="${HLT_REQUIRE_FULL_TARGET_SUCCESS:-1}"
    HLT_MIN_FULL_SUCCESS_SHARE="${HLT_MIN_FULL_SUCCESS_SHARE:-1.0}"
    HLT_MIN_ACCEPTED_SAMPLES="${HLT_MIN_ACCEPTED_SAMPLES:-$((HLT_MIN_RUNTIME_SUCCESSFUL_THETA * HLT_PERIODS))}"
    HLT_MIN_VALIDATION_IMPROVEMENT_MEAN="${HLT_MIN_VALIDATION_IMPROVEMENT_MEAN:-0.10}"
    HLT_REQUIRE_JAX_PARITY="${HLT_REQUIRE_JAX_PARITY:-1}"
    HLT_REQUIRE_HMC="${HLT_REQUIRE_HMC:-1}"
    HLT_MAX_HMC_ACCEPTED_SHARE="${HLT_MAX_HMC_ACCEPTED_SHARE:-0.995}"
    HLT_REQUIRE_SOLVED_STEADY_STATE="${HLT_REQUIRE_SOLVED_STEADY_STATE:-1}"
    FAIL_ON_QUALITY_GATE="${FAIL_ON_QUALITY_GATE:-1}"
    HLT_STEADY_STATE_MODE="${HLT_STEADY_STATE_MODE:-solve-or-reference}"
    HLT_STEADY_STATE_MAX_ITER="${HLT_STEADY_STATE_MAX_ITER:-200}"
    LIKELIHOOD_RUNTIME_MODE="${LIKELIHOOD_RUNTIME_MODE:-full-jax}"
    LIKELIHOOD_QME_ALGORITHM="${LIKELIHOOD_QME_ALGORITHM:-schur_gpu}"
    LIKELIHOOD_STATIC_ROWS_MODE="${LIKELIHOOD_STATIC_ROWS_MODE:-reference}"
    ONLY_FULL_SUCCESS="${ONLY_FULL_SUCCESS:-1}"
    SEP_PERIODS="${SEP_PERIODS:-4}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-20}"
    SEP_ACCEPT_TOL="${SEP_ACCEPT_TOL:-1e-4}"
    SEP_LINEAR_SOLVER="${SEP_LINEAR_SOLVER:-qr}"
    EPOCHS="${EPOCHS:-300}"
    HIDDEN="${HIDDEN:-192}"
    BLOCKS="${BLOCKS:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
    VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-80}"
    JAX_LOG_DENSITY_REPEAT_EVALS="${JAX_LOG_DENSITY_REPEAT_EVALS:-4}"
    # Full-JAX log density recomputes steady state/ROM inside the traced graph.
    # Large vmapped timing batches can request terabytes of HLO memory, so the
    # correctness-first run defaults to scalar parity/HMC diagnostics.
    JAX_LOG_DENSITY_BATCH_SIZE="${JAX_LOG_DENSITY_BATCH_SIZE:-0}"
    JAX_LOG_DENSITY_BATCH_REPEAT_EVALS="${JAX_LOG_DENSITY_BATCH_REPEAT_EVALS:-0}"
    HMC_WARMUP="${HMC_WARMUP:-500}"
    HMC_SAMPLES="${HMC_SAMPLES:-1000}"
    HMC_CHAINS="${HMC_CHAINS:-64}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-6}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.003}"
    HMC_MAX_RETRIES="${HMC_MAX_RETRIES:-4}"
    HMC_RETRY_STEP_SIZE_FACTOR="${HMC_RETRY_STEP_SIZE_FACTOR:-0.5}"
    ;;
  full_hlt)
    # Correctness-first all-parameter HLT stress run. This is the expensive
    # version: all parsed HLT parameters are perturbed around the reference,
    # the SEP horizon is longer than final_nonlinear, and strict quality gates
    # are enabled. Start with smoke_then_full_hlt on GPUHUB.
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-batched-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-2}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-all}"
    HLT_THETA_DESIGN="${HLT_THETA_DESIGN:-perturbation}"
    HLT_THETA_DESIGN_SET="${HLT_THETA_DESIGN_SET:-auto}"
    HLT_THETA_INCLUDE_REFERENCE="${HLT_THETA_INCLUDE_REFERENCE:-0}"
    HLT_DROP_RUNTIME_FAILURES="${HLT_DROP_RUNTIME_FAILURES:-1}"
    HLT_MIN_RUNTIME_SUCCESSFUL_THETA="${HLT_MIN_RUNTIME_SUCCESSFUL_THETA:-128}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-288}"
    HLT_PERIODS="${HLT_PERIODS:-8}"
    HLT_SHOCK_SCALE="${HLT_SHOCK_SCALE:-0.03}"
    HLT_PARAMETER_PERTURBATION="${HLT_PARAMETER_PERTURBATION:-1e-6}"
    HLT_TARGET_MIN_STABLE_PERIODS="${HLT_TARGET_MIN_STABLE_PERIODS:--1}"
    HLT_REQUIRE_FULL_TARGET_SUCCESS="${HLT_REQUIRE_FULL_TARGET_SUCCESS:-1}"
    HLT_MIN_FULL_SUCCESS_SHARE="${HLT_MIN_FULL_SUCCESS_SHARE:-0.95}"
    HLT_MIN_ACCEPTED_SAMPLES="${HLT_MIN_ACCEPTED_SAMPLES:-$((HLT_MIN_RUNTIME_SUCCESSFUL_THETA * HLT_PERIODS))}"
    HLT_MIN_VALIDATION_IMPROVEMENT_MEAN="${HLT_MIN_VALIDATION_IMPROVEMENT_MEAN:-0.10}"
    HLT_REQUIRE_JAX_PARITY="${HLT_REQUIRE_JAX_PARITY:-1}"
    HLT_REQUIRE_HMC="${HLT_REQUIRE_HMC:-1}"
    HLT_MAX_HMC_ACCEPTED_SHARE="${HLT_MAX_HMC_ACCEPTED_SHARE:-0.995}"
    HLT_REQUIRE_SOLVED_STEADY_STATE="${HLT_REQUIRE_SOLVED_STEADY_STATE:-1}"
    FAIL_ON_QUALITY_GATE="${FAIL_ON_QUALITY_GATE:-1}"
    HLT_STEADY_STATE_MODE="${HLT_STEADY_STATE_MODE:-solve-or-reference}"
    HLT_STEADY_STATE_MAX_ITER="${HLT_STEADY_STATE_MAX_ITER:-200}"
    LIKELIHOOD_RUNTIME_MODE="${LIKELIHOOD_RUNTIME_MODE:-full-jax}"
    LIKELIHOOD_QME_ALGORITHM="${LIKELIHOOD_QME_ALGORITHM:-schur_gpu}"
    LIKELIHOOD_STATIC_ROWS_MODE="${LIKELIHOOD_STATIC_ROWS_MODE:-reference}"
    ONLY_FULL_SUCCESS="${ONLY_FULL_SUCCESS:-1}"
    SEP_PERIODS="${SEP_PERIODS:-8}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-12}"
    SEP_ACCEPT_TOL="${SEP_ACCEPT_TOL:-1e-4}"
    SEP_LINEAR_SOLVER="${SEP_LINEAR_SOLVER:-qr}"
    EPOCHS="${EPOCHS:-300}"
    HIDDEN="${HIDDEN:-192}"
    BLOCKS="${BLOCKS:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
    VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-80}"
    JAX_LOG_DENSITY_REPEAT_EVALS="${JAX_LOG_DENSITY_REPEAT_EVALS:-4}"
    JAX_LOG_DENSITY_BATCH_SIZE="${JAX_LOG_DENSITY_BATCH_SIZE:-0}"
    JAX_LOG_DENSITY_BATCH_REPEAT_EVALS="${JAX_LOG_DENSITY_BATCH_REPEAT_EVALS:-0}"
    HMC_WARMUP="${HMC_WARMUP:-500}"
    HMC_SAMPLES="${HMC_SAMPLES:-1000}"
    HMC_CHAINS="${HMC_CHAINS:-64}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-6}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.003}"
    HMC_MAX_RETRIES="${HMC_MAX_RETRIES:-4}"
    HMC_RETRY_STEP_SIZE_FACTOR="${HMC_RETRY_STEP_SIZE_FACTOR:-0.5}"
    ;;
  full)
    DEFAULT_HLT_TARGET_BUILDER="${DEFAULT_HLT_TARGET_BUILDER:-batched-sep}"
    DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE="${DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE:-2}"
    HLT_PARAMETER_SET="${HLT_PARAMETER_SET:-all}"
    HLT_THETA_DRAWS="${HLT_THETA_DRAWS:-288}"
    HLT_PERIODS="${HLT_PERIODS:-8}"
    SEP_PERIODS="${SEP_PERIODS:-8}"
    SEP_ORDER="${SEP_ORDER:-1}"
    SEP_NNODES="${SEP_NNODES:-3}"
    SEP_MAX_ITER="${SEP_MAX_ITER:-8}"
    EPOCHS="${EPOCHS:-250}"
    HIDDEN="${HIDDEN:-192}"
    BLOCKS="${BLOCKS:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
    TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-80}"
    HMC_WARMUP="${HMC_WARMUP:-500}"
    HMC_SAMPLES="${HMC_SAMPLES:-1000}"
    HMC_CHAINS="${HMC_CHAINS:-32}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-6}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.003}"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use smoke, calibration, pilot, estimation_pilot, final_nonlinear, full_hlt, or full." >&2
    exit 2
    ;;
esac

HLT_TARGET_BUILDER_EFFECTIVE="${HLT_TARGET_BUILDER:-$DEFAULT_HLT_TARGET_BUILDER}"
if [[ -z "${HLT_SEP_BATCH_CHUNK_SIZE:-}" ]]; then
  if [[ "$HLT_TARGET_BUILDER_EFFECTIVE" == "batched-sep" || "$HLT_TARGET_BUILDER_EFFECTIVE" == "grid-batched-sep" ]]; then
    # HLT sparse-tree SEP still forms dense Newton Jacobians. Chunking theta
    # draws preserves the batched GPU path while bounding QR workspace.
    HLT_SEP_BATCH_CHUNK_SIZE_EFFECTIVE="$DEFAULT_HLT_SEP_BATCH_CHUNK_SIZE"
  else
    HLT_SEP_BATCH_CHUNK_SIZE_EFFECTIVE="0"
  fi
else
  HLT_SEP_BATCH_CHUNK_SIZE_EFFECTIVE="$HLT_SEP_BATCH_CHUNK_SIZE"
fi
HLT_SURROGATE_BUNDLE_PATH="${HLT_SURROGATE_BUNDLE_PATH:-$RESULT_ROOT/hlt_${MODE}_surrogate_bundle.snn.npz}"
HLT_TARGET_ARRAYS_CHECKPOINT_PATH="${HLT_TARGET_ARRAYS_CHECKPOINT_PATH:-$RESULT_ROOT/hlt_${MODE}_target_arrays.npz}"
HLT_SURROGATE_HMC_DRAWS_PATH="${HLT_SURROGATE_HMC_DRAWS_PATH:-$RESULT_ROOT/hlt_${MODE}_surrogate_hmc_draws.npz}"
ROM1_HMC_DRAWS_PATH="${ROM1_HMC_DRAWS_PATH:-$RESULT_ROOT/hlt_${MODE}_rom1_hmc_draws.npz}"
ROM1_HMC_OUTPUT_PATH="${ROM1_HMC_OUTPUT_PATH:-$RESULT_ROOT/hlt_${MODE}_rom1_static_hmc.json}"
POSTERIOR_COMPARISON_PATH="${POSTERIOR_COMPARISON_PATH:-$RESULT_ROOT/hlt_${MODE}_posterior_comparison.json}"
POSTERIOR_COMPARISON_CSV_PATH="${POSTERIOR_COMPARISON_CSV_PATH:-$RESULT_ROOT/hlt_${MODE}_posterior_comparison.csv}"

OPTIONAL_FLAGS=()
if [[ "${ONLY_FULL_SUCCESS:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--only-full-success)
fi
if [[ "${HLT_THETA_INCLUDE_REFERENCE:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-theta-include-reference)
else
  OPTIONAL_FLAGS+=(--no-hlt-theta-include-reference)
fi
if [[ "${HLT_DROP_RUNTIME_FAILURES:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-drop-runtime-failures)
else
  OPTIONAL_FLAGS+=(--no-hlt-drop-runtime-failures)
fi
if [[ "${HLT_RUNTIME_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-runtime-preflight-only)
else
  OPTIONAL_FLAGS+=(--no-hlt-runtime-preflight-only)
fi
if [[ "${HLT_REUSE_SURROGATE_BUNDLE:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-reuse-surrogate-bundle)
else
  OPTIONAL_FLAGS+=(--no-hlt-reuse-surrogate-bundle)
fi
if [[ "${HLT_SAVE_TARGET_ARRAYS_CHECKPOINT:-1}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-save-target-arrays-checkpoint)
else
  OPTIONAL_FLAGS+=(--no-hlt-save-target-arrays-checkpoint)
fi
if [[ "${HLT_REUSE_TARGET_ARRAYS_CHECKPOINT:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-reuse-target-arrays-checkpoint)
else
  OPTIONAL_FLAGS+=(--no-hlt-reuse-target-arrays-checkpoint)
fi
if [[ "${HLT_REQUIRE_FULL_TARGET_SUCCESS:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-require-full-target-success)
else
  OPTIONAL_FLAGS+=(--no-hlt-require-full-target-success)
fi
if [[ "${HLT_REQUIRE_JAX_PARITY:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-require-jax-parity)
else
  OPTIONAL_FLAGS+=(--no-hlt-require-jax-parity)
fi
if [[ "${HLT_REQUIRE_HMC:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-require-hmc)
else
  OPTIONAL_FLAGS+=(--no-hlt-require-hmc)
fi
if [[ "${HLT_REQUIRE_SOLVED_STEADY_STATE:-0}" == "1" ]]; then
  OPTIONAL_FLAGS+=(--hlt-require-solved-steady-state)
else
  OPTIONAL_FLAGS+=(--no-hlt-require-solved-steady-state)
fi
if [[ -n "${HLT_MIN_VALIDATION_IMPROVEMENT_MEAN:-}" ]]; then
  OPTIONAL_FLAGS+=(--hlt-min-validation-improvement-mean "$HLT_MIN_VALIDATION_IMPROVEMENT_MEAN")
fi
if [[ -n "${HLT_MAX_VALIDATION_RMSE_MEAN:-}" ]]; then
  OPTIONAL_FLAGS+=(--hlt-max-validation-rmse-mean "$HLT_MAX_VALIDATION_RMSE_MEAN")
fi
if [[ -n "${HLT_MAX_HMC_ACCEPTED_SHARE:-}" ]]; then
  OPTIONAL_FLAGS+=(--hlt-max-hmc-accepted-share "$HLT_MAX_HMC_ACCEPTED_SHARE")
fi

mkdir -p "$RESULT_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
HLT_GPU_ESTIMATION_DRY_RUN=1
MODE=$MODE
DEVICE=$DEVICE
REQUIRE_GPU=$REQUIRE_GPU
RESULT_ROOT=$RESULT_ROOT
HLT_TARGET_BUILDER=$HLT_TARGET_BUILDER_EFFECTIVE
HLT_SEP_BATCH_CHUNK_SIZE=$HLT_SEP_BATCH_CHUNK_SIZE_EFFECTIVE
HLT_PARAMETER_SET=$HLT_PARAMETER_SET
HLT_THETA_DESIGN=${HLT_THETA_DESIGN:-perturbation}
HLT_THETA_DESIGN_SET=${HLT_THETA_DESIGN_SET:-auto}
HLT_THETA_INCLUDE_REFERENCE=${HLT_THETA_INCLUDE_REFERENCE:-0}
HLT_DROP_RUNTIME_FAILURES=${HLT_DROP_RUNTIME_FAILURES:-0}
HLT_RUNTIME_PREFLIGHT_ONLY=${HLT_RUNTIME_PREFLIGHT_ONLY:-0}
HLT_MIN_RUNTIME_SUCCESSFUL_THETA=${HLT_MIN_RUNTIME_SUCCESSFUL_THETA:-1}
HLT_THETA_DRAWS=$HLT_THETA_DRAWS
HLT_PERIODS=$HLT_PERIODS
HLT_SHOCK_SCALE=${HLT_SHOCK_SCALE:-0.02}
HLT_PARAMETER_PERTURBATION=${HLT_PARAMETER_PERTURBATION:-1e-6}
HLT_TARGET_MIN_STABLE_PERIODS=${HLT_TARGET_MIN_STABLE_PERIODS:--1}
HLT_SURROGATE_BUNDLE_PATH=$HLT_SURROGATE_BUNDLE_PATH
HLT_TARGET_ARRAYS_CHECKPOINT_PATH=$HLT_TARGET_ARRAYS_CHECKPOINT_PATH
HLT_SAVE_TARGET_ARRAYS_CHECKPOINT=${HLT_SAVE_TARGET_ARRAYS_CHECKPOINT:-1}
HLT_REUSE_TARGET_ARRAYS_CHECKPOINT=${HLT_REUSE_TARGET_ARRAYS_CHECKPOINT:-0}
HLT_SURROGATE_HMC_DRAWS_PATH=$HLT_SURROGATE_HMC_DRAWS_PATH
HLT_REUSE_SURROGATE_BUNDLE=${HLT_REUSE_SURROGATE_BUNDLE:-0}
HLT_REQUIRE_FULL_TARGET_SUCCESS=${HLT_REQUIRE_FULL_TARGET_SUCCESS:-0}
HLT_MIN_FULL_SUCCESS_SHARE=${HLT_MIN_FULL_SUCCESS_SHARE:-0.0}
HLT_MIN_ACCEPTED_SAMPLES=${HLT_MIN_ACCEPTED_SAMPLES:-1}
HLT_MIN_VALIDATION_IMPROVEMENT_MEAN=${HLT_MIN_VALIDATION_IMPROVEMENT_MEAN:-}
HLT_REQUIRE_JAX_PARITY=${HLT_REQUIRE_JAX_PARITY:-0}
HLT_REQUIRE_HMC=${HLT_REQUIRE_HMC:-0}
HLT_REQUIRE_SOLVED_STEADY_STATE=${HLT_REQUIRE_SOLVED_STEADY_STATE:-0}
HLT_STEADY_STATE_MAX_ITER=${HLT_STEADY_STATE_MAX_ITER:-100}
HLT_MAX_HMC_ACCEPTED_SHARE=${HLT_MAX_HMC_ACCEPTED_SHARE:-}
SEP_PERIODS=$SEP_PERIODS
SEP_ORDER=$SEP_ORDER
SEP_NNODES=$SEP_NNODES
SEP_SHOCK_SCALE=${SEP_SHOCK_SCALE:-1.0}
SEP_MAX_ITER=$SEP_MAX_ITER
SEP_LINEAR_SOLVER=${SEP_LINEAR_SOLVER:-qr}
SEP_LINE_SEARCH=${SEP_LINE_SEARCH:-1}
VERBOSE_PROGRESS=${VERBOSE_PROGRESS:-0}
PROGRESS_CHUNK_INTERVAL=${PROGRESS_CHUNK_INTERVAL:-1}
EPOCHS=$EPOCHS
HIDDEN=$HIDDEN
BLOCKS=$BLOCKS
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
TRAIN_DTYPE=$TRAIN_DTYPE
LIKELIHOOD_PERIODS=$LIKELIHOOD_PERIODS
LIKELIHOOD_RUNTIME_MODE=${LIKELIHOOD_RUNTIME_MODE:-fixed-reference}
LIKELIHOOD_QME_ALGORITHM=${LIKELIHOOD_QME_ALGORITHM:-schur}
LIKELIHOOD_STATIC_ROWS_MODE=${LIKELIHOOD_STATIC_ROWS_MODE:-reference}
JAX_LOG_DENSITY_REPEAT_EVALS=${JAX_LOG_DENSITY_REPEAT_EVALS:-0}
JAX_LOG_DENSITY_BATCH_SIZE=${JAX_LOG_DENSITY_BATCH_SIZE:-0}
JAX_LOG_DENSITY_BATCH_REPEAT_EVALS=${JAX_LOG_DENSITY_BATCH_REPEAT_EVALS:-0}
HMC_WARMUP=$HMC_WARMUP
HMC_SAMPLES=$HMC_SAMPLES
HMC_CHAINS=$HMC_CHAINS
HMC_LEAPFROG_STEPS=$HMC_LEAPFROG_STEPS
HMC_STEP_SIZE=$HMC_STEP_SIZE
HMC_MAX_RETRIES=${HMC_MAX_RETRIES:-3}
RUN_ROM1_COMPARISON=${RUN_ROM1_COMPARISON:-0}
ROM1_HMC_DRAWS_PATH=$ROM1_HMC_DRAWS_PATH
ROM1_HMC_OUTPUT_PATH=$ROM1_HMC_OUTPUT_PATH
POSTERIOR_COMPARISON_PATH=$POSTERIOR_COMPARISON_PATH
FAIL_ON_QUALITY_GATE=${FAIL_ON_QUALITY_GATE:-0}
EOF
  exit 0
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON" -m pip install --upgrade pip wheel setuptools
  "$PYTHON" -m pip install -e '.[dev,cuda13]'
fi

"$PYTHON" - <<'PY' | tee "$RESULT_ROOT/environment.txt"
import jax, os, platform, sys
print("python", sys.version)
print("platform", platform.platform())
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("x64", jax.config.read("jax_enable_x64"))
print("XLA_PYTHON_CLIENT_PREALLOCATE", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
print("XLA_FLAGS", os.environ.get("XLA_FLAGS"))
PY
nvidia-smi > "$RESULT_ROOT/nvidia_smi_start.txt" 2>&1 || true

REQUIRE_GPU_FLAG=""
if [[ "$REQUIRE_GPU" == "1" ]]; then
  REQUIRE_GPU_FLAG="--require-gpu"
fi

echo "Running HLT GPU estimation MODE=$MODE into $RESULT_ROOT"
"$PYTHON" benchmarks/profile_surrogate_pipeline_gpu.py \
  --mode hlt-fixed-ss-smoke \
  --device "$DEVICE" $REQUIRE_GPU_FLAG \
  --hlt-case-name medium_sw07_hlt \
  --hlt-parameter-set "$HLT_PARAMETER_SET" \
  --hlt-steady-state-mode "${HLT_STEADY_STATE_MODE:-fixed-reference}" \
  --hlt-steady-state-tol "${HLT_STEADY_STATE_TOL:-1e-10}" \
  --hlt-steady-state-max-iter "${HLT_STEADY_STATE_MAX_ITER:-100}" \
  --hlt-theta-draws "$HLT_THETA_DRAWS" \
  --hlt-periods "$HLT_PERIODS" \
  --hlt-shock-scale "${HLT_SHOCK_SCALE:-0.02}" \
  --hlt-parameter-perturbation "${HLT_PARAMETER_PERTURBATION:-1e-6}" \
  --hlt-theta-design "${HLT_THETA_DESIGN:-perturbation}" \
  --hlt-theta-design-set "${HLT_THETA_DESIGN_SET:-auto}" \
  --hlt-min-runtime-successful-theta "${HLT_MIN_RUNTIME_SUCCESSFUL_THETA:-1}" \
  --hlt-surrogate-bundle-path "$HLT_SURROGATE_BUNDLE_PATH" \
  --hlt-target-arrays-checkpoint-path "$HLT_TARGET_ARRAYS_CHECKPOINT_PATH" \
  --hlt-target-builder "$HLT_TARGET_BUILDER_EFFECTIVE" \
  --hlt-target-min-stable-periods "${HLT_TARGET_MIN_STABLE_PERIODS:--1}" \
  --hlt-min-full-success-share "${HLT_MIN_FULL_SUCCESS_SHARE:-0.0}" \
  --hlt-min-accepted-samples "${HLT_MIN_ACCEPTED_SAMPLES:-1}" \
  --hlt-sep-batch-chunk-size "$HLT_SEP_BATCH_CHUNK_SIZE_EFFECTIVE" \
  --hlt-sep-order-ladder "${HLT_SEP_ORDER_LADDER:-auto}" \
  --hlt-sep-periods-ladder "${HLT_SEP_PERIODS_LADDER:-auto}" \
  --hlt-sep-max-iter-ladder "${HLT_SEP_MAX_ITER_LADDER:-auto}" \
  --hlt-sep-shock-scale-ladder "${HLT_SEP_SHOCK_SCALE_LADDER:-1.0,0.5,0.25,0.1,0.0}" \
  --hlt-target-max-logged-failures "${HLT_TARGET_MAX_LOGGED_FAILURES:-20}" \
  --sep-periods "$SEP_PERIODS" \
  --sep-order "$SEP_ORDER" \
  --sep-nnodes "$SEP_NNODES" \
  --sep-shock-scale "${SEP_SHOCK_SCALE:-1.0}" \
  --sep-max-iter "$SEP_MAX_ITER" \
  --sep-tol "${SEP_TOL:-1e-8}" \
  --sep-accept-tol "${SEP_ACCEPT_TOL:-1e-5}" \
  --sep-linear-solver "${SEP_LINEAR_SOLVER:-qr}" \
  "$SEP_LINE_SEARCH_FLAG" \
  "$VERBOSE_PROGRESS_FLAG" \
  --progress-chunk-interval "${PROGRESS_CHUNK_INTERVAL:-1}" \
  --epochs "$EPOCHS" \
  --hidden "$HIDDEN" \
  --blocks "$BLOCKS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --train-dtype "$TRAIN_DTYPE" \
  --validation-fraction "${VALIDATION_FRACTION:-0.1}" \
  --split-by-theta \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --hlt-likelihood-periods "$LIKELIHOOD_PERIODS" \
  --hlt-likelihood-runtime-mode "${LIKELIHOOD_RUNTIME_MODE:-fixed-reference}" \
  --hlt-likelihood-qme-algorithm "${LIKELIHOOD_QME_ALGORITHM:-schur}" \
  --hlt-likelihood-static-rows-mode "${LIKELIHOOD_STATIC_ROWS_MODE:-reference}" \
  --hlt-surrogate-inversion-maxit "${INVERSION_MAXIT:-4}" \
  --hlt-surrogate-inversion-tol "${INVERSION_TOL:-1e-5}" \
  --hlt-surrogate-inversion-lambda "${INVERSION_LAMBDA:-1e-4}" \
  --hlt-jax-log-density-smoke \
  "$JAX_LOG_DENSITY_GRADIENT_FLAG" \
  --hlt-jax-log-density-repeat-evals "${JAX_LOG_DENSITY_REPEAT_EVALS:-0}" \
  --hlt-jax-log-density-repeat-perturbation "${JAX_LOG_DENSITY_REPEAT_PERTURBATION:-0.0}" \
  --hlt-jax-log-density-batch-size "${JAX_LOG_DENSITY_BATCH_SIZE:-0}" \
  --hlt-jax-log-density-batch-repeat-evals "${JAX_LOG_DENSITY_BATCH_REPEAT_EVALS:-0}" \
  --hlt-jax-log-density-batch-perturbation "${JAX_LOG_DENSITY_BATCH_PERTURBATION:-0.0}" \
  --hlt-jax-shock-solver "${SHOCK_SOLVER:-rom}" \
  --hlt-jax-batch-replay \
  "$DIFFERENTIATE_SHOCKS_FLAG" \
  --hlt-surrogate-hmc-warmup "$HMC_WARMUP" \
  --hlt-surrogate-hmc-samples "$HMC_SAMPLES" \
  --hlt-surrogate-hmc-chains "$HMC_CHAINS" \
  --hlt-surrogate-hmc-leapfrog-steps "$HMC_LEAPFROG_STEPS" \
  --hlt-surrogate-hmc-step-size "$HMC_STEP_SIZE" \
  --hlt-surrogate-hmc-target-accept-prob "${HMC_TARGET_ACCEPT:-0.8}" \
  --hlt-surrogate-hmc-initial-jitter "${HMC_INITIAL_JITTER:-0.02}" \
  --hlt-surrogate-hmc-prior-width-scale "${HMC_PRIOR_WIDTH_SCALE:-0.01}" \
  --hlt-surrogate-hmc-prior-width-floor "${HMC_PRIOR_WIDTH_FLOOR:-1e-4}" \
  --hlt-surrogate-hmc-min-accepted-share "${HMC_MIN_ACCEPTED_SHARE:-0.01}" \
  --hlt-surrogate-hmc-max-retries "${HMC_MAX_RETRIES:-3}" \
  --hlt-surrogate-hmc-retry-step-size-factor "${HMC_RETRY_STEP_SIZE_FACTOR:-0.25}" \
  --hlt-surrogate-hmc-draws-output "$HLT_SURROGATE_HMC_DRAWS_PATH" \
  --hlt-surrogate-hmc-seed "${HMC_SEED:-20260923}" \
  "${OPTIONAL_FLAGS[@]}" \
  --output "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.json" \
  2>&1 | tee "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.log"

if [[ "${RUN_ROM1_COMPARISON:-0}" == "1" ]]; then
  ROM1_REQUIRE_GPU_FLAG=""
  if [[ "$REQUIRE_GPU" == "1" ]]; then
    ROM1_REQUIRE_GPU_FLAG="--force-gpu"
  fi
  echo "Running comparable ROM1 static-HMC posterior into $ROM1_HMC_OUTPUT_PATH"
  "$PYTHON" benchmarks/static_hmc_sampling_speed.py \
    --preset sw07_hlt \
    --case medium_sw07_hlt \
    --parameters "$HLT_PARAMETER_SET" \
    --periods "$LIKELIHOOD_PERIODS" \
    --chains "${ROM1_HMC_CHAINS:-$HMC_CHAINS}" \
    --warmup "${ROM1_HMC_WARMUP:-$HMC_WARMUP}" \
    --samples "${ROM1_HMC_SAMPLES:-$HMC_SAMPLES}" \
    --leapfrog-steps "${ROM1_HMC_LEAPFROG_STEPS:-$HMC_LEAPFROG_STEPS}" \
    --step-size "${ROM1_HMC_STEP_SIZE:-$HMC_STEP_SIZE}" \
    --target-accept-prob "${ROM1_HMC_TARGET_ACCEPT:-${HMC_TARGET_ACCEPT:-0.8}}" \
    --initial-jitter "${ROM1_HMC_INITIAL_JITTER:-${HMC_INITIAL_JITTER:-0.02}}" \
    --prior-width-scale "${ROM1_HMC_PRIOR_WIDTH_SCALE:-${HMC_PRIOR_WIDTH_SCALE:-0.01}}" \
    --prior-width-floor "${ROM1_HMC_PRIOR_WIDTH_FLOOR:-${HMC_PRIOR_WIDTH_FLOOR:-1e-4}}" \
    --dtype "${ROM1_DTYPE:-float64}" \
    --platform "$DEVICE" $ROM1_REQUIRE_GPU_FLAG \
    --qme-algorithm "${ROM1_QME_ALGORITHM:-${LIKELIHOOD_QME_ALGORITHM:-schur_gpu}}" \
    --posterior-draws-output "$ROM1_HMC_DRAWS_PATH" \
    --verbose \
    --output "$ROM1_HMC_OUTPUT_PATH" \
    2>&1 | tee "$RESULT_ROOT/hlt_${MODE}_rom1_static_hmc.log"

  echo "Comparing ROM1 and surrogate posterior draws into $POSTERIOR_COMPARISON_PATH"
  "$PYTHON" benchmarks/compare_posterior_draws.py \
    --left "$ROM1_HMC_DRAWS_PATH" \
    --right "$HLT_SURROGATE_HMC_DRAWS_PATH" \
    --left-label "linear_rom1" \
    --right-label "sep_resnn_surrogate" \
    --output "$POSTERIOR_COMPARISON_PATH" \
    --csv-output "$POSTERIOR_COMPARISON_CSV_PATH" \
    2>&1 | tee "$RESULT_ROOT/hlt_${MODE}_posterior_comparison.log"
fi

nvidia-smi > "$RESULT_ROOT/nvidia_smi_end.txt" 2>&1 || true

"$PYTHON" - \
  "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.json" \
  "$ROM1_HMC_DRAWS_PATH" \
  "$POSTERIOR_COMPARISON_PATH" <<'PY' | tee "$RESULT_ROOT/summary.txt"
import json, sys
path = sys.argv[1]
rom1_draws_path = sys.argv[2]
posterior_comparison_path = sys.argv[3]
payload = json.load(open(path))
result = payload["results"]["hlt_fixed_ss_smoke"]
hmc = result["surrogate_hmc"]
log_density = result["jax_surrogate_log_density"]
lik = result["surrogate_inversion_likelihood"]
target = result.get("target_diagnostics", {})
print("status", result["status"])
print("backend", result["backend"])
print("hlt_parameter_set", result["hlt_parameter_set"])
print("parameter_count", len(result["parameter_subset"]))
print("theta_draws", result["theta_draws"])
print("train_size", result["train_size"], "val_size", result["val_size"])
print("train_dtype", result.get("train_dtype"))
print("training_loop", result.get("training_metadata", {}).get("training_loop"))
print("surrogate_bundle_path", result.get("surrogate_bundle_path"))
print("surrogate_bundle_reused", result.get("surrogate_bundle_reused"))
print("pipeline_s", result["pipeline_s"])
print("target_builder", target.get("builder"), "accepted_samples", target.get("accepted_samples"))
print("target_runtime_prepared_theta", target.get("runtime_prepared_theta_draws"))
print("target_runtime_dropped_theta", target.get("runtime_dropped_theta_count"))
print("target_theta_full_success_count", target.get("theta_full_success_count"))
print("target_fallback_share", target.get("fallback_share"))
print("target_accepted_by_order", target.get("accepted_by_branching_order"))
print("steady_state_solved_count", result.get("steady_state_solved_count"))
print("steady_state_fallback_count", result.get("steady_state_fallback_count"))
print("steady_state_attempted_solved_count", result.get("steady_state_attempted_solved_count"))
print("steady_state_attempted_fallback_count", result.get("steady_state_attempted_fallback_count"))
print("strict_solved_steady_state_preflight", result.get("strict_solved_steady_state_preflight"))
print("likelihood_status", lik.get("status"), "likelihood", lik.get("total_loglikelihood"))
print("jax_log_density_status", log_density.get("status"), "parity_ok", log_density.get("parity_ok"))
print("hmc_status", hmc.get("status"))
print("hmc_draws", hmc.get("post_warmup_draws"), "hmc_elapsed_s", hmc.get("elapsed_s"))
print("hmc_draws_per_second", hmc.get("draws_per_second"))
print("hmc_accepted_share", hmc.get("accepted_share"))
print("hmc_retry_count", hmc.get("retry_count"), "hmc_initial_step_size", hmc.get("initial_step_size"))
print("surrogate_hmc_draws", hmc.get("posterior_draws"))
print("rom1_hmc_draws", rom1_draws_path)
print("posterior_comparison", posterior_comparison_path)
quality = result.get("quality_gate", {})
print("quality_gate_status", quality.get("status"))
print("quality_gate_issues", quality.get("issues"))
print("validation_improvement_mean", result.get("validation_improvement_mean"))
print("validation_rmse_mean", result.get("validation_rmse_mean"))
print("output", path)
PY

if [[ "${FAIL_ON_QUALITY_GATE:-0}" == "1" ]]; then
  "$PYTHON" - "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.json" <<'PY'
import json
import sys
path = sys.argv[1]
payload = json.load(open(path))
result = payload["results"]["hlt_fixed_ss_smoke"]
quality = result.get("quality_gate", {})
if result.get("status") != "ok" or quality.get("status") == "failed":
    print("Quality gate failed; see summary.txt and the full JSON for diagnostics.", file=sys.stderr)
    sys.exit(1)
PY
fi

echo "Wrote $RESULT_ROOT"
