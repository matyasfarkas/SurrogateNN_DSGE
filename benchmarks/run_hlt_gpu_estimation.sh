#!/usr/bin/env bash
set -euo pipefail

# Run the HLT nonlinear surrogate-estimation pipeline on a CUDA/JAX host.
#
# Usage:
#   MODE=smoke bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=pilot HLT_THETA_DRAWS=64 HMC_SAMPLES=64 bash benchmarks/run_hlt_gpu_estimation.sh
#   MODE=full bash benchmarks/run_hlt_gpu_estimation.sh
#
# The current posterior stage samples the trained-surrogate inversion likelihood
# with fixed reference steady state and fixed first-order ROM matrices. It does
# not yet recompute parameter-specific steady states/ROMs inside HMC.

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

case "$MODE" in
  smoke)
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
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-1}"
    HMC_WARMUP="${HMC_WARMUP:-0}"
    HMC_SAMPLES="${HMC_SAMPLES:-1}"
    HMC_CHAINS="${HMC_CHAINS:-1}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-1}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.001}"
    ;;
  pilot)
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
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-20}"
    HMC_WARMUP="${HMC_WARMUP:-32}"
    HMC_SAMPLES="${HMC_SAMPLES:-64}"
    HMC_CHAINS="${HMC_CHAINS:-16}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-4}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.005}"
    ;;
  full)
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
    LIKELIHOOD_PERIODS="${LIKELIHOOD_PERIODS:-80}"
    HMC_WARMUP="${HMC_WARMUP:-500}"
    HMC_SAMPLES="${HMC_SAMPLES:-1000}"
    HMC_CHAINS="${HMC_CHAINS:-32}"
    HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-6}"
    HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.003}"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use smoke, pilot, or full." >&2
    exit 2
    ;;
esac

mkdir -p "$RESULT_ROOT"

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
  --hlt-steady-state-mode fixed-reference \
  --hlt-theta-draws "$HLT_THETA_DRAWS" \
  --hlt-periods "$HLT_PERIODS" \
  --hlt-parameter-perturbation "${HLT_PARAMETER_PERTURBATION:-1e-6}" \
  --hlt-target-builder "${HLT_TARGET_BUILDER:-adaptive-sep}" \
  --hlt-target-min-stable-periods "${HLT_TARGET_MIN_STABLE_PERIODS:--1}" \
  --hlt-sep-order-ladder "${HLT_SEP_ORDER_LADDER:-auto}" \
  --hlt-sep-periods-ladder "${HLT_SEP_PERIODS_LADDER:-auto}" \
  --hlt-sep-max-iter-ladder "${HLT_SEP_MAX_ITER_LADDER:-auto}" \
  --hlt-sep-shock-scale-ladder "${HLT_SEP_SHOCK_SCALE_LADDER:-1.0,0.5,0.25,0.1,0.0}" \
  --hlt-target-max-logged-failures "${HLT_TARGET_MAX_LOGGED_FAILURES:-20}" \
  --sep-periods "$SEP_PERIODS" \
  --sep-order "$SEP_ORDER" \
  --sep-nnodes "$SEP_NNODES" \
  --sep-max-iter "$SEP_MAX_ITER" \
  --sep-tol "${SEP_TOL:-1e-8}" \
  --sep-accept-tol "${SEP_ACCEPT_TOL:-1e-5}" \
  --epochs "$EPOCHS" \
  --hidden "$HIDDEN" \
  --blocks "$BLOCKS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --validation-fraction "${VALIDATION_FRACTION:-0.1}" \
  --split-by-theta \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --hlt-likelihood-periods "$LIKELIHOOD_PERIODS" \
  --hlt-surrogate-inversion-maxit "${INVERSION_MAXIT:-4}" \
  --hlt-surrogate-inversion-tol "${INVERSION_TOL:-1e-5}" \
  --hlt-surrogate-inversion-lambda "${INVERSION_LAMBDA:-1e-4}" \
  --hlt-jax-log-density-smoke \
  --hlt-jax-shock-solver "${SHOCK_SOLVER:-rom}" \
  --hlt-jax-batch-replay \
  --no-hlt-jax-differentiate-shocks \
  --hlt-surrogate-hmc-warmup "$HMC_WARMUP" \
  --hlt-surrogate-hmc-samples "$HMC_SAMPLES" \
  --hlt-surrogate-hmc-chains "$HMC_CHAINS" \
  --hlt-surrogate-hmc-leapfrog-steps "$HMC_LEAPFROG_STEPS" \
  --hlt-surrogate-hmc-step-size "$HMC_STEP_SIZE" \
  --hlt-surrogate-hmc-target-accept-prob "${HMC_TARGET_ACCEPT:-0.8}" \
  --hlt-surrogate-hmc-initial-jitter "${HMC_INITIAL_JITTER:-0.02}" \
  --hlt-surrogate-hmc-prior-width-scale "${HMC_PRIOR_WIDTH_SCALE:-0.01}" \
  --hlt-surrogate-hmc-prior-width-floor "${HMC_PRIOR_WIDTH_FLOOR:-1e-4}" \
  --hlt-surrogate-hmc-seed "${HMC_SEED:-20260923}" \
  --output "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.json" \
  2>&1 | tee "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.log"

nvidia-smi > "$RESULT_ROOT/nvidia_smi_end.txt" 2>&1 || true

"$PYTHON" - "$RESULT_ROOT/hlt_${MODE}_surrogate_estimation.json" <<'PY' | tee "$RESULT_ROOT/summary.txt"
import json, sys
path = sys.argv[1]
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
print("pipeline_s", result["pipeline_s"])
print("target_builder", target.get("builder"), "accepted_samples", target.get("accepted_samples"))
print("target_theta_full_success_count", target.get("theta_full_success_count"))
print("target_fallback_share", target.get("fallback_share"))
print("target_accepted_by_order", target.get("accepted_by_branching_order"))
print("likelihood_status", lik.get("status"), "likelihood", lik.get("total_loglikelihood"))
print("jax_log_density_status", log_density.get("status"), "parity_ok", log_density.get("parity_ok"))
print("hmc_status", hmc.get("status"))
print("hmc_draws", hmc.get("post_warmup_draws"), "hmc_elapsed_s", hmc.get("elapsed_s"))
print("hmc_draws_per_second", hmc.get("draws_per_second"))
print("hmc_accepted_share", hmc.get("accepted_share"))
print("output", path)
PY

echo "Wrote $RESULT_ROOT"
