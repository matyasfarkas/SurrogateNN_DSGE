#!/usr/bin/env bash
set -euo pipefail

# Resource-feasibility pilot for the parameter-specific full-JAX HLT posterior.
#
# The pilot reuses a trained surrogate bundle and runs two chains sequentially,
# one chain per compiled batch.  Fixed-reference gradients generate reversible
# leapfrog proposals; the full-JAX likelihood supplies the exact endpoint
# Metropolis correction.  Its purpose is to establish that this corrected HMC
# graph compiles and executes within device memory.
# The resulting four post-warmup draws are not suitable for inference.
#
# Usage:
#   HLT_SURROGATE_BUNDLE_PATH=/path/to/hlt_final_nonlinear_surrogate_bundle.snn.npz \
#     bash benchmarks/run_full_jax_hmc_resource_pilot.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUNDLE_PATH="${HLT_SURROGATE_BUNDLE_PATH:-}"
if [[ -z "$BUNDLE_PATH" ]]; then
  echo "HLT_SURROGATE_BUNDLE_PATH must name an existing trained surrogate bundle." >&2
  exit 2
fi
if [[ ! -f "$BUNDLE_PATH" ]]; then
  echo "Surrogate bundle does not exist: $BUNDLE_PATH" >&2
  exit 2
fi

RUN_LABEL="${RUN_LABEL:-full_jax_hmc_resource_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULT_ROOT="${RESULT_ROOT:-benchmarks/results/$RUN_LABEL}"

# Reproduce the final-nonlinear parameter design so that the saved bundle and
# runtime-preflight theta set agree.  Skip target construction and training.
export MODE=final_nonlinear
export HLT_REUSE_SURROGATE_BUNDLE=1
export HLT_SAVE_TARGET_ARRAYS_CHECKPOINT=0

# The reused bundle records five failed SEP paths.  Those failures remain part
# of the saved diagnostics, but they are not the question tested here.  Keep
# the parity and HMC gates strict so a zero exit status has a narrow meaning.
export HLT_REQUIRE_FULL_TARGET_SUCCESS=0
export HLT_MIN_FULL_SUCCESS_SHARE=0.0
export HLT_MIN_ACCEPTED_SAMPLES=1
export HLT_REQUIRE_JAX_PARITY=1
export HLT_REQUIRE_HMC=1
export HLT_MAX_HMC_ACCEPTED_SHARE=1.0
export FAIL_ON_QUALITY_GATE=1

# Compile a scalar-chain graph and reuse it for the second chain.  The full-JAX
# target is evaluated once per transition but is not differentiated; the cheap
# fixed-reference density supplies every leapfrog gradient.
export HMC_CHAINS="${HMC_CHAINS:-2}"
export HMC_CHAIN_BATCH_SIZE="${HMC_CHAIN_BATCH_SIZE:-1}"
export HMC_GRADIENT_MODE="${HMC_GRADIENT_MODE:-fixed-reference-corrected}"
export HMC_WARMUP="${HMC_WARMUP:-1}"
export HMC_SAMPLES="${HMC_SAMPLES:-2}"
export HMC_LEAPFROG_STEPS="${HMC_LEAPFROG_STEPS:-1}"
export HMC_STEP_SIZE="${HMC_STEP_SIZE:-0.001}"
export HMC_MAX_RETRIES="${HMC_MAX_RETRIES:-0}"

if (( HMC_CHAIN_BATCH_SIZE < 1 || HMC_CHAIN_BATCH_SIZE >= HMC_CHAINS )); then
  echo "Resource pilot requires 1 <= HMC_CHAIN_BATCH_SIZE < HMC_CHAINS." >&2
  echo "For a two-chain batch, for example, use HMC_CHAINS=4 HMC_CHAIN_BATCH_SIZE=2." >&2
  exit 2
fi

# One repeat supplies a post-compilation scalar timing without extending an
# already expensive resource test.  Batched log-density timing remains off.
export JAX_LOG_DENSITY_SMOKE=1
# Smoke-test the full-JAX target value and parity without constructing the
# reverse-mode graph that exhausted GPU memory in the original run.
export JAX_LOG_DENSITY_GRADIENT=0
export JAX_LOG_DENSITY_REPEAT_EVALS="${JAX_LOG_DENSITY_REPEAT_EVALS:-1}"
export JAX_LOG_DENSITY_BATCH_SIZE=0
export JAX_LOG_DENSITY_BATCH_REPEAT_EVALS=0
export RUN_ROM1_COMPARISON=0

echo "Full-JAX HMC resource pilot"
echo "bundle=$BUNDLE_PATH"
echo "result_root=$RESULT_ROOT"
echo "chains=$HMC_CHAINS chain_batch_size=$HMC_CHAIN_BATCH_SIZE"
echo "gradient_mode=$HMC_GRADIENT_MODE"
echo "warmup=$HMC_WARMUP samples=$HMC_SAMPLES leapfrog_steps=$HMC_LEAPFROG_STEPS"

bash benchmarks/run_hlt_gpu_estimation.sh
