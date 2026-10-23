#!/usr/bin/env bash
set -euo pipefail

# Reduced experiment on the complete 23-equation nonlinear Galí model.
# Only the estimated parameter subset and numerical budget are reduced.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cpu}"
RUN_LABEL="${RUN_LABEL:-gali_full_nonlinear_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-benchmarks/results/$RUN_LABEL}"
mkdir -p "$RESULT_ROOT"
GALI_SEP_ORDER="${GALI_SEP_ORDER:-1}"
GALI_SEP_NNODES="${GALI_SEP_NNODES:-3}"
GALI_SEED="${GALI_SEED:-20261019}"
GALI_HMC_SEED="${GALI_HMC_SEED:-20261023}"

ARGS=(
  --mode hlt-fixed-ss-smoke
  --device "$DEVICE"
  --hlt-model-source benchmarks/model_sources/Gali_2015_chapter_3_nonlinear.jl
  --hlt-payload benchmarks/gali_nonlinear_pilot_payload.json
  --hlt-case-name gali_full_nonlinear_reduced_pilot
  --hlt-parameter-set payload
  --hlt-theta-design perturbation
  --hlt-theta-include-reference
  --hlt-theta-draws "${GALI_THETA_DRAWS:-8}"
  --hlt-parameter-perturbation "${GALI_PARAMETER_PERTURBATION:-0.12}"
  --hlt-periods "${GALI_PERIODS:-10}"
  --hlt-shock-scale "${GALI_SHOCK_SCALE:-0.12}"
  --hlt-steady-state-mode solve
  --hlt-require-solved-steady-state
  --hlt-steady-state-max-iter 100
  --hlt-first-order-qme-algorithm schur
  --hlt-target-builder batched-sep
  --hlt-target-min-stable-periods "${GALI_PERIODS:-10}"
  --hlt-require-full-target-success
  --hlt-min-runtime-successful-theta "${GALI_THETA_DRAWS:-8}"
  --hlt-min-full-success-share 1.0
  --hlt-min-accepted-samples "${GALI_THETA_DRAWS:-8}"
  --sep-periods "${GALI_PERIODS:-10}"
  --sep-order "$GALI_SEP_ORDER"
  --sep-nnodes "$GALI_SEP_NNODES"
  --sep-max-iter 50
  --sep-tol 1e-8
  --sep-accept-tol 1e-5
  --sep-linear-solver qr
  --sep-line-search
  --epochs "${GALI_EPOCHS:-400}"
  --hidden "${GALI_HIDDEN:-64}"
  --blocks "${GALI_BLOCKS:-1}"
  --batch-size "${GALI_TRAIN_BATCH_SIZE:-32}"
  --learning-rate "${GALI_LEARNING_RATE:-0.002}"
  --train-dtype float64
  --validation-fraction 0.25
  --split-by-theta
  --hlt-likelihood-periods "${GALI_PERIODS:-10}"
  --hlt-likelihood-runtime-mode "${GALI_LIKELIHOOD_RUNTIME_MODE:-fixed-reference}"
  --hlt-jax-log-density-smoke
  --hlt-require-jax-parity
  --hlt-surrogate-hmc-warmup "${GALI_HMC_WARMUP:-25}"
  --hlt-surrogate-hmc-samples "${GALI_HMC_SAMPLES:-50}"
  --hlt-surrogate-hmc-chains "${GALI_HMC_CHAINS:-1}"
  --hlt-surrogate-hmc-chain-batch-size "${GALI_HMC_CHAIN_BATCH_SIZE:-1}"
  --hlt-surrogate-hmc-gradient-mode "${GALI_HMC_GRADIENT_MODE:-target}"
  --hlt-surrogate-hmc-leapfrog-steps "${GALI_HMC_LEAPFROG_STEPS:-3}"
  --hlt-surrogate-hmc-step-size "${GALI_HMC_STEP_SIZE:-0.02}"
  --hlt-surrogate-hmc-seed "$GALI_HMC_SEED"
  --hlt-surrogate-hmc-target-accept-prob 0.8
  --hlt-surrogate-hmc-prior-width-scale 0.12
  --hlt-surrogate-hmc-prior-width-floor 1e-4
  --hlt-surrogate-hmc-min-accepted-share 0.05
  --hlt-surrogate-hmc-max-retries 2
  --hlt-require-hmc
  --hlt-surrogate-hmc-draws-output "$RESULT_ROOT/posterior_draws.npz"
  --hlt-surrogate-bundle-path "$RESULT_ROOT/surrogate_bundle.snn.npz"
  --output "$RESULT_ROOT/result.json"
  --seed "$GALI_SEED"
)

if [[ "${GALI_JAX_LOG_DENSITY_GRADIENT:-1}" == "1" ]]; then
  ARGS+=(--hlt-jax-log-density-gradient)
else
  ARGS+=(--no-hlt-jax-log-density-gradient)
fi

if [[ "$DEVICE" == "gpu" ]]; then
  ARGS+=(--require-gpu)
fi

COMMAND=("${PYTHON:-python}" benchmarks/profile_surrogate_pipeline_gpu.py "${ARGS[@]}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

MANIFEST_PATH="$RESULT_ROOT/run_manifest.json"
MANIFEST_PRE_ARGS=(
  pre
  --output "$MANIFEST_PATH"
  --input benchmarks/publication_run_manifest.py
  --input benchmarks/run_nonlinear_gali_pilot.sh
  --input benchmarks/profile_surrogate_pipeline_gpu.py
  --input benchmarks/model_sources/Gali_2015_chapter_3_nonlinear.jl
  --input benchmarks/gali_nonlinear_pilot_payload.json
  --seed "design=$GALI_SEED"
  --seed "hmc=$GALI_HMC_SEED"
)
if [[ "${ALLOW_DIRTY_MANIFEST:-0}" == "1" ]]; then
  MANIFEST_PRE_ARGS+=(--allow-dirty)
fi
MANIFEST_PRE_ARGS+=(--command "${COMMAND[@]}")
"${PYTHON:-python}" benchmarks/publication_run_manifest.py "${MANIFEST_PRE_ARGS[@]}"

set +e
"${COMMAND[@]}" 2>&1 | tee "$RESULT_ROOT/run.log"
PIPE_RESULT=("${PIPESTATUS[@]}")
set -e
RUN_RC="${PIPE_RESULT[0]}"
if [[ "$RUN_RC" == "0" && "${PIPE_RESULT[1]}" != "0" ]]; then
  RUN_RC="${PIPE_RESULT[1]}"
fi

if [[ "$RUN_RC" == "0" ]]; then
  set +e
  "${PYTHON:-python}" - "$RESULT_ROOT/result.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
result = payload["results"]["hlt_fixed_ss_smoke"]
if result.get("status") != "ok":
    issues = result.get("quality_gate", {}).get("issues", [])
    raise SystemExit("Galí pilot quality gate failed: " + "; ".join(map(str, issues)))
PY
  QUALITY_RC="$?"
  set -e
  if [[ "$QUALITY_RC" != "0" ]]; then
    RUN_RC="$QUALITY_RC"
  fi
fi

"${PYTHON:-python}" benchmarks/publication_run_manifest.py finalize \
  --manifest "$MANIFEST_PATH" \
  --exit-code "$RUN_RC" \
  --artifact "$RESULT_ROOT/result.json" \
  --artifact "$RESULT_ROOT/posterior_draws.npz" \
  --artifact "$RESULT_ROOT/surrogate_bundle.snn.npz" \
  --artifact "$RESULT_ROOT/run.log"

if [[ "$RUN_RC" != "0" ]]; then
  exit "$RUN_RC"
fi

echo "Galí pilot result: $RESULT_ROOT/result.json"
