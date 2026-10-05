# GPUHUB HLT Nonlinear ResNN Runbook

This runbook prepares the current executable HLT nonlinear SEP/ResNN pipeline
for GPUHUB. It is a staged paid-run workflow: setup, smoke, calibration,
pilot, estimation-pilot, `final_nonlinear`, then `full_hlt`.

Important distinction: `MODE=estimation_pilot` is a speed/stress run and can use
fixed-reference shortcuts. `MODE=final_nonlinear` is the correctness-first run:
it samples the Julia-comparable narrow 18-parameter support, solves
parameter-specific steady states when possible, requires full SEP target paths,
trains only on fully successful theta draws, evaluates the full-JAX SS/ROM
surrogate likelihood, and rejects the run if any quality gate fails.
`MODE=full_hlt` is the expensive all-parameter HLT stress run: it uses the full
parsed HLT parameter vector, an 8-period SEP horizon, GPU Schur, strict quality
gates, and a smoke-first staged launch. Use it only after `final_nonlinear`
passes or when you explicitly want to test the full HLT envelope.

## First Command After Restart

Run this in SSH or a GPUHUB JupyterLab terminal:

```bash
# Cancel any stale safety shutdown left by a previous profiling wrapper.
shutdown -c || true
cd /root/autodl-tmp
if [ ! -d SurrogateNN_DSGE ]; then
  git clone --branch codex/nonlinear-sep-surrogate-port \
    https://github.com/matyasfarkas/SurrogateNN_DSGE.git
fi
cd SurrogateNN_DSGE
git fetch origin codex/nonlinear-sep-surrogate-port
git checkout codex/nonlinear-sep-surrogate-port
git pull --ff-only origin codex/nonlinear-sep-surrogate-port
python scripts/gpuhub_hlt_resnn_runner.py --mode setup --jax-extra auto
```

Setup installs the CUDA JAX stack, reinstalls the repo editable, and refuses to
continue if JAX cannot see the GPU.

## Recommended Staged Run

Run this first on any paid GPU. It verifies the GPU stack, runs a one-theta
smoke, then runs a bounded HLT calibration before attempting the larger pilot.

```bash
RUN_LABEL="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke_then_calibration \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --hlt-target-builder batched-sep \
  --likelihood-runtime-mode fixed-reference \
  --steady-state-mode fixed-reference \
  --env JAX_LOG_DENSITY_REPEAT_EVALS=3 \
  --env JAX_LOG_DENSITY_BATCH_SIZE=64 \
  --env JAX_LOG_DENSITY_BATCH_REPEAT_EVALS=3 \
  --run-label "$RUN_LABEL" \
  > "$RESULTS_BASE/master.log" 2>&1 &
echo $! > "$RESULTS_BASE/runner.pid"
tail -f "$RESULTS_BASE/master.log"
```

The runner stops if `smoke` fails. If `smoke` passes, it runs `calibration`.
Only run `pilot` after calibration produces finite targets, JAX parity, and
nonzero HMC acceptance.

## Known SEP Memory Bottleneck

The current batched SEP target generator forms a dense Newton Jacobian. For HLT
with `SEP_PERIODS=8`, `SEP_ORDER=1`, sparse Gauss-Hermite nodes, and 66 model
variables, the stacked SEP unknown count is 9,768. One all-theta batch of 32
draws tried to allocate a `f64[32,9768,9768]` Jacobian and OOMed on a 32 GB
5090D. A chunk size of 4 reached about 31 GB VRAM and was still slow in the
first target-generation period.

The new `MODE=estimation_pilot` deliberately uses a shorter 4-period SEP horizon
and defaults `HLT_SEP_BATCH_CHUNK_SIZE=8`, which should keep rough QR workspace
near 12.8 GiB per chunk for 128 theta draws while still testing batched target
generation. The heavier `MODE=full` keeps the 8-period horizon and defaults
chunk size 2, roughly 17.1 GiB per chunk for the same 128-draw thought
experiment. The output JSON records a `target_diagnostics.memory_estimate` block
with the estimated dense Jacobian and rough workspace sizes. Treat this estimate
as a preflight warning, not an exact allocator forecast.

## Pilot After Calibration

If calibration succeeds, run:

```bash
RUN_LABEL="pilot_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode pilot \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --hlt-target-builder batched-sep \
  --likelihood-runtime-mode fixed-reference \
  --steady-state-mode fixed-reference \
  --env HLT_SEP_BATCH_CHUNK_SIZE=2 \
  --env JAX_LOG_DENSITY_REPEAT_EVALS=3 \
  --env JAX_LOG_DENSITY_BATCH_SIZE=64 \
  --env JAX_LOG_DENSITY_BATCH_REPEAT_EVALS=3 \
  --run-label "$RUN_LABEL" \
  > "$RESULTS_BASE/master.log" 2>&1 &
echo $! > "$RESULTS_BASE/runner.pid"
tail -f "$RESULTS_BASE/master.log"
```

Do not use `SEP_LINEAR_SOLVER=normal_equations` or `SEP_LINE_SEARCH=0` as a
default. A local HLT smoke accepted targets with the robust QR + line-search
solver, while the aggressive normal-equations/no-line-search shortcut generated
zero accepted SEP targets.

## Estimation-Like Parallel Pilot

This is the recommended large GPU test before `MODE=full`. It is intended to be
large enough that a local CPU run can take hours, while a 32GB+ GPU should expose
parallel SEP target generation, ResNN training, batched log-density evaluation,
and vectorized HMC chains.

Default `MODE=estimation_pilot` settings:

- `HLT_PARAMETER_SET=sw07_safe_27`
- `HLT_THETA_DRAWS=128`
- `HLT_PERIODS=4`
- `SEP_PERIODS=4`
- `SEP_ORDER=1`
- `SEP_NNODES=3`
- `HLT_TARGET_BUILDER=batched-sep`
- `HLT_SEP_BATCH_CHUNK_SIZE=8`
- `EPOCHS=200`
- `HIDDEN=192`
- `BLOCKS=4`
- `TRAIN_BATCH_SIZE=2048`
- `LIKELIHOOD_PERIODS=80`
- `JAX_LOG_DENSITY_BATCH_SIZE=2048`
- `HMC_WARMUP=256`
- `HMC_SAMPLES=512`
- `HMC_CHAINS=64`

Dry-run locally or on GPUHUB to verify resolved settings without launching:

```bash
MODE=estimation_pilot DRY_RUN=1 bash benchmarks/run_hlt_gpu_estimation.sh
```

Run smoke first, then the estimation pilot:

```bash
RUN_LABEL="estimation_pilot_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke_then_estimation_pilot \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --likelihood-runtime-mode fixed-reference \
  --steady-state-mode fixed-reference \
  --run-label "$RUN_LABEL" \
  > "$RESULTS_BASE/master.log" 2>&1 &
echo $! > "$RESULTS_BASE/runner.pid"
tail -f "$RESULTS_BASE/master.log"
```

On 16 GB GPUs, add `--env HLT_SEP_BATCH_CHUNK_SIZE=4`. On 48 GB or larger
GPUs, try `--env HLT_SEP_BATCH_CHUNK_SIZE=16` only after the default completes.
For a stronger chain-parallel HMC stress, add `--env HMC_CHAINS=128`; keep all
other settings fixed so the speed comparison remains interpretable.

## Final Nonlinear Estimation Run

Use this after `smoke_then_estimation_pilot` has passed. This is the first mode
intended to be interpreted as a full nonlinear HLT ResNN estimation attempt.

Default `MODE=final_nonlinear` settings:

- `HLT_PARAMETER_SET=phase1_18params_narrow`
- `HLT_THETA_DESIGN=prior`
- `HLT_THETA_INCLUDE_REFERENCE=1`
- `HLT_TARGET_BUILDER=batched-sep`
- `HLT_THETA_DRAWS=192`
- `HLT_PERIODS=4`
- `SEP_PERIODS=4`
- `SEP_MAX_ITER=20`
- `ONLY_FULL_SUCCESS=1`
- `HLT_STEADY_STATE_MODE=solve-or-reference`
- `HLT_STEADY_STATE_MAX_ITER=200`
- `LIKELIHOOD_RUNTIME_MODE=full-jax`
- `LIKELIHOOD_QME_ALGORITHM=schur_gpu`
- `JAX_LOG_DENSITY_BATCH_SIZE=0`
- `HMC_WARMUP=500`
- `HMC_SAMPLES=1000`
- `HMC_CHAINS=64`

The final mode writes the full JSON and `summary.txt` even when a quality gate
fails, then exits nonzero by default (`FAIL_ON_QUALITY_GATE=1`). A valid final
run must have:

- `status ok`
- `quality_gate_status ok`
- `target_theta_full_success_count == theta_draws`
- `jax_log_density_status ok` and `parity_ok True`
- `hmc_status ok`
- `hmc_accepted_share <= HLT_MAX_HMC_ACCEPTED_SHARE`
- no solved-steady-state fallback if `HLT_REQUIRE_SOLVED_STEADY_STATE=1`

Dry-run the resolved final settings first:

```bash
MODE=final_nonlinear DRY_RUN=1 bash benchmarks/run_hlt_gpu_estimation.sh
```

Launch final mode through the staged runner:

```bash
RUN_LABEL="final_nonlinear_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke_then_final_nonlinear \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --run-label "$RUN_LABEL" \
  > "$RESULTS_BASE/master.log" 2>&1 &
echo $! > "$RESULTS_BASE/runner.pid"
tail -f "$RESULTS_BASE/master.log"
```

If the solved-steady-state gate fails but all other gates pass, rerun with
`--env HLT_REQUIRE_SOLVED_STEADY_STATE=0` only for diagnostic profiling. Do not
use that override for the final claim.

### Reusing a Saved ResNN Bundle

Every non-preflight HLT stage now writes a portable NN bundle next to the JSON:

```text
$RESULT_ROOT/hlt_${MODE}_surrogate_bundle.snn.npz
```

The bundle stores the frozen ResNN weights, normalization statistics, validation
diagnostics, parameter subset, input/output names, target diagnostics, and
dataset summary. This lets a restarted GPUHUB VM debug likelihood/HMC without
paying again for SEP target generation and training.

Example: rerun final likelihood/HMC diagnostics using an existing final bundle:

```bash
MODE=final_nonlinear \
HLT_REUSE_SURROGATE_BUNDLE=1 \
HLT_SURROGATE_BUNDLE_PATH=/root/autodl-tmp/SurrogateNN_DSGE/benchmarks/results/<run>/final_nonlinear/hlt_final_nonlinear_surrogate_bundle.snn.npz \
JAX_LOG_DENSITY_BATCH_SIZE=0 \
JAX_LOG_DENSITY_BATCH_REPEAT_EVALS=0 \
bash benchmarks/run_hlt_gpu_estimation.sh
```

The reuse path still reruns the HLT steady-state/first-order runtime preflight
and validates the loaded bundle dimensions and parameter subset. The summary
must show `surrogate_bundle_reused True` and `target_builder reused_bundle`.
If the original bundle passed target-generation gates, those target diagnostics
are preserved in the reused run. The currently recovered failed GPUHUB archive
did not contain a bundle because this checkpointing was not enabled yet, so it
cannot be reused retroactively.

## More Aggressive Full Overrides

The older script defaults for `MODE=full` are retained as an aggressive
all-parameter stress profile, not as the current correctness-first final mode.
Prefer `MODE=full_hlt` for an interpretable all-parameter HLT run.

Default `MODE=full_hlt` settings:

- `HLT_PARAMETER_SET=all`
- `HLT_THETA_DRAWS=288`
- `HLT_PERIODS=8`
- `SEP_PERIODS=8`
- `SEP_ORDER=1`
- `SEP_NNODES=3`
- `HLT_SEP_BATCH_CHUNK_SIZE=2`
- `LIKELIHOOD_RUNTIME_MODE=full-jax`
- `LIKELIHOOD_QME_ALGORITHM=schur_gpu`
- `JAX_LOG_DENSITY_BATCH_SIZE=0`
- `HMC_WARMUP=500`
- `HMC_SAMPLES=1000`
- `HMC_CHAINS=64`

Dry-run the full HLT settings first:

```bash
MODE=full_hlt DRY_RUN=1 bash benchmarks/run_hlt_gpu_estimation.sh
```

Launch through the staged runner:

```bash
RUN_LABEL="full_hlt_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke_then_full_hlt \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --run-label "$RUN_LABEL" \
  > "$RESULTS_BASE/master.log" 2>&1 &
echo $! > "$RESULTS_BASE/runner.pid"
tail -f "$RESULTS_BASE/master.log"
```

For a 16 GB GPU, add `--env HLT_SEP_BATCH_CHUNK_SIZE=1`. For a 48 GB or larger
GPU, try `--env HLT_SEP_BATCH_CHUNK_SIZE=4` only after the default chunk size
has completed without OOM.

The older `MODE=full` defaults are:

- `HLT_PARAMETER_SET=all`
- `HLT_THETA_DRAWS=288`
- `HLT_PERIODS=8`
- `SEP_PERIODS=8`
- `SEP_ORDER=1`
- `SEP_NNODES=3`
- `EPOCHS=250`
- `HMC_WARMUP=500`
- `HMC_SAMPLES=1000`
- `HMC_CHAINS=32`

To push harder on a large GPU, append overrides:

```bash
  --env HMC_CHAINS=64 \
  --env TRAIN_BATCH_SIZE=2048 \
  --env HLT_THETA_DRAWS=384 \
```

Do not start with these overrides until the default staged run has produced a
finite likelihood, JAX parity, and nonzero HMC acceptance.

## Monitoring

Use:

```bash
tail -f "$RESULTS_BASE/master.log"
cat "$RESULTS_BASE/master_summary.json"
nvidia-smi
```

Each stage writes:

```text
benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}/{smoke,calibration,pilot,full}/
```

The most important fields are:

- `target_status`
- `accepted_samples`
- `fallback_share`
- `target_diagnostics.memory_estimate`
- `jax_log_density_parity_ok`
- `hmc_status`
- `hmc_draws_per_second`
- `hmc_accepted_share`

## Experimental Full-JAX Diagnostic

Only run this after the fixed-reference staged run succeeds:

```bash
python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke \
  --skip-repo-sync \
  --skip-install \
  --likelihood-runtime-mode full-jax \
  --steady-state-mode solve-or-reference \
  --no-jax-log-density-gradient \
  --env HMC_SAMPLES=0 \
  --run-label "hlt_full_jax_value_only_$(date -u +%Y%m%dT%H%M%SZ)"
```

If value-only full-JAX works, try the same command without
`--no-jax-log-density-gradient`. On the M4 CPU, the tiny full-JAX gradient path
was still compiling after about 2.5 minutes, so do not spend GPU budget on a
large full-JAX HMC run until the tiny diagnostic is acceptable.

## Shutdown

GPUHUB instances are container-style guests. A guest `poweroff` command may not
stop billing. Shut the instance down from the GPUHUB console after the run, or
ask Codex to use the signed-in browser console to stop it.
