# GPUHUB HLT Nonlinear ResNN Runbook

This runbook prepares the current executable HLT nonlinear SEP/ResNN pipeline
for GPUHUB. It is a staged paid-run workflow: setup, smoke, pilot, then full.

Current estimator caveat: the default posterior stage samples the trained
surrogate inversion likelihood with a fixed reference steady state and fixed
first-order ROM matrices. Use `--likelihood-runtime-mode full-jax` only as an
experimental diagnostic because full parameter-specific SS/ROM differentiation
is still compile-heavy.

## First Command After Restart

Run this in SSH or a GPUHUB JupyterLab terminal:

```bash
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

## Recommended Full Staged Run

This is the command I would run first on the 6000D or a similarly large GPU:

```bash
RUN_LABEL="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_BASE="benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}"
mkdir -p "$RESULTS_BASE"
nohup python scripts/gpuhub_hlt_resnn_runner.py \
  --mode smoke_pilot_full \
  --skip-repo-sync \
  --skip-install \
  --jax-extra auto \
  --hlt-target-builder adaptive-sep \
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

The runner stops if `smoke` fails. If `smoke` passes, it runs `pilot`; if
`pilot` passes, it starts `full`.

## More Aggressive Full Overrides

The script defaults for `MODE=full` are:

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
benchmarks/results/gpuhub_hlt_resnn_${RUN_LABEL}/{smoke,pilot,full}/
```

The most important fields are:

- `target_status`
- `accepted_samples`
- `fallback_share`
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
