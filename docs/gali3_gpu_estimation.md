# Gali 3-Equation GPU Estimation Smoke

This benchmark is the smallest release-facing GPU estimation check. It uses a
MacroModelling-style Gali three-equation New Keynesian source file at
`benchmarks/model_sources/Gali_3eq_linear.jl`, generates synthetic observations
from the Schur-certified first-order solution, and runs the existing
NumPyro/JAX posterior benchmark with vectorized chains.

It is meant to answer whether the package can do the full compiled estimation
loop on a GPU:

- parse MacroModelling-style source syntax
- solve the first-order model
- build the Kalman likelihood
- differentiate the log posterior in JAX
- run vectorized HMC/NUTS chains on the selected JAX backend
- optionally audit sampled draws against Schur/QZ determinacy

It is not by itself evidence that full nonlinear HLT/SW07 global estimation is
economically feasible. That still requires full-size sparse-tree SEP target
generation, residual-surrogate training, and a switching ROM/FOM likelihood.

## Local CPU Calibration

Run this first on any machine:

```bash
python benchmarks/posterior_sampling_speed.py \
  --preset gali3_nk \
  --periods 32 \
  --parameters gali3_policy_4 \
  --kernel hmc \
  --hmc-num-steps 4 \
  --hmc-step-size 0.02 \
  --warmup 8 \
  --samples 8 \
  --chains 2 \
  --chain-method vectorized \
  --qme-algorithm doubling \
  --preflight \
  --preflight-reps 2 \
  --schur-support-draws 8 \
  --output benchmarks/results/gali3_cpu_smoke.json \
  --verbose
```

## GPU Run

On Colab or GPUHUB, install a CUDA-capable JAX wheel first, then run:

```bash
python benchmarks/posterior_sampling_speed.py \
  --preset gali3_nk \
  --periods 80 \
  --parameters gali3_policy_7 \
  --kernel hmc \
  --hmc-num-steps 8 \
  --hmc-step-size 0.02 \
  --warmup 64 \
  --samples 128 \
  --chains 32 \
  --chain-method vectorized \
  --dtype float32 \
  --platform gpu \
  --force-gpu \
  --qme-algorithm doubling \
  --preflight \
  --preflight-reps 5 \
  --schur-support-draws 64 \
  --output benchmarks/results/gali3_gpu_hmc.json \
  --verbose
```

Use `--qme-algorithm doubling` for the fully JAX-native fast path. Keep
`--schur-support-draws` positive so the output reports whether any doubling
draws are accepted outside Schur/QZ unique-stable support.

## Sufficiency

Passing this benchmark is sufficient for release smoke coverage of GPU-based
first-order DSGE estimation. It is not sufficient for the paper's home-run
claim. For that, the same profiling structure must be scaled to HLT/SW07 with:

- batched sparse-tree SEP target generation
- residual ResNet training on generated ROM/FOM errors
- switching-order likelihood evaluation
- end-to-end posterior ESS per second
- Schur/QZ support audits when the HMC inner loop uses doubling
