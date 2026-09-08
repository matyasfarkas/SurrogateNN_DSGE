# GPUHub Posterior Sampling Speed Benchmark

This benchmark measures NumPyro/JAX posterior sampling throughput for the
Python SW07/HLT translation as seconds per effective sample.

## Instance Setup

Use a single RTX 5090 instance first. GPUHub instances are containers, and
JupyterLab starts in `/root`; use `/root/gpuhub-tmp` for benchmark code and
outputs so the run is not constrained by the small system disk.

Open a GPUHub JupyterLab terminal or SSH session and run:

```bash
cd /root/gpuhub-tmp
git clone --depth 1 --branch codex/colab-jax-gemini-profile \
  https://github.com/matyasfarkas/SurrogateNN_DSGE.git
cd SurrogateNN_DSGE

python -m pip install --upgrade pip
python -m pip install --upgrade --no-cache-dir \
  'numpy>=2.1,<2.3' 'jax[cuda13]>=0.6' 'numpyro>=0.20' \
  'scipy>=1.14,<2' 'sympy>=1.13,<2'
python -m pip install -e . --no-deps

nvidia-smi
python - <<'PY'
import jax
import jax.numpy as jnp
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
print((jnp.ones((1024, 1024), dtype=jnp.float32) @ jnp.ones((1024, 1024), dtype=jnp.float32)).block_until_ready().dtype)
PY
```

## Calibration Run

Run this first. It keeps the parameter set small and audits the first posterior
draws against Schur determinacy.

```bash
python benchmarks/posterior_sampling_speed.py \
  --preset sw07_hlt \
  --periods 80 \
  --parameters calfa,cg,cgy,crdy,crhob,crpi \
  --warmup 16 \
  --samples 16 \
  --chains 2 \
  --chain-method vectorized \
  --dtype float32 \
  --qme-algorithm doubling \
  --target-accept-prob 0.8 \
  --max-tree-depth 8 \
  --prior-width-scale 0.0025 \
  --prior-width-floor 0.0001 \
  --preflight \
  --preflight-reps 1 \
  --schur-support-draws 16 \
  --force-gpu \
  --output benchmarks/results/gpuhub_sw07_calibration_ess.json
```

## Proper RTX 5090 Run

Run this only after calibration has finite gradients and no Schur support
violations.

```bash
python benchmarks/posterior_sampling_speed.py \
  --preset sw07_hlt \
  --periods 160 \
  --parameters sw07_safe_15 \
  --warmup 256 \
  --samples 256 \
  --chains 4 \
  --chain-method vectorized \
  --dtype float32 \
  --qme-algorithm doubling \
  --target-accept-prob 0.8 \
  --max-tree-depth 8 \
  --prior-width-scale 0.0025 \
  --prior-width-floor 0.0001 \
  --preflight \
  --preflight-reps 1 \
  --schur-support-draws 64 \
  --force-gpu \
  --output benchmarks/results/gpuhub_sw07_proper_5090_ess.json
```

## Headline Metric

Use:

```text
throughput.seconds_per_min_ess
```

This is more informative than raw draws per second because the slowest-mixing
parameter controls posterior accuracy. Also inspect:

```text
extra_fields.diverging.count
extra_fields.accept_prob.mean
posterior_diagnostics.max_r_hat
schur_support_audit.doubling_accepts_non_unique_count
```

If `doubling_accepts_non_unique_count` is positive, treat the run as a speed
experiment only. It is not a valid DSGE posterior until the doubling likelihood
is gated by Schur determinacy or another equivalent stability certificate.
