# GPUHub Posterior Sampling Speed Benchmark

This benchmark measures NumPyro/JAX posterior sampling throughput for the
Python SW07/HLT translation as seconds per effective sample.

## Instance Setup

Use a single RTX 5090 instance first. GPUHub instances are containers, and
JupyterLab starts in `/root`; use `/root/gpuhub-tmp` for benchmark code and
outputs so the run is not constrained by the small system disk.

Do not rely on a provider image that advertises `JAX 0.3.10`, Ubuntu 18.04, or
CUDA 11.1. That stack is too old for this repository and for a Blackwell-class
RTX 5090 benchmark. It is acceptable to launch such an image only as a shell if
`nvidia-smi` exposes a modern host driver; immediately reinstall the Python
stack with the bootstrap below.

Open a GPUHub JupyterLab terminal or SSH session and run:

```bash
cd /root/gpuhub-tmp
git clone --depth 1 --branch codex/colab-jax-gemini-profile \
  https://github.com/matyasfarkas/SurrogateNN_DSGE.git
cd SurrogateNN_DSGE

python -m pip install --upgrade pip
python scripts/gpuhub_bootstrap.py --mode setup --jax-extra auto
```

The bootstrap parses `nvidia-smi`, chooses `jax[cuda13]>=0.6` when the host
driver supports CUDA 13, falls back to `jax[cuda12]>=0.6` for CUDA 12-capable
drivers, uninstalls stale JAX/JAXlib packages, installs this repo without
dependency downgrades, and refuses to continue if JAX does not see a GPU.

## Calibration Run

Run this first. It keeps the parameter set small and audits the first posterior
draws against Schur determinacy.

```bash
python scripts/gpuhub_bootstrap.py --mode calibration --skip-repo-sync --skip-install
```

## Proper RTX 5090 Run

Run this only after calibration has finite gradients and no Schur support
violations.

```bash
python scripts/gpuhub_bootstrap.py --mode proper_5090 --skip-repo-sync --skip-install
```

To run calibration and then automatically continue to the proper run only if
the Schur support audit passes:

```bash
python scripts/gpuhub_bootstrap.py --mode both
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
