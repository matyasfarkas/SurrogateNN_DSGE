# GPUHub Posterior Sampling Speed Benchmark

This benchmark measures NumPyro/JAX posterior sampling throughput for the
Python SW07/HLT translation as seconds per effective sample.

## Pre-Deployment Workflow

The correct workflow is:

1. Choose the host and image in the GPUHub create-instance form.
2. Confirm the price and only then create the instance.
3. Open JupyterLab from `Console -> Container Instances -> Jupyterlab`.
4. Open a JupyterLab terminal.
5. Clone the GitHub repository into the data disk, usually
   `/root/autodl-tmp` on current PyTorch images.
6. Run the bootstrap setup/probe.
7. Open `notebooks/gpuhub_sw07_posterior_ess.ipynb` from the cloned repo if an
   interactive notebook run is preferred, or run the CLI benchmark directly.
8. Download benchmark JSON outputs or push/commit them if they should be
   preserved before shutting the instance down.

GPUHub's docs state that JupyterLab's working directory is `/root`, while the
data disk is mounted under `/root`. On the inspected PyTorch 2.12.1 image, the
actual path is `/root/autodl-tmp`; older docs/images may call the analogous
path `/root/gpuhub-tmp`. Use the data disk for the repo and benchmark outputs
so the small system disk is not the bottleneck. JupyterLab upload is useful for
individual files, but GPUHub documents that it does not support folders. For
this project, clone the GitHub repo from a JupyterLab terminal instead of
uploading the repo manually.

The current launch form choice inspected on GPUHub is:

```text
Host: 282b4b870f / G030-R
GPU: 1x RTX 5090 32GB
CPU/RAM: 25 cores / 90GB RAM
Driver/CUDA: 580.105.08 / CUDA 13.0
Image: PyTorch 2.12.1 / Python 3.12(ubuntu22.04) / CUDA 13.0
Billing: Pay-as-you-go
Shown price: $0.46/hour
Account budget: $10.00
```

This image is preferred over the platform's JAX image because GPUHub's
documented JAX image is `JAX 0.3.10 / Python 3.8 / CUDA 11.1`, which is too old
for the current repository. The PyTorch 2.12.1 image is only a modern base
environment; the bootstrap still uninstalls stale JAX packages and installs the
current JAX CUDA wheel explicitly.

## Instance Setup

Use a single RTX 5090 instance first. GPUHub instances are containers, and
JupyterLab starts in `/root`; use `/root/autodl-tmp` when present, otherwise
`/root/gpuhub-tmp`, for benchmark code and outputs so the run is not
constrained by the small system disk.

Do not rely on a provider image that advertises `JAX 0.3.10`, Ubuntu 18.04, or
CUDA 11.1. That stack is too old for this repository and for a Blackwell-class
RTX 5090 benchmark. It is acceptable to launch such an image only as a shell if
`nvidia-smi` exposes a modern host driver; immediately reinstall the Python
stack with the bootstrap below.

Open a GPUHub JupyterLab terminal or SSH session and run:

```bash
cd /root/autodl-tmp
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
