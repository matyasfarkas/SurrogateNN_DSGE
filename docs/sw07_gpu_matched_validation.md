# Paper-matched SW07 GPU validation

This protocol converts the Smets--Wouters (2007) experiment in Farkas and
Tatar, *Bayesian Estimation of DSGE Models with Hamiltonian Monte Carlo*, into
an executable external validation target.  It is deliberately separate from
the synthetic `sw07_hlt` throughput benchmark and from the nonlinear
HLT/SEP/ResNN estimator.

Reference: <https://www.matyasfarkas.eu/publication/hmc/hmc.pdf>

## What “matched” means

The `sw07_paper` preset requires all of the following:

- seven observables in the published order: output growth, consumption growth,
  investment growth, hours, inflation, real-wage growth, and the policy rate;
- the 180-quarter 1960Q1--2004Q4 slice from the archived 230-row data file;
- an exact SHA256 match for the archived `usmodel_data.mat`;
- all 36 published estimands;
- the archived Dynare hard bounds and Normal, Beta, Gamma, and inverse-Gamma
  prior densities;
- six presample periods;
- a parameter-dependent JAX steady-state solve inside the likelihood;
- comparison with the published original-mode posterior means and 5/95
  percentiles.

The data are not copied into this repository.  Every result records the input
path, SHA256, row slice, series names, and the `dw -> dwobs` mapping.

## Evidence layers

### A. CPU/GPU numerical parity

At the same 36-parameter initialization, CPU and GPU must agree on the
log-likelihood to `1e-6` absolute error and on every gradient component to
`1e-5`.  Both runs must use float64, the same data fingerprint, the dynamic
steady-state path, zero measurement error, zero numerical jitter, and the
determinacy-aware Schur solver.  The independent Dynare 7.1 likelihood at this
point is `-1045.8945641261721`; each backend must match it within `1e-8`.

The local CPU reference generated on 2026-10-29 gives
`-1045.8945641261862`, an absolute Dynare discrepancy of approximately
`1.4e-11`, with all 36 gradient components finite.  Its first compiled value
and gradient calls took 7.48 and 329.85 seconds, respectively.  The transition
and shock-impact matrices agree with Dynare to maximum absolute errors of
`4.7e-13` and `3.7e-14`.

### B. Empirical posterior replication

The publication run uses at least four chains, 500 warmup iterations, and
1,000 retained draws per chain.  Required diagnostics are zero divergences,
maximum split R-hat at most 1.01, minimum parameter ESS at least 400, and no
maximum-tree-depth saturation.  The original-mode comparison requires:

- all 36 parameters present;
- median posterior-mean error no larger than one quarter of the published 90%
  interval width;
- maximum posterior-mean error no larger than one published interval width;
- at least 32 of 36 GPU posterior means inside the published 90% intervals.

Publication artifacts must include the finite chain-by-draw arrays for all 36
parameters.  The validator recomputes means, quantiles, ESS, and split R-hat
from those raw draws and rejects a result whose stored summaries disagree.  It
also checks the ordered estimand block, GPU backend, float64 precision, Schur
solver, covariance initialization, presample, sampler settings, initialization
basin, and model-source fingerprint; a summary-only JSON cannot pass.

Run the experiment from both the archived Dynare initialization and the prior
means.  This is a basin search, not proof of unimodality.  The HMC paper found
a second SW07 mode and correctly notes that one HMC chain cannot cross widely
separated modes.

### C. Nonlinear estimator validation

Passing A and B validates the GPU/JAX estimation stack on the linearized
empirical SW07 model.  It does **not** validate the nonlinear HLT/SEP/ResNN
likelihood.  That claim still requires the repository's independent nonlinear
trajectory, likelihood, posterior, and same-work timing gates.

The complete upstream MacroModelling.jl source is pinned at
`benchmarks/model_sources/Smets_Wouters_2007.jl` for that later nonlinear
extension.  The present empirical posterior uses the separately pinned
paper-matched linear state-space translation
`benchmarks/model_sources/Smets_Wouters_2007_linear.jl`; no result below is
silently attributed to the nonlinear source.

## Commands

Set the user-owned data path once:

```bash
export SW07_DATA_MAT=/absolute/path/to/usmodel_data.mat
```

Run CPU and GPU preflights.  CPU uses the same launcher with GPU enforcement
disabled by invoking the Python benchmark directly:

```bash
.venv/bin/python benchmarks/posterior_sampling_speed.py \
  --preset sw07_paper --sw07-data-mat "$SW07_DATA_MAT" \
  --parameters sw07_paper_36 --steady-state-mode solve \
  --steady-state-max-iter 20 \
  --platform cpu --qme-algorithm schur --dtype float64 \
  --preflight-only --preflight-reps 3 \
  --output benchmarks/results/sw07_cpu_preflight.json

MODE=preflight INIT=dynare \
  bash benchmarks/run_sw07_paper_gpu_validation.sh
```

After a short pilot succeeds, run both publication initializations:

```bash
MODE=pilot INIT=dynare bash benchmarks/run_sw07_paper_gpu_validation.sh
MODE=pilot INIT=prior_mean bash benchmarks/run_sw07_paper_gpu_validation.sh

MODE=publication INIT=dynare bash benchmarks/run_sw07_paper_gpu_validation.sh
MODE=publication INIT=prior_mean bash benchmarks/run_sw07_paper_gpu_validation.sh
```

Combine the two posterior runs and the CPU/GPU preflights into one fail-closed
report:

```bash
.venv/bin/python benchmarks/validate_sw07_paper_results.py \
  --dynare-run /path/to/publication_dynare.json \
  --prior-mean-run /path/to/publication_prior_mean.json \
  --cpu-preflight /path/to/sw07_cpu_preflight.json \
  --gpu-preflight /path/to/preflight_dynare.json \
  --output benchmarks/results/sw07_publication_gate.json
```

## Reproducibility discrepancy to resolve

The paper declares a 1960Q1--2004Q4 sample, while the archived data file has 230
quarters beginning in 1947 and some archived Dynare command files say
`first_obs=1`.  The executable preset follows the paper and selects rows 50--229
(zero-based), giving 180 quarters.  The same-slice fixed-point likelihood now
matches Dynare, but that numerical parity does not establish which sample was
used to produce the published posterior table.  If the table actually used all
230 observations, report that as a correction and rerun both backends; never
conceal the discrepancy by changing the slice silently.

## Claim boundary

Until the combined gate reports `publication_ready: true`, the defensible text
is: “We implemented a paper-matched SW07 GPU replication protocol.”  After it
passes, the defensible text is: “The GPU/JAX implementation reproduces the
linearized empirical SW07 posterior within preregistered numerical and
statistical tolerances.”  Neither sentence alone establishes the nonlinear
surrogate-estimation result.
