# Full nonlinear Galí pilot

This pilot uses the complete 23-equation, 16-parameter, three-shock
`Gali_2015_chapter_3_nonlinear` model. The model is not replaced by the
three-equation linear approximation. The initial experiment is made small by
estimating only the Taylor-rule coefficients `ϕᵖⁱ` and `ϕʸ`, using a short
horizon, and limiting the training and HMC budgets.

The purpose of the first run is to verify that the complete numerical path is
composable before paying for a large GPU experiment:

1. parse the full MacroModelling source;
2. solve a parameter-specific steady state and first-order Schur solution for
   every design point;
3. generate nonlinear SEP targets with fixed-shape batched JAX solves;
4. hold out complete parameter vectors for validation;
5. train the residual network;
6. check Python/JAX likelihood parity; and
7. run a short static-HMC smoke test and save its draws.

## Run it locally

Use the project environment if it is already installed:

```bash
PYTHON=.venv/bin/python bash benchmarks/run_nonlinear_gali_pilot.sh
```

The defaults use CPU, eight parameter vectors, ten observed periods, a
15-period SEP terminal horizon, 400 training epochs, one chain, 25 warmup
draws, and 50 retained draws. The longer SEP horizon is deliberate: controlled
CPU/GPU parity experiments showed that a ten-period terminal horizon truncates
the persistent price-dispersion response, whereas the otherwise identical
15-period solve reaches the strict residual tolerance. Change it independently
with `GALI_SEP_PERIODS`; do not infer it from the data horizon. A fast
integration check is:

```bash
PYTHON=.venv/bin/python \
GALI_THETA_DRAWS=4 \
GALI_PERIODS=4 \
GALI_SEP_PERIODS=6 \
GALI_EPOCHS=5 \
GALI_HMC_WARMUP=1 \
GALI_HMC_SAMPLES=2 \
GALI_HMC_LEAPFROG_STEPS=1 \
bash benchmarks/run_nonlinear_gali_pilot.sh
```

Set `DRY_RUN=1` to print the resolved command without starting the experiment.
Each run creates a timestamped directory below `benchmarks/results/` containing
`result.json`, `run.log`, `surrogate_bundle.snn.npz`, and
`posterior_draws.npz`. The script exits nonzero if its numerical quality gate
does not pass.

## Run the same pilot on a GPU

```bash
DEVICE=gpu PYTHON=.venv/bin/python \
bash benchmarks/run_nonlinear_gali_pilot.sh
```

The runner then requires JAX to expose a GPU. Batched SEP is the principal
accelerated workload; its theta chunk can be controlled in the underlying
profile command if device memory is limited.

## Freeze confirmatory synthetic data

The original pilot payload duplicated the equation-level `std_a`, `std_nu`,
and `std_z` scales in the inversion prior. The corrected payload treats
`eps_a`, `eps_nu`, and `eps_z` as unit-variance innovations because the full
nonlinear equations already apply those scale parameters. Archived pilots
remain useful only for software feasibility and Python/JAX parity of the same
mis-scaled objective.

Generate the confirmatory synthetic-data artifact only from a clean revision:

```bash
.venv/bin/python benchmarks/generate_gali_synthetic_data.py \
  --periods 100 \
  --burn-in 50 \
  --output benchmarks/data/gali_full_nonlinear_confirmatory_v1.json
```

The generator records the exact parameter vector, steady state, independent
shock and measurement seeds, all latent innovations, burn-in states, retained
observations, SEP diagnostics, Git revision, dirty-worktree state, and content
hashes. Use the frozen artifact without editing it:

```bash
GALI_PAYLOAD=benchmarks/data/gali_full_nonlinear_confirmatory_v1.json \
GALI_CASE_NAME=gali_full_nonlinear_confirmatory_v1 \
DEVICE=gpu PYTHON=.venv/bin/python \
bash benchmarks/run_nonlinear_gali_pilot.sh
```

## What the first pilot does and does not establish

The default likelihood and HMC stage uses a fixed steady state and fixed
first-order ROM after the parameter-specific preflight and nonlinear target
generation. It is a pipeline and surrogate-validation pilot, not an exact
posterior result and not publication evidence.

For the next resource pilot, set:

```bash
DEVICE=gpu \
GALI_LIKELIHOOD_RUNTIME_MODE=full-jax \
GALI_JAX_LOG_DENSITY_GRADIENT=0 \
GALI_HMC_GRADIENT_MODE=fixed-reference-corrected \
GALI_HMC_CHAINS=2 \
GALI_HMC_CHAIN_BATCH_SIZE=1 \
GALI_HMC_WARMUP=1 \
GALI_HMC_SAMPLES=2 \
GALI_HMC_LEAPFROG_STEPS=1 \
bash benchmarks/run_nonlinear_gali_pilot.sh
```

That configuration uses the cheap fixed-reference density for leapfrog
gradients but applies Metropolis correction with the parameter-specific
full-JAX target Hamiltonian at each endpoint. The target is never
differentiated. The correction is exact for the selected target density because
the proposal map remains reversible and volume preserving. This test answers
whether scalar full-JAX target evaluations fit and are fast enough; it is still
far too short for inference.

## Promotion criteria

The machine-readable acceptance contract is
`benchmarks/publication_readiness_gate.py`. It distinguishes `smoke`,
`evidence`, and `publication` modes and deliberately scopes any pass to a
reduced experiment using the complete Galí equation set. Publication mode
requires, among other checks, independent Julia comparisons, 200 held-out
parameter vectors, likelihood and gradient parity, four chains with at least
1,000 warmup and 1,000 retained draws per chain, R-hat/ESS diagnostics, exact
posterior comparisons, repeated timings, hashes, and a clean recorded Git
revision.

The recommended progression is therefore:

1. local quick integration check;
2. default full-model CPU pilot;
3. tiny corrected full-JAX GPU resource pilot;
4. moderate `evidence` design with independent Julia cross-checks; and
5. only then the preregistered `publication` run.

The first three stages are designed to fail cheaply. More GPU memory is useful
for larger batched SEP designs and concurrent chains, but it is not a substitute
for the validation, convergence, and cross-implementation gates.
