# Methods-paper working package

This directory is a claim-scoped conference working-paper package for the
nonlinear Gal\'i estimation pipeline. The current draft supports an auditable
validation-method contribution and reduced-horizon algorithmic parity. It does
not claim production-horizon likelihood parity, posterior fidelity, or an
end-to-end speed advantage while the confirmatory evidence is incomplete.

## Rebuild the preserved pilot evidence

From the repository root:

```bash
.venv/bin/python benchmarks/build_methods_paper_evidence.py \
  benchmarks/results/gpuhub_exports/gali_gpu_suite_20261021T220300Z \
  --output-dir paper/generated
```

The command reads the preserved raw results and writes:

- `generated/pilot_evidence.json`: machine-readable evidence and source hashes;
- `generated/pilot_evidence.md`: human-readable claim ledger; and
- `generated/pilot_results.tex`: the pilot manuscript table; and
- `generated/gpu_profile_results.tex`: the claim-scoped archived GPU execution
  profile and telemetry boundary.

Normalize the first provenance-captured stochastic pilot separately:

```bash
.venv/bin/python benchmarks/build_stochastic_pilot_evidence.py \
  benchmarks/results/gpuhub_exports/gali_stochastic_sep_pilot_20261023T215504Z \
  --archive benchmarks/results/gpuhub_exports/gali_stochastic_sep_pilot_20261023T215504Z.tar.gz \
  --output-dir paper/generated
```

This writes the hash-verified negative-result record and
`generated/stochastic_pilot_results.tex`. A failed strict gate remains a failed
gate in both the machine-readable evidence and the manuscript.

Build the reduced-horizon stochastic target evidence from the clean preflight
and clean horizon diagnostics with:

```bash
.venv/bin/python benchmarks/build_stochastic_horizon_evidence.py \
  benchmarks/results/local_gali_h5_stochastic_preflight_95122e6 \
  --horizon-result benchmarks/results/local_gali_alltheta_p12_h3_level_inf_95122e6.json \
  --horizon-result benchmarks/results/local_gali_alltheta_p12_h4_level_inf_95122e6.json \
  --horizon-result benchmarks/results/local_gali_alltheta_p12_h5_level_inf_95122e6.json \
  --horizon-six-result benchmarks/results/local_gali_theta1_p12_h6_both_level_inf_95122e6.json \
  --output-dir paper/generated
```

The builder checks source hashes, clean-revision metadata, recorded commands,
the exact stochastic SEP configuration, all eight path-level acceptance
decisions across seven unique parameter points, and the horizon-six failure
under both solver implementations. It emits JSON, Markdown,
and `generated/stochastic_horizon_results.tex`. The evidence supports a reduced
horizon-five target pilot but explicitly does not support terminal-horizon,
likelihood, posterior, or speed claims.

Build the independent full-model Julia/Python core-parity evidence with:

```bash
.venv/bin/python benchmarks/validate_gali_full_julia_core_parity.py \
  --upstream-repo /path/to/clean/MacroModelling.jl \
  --output /tmp/gali_full_core_parity.json
.venv/bin/python benchmarks/build_gali_core_parity_evidence.py \
  /tmp/gali_full_core_parity.json \
  --output-dir paper/generated
```

The validator requires byte-identical local and upstream Gal\'i sources, keeps
all eight evaluations in the denominator, aligns variables and state/shock
columns by name, and refuses the evidence gate when either repository is dirty.
The generated table supports non-stochastic steady-state and first-order Schur
parity only; it does not by itself support nonlinear trajectory or likelihood
parity.

Build the independent reduced-horizon nonlinear-tree evidence with:

```bash
.venv/bin/python benchmarks/validate_gali_full_julia_sep_parity.py \
  --upstream-repo /path/to/clean/MacroModelling.jl \
  --theta-indices 0,1,2,3,4,5,6,7 \
  --output /tmp/gali_full_sep_parity.json
.venv/bin/python benchmarks/build_gali_sep_parity_evidence.py \
  /tmp/gali_full_sep_parity.json \
  --output-dir paper/generated
```

This validator compares the complete five-period sparse scenario tree, not
only a central path, with order one, three Gauss--Hermite nodes per shock, and
all three first-period shock channels active. It requires 8/8 joint successes,
clean repositories, frozen tolerances and configuration, and a reproducible
duplicate reference case. The evidence builder recomputes tree and expected-path
discrepancies from the archived raw states rather than trusting reported summary
metrics. The generated evidence does not support a production-horizon
trajectory, multi-period nonlinear likelihood, posterior, or GPU claim.

Build the independent one-transition nonlinear likelihood evidence with:

```bash
.venv/bin/python benchmarks/validate_gali_full_julia_sep_likelihood.py \
  --upstream-repo /path/to/clean/MacroModelling.jl \
  --output /tmp/gali_sep_likelihood_parity.json
.venv/bin/python benchmarks/build_gali_sep_likelihood_evidence.py \
  /tmp/gali_sep_likelihood_parity.json \
  --output-dir paper/generated
```

The validator generates one transition from the complete nonlinear Gal\'i
model, then makes the Python public likelihood and an independent Julia adapter
recover all three standardized innovations. The gate compares recovered shocks,
the complete finite-difference observation Jacobian, residuals, and the scalar
likelihood. It requires clean repositories and a $10^{-12}$ nonlinear-solver
tolerance. This is one-transition algorithmic parity; multi-period filtering,
the parameter-surface likelihood, and the production horizon remain open.

Run the focused validation tests with:

```bash
.venv/bin/python -m pytest -q \
  tests/test_methods_paper_evidence.py \
  tests/test_stochastic_pilot_evidence.py \
  tests/test_stochastic_horizon_evidence.py \
  tests/test_gali_full_julia_core_parity.py \
  tests/test_gali_core_parity_evidence.py \
  tests/test_gali_full_julia_sep_parity.py \
  tests/test_gali_sep_parity_evidence.py \
  tests/test_gali_full_julia_sep_likelihood.py \
  tests/test_gali_sep_likelihood_evidence.py \
  tests/test_model_sep.py \
  tests/test_gali_synthetic_data.py \
  tests/test_static_hmc.py \
  tests/test_publication_readiness_gate.py \
  tests/test_gpu_run_profile.py \
  tests/test_surrogate_pipeline_profile.py
```

## Evidence status

The preserved suite is an engineering pilot. The new clean preflight uses the
complete 23-equation model, order-one stochastic SEP, and three quadrature
nodes; all eight path evaluations across seven unique parameter points pass at
horizon five. The hard design fails at horizon
six under both solvers, and first-step outcomes still move from horizon four to
five. The corrected HMC run targets a parameter-dependent surrogate profile
objective, not a full nonlinear SEP likelihood. Its Schur QME path also invokes
a SciPy host callback and is not GPU-native. The preserved RMSE uses a parameter
block that calibrated per-output residual shrinkage, so it remains a calibration
metric rather than an untouched test result. The archived payload also
duplicated equation-level innovation scales in the inversion prior; future
evidence uses unit-variance standardized innovations and a provenance-captured
synthetic-data generator.

The independent core-parity gate now passes on the complete Gal\'i source:
8/8 evaluations agree between Python and a clean MacroModelling.jl checkout for
the non-stochastic steady state and first-order Schur solution. A separate 8/8
comparison also matches the complete five-period nonlinear sparse scenario
tree within its frozen tolerances. A direct one-transition likelihood adapter
also passes recovered-shock, Jacobian, residual, and scalar-likelihood parity.
This removes core local-solution, reduced-horizon nonlinear-tree, and
one-transition likelihood uncertainties. Those results support the narrow
methods-paper claim. Multi-period and production-horizon likelihood, posterior,
and same-work GPU claims remain incomplete and are explicitly excluded.

The confirmatory plan is in `../docs/submission_readiness_plan.md`. The paper
should be promoted from an algorithmic-parity methods result to a production
structural-posterior and speed claim only when every corresponding hard gate is
backed by raw artifacts and a frozen protocol.

## Run a provenance-captured stochastic pilot

From a clean commit, the reduced full-equation Gal\'i launcher now defaults to
stochastic SEP order one and three quadrature nodes:

```bash
DEVICE=gpu PYTHON=.venv/bin/python \
  bash benchmarks/run_nonlinear_gali_pilot.sh
```

The launcher refuses a dirty worktree, records the exact command, Git revision,
allowlisted numerical environment, design and HMC seeds, and input hashes before
execution, then records the exit code and output hashes in
`run_manifest.json`. The publication launcher disables residual-shrinkage
calibration so the held-out parameter block remains untouched. Budgets remain
intentionally small; a successful pilot is still not confirmatory evidence.

On `DEVICE=gpu`, the launcher also wraps the workload with
`benchmarks/profile_gpu_run.py` by default. It samples device-wide NVIDIA
utilization, memory, power, temperature, and clock counters every 0.5 seconds,
then writes `gpu_profile/gpu_profile_samples.csv` and
`gpu_profile/gpu_profile_summary.json`. The summary includes upper-tail
utilization, peak memory, and trapezoidal energy estimates. These counters are
system-wide, so publication runs require a dedicated GPU. Set
`GPU_PROFILE_INTERVAL_S` to change the interval, or `GPU_PROFILE=0` only for a
non-evidence diagnostic run. A single profile never establishes acceleration;
the timing gate still requires repeated same-work baseline and treatment runs.
Each sample launches an `nvidia-smi` query, so comparison arms must use the same
interval and the final timing appendix should include unprofiled sensitivity
runs to bound monitoring overhead.

## Generate the frozen confirmatory data

After committing the protocol, run:

```bash
.venv/bin/python benchmarks/generate_gali_synthetic_data.py \
  --periods 100 --burn-in 50 \
  --output benchmarks/data/gali_full_nonlinear_confirmatory_v1.json
```

The generator refuses a dirty worktree by default and records the full latent
simulation path and solver diagnostics. The resulting payload is then selected
with `GALI_PAYLOAD` and `GALI_CASE_NAME` in the launcher.
