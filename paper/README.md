# Methods-paper working package

This directory is a claim-scoped working paper package for the nonlinear Gal\'i
estimation pipeline. It is deliberately not labelled submission-ready while the
confirmatory evidence is incomplete.

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
- `generated/pilot_results.tex`: the manuscript table.

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

Run the focused validation tests with:

```bash
.venv/bin/python -m pytest -q \
  tests/test_methods_paper_evidence.py \
  tests/test_stochastic_pilot_evidence.py \
  tests/test_gali_synthetic_data.py \
  tests/test_static_hmc.py \
  tests/test_publication_readiness_gate.py \
  tests/test_surrogate_pipeline_profile.py
```

## Evidence status

The preserved suite is an engineering pilot. It uses the complete 23-equation
model, but its target construction has branching order zero and one node. The
corrected HMC run targets a parameter-dependent surrogate profile objective,
not a full nonlinear SEP likelihood. Its Schur QME path also invokes a SciPy host
callback and is not GPU-native. The preserved RMSE also uses a parameter block
that calibrated per-output residual shrinkage, so it is labelled a calibration
metric rather than an untouched test result. These limitations are stated in
the abstract, results table, limitations section, and generated claim ledger.
The archived payload also duplicated equation-level innovation scales in the
inversion prior; future evidence uses unit-variance standardized innovations
and a provenance-captured synthetic-data generator.

The confirmatory plan is in `../docs/submission_readiness_plan.md`. The paper
should be promoted to a submission draft only when every hard gate is backed by
raw artifacts and a frozen protocol.

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
