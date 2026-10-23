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

Run the focused validation tests with:

```bash
.venv/bin/python -m pytest -q \
  tests/test_methods_paper_evidence.py \
  tests/test_static_hmc.py \
  tests/test_publication_readiness_gate.py \
  tests/test_surrogate_pipeline_profile.py
```

## Evidence status

The preserved suite is an engineering pilot. It uses the complete 23-equation
model, but its target construction has branching order zero and one node. The
corrected HMC run targets a parameter-dependent surrogate inversion likelihood,
not a full nonlinear SEP likelihood. Its Schur QME path also invokes a SciPy host
callback and is not GPU-native. These limitations are stated in the abstract,
results table, limitations section, and generated claim ledger.

The confirmatory plan is in `../docs/submission_readiness_plan.md`. The paper
should be promoted to a submission draft only when every hard gate is backed by
raw artifacts and a frozen protocol.
