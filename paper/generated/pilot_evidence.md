# Methods-paper evidence snapshot

Schema: `surrogatenn-dsge-methods-paper-evidence/v1`

**Evidence level:** engineering pilot

**Publication ready:** no

## Claim ledger

| Claim | Status | Auditable statement |
|---|---|---|
| `full_model_pipeline_executes` | `supported_by_pilot` | The complete 23-equation, 16-parameter, three-shock nonlinear Galí source executes through target generation, surrogate fitting, likelihood evaluation, and a sampling smoke test. |
| `nonlinear_extended_path_targets_solve_in_pilot_design` | `supported_by_pilot` | The order-0 nonlinear extended-path pilot accepted 80 targets across 8 parameter paths with maximum recorded residual 9.924e-09. This is not yet a stochastic-expectations SEP validation. |
| `surrogate_improves_rom_in_pilot_design` | `calibration_set_only` | The default pilot reports mean calibration-set improvement over the ROM of 53.9 percent after fitting per-output shrinkage on that same parameter block. The value is not an untouched out-of-sample estimate. |
| `jax_python_likelihood_implementation_parity` | `supported_by_pilot` | JAX and Python evaluations of the same surrogate profile inversion objective agree within 1.705e-13 at the audited calibration. |
| `target_corrected_sampler_executes` | `resource_feasibility_only` | A target-corrected sampler can use fixed-reference proposal gradients and evaluate the parameter-dependent full-JAX surrogate profile objective only at trajectory endpoints without differentiating that target. |
| `exact_full_nonlinear_posterior` | `not_supported` | The pilot does not establish exact sampling from a full nonlinear SEP posterior because its endpoint target remains a surrogate profile objective. |
| `posterior_convergence_or_fidelity` | `not_supported` | The pilot budgets are too small for R-hat, effective-sample-size, posterior moment, or coverage claims. |
| `computational_speedup` | `not_supported` | The suite lacks repeated same-work exact-baseline timings and therefore does not establish an end-to-end speedup. |

## Hard blockers

- A frozen, reproducible synthetic-data generator with recorded seed, burn-in, initial condition, and data-generating parameters.
- Independent MacroModelling.jl trajectory and likelihood comparisons.
- Positive-order stochastic SEP with multiple quadrature nodes and all shocks active.
- An untouched test split; residual-shrinkage calibration must use a separate partition or be disabled.
- A full nonlinear likelihood or an explicitly surrogate-defined estimand.
- Explicit invalid-solve support handling rather than a finite log-density plateau.
- Four tuned chains with convergence and effective-sample-size diagnostics.
- Posterior comparison with an exact or independently validated reference.
- Multi-seed, repeated, synchronized end-to-end timing against the same-work baseline.
- Galí-specific parity validation for the GPU-native QME solver; the audited Schur path uses a sequential SciPy host callback.
- A clean, content-addressed software environment and revision.
