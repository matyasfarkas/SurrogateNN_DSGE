# End-to-end estimator-validation protocol

## Why this gate exists

Component parity is necessary but does not validate an estimator. Agreement on
a steady state, Schur solution, short nonlinear tree, or isolated likelihood
contribution can coexist with biased shock recovery or a distorted posterior.
The publication claim therefore requires a matched, known-truth validation of
the complete estimation pipeline.

This protocol ports the decisive logic of the original
`SurrogateNN_Estimation.jl` validation package. It applies to the complete
23-equation nonlinear Gal\'i model while permitting a prespecified reduced
parameter experiment. The current primary design estimates
$(\phi_\pi,\phi_y)$; reducing the parameter dimension does not permit reducing
the model equations or replacing the nonlinear data-generating process.

## Matched comparison

Generate one frozen synthetic panel from the declared nonlinear reference at a
known truth. The direct-reference and residual-surrogate arms must then use:

- byte-identical observations and initial conditions;
- identical priors, parameter transforms, and support rules;
- identical shock treatment and every other objective term;
- identical initializations, chain lengths, adaptation rules, and random seeds;
- the same attempted cases and denominators.

Only the nonlinear transition evaluator may differ. If shocks are optimized
rather than integrated out, both arms must be described as matched profiled
objectives. Agreement between two profiled targets does not turn either target
into a marginal likelihood.

## Sequential validation layers

1. **Reference stress path.** Verify that the direct nonlinear solver is finite
   and stable at the truth, posterior-region points, and frozen boundary cases.
2. **Held-out residual layer.** Evaluate the residual surrogate on parameter,
   shock, and trajectory groups excluded from fitting and calibration.
3. **Common-shock inversion grid.** Hold the synthetic panel and shock rule
   fixed, evaluate direct and surrogate objectives on the same prespecified
   grid, and double the grid resolution until posterior summaries stabilize.
4. **Matched HMC.** Run direct-reference and surrogate chains with common data,
   priors, transforms, objective terms, initializations, and seeds. Compare
   intervals, truth coverage, posterior means in combined-MCSE units,
   convergence diagnostics, and every numerical failure.
5. **Identification probes.** Preserve cases in which both objectives agree but
   fail to recover a weakly identified truth. Such agreement is an
   identification failure, not evidence of surrogate validity.
6. **Computational value.** Only after the correctness layers pass, compare
   identical GPU workloads, effective samples per second, and continuous device
   telemetry. GPU execution alone is not estimator validation.

## Publication acceptance record

The publication-mode readiness gate requires:

- finite objective values at 100 percent of the frozen grid points;
- a passed grid-resolution convergence check;
- direct/surrogate interval overlap for every estimated parameter;
- truth coverage by both arms for every identified parameter;
- maximum posterior-mean difference no larger than 0.5 combined MCSE;
- direct numerical-failure share no larger than 0.5 percent;
- surrogate numerical-failure share no larger than 0.1 percent;
- at least 4,000 retained draws per objective;
- all attempted cases and expected identification failures preserved.

The required raw artifacts are the frozen data-generating-process record,
direct and surrogate grid values, resolution-doubling summaries, chain draws,
adaptation and convergence diagnostics, failure ledgers, posterior comparison
table, seeds, manifests, and content hashes. A summary without these inputs does
not pass the gate.
