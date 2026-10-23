# Submission-readiness plan for the nonlinear DSGE methods paper

## Frozen claim

The proposed contribution is an auditable nonlinear-DSGE estimation workflow
that combines a first-order reduced-order model, a learned full-minus-reduced
residual, and target-corrected Hamiltonian proposals. The generic endpoint
Metropolis correction is prior art. The paper's contribution must come from the
DSGE integration, numerical validation, posterior fidelity, and measured
speed--accuracy tradeoff.

The current pilot is not publication evidence. It uses deterministic order-zero
extended path, and its corrected target is still a surrogate profile inversion
objective. No manuscript text may call that run an exact nonlinear posterior.

## Critical path

### Experiment A: two-parameter correctness reference

1. Freeze the priors, synthetic data, observables, shock scaling, SEP
   configuration, training design, and case identifiers.
2. Generate positive-order stochastic-SEP targets with multiple quadrature
   nodes and nonzero variation in every shock dimension.
3. Cross-check common steady states, QZ solutions, trajectories, and
   likelihoods against MacroModelling.jl. Preserve failures from either
   implementation in the denominator.
4. Build a full nonlinear $(\phi_\pi,\phi_y)$ likelihood grid. Double the grid
   resolution until posterior moments and interval endpoints stabilize.
5. Compare ROM, residual-surrogate, surrogate-target HMC, and nonlinear-target
   corrected HMC against the grid. Report numerical uncertainty or pooled Monte
   Carlo error.
6. Stop if posterior fidelity fails. Do not scale the parameter dimension to
   compensate for a failed correctness reference.

### Experiment B: computational scaling

1. Prespecify a four-to-six parameter subset and justify it economically.
2. Validate the GPU-native `schur_gpu` QME path against the audited Schur
   implementation on the Gal\'i model before using it in any GPU speed claim.
3. Run four independently initialized, tuned chains. Report rank-normalized
   split R-hat, bulk ESS, tail ESS, numerical failures, ESS per gradient
   evaluation, and ESS per second.
4. Measure identical workloads and precision. Separate cold compilation, warm
   evaluation, transfers, host callbacks, training, and end-to-end runtime.
5. Repeat warm microbenchmarks 20--30 times and full independent processes at
   least three times. Use five end-to-end repetitions for the final table if
   feasible.

## Hard exit criteria

- All evidence is produced from a clean, content-addressed revision with a
  complete environment and command manifest.
- The stochastic nonlinear solver reports raw attempted, successful, fallback,
  and non-finite counts by prior-interior, posterior, and boundary-stress strata.
- The Julia comparison uses common case identifiers and reports discrepancies
  and joint failures, not only successful selected cases.
- Held-out tests are disjoint in parameter vectors, shock sequences, and
  trajectories. Per-output and tail errors, paired ROM improvement with a 95%
  interval, and learning curves are included.
- Likelihood comparisons include centered error, tail error, near-mode rank
  reversals, and normalized gradient error with a declared reference-gradient
  floor.
- The two-parameter nonlinear grid passes resolution-doubling checks, and the
  approximate/corrected posterior agrees within prespecified numerical or Monte
  Carlo uncertainty.
- The scaling run passes multi-chain diagnostics and reports effective samples
  per second.
- Timing claims use same-work baselines and uncertainty intervals. A GPU label
  is not used for a path whose dominant operation is a sequential CPU callback.
- Every table value is generated from archived raw artifacts; every failed hard
  gate remains visible.

## Decision rule

Proceed to journal submission only if Experiment A establishes nonlinear
posterior fidelity and Experiment B establishes a meaningful computational
advantage after full accounting. If correctness passes but speed does not,
submit as a validation/infrastructure contribution only if that narrower claim
is valuable for the chosen journal. If posterior fidelity fails, report the
failure and redesign the surrogate; do not weaken the reference or remove hard
cases post hoc.
