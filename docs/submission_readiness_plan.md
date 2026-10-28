# Submission-readiness plan for the nonlinear DSGE methods paper

## Frozen claim

The current conference-paper contribution is an auditable validation
architecture for nonlinear-DSGE surrogates. It combines a first-order
reduced-order model, a learned full-minus-reduced residual, and target-corrected
Hamiltonian proposals, then separates solution, trajectory, likelihood,
posterior, and timing claims into executable gates. The generic endpoint
Metropolis correction is prior art. The evidence supports complete-model core
parity, a reduced-horizon nonlinear tree, and one inversion-likelihood
contribution. It does not support a production structural posterior or an
end-to-end speed advantage.

The deterministic order-zero pilots are not publication evidence, and their
corrected target is still a surrogate profile inversion objective. A later
clean, order-one, three-node preflight accepts all eight path evaluations across
seven unique policy-rule points at a five-period terminal horizon. The reference
calibration is evaluated twice. This supports reduced stochastic target
construction only: first-step outcomes still move from horizons four to five,
and the hard design fails under both solvers at horizon six. No manuscript text
may call either result an exact nonlinear posterior or terminal-horizon
validation.

The independent core-parity subgate is complete on the full source. A clean
MacroModelling.jl revision and a clean SurrogateNN_DSGE revision use
byte-identical Gal\'i files and agree at all eight evaluations on the
non-stochastic steady state and first-order Schur solution. A second clean
comparison agrees on the complete five-period, order-one sparse nonlinear tree
with all three first-period shock channels active. This closes the
reduced-horizon nonlinear-tree subgate, not production-horizon trajectory
parity.

At the pinned clean MacroModelling.jl revision, the public likelihood interface
does not expose SEP controls or per-period nonlinear likelihood contributions.
A separately specified reference adapter now reconstructs one nonlinear
inversion-likelihood contribution directly from the upstream SEP solver. The
clean comparison passes its recovered-shock, finite-difference Jacobian,
residual, and likelihood gates. This closes one-transition algorithmic parity;
multi-period state propagation, parameter-surface likelihood parity, and the
production horizon remain open.

The original `SurrogateNN_Estimation.jl` paper treats those component checks as
prerequisites, not as estimator validation. Its decisive evidence is a matched,
known-truth pipeline: nonlinear reference data, held-out residual trajectories,
a common-shock inversion grid, and direct-reference versus surrogate HMC using
the same data, priors, transforms, objective terms, and chain seeds. This
project has not yet completed that package. GPU execution and profiling remain
computational-feasibility evidence until the matched pipeline passes.

## Critical path

Every new Gal\'i pilot is launched from a clean revision through
`benchmarks/run_nonlinear_gali_pilot.sh`. The launcher defaults to stochastic
SEP order one with three quadrature nodes, records the exact argv and independent
design/HMC seeds, hashes the model, payload, launcher, and benchmark source before
execution, and hashes declared outputs afterward in `run_manifest.json`. Setting
`ALLOW_DIRTY_MANIFEST=1` is permitted only for local debugging; an evidence or
publication run with that flag cannot satisfy the clean-revision gate.

### Experiment A: two-parameter correctness reference

1. Freeze the priors, synthetic data, observables, shock scaling, SEP
   configuration, training design, and case identifiers. Generate the data from
   a clean revision with `benchmarks/generate_gali_synthetic_data.py`; the
   artifact records the two random seeds, burn-in, initial condition, complete
   shock and measurement-error paths, resolved parameters, solver diagnostics,
   and content hashes. The `eps_*` innovations have unit variance because the
   model equations already apply `std_a`, `std_nu`, and `std_z`.
2. Generate positive-order stochastic-SEP targets with multiple quadrature
   nodes and nonzero variation in every shock dimension.
3. Keep an untouched parameter-group test partition. Disable residual-shrinkage
   calibration unless a distinct calibration partition is frozen in advance;
   never tune shrinkage on the block used for the reported test error.
4. Cross-check common steady states, QZ solutions, trajectories, and
   likelihoods against MacroModelling.jl. Preserve failures from either
   implementation in the denominator. The steady-state and first-order Schur
   checks now pass for 8/8 evaluations across seven unique policy-rule points.
   The complete reduced-horizon nonlinear tree also passes 8/8, and one direct
   nonlinear inversion-likelihood contribution passes its cross-language gate.
   Production-horizon trajectories, multi-period likelihood recursion, and the
   likelihood surface remain open.
5. Build a full nonlinear $(\phi_\pi,\phi_y)$ likelihood grid. Double the grid
   resolution until posterior moments and interval endpoints stabilize.
6. Compare direct-reference and residual-surrogate objectives against the grid,
   then run matched HMC with the same observations, priors, transforms, shock
   treatment, objective terms, initialization rules, and chain seeds. Report
   truth coverage, interval overlap, posterior-mean differences in combined
   MCSE units, convergence diagnostics, and every numerical failure.
7. Require the held-out residual, common-shock inversion-grid, and matched-HMC
   layers to pass sequentially. Preserve cases in which both objectives agree
   but miss a weakly identified truth as identification failures.
8. Stop if posterior fidelity fails. Do not scale the parameter dimension to
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
5. On a dedicated device, record continuous GPU utilization, memory, power,
   temperature, and clock telemetry. Report upper-tail utilization, peak memory,
   and integrated energy alongside the raw trace. Use the same sampling interval
   for every comparison arm and add unprofiled sensitivity timings to bound the
   monitoring overhead.
6. Repeat warm microbenchmarks 20--30 times and full independent processes at
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
- Residual-shrinkage factors are either disabled or fitted on a prespecified
  calibration partition that is disjoint from the untouched test partition.
- Likelihood comparisons include centered error, tail error, near-mode rank
  reversals, and normalized gradient error with a declared reference-gradient
  floor.
- The two-parameter nonlinear grid passes resolution-doubling checks, and the
  matched direct-reference and surrogate arms use identical data, priors,
  transforms, shock treatment, objective terms, and chain seeds.
- Direct and surrogate intervals overlap and both cover every identified truth;
  posterior-mean differences are no larger than 0.5 combined MCSE. All
  attempted evaluations and expected identification failures remain visible.
- The scaling run passes multi-chain diagnostics and reports effective samples
  per second.
- Timing claims use same-work baselines and uncertainty intervals. A GPU label
  is not used for a path whose dominant operation is a sequential CPU callback.
  Point-in-time `nvidia-smi` snapshots are environment metadata, not utilization
  or peak-memory measurements.
- Every table value is generated from archived raw artifacts; every failed hard
  gate remains visible.

## Decision rule

The conference working paper may be submitted as a validation-method result
because its narrow claims are backed by clean, archived, cross-language
evidence and its negative horizon result is reported explicitly. The title,
abstract, introduction, and conclusion must exclude a production-posterior or
speed claim.

Proceed to a journal submission that claims posterior fidelity or computational
advantage only if Experiment A establishes the former and Experiment B
establishes the latter after full accounting. If correctness passes but speed
does not, retain the validation/infrastructure framing. If posterior fidelity
fails, report the failure and redesign the surrogate; do not weaken the
reference or remove hard cases post hoc.
