# SurrogateNN_DSGE

This repository is a gradual Python/JAX port of the Julia code in `MacroModelling` as used inside `/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl`.

The porting rule for this repository is strict:

- work feature by feature
- keep the Julia source read-only
- only expose behavior that has been implemented and tested
- prefer JAX-native kernels that can later feed NumPyro and GPU execution

## Current status

Implemented:

- discrete Lyapunov solver for `A X A^T + C = X`
- JAX-native doubling kernel
- JAX-native dense direct solver fallback
- discrete Sylvester solver for `A X B + C = X`
- linear Gaussian state-space simulation, Kalman likelihood, filtering, and RTS smoothing
- first-order inversion-filter likelihoods with per-period contributions and warmup support
- parsed-model observable-name resolution and first-order state-space construction for Kalman estimation
- parsed-model Kalman loglikelihood helpers for named observable data in levels, including per-period likelihood contributions and failure fallbacks
- parsed-model inversion loglikelihood helpers for both first-order and stochastic extended path likelihoods, including SEP inversion diagnostics and Julia-style runtime override keywords
- regime-switching likelihood mixing with supplied hard masks or gate probabilities, plus a parsed-model bridge that mixes ROM Kalman and FOM inversion per-period likelihoods
- gate-stat computation, threshold calibration, probability mapping, padding, and automatic hard-regime assignment utilities for the switching layer
- first-order rollout helpers and parsed-model linear gate-stat computation from supplied shock paths, including named observable/shock sigma inputs
- JAX-native gate-stat kernels and a compiled parsed-model first-order supplied-shock gate-stat helper, so linear gate diagnostics can now be built under `jax.jit` from solved first-order paths as well
- compiled first-order filter helpers for observed-shock reconstruction, filtered/smoothed state extraction, filter-derived gate statistics, and JAX-native gate-probability construction, so the switching-order first-order filter workflow can now stay inside JAX as well
- Julia-style first-order observed-shock / observed-variable estimation helpers, linear filter state extractors, and filter-based linear gate-stat workflows for both Kalman and inversion filters
- switching diagnostics and comparison helpers for gated episodes, loglikelihood decompositions, and runtime summaries
- callback-based switching likelihood helpers including named-parameter selection/override, conditional and additive-residual loglikelihood utilities, generic inversion-step helpers, linear-reference likelihood comparison helpers, Julia-style shock reconstruction from epsilon means, and chunked-sampling orchestration
- posterior-chain convenience helpers for switching workflows, including `theta_draws`, `epsilon_means_from_chain`, and `chunk_stats`, with support for NumPyro `MCMC` objects plus raw/checkpoint-style sample mappings
- optional NumPyro inference helpers for subset priors, parameter-vector assembly, and concrete log-density evaluation on top of the parsed-model Kalman likelihood
- JAX first-order structural likelihood and NumPyro wrappers that can run compiled kernels like `NUTS` on the parsed-model first-order path with either an explicit steady state or automatic JAX steady-state and calibration-equation solves
- JAX first-order switching likelihood and NumPyro wrappers for fixed gate probabilities or hard masks, so ROM Kalman and FOM inversion can now be mixed on the compiled first-order path as well
- quadratic matrix equation doubling solver plus a generalized Schur / ordered-QZ solver for the Julia `:schur` path
- explicit Schur / ordered-QZ determinacy diagnostics for first-order models, including stable-root counts, unique/indeterminate/no-stable classification, and parsed-model wrappers to inspect the Schur branch directly
- parsed-model state-space, likelihood, filtering, gate-stat, and concrete/compiled NumPyro helpers now expose `qme_algorithm` so first-order workflows can explicitly choose between the doubling and Schur/QZ solution branches
- MacroModelling-style `@model` / `@parameters` source parsing for the first-order path
- basic `max` / `min` OBC syntax parsing with parsed-model `has_obc` detection, steady-state support, inactive-branch first-order solves, and SEP path solves on simple bound models
- branch-frozen OBC derivative evaluation around active steady-state branches, so binding `max` / `min` constraints no longer linearize with spurious `0.5` derivatives at the kink
- SEP Jacobian selection via `jacobian_method=\"auto\" | \"autodiff\" | \"finite_difference\" | \"subgradient\"`, with parsed OBC SEP solves automatically switching `auto` to a branch-frozen subgradient Jacobian on the Gauss-Hermite path
- parsed-model OBC violation diagnostics, including single-period violation evaluation, full-path violation checks, and first-order rollout diagnostics that flag when the linearized path breaches a `max` / `min` bound
- parsed-model `get_irf` and `simulate_model` runtime helpers for first-order and deterministic SEP paths, including named shock selection, explicit shock histories, variable selection, and OBC-aware `ignore_obc` routing
- MacroModelling-style inline time-index `for` loops inside `@model` equations, including additive and `operator = :*` forms
- explicit curly-brace indexed identifiers such as `y{H}[0]` and `rho{H}{F}` across parsed model and parameter blocks
- top-level `for`-block expansion in `@model` for explicit identifier lists like `[H, F]`, named source-level collections like `countries = [:H, :F]`, and integer ranges
- order-independent parameter definitions in `@parameters` blocks
- calibration equations in `@parameters` blocks, including both `par | lhs = rhs` and `lhs = rhs | par` forms
- indexed parameter-family broadcasting in `@parameters`, including `alpha = 0.3` style direct definitions and one-family calibration equations like `y[ss] = target | beta`
- `@parameters ... guess = Dict(...)` default steady-state guesses for variables and calibrated parameters
- inequality bounds in `@parameters` such as `0 < alpha < 1`, `x >= 0`, and `10 >= R`, with bounded Newton projection for steady-state and calibration solves
- MacroModelling special-function aliases including `normpdf`, `dnorm`, `pnorm`, `normlogpdf`, `erfinv`, and `erfcinv`
- lazy symbolic matrix construction and cached `sympy.lambdify` compilation for parsed models, so large MacroModelling-style sources parse without eagerly compiling every derivative object
- symbolic Jacobian, Hessian, and third-order derivative evaluation with Julia-compatible compressed ordering
- damped Newton non-stochastic steady-state solver
- calibrated-parameter resolution from either the joint steady-state solve or a supplied steady state
- first-order DSGE perturbation solver with Julia `RBC_CME` fixture coverage
- second-order DSGE perturbation solver with Julia-compatible compression matrices and Sylvester solve
- second-order stochastic steady-state solver for both standard and pruned second-order updates
- parsed-model second-order solve path from MacroModelling-style source through symbolic Hessians to the perturbation solution
- third-order DSGE perturbation solver with Julia-compatible tensor compression/permutation machinery and Sylvester solve
- third-order stochastic steady-state solver for standard third-order solutions plus pruned third-order state updates
- parsed-model third-order solve path from MacroModelling-style source through symbolic third derivatives to the perturbation solution
- generic callback-based stochastic extended path solver with Gauss-Hermite branching
- sparse fishbone stochastic extended path branching for callback-based and parsed-model SEP solves, including real `sep_sparse_tree` runtime support in the inversion bridge
- adaptive Levenberg-Marquardt damping, explicit SEP config validation, and checked warm-start support for more robust nonlinear SEP solves
- HMC expectation backend for both SEP callback APIs, including parsed-model SEP solves and optional parallel tempering
- parsed-model stochastic extended path solve path with JAX dynamic residual evaluation and residual-expectation averaging over future branches
- focused tests for residuals, symmetry, fallback behavior, JIT, autodiff, parser parity, inversion filtering, switching likelihoods, gate calibration, and multi-model JAX compile smoke across upstream model files

Not implemented yet:

- perturbation orders above third
- ambiguous multi-family calibration-equation broadcasting remains guarded rather than inferred
- the remaining non-equation `@parameters` directives from the Julia macro layer beyond `guess` and bounds
- full Julia-style occasionally binding constraint enforcement around kinks, including horizon-level OBC shock-sequence optimization, full kink-aware runtime switching, and the broader OBC runtime surface beyond the new parsed-model violation diagnostics
- full Julia `simulate(..., shocks = :simulate)` random-shock semantics and the exact first-order OBC shock-enforcement solver are still unported; the current runtime helpers enforce OBC models by routing first-order requests through deterministic SEP when `ignore_obc = false`
- the remaining sparse-tree-specific Jacobian/runtime optimizations and the broader OBC-specific SEP machinery beyond the new subgradient / finite-difference Jacobian safeguards
- fully GPU-native generalized QZ / ordered-QZ primitives; the current JAX-facing `schur` QME path uses a SciPy host callback in the primal solve because JAX does not yet expose generalized `qz` / `ordqz`
- the public first-order default now matches Julia and uses `schur`; request `qme_algorithm=\"doubling\"` explicitly if you want to stay on the fully JAX-native doubling path
- the new determinacy diagnostics are currently tied to the Schur/QZ path; the doubling path still solves the QME but does not provide a comparable stable-root decomposition report
- fully JAX-traceable parsed-model structural likelihoods beyond the first-order path, including higher-order estimation edges for compiled NumPyro kernels like `NUTS` and `HMC`
- the older high-level parsed-model switching/filter bridge in `model.py` still exists for concrete NumPy-oriented use, but the first-order switching-order filter ingredients now have compiled JAX counterparts; the remaining gaps are mainly higher-order/nonlinear switching surfaces rather than the first-order filter gate path
- the broader regime-switching estimation harness beyond the currently ported likelihood mixer and gate/regime utilities

Progress is tracked in [docs/porting_progress.md](docs/porting_progress.md).
