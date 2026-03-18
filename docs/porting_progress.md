# Porting Progress

## Scope

Source repository:

- `/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl`

Target repository:

- `/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE`

## Porting policy

- Port one feature at a time.
- Keep the Julia source read-only.
- Do not claim feature parity before tests exist in Python.
- Prefer JAX-native implementations over thin SciPy wrappers.

## Completed features

### 1. Discrete Lyapunov solver

Julia reference:

- `src/algorithms/lyapunov.jl`

Python/JAX status:

- `solve_discrete_lyapunov_doubling` implemented
- `solve_discrete_lyapunov_direct` implemented
- `solve_discrete_lyapunov` wrapper implemented with controlled fallback
- tests cover scalar closed form, matrix residuals, symmetry, fallback, JIT, and autodiff

### 2. Discrete Sylvester solver

Julia reference:

- `src/algorithms/sylvester.jl`

Python/JAX status:

- `solve_discrete_sylvester_doubling` implemented
- `solve_discrete_sylvester_direct` implemented
- `solve_discrete_sylvester` wrapper implemented with initial-guess fast path and direct fallback
- tests cover closed form, residuals, initial-guess reuse, fallback behavior, JIT, and autodiff

### 3. Linear Gaussian state-space layer

Julia reference:

- `src/filter/kalman.jl`
- `test/test_kalman_filter.jl`

Python/JAX status:

- `build_linear_gaussian_state_space` implemented
- `simulate_linear_gaussian_state_space` implemented
- `kalman_filter`, `kalman_loglikelihood`, and `kalman_loglikelihood_per_period` implemented
- `kalman_smoother` implemented with Rauch-Tung-Striebel backward pass
- tests cover finiteness, deterministic replay, short samples, likelihood ordering, JIT, and autodiff

### 4. Quadratic matrix equation and first-order DSGE solver

Julia reference:

- `src/algorithms/quadratic_matrix_equation.jl`
- `src/perturbation.jl`
- `test/test_standalone_function.jl`

Python/JAX status:

- `solve_quadratic_matrix_equation_doubling` implemented
- `solve_quadratic_matrix_equation_schur` implemented with the same companion-pencil generalized Schur / ordered-QZ construction as the Julia `:schur` path
- `solve_quadratic_matrix_equation_schur_jax` implemented with an implicit reverse-mode pullback, so JAX can differentiate the Schur-selected solution even though the primal ordered-QZ solve currently runs through SciPy
- explicit Schur / ordered-QZ determinacy diagnostics are implemented via `analyze_quadratic_matrix_equation_schur`, `analyze_first_order_dsge_determinacy`, and `analyze_first_order_model_determinacy`, including stable-root counts and `unique_stable_solution` / `indeterminate` / `no_stable_solution` classification
- `DSGETimings` implemented for low-level timing metadata
- `solve_first_order_dsge_solution` implemented with `qme_algorithm="doubling"` and `qme_algorithm="schur"` options, and the default now matches Julia's `:schur` path
- `linear_state_space_from_first_order_solution` implemented to connect first-order solutions to the Kalman layer
- tests include the Julia `RBC_CME` Jacobian/timing fixture and verify the resulting solution matrix against upstream reference values, Schur-vs-doubling parity, JIT coverage for the Schur first-order path, reverse-mode autodiff through the Schur QME solution, and toy-model determinacy classification for unique, indeterminate, and no-stable regimes

### 5. Generic stochastic extended path core

Julia reference:

- `src/sep_solver.jl`
- `src/sep_simulation.jl`

Python/JAX status:

- `gauss_hermite_rule` implemented
- `solve_stochastic_extended_path` implemented for callback-based residuals with Gauss-Hermite branching
- tests cover quadrature normalization, zero-shock linear solutions, deterministic-vs-stochastic mean-path equivalence for zero-mean shocks, and a nonlinear expectational toy model

### 6. MacroModelling-style parser, symbolic derivatives, and high-level first-order solve

Julia reference:

- `src/macros.jl`
- `src/structures.jl`
- `test/models/RBC_CME.jl`
- `test/test_standalone_function.jl`

Python/JAX status:

- `parse_macro_model` implemented for `@model ... begin ... end` and `@parameters ... begin ... end` source blocks
- core MacroModelling timing syntax implemented: endogenous `[-k, 0, +k]`, steady-state `[ss]`, and shock tags `[x]`, `[x+k]`, `[x-k]`
- automatic auxiliary lead/lag expansion implemented for larger endogenous leads/lags and shifted shocks
- symbolic Jacobian, Hessian, and third-order derivative evaluation implemented with Julia-compatible compressed column ordering
- damped Newton non-stochastic steady-state solver implemented for parsed models
- high-level parsed-source first-order solve implemented and tested against the Julia `RBC_CME` fixture
- tests verify timing metadata, steady state, Jacobian, Hessian, third-order derivatives, first-order solution, and auxiliary-variable expansion

### 7. Second-order perturbation and stochastic steady state

Julia reference:

- `src/perturbation.jl`
- `src/MacroModelling.jl`
- `test/test_standalone_function.jl`

Python/JAX status:

- `create_second_order_auxiliary_matrices` implemented with Julia-compatible compression, uncompression, and volatility-routing matrices
- `solve_second_order_dsge_solution` implemented using the same compressed Sylvester formulation as the Julia code path
- `solve_second_order_stochastic_steady_state` implemented for standard and pruned second-order solutions
- `second_order_state_update` and `pruned_second_order_state_update` implemented for direct simulation/IRF use
- high-level parsed-source second-order solve implemented via `solve_second_order_model`
- tests verify auxiliary-matrix roundtrips, second-order solution parity on the Julia `RBC_CME` fixture, deterministic second-order and pruned-second-order responses, stochastic steady-state fixed-point behavior, and parsed-model parity versus the low-level fixture path

### 8. Third-order perturbation and pruned third-order simulation

Julia reference:

- `src/perturbation.jl`
- `src/MacroModelling.jl`
- `test/test_standalone_function.jl`

Python/JAX status:

- `create_third_order_auxiliary_matrices` implemented with Julia-compatible cubic compression matrices and tensor-permutation operators
- `solve_third_order_dsge_solution` implemented using the Julia third-order Sylvester formulation on top of the ported first-order and second-order kernels
- `solve_third_order_stochastic_steady_state` implemented for standard third-order solutions
- `third_order_state_update` and `pruned_third_order_state_update` implemented for direct simulation/IRF use
- high-level parsed-source third-order solve implemented via `solve_third_order_model`
- tests verify auxiliary-matrix roundtrips, third-order solution parity on the Julia `RBC_CME` fixture, deterministic third-order and pruned-third-order responses, third-order stochastic steady-state fixed-point behavior, and parsed-model parity versus the low-level fixture path

### 9. Parameter definitions, calibration equations, and calibrated parsed-model solves

Julia reference:

- `src/macros.jl`
- `src/get_functions.jl`
- `test/models/RBC_CME_calibration_equations.jl`
- `test/models/RBC_CME_calibration_equations_and_parameter_definitions.jl`

Python/JAX status:

- `@parameters` parsing now supports order-independent parameter definitions as a solved system rather than only sequential numeric assignments
- calibration equations now support both MacroModelling forms: `par | lhs = rhs` and `lhs = rhs | par`
- parsed models now expose `resolve_parameter_values` to recover consistent parameter vectors from either default parameter guesses or a supplied steady state
- `solve_steady_state` now solves the full non-stochastic steady state together with calibrated parameters when calibration equations are present
- high-level parsed first-order solves now propagate resolved calibrated parameters through Jacobian evaluation and the perturbation solution
- tests verify out-of-order parameter definitions, end-target calibration syntax, Julia-fixture parity for `RBC_CME` calibration equations plus parameter definitions, and JAX JIT/device accessibility on calibrated first-order state-space outputs

### 10. Inline time-index `for` loops in parsed `@model` equations

Julia reference:

- `src/macros.jl`
- `src/preprocess_model.jl`

Python/JAX status:

- parsed `@model` blocks now support inline time-index `for ... end` expressions inside equations
- additive loop expansion and `operator = :*` product expansion are both implemented
- multiline equations containing multiple inline time loops are normalized and expanded before symbolic differentiation
- that slice did not yet cover top-level symbolic/indexed `for`-loop blocks; later sections cover the broader top-level expansion work
- tests verify additive-loop parity, product-loop parity through Hessians and first-order solutions, multiline loop parity, and the explicit unsupported-block boundary

### 11. Parsed-model stochastic extended path wiring

Julia reference:

- `src/sep_solver.jl`
- `src/sep_simulation.jl`

Python/JAX status:

- `solve_stochastic_extended_path_residual_expectation` implemented to average conditional nonlinear residuals over future SEP branches
- parsed models now expose `evaluate_dynamic_residual` for arbitrary lag/current/lead states and current shocks
- high-level parsed SEP solves are implemented via `solve_stochastic_extended_path_model`
- parsed SEP solves resolve steady states and calibrated parameters first, then evaluate JAX-native symbolic residuals directly inside the SEP Newton solve
- deterministic shocks can be passed either as a matrix in model exogenous order or as a mapping from shock names to time series
- tests verify low-level residual-expectation equivalence, direct dynamic-residual evaluation, and parsed-model SEP parity against the same nonlinear equation written as a manual conditional residual callback

### 12. Explicit indexed identifiers and top-level symbolic `for` expansion

Julia reference:

- `src/MacroModelling.jl` (`replace_indices_inside_for_loop`, `replace_indices`, `parse_for_loops`)
- `test/models/Backus_Kehoe_Kydland_1992.jl`

Python/JAX status:

- parsed `@model` and `@parameters` blocks now accept explicit curly-brace indexed identifiers such as `y{H}[0]`, `u{F}[x]`, and `rho{H}{F}`
- brace-indexed parameter names are sanitized only for symbolic parsing, while the public `parameter_names` and timing metadata preserve the original Julia-style syntax
- top-level `for` blocks in `@model` now expand into multiple equations for explicit identifier lists like `[H, F]` and integer ranges
- inline symbolic sums such as `for co in [H, F] y{co}[0] end` now expand inside equations alongside the previously ported time-index loops
- tests verify top-level indexed-loop parity, inline indexed-sum parity, and nested brace-index parameter resolution

### 13. Indexed parameter-family broadcasting in `@parameters`

Julia reference:

- `src/macros.jl`
- `src/MacroModelling.jl` (`expand_indices`, `expand_calibration_equations`)
- `models/Backus_Kehoe_Kydland_1992.jl`

Python/JAX status:

- direct `@parameters` definitions now broadcast unindexed parameter targets such as `alpha = 0.3` across indexed families already referenced by the parsed model
- one-family calibration equations now expand generic indexed names consistently, so expressions like `y[ss] = target | beta` become indexed calibration conditions for `y{...}` and `beta{...}`
- generic steady-state references inside those one-family calibration equations are rewritten before symbolic parsing, preserving JAX-native steady-state and first-order solution paths
- tests verify direct-definition family broadcasting, calibration-equation family broadcasting, resolved parameter values, steady states, Jacobians, and first-order solutions against explicit fully indexed equivalents

### 14. Lazy symbolic compilation for parsed models

Julia reference:

- `src/macros.jl`
- `models/Backus_Kehoe_Kydland_1992.jl`

Python/JAX status:

- `parse_macro_model` no longer eagerly builds every symbolic matrix and `sympy.lambdify` artifact during parse
- parsed models now compile steady-state, calibration, residual, Jacobian, Hessian, and third-order derivative callables lazily on first use and cache them afterward
- repeated steady-state and derivative evaluations reuse the cached compiled functions rather than recompiling symbolic kernels
- targeted tests verify zero eager `lambdify` calls at parse time and cache reuse across steady-state, Jacobian, Hessian, third-order, and dynamic-residual evaluation paths
- the upstream `Backus_Kehoe_Kydland_1992.jl` model now parses successfully as a smoke check instead of stalling in eager compilation

### 15. `@parameters` default `guess = Dict(...)` support

Julia reference:

- `src/macros.jl`
- `test/runtests.jl` (`guess = Dict(:alpha => .2, :beta => .99)`)

Python/JAX status:

- parsed `@parameters` options now read `guess = Dict(...)` entries with Julia-style symbol or string keys
- default guesses are applied to steady-state variables when no explicit `initial_guess` is passed to `solve_steady_state`
- guesses for calibrated parameters now seed the joint steady-state/calibration solve, while explicit `initial_guess` and `parameter_values` arguments still override the defaults
- tests verify branch-sensitive steady-state behavior from the default guess and direct capture of calibrated-parameter guesses in the joint Newton initial condition

### 16. `@parameters` inequality bounds

Julia reference:

- `src/macros.jl` (bound parsing in the `@parameters` macro)
- `test/models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl`

Python/JAX status:

- parsed `@parameters` blocks now read one-sided and chained inequality bounds such as `x >= 0`, `10 >= R`, `0 < alpha < 1`, and `1 > y > -1`
- bounds are stored on parsed models and intersected when multiple bound lines target the same variable or parameter
- steady-state, parameter-only calibration, and joint steady-state/calibration Newton solves now project iterates into the parsed box constraints
- tests verify open/closed bound parsing, feasible-branch selection under a lower bound, and bound vectors passed into the joint steady-state/calibration solve

### 17. MacroModelling special-function aliases

Julia reference:

- `test/models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl`
- `src/MacroModelling.jl` symbolic registrations for normal-distribution helpers

Python/JAX status:

- the parser now recognizes MacroModelling aliases `normpdf`, `dnorm`, `pnorm`, `normlogpdf`, and `erfcinv` in addition to the previously ported `normcdf`, `norminv`, `norminvcdf`, and `qnorm`
- custom lambdify modules now provide a numeric implementation for `erfcinv`, so steady-state and derivative evaluation do not fail at runtime when those functions appear in parsed symbolic expressions
- tests verify parsed-model steady-state evaluation for the special-function aliases, and the upstream `RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl` fixture now parses successfully as a smoke check

### 18. Named source-level loop collections

Julia reference:

- `src/MacroModelling.jl` (`replace_indices_inside_for_loop`, `parse_for_loops`)

Python/JAX status:

- the parsed front end now reads simple source-level collection definitions outside `@model` and `@parameters`, such as `countries = [:H, :F]` and `lags = -2:0`
- top-level and inline `for` expansion in `@model` now resolve those named collections in addition to explicit identifier lists and explicit integer ranges
- symbol literals in source-level collections now support Julia-style `:H` syntax before substitution into brace-indexed identifiers
- tests verify top-level named-collection parity, named-range parity, and explicit failure for undefined collection names rather than silent inference

### 19. Parsed-model observable resolution and first-order state-space bridge

Julia reference:

- `src/MacroModelling.jl` (`get_and_check_observables`)
- first-order state-space usage around the Kalman layer

Python/JAX status:

- parsed models now resolve observable names like `(\"y\", \"c\")` into first-order solution row indices without exposing integer indexing in user code
- parsed models now build `LinearGaussianStateSpace` objects directly from a parsed first-order solution, preserving the existing low-level transition, shock, and observation construction
- custom measurement-error covariance matrices can now be injected at the parsed-model layer instead of only the low-level index-based helper
- tests verify observable-order preservation, explicit failure for unknown names, parity with the low-level state-space helper, and JAX JIT/device accessibility for the resulting state-space objects

### 20. Parsed-model Kalman loglikelihood entry points

Julia reference:

- `src/get_functions.jl` (`get_loglikelihood`, `get_loglikelihood_per_period`)
- `src/MacroModelling.jl` (`get_and_check_observables`)

Python/JAX status:

- parsed models now expose high-level Kalman loglikelihood helpers that accept named observable data either as a matrix plus explicit observable order or as a mapping from observable names to time series
- the parsed-model likelihood path demeans level data by the solved model steady state before passing deviations into the existing JAX Kalman layer, matching the Julia estimation flow
- first-order solution failure and parameter-bound violations now return configurable failure values instead of forcing every caller to handle solver internals manually
- tests verify parity with the low-level deviation-based Kalman path, mapping-input handling, per-period contributions summing to the total likelihood, and Julia-style failure fallbacks on bound violations

### 21. Optional NumPyro subset-parameter inference helpers

Julia reference:

- no direct Julia/NumPyro analogue; this is the Python-side estimation bridge on top of the ported parsed-model likelihood path

Python/JAX status:

- an optional inference module now provides subset-parameter vector assembly, a NumPyro model factory for the parsed-model Kalman likelihood, and concrete log-density evaluation via `numpyro.infer.util.log_density`
- the NumPyro bridge is explicit about the current boundary: it supports prior wiring and concrete likelihood evaluation, but it fails fast for compiled structural kernels once parameters become JAX tracers
- `pyproject.toml` now exposes an `inference` extra for `numpyro`, and the dev extra includes that dependency so the integration tests are reproducible
- tests verify subset parameter assembly, NumPyro log-density parity against manual prior-plus-likelihood calculations, deterministic sites for the assembled parameter vector and loglikelihood, and the explicit fast-fail behavior for compiled kernels

### 22. Fixed-steady-state compiled JAX first-order estimation path

Julia reference:

- this is a Python/JAX bridge around the existing first-order perturbation and Kalman stack rather than a direct one-to-one Julia API surface

Python/JAX status:

- the port now includes a traceable quadratic matrix equation doubling solver and a JAX-first-order solution path for compiled use
- parsed models can now evaluate a first-order Kalman loglikelihood entirely in JAX when the steady state is supplied explicitly, avoiding the current NumPy/SymPy steady-state and calibration bottleneck
- a dedicated NumPyro wrapper now supports compiled kernels like `NUTS` on that fixed-steady-state first-order path, while preserving the older concrete-only wrapper for the general parsed-model likelihood
- tests verify JIT parity for the new first-order solver, fixed-steady-state JAX likelihood parity against the existing parsed-model likelihood, NumPyro log-density parity, and an actual compiled `NUTS` run

### 23. Automatic JAX steady-state solves for non-calibrated first-order estimation

Julia reference:

- this advances the Python/JAX estimation bridge rather than porting a separate Julia API surface

Python/JAX status:

- parsed models now expose a JAX steady-state Newton solver for non-calibrated models, including bounded backtracking and JIT coverage
- the compiled first-order Kalman likelihood and NumPyro wrapper no longer require an explicit steady state for that subset; they can solve the steady state inside the JAX trace from an initial guess
- the calibration-equation boundary remains explicit: models with `@parameters` calibration equations still require an explicit steady state or the existing NumPy-based path
- tests verify steady-state parity against the NumPy solver, JIT compatibility, likelihood parity with automatic steady-state solves, and a compiled `NUTS` run on the auto-steady-state path

### 24. Automatic JAX calibration-equation solves for first-order compiled estimation

Julia reference:

- this extends the Python/JAX estimation bridge over the existing calibrated parsed-model path rather than introducing a separate Julia-facing API

Python/JAX status:

- parsed models now expose JAX parameter-resolution and joint steady-state/calibration solves for `@parameters` calibration equations
- the compiled first-order Kalman likelihood and NumPyro wrapper now cover calibrated parsed models as well, using the JAX steady-state/calibration solve inside the trace
- tests verify JAX parity for direct calibrated-parameter resolution, joint calibrated steady states on both a minimal target-calibration model and the existing `RBC_CME` fixture, calibrated auto-steady-state likelihood parity, and a compiled `NUTS` run on a calibrated model

### 25. Inversion filters and switching-likelihood bridge

Julia reference:

- `src/filter/inversion.jl`
- `src/regime_switching/likelihood.jl`
- `test/test_inversion_filter.jl`
- `test/test_sep_inversion_filter_likelihood.jl`
- `test/test_sw07_hlt_estimation_switching.jl`

Python/JAX status:

- a new inversion module now ports the first-order inversion filter with total and per-period likelihood helpers, square and rectangular shock Jacobian handling, and Julia-style warmup support
- parsed models now expose named-observable inversion likelihood entry points for both `first_order` and `stochastic_extended_path`, returning configurable failure values just like the existing Kalman helpers
- the SEP path now includes the missing inversion-filter layer on top of the parsed full-tree SEP solver, together with `reset_sep_inversion_last_diagnostics` / `get_sep_inversion_last_diagnostics` and Julia-style runtime overrides such as `sep_periods`, `sep_order`, `sep_nnodes`, `sep_accept_tol`, and the SEP inversion LM tolerances
- a switching module now ports the regime-switching likelihood combiner `compute_switching_loglikelihood` / `mix_loglikelihood`, and parsed models can mix ROM Kalman and FOM inversion per-period likelihoods directly when supplied with gate probabilities or a hard mask
- tests verify low-level vs parsed first-order inversion parity, JIT coverage for the first-order inversion kernel, failure fallbacks, SEP inversion diagnostics and determinism, runtime override acceptance, and parsed-model switching parity against manual ROM/FOM mixtures

### 26. Switching gate utilities, parser hardening, and multi-model JAX compile smoke

Julia reference:

- `src/regime_switching/gating.jl`
- `src/regime_switching/types.jl`
- `test/models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl`
- `test/models/Backus_Kehoe_Kydland_1992.jl`
- `models/RBC_Dynare.jl`
- `models/FS2000.jl`
- `models/RBC_baseline.jl`

Python/JAX status:

- the switching layer now ports the gate-stat and assignment utilities around the existing likelihood mixer: `compute_gate_stat_series`, `gate_share`, `calibrate_gate`, `calibrate_tau_y`, `calibrate_tau_eps`, `apply_gate_padding`, `assign_regimes`, `logistic`, `logit`, `calibrate_gate_bias`, and `gate_probabilities`
- parsed-model symbolic internals now sanitize all parameter names to ASCII-safe parse tokens, which fixes Python-keyword parameters like `del`, Unicode shock names like `ϵ[x]`, and combining-mark identifiers like `ḡ` without changing the public Julia-style names exposed by parsed models
- the JAX lambdify path now overrides `erf`, `erfc`, `erfinv`, and `erfcinv` with tracer-safe implementations, so the tested special-function models compile on the JAX first-order likelihood path instead of falling back to SciPy behavior that breaks tracing
- upstream compile smoke now covers first-order solve plus JAX-compiled Kalman likelihoods on `RBC_CME`, calibrated `RBC_CME`, lead-lag and special-function `RBC_CME` variants, `RBC_Dynare`, `FS2000`, `RBC_baseline`, and `Backus_Kehoe_Kydland_1992`
- tests verify gate-stat normalization and calibration behavior, hard vs soft gate probabilities, special-function steady-state evaluation with safe seeds, and successful parse/solve/JAX-compile smoke across the multi-model upstream fixture set

### 27. First-order switching diagnostics and supplied-shock linear gate stats

Julia reference:

- `src/regime_switching/gating.jl` (`compute_linear_gate_stats_from_shocks`)
- `src/regime_switching/diagnostics.jl`
- `src/regime_switching/likelihood.jl` (`evaluate_switching_vs_fom`)

Python/JAX status:

- the DSGE core now includes `first_order_state_update` and `rollout_first_order_solution`, so first-order solution matrices can be simulated directly without routing everything through the Kalman wrapper
- parsed models now expose `compute_linear_gate_stats_from_shocks_model`, which rolls a first-order solution forward under a supplied shock path, reconstructs the matching linear observable path, and evaluates `e_stat` / `f_stat` with either ordered sigma vectors or name-indexed sigma mappings
- switching diagnostics now cover contiguous gated runs, longest/first/last block selection, context-window extraction, gate-mask summary statistics, window overlap summaries, loglikelihood decomposition summaries, runtime summaries, and switching-vs-FOM comparison metrics
- tests verify the diagnostics against manual episode accounting and verify parsed-model linear gate stats on the upstream `RBC_CME` model against an explicit manual first-order rollout

### 28. First-order observed-shock / observed-variable helpers and filter-based linear gate stats

Julia reference:

- `src/regime_switching/gating.jl` (`estimate_observed_shocks_matrix`, `estimate_observed_variables_matrix`, `linear_filter_initial_state`, `linear_filter_full_state_initial`, `compute_linear_gate_stats_from_filter`)
- `src/get_functions.jl` (`get_estimated_shocks`, `get_estimated_variables`)
- `src/filter/kalman.jl`
- `src/filter/inversion.jl`
- `test/test_regime_switching_api.jl`

Python/JAX status:

- parsed models now expose Julia-style first-order helper entry points for observed shocks and observed variables, with exact public names: `estimate_observed_shocks_matrix`, `estimate_observed_variables_matrix`, `linear_filter_initial_state`, `linear_filter_full_state_initial`, and `compute_linear_gate_stats_from_filter`
- the first-order inversion path mirrors the Julia inversion filter recurrence directly, while the Kalman path uses the existing JAX Kalman filter / smoother and reconstructs the matching shock path before rolling the full first-order solution forward
- the shock coercion layer now accepts both Python-oriented `(periods, n_exo)` arrays and Julia-oriented `(n_exo, periods)` shock matrices, which is needed for parity across the parsed-model SEP and switching helpers
- the helper layer now short-circuits exact steady-state data to zero shocks and steady-state variables, which materially improves robustness on large upstream fixture models used only as compile smoke
- tests verify exact one-shock recovery for both inversion and Kalman helpers, manual parity for the filter-based gate-stat composition, shape/error guards, and multi-model smoke coverage on `RBC_CME`, `RBC_Dynare`, `FS2000`, `RBC_baseline`, and `Backus_Kehoe_Kydland_1992`

### 29. Callback-based switching likelihood utilities and named-parameter helpers

Julia reference:

- `src/regime_switching/gating.jl` (`extract_named_parameters`, `override_named_parameters`, `parameters_with_theta_mode`)
- `src/regime_switching/likelihood.jl` (`linear_model_loglik_per_period`, `conditional_loglik_per_period`, `split_observation_state`, `predict_from_full`, `predict_additive_residual`, `additive_residual_loglik_per_period`, `rollout_observations`, `advance_state`, `linear_reference_loglik_per_period`, `inversion_step`, `inversion_loglik_per_period`, `build_shocks_from_eps`)
- `src/regime_switching/diagnostics.jl` (`run_chunked_sampling`)
- `test/test_regime_switching_api.jl`

Python/JAX status:

- the port now includes the missing named-parameter helpers `extract_named_parameters`, `override_named_parameters`, `parameters_with_theta_mode`, and the indexed override path used by `linear_model_loglik_per_period`
- a new callback-oriented regime-switching helper module now ports the Julia toy-API layer for conditional likelihoods, additive residuals, observation rollout, state advancement, Levenberg-Marquardt inversion steps, inversion-filter likelihood construction, linear-reference comparison likelihoods, and epsilon-to-shock reconstruction
- `linear_model_loglik_per_period` now bridges these helpers back to parsed models, using the existing parsed-model Kalman and inversion per-period likelihood paths with explicit parameter overrides
- `run_chunked_sampling` is now ported as a generic orchestration helper for chunk-wise posterior or likelihood workflows
- for robustness, the inversion-step helper first attempts a JAX Jacobian and falls back to finite-difference Jacobians when the callback is not JAX-traceable
- tests mirror the upstream Julia toy cases for parameter overrides, conditional and additive-residual likelihoods, rollout/state helpers, inversion helpers, linear-reference dispatch, shock reconstruction, and chunked sampling

### 30. Schur/QZ selection through parsed-model first-order APIs

Julia reference:

- `src/get_functions.jl`
- `src/filter/kalman.jl`
- `src/filter/inversion.jl`
- `src/regime_switching/gating.jl`
- `src/regime_switching/likelihood.jl`

Python/JAX status:

- parsed-model first-order callers that build solutions internally now expose `qme_algorithm`, including state-space construction, Kalman likelihoods, inversion likelihoods, switching likelihoods, observed-shock / observed-variable helpers, filter-state extractors, and linear gate-stat helpers
- the default first-order branch selection on those public APIs now matches Julia and uses `schur`, while `doubling` remains available as an explicit opt-in
- the concrete NumPyro bridge now forwards the same `qme_algorithm` choice, so both the concrete log-density path and the compiled JAX likelihood bridge can use the Schur/QZ branch consistently
- upstream compile smoke now also covers explicit Schur/QZ first-order solves plus JAX-compiled Kalman likelihoods on `RBC_CME`, `RBC_Dynare`, `FS2000`, and `Backus_Kehoe_Kydland_1992`
- tests verify high-level Schur parity for parsed-model state-space construction, Kalman likelihoods, first-order filter helpers, concrete NumPyro log-density evaluation, the new multi-model Schur compile smoke, and explicit default-equals-Schur behavior for low-level first-order solves plus compiled Kalman likelihoods

### 31. Posterior-chain convenience helpers for switching workflows

Julia reference:

- `scripts/hlt_sep_surrogate_synthetic_estimation.jl` (`theta_draws`, `epsilon_means_from_chain`, `chunk_stats`)

Python/JAX status:

- the switching helper module now ports `theta_draws`, `epsilon_means_from_chain`, and `chunk_stats`
- the chain coercion layer accepts NumPyro `MCMC` objects, raw sample mappings, and checkpoint-like payloads containing `chain` or `samples`
- `theta_draws` now returns Julia-style dense draw matrices in user-specified parameter order, with explicit guards for missing names and non-scalar sample sites
- `epsilon_means_from_chain` now parses Greek epsilon site names such as `ε[1,3]` and `ϵ[2,5]`, averages them across posterior draws, and returns the same shock-by-time mean matrix shape expected by the existing `build_shocks_from_eps` helper
- `chunk_stats` now extracts acceptance, divergence, and step-size diagnostics from finished NumPyro runs when those diagnostics are available, while also honoring lightweight synthetic/checkpoint payloads for downstream summaries
- tests cover raw sample-mapping parity, epsilon-site parsing and mismatch warnings, and real NumPyro `MCMC` extraction for both draw matrices and chunk statistics

### 32. Sparse fishbone SEP tree

Julia reference:

- `src/sep_solver.jl` (`SEPLayout`, `parent_group_sparse`, `child_groups_sparse`, `branch_info_sparse`, `build_sparse_shock_nodes`)

Python/JAX status:

- the generic SEP core now implements a real sparse fishbone tree instead of treating `sep_sparse_tree` as an API-compatibility no-op
- `SEPConfig` now includes `sparse_tree`, and the low-level solver now switches between the full tensor-product Gauss-Hermite tree and a monomial sparse-tree rule with Julia-style fishbone branching semantics
- sparse-tree group counts, branch identities, parent/child navigation, probabilities, and one-period node shocks now follow the Julia convention where only the trunk branches and new side branches appear from period 2 onward
- the parsed-model SEP solve path uses the same sparse-tree core automatically when `SEPConfig(sparse_tree=True)` is supplied
- the SEP inversion filter now honors sparse-tree configuration both through `SEPConfig(sparse_tree=True)` and the Julia-style `sep_sparse_tree` runtime override, instead of always forcing the full tree
- tests verify the sparse fishbone group-count layout directly, confirm that sparse and full trees deliver the same linear mean path under zero-mean shocks, and verify that the parsed SEP inversion bridge now preserves `config.sparse_tree`

### 33. SEP robustness hardening

Julia reference:

- `src/sep_solver.jl` (`line_search`, `lm_lambda`, `lm_lambda_scale`, `lm_lambda_min`, `lm_lambda_max`, warm-start handling)

Python/JAX status:

- the SEP Newton loop now uses adaptive Levenberg-Marquardt damping on the normal equations instead of a fixed diagonal regularization, which materially improves robustness on harder nonlinear solves while staying close to the Julia solver structure
- `SEPConfig` now validates core runtime invariants up front, including positive tolerances and iteration counts, valid line-search ranges, valid LM bounds, and the sparse-tree requirement that `nnodes` be odd so the trunk can use a true zero Gauss-Hermite node
- low-level SEP calls now validate `initial_guess` size explicitly instead of silently reshaping mismatched warm starts
- exact warm starts from prior `stacked_states` now short-circuit cleanly with zero additional iterations when the supplied solution already satisfies the nonlinear system
- tests cover warm-start exact reuse, invalid warm-start shape errors, sparse-tree `nnodes` validation, bad line-search configuration rejection, plus regression checks that parsed-model SEP and SEP inversion still pass on top of the hardened core

### 34. HMC expectation backend for SEP

Julia reference:

- `src/hmc_sep.jl`
- `src/sep_solver.jl` (`sep_expectation_method = :hmc`)

Python/JAX status:

- `SEPConfig` now exposes `expectation_method="gauss_hermite" | "hmc"` together with HMC controls for sample count, warmup, leapfrog steps, step size, tempering, temperatures, swap interval, and seed
- the residual-expectation SEP path now supports an HMC expectation backend that replaces Gauss-Hermite branching with adaptive shock sampling and uses a finite-difference Newton Jacobian, which is closer to the Julia HMC SEP strategy than tracing through the sampler
- optional parallel tempering is implemented for the HMC backend and can be enabled with `hmc_use_tempering=True`
- parsed-model SEP solves now inherit the same HMC backend automatically when `solve_stochastic_extended_path_model` is called with `SEPConfig(expectation_method="hmc", ...)`
- the older `solve_stochastic_extended_path` API based on `residual_fn` plus `expectation_fn` now also supports the HMC backend by averaging residuals over sampled future shocks without branching the state tree, matching the same no-tree HMC strategy used by the residual-expectation path
- tests cover deterministic fixed-seed HMC behavior, HMC tempering smoke, legacy callback-API HMC regression, and parsed-model HMC SEP smoke

### 35. Basic OBC syntax coverage

Julia reference:

- `src/macros.jl`
- `src/MacroModelling.jl` (`check_for_minmax`, `replace_min_max`)

Python/JAX status:

- parsed-model source already supported simple `max` / `min` expressions through the SymPy front end; this slice formalizes that support with explicit tests and parsed-model metadata
- `MacroModel` now exposes `has_obc`, which is set when dynamic equations contain `sp.Max` or `sp.Min`
- tests now cover parsed-model `max` and `min` residual evaluation, simple OBC steady states, inactive-branch first-order solves away from the kink, and a parsed-model SEP path where a lower bound becomes active under a large negative deterministic shock
- this does not claim full Julia OBC parity; it only narrows the boundary from “syntax unsupported” to “basic syntax works, while OBC-specific enforcement around kinks is still missing”

### 36. Branch-frozen OBC linearization at the steady state

Julia reference:

- `src/MacroModelling.jl` (`replace_min_max`, steady-state min/max elimination)

Python/JAX status:

- derivative extraction for parsed OBC models now freezes `sp.Max` / `sp.Min` branches at the evaluation point before differentiating, instead of accepting SymPy's generic kink derivative
- ties at the steady state now prefer the branch with fewer dynamic steady-state symbols, which correctly pins standard bound formulations like `max(r_star, zlb)` and `min(q_star, q_cap)` to the constant constraint branch when they bind
- non-JAX steady-state and joint steady-state/calibration Newton solves now use the same branch-frozen Jacobians for OBC models, so first-order solution prep does not linearize with the previous spurious `0.5` derivatives at binding points
- first-, second-, and third-order parsed derivative extraction now inherit the same branch-frozen OBC handling because `calculate_jacobian`, `calculate_hessian`, and `calculate_third_order_derivatives` resolve min/max branches before symbolic differentiation when `model.has_obc`
- tests cover binding `max` and `min` steady states where the constrained variable stays fixed in the linearization, including regression coverage that the first equation Jacobian row is `[1, 0, 0, 0]` instead of a half-weighted kink derivative

### 37. SEP Jacobian selection and OBC Jacobian safeguards

Julia reference:

- `src/sep_solver.jl`
- `src/MacroModelling.jl` (OBC residual handling around `min` / `max`)

Python/JAX status:

- `SEPConfig` now exposes `jacobian_method="auto" | "autodiff" | "finite_difference" | "subgradient"` so SEP Newton steps can choose between traced Jacobians, explicit finite differences, and caller-supplied generalized Jacobians
- the low-level SEP solver now reports the effective choice in `SEPSolution.jacobian_method`, which makes runtime behavior inspectable in tests and inversion wrappers
- `expectation_method="hmc"` now rejects `jacobian_method="autodiff"` and `"subgradient"` explicitly and keeps the existing finite-difference requirement for sampled expectations
- parsed-model SEP solves now switch `jacobian_method="auto"` to a branch-frozen subgradient Jacobian when `model.has_obc` and the expectation path is Gauss-Hermite, while keeping the finite-difference fallback on HMC paths
- the parsed-model OBC subgradient Jacobian is assembled from branch-frozen dynamic Jacobians at each SEP node, which is materially closer to the Julia subdifferential Newton idea than the earlier pure finite-difference fallback
- smooth-model tests verify that finite-difference and autodiff Jacobians agree on the same SEP path, and parsed OBC SEP tests now assert that the automatic subgradient path is engaged and matches the finite-difference solve on the same bound model

### 38. Parsed-model OBC violation diagnostics

Julia reference:

- `src/MacroModelling.jl` (`transform_obc`, `set_up_obc_violation_function!`, `write_obc_violation_equations`)
- `src/get_functions.jl`

Python/JAX status:

- parsed models now expose `evaluate_obc_violations` for single-period OBC complementarity diagnostics on `max` / `min` equations, using transformed branch-gap expressions rather than only checking the raw dynamic residual
- `evaluate_obc_violations_along_path` is implemented for full state paths plus named deterministic shocks, so parsed-model OBC paths can be checked period by period after SEP or other runtime solves
- `compute_first_order_obc_violation_path` is implemented for first-order rollouts, which makes it possible to quantify where the linearized solution breaches an occasionally binding constraint before engaging a nonlinear/OBC solver
- the current transformation supports one binary `min` or `max` call per equation, which is enough for the tested simple bound formulations but does not yet claim the full Julia OBC transformation surface
- tests verify inactive-branch steady-state violations are nonpositive, deterministic SEP paths satisfy the transformed OBC diagnostics along the whole path, and the first-order linear rollout correctly flags a large-shock lower-bound breach

### 39. Parsed-model `get_irf` and simulation helpers

Julia reference:

- `src/get_functions.jl` (`get_irf`, `simulate`)
- `src/common_docstrings.jl` (`ignore_obc`, `initial_state`, shock handling)
- `src/macros.jl` (runtime note that first-order OBC enforcement is active unless `ignore_obc = true`)

Python/JAX status:

- parsed models now expose `get_irf` and `simulate`, with top-level wrappers `get_irf(...)` and `simulate_model(...)`
- the new runtime helpers support named variable selection, named shock selection such as `"all"` or a specific exogenous name, explicit deterministic shock histories as matrices or mappings, `levels` vs deviation output, `initial_state`, and the existing first-order `qme_algorithm` selection
- first-order runtime requests on OBC models now honor `ignore_obc`; when `ignore_obc = false`, the helper routes through a deterministic `SEPConfig(branching_order = 0)` solve instead of silently returning the unconstrained linear path
- this narrows the user-facing MacroModelling gap materially, but it is not yet exact Julia parity because the current OBC runtime route uses deterministic SEP rather than Julia's dedicated first-order artificial-shock enforcement solver
- tests verify a closed-form linear AR(1) IRF, simulation-vs-rollout parity on the same linear model, OBC routing behavior for a simple bound model, and runtime smoke across upstream `RBC_Dynare`, `FS2000`, and `RBC_baseline`

## Explicit gaps

- The Julia `:bartels_stewart`, `:bicgstab`, and `:gmres` Lyapunov variants are not ported yet.
- The Julia `:bartels_stewart`, `:bicgstab`, `:dqgmres`, and `:gmres` Sylvester variants are not ported yet.
- The Julia QME `:schur` variant is now ported, but the JAX-facing primal solve is not fully GPU-native yet; until JAX exposes generalized `qz` / `ordqz`, the ordered-QZ step runs through SciPy on the host and only the reverse-mode derivative is native JAX.
- The Python port now defaults public first-order workflows to `schur` for Julia parity, which means the default compiled JAX likelihood path also uses the host-callback ordered-QZ branch unless `qme_algorithm="doubling"` is requested explicitly.
- The current dense Sylvester fallback is a direct Kronecker solve, not a Bartels-Stewart implementation.
- The current dense Lyapunov fallback is also a direct Kronecker solve.
- Ambiguous calibration equations that mix more than one indexed family still raise instead of inferring a broadcast pattern.
- The current parsed front end does not yet port the remaining non-equation `@parameters` directives from the Julia macro layer beyond `guess` and bounds.
- The current parsed front end now supports basic `max` / `min` OBC syntax, branch-frozen linearization around the active steady-state branch, and parsed-model OBC violation diagnostics, but the Julia-specific enforcement layer around kinks, including explicit OBC shock-sequence optimization, subdifferential Newton options, and the broader OBC runtime surface, is still unported.
- Parsed-model `get_irf` and `simulate_model` are now available, but the full Julia runtime surface is still broader, especially `simulate(..., shocks = :simulate)` random-shock semantics, the richer shock/variable selection API, and the exact first-order OBC artificial-shock enforcement path.
- Parameter-derivative pullbacks and the Julia reverse-rule machinery around symbolic derivatives are not ported yet.
- The parsed-model structural likelihood is still not pure-JAX end to end beyond the first-order path; compiled NumPyro kernels currently cover the first-order path with explicit steady states or automatic JAX steady-state/calibration solves only.
- The parsed SEP path now covers the full-tree Gauss-Hermite solver, the sparse fishbone tree, the HMC backend across both low-level callback APIs plus parsed-model solves, and branch-frozen subgradient Jacobians for parsed OBC models on the Gauss-Hermite path, but the remaining sparse-tree-specific Jacobian/runtime optimizations and broader OBC-specific subdifferential SEP machinery are still unported.
- Regime-switching likelihood mixing, gate-stat computation, gate calibration, probability mapping, automatic hard-regime assignment, and the first-order observed-shock / observed-variable helper surface are now ported, but the broader switching-estimation harness is not ported yet.
- Perturbation orders above third and the broader Julia higher-order moment/statistics machinery remain unported.
- No claim is made yet about full MacroModelling feature parity beyond the tested kernels, Kalman/state-space layer, parsed-model perturbation path through third order, parsed inversion filters, switching-likelihood mixer, and the parsed SEP path with both full-tree and sparse fishbone branching.

## Environment note

- A local git repository can be maintained here.
- A remote GitHub repository has not been created from this environment because `gh` is not installed/configured.
