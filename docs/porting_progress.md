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
- `DSGETimings` implemented for low-level timing metadata
- `solve_first_order_dsge_solution` implemented
- `linear_state_space_from_first_order_solution` implemented to connect first-order solutions to the Kalman layer
- tests include the Julia `RBC_CME` Jacobian/timing fixture and verify the resulting solution matrix against upstream reference values

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

## Explicit gaps

- The Julia `:bartels_stewart`, `:bicgstab`, and `:gmres` Lyapunov variants are not ported yet.
- The Julia `:bartels_stewart`, `:bicgstab`, `:dqgmres`, and `:gmres` Sylvester variants are not ported yet.
- The Julia QME `:schur` variant is not ported yet; the Python port currently uses the doubling solver.
- The current dense Sylvester fallback is a direct Kronecker solve, not a Bartels-Stewart implementation.
- The current dense Lyapunov fallback is also a direct Kronecker solve.
- Ambiguous calibration equations that mix more than one indexed family still raise instead of inferring a broadcast pattern.
- The current parsed front end does not yet port the remaining non-equation `@parameters` directives from the Julia macro layer beyond `guess` and bounds.
- The current parsed front end does not yet port occasionally binding constraint parsing (`max`/`min` OBC machinery) from the Julia macro layer.
- Parameter-derivative pullbacks and the Julia reverse-rule machinery around symbolic derivatives are not ported yet.
- The parsed-model structural likelihood is still not pure-JAX end to end once automatic steady-state or calibration solves are required; compiled NumPyro kernels are currently available only on the fixed-steady-state first-order path.
- The parsed SEP path currently covers the full-tree Gauss-Hermite residual-expectation solver only; Julia sparse-tree/fishbone layouts, HMC expectations, and OBC-specific subdifferential SEP machinery are not ported yet.
- Perturbation orders above third and the broader Julia higher-order moment/statistics machinery remain unported.
- No claim is made yet about full MacroModelling feature parity beyond the tested kernels, Kalman/state-space layer, parsed-model perturbation path through third order, and the parsed full-tree SEP path.

## Environment note

- A local git repository can be maintained here.
- A remote GitHub repository has not been created from this environment because `gh` is not installed/configured.
