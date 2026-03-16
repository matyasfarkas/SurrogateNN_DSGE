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
- top-level symbolic/indexed `for`-loop blocks remain explicitly unported and raise `NotImplementedError`
- tests verify additive-loop parity, product-loop parity through Hessians and first-order solutions, multiline loop parity, and the explicit unsupported-block boundary

## Explicit gaps

- The Julia `:bartels_stewart`, `:bicgstab`, and `:gmres` Lyapunov variants are not ported yet.
- The Julia `:bartels_stewart`, `:bicgstab`, `:dqgmres`, and `:gmres` Sylvester variants are not ported yet.
- The Julia QME `:schur` variant is not ported yet; the Python port currently uses the doubling solver.
- The current dense Sylvester fallback is a direct Kronecker solve, not a Bartels-Stewart implementation.
- The current dense Lyapunov fallback is also a direct Kronecker solve.
- The current parsed front end only supports inline time-index `for` loops inside equations; programmatic symbolic/indexed `for`-loop model generation is not ported yet.
- The current parsed front end does not yet port parameter bounds and other non-equation `@parameters` directives from the Julia macro layer.
- The current parsed front end does not yet port occasionally binding constraint parsing (`max`/`min` OBC machinery) from the Julia macro layer.
- Parameter-derivative pullbacks and the Julia reverse-rule machinery around symbolic derivatives are not ported yet.
- The current SEP solver is callback-based and generic; it is not yet wired to the parsed model objects or the full Julia tree-layout machinery.
- Perturbation orders above third and the broader Julia higher-order moment/statistics machinery remain unported.
- No claim is made yet about full MacroModelling feature parity beyond the tested kernels, Kalman/state-space layer, parsed-model perturbation path through third order, and generic SEP core.

## Environment note

- A local git repository can be maintained here.
- A remote GitHub repository has not been created from this environment because `gh` is not installed/configured.
