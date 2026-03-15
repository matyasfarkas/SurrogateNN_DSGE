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

## Explicit gaps

- The Julia `:bartels_stewart`, `:bicgstab`, and `:gmres` Lyapunov variants are not ported yet.
- The Julia `:bartels_stewart`, `:bicgstab`, `:dqgmres`, and `:gmres` Sylvester variants are not ported yet.
- The Julia QME `:schur` variant is not ported yet; the Python port currently uses the doubling solver.
- The current dense Sylvester fallback is a direct Kronecker solve, not a Bartels-Stewart implementation.
- The current dense Lyapunov fallback is also a direct Kronecker solve.
- The current parsed front end does not yet port calibration equations in `@parameters` blocks.
- The current parsed front end does not yet port programmatic `for`-loop model generation.
- The current parsed front end does not yet port occasionally binding constraint parsing (`max`/`min` OBC machinery) from the Julia macro layer.
- Parameter-derivative pullbacks and the Julia reverse-rule machinery around symbolic derivatives are not ported yet.
- The current SEP solver is callback-based and generic; it is not yet wired to the parsed model objects or the full Julia tree-layout machinery.
- Higher-order perturbation solution operators are not ported yet, even though symbolic Hessian and third-order derivative evaluation now exist.
- No claim is made yet about full MacroModelling feature parity beyond the tested kernels, Kalman/state-space layer, parsed-model first-order path, and generic SEP core.

## Environment note

- A local git repository can be maintained here.
- A remote GitHub repository has not been created from this environment because `gh` is not installed/configured.
