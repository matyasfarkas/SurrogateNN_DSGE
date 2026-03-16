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
- parsed-model observable-name resolution and first-order state-space construction for Kalman estimation
- quadratic matrix equation doubling solver
- MacroModelling-style `@model` / `@parameters` source parsing for the first-order path
- MacroModelling-style inline time-index `for` loops inside `@model` equations, including additive and `operator = :*` forms
- explicit curly-brace indexed identifiers such as `y{H}[0]` and `rho{H}{F}` across parsed model and parameter blocks
- top-level `for`-block expansion in `@model` for explicit identifier lists like `[H, F]`, named source-level collections like `countries = [:H, :F]`, and integer ranges
- order-independent parameter definitions in `@parameters` blocks
- calibration equations in `@parameters` blocks, including both `par | lhs = rhs` and `lhs = rhs | par` forms
- indexed parameter-family broadcasting in `@parameters`, including `alpha = 0.3` style direct definitions and one-family calibration equations like `y[ss] = target | beta`
- `@parameters ... guess = Dict(...)` default steady-state guesses for variables and calibrated parameters
- inequality bounds in `@parameters` such as `0 < alpha < 1`, `x >= 0`, and `10 >= R`, with bounded Newton projection for steady-state and calibration solves
- MacroModelling special-function aliases including `normpdf`, `dnorm`, `pnorm`, `normlogpdf`, and `erfcinv`
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
- parsed-model stochastic extended path solve path with JAX dynamic residual evaluation and residual-expectation averaging over future branches
- focused tests for residuals, symmetry, fallback behavior, JIT, autodiff, parser parity, and JAX device accessibility

Not implemented yet:

- perturbation orders above third
- ambiguous multi-family calibration-equation broadcasting remains guarded rather than inferred
- the remaining non-equation `@parameters` directives from the Julia macro layer beyond `guess` and bounds
- occasionally binding constraint parsing from the Julia macro layer
- Julia sparse-tree / HMC SEP variants and OBC-specific SEP machinery
- regime switching

Progress is tracked in [docs/porting_progress.md](docs/porting_progress.md).
