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
- quadratic matrix equation doubling solver
- MacroModelling-style `@model` / `@parameters` source parsing for the first-order path
- order-independent parameter definitions in `@parameters` blocks
- calibration equations in `@parameters` blocks, including both `par | lhs = rhs` and `lhs = rhs | par` forms
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
- focused tests for residuals, symmetry, fallback behavior, JIT, autodiff, parser parity, and JAX device accessibility

Not implemented yet:

- perturbation orders above third
- MacroModelling programmatic `for`-loop parsing
- parameter bounds and other non-equation `@parameters` directives from the Julia macro layer
- occasionally binding constraint parsing from the Julia macro layer
- full SEP integration with symbolic model objects
- regime switching

Progress is tracked in [docs/porting_progress.md](docs/porting_progress.md).
