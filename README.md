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
- first-order DSGE perturbation solver with Julia `RBC_CME` fixture coverage
- generic callback-based stochastic extended path solver with Gauss-Hermite branching
- focused tests for residuals, symmetry, fallback behavior, JIT, and autodiff

Not implemented yet:

- model parser/macros
- higher-order perturbation solvers
- MacroModelling-style model-to-Jacobian plumbing
- full SEP integration with symbolic model objects
- regime switching

Progress is tracked in [docs/porting_progress.md](docs/porting_progress.md).
