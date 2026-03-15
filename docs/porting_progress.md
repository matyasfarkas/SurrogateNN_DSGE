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

## Explicit gaps

- The Julia `:bartels_stewart`, `:bicgstab`, and `:gmres` variants are not ported yet.
- The current dense fallback is a direct Kronecker solve, not a Bartels-Stewart implementation.
- No claim is made yet about feature parity beyond the Lyapunov solver.

## Environment note

- A local git repository can be maintained here.
- A remote GitHub repository has not been created from this environment because `gh` is not installed/configured.

