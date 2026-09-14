# DSGE Speed Comparison

Validation directory: `/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/benchmarks/results/20260712T083357`

## Scope

- Julia numbers are stage/profile timings from the MacroModelling validation harness; a Julia Turing/AdvancedHMC posterior ESS/sec sweep has not been measured yet.
- NumPyro rows are full sampler wall times including JAX compilation and warmup for the saved runs.
- JAX static-HMC rows use the fixed-shape chain-parallel benchmark. For step-size grids, rows after the first are post-compile runs with the same compiled shape.
- ESS diagnostics from very short chains are noisy; use seconds per minimum ESS only as a screening metric until longer chains are run.

## Main Result

Best measured GPU static-HMC is 3.12x faster than the best measured M4 NumPyro sampler on seconds per minimum ESS. Raw posterior draws/sec ratio is 8.29x.

## Stage Timings

| Language | Stage | Status | First call s | Steady median s | Steady reps |
| --- | --- | --- | --- | --- | --- |
| Julia | first_order_solve | ok | 0.371 | 7.11e-05 | 1 |
| Julia | kalman_value | ok | 0.001 | 0.011 | 6 |
| Julia | kalman_grad | ok | 2.920 | 0.077 | 2 |
| Julia | switching_fixed | ok | 0.029 | 0.013 | 2 |
| Julia | switching_value | ok | 0.371 | 0.207 | 2 |
| Python/JAX | first_order_solve | ok | 2.313 | 0.023 | 1 |
| Python/JAX | kalman_value | ok | 12.20 | 0.398 | 6 |
| Python/JAX | kalman_grad | error |  |  |  |
| Python/JAX | switching_fixed | ok | 12.04 | 0.269 | 2 |
| Python/JAX | switching_value | ok | 24.81 | 1.294 | 2 |

## Posterior Samplers

| Environment | Sampler | Chains | Warmup | Samples | Periods | Params | Step size | Timing scope | Wall s | Draws/s | Min ESS | s/min ESS | Max Rhat | Accept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macOS / JAX cpu | JAX static HMC | 2 | 1 | 2 | 10 | 15 | 0.020 | cold compile+run | 45.94 | 0.087 | 1.335 | 34.41 |  | 1.000 |
| macOS / JAX cpu | JAX static HMC | 4 | 4 | 8 | 10 | 1 | 0.050 | steady replay | 9.969 | 3.210 | 2.217 | 4.496 | 5.440 | 1.000 |
| Linux / JAX gpu | JAX static HMC | 32 | 16 | 32 | 40 | 15 | 0.200 | cold compile+run | 298.5 | 3.430 | 170.7 | 1.749 | 1.116 | 0.994 |
| Linux / JAX gpu | JAX static HMC | 32 | 16 | 32 | 40 | 15 | 0.600 | post-compile run | 155.3 | 6.595 |  |  | 0.972 | 0.969 |
| Linux / JAX gpu | JAX static HMC | 32 | 16 | 32 | 40 | 15 | 1.000 | post-compile run | 155.0 | 6.605 | 548.3 | 0.283 | 1.035 | 0.873 |
| macOS / JAX cpu | NumPyro HMC | 8 | 16 | 32 | 40 | 15 | 0.100 | full MCMC run | 321.2 | 0.797 | 364.1 | 0.882 | 1.004 | 0.942 |
| macOS / JAX cpu | NumPyro NUTS | 4 | 8 | 8 | 10 | 1 |  | full MCMC run | 85.59 | 0.374 | 4.052 | 21.12 | 1.445 | 0.859 |
| macOS / JAX cpu | NumPyro NUTS | 8 | 16 | 32 | 40 | 15 |  | full MCMC run | 522.1 | 0.490 | 294.0 | 1.775 | 1.021 | 0.892 |

## Batched Likelihood / Gradient

| Environment | Batch | Periods | Params | Value evals/s | Gradient evals/s | Value first s | Gradient first s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macOS / JAX cpu | 1 | 10 | 1 | 34.83 | 15.22 | 3.816 | 8.655 |
| macOS / JAX cpu | 1 | 40 | 15 | 14.26 | 11.34 | 4.105 | 8.997 |
| macOS / JAX cpu | 4 | 10 | 1 | 44.83 | 14.77 | 4.688 | 12.31 |
| macOS / JAX cpu | 8 | 40 | 15 | 32.38 | 14.05 | 4.952 | 13.11 |
| macOS / JAX cpu | 16 | 10 | 1 | 38.63 | 20.25 | 14.48 | 32.31 |
| macOS / JAX cpu | 32 | 40 | 15 | 44.53 | 19.82 | 14.12 | 33.32 |
| macOS / JAX cpu | 64 | 10 | 1 | 50.37 | 21.03 | 14.51 | 32.53 |
