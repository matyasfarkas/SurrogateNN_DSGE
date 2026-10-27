# Full Galí Julia/Python core parity evidence

**Evidence level:** independent core parity

- Joint cases passed: 8/8 across seven unique policy-rule points.
- Worst steady-state maximum absolute difference: 9.059420e-14 (threshold 1.0e-10).
- Worst first-order Schur maximum absolute difference: 2.890488e-11 (threshold 1.0e-08).
- Worst Python steady-state residual norm: 1.110223e-16.
- SurrogateNN_DSGE revision: `e6a52ff7f01922885009f3b81fae21447b7c1205` (clean).
- MacroModelling.jl revision: `b224f74fb892b24918aea27f0c22f609cdba2059` (clean).
- Model SHA-256: `53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55`.

## Claim boundary

This result validates the shared model source, non-stochastic steady state, and first-order Schur solution. It does not validate nonlinear trajectories, nonlinear likelihoods, a posterior, or GPU performance.

## Per-case discrepancies

| Case | $\phi_\pi$ | $\phi_y$ | SS max abs | First-order max abs |
|---|---:|---:|---:|---:|
| `theta_0` | 1.500 | 0.125 | 9.059e-14 | 1.240e-12 |
| `theta_1` | 1.320 | 0.140 | 9.059e-14 | 2.890e-11 |
| `theta_2` | 1.380 | 0.135 | 9.059e-14 | 2.956e-12 |
| `theta_3` | 1.440 | 0.130 | 9.059e-14 | 2.682e-12 |
| `theta_4` | 1.500 | 0.125 | 9.059e-14 | 1.240e-12 |
| `theta_5` | 1.560 | 0.120 | 9.059e-14 | 9.805e-13 |
| `theta_6` | 1.620 | 0.115 | 9.059e-14 | 1.862e-12 |
| `theta_7` | 1.680 | 0.110 | 9.059e-14 | 5.819e-12 |
