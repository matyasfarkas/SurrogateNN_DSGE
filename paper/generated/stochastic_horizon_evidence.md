# Stochastic Galí horizon evidence

Schema: `surrogatenn-dsge-stochastic-horizon-evidence/v1`

**Publication ready:** no

## Residual gates

| SEP horizon | Accepted paths | Maximum residual | Newton iterations |
|---:|---:|---:|---:|
| 3 | 8/8 | 1.728e-12 | 5--5 |
| 4 | 8/8 | 7.444e-09 | 4--5 |
| 5 | 8/8 | 1.928e-10 | 5--7 |

## First-step horizon sensitivity

| Comparison | All-state max | log output max | Inflation max | Policy-rate max |
|---:|---:|---:|---:|---:|
| 3→4 | 1.373e-02 | 2.905e-04 | 9.682e-04 | 1.115e-03 |
| 4→5 | 2.118e-02 | 3.957e-04 | 1.224e-03 | 1.394e-03 |
| 3→5 | 3.490e-02 | 6.862e-04 | 2.192e-03 | 2.510e-03 |

## Horizon-six boundary

The hard design (theta index 1) fails both solvers: sequential residual 1.020e-04; batched residual 1.024e-04.

## Claim boundary

- Supported: All eight prespecified path evaluations, spanning seven unique policy-rule points in the 12% perturbation design, solve the complete 23-equation model at stochastic SEP horizons three through five under the strict 1e-8 residual gate.
- Not supported: The evidence does not establish terminal-horizon convergence, a nonlinear likelihood, posterior fidelity, or a speed advantage. The hard design fails at horizon six under both solver implementations.
