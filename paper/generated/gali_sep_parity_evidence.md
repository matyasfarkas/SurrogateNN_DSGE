# Full Galí nonlinear SEP Julia/Python parity evidence

**Evidence level:** independent reduced-horizon nonlinear scenario-tree parity

- Joint cases passed: 8/8 across seven unique policy-rule points.
- Worst complete-tree maximum absolute difference: 1.753452e-07 (threshold 1.0e-06).
- Worst expected-path maximum absolute difference: 7.217007e-09 (threshold 1.0e-08).
- Worst Python/Julia solver residuals: 9.233999e-09 / 9.213551e-09.
- SurrogateNN_DSGE revision: `a68b996c03486faa4e31059cfba04e7334c90f13` (clean).
- MacroModelling.jl revision: `b224f74fb892b24918aea27f0c22f609cdba2059` (clean).
- Model SHA-256: `53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55`.

## Claim boundary

This result validates a three-period, order-one, three-node-per-shock sparse nonlinear SEP tree with all three deterministic shock channels active. It does not validate the production horizon, nonlinear likelihood, posterior, or GPU performance.

## Per-case discrepancies

| Case | $\phi_\pi$ | $\phi_y$ | Tree max abs | Mean-path max abs | Python residual | Julia residual |
|---|---:|---:|---:|---:|---:|---:|
| `theta_0` | 1.500 | 0.125 | 1.217e-07 | 5.853e-09 | 6.530e-09 | 9.214e-09 |
| `theta_1` | 1.320 | 0.140 | 1.753e-07 | 5.609e-09 | 5.008e-14 | 9.214e-09 |
| `theta_2` | 1.380 | 0.135 | 1.716e-07 | 5.411e-09 | 3.921e-14 | 9.214e-09 |
| `theta_3` | 1.440 | 0.130 | 1.198e-07 | 7.217e-09 | 9.234e-09 | 9.214e-09 |
| `theta_4` | 1.500 | 0.125 | 1.217e-07 | 5.853e-09 | 6.530e-09 | 9.214e-09 |
| `theta_5` | 1.560 | 0.120 | 1.228e-07 | 4.917e-09 | 4.650e-09 | 9.214e-09 |
| `theta_6` | 1.620 | 0.115 | 1.234e-07 | 4.700e-09 | 3.331e-09 | 9.214e-09 |
| `theta_7` | 1.680 | 0.110 | 1.237e-07 | 4.564e-09 | 2.397e-09 | 9.214e-09 |
