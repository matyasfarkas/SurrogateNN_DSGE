# Full Galí nonlinear SEP Julia/Python parity evidence

**Evidence level:** independent reduced-horizon nonlinear scenario-tree parity

- Joint cases passed: 8/8 across seven unique policy-rule points.
- Worst complete-tree maximum absolute difference: 6.399040e-08 (threshold 1.0e-06).
- Worst expected-path maximum absolute difference: 6.500682e-09 (threshold 1.0e-08).
- Worst Python/Julia solver residuals: 5.634816e-11 / 8.497987e-11.
- SurrogateNN_DSGE revision: `790fe8011b8a46f55ff0187867b1c60f4b898f34` (clean).
- MacroModelling.jl revision: `b224f74fb892b24918aea27f0c22f609cdba2059` (clean).
- Model SHA-256: `53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55`.

## Claim boundary

This result validates a five-period, order-one, three-node-per-shock sparse nonlinear SEP tree with all three deterministic shock channels active. It does not validate the production horizon, nonlinear likelihood, posterior, or GPU performance.

## Per-case discrepancies

| Case | $\phi_\pi$ | $\phi_y$ | Tree max abs | Mean-path max abs | Python residual | Julia residual |
|---|---:|---:|---:|---:|---:|---:|
| `theta_0` | 1.500 | 0.125 | 2.223e-08 | 2.462e-09 | 2.363e-11 | 8.498e-11 |
| `theta_1` | 1.320 | 0.140 | 6.125e-10 | 1.404e-10 | 2.052e-12 | 8.498e-11 |
| `theta_2` | 1.380 | 0.135 | 2.720e-09 | 9.683e-11 | 4.827e-13 | 8.498e-11 |
| `theta_3` | 1.440 | 0.130 | 6.399e-08 | 6.501e-09 | 5.635e-11 | 8.498e-11 |
| `theta_4` | 1.500 | 0.125 | 2.223e-08 | 2.462e-09 | 2.363e-11 | 8.498e-11 |
| `theta_5` | 1.560 | 0.120 | 7.823e-09 | 1.016e-09 | 1.086e-11 | 8.498e-11 |
| `theta_6` | 1.620 | 0.115 | 2.327e-09 | 4.432e-10 | 5.374e-12 | 8.498e-11 |
| `theta_7` | 1.680 | 0.110 | 4.146e-10 | 1.974e-10 | 2.826e-12 | 8.498e-11 |
