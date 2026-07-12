# Python/JAX vs Julia Profile

Date: 2026-07-12

## Environment

- CPU: Apple M4
- Logical cores: 10
- Performance cores: 4
- Efficiency cores: 6
- Memory (bytes): 17179869184
- JAX devices: TFRT_CPU_0
- Julia version: 1.12.6
- Threads requested: 8
- JAX platform setting: auto

## Scope

- Small model: `FS2000`.
- Medium model: `Smets_Wouters_2007_HLT`.
- Shared synthetic observations are generated once from the reference first-order solution and reused in both environments.
- The report validates solution parity first, then Kalman likelihood/filter parity, then gate/switching parity, and only then compares timings.
- Important caveat: the Python HLT benchmark still uses a Julia-exported reference steady state so the comparison isolates already-ported solve/filter code instead of cold-start steady-state recovery.

## Key Findings

- Whole-process wall time: Python finished in 197.85s and Julia in 150.03s, so Julia/Python = 0.76x.
- Small-model first-order solution parity max abs diff: 1.776e-14.
- Medium-model first-order solution parity max abs diff: 2.955e-10.
- Small-model Kalman total loglikelihood abs/rel diff: 7.019e-03 / 1.170e-05.
- Medium-model Kalman total loglikelihood abs/rel diff: 4.144e-06 / 6.899e-09.
- Small-model filtered-variable path parity max abs diff: 2.623e-06.
- Medium-model filtered-variable path parity max abs diff: 5.342e-07.
- Small-model fixed-gate switching abs/rel diff: 6.933e-03 / 1.241e-05.
- Medium-model fixed-gate switching abs/rel diff: 9.229e-06 / 1.492e-08.
- Small-model automatic switching abs/rel diff: 6.269e-03 / 1.122e-05.
- Medium-model automatic switching abs/rel diff: 8.376e-06 / 1.354e-08.
- Python/JAX reverse-mode likelihood differentiation still failed during the benchmark with `NotImplementedError: Unimplemented case of QR decomposition derivative`.
- Small-model NumPyro/JAX log densities evaluated successfully: Kalman 613.423935, switching 572.102963.
- NumPyro NUTS smoke did not complete on the benchmark DSGE model: `NotImplementedError: Unimplemented case of QR decomposition derivative`.
- JAX only exposed `TFRT_CPU_0` in this environment, so this remains a CPU benchmark rather than a live GPU benchmark.
- SEP remains a bounded robustness smoke stage here; it is still reported for runtime coverage, not as a validated matched-likelihood parity stage on these large models.

## Whole Process

- Python wall/user/sys: 197.850s / 411.920s / 31.840s
- Python max RSS (raw `time -l` units): 2233466880
- Python peak memory footprint: 3117.3 MiB
- Julia wall/user/sys: 150.030s / 188.150s / 4.230s
- Julia max RSS (raw `time -l` units): 2516172800
- Julia peak memory footprint: 2142.4 MiB

## Parity Results

### small_fs2000 Parity

| Check | Max abs diff |
| --- | ---: |
| First-order solution matrix | 1.776e-14 |
| Kalman loglikelihood | 7.019e-03 |
| Kalman per-period path | 7.839e-05 |
| Kalman grad value | n/a |
| Kalman grad vector | n/a |
| Filtered variables | 2.623e-06 |
| Smoothed variables | 4.541e-06 |
| Filtered shocks | 7.775e-06 |
| Smoothed shocks | 7.775e-06 |
| Gate linear observations | 5.603e-08 |
| Gate shocks | 7.775e-06 |
| Gate e-stat | 3.026e-05 |
| Gate f-stat | 1.034e-05 |
| Switching fixed-gate | 6.933e-03 |
| Switching auto-gated | 6.269e-03 |

### medium_sw07_hlt Parity

| Check | Max abs diff |
| --- | ---: |
| First-order solution matrix | 2.955e-10 |
| Kalman loglikelihood | 4.144e-06 |
| Kalman per-period path | 2.461e-06 |
| Kalman grad value | n/a |
| Kalman grad vector | n/a |
| Filtered variables | 5.342e-07 |
| Smoothed variables | 3.788e-07 |
| Filtered shocks | 1.648e-07 |
| Smoothed shocks | 1.437e-07 |
| Gate linear observations | 2.982e-08 |
| Gate shocks | 1.648e-07 |
| Gate e-stat | 1.269e-07 |
| Gate f-stat | 2.140e-08 |
| Switching fixed-gate | 9.229e-06 |
| Switching auto-gated | 8.376e-06 |

## Stage Timings

### small_fs2000

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.098375 | - | 29.256335 | - | n/a |
| first_order_solve | 1.211021 | 0.010248 | 13.659106 | 0.000049 | Julia |
| kalman_value | 2.728017 | 0.001642 | 4.171216 | 0.000106 | Julia |
| kalman_per_period | 0.764744 | 0.197521 | 0.184863 | 0.000172 | Julia |
| kalman_paths | 0.583990 | 0.382377 | 2.984679 | 0.019028 | Julia |
| kalman_grad | error | error | 20.784987 | 0.011811 | n/a |
| gate_stats | 0.132001 | 0.145138 | 13.411137 | 0.005812 | Julia |
| switching_fixed | 2.750274 | 0.000935 | 0.431250 | 0.000155 | Julia |
| switching_value | 5.739379 | 0.079418 | 0.278795 | 0.009481 | Julia |
| numpyro_kalman_log_density | 2.461988 | 0.000615 | - | - | n/a |
| numpyro_switching_log_density | 2.727250 | 0.000967 | - | - | n/a |
| numpyro_nuts_smoke | error | error | - | - | n/a |
| sep_inversion | 23.801456 | 22.694233 | 6.202743 | 0.006137 | Julia |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative
- Python `numpyro_nuts_smoke` error: NotImplementedError: Unimplemented case of QR decomposition derivative

### medium_sw07_hlt

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.416412 | - | 10.637170 | - | n/a |
| first_order_solve | 2.312875 | 0.023115 | 0.370939 | 0.000071 | Julia |
| kalman_value | 12.196485 | 0.398475 | 0.001173 | 0.010630 | Julia |
| kalman_per_period | 0.976049 | 0.420645 | 0.001040 | 0.001008 | Julia |
| kalman_paths | 4.215009 | 5.366014 | 0.366561 | 0.424824 | Julia |
| kalman_grad | error | error | 2.919949 | 0.077195 | n/a |
| gate_stats | 1.201527 | 1.162841 | 0.175816 | 0.240152 | Julia |
| switching_fixed | 12.044920 | 0.269460 | 0.028637 | 0.012955 | Julia |
| switching_value | 24.808983 | 1.293763 | 0.371279 | 0.207076 | Julia |
| numpyro_kalman_log_density | 12.179868 | 0.295255 | - | - | n/a |
| numpyro_switching_log_density | 11.291073 | 0.211926 | - | - | n/a |
| numpyro_nuts_smoke | - | - | - | - | n/a |
| sep_inversion | 1.460720 | - | 13.157902 | - | n/a |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

## Interpretation

- `first_call` includes compilation and one-off setup overhead.
- `steady median` is the more relevant figure for repeated estimation inner loops.
- `kalman_grad` still uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.
- Fixed-gate switching isolates the likelihood mixer on shared gates; automatic switching also exercises gate reconstruction and filtering in each environment.
- NumPyro stages use the Python/JAX likelihood wrappers with calibrated-parameter-centered benchmark priors; Julia has no matching NumPyro stage, so those rows validate JAX+NumPyro runtime coverage rather than cross-language sampler parity.
- The parity tables should be read before the timing tables. Any stage with weak parity should not be used to make strong runtime claims.
