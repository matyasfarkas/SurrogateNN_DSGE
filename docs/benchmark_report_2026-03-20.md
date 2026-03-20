# Python/JAX vs Julia Profile

Date: 2026-03-20

## Environment

- CPU: Apple M4
- Logical cores: 10
- Performance cores: 4
- Efficiency cores: 6
- Memory (bytes): 17179869184
- JAX devices: TFRT_CPU_0
- Julia version: 1.12.4
- Threads requested: 8

## Scope

- Small model: `FS2000`.
- Medium model: `Smets_Wouters_2007_HLT`.
- Shared synthetic observations are generated once from the reference first-order solution and reused in both environments.
- The report validates solution parity first, then Kalman likelihood/filter parity, then gate/switching parity, and only then compares timings.
- Important caveat: the Python HLT benchmark still uses a Julia-exported reference steady state so the comparison isolates already-ported solve/filter code instead of cold-start steady-state recovery.

## Key Findings

- Whole-process wall time: Python finished in 77.83s and Julia in 114.06s, so Julia/Python = 1.47x.
- Small-model first-order solution parity max abs diff: 9.948e-14.
- Medium-model first-order solution parity max abs diff: 2.257e-10.
- Small-model Kalman total loglikelihood abs/rel diff: 7.019e-03 / 1.170e-05.
- Medium-model Kalman total loglikelihood abs/rel diff: 4.144e-06 / 6.899e-09.
- Small-model filtered-variable path parity max abs diff: 2.623e-06.
- Medium-model filtered-variable path parity max abs diff: 5.343e-07.
- Small-model fixed-gate switching abs/rel diff: 6.933e-03 / 1.241e-05.
- Medium-model fixed-gate switching abs/rel diff: 9.229e-06 / 1.492e-08.
- Small-model automatic switching abs/rel diff: 6.269e-03 / 1.122e-05.
- Medium-model automatic switching abs/rel diff: 8.376e-06 / 1.354e-08.
- Python/JAX reverse-mode likelihood differentiation still failed during the benchmark with `NotImplementedError: Unimplemented case of QR decomposition derivative`.
- JAX only exposed `TFRT_CPU_0` in this environment, so this remains a CPU benchmark rather than a live GPU benchmark.
- SEP remains a bounded robustness smoke stage here; it is still reported for runtime coverage, not as a validated matched-likelihood parity stage on these large models.

## Whole Process

- Python wall/user/sys: 77.830s / 180.010s / 34.450s
- Python max RSS (raw `time -l` units): 1781841920
- Python peak memory footprint: 1552.5 MiB
- Julia wall/user/sys: 114.060s / 145.390s / 8.390s
- Julia max RSS (raw `time -l` units): 2747678720
- Julia peak memory footprint: 2050.6 MiB

## Parity Results

### small_fs2000 Parity

| Check | Max abs diff |
| --- | ---: |
| First-order solution matrix | 9.948e-14 |
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
| First-order solution matrix | 2.257e-10 |
| Kalman loglikelihood | 4.144e-06 |
| Kalman per-period path | 2.461e-06 |
| Kalman grad value | n/a |
| Kalman grad vector | n/a |
| Filtered variables | 5.343e-07 |
| Smoothed variables | 3.789e-07 |
| Filtered shocks | 1.648e-07 |
| Smoothed shocks | 1.437e-07 |
| Gate linear observations | 2.982e-08 |
| Gate shocks | 1.648e-07 |
| Gate e-stat | 1.270e-07 |
| Gate f-stat | 2.140e-08 |
| Switching fixed-gate | 9.229e-06 |
| Switching auto-gated | 8.376e-06 |

## Stage Timings

### small_fs2000

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.100400 | - | 23.521722 | - | n/a |
| first_order_solve | 0.925822 | 0.015419 | 9.697571 | 0.000050 | Julia |
| kalman_value | 1.055343 | 0.000409 | 3.465075 | 0.000105 | Julia |
| kalman_per_period | 0.616040 | 0.162485 | 0.170154 | 0.000108 | Julia |
| kalman_paths | 0.250249 | 0.067537 | 2.570652 | 0.017581 | Julia |
| kalman_grad | error | error | 17.019021 | 0.002235 | n/a |
| gate_stats | 0.077279 | 0.068586 | 10.993294 | 0.004799 | Julia |
| switching_fixed | 1.067347 | 0.000411 | 0.449025 | 0.000156 | Julia |
| switching_value | 2.210615 | 0.001492 | 0.236011 | 0.005555 | Python |
| sep_inversion | 17.448895 | 16.049545 | 4.388839 | 0.006208 | Julia |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

### medium_sw07_hlt

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.323449 | - | 8.865057 | - | n/a |
| first_order_solve | 1.895753 | 0.009453 | 0.344250 | 0.000082 | Julia |
| kalman_value | 3.867001 | 0.019114 | 0.001197 | 0.001021 | Julia |
| kalman_per_period | 0.709711 | 0.301544 | 0.001013 | 0.002744 | Julia |
| kalman_paths | 2.225453 | 2.026809 | 0.229364 | 0.244405 | Julia |
| kalman_grad | error | error | 2.237489 | 0.009056 | n/a |
| gate_stats | 0.591784 | 0.562068 | 0.135807 | 0.124583 | Julia |
| switching_fixed | 4.015816 | 0.021419 | 0.001435 | 0.001225 | Julia |
| switching_value | 8.416332 | 0.503420 | 0.112390 | 0.110900 | Julia |
| sep_inversion | 0.925942 | - | 4.688666 | - | n/a |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

## Interpretation

- `first_call` includes compilation and one-off setup overhead.
- `steady median` is the more relevant figure for repeated estimation inner loops.
- `kalman_grad` still uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.
- Fixed-gate switching isolates the likelihood mixer on shared gates; automatic switching also exercises gate reconstruction and filtering in each environment.
- The parity tables should be read before the timing tables. Any stage with weak parity should not be used to make strong runtime claims.
