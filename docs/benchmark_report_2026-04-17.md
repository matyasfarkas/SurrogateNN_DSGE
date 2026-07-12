# Python/JAX vs Julia Profile

Date: 2026-04-17

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

- Whole-process wall time: Python finished in 104.95s and Julia in 108.64s, so Julia/Python = 1.04x.
- Small-model first-order solution parity max abs diff: 1.776e-14.
- Medium-model first-order solution parity max abs diff: 6.816e-10.
- Small-model Kalman total loglikelihood abs/rel diff: 7.019e-03 / 1.170e-05.
- Medium-model Kalman total loglikelihood abs/rel diff: 4.144e-06 / 6.899e-09.
- Small-model filtered-variable path parity max abs diff: 2.623e-06.
- Medium-model filtered-variable path parity max abs diff: 5.338e-07.
- Small-model fixed-gate switching abs/rel diff: 6.933e-03 / 1.241e-05.
- Medium-model fixed-gate switching abs/rel diff: 9.229e-06 / 1.492e-08.
- Small-model automatic switching abs/rel diff: 6.269e-03 / 1.122e-05.
- Medium-model automatic switching abs/rel diff: 8.376e-06 / 1.354e-08.
- Python/JAX reverse-mode likelihood differentiation still failed during the benchmark with `NotImplementedError: Unimplemented case of QR decomposition derivative`.
- JAX only exposed `TFRT_CPU_0` in this environment, so this remains a CPU benchmark rather than a live GPU benchmark.
- SEP remains a bounded robustness smoke stage here; it is still reported for runtime coverage, not as a validated matched-likelihood parity stage on these large models.

## Whole Process

- Python wall/user/sys: 104.950s / 192.130s / 64.830s
- Python max RSS (raw `time -l` units): 2586755072
- Python peak memory footprint: 2789.8 MiB
- Julia wall/user/sys: 108.640s / 137.360s / 8.150s
- Julia max RSS (raw `time -l` units): 2718646272
- Julia peak memory footprint: 2231.7 MiB

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
| First-order solution matrix | 6.816e-10 |
| Kalman loglikelihood | 4.144e-06 |
| Kalman per-period path | 2.461e-06 |
| Kalman grad value | n/a |
| Kalman grad vector | n/a |
| Filtered variables | 5.338e-07 |
| Smoothed variables | 3.784e-07 |
| Filtered shocks | 1.647e-07 |
| Smoothed shocks | 1.437e-07 |
| Gate linear observations | 2.981e-08 |
| Gate shocks | 1.647e-07 |
| Gate e-stat | 1.270e-07 |
| Gate f-stat | 2.140e-08 |
| Switching fixed-gate | 9.229e-06 |
| Switching auto-gated | 8.376e-06 |

## Stage Timings

### small_fs2000

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.099493 | - | 22.069143 | - | n/a |
| first_order_solve | 0.833666 | 0.007885 | 10.303581 | 0.000049 | Julia |
| kalman_value | 2.004515 | 0.000409 | 3.219308 | 0.000105 | Julia |
| kalman_per_period | 0.523441 | 0.143880 | 0.141818 | 0.000108 | Julia |
| kalman_paths | 0.256636 | 0.071154 | 2.287649 | 0.015641 | Julia |
| kalman_grad | error | error | 15.615006 | 0.001903 | n/a |
| gate_stats | 0.069766 | 0.060717 | 9.827270 | 0.004953 | Julia |
| switching_fixed | 2.028555 | 0.000443 | 0.328441 | 0.000153 | Julia |
| switching_value | 4.050980 | 0.001432 | 0.196523 | 0.004859 | Python |
| sep_inversion | 16.617322 | 15.436592 | 7.594849 | 0.016999 | Julia |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

### medium_sw07_hlt

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.313387 | - | 8.423470 | - | n/a |
| first_order_solve | 1.825858 | 0.010396 | 0.281990 | 0.000068 | Julia |
| kalman_value | 8.623434 | 0.019336 | 0.001194 | 0.001008 | Julia |
| kalman_per_period | 0.641506 | 0.209148 | 0.001023 | 0.001002 | Julia |
| kalman_paths | 2.006587 | 1.680577 | 0.175658 | 0.171471 | Julia |
| kalman_grad | error | error | 2.066126 | 0.010973 | n/a |
| gate_stats | 0.500670 | 0.527284 | 0.106698 | 0.091011 | Julia |
| switching_fixed | 8.885778 | 0.020258 | 0.001809 | 0.001469 | Julia |
| switching_value | 18.643474 | 0.378076 | 0.110501 | 0.102096 | Julia |
| sep_inversion | 0.949411 | - | 2.055847 | - | n/a |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

## Interpretation

- `first_call` includes compilation and one-off setup overhead.
- `steady median` is the more relevant figure for repeated estimation inner loops.
- `kalman_grad` still uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.
- Fixed-gate switching isolates the likelihood mixer on shared gates; automatic switching also exercises gate reconstruction and filtering in each environment.
- The parity tables should be read before the timing tables. Any stage with weak parity should not be used to make strong runtime claims.
