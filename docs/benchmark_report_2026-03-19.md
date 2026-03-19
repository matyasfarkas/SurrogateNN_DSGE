# Python/JAX vs Julia Profile

Date: 2026-03-19

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
- Workload: first-order Schur solve, Kalman loglikelihood, Kalman gradient on the differentiable doubling path, automatic filter-gated switching likelihood, and bounded SEP inversion smoke.
- Shared synthetic observations were generated once in Python from the reference first-order solution and then reused in both environments.
- Important caveat: the Python HLT benchmark uses a Julia-exported reference steady state because cold-start HLT steady-state recovery is still not robust enough in the port for a fair timing comparison.

## Key Findings

- Whole-process wall time: Python finished in 69.53s and Julia in 177.82s, so Python was 2.56x faster on this benchmark harness.
- Small-model Kalman likelihood parity is good: FS2000 differs by 0.001% between Python and Julia.
- Medium-model Kalman likelihood parity is not yet good: HLT differs by 106837.749% between Python and Julia on the shared payload, so medium-model timings should be treated as a runtime stress test, not a validated apples-to-apples estimation benchmark.
- Python/JAX reverse-mode likelihood differentiation failed on these benchmark models with `NotImplementedError: Unimplemented case of QR decomposition derivative`. Julia completed the same stage.
- On the small model, steady-state switching likelihood evaluation was faster in Python (0.001862s median) than Julia (0.017639s median).
- JAX only exposed `TFRT_CPU_0` in this environment, so none of these runs exercised a live GPU backend.
- SEP here is a bounded robustness smoke stage. Both environments produced non-finite or failure-style SEP outputs on these settings, so SEP timings are not a validated matched-likelihood comparison.

## Whole Process

- Python wall/user/sys: 69.530s / 97.170s / 7.190s
- Python max RSS (raw `time -l` units): 638287872
- Python peak memory footprint: 753.8 MiB
- Julia wall/user/sys: 177.820s / 159.170s / 56.140s
- Julia max RSS (raw `time -l` units): 2066186240
- Julia peak memory footprint: 2514.0 MiB

## Stage Results

### small_fs2000

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.092725 | - | 33.532416 | - | n/a |
| first_order_solve | 1.413073 | 0.020825 | 11.899887 | 0.000495 | Julia |
| kalman_value | 0.997425 | 0.000446 | 5.931159 | 0.000215 | Julia |
| kalman_grad | error | error | 30.039975 | 0.002852 | n/a |
| switching_value | 1.876565 | 0.001862 | 32.771819 | 0.017639 | Python |
| sep_inversion | 22.634325 | 20.674319 | 6.254147 | 0.009754 | Julia |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

### medium_sw07_hlt

| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |
| --- | ---: | ---: | ---: | ---: | --- |
| model_load | 0.432591 | - | 12.386417 | - | n/a |
| first_order_solve | 2.847268 | 0.015938 | 0.912899 | 0.000165 | Julia |
| kalman_value | 2.104578 | 0.052763 | 0.014942 | 0.002783 | Julia |
| kalman_grad | error | error | 3.341544 | 0.026135 | n/a |
| switching_value | 4.763191 | 0.086138 | 0.824269 | 0.885618 | Python |
| sep_inversion | 6.921591 | - | 6.807737 | - | n/a |

- Python `kalman_grad` error: NotImplementedError: Unimplemented case of QR decomposition derivative

## Interpretation

- `first_call` includes language-specific JIT or XLA compilation overhead.
- `steady median` is the more relevant figure for repeated estimation inner loops.
- `kalman_grad` uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.
- Julia stage timings are public-API timings. They include its own steady-state and solution preparation work when the API does so.
- Python medium-model timings are inner-loop timings conditional on a supplied reference steady state.
