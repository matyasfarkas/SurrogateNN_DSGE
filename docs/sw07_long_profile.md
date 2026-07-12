# SW07 Long Benchmark Profile

`benchmarks/sw07_long_profile.toml` is an opt-in long benchmark for the existing
`Smets_Wouters_2007_HLT` fixture. It does not introduce a hand-written Python
SW07 model; it reuses the upstream MacroModelling-style model file through the
same parser, reference steady-state export, Schur first-order solution, Kalman
filter, switching-order filter, NumPyro log-density wrappers, and bounded SEP
inversion path used by the standard benchmark harness.

Run it from the Python repo:

```bash
.venv/bin/python benchmarks/profile_validation.py benchmarks/sw07_long_profile.toml
```

The same override can be supplied by environment variable:

```bash
SURROGATENN_DSGE_BENCHMARK_CONFIG=benchmarks/sw07_long_profile.toml \
  .venv/bin/python benchmarks/profile_validation.py
```

The profile is calibrated from the 2026-07-12 Apple M4 CPU report and targets a
rough 2.5-3.5 hour wall-clock range on similar hardware. The target is not a
guarantee: JAX backend, BLAS threading, Colab GPU type, thermal state, and Julia
package precompilation can all move the runtime. After one calibration run, scale
the `*_reps` fields in the TOML by a common factor if a tighter target is needed.

NUTS is intentionally disabled in this long profile. NumPyro log densities are
covered, but the full SW07 NUTS path can still hit JAX derivative limitations in
the current Schur/doubling first-order solve stack. A long run should stress the
validated likelihood/filter/switching path rather than spend hours in a known
fragile sampler configuration.
