#!/usr/bin/env python3
"""Pilot the GPU-native Cayley generalized-eigen screen against CPU ordqz.

This is not a full ordered QZ implementation. It benchmarks the JAX-native
Cayley pencil transform used by ``schur_gpu`` and compares a small CPU ordqz
audit subset for the same QME fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from surrogatenn_dsge import (
    DSGETimings,
    analyze_quadratic_matrix_equation_cayley_gpu_jax,
    analyze_quadratic_matrix_equation_schur,
)


def _dense_from_julia_sparse(rows: list[int], cols: list[int], values: list[float], shape: tuple[int, int]) -> jax.Array:
    matrix = np.zeros(shape, dtype=np.float64)
    for row, col, value in zip(rows, cols, values):
        matrix[row - 1, col - 1] = value
    return jnp.asarray(matrix)


def _rbc_qme_fixture() -> tuple[DSGETimings, jax.Array, jax.Array, jax.Array]:
    timings = DSGETimings.from_julia(
        present_only=["R", "y"],
        future_not_past=["Pi", "c"],
        past_not_future=["k", "z_delta"],
        mixed=["A"],
        future_not_past_and_mixed=["A", "Pi", "c"],
        past_not_future_and_mixed=["A", "k", "z_delta"],
        present_but_not_only=["A", "Pi", "c", "k", "z_delta"],
        mixed_in_past=["A"],
        not_mixed_in_past=["k", "z_delta"],
        mixed_in_future=["A"],
        exo=["delta_eps", "eps_z"],
        var=["A", "Pi", "R", "c", "k", "y", "z_delta"],
        aux=[],
        exo_present=[],
        nPresent_only=2,
        nMixed=1,
        nFuture_not_past_and_mixed=3,
        nPast_not_future_and_mixed=3,
        nPresent_but_not_only=5,
        nVars=7,
        nExo=2,
        present_only_idx=[3, 6],
        present_but_not_only_idx=[1, 2, 4, 5, 7],
        future_not_past_and_mixed_idx=[1, 2, 4],
        not_mixed_in_past_idx=[2, 3],
        past_not_future_and_mixed_idx=[1, 5, 7],
        mixed_in_past_idx=[1],
        mixed_in_future_idx=[1],
        past_not_future_idx=[5, 7],
        reorder=[5, 6, 1, 7, 3, 2, 4],
        dynamic_order=[3, 4, 5, 1, 2],
    )
    jacobian = _dense_from_julia_sparse(
        rows=[2, 3, 2, 3, 1, 5, 7, 4, 3, 4, 2, 3, 5, 2, 5, 1, 5, 6, 7, 1, 5, 6, 6, 7],
        cols=[1, 2, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15],
        values=[
            -0.01949762952125457,
            0.8249811251976374,
            0.6838672481452972,
            0.683867248150105,
            -1.42321160651834,
            1.42321160651834,
            1.0,
            -1.5000000000190756,
            -0.8241561440666456,
            0.999,
            -0.6838672481452971,
            -0.6838672481452971,
            -1.0,
            0.0017360837926088356,
            -1.0,
            1.0,
            -0.21396717122439846,
            1.0,
            -0.9,
            -0.023601001001000967,
            1.001001001001001,
            -0.9,
            -0.005,
            -0.0068,
        ],
        shape=(7, 15),
    )
    grad = jnp.asarray(jacobian, dtype=jnp.float64)
    dyn_index = np.arange(timings.nPresent_only, timings.nVars)
    comb = tuple(sorted(set(timings.future_not_past_and_mixed_idx) | set(timings.past_not_future_idx)))
    ref_map = {value: idx for idx, value in enumerate(comb)}
    future_in_comb = tuple(ref_map[value] for value in timings.future_not_past_and_mixed_idx)
    past_in_comb = tuple(ref_map[value] for value in timings.past_not_future_and_mixed_idx)
    selector = jnp.eye(len(comb), dtype=grad.dtype)

    n_future = timings.nFuture_not_past_and_mixed
    n_vars = timings.nVars
    n_past = timings.nPast_not_future_and_mixed
    grad_plus = grad[:, :n_future]
    grad_zero = grad[:, n_future : n_future + n_vars]
    grad_minus = grad[:, n_future + n_vars : n_future + n_vars + n_past]
    q_complete = jnp.linalg.qr(grad_zero[:, timings.present_only_idx], mode="complete")[0]
    a_plus = q_complete.T @ grad_plus
    a_zero = q_complete.T @ grad_zero
    a_minus = q_complete.T @ grad_minus
    return (
        timings,
        a_plus[dyn_index] @ selector[list(future_in_comb), :],
        a_zero[dyn_index][:, comb],
        a_minus[dyn_index] @ selector[list(past_in_comb), :],
    )


def _block(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--steady-repeats", type=int, default=5)
    parser.add_argument("--cpu-audit-draws", type=int, default=64)
    parser.add_argument("--shift-scale", type=float, default=2.0e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    timings, a_plus, a_zero, a_minus = _rbc_qme_fixture()
    shifts = jnp.linspace(-float(args.shift_scale), float(args.shift_scale), int(args.draws), dtype=jnp.float64)

    def analyze_shift(shift: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        result = analyze_quadratic_matrix_equation_cayley_gpu_jax(
            a_plus,
            a_zero.at[0, 0].add(shift),
            a_minus,
            timings,
        )
        return (
            result.solution,
            result.diagnostics.success,
            result.diagnostics.spectral_margin,
            result.diagnostics.relative_residual,
        )

    batched = jax.jit(jax.vmap(analyze_shift))
    started = time.perf_counter()
    solution, success, margin, residual = _block(batched(shifts))
    first_s = time.perf_counter() - started

    steady_times = []
    for _ in range(int(args.steady_repeats)):
        started = time.perf_counter()
        _block(batched(shifts))
        steady_times.append(time.perf_counter() - started)

    audit_n = min(int(args.cpu_audit_draws), int(args.draws))
    cpu_started = time.perf_counter()
    cpu_success = []
    cpu_margin = []
    cpu_solution_errors = []
    for idx, shift in enumerate(np.asarray(shifts[:audit_n], dtype=np.float64)):
        expected = analyze_quadratic_matrix_equation_schur(
            a_plus,
            a_zero.at[0, 0].add(float(shift)),
            a_minus,
            timings,
        )
        cpu_success.append(bool(expected.diagnostics.unique_stable_solution))
        cpu_roots = np.abs(np.asarray(expected.diagnostics.companion_roots, dtype=np.complex128))
        cpu_margin.append(float(np.min(np.abs(cpu_roots - 1.0))))
        cpu_solution_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(solution[idx], dtype=np.float64)
                        - np.asarray(expected.solution, dtype=np.float64)
                    )
                )
            )
        )
    cpu_s = time.perf_counter() - cpu_started

    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "draws": int(args.draws),
        "cpu_audit_draws": int(audit_n),
        "first_call_s": float(first_s),
        "steady_times_s": [float(x) for x in steady_times],
        "steady_median_s": float(np.median(steady_times)) if steady_times else None,
        "steady_draws_per_s": float(int(args.draws) / np.median(steady_times)) if steady_times else None,
        "cpu_audit_s": float(cpu_s),
        "cpu_audit_draws_per_s": float(audit_n / cpu_s) if cpu_s > 0 else None,
        "success_share": float(np.mean(np.asarray(success, dtype=bool))),
        "min_spectral_margin": float(np.min(np.asarray(margin, dtype=np.float64))),
        "median_spectral_margin": float(np.median(np.asarray(margin, dtype=np.float64))),
        "max_relative_residual": float(np.max(np.asarray(residual, dtype=np.float64))),
        "cpu_success_share": float(np.mean(np.asarray(cpu_success, dtype=bool))) if cpu_success else None,
        "cpu_min_spectral_margin": float(np.min(np.asarray(cpu_margin, dtype=np.float64))) if cpu_margin else None,
        "max_cpu_solution_error": float(np.max(np.asarray(cpu_solution_errors, dtype=np.float64)))
        if cpu_solution_errors
        else None,
    }
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
