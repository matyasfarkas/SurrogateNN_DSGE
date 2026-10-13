#!/usr/bin/env python3
"""Audit CPU ordered QZ, GPU-native Cayley, and doubling over model support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import jax
import jax.numpy as jnp
import scipy.linalg as scipy_linalg

import posterior_sampling_speed as posterior_speed
import surrogatenn_dsge as sdsge
from surrogatenn_dsge.dsge import (
    _classify_schur_determinacy,
    _indexin,
    _prepare_first_order_qme_system,
    _schur_stable_selection,
    quadratic_matrix_equation_residual,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_hlt_case(payload_path: Path, case_name: str) -> dict[str, Any]:
    return posterior_speed._load_payload_case(payload_path, case_name)


def _parameter_names(model: Any, case: dict[str, Any], spec: str) -> tuple[str, ...]:
    if spec == "payload":
        return tuple(str(name) for name in case["parameter_subset"])
    return posterior_speed._select_parameter_names(model, spec)


def _centered_uniform_intervals(
    parameter_names: Sequence[str],
    center: np.ndarray,
    *,
    width_scale: float,
    width_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(parameter_names),), dtype=np.float64)
    upper = np.empty_like(lower)
    for idx, name in enumerate(parameter_names):
        lower[idx], upper[idx] = posterior_speed._prior_interval(
            str(name),
            float(center[idx]),
            float(width_scale),
            float(width_floor),
        )
    return lower, upper


def _draw_theta(
    *,
    model: Any,
    case: dict[str, Any],
    parameter_names: Sequence[str],
    parameter_set: str,
    draw_mode: str,
    draws: int,
    seed: int,
    width_scale: float,
    width_floor: float,
    include_reference: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    base_values = np.asarray(model.parameter_values, dtype=np.float64)
    base_subset = np.asarray([base_values[parameter_index[name]] for name in parameter_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_random = max(0, int(draws) - (1 if include_reference else 0))
    if draw_mode == "package-prior":
        design = sdsge.sample_prior_parameters(parameter_set, n_random, seed=seed)
        if tuple(design.names) != tuple(parameter_names):
            raise ValueError(
                "Package-prior parameter names do not match selected model parameters: "
                f"{design.names} vs {tuple(parameter_names)}."
            )
        theta = np.asarray(design.theta, dtype=np.float64)
        lower = np.asarray(design.lower, dtype=np.float64)
        upper = np.asarray(design.upper, dtype=np.float64)
        support_label = parameter_set
    elif draw_mode == "centered-uniform":
        lower, upper = _centered_uniform_intervals(
            parameter_names,
            base_subset,
            width_scale=width_scale,
            width_floor=width_floor,
        )
        unit = rng.random((len(parameter_names), n_random))
        theta = lower[:, None] + (upper - lower)[:, None] * unit
        support_label = f"centered_uniform(scale={width_scale}, floor={width_floor})"
    else:
        raise ValueError(f"Unsupported draw_mode {draw_mode!r}.")
    if include_reference:
        theta = np.concatenate([base_subset[:, None], theta], axis=1)
    return theta, lower, upper, support_label


def _full_parameter_vector(
    model: Any,
    parameter_names: Sequence[str],
    theta_column: np.ndarray,
) -> np.ndarray:
    values = np.asarray(model.parameter_values, dtype=np.float64).copy()
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value in zip(parameter_names, theta_column):
        values[parameter_index[name]] = float(value)
    return values


def _analyze_qme_real_qz(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    timings: Any,
    *,
    acceptance_tol: float,
) -> dict[str, Any]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    solution = np.array(a_arr, copy=True)
    roots = np.asarray([], dtype=np.complex128)
    stable_mask = np.asarray([], dtype=bool)
    decomposition_succeeded = False
    invariant_subspace_invertible = False
    solution_extracted = False
    invariant_subspace_rcond = 0.0
    relative_residual = float("inf")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        comb = tuple(
            sorted(
                set(timings.future_not_past_and_mixed_idx)
                | set(timings.past_not_future_idx)
            )
        )
        future_in_comb = _indexin(timings.future_not_past_and_mixed_idx, comb)
        past_not_future_and_mixed_in_comb = _indexin(
            timings.past_not_future_and_mixed_idx,
            comb,
        )
        past_not_future_in_comb = _indexin(timings.past_not_future_idx, comb)
        a_tilde_plus = a_arr[:, list(future_in_comb)]
        a_tilde_minus = c_arr[:, list(past_not_future_and_mixed_in_comb)]
        a_tilde_zero_plus = b_arr[:, list(future_in_comb)]
        a_tilde_zero_minus = b_arr[:, list(past_not_future_in_comb)] @ np.eye(
            timings.nPast_not_future_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.not_mixed_in_past_idx), :]
        z_minus = np.zeros(
            (timings.nMixed, timings.nPast_not_future_and_mixed),
            dtype=a_arr.dtype,
        )
        i_minus = np.eye(
            timings.nPast_not_future_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.mixed_in_past_idx), :]
        z_plus = np.zeros(
            (timings.nMixed, timings.nFuture_not_past_and_mixed),
            dtype=a_arr.dtype,
        )
        i_plus = np.eye(
            timings.nFuture_not_past_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.mixed_in_future_idx), :]
        d_pencil = np.block(
            [
                [a_tilde_zero_minus, a_tilde_plus],
                [i_minus, z_plus],
            ]
        )
        e_pencil = np.block(
            [
                [-a_tilde_minus, -a_tilde_zero_plus],
                [z_minus, i_plus],
            ]
        )
        if d_pencil.shape == e_pencil.shape and d_pencil.ndim == 2 and d_pencil.shape[0] == d_pencil.shape[1]:
            if d_pencil.shape[0] == 0:
                decomposition_succeeded = True
                invariant_subspace_invertible = True
                solution_extracted = True
                solution = np.array(a_arr, copy=True)
                relative_residual = 0.0
            else:
                try:
                    s_matrix, t_matrix, alpha, beta, _, z_matrix = scipy_linalg.ordqz(
                        d_pencil,
                        e_pencil,
                        sort=_schur_stable_selection,
                        output="real",
                        check_finite=False,
                    )
                    decomposition_succeeded = True
                    roots = np.asarray(beta / alpha, dtype=np.complex128)
                    stable_mask = _schur_stable_selection(alpha, beta)
                except Exception:
                    decomposition_succeeded = False
                if decomposition_succeeded and int(np.count_nonzero(stable_mask)) == timings.nPast_not_future_and_mixed:
                    n_past = timings.nPast_not_future_and_mixed
                    z21 = z_matrix[n_past:, :n_past]
                    z11 = z_matrix[:n_past, :n_past]
                    s11 = s_matrix[:n_past, :n_past]
                    t11 = t_matrix[:n_past, :n_past]
                    try:
                        z11_cond = np.linalg.cond(z11)
                        invariant_subspace_rcond = (
                            float(1.0 / z11_cond) if np.isfinite(z11_cond) else 0.0
                        )
                        d_block = scipy_linalg.solve(z11.T, z21.T, check_finite=False).T
                        l_core = scipy_linalg.solve(s11, t11, check_finite=False)
                        l_block = scipy_linalg.solve(
                            z11.T,
                            (z11 @ l_core).T,
                            check_finite=False,
                        ).T
                        invariant_subspace_invertible = True
                        sol = np.vstack([l_block[list(timings.not_mixed_in_past_idx), :], d_block])
                        selection = np.eye(len(comb), dtype=a_arr.dtype)[
                            list(past_not_future_and_mixed_in_comb),
                            :,
                        ]
                        solution = np.asarray(sol[list(timings.dynamic_order), :] @ selection, dtype=np.float64)
                        if np.all(np.isfinite(solution)):
                            solution_extracted = True
                            relative_residual = float(
                                np.asarray(
                                    quadratic_matrix_equation_residual(a_arr, solution, b_arr, c_arr)
                                )
                            )
                    except Exception:
                        invariant_subspace_invertible = False
                        solution_extracted = False
    stable_count = int(np.count_nonzero(stable_mask))
    expected_stable_count = int(timings.nPast_not_future_and_mixed)
    classification = _classify_schur_determinacy(
        expected_stable_count=expected_stable_count,
        stable_count=stable_count,
        decomposition_succeeded=decomposition_succeeded,
        invariant_subspace_invertible=invariant_subspace_invertible,
        solution_extracted=solution_extracted,
        relative_residual=relative_residual,
        acceptance_tol=acceptance_tol,
    )
    return {
        "solution": solution,
        "classification": classification,
        "unique": classification == "unique_stable_solution",
        "stable_count": stable_count,
        "expected_stable_count": expected_stable_count,
        "decomposition_succeeded": bool(decomposition_succeeded),
        "invariant_subspace_invertible": bool(invariant_subspace_invertible),
        "solution_extracted": bool(solution_extracted),
        "invariant_subspace_rcond": float(invariant_subspace_rcond),
        "relative_residual": float(relative_residual),
        "companion_roots": roots,
    }


def _audit_draw(
    *,
    model: Any,
    steady_state: np.ndarray,
    parameter_names: Sequence[str],
    theta_column: np.ndarray,
    qme_acceptance_tol: float,
    unit_root_tol: float,
    cayley_safe_min_spectral_margin: float,
    cayley_safe_min_rcond: float,
    cayley_safe_imag_tol: float,
) -> dict[str, Any]:
    parameter_values = _full_parameter_vector(model, parameter_names, theta_column)
    jacobian = model.calculate_jacobian(
        parameter_values=parameter_values,
        steady_state=steady_state,
    )
    _, a_plus, a_zero, a_minus = _prepare_first_order_qme_system(jacobian, model.timings)
    qz = sdsge.analyze_quadratic_matrix_equation_schur(
        a_plus,
        a_zero,
        a_minus,
        model.timings,
        acceptance_tol=qme_acceptance_tol,
    )
    real_qz = _analyze_qme_real_qz(
        np.asarray(a_plus),
        np.asarray(a_zero),
        np.asarray(a_minus),
        model.timings,
        acceptance_tol=qme_acceptance_tol,
    )
    cayley = sdsge.analyze_quadratic_matrix_equation_cayley_gpu_jax(
        a_plus,
        a_zero,
        a_minus,
        model.timings,
        acceptance_tol=qme_acceptance_tol,
        unit_root_tol=unit_root_tol,
    )
    doubling = sdsge.solve_quadratic_matrix_equation_doubling(
        a_plus,
        a_zero,
        a_minus,
        acceptance_tol=qme_acceptance_tol,
    )
    qz_unique = bool(qz.diagnostics.unique_stable_solution)
    real_qz_unique = bool(real_qz["unique"])
    cayley_success = bool(np.asarray(cayley.diagnostics.success))
    doubling_success = bool(np.asarray(doubling.converged))
    cayley_spectral_margin = float(np.asarray(cayley.diagnostics.spectral_margin))
    cayley_near_unit_count = int(np.asarray(cayley.diagnostics.near_unit_count))
    cayley_invariant_subspace_rcond = float(
        np.asarray(cayley.diagnostics.invariant_subspace_rcond)
    )
    cayley_solution_imag_max_abs = float(np.asarray(cayley.diagnostics.solution_imag_max_abs))
    cayley_relative_residual = float(np.asarray(cayley.diagnostics.relative_residual))
    cayley_solution = np.asarray(cayley.solution, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cayley_realpart_relative_residual = float(
                np.asarray(
                    quadratic_matrix_equation_residual(
                        np.asarray(a_plus),
                        cayley_solution,
                        np.asarray(a_zero),
                        np.asarray(a_minus),
                    )
                )
            )
    cayley_stable_count = int(np.asarray(cayley.diagnostics.stable_count))
    cayley_expected_stable_count = int(np.asarray(cayley.diagnostics.expected_stable_count))
    cayley_realpart_accept = bool(
        cayley_stable_count == cayley_expected_stable_count
        and np.all(np.isfinite(cayley_solution))
        and np.isfinite(cayley_realpart_relative_residual)
        and cayley_realpart_relative_residual < float(qme_acceptance_tol)
    )
    cayley_relaxed_accept = bool(
        cayley_realpart_accept
        and np.isfinite(cayley_solution_imag_max_abs)
        and cayley_solution_imag_max_abs <= 1.0e-6
    )
    cayley_safe_accept = bool(
        cayley_success
        and cayley_near_unit_count == 0
        and np.isfinite(cayley_spectral_margin)
        and np.isfinite(cayley_invariant_subspace_rcond)
        and np.isfinite(cayley_solution_imag_max_abs)
        and np.isfinite(cayley_relative_residual)
        and cayley_spectral_margin >= float(cayley_safe_min_spectral_margin)
        and cayley_invariant_subspace_rcond >= float(cayley_safe_min_rcond)
        and cayley_solution_imag_max_abs <= float(cayley_safe_imag_tol)
        and cayley_relative_residual < float(qme_acceptance_tol)
    )
    qz_solution = np.asarray(qz.solution, dtype=np.float64)
    real_qz_solution = np.asarray(real_qz["solution"], dtype=np.float64)
    doubling_solution = np.asarray(doubling.solution, dtype=np.float64)
    return {
        "qz_classification": str(qz.diagnostics.classification),
        "qz_unique": qz_unique,
        "qz_stable_count": int(qz.diagnostics.stable_count),
        "qz_expected_stable_count": int(qz.diagnostics.expected_stable_count),
        "qz_relative_residual": float(qz.diagnostics.relative_residual),
        "real_qz_classification": str(real_qz["classification"]),
        "real_qz_unique": real_qz_unique,
        "real_qz_stable_count": int(real_qz["stable_count"]),
        "real_qz_expected_stable_count": int(real_qz["expected_stable_count"]),
        "real_qz_decomposition_succeeded": bool(real_qz["decomposition_succeeded"]),
        "real_qz_invariant_subspace_invertible": bool(real_qz["invariant_subspace_invertible"]),
        "real_qz_solution_extracted": bool(real_qz["solution_extracted"]),
        "real_qz_invariant_subspace_rcond": float(real_qz["invariant_subspace_rcond"]),
        "real_qz_relative_residual": float(real_qz["relative_residual"]),
        "cayley_success": cayley_success,
        "cayley_stable_count": cayley_stable_count,
        "cayley_expected_stable_count": cayley_expected_stable_count,
        "cayley_spectral_margin": cayley_spectral_margin,
        "cayley_near_unit_count": cayley_near_unit_count,
        "cayley_invariant_subspace_rcond": cayley_invariant_subspace_rcond,
        "cayley_solution_imag_max_abs": cayley_solution_imag_max_abs,
        "cayley_relative_residual": cayley_relative_residual,
        "cayley_realpart_relative_residual": cayley_realpart_relative_residual,
        "cayley_realpart_accept": cayley_realpart_accept,
        "cayley_relaxed_accept": cayley_relaxed_accept,
        "cayley_safe_accept": cayley_safe_accept,
        "doubling_success": doubling_success,
        "doubling_relative_residual": float(np.asarray(doubling.relative_residual)),
        "qz_real_qz_disagree": qz_unique != real_qz_unique,
        "real_qz_accepts_complex_qz_reject": real_qz_unique and not qz_unique,
        "real_qz_rejects_complex_qz_unique": qz_unique and not real_qz_unique,
        "real_qz_cayley_disagree": real_qz_unique != cayley_success,
        "real_qz_cayley_safe_disagree": real_qz_unique != cayley_safe_accept,
        "real_qz_cayley_realpart_disagree": real_qz_unique != cayley_realpart_accept,
        "real_qz_cayley_relaxed_disagree": real_qz_unique != cayley_relaxed_accept,
        "qz_cayley_disagree": qz_unique != cayley_success,
        "qz_doubling_disagree": qz_unique != doubling_success,
        "doubling_accepts_non_unique": doubling_success and not qz_unique,
        "cayley_accepts_non_unique": cayley_success and not qz_unique,
        "cayley_rejects_qz_unique": qz_unique and not cayley_success,
        "cayley_safe_accepts_non_unique": cayley_safe_accept and not qz_unique,
        "cayley_safe_rejects_qz_unique": qz_unique and not cayley_safe_accept,
        "doubling_rejects_qz_unique": qz_unique and not doubling_success,
        "doubling_accepts_real_qz_non_unique": doubling_success and not real_qz_unique,
        "cayley_accepts_real_qz_non_unique": cayley_success and not real_qz_unique,
        "cayley_safe_accepts_real_qz_non_unique": cayley_safe_accept and not real_qz_unique,
        "cayley_realpart_accepts_real_qz_non_unique": cayley_realpart_accept and not real_qz_unique,
        "cayley_relaxed_accepts_real_qz_non_unique": cayley_relaxed_accept and not real_qz_unique,
        "cayley_rejects_real_qz_unique": real_qz_unique and not cayley_success,
        "cayley_safe_rejects_real_qz_unique": real_qz_unique and not cayley_safe_accept,
        "cayley_realpart_rejects_real_qz_unique": real_qz_unique and not cayley_realpart_accept,
        "cayley_relaxed_rejects_real_qz_unique": real_qz_unique and not cayley_relaxed_accept,
        "max_abs_qz_cayley_solution_diff": float(np.max(np.abs(qz_solution - cayley_solution)))
        if qz_solution.shape == cayley_solution.shape
        else None,
        "max_abs_real_qz_cayley_solution_diff": float(
            np.max(np.abs(real_qz_solution - cayley_solution))
        )
        if real_qz_solution.shape == cayley_solution.shape
        else None,
        "max_abs_qz_real_qz_solution_diff": float(np.max(np.abs(qz_solution - real_qz_solution)))
        if qz_solution.shape == real_qz_solution.shape
        else None,
        "max_abs_qz_doubling_solution_diff": float(np.max(np.abs(qz_solution - doubling_solution)))
        if qz_solution.shape == doubling_solution.shape
        else None,
    }


def _example_bucket_names(row: dict[str, Any]) -> tuple[str, ...]:
    buckets: list[str] = []
    if bool(row["real_qz_accepts_complex_qz_reject"]):
        buckets.append("real_qz_accepts_complex_qz_reject")
    if bool(row["real_qz_rejects_complex_qz_unique"]):
        buckets.append("real_qz_rejects_complex_qz_unique")
    if bool(row["cayley_accepts_non_unique"]):
        buckets.append("cayley_accepts_non_unique")
    if bool(row["cayley_accepts_real_qz_non_unique"]):
        buckets.append("cayley_accepts_real_qz_non_unique")
    if bool(row["cayley_rejects_qz_unique"]):
        buckets.append("cayley_rejects_qz_unique")
    if bool(row["cayley_rejects_real_qz_unique"]):
        buckets.append("cayley_rejects_real_qz_unique")
    if bool(row["cayley_safe_accepts_non_unique"]):
        buckets.append("cayley_safe_accepts_non_unique")
    if bool(row["cayley_safe_accepts_real_qz_non_unique"]):
        buckets.append("cayley_safe_accepts_real_qz_non_unique")
    if bool(row["cayley_realpart_accepts_real_qz_non_unique"]):
        buckets.append("cayley_realpart_accepts_real_qz_non_unique")
    if bool(row["cayley_relaxed_accepts_real_qz_non_unique"]):
        buckets.append("cayley_relaxed_accepts_real_qz_non_unique")
    if bool(row["cayley_safe_rejects_qz_unique"]):
        buckets.append("cayley_safe_rejects_qz_unique")
    if bool(row["cayley_safe_rejects_real_qz_unique"]):
        buckets.append("cayley_safe_rejects_real_qz_unique")
    if bool(row["cayley_realpart_rejects_real_qz_unique"]):
        buckets.append("cayley_realpart_rejects_real_qz_unique")
    if bool(row["cayley_relaxed_rejects_real_qz_unique"]):
        buckets.append("cayley_relaxed_rejects_real_qz_unique")
    if bool(row["doubling_accepts_non_unique"]):
        buckets.append("doubling_accepts_non_unique")
    if bool(row["doubling_accepts_real_qz_non_unique"]):
        buckets.append("doubling_accepts_real_qz_non_unique")
    if bool(row["doubling_rejects_qz_unique"]):
        buckets.append("doubling_rejects_qz_unique")
    if int(row["cayley_near_unit_count"]) > 0:
        buckets.append("cayley_near_unit")
    return tuple(buckets)


def _finite_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray(
        [float(row[key]) for row in rows if np.isfinite(float(row[key]))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"min": None, "p01": None, "median": None, "p99": None, "max": None}
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("sw07_hlt",), default="sw07_hlt")
    parser.add_argument("--model-source", type=Path, default=posterior_speed.DEFAULT_SW07_MODEL_SOURCE_PATH)
    parser.add_argument("--payload-path", type=Path, default=posterior_speed.DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--parameters", default="phase1_18params_narrow")
    parser.add_argument("--draw-mode", choices=("package-prior", "centered-uniform"), default="package-prior")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--prior-width-scale", type=float, default=0.01)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--qme-acceptance-tol", type=float, default=1.0e-8)
    parser.add_argument("--unit-root-tol", type=float, default=1.0e-6)
    parser.add_argument("--cayley-safe-min-spectral-margin", type=float, default=1.0e-6)
    parser.add_argument("--cayley-safe-min-rcond", type=float, default=2.0e-4)
    parser.add_argument("--cayley-safe-imag-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-examples-per-type", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results" / "qz_cayley_support_audit.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    case = _load_hlt_case(args.payload_path, args.case)
    model = sdsge.parse_macro_model(args.model_source.read_text())
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    parameter_names = _parameter_names(model, case, args.parameters)
    theta, lower, upper, support_label = _draw_theta(
        model=model,
        case=case,
        parameter_names=parameter_names,
        parameter_set=args.parameters,
        draw_mode=args.draw_mode,
        draws=int(args.draws),
        seed=int(args.seed),
        width_scale=float(args.prior_width_scale),
        width_floor=float(args.prior_width_floor),
        include_reference=bool(args.include_reference),
    )
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    examples_by_type: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    example_cap = int(args.max_examples_per_type or args.max_examples)
    for draw_idx in range(theta.shape[1]):
        if args.verbose:
            print(f"audit draw {draw_idx + 1}/{theta.shape[1]}", flush=True)
        theta_column = theta[:, draw_idx]
        try:
            row = _audit_draw(
                model=model,
                steady_state=steady_state,
                parameter_names=parameter_names,
                theta_column=theta_column,
                qme_acceptance_tol=float(args.qme_acceptance_tol),
                unit_root_tol=float(args.unit_root_tol),
                cayley_safe_min_spectral_margin=float(args.cayley_safe_min_spectral_margin),
                cayley_safe_min_rcond=float(args.cayley_safe_min_rcond),
                cayley_safe_imag_tol=float(args.cayley_safe_imag_tol),
            )
        except Exception as exc:
            errors.append(
                {
                    "draw_index": draw_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                    "theta": {name: float(value) for name, value in zip(parameter_names, theta_column)},
                }
            )
            continue
        row["draw_index"] = draw_idx
        row["theta"] = {name: float(value) for name, value in zip(parameter_names, theta_column)}
        rows.append(row)
        bucket_names = _example_bucket_names(row)
        for bucket_name in bucket_names:
            bucket = examples_by_type.setdefault(bucket_name, [])
            if len(bucket) < example_cap:
                bucket.append(row)
        if bucket_names and len(examples) < int(args.max_examples):
            examples.append(row)

    qz_counts: dict[str, int] = {}
    real_qz_counts: dict[str, int] = {}
    for row in rows:
        cls = str(row["qz_classification"])
        qz_counts[cls] = qz_counts.get(cls, 0) + 1
        real_cls = str(row["real_qz_classification"])
        real_qz_counts[real_cls] = real_qz_counts.get(real_cls, 0) + 1
    audited = len(rows)
    payload = {
        "status": "ok" if not errors else "partial",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "elapsed_s": float(time.perf_counter() - started),
        "case": args.case,
        "model_source": args.model_source,
        "payload_path": args.payload_path,
        "parameters": tuple(parameter_names),
        "support_label": support_label,
        "draw_mode": args.draw_mode,
        "qme_acceptance_tol": float(args.qme_acceptance_tol),
        "unit_root_tol": float(args.unit_root_tol),
        "cayley_safe_min_spectral_margin": float(args.cayley_safe_min_spectral_margin),
        "cayley_safe_min_rcond": float(args.cayley_safe_min_rcond),
        "cayley_safe_imag_tol": float(args.cayley_safe_imag_tol),
        "requested_draws": int(args.draws),
        "audited_draws": audited,
        "error_count": len(errors),
        "qz_classification_counts": qz_counts,
        "real_qz_classification_counts": real_qz_counts,
        "qz_unique_count": int(sum(bool(row["qz_unique"]) for row in rows)),
        "real_qz_unique_count": int(sum(bool(row["real_qz_unique"]) for row in rows)),
        "cayley_success_count": int(sum(bool(row["cayley_success"]) for row in rows)),
        "cayley_realpart_accept_count": int(
            sum(bool(row["cayley_realpart_accept"]) for row in rows)
        ),
        "cayley_relaxed_accept_count": int(sum(bool(row["cayley_relaxed_accept"]) for row in rows)),
        "cayley_safe_accept_count": int(sum(bool(row["cayley_safe_accept"]) for row in rows)),
        "doubling_success_count": int(sum(bool(row["doubling_success"]) for row in rows)),
        "qz_real_qz_disagreement_count": int(sum(bool(row["qz_real_qz_disagree"]) for row in rows)),
        "real_qz_accepts_complex_qz_reject_count": int(
            sum(bool(row["real_qz_accepts_complex_qz_reject"]) for row in rows)
        ),
        "real_qz_rejects_complex_qz_unique_count": int(
            sum(bool(row["real_qz_rejects_complex_qz_unique"]) for row in rows)
        ),
        "real_qz_cayley_disagreement_count": int(
            sum(bool(row["real_qz_cayley_disagree"]) for row in rows)
        ),
        "real_qz_cayley_safe_disagreement_count": int(
            sum(bool(row["real_qz_cayley_safe_disagree"]) for row in rows)
        ),
        "real_qz_cayley_realpart_disagreement_count": int(
            sum(bool(row["real_qz_cayley_realpart_disagree"]) for row in rows)
        ),
        "real_qz_cayley_relaxed_disagreement_count": int(
            sum(bool(row["real_qz_cayley_relaxed_disagree"]) for row in rows)
        ),
        "qz_cayley_disagreement_count": int(sum(bool(row["qz_cayley_disagree"]) for row in rows)),
        "qz_cayley_safe_disagreement_count": int(
            sum(bool(row["qz_unique"]) != bool(row["cayley_safe_accept"]) for row in rows)
        ),
        "qz_doubling_disagreement_count": int(sum(bool(row["qz_doubling_disagree"]) for row in rows)),
        "doubling_accepts_non_unique_count": int(
            sum(bool(row["doubling_accepts_non_unique"]) for row in rows)
        ),
        "doubling_rejects_qz_unique_count": int(
            sum(bool(row["doubling_rejects_qz_unique"]) for row in rows)
        ),
        "doubling_accepts_real_qz_non_unique_count": int(
            sum(bool(row["doubling_accepts_real_qz_non_unique"]) for row in rows)
        ),
        "cayley_accepts_non_unique_count": int(
            sum(bool(row["cayley_accepts_non_unique"]) for row in rows)
        ),
        "cayley_accepts_real_qz_non_unique_count": int(
            sum(bool(row["cayley_accepts_real_qz_non_unique"]) for row in rows)
        ),
        "cayley_rejects_qz_unique_count": int(
            sum(bool(row["cayley_rejects_qz_unique"]) for row in rows)
        ),
        "cayley_rejects_real_qz_unique_count": int(
            sum(bool(row["cayley_rejects_real_qz_unique"]) for row in rows)
        ),
        "cayley_safe_accepts_non_unique_count": int(
            sum(bool(row["cayley_safe_accepts_non_unique"]) for row in rows)
        ),
        "cayley_safe_accepts_real_qz_non_unique_count": int(
            sum(bool(row["cayley_safe_accepts_real_qz_non_unique"]) for row in rows)
        ),
        "cayley_realpart_accepts_real_qz_non_unique_count": int(
            sum(bool(row["cayley_realpart_accepts_real_qz_non_unique"]) for row in rows)
        ),
        "cayley_relaxed_accepts_real_qz_non_unique_count": int(
            sum(bool(row["cayley_relaxed_accepts_real_qz_non_unique"]) for row in rows)
        ),
        "cayley_safe_rejects_qz_unique_count": int(
            sum(bool(row["cayley_safe_rejects_qz_unique"]) for row in rows)
        ),
        "cayley_safe_rejects_real_qz_unique_count": int(
            sum(bool(row["cayley_safe_rejects_real_qz_unique"]) for row in rows)
        ),
        "cayley_realpart_rejects_real_qz_unique_count": int(
            sum(bool(row["cayley_realpart_rejects_real_qz_unique"]) for row in rows)
        ),
        "cayley_relaxed_rejects_real_qz_unique_count": int(
            sum(bool(row["cayley_relaxed_rejects_real_qz_unique"]) for row in rows)
        ),
        "near_unit_count": int(sum(int(row["cayley_near_unit_count"]) > 0 for row in rows)),
        "cayley_spectral_margin_summary": _finite_summary(rows, "cayley_spectral_margin"),
        "cayley_invariant_subspace_rcond_summary": _finite_summary(
            rows,
            "cayley_invariant_subspace_rcond",
        ),
        "cayley_solution_imag_max_abs_summary": _finite_summary(
            rows,
            "cayley_solution_imag_max_abs",
        ),
        "cayley_realpart_relative_residual_summary": _finite_summary(
            rows,
            "cayley_realpart_relative_residual",
        ),
        "real_qz_invariant_subspace_rcond_summary": _finite_summary(
            rows,
            "real_qz_invariant_subspace_rcond",
        ),
        "real_qz_relative_residual_summary": _finite_summary(rows, "real_qz_relative_residual"),
        "lower": lower,
        "upper": upper,
        "examples": examples,
        "examples_by_type": examples_by_type,
        "errors": errors[: int(args.max_examples)],
    }
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
