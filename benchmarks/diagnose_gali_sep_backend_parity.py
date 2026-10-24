#!/usr/bin/env python3
"""Diagnose sequential-versus-batched SEP parity on the full nonlinear Galí model.

Run this program in separate processes for CPU and GPU backends.  It keeps the
model, parameter design, shock, horizon, tolerances, and precision fixed and
records enough solver state to distinguish an algorithm mismatch from a JAX
backend numerical difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks import profile_surrogate_pipeline_gpu as profile  # noqa: E402
from surrogatenn_dsge import (  # noqa: E402
    SEPConfig,
    parse_macro_model,
    solve_batched_stochastic_extended_path_model,
    solve_stochastic_extended_path_model,
)


MODEL_PATH = ROOT / "benchmarks" / "model_sources" / "Gali_2015_chapter_3_nonlinear.jl"
PAYLOAD_PATH = ROOT / "benchmarks" / "gali_nonlinear_pilot_payload.json"
CASE_NAME = "gali_full_nonlinear_reduced_pilot"
TERMINATION_LABELS = {
    0: "max_iterations",
    1: "converged",
    2: "line_search_failed",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(message: str) -> None:
    print(f"[gali-sep-parity] {_utc_now()} {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _block_tree(value: Any) -> None:
    jax.tree_util.tree_map(
        lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item,
        value,
    )


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": repr(exc)}


def _parse_indices(raw: str, *, upper_bound: int) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(part.strip()) for part in raw.split(",") if part.strip()))
    if not values:
        raise ValueError("--theta-indices must contain at least one index.")
    invalid = [value for value in values if value < 0 or value >= upper_bound]
    if invalid:
        raise ValueError(
            f"theta indices {invalid} fall outside the generated design [0, {upper_bound - 1}]."
        )
    return values


def _base_solution_summary(solution: Any, elapsed_s: float) -> dict[str, Any]:
    stacked = np.asarray(solution.stacked_states, dtype=np.float64)
    mean_path = np.asarray(solution.mean_path, dtype=np.float64)
    return {
        "elapsed_s": float(elapsed_s),
        "residual_norm": _jsonable(np.asarray(solution.residual_norm, dtype=np.float64)),
        "accepted": _jsonable(np.asarray(solution.accepted, dtype=bool)),
        "converged": _jsonable(np.asarray(solution.converged, dtype=bool)),
        "iterations": _jsonable(np.asarray(solution.iterations, dtype=np.int64)),
        "group_counts": [int(value) for value in solution.group_counts],
        "stacked_states_shape": list(stacked.shape),
        "stacked_states_sha256": _sha256_array(stacked),
        "stacked_states_finite": bool(np.isfinite(stacked).all()),
        "mean_path_shape": list(mean_path.shape),
        "mean_path_sha256": _sha256_array(mean_path),
        "mean_path_finite": bool(np.isfinite(mean_path).all()),
    }


def _run_sequential(
    *,
    model: Any,
    parameters: np.ndarray,
    steady_state: np.ndarray,
    deterministic: np.ndarray,
    initial_guess: np.ndarray,
    config: SEPConfig,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    with jax.default_device(device):
        result = solve_stochastic_extended_path_model(
            model,
            parameter_values=parameters,
            steady_state=steady_state,
            initial_state=steady_state,
            terminal_state=steady_state,
            deterministic_shocks=deterministic,
            initial_guess=initial_guess,
            config=config,
        )
    _block_tree(result.solution)
    elapsed = time.perf_counter() - started
    summary = _base_solution_summary(result.solution, elapsed)
    return (
        summary,
        np.asarray(result.solution.stacked_states, dtype=np.float64),
        np.asarray(result.solution.mean_path, dtype=np.float64),
    )


def _run_batched(
    *,
    model: Any,
    parameters: np.ndarray,
    steady_state: np.ndarray,
    deterministic: np.ndarray,
    initial_guess: np.ndarray,
    config: SEPConfig,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    with jax.default_device(device):
        result = solve_batched_stochastic_extended_path_model(
            model,
            parameter_values=parameters[None, :],
            steady_state=steady_state[None, :],
            initial_state=steady_state[None, :],
            terminal_state=steady_state[None, :],
            deterministic_shocks=deterministic[None, :, :],
            initial_guess=initial_guess[None, ...],
            config=config,
        )
    _block_tree(result.solution)
    elapsed = time.perf_counter() - started
    summary = _base_solution_summary(result.solution, elapsed)
    initial_residual = result.solution.initial_residual_norm
    final_lambda = result.solution.final_lambda
    termination_code = result.solution.termination_code
    summary.update(
        {
            "initial_residual_norm": None
            if initial_residual is None
            else _jsonable(np.asarray(initial_residual, dtype=np.float64)),
            "final_lambda": None
            if final_lambda is None
            else _jsonable(np.asarray(final_lambda, dtype=np.float64)),
            "termination_code": None
            if termination_code is None
            else _jsonable(np.asarray(termination_code, dtype=np.int8)),
            "termination": None
            if termination_code is None
            else [
                TERMINATION_LABELS.get(int(code), "unknown")
                for code in np.asarray(termination_code, dtype=np.int8)
            ],
        }
    )
    return (
        summary,
        np.asarray(result.solution.stacked_states[0], dtype=np.float64),
        np.asarray(result.solution.mean_path[0], dtype=np.float64),
    )


def _difference_summary(
    sequential_stacked: np.ndarray,
    batched_stacked: np.ndarray,
    sequential_mean_path: np.ndarray,
    batched_mean_path: np.ndarray,
) -> dict[str, Any]:
    stacked_delta = np.asarray(batched_stacked - sequential_stacked, dtype=np.float64)
    mean_path_delta = np.asarray(batched_mean_path - sequential_mean_path, dtype=np.float64)
    return {
        "stacked_states_max_abs": float(np.max(np.abs(stacked_delta))),
        "stacked_states_l2": float(np.linalg.norm(stacked_delta)),
        "mean_path_max_abs": float(np.max(np.abs(mean_path_delta))),
        "mean_path_l2": float(np.linalg.norm(mean_path_delta)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    devices = jax.devices() if args.device == "auto" else jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No JAX devices are available for {args.device!r}.")
    device = devices[0]
    model = parse_macro_model(MODEL_PATH.read_text(encoding="utf-8"))
    case = profile._load_hlt_payload_case(
        Namespace(hlt_payload=PAYLOAD_PATH, hlt_case_name=CASE_NAME)
    )
    profile._validate_gali_standardized_innovations(model, case)
    base_parameters = np.asarray(model.parameter_values, dtype=np.float64)
    subset_names = profile._select_hlt_parameter_subset(model, case, "payload")
    theta, subset_indices, theta_design = profile._make_hlt_theta_design(
        base_parameters=base_parameters,
        parameter_names=model.parameter_names,
        subset_names=subset_names,
        draws=args.theta_draws,
        perturbation=args.parameter_perturbation,
        design="perturbation",
        seed=args.seed,
        include_reference=True,
    )
    selected_indices = _parse_indices(args.theta_indices, upper_bound=theta.shape[1])
    config = SEPConfig(
        periods=args.periods,
        branching_order=args.order,
        nnodes=args.nnodes,
        shock_scale=args.stochastic_shock_scale,
        sparse_tree=args.sparse_tree,
        max_iter=args.max_iter,
        tol=args.tol,
        accept_tol=args.accept_tol,
        linear_solver=args.linear_solver,
        line_search=True,
    )
    output: dict[str, Any] = {
        "started_utc": _utc_now(),
        "status": "running",
        "backend": jax.default_backend(),
        "selected_device": str(device),
        "devices": [str(item) for item in jax.devices()],
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "git": _git_metadata(),
        "model": {
            "name": model.name,
            "variables": int(model.timings.nVars),
            "shocks": int(model.timings.nExo),
            "parameters": len(model.parameter_names),
            "source_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        },
        "theta_design": theta_design,
        "theta_indices": list(selected_indices),
        "theta_values": theta[:, selected_indices],
        "config": {
            "solver": args.solver,
            "periods": args.periods,
            "branching_order": args.order,
            "nnodes": args.nnodes,
            "sparse_tree": args.sparse_tree,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "accept_tol": args.accept_tol,
            "linear_solver": args.linear_solver,
            "deterministic_first_shock": args.deterministic_shock,
            "stochastic_shock_scale": args.stochastic_shock_scale,
        },
        "draws": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_partial() -> None:
        output["last_update_utc"] = _utc_now()
        args.output.write_text(
            json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_partial()
    reference_steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    for theta_index in selected_indices:
        parameters = base_parameters.copy()
        parameters[subset_indices] = theta[:, theta_index]
        _log(f"theta={theta_index} solving steady state")
        steady_started = time.perf_counter()
        steady_result = model.solve_steady_state(
            parameter_values=parameters,
            initial_guess=reference_steady_state,
            tol=1e-10,
            max_iter=100,
        )
        steady_state = np.asarray(steady_result.steady_state, dtype=np.float64)
        draw: dict[str, Any] = {
            "theta_index": theta_index,
            "theta": theta[:, theta_index],
            "steady_state": {
                "converged": bool(steady_result.converged),
                "iterations": int(steady_result.iterations),
                "residual_norm": float(steady_result.residual_norm),
                "elapsed_s": time.perf_counter() - steady_started,
            },
        }
        output["draws"].append(draw)
        write_partial()
        if not bool(steady_result.converged):
            draw["status"] = "steady_state_failed"
            write_partial()
            continue

        deterministic = np.zeros((config.periods, model.timings.nExo), dtype=np.float64)
        deterministic[0, 0] = args.deterministic_shock
        initial_guess = model._build_sep_linear_initial_guess(
            parameter_values=parameters,
            steady_state=steady_state,
            initial_state=steady_state,
            deterministic_shocks=deterministic,
            config=config,
        )
        if initial_guess is None:
            raise RuntimeError("The shared first-order SEP initial guess could not be constructed.")
        initial_guess = np.asarray(initial_guess, dtype=np.float64)
        draw["shared_initial_guess"] = {
            "shape": list(initial_guess.shape),
            "sha256": _sha256_array(initial_guess),
            "finite": bool(np.isfinite(initial_guess).all()),
        }
        write_partial()
        sequential_arrays: tuple[np.ndarray, np.ndarray] | None = None
        batched_arrays: tuple[np.ndarray, np.ndarray] | None = None
        if args.solver in {"sequential", "both"}:
            _log(f"theta={theta_index} starting sequential SEP")
            summary, stacked, mean_path = _run_sequential(
                model=model,
                parameters=parameters,
                steady_state=steady_state,
                deterministic=deterministic,
                initial_guess=initial_guess,
                config=config,
                device=device,
            )
            draw["sequential"] = summary
            sequential_arrays = (stacked, mean_path)
            _log(
                f"theta={theta_index} sequential residual={summary['residual_norm']} "
                f"accepted={summary['accepted']} elapsed_s={summary['elapsed_s']:.3f}"
            )
            write_partial()
        if args.solver in {"batched", "both"}:
            _log(f"theta={theta_index} starting batched SEP with batch_size=1")
            summary, stacked, mean_path = _run_batched(
                model=model,
                parameters=parameters,
                steady_state=steady_state,
                deterministic=deterministic,
                initial_guess=initial_guess,
                config=config,
                device=device,
            )
            draw["batched"] = summary
            batched_arrays = (stacked, mean_path)
            _log(
                f"theta={theta_index} batched residual={summary['residual_norm']} "
                f"accepted={summary['accepted']} termination={summary['termination']} "
                f"elapsed_s={summary['elapsed_s']:.3f}"
            )
            write_partial()
        if sequential_arrays is not None and batched_arrays is not None:
            draw["sequential_batched_difference"] = _difference_summary(
                sequential_arrays[0],
                batched_arrays[0],
                sequential_arrays[1],
                batched_arrays[1],
            )
            write_partial()

    output["status"] = "complete"
    output["finished_utc"] = _utc_now()
    write_partial()
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--solver", choices=("sequential", "batched", "both"), default="both")
    parser.add_argument("--theta-draws", type=int, default=8)
    parser.add_argument("--theta-indices", default="0")
    parser.add_argument("--parameter-perturbation", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--periods", type=int, default=10)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=3)
    parser.add_argument("--sparse-tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--accept-tol", type=float, default=1e-5)
    parser.add_argument("--linear-solver", choices=("qr", "normal_equations"), default="qr")
    parser.add_argument("--deterministic-shock", type=float, default=0.12)
    parser.add_argument("--stochastic-shock-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmarks"
        / "results"
        / f"gali_sep_backend_parity_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
