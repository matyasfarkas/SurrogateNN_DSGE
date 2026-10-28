#!/usr/bin/env python3
"""Validate one-transition full-Gali nonlinear SEP likelihood parity.

The gate generates one observation from the complete nonlinear Galí model and
then independently reconstructs the inversion likelihood in Python and in a
clean, pinned MacroModelling.jl checkout.  It compares the recovered shocks,
finite-difference Jacobian, residual, and scalar likelihood.  This is a scoped
algorithmic gate; it is not a production-horizon likelihood validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogatenn_dsge import (  # noqa: E402
    SEPConfig,
    get_sep_inversion_last_diagnostics,
    parse_macro_model,
    reset_sep_inversion_last_diagnostics,
)
from surrogatenn_dsge import inversion as inversion_module  # noqa: E402


SCHEMA_VERSION = "surrogatenn-dsge-gali-sep-likelihood-parity/v1"
JULIA_SCHEMA_VERSION = (
    "surrogatenn-dsge-gali-julia-sep-likelihood-export/v1"
)
MODEL_SYMBOL = "Gali_2015_chapter_3_nonlinear"
MODEL_PATH = ROOT / "benchmarks" / "model_sources" / f"{MODEL_SYMBOL}.jl"
PAYLOAD_PATH = ROOT / "benchmarks" / "gali_nonlinear_pilot_payload.json"
JULIA_EXPORTER = (
    ROOT / "benchmarks" / "export_gali_full_julia_sep_likelihood.jl"
)
DEFAULT_UPSTREAM_REPO = ROOT.parent / "MacroModelling.jl"
PARAMETERS = {"ϕᵖⁱ": 1.5, "ϕʸ": 0.125}
OBSERVABLE_NAMES = ("log_y", "pi_ann", "i_ann")
KNOWN_SHOCKS = np.asarray((0.02, -0.01, 0.015), dtype=np.float64)

STEADY_STATE_MAX_ABS_TOL = 1.0e-10
PUBLIC_AUDIT_ABS_TOL = 1.0e-8
RECOVERED_SHOCK_MAX_ABS_TOL = 1.0e-7
KNOWN_SHOCK_MAX_ABS_TOL = 1.0e-7
JACOBIAN_MAX_ABS_TOL = 1.0e-5
LOGLIKELIHOOD_ABS_TOL = 1.0e-4
INVERSION_RESIDUAL_TOL = 1.0e-9
SOLVER_RESIDUAL_TOL = 1.0e-12


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(argv: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        list(argv), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_state(repo: Path) -> dict[str, Any]:
    status = _run_text(("git", "status", "--short"), cwd=repo).splitlines()
    return {
        "path": str(repo.resolve()),
        "commit": _run_text(("git", "rev-parse", "HEAD"), cwd=repo),
        "dirty": bool(status),
        "status_short": status,
    }


def _load_reference_steady_state() -> np.ndarray:
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    matches = [
        case
        for case in payload.get("cases", [])
        if case.get("model_symbol") == MODEL_SYMBOL
    ]
    if len(matches) != 1:
        raise ValueError("Could not identify exactly one full-Gali payload case.")
    return np.asarray(matches[0]["reference_steady_state"], dtype=np.float64)


def _inject_parameters(
    base_values: Sequence[float],
    names: Sequence[str],
    overrides: Mapping[str, float],
) -> np.ndarray:
    values = np.asarray(base_values, dtype=np.float64).copy()
    index = {str(name): position for position, name in enumerate(names)}
    missing = sorted(set(overrides) - set(index))
    if missing:
        raise ValueError(f"Unknown parameter override(s): {missing}")
    for name, raw_value in overrides.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Parameter {name} is not finite.")
        values[index[name]] = value
    return values


def _difference_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {left.shape} and {right.shape}.")
    delta = np.asarray(left - right, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _config() -> SEPConfig:
    return SEPConfig(
        periods=5,
        branching_order=1,
        nnodes=3,
        sparse_tree=True,
        max_iter=120,
        tol=SOLVER_RESIDUAL_TOL,
        accept_tol=SOLVER_RESIDUAL_TOL,
        linear_solver="qr",
        fallback_solver=None,
        stall_iters=25,
        stall_rel_tol=1.0e-4,
        stall_abs_tol=1.0e-10,
        line_search=True,
        line_search_norm="inf",
    )


def _config_json(config: SEPConfig) -> dict[str, Any]:
    return {
        "periods": int(config.periods),
        "branching_order": int(config.branching_order),
        "nnodes": int(config.nnodes),
        "sparse_tree": bool(config.sparse_tree),
        "max_iter": int(config.max_iter),
        "tol": float(config.tol),
        "accept_tol": SOLVER_RESIDUAL_TOL,
        "linear_solver": str(config.linear_solver),
        "fallback_solver": config.fallback_solver,
        "stall_iters": int(config.stall_iters),
        "stall_rel_tol": float(config.stall_rel_tol),
        "stall_abs_tol": float(config.stall_abs_tol),
    }


def _inversion_json() -> dict[str, Any]:
    return {
        "max_iter": 12,
        "step_tol": 1.0e-10,
        "residual_tol": INVERSION_RESIDUAL_TOL,
        "lambda": 1.0e-8,
        "finite_difference_step_rule": "max(cbrt(eps)*max(1,abs(shock),sigma),1e-6)",
    }


def _python_audit_inversion(
    model: Any,
    *,
    observation_deviation: np.ndarray,
    observable_indices: Sequence[int],
    parameter_values: np.ndarray,
    steady_state: np.ndarray,
    config: SEPConfig,
    shock_sigmas: np.ndarray,
) -> dict[str, Any]:
    inversion = _inversion_json()
    structural_indices = np.flatnonzero(shock_sigmas > 0.0)
    if structural_indices.size != model.timings.nExo:
        raise ValueError("The parity adapter requires every model shock to be structural.")
    eps_structural = np.zeros((structural_indices.size,), dtype=np.float64)
    state_deviation = np.zeros_like(steady_state)
    base_guess = None
    final_eval = None
    final_jacobian = None
    final_evaluated_shocks = None
    iteration_trace: list[dict[str, Any]] = []
    total_predict_calls = 0
    stop_reason = "max_iterations"

    for iteration in range(1, int(inversion["max_iter"]) + 1):
        evaluated_shocks = eps_structural.copy()
        ok, jacobian, base_eval, predict_calls = inversion_module._sep_fd_jacobian(
            model,
            state_deviation,
            eps_structural,
            structural_indices,
            observable_indices,
            parameter_values=parameter_values,
            steady_state=steady_state,
            terminal_state=steady_state,
            config=config,
            shock_sigmas=shock_sigmas,
            sep_accept_tol=SOLVER_RESIDUAL_TOL,
            initial_guess=base_guess,
        )
        total_predict_calls += int(predict_calls)
        if not ok:
            raise RuntimeError(
                f"Python finite-difference SEP prediction failed at iteration {iteration}."
            )
        residual = observation_deviation - base_eval.obs_dev
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise RuntimeError("Python inversion produced a non-finite value.")
        final_eval = base_eval
        final_jacobian = np.asarray(jacobian, dtype=np.float64).copy()
        final_evaluated_shocks = evaluated_shocks
        base_guess = base_eval.solution_guess
        residual_norm = float(np.linalg.norm(residual))
        record: dict[str, Any] = {
            "iteration": iteration,
            "evaluated_shocks": evaluated_shocks.tolist(),
            "predicted_observation_deviation": base_eval.obs_dev.tolist(),
            "residual": residual.tolist(),
            "residual_norm": residual_norm,
            "jacobian": np.asarray(jacobian).tolist(),
            "predict_calls": int(predict_calls),
            "base_sep_flag": int(base_eval.sep_flag),
            "base_sep_error": float(base_eval.sep_err),
        }
        if residual_norm <= float(inversion["residual_tol"]):
            stop_reason = "residual_tolerance"
            record["stop_reason"] = stop_reason
            iteration_trace.append(record)
            break

        lhs = jacobian.T @ jacobian
        lhs = lhs + float(inversion["lambda"]) * np.eye(
            structural_indices.size, dtype=np.float64
        )
        rhs = jacobian.T @ residual
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(jacobian) @ residual
        if not np.isfinite(step).all():
            raise RuntimeError("Python inversion step is non-finite.")
        eps_structural = eps_structural + step
        step_norm = float(np.linalg.norm(step))
        record["step"] = step.tolist()
        record["step_norm"] = step_norm
        if step_norm <= float(inversion["step_tol"]) * (
            1.0 + float(np.linalg.norm(eps_structural))
        ):
            stop_reason = "step_tolerance"
            record["stop_reason"] = stop_reason
            iteration_trace.append(record)
            break
        iteration_trace.append(record)

    if final_eval is None or final_jacobian is None:
        raise RuntimeError("Python inversion did not produce a final evaluation.")
    residual = observation_deviation - final_eval.obs_dev
    standardized_jacobian = (
        final_jacobian * shock_sigmas[structural_indices][None, :]
    )
    logabsdet = inversion_module._pseudo_logabsdet_numpy(standardized_jacobian)
    standardized_shocks = eps_structural / shock_sigmas[structural_indices]
    shocks2 = float(np.sum(standardized_shocks**2))
    loglikelihood = -0.5 * (
        logabsdet
        + shocks2
        + len(observable_indices) * float(np.log(2.0 * np.pi))
    )
    return {
        "status": "ok",
        "recovered_shocks": eps_structural.tolist(),
        "final_evaluated_shocks": np.asarray(final_evaluated_shocks).tolist(),
        "predicted_observation_deviation": final_eval.obs_dev.tolist(),
        "residual": residual.tolist(),
        "residual_norm": float(np.linalg.norm(residual)),
        "jacobian": final_jacobian.tolist(),
        "standardized_jacobian": standardized_jacobian.tolist(),
        "logabsdet": float(logabsdet),
        "shocks2": shocks2,
        "loglikelihood": float(loglikelihood),
        "iterations": len(iteration_trace),
        "predict_calls": total_predict_calls,
        "stop_reason": stop_reason,
        "final_sep_convergence_flag": int(final_eval.sep_flag),
        "final_sep_error": float(final_eval.sep_err),
        "iteration_trace": iteration_trace,
    }


def _solve_python() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model = parse_macro_model(MODEL_PATH.read_text(encoding="utf-8"))
    parameters = _inject_parameters(
        model.parameter_values, model.parameter_names, PARAMETERS
    )
    steady_result = model.solve_steady_state(
        parameter_values=parameters,
        initial_guess=_load_reference_steady_state(),
        tol=1.0e-12,
        max_iter=100,
    )
    steady_state = np.asarray(steady_result.steady_state, dtype=np.float64)
    config = _config()
    observable_indices = tuple(model.resolve_observable_indices(OBSERVABLE_NAMES))
    generation = inversion_module._sep_predict_step(
        model,
        np.zeros_like(steady_state),
        KNOWN_SHOCKS,
        observable_indices,
        parameter_values=parameters,
        steady_state=steady_state,
        terminal_state=steady_state,
        config=config,
        sep_accept_tol=SOLVER_RESIDUAL_TOL,
        initial_guess=None,
    )
    if not generation.ok:
        raise RuntimeError(
            "Python could not generate the parity observation: "
            f"flag={generation.sep_flag}, error={generation.sep_err}."
        )
    observation_deviation = np.asarray(generation.obs_dev, dtype=np.float64)
    observation_levels = steady_state[list(observable_indices)] + observation_deviation
    shock_sigmas = np.asarray(
        inversion_module._sep_inversion_shock_sigmas(model, parameters),
        dtype=np.float64,
    )
    audit = _python_audit_inversion(
        model,
        observation_deviation=observation_deviation,
        observable_indices=observable_indices,
        parameter_values=parameters,
        steady_state=steady_state,
        config=config,
        shock_sigmas=shock_sigmas,
    )

    reset_sep_inversion_last_diagnostics()
    public_values = np.asarray(
        model.inversion_loglikelihood_per_period(
            observation_levels[:, None],
            observables=OBSERVABLE_NAMES,
            algorithm="stochastic_extended_path",
            parameter_values=parameters,
            steady_state=steady_state,
            initial_state=steady_state,
            terminal_state=steady_state,
            config=config,
            sep_accept_tol=SOLVER_RESIDUAL_TOL,
            sep_inv_maxit=int(_inversion_json()["max_iter"]),
            sep_inv_step_tol=float(_inversion_json()["step_tol"]),
            sep_inv_resid_tol=float(_inversion_json()["residual_tol"]),
            sep_inv_lambda=float(_inversion_json()["lambda"]),
            on_failure_loglikelihood=-np.inf,
        ),
        dtype=np.float64,
    )
    public_diagnostics = get_sep_inversion_last_diagnostics()
    if public_values.shape != (1,) or not np.isfinite(public_values).all():
        raise RuntimeError("The public Python SEP likelihood call failed.")
    public_audit_abs_diff = abs(float(public_values[0]) - audit["loglikelihood"])
    python = {
        "status": "ok",
        "parameter_values": parameters.tolist(),
        "steady_state": steady_state.tolist(),
        "steady_state_residual_norm": float(steady_result.residual_norm),
        "observable_indices_zero_based": list(observable_indices),
        "observation_levels": observation_levels.tolist(),
        "observation_deviation": observation_deviation.tolist(),
        "known_generation_shocks": KNOWN_SHOCKS.tolist(),
        "generation_sep_flag": int(generation.sep_flag),
        "generation_sep_error": float(generation.sep_err),
        "shock_sigmas": shock_sigmas.tolist(),
        "public_loglikelihood_per_period": public_values.tolist(),
        "public_audit_abs_diff": public_audit_abs_diff,
        "public_diagnostics": public_diagnostics,
        "audit": audit,
    }
    spec = {
        "model_symbol": MODEL_SYMBOL,
        "parameters": PARAMETERS,
        "observable_names": list(OBSERVABLE_NAMES),
        "structural_shocks": [str(name) for name in model.timings.exo],
        "shock_sigmas": shock_sigmas.tolist(),
        "observation_levels": observation_levels.tolist(),
        "known_generation_shocks": KNOWN_SHOCKS.tolist(),
        "config": _config_json(config),
        "inversion": _inversion_json(),
    }
    return model, python, spec


def _run_julia_export(
    *, julia: str, upstream_repo: Path, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gali-sep-likelihood-parity-") as name:
        temp_dir = Path(name)
        spec_path = temp_dir / "spec.json"
        output_path = temp_dir / "julia.json"
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        argv = [
            julia,
            f"--project={upstream_repo}",
            str(JULIA_EXPORTER),
            str(MODEL_PATH),
            str(spec_path),
            str(output_path),
        ]
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        invocation = {
            "argv": argv,
            "shell_escaped": shlex.join(argv),
            "cwd": str(ROOT),
            "elapsed_s": time.perf_counter() - started,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if not output_path.is_file():
            raise RuntimeError(
                "Julia exporter did not create output. "
                f"Exit {completed.returncode}; stderr: {completed.stderr[-4000:]}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0:
            raise RuntimeError(
                "Julia exporter failed. "
                f"Exit {completed.returncode}; result: {payload.get('result')}"
            )
        return payload, invocation


def _align_julia_steady_state(
    values: Sequence[float],
    *,
    julia_names: Sequence[str],
    python_names: Sequence[str],
) -> np.ndarray:
    index = {str(name): position for position, name in enumerate(julia_names)}
    if set(index) != set(python_names):
        raise ValueError("Julia and Python variable-name sets differ.")
    return np.asarray(values, dtype=np.float64)[
        [index[str(name)] for name in python_names]
    ]


def _compare(
    *, model: Any, python: Mapping[str, Any], julia_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    julia = julia_payload["result"]
    if julia.get("status") != "ok":
        raise ValueError(f"Julia result status is {julia.get('status')!r}.")
    python_audit = python["audit"]
    julia_steady = _align_julia_steady_state(
        julia["steady_state"],
        julia_names=julia_payload["model"]["variable_names"],
        python_names=model.timings.var,
    )
    steady = _difference_metrics(
        np.asarray(python["steady_state"], dtype=np.float64), julia_steady
    )
    recovered = _difference_metrics(
        np.asarray(python_audit["recovered_shocks"], dtype=np.float64),
        np.asarray(julia["recovered_shocks"], dtype=np.float64),
    )
    python_known = _difference_metrics(
        np.asarray(python_audit["recovered_shocks"], dtype=np.float64),
        KNOWN_SHOCKS,
    )
    julia_known = _difference_metrics(
        np.asarray(julia["recovered_shocks"], dtype=np.float64), KNOWN_SHOCKS
    )
    jacobian = _difference_metrics(
        np.asarray(python_audit["jacobian"], dtype=np.float64),
        np.asarray(julia["jacobian"], dtype=np.float64),
    )
    ll_abs_diff = abs(
        float(python_audit["loglikelihood"]) - float(julia["loglikelihood"])
    )
    logabsdet_abs_diff = abs(
        float(python_audit["logabsdet"]) - float(julia["logabsdet"])
    )
    observation_deviation = _difference_metrics(
        np.asarray(python["observation_deviation"], dtype=np.float64),
        np.asarray(julia["observation_deviation"], dtype=np.float64),
    )
    comparison = {
        "steady_state": steady,
        "observation_deviation": observation_deviation,
        "recovered_shocks": recovered,
        "python_recovered_vs_known": python_known,
        "julia_recovered_vs_known": julia_known,
        "jacobian": jacobian,
        "loglikelihood_abs_diff": ll_abs_diff,
        "logabsdet_abs_diff": logabsdet_abs_diff,
        "python_loglikelihood": float(python_audit["loglikelihood"]),
        "julia_loglikelihood": float(julia["loglikelihood"]),
        "python_residual_norm": float(python_audit["residual_norm"]),
        "julia_residual_norm": float(julia["residual_norm"]),
        "python_final_sep_error": float(python_audit["final_sep_error"]),
        "julia_final_sep_error": float(julia["final_sep_error"]),
        "python_iterations": int(python_audit["iterations"]),
        "julia_iterations": int(julia["iterations"]),
        "python_predict_calls": int(python_audit["predict_calls"]),
        "julia_predict_calls": int(julia["predict_calls"]),
        "python_stop_reason": python_audit["stop_reason"],
        "julia_stop_reason": julia["stop_reason"],
    }
    gates = {
        "steady_state_parity": steady["max_abs"] <= STEADY_STATE_MAX_ABS_TOL,
        "observation_demeaning_parity": (
            observation_deviation["max_abs"] <= STEADY_STATE_MAX_ABS_TOL
        ),
        "public_python_matches_audit": (
            float(python["public_audit_abs_diff"]) <= PUBLIC_AUDIT_ABS_TOL
        ),
        "recovered_shock_parity": (
            recovered["max_abs"] <= RECOVERED_SHOCK_MAX_ABS_TOL
        ),
        "python_recovers_known_shocks": (
            python_known["max_abs"] <= KNOWN_SHOCK_MAX_ABS_TOL
        ),
        "julia_recovers_known_shocks": (
            julia_known["max_abs"] <= KNOWN_SHOCK_MAX_ABS_TOL
        ),
        "finite_difference_jacobian_parity": (
            jacobian["max_abs"] <= JACOBIAN_MAX_ABS_TOL
        ),
        "loglikelihood_parity": ll_abs_diff <= LOGLIKELIHOOD_ABS_TOL,
        "python_inversion_residual": (
            float(python_audit["residual_norm"]) <= INVERSION_RESIDUAL_TOL
        ),
        "julia_inversion_residual": (
            float(julia["residual_norm"]) <= INVERSION_RESIDUAL_TOL
        ),
        "python_sep_solver_residual": (
            float(python_audit["final_sep_error"]) <= SOLVER_RESIDUAL_TOL
        ),
        "julia_sep_solver_residual": (
            float(julia["final_sep_error"]) <= SOLVER_RESIDUAL_TOL
        ),
    }
    return comparison, gates


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    upstream_repo = args.upstream_repo.resolve()
    upstream_model = upstream_repo / "models" / MODEL_PATH.name
    model, python, spec = _solve_python()
    julia_payload, julia_invocation = _run_julia_export(
        julia=args.julia, upstream_repo=upstream_repo, spec=spec
    )
    if julia_payload.get("schema_version") != JULIA_SCHEMA_VERSION:
        raise ValueError("Unexpected Julia SEP likelihood export schema.")
    comparison, numerical_gates = _compare(
        model=model, python=python, julia_payload=julia_payload
    )
    local_hash = _sha256(MODEL_PATH)
    upstream_hash = _sha256(upstream_model)
    repository_git = _git_state(ROOT)
    upstream_git = _git_state(upstream_repo)
    gates = {
        "source_identity": bool(
            local_hash == upstream_hash == julia_payload["model"]["sha256"]
        ),
        "model_identity": bool(
            str(model.name) == MODEL_SYMBOL
            and julia_payload["model"]["symbol"] == MODEL_SYMBOL
            and [str(name) for name in model.timings.exo]
            == julia_payload["model"]["shock_names"]
        ),
        "clean_provenance": bool(
            not repository_git["dirty"] and not upstream_git["dirty"]
        ),
        **numerical_gates,
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": "pass" if passed else "fail",
        "publication_ready": False,
        "scope": {
            "validated": [
                "full nonlinear Gali source identity",
                "one-transition nonlinear SEP inversion likelihood",
                "all three standardized structural innovations",
                "recovered shocks, finite-difference Jacobian, and likelihood",
            ],
            "not_validated": [
                "multi-period nonlinear likelihood recursion",
                "production-horizon likelihood parity",
                "posterior recovery",
                "GPU throughput",
            ],
        },
        "thresholds": {
            "steady_state_max_abs": STEADY_STATE_MAX_ABS_TOL,
            "public_audit_abs": PUBLIC_AUDIT_ABS_TOL,
            "recovered_shock_max_abs": RECOVERED_SHOCK_MAX_ABS_TOL,
            "known_shock_max_abs": KNOWN_SHOCK_MAX_ABS_TOL,
            "jacobian_max_abs": JACOBIAN_MAX_ABS_TOL,
            "loglikelihood_abs": LOGLIKELIHOOD_ABS_TOL,
            "inversion_residual": INVERSION_RESIDUAL_TOL,
            "solver_residual": SOLVER_RESIDUAL_TOL,
        },
        "design": spec,
        "sources": {
            "local_model": {"path": str(MODEL_PATH), "sha256": local_hash},
            "upstream_model": {"path": str(upstream_model), "sha256": upstream_hash},
        },
        "provenance": {
            "repository": repository_git,
            "upstream_repository": upstream_git,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "gates": gates,
        "comparison": comparison,
        "python": python,
        "julia_export": julia_payload,
        "julia_invocation": julia_invocation,
        "elapsed_s": time.perf_counter() - started,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
