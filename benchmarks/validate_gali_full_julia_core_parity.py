#!/usr/bin/env python3
"""Validate full-Gali steady-state and first-order parity against Julia.

This is deliberately a narrow, independent core gate.  It does not validate
nonlinear extended-path trajectories, nonlinear likelihoods, posterior draws,
or GPU throughput.  Those remain separate publication requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
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
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
)


SCHEMA_VERSION = "surrogatenn-dsge-gali-core-parity/v1"
MODEL_SYMBOL = "Gali_2015_chapter_3_nonlinear"
MODEL_PATH = ROOT / "benchmarks" / "model_sources" / f"{MODEL_SYMBOL}.jl"
PAYLOAD_PATH = ROOT / "benchmarks" / "gali_nonlinear_pilot_payload.json"
JULIA_EXPORTER = ROOT / "benchmarks" / "export_gali_full_julia_core_parity.jl"
DEFAULT_UPSTREAM_REPO = ROOT.parent / "MacroModelling.jl"
EXPECTED_THETA_POINTS = (
    (1.50, 0.125),
    (1.32, 0.140),
    (1.38, 0.135),
    (1.44, 0.130),
    (1.50, 0.125),
    (1.56, 0.120),
    (1.62, 0.115),
    (1.68, 0.110),
)
SS_MAX_ABS_TOL = 1.0e-10
FIRST_ORDER_MAX_ABS_TOL = 1.0e-8
PYTHON_SS_RESIDUAL_TOL = 1.0e-10


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_state(repo: Path) -> dict[str, Any]:
    commit = _run_text(("git", "rev-parse", "HEAD"), cwd=repo)
    status = _run_text(("git", "status", "--short"), cwd=repo).splitlines()
    diff = subprocess.run(
        ("git", "diff", "HEAD", "--binary"),
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": str(repo.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def inject_parameter_values(
    base_values: Sequence[float],
    parameter_names: Sequence[str],
    overrides: Mapping[str, float],
) -> np.ndarray:
    """Return a copied parameter vector with name-addressed overrides."""

    values = np.asarray(base_values, dtype=np.float64).copy()
    if values.shape != (len(parameter_names),):
        raise ValueError("Parameter vector length does not match parameter names.")
    index = {str(name): position for position, name in enumerate(parameter_names)}
    missing = sorted(set(overrides) - set(index))
    if missing:
        raise ValueError(f"Unknown parameter override(s): {missing}")
    for name, raw_value in overrides.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Parameter {name} is not finite.")
        values[index[name]] = value
    return values


def _normalize_julia_solution_column(name: str) -> str:
    for suffix in ("₍₋₁₎", "₍ₓ₎"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _aligned_matrix(
    values: Any,
    row_names: Sequence[str],
    column_names: Sequence[str],
    expected_rows: Sequence[str],
    expected_columns: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (len(row_names), len(column_names)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match labels "
            f"{(len(row_names), len(column_names))}."
        )
    row_index = {name: i for i, name in enumerate(row_names)}
    column_index = {name: i for i, name in enumerate(column_names)}
    if set(row_index) != set(expected_rows):
        raise ValueError("Julia and Python first-order row names differ.")
    if set(column_index) != set(expected_columns):
        raise ValueError("Julia and Python first-order column names differ.")
    return matrix[
        np.ix_(
            [row_index[name] for name in expected_rows],
            [column_index[name] for name in expected_columns],
        )
    ]


def _aligned_vector(
    values: Any,
    names: Sequence[str],
    expected_names: Sequence[str],
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(names),):
        raise ValueError("Steady-state vector shape does not match its labels.")
    index = {name: i for i, name in enumerate(names)}
    if set(index) != set(expected_names):
        raise ValueError("Julia and Python steady-state names differ.")
    return vector[[index[name] for name in expected_names]]


def _difference_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {left.shape} and {right.shape}.")
    delta = np.asarray(left - right, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def compare_case(
    python_case: Mapping[str, Any],
    julia_case: Mapping[str, Any],
    *,
    variable_names: Sequence[str],
    solution_column_names: Sequence[str],
) -> dict[str, Any]:
    """Compare one independently solved case without dropping failures."""

    result: dict[str, Any] = {
        "case_id": str(python_case["case_id"]),
        "theta_index": int(python_case["theta_index"]),
        "parameters": dict(python_case["parameters"]),
        "python_status": str(python_case["status"]),
        "julia_status": str(julia_case["status"]),
        "passed": False,
    }
    if python_case["case_id"] != julia_case.get("case_id"):
        result["error"] = "Case identifiers do not match."
        return result
    if python_case["status"] != "ok" or julia_case.get("status") != "ok":
        result["error"] = "At least one implementation failed this case."
        return result

    julia_ss = _aligned_vector(
        julia_case["steady_state"]["values"],
        julia_case["steady_state"]["names"],
        variable_names,
    )
    julia_columns = [
        _normalize_julia_solution_column(str(name))
        for name in julia_case["first_order"]["column_names_raw"]
    ]
    julia_solution = _aligned_matrix(
        julia_case["first_order"]["solution_matrix"],
        julia_case["first_order"]["row_names"],
        julia_columns,
        variable_names,
        solution_column_names,
    )
    ss_metrics = _difference_metrics(
        np.asarray(python_case["steady_state"], dtype=np.float64), julia_ss
    )
    solution_metrics = _difference_metrics(
        np.asarray(python_case["first_order_solution"], dtype=np.float64),
        julia_solution,
    )
    result.update(
        {
            "steady_state": ss_metrics,
            "first_order_solution": solution_metrics,
            "python_steady_state_residual_norm": float(
                python_case["steady_state_residual_norm"]
            ),
            "thresholds": {
                "steady_state_max_abs": SS_MAX_ABS_TOL,
                "first_order_max_abs": FIRST_ORDER_MAX_ABS_TOL,
                "python_steady_state_residual_norm": PYTHON_SS_RESIDUAL_TOL,
            },
        }
    )
    result["passed"] = bool(
        ss_metrics["max_abs"] <= SS_MAX_ABS_TOL
        and solution_metrics["max_abs"] <= FIRST_ORDER_MAX_ABS_TOL
        and float(python_case["steady_state_residual_norm"])
        <= PYTHON_SS_RESIDUAL_TOL
        and bool(python_case["first_order_converged"])
    )
    return result


def _cases_spec() -> dict[str, Any]:
    return {
        "model_symbol": MODEL_SYMBOL,
        "cases": [
            {
                "case_id": f"theta_{theta_index}",
                "theta_index": theta_index,
                "parameters": {"ϕᵖⁱ": phi_pi, "ϕʸ": phi_y},
            }
            for theta_index, (phi_pi, phi_y) in enumerate(EXPECTED_THETA_POINTS)
        ],
    }


def _load_reference_steady_state() -> list[float]:
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    matching = [case for case in cases if case.get("model_symbol") == MODEL_SYMBOL]
    if len(matching) != 1:
        raise ValueError("Could not identify one full-Gali payload case.")
    return [float(value) for value in matching[0]["reference_steady_state"]]


def _solve_python_cases(spec: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    model = parse_macro_model(MODEL_PATH.read_text(encoding="utf-8"))
    reference_ss = _load_reference_steady_state()
    output: list[dict[str, Any]] = []
    for case in spec["cases"]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "theta_index": case["theta_index"],
            "parameters": case["parameters"],
        }
        try:
            parameters = inject_parameter_values(
                model.parameter_values,
                model.parameter_names,
                case["parameters"],
            )
            steady = solve_steady_state(
                model,
                parameter_values=parameters,
                initial_guess=reference_ss,
                tol=1.0e-12,
                max_iter=100,
            )
            first_order = solve_first_order_model(
                model,
                parameter_values=parameters,
                steady_state=np.asarray(steady.steady_state, dtype=np.float64),
                qme_algorithm="schur",
            )
            steady_values = np.asarray(steady.steady_state, dtype=np.float64)
            solution_values = np.asarray(
                first_order.solution.solution_matrix, dtype=np.float64
            )
            if not np.all(np.isfinite(steady_values)) or not np.all(
                np.isfinite(solution_values)
            ):
                raise ValueError("Python solver output contains a non-finite value.")
            record.update(
                {
                    "status": "ok",
                    "elapsed_s": time.perf_counter() - started,
                    "parameter_values": np.asarray(parameters).tolist(),
                    "steady_state": steady_values.tolist(),
                    "steady_state_converged": bool(steady.converged),
                    "steady_state_iterations": int(steady.iterations),
                    "steady_state_residual_norm": float(steady.residual_norm),
                    "first_order_converged": bool(first_order.solution.converged),
                    "first_order_solution": solution_values.tolist(),
                }
            )
        except Exception as exc:  # preserve every failed design point
            record.update(
                {
                    "status": "error",
                    "elapsed_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        output.append(record)
    return model, output


def _run_julia_export(
    *,
    julia_executable: str,
    upstream_repo: Path,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gali-core-parity-") as temp_name:
        temp_dir = Path(temp_name)
        cases_path = temp_dir / "cases.json"
        output_path = temp_dir / "julia.json"
        cases_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        argv = [
            julia_executable,
            f"--project={upstream_repo}",
            str(JULIA_EXPORTER),
            str(MODEL_PATH),
            str(cases_path),
            str(output_path),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
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
                f"Exit code {completed.returncode}; stderr: {completed.stderr[-4000:]}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return payload, invocation


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    upstream_repo = args.upstream_repo.resolve()
    upstream_model = upstream_repo / "models" / MODEL_PATH.name
    if not upstream_model.is_file():
        raise FileNotFoundError(f"Missing upstream model source: {upstream_model}")
    local_hash = _sha256(MODEL_PATH)
    upstream_hash = _sha256(upstream_model)
    spec = _cases_spec()
    model, python_cases = _solve_python_cases(spec)
    julia_payload, julia_invocation = _run_julia_export(
        julia_executable=args.julia,
        upstream_repo=upstream_repo,
        spec=spec,
    )
    julia_cases = {case["case_id"]: case for case in julia_payload["cases"]}
    expected_columns = tuple(model.timings.past_not_future_and_mixed) + tuple(
        model.timings.exo
    )
    comparisons = [
        compare_case(
            python_case,
            julia_cases.get(
                python_case["case_id"],
                {
                    "case_id": python_case["case_id"],
                    "status": "missing",
                },
            ),
            variable_names=model.timings.var,
            solution_column_names=expected_columns,
        )
        for python_case in python_cases
    ]
    upstream_git = _git_state(upstream_repo)
    repository_git = _git_state(ROOT)
    source_gate = local_hash == upstream_hash == julia_payload["model"]["sha256"]
    numerical_gate = len(comparisons) == len(EXPECTED_THETA_POINTS) and all(
        comparison["passed"] for comparison in comparisons
    )
    provenance_gate = not repository_git["dirty"] and not upstream_git["dirty"]
    passed = source_gate and numerical_gate and provenance_gate
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": "pass" if passed else "fail",
        "scope": {
            "validated": [
                "full nonlinear Gali source identity",
                "non-stochastic steady state",
                "first-order Schur solution",
            ],
            "not_validated": [
                "nonlinear extended-path trajectory parity",
                "nonlinear likelihood parity",
                "posterior recovery",
                "GPU throughput",
            ],
        },
        "thresholds": {
            "steady_state_max_abs": SS_MAX_ABS_TOL,
            "first_order_max_abs": FIRST_ORDER_MAX_ABS_TOL,
            "python_steady_state_residual_norm": PYTHON_SS_RESIDUAL_TOL,
        },
        "design": {
            "theta_parameter_names": ["ϕᵖⁱ", "ϕʸ"],
            "theta_points": [list(point) for point in EXPECTED_THETA_POINTS],
            "case_count": len(EXPECTED_THETA_POINTS),
            "unique_theta_count": len(set(EXPECTED_THETA_POINTS)),
            "duplicate_reference_indices": [0, 4],
        },
        "sources": {
            "local_model": {"path": str(MODEL_PATH), "sha256": local_hash},
            "upstream_model": {
                "path": str(upstream_model),
                "sha256": upstream_hash,
            },
            "source_identity_passed": source_gate,
        },
        "repositories": {
            "surrogatenn_dsge": repository_git,
            "macromodelling_jl": upstream_git,
            "clean_provenance_passed": provenance_gate,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": np.__version__,
            "julia": julia_payload.get("julia_version"),
            "jax_platform": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "command": {
            "argv": sys.argv,
            "shell_escaped": shlex.join(sys.argv),
            "cwd": os.getcwd(),
        },
        "julia_invocation": julia_invocation,
        "python_cases": python_cases,
        "julia_export": julia_payload,
        "comparisons": comparisons,
        "gates": {
            "source_identity": source_gate,
            "all_design_points_numerically_pass": numerical_gate,
            "clean_repository_provenance": provenance_gate,
        },
        "elapsed_s": time.perf_counter() - started,
        "publication_ready": False,
        "publication_readiness_note": (
            "Passing this report establishes only the declared core parity gate; "
            "the nonlinear, likelihood, posterior, and GPU gates remain open."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=DEFAULT_UPSTREAM_REPO,
        help="MacroModelling.jl checkout used for the independent Julia solve.",
    )
    parser.add_argument("--julia", default="julia", help="Julia executable.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
