#!/usr/bin/env python3
"""Validate a reduced-horizon full-Gali nonlinear SEP tree against Julia.

This gate compares independent nonlinear solves from SurrogateNN_DSGE and the
pinned MacroModelling.jl implementation.  It deliberately targets the complete
sparse scenario tree, not only a central path or a first-order approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parse_macro_model,
    solve_stochastic_extended_path_model,
)


SCHEMA_VERSION = "surrogatenn-dsge-gali-sep-parity/v1"
JULIA_SCHEMA_VERSION = "surrogatenn-dsge-gali-julia-sep-export/v1"
MODEL_SYMBOL = "Gali_2015_chapter_3_nonlinear"
MODEL_PATH = ROOT / "benchmarks" / "model_sources" / f"{MODEL_SYMBOL}.jl"
PAYLOAD_PATH = ROOT / "benchmarks" / "gali_nonlinear_pilot_payload.json"
JULIA_EXPORTER = ROOT / "benchmarks" / "export_gali_full_julia_sep_parity.jl"
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
DEFAULT_SHOCK = (0.12, -0.08, 0.10)
SS_MAX_ABS_TOL = 1.0e-10
TREE_MAX_ABS_TOL = 1.0e-6
MEAN_PATH_MAX_ABS_TOL = 1.0e-8
SOLVER_RESIDUAL_TOL = 1.0e-8


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


def _sparse_probabilities(
    *, periods: int, branching_order: int, nnodes: int, shock_dim: int
) -> tuple[np.ndarray, ...]:
    _, one_d_weights = np.polynomial.hermite.hermgauss(nnodes)
    one_d_weights = one_d_weights / np.sqrt(np.pi)
    zero_index = int(np.argmin(np.abs(np.polynomial.hermite.hermgauss(nnodes)[0])))
    if zero_index != 0:
        one_d_weights[[0, zero_index]] = one_d_weights[[zero_index, 0]]
    rule_weights = np.tile(one_d_weights / shock_dim, shock_dim)
    rule_weights /= np.sum(rule_weights)
    probabilities: list[np.ndarray] = [np.ones(1, dtype=np.float64)]
    trunk_probability = 1.0
    side_probabilities = np.zeros(0, dtype=np.float64)
    for period in range(1, periods + 1):
        if period > 1 and period <= branching_order + 1:
            branch_probability = trunk_probability
            trunk_probability = branch_probability * rule_weights[0]
            side_probabilities = np.concatenate(
                (side_probabilities, branch_probability * rule_weights[1:])
            )
        probabilities.append(
            np.concatenate(
                (np.asarray([trunk_probability]), side_probabilities)
            )
        )
    return tuple(probabilities)


def _mean_path(
    tree_states: Sequence[np.ndarray], probabilities: Sequence[np.ndarray]
) -> np.ndarray:
    return np.stack(
        [
            np.sum(probability[:, None] * states, axis=0)
            for probability, states in zip(probabilities, tree_states)
        ],
        axis=1,
    )


def _parse_indices(raw: str) -> tuple[int, ...]:
    indices = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not indices:
        raise ValueError("--theta-indices must contain at least one index.")
    invalid = [index for index in indices if not 0 <= index < len(EXPECTED_THETA_POINTS)]
    if invalid:
        raise ValueError(f"Theta indices out of range: {invalid}")
    return indices


def _build_spec(theta_indices: Sequence[int]) -> dict[str, Any]:
    config = {
        "periods": 3,
        "branching_order": 1,
        "nnodes": 3,
        "sparse_tree": True,
        "max_iter": 80,
        "tol": SOLVER_RESIDUAL_TOL,
        "accept_tol": SOLVER_RESIDUAL_TOL,
        "linear_solver": "qr",
        "fallback_solver": None,
        "stall_iters": 25,
        "stall_rel_tol": 1.0e-4,
        "stall_abs_tol": 1.0e-10,
    }
    shocks = np.zeros((config["periods"], 3), dtype=np.float64)
    shocks[0, :] = np.asarray(DEFAULT_SHOCK, dtype=np.float64)
    return {
        "model_symbol": MODEL_SYMBOL,
        "config": config,
        "cases": [
            {
                "case_id": f"theta_{theta_index}",
                "theta_index": theta_index,
                "parameters": {
                    "ϕᵖⁱ": EXPECTED_THETA_POINTS[theta_index][0],
                    "ϕʸ": EXPECTED_THETA_POINTS[theta_index][1],
                },
                "deterministic_shocks": shocks.tolist(),
            }
            for theta_index in theta_indices
        ],
    }


def _solve_python_cases(
    spec: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    model = parse_macro_model(MODEL_PATH.read_text(encoding="utf-8"))
    reference_ss = _load_reference_steady_state()
    raw_config = spec["config"]
    config = SEPConfig(
        periods=int(raw_config["periods"]),
        branching_order=int(raw_config["branching_order"]),
        nnodes=int(raw_config["nnodes"]),
        sparse_tree=bool(raw_config["sparse_tree"]),
        max_iter=int(raw_config["max_iter"]),
        tol=float(raw_config["tol"]),
        accept_tol=float(raw_config["accept_tol"]),
        linear_solver=str(raw_config["linear_solver"]),
        fallback_solver=raw_config["fallback_solver"],
        stall_iters=int(raw_config["stall_iters"]),
        stall_rel_tol=float(raw_config["stall_rel_tol"]),
        stall_abs_tol=float(raw_config["stall_abs_tol"]),
        line_search=True,
        line_search_norm="inf",
    )
    output: list[dict[str, Any]] = []
    for case in spec["cases"]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "theta_index": case["theta_index"],
            "parameters": case["parameters"],
        }
        try:
            parameters = _inject_parameters(
                model.parameter_values, model.parameter_names, case["parameters"]
            )
            steady = model.solve_steady_state(
                parameter_values=parameters,
                initial_guess=reference_ss,
                tol=1.0e-12,
                max_iter=100,
            )
            result = solve_stochastic_extended_path_model(
                model,
                parameter_values=parameters,
                steady_state=np.asarray(steady.steady_state, dtype=np.float64),
                initial_state=np.asarray(steady.steady_state, dtype=np.float64),
                terminal_state=np.asarray(steady.steady_state, dtype=np.float64),
                config=config,
                deterministic_shocks=case["deterministic_shocks"],
            )
            solution = result.solution
            stacked = np.asarray(solution.stacked_states, dtype=np.float64)
            group_counts = tuple(int(value) for value in solution.group_counts)
            tree_states = [
                np.asarray(steady.steady_state, dtype=np.float64)[None, :]
            ]
            offset = 0
            for period in range(1, config.periods + 1):
                block_size = group_counts[period] * model.timings.nVars
                tree_states.append(
                    stacked[offset : offset + block_size].reshape(
                        group_counts[period], model.timings.nVars
                    )
                )
                offset += block_size
            if offset != stacked.size:
                raise ValueError("Python SEP tree did not consume its full state vector.")
            record.update(
                {
                    "status": "ok",
                    "elapsed_s": time.perf_counter() - started,
                    "steady_state": np.asarray(steady.steady_state).tolist(),
                    "steady_state_residual_norm": float(steady.residual_norm),
                    "residual_norm": float(solution.residual_norm),
                    "converged": bool(solution.converged),
                    "accepted": bool(solution.accepted),
                    "iterations": int(solution.iterations),
                    "group_counts": list(group_counts),
                    "tree_states": [values.tolist() for values in tree_states],
                    "mean_path": np.asarray(solution.mean_path).tolist(),
                }
            )
        except Exception as exc:
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
    *, julia: str, upstream_repo: Path, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gali-sep-parity-") as temp_name:
        temp_dir = Path(temp_name)
        cases_path = temp_dir / "cases.json"
        output_path = temp_dir / "julia.json"
        cases_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        argv = [
            julia,
            f"--project={upstream_repo}",
            str(JULIA_EXPORTER),
            str(MODEL_PATH),
            str(cases_path),
            str(output_path),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True
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
                f"Exit {completed.returncode}; stderr: {completed.stderr[-4000:]}"
            )
        return json.loads(output_path.read_text(encoding="utf-8")), invocation


def _align_julia_tree(
    case: Mapping[str, Any],
    *,
    julia_names: Sequence[str],
    python_names: Sequence[str],
) -> list[np.ndarray]:
    index = {str(name): position for position, name in enumerate(julia_names)}
    if set(index) != set(python_names):
        raise ValueError("Julia and Python SEP variable names differ.")
    order = [index[name] for name in python_names]
    return [
        np.asarray(period, dtype=np.float64)[:, order]
        for period in case["tree_states"]
    ]


def _compare_case(
    python_case: Mapping[str, Any],
    julia_case: Mapping[str, Any],
    *,
    julia_names: Sequence[str],
    python_names: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": python_case["case_id"],
        "theta_index": python_case["theta_index"],
        "parameters": python_case["parameters"],
        "python_status": python_case["status"],
        "julia_status": julia_case.get("status", "missing"),
        "passed": False,
    }
    if python_case["status"] != "ok" or julia_case.get("status") != "ok":
        result["error"] = "At least one implementation failed this case."
        return result
    try:
        python_tree = [
            np.asarray(period, dtype=np.float64)
            for period in python_case["tree_states"]
        ]
        julia_tree = _align_julia_tree(
            julia_case, julia_names=julia_names, python_names=python_names
        )
        if python_case["group_counts"] != julia_case["group_counts"]:
            raise ValueError("Julia and Python group counts differ.")
        period_metrics = [
            _difference_metrics(python_period, julia_period)
            for python_period, julia_period in zip(python_tree, julia_tree)
        ]
        probabilities = _sparse_probabilities(
            periods=int(config["periods"]),
            branching_order=int(config["branching_order"]),
            nnodes=int(config["nnodes"]),
            shock_dim=3,
        )
        julia_mean = _mean_path(julia_tree, probabilities)
        python_mean = np.asarray(python_case["mean_path"], dtype=np.float64)
        tree_metrics = _difference_metrics(
            np.concatenate(python_tree, axis=0),
            np.concatenate(julia_tree, axis=0),
        )
        mean_metrics = _difference_metrics(python_mean, julia_mean)
        ss_metrics = _difference_metrics(python_tree[0], julia_tree[0])
        result.update(
            {
                "group_counts": python_case["group_counts"],
                "steady_state": ss_metrics,
                "tree": tree_metrics,
                "mean_path": mean_metrics,
                "period_metrics": period_metrics,
                "python": {
                    "residual_norm": float(python_case["residual_norm"]),
                    "converged": bool(python_case["converged"]),
                    "accepted": bool(python_case["accepted"]),
                    "iterations": int(python_case["iterations"]),
                },
                "julia": {
                    "final_error": float(julia_case["final_error"]),
                    "convergence_flag": int(julia_case["convergence_flag"]),
                },
            }
        )
        result["passed"] = bool(
            ss_metrics["max_abs"] <= SS_MAX_ABS_TOL
            and tree_metrics["max_abs"] <= TREE_MAX_ABS_TOL
            and mean_metrics["max_abs"] <= MEAN_PATH_MAX_ABS_TOL
            and float(python_case["residual_norm"]) <= SOLVER_RESIDUAL_TOL
            and bool(python_case["accepted"])
            and int(julia_case["convergence_flag"]) == 0
            and float(julia_case["final_error"]) <= SOLVER_RESIDUAL_TOL
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    upstream_repo = args.upstream_repo.resolve()
    upstream_model = upstream_repo / "models" / MODEL_PATH.name
    theta_indices = _parse_indices(args.theta_indices)
    spec = _build_spec(theta_indices)
    model, python_cases = _solve_python_cases(spec)
    julia_payload, julia_invocation = _run_julia_export(
        julia=args.julia, upstream_repo=upstream_repo, spec=spec
    )
    if julia_payload.get("schema_version") != JULIA_SCHEMA_VERSION:
        raise ValueError("Unexpected Julia SEP export schema.")
    julia_cases = {case["case_id"]: case for case in julia_payload["cases"]}
    comparisons = [
        _compare_case(
            python_case,
            julia_cases.get(python_case["case_id"], {}),
            julia_names=julia_payload["model"]["variable_names"],
            python_names=model.timings.var,
            config=spec["config"],
        )
        for python_case in python_cases
    ]
    local_hash = _sha256(MODEL_PATH)
    upstream_hash = _sha256(upstream_model)
    repository_git = _git_state(ROOT)
    upstream_git = _git_state(upstream_repo)
    gates = {
        "source_identity": bool(
            local_hash == upstream_hash == julia_payload["model"]["sha256"]
        ),
        "all_cases_pass": bool(
            len(comparisons) == len(theta_indices)
            and all(item["passed"] for item in comparisons)
        ),
        "clean_provenance": bool(
            not repository_git["dirty"] and not upstream_git["dirty"]
        ),
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
                "reduced-horizon nonlinear stochastic extended-path tree",
                "all three deterministic shock channels",
            ],
            "not_validated": [
                "production-horizon trajectory parity",
                "nonlinear likelihood parity",
                "posterior recovery",
                "GPU throughput",
            ],
        },
        "thresholds": {
            "steady_state_max_abs": SS_MAX_ABS_TOL,
            "tree_max_abs": TREE_MAX_ABS_TOL,
            "mean_path_max_abs": MEAN_PATH_MAX_ABS_TOL,
            "solver_residual": SOLVER_RESIDUAL_TOL,
        },
        "design": {
            "theta_indices": list(theta_indices),
            "theta_points": [list(EXPECTED_THETA_POINTS[index]) for index in theta_indices],
            "deterministic_first_period_shock": list(DEFAULT_SHOCK),
            "config": spec["config"],
        },
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
        "gates": gates,
        "comparisons": comparisons,
        "python_cases": python_cases,
        "julia_export": julia_payload,
        "julia_invocation": julia_invocation,
        "elapsed_s": time.perf_counter() - started,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--theta-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"WROTE {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
