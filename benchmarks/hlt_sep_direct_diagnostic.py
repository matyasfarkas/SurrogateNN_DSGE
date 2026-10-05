#!/usr/bin/env python3
"""Direct HLT SEP diagnostics for GPUHub runs.

This avoids the full training/HMC wrapper and measures the specific HLT SEP
solve shapes that caused the residual-floor investigation.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks import profile_surrogate_pipeline_gpu as prof  # noqa: E402
from surrogatenn_dsge import (  # noqa: E402
    SEPConfig,
    parse_macro_model,
    solve_batched_stochastic_extended_path_model,
)
from surrogatenn_dsge.sep import _gauss_hermite_sparse_rule, _group_counts  # noqa: E402


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[hlt-sep-diagnostic] {stamp} {message}", flush=True)


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


def _block_solution(solution: Any) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        solution,
    )


def run_diagnostics(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gpu_sep_diagnostics.json"
    out: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.read("jax_enable_x64")),
        "output_dir": str(output_dir),
    }

    def write_partial(stage: str) -> None:
        out["last_stage"] = stage
        out["last_update_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        output_path.write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _log(f"wrote partial diagnostics stage={stage} path={output_path}")

    _log(f"starting diagnostics backend={out['backend']} devices={out['devices']}")
    write_partial("started")
    _log("loading HLT payload and parsing model")
    args = Namespace(
        hlt_payload=ROOT / "benchmarks" / "results" / "test_payloads.json",
        hlt_case_name="medium_sw07_hlt",
        hlt_parameter_set="all",
        hlt_theta_draws=1,
        hlt_parameter_perturbation=1e-6,
        hlt_theta_design="perturbation",
        hlt_theta_design_set="auto",
        seed=20260918,
        hlt_theta_include_reference=False,
        hlt_model_source=ROOT / "benchmarks" / "model_sources" / "Smets_Wouters_2007_HLT.jl",
    )
    case = prof._load_hlt_payload_case(args)
    model = parse_macro_model(args.hlt_model_source.read_text())
    _log(
        "parsed HLT model "
        f"n_vars={model.timings.nVars} n_exo={model.timings.nExo} "
        f"n_params={len(model.parameter_names)}"
    )
    reference_ss = np.asarray(case["reference_steady_state"], dtype=np.float64)
    base_params = np.asarray(model.parameter_values, dtype=np.float64)
    subset = list(prof._select_hlt_parameter_subset(model, case, args.hlt_parameter_set))
    theta, subset_idx, theta_diag = prof._make_hlt_theta_design(
        base_parameters=base_params,
        parameter_names=model.parameter_names,
        subset_names=subset,
        draws=1,
        perturbation=1e-6,
        design="perturbation",
        design_set="auto",
        seed=20260918,
        include_reference=False,
    )
    params = base_params.copy()
    params[subset_idx] = theta[:, 0]
    out["parameter_count"] = int(len(subset))
    out["theta_design"] = theta_diag
    write_partial("model_parsed")

    _log("solving HLT steady state")
    ss_t0 = time.perf_counter()
    ss_result = model.solve_steady_state(
        parameter_values=params,
        initial_guess=reference_ss,
        tol=1e-10,
        max_iter=100,
    )
    ss = np.asarray(
        ss_result.steady_state if ss_result.converged else reference_ss,
        dtype=np.float64,
    )
    out["steady_state"] = {
        "converged": bool(ss_result.converged),
        "iterations": int(ss_result.iterations),
        "residual_norm": float(ss_result.residual_norm),
        "elapsed_s": time.perf_counter() - ss_t0,
    }
    _log(
        "steady state finished "
        f"converged={bool(ss_result.converged)} "
        f"residual={float(ss_result.residual_norm):.3e} "
        f"elapsed_s={out['steady_state']['elapsed_s']:.3f}"
    )
    write_partial("steady_state")

    shock_dim = model.timings.nExo
    shock = np.zeros((shock_dim,), dtype=np.float64)
    shock[0] = 0.03
    gpu = jax.devices("gpu")[0]

    def run_batched(
        label: str,
        config: SEPConfig,
        *,
        initial_guess: np.ndarray | None = None,
    ) -> dict[str, Any]:
        deterministic = np.zeros((config.periods, shock_dim), dtype=np.float64)
        deterministic[0] = shock
        _log(
            f"starting {label} periods={config.periods} "
            f"order={config.branching_order} nnodes={config.nnodes} "
            f"max_iter={config.max_iter} tol={config.tol} "
            f"accept_tol={config.accept_tol}"
        )
        started = time.perf_counter()
        try:
            with jax.default_device(gpu):
                result = solve_batched_stochastic_extended_path_model(
                    model,
                    parameter_values=params[None, :],
                    steady_state=ss[None, :],
                    initial_state=ss[None, :],
                    terminal_state=ss[None, :],
                    deterministic_shocks=deterministic[None, :, :],
                    config=config,
                    initial_guess=initial_guess,
                )
            _block_solution(result.solution)
            solution = result.solution
            elapsed = time.perf_counter() - started
            residual_norm = np.asarray(solution.residual_norm, dtype=np.float64)
            accepted = np.asarray(solution.accepted, dtype=bool)
            iterations = np.asarray(solution.iterations, dtype=np.int64)
            _log(
                f"finished {label} status=ok elapsed_s={elapsed:.3f} "
                f"accepted={accepted.tolist()} "
                f"residual={residual_norm.tolist()} "
                f"iterations={iterations.tolist()}"
            )
            return {
                "status": "ok",
                "label": label,
                "elapsed_s": elapsed,
                "residual_norm": residual_norm,
                "accepted": accepted,
                "converged": np.asarray(solution.converged, dtype=bool),
                "iterations": iterations,
                "group_counts": tuple(int(x) for x in solution.group_counts),
                "config": {
                    "periods": int(config.periods),
                    "branching_order": int(config.branching_order),
                    "nnodes": int(config.nnodes),
                    "shock_scale": float(config.shock_scale),
                    "max_iter": int(config.max_iter),
                    "tol": float(config.tol),
                    "accept_tol": None
                    if config.accept_tol is None
                    else float(config.accept_tol),
                    "linear_solver": str(config.linear_solver),
                    "line_search_maxit": int(config.line_search_maxit),
                    "lm_lambda_max": float(config.lm_lambda_max),
                },
            }
        except Exception as exc:
            _log(
                f"finished {label} status=error elapsed_s={time.perf_counter() - started:.3f} "
                f"error={exc!r}"
            )
            return {
                "status": "error",
                "label": label,
                "elapsed_s": time.perf_counter() - started,
                "error": repr(exc),
            }

    p4_config = SEPConfig(
        periods=4,
        branching_order=1,
        nnodes=3,
        shock_scale=1.0,
        sparse_tree=True,
        max_iter=12,
        tol=1e-8,
        accept_tol=1e-4,
        linear_solver="qr",
        line_search=True,
    )
    out["p4_batched_full_solve"] = run_batched("p4_batched_full_solve", p4_config)
    write_partial("p4_batched_full_solve")

    p8_initial_config = SEPConfig(
        periods=8,
        branching_order=1,
        nnodes=3,
        shock_scale=1.0,
        sparse_tree=True,
        max_iter=1,
        tol=1e9,
        accept_tol=1e9,
        linear_solver="qr",
        line_search=True,
        line_search_maxit=1,
        lm_lambda_max=1e-8,
    )
    out["p8_selected_initial_residual"] = run_batched(
        "p8_selected_initial_residual",
        p8_initial_config,
    )
    write_partial("p8_selected_initial_residual")

    rule = _gauss_hermite_sparse_rule(
        p8_initial_config.nnodes,
        shock_dim,
        p8_initial_config.shock_scale,
    )
    counts = _group_counts(
        p8_initial_config.periods,
        p8_initial_config.branching_order,
        int(rule.weights.shape[0]),
        sparse_tree=True,
    )
    terminal_guess = np.stack(
        [
            np.vstack(
                [
                    np.tile(ss, (counts[t + 1], 1))
                    for t in range(p8_initial_config.periods)
                ]
            )
        ],
        axis=0,
    )
    out["p8_terminal_initial_residual"] = run_batched(
        "p8_terminal_initial_residual",
        p8_initial_config,
        initial_guess=terminal_guess,
    )
    write_partial("p8_terminal_initial_residual")

    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_path.write_text(
        json.dumps(_jsonable(out), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log(f"finished all diagnostics path={output_path}")
    return out


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "benchmarks"
        / "results"
        / f"gpuhub_hlt_sep_direct_diag_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
    )
    args = parser.parse_args()
    result = run_diagnostics(args.output_dir)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    print("WROTE", args.output_dir / "gpu_sep_diagnostics.json")


if __name__ == "__main__":
    main()
