from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        import toml as tomllib  # type: ignore[no-redef]

from surrogatenn_dsge import (
    RegimeSwitchConfig,
    SEPConfig,
    compute_linear_gate_stats_from_filter_model_jax,
    gate_probabilities,
    parse_macro_model,
    rollout_first_order_solution,
    solve_first_order_model,
)


ROOT = Path("/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE")
BENCH_DIR = ROOT / "benchmarks"
CONFIG_PATH = BENCH_DIR / "nn_surrogate_validation_profile.toml"
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def _system_info() -> dict[str, str]:
    keys = {
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "logical_cpu": ["sysctl", "-n", "hw.ncpu"],
        "perf_cores": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
        "eff_cores": ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
        "mem_bytes": ["sysctl", "-n", "hw.memsize"],
    }
    out: dict[str, str] = {}
    for name, cmd in keys.items():
        try:
            out[name] = _run(cmd).stdout.strip()
        except Exception:
            out[name] = "unknown"
    return out


def _load_config(path: Path) -> dict[str, Any]:
    if tomllib.__name__ == "toml":
        return tomllib.load(str(path))
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _reference_states_by_case(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {entry["name"]: entry for entry in payload["entries"]}


def _normalize_reference_name(name: str) -> str:
    return re.sub(
        r"ᴸ⁽([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)⁾",
        lambda match: "__L" + match.group(1).translate(_SUPERSCRIPT_DIGITS),
        name,
    )


def _build_payloads(
    config: dict[str, Any],
    reference_state_path: Path,
    payload_path: Path,
) -> dict[str, Any]:
    reference_states = _reference_states_by_case(reference_state_path)
    gate_quantile = float(config["global"]["gate_quantile"])
    cases_payload: list[dict[str, Any]] = []

    for case in config["case"]:
        model_path = Path(case["model_path"])
        model = parse_macro_model(model_path.read_text())
        reference = reference_states[case["name"]]
        reference_lookup = {
            key: value for key, value in zip(reference["var"], reference["steady_state"])
        }
        reference_lookup.update(
            {
                _normalize_reference_name(key): value
                for key, value in zip(reference["var"], reference["steady_state"])
            }
        )
        steady_state = np.asarray(
            [reference_lookup[name] for name in model.timings.var],
            dtype=np.float64,
        )
        first_order = solve_first_order_model(
            model,
            steady_state=steady_state,
            qme_algorithm="schur",
        )
        if not first_order.solution.converged:
            raise RuntimeError(f"First-order solve failed for {case['name']}.")

        shock_names = tuple(model.timings.exo)
        observation_names = tuple(sorted(case["observables"]))
        state_names = tuple(case["state_names"])
        shock_scales = np.asarray(
            [float(case["shock_scale_by_exo"][name]) for name in shock_names],
            dtype=np.float64,
        )
        rng = np.random.default_rng(int(case["shock_seed"]))
        shock_matrix = shock_scales[:, None] * rng.standard_normal(
            (len(shock_names), int(case["periods"]))
        )
        full_levels = (
            np.asarray(
                rollout_first_order_solution(
                    first_order.solution.solution_matrix,
                    model.timings,
                    shock_matrix,
                ),
                dtype=np.float64,
            )
            + steady_state[:, None]
        )
        observation_indices = [model.timings.var.index(name) for name in observation_names]
        observations = full_levels[observation_indices, :]
        obs_sigma_values = np.maximum(observations.std(axis=1), 1.0e-3)
        obs_sigma = {
            name: float(obs_sigma_values[idx]) for idx, name in enumerate(observation_names)
        }
        shock_sigmas = {
            name: float(abs(shock_scales[idx])) for idx, name in enumerate(shock_names)
        }

        gate_stats = compute_linear_gate_stats_from_filter_model_jax(
            model,
            observations,
            obs_sigma,
            shock_sigmas,
            state_names,
            observables=observation_names,
            parameter_values=np.asarray(first_order.parameter_values, dtype=np.float64),
            steady_state=steady_state,
            qme_algorithm="schur",
            filter="kalman",
            algorithm="first_order",
            smooth=False,
            on_failure_fill_value=np.nan,
        )
        e_stat = np.asarray(gate_stats.e_stat, dtype=np.float64)
        f_stat = np.asarray(gate_stats.f_stat, dtype=np.float64)
        tau_eps = float(np.quantile(e_stat, gate_quantile))
        tau_y = float(np.quantile(f_stat, gate_quantile))
        regime_switch_config = RegimeSwitchConfig(
            gate_mode=str(case["gate_mode"]),
            tau_eps=tau_eps,
            tau_y=tau_y,
            beta_eps=float(case["gate_beta_eps"]),
            beta_y=float(case["gate_beta_y"]),
            hard_threshold=float(case["gate_hard_threshold"]),
            prob_floor=float(case["gate_prob_floor"]),
            prob_ceiling=float(case["gate_prob_ceiling"]),
            soft_mixture=str(case["soft_mixture"]),
        )
        shared_gate_probs = np.asarray(
            gate_probabilities(e_stat, f_stat, regime_switch_config),
            dtype=np.float64,
        )

        cases_payload.append(
            {
                "name": case["name"],
                "model_symbol": case["model_symbol"],
                "model_path": str(model_path),
                "reference_steady_state": steady_state.tolist(),
                "observables": list(observation_names),
                "state_names": list(state_names),
                "shock_names": list(shock_names),
                "observations": observations.tolist(),
                "shock_matrix": shock_matrix.tolist(),
                "obs_sigma": obs_sigma,
                "shock_sigmas": shock_sigmas,
                "parameter_subset": list(case["parameter_subset"]),
                "numpyro_parameter_subset": list(
                    case.get("numpyro_parameter_subset", case["parameter_subset"])
                ),
                "numpyro_prior_width_scale": float(
                    case.get("numpyro_prior_width_scale", 0.01)
                ),
                "numpyro_prior_width_floor": float(
                    case.get("numpyro_prior_width_floor", 1e-4)
                ),
                "numpyro_log_density_reps": int(case.get("numpyro_log_density_reps", 0)),
                "numpyro_switching_log_density_reps": int(
                    case.get("numpyro_switching_log_density_reps", 0)
                ),
                "numpyro_nuts_warmup": int(case.get("numpyro_nuts_warmup", 0)),
                "numpyro_nuts_samples": int(case.get("numpyro_nuts_samples", 0)),
                "numpyro_nuts_chains": int(case.get("numpyro_nuts_chains", 1)),
                "numpyro_target_accept_prob": float(
                    case.get("numpyro_target_accept_prob", 0.8)
                ),
                "numpyro_seed": int(case.get("numpyro_seed", 0)),
                "measurement_error_scale": float(case["measurement_error_scale"]),
                "jitter": float(case["jitter"]),
                "solve_reps": int(case["solve_reps"]),
                "kalman_value_reps": int(case["kalman_value_reps"]),
                "kalman_per_period_reps": int(case["kalman_per_period_reps"]),
                "kalman_paths_reps": int(case["kalman_paths_reps"]),
                "kalman_grad_reps": int(case["kalman_grad_reps"]),
                "gate_stats_reps": int(case["gate_stats_reps"]),
                "switching_fixed_reps": int(case["switching_fixed_reps"]),
                "switching_reps": int(case["switching_reps"]),
                "sep_reps": int(case["sep_reps"]),
                "gate_periods": None,
                "gate_mode": str(case["gate_mode"]),
                "gate_hard_threshold": float(case["gate_hard_threshold"]),
                "gate_prob_floor": float(case["gate_prob_floor"]),
                "gate_prob_ceiling": float(case["gate_prob_ceiling"]),
                "soft_mixture": str(case["soft_mixture"]),
                "shared_gate_probs": shared_gate_probs.tolist(),
                "regime_switch_config": {
                    "tau_eps": tau_eps,
                    "tau_y": tau_y,
                    "beta_eps": float(case["gate_beta_eps"]),
                    "beta_y": float(case["gate_beta_y"]),
                },
                "sep_eval_periods": int(case["sep_eval_periods"]),
                "sep_periods": int(case["sep_periods"]),
                "sep_branching_order": int(case["sep_branching_order"]),
                "sep_nnodes": int(case["sep_nnodes"]),
                "sep_sparse_tree": bool(case["sep_sparse_tree"]),
                "sep_maxit": int(case["sep_maxit"]),
                "sep_tol": float(case["sep_tol"]),
                "sep_accept_tol": float(case["sep_accept_tol"]),
                "sep_inv_maxit": int(case["sep_inv_maxit"]),
                "sep_inv_step_tol": float(case["sep_inv_step_tol"]),
                "sep_inv_resid_tol": float(case["sep_inv_resid_tol"]),
                "sep_inv_lambda": float(case["sep_inv_lambda"]),
            }
        )

    payload = {"cases": cases_payload}
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _timed_process(
    cmd: list[str],
    *,
    env: dict[str, str],
    time_output_path: Path,
) -> subprocess.CompletedProcess[str]:
    wrapped = ["/usr/bin/time", "-l", "-o", str(time_output_path), *cmd]
    return _run(wrapped, env=env)


def _parse_time_output(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    text = path.read_text()
    real_match = re.search(r"([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", text)
    if real_match:
        metrics["real_s"] = float(real_match.group(1))
        metrics["user_s"] = float(real_match.group(2))
        metrics["sys_s"] = float(real_match.group(3))
    rss_match = re.search(r"([0-9]+)\s+maximum resident set size", text)
    if rss_match:
        metrics["max_rss_raw"] = float(rss_match.group(1))
    peak_match = re.search(r"([0-9]+)\s+peak memory footprint", text)
    if peak_match:
        metrics["peak_memory_bytes"] = float(peak_match.group(1))
    return metrics


def _stage_table(
    case_name: str,
    python_case: dict[str, Any],
    julia_case: dict[str, Any],
) -> str:
    lines = [
        f"### {case_name}",
        "",
        "| Stage | Python first call (s) | Python steady median (s) | Julia first call (s) | Julia steady median (s) | Steady faster |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    stages = [
        "model_load",
        "first_order_solve",
        "kalman_value",
        "kalman_per_period",
        "kalman_paths",
        "kalman_grad",
        "gate_stats",
        "switching_fixed",
        "switching_value",
        "numpyro_kalman_log_density",
        "numpyro_switching_log_density",
        "numpyro_nuts_smoke",
        "sep_inversion",
    ]
    for stage in stages:
        p = python_case["stages"].get(stage, {})
        j = julia_case["stages"].get(stage, {})
        p_status = p.get("status")
        j_status = j.get("status")
        p_first = p.get("first_call_s")
        j_first = j.get("first_call_s")
        p_med = p.get("steady", {}).get("median_s")
        j_med = j.get("steady", {}).get("median_s")
        faster = "n/a"
        if (
            p_status == "ok"
            and j_status == "ok"
            and isinstance(p_med, (int, float))
            and isinstance(j_med, (int, float))
            and p_med > 0
            and j_med > 0
        ):
            faster = "Python" if p_med < j_med else "Julia"
        lines.append(
            "| {stage} | {p_first} | {p_med} | {j_first} | {j_med} | {faster} |".format(
                stage=stage,
                p_first=(
                    "error"
                    if p_status == "error"
                    else "-" if p_first is None else f"{p_first:.6f}"
                ),
                p_med=(
                    "error"
                    if p_status == "error"
                    else "-" if p_med is None else f"{p_med:.6f}"
                ),
                j_first=(
                    "error"
                    if j_status == "error"
                    else "-" if j_first is None else f"{j_first:.6f}"
                ),
                j_med=(
                    "error"
                    if j_status == "error"
                    else "-" if j_med is None else f"{j_med:.6f}"
                ),
                faster=faster,
            )
        )
    error_notes: list[str] = []
    for stage in stages:
        p = python_case["stages"].get(stage, {})
        j = julia_case["stages"].get(stage, {})
        if p.get("status") == "error":
            error_notes.append(f"- Python `{stage}` error: {p.get('error', 'unknown error')}")
        if j.get("status") == "error":
            error_notes.append(f"- Julia `{stage}` error: {j.get('error', 'unknown error')}")
    if error_notes:
        lines.append("")
        lines.extend(error_notes)
    lines.append("")
    return "\n".join(lines)


def _stage_value(case_results: dict[str, Any], stage: str) -> float | None:
    value = case_results["stages"].get(stage, {}).get("result", {}).get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _stage_median(case_results: dict[str, Any], stage: str) -> float | None:
    value = case_results["stages"].get(stage, {}).get("steady", {}).get("median_s")
    return float(value) if isinstance(value, (int, float)) else None


def _stage_error(case_results: dict[str, Any], stage: str) -> str | None:
    error = case_results["stages"].get(stage, {}).get("error")
    return str(error) if isinstance(error, str) else None


def _stage_result(case_results: dict[str, Any], stage: str) -> dict[str, Any]:
    return case_results["stages"].get(stage, {}).get("result", {})


def _array_diff(a: Any, b: Any) -> float | None:
    try:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
    except Exception:
        return None
    if arr_a.shape != arr_b.shape:
        return None
    if arr_a.size == 0:
        return 0.0
    return float(np.max(np.abs(arr_a - arr_b)))


def _scalar_diff(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    abs_diff = abs(a - b)
    rel_diff = abs_diff / max(abs(b), 1.0)
    return abs_diff, rel_diff


def _parity_table(
    case_name: str,
    python_case: dict[str, Any],
    julia_case: dict[str, Any],
) -> str:
    python_solve = _stage_result(python_case, "first_order_solve")
    julia_solve = _stage_result(julia_case, "first_order_solve")
    python_per_period = _stage_result(python_case, "kalman_per_period")
    julia_per_period = _stage_result(julia_case, "kalman_per_period")
    python_paths = _stage_result(python_case, "kalman_paths")
    julia_paths = _stage_result(julia_case, "kalman_paths")
    python_grad = _stage_result(python_case, "kalman_grad")
    julia_grad = _stage_result(julia_case, "kalman_grad")
    python_gate = _stage_result(python_case, "gate_stats")
    julia_gate = _stage_result(julia_case, "gate_stats")

    rows = [
        ("First-order solution matrix", _array_diff(
            python_solve.get("solution_matrix"),
            julia_solve.get("solution_matrix"),
        )),
        ("Kalman loglikelihood", _scalar_diff(
            _stage_value(python_case, "kalman_value"),
            _stage_value(julia_case, "kalman_value"),
        )[0]),
        ("Kalman per-period path", _array_diff(
            python_per_period.get("per_period"),
            julia_per_period.get("per_period"),
        )),
        ("Kalman grad value", _scalar_diff(
            python_grad.get("value"),
            julia_grad.get("value"),
        )[0]),
        ("Kalman grad vector", _array_diff(
            python_grad.get("grad"),
            julia_grad.get("grad"),
        )),
        ("Filtered variables", _array_diff(
            python_paths.get("filtered_variables"),
            julia_paths.get("filtered_variables"),
        )),
        ("Smoothed variables", _array_diff(
            python_paths.get("smoothed_variables"),
            julia_paths.get("smoothed_variables"),
        )),
        ("Filtered shocks", _array_diff(
            python_paths.get("filtered_shocks"),
            julia_paths.get("filtered_shocks"),
        )),
        ("Smoothed shocks", _array_diff(
            python_paths.get("smoothed_shocks"),
            julia_paths.get("smoothed_shocks"),
        )),
        ("Gate linear observations", _array_diff(
            python_gate.get("linear_observations"),
            julia_gate.get("linear_observations"),
        )),
        ("Gate shocks", _array_diff(
            python_gate.get("shocks"),
            julia_gate.get("shocks"),
        )),
        ("Gate e-stat", _array_diff(
            python_gate.get("e_stat"),
            julia_gate.get("e_stat"),
        )),
        ("Gate f-stat", _array_diff(
            python_gate.get("f_stat"),
            julia_gate.get("f_stat"),
        )),
        ("Switching fixed-gate", _scalar_diff(
            _stage_value(python_case, "switching_fixed"),
            _stage_value(julia_case, "switching_fixed"),
        )[0]),
        ("Switching auto-gated", _scalar_diff(
            _stage_value(python_case, "switching_value"),
            _stage_value(julia_case, "switching_value"),
        )[0]),
    ]

    lines = [
        f"### {case_name} Parity",
        "",
        "| Check | Max abs diff |",
        "| --- | ---: |",
    ]
    for label, diff in rows:
        lines.append(
            f"| {label} | {'n/a' if diff is None else f'{diff:.3e}'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _report_text(
    config: dict[str, Any],
    payload: dict[str, Any],
    python_results: dict[str, Any],
    julia_results: dict[str, Any],
    python_process: dict[str, float],
    julia_process: dict[str, float],
    system_info: dict[str, str],
) -> str:
    small_case_name = payload["cases"][0]["name"]
    medium_case_name = payload["cases"][1]["name"]
    small_python = python_results["cases"][small_case_name]
    small_julia = julia_results["cases"][small_case_name]
    medium_python = python_results["cases"][medium_case_name]
    medium_julia = julia_results["cases"][medium_case_name]
    python_wall = python_process.get("real_s")
    julia_wall = julia_process.get("real_s")

    small_solution_diff = _array_diff(
        _stage_result(small_python, "first_order_solve").get("solution_matrix"),
        _stage_result(small_julia, "first_order_solve").get("solution_matrix"),
    )
    medium_solution_diff = _array_diff(
        _stage_result(medium_python, "first_order_solve").get("solution_matrix"),
        _stage_result(medium_julia, "first_order_solve").get("solution_matrix"),
    )
    small_kalman_abs, small_kalman_rel = _scalar_diff(
        _stage_value(small_python, "kalman_value"),
        _stage_value(small_julia, "kalman_value"),
    )
    medium_kalman_abs, medium_kalman_rel = _scalar_diff(
        _stage_value(medium_python, "kalman_value"),
        _stage_value(medium_julia, "kalman_value"),
    )
    small_filter_diff = _array_diff(
        _stage_result(small_python, "kalman_paths").get("filtered_variables"),
        _stage_result(small_julia, "kalman_paths").get("filtered_variables"),
    )
    medium_filter_diff = _array_diff(
        _stage_result(medium_python, "kalman_paths").get("filtered_variables"),
        _stage_result(medium_julia, "kalman_paths").get("filtered_variables"),
    )
    small_switch_fixed_abs, small_switch_fixed_rel = _scalar_diff(
        _stage_value(small_python, "switching_fixed"),
        _stage_value(small_julia, "switching_fixed"),
    )
    medium_switch_fixed_abs, medium_switch_fixed_rel = _scalar_diff(
        _stage_value(medium_python, "switching_fixed"),
        _stage_value(medium_julia, "switching_fixed"),
    )
    small_switch_auto_abs, small_switch_auto_rel = _scalar_diff(
        _stage_value(small_python, "switching_value"),
        _stage_value(small_julia, "switching_value"),
    )
    medium_switch_auto_abs, medium_switch_auto_rel = _scalar_diff(
        _stage_value(medium_python, "switching_value"),
        _stage_value(medium_julia, "switching_value"),
    )
    grad_error = _stage_error(small_python, "kalman_grad") or _stage_error(
        medium_python,
        "kalman_grad",
    )
    nuts_error = _stage_error(small_python, "numpyro_nuts_smoke") or _stage_error(
        medium_python,
        "numpyro_nuts_smoke",
    )
    small_numpyro_kalman = _stage_value(small_python, "numpyro_kalman_log_density")
    small_numpyro_switching = _stage_value(small_python, "numpyro_switching_log_density")

    report_lines = [
        "# Python/JAX vs Julia Profile",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## Environment",
        "",
        f"- CPU: {system_info['cpu_brand']}",
        f"- Logical cores: {system_info['logical_cpu']}",
        f"- Performance cores: {system_info['perf_cores']}",
        f"- Efficiency cores: {system_info['eff_cores']}",
        f"- Memory (bytes): {system_info['mem_bytes']}",
        f"- JAX devices: {', '.join(python_results['cases'][payload['cases'][0]['name']]['model_info']['jax_devices'])}",
        f"- Julia version: {julia_results['julia_version']}",
        f"- Threads requested: {config['global']['thread_count']}",
        f"- JAX platform setting: {config['global'].get('jax_platform_name', 'auto')}",
        "",
        "## Scope",
        "",
        "- Small model: `FS2000`.",
        "- Medium model: `Smets_Wouters_2007_HLT`.",
        "- Shared synthetic observations are generated once from the reference first-order solution and reused in both environments.",
        "- The report validates solution parity first, then Kalman likelihood/filter parity, then gate/switching parity, and only then compares timings.",
        "- Important caveat: the Python HLT benchmark still uses a Julia-exported reference steady state so the comparison isolates already-ported solve/filter code instead of cold-start steady-state recovery.",
        "",
        "## Key Findings",
        "",
    ]
    if isinstance(python_wall, float) and isinstance(julia_wall, float) and python_wall > 0:
        report_lines.append(
            f"- Whole-process wall time: Python finished in {python_wall:.2f}s and Julia in {julia_wall:.2f}s, so Julia/Python = {julia_wall / python_wall:.2f}x."
        )
    if small_solution_diff is not None:
        report_lines.append(
            f"- Small-model first-order solution parity max abs diff: {small_solution_diff:.3e}."
        )
    if medium_solution_diff is not None:
        report_lines.append(
            f"- Medium-model first-order solution parity max abs diff: {medium_solution_diff:.3e}."
        )
    if small_kalman_abs is not None and small_kalman_rel is not None:
        report_lines.append(
            f"- Small-model Kalman total loglikelihood abs/rel diff: {small_kalman_abs:.3e} / {small_kalman_rel:.3e}."
        )
    if medium_kalman_abs is not None and medium_kalman_rel is not None:
        report_lines.append(
            f"- Medium-model Kalman total loglikelihood abs/rel diff: {medium_kalman_abs:.3e} / {medium_kalman_rel:.3e}."
        )
    if small_filter_diff is not None:
        report_lines.append(
            f"- Small-model filtered-variable path parity max abs diff: {small_filter_diff:.3e}."
        )
    if medium_filter_diff is not None:
        report_lines.append(
            f"- Medium-model filtered-variable path parity max abs diff: {medium_filter_diff:.3e}."
        )
    if small_switch_fixed_abs is not None and small_switch_fixed_rel is not None:
        report_lines.append(
            f"- Small-model fixed-gate switching abs/rel diff: {small_switch_fixed_abs:.3e} / {small_switch_fixed_rel:.3e}."
        )
    if medium_switch_fixed_abs is not None and medium_switch_fixed_rel is not None:
        report_lines.append(
            f"- Medium-model fixed-gate switching abs/rel diff: {medium_switch_fixed_abs:.3e} / {medium_switch_fixed_rel:.3e}."
        )
    if small_switch_auto_abs is not None and small_switch_auto_rel is not None:
        report_lines.append(
            f"- Small-model automatic switching abs/rel diff: {small_switch_auto_abs:.3e} / {small_switch_auto_rel:.3e}."
        )
    if medium_switch_auto_abs is not None and medium_switch_auto_rel is not None:
        report_lines.append(
            f"- Medium-model automatic switching abs/rel diff: {medium_switch_auto_abs:.3e} / {medium_switch_auto_rel:.3e}."
        )
    if grad_error is not None:
        report_lines.append(
            f"- Python/JAX reverse-mode likelihood differentiation still failed during the benchmark with `{grad_error}`."
        )
    if small_numpyro_kalman is not None and small_numpyro_switching is not None:
        report_lines.append(
            f"- Small-model NumPyro/JAX log densities evaluated successfully: Kalman {small_numpyro_kalman:.6f}, switching {small_numpyro_switching:.6f}."
        )
    if nuts_error is not None:
        report_lines.append(
            f"- NumPyro NUTS smoke did not complete on the benchmark DSGE model: `{nuts_error}`."
        )
    report_lines.extend(
        [
            "- JAX only exposed `TFRT_CPU_0` in this environment, so this remains a CPU benchmark rather than a live GPU benchmark.",
            "- SEP remains a bounded robustness smoke stage here; it is still reported for runtime coverage, not as a validated matched-likelihood parity stage on these large models.",
            "",
            "## Whole Process",
            "",
            f"- Python wall/user/sys: {python_process.get('real_s', float('nan')):.3f}s / {python_process.get('user_s', float('nan')):.3f}s / {python_process.get('sys_s', float('nan')):.3f}s",
            f"- Python max RSS (raw `time -l` units): {python_process.get('max_rss_raw', float('nan')):.0f}",
            f"- Python peak memory footprint: {python_process.get('peak_memory_bytes', float('nan')) / (1024.0 ** 2):.1f} MiB",
            f"- Julia wall/user/sys: {julia_process.get('real_s', float('nan')):.3f}s / {julia_process.get('user_s', float('nan')):.3f}s / {julia_process.get('sys_s', float('nan')):.3f}s",
            f"- Julia max RSS (raw `time -l` units): {julia_process.get('max_rss_raw', float('nan')):.0f}",
            f"- Julia peak memory footprint: {julia_process.get('peak_memory_bytes', float('nan')) / (1024.0 ** 2):.1f} MiB",
            "",
            "## Parity Results",
            "",
        ]
    )
    for case in payload["cases"]:
        report_lines.append(
            _parity_table(
                case["name"],
                python_results["cases"][case["name"]],
                julia_results["cases"][case["name"]],
            )
        )
    report_lines.extend(
        [
            "## Stage Timings",
            "",
        ]
    )
    for case in payload["cases"]:
        report_lines.append(
            _stage_table(
                case["name"],
                python_results["cases"][case["name"]],
                julia_results["cases"][case["name"]],
            )
        )
    report_lines.extend(
        [
            "## Interpretation",
            "",
            "- `first_call` includes compilation and one-off setup overhead.",
            "- `steady median` is the more relevant figure for repeated estimation inner loops.",
            "- `kalman_grad` still uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.",
            "- Fixed-gate switching isolates the likelihood mixer on shared gates; automatic switching also exercises gate reconstruction and filtering in each environment.",
            "- NumPyro stages use the Python/JAX likelihood wrappers with calibrated-parameter-centered benchmark priors; Julia has no matching NumPyro stage, so those rows validate JAX+NumPyro runtime coverage rather than cross-language sampler parity.",
            "- The parity tables should be read before the timing tables. Any stage with weak parity should not be used to make strong runtime claims.",
        ]
    )
    return "\n".join(report_lines) + "\n"


def main() -> None:
    config = _load_config(CONFIG_PATH)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    results_dir = BENCH_DIR / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    reference_state_path = results_dir / "reference_steady_states.json"
    payload_path = results_dir / "validation_payloads.json"
    python_results_path = results_dir / "python_results.json"
    julia_results_path = results_dir / "julia_results.json"
    python_time_path = results_dir / "python_time.txt"
    julia_time_path = results_dir / "julia_time.txt"
    report_path = Path(config["global"]["report_path"])
    combined_path = results_dir / "combined_results.json"

    export_cmd = [
        str(config["global"]["julia_executable"]),
        f"--project={config['global']['upstream_repo']}",
        str(BENCH_DIR / "export_reference_steady_states.jl"),
        str(CONFIG_PATH),
        str(reference_state_path),
    ]
    _run(export_cmd)
    payload = _build_payloads(config, reference_state_path, payload_path)

    thread_count = str(config["global"]["thread_count"])
    python_env = os.environ.copy()
    python_env.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
            "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={thread_count}",
        }
    )
    jax_platform_name = str(config["global"].get("jax_platform_name", "auto"))
    if jax_platform_name and jax_platform_name != "auto":
        python_env["JAX_PLATFORM_NAME"] = jax_platform_name
    python_cmd = [
        str(config["global"]["python_executable"]),
        str(BENCH_DIR / "profile_validation_python.py"),
        str(payload_path),
        str(python_results_path),
    ]
    _timed_process(python_cmd, env=python_env, time_output_path=python_time_path)

    julia_env = os.environ.copy()
    julia_env.update(
        {
            "JULIA_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
        }
    )
    julia_cmd = [
        str(config["global"]["julia_executable"]),
        f"--project={config['global']['upstream_repo']}",
        str(BENCH_DIR / "profile_validation_julia.jl"),
        str(payload_path),
        str(julia_results_path),
    ]
    _timed_process(julia_cmd, env=julia_env, time_output_path=julia_time_path)

    python_results = json.loads(python_results_path.read_text())
    julia_results = json.loads(julia_results_path.read_text())
    python_process = _parse_time_output(python_time_path)
    julia_process = _parse_time_output(julia_time_path)
    system_info = _system_info()

    combined = {
        "config_path": str(CONFIG_PATH),
        "payload_path": str(payload_path),
        "python_results": python_results,
        "julia_results": julia_results,
        "python_process": python_process,
        "julia_process": julia_process,
        "system_info": system_info,
    }
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True))

    report_text = _report_text(
        config,
        payload,
        python_results,
        julia_results,
        python_process,
        julia_process,
        system_info,
    )
    report_path.write_text(report_text)
    print(str(report_path))
    print(str(combined_path))


if __name__ == "__main__":
    main()
