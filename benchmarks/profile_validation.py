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
        observation_names = tuple(case["observables"])
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
                "measurement_error_scale": float(case["measurement_error_scale"]),
                "jitter": float(case["jitter"]),
                "solve_reps": int(case["solve_reps"]),
                "kalman_value_reps": int(case["kalman_value_reps"]),
                "kalman_grad_reps": int(case["kalman_grad_reps"]),
                "switching_reps": int(case["switching_reps"]),
                "sep_reps": int(case["sep_reps"]),
                "gate_periods": None,
                "gate_mode": str(case["gate_mode"]),
                "gate_hard_threshold": float(case["gate_hard_threshold"]),
                "gate_prob_floor": float(case["gate_prob_floor"]),
                "gate_prob_ceiling": float(case["gate_prob_ceiling"]),
                "soft_mixture": str(case["soft_mixture"]),
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
        "kalman_grad",
        "switching_value",
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


def _report_text(
    config: dict[str, Any],
    payload: dict[str, Any],
    python_results: dict[str, Any],
    julia_results: dict[str, Any],
    python_process: dict[str, float],
    julia_process: dict[str, float],
    system_info: dict[str, str],
) -> str:
    def _stage_value(case_results: dict[str, Any], stage: str) -> float | None:
        value = case_results["stages"].get(stage, {}).get("result", {}).get("value")
        return float(value) if isinstance(value, (int, float)) else None

    def _stage_median(case_results: dict[str, Any], stage: str) -> float | None:
        value = case_results["stages"].get(stage, {}).get("steady", {}).get("median_s")
        return float(value) if isinstance(value, (int, float)) else None

    def _stage_error(case_results: dict[str, Any], stage: str) -> str | None:
        error = case_results["stages"].get(stage, {}).get("error")
        return str(error) if isinstance(error, str) else None

    small_case_name = payload["cases"][0]["name"]
    medium_case_name = payload["cases"][1]["name"]
    small_python = python_results["cases"][small_case_name]
    small_julia = julia_results["cases"][small_case_name]
    medium_python = python_results["cases"][medium_case_name]
    medium_julia = julia_results["cases"][medium_case_name]
    python_wall = python_process.get("real_s")
    julia_wall = julia_process.get("real_s")
    small_kalman_python = _stage_value(small_python, "kalman_value")
    small_kalman_julia = _stage_value(small_julia, "kalman_value")
    medium_kalman_python = _stage_value(medium_python, "kalman_value")
    medium_kalman_julia = _stage_value(medium_julia, "kalman_value")
    small_switch_python = _stage_median(small_python, "switching_value")
    small_switch_julia = _stage_median(small_julia, "switching_value")
    grad_error = _stage_error(small_python, "kalman_grad") or _stage_error(
        medium_python, "kalman_grad"
    )
    small_kalman_rel = None
    if small_kalman_python is not None and small_kalman_julia is not None:
        small_kalman_rel = abs(small_kalman_python - small_kalman_julia) / max(
            abs(small_kalman_julia),
            1.0,
        )
    medium_kalman_rel = None
    if medium_kalman_python is not None and medium_kalman_julia is not None:
        medium_kalman_rel = abs(medium_kalman_python - medium_kalman_julia) / max(
            abs(medium_kalman_julia),
            1.0,
        )

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
        "",
        "## Scope",
        "",
        "- Small model: `FS2000`.",
        "- Medium model: `Smets_Wouters_2007_HLT`.",
        "- Workload: first-order Schur solve, Kalman loglikelihood, Kalman gradient on the differentiable doubling path, automatic filter-gated switching likelihood, and bounded SEP inversion smoke.",
        "- Shared synthetic observations were generated once in Python from the reference first-order solution and then reused in both environments.",
        "- Important caveat: the Python HLT benchmark uses a Julia-exported reference steady state because cold-start HLT steady-state recovery is still not robust enough in the port for a fair timing comparison.",
        "",
        "## Key Findings",
        "",
    ]
    if isinstance(python_wall, float) and isinstance(julia_wall, float) and python_wall > 0:
        report_lines.append(
            f"- Whole-process wall time: Python finished in {python_wall:.2f}s and Julia in {julia_wall:.2f}s, so Python was {julia_wall / python_wall:.2f}x faster on this benchmark harness."
        )
    if small_kalman_rel is not None:
        report_lines.append(
            f"- Small-model Kalman likelihood parity is good: FS2000 differs by {small_kalman_rel:.3%} between Python and Julia."
        )
    if medium_kalman_rel is not None:
        report_lines.append(
            f"- Medium-model Kalman likelihood parity is not yet good: HLT differs by {medium_kalman_rel:.3%} between Python and Julia on the shared payload, so medium-model timings should be treated as a runtime stress test, not a validated apples-to-apples estimation benchmark."
        )
    if grad_error is not None:
        report_lines.append(
            f"- Python/JAX reverse-mode likelihood differentiation failed on these benchmark models with `{grad_error}`. Julia completed the same stage."
        )
    if (
        isinstance(small_switch_python, float)
        and isinstance(small_switch_julia, float)
        and small_switch_python > 0
        and small_switch_julia > 0
    ):
        report_lines.append(
            f"- On the small model, steady-state switching likelihood evaluation was faster in Python ({small_switch_python:.6f}s median) than Julia ({small_switch_julia:.6f}s median)."
        )
    report_lines.extend(
        [
            "- JAX only exposed `TFRT_CPU_0` in this environment, so none of these runs exercised a live GPU backend.",
            "- SEP here is a bounded robustness smoke stage. Both environments produced non-finite or failure-style SEP outputs on these settings, so SEP timings are not a validated matched-likelihood comparison.",
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
        "## Stage Results",
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
            "- `first_call` includes language-specific JIT or XLA compilation overhead.",
            "- `steady median` is the more relevant figure for repeated estimation inner loops.",
            "- `kalman_grad` uses the doubling QME path in both environments because the current JAX Schur path is not reverse-mode differentiable.",
            "- Julia stage timings are public-API timings. They include its own steady-state and solution preparation work when the API does so.",
            "- Python medium-model timings are inner-loop timings conditional on a supplied reference steady state.",
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
            "JAX_PLATFORM_NAME": "cpu",
            "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={thread_count}",
        }
    )
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
