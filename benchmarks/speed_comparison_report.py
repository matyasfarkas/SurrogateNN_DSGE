from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "benchmarks" / "results"
DEFAULT_VALIDATION_DIR = DEFAULT_RESULTS_ROOT / "20260712T083357"
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_ROOT / "speed_comparison_report.md"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_ROOT / "speed_comparison_report.json"


def _safe_get(mapping: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    if 0.0 < abs(number) < 0.001:
        return f"{number:.2e}"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def _runtime_label(runtime: Mapping[str, Any] | None) -> str:
    if not runtime:
        return ""
    backend = runtime.get("jax_default_backend") or runtime.get("backend")
    platform = str(runtime.get("platform", "")).split("-", maxsplit=1)[0]
    devices = runtime.get("jax_devices") or []
    if backend:
        return f"{platform} / JAX {backend}"
    if devices:
        return f"{platform} / {devices[0]}"
    return platform


def _latest_validation_dir(results_root: Path) -> Path | None:
    candidates = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and (path / "julia_results.json").exists()
        and (path / "python_results.json").exists()
    )
    return candidates[-1] if candidates else None


def _stage_rows(validation_dir: Path | None, case: str) -> list[dict[str, Any]]:
    if validation_dir is None:
        return []
    rows: list[dict[str, Any]] = []
    for language, filename in (("Julia", "julia_results.json"), ("Python/JAX", "python_results.json")):
        path = validation_dir / filename
        if not path.exists():
            continue
        data = _load_json(path)
        stages = _safe_get(data, "cases", case, "stages")
        if not isinstance(stages, Mapping):
            continue
        for stage_name in (
            "first_order_solve",
            "kalman_value",
            "kalman_grad",
            "switching_fixed",
            "switching_value",
            "sep_inversion",
        ):
            stage = stages.get(stage_name)
            if not isinstance(stage, Mapping):
                continue
            steady = stage.get("steady") if isinstance(stage.get("steady"), Mapping) else {}
            rows.append(
                {
                    "language": language,
                    "stage": stage_name,
                    "status": stage.get("status"),
                    "first_call_s": stage.get("first_call_s"),
                    "steady_median_s": steady.get("median_s"),
                    "steady_mean_s": steady.get("mean_s"),
                    "steady_reps": steady.get("reps"),
                    "value": _safe_get(stage, "result", "value"),
                    "source": str(path),
                }
            )
    return rows


def _posterior_row(path: Path, data: Mapping[str, Any]) -> dict[str, Any] | None:
    throughput = data.get("throughput")
    benchmark = data.get("benchmark")
    if not isinstance(throughput, Mapping) or not isinstance(benchmark, Mapping):
        return None
    diagnostics = data.get("posterior_diagnostics")
    extra_fields = data.get("extra_fields")
    accept_prob = _safe_get(extra_fields, "accept_prob", "mean")
    kernel = str(benchmark.get("kernel") or "nuts").upper()
    return {
        "source": str(path),
        "environment": _runtime_label(data.get("runtime")),
        "sampler": f"NumPyro {kernel}",
        "chains": benchmark.get("chains"),
        "warmup": benchmark.get("warmup"),
        "samples": benchmark.get("samples"),
        "post_warmup_draws": throughput.get("post_warmup_draws"),
        "periods": benchmark.get("periods"),
        "parameters": benchmark.get("parameter_count"),
        "wall_s": throughput.get("sampling_wall_s"),
        "draws_per_second": throughput.get("draws_per_second"),
        "min_ess": throughput.get("min_ess"),
        "mean_ess": throughput.get("mean_ess"),
        "seconds_per_min_ess": throughput.get("seconds_per_min_ess"),
        "max_r_hat": _safe_get(diagnostics, "max_r_hat"),
        "accept_prob_mean": accept_prob,
        "step_size": benchmark.get("hmc_step_size"),
        "timing_scope": "full MCMC run",
    }


def _static_rows(path: Path, data: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmark = data.get("benchmark")
    runs = data.get("step_size_runs")
    if not isinstance(benchmark, Mapping):
        return []
    if not isinstance(runs, list):
        runs = [
            {
                "initial_step_size": benchmark.get("initial_step_size"),
                "first_run": data.get("cold_run"),
                "steady_last_run": data.get("steady_last_run"),
            }
        ]
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            continue
        steady = run.get("steady_last_run")
        first = run.get("first_run")
        selected = steady if isinstance(steady, Mapping) else first
        if not isinstance(selected, Mapping):
            continue
        diagnostics = selected.get("posterior_diagnostics")
        acceptance = selected.get("acceptance")
        rows.append(
            {
                "source": str(path),
                "environment": _runtime_label(data.get("runtime")),
                "sampler": "JAX static HMC",
                "chains": benchmark.get("chains"),
                "warmup": benchmark.get("warmup"),
                "samples": benchmark.get("samples"),
                "post_warmup_draws": selected.get("post_warmup_draws"),
                "periods": benchmark.get("periods"),
                "parameters": benchmark.get("parameter_count"),
                "wall_s": selected.get("timing_s"),
                "draws_per_second": selected.get("draws_per_second"),
                "min_ess": selected.get("min_ess"),
                "mean_ess": selected.get("mean_ess"),
                "seconds_per_min_ess": selected.get("seconds_per_min_ess"),
                "max_r_hat": _safe_get(diagnostics, "max_r_hat"),
                "accept_prob_mean": _safe_get(acceptance, "accept_prob", "mean"),
                "step_size": run.get("initial_step_size"),
                "timing_scope": "steady replay" if isinstance(steady, Mapping) else (
                    "cold compile+run" if index == 0 else "post-compile run"
                ),
            }
        )
    return rows


def _posterior_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        row = _posterior_row(path, data)
        if row is not None:
            rows.append(row)
        rows.extend(_static_rows(path, data))
    return rows


def _batched_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.exists() or path.suffix != ".json":
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        benchmark = data.get("benchmark")
        batches = data.get("batches")
        if not isinstance(benchmark, Mapping) or not isinstance(batches, Mapping):
            continue
        for batch_size, payload in batches.items():
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "source": str(path),
                    "environment": _runtime_label(data.get("runtime")),
                    "batch_size": int(batch_size),
                    "periods": benchmark.get("periods"),
                    "parameters": benchmark.get("parameter_count"),
                    "value_draws_per_second": payload.get("value_draws_per_second_median"),
                    "gradient_draws_per_second": payload.get("gradient_draws_per_second_median"),
                    "value_first_call_s": payload.get("value_first_call_s"),
                    "gradient_first_call_s": payload.get("gradient_first_call_s"),
                }
            )
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            cells.append(_format_number(value) if isinstance(value, (int, float)) else str(value or ""))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _best_static_gpu_row(posterior_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in posterior_rows
        if row.get("sampler") == "JAX static HMC"
        and "gpu" in str(row.get("environment", "")).lower()
        and isinstance(row.get("seconds_per_min_ess"), (int, float))
    ]
    return min(candidates, key=lambda row: float(row["seconds_per_min_ess"])) if candidates else None


def _same_benchmark_shape(row: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return (
        row.get("periods") == reference.get("periods")
        and row.get("parameters") == reference.get("parameters")
    )


def _best_m4_numpyro_row(
    posterior_rows: Sequence[Mapping[str, Any]],
    *,
    reference: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in posterior_rows
        if str(row.get("sampler", "")).startswith("NumPyro")
        and "cpu" in str(row.get("environment", "")).lower()
        and isinstance(row.get("seconds_per_min_ess"), (int, float))
    ]
    if reference is not None:
        same_shape = [row for row in candidates if _same_benchmark_shape(row, reference)]
        if same_shape:
            candidates = same_shape
    return min(candidates, key=lambda row: float(row["seconds_per_min_ess"])) if candidates else None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = args.validation_dir
    if validation_dir is None:
        validation_dir = _latest_validation_dir(args.results_root)
    result_paths = list(args.results_root.rglob("*.json"))
    if args.extra_result:
        result_paths.extend(args.extra_result)
    stage_rows = _stage_rows(validation_dir, args.case)
    posterior_rows = _posterior_rows(result_paths)
    batched_rows = _batched_rows(result_paths)
    best_gpu = _best_static_gpu_row(posterior_rows)
    return {
        "validation_dir": str(validation_dir) if validation_dir is not None else None,
        "case": args.case,
        "stage_rows": stage_rows,
        "posterior_rows": posterior_rows,
        "batched_likelihood_rows": batched_rows,
        "best_gpu_static_hmc": best_gpu,
        "best_m4_numpyro": _best_m4_numpyro_row(posterior_rows, reference=best_gpu),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    posterior_rows = sorted(
        report["posterior_rows"],
        key=lambda row: (
            str(row.get("sampler", "")),
            int(row.get("chains") or 0),
            float(row.get("step_size") or 0.0),
        ),
    )
    stage_rows = [
        row
        for row in report["stage_rows"]
        if row.get("stage")
        in {"first_order_solve", "kalman_value", "kalman_grad", "switching_fixed", "switching_value"}
    ]
    batched_rows = sorted(
        report["batched_likelihood_rows"],
        key=lambda row: (int(row.get("batch_size") or 0), str(row.get("environment", ""))),
    )
    best_gpu = report.get("best_gpu_static_hmc")
    best_cpu = report.get("best_m4_numpyro")
    ratio_text = "Not available."
    if isinstance(best_gpu, Mapping) and isinstance(best_cpu, Mapping):
        gpu_spme = best_gpu.get("seconds_per_min_ess")
        cpu_spme = best_cpu.get("seconds_per_min_ess")
        gpu_dps = best_gpu.get("draws_per_second")
        cpu_dps = best_cpu.get("draws_per_second")
        if isinstance(gpu_spme, (int, float)) and isinstance(cpu_spme, (int, float)):
            ess_ratio = float(cpu_spme) / float(gpu_spme)
            ratio_text = (
                f"Best measured GPU static-HMC is {_format_number(ess_ratio, 2)}x faster "
                "than the best measured M4 NumPyro sampler on seconds per minimum ESS."
            )
            if isinstance(gpu_dps, (int, float)) and isinstance(cpu_dps, (int, float)):
                ratio_text += (
                    f" Raw posterior draws/sec ratio is "
                    f"{_format_number(float(gpu_dps) / float(cpu_dps), 2)}x."
                )
    lines = [
        "# DSGE Speed Comparison",
        "",
        f"Validation directory: `{report.get('validation_dir')}`",
        "",
        "## Scope",
        "",
        "- Julia numbers are stage/profile timings from the MacroModelling validation harness; a Julia Turing/AdvancedHMC posterior ESS/sec sweep has not been measured yet.",
        "- NumPyro rows are full sampler wall times including JAX compilation and warmup for the saved runs.",
        "- JAX static-HMC rows use the fixed-shape chain-parallel benchmark. For step-size grids, rows after the first are post-compile runs with the same compiled shape.",
        "- ESS diagnostics from very short chains are noisy; use seconds per minimum ESS only as a screening metric until longer chains are run.",
        "",
        "## Main Result",
        "",
        ratio_text,
        "",
        "## Stage Timings",
        "",
        _markdown_table(
            stage_rows,
            (
                ("Language", "language"),
                ("Stage", "stage"),
                ("Status", "status"),
                ("First call s", "first_call_s"),
                ("Steady median s", "steady_median_s"),
                ("Steady reps", "steady_reps"),
            ),
        ),
        "",
        "## Posterior Samplers",
        "",
        _markdown_table(
            posterior_rows,
            (
                ("Environment", "environment"),
                ("Sampler", "sampler"),
                ("Chains", "chains"),
                ("Warmup", "warmup"),
                ("Samples", "samples"),
                ("Periods", "periods"),
                ("Params", "parameters"),
                ("Step size", "step_size"),
                ("Timing scope", "timing_scope"),
                ("Wall s", "wall_s"),
                ("Draws/s", "draws_per_second"),
                ("Min ESS", "min_ess"),
                ("s/min ESS", "seconds_per_min_ess"),
                ("Max Rhat", "max_r_hat"),
                ("Accept", "accept_prob_mean"),
            ),
        ),
        "",
        "## Batched Likelihood / Gradient",
        "",
        _markdown_table(
            batched_rows,
            (
                ("Environment", "environment"),
                ("Batch", "batch_size"),
                ("Periods", "periods"),
                ("Params", "parameters"),
                ("Value evals/s", "value_draws_per_second"),
                ("Gradient evals/s", "gradient_draws_per_second"),
                ("Value first s", "value_first_call_s"),
                ("Gradient first s", "gradient_first_call_s"),
            ),
        ),
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Julia/Python/JAX DSGE speed benchmark JSON files."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--extra-result", type=Path, action="append", default=[])
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = build_report(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True))
    markdown = render_markdown(report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
