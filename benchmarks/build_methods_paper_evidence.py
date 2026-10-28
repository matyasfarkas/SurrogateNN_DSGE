#!/usr/bin/env python3
"""Build an auditable paper-evidence snapshot from a Galí GPU suite.

The builder deliberately distinguishes three objects that are easy to conflate:

1. nonlinear SEP targets used to train and validate the residual surrogate;
2. the parameter-dependent, full-JAX surrogate profile inversion objective; and
3. an exact nonlinear SEP likelihood, which the current pilot does not evaluate
   inside HMC.

The generated files are suitable for inclusion in the methods-paper working
draft.  They never promote a smoke/resource run to publication evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-methods-paper-evidence/v2"
EXPECTED_MODEL = "Gali_2015_chapter_3_nonlinear"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}.")
    return result


def _integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    return int(value)


def _optional_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field=field)


def _find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {name!r} below {root}, found {len(paths)}.")
    return paths[0]


def _raw_run(result_path: Path) -> Mapping[str, Any]:
    result = _read_json(result_path)
    try:
        run = result["results"]["hlt_fixed_ss_smoke"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing results.hlt_fixed_ss_smoke in {result_path}.") from exc
    if not isinstance(run, Mapping):
        raise ValueError(f"Invalid run payload in {result_path}.")
    if run.get("model_name") != EXPECTED_MODEL:
        raise ValueError(
            f"Unexpected model in {result_path}: {run.get('model_name')!r}; "
            f"expected {EXPECTED_MODEL!r}."
        )
    if run.get("equation_count") != 23 or run.get("parameter_count") != 16:
        raise ValueError(
            f"The run in {result_path} is not the declared full Galí equation set."
        )
    return run


def _parse_nvidia_smi_snapshot(text: str, *, field: str) -> dict[str, Any]:
    """Parse the archived point-in-time human-readable NVIDIA snapshot."""

    header = re.search(
        r"Driver Version:\s*([^\s]+)\s+CUDA Version:\s*([^\s]+)", text
    )
    gpu_line = re.search(r"\|\s*0\s+(NVIDIA.+?)\s+(?:On|Off)\s+\|", text)
    metrics = re.search(
        r"\|\s*\d+%\s+(\d+)C\s+\S+\s+(\d+)W\s*/\s*(\d+)W\s*\|"
        r"\s*(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%",
        text,
    )
    if header is None or gpu_line is None or metrics is None:
        raise ValueError(f"Could not parse archived nvidia-smi snapshot for {field}.")
    return {
        "telemetry_kind": "point_in_time_start_snapshot",
        "gpu_index": 0,
        "gpu_name_as_reported": gpu_line.group(1).strip(),
        "driver_version": header.group(1),
        "cuda_version": header.group(2),
        "temperature_c": int(metrics.group(1)),
        "power_draw_w": int(metrics.group(2)),
        "power_limit_w": int(metrics.group(3)),
        "memory_used_mib": int(metrics.group(4)),
        "memory_total_mib": int(metrics.group(5)),
        "gpu_utilization_percent": int(metrics.group(6)),
    }


def _gpu_environment(result_path: Path) -> dict[str, Any]:
    result = _read_json(result_path)
    environment = result.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"Missing environment object in {result_path}.")
    if environment.get("jax_default_backend") != "gpu":
        raise ValueError(f"Archived run did not use the JAX GPU backend: {result_path}.")
    devices = environment.get("jax_devices")
    if not isinstance(devices, list) or not devices or not all(
        str(device).startswith("cuda:") for device in devices
    ):
        raise ValueError(f"Archived run has invalid CUDA device metadata: {result_path}.")
    if environment.get("jax_enable_x64") is not True:
        raise ValueError(f"Archived GPU run did not enable JAX 64-bit mode: {result_path}.")
    snapshot_text = environment.get("nvidia_smi")
    if not isinstance(snapshot_text, str) or not snapshot_text.strip():
        raise ValueError(f"Archived run has no nvidia-smi snapshot: {result_path}.")
    return {
        "jax_version": str(environment.get("jax")),
        "jax_default_backend": "gpu",
        "jax_devices": [str(device) for device in devices],
        "jax_enable_x64": True,
        "xla_preallocate": str(environment.get("xla_preallocate")),
        "xla_mem_fraction": str(environment.get("xla_mem_fraction")),
        "snapshot": _parse_nvidia_smi_snapshot(
            snapshot_text, field=str(result_path)
        ),
    }


def _run_metrics(label: str, summary: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    hmc = raw.get("surrogate_hmc", {})
    jax_density = raw.get("jax_surrogate_log_density", {})
    target = raw.get("target_diagnostics", {})
    gate = raw.get("quality_gate", {})
    if not all(isinstance(item, Mapping) for item in (hmc, jax_density, target, gate)):
        raise ValueError(f"Malformed diagnostics for run {label!r}.")
    sep_config = target.get("sep_config", {})
    if not isinstance(sep_config, Mapping):
        raise ValueError(f"Malformed SEP configuration for run {label!r}.")

    metrics = {
        "label": label,
        "status": str(summary.get("status")),
        "quality_gate": str(summary.get("quality_gate")),
        "likelihood_runtime_mode": str(summary.get("likelihood_runtime_mode")),
        "theta_draws": _integer(summary.get("theta_draws"), field=f"{label}.theta_draws"),
        "periods": _integer(summary.get("periods"), field=f"{label}.periods"),
        "accepted_targets": _integer(
            summary.get("accepted_targets"), field=f"{label}.accepted_targets"
        ),
        "full_paths": _integer(summary.get("full_paths"), field=f"{label}.full_paths"),
        "max_sep_residual": _finite(
            summary.get("max_sep_residual"), field=f"{label}.max_sep_residual"
        ),
        "validation_rmse_mean": _finite(
            summary.get("validation_rmse_mean"), field=f"{label}.validation_rmse_mean"
        ),
        "validation_improvement_mean": _finite(
            summary.get("validation_improvement_mean"),
            field=f"{label}.validation_improvement_mean",
        ),
        "jax_parity_abs_diff": _finite(
            summary.get("jax_parity_abs_diff"), field=f"{label}.jax_parity_abs_diff"
        ),
        "hmc_chains": _integer(summary.get("hmc_chains"), field=f"{label}.hmc_chains"),
        "hmc_warmup": _integer(summary.get("hmc_warmup"), field=f"{label}.hmc_warmup"),
        "hmc_samples": _integer(summary.get("hmc_samples"), field=f"{label}.hmc_samples"),
        "hmc_accepted_share": _finite(
            summary.get("hmc_accepted_share"), field=f"{label}.hmc_accepted_share"
        ),
        "hmc_draws_per_second": _finite(
            summary.get("hmc_draws_per_second"), field=f"{label}.hmc_draws_per_second"
        ),
        "hmc_elapsed_s": _finite(
            summary.get("hmc_elapsed_s"), field=f"{label}.hmc_elapsed_s"
        ),
        "jax_value_elapsed_s": _finite(
            summary.get("jax_value_elapsed_s"), field=f"{label}.jax_value_elapsed_s"
        ),
        "target_evaluations": _optional_integer(
            summary.get("target_evaluations"), field=f"{label}.target_evaluations"
        ),
        "exact_endpoint_correction": bool(summary.get("exact_endpoint_correction")),
        "target_gradient_evaluated": summary.get("target_gradient_evaluated"),
        "pipeline_s": _finite(summary.get("pipeline_s"), field=f"{label}.pipeline_s"),
        "target_builder": str(target.get("builder")),
        "sep_branching_order": _integer(
            sep_config.get("branching_order"), field=f"{label}.sep_branching_order"
        ),
        "sep_nodes": _integer(sep_config.get("nnodes"), field=f"{label}.sep_nodes"),
        "likelihood_kind": "surrogate_profile_inversion_objective",
        "full_jax_recomputes_parameter_runtime": (
            summary.get("likelihood_runtime_mode") == "full-jax"
        ),
        "full_nonlinear_likelihood_in_hmc": False,
        "jax_python_comparison_scope": "same surrogate profile objective implementation",
        "hmc_kind": str(hmc.get("kind")),
        "raw_quality_gate_status": str(gate.get("status")),
        "jax_density_caveat": str(jax_density.get("caveat", "")),
        "jax_backend": str(jax_density.get("backend", "")),
        "qme_algorithm": str(jax_density.get("qme_algorithm", "")),
        "gpu_native_qme": str(jax_density.get("qme_algorithm", "")) == "schur_gpu",
    }
    if metrics["status"] != "ok" or metrics["quality_gate"] != "ok":
        raise ValueError(f"Run {label!r} is not a successful pilot run.")
    return metrics


def build_evidence(suite_root: Path) -> dict[str, Any]:
    """Return a deterministic, claim-scoped evidence record."""

    suite_root = suite_root.resolve()
    summary_path = _find_one(suite_root, "scientific_summary.json")
    summary = _read_json(summary_path)
    if summary.get("model") != EXPECTED_MODEL:
        raise ValueError(
            f"Unexpected suite model {summary.get('model')!r}; expected {EXPECTED_MODEL!r}."
        )
    summary_runs = summary.get("runs")
    if not isinstance(summary_runs, Mapping):
        raise ValueError("scientific_summary.json is missing its runs object.")

    run_paths = {path.parent.name: path for path in sorted(suite_root.glob("gali_*/*result.json"))}
    label_to_dir_fragment = {
        "quick": "gali_quick_",
        "corrected_full_jax_resource": "gali_full_jax_resource_",
        "default": "gali_default_",
    }
    runs: dict[str, Any] = {}
    sources = [
        {
            "path": str(summary_path.relative_to(suite_root)),
            "sha256": _sha256(summary_path),
        }
    ]
    for label, fragment in label_to_dir_fragment.items():
        matches = [(name, path) for name, path in run_paths.items() if name.startswith(fragment)]
        if len(matches) != 1:
            raise ValueError(f"Expected one result for {label!r}, found {len(matches)}.")
        _, result_path = matches[0]
        raw = _raw_run(result_path)
        run_summary = summary_runs.get(label)
        if not isinstance(run_summary, Mapping):
            raise ValueError(f"Missing summary entry for {label!r}.")
        runs[label] = _run_metrics(label, run_summary, raw)
        runs[label]["gpu_environment"] = _gpu_environment(result_path)
        sources.append(
            {
                "path": str(result_path.resolve().relative_to(suite_root)),
                "sha256": _sha256(result_path),
            }
        )

    corrected = runs["corrected_full_jax_resource"]
    default = runs["default"]
    if not corrected["exact_endpoint_correction"]:
        raise ValueError("The corrected resource run did not record endpoint correction.")
    if corrected["likelihood_runtime_mode"] != "full-jax":
        raise ValueError("The corrected resource run did not use the full-JAX runtime mode.")
    if corrected["target_gradient_evaluated"] is not False:
        raise ValueError("The corrected resource run must record that target gradients were disabled.")

    environment_fingerprints = {
        (
            run["gpu_environment"]["jax_version"],
            tuple(run["gpu_environment"]["jax_devices"]),
            run["gpu_environment"]["snapshot"]["gpu_name_as_reported"],
            run["gpu_environment"]["snapshot"]["driver_version"],
            run["gpu_environment"]["snapshot"]["cuda_version"],
            run["gpu_environment"]["snapshot"]["memory_total_mib"],
        )
        for run in runs.values()
    }
    if len(environment_fingerprints) != 1:
        raise ValueError("Archived pilots do not share one recorded GPU environment.")
    common_environment = next(iter(environment_fingerprints))

    claims = [
        {
            "id": "full_model_pipeline_executes",
            "status": "supported_by_pilot",
            "statement": (
                "The complete 23-equation, 16-parameter, three-shock nonlinear Galí "
                "source executes through target generation, surrogate fitting, likelihood "
                "evaluation, and a sampling smoke test."
            ),
        },
        {
            "id": "nonlinear_extended_path_targets_solve_in_pilot_design",
            "status": "supported_by_pilot",
            "statement": (
                f"The order-{default['sep_branching_order']} nonlinear extended-path pilot "
                f"accepted {default['accepted_targets']} targets across "
                f"{default['full_paths']} parameter paths with maximum recorded residual "
                f"{default['max_sep_residual']:.3e}. This is not yet a stochastic-"
                "expectations SEP validation."
            ),
        },
        {
            "id": "surrogate_improves_rom_in_pilot_design",
            "status": "calibration_set_only",
            "statement": (
                "The default pilot reports mean calibration-set improvement over the ROM of "
                f"{100.0 * default['validation_improvement_mean']:.1f} percent after fitting "
                "per-output shrinkage on that same parameter block. The value is not an "
                "untouched out-of-sample estimate."
            ),
        },
        {
            "id": "jax_python_likelihood_implementation_parity",
            "status": "supported_by_pilot",
            "statement": (
                "JAX and Python evaluations of the same surrogate profile inversion objective agree "
                f"within {default['jax_parity_abs_diff']:.3e} at the audited calibration."
            ),
        },
        {
            "id": "target_corrected_sampler_executes",
            "status": "resource_feasibility_only",
            "statement": (
                "A target-corrected sampler can use fixed-reference proposal gradients and "
                "evaluate the parameter-dependent full-JAX surrogate profile objective only at "
                "trajectory endpoints without differentiating that target."
            ),
        },
        {
            "id": "exact_full_nonlinear_posterior",
            "status": "not_supported",
            "statement": (
                "The pilot does not establish exact sampling from a full nonlinear SEP "
                "posterior because its endpoint target remains a surrogate profile objective."
            ),
        },
        {
            "id": "posterior_convergence_or_fidelity",
            "status": "not_supported",
            "statement": (
                "The pilot budgets are too small for R-hat, effective-sample-size, posterior "
                "moment, or coverage claims."
            ),
        },
        {
            "id": "computational_speedup",
            "status": "not_supported",
            "statement": (
                "The suite lacks repeated same-work exact-baseline timings and therefore does "
                "not establish an end-to-end speedup."
            ),
        },
        {
            "id": "gpu_execution_profile",
            "status": "environment_and_execution_only",
            "statement": (
                "The archived suite records a CUDA backend, 64-bit JAX execution, "
                "stage timings, and one nvidia-smi snapshot at run startup. The snapshot "
                "is not a utilization time series and cannot establish peak memory, "
                "average utilization, energy use, or a speed advantage."
            ),
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "engineering_pilot",
        "publication_ready": False,
        "model": {
            "name": EXPECTED_MODEL,
            "equations": 23,
            "parameters": 16,
            "shocks": 3,
            "experiment_scope": "two-parameter reduced experiment",
        },
        "runs": runs,
        "gpu_profile": {
            "evidence_level": "archived_execution_profile_with_start_snapshot",
            "continuous_telemetry_available": False,
            "same_work_comparison_available": False,
            "speedup_claim_supported": False,
            "common_environment": {
                "jax_version": common_environment[0],
                "jax_devices": list(common_environment[1]),
                "gpu_name_as_reported": common_environment[2],
                "driver_version": common_environment[3],
                "cuda_version": common_environment[4],
                "memory_total_mib": common_environment[5],
            },
            "interpretation": (
                "Run-start snapshots document the device environment only. Continuous "
                "nvidia-smi sampling is required for utilization, peak-memory, power, and "
                "energy claims. The three archived workloads are not same-work comparisons."
            ),
        },
        "claim_ledger": claims,
        "hard_blockers": [
            "A frozen, reproducible synthetic-data generator with recorded seed, burn-in, initial condition, and data-generating parameters.",
            "Standardized innovation variances consistent with the model equations; archived pilots duplicated equation-level shock scales in the inversion prior.",
            "Independent MacroModelling.jl trajectory and likelihood comparisons.",
            "Positive-order stochastic SEP with multiple quadrature nodes and all shocks active.",
            "An untouched test split; residual-shrinkage calibration must use a separate partition or be disabled.",
            "A full nonlinear likelihood or an explicitly surrogate-defined estimand.",
            "Explicit invalid-solve support handling rather than a finite log-density plateau.",
            "Four tuned chains with convergence and effective-sample-size diagnostics.",
            "Posterior comparison with an exact or independently validated reference.",
            "Multi-seed, repeated, synchronized end-to-end timing against the same-work baseline.",
            "Continuous dedicated-device telemetry for GPU utilization, peak memory, power, and estimated energy during each timed run.",
            "Galí-specific parity validation for the GPU-native QME solver; the audited Schur path uses a sequential SciPy host callback.",
            "A clean, content-addressed software environment and revision.",
        ],
        "sources": sources,
    }


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_latex_table(evidence: Mapping[str, Any]) -> str:
    labels = ["quick", "corrected_full_jax_resource", "default"]
    display = {
        "quick": "Quick smoke",
        "corrected_full_jax_resource": "Corrected resource",
        "default": "Default pilot",
    }
    rows: list[str] = []
    for label in labels:
        run = evidence["runs"][label]
        correction = "Yes" if run["exact_endpoint_correction"] else "No"
        rows.append(
            " & ".join(
                [
                    _latex_escape(display[label]),
                    _latex_escape(str(run["likelihood_runtime_mode"])),
                    f"{run['theta_draws']} $\\times$ {run['periods']}",
                    f"{run['max_sep_residual']:.2e}",
                    f"{run['validation_rmse_mean']:.2e}",
                    correction,
                    f"{run['hmc_accepted_share']:.3f}",
                    f"{run['hmc_draws_per_second']:.3g}",
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    return rf"""% Generated by benchmarks/build_methods_paper_evidence.py. Do not edit.
\begin{{table}}[t]
\centering
\caption{{Preserved full-model Galí engineering pilots}}
\label{{tab:pilot-results}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
Run & Target runtime & $N_\theta \times T$ & Max. residual & Calib. RMSE & Corrected & Accept. & Draws/s \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\parbox{{0.98\textwidth}}{{\footnotesize\textit{{Notes:}} All runs use the complete 23-equation nonlinear Galí source but estimate only two policy-rule parameters. $N_\theta$ is the number of parameter vectors and $T$ is the observation horizon. Target construction used branching order zero and one node, so these are deterministic nonlinear extended-path pilots, not stochastic-expectations SEP validation. The reported calibration RMSE is computed on the parameter block also used to fit per-output residual-shrinkage factors; it is not an untouched test error. The archived likelihood payload also duplicated equation-level innovation scales in the shock prior. ``Full-JAX'' recomputes the parameter-dependent steady state and first-order ROM inside the surrogate profile inversion objective. It does not evaluate a full nonlinear SEP likelihood inside HMC. Correction therefore targets the declared surrogate objective, not an exact full-nonlinear posterior. The quick, corrected-resource, and default runs use respectively 1, 2, and 1 chains with 2, 2, and 50 post-warmup draws per chain. These runs establish execution and resource feasibility only.}}
\end{{table}}
"""


def render_gpu_profile_table(evidence: Mapping[str, Any]) -> str:
    labels = ["quick", "corrected_full_jax_resource", "default"]
    display = {
        "quick": "Quick smoke",
        "corrected_full_jax_resource": "Corrected resource",
        "default": "Default pilot",
    }
    rows: list[str] = []
    for label in labels:
        run = evidence["runs"][label]
        snapshot = run["gpu_environment"]["snapshot"]
        target_evaluations = run["target_evaluations"]
        target_text = "--" if target_evaluations is None else str(target_evaluations)
        total_draws = int(run["hmc_chains"]) * int(run["hmc_samples"])
        rows.append(
            " & ".join(
                [
                    _latex_escape(display[label]),
                    _latex_escape(str(run["likelihood_runtime_mode"])),
                    target_text,
                    f"{run['pipeline_s'] / 60.0:.2f}",
                    str(total_draws),
                    f"{run['hmc_elapsed_s']:.2f}",
                    f"{run['hmc_draws_per_second']:.4g}",
                    f"{snapshot['gpu_utilization_percent']} / {snapshot['memory_used_mib']}",
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    environment = evidence["gpu_profile"]["common_environment"]
    return rf"""% Generated by benchmarks/build_methods_paper_evidence.py. Do not edit.
\begin{{table}}[t]
\centering
\caption{{Archived Galí GPU execution profile and telemetry boundary}}
\label{{tab:gali-gpu-profile}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
Run & Runtime & Endpoint evals & Pipeline min & Draws & HMC s & Draws/s & Start \% / MiB \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\parbox{{0.98\textwidth}}{{\footnotesize\textit{{Notes:}} All rows report JAX {environment['jax_version']} on {environment['jax_devices'][0]} in 64-bit mode. The archived device reports {environment['memory_total_mib']:,} MiB, NVIDIA driver {_latex_escape(str(environment['driver_version']))}, and CUDA {_latex_escape(str(environment['cuda_version']))}. ``Pipeline min'' is the recorded target-generation and training stage, not a common end-to-end workload. HMC draw counts, chains, likelihood runtimes, trajectory budgets, and endpoint evaluations differ across rows. The final column is a single \texttt{{nvidia-smi}} snapshot taken at run startup; it is not an average or peak. Consequently these data establish GPU execution and document a high corrected-run burden, but they support no utilization, memory-capacity, energy, or speedup claim. The replication code now provides continuous dedicated-device sampling for future same-work runs.}}
\end{{table}}
"""


def render_markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Methods-paper evidence snapshot",
        "",
        f"Schema: `{evidence['schema_version']}`",
        "",
        "**Evidence level:** engineering pilot",
        "",
        "**Publication ready:** no",
        "",
        "## Claim ledger",
        "",
        "| Claim | Status | Auditable statement |",
        "|---|---|---|",
    ]
    for claim in evidence["claim_ledger"]:
        statement = str(claim["statement"]).replace("|", "\\|")
        lines.append(f"| `{claim['id']}` | `{claim['status']}` | {statement} |")
    lines.extend(["", "## Hard blockers", ""])
    lines.extend(f"- {blocker}" for blocker in evidence["hard_blockers"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(evidence: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )
    (output_dir / "pilot_results.tex").write_text(
        render_latex_table(evidence), encoding="utf-8"
    )
    (output_dir / "gpu_profile_results.tex").write_text(
        render_gpu_profile_table(evidence), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    args = parser.parse_args(argv)
    evidence = build_evidence(args.suite_root)
    write_outputs(evidence, args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
