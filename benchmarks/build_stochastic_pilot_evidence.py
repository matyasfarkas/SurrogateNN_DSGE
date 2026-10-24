#!/usr/bin/env python3
"""Normalize one provenance-captured stochastic Galí pilot for the paper.

This builder treats a failed pilot as evidence about feasibility and numerical
failure modes.  It does not reinterpret a failed quality gate, a calibrated
validation score, or same-object Python/JAX parity as confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-stochastic-pilot-evidence/v1"
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


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


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


def _verify_manifest_artifacts(run_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("run_manifest.json has no artifact inventory.")
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        record = _mapping(item, field=f"artifacts[{index}]")
        expected = record.get("sha256")
        remote_path = record.get("path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"artifacts[{index}].sha256 is invalid.")
        if not isinstance(remote_path, str) or not remote_path:
            raise ValueError(f"artifacts[{index}].path is invalid.")
        local_path = run_dir / Path(remote_path).name
        if not local_path.is_file():
            raise ValueError(f"Missing archived artifact {local_path}.")
        actual = _sha256(local_path)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {local_path.name}: expected {expected}, got {actual}."
            )
        verified.append(
            {
                "name": local_path.name,
                "sha256": actual,
                "size_bytes": local_path.stat().st_size,
            }
        )
    return verified


def build_evidence(run_dir: Path, *, archive: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "run_manifest.json"
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    try:
        run = _mapping(
            result["results"]["hlt_fixed_ss_smoke"],
            field="results.hlt_fixed_ss_smoke",
        )
    except KeyError as exc:
        raise ValueError("result.json is missing results.hlt_fixed_ss_smoke.") from exc

    if run.get("model_name") != EXPECTED_MODEL:
        raise ValueError(f"Unexpected model {run.get('model_name')!r}.")
    if run.get("equation_count") != 23 or run.get("parameter_count") != 16:
        raise ValueError("The run is not the declared full 23-equation Galí model.")
    if manifest.get("schema_version") != "surrogatenn-dsge-publication-run-manifest/v1":
        raise ValueError("Unexpected or missing run-manifest schema.")

    repository = _mapping(manifest.get("repository"), field="repository")
    repository_after = _mapping(manifest.get("repository_after"), field="repository_after")
    if repository.get("dirty_worktree") is not False or repository_after.get(
        "dirty_worktree"
    ) is not False:
        raise ValueError("The pilot did not run from a clean repository revision.")
    if repository.get("commit") != repository_after.get("commit"):
        raise ValueError("The repository revision changed during the pilot.")

    target = _mapping(run.get("target_diagnostics"), field="target_diagnostics")
    sep = _mapping(target.get("sep_config"), field="target_diagnostics.sep_config")
    gate = _mapping(run.get("quality_gate"), field="quality_gate")
    dataset = _mapping(run.get("dataset_summary"), field="dataset_summary")
    density = _mapping(run.get("jax_surrogate_log_density"), field="jax_surrogate_log_density")
    hmc = _mapping(run.get("surrogate_hmc"), field="surrogate_hmc")
    training = _mapping(run.get("training_metadata"), field="training_metadata")
    memory = _mapping(target.get("memory_estimate"), field="target_diagnostics.memory_estimate")

    theta_draws = _integer(run.get("original_theta_draws"), field="original_theta_draws")
    full_paths = _integer(target.get("theta_full_success_count"), field="theta_full_success_count")
    accepted_targets = _integer(target.get("accepted_samples"), field="accepted_samples")
    periods = _integer(run.get("periods"), field="periods")
    expected_targets = theta_draws * periods
    success_share = _finite(dataset.get("theta_success_rate"), field="theta_success_rate")
    if not math.isclose(success_share, full_paths / theta_draws, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Reported theta success share is inconsistent with the raw counts.")

    failures = target.get("failure_log")
    if not isinstance(failures, list):
        raise ValueError("target_diagnostics.failure_log must be a list.")
    first_period_failures = sum(
        1
        for item in failures
        if isinstance(item, Mapping) and item.get("period") == 0 and item.get("accepted") is False
    )
    reference_failed = any(
        isinstance(item, Mapping)
        and item.get("theta_index") == 0
        and item.get("period") == 0
        and item.get("accepted") is False
        for item in failures
    )

    artifacts = _verify_manifest_artifacts(run_dir, manifest)
    sources: list[dict[str, Any]] = [
        {"name": "run_manifest.json", "sha256": _sha256(manifest_path)},
        *artifacts,
    ]
    if archive is not None:
        archive = archive.resolve()
        if not archive.is_file():
            raise ValueError(f"Archive does not exist: {archive}.")
        sources.append(
            {
                "name": archive.name,
                "sha256": _sha256(archive),
                "size_bytes": archive.stat().st_size,
            }
        )

    nvidia = _mapping(
        _mapping(manifest.get("environment"), field="environment").get("nvidia_runtime"),
        field="environment.nvidia_runtime",
    )
    gpu_rows = nvidia.get("rows")
    gpu = gpu_rows[0] if isinstance(gpu_rows, list) and gpu_rows else "unavailable"

    gate_issues = gate.get("issues")
    if not isinstance(gate_issues, list) or not all(isinstance(x, str) for x in gate_issues):
        raise ValueError("quality_gate.issues must be a list of strings.")

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "engineering_pilot_negative_result",
        "publication_ready": False,
        "model": {
            "name": EXPECTED_MODEL,
            "equations": 23,
            "parameters": 16,
            "estimated_parameters": list(run.get("parameter_subset", [])),
        },
        "provenance": {
            "commit": str(repository.get("commit")),
            "clean_worktree": True,
            "manifest_status": str(manifest.get("status")),
            "exit_code": manifest.get("exit_code"),
            "prepared_at_utc": manifest.get("prepared_at_utc"),
            "completed_at_utc": manifest.get("completed_at_utc"),
            "gpu": gpu,
            "sources": sources,
        },
        "solver": {
            "branching_order": _integer(sep.get("branching_order"), field="branching_order"),
            "quadrature_nodes": _integer(sep.get("nnodes"), field="nnodes"),
            "planning_horizon": periods,
            "theta_draws": theta_draws,
            "full_paths": full_paths,
            "full_path_share": success_share,
            "accepted_targets": accepted_targets,
            "expected_targets": expected_targets,
            "first_period_failures": first_period_failures,
            "reference_path_failed": reference_failed,
            "accepted_residual_max": _finite(
                target.get("residual_norm_max"), field="residual_norm_max"
            ),
            "pipeline_seconds": _finite(run.get("pipeline_s"), field="pipeline_s"),
            "stacked_unknowns": _integer(memory.get("stacked_unknowns"), field="stacked_unknowns"),
            "rough_dense_workspace_gib": _finite(
                memory.get("rough_chunk_workspace_gib"), field="rough_chunk_workspace_gib"
            ),
            "shifted_path_warm_start_supported": bool(
                _mapping(
                    target.get("shifted_path_warm_start"),
                    field="shifted_path_warm_start",
                ).get("supported")
            ),
        },
        "quality_gate": {
            "status": str(gate.get("status")),
            "issues": list(gate_issues),
        },
        "diagnostics": {
            "steady_states_solved": _integer(
                run.get("steady_state_solved_count"), field="steady_state_solved_count"
            ),
            "steady_state_fallbacks": _integer(
                run.get("steady_state_fallback_count"), field="steady_state_fallback_count"
            ),
            "same_object_python_jax_parity_abs_diff": _finite(
                density.get("parity_abs_diff"), field="parity_abs_diff"
            ),
            "hmc_chains": _integer(hmc.get("chains"), field="hmc.chains"),
            "hmc_warmup": _integer(hmc.get("warmup"), field="hmc.warmup"),
            "hmc_retained_draws": _integer(
                hmc.get("post_warmup_draws"), field="hmc.post_warmup_draws"
            ),
            "hmc_acceptance": _finite(hmc.get("accepted_share"), field="hmc.accepted_share"),
            "hmc_draws_per_second": _finite(
                hmc.get("draws_per_second"), field="hmc.draws_per_second"
            ),
            "reported_validation_rmse": _finite(
                run.get("validation_rmse_mean"), field="validation_rmse_mean"
            ),
            "reported_validation_improvement": _finite(
                run.get("validation_improvement_mean"), field="validation_improvement_mean"
            ),
            "validation_is_untouched": not bool(training.get("residual_shrinkage_calibrated")),
        },
        "known_design_defects": [
            {
                "id": "double_scaled_innovation_prior",
                "effect": (
                    "The archived likelihood payload applied innovation-scale parameters both "
                    "inside the model equations and again as shock-prior standard deviations."
                ),
            },
            {
                "id": "validation_shrinkage_leakage",
                "effect": (
                    "Per-output residual shrinkage was calibrated on the reported validation "
                    "block, so its RMSE and improvement are calibration diagnostics only."
                ),
            },
        ],
        "interpretation": {
            "supported": [
                "The full equation source, nonlinear steady states, stochastic SEP target builder, surrogate, likelihood implementation, and HMC smoke path compose on a GPU.",
                "All eight parameter-specific steady states solved without fallback.",
                "The strict full-path gate exposed a localized nonlinear-solver robustness failure rather than silently dropping failed parameter paths.",
            ],
            "not_supported": [
                "publication-ready estimation evidence",
                "an exact nonlinear posterior",
                "out-of-sample surrogate accuracy",
                "HMC convergence or efficiency",
                "the claim that additional GPU memory alone resolves the bottleneck",
            ],
        },
    }
    if evidence["quality_gate"]["status"] != "failed" or manifest.get("exit_code") == 0:
        raise ValueError("Expected the preserved stochastic pilot to retain its failed gate.")
    return evidence


def render_markdown(evidence: Mapping[str, Any]) -> str:
    solver = _mapping(evidence["solver"], field="solver")
    diagnostics = _mapping(evidence["diagnostics"], field="diagnostics")
    gate = _mapping(evidence["quality_gate"], field="quality_gate")
    lines = [
        "# Stochastic Galí pilot evidence",
        "",
        "**Publication ready:** no",
        "",
        f"- Strict quality gate: **{gate['status']}**",
        (
            f"- Complete nonlinear paths: {solver['full_paths']}/{solver['theta_draws']} "
            f"({100.0 * float(solver['full_path_share']):.1f}%)"
        ),
        f"- Accepted targets: {solver['accepted_targets']}/{solver['expected_targets']}",
        f"- First-period failures: {solver['first_period_failures']}",
        f"- Reference path failed: {str(solver['reference_path_failed']).lower()}",
        f"- Pipeline time: {float(solver['pipeline_seconds']) / 60.0:.1f} minutes",
        f"- Same-object Python/JAX likelihood difference: {float(diagnostics['same_object_python_jax_parity_abs_diff']):.3e}",
        "",
        "The result is a provenance-captured negative engineering result. The archived "
        "validation metric is calibration-only, and the archived likelihood payload "
        "double-scaled standardized innovations. Neither statistic supports an empirical claim.",
        "",
        "## Gate issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in gate["issues"])
    return "\n".join(lines) + "\n"


def render_latex_table(evidence: Mapping[str, Any]) -> str:
    solver = _mapping(evidence["solver"], field="solver")
    diagnostics = _mapping(evidence["diagnostics"], field="diagnostics")
    return rf"""\begin{{table}}[t]
\centering
\caption{{First provenance-captured stochastic extended-path pilot}}
\label{{tab:stochastic-pilot}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Recorded value \\
\midrule
SEP order; nodes; horizon & {solver['branching_order']}; {solver['quadrature_nodes']}; {solver['planning_horizon']} \\
Complete parameter paths & {solver['full_paths']}/{solver['theta_draws']} ({100.0 * float(solver['full_path_share']):.1f}\%) \\
Accepted targets & {solver['accepted_targets']}/{solver['expected_targets']} \\
Accepted-target maximum residual & {float(solver['accepted_residual_max']):.2e} \\
Pipeline time & {float(solver['pipeline_seconds']) / 60.0:.1f} minutes \\
Same-object Python/JAX difference & {float(diagnostics['same_object_python_jax_parity_abs_diff']):.2e} \\
HMC smoke: chains; warmup; retained & {diagnostics['hmc_chains']}; {diagnostics['hmc_warmup']}; {diagnostics['hmc_retained_draws']} \\
HMC smoke: acceptance; draws/s & {100.0 * float(diagnostics['hmc_acceptance']):.1f}\%; {float(diagnostics['hmc_draws_per_second']):.2f} \\
Strict quality gate & failed \\
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.94\linewidth}}
\footnotesize \emph{{Notes:}} Five designs, including the reference path, failed at the first simulated period. The recorded RMSE used shrinkage calibrated on the reported block, and the archived likelihood payload applied shock scales twice. The run therefore diagnoses numerical feasibility only; it is not estimation evidence.
\end{{minipage}}
\end{{table}}
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    evidence = build_evidence(args.run_dir, archive=args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stochastic_pilot_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "stochastic_pilot_evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )
    (args.output_dir / "stochastic_pilot_results.tex").write_text(
        render_latex_table(evidence), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
