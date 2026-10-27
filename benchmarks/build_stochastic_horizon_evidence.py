#!/usr/bin/env python3
"""Build hash-verified evidence for the reduced-horizon stochastic Galí gate.

The builder deliberately separates equation-residual convergence from terminal-
horizon stabilization.  A clean horizon-five target preflight can support a
reduced numerical pilot while horizon sensitivity remains too large for a
confirmatory likelihood or posterior claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-stochastic-horizon-evidence/v1"
MANIFEST_SCHEMA = "surrogatenn-dsge-publication-run-manifest/v1"
EXPECTED_MODEL = "Gali_2015_chapter_3_nonlinear"
EXPECTED_EQUATIONS = 23
EXPECTED_PARAMETERS = 16
EXPECTED_THETA_INDICES = tuple(range(8))
SENSITIVITY_VARIABLES = ("log_y", "pi_ann", "i_ann")


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
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}.")
    return int(value)


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean, got {value!r}.")
    return value


def _singleton(value: Any, *, field: str) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{field} must be a one-element list.")
    return value[0]


def _verify_manifest_artifacts(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("run_manifest.json has no artifact inventory.")
    verified: list[dict[str, Any]] = []
    for index, raw_record in enumerate(artifacts):
        record = _mapping(raw_record, field=f"artifacts[{index}]")
        expected = record.get("sha256")
        recorded_path = record.get("path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"artifacts[{index}].sha256 is invalid.")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ValueError(f"artifacts[{index}].path is invalid.")
        local_path = run_dir / Path(recorded_path).name
        if not local_path.is_file():
            raise ValueError(f"Missing preflight artifact {local_path}.")
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


def _validate_common_solver_config(
    config: Mapping[str, Any],
    *,
    periods: int,
    field: str,
) -> None:
    required = {
        "periods": periods,
        "branching_order": 1,
        "nnodes": 3,
        "sparse_tree": True,
        "shock_scale": 1.0,
        "linear_solver": "qr",
        "line_search_norm": "inf",
        "positive_state_indices": [],
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(
                f"{field}.{name} must equal {expected!r}, got {config.get(name)!r}."
            )
    if _finite(config.get("tol"), field=f"{field}.tol") > 1e-8:
        raise ValueError(f"{field}.tol is weaker than 1e-8.")
    if _finite(config.get("accept_tol"), field=f"{field}.accept_tol") > 1e-8:
        raise ValueError(f"{field}.accept_tol is weaker than 1e-8.")


def _validate_preflight(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "run_manifest.json"
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected or missing run-manifest schema.")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError("The target preflight manifest did not complete successfully.")
    command = _mapping(manifest.get("command"), field="command")
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("The target preflight manifest has no command argv.")

    def option(name: str) -> str:
        try:
            index = argv.index(name)
            return argv[index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"The target preflight command is missing {name}.") from exc

    exact_options = {
        "--hlt-target-builder": "batched-sep",
        "--hlt-theta-draws": "8",
        "--hlt-periods": "1",
        "--sep-periods": "5",
        "--sep-order": "1",
        "--sep-nnodes": "3",
        "--sep-tol": "1e-8",
        "--sep-accept-tol": "1e-8",
        "--sep-line-search-norm": "inf",
        "--hlt-likelihood-periods": "0",
        "--hlt-surrogate-hmc-samples": "0",
    }
    for name, expected in exact_options.items():
        actual = option(name)
        if actual != expected:
            raise ValueError(
                f"The target preflight command uses {name}={actual!r}, "
                f"expected {expected!r}."
            )
    for name in ("--hlt-parameter-perturbation", "--hlt-shock-scale"):
        if not math.isclose(float(option(name)), 0.12, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"The target preflight command uses the wrong {name}.")
    if "--hlt-require-full-target-success" not in argv:
        raise ValueError("The target preflight command did not require full target success.")
    repository = _mapping(manifest.get("repository"), field="repository")
    repository_after = _mapping(
        manifest.get("repository_after"), field="repository_after"
    )
    if repository.get("dirty_worktree") is not False or repository_after.get(
        "dirty_worktree"
    ) is not False:
        raise ValueError("The target preflight did not run from a clean revision.")
    commit = repository.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("The target preflight commit is invalid.")
    if repository_after.get("commit") != commit:
        raise ValueError("The repository revision changed during the target preflight.")

    try:
        run = _mapping(
            result["results"]["hlt_fixed_ss_smoke"],
            field="results.hlt_fixed_ss_smoke",
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("result.json is missing results.hlt_fixed_ss_smoke.") from exc
    if run.get("model_name") != EXPECTED_MODEL:
        raise ValueError(f"Unexpected model {run.get('model_name')!r}.")
    if run.get("equation_count") != EXPECTED_EQUATIONS or run.get(
        "parameter_count"
    ) != EXPECTED_PARAMETERS:
        raise ValueError("The preflight is not the declared full Galí equation system.")
    if run.get("status") != "ok":
        raise ValueError("The target preflight result status is not ok.")

    sep = _mapping(run.get("sep_config"), field="sep_config")
    _validate_common_solver_config(sep, periods=5, field="sep_config")
    target = _mapping(run.get("target_diagnostics"), field="target_diagnostics")
    gate = _mapping(run.get("quality_gate"), field="quality_gate")
    if gate.get("status") != "ok" or gate.get("issues") != []:
        raise ValueError("The target preflight quality gate did not pass cleanly.")
    if target.get("status") != "ok" or target.get("failure_log") != []:
        raise ValueError("The target preflight contains target failures.")
    if target.get("theta_full_success_count") != 8 or target.get(
        "accepted_samples"
    ) != 8:
        raise ValueError("The target preflight did not accept all eight designs.")
    if run.get("steady_state_solved_count") != 8 or run.get(
        "steady_state_fallback_count"
    ) != 0:
        raise ValueError("The target preflight did not solve all steady states directly.")
    residual_summary = _mapping(
        target.get("attempted_residual_summary"),
        field="target_diagnostics.attempted_residual_summary",
    )
    residual_max = _finite(residual_summary.get("max"), field="residual_summary.max")
    if residual_max > _finite(sep.get("accept_tol"), field="sep_config.accept_tol"):
        raise ValueError("The target preflight residual exceeds its acceptance gate.")

    return {
        "commit": commit,
        "branch": repository.get("branch"),
        "result_sha256": _sha256(result_path),
        "manifest_sha256": _sha256(manifest_path),
        "verified_artifacts": _verify_manifest_artifacts(run_dir, manifest),
        "prepared_at_utc": manifest.get("prepared_at_utc"),
        "completed_at_utc": manifest.get("completed_at_utc"),
        "solver": dict(sep),
        "steady_states_solved": 8,
        "steady_state_fallbacks": 0,
        "accepted_targets": 8,
        "residual_max": residual_max,
        "pipeline_seconds": _finite(run.get("pipeline_s"), field="pipeline_s"),
    }


def _validate_diagnostic(
    path: Path,
    *,
    expected_commit: str,
    expected_periods: int,
    expected_solver: str,
    expected_indices: tuple[int, ...],
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"Diagnostic {path.name} is not complete.")
    git = _mapping(payload.get("git"), field=f"{path.name}.git")
    if git.get("dirty") is not False or git.get("status_short") != []:
        raise ValueError(f"Diagnostic {path.name} was not run from a clean revision.")
    if git.get("commit") != expected_commit:
        raise ValueError(f"Diagnostic {path.name} uses a different revision.")
    command = _mapping(payload.get("command"), field=f"{path.name}.command")
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError(f"Diagnostic {path.name} has no recorded command argv.")
    inputs = _mapping(payload.get("inputs"), field=f"{path.name}.inputs")
    for input_name in ("model_source", "payload"):
        record = _mapping(inputs.get(input_name), field=f"inputs.{input_name}")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError(f"Diagnostic {path.name} has an invalid {input_name} hash.")
    model = _mapping(payload.get("model"), field=f"{path.name}.model")
    if model.get("name") != EXPECTED_MODEL or model.get("variables") != 23 or model.get(
        "parameters"
    ) != 16:
        raise ValueError(f"Diagnostic {path.name} uses the wrong model.")
    config = _mapping(payload.get("config"), field=f"{path.name}.config")
    _validate_common_solver_config(
        config,
        periods=expected_periods,
        field=f"{path.name}.config",
    )
    if config.get("solver") != expected_solver:
        raise ValueError(f"Diagnostic {path.name} uses the wrong solver selection.")
    theta_indices = payload.get("theta_indices")
    if theta_indices != list(expected_indices):
        raise ValueError(f"Diagnostic {path.name} uses unexpected theta indices.")
    draws = payload.get("draws")
    if not isinstance(draws, list) or len(draws) != len(expected_indices):
        raise ValueError(f"Diagnostic {path.name} has an unexpected draw count.")
    return payload


def _extract_passing_horizon(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(payload.get("config"), field="config")
    periods = _integer(config.get("periods"), field="config.periods")
    residuals: list[float] = []
    iterations: list[int] = []
    first_steps: dict[int, dict[str, float]] = {}
    for row_index, raw_draw in enumerate(payload["draws"]):
        draw = _mapping(raw_draw, field=f"draws[{row_index}]")
        theta_index = _integer(draw.get("theta_index"), field="theta_index")
        batched = _mapping(draw.get("batched"), field="batched")
        accepted = _boolean(
            _singleton(batched.get("accepted"), field="batched.accepted"),
            field="batched.accepted[0]",
        )
        converged = _boolean(
            _singleton(batched.get("converged"), field="batched.converged"),
            field="batched.converged[0]",
        )
        if not accepted or not converged:
            raise ValueError(f"Horizon {periods} theta {theta_index} did not pass.")
        residuals.append(
            _finite(
                _singleton(batched.get("residual_norm"), field="batched.residual_norm"),
                field="batched.residual_norm[0]",
            )
        )
        iterations.append(
            _integer(
                _singleton(batched.get("iterations"), field="batched.iterations"),
                field="batched.iterations[0]",
            )
        )
        first_raw = _mapping(
            batched.get("mean_path_first_step"), field="batched.mean_path_first_step"
        )
        first_steps[theta_index] = {
            str(name): _finite(value, field=f"first_step.{name}")
            for name, value in first_raw.items()
        }
    accept_tol = _finite(config.get("accept_tol"), field="config.accept_tol")
    if max(residuals) > accept_tol:
        raise ValueError(f"Horizon {periods} residual exceeds the acceptance gate.")
    return {
        "periods": periods,
        "attempted": len(residuals),
        "accepted": len(residuals),
        "residual_min": min(residuals),
        "residual_max": max(residuals),
        "iterations_min": min(iterations),
        "iterations_max": max(iterations),
        "first_steps": first_steps,
    }


def _first_step_sensitivity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_steps = _mapping(left.get("first_steps"), field="left.first_steps")
    right_steps = _mapping(right.get("first_steps"), field="right.first_steps")
    if set(left_steps) != set(right_steps):
        raise ValueError("Horizon diagnostics do not contain the same theta designs.")
    all_differences: list[float] = []
    named: dict[str, list[float]] = {name: [] for name in SENSITIVITY_VARIABLES}
    for theta_index in sorted(left_steps, key=int):
        left_row = _mapping(left_steps[theta_index], field="left.first_step")
        right_row = _mapping(right_steps[theta_index], field="right.first_step")
        if set(left_row) != set(right_row):
            raise ValueError("First-step state names differ across horizons.")
        for name in left_row:
            difference = _finite(right_row[name], field=f"right.{name}") - _finite(
                left_row[name], field=f"left.{name}"
            )
            all_differences.append(difference)
            if name in named:
                named[name].append(difference)
    if any(not values for values in named.values()):
        raise ValueError("A required horizon-sensitivity variable is absent.")

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "max_abs": max(abs(value) for value in values),
            "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        }

    return {
        "from_periods": left["periods"],
        "to_periods": right["periods"],
        "all_states": summary(all_differences),
        "variables": {name: summary(values) for name, values in named.items()},
    }


def _extract_horizon_six_failure(payload: Mapping[str, Any]) -> dict[str, Any]:
    draw = _mapping(payload["draws"][0], field="draws[0]")
    sequential = _mapping(draw.get("sequential"), field="sequential")
    batched = _mapping(draw.get("batched"), field="batched")
    sequential_accepted = _boolean(sequential.get("accepted"), field="sequential.accepted")
    batched_accepted = _boolean(
        _singleton(batched.get("accepted"), field="batched.accepted"),
        field="batched.accepted[0]",
    )
    if sequential_accepted or batched_accepted:
        raise ValueError("The declared horizon-six hard case did not fail both solvers.")
    return {
        "periods": 6,
        "theta_index": _integer(draw.get("theta_index"), field="theta_index"),
        "theta": list(draw.get("theta", [])),
        "sequential": {
            "accepted": False,
            "residual_norm": _finite(
                sequential.get("residual_norm"), field="sequential.residual_norm"
            ),
            "iterations": _integer(
                sequential.get("iterations"), field="sequential.iterations"
            ),
        },
        "batched": {
            "accepted": False,
            "residual_norm": _finite(
                _singleton(batched.get("residual_norm"), field="batched.residual_norm"),
                field="batched.residual_norm[0]",
            ),
            "iterations": _integer(
                _singleton(batched.get("iterations"), field="batched.iterations"),
                field="batched.iterations[0]",
            ),
            "termination": _singleton(
                batched.get("termination"), field="batched.termination"
            ),
        },
    }


def build_evidence(
    preflight_dir: Path,
    *,
    horizon_paths: Sequence[Path],
    horizon_six_path: Path,
) -> dict[str, Any]:
    preflight_dir = preflight_dir.resolve()
    preflight = _validate_preflight(preflight_dir)
    if len(horizon_paths) != 3:
        raise ValueError("Exactly three horizon diagnostics (3, 4, and 5) are required.")

    horizon_payloads: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = [
        {
            "name": "preflight/run_manifest.json",
            "sha256": preflight["manifest_sha256"],
        },
        {
            "name": "preflight/result.json",
            "sha256": preflight["result_sha256"],
        },
    ]
    for path in horizon_paths:
        resolved = path.resolve()
        payload = _validate_diagnostic(
            resolved,
            expected_commit=preflight["commit"],
            expected_periods=_integer(
                _mapping(_read_json(resolved).get("config"), field="config").get("periods"),
                field="config.periods",
            ),
            expected_solver="batched",
            expected_indices=EXPECTED_THETA_INDICES,
        )
        horizon_payloads.append(_extract_passing_horizon(payload))
        sources.append(
            {
                "name": resolved.name,
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    horizon_payloads.sort(key=lambda item: item["periods"])
    if [row["periods"] for row in horizon_payloads] != [3, 4, 5]:
        raise ValueError("Horizon diagnostics must cover periods 3, 4, and 5 exactly.")

    h6_resolved = horizon_six_path.resolve()
    h6_payload = _validate_diagnostic(
        h6_resolved,
        expected_commit=preflight["commit"],
        expected_periods=6,
        expected_solver="both",
        expected_indices=(1,),
    )
    horizon_six = _extract_horizon_six_failure(h6_payload)
    sources.append(
        {
            "name": h6_resolved.name,
            "sha256": _sha256(h6_resolved),
            "size_bytes": h6_resolved.stat().st_size,
        }
    )

    public_horizons = [
        {key: value for key, value in row.items() if key != "first_steps"}
        for row in horizon_payloads
    ]
    sensitivity = [
        _first_step_sensitivity(horizon_payloads[0], horizon_payloads[1]),
        _first_step_sensitivity(horizon_payloads[1], horizon_payloads[2]),
        _first_step_sensitivity(horizon_payloads[0], horizon_payloads[2]),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_level": "reduced_horizon_stochastic_target_preflight",
        "publication_ready": False,
        "model": {
            "name": EXPECTED_MODEL,
            "equations": EXPECTED_EQUATIONS,
            "parameters": EXPECTED_PARAMETERS,
            "estimated_parameters": ["ϕᵖⁱ", "ϕʸ"],
        },
        "provenance": {
            "commit": preflight["commit"],
            "branch": preflight["branch"],
            "clean_worktree": True,
            "sources": sources,
            "verified_preflight_artifacts": preflight["verified_artifacts"],
        },
        "target_preflight": {
            key: value
            for key, value in preflight.items()
            if key
            not in {
                "commit",
                "branch",
                "result_sha256",
                "manifest_sha256",
                "verified_artifacts",
            }
        },
        "horizon_results": public_horizons,
        "first_step_sensitivity": sensitivity,
        "horizon_six_hard_case": horizon_six,
        "interpretation": {
            "supported": (
                "All eight prespecified 12% policy-rule designs solve the complete "
                "23-equation model at stochastic SEP horizons three through five under "
                "the strict 1e-8 residual gate."
            ),
            "not_supported": (
                "The evidence does not establish terminal-horizon convergence, a "
                "nonlinear likelihood, posterior fidelity, or a speed advantage. The "
                "hard design fails at horizon six under both solver implementations."
            ),
        },
    }


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"


def render_markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Stochastic Galí horizon evidence",
        "",
        f"Schema: `{evidence['schema_version']}`",
        "",
        f"**Publication ready:** {'yes' if evidence['publication_ready'] else 'no'}",
        "",
        "## Residual gates",
        "",
        "| SEP horizon | Accepted designs | Maximum residual | Newton iterations |",
        "|---:|---:|---:|---:|",
    ]
    for row in evidence["horizon_results"]:
        lines.append(
            f"| {row['periods']} | {row['accepted']}/{row['attempted']} | "
            f"{_format_scientific(row['residual_max'])} | "
            f"{row['iterations_min']}--{row['iterations_max']} |"
        )
    lines.extend(
        [
            "",
            "## First-step horizon sensitivity",
            "",
            "| Comparison | All-state max | log output max | Inflation max | Policy-rate max |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evidence["first_step_sensitivity"]:
        lines.append(
            f"| {row['from_periods']}→{row['to_periods']} | "
            f"{_format_scientific(row['all_states']['max_abs'])} | "
            f"{_format_scientific(row['variables']['log_y']['max_abs'])} | "
            f"{_format_scientific(row['variables']['pi_ann']['max_abs'])} | "
            f"{_format_scientific(row['variables']['i_ann']['max_abs'])} |"
        )
    h6 = evidence["horizon_six_hard_case"]
    lines.extend(
        [
            "",
            "## Horizon-six boundary",
            "",
            f"The hard design (theta index {h6['theta_index']}) fails both solvers: "
            f"sequential residual {_format_scientific(h6['sequential']['residual_norm'])}; "
            f"batched residual {_format_scientific(h6['batched']['residual_norm'])}.",
            "",
            "## Claim boundary",
            "",
            f"- Supported: {evidence['interpretation']['supported']}",
            f"- Not supported: {evidence['interpretation']['not_supported']}",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(evidence: Mapping[str, Any]) -> str:
    lines = [
        "% Generated by benchmarks/build_stochastic_horizon_evidence.py. Do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Reduced-horizon stochastic SEP evidence}",
        "\\label{tab:stochastic-horizon-evidence}",
        "\\small",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "SEP horizon & Accepted designs & Maximum residual & Newton iterations \\\\",
        "\\midrule",
    ]
    for row in evidence["horizon_results"]:
        lines.append(
            f"{row['periods']} & {row['accepted']}/{row['attempted']} & "
            f"${row['residual_max']:.2e}$ & {row['iterations_min']}--{row['iterations_max']} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "Horizon comparison & All-state max & $\\log Y$ max & $\\pi^{ann}$ / $i^{ann}$ max \\\\",
            "\\midrule",
        ]
    )
    for row in evidence["first_step_sensitivity"]:
        lines.append(
            f"{row['from_periods']}$\\rightarrow${row['to_periods']} & "
            f"{row['all_states']['max_abs']:.3g} & "
            f"{row['variables']['log_y']['max_abs']:.3g} & "
            f"{row['variables']['pi_ann']['max_abs']:.3g} / "
            f"{row['variables']['i_ann']['max_abs']:.3g} \\\\"
        )
    h6 = evidence["horizon_six_hard_case"]
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{0.4em}",
            "\\parbox{0.96\\textwidth}{\\footnotesize\\textit{Notes:} "
            "Horizons three through five use all eight prespecified 12 percent "
            "perturbations of the two policy-rule parameters, order-one stochastic "
            "extended path, three Gauss--Hermite nodes, and an $L_\\infty$ residual "
            "gate of $10^{-8}$. The horizon-six hard design fails under both the "
            f"sequential ($ {h6['sequential']['residual_norm']:.2e} $) and batched "
            f"($ {h6['batched']['residual_norm']:.2e} $) solvers. First-step changes "
            "show that residual convergence is not terminal-horizon convergence. "
            "This table supports a reduced numerical pilot only; it is not likelihood "
            "or posterior evidence.}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preflight_dir", type=Path)
    parser.add_argument(
        "--horizon-result",
        action="append",
        type=Path,
        required=True,
        help="Clean diagnostic JSON for horizon 3, 4, or 5; pass exactly three times.",
    )
    parser.add_argument("--horizon-six-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = build_evidence(
        args.preflight_dir,
        horizon_paths=args.horizon_result,
        horizon_six_path=args.horizon_six_result,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stochastic_horizon_evidence.json"
    markdown_path = args.output_dir / "stochastic_horizon_evidence.md"
    latex_path = args.output_dir / "stochastic_horizon_results.tex"
    json_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(evidence), encoding="utf-8")
    latex_path.write_text(render_latex(evidence), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
