#!/usr/bin/env python3
"""Build manuscript evidence from the full-Gali Julia/Python core parity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "surrogatenn-dsge-gali-core-parity/v1"
EXPORT_SCHEMA = "surrogatenn-dsge-gali-julia-core-export/v1"
EVIDENCE_SCHEMA = "surrogatenn-dsge-gali-core-parity-evidence/v1"
EXPECTED_MODEL = "Gali_2015_chapter_3_nonlinear"
EXPECTED_MODEL_SHA256 = "53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55"
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
EXPECTED_THRESHOLDS = {
    "steady_state_max_abs": 1.0e-10,
    "first_order_max_abs": 1.0e-8,
    "python_steady_state_residual_norm": 1.0e-10,
}


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
        raise ValueError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _validate_commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{field} must be a 40-character Git commit.")
    if any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field} is not hexadecimal.")
    return value


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected core-parity report schema.")
    if report.get("status") != "pass":
        raise ValueError("Core-parity report did not pass.")
    if report.get("publication_ready") is not False:
        raise ValueError("The scoped report must not claim publication readiness.")

    gates = _mapping(report.get("gates"), field="gates")
    expected_gates = {
        "source_identity",
        "all_design_points_numerically_pass",
        "clean_repository_provenance",
    }
    if set(gates) != expected_gates or not all(gates.values()):
        raise ValueError("Every declared core-parity gate must pass.")

    thresholds = _mapping(report.get("thresholds"), field="thresholds")
    for name, expected in EXPECTED_THRESHOLDS.items():
        actual = _finite(thresholds.get(name), field=f"thresholds.{name}")
        if actual != expected:
            raise ValueError(f"Unexpected threshold {name}={actual!r}.")

    design = _mapping(report.get("design"), field="design")
    theta_points = tuple(tuple(float(value) for value in point) for point in design.get("theta_points", []))
    if theta_points != EXPECTED_THETA_POINTS:
        raise ValueError("The theta design does not match the frozen eight-point design.")
    if design.get("case_count") != 8 or design.get("unique_theta_count") != 7:
        raise ValueError("The design must contain eight evaluations and seven unique points.")
    if design.get("duplicate_reference_indices") != [0, 4]:
        raise ValueError("The duplicated reference indices are not [0, 4].")

    sources = _mapping(report.get("sources"), field="sources")
    if sources.get("source_identity_passed") is not True:
        raise ValueError("Source identity did not pass.")
    for field in ("local_model", "upstream_model"):
        source = _mapping(sources.get(field), field=f"sources.{field}")
        if source.get("sha256") != EXPECTED_MODEL_SHA256:
            raise ValueError(f"Unexpected {field} content hash.")

    repositories = _mapping(report.get("repositories"), field="repositories")
    repository_summary: dict[str, Any] = {}
    for name in ("surrogatenn_dsge", "macromodelling_jl"):
        repository = _mapping(repositories.get(name), field=f"repositories.{name}")
        if repository.get("dirty") is not False or repository.get("status_short") != []:
            raise ValueError(f"{name} was not clean.")
        repository_summary[name] = {
            "commit": _validate_commit(repository.get("commit"), field=f"{name}.commit"),
            "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        }

    julia_export = _mapping(report.get("julia_export"), field="julia_export")
    if julia_export.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("Unexpected Julia-export schema.")
    model = _mapping(julia_export.get("model"), field="julia_export.model")
    if model.get("symbol") != EXPECTED_MODEL:
        raise ValueError("Unexpected Julia model symbol.")
    if model.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("Unexpected Julia model hash.")
    if len(model.get("variable_names", [])) != 23:
        raise ValueError("The Julia export is not the 23-variable full model.")
    if len(model.get("parameter_names", [])) != 16:
        raise ValueError("The Julia export does not declare 16 parameters.")
    julia_cases = julia_export.get("cases")
    if not isinstance(julia_cases, list) or len(julia_cases) != 8:
        raise ValueError("The Julia export must contain all eight cases.")
    if any(case.get("status") != "ok" for case in julia_cases):
        raise ValueError("At least one Julia case failed.")

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 8:
        raise ValueError("The report must contain all eight comparisons.")
    expected_case_ids = [f"theta_{index}" for index in range(8)]
    if [case.get("case_id") for case in comparisons] != expected_case_ids:
        raise ValueError("Comparison case identifiers or ordering changed.")
    per_case: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparisons):
        if comparison.get("passed") is not True:
            raise ValueError(f"Comparison theta_{index} did not pass.")
        if comparison.get("python_status") != "ok" or comparison.get("julia_status") != "ok":
            raise ValueError(f"Comparison theta_{index} has a failed implementation.")
        parameters = _mapping(comparison.get("parameters"), field=f"comparisons[{index}].parameters")
        expected_phi_pi, expected_phi_y = EXPECTED_THETA_POINTS[index]
        if float(parameters.get("ϕᵖⁱ")) != expected_phi_pi or float(parameters.get("ϕʸ")) != expected_phi_y:
            raise ValueError(f"Comparison theta_{index} uses the wrong parameters.")
        ss = _mapping(comparison.get("steady_state"), field=f"comparisons[{index}].steady_state")
        first_order = _mapping(
            comparison.get("first_order_solution"),
            field=f"comparisons[{index}].first_order_solution",
        )
        ss_max = _finite(ss.get("max_abs"), field=f"comparisons[{index}].steady_state.max_abs")
        ss_rmse = _finite(ss.get("rmse"), field=f"comparisons[{index}].steady_state.rmse")
        first_order_max = _finite(
            first_order.get("max_abs"),
            field=f"comparisons[{index}].first_order_solution.max_abs",
        )
        first_order_rmse = _finite(
            first_order.get("rmse"),
            field=f"comparisons[{index}].first_order_solution.rmse",
        )
        python_residual = _finite(
            comparison.get("python_steady_state_residual_norm"),
            field=f"comparisons[{index}].python_steady_state_residual_norm",
        )
        if ss_max > EXPECTED_THRESHOLDS["steady_state_max_abs"]:
            raise ValueError(f"Comparison theta_{index} fails steady-state parity.")
        if first_order_max > EXPECTED_THRESHOLDS["first_order_max_abs"]:
            raise ValueError(f"Comparison theta_{index} fails first-order parity.")
        if python_residual > EXPECTED_THRESHOLDS["python_steady_state_residual_norm"]:
            raise ValueError(f"Comparison theta_{index} fails Python residual tolerance.")
        per_case.append(
            {
                "case_id": comparison["case_id"],
                "theta_index": index,
                "phi_pi": expected_phi_pi,
                "phi_y": expected_phi_y,
                "steady_state_max_abs": ss_max,
                "steady_state_rmse": ss_rmse,
                "first_order_max_abs": first_order_max,
                "first_order_rmse": first_order_rmse,
                "python_steady_state_residual_norm": python_residual,
            }
        )

    invocation = _mapping(report.get("julia_invocation"), field="julia_invocation")
    if invocation.get("exit_code") != 0:
        raise ValueError("The Julia exporter did not exit successfully.")
    return {
        "repositories": repository_summary,
        "julia_version": report.get("environment", {}).get("julia"),
        "python_version": report.get("environment", {}).get("python"),
        "jax_version": report.get("environment", {}).get("jax"),
        "per_case": per_case,
        "worst_steady_state_max_abs": max(row["steady_state_max_abs"] for row in per_case),
        "worst_first_order_max_abs": max(row["first_order_max_abs"] for row in per_case),
        "worst_python_steady_state_residual_norm": max(
            row["python_steady_state_residual_norm"] for row in per_case
        ),
    }


def _scientific(value: float) -> str:
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times 10^{{{int(exponent)}}}$"


def build_evidence(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    validated = validate_report(report)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "evidence_level": "independent_core_parity",
        "source_report": {
            "name": report_path.name,
            "sha256": _sha256(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "model": {
            "name": EXPECTED_MODEL,
            "equations_and_variables": 23,
            "parameters": 16,
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "design": {
            "evaluations": 8,
            "unique_parameter_points": 7,
            "duplicate_reference_indices": [0, 4],
            "parameter_names": ["ϕᵖⁱ", "ϕʸ"],
            "theta_points": [list(point) for point in EXPECTED_THETA_POINTS],
        },
        "repositories": validated["repositories"],
        "environment": {
            "python": validated["python_version"],
            "jax": validated["jax_version"],
            "julia": validated["julia_version"],
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "metrics": {
            "joint_cases_passed": 8,
            "joint_cases_attempted": 8,
            "worst_steady_state_max_abs": validated["worst_steady_state_max_abs"],
            "worst_first_order_max_abs": validated["worst_first_order_max_abs"],
            "worst_python_steady_state_residual_norm": validated[
                "worst_python_steady_state_residual_norm"
            ],
            "per_case": validated["per_case"],
        },
        "claims": {
            "supported": [
                "The local and upstream files are byte-identical copies of the declared full nonlinear Gali model.",
                "Python and MacroModelling.jl agree on the non-stochastic steady state at all eight evaluations within the frozen tolerance.",
                "Python and MacroModelling.jl agree on the first-order Schur solution at all eight evaluations within the frozen tolerance.",
            ],
            "not_supported": [
                "nonlinear extended-path trajectory parity",
                "nonlinear likelihood parity",
                "posterior fidelity",
                "GPU correctness or throughput",
                "publication readiness",
            ],
        },
        "publication_ready": False,
    }


def render_markdown(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    repositories = evidence["repositories"]
    lines = [
        "# Full Galí Julia/Python core parity evidence",
        "",
        "**Evidence level:** independent core parity",
        "",
        f"- Joint cases passed: {metrics['joint_cases_passed']}/{metrics['joint_cases_attempted']} across seven unique policy-rule points.",
        f"- Worst steady-state maximum absolute difference: {metrics['worst_steady_state_max_abs']:.6e} (threshold {EXPECTED_THRESHOLDS['steady_state_max_abs']:.1e}).",
        f"- Worst first-order Schur maximum absolute difference: {metrics['worst_first_order_max_abs']:.6e} (threshold {EXPECTED_THRESHOLDS['first_order_max_abs']:.1e}).",
        f"- Worst Python steady-state residual norm: {metrics['worst_python_steady_state_residual_norm']:.6e}.",
        f"- SurrogateNN_DSGE revision: `{repositories['surrogatenn_dsge']['commit']}` (clean).",
        f"- MacroModelling.jl revision: `{repositories['macromodelling_jl']['commit']}` (clean).",
        f"- Model SHA-256: `{EXPECTED_MODEL_SHA256}`.",
        "",
        "## Claim boundary",
        "",
        "This result validates the shared model source, non-stochastic steady state, and first-order Schur solution. It does not validate nonlinear trajectories, nonlinear likelihoods, a posterior, or GPU performance.",
        "",
        "## Per-case discrepancies",
        "",
        "| Case | $\\phi_\\pi$ | $\\phi_y$ | SS max abs | First-order max abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics["per_case"]:
        lines.append(
            f"| `{row['case_id']}` | {row['phi_pi']:.3f} | {row['phi_y']:.3f} | "
            f"{row['steady_state_max_abs']:.3e} | {row['first_order_max_abs']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def render_latex(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    ss = _scientific(metrics["worst_steady_state_max_abs"])
    first_order = _scientific(metrics["worst_first_order_max_abs"])
    residual = _scientific(metrics["worst_python_steady_state_residual_norm"])
    return rf"""% Generated by benchmarks/build_gali_core_parity_evidence.py. Do not edit.
\providecommand{{\GaliCoreSSMax}}{{{ss}}}
\providecommand{{\GaliCoreFirstOrderMax}}{{{first_order}}}
\providecommand{{\GaliCorePythonSSResidual}}{{{residual}}}
\begin{{table}}[t]
\centering
\caption{{Independent Julia--Python core parity on the complete nonlinear Gal\'i source}}
\label{{tab:gali-core-parity}}
\small
\begin{{tabular}}{{lrrl}}
\toprule
Gate & Observed & Threshold & Result \\
\midrule
Joint design evaluations & 8/8 & 8/8 & Pass \\
Steady-state maximum absolute difference & \GaliCoreSSMax & $10^{{-10}}$ & Pass \\
First-order Schur maximum absolute difference & \GaliCoreFirstOrderMax & $10^{{-8}}$ & Pass \\
Python steady-state residual norm & \GaliCorePythonSSResidual & $10^{{-10}}$ & Pass \\
Clean source revisions & 2/2 & 2/2 & Pass \\
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\parbox{{0.91\textwidth}}{{\footnotesize\textit{{Notes:}} Eight evaluations cover seven unique $(\phi_\pi,\phi_y)$ points because the reference calibration is repeated at indices 0 and 4. The local and upstream model files share SHA-256 \texttt{{{EXPECTED_MODEL_SHA256[:16]}\ldots}}. This gate covers the non-stochastic steady state and first-order Schur solution only; nonlinear trajectory, likelihood, posterior, and GPU claims remain open.}}
\end{{table}}
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Raw core-parity JSON report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/generated"),
        help="Directory for JSON, Markdown, LaTeX, and raw evidence.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence = build_evidence(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gali_core_parity_raw.json").write_bytes(args.report.read_bytes())
    (args.output_dir / "gali_core_parity_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gali_core_parity_evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )
    (args.output_dir / "gali_core_parity_results.tex").write_text(
        render_latex(evidence), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
