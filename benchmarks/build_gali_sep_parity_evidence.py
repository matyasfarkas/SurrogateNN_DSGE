#!/usr/bin/env python3
"""Build manuscript evidence from the full-Gali nonlinear SEP parity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "surrogatenn-dsge-gali-sep-parity/v1"
EXPORT_SCHEMA = "surrogatenn-dsge-gali-julia-sep-export/v1"
EVIDENCE_SCHEMA = "surrogatenn-dsge-gali-sep-parity-evidence/v1"
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
    "tree_max_abs": 1.0e-6,
    "mean_path_max_abs": 1.0e-8,
    "solver_residual": 1.0e-8,
}
EXPECTED_CONFIG = {
    "periods": 3,
    "branching_order": 1,
    "nnodes": 3,
    "sparse_tree": True,
    "max_iter": 80,
    "tol": 1.0e-8,
    "accept_tol": 1.0e-8,
    "linear_solver": "qr",
    "fallback_solver": None,
    "stall_iters": 25,
    "stall_rel_tol": 1.0e-4,
    "stall_abs_tol": 1.0e-10,
}
EXPECTED_FIRST_PERIOD_SHOCK = (0.12, -0.08, 0.10)
EXPECTED_SHOCK_MATRIX = (
    EXPECTED_FIRST_PERIOD_SHOCK,
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
EXPECTED_GROUP_COUNTS = [1, 1, 9, 9]


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


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_config(value: Any) -> None:
    config = _mapping(value, field="design.config")
    if set(config) != set(EXPECTED_CONFIG):
        raise ValueError("The SEP configuration fields changed.")
    for name, expected in EXPECTED_CONFIG.items():
        if config.get(name) != expected:
            raise ValueError(f"Unexpected SEP configuration {name}={config.get(name)!r}.")


def _validate_shock_matrix(value: Any, *, field: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must have three period rows.")
    rows: list[tuple[float, ...]] = []
    for period, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{field}[{period}] must contain three shocks.")
        rows.append(
            tuple(
                _finite(item, field=f"{field}[{period}][{shock}]")
                for shock, item in enumerate(row)
            )
        )
    if tuple(rows) != EXPECTED_SHOCK_MATRIX:
        raise ValueError(f"{field} does not match the frozen three-shock scenario.")


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected nonlinear SEP parity report schema.")
    if report.get("status") != "pass":
        raise ValueError("Nonlinear SEP parity report did not pass.")
    if report.get("publication_ready") is not False:
        raise ValueError("The scoped report must not claim publication readiness.")

    gates = _mapping(report.get("gates"), field="gates")
    if set(gates) != {"source_identity", "all_cases_pass", "clean_provenance"}:
        raise ValueError("The declared nonlinear SEP gates changed.")
    if not all(value is True for value in gates.values()):
        raise ValueError("Every declared nonlinear SEP gate must pass.")

    thresholds = _mapping(report.get("thresholds"), field="thresholds")
    if set(thresholds) != set(EXPECTED_THRESHOLDS):
        raise ValueError("The nonlinear SEP threshold fields changed.")
    for name, expected in EXPECTED_THRESHOLDS.items():
        actual = _finite(thresholds.get(name), field=f"thresholds.{name}")
        if actual != expected:
            raise ValueError(f"Unexpected threshold {name}={actual!r}.")

    design = _mapping(report.get("design"), field="design")
    if design.get("theta_indices") != list(range(8)):
        raise ValueError("The nonlinear SEP report must contain theta indices 0 through 7.")
    theta_points = tuple(
        tuple(float(item) for item in point)
        for point in design.get("theta_points", [])
    )
    if theta_points != EXPECTED_THETA_POINTS:
        raise ValueError("The theta design does not match the frozen eight-point design.")
    shock = tuple(float(item) for item in design.get("deterministic_first_period_shock", []))
    if shock != EXPECTED_FIRST_PERIOD_SHOCK:
        raise ValueError("The declared three-shock scenario changed.")
    _validate_shock_matrix(
        design.get("deterministic_shocks"), field="design.deterministic_shocks"
    )
    _validate_config(design.get("config"))

    sources = _mapping(report.get("sources"), field="sources")
    for name in ("local_model", "upstream_model"):
        source = _mapping(sources.get(name), field=f"sources.{name}")
        if source.get("sha256") != EXPECTED_MODEL_SHA256:
            raise ValueError(f"Unexpected {name} content hash.")

    provenance = _mapping(report.get("provenance"), field="provenance")
    repositories: dict[str, Any] = {}
    for key, output_name in (
        ("repository", "surrogatenn_dsge"),
        ("upstream_repository", "macromodelling_jl"),
    ):
        repository = _mapping(provenance.get(key), field=f"provenance.{key}")
        if repository.get("dirty") is not False or repository.get("status_short") != []:
            raise ValueError(f"{output_name} was not clean.")
        repositories[output_name] = {
            "commit": _validate_commit(
                repository.get("commit"), field=f"{output_name}.commit"
            )
        }

    julia_export = _mapping(report.get("julia_export"), field="julia_export")
    if julia_export.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("Unexpected Julia nonlinear SEP export schema.")
    model = _mapping(julia_export.get("model"), field="julia_export.model")
    if model.get("symbol") != EXPECTED_MODEL or model.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("Unexpected Julia model identity.")
    if len(model.get("variable_names", [])) != 23:
        raise ValueError("The Julia export is not the 23-variable full model.")
    if len(model.get("shock_names", [])) != 3:
        raise ValueError("The Julia export does not declare three shocks.")
    if len(model.get("parameter_names", [])) != 16:
        raise ValueError("The Julia export does not declare 16 parameters.")
    _validate_config(julia_export.get("config"))

    python_cases = report.get("python_cases")
    julia_cases = julia_export.get("cases")
    comparisons = report.get("comparisons")
    for name, cases in (
        ("Python", python_cases),
        ("Julia", julia_cases),
        ("comparison", comparisons),
    ):
        if not isinstance(cases, list) or len(cases) != 8:
            raise ValueError(f"The {name} results must contain all eight cases.")
    expected_ids = [f"theta_{index}" for index in range(8)]
    for name, cases in (
        ("Python", python_cases),
        ("Julia", julia_cases),
        ("comparison", comparisons),
    ):
        if [case.get("case_id") for case in cases] != expected_ids:
            raise ValueError(f"The {name} case identifiers or ordering changed.")

    for index, (python_case, julia_case) in enumerate(zip(python_cases, julia_cases)):
        if python_case.get("status") != "ok" or julia_case.get("status") != "ok":
            raise ValueError(f"At least one implementation failed theta_{index}.")

    for index, julia_case in enumerate(julia_cases):
        if julia_case.get("group_counts") != EXPECTED_GROUP_COUNTS:
            raise ValueError(f"Julia theta_{index} group counts changed.")
        if int(julia_case.get("convergence_flag", -1)) != 0:
            raise ValueError(f"Julia theta_{index} did not converge.")

    per_case: list[dict[str, Any]] = []
    for index, comparison in enumerate(comparisons):
        if comparison.get("passed") is not True:
            raise ValueError(f"Comparison theta_{index} did not pass.")
        if comparison.get("python_status") != "ok" or comparison.get("julia_status") != "ok":
            raise ValueError(f"Comparison theta_{index} has a failed implementation.")
        if comparison.get("group_counts") != EXPECTED_GROUP_COUNTS:
            raise ValueError(f"Comparison theta_{index} group counts changed.")
        parameters = _mapping(
            comparison.get("parameters"), field=f"comparisons[{index}].parameters"
        )
        expected_phi_pi, expected_phi_y = EXPECTED_THETA_POINTS[index]
        if (
            float(parameters.get("ϕᵖⁱ")) != expected_phi_pi
            or float(parameters.get("ϕʸ")) != expected_phi_y
        ):
            raise ValueError(f"Comparison theta_{index} uses the wrong parameters.")
        steady = _mapping(
            comparison.get("steady_state"), field=f"comparisons[{index}].steady_state"
        )
        tree = _mapping(comparison.get("tree"), field=f"comparisons[{index}].tree")
        mean = _mapping(
            comparison.get("mean_path"), field=f"comparisons[{index}].mean_path"
        )
        python = _mapping(
            comparison.get("python"), field=f"comparisons[{index}].python"
        )
        julia = _mapping(comparison.get("julia"), field=f"comparisons[{index}].julia")
        steady_max = _finite(
            steady.get("max_abs"), field=f"comparisons[{index}].steady_state.max_abs"
        )
        tree_max = _finite(tree.get("max_abs"), field=f"comparisons[{index}].tree.max_abs")
        tree_rmse = _finite(tree.get("rmse"), field=f"comparisons[{index}].tree.rmse")
        mean_max = _finite(
            mean.get("max_abs"), field=f"comparisons[{index}].mean_path.max_abs"
        )
        mean_rmse = _finite(
            mean.get("rmse"), field=f"comparisons[{index}].mean_path.rmse"
        )
        python_residual = _finite(
            python.get("residual_norm"), field=f"comparisons[{index}].python.residual_norm"
        )
        julia_residual = _finite(
            julia.get("final_error"), field=f"comparisons[{index}].julia.final_error"
        )
        if steady_max > EXPECTED_THRESHOLDS["steady_state_max_abs"]:
            raise ValueError(f"Comparison theta_{index} fails steady-state parity.")
        if tree_max > EXPECTED_THRESHOLDS["tree_max_abs"]:
            raise ValueError(f"Comparison theta_{index} fails full-tree parity.")
        if mean_max > EXPECTED_THRESHOLDS["mean_path_max_abs"]:
            raise ValueError(f"Comparison theta_{index} fails mean-path parity.")
        if python_residual > EXPECTED_THRESHOLDS["solver_residual"]:
            raise ValueError(f"Python theta_{index} fails the residual gate.")
        if julia_residual > EXPECTED_THRESHOLDS["solver_residual"]:
            raise ValueError(f"Julia theta_{index} fails the residual gate.")
        if python.get("accepted") is not True or python.get("converged") is not True:
            raise ValueError(f"Python theta_{index} was not accepted and converged.")
        if int(julia.get("convergence_flag", -1)) != 0:
            raise ValueError(f"Julia theta_{index} did not report convergence.")
        per_case.append(
            {
                "case_id": comparison["case_id"],
                "theta_index": index,
                "phi_pi": expected_phi_pi,
                "phi_y": expected_phi_y,
                "steady_state_max_abs": steady_max,
                "tree_max_abs": tree_max,
                "tree_rmse": tree_rmse,
                "mean_path_max_abs": mean_max,
                "mean_path_rmse": mean_rmse,
                "python_residual_norm": python_residual,
                "julia_residual_norm": julia_residual,
            }
        )

    if _canonical_hash(python_cases[0]["tree_states"]) != _canonical_hash(
        python_cases[4]["tree_states"]
    ):
        raise ValueError("The duplicated Python reference case is not reproducible.")
    if _canonical_hash(julia_cases[0]["tree_states"]) != _canonical_hash(
        julia_cases[4]["tree_states"]
    ):
        raise ValueError("The duplicated Julia reference case is not reproducible.")

    invocation = _mapping(report.get("julia_invocation"), field="julia_invocation")
    if invocation.get("exit_code") != 0:
        raise ValueError("The Julia exporter did not exit successfully.")
    environment = _mapping(report.get("environment"), field="environment")
    if environment.get("jax_enable_x64") is not True:
        raise ValueError("The report was not generated with JAX 64-bit mode enabled.")
    return {
        "repositories": repositories,
        "environment": dict(environment),
        "julia_version": julia_export.get("julia_version"),
        "per_case": per_case,
        "duplicate_reference": {
            "indices": [0, 4],
            "python_tree_sha256": _canonical_hash(python_cases[0]["tree_states"]),
            "julia_tree_sha256": _canonical_hash(julia_cases[0]["tree_states"]),
        },
    }


def build_evidence(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    validated = validate_report(report)
    rows = validated["per_case"]
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "evidence_level": "independent_reduced_horizon_nonlinear_tree_parity",
        "source_report": {
            "name": report_path.name,
            "sha256": _sha256(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "model": {
            "name": EXPECTED_MODEL,
            "equations_and_variables": 23,
            "shocks": 3,
            "parameters": 16,
            "sha256": EXPECTED_MODEL_SHA256,
        },
        "design": {
            "evaluations": 8,
            "unique_parameter_points": 7,
            "duplicate_reference_indices": [0, 4],
            "theta_points": [list(point) for point in EXPECTED_THETA_POINTS],
            "deterministic_first_period_shock": list(EXPECTED_FIRST_PERIOD_SHOCK),
            "configuration": EXPECTED_CONFIG,
            "group_counts": EXPECTED_GROUP_COUNTS,
        },
        "repositories": validated["repositories"],
        "environment": {
            **validated["environment"],
            "julia": validated["julia_version"],
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "metrics": {
            "joint_cases_passed": 8,
            "joint_cases_attempted": 8,
            "worst_steady_state_max_abs": max(row["steady_state_max_abs"] for row in rows),
            "worst_tree_max_abs": max(row["tree_max_abs"] for row in rows),
            "worst_mean_path_max_abs": max(row["mean_path_max_abs"] for row in rows),
            "worst_python_residual_norm": max(row["python_residual_norm"] for row in rows),
            "worst_julia_residual_norm": max(row["julia_residual_norm"] for row in rows),
            "per_case": rows,
        },
        "duplicate_reference": validated["duplicate_reference"],
        "claims": {
            "supported": [
                "The local and upstream files are byte-identical copies of the declared full nonlinear Gali model.",
                "Independent Python and MacroModelling.jl nonlinear SEP solves agree on the complete reduced-horizon sparse scenario tree at all eight evaluations within the frozen tolerance.",
                "The expected paths implied by those trees agree at all eight evaluations within the frozen tolerance.",
                "The test activates all three deterministic shock channels and preserves every attempted design point.",
            ],
            "not_supported": [
                "production-horizon nonlinear trajectory parity",
                "nonlinear likelihood parity",
                "posterior fidelity or parameter recovery",
                "GPU correctness or throughput",
                "publication readiness",
            ],
        },
        "publication_ready": False,
    }


def _scientific(value: float) -> str:
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times 10^{{{int(exponent)}}}$"


def render_markdown(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    repositories = evidence["repositories"]
    lines = [
        "# Full Galí nonlinear SEP Julia/Python parity evidence",
        "",
        "**Evidence level:** independent reduced-horizon nonlinear scenario-tree parity",
        "",
        f"- Joint cases passed: {metrics['joint_cases_passed']}/{metrics['joint_cases_attempted']} across seven unique policy-rule points.",
        f"- Worst complete-tree maximum absolute difference: {metrics['worst_tree_max_abs']:.6e} (threshold {EXPECTED_THRESHOLDS['tree_max_abs']:.1e}).",
        f"- Worst expected-path maximum absolute difference: {metrics['worst_mean_path_max_abs']:.6e} (threshold {EXPECTED_THRESHOLDS['mean_path_max_abs']:.1e}).",
        f"- Worst Python/Julia solver residuals: {metrics['worst_python_residual_norm']:.6e} / {metrics['worst_julia_residual_norm']:.6e}.",
        f"- SurrogateNN_DSGE revision: `{repositories['surrogatenn_dsge']['commit']}` (clean).",
        f"- MacroModelling.jl revision: `{repositories['macromodelling_jl']['commit']}` (clean).",
        f"- Model SHA-256: `{EXPECTED_MODEL_SHA256}`.",
        "",
        "## Claim boundary",
        "",
        "This result validates a three-period, order-one, three-node-per-shock sparse nonlinear SEP tree with all three deterministic shock channels active. It does not validate the production horizon, nonlinear likelihood, posterior, or GPU performance.",
        "",
        "## Per-case discrepancies",
        "",
        "| Case | $\\phi_\\pi$ | $\\phi_y$ | Tree max abs | Mean-path max abs | Python residual | Julia residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["per_case"]:
        lines.append(
            f"| `{row['case_id']}` | {row['phi_pi']:.3f} | {row['phi_y']:.3f} | "
            f"{row['tree_max_abs']:.3e} | {row['mean_path_max_abs']:.3e} | "
            f"{row['python_residual_norm']:.3e} | {row['julia_residual_norm']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def render_latex(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    tree = _scientific(metrics["worst_tree_max_abs"])
    mean = _scientific(metrics["worst_mean_path_max_abs"])
    py_residual = _scientific(metrics["worst_python_residual_norm"])
    jl_residual = _scientific(metrics["worst_julia_residual_norm"])
    return rf"""% Generated by benchmarks/build_gali_sep_parity_evidence.py. Do not edit.
\providecommand{{\GaliSEPTreeMax}}{{{tree}}}
\providecommand{{\GaliSEPMeanMax}}{{{mean}}}
\providecommand{{\GaliSEPPythonResidual}}{{{py_residual}}}
\providecommand{{\GaliSEPJuliaResidual}}{{{jl_residual}}}
\begin{{table}}[t]
\centering
\caption{{Independent Julia--Python nonlinear SEP parity on the complete Gal\'i source}}
\label{{tab:gali-sep-parity}}
\small
\begin{{tabular}}{{lrrl}}
\toprule
Gate & Observed & Threshold & Result \\
\midrule
Joint design evaluations & 8/8 & 8/8 & Pass \\
Complete scenario-tree maximum absolute difference & \GaliSEPTreeMax & $10^{{-6}}$ & Pass \\
Expected-path maximum absolute difference & \GaliSEPMeanMax & $10^{{-8}}$ & Pass \\
Python solver residual norm & \GaliSEPPythonResidual & $10^{{-8}}$ & Pass \\
Julia solver residual norm & \GaliSEPJuliaResidual & $10^{{-8}}$ & Pass \\
Clean source revisions & 2/2 & 2/2 & Pass \\
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\parbox{{0.91\textwidth}}{{\footnotesize\textit{{Notes:}} The test uses a three-period, order-one sparse tree with three Gauss--Hermite nodes per shock, group counts $(1,1,9,9)$, and first-period standardized innovations $(0.12,-0.08,0.10)$ across all three shock channels. Eight evaluations cover seven unique $(\phi_\pi,\phi_y)$ points; indices 0 and 4 repeat the reference calibration and reproduce bit-for-bit within each implementation. This closes only the reduced-horizon nonlinear-tree subgate; production-horizon likelihood, posterior, and GPU gates remain open.}}
\end{{table}}
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Raw nonlinear SEP parity JSON report.")
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
    (args.output_dir / "gali_sep_parity_raw.json").write_bytes(args.report.read_bytes())
    (args.output_dir / "gali_sep_parity_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gali_sep_parity_evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )
    (args.output_dir / "gali_sep_parity_results.tex").write_text(
        render_latex(evidence), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
