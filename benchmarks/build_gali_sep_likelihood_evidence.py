#!/usr/bin/env python3
"""Build hash-verified manuscript evidence for full-Gali SEP likelihood parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "surrogatenn-dsge-gali-sep-likelihood-parity/v1"
EXPORT_SCHEMA = "surrogatenn-dsge-gali-julia-sep-likelihood-export/v1"
EVIDENCE_SCHEMA = "surrogatenn-dsge-gali-sep-likelihood-evidence/v1"
EXPECTED_MODEL = "Gali_2015_chapter_3_nonlinear"
EXPECTED_MODEL_SHA256 = "53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55"
EXPECTED_GATES = {
    "source_identity",
    "model_identity",
    "clean_provenance",
    "steady_state_parity",
    "observation_demeaning_parity",
    "public_python_matches_audit",
    "recovered_shock_parity",
    "python_recovers_known_shocks",
    "julia_recovers_known_shocks",
    "finite_difference_jacobian_parity",
    "loglikelihood_parity",
    "python_inversion_residual",
    "julia_inversion_residual",
    "python_sep_solver_residual",
    "julia_sep_solver_residual",
}
EXPECTED_THRESHOLDS = {
    "steady_state_max_abs": 1.0e-10,
    "public_audit_abs": 1.0e-8,
    "recovered_shock_max_abs": 1.0e-7,
    "known_shock_max_abs": 1.0e-7,
    "jacobian_max_abs": 1.0e-5,
    "loglikelihood_abs": 1.0e-4,
    "inversion_residual": 1.0e-9,
    "solver_residual": 1.0e-12,
}
EXPECTED_CONFIG = {
    "periods": 5,
    "branching_order": 1,
    "nnodes": 3,
    "sparse_tree": True,
    "max_iter": 120,
    "tol": 1.0e-12,
    "accept_tol": 1.0e-12,
    "linear_solver": "qr",
    "fallback_solver": None,
    "stall_iters": 25,
    "stall_rel_tol": 1.0e-4,
    "stall_abs_tol": 1.0e-10,
}
EXPECTED_INVERSION = {
    "max_iter": 12,
    "step_tol": 1.0e-10,
    "residual_tol": 1.0e-9,
    "lambda": 1.0e-8,
    "finite_difference_step_rule": "max(cbrt(eps)*max(1,abs(shock),sigma),1e-6)",
}
EXPECTED_OBSERVABLES = ["log_y", "pi_ann", "i_ann"]
EXPECTED_SHOCKS = ["eps_a", "eps_nu", "eps_z"]
EXPECTED_KNOWN_SHOCKS = np.asarray([0.02, -0.01, 0.015], dtype=np.float64)


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


def _array(value: Any, *, field: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite with shape {shape}, got {result.shape}.")
    return result


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {left.shape} and {right.shape}.")
    return float(np.max(np.abs(left - right)))


def _validate_commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{field} must be a 40-character Git commit.")
    if any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field} is not hexadecimal.")
    return value


def _validate_exact_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, field: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{field} fields changed.")
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(expected_value, float):
            actual_value = _finite(actual_value, field=f"{field}.{name}")
        if actual_value != expected_value:
            raise ValueError(
                f"Unexpected {field}.{name}={actual_value!r}; expected {expected_value!r}."
            )


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unexpected SEP likelihood report schema.")
    if report.get("status") != "pass":
        raise ValueError("SEP likelihood parity report did not pass.")
    if report.get("publication_ready") is not False:
        raise ValueError("The scoped report must not claim publication readiness.")

    gates = _mapping(report.get("gates"), field="gates")
    if set(gates) != EXPECTED_GATES or not all(value is True for value in gates.values()):
        raise ValueError("Every declared SEP likelihood parity gate must pass.")
    _validate_exact_mapping(
        _mapping(report.get("thresholds"), field="thresholds"),
        EXPECTED_THRESHOLDS,
        field="thresholds",
    )

    design = _mapping(report.get("design"), field="design")
    if design.get("model_symbol") != EXPECTED_MODEL:
        raise ValueError("Unexpected model symbol in the design.")
    if design.get("observable_names") != EXPECTED_OBSERVABLES:
        raise ValueError("The observable design changed.")
    if design.get("structural_shocks") != EXPECTED_SHOCKS:
        raise ValueError("The structural-shock order changed.")
    if design.get("parameters") != {"ϕᵖⁱ": 1.5, "ϕʸ": 0.125}:
        raise ValueError("The policy-rule calibration changed.")
    known_shocks = _array(
        design.get("known_generation_shocks"), field="known shocks", shape=(3,)
    )
    if not np.array_equal(known_shocks, EXPECTED_KNOWN_SHOCKS):
        raise ValueError("The known shock vector changed.")
    shock_sigmas = _array(
        design.get("shock_sigmas"), field="shock sigmas", shape=(3,)
    )
    if not np.array_equal(shock_sigmas, np.ones(3)):
        raise ValueError("The standardized shock scales changed.")
    _validate_exact_mapping(
        _mapping(design.get("config"), field="design.config"),
        EXPECTED_CONFIG,
        field="design.config",
    )
    _validate_exact_mapping(
        _mapping(design.get("inversion"), field="design.inversion"),
        EXPECTED_INVERSION,
        field="design.inversion",
    )

    sources = _mapping(report.get("sources"), field="sources")
    for name in ("local_model", "upstream_model"):
        source = _mapping(sources.get(name), field=f"sources.{name}")
        if source.get("sha256") != EXPECTED_MODEL_SHA256:
            raise ValueError(f"Unexpected {name} source hash.")

    provenance = _mapping(report.get("provenance"), field="provenance")
    repositories: dict[str, Any] = {}
    for source_name, evidence_name in (
        ("repository", "surrogatenn_dsge"),
        ("upstream_repository", "macromodelling_jl"),
    ):
        repository = _mapping(provenance.get(source_name), field=source_name)
        if repository.get("dirty") is not False or repository.get("status_short") != []:
            raise ValueError(f"{source_name} was not clean.")
        repositories[evidence_name] = {
            "commit": _validate_commit(
                repository.get("commit"), field=f"{source_name}.commit"
            ),
            "dirty": False,
        }

    julia_export = _mapping(report.get("julia_export"), field="julia_export")
    if julia_export.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("Unexpected Julia export schema.")
    julia_model = _mapping(julia_export.get("model"), field="julia_export.model")
    if julia_model.get("symbol") != EXPECTED_MODEL:
        raise ValueError("Unexpected Julia model symbol.")
    if julia_model.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("Unexpected Julia model hash.")
    variable_names = julia_model.get("variable_names")
    if not isinstance(variable_names, list) or len(variable_names) != 23:
        raise ValueError("The Julia export is not the 23-variable full model.")
    if julia_model.get("shock_names") != EXPECTED_SHOCKS:
        raise ValueError("The Julia shock order changed.")
    if len(julia_model.get("parameter_names", [])) != 16:
        raise ValueError("The Julia export does not declare 16 parameters.")
    julia = _mapping(julia_export.get("result"), field="julia_export.result")
    python = _mapping(report.get("python"), field="python")
    python_audit = _mapping(python.get("audit"), field="python.audit")
    if julia.get("status") != "ok" or python.get("status") != "ok":
        raise ValueError("At least one implementation failed.")

    python_steady = _array(python.get("steady_state"), field="python steady state", shape=(23,))
    julia_steady_raw = _array(julia.get("steady_state"), field="Julia steady state", shape=(23,))
    local_variable_names = [
        "A", "C", "MC", "M_real", "N", "Pi", "Pi_star", "Q", "R", "S",
        "W_real", "Y", "Z", "i_ann", "log_N", "log_W_real", "log_y", "nu",
        "pi_ann", "r_real_ann", "realinterest", "x_aux_1", "x_aux_2",
    ]
    if set(variable_names) != set(local_variable_names):
        raise ValueError("The full-model variable-name set changed.")
    julia_index = {name: index for index, name in enumerate(variable_names)}
    julia_steady = julia_steady_raw[[julia_index[name] for name in local_variable_names]]
    steady_state_max_abs = _max_abs(python_steady, julia_steady)

    python_observation = _array(
        python.get("observation_deviation"), field="Python observation deviation", shape=(3,)
    )
    julia_observation = _array(
        julia.get("observation_deviation"), field="Julia observation deviation", shape=(3,)
    )
    observation_max_abs = _max_abs(python_observation, julia_observation)
    python_shocks = _array(
        python_audit.get("recovered_shocks"), field="Python recovered shocks", shape=(3,)
    )
    julia_shocks = _array(
        julia.get("recovered_shocks"), field="Julia recovered shocks", shape=(3,)
    )
    recovered_shock_max_abs = _max_abs(python_shocks, julia_shocks)
    python_known_max_abs = _max_abs(python_shocks, EXPECTED_KNOWN_SHOCKS)
    julia_known_max_abs = _max_abs(julia_shocks, EXPECTED_KNOWN_SHOCKS)
    python_jacobian = _array(
        python_audit.get("jacobian"), field="Python Jacobian", shape=(3, 3)
    )
    julia_jacobian = _array(julia.get("jacobian"), field="Julia Jacobian", shape=(3, 3))
    jacobian_max_abs = _max_abs(python_jacobian, julia_jacobian)
    python_ll = _finite(python_audit.get("loglikelihood"), field="Python likelihood")
    julia_ll = _finite(julia.get("loglikelihood"), field="Julia likelihood")
    loglikelihood_abs = abs(python_ll - julia_ll)
    public_ll = _array(
        python.get("public_loglikelihood_per_period"), field="public Python likelihood", shape=(1,)
    )[0]
    public_audit_abs = abs(float(public_ll) - python_ll)
    python_inversion_residual = _finite(
        python_audit.get("residual_norm"), field="Python inversion residual"
    )
    julia_inversion_residual = _finite(
        julia.get("residual_norm"), field="Julia inversion residual"
    )
    python_solver_residual = _finite(
        python_audit.get("final_sep_error"), field="Python solver residual"
    )
    julia_solver_residual = _finite(
        julia.get("final_sep_error"), field="Julia solver residual"
    )

    recomputed = {
        "steady_state_max_abs": steady_state_max_abs,
        "observation_max_abs": observation_max_abs,
        "public_audit_abs": public_audit_abs,
        "recovered_shock_max_abs": recovered_shock_max_abs,
        "python_known_shock_max_abs": python_known_max_abs,
        "julia_known_shock_max_abs": julia_known_max_abs,
        "jacobian_max_abs": jacobian_max_abs,
        "loglikelihood_abs": loglikelihood_abs,
        "python_inversion_residual": python_inversion_residual,
        "julia_inversion_residual": julia_inversion_residual,
        "python_solver_residual": python_solver_residual,
        "julia_solver_residual": julia_solver_residual,
        "python_loglikelihood": python_ll,
        "julia_loglikelihood": julia_ll,
        "python_iterations": int(python_audit.get("iterations")),
        "julia_iterations": int(julia.get("iterations")),
        "python_predict_calls": int(python_audit.get("predict_calls")),
        "julia_predict_calls": int(julia.get("predict_calls")),
    }
    threshold_checks = {
        "steady state": steady_state_max_abs <= EXPECTED_THRESHOLDS["steady_state_max_abs"],
        "observation": observation_max_abs <= EXPECTED_THRESHOLDS["steady_state_max_abs"],
        "public audit": public_audit_abs <= EXPECTED_THRESHOLDS["public_audit_abs"],
        "shock parity": recovered_shock_max_abs <= EXPECTED_THRESHOLDS["recovered_shock_max_abs"],
        "Python known shocks": python_known_max_abs <= EXPECTED_THRESHOLDS["known_shock_max_abs"],
        "Julia known shocks": julia_known_max_abs <= EXPECTED_THRESHOLDS["known_shock_max_abs"],
        "Jacobian": jacobian_max_abs <= EXPECTED_THRESHOLDS["jacobian_max_abs"],
        "likelihood": loglikelihood_abs <= EXPECTED_THRESHOLDS["loglikelihood_abs"],
        "Python inversion residual": python_inversion_residual <= EXPECTED_THRESHOLDS["inversion_residual"],
        "Julia inversion residual": julia_inversion_residual <= EXPECTED_THRESHOLDS["inversion_residual"],
        "Python solver residual": python_solver_residual <= EXPECTED_THRESHOLDS["solver_residual"],
        "Julia solver residual": julia_solver_residual <= EXPECTED_THRESHOLDS["solver_residual"],
    }
    failed = [name for name, passed in threshold_checks.items() if not passed]
    if failed:
        raise ValueError("Recomputed numerical gates failed: " + ", ".join(failed))
    comparison = _mapping(report.get("comparison"), field="comparison")
    for name in (
        "loglikelihood_abs_diff", "python_residual_norm", "julia_residual_norm",
        "python_final_sep_error", "julia_final_sep_error",
    ):
        _finite(comparison.get(name), field=f"comparison.{name}")
    invocation = _mapping(report.get("julia_invocation"), field="julia_invocation")
    if invocation.get("exit_code") != 0:
        raise ValueError("The Julia exporter did not exit successfully.")
    return {
        "repositories": repositories,
        "metrics": recomputed,
        "julia_version": julia_export.get("julia_version"),
        "environment": report.get("environment"),
    }


def build_evidence(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    validated = validate_report(report)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "publication_ready": False,
        "evidence_level": "one-transition full-model nonlinear SEP likelihood parity",
        "raw_report": {
            "filename": report_path.name,
            "sha256": _sha256(report_path),
            "bytes": report_path.stat().st_size,
        },
        "repositories": validated["repositories"],
        "model": {
            "symbol": EXPECTED_MODEL,
            "sha256": EXPECTED_MODEL_SHA256,
            "variables": 23,
            "parameters": 16,
            "shocks": 3,
        },
        "design": {
            "sep_periods": 5,
            "branching_order": 1,
            "nodes_per_shock": 3,
            "observables": EXPECTED_OBSERVABLES,
            "known_shocks": EXPECTED_KNOWN_SHOCKS.tolist(),
            "solver_tolerance": EXPECTED_THRESHOLDS["solver_residual"],
        },
        "thresholds": EXPECTED_THRESHOLDS,
        "metrics": validated["metrics"],
        "claims": {
            "supported": [
                "one-transition nonlinear SEP inversion-likelihood parity",
                "three-shock recovery parity",
                "finite-difference observation-Jacobian parity",
            ],
            "not_supported": [
                "multi-period nonlinear likelihood recursion",
                "production-horizon likelihood parity",
                "posterior recovery",
                "GPU speed or efficiency",
                "overall publication readiness",
            ],
        },
        "environment": validated["environment"],
        "julia_version": validated["julia_version"],
    }


def _scientific(value: float) -> str:
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times 10^{{{int(exponent)}}}$"


def render_markdown(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    repositories = evidence["repositories"]
    return "\n".join(
        [
            "# Full Galí nonlinear SEP likelihood parity evidence",
            "",
            "**Evidence level:** one-transition full-model nonlinear SEP inversion-likelihood parity",
            "",
            f"- Recovered-shock maximum absolute difference: {metrics['recovered_shock_max_abs']:.6e}.",
            f"- Finite-difference Jacobian maximum absolute difference: {metrics['jacobian_max_abs']:.6e}.",
            f"- Log-likelihood absolute difference: {metrics['loglikelihood_abs']:.6e}.",
            f"- Python/Julia inversion residuals: {metrics['python_inversion_residual']:.6e} / {metrics['julia_inversion_residual']:.6e}.",
            f"- SurrogateNN_DSGE revision: `{repositories['surrogatenn_dsge']['commit']}` (clean).",
            f"- MacroModelling.jl revision: `{repositories['macromodelling_jl']['commit']}` (clean).",
            f"- Raw report SHA-256: `{evidence['raw_report']['sha256']}`.",
            "",
            "## Claim boundary",
            "",
            "This establishes one-transition nonlinear likelihood parity for the complete Galí equation system under the frozen five-period SEP solve. It does not establish multi-period filtering, production-horizon likelihood parity, posterior recovery, or GPU speed.",
            "",
        ]
    )


def render_latex(evidence: Mapping[str, Any]) -> str:
    metrics = evidence["metrics"]
    return rf"""% Generated by benchmarks/build_gali_sep_likelihood_evidence.py. Do not edit.
\providecommand{{\GaliSEPLLShockMax}}{{{_scientific(metrics['recovered_shock_max_abs'])}}}
\providecommand{{\GaliSEPLLJacobianMax}}{{{_scientific(metrics['jacobian_max_abs'])}}}
\providecommand{{\GaliSEPLLAbs}}{{{_scientific(metrics['loglikelihood_abs'])}}}
\providecommand{{\GaliSEPLLPythonResidual}}{{{_scientific(metrics['python_inversion_residual'])}}}
\providecommand{{\GaliSEPLLJuliaResidual}}{{{_scientific(metrics['julia_inversion_residual'])}}}
\begin{{table}}[t]
\centering
\caption{{Independent one-transition Julia--Python nonlinear SEP likelihood parity}}
\label{{tab:gali-sep-likelihood-parity}}
\small
\begin{{tabular}}{{@{{}}p{{0.49\linewidth}}rrl@{{}}}}
\toprule
Gate & Observed & Threshold & Result \\
\midrule
Recovered-shock maximum absolute difference & \GaliSEPLLShockMax & $10^{{-7}}$ & Pass \\
Jacobian maximum absolute difference & \GaliSEPLLJacobianMax & $10^{{-5}}$ & Pass \\
Log-likelihood absolute difference & \GaliSEPLLAbs & $10^{{-4}}$ & Pass \\
Python inversion residual norm & \GaliSEPLLPythonResidual & $10^{{-9}}$ & Pass \\
Julia inversion residual norm & \GaliSEPLLJuliaResidual & $10^{{-9}}$ & Pass \\
Clean source revisions & 2/2 & 2/2 & Pass \\
\bottomrule
\end{{tabular}}

\vspace{{0.4em}}
\parbox{{0.91\textwidth}}{{\footnotesize\textit{{Notes:}} The observation is generated from standardized innovations $(0.02,-0.01,0.015)$ in the complete 23-equation Gal\'i model. Both adapters invert the same $(\log Y,4\log\Pi,4\log R)$ transition using a five-period, order-one sparse SEP solve, three Gauss--Hermite nodes per shock, a $10^{{-12}}$ nonlinear-solver tolerance, and the same forward-difference rule. The Python public likelihood agrees with its instrumented audit call within $10^{{-8}}$. This closes only the one-transition likelihood subgate; multi-period recursion, production-horizon likelihood, posterior, and GPU gates remain open.}}
\end{{table}}
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence = build_evidence(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gali_sep_likelihood_raw.json").write_bytes(args.report.read_bytes())
    (args.output_dir / "gali_sep_likelihood_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gali_sep_likelihood_evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )
    (args.output_dir / "gali_sep_likelihood_results.tex").write_text(
        render_latex(evidence), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
