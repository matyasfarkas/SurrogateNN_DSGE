from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "build_gali_core_parity_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_gali_core_parity_evidence", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> dict[str, Any]:
    module = _load_module()
    comparisons = []
    julia_cases = []
    for index, (phi_pi, phi_y) in enumerate(module.EXPECTED_THETA_POINTS):
        case_id = f"theta_{index}"
        comparisons.append(
            {
                "case_id": case_id,
                "theta_index": index,
                "parameters": {"ϕᵖⁱ": phi_pi, "ϕʸ": phi_y},
                "python_status": "ok",
                "julia_status": "ok",
                "passed": True,
                "steady_state": {"max_abs": 9.0e-14, "rmse": 2.0e-14},
                "first_order_solution": {"max_abs": 2.0e-11, "rmse": 2.0e-12},
                "python_steady_state_residual_norm": 1.0e-16,
            }
        )
        julia_cases.append({"case_id": case_id, "status": "ok"})
    clean_repo = {
        "commit": "a" * 40,
        "dirty": False,
        "status_short": [],
        "tracked_diff_sha256": "0" * 64,
    }
    return {
        "schema_version": module.REPORT_SCHEMA,
        "status": "pass",
        "publication_ready": False,
        "gates": {
            "source_identity": True,
            "all_design_points_numerically_pass": True,
            "clean_repository_provenance": True,
        },
        "thresholds": dict(module.EXPECTED_THRESHOLDS),
        "design": {
            "theta_points": [list(point) for point in module.EXPECTED_THETA_POINTS],
            "case_count": 8,
            "unique_theta_count": 7,
            "duplicate_reference_indices": [0, 4],
        },
        "sources": {
            "source_identity_passed": True,
            "local_model": {"sha256": module.EXPECTED_MODEL_SHA256},
            "upstream_model": {"sha256": module.EXPECTED_MODEL_SHA256},
        },
        "repositories": {
            "surrogatenn_dsge": clean_repo,
            "macromodelling_jl": {**clean_repo, "commit": "b" * 40},
        },
        "environment": {"python": "3.13", "jax": "0.8", "julia": "1.12"},
        "julia_invocation": {"exit_code": 0},
        "julia_export": {
            "schema_version": module.EXPORT_SCHEMA,
            "model": {
                "symbol": module.EXPECTED_MODEL,
                "sha256": module.EXPECTED_MODEL_SHA256,
                "variable_names": [f"x{i}" for i in range(23)],
                "parameter_names": [f"p{i}" for i in range(16)],
            },
            "cases": julia_cases,
        },
        "comparisons": comparisons,
    }


def test_build_evidence_accepts_complete_scoped_pass(tmp_path: Path) -> None:
    module = _load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    evidence = module.build_evidence(report_path)
    assert evidence["metrics"]["joint_cases_passed"] == 8
    assert evidence["metrics"]["worst_first_order_max_abs"] == 2.0e-11
    assert evidence["publication_ready"] is False
    assert "nonlinear likelihood parity" in evidence["claims"]["not_supported"]


def test_rejects_dropped_design_case() -> None:
    module = _load_module()
    report = _report()
    report["comparisons"].pop()
    with pytest.raises(ValueError, match="all eight comparisons"):
        module.validate_report(report)


def test_rejects_dirty_upstream() -> None:
    module = _load_module()
    report = _report()
    report["repositories"]["macromodelling_jl"]["dirty"] = True
    with pytest.raises(ValueError, match="was not clean"):
        module.validate_report(report)


def test_rejects_relaxed_threshold() -> None:
    module = _load_module()
    report = _report()
    report["thresholds"]["first_order_max_abs"] = 1.0e-6
    with pytest.raises(ValueError, match="Unexpected threshold"):
        module.validate_report(report)


def test_rejects_material_case_error() -> None:
    module = _load_module()
    report = _report()
    report["comparisons"][3]["first_order_solution"]["max_abs"] = 2.0e-7
    with pytest.raises(ValueError, match="fails first-order parity"):
        module.validate_report(report)
