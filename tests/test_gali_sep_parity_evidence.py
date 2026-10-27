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
        / "build_gali_sep_parity_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_gali_sep_parity_evidence", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(index: int) -> list[list[list[float]]]:
    offset = 0.0 if index in (0, 4) else index * 0.01
    return [
        [[offset, 1.0 + offset]],
        [[0.1 + offset, 1.1 + offset]],
        [[0.2 + offset, 1.2 + offset] for _ in range(9)],
        [[0.3 + offset, 1.3 + offset] for _ in range(9)],
    ]


def _report() -> dict[str, Any]:
    module = _load_module()
    comparisons = []
    python_cases = []
    julia_cases = []
    for index, (phi_pi, phi_y) in enumerate(module.EXPECTED_THETA_POINTS):
        case_id = f"theta_{index}"
        python_cases.append(
            {
                "case_id": case_id,
                "status": "ok",
                "tree_states": _tree(index),
            }
        )
        julia_cases.append(
            {
                "case_id": case_id,
                "status": "ok",
                "group_counts": list(module.EXPECTED_GROUP_COUNTS),
                "convergence_flag": 0,
                "final_error": 8.0e-9,
                "tree_states": _tree(index),
            }
        )
        comparisons.append(
            {
                "case_id": case_id,
                "theta_index": index,
                "parameters": {"ϕᵖⁱ": phi_pi, "ϕʸ": phi_y},
                "python_status": "ok",
                "julia_status": "ok",
                "passed": True,
                "group_counts": list(module.EXPECTED_GROUP_COUNTS),
                "steady_state": {"max_abs": 9.0e-14, "rmse": 2.0e-14},
                "tree": {"max_abs": 1.7e-7, "rmse": 2.0e-8},
                "mean_path": {"max_abs": 7.0e-9, "rmse": 8.0e-10},
                "python": {
                    "residual_norm": 9.0e-9,
                    "converged": True,
                    "accepted": True,
                    "iterations": 4,
                },
                "julia": {"final_error": 8.0e-9, "convergence_flag": 0},
            }
        )
    clean_repo = {
        "commit": "a" * 40,
        "dirty": False,
        "status_short": [],
    }
    return {
        "schema_version": module.REPORT_SCHEMA,
        "status": "pass",
        "publication_ready": False,
        "gates": {
            "source_identity": True,
            "all_cases_pass": True,
            "clean_provenance": True,
        },
        "thresholds": dict(module.EXPECTED_THRESHOLDS),
        "design": {
            "theta_indices": list(range(8)),
            "theta_points": [list(point) for point in module.EXPECTED_THETA_POINTS],
            "deterministic_first_period_shock": list(
                module.EXPECTED_FIRST_PERIOD_SHOCK
            ),
            "deterministic_shocks": [
                list(row) for row in module.EXPECTED_SHOCK_MATRIX
            ],
            "config": dict(module.EXPECTED_CONFIG),
        },
        "sources": {
            "local_model": {"sha256": module.EXPECTED_MODEL_SHA256},
            "upstream_model": {"sha256": module.EXPECTED_MODEL_SHA256},
        },
        "provenance": {
            "repository": clean_repo,
            "upstream_repository": {**clean_repo, "commit": "b" * 40},
        },
        "environment": {
            "python": "3.13",
            "platform": "test-platform",
            "jax": "0.8",
            "numpy": "2.0",
            "jax_enable_x64": True,
        },
        "julia_invocation": {"exit_code": 0},
        "julia_export": {
            "schema_version": module.EXPORT_SCHEMA,
            "julia_version": "1.12",
            "model": {
                "symbol": module.EXPECTED_MODEL,
                "sha256": module.EXPECTED_MODEL_SHA256,
                "variable_names": [f"x{i}" for i in range(23)],
                "shock_names": ["eps_a", "eps_nu", "eps_z"],
                "parameter_names": [f"p{i}" for i in range(16)],
            },
            "config": dict(module.EXPECTED_CONFIG),
            "cases": julia_cases,
        },
        "python_cases": python_cases,
        "comparisons": comparisons,
    }


def test_build_evidence_accepts_complete_scoped_pass(tmp_path: Path) -> None:
    module = _load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    evidence = module.build_evidence(report_path)
    assert evidence["metrics"]["joint_cases_passed"] == 8
    assert evidence["metrics"]["worst_tree_max_abs"] == 1.7e-7
    assert evidence["duplicate_reference"]["indices"] == [0, 4]
    assert evidence["publication_ready"] is False
    assert "nonlinear likelihood parity" in evidence["claims"]["not_supported"]


def test_rejects_dropped_design_case() -> None:
    module = _load_module()
    report = _report()
    report["python_cases"].pop()
    with pytest.raises(ValueError, match="all eight cases"):
        module.validate_report(report)


def test_rejects_dirty_upstream() -> None:
    module = _load_module()
    report = _report()
    report["provenance"]["upstream_repository"]["dirty"] = True
    with pytest.raises(ValueError, match="was not clean"):
        module.validate_report(report)


def test_rejects_relaxed_threshold() -> None:
    module = _load_module()
    report = _report()
    report["thresholds"]["tree_max_abs"] = 1.0e-4
    with pytest.raises(ValueError, match="Unexpected threshold"):
        module.validate_report(report)


def test_rejects_material_tree_error() -> None:
    module = _load_module()
    report = _report()
    report["comparisons"][3]["tree"]["max_abs"] = 2.0e-6
    with pytest.raises(ValueError, match="fails full-tree parity"):
        module.validate_report(report)


def test_rejects_nonreproducible_duplicate_reference() -> None:
    module = _load_module()
    report = _report()
    report["julia_export"]["cases"][4]["tree_states"][1][0][0] += 1.0e-12
    with pytest.raises(ValueError, match="duplicated Julia reference"):
        module.validate_report(report)


def test_rejects_altered_shock_scenario() -> None:
    module = _load_module()
    report = _report()
    report["design"]["deterministic_shocks"][0][1] = -0.07
    with pytest.raises(ValueError, match="frozen three-shock scenario"):
        module.validate_report(report)


def test_rejects_altered_solver_configuration() -> None:
    module = _load_module()
    report = _report()
    report["design"]["config"]["periods"] = 4
    with pytest.raises(ValueError, match="Unexpected SEP configuration periods"):
        module.validate_report(report)
