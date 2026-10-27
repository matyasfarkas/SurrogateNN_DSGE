from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "validate_gali_full_julia_sep_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_gali_full_julia_sep_parity", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matching_cases(module: ModuleType) -> tuple[dict, dict, dict]:
    config = module._build_spec((0,))["config"]
    probabilities = module._sparse_probabilities(
        periods=config["periods"],
        branching_order=config["branching_order"],
        nnodes=config["nnodes"],
        shock_dim=3,
    )
    tree = [
        np.asarray([[1.0, 2.0]], dtype=np.float64),
        np.asarray([[1.1, 2.1]], dtype=np.float64),
        np.arange(18, dtype=np.float64).reshape(9, 2) / 10.0,
        np.arange(18, 36, dtype=np.float64).reshape(9, 2) / 10.0,
    ]
    mean_path = module._mean_path(tree, probabilities)
    python_case = {
        "case_id": "theta_0",
        "theta_index": 0,
        "parameters": {"ϕᵖⁱ": 1.5, "ϕʸ": 0.125},
        "status": "ok",
        "residual_norm": 1.0e-12,
        "converged": True,
        "accepted": True,
        "iterations": 4,
        "group_counts": [1, 1, 9, 9],
        "tree_states": [values.tolist() for values in tree],
        "mean_path": mean_path.tolist(),
    }
    julia_case = {
        "case_id": "theta_0",
        "status": "ok",
        "final_error": 1.0e-12,
        "convergence_flag": 0,
        "group_counts": [1, 1, 9, 9],
        "tree_states": [values[:, ::-1].tolist() for values in tree],
    }
    return python_case, julia_case, config


def test_spec_uses_period_major_square_shock_matrix() -> None:
    module = _load_module()
    spec = module._build_spec((0, 4))

    assert len(spec["cases"]) == 2
    assert spec["cases"][0]["parameters"] == spec["cases"][1]["parameters"]
    shocks = np.asarray(spec["cases"][0]["deterministic_shocks"])
    assert shocks.shape == (3, 3)
    np.testing.assert_allclose(shocks[0], module.DEFAULT_SHOCK)
    np.testing.assert_allclose(shocks[1:], 0.0)


def test_sparse_probabilities_match_expected_group_counts_and_normalize() -> None:
    module = _load_module()
    probabilities = module._sparse_probabilities(
        periods=3,
        branching_order=1,
        nnodes=3,
        shock_dim=3,
    )

    assert tuple(len(values) for values in probabilities) == (1, 1, 9, 9)
    np.testing.assert_allclose([np.sum(values) for values in probabilities], 1.0)


def test_comparison_aligns_variable_names_and_accepts_matching_trees() -> None:
    module = _load_module()
    python_case, julia_case, config = _matching_cases(module)

    result = module._compare_case(
        python_case,
        julia_case,
        julia_names=("y", "x"),
        python_names=("x", "y"),
        config=config,
    )

    assert result["passed"] is True
    assert result["tree"]["max_abs"] == 0.0
    assert result["mean_path"]["max_abs"] <= 1.0e-14


def test_comparison_rejects_material_tree_mismatch() -> None:
    module = _load_module()
    python_case, julia_case, config = _matching_cases(module)
    julia_case["tree_states"][2][3][0] += 2.0e-6

    result = module._compare_case(
        python_case,
        julia_case,
        julia_names=("y", "x"),
        python_names=("x", "y"),
        config=config,
    )

    assert result["passed"] is False
    assert result["tree"]["max_abs"] > module.TREE_MAX_ABS_TOL


@pytest.mark.parametrize("failed_side", ["python", "julia"])
def test_comparison_preserves_solver_failures(failed_side: str) -> None:
    module = _load_module()
    python_case, julia_case, config = _matching_cases(module)
    if failed_side == "python":
        python_case["status"] = "error"
    else:
        julia_case["status"] = "error"

    result = module._compare_case(
        python_case,
        julia_case,
        julia_names=("y", "x"),
        python_names=("x", "y"),
        config=config,
    )

    assert result["passed"] is False
    assert "failed" in result["error"]
