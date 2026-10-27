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
        / "validate_gali_full_julia_core_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_gali_full_julia_core_parity", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parameter_injection_uses_names_not_positions() -> None:
    module = _load_module()
    names = ("std_a", "ϕʸ", "β", "ϕᵖⁱ")
    values = module.inject_parameter_values(
        [0.1, 0.2, 0.99, 1.5],
        names,
        {"ϕᵖⁱ": 1.32, "ϕʸ": 0.14},
    )
    np.testing.assert_allclose(values, [0.1, 0.14, 0.99, 1.32])


def test_comparison_aligns_labels_and_accepts_small_differences() -> None:
    module = _load_module()
    python_case = {
        "case_id": "theta_0",
        "theta_index": 0,
        "parameters": {"ϕᵖⁱ": 1.5, "ϕʸ": 0.125},
        "status": "ok",
        "steady_state": [1.0, 2.0],
        "steady_state_residual_norm": 1.0e-13,
        "first_order_converged": True,
        "first_order_solution": [[3.0, 4.0], [5.0, 6.0]],
    }
    julia_case = {
        "case_id": "theta_0",
        "status": "ok",
        "steady_state": {"names": ["y", "x"], "values": [2.0, 1.0]},
        "first_order": {
            "row_names": ["y", "x"],
            "column_names_raw": ["eps₍ₓ₎", "x₍₋₁₎"],
            "solution_matrix": [[6.0, 5.0], [4.0, 3.0]],
        },
    }
    result = module.compare_case(
        python_case,
        julia_case,
        variable_names=("x", "y"),
        solution_column_names=("x", "eps"),
    )
    assert result["passed"] is True
    assert result["steady_state"]["max_abs"] == 0.0
    assert result["first_order_solution"]["max_abs"] == 0.0


@pytest.mark.parametrize("failed_side", ["python", "julia"])
def test_comparison_preserves_solver_failures(failed_side: str) -> None:
    module = _load_module()
    python_case = {
        "case_id": "theta_1",
        "theta_index": 1,
        "parameters": {"ϕᵖⁱ": 1.32, "ϕʸ": 0.14},
        "status": "error" if failed_side == "python" else "ok",
    }
    julia_case = {
        "case_id": "theta_1",
        "status": "error" if failed_side == "julia" else "ok",
    }
    result = module.compare_case(
        python_case,
        julia_case,
        variable_names=("x",),
        solution_column_names=("x",),
    )
    assert result["passed"] is False
    assert "failed" in result["error"]


def test_comparison_rejects_material_matrix_mismatch() -> None:
    module = _load_module()
    python_case = {
        "case_id": "theta_2",
        "theta_index": 2,
        "parameters": {"ϕᵖⁱ": 1.38, "ϕʸ": 0.135},
        "status": "ok",
        "steady_state": [1.0],
        "steady_state_residual_norm": 0.0,
        "first_order_converged": True,
        "first_order_solution": [[1.0]],
    }
    julia_case = {
        "case_id": "theta_2",
        "status": "ok",
        "steady_state": {"names": ["x"], "values": [1.0]},
        "first_order": {
            "row_names": ["x"],
            "column_names_raw": ["x₍₋₁₎"],
            "solution_matrix": [[1.0 + 1.0e-5]],
        },
    }
    result = module.compare_case(
        python_case,
        julia_case,
        variable_names=("x",),
        solution_column_names=("x",),
    )
    assert result["passed"] is False
    assert result["first_order_solution"]["max_abs"] > 1.0e-8
