from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "validate_gali_full_julia_sep_likelihood.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_gali_full_julia_sep_likelihood", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matching_payloads(module: ModuleType) -> tuple[SimpleNamespace, dict, dict]:
    model = SimpleNamespace(
        timings=SimpleNamespace(var=("x", "y", "z"), exo=("e1", "e2", "e3"))
    )
    known = module.KNOWN_SHOCKS.tolist()
    audit = {
        "recovered_shocks": known.copy(),
        "jacobian": np.eye(3).tolist(),
        "loglikelihood": 1.25,
        "logabsdet": 0.0,
        "residual_norm": 1.0e-12,
        "final_sep_error": 2.0e-13,
        "iterations": 3,
        "predict_calls": 12,
        "stop_reason": "residual_tolerance",
    }
    python = {
        "steady_state": [1.0, 2.0, 3.0],
        "observation_deviation": [0.1, 0.2, 0.3],
        "public_audit_abs_diff": 0.0,
        "audit": audit,
    }
    julia_payload = {
        "model": {"variable_names": ["z", "x", "y"]},
        "result": {
            "status": "ok",
            "steady_state": [3.0, 1.0, 2.0],
            "observation_deviation": [0.1, 0.2, 0.3],
            "recovered_shocks": known.copy(),
            "jacobian": np.eye(3).tolist(),
            "loglikelihood": 1.25,
            "logabsdet": 0.0,
            "residual_norm": 1.0e-12,
            "final_sep_error": 2.0e-13,
            "iterations": 3,
            "predict_calls": 12,
            "stop_reason": "residual_tolerance",
        },
    }
    return model, python, julia_payload


def test_frozen_likelihood_design_uses_five_period_sep_and_strict_inversion() -> None:
    module = _load_module()
    config = module._config()
    inversion = module._inversion_json()

    assert config.periods == 5
    assert config.branching_order == 1
    assert config.nnodes == 3
    assert config.sparse_tree is True
    assert config.max_iter == 120
    assert config.tol == module.SOLVER_RESIDUAL_TOL
    assert inversion["max_iter"] == 12
    assert inversion["residual_tol"] == module.INVERSION_RESIDUAL_TOL
    assert inversion["step_tol"] == 1.0e-10
    assert inversion["lambda"] == 1.0e-8
    assert module.OBSERVABLE_NAMES == ("log_y", "pi_ann", "i_ann")
    np.testing.assert_allclose(module.KNOWN_SHOCKS, [0.02, -0.01, 0.015])


def test_comparison_aligns_steady_state_and_accepts_matching_audit_data() -> None:
    module = _load_module()
    model, python, julia_payload = _matching_payloads(module)

    comparison, gates = module._compare(
        model=model, python=python, julia_payload=julia_payload
    )

    assert all(gates.values())
    assert comparison["steady_state"]["max_abs"] == 0.0
    assert comparison["recovered_shocks"]["max_abs"] == 0.0
    assert comparison["jacobian"]["max_abs"] == 0.0
    assert comparison["loglikelihood_abs_diff"] == 0.0


def test_comparison_rejects_material_jacobian_or_likelihood_difference() -> None:
    module = _load_module()
    model, python, julia_payload = _matching_payloads(module)
    julia_payload["result"]["jacobian"][1][2] = 2.0e-4
    julia_payload["result"]["loglikelihood"] += 2.0e-4

    _, gates = module._compare(
        model=model, python=python, julia_payload=julia_payload
    )

    assert gates["finite_difference_jacobian_parity"] is False
    assert gates["loglikelihood_parity"] is False


def test_comparison_rejects_recovered_shock_that_misses_known_generator() -> None:
    module = _load_module()
    model, python, julia_payload = _matching_payloads(module)
    julia_payload["result"]["recovered_shocks"][0] += 2.0e-6

    _, gates = module._compare(
        model=model, python=python, julia_payload=julia_payload
    )

    assert gates["recovered_shock_parity"] is False
    assert gates["julia_recovers_known_shocks"] is False
