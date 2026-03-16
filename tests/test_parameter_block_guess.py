from __future__ import annotations

import numpy as np

import surrogatenn_dsge.model as model_module
from surrogatenn_dsge import parse_macro_model, solve_steady_state


STEADY_STATE_GUESS_SOURCE = """
@model steady_state_guess begin
    x[0]^2 = level
end

@parameters steady_state_guess guess = Dict(:x => -3.0) begin
    level = 4.0
end
"""


CALIBRATED_PARAMETER_GUESS_SOURCE = """
@model calibrated_parameter_guess begin
    x[0] = a
end

@parameters calibrated_parameter_guess guess = Dict("x" => -3.0, "a" => 5.0) begin
    x[ss] = target | a
    target = 2.0
end
"""


def test_parameter_block_guess_sets_default_steady_state_initial_guess() -> None:
    model = parse_macro_model(STEADY_STATE_GUESS_SOURCE)

    default_result = solve_steady_state(model)
    override_result = solve_steady_state(model, initial_guess={"x": 3.0})

    assert default_result.converged
    assert override_result.converged
    np.testing.assert_allclose(
        default_result.steady_state,
        np.asarray([-2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        override_result.steady_state,
        np.asarray([2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )


def test_parameter_block_guess_seeds_joint_solve_with_calibrated_parameter_values(
    monkeypatch,
) -> None:
    model = parse_macro_model(CALIBRATED_PARAMETER_GUESS_SOURCE)
    captured: dict[str, np.ndarray] = {}
    original = model_module._solve_newton_system

    def fake_solve_newton_system(initial, **kwargs):
        captured["initial"] = np.asarray(initial, dtype=np.float64)
        solution = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
        return solution, True, 0, 0.0

    monkeypatch.setattr(model_module, "_solve_newton_system", fake_solve_newton_system)
    try:
        result = solve_steady_state(model)
    finally:
        monkeypatch.setattr(model_module, "_solve_newton_system", original)

    assert result.converged
    np.testing.assert_allclose(
        captured["initial"],
        np.asarray([-3.0, 5.0, 2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.steady_state,
        np.asarray([2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.parameter_values,
        np.asarray([2.0, 2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
