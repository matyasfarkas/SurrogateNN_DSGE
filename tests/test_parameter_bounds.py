from __future__ import annotations

import numpy as np

import surrogatenn_dsge.model as model_module
from surrogatenn_dsge import parse_macro_model, solve_steady_state


BOUND_PARSE_SOURCE = """
@model bound_parse begin
    x[0] = alpha
    y[0] = beta
end

@parameters bound_parse begin
    alpha = 0.5
    beta = 2.0
    0 < alpha < 1
    10 >= beta
    x >= 0
    1 > y > -1
end
"""


BOUND_SOLVE_SOURCE = """
@model bound_solve begin
    x[0]^2 = level
end

@parameters bound_solve guess = Dict(:x => -3.0) begin
    level = 4.0
    x >= 1
end
"""


BOUND_JOINT_SOURCE = """
@model bound_joint begin
    x[0] = a
end

@parameters bound_joint guess = Dict("x" => -3.0, "a" => -5.0) begin
    x[ss] = target | a
    target = 2.0
    x >= 0
    a >= 0
end
"""


def test_parameter_bounds_parse_common_open_and_closed_forms() -> None:
    model = parse_macro_model(BOUND_PARSE_SOURCE)

    alpha_lower, alpha_upper = model.bounds["alpha"]
    assert 0.0 < alpha_lower < 1e-12
    assert 1.0 - 1e-12 < alpha_upper < 1.0
    np.testing.assert_allclose(model.bounds["beta"], (-np.inf, 10.0), rtol=0, atol=0)
    np.testing.assert_allclose(model.bounds["x"], (0.0, np.inf), rtol=0, atol=0)
    y_lower, y_upper = model.bounds["y"]
    assert -1.0 < y_lower < -1.0 + 1e-12
    assert 1.0 - 1e-12 < y_upper < 1.0


def test_variable_bounds_project_steady_state_solve_to_feasible_branch() -> None:
    model = parse_macro_model(BOUND_SOLVE_SOURCE)

    result = solve_steady_state(model)

    assert result.converged
    np.testing.assert_allclose(
        result.steady_state,
        np.asarray([2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )


def test_bounds_clip_joint_initial_condition_for_variables_and_parameters(
    monkeypatch,
) -> None:
    model = parse_macro_model(BOUND_JOINT_SOURCE)
    captured: dict[str, np.ndarray] = {}
    original = model_module._solve_newton_system

    def fake_solve_newton_system(initial, **kwargs):
        captured["initial"] = np.asarray(initial, dtype=np.float64)
        captured["lower_bounds"] = np.asarray(kwargs["lower_bounds"], dtype=np.float64)
        captured["upper_bounds"] = np.asarray(kwargs["upper_bounds"], dtype=np.float64)
        solution = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
        return solution, True, 0, 0.0

    monkeypatch.setattr(model_module, "_solve_newton_system", fake_solve_newton_system)
    try:
        result = solve_steady_state(model)
    finally:
        monkeypatch.setattr(model_module, "_solve_newton_system", original)

    assert result.converged
    np.testing.assert_allclose(captured["initial"], np.asarray([-3.0, -5.0, 2.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        captured["lower_bounds"],
        np.asarray([0.0, 0.0, -np.inf], dtype=np.float64),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        captured["upper_bounds"],
        np.asarray([np.inf, np.inf, np.inf], dtype=np.float64),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        result.parameter_values,
        np.asarray([2.0, 2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
