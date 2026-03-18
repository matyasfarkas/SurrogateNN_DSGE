from __future__ import annotations

import numpy as np

from surrogatenn_dsge import (
    SEPConfig,
    evaluate_dynamic_residual,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
    solve_stochastic_extended_path_model,
)


OBC_MAX_SOURCE = """
@model obc_linear begin
    r[0] = max(r_star[0], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_linear begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


OBC_MIN_SOURCE = """
@model obc_cap begin
    q[0] = min(q_star[0], q_cap)
    q_star[0] = rho * q_star[-1] + (1-rho) * mu + eps_q[x]
end

@parameters obc_cap begin
    rho = 0.7
    mu = 0.8
    q_cap = 1.0
end
"""


OBC_BINDING_MAX_SOURCE = """
@model obc_binding begin
    r[0] = max(r_star[0], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_binding begin
    rho = 0.8
    mu = 1.0
    zlb = 1.0
end
"""


OBC_BINDING_MIN_SOURCE = """
@model obc_binding_cap begin
    q[0] = min(q_star[0], q_cap)
    q_star[0] = rho * q_star[-1] + (1-rho) * mu + eps_q[x]
end

@parameters obc_binding_cap begin
    rho = 0.7
    mu = 1.0
    q_cap = 1.0
end
"""


def test_parse_macro_model_flags_obc_and_evaluates_max_residual() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)

    assert model.has_obc is True
    residual = evaluate_dynamic_residual(
        model,
        lag_state=[1.2, 1.2],
        current_state=[1.2, 1.2],
        lead_state=[1.2, 1.2],
        shock=[0.0],
        steady_state=[1.2, 1.2],
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-12)


def test_parse_macro_model_flags_min_obc() -> None:
    model = parse_macro_model(OBC_MIN_SOURCE)

    assert model.has_obc is True
    residual = evaluate_dynamic_residual(
        model,
        lag_state=[0.8, 0.8],
        current_state=[0.8, 0.8],
        lead_state=[0.8, 0.8],
        shock=[0.0],
        steady_state=[0.8, 0.8],
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-12)


def test_obc_model_supports_steady_state_and_inactive_first_order_solution() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)

    steady_state = solve_steady_state(
        model,
        initial_guess={"r": 1.2, "r_star": 1.2},
    )
    first_order = solve_first_order_model(
        model,
        steady_state_initial_guess={"r": 1.2, "r_star": 1.2},
    )

    assert steady_state.converged
    np.testing.assert_allclose(steady_state.steady_state, [1.2, 1.2], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(first_order.steady_state, [1.2, 1.2], rtol=0.0, atol=1e-10)
    assert np.all(np.isfinite(first_order.solution.solution_matrix))


def test_obc_model_sep_enforces_max_constraint_along_path() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)
    result = solve_stochastic_extended_path_model(
        model,
        steady_state=[1.2, 1.2],
        initial_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
        config=SEPConfig(periods=3, branching_order=1, tol=1e-8),
        deterministic_shocks={"eps_r": [-2.0, 0.0, 0.0]},
    )

    assert result.solution.converged
    assert result.solution.jacobian_method == "finite_difference"
    r_index = model.timings.var.index("r")
    r_path = np.asarray(result.solution.mean_path[r_index, 1:], dtype=np.float64)
    assert np.all(r_path >= 1.0 - 1e-8)
    assert np.any(np.isclose(r_path, 1.0, atol=1e-4))


def test_obc_binding_max_linearization_freezes_constraint_branch() -> None:
    model = parse_macro_model(OBC_BINDING_MAX_SOURCE)
    first_order = solve_first_order_model(
        model,
        steady_state_initial_guess={"r": 1.0, "r_star": 1.0},
    )

    assert first_order.solution.converged
    np.testing.assert_allclose(first_order.steady_state, [1.0, 1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(first_order.jacobian[0], dtype=np.float64),
        [1.0, 0.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(first_order.solution.solution_matrix[0], dtype=np.float64),
        [0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_obc_binding_min_linearization_freezes_constraint_branch() -> None:
    model = parse_macro_model(OBC_BINDING_MIN_SOURCE)
    steady_state = solve_steady_state(
        model,
        initial_guess={"q": 1.0, "q_star": 1.0},
    )
    jacobian = model.calculate_jacobian(
        steady_state=steady_state.steady_state,
        parameter_values=steady_state.parameter_values,
    )

    assert steady_state.converged
    np.testing.assert_allclose(steady_state.steady_state, [1.0, 1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jacobian[0], dtype=np.float64),
        [1.0, 0.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
