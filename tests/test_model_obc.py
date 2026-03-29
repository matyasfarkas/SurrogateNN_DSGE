from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import surrogatenn_dsge.model as model_module

from surrogatenn_dsge import (
    compute_first_order_obc_violation_path,
    SEPConfig,
    evaluate_dynamic_residual,
    evaluate_obc_violations,
    evaluate_obc_violations_along_path,
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


OBC_SEP_AUX_SHOCK_SOURCE = """
@model obc_sep_aux begin
    r[0] = max(r_star[0] + eps_zlbᵒᵇᶜ[x], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_sep_aux begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
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
    violations = evaluate_obc_violations(
        model,
        lag_state=[1.2, 1.2],
        current_state=[1.2, 1.2],
        lead_state=[1.2, 1.2],
        shock=[0.0],
        steady_state=[1.2, 1.2],
    )
    assert np.all(np.asarray(violations, dtype=np.float64) <= 1e-12)


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
    assert result.solution.jacobian_method == "subgradient"
    r_index = model.timings.var.index("r")
    r_path = np.asarray(result.solution.mean_path[r_index, 1:], dtype=np.float64)
    assert np.all(r_path >= 1.0 - 1e-8)
    assert np.any(np.isclose(r_path, 1.0, atol=1e-4))


def test_obc_violation_diagnostics_hold_on_deterministic_sep_path() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)
    result = solve_stochastic_extended_path_model(
        model,
        steady_state=[1.2, 1.2],
        initial_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
        deterministic_shocks={"eps_r": [-2.0, 0.0, 0.0]},
    )

    assert result.solution.converged
    violations = evaluate_obc_violations_along_path(
        model,
        result.solution.mean_path.T,
        shocks={"eps_r": [-2.0, 0.0, 0.0]},
        steady_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
    )
    assert np.all(np.asarray(violations, dtype=np.float64) <= 1e-8)


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


def test_obc_model_sep_subgradient_matches_finite_difference() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)
    subgradient = solve_stochastic_extended_path_model(
        model,
        steady_state=[1.2, 1.2],
        initial_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
        config=SEPConfig(
            periods=3,
            branching_order=1,
            jacobian_method="subgradient",
            tol=1e-8,
        ),
        deterministic_shocks={"eps_r": [-2.0, 0.0, 0.0]},
    )
    finite_difference = solve_stochastic_extended_path_model(
        model,
        steady_state=[1.2, 1.2],
        initial_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
        config=SEPConfig(
            periods=3,
            branching_order=1,
            jacobian_method="finite_difference",
            tol=1e-8,
        ),
        deterministic_shocks={"eps_r": [-2.0, 0.0, 0.0]},
    )

    assert subgradient.solution.converged
    assert finite_difference.solution.converged
    assert subgradient.solution.jacobian_method == "subgradient"
    assert finite_difference.solution.jacobian_method == "finite_difference"
    np.testing.assert_allclose(
        subgradient.solution.mean_path,
        finite_difference.solution.mean_path,
        rtol=1e-8,
        atol=1e-8,
    )


def test_first_order_obc_violation_path_detects_linear_constraint_breach() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)
    first_order = solve_first_order_model(
        model,
        steady_state_initial_guess={"r": 1.2, "r_star": 1.2},
    )
    violation_path = compute_first_order_obc_violation_path(
        model,
        {"eps_r": [-2.0, 0.0, 0.0]},
        first_order_result=first_order,
        steady_state=[1.2, 1.2],
    )

    assert violation_path.state_path.shape == (2, 4)
    assert violation_path.violations.shape[1] == 3
    assert np.max(np.asarray(violation_path.violations, dtype=np.float64)) > 1e-3


def test_sep_obc_reinjection_uses_linear_obc_shocks_before_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(OBC_SEP_AUX_SHOCK_SOURCE)
    calls: list[np.ndarray] = []
    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")

    def fake_sep_core(
        self,
        *,
        full_steady_state: np.ndarray,
        parameter_values: np.ndarray,
        initial_state: np.ndarray,
        terminal_state: np.ndarray,
        config: SEPConfig,
        deterministic_shocks: np.ndarray | None,
        initial_guess: object = None,
    ) -> model_module.ParsedModelSEPResult:
        shock_matrix = (
            np.zeros((config.periods, self.timings.nExo), dtype=np.float64)
            if deterministic_shocks is None
            else np.asarray(deterministic_shocks, dtype=np.float64)
        )
        calls.append(shock_matrix.copy())
        if len(calls) == 1:
            mean_path = np.asarray(
                [
                    [1.2, 0.4, 0.72, 0.976],
                    [1.2, 0.4, 0.72, 0.976],
                ],
                dtype=np.float64,
            )
        else:
            assert float(np.max(calls[-1][:, obc_index])) > 0.0
            implied_r = np.asarray(
                [
                    0.4 + shock_matrix[0, obc_index],
                    0.72 + shock_matrix[1, obc_index],
                    0.976 + shock_matrix[2, obc_index],
                ],
                dtype=np.float64,
            )
            mean_path = np.asarray(
                [
                    [1.2, *implied_r],
                    [1.2, 0.4, 0.72, 0.976],
                ],
                dtype=np.float64,
            )
        return model_module.ParsedModelSEPResult(
            steady_state=np.asarray(full_steady_state, dtype=np.float64),
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            solution=model_module.SEPSolution(
                stacked_states=jnp.asarray(mean_path[:, 1:].T.reshape(-1), dtype=jnp.float64),
                mean_path=jnp.asarray(mean_path, dtype=jnp.float64),
                residual_norm=1e-8,
                converged=True,
                accepted=True,
                iterations=2,
                group_counts=(1, 1, 1, 1),
                jacobian_method="subgradient",
            ),
        )

    monkeypatch.setattr(
        model_module.MacroModel,
        "_solve_stochastic_extended_path_core",
        fake_sep_core,
    )

    result = solve_stochastic_extended_path_model(
        model,
        steady_state=[1.2, 1.2],
        initial_state=[1.2, 1.2],
        terminal_state=[1.2, 1.2],
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
        deterministic_shocks={"eps_r": [-2.0, 0.0, 0.0]},
    )

    assert len(calls) >= 2
    assert result.solution.accepted
    assert np.all(np.asarray(result.solution.mean_path[0, 1:], dtype=np.float64) >= 1.0)
