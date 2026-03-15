from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import SEPConfig, gauss_hermite_rule, solve_stochastic_extended_path


def test_gauss_hermite_rule_has_unit_total_weight() -> None:
    rule = gauss_hermite_rule(3, 2)
    np.testing.assert_allclose(jnp.sum(rule.weights), 1.0, rtol=1e-12, atol=1e-12)


def test_sep_zero_shock_linear_model_returns_zero_mean_path() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.5 * y_prev + 0.3 * expected_next + shock)

    solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=5, branching_order=1, nnodes=3, tol=1e-10),
    )

    assert solution.converged
    np.testing.assert_allclose(solution.mean_path, 0.0, rtol=1e-10, atol=1e-10)


def test_sep_mean_path_matches_deterministic_path_for_linear_zero_mean_shocks() -> None:
    deterministic_shocks = jnp.array([[1.0], [0.2], [0.0], [0.0]])

    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    stochastic_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=4, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )
    deterministic_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=4, branching_order=0, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )

    assert stochastic_solution.converged
    assert deterministic_solution.converged
    np.testing.assert_allclose(
        stochastic_solution.mean_path,
        deterministic_solution.mean_path,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sep_handles_nonlinear_expectational_equation() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (
            0.3 * y_prev
            + 0.4 * expected_next
            - 0.05 * y_curr**2
            + shock
        )

    solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=1, nnodes=3, tol=1e-8),
        deterministic_shocks=[[0.5], [0.0], [0.0]],
    )

    assert solution.converged
    assert solution.residual_norm < 1e-8
    assert np.all(np.isfinite(solution.mean_path))
