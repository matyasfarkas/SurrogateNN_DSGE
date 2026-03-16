from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    SEPConfig,
    evaluate_dynamic_residual,
    parse_macro_model,
    solve_stochastic_extended_path_model,
    solve_stochastic_extended_path_residual_expectation,
)


NONLINEAR_SEP_SOURCE = """
@model nonlinear_sep begin
    y[0] = rho * y[-1] + gamma * y[1]^2 + u[x]
end

@parameters nonlinear_sep begin
    gamma = 0.15
    rho = 0.25
end
"""


def test_evaluate_dynamic_residual_matches_manual_equation() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)

    residual = evaluate_dynamic_residual(
        model,
        lag_state=[0.5],
        current_state=[0.7],
        lead_state=[-0.2],
        shock=[0.1],
        steady_state=[0.0],
    )

    expected = 0.7 - (0.25 * 0.5 + 0.15 * (-0.2) ** 2 + 0.1)
    np.testing.assert_allclose(
        residual,
        jnp.asarray([expected], dtype=jnp.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parsed_model_sep_matches_manual_conditional_residual_solver() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    config = SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10)
    deterministic = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)

    def conditional_residual(
        y_prev: jnp.ndarray,
        y_curr: jnp.ndarray,
        y_next: jnp.ndarray,
        shock: jnp.ndarray,
        params: tuple[float, float],
    ) -> jnp.ndarray:
        rho, gamma = params
        return y_curr - (rho * y_prev + gamma * y_next**2 + shock)

    manual = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=config,
        deterministic_shocks=deterministic,
        params=(0.25, 0.15),
    )
    parsed = solve_stochastic_extended_path_model(
        model,
        config=config,
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )

    assert manual.converged
    assert parsed.solution.converged
    np.testing.assert_allclose(parsed.steady_state, 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        parsed.solution.stacked_states,
        manual.stacked_states,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed.solution.mean_path,
        manual.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )
