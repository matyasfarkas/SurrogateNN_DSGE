from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    parse_macro_model,
    solve_steady_state,
    solve_steady_state_jax,
)


STEADY_STATE_SOURCE = """
@model steady_state_jax begin
    a[0] = rho_a * a[-1] + (1 - rho_a) * a_bar + eps_a[x]
    y[0] = rho_y * y[-1] + (1 - rho_y) * y_bar + alpha * (a[0] - a_bar) + eps_y[x]
end

@parameters steady_state_jax begin
    0 < rho_a < 1
    0 < rho_y < 1
    alpha = 0.4
    a_bar = 1.5
    y_bar = 2.0
    rho_a = 0.8
    rho_y = 0.6
end
"""


CALIBRATION_SOURCE = """
@model end_target begin
    x[0] = a
end

@parameters end_target begin
    target = theta + 1
    x[ss] = target | a
    theta = 2
end
"""


def test_jax_steady_state_solver_matches_numpy_path() -> None:
    model = parse_macro_model(STEADY_STATE_SOURCE)
    parameter_vector = jnp.asarray(model.parameter_values, dtype=jnp.float64)

    jax_result = solve_steady_state_jax(
        model,
        parameter_values=parameter_vector,
        initial_guess={"a": 1.2, "y": 1.9},
    )
    numpy_result = solve_steady_state(
        model,
        parameter_values=parameter_vector,
        initial_guess={"a": 1.2, "y": 1.9},
    )

    assert bool(np.asarray(jax_result.converged))
    np.testing.assert_allclose(
        jax_result.steady_state,
        numpy_result.steady_state,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_steady_state_solver_is_jittable() -> None:
    model = parse_macro_model(STEADY_STATE_SOURCE)
    compiled = jax.jit(
        lambda theta: solve_steady_state_jax(
            model,
            parameter_values=theta,
            initial_guess={"a": 1.2, "y": 1.9},
        ).steady_state
    )

    steady_state = compiled(
        jnp.asarray(model.parameter_values, dtype=jnp.float64)
    )

    np.testing.assert_allclose(
        steady_state,
        jnp.asarray([1.5, 2.0], dtype=jnp.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_steady_state_solver_rejects_calibration_equations() -> None:
    model = parse_macro_model(CALIBRATION_SOURCE)

    with pytest.raises(NotImplementedError, match="calibration equations"):
        solve_steady_state_jax(
            model,
            initial_guess={"x": 2.0},
        )
