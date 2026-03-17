from __future__ import annotations

import jax
import numpy as np

from surrogatenn_dsge import (
    build_linear_state_space_from_model,
    kalman_loglikelihood,
    kalman_loglikelihood_from_model,
    kalman_loglikelihood_per_period_from_model,
    parse_macro_model,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
)


LIKELIHOOD_SOURCE = """
@model parsed_loglikelihood begin
    a[0] = rho_a * a[-1] + (1 - rho_a) * a_bar + eps_a[x]
    y[0] = rho_y * y[-1] + (1 - rho_y) * y_bar + alpha * (a[0] - a_bar) + eps_y[x]
end

@parameters parsed_loglikelihood begin
    0 < rho_a < 1
    0 < rho_y < 1
    alpha = 0.4
    a_bar = 1.5
    y_bar = 2.0
    rho_a = 0.8
    rho_y = 0.6
end
"""


def _loglikelihood_fixture():
    model = parse_macro_model(LIKELIHOOD_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )
    observables = ("y", "a")
    state_space = build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order_result,
    )
    simulation = simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(0),
        num_periods=25,
    )
    steady_lookup = dict(zip(model.timings.var, np.asarray(first_order_result.steady_state)))
    levels = simulation.observations + np.asarray(
        [[steady_lookup[name]] for name in observables],
        dtype=np.float64,
    )
    return model, first_order_result, observables, state_space, simulation, levels


def test_model_loglikelihood_matches_low_level_kalman_path() -> None:
    model, first_order_result, observables, state_space, simulation, levels = (
        _loglikelihood_fixture()
    )

    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
    )
    low_level = kalman_loglikelihood(state_space, simulation.observations)

    np.testing.assert_allclose(high_level, low_level, rtol=1e-10, atol=1e-10)


def test_model_loglikelihood_accepts_mapping_input() -> None:
    model, first_order_result, _, _, simulation, levels = _loglikelihood_fixture()
    mapping_data = {
        "a": levels[1],
        "y": levels[0],
    }

    high_level = kalman_loglikelihood_from_model(
        model,
        mapping_data,
        first_order_result=first_order_result,
    )
    sorted_observables = ("a", "y")
    sorted_state_space = build_linear_state_space_from_model(
        model,
        sorted_observables,
        first_order_result=first_order_result,
    )
    sorted_deviations = np.vstack([simulation.observations[1], simulation.observations[0]])
    low_level = kalman_loglikelihood(sorted_state_space, sorted_deviations)

    np.testing.assert_allclose(high_level, low_level, rtol=1e-10, atol=1e-10)


def test_model_loglikelihood_per_period_matches_total_and_failure_value() -> None:
    model, first_order_result, observables, _, _, levels = _loglikelihood_fixture()

    total = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
    )
    per_period = kalman_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
    )

    np.testing.assert_allclose(total, np.sum(per_period), rtol=1e-10, atol=1e-10)

    bad_parameter_values = np.asarray(model.parameter_values).copy()
    bad_parameter_values[model.parameter_names.index("rho_a")] = 1.2
    failure_value = -1e9

    failed_total = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        parameter_values=bad_parameter_values,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        on_failure_loglikelihood=failure_value,
    )
    failed_per_period = kalman_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=observables,
        parameter_values=bad_parameter_values,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        on_failure_loglikelihood=failure_value,
    )

    np.testing.assert_allclose(failed_total, failure_value, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        failed_per_period,
        np.full((levels.shape[1],), failure_value),
        rtol=0.0,
        atol=0.0,
    )
