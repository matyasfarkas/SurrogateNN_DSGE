from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    build_linear_state_space_from_model,
    kalman_loglikelihood,
    kalman_loglikelihood_from_model,
    kalman_loglikelihood_per_period_from_model,
    parse_macro_model,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
    solve_first_order_model_jax,
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

PRESENT_ONLY_SOURCE = """
@model present_only_first_order begin
    x[0] = rho * x[-1] + eps_x[x]
    y[0] = alpha * x[0]
end

@parameters present_only_first_order begin
    0 < rho < 1
    alpha = 0.5
    rho = 0.7
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


def _sorted_low_level_reference(
    model,
    first_order_result,
    observables,
    simulation,
):
    sorted_observables = tuple(sorted(observables))
    row_lookup = {name: idx for idx, name in enumerate(observables)}
    sorted_state_space = build_linear_state_space_from_model(
        model,
        sorted_observables,
        first_order_result=first_order_result,
    )
    sorted_deviations = np.vstack(
        [simulation.observations[row_lookup[name]] for name in sorted_observables]
    )
    return sorted_state_space, sorted_deviations


def test_model_loglikelihood_matches_low_level_kalman_path() -> None:
    model, first_order_result, observables, _, simulation, levels = (
        _loglikelihood_fixture()
    )
    sorted_state_space, sorted_deviations = _sorted_low_level_reference(
        model,
        first_order_result,
        observables,
        simulation,
    )

    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
    )
    low_level = kalman_loglikelihood(sorted_state_space, sorted_deviations)

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


def test_model_loglikelihood_accepts_schur_qme_algorithm() -> None:
    model, first_order_result, observables, _, simulation, levels = (
        _loglikelihood_fixture()
    )
    sorted_state_space, sorted_deviations = _sorted_low_level_reference(
        model,
        first_order_result,
        observables,
        simulation,
    )

    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    explicit_first_order = solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    explicit = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=explicit_first_order,
    )
    low_level = kalman_loglikelihood(sorted_state_space, sorted_deviations)

    np.testing.assert_allclose(high_level, explicit, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(high_level, low_level, rtol=1e-10, atol=1e-10)


def test_solve_first_order_model_jax_matches_python_explicit_steady_state() -> None:
    model, first_order_result, _, _, _, _ = _loglikelihood_fixture()
    parameters = np.asarray(model.parameter_values, dtype=np.float64).copy()

    jax_result = solve_first_order_model_jax(
        model,
        parameter_values=parameters,
        steady_state=np.asarray(first_order_result.steady_state, dtype=np.float64),
        qme_algorithm="schur",
        check_parameter_bounds=False,
    )

    assert bool(np.asarray(jax_result.converged))
    np.testing.assert_allclose(
        jax_result.steady_state,
        first_order_result.steady_state,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jax_result.state_transition,
        first_order_result.solution.state_transition,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jax_result.shock_impact,
        first_order_result.solution.shock_impact,
        rtol=1e-10,
        atol=1e-10,
    )


def test_solve_first_order_model_jax_solves_steady_state_and_has_finite_gradient() -> None:
    model = parse_macro_model(LIKELIHOOD_SOURCE)
    parameters = jnp.asarray(model.parameter_values, dtype=jnp.float64)
    initial_guess = {"a": 1.5, "y": 2.0}

    def objective(theta: jax.Array) -> jax.Array:
        result = solve_first_order_model_jax(
            model,
            parameter_values=theta,
            steady_state_initial_guess=initial_guess,
            qme_algorithm="schur",
            check_parameter_bounds=False,
        )
        return jnp.sum(result.steady_state) + jnp.sum(result.state_transition)

    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)

    assert np.isfinite(float(value))
    assert gradient.shape == parameters.shape
    assert np.isfinite(np.asarray(gradient)).all()


def test_solve_first_order_model_jax_static_rows_avoid_qr_ad_limitation() -> None:
    model = parse_macro_model(PRESENT_ONLY_SOURCE)
    parameters = jnp.asarray(model.parameter_values, dtype=jnp.float64)
    steady_state = np.zeros((model.timings.nVars,), dtype=np.float64)
    static_rows = model._first_order_static_equation_rows_for_values(
        steady_state=steady_state,
        parameter_values=np.asarray(model.parameter_values, dtype=np.float64),
    )

    def objective(theta: jax.Array) -> jax.Array:
        result = solve_first_order_model_jax(
            model,
            parameter_values=theta,
            steady_state_initial_guess={"x": 0.0, "y": 0.0},
            qme_algorithm="schur",
            static_equation_rows=static_rows,
            check_parameter_bounds=False,
        )
        return jnp.sum(result.state_transition) + jnp.sum(result.shock_impact)

    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)

    assert static_rows
    assert np.isfinite(float(value))
    assert gradient.shape == parameters.shape
    assert np.isfinite(np.asarray(gradient)).all()


def test_solve_first_order_model_jax_rejects_out_of_bounds_parameters() -> None:
    model = parse_macro_model(LIKELIHOOD_SOURCE)
    parameters = np.asarray(model.parameter_values, dtype=np.float64).copy()
    parameters[model.parameter_names.index("rho_a")] = 1.2

    result = solve_first_order_model_jax(
        model,
        parameter_values=parameters,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
        check_parameter_bounds=True,
    )

    assert not bool(np.asarray(result.converged))
    assert not bool(np.asarray(result.steady_state_converged))
    assert not bool(np.asarray(result.first_order_converged))
    assert np.isinf(float(np.asarray(result.steady_state_residual_norm)))


def test_model_loglikelihood_sorts_array_observables_like_julia() -> None:
    model, first_order_result, observables, _, simulation, levels = _loglikelihood_fixture()
    sorted_state_space, sorted_deviations = _sorted_low_level_reference(
        model,
        first_order_result,
        observables,
        simulation,
    )

    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
    )

    np.testing.assert_allclose(
        high_level,
        kalman_loglikelihood(sorted_state_space, sorted_deviations),
        rtol=1e-10,
        atol=1e-10,
    )
