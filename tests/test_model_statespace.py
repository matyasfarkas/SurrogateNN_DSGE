from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    build_linear_state_space_from_model,
    kalman_loglikelihood,
    linear_state_space_from_first_order_solution,
    parse_macro_model,
    resolve_observable_indices,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
)


STATE_SPACE_SOURCE = """
@model parsed_statespace begin
    c[0] = rho_c * c[-1] + eps_c[x]
    y[0] = rho_y * y[-1] + alpha * c[0] + eps_y[x]
end

@parameters parsed_statespace begin
    rho_c = 0.8
    rho_y = 0.7
    alpha = 0.3
end
"""


def _parsed_statespace_fixture():
    model = parse_macro_model(STATE_SPACE_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"c": 0.0, "y": 0.0},
    )
    return model, first_order_result


def test_resolve_observable_indices_preserves_requested_order() -> None:
    model, _ = _parsed_statespace_fixture()
    observables = ("y", "c")

    indices = resolve_observable_indices(model, observables)

    expected = tuple(model.timings.var.index(name) for name in observables)
    assert indices == expected


def test_resolve_observable_indices_rejects_unknown_names() -> None:
    model, _ = _parsed_statespace_fixture()

    with pytest.raises(ValueError, match="Unknown observable names"):
        resolve_observable_indices(model, ("z",))


def test_build_linear_state_space_from_model_matches_low_level_helper() -> None:
    model, first_order_result = _parsed_statespace_fixture()
    observables = ("y", "c")
    measurement_error_covariance = jnp.array(
        [[0.05, 0.01], [0.01, 0.02]],
        dtype=jnp.float64,
    )

    parsed_state_space = build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order_result,
        measurement_error_covariance=measurement_error_covariance,
    )
    low_level_state_space = linear_state_space_from_first_order_solution(
        first_order_result.solution.solution_matrix,
        model.timings,
        observable_indices=resolve_observable_indices(model, observables),
        measurement_error_scale=0.0,
    )

    np.testing.assert_allclose(
        parsed_state_space.transition_matrix,
        low_level_state_space.transition_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed_state_space.process_noise_covariance,
        low_level_state_space.process_noise_covariance,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed_state_space.observation_matrix,
        low_level_state_space.observation_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed_state_space.observation_noise_covariance,
        measurement_error_covariance,
        rtol=1e-10,
        atol=1e-10,
    )


def test_build_linear_state_space_from_model_accepts_schur_qme_algorithm() -> None:
    model, _ = _parsed_statespace_fixture()
    observables = ("y", "c")

    parsed_state_space = build_linear_state_space_from_model(
        model,
        observables,
        steady_state_initial_guess={"c": 0.0, "y": 0.0},
        qme_algorithm="schur",
    )
    explicit_first_order = solve_first_order_model(
        model,
        steady_state_initial_guess={"c": 0.0, "y": 0.0},
        qme_algorithm="schur",
    )
    explicit_state_space = build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=explicit_first_order,
    )

    np.testing.assert_allclose(
        parsed_state_space.transition_matrix,
        explicit_state_space.transition_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed_state_space.process_noise_covariance,
        explicit_state_space.process_noise_covariance,
        rtol=1e-10,
        atol=1e-10,
    )


def test_parsed_model_state_space_is_jittable_and_device_accessible() -> None:
    model, first_order_result = _parsed_statespace_fixture()
    state_space = build_linear_state_space_from_model(
        model,
        ("y",),
        first_order_result=first_order_result,
    )

    jit_simulate = jax.jit(
        simulate_linear_gaussian_state_space,
        static_argnames=("num_periods",),
    )
    simulation = jit_simulate(state_space, jax.random.PRNGKey(0), num_periods=8)
    loglikelihood = kalman_loglikelihood(state_space, simulation.observations)

    assert simulation.states.shape == (state_space.transition_matrix.shape[0], 8)
    assert simulation.observations.shape == (1, 8)
    assert np.isfinite(loglikelihood)

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if gpu_devices:
        gpu_state_space = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, gpu_devices[0]),
            state_space,
        )
        gpu_key = jax.device_put(jax.random.PRNGKey(0), gpu_devices[0])
        gpu_simulation = jit_simulate(gpu_state_space, gpu_key, num_periods=8)
        assert gpu_simulation.states.shape == simulation.states.shape
        assert gpu_simulation.observations.shape == simulation.observations.shape
