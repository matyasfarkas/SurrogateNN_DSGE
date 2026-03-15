from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    LinearGaussianStateSpace,
    build_linear_gaussian_state_space,
    kalman_filter,
    kalman_loglikelihood,
    kalman_loglikelihood_per_period,
    kalman_smoother,
    simulate_linear_gaussian_state_space,
    solve_discrete_lyapunov_direct,
)


def _build_test_model(phi: float = 0.8) -> tuple:
    model = build_linear_gaussian_state_space(
        transition_matrix=jnp.array([[phi, 0.1], [0.0, 0.6]]),
        process_noise_covariance=jnp.array([[0.2, 0.0], [0.0, 0.1]]),
        observation_matrix=jnp.array([[1.0, 0.0], [0.2, 1.0]]),
        observation_noise_covariance=jnp.array([[0.05, 0.0], [0.0, 0.04]]),
    )
    key = jax.random.PRNGKey(123)
    simulation = simulate_linear_gaussian_state_space(model, key, 100)
    return model, simulation


def test_theoretical_initial_covariance_is_finite() -> None:
    model, _ = _build_test_model()
    eigvals = np.linalg.eigvalsh(np.asarray(model.initial_covariance))

    assert np.all(np.isfinite(model.initial_covariance))
    assert np.all(eigvals >= -1e-10)


def test_kalman_loglikelihood_matches_sum_of_per_period_values() -> None:
    model, simulation = _build_test_model()

    total = kalman_loglikelihood(model, simulation.observations)
    per_period = kalman_loglikelihood_per_period(model, simulation.observations)

    np.testing.assert_allclose(total, jnp.sum(per_period), rtol=1e-12, atol=1e-12)


def test_filter_and_smoother_return_finite_outputs() -> None:
    model, simulation = _build_test_model()

    filtered = kalman_filter(model, simulation.observations)
    smoothed = kalman_smoother(model, filtered)

    assert filtered.filtered_means.shape == (2, 100)
    assert smoothed.smoothed_means.shape == (2, 100)
    assert np.all(np.isfinite(filtered.filtered_means))
    assert np.all(np.isfinite(smoothed.smoothed_means))


def test_smoothing_differs_from_filtering() -> None:
    model, simulation = _build_test_model()

    filtered = kalman_filter(model, simulation.observations)
    smoothed = kalman_smoother(model, filtered)

    assert not np.allclose(filtered.filtered_means, smoothed.smoothed_means)


def test_repeated_filtering_is_deterministic() -> None:
    model, simulation = _build_test_model()

    result_1 = kalman_filter(model, simulation.observations)
    result_2 = kalman_filter(model, simulation.observations)

    np.testing.assert_allclose(
        result_1.filtered_means,
        result_2.filtered_means,
        rtol=1e-12,
        atol=1e-12,
    )


def test_short_data_is_supported() -> None:
    model, _ = _build_test_model()
    key = jax.random.PRNGKey(321)
    simulation = simulate_linear_gaussian_state_space(model, key, 10)

    filtered = kalman_filter(model, simulation.observations)

    assert filtered.filtered_means.shape == (2, 10)
    assert np.all(np.isfinite(filtered.filtered_means))


def test_true_parameter_has_higher_likelihood_than_bad_perturbation() -> None:
    true_model, simulation = _build_test_model(phi=0.8)
    perturbed_model = build_linear_gaussian_state_space(
        transition_matrix=jnp.array([[0.3, 0.1], [0.0, 0.6]]),
        process_noise_covariance=true_model.process_noise_covariance,
        observation_matrix=true_model.observation_matrix,
        observation_noise_covariance=true_model.observation_noise_covariance,
    )

    ll_true = kalman_loglikelihood(true_model, simulation.observations)
    ll_bad = kalman_loglikelihood(perturbed_model, simulation.observations)

    assert float(np.asarray(ll_true)) > float(np.asarray(ll_bad))


def test_kalman_loglikelihood_is_autodiff_friendly() -> None:
    _, simulation = _build_test_model(phi=0.75)

    def objective(phi: jax.Array) -> jax.Array:
        transition = jnp.array([[phi, 0.1], [0.0, 0.6]])
        process_cov = jnp.array([[0.2, 0.0], [0.0, 0.1]])
        model = LinearGaussianStateSpace(
            transition_matrix=transition,
            process_noise_covariance=process_cov,
            observation_matrix=jnp.array([[1.0, 0.0], [0.2, 1.0]]),
            observation_noise_covariance=jnp.array([[0.05, 0.0], [0.0, 0.04]]),
            initial_mean=jnp.zeros((2,)),
            initial_covariance=solve_discrete_lyapunov_direct(
                transition,
                process_cov,
            ).solution,
        )
        return kalman_loglikelihood(model, simulation.observations)

    grad_value = jax.grad(objective)(jnp.array(0.7))

    assert np.isfinite(grad_value)


def test_kalman_filter_is_jittable() -> None:
    model, simulation = _build_test_model()

    compiled = jax.jit(kalman_filter)
    result = compiled(model, simulation.observations)

    assert np.all(np.isfinite(result.filtered_means))
