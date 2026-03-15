from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .linalg import solve_discrete_lyapunov_direct

InitialCovarianceStrategy = Literal["theoretical", "diagonal"]
LOG_2PI = float(np.log(2.0 * np.pi))


class LinearGaussianStateSpace(NamedTuple):
    transition_matrix: jax.Array
    process_noise_covariance: jax.Array
    observation_matrix: jax.Array
    observation_noise_covariance: jax.Array
    initial_mean: jax.Array
    initial_covariance: jax.Array


class StateSpaceSimulation(NamedTuple):
    states: jax.Array
    observations: jax.Array


class KalmanFilterResult(NamedTuple):
    filtered_means: jax.Array
    filtered_covariances: jax.Array
    predicted_means: jax.Array
    predicted_covariances: jax.Array
    innovations: jax.Array
    innovation_covariances: jax.Array
    loglikelihood_per_period: jax.Array
    total_loglikelihood: jax.Array


class KalmanSmootherResult(NamedTuple):
    smoothed_means: jax.Array
    smoothed_covariances: jax.Array


def _cast_statespace_matrix(
    value: Union[jax.Array, np.ndarray],
    name: str,
) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {array.shape}.")
    return jnp.asarray(array, dtype=jnp.result_type(array, jnp.float64))


def build_linear_gaussian_state_space(
    transition_matrix: Union[jax.Array, np.ndarray],
    process_noise_covariance: Union[jax.Array, np.ndarray],
    observation_matrix: Union[jax.Array, np.ndarray],
    observation_noise_covariance: Optional[Union[jax.Array, np.ndarray]] = None,
    *,
    initial_mean: Optional[Union[jax.Array, np.ndarray]] = None,
    initial_covariance: Optional[Union[jax.Array, np.ndarray]] = None,
    initial_covariance_strategy: InitialCovarianceStrategy = "theoretical",
    diagonal_scale: float = 10.0,
) -> LinearGaussianStateSpace:
    transition = _cast_statespace_matrix(transition_matrix, "transition_matrix")
    process_cov = _cast_statespace_matrix(
        process_noise_covariance,
        "process_noise_covariance",
    )
    observation = _cast_statespace_matrix(observation_matrix, "observation_matrix")

    state_dim = transition.shape[0]
    obs_dim = observation.shape[0]

    if transition.shape[1] != state_dim:
        raise ValueError("transition_matrix must be square.")
    if process_cov.shape != (state_dim, state_dim):
        raise ValueError(
            "process_noise_covariance must match transition_matrix, "
            f"got {process_cov.shape} and {(state_dim, state_dim)}."
        )
    if observation.shape[1] != state_dim:
        raise ValueError(
            "observation_matrix must have shape (obs_dim, state_dim), "
            f"got {observation.shape}."
        )

    if observation_noise_covariance is None:
        observation_cov = jnp.zeros((obs_dim, obs_dim), dtype=transition.dtype)
    else:
        observation_cov = _cast_statespace_matrix(
            observation_noise_covariance,
            "observation_noise_covariance",
        )
        if observation_cov.shape != (obs_dim, obs_dim):
            raise ValueError(
                "observation_noise_covariance must be square in observation dimension, "
                f"got {observation_cov.shape} and {(obs_dim, obs_dim)}."
            )

    if initial_mean is None:
        initial_mean_arr = jnp.zeros((state_dim,), dtype=transition.dtype)
    else:
        initial_mean_arr = jnp.asarray(initial_mean, dtype=transition.dtype)
        if initial_mean_arr.shape != (state_dim,):
            raise ValueError(
                f"initial_mean must have shape {(state_dim,)}, got {initial_mean_arr.shape}."
            )

    if initial_covariance is None:
        if initial_covariance_strategy == "theoretical":
            initial_cov_arr = solve_discrete_lyapunov_direct(
                transition,
                process_cov,
            ).solution
        elif initial_covariance_strategy == "diagonal":
            initial_cov_arr = diagonal_scale * jnp.eye(state_dim, dtype=transition.dtype)
        else:
            raise ValueError(
                "initial_covariance_strategy must be 'theoretical' or 'diagonal'."
            )
    else:
        initial_cov_arr = _cast_statespace_matrix(initial_covariance, "initial_covariance")
        if initial_cov_arr.shape != (state_dim, state_dim):
            raise ValueError(
                "initial_covariance must be square in state dimension, "
                f"got {initial_cov_arr.shape} and {(state_dim, state_dim)}."
            )

    for name, value in (
        ("transition_matrix", transition),
        ("process_noise_covariance", process_cov),
        ("observation_matrix", observation),
        ("observation_noise_covariance", observation_cov),
        ("initial_mean", initial_mean_arr),
        ("initial_covariance", initial_cov_arr),
    ):
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"{name} must contain only finite values.")

    return LinearGaussianStateSpace(
        transition_matrix=transition,
        process_noise_covariance=process_cov,
        observation_matrix=observation,
        observation_noise_covariance=observation_cov,
        initial_mean=initial_mean_arr,
        initial_covariance=initial_cov_arr,
    )


def simulate_linear_gaussian_state_space(
    model: LinearGaussianStateSpace,
    key: jax.Array,
    num_periods: int,
    *,
    sampling_jitter: float = 1e-10,
) -> StateSpaceSimulation:
    state_dim = model.transition_matrix.shape[0]
    obs_dim = model.observation_matrix.shape[0]
    process_key, observation_key = jax.random.split(key)
    process_cov = model.process_noise_covariance + sampling_jitter * jnp.eye(
        state_dim,
        dtype=model.transition_matrix.dtype,
    )
    observation_cov = model.observation_noise_covariance + sampling_jitter * jnp.eye(
        obs_dim,
        dtype=model.transition_matrix.dtype,
    )
    process_noise = jax.random.multivariate_normal(
        process_key,
        mean=jnp.zeros((state_dim,), dtype=model.transition_matrix.dtype),
        cov=process_cov,
        shape=(num_periods,),
    )
    observation_noise = jax.random.multivariate_normal(
        observation_key,
        mean=jnp.zeros((obs_dim,), dtype=model.transition_matrix.dtype),
        cov=observation_cov,
        shape=(num_periods,),
    )

    def step(state: jax.Array, noises: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        process_eps, observation_eps = noises
        next_state = model.transition_matrix @ state + process_eps
        observation = model.observation_matrix @ next_state + observation_eps
        return next_state, (next_state, observation)

    _, (states, observations) = lax.scan(
        step,
        model.initial_mean,
        (process_noise, observation_noise),
    )
    return StateSpaceSimulation(states=states.T, observations=observations.T)


def kalman_filter(
    model: LinearGaussianStateSpace,
    observations: Union[jax.Array, np.ndarray],
    *,
    presample_periods: int = 0,
    jitter: float = 1e-9,
) -> KalmanFilterResult:
    y = jnp.asarray(observations, dtype=model.transition_matrix.dtype)
    if y.ndim != 2:
        raise ValueError(f"observations must be rank-2, got shape {y.shape}.")
    obs_dim = model.observation_matrix.shape[0]
    if y.shape[0] != obs_dim:
        raise ValueError(
            f"observations must have shape ({obs_dim}, T), got {y.shape}."
        )

    state_dim = model.transition_matrix.shape[0]
    identity = jnp.eye(state_dim, dtype=model.transition_matrix.dtype)
    obs_identity = jnp.eye(obs_dim, dtype=model.transition_matrix.dtype)

    def step(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        filtered_mean_prev, filtered_cov_prev = carry
        y_t, active = inputs

        predicted_mean = model.transition_matrix @ filtered_mean_prev
        predicted_cov = (
            model.transition_matrix @ filtered_cov_prev @ model.transition_matrix.T
            + model.process_noise_covariance
        )

        innovation = y_t - model.observation_matrix @ predicted_mean
        innovation_cov = (
            model.observation_matrix @ predicted_cov @ model.observation_matrix.T
            + model.observation_noise_covariance
            + jitter * obs_identity
        )
        innovation_precision = jnp.linalg.inv(innovation_cov)
        kalman_gain = predicted_cov @ model.observation_matrix.T @ innovation_precision

        filtered_mean = predicted_mean + kalman_gain @ innovation
        joseph_left = identity - kalman_gain @ model.observation_matrix
        filtered_cov = (
            joseph_left @ predicted_cov @ joseph_left.T
            + kalman_gain @ model.observation_noise_covariance @ kalman_gain.T
        )

        sign, logdet = jnp.linalg.slogdet(innovation_cov)
        quadratic_form = innovation @ innovation_precision @ innovation
        base_loglik = -0.5 * (obs_dim * LOG_2PI + logdet + quadratic_form)
        loglik_t = jnp.where(jnp.logical_and(active, sign > 0), base_loglik, 0.0)

        outputs = (
            filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            innovation,
            innovation_cov,
            loglik_t,
        )
        return (filtered_mean, filtered_cov), outputs

    active_periods = jnp.arange(y.shape[1]) >= presample_periods
    (_, _), outputs = lax.scan(
        step,
        (model.initial_mean, model.initial_covariance),
        (y.T, active_periods),
    )

    (
        filtered_means_t,
        filtered_covariances_t,
        predicted_means_t,
        predicted_covariances_t,
        innovations_t,
        innovation_covariances_t,
        loglikelihood_per_period,
    ) = outputs

    return KalmanFilterResult(
        filtered_means=filtered_means_t.T,
        filtered_covariances=filtered_covariances_t,
        predicted_means=predicted_means_t.T,
        predicted_covariances=predicted_covariances_t,
        innovations=innovations_t.T,
        innovation_covariances=innovation_covariances_t,
        loglikelihood_per_period=loglikelihood_per_period,
        total_loglikelihood=jnp.sum(loglikelihood_per_period),
    )


def kalman_loglikelihood(
    model: LinearGaussianStateSpace,
    observations: Union[jax.Array, np.ndarray],
    *,
    presample_periods: int = 0,
    jitter: float = 1e-9,
) -> jax.Array:
    return kalman_filter(
        model,
        observations,
        presample_periods=presample_periods,
        jitter=jitter,
    ).total_loglikelihood


def kalman_loglikelihood_per_period(
    model: LinearGaussianStateSpace,
    observations: Union[jax.Array, np.ndarray],
    *,
    presample_periods: int = 0,
    jitter: float = 1e-9,
) -> jax.Array:
    return kalman_filter(
        model,
        observations,
        presample_periods=presample_periods,
        jitter=jitter,
    ).loglikelihood_per_period


def kalman_smoother(
    model: LinearGaussianStateSpace,
    filter_result: KalmanFilterResult,
    *,
    jitter: float = 1e-9,
) -> KalmanSmootherResult:
    state_dim = model.transition_matrix.shape[0]
    filtered_means = filter_result.filtered_means.T
    filtered_covariances = filter_result.filtered_covariances
    predicted_means = filter_result.predicted_means.T
    predicted_covariances = filter_result.predicted_covariances

    smoothed_means = [filtered_means[-1]]
    smoothed_covariances = [filtered_covariances[-1]]
    identity = jnp.eye(state_dim, dtype=model.transition_matrix.dtype)

    for t in range(filtered_means.shape[0] - 2, -1, -1):
        predicted_cov_next = predicted_covariances[t + 1] + jitter * identity
        smoother_gain = (
            filtered_covariances[t]
            @ model.transition_matrix.T
            @ jnp.linalg.inv(predicted_cov_next)
        )
        next_smoothed_mean = smoothed_means[-1]
        next_smoothed_cov = smoothed_covariances[-1]
        smoothed_mean = filtered_means[t] + smoother_gain @ (
            next_smoothed_mean - predicted_means[t + 1]
        )
        smoothed_cov = filtered_covariances[t] + smoother_gain @ (
            next_smoothed_cov - predicted_covariances[t + 1]
        ) @ smoother_gain.T
        smoothed_means.append(smoothed_mean)
        smoothed_covariances.append(smoothed_cov)

    smoothed_means_arr = jnp.stack(smoothed_means[::-1], axis=0)
    smoothed_covariances_arr = jnp.stack(smoothed_covariances[::-1], axis=0)
    return KalmanSmootherResult(
        smoothed_means=smoothed_means_arr.T,
        smoothed_covariances=smoothed_covariances_arr,
    )
