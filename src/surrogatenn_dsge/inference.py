from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
from jax import core as jax_core
from jax import lax
import jax.numpy as jnp
import numpy as np

from .dsge import solve_first_order_dsge_solution_jax
from .linalg import solve_discrete_lyapunov_direct
from .model import MacroModel, kalman_loglikelihood_from_model
from .statespace import LinearGaussianStateSpace, kalman_loglikelihood as _statespace_kalman_loglikelihood


def _require_numpyro() -> tuple[Any, Any, Any]:
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer.util import log_density
    except ImportError as exc:
        raise ImportError(
            "NumPyro integration requires the optional `numpyro` dependency. "
            "Install the `inference` extra or add `numpyro` to the environment."
        ) from exc
    return numpyro, dist, log_density


def _coerce_base_parameter_vector(
    model: MacroModel,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]],
) -> jax.Array:
    if base_parameter_values is None:
        return jnp.asarray(model.parameter_values, dtype=jnp.float64)
    if isinstance(base_parameter_values, Mapping):
        unknown = tuple(
            sorted(set(base_parameter_values).difference(model.parameter_names))
        )
        if unknown:
            raise ValueError(
                "Unknown parameter names in `base_parameter_values`: "
                + ", ".join(unknown)
                + "."
            )
        base = np.asarray(model.parameter_values, dtype=np.float64).copy()
        index_lookup = {name: idx for idx, name in enumerate(model.parameter_names)}
        for name, value in base_parameter_values.items():
            base[index_lookup[name]] = float(value)
        return jnp.asarray(base, dtype=jnp.float64)
    base = jnp.asarray(base_parameter_values, dtype=jnp.float64)
    expected_shape = (len(model.parameter_names),)
    if base.shape != expected_shape:
        raise ValueError(
            "base_parameter_values must have shape "
            f"{expected_shape}, got {base.shape}."
        )
    return base


def assemble_parameter_vector(
    model: MacroModel,
    updated_parameter_values: Mapping[str, Any],
    *,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
) -> jax.Array:
    unknown = tuple(
        sorted(set(updated_parameter_values).difference(model.parameter_names))
    )
    if unknown:
        raise ValueError(
            "Unknown parameter names in `updated_parameter_values`: "
            + ", ".join(unknown)
            + "."
        )

    parameter_vector = _coerce_base_parameter_vector(model, base_parameter_values)
    index_lookup = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value in updated_parameter_values.items():
        parameter_vector = parameter_vector.at[index_lookup[name]].set(
            jnp.asarray(value, dtype=jnp.float64)
        )
    return parameter_vector


def _coerce_parameter_vector_for_jax(
    model: MacroModel,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]],
    *,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
) -> jax.Array:
    if parameter_values is None:
        return _coerce_base_parameter_vector(model, base_parameter_values)
    if isinstance(parameter_values, Mapping):
        return assemble_parameter_vector(
            model,
            parameter_values,
            base_parameter_values=base_parameter_values,
        )
    vector = jnp.asarray(parameter_values, dtype=jnp.float64)
    expected_shape = (len(model.parameter_names),)
    if vector.shape != expected_shape:
        raise ValueError(
            "parameter_values must have shape "
            f"{expected_shape}, got {vector.shape}."
        )
    return vector


def _linear_state_space_from_first_order_solution_jax(
    solution_matrix: jax.Array,
    model: MacroModel,
    observable_indices: Sequence[int],
    *,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
) -> LinearGaussianStateSpace:
    timings = model.timings
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    obs_zero = tuple(int(i) for i in observable_indices)
    observables_and_states = tuple(
        sorted(set(timings.past_not_future_and_mixed_idx) | set(obs_zero))
    )
    selector = jnp.eye(len(observables_and_states), dtype=solution.dtype)[
        list(observables_and_states.index(idx) for idx in timings.past_not_future_and_mixed_idx),
        :,
    ]
    transition = (
        solution[list(observables_and_states), : timings.nPast_not_future_and_mixed]
        @ selector
    )
    shock_impact = solution[
        list(observables_and_states),
        timings.nPast_not_future_and_mixed :,
    ]
    observation = jnp.eye(len(observables_and_states), dtype=solution.dtype)[
        list(observables_and_states.index(idx) for idx in tuple(sorted(obs_zero))),
        :,
    ]
    if measurement_error_covariance is None:
        observation_noise = measurement_error_scale * jnp.eye(
            observation.shape[0],
            dtype=solution.dtype,
        )
    else:
        observation_noise = jnp.asarray(
            measurement_error_covariance,
            dtype=solution.dtype,
        )
        expected_shape = (len(obs_zero), len(obs_zero))
        if observation_noise.shape != expected_shape:
            raise ValueError(
                "measurement_error_covariance must have shape "
                f"{expected_shape}, got {observation_noise.shape}."
            )
    process_noise_covariance = shock_impact @ shock_impact.T
    if initial_covariance_strategy == "theoretical":
        initial_covariance = solve_discrete_lyapunov_direct(
            transition,
            process_noise_covariance,
        ).solution
    elif initial_covariance_strategy == "diagonal":
        initial_covariance = 10.0 * jnp.eye(
            transition.shape[0],
            dtype=transition.dtype,
        )
    else:
        raise ValueError(
            "initial_covariance_strategy must be 'theoretical' or 'diagonal'."
        )
    return LinearGaussianStateSpace(
        transition_matrix=transition,
        process_noise_covariance=process_noise_covariance,
        observation_matrix=observation,
        observation_noise_covariance=observation_noise,
        initial_mean=jnp.zeros((transition.shape[0],), dtype=transition.dtype),
        initial_covariance=initial_covariance,
    )


def kalman_loglikelihood_from_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    observables: Optional[Sequence[str] | str] = None,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    observable_names, observation_data = model._coerce_observations(
        observations,
        observables=observables,
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    observations_array = jnp.asarray(observation_data, dtype=jnp.float64)
    lower_bounds, upper_bounds = model._bounds_vector(model.parameter_names)
    lower_bounds_array = jnp.asarray(lower_bounds, dtype=jnp.float64)
    upper_bounds_array = jnp.asarray(upper_bounds, dtype=jnp.float64)
    failure_value = jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
    parameter_vector = _coerce_parameter_vector_for_jax(
        model,
        parameter_values,
        base_parameter_values=base_parameter_values,
    )
    future_index_array = jnp.asarray(
        model.timings.future_not_past_and_mixed_idx,
        dtype=jnp.int32,
    )
    past_index_array = jnp.asarray(
        model.timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )
    explicit_steady_state = (
        None
        if steady_state is None
        else jnp.asarray(model._coerce_full_steady_state(steady_state), dtype=jnp.float64)
    )

    def _loglikelihood_from_full_steady_state(
        full_steady_state: jax.Array,
        parameters: jax.Array,
    ) -> jax.Array:
        steady_reference_values = model._steady_reference_values_jax(full_steady_state)
        observable_steady_state = full_steady_state[observable_index_array]
        demeaned_observations = observations_array - observable_steady_state[:, None]
        dynamic_point = jnp.concatenate(
            [
                full_steady_state[future_index_array],
                full_steady_state,
                full_steady_state[past_index_array],
                jnp.zeros((model.timings.nExo,), dtype=jnp.float64),
            ]
        )

        def residual_from_dynamic_vector(dynamic_vector: jax.Array) -> jax.Array:
            lead_state = full_steady_state.at[future_index_array].set(
                dynamic_vector[: model.timings.nFuture_not_past_and_mixed]
            )
            current_start = model.timings.nFuture_not_past_and_mixed
            current_end = current_start + model.timings.nVars
            current_state = dynamic_vector[current_start:current_end]
            lag_state = full_steady_state.at[past_index_array].set(
                dynamic_vector[
                    current_end : current_end + model.timings.nPast_not_future_and_mixed
                ]
            )
            shock = dynamic_vector[current_end + model.timings.nPast_not_future_and_mixed :]
            return model._evaluate_dynamic_residual_with_context(
                lag_state,
                current_state,
                lead_state,
                shock,
                parameter_values=parameters,
                steady_reference_values=steady_reference_values,
            )

        jacobian = jax.jacrev(residual_from_dynamic_vector)(dynamic_point)
        first_order_result = solve_first_order_dsge_solution_jax(
            jacobian,
            model.timings,
        )

        def _success(result) -> jax.Array:
            state_space = _linear_state_space_from_first_order_solution_jax(
                result.solution_matrix,
                model,
                observable_indices,
                initial_covariance_strategy=initial_covariance_strategy,
                measurement_error_scale=measurement_error_scale,
                measurement_error_covariance=measurement_error_covariance,
            )
            return _statespace_kalman_loglikelihood(
                state_space,
                demeaned_observations,
                presample_periods=presample_periods,
                jitter=jitter,
            )

        return lax.cond(
            first_order_result.converged,
            _success,
            lambda _: failure_value,
            first_order_result,
        )

    def _valid_loglikelihood(parameters: jax.Array) -> jax.Array:
        if explicit_steady_state is not None:
            return _loglikelihood_from_full_steady_state(explicit_steady_state, parameters)

        steady_state_result = model.solve_steady_state_jax(
            parameter_values=parameters,
            initial_guess=steady_state_initial_guess,
            tol=steady_state_tol,
            max_iter=steady_state_max_iter,
        )
        return lax.cond(
            steady_state_result.converged,
            lambda result: _loglikelihood_from_full_steady_state(
                result.steady_state,
                parameters,
            ),
            lambda _: failure_value,
            steady_state_result,
        )

    within_bounds = jnp.all(
        (parameter_vector >= lower_bounds_array) & (parameter_vector <= upper_bounds_array)
    )
    return lax.cond(
        within_bounds,
        _valid_loglikelihood,
        lambda _: failure_value,
        parameter_vector,
    )


def build_numpyro_kalman_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    *,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
):
    numpyro, _, _ = _require_numpyro()

    prior_names = tuple(priors)
    if not prior_names:
        raise ValueError("priors must contain at least one parameter prior.")
    unknown = tuple(sorted(set(prior_names).difference(model.parameter_names)))
    if unknown:
        raise ValueError(
            "Unknown parameter names in `priors`: "
            + ", ".join(unknown)
            + "."
        )
    base_parameters = _coerce_base_parameter_vector(model, base_parameter_values)

    def numpyro_model() -> None:
        sampled_values = {
            name: numpyro.sample(name, priors[name])
            for name in prior_names
        }
        if any(isinstance(value, jax_core.Tracer) for value in sampled_values.values()):
            raise NotImplementedError(
                "Parsed-model NumPyro estimation is not yet JAX-traceable enough "
                "for compiled kernels like NUTS. The current wrapper supports "
                "concrete log-density evaluation and explicit parameter-vector "
                "assembly, but the steady-state / symbolic derivative path still "
                "needs a pure-JAX port."
            )

        parameter_vector = assemble_parameter_vector(
            model,
            sampled_values,
            base_parameter_values=base_parameters,
        )
        loglikelihood = kalman_loglikelihood_from_model(
            model,
            observations,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("kalman_loglikelihood", loglikelihood)

    return numpyro_model


def build_numpyro_kalman_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    *,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
    ):
    numpyro, _, _ = _require_numpyro()

    prior_names = tuple(priors)
    if not prior_names:
        raise ValueError("priors must contain at least one parameter prior.")
    unknown = tuple(sorted(set(prior_names).difference(model.parameter_names)))
    if unknown:
        raise ValueError(
            "Unknown parameter names in `priors`: "
            + ", ".join(unknown)
            + "."
        )
    base_parameters = _coerce_base_parameter_vector(model, base_parameter_values)

    def numpyro_model() -> None:
        sampled_values = {
            name: numpyro.sample(name, priors[name])
            for name in prior_names
        }
        parameter_vector = assemble_parameter_vector(
            model,
            sampled_values,
            base_parameter_values=base_parameters,
        )
        loglikelihood = kalman_loglikelihood_from_model_jax(
            model,
            observations,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("kalman_loglikelihood", loglikelihood)

    return numpyro_model


def evaluate_numpyro_kalman_log_density(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    *,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    _, _, log_density = _require_numpyro()
    numpyro_model = build_numpyro_kalman_model(
        model,
        observations,
        priors,
        observables=observables,
        base_parameter_values=base_parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)


def evaluate_numpyro_kalman_log_density_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    *,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    _, _, log_density = _require_numpyro()
    numpyro_model = build_numpyro_kalman_model_jax(
        model,
        observations,
        priors,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        observables=observables,
        base_parameter_values=base_parameter_values,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)
