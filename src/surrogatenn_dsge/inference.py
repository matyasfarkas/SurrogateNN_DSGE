from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Sequence

import jax
from jax import core as jax_core
from jax import lax
import jax.numpy as jnp
import numpy as np

from .dsge import rollout_first_order_solution, solve_first_order_dsge_solution_jax
from .inversion import first_order_inversion_loglikelihood_per_period
from .linalg import solve_discrete_lyapunov_direct
from .model import MacroModel, kalman_loglikelihood_from_model
from .statespace import (
    LinearGaussianStateSpace,
    kalman_filter,
    kalman_loglikelihood as _statespace_kalman_loglikelihood,
    kalman_loglikelihood_per_period as _statespace_kalman_loglikelihood_per_period,
    kalman_smoother,
)
from .switching import (
    LinearGateStatsResult,
    RegimeSwitchConfig,
    SwitchingLikelihoodConfig,
    compute_gate_stat_series_jax,
    compute_switching_loglikelihood,
    gate_probabilities_jax,
)


class _LinearFilterPathResult(NamedTuple):
    ok: jax.Array
    shocks: jax.Array
    variables: jax.Array


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
    qme_algorithm: str = "schur",
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
            qme_algorithm=qme_algorithm,
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
            resolved_parameters = model.resolve_parameter_values_jax(
                parameter_values=parameters,
                steady_state=explicit_steady_state,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            return _loglikelihood_from_full_steady_state(
                explicit_steady_state,
                resolved_parameters,
            )

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
                result.parameter_values,
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


def switching_loglikelihood_from_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> jax.Array:
    if fom_algorithm != "first_order":
        raise NotImplementedError(
            "The compiled JAX switching likelihood currently supports "
            "fom_algorithm='first_order' only."
        )
    if (gate_probs is None) == (hard_mask is None):
        raise ValueError("Provide exactly one of gate_probs or hard_mask.")

    observable_names, observation_data = model._coerce_observations(
        observations,
        observables=observables,
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    observations_array = jnp.asarray(observation_data, dtype=jnp.float64)
    n_periods = observations_array.shape[1]

    gate_probs_array = None
    if gate_probs is not None:
        gate_probs_array = jnp.asarray(gate_probs, dtype=jnp.float64).reshape(-1)
        if gate_probs_array.shape != (n_periods,):
            raise ValueError(
                f"gate_probs must have shape ({n_periods},), got {gate_probs_array.shape}."
            )
    hard_mask_array = None
    if hard_mask is not None:
        hard_mask_array = jnp.asarray(hard_mask, dtype=bool).reshape(-1)
        if hard_mask_array.shape != (n_periods,):
            raise ValueError(
                f"hard_mask must have shape ({n_periods},), got {hard_mask_array.shape}."
            )

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
    explicit_initial_state = (
        None
        if initial_state is None
        else jnp.asarray(
            model._coerce_dynamic_state_vector(initial_state, label="initial_state"),
            dtype=jnp.float64,
        )
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
            qme_algorithm=qme_algorithm,
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
            rom = _statespace_kalman_loglikelihood_per_period(
                state_space,
                demeaned_observations,
                presample_periods=presample_periods,
                jitter=jitter,
            )
            inversion_initial_state = (
                jnp.zeros((model.timings.nVars,), dtype=jnp.float64)
                if explicit_initial_state is None
                else explicit_initial_state - full_steady_state
            )
            fom = first_order_inversion_loglikelihood_per_period(
                result.solution_matrix,
                model.timings,
                demeaned_observations,
                observable_indices,
                initial_state=inversion_initial_state,
                presample_periods=presample_periods,
                on_failure_loglikelihood=on_failure_loglikelihood,
            )
            switching = compute_switching_loglikelihood(
                rom,
                fom,
                hard_mask=hard_mask_array,
                gate_probs=gate_probs_array,
                config=switching_config,
            )
            return switching.total

        return lax.cond(
            first_order_result.converged,
            _success,
            lambda _: failure_value,
            first_order_result,
        )

    def _valid_loglikelihood(parameters: jax.Array) -> jax.Array:
        if explicit_steady_state is not None:
            resolved_parameters = model.resolve_parameter_values_jax(
                parameter_values=parameters,
                steady_state=explicit_steady_state,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            return _loglikelihood_from_full_steady_state(
                explicit_steady_state,
                resolved_parameters,
            )

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
                result.parameter_values,
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


def _estimate_first_order_inversion_filter_paths_jax(
    solution_matrix: jax.Array,
    model: MacroModel,
    observation_deviations: jax.Array,
    observable_indices: Sequence[int],
    *,
    fill_value: jax.Array,
) -> _LinearFilterPathResult:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    observations = jnp.asarray(observation_deviations, dtype=jnp.float64)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    state_index_array = jnp.asarray(
        model.timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )
    n_past = model.timings.nPast_not_future_and_mixed
    periods = observations.shape[1]
    observable_transition = solution[observable_index_array, :n_past]
    shock_jacobian = solution[observable_index_array, n_past:]
    if model.timings.nExo == len(observable_indices):
        shock_map = jnp.linalg.inv(shock_jacobian)
    else:
        shock_map = jnp.linalg.pinv(shock_jacobian)

    empty_shocks = jnp.full(
        (model.timings.nExo, periods),
        fill_value,
        dtype=jnp.float64,
    )
    empty_variables = jnp.full(
        (model.timings.nVars, periods),
        fill_value,
        dtype=jnp.float64,
    )
    map_ok = jnp.all(jnp.isfinite(shock_jacobian)) & jnp.all(jnp.isfinite(shock_map))

    def _success(_: Any) -> _LinearFilterPathResult:
        def step(
            carry: tuple[jax.Array, jax.Array],
            observation_t: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            reduced_state, valid = carry

            def _active(_: Any) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                residual = observation_t - observable_transition @ reduced_state
                shock_t = shock_map @ residual
                next_state = solution @ jnp.concatenate([reduced_state, shock_t], axis=0)
                finite = (
                    jnp.all(jnp.isfinite(residual))
                    & jnp.all(jnp.isfinite(shock_t))
                    & jnp.all(jnp.isfinite(next_state))
                )
                safe_reduced = jnp.where(
                    finite,
                    next_state[state_index_array],
                    reduced_state,
                )
                shock_out = jnp.where(
                    finite,
                    shock_t,
                    jnp.full_like(shock_t, fill_value),
                )
                state_out = jnp.where(
                    finite,
                    next_state,
                    jnp.full_like(next_state, fill_value),
                )
                return (safe_reduced, finite), (shock_out, state_out)

            def _inactive(_: Any) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                return (
                    (reduced_state, jnp.asarray(False, dtype=jnp.bool_)),
                    (
                        jnp.full((model.timings.nExo,), fill_value, dtype=solution.dtype),
                        jnp.full((model.timings.nVars,), fill_value, dtype=solution.dtype),
                    ),
                )

            return lax.cond(valid, _active, _inactive, operand=None)

        initial_carry = (
            jnp.zeros((n_past,), dtype=solution.dtype),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        (_, valid), (shocks_t, variables_t) = lax.scan(step, initial_carry, observations.T)
        return _LinearFilterPathResult(
            ok=valid,
            shocks=shocks_t.T,
            variables=variables_t.T,
        )

    return lax.cond(
        map_ok,
        _success,
        lambda _: _LinearFilterPathResult(
            ok=jnp.asarray(False, dtype=jnp.bool_),
            shocks=empty_shocks,
            variables=empty_variables,
        ),
        operand=None,
    )


def _estimate_first_order_kalman_filter_paths_jax(
    solution_matrix: jax.Array,
    model: MacroModel,
    observation_deviations: jax.Array,
    observable_indices: Sequence[int],
    *,
    smooth: bool,
    initial_covariance_strategy: str,
    jitter: float,
    fill_value: jax.Array,
) -> _LinearFilterPathResult:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    observations = jnp.asarray(observation_deviations, dtype=jnp.float64)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    state_index_array = jnp.asarray(
        model.timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )
    n_past = model.timings.nPast_not_future_and_mixed
    periods = observations.shape[1]

    state_space = _linear_state_space_from_first_order_solution_jax(
        solution,
        model,
        observable_indices,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=0.0,
    )
    filter_result = kalman_filter(
        state_space,
        observations,
        presample_periods=0,
        jitter=jitter,
    )
    latent_path = (
        kalman_smoother(
            state_space,
            filter_result,
            jitter=jitter,
        ).smoothed_means
        if smooth
        else filter_result.filtered_means
    )

    latent_indices = tuple(
        sorted(set(model.timings.past_not_future_and_mixed_idx) | set(observable_indices))
    )
    latent_index_array = jnp.asarray(latent_indices, dtype=jnp.int32)
    latent_transition = solution[latent_index_array, :n_past]
    latent_shock_impact = solution[latent_index_array, n_past:]
    if len(latent_indices) == model.timings.nExo:
        shock_map = jnp.linalg.inv(latent_shock_impact)
    else:
        shock_map = jnp.linalg.pinv(latent_shock_impact)

    empty_shocks = jnp.full(
        (model.timings.nExo, periods),
        fill_value,
        dtype=jnp.float64,
    )
    empty_variables = jnp.full(
        (model.timings.nVars, periods),
        fill_value,
        dtype=jnp.float64,
    )
    map_ok = (
        jnp.all(jnp.isfinite(latent_path))
        & jnp.all(jnp.isfinite(latent_shock_impact))
        & jnp.all(jnp.isfinite(shock_map))
    )

    def _success(_: Any) -> _LinearFilterPathResult:
        def step(
            carry: tuple[jax.Array, jax.Array],
            latent_t: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            reduced_state, valid = carry

            def _active(_: Any) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                residual = latent_t - latent_transition @ reduced_state
                shock_t = shock_map @ residual
                next_state = solution @ jnp.concatenate([reduced_state, shock_t], axis=0)
                finite = (
                    jnp.all(jnp.isfinite(residual))
                    & jnp.all(jnp.isfinite(shock_t))
                    & jnp.all(jnp.isfinite(next_state))
                )
                safe_reduced = jnp.where(
                    finite,
                    next_state[state_index_array],
                    reduced_state,
                )
                shock_out = jnp.where(
                    finite,
                    shock_t,
                    jnp.full_like(shock_t, fill_value),
                )
                state_out = jnp.where(
                    finite,
                    next_state,
                    jnp.full_like(next_state, fill_value),
                )
                return (safe_reduced, finite), (shock_out, state_out)

            def _inactive(_: Any) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                return (
                    (reduced_state, jnp.asarray(False, dtype=jnp.bool_)),
                    (
                        jnp.full((model.timings.nExo,), fill_value, dtype=solution.dtype),
                        jnp.full((model.timings.nVars,), fill_value, dtype=solution.dtype),
                    ),
                )

            return lax.cond(valid, _active, _inactive, operand=None)

        initial_carry = (
            jnp.zeros((n_past,), dtype=solution.dtype),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        (_, valid), (shocks_t, variables_t) = lax.scan(step, initial_carry, latent_path.T)
        return _LinearFilterPathResult(
            ok=valid,
            shocks=shocks_t.T,
            variables=variables_t.T,
        )

    return lax.cond(
        map_ok,
        _success,
        lambda _: _LinearFilterPathResult(
            ok=jnp.asarray(False, dtype=jnp.bool_),
            shocks=empty_shocks,
            variables=empty_variables,
        ),
        operand=None,
    )


def _estimate_observed_shocks_and_variables_matrix_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    data_in_levels: bool = True,
    levels: bool = True,
    smooth: bool = False,
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> tuple[tuple[str, ...], jax.Array, jax.Array, jax.Array, jax.Array]:
    filter_name = str(filter)
    if filter_name not in {"kalman", "inversion"}:
        raise ValueError(
            f"Unsupported filter {filter!r}. Use 'kalman' or 'inversion'."
        )
    if str(algorithm) != "first_order":
        raise ValueError(
            "Only the first-order filter helper path is currently ported. "
            f"Got algorithm={algorithm!r}."
        )

    observable_names, observation_data = model._coerce_observations(
        observations,
        observables=observables,
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    observations_array = jnp.asarray(observation_data, dtype=jnp.float64)
    periods = observations_array.shape[1]
    fill_value = jnp.asarray(on_failure_fill_value, dtype=jnp.float64)

    lower_bounds, upper_bounds = model._bounds_vector(model.parameter_names)
    lower_bounds_array = jnp.asarray(lower_bounds, dtype=jnp.float64)
    upper_bounds_array = jnp.asarray(upper_bounds, dtype=jnp.float64)
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

    def _failure_result(_: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
            jnp.full((model.timings.nVars,), fill_value, dtype=jnp.float64),
            jnp.full(
                (
                    model.timings.nVars,
                    model.timings.nPast_not_future_and_mixed + model.timings.nExo,
                ),
                fill_value,
                dtype=jnp.float64,
            ),
            jnp.full((model.timings.nExo, periods), fill_value, dtype=jnp.float64),
            jnp.full((model.timings.nVars, periods), fill_value, dtype=jnp.float64),
        )

    def _paths_from_full_steady_state(
        full_steady_state: jax.Array,
        parameters: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        steady_reference_values = model._steady_reference_values_jax(full_steady_state)
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
            qme_algorithm=qme_algorithm,
        )

        def _success(result: Any) -> tuple[jax.Array, jax.Array]:
            observable_steady_state = full_steady_state[observable_index_array]
            observation_deviations = (
                observations_array - observable_steady_state[:, None]
                if data_in_levels
                else observations_array
            )

            def _zero_case(_: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                zero_shocks = jnp.zeros((model.timings.nExo, periods), dtype=jnp.float64)
                zero_variables = jnp.zeros((model.timings.nVars, periods), dtype=jnp.float64)
                return (
                    full_steady_state,
                    result.solution_matrix,
                    zero_shocks,
                    (
                        zero_variables + full_steady_state[:, None]
                        if levels and periods > 0
                        else zero_variables
                    ),
                )

            def _nonzero_case(_: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                path_result = (
                    _estimate_first_order_inversion_filter_paths_jax(
                        result.solution_matrix,
                        model,
                        observation_deviations,
                        observable_indices,
                        fill_value=fill_value,
                    )
                    if filter_name == "inversion"
                    else _estimate_first_order_kalman_filter_paths_jax(
                        result.solution_matrix,
                        model,
                        observation_deviations,
                        observable_indices,
                        smooth=smooth,
                        initial_covariance_strategy=initial_covariance_strategy,
                        jitter=jitter,
                        fill_value=fill_value,
                    )
                )

                def _path_success(
                    paths: _LinearFilterPathResult,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    return (
                        full_steady_state,
                        result.solution_matrix,
                        paths.shocks,
                        paths.variables + full_steady_state[:, None]
                        if levels
                        else paths.variables,
                    )

                return lax.cond(
                    path_result.ok,
                    _path_success,
                    _failure_result,
                    path_result,
                )

            if periods == 0:
                return _zero_case(None)
            near_zero = jnp.max(jnp.abs(observation_deviations)) <= 1e-14
            return lax.cond(near_zero, _zero_case, _nonzero_case, operand=None)

        return lax.cond(
            first_order_result.converged,
            _success,
            _failure_result,
            first_order_result,
        )

    def _valid_paths(parameters: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if explicit_steady_state is not None:
            resolved_parameters = model.resolve_parameter_values_jax(
                parameter_values=parameters,
                steady_state=explicit_steady_state,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            return _paths_from_full_steady_state(
                explicit_steady_state,
                resolved_parameters,
            )

        steady_state_result = model.solve_steady_state_jax(
            parameter_values=parameters,
            initial_guess=steady_state_initial_guess,
            tol=steady_state_tol,
            max_iter=steady_state_max_iter,
        )
        return lax.cond(
            steady_state_result.converged,
            lambda result: _paths_from_full_steady_state(
                result.steady_state,
                result.parameter_values,
            ),
            _failure_result,
            steady_state_result,
        )

    within_bounds = jnp.all(
        (parameter_vector >= lower_bounds_array) & (parameter_vector <= upper_bounds_array)
    )
    full_steady_state_out, solution_matrix_out, shocks_matrix, variables_matrix = lax.cond(
        within_bounds,
        _valid_paths,
        _failure_result,
        parameter_vector,
    )
    return (
        observable_names,
        full_steady_state_out,
        solution_matrix_out,
        shocks_matrix,
        variables_matrix,
    )


def estimate_observed_shocks_matrix_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    data_in_levels: bool = True,
    smooth: bool = False,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    label: str = "Estimated shocks",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> jax.Array:
    _, _, _, shocks_matrix, _ = _estimate_observed_shocks_and_variables_matrix_model_jax(
        model,
        observations,
        observables=observables,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        parameter_values=parameter_values,
        base_parameter_values=base_parameter_values,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=data_in_levels,
        levels=False,
        smooth=smooth,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
        on_failure_fill_value=on_failure_fill_value,
    )
    if expected_rows is not None and shocks_matrix.shape[0] != int(expected_rows):
        raise ValueError(
            f"{label} row mismatch: got {shocks_matrix.shape[0]}, expected {int(expected_rows)}."
        )
    if expected_cols is not None and shocks_matrix.shape[1] != int(expected_cols):
        raise ValueError(
            f"{label} length mismatch: got {shocks_matrix.shape[1]}, expected {int(expected_cols)}."
        )
    return shocks_matrix


def estimate_observed_variables_matrix_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    data_in_levels: bool = True,
    levels: bool = True,
    smooth: bool = False,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    label: str = "Estimated variables",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> jax.Array:
    _, _, _, _, variables_matrix = _estimate_observed_shocks_and_variables_matrix_model_jax(
        model,
        observations,
        observables=observables,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        parameter_values=parameter_values,
        base_parameter_values=base_parameter_values,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=data_in_levels,
        levels=levels,
        smooth=smooth,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
        on_failure_fill_value=on_failure_fill_value,
    )
    if expected_rows is not None and variables_matrix.shape[0] != int(expected_rows):
        raise ValueError(
            f"{label} row mismatch: got {variables_matrix.shape[0]}, expected {int(expected_rows)}."
        )
    if expected_cols is not None and variables_matrix.shape[1] != int(expected_cols):
        raise ValueError(
            f"{label} length mismatch: got {variables_matrix.shape[1]}, expected {int(expected_cols)}."
        )
    return variables_matrix


def linear_filter_initial_state_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    state_names: Sequence[str] | str,
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    label: str = "Linear filter variables",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> jax.Array:
    names = model._coerce_observable_names(state_names)
    variables_matrix = estimate_observed_variables_matrix_jax(
        model,
        observations,
        observables=observables,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        parameter_values=parameter_values,
        base_parameter_values=base_parameter_values,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=True,
        levels=True,
        smooth=smooth,
        expected_cols=model._coerce_observations(
            observations,
            observables=observables,
        )[1].shape[1],
        label=label,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
        on_failure_fill_value=on_failure_fill_value,
    )
    variable_lookup = {name: idx for idx, name in enumerate(model.timings.var)}
    missing = tuple(name for name in names if name not in variable_lookup)
    if missing:
        raise ValueError(
            "state_names not found in linear filter output: "
            + ", ".join(missing)
            + "."
        )
    return variables_matrix[
        jnp.asarray([variable_lookup[name] for name in names], dtype=jnp.int32),
        -1,
    ]


def linear_filter_full_state_initial_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    label: str = "Linear filter variables",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> jax.Array:
    variables_matrix = estimate_observed_variables_matrix_jax(
        model,
        observations,
        observables=observables,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        parameter_values=parameter_values,
        base_parameter_values=base_parameter_values,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=True,
        levels=True,
        smooth=smooth,
        expected_cols=model._coerce_observations(
            observations,
            observables=observables,
        )[1].shape[1],
        label=label,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
        on_failure_fill_value=on_failure_fill_value,
    )
    return variables_matrix[:, 0]


def compute_linear_gate_stats_from_filter_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    state_names: Optional[Sequence[str] | str] = None,
    *,
    observables: Optional[Sequence[str] | str] = None,
    periods: Optional[int] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    shock_norm: str = "l2",
    error_norm: str = "l2",
    label: str = "Linear gate stats",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
    on_failure_fill_value: float = np.nan,
) -> LinearGateStatsResult:
    del state_names
    if periods is not None and int(periods) <= 0:
        raise ValueError(f"periods must be positive, got {periods}.")

    observable_names, full_steady_state, solution_matrix, shocks_matrix, variables_matrix = (
        _estimate_observed_shocks_and_variables_matrix_model_jax(
            model,
            observations,
            observables=observables,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            parameter_values=parameter_values,
            base_parameter_values=base_parameter_values,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=True,
            levels=True,
            smooth=smooth,
            initial_covariance_strategy=initial_covariance_strategy,
            jitter=jitter,
            on_failure_fill_value=on_failure_fill_value,
        )
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    state_index_array = jnp.asarray(
        model.timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    observation_data = jnp.asarray(
        model._coerce_observations(
            observations,
            observables=observables,
        )[1],
        dtype=jnp.float64,
    )
    obs_sigma_vector = jnp.asarray(
        model._coerce_named_values(
            obs_sigma,
            observable_names,
            label="obs_sigma",
        ),
        dtype=jnp.float64,
    )
    shock_sigma_vector = jnp.asarray(
        model._coerce_named_values(
            shock_sigmas,
            model.timings.exo,
            label="shock_sigmas",
        ),
        dtype=jnp.float64,
    )
    reduced_initial_state = (
        variables_matrix[:, 0][state_index_array] - full_steady_state[state_index_array]
    )
    linear_deviations = rollout_first_order_solution(
        solution_matrix,
        model.timings,
        shocks_matrix,
        initial_reduced_state=reduced_initial_state,
    )
    linear_observations = linear_deviations[observable_index_array, :] + (
        full_steady_state[observable_index_array][:, None]
    )
    e_stat, f_stat = compute_gate_stat_series_jax(
        observation_data,
        linear_observations,
        shocks_matrix,
        obs_sigma_vector,
        shock_sigma_vector,
        shock_norm=shock_norm,
        error_norm=error_norm,
    )
    gate_stats = LinearGateStatsResult(
        linear_observations=linear_observations,
        shocks=shocks_matrix,
        e_stat=e_stat,
        f_stat=f_stat,
    )
    if periods is not None and gate_stats.linear_observations.shape[1] != int(periods):
        raise ValueError(
            f"{label} length mismatch: got {gate_stats.linear_observations.shape[1]}, expected {int(periods)}."
        )
    return gate_stats


def switching_loglikelihood_from_model_filter_gates_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    regime_switch_config: RegimeSwitchConfig,
    state_names: Optional[Sequence[str] | str] = None,
    periods: Optional[int] = None,
    gate_filter: str = "kalman",
    gate_algorithm: str = "first_order",
    gate_smooth: bool = False,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> jax.Array:
    gate_stats = compute_linear_gate_stats_from_filter_model_jax(
        model,
        observations,
        obs_sigma,
        shock_sigmas,
        state_names,
        observables=observables,
        periods=periods,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        parameter_values=parameter_values,
        base_parameter_values=base_parameter_values,
        qme_algorithm=qme_algorithm,
        filter=gate_filter,
        algorithm=gate_algorithm,
        smooth=gate_smooth,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
        on_failure_fill_value=np.nan,
    )
    gate_probs = gate_probabilities_jax(
        gate_stats.e_stat,
        gate_stats.f_stat,
        regime_switch_config,
    )
    failure_value = jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
    valid_gate_probs = jnp.all(jnp.isfinite(gate_probs))

    def _success(probs: jax.Array) -> jax.Array:
        return switching_loglikelihood_from_model_jax(
            model,
            observations,
            gate_probs=probs,
            fom_algorithm=fom_algorithm,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            observables=observables,
            parameter_values=parameter_values,
            base_parameter_values=base_parameter_values,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            switching_config=switching_config,
        )

    return lax.cond(
        valid_gate_probs,
        _success,
        lambda _: failure_value,
        gate_probs,
    )


def compute_linear_gate_stats_from_shocks_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    observables: Optional[Sequence[str] | str] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    parameter_values: Optional[Sequence[float] | Mapping[str, Any]] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    shock_norm: str = "l2",
    error_norm: str = "l2",
    on_failure_fill_value: float = np.nan,
) -> LinearGateStatsResult:
    observable_names, observation_data = model._coerce_observations(
        observations,
        observables=observables,
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    observations_array = jnp.asarray(observation_data, dtype=jnp.float64)
    periods = int(observations_array.shape[1])

    shock_values = model._coerce_sep_deterministic_shocks(
        shocks,
        periods=periods,
    )
    if shock_values is None:
        raise ValueError("shocks must be provided for linear gate statistics.")
    shock_matrix = jnp.asarray(shock_values, dtype=jnp.float64).T
    obs_sigma_vector = jnp.asarray(
        model._coerce_named_values(
            obs_sigma,
            observable_names,
            label="obs_sigma",
        ),
        dtype=jnp.float64,
    )
    shock_sigma_vector = jnp.asarray(
        model._coerce_named_values(
            shock_sigmas,
            model.timings.exo,
            label="shock_sigmas",
        ),
        dtype=jnp.float64,
    )

    lower_bounds, upper_bounds = model._bounds_vector(model.parameter_names)
    lower_bounds_array = jnp.asarray(lower_bounds, dtype=jnp.float64)
    upper_bounds_array = jnp.asarray(upper_bounds, dtype=jnp.float64)
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
    state_index_array = jnp.asarray(
        model.timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )
    explicit_steady_state = (
        None
        if steady_state is None
        else jnp.asarray(model._coerce_full_steady_state(steady_state), dtype=jnp.float64)
    )
    explicit_initial_state = (
        None
        if initial_state is None
        else jnp.asarray(
            model._coerce_dynamic_state_vector(initial_state, label="initial_state"),
            dtype=jnp.float64,
        )
    )
    fill_value = jnp.asarray(on_failure_fill_value, dtype=jnp.float64)

    def _failure_result(_: Any) -> LinearGateStatsResult:
        return LinearGateStatsResult(
            linear_observations=jnp.full(
                observations_array.shape,
                fill_value,
                dtype=jnp.float64,
            ),
            shocks=shock_matrix,
            e_stat=jnp.full((periods,), fill_value, dtype=jnp.float64),
            f_stat=jnp.full((periods,), fill_value, dtype=jnp.float64),
        )

    def _stats_from_full_steady_state(
        full_steady_state: jax.Array,
        parameters: jax.Array,
    ) -> LinearGateStatsResult:
        steady_reference_values = model._steady_reference_values_jax(full_steady_state)
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
            qme_algorithm=qme_algorithm,
        )

        def _success(result: Any) -> LinearGateStatsResult:
            initial_state_values = (
                full_steady_state
                if explicit_initial_state is None
                else explicit_initial_state
            )
            reduced_initial_state = (
                initial_state_values[state_index_array] - full_steady_state[state_index_array]
            )
            linear_deviations = rollout_first_order_solution(
                result.solution_matrix,
                model.timings,
                shock_matrix,
                initial_reduced_state=reduced_initial_state,
            )
            linear_observations = linear_deviations[observable_index_array, :] + (
                full_steady_state[observable_index_array][:, None]
            )
            e_stat, f_stat = compute_gate_stat_series_jax(
                observations_array,
                linear_observations,
                shock_matrix,
                obs_sigma_vector,
                shock_sigma_vector,
                shock_norm=shock_norm,
                error_norm=error_norm,
            )
            return LinearGateStatsResult(
                linear_observations=linear_observations,
                shocks=shock_matrix,
                e_stat=e_stat,
                f_stat=f_stat,
            )

        return lax.cond(
            first_order_result.converged,
            _success,
            _failure_result,
            first_order_result,
        )

    def _valid_stats(parameters: jax.Array) -> LinearGateStatsResult:
        if explicit_steady_state is not None:
            resolved_parameters = model.resolve_parameter_values_jax(
                parameter_values=parameters,
                steady_state=explicit_steady_state,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            return _stats_from_full_steady_state(
                explicit_steady_state,
                resolved_parameters,
            )

        steady_state_result = model.solve_steady_state_jax(
            parameter_values=parameters,
            initial_guess=steady_state_initial_guess,
            tol=steady_state_tol,
            max_iter=steady_state_max_iter,
        )
        return lax.cond(
            steady_state_result.converged,
            lambda result: _stats_from_full_steady_state(
                result.steady_state,
                result.parameter_values,
            ),
            _failure_result,
            steady_state_result,
        )

    within_bounds = jnp.all(
        (parameter_vector >= lower_bounds_array) & (parameter_vector <= upper_bounds_array)
    )
    return lax.cond(
        within_bounds,
        _valid_stats,
        _failure_result,
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
    qme_algorithm: str = "schur",
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
            qme_algorithm=qme_algorithm,
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
    qme_algorithm: str = "schur",
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
            qme_algorithm=qme_algorithm,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("kalman_loglikelihood", loglikelihood)

    return numpyro_model


def build_numpyro_switching_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    *,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
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
        loglikelihood = switching_loglikelihood_from_model_jax(
            model,
            observations,
            gate_probs=gate_probs,
            hard_mask=hard_mask,
            fom_algorithm=fom_algorithm,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            observables=observables,
            parameter_values=parameter_vector,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            switching_config=switching_config,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("switching_loglikelihood", loglikelihood)

    return numpyro_model


def build_numpyro_switching_filter_model_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    regime_switch_config: RegimeSwitchConfig,
    state_names: Optional[Sequence[str] | str] = None,
    periods: Optional[int] = None,
    gate_filter: str = "kalman",
    gate_algorithm: str = "first_order",
    gate_smooth: bool = False,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
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
        loglikelihood = switching_loglikelihood_from_model_filter_gates_jax(
            model,
            observations,
            obs_sigma,
            shock_sigmas,
            regime_switch_config=regime_switch_config,
            state_names=state_names,
            periods=periods,
            gate_filter=gate_filter,
            gate_algorithm=gate_algorithm,
            gate_smooth=gate_smooth,
            fom_algorithm=fom_algorithm,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            observables=observables,
            parameter_values=parameter_vector,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            switching_config=switching_config,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("switching_loglikelihood", loglikelihood)

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
    qme_algorithm: str = "schur",
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
        qme_algorithm=qme_algorithm,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)


def evaluate_numpyro_switching_log_density_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    *,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> jax.Array:
    _, _, log_density = _require_numpyro()
    numpyro_model = build_numpyro_switching_model_jax(
        model,
        observations,
        priors,
        gate_probs=gate_probs,
        hard_mask=hard_mask,
        fom_algorithm=fom_algorithm,
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
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        switching_config=switching_config,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)


def evaluate_numpyro_switching_filter_log_density_jax(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    regime_switch_config: RegimeSwitchConfig,
    state_names: Optional[Sequence[str] | str] = None,
    periods: Optional[int] = None,
    gate_filter: str = "kalman",
    gate_algorithm: str = "first_order",
    gate_smooth: bool = False,
    fom_algorithm: str = "first_order",
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
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> jax.Array:
    _, _, log_density = _require_numpyro()
    numpyro_model = build_numpyro_switching_filter_model_jax(
        model,
        observations,
        priors,
        obs_sigma,
        shock_sigmas,
        regime_switch_config=regime_switch_config,
        state_names=state_names,
        periods=periods,
        gate_filter=gate_filter,
        gate_algorithm=gate_algorithm,
        gate_smooth=gate_smooth,
        fom_algorithm=fom_algorithm,
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
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        switching_config=switching_config,
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
    qme_algorithm: str = "schur",
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
        qme_algorithm=qme_algorithm,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)
