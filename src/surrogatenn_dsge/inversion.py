from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, NamedTuple, Optional, Sequence

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from .dsge import DSGETimings
from .sep import SEPConfig


class _FirstOrderInversionSetup(NamedTuple):
    solution_matrix: jax.Array
    observable_matrix: jax.Array
    shock_jacobian: jax.Array
    inverse_jacobian: jax.Array
    logabsdet_jacobian: jax.Array
    jacobian_valid: jax.Array
    state_indices: tuple[int, ...]
    n_observables: int
    n_exogenous: int


class _SEPInversionPredictResult(NamedTuple):
    ok: bool
    obs_dev: np.ndarray
    state_next_dev: np.ndarray
    solution_guess: Optional[np.ndarray]
    carry_guess: Optional[np.ndarray]
    sep_flag: int
    sep_err: float


_SEP_INVERSION_LAST_DIAGNOSTICS: Optional[dict[str, Any]] = None


def reset_sep_inversion_last_diagnostics() -> None:
    global _SEP_INVERSION_LAST_DIAGNOSTICS
    _SEP_INVERSION_LAST_DIAGNOSTICS = None


def get_sep_inversion_last_diagnostics() -> Optional[dict[str, Any]]:
    if _SEP_INVERSION_LAST_DIAGNOSTICS is None:
        return None
    return deepcopy(_SEP_INVERSION_LAST_DIAGNOSTICS)


def _set_sep_inversion_last_diagnostics(diagnostics: dict[str, Any]) -> None:
    global _SEP_INVERSION_LAST_DIAGNOSTICS
    _SEP_INVERSION_LAST_DIAGNOSTICS = deepcopy(diagnostics)


def _pseudo_logabsdet_jax(matrix: jax.Array) -> tuple[jax.Array, jax.Array]:
    array = jnp.asarray(matrix, dtype=jnp.float64)
    if array.shape[0] == array.shape[1]:
        sign, logabsdet = jnp.linalg.slogdet(array)
        valid = jnp.isfinite(logabsdet) & (sign != 0)
        return logabsdet, valid

    singular_values = jnp.linalg.svd(array, compute_uv=False)
    tolerance = jnp.sqrt(jnp.finfo(array.dtype).eps)
    positive = singular_values > tolerance
    logabsdet = jnp.sum(jnp.log(jnp.where(positive, singular_values, 1.0)))
    valid = jnp.isfinite(logabsdet) & jnp.any(positive) & jnp.all(
        jnp.isfinite(singular_values)
    )
    return logabsdet, valid


def _pseudo_logabsdet_numpy(matrix: np.ndarray) -> float:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape[0] == array.shape[1]:
        sign, logabsdet = np.linalg.slogdet(array)
        if sign == 0 or not np.isfinite(logabsdet):
            return float("nan")
        return float(logabsdet)

    try:
        singular_values = np.linalg.svd(array, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    tolerance = np.sqrt(np.finfo(np.float64).eps)
    singular_values = singular_values[singular_values > tolerance]
    if singular_values.size == 0:
        return float("nan")
    return float(np.sum(np.log(singular_values)))


def _coerce_observations(
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
) -> jax.Array:
    array = jnp.asarray(observations, dtype=jnp.float64)
    if array.ndim != 2:
        raise ValueError(f"observations must be rank-2, got shape {array.shape}.")
    return array


def _prepare_first_order_inversion_setup(
    solution_matrix: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    timings: DSGETimings,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
) -> tuple[_FirstOrderInversionSetup, jax.Array, jax.Array]:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    observation_array = _coerce_observations(observations)
    indices = tuple(int(index) for index in observable_indices)
    if not indices:
        raise ValueError("observable_indices must contain at least one index.")

    expected_shape = (
        timings.nVars,
        timings.nPast_not_future_and_mixed + timings.nExo,
    )
    if solution.shape != expected_shape:
        raise ValueError(
            "solution_matrix must have shape "
            f"{expected_shape}, got {solution.shape}."
        )
    if observation_array.shape[0] != len(indices):
        raise ValueError(
            "observations must have one row per observable, got "
            f"{observation_array.shape[0]} rows for {len(indices)} observables."
        )

    if initial_state is None:
        state = jnp.zeros((timings.nVars,), dtype=solution.dtype)
    else:
        state = jnp.asarray(initial_state, dtype=solution.dtype)
        if state.shape != (timings.nVars,):
            raise ValueError(
                f"initial_state must have shape ({timings.nVars},), got {state.shape}."
            )

    state_indices = tuple(int(index) for index in timings.past_not_future_and_mixed_idx)
    shock_jacobian = solution[list(indices), timings.nPast_not_future_and_mixed :]
    if timings.nExo == len(indices):
        inverse_jacobian = jnp.linalg.inv(shock_jacobian)
    else:
        inverse_jacobian = jnp.linalg.pinv(shock_jacobian)
    logabsdet_jacobian, logabsdet_valid = _pseudo_logabsdet_jax(shock_jacobian)
    jacobian_valid = (
        logabsdet_valid
        & jnp.all(jnp.isfinite(shock_jacobian))
        & jnp.all(jnp.isfinite(inverse_jacobian))
    )

    return (
        _FirstOrderInversionSetup(
            solution_matrix=solution,
            observable_matrix=solution[list(indices), : timings.nPast_not_future_and_mixed],
            shock_jacobian=shock_jacobian,
            inverse_jacobian=inverse_jacobian,
            logabsdet_jacobian=logabsdet_jacobian,
            jacobian_valid=jacobian_valid,
            state_indices=state_indices,
            n_observables=len(indices),
            n_exogenous=timings.nExo,
        ),
        observation_array,
        state,
    )


def _warmup_first_order_state(
    setup: _FirstOrderInversionSetup,
    observations: jax.Array,
    initial_state: jax.Array,
    *,
    warmup_iterations: int,
) -> tuple[jax.Array, jax.Array, bool]:
    if warmup_iterations <= 0:
        return initial_state, jnp.asarray(0.0, dtype=jnp.float64), True
    if observations.shape[1] == 0:
        return initial_state, jnp.asarray(0.0, dtype=jnp.float64), True

    solution = np.asarray(setup.solution_matrix, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    observation_array = np.asarray(observations, dtype=np.float64)
    n_past = len(setup.state_indices)
    blocks = [np.asarray(setup.shock_jacobian, dtype=np.float64)]

    if warmup_iterations >= 2:
        obs_matrix = np.asarray(setup.observable_matrix, dtype=np.float64)
        shock_to_state = solution[list(setup.state_indices), n_past:]
        blocks.insert(0, obs_matrix @ shock_to_state)
        if warmup_iterations >= 3:
            transition_reduced = solution[list(setup.state_indices), :n_past]
            transition_power = transition_reduced.copy()
            for _ in range(warmup_iterations - 2):
                blocks.insert(0, obs_matrix @ transition_power @ shock_to_state)
                transition_power = transition_power @ transition_reduced

    warmup_jacobian = np.hstack(blocks)
    try:
        warmup_solution, *_ = np.linalg.lstsq(
            warmup_jacobian,
            observation_array[:, 0],
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return initial_state, jnp.asarray(0.0, dtype=jnp.float64), False

    if not np.isfinite(warmup_solution).all():
        return initial_state, jnp.asarray(0.0, dtype=jnp.float64), False

    warmup_shocks = np.reshape(
        warmup_solution,
        (setup.n_exogenous, warmup_iterations),
        order="F",
    )
    log_two_pi = float(np.log(2.0 * np.pi))
    warmup_total = 0.0
    for block_index, block in enumerate(blocks):
        logabsdet = _pseudo_logabsdet_numpy(block)
        if not np.isfinite(logabsdet):
            return initial_state, jnp.asarray(0.0, dtype=jnp.float64), False
        shocks_sq = float(np.sum(warmup_shocks[:, block_index] ** 2))
        warmup_total += -0.5 * (
            logabsdet + shocks_sq + setup.n_observables * log_two_pi
        )

    for block_index in range(max(0, warmup_iterations - 1)):
        reduced_state = state[list(setup.state_indices)]
        state = solution @ np.concatenate([reduced_state, warmup_shocks[:, block_index]])
        if not np.isfinite(state).all():
            return initial_state, jnp.asarray(0.0, dtype=jnp.float64), False

    return jnp.asarray(state, dtype=jnp.float64), jnp.asarray(
        warmup_total,
        dtype=jnp.float64,
    ), True


def _first_order_inversion_loglikelihood_per_period_impl(
    setup: _FirstOrderInversionSetup,
    observations: jax.Array,
    initial_state: jax.Array,
    *,
    presample_periods: int,
    on_failure_loglikelihood: float,
) -> tuple[jax.Array, jax.Array]:
    state_indices = jnp.asarray(setup.state_indices, dtype=jnp.int32)
    failure_value = jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
    log_two_pi = jnp.asarray(np.log(2.0 * np.pi), dtype=jnp.float64)
    shock_constant = jnp.asarray(setup.n_observables, dtype=jnp.float64) * log_two_pi
    time_index = jnp.arange(observations.shape[1], dtype=jnp.int32)

    def step(
        carry: tuple[jax.Array, jax.Array],
        xs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        state, valid = carry
        t_idx, observation_t = xs

        def _active(
            active_carry: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            active_state, active_valid = active_carry
            reduced_state = active_state[state_indices]
            residual = observation_t - setup.observable_matrix @ reduced_state
            shocks = setup.inverse_jacobian @ residual
            next_state = setup.solution_matrix @ jnp.concatenate(
                [reduced_state, shocks],
                axis=0,
            )
            finite = (
                jnp.all(jnp.isfinite(residual))
                & jnp.all(jnp.isfinite(shocks))
                & jnp.all(jnp.isfinite(next_state))
            )
            ll_t = jnp.where(
                t_idx >= presample_periods,
                -0.5
                * (
                    setup.logabsdet_jacobian
                    + jnp.sum(shocks**2)
                    + shock_constant
                ),
                0.0,
            )
            safe_state = jnp.where(finite, next_state, active_state)
            return (safe_state, active_valid & finite), ll_t

        return lax.cond(
            valid,
            _active,
            lambda inactive_carry: (inactive_carry, jnp.asarray(0.0, dtype=jnp.float64)),
            (state, valid),
        )

    initial_carry = (
        initial_state,
        jnp.asarray(setup.jacobian_valid, dtype=jnp.bool_),
    )
    (_, valid), ll_per_period = lax.scan(step, initial_carry, (time_index, observations.T))
    failed = jnp.full(
        (observations.shape[1],),
        failure_value,
        dtype=jnp.float64,
    )
    return jnp.where(valid, ll_per_period, failed), valid


def first_order_inversion_loglikelihood(
    solution_matrix: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    timings: DSGETimings,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    warmup_iterations: int = 0,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    setup, observation_array, state = _prepare_first_order_inversion_setup(
        solution_matrix,
        timings,
        observations,
        observable_indices,
        initial_state=initial_state,
    )
    state, warmup_total, warmup_ok = _warmup_first_order_state(
        setup,
        observation_array,
        state,
        warmup_iterations=warmup_iterations,
    )
    if not warmup_ok:
        return jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)

    ll_per_period, valid = _first_order_inversion_loglikelihood_per_period_impl(
        setup,
        observation_array,
        state,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    total = warmup_total + jnp.sum(ll_per_period)
    return jnp.where(
        valid,
        total,
        jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64),
    )


def first_order_inversion_loglikelihood_per_period(
    solution_matrix: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    timings: DSGETimings,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    warmup_iterations: int = 0,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    _ = warmup_iterations
    setup, observation_array, state = _prepare_first_order_inversion_setup(
        solution_matrix,
        timings,
        observations,
        observable_indices,
        initial_state=initial_state,
    )
    ll_per_period, _ = _first_order_inversion_loglikelihood_per_period_impl(
        setup,
        observation_array,
        state,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    return ll_per_period


def _sep_inversion_shock_sigmas(model: Any, parameter_values: np.ndarray) -> np.ndarray:
    sigmas = np.zeros((model.timings.nExo,), dtype=np.float64)
    parameter_lookup = {
        name: float(parameter_values[idx])
        for idx, name in enumerate(model.parameter_names)
    }
    for shock_index, shock_name in enumerate(model.timings.exo):
        if "ᵒᵇᶜ" in str(shock_name):
            sigmas[shock_index] = 0.0
            continue
        parameter_name = f"z_{shock_name}"
        sigmas[shock_index] = abs(parameter_lookup.get(parameter_name, 1.0))
    return sigmas


def _coerce_sep_runtime_config(
    config: SEPConfig,
    *,
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_shock_scale: Optional[float] = None,
) -> SEPConfig:
    updated = config
    if sep_periods is not None:
        updated = replace(updated, periods=int(sep_periods))
    if sep_order is not None:
        updated = replace(updated, branching_order=int(sep_order))
    if sep_nnodes is not None:
        updated = replace(updated, nnodes=int(sep_nnodes))
    if sep_sparse_tree is not None:
        updated = replace(updated, sparse_tree=bool(sep_sparse_tree))
    if sep_maxit is not None:
        updated = replace(updated, max_iter=int(sep_maxit))
    if sep_tol is not None:
        updated = replace(updated, tol=float(sep_tol))
    if sep_shock_scale is not None:
        updated = replace(updated, shock_scale=float(sep_shock_scale))
    return updated


def _sep_predict_step(
    model: Any,
    state_dev: np.ndarray,
    shocks_full: np.ndarray,
    observable_indices: Sequence[int],
    *,
    parameter_values: np.ndarray,
    steady_state: np.ndarray,
    terminal_state: np.ndarray,
    config: SEPConfig,
    sep_accept_tol: float,
    initial_guess: Optional[np.ndarray],
) -> _SEPInversionPredictResult:
    deterministic_shocks = np.zeros((config.periods, model.timings.nExo), dtype=np.float64)
    deterministic_shocks[0, :] = shocks_full
    initial_state = steady_state + state_dev
    result = model.solve_stochastic_extended_path(
        parameter_values=parameter_values,
        steady_state=steady_state,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        deterministic_shocks=deterministic_shocks,
        initial_guess=initial_guess,
    )
    solution = result.solution
    sep_err = float(solution.residual_norm)
    sep_flag = 0 if solution.converged else 1
    ok = bool(solution.converged or (np.isfinite(sep_err) and sep_err <= sep_accept_tol))
    if not ok or not np.isfinite(sep_err):
        return _SEPInversionPredictResult(
            ok=False,
            obs_dev=np.zeros((len(observable_indices),), dtype=np.float64),
            state_next_dev=np.zeros_like(steady_state),
            solution_guess=None,
            carry_guess=None,
            sep_flag=sep_flag,
            sep_err=sep_err,
        )

    mean_path = np.asarray(solution.mean_path, dtype=np.float64)
    if mean_path.shape[1] < 2:
        return _SEPInversionPredictResult(
            ok=False,
            obs_dev=np.zeros((len(observable_indices),), dtype=np.float64),
            state_next_dev=np.zeros_like(steady_state),
            solution_guess=None,
            carry_guess=None,
            sep_flag=sep_flag,
            sep_err=sep_err,
        )

    next_state = mean_path[:, 1]
    next_state_dev = next_state - steady_state
    obs_dev = next_state_dev[list(observable_indices)]
    stacked_states = np.asarray(solution.stacked_states, dtype=np.float64)
    carry_guess = _build_sep_period_warm_start(
        stacked_states,
        np.asarray(solution.mean_path, dtype=np.float64),
        tuple(int(count) for count in solution.group_counts),
        terminal_state,
    )
    return _SEPInversionPredictResult(
        ok=bool(np.isfinite(obs_dev).all() and np.isfinite(next_state_dev).all()),
        obs_dev=obs_dev,
        state_next_dev=next_state_dev,
        solution_guess=stacked_states,
        carry_guess=carry_guess,
        sep_flag=sep_flag,
        sep_err=sep_err,
    )


def _sep_fd_jacobian(
    model: Any,
    state_dev: np.ndarray,
    eps_struct: np.ndarray,
    structural_indices: np.ndarray,
    observable_indices: Sequence[int],
    *,
    parameter_values: np.ndarray,
    steady_state: np.ndarray,
    terminal_state: np.ndarray,
    config: SEPConfig,
    shock_sigmas: np.ndarray,
    sep_accept_tol: float,
    initial_guess: Optional[np.ndarray],
) -> tuple[bool, np.ndarray, _SEPInversionPredictResult, int]:
    n_observables = len(observable_indices)
    n_structural = len(structural_indices)
    jacobian = np.zeros((n_observables, n_structural), dtype=np.float64)
    predict_calls = 1
    base_full = np.zeros((model.timings.nExo,), dtype=np.float64)
    base_full[structural_indices] = eps_struct
    base_eval = _sep_predict_step(
        model,
        state_dev,
        base_full,
        observable_indices,
        parameter_values=parameter_values,
        steady_state=steady_state,
        terminal_state=terminal_state,
        config=config,
        sep_accept_tol=sep_accept_tol,
        initial_guess=initial_guess,
    )
    if not base_eval.ok:
        return False, jacobian, base_eval, predict_calls

    for column_index in range(n_structural):
        step_scale = max(
            1.0,
            abs(float(eps_struct[column_index])),
            float(shock_sigmas[structural_indices[column_index]]),
        )
        step_size = np.cbrt(np.finfo(np.float64).eps) * step_scale
        step_size = max(step_size, 1e-6)
        perturbed = np.asarray(eps_struct, dtype=np.float64).copy()
        perturbed[column_index] += step_size
        full_shocks = np.zeros((model.timings.nExo,), dtype=np.float64)
        full_shocks[structural_indices] = perturbed
        predict_calls += 1
        perturbed_eval = _sep_predict_step(
            model,
            state_dev,
            full_shocks,
            observable_indices,
            parameter_values=parameter_values,
            steady_state=steady_state,
            terminal_state=terminal_state,
            config=config,
            sep_accept_tol=sep_accept_tol,
            initial_guess=base_eval.solution_guess,
        )
        if not perturbed_eval.ok:
            return False, jacobian, base_eval, predict_calls
        jacobian[:, column_index] = (
            perturbed_eval.obs_dev - base_eval.obs_dev
        ) / step_size

    return True, jacobian, base_eval, predict_calls


def _build_sep_period_warm_start(
    stacked_states: np.ndarray,
    mean_path: np.ndarray,
    group_counts: Sequence[int],
    terminal_state: np.ndarray,
) -> Optional[np.ndarray]:
    stacked_array = np.asarray(stacked_states, dtype=np.float64).reshape(-1)
    mean_path_array = np.asarray(mean_path, dtype=np.float64)
    terminal_state_array = np.asarray(terminal_state, dtype=np.float64).reshape(-1)
    counts = tuple(int(count) for count in group_counts)
    if len(counts) < 2:
        return None
    state_dim = int(terminal_state_array.size)
    if mean_path_array.shape[0] != state_dim:
        return None
    offsets = [0]
    for time_index in range(1, len(counts)):
        offsets.append(offsets[-1] + counts[time_index] * state_dim)
    if stacked_array.shape != (offsets[-1],):
        return None

    states_by_time = []
    for time_index in range(1, len(counts)):
        start = offsets[time_index - 1]
        end = offsets[time_index]
        states_by_time.append(
            stacked_array[start:end].reshape((counts[time_index], state_dim))
        )

    periods = len(counts) - 1
    shifted_blocks = []
    for time_index in range(1, periods + 1):
        target_count = counts[time_index]
        if time_index < periods:
            source_states = states_by_time[time_index]
            if mean_path_array.shape[1] > time_index + 1:
                source_mean = mean_path_array[:, time_index + 1]
            else:
                source_mean = np.mean(source_states, axis=0)
            if source_states.shape[0] == target_count:
                block = source_states
            elif source_states.shape[0] == 1:
                block = np.repeat(source_states, target_count, axis=0)
            elif target_count == 1:
                block = np.asarray(source_mean, dtype=np.float64)[None, :]
            else:
                block = np.repeat(
                    np.asarray(source_mean, dtype=np.float64)[None, :],
                    target_count,
                    axis=0,
                )
        else:
            block = np.repeat(terminal_state_array[None, :], target_count, axis=0)
        shifted_blocks.append(block.reshape(-1))

    if not shifted_blocks:
        return None
    shifted = np.concatenate(shifted_blocks, axis=0)
    if not np.isfinite(shifted).all():
        return None
    return shifted


def _run_sep_inversion_filter(
    model: Any,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    parameter_values: Sequence[float] | jax.Array | np.ndarray,
    steady_state: Sequence[float] | jax.Array | np.ndarray,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    terminal_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> tuple[bool, float, np.ndarray]:
    reset_sep_inversion_last_diagnostics()
    observation_array = np.asarray(observations, dtype=np.float64)
    if observation_array.ndim != 2:
        raise ValueError(
            f"observations must be rank-2, got shape {observation_array.shape}."
        )
    if observation_array.shape[0] != len(observable_indices):
        raise ValueError(
            "observations must have one row per observable, got "
            f"{observation_array.shape[0]} rows for {len(observable_indices)} observables."
        )

    parameter_array = np.asarray(parameter_values, dtype=np.float64)
    steady_state_array = np.asarray(steady_state, dtype=np.float64)
    if steady_state_array.shape != (model.timings.nVars,):
        raise ValueError(
            f"steady_state must have shape ({model.timings.nVars},), got {steady_state_array.shape}."
        )
    if initial_state is None:
        state_dev = np.zeros_like(steady_state_array)
    else:
        initial_state_array = np.asarray(initial_state, dtype=np.float64)
        if initial_state_array.shape != steady_state_array.shape:
            raise ValueError(
                "initial_state must have the same shape as steady_state, got "
                f"{initial_state_array.shape} and {steady_state_array.shape}."
            )
        state_dev = initial_state_array - steady_state_array
    if terminal_state is None:
        terminal_state_array = steady_state_array.copy()
    else:
        terminal_state_array = np.asarray(terminal_state, dtype=np.float64)
        if terminal_state_array.shape != steady_state_array.shape:
            raise ValueError(
                "terminal_state must have the same shape as steady_state, got "
                f"{terminal_state_array.shape} and {steady_state_array.shape}."
            )

    runtime_config = _coerce_sep_runtime_config(
        config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_shock_scale=sep_shock_scale,
    )
    if runtime_config.periods < 1:
        raise ValueError("SEP inversion requires config.periods >= 1.")

    n_periods = int(observation_array.shape[1])
    n_observables = len(observable_indices)
    if n_periods == 0:
        _set_sep_inversion_last_diagnostics(
            {
                "kind": "sep_inversion_filter",
                "status": "ok",
                "reason": "empty_sample",
                "n_periods": 0,
                "presample_periods": int(presample_periods),
                "ll_total": 0.0,
            }
        )
        return True, 0.0, np.zeros((0,), dtype=np.float64)

    shock_sigmas = _sep_inversion_shock_sigmas(model, parameter_array)
    structural_indices = np.flatnonzero(shock_sigmas > 0.0)
    n_structural = int(structural_indices.size)
    eps_struct = np.zeros((n_structural,), dtype=np.float64)
    per_period = np.zeros((n_periods,), dtype=np.float64)
    ll_total = 0.0
    log_two_pi = float(np.log(2.0 * np.pi))
    period_predict_calls = np.zeros((n_periods,), dtype=np.int64)
    period_carry_warm_start_used = [False] * n_periods
    total_predict_calls = 0

    def fail(
        code: str,
        *,
        period_index: int = 0,
        iteration: int = 0,
        pred: Optional[_SEPInversionPredictResult] = None,
        residual: Optional[np.ndarray] = None,
        step: Optional[np.ndarray] = None,
        jacobian: Optional[np.ndarray] = None,
        logabsdet: Optional[float] = None,
        shocks2: Optional[float] = None,
        message: Optional[str] = None,
    ) -> tuple[bool, float, np.ndarray]:
        diagnostics: dict[str, Any] = {
            "kind": "sep_inversion_filter",
            "status": "failure",
            "failure_code": code,
            "period_index": int(period_index),
            "iteration": int(iteration),
            "n_periods": n_periods,
            "presample_periods": int(presample_periods),
            "n_obs": n_observables,
            "n_struct": n_structural,
            "sep_periods": runtime_config.periods,
            "sep_order": runtime_config.branching_order,
            "sep_nnodes": runtime_config.nnodes,
            "sep_sparse_tree": bool(runtime_config.sparse_tree),
            "sep_maxit": runtime_config.max_iter,
            "sep_tol": runtime_config.tol,
            "sep_accept_tol": float(sep_accept_tol),
            "sep_shock_scale": runtime_config.shock_scale,
            "sep_inv_maxit": int(sep_inv_maxit),
            "sep_inv_step_tol": float(sep_inv_step_tol),
            "sep_inv_resid_tol": float(sep_inv_resid_tol),
            "sep_inv_lambda": float(sep_inv_lambda),
            "sep_carry_warm_start_strategy": "shifted_tree",
            "sep_period_carry_warm_start_used": list(period_carry_warm_start_used),
            "sep_period_predict_calls": period_predict_calls.tolist(),
            "sep_total_predict_calls": int(total_predict_calls),
        }
        if message is not None:
            diagnostics["message"] = message
        if pred is not None:
            diagnostics["sep_flag"] = int(pred.sep_flag)
            diagnostics["sep_err"] = float(pred.sep_err)
        if residual is not None:
            diagnostics["resid_norm"] = float(np.linalg.norm(residual))
            diagnostics["resid_finite"] = bool(np.isfinite(residual).all())
        if step is not None:
            diagnostics["step_norm"] = float(np.linalg.norm(step))
            diagnostics["step_finite"] = bool(np.isfinite(step).all())
        if jacobian is not None:
            diagnostics["jacobian_finite"] = bool(np.isfinite(jacobian).all())
            diagnostics["jacobian_size"] = list(jacobian.shape)
        if logabsdet is not None:
            diagnostics["logabsdet"] = float(logabsdet)
        if shocks2 is not None:
            diagnostics["shocks2"] = float(shocks2)
        _set_sep_inversion_last_diagnostics(diagnostics)
        return (
            False,
            float(on_failure_loglikelihood),
            np.full((n_periods,), on_failure_loglikelihood, dtype=np.float64),
        )

    carry_guess = None
    for period_index in range(n_periods):
        observation_t = observation_array[:, period_index]
        base_guess = carry_guess
        period_carry_warm_start_used[period_index] = base_guess is not None
        final_eval: Optional[_SEPInversionPredictResult] = None
        final_jacobian: Optional[np.ndarray] = None

        if n_structural == 0:
            period_predict_calls[period_index] += 1
            total_predict_calls += 1
            pred = _sep_predict_step(
                model,
                state_dev,
                np.zeros((model.timings.nExo,), dtype=np.float64),
                observable_indices,
                parameter_values=parameter_array,
                steady_state=steady_state_array,
                terminal_state=terminal_state_array,
                config=runtime_config,
                sep_accept_tol=sep_accept_tol,
                initial_guess=base_guess,
            )
            if not pred.ok:
                return fail(
                    "predict_no_struct",
                    period_index=period_index + 1,
                    pred=pred,
                    message="SEP predict step failed with no structural shocks.",
                )
            residual = observation_t - pred.obs_dev
            if period_index >= presample_periods:
                per_period[period_index] = -0.5 * (
                    float(np.sum(residual**2)) + n_observables * log_two_pi
                )
            state_dev = pred.state_next_dev
            carry_guess = pred.carry_guess
            continue

        last_iteration = 0
        for iteration in range(1, sep_inv_maxit + 1):
            last_iteration = iteration
            ok, jacobian, base_eval, predict_calls = _sep_fd_jacobian(
                model,
                state_dev,
                eps_struct,
                structural_indices,
                observable_indices,
                parameter_values=parameter_array,
                steady_state=steady_state_array,
                terminal_state=terminal_state_array,
                config=runtime_config,
                shock_sigmas=shock_sigmas,
                sep_accept_tol=sep_accept_tol,
                initial_guess=base_guess,
            )
            period_predict_calls[period_index] += predict_calls
            total_predict_calls += predict_calls
            if not ok:
                return fail(
                    "fd_jacobian_failed",
                    period_index=period_index + 1,
                    iteration=iteration,
                    pred=base_eval,
                    message="Finite-difference Jacobian construction failed.",
                )

            residual = observation_t - base_eval.obs_dev
            final_eval = base_eval
            final_jacobian = jacobian
            base_guess = base_eval.solution_guess

            if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
                return fail(
                    "nonfinite_resid_or_jacobian",
                    period_index=period_index + 1,
                    iteration=iteration,
                    pred=base_eval,
                    residual=residual,
                    jacobian=jacobian,
                    message="Non-finite residual or Jacobian.",
                )
            if np.linalg.norm(residual) <= sep_inv_resid_tol:
                break

            lhs = jacobian.T @ jacobian
            rhs = jacobian.T @ residual
            lhs = lhs + sep_inv_lambda * np.eye(n_structural, dtype=np.float64)
            try:
                step = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError as lhs_error:
                try:
                    step = np.linalg.pinv(jacobian) @ residual
                except np.linalg.LinAlgError as pinv_error:
                    return fail(
                        "linear_solve_failed",
                        period_index=period_index + 1,
                        iteration=iteration,
                        pred=base_eval,
                        residual=residual,
                        jacobian=jacobian,
                        message=(
                            "Failed solving LM step. "
                            f"solve: {lhs_error}; pinv fallback: {pinv_error}"
                        ),
                    )
            if not np.isfinite(step).all():
                return fail(
                    "nonfinite_step",
                    period_index=period_index + 1,
                    iteration=iteration,
                    pred=base_eval,
                    residual=residual,
                    jacobian=jacobian,
                    step=step,
                    message="LM step contains non-finite values.",
                )
            eps_struct = eps_struct + step
            if np.linalg.norm(step) <= sep_inv_step_tol * (1.0 + np.linalg.norm(eps_struct)):
                break

        if final_eval is None or final_jacobian is None:
            return fail(
                "missing_final_eval",
                period_index=period_index + 1,
                iteration=last_iteration,
                message="No final prediction or Jacobian was produced.",
            )

        residual = observation_t - final_eval.obs_dev
        if not np.isfinite(residual).all():
            return fail(
                "nonfinite_residual_postsolve",
                period_index=period_index + 1,
                iteration=last_iteration,
                pred=final_eval,
                residual=residual,
                jacobian=final_jacobian,
                message="Residual became non-finite after inversion iterations.",
            )

        if period_index >= presample_periods:
            standardized_jacobian = final_jacobian * shock_sigmas[structural_indices][None, :]
            logabsdet = _pseudo_logabsdet_numpy(standardized_jacobian)
            if not np.isfinite(logabsdet):
                return fail(
                    "invalid_logabsdet",
                    period_index=period_index + 1,
                    iteration=last_iteration,
                    pred=final_eval,
                    residual=residual,
                    jacobian=final_jacobian,
                    logabsdet=logabsdet,
                    message="Jacobian logabsdet is non-finite.",
                )
            standardized_shocks = eps_struct / shock_sigmas[structural_indices]
            shocks2 = float(np.sum(standardized_shocks**2))
            if not np.isfinite(shocks2):
                return fail(
                    "invalid_shocks2",
                    period_index=period_index + 1,
                    iteration=last_iteration,
                    pred=final_eval,
                    residual=residual,
                    jacobian=final_jacobian,
                    shocks2=shocks2,
                    message="Standardized shock norm is non-finite.",
                )
            per_period[period_index] = -0.5 * (
                logabsdet + shocks2 + n_observables * log_two_pi
            )

        state_dev = final_eval.state_next_dev
        carry_guess = final_eval.carry_guess
        if not np.isfinite(state_dev).all():
            return fail(
                "nonfinite_state_update",
                period_index=period_index + 1,
                iteration=last_iteration,
                pred=final_eval,
                residual=residual,
                jacobian=final_jacobian,
                message="State update produced non-finite values.",
            )

    ll_total = float(np.sum(per_period))
    _set_sep_inversion_last_diagnostics(
        {
            "kind": "sep_inversion_filter",
            "status": "ok",
            "n_periods": n_periods,
            "presample_periods": int(presample_periods),
            "n_obs": n_observables,
            "n_struct": n_structural,
            "sep_periods": runtime_config.periods,
            "sep_order": runtime_config.branching_order,
            "sep_nnodes": runtime_config.nnodes,
            "sep_sparse_tree": bool(runtime_config.sparse_tree),
            "sep_maxit": runtime_config.max_iter,
            "sep_tol": runtime_config.tol,
            "sep_accept_tol": float(sep_accept_tol),
            "sep_shock_scale": runtime_config.shock_scale,
            "sep_inv_maxit": int(sep_inv_maxit),
            "sep_inv_step_tol": float(sep_inv_step_tol),
            "sep_inv_resid_tol": float(sep_inv_resid_tol),
            "sep_inv_lambda": float(sep_inv_lambda),
            "sep_carry_warm_start_strategy": "shifted_tree",
            "sep_period_carry_warm_start_used": list(period_carry_warm_start_used),
            "sep_period_predict_calls": period_predict_calls.tolist(),
            "sep_total_predict_calls": int(total_predict_calls),
            "ll_total": ll_total,
            "final_state_finite": bool(np.isfinite(state_dev).all()),
        }
    )
    return True, ll_total, per_period


def sep_inversion_loglikelihood(
    model: Any,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    parameter_values: Sequence[float] | jax.Array | np.ndarray,
    steady_state: Sequence[float] | jax.Array | np.ndarray,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    terminal_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    success, total, _ = _run_sep_inversion_filter(
        model,
        observations,
        observable_indices,
        parameter_values=parameter_values,
        steady_state=steady_state,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    if not success:
        return jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
    return jnp.asarray(total, dtype=jnp.float64)


def sep_inversion_loglikelihood_per_period(
    model: Any,
    observations: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    observable_indices: Sequence[int],
    *,
    parameter_values: Sequence[float] | jax.Array | np.ndarray,
    steady_state: Sequence[float] | jax.Array | np.ndarray,
    initial_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    terminal_state: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    _, _, per_period = _run_sep_inversion_filter(
        model,
        observations,
        observable_indices,
        parameter_values=parameter_values,
        steady_state=steady_state,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    return jnp.asarray(per_period, dtype=jnp.float64)
