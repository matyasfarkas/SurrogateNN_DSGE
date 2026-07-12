from __future__ import annotations

import re
from time import time
from typing import Any, Callable, Mapping, Optional, Sequence
import warnings

import jax
import jax.numpy as jnp
import numpy as np


_EPSILON_SITE_RE = re.compile(r"^[εϵ]\[(\d+),(\d+)\]$")


def _as_float_vector(
    values: Sequence[float] | np.ndarray | jax.Array,
    *,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {array.shape}.")
    return array


def _as_float_matrix(
    values: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    *,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{label} must be rank-2, got shape {array.shape}.")
    return array


def _call_predict(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    output = predict_fn(state, shocks, theta)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError(
            "predict_fn must return a pair `(obs_pred, state_next)`."
        )
    obs_pred = _as_float_vector(output[0], label="predict_fn observation output")
    state_next = _as_float_vector(output[1], label="predict_fn state output")
    return obs_pred, state_next


def _call_predict_jax(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: jax.Array,
    shocks: jax.Array,
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    output = predict_fn(state, shocks, theta)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError(
            "predict_fn must return a pair `(obs_pred, state_next)`."
        )
    obs_pred = jnp.asarray(output[0], dtype=jnp.float64).reshape(-1)
    state_next = jnp.asarray(output[1], dtype=jnp.float64).reshape(-1)
    return obs_pred, state_next


def _call_full_predict(
    full_predict: Callable[[Any, Any, Any], Any],
    state: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    *,
    label: str,
) -> np.ndarray:
    output = full_predict(state, shocks, theta)
    return _as_float_vector(output, label=label)


def _resolve_named_indices(
    names: Sequence[str],
    available_names: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    if not names:
        return np.zeros((0,), dtype=np.int64)
    lookup = {str(name): idx for idx, name in enumerate(available_names)}
    indices: list[int] = []
    missing: list[str] = []
    for name in names:
        key = str(name)
        if key not in lookup:
            missing.append(key)
        else:
            indices.append(lookup[key])
    if missing:
        raise ValueError(f"{label} not found in model parameters.")
    return np.asarray(indices, dtype=np.int64)


def _normalize_julia_indices(
    indices: Sequence[int] | np.ndarray,
    *,
    upper: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return values
    if np.min(values) < 1:
        raise ValueError(f"{label} contains an index < 1.")
    if np.max(values) > upper:
        raise ValueError(
            f"{label} contains an index > number of model parameters ({upper})."
        )
    return values - 1


def extract_named_parameters(
    params: Sequence[float] | np.ndarray,
    model_parameter_names: Sequence[str],
    names: Sequence[str],
    *,
    label: str = "Parameter names",
) -> np.ndarray:
    params_array = _as_float_vector(params, label="params")
    resolved = _resolve_named_indices(names, model_parameter_names, label=label)
    return params_array[resolved]


def override_named_parameters(
    base_params: Sequence[float] | np.ndarray,
    model_parameter_names: Sequence[str],
    names: Sequence[str],
    values: Sequence[float] | np.ndarray,
    *,
    label: str = "Parameter names",
) -> np.ndarray:
    base = _as_float_vector(base_params, label="base_params").copy()
    updates = _as_float_vector(values, label="values")
    if updates.shape[0] != len(names):
        raise ValueError(
            f"{label}/value length mismatch: {len(names)} names vs {updates.shape[0]} values."
        )
    resolved = _resolve_named_indices(names, model_parameter_names, label=label)
    base[resolved] = updates
    return base


def parameters_with_theta_mode(
    base_params: Sequence[float] | np.ndarray,
    model_parameter_names: Sequence[str],
    theta_names: Sequence[str],
    theta_values: Optional[Sequence[float] | np.ndarray],
    *,
    theta_mode: str,
    mode_label: str = "theta_mode",
    theta_label: str = "Theta names",
) -> np.ndarray:
    mode = str(theta_mode)
    if mode == "baseline":
        return _as_float_vector(base_params, label="base_params").copy()
    if mode in {"synthetic", "current"}:
        if theta_values is None:
            raise ValueError(f"{mode_label}='{mode}' requires theta values.")
        return override_named_parameters(
            base_params,
            model_parameter_names,
            theta_names,
            theta_values,
            label=theta_label,
        )
    raise ValueError(
        f"Unknown {mode_label}={theta_mode!r}. Use 'baseline', 'current', or 'synthetic'."
    )


def _override_named_parameters_with_index(
    base_parameters: Sequence[float] | np.ndarray,
    theta_idx: Sequence[int] | np.ndarray,
    theta: Sequence[float] | np.ndarray,
    *,
    theta_label: str,
) -> np.ndarray:
    params = _as_float_vector(base_parameters, label="base_parameters").copy()
    values = _as_float_vector(theta, label="theta")
    resolved = _normalize_julia_indices(
        theta_idx,
        upper=params.shape[0],
        label=theta_label,
    )
    if values.shape[0] != resolved.shape[0]:
        raise ValueError(
            f"{theta_label}/value length mismatch: {resolved.shape[0]} indices vs {values.shape[0]} values."
        )
    params[resolved] = values
    return params


def split_observation_state(
    y_full: Sequence[float] | np.ndarray | jax.Array,
    d_obs: int,
    *,
    label: str = "predict output",
) -> tuple[np.ndarray, np.ndarray]:
    if int(d_obs) <= 0:
        raise ValueError(f"d_obs must be positive, got {d_obs}.")
    values = _as_float_vector(y_full, label=label)
    if values.shape[0] < int(d_obs):
        raise ValueError(
            f"{label} length mismatch: got {values.shape[0]}, expected at least d_obs={int(d_obs)}."
        )
    return values[: int(d_obs)], values[int(d_obs) :]


def predict_from_full(
    full_predict: Callable[[Any, Any, Any], Any],
    state: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    d_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_full = _call_full_predict(
        full_predict,
        state,
        shocks,
        theta,
        label="full_predict output",
    )
    return split_observation_state(y_full, d_obs, label="full_predict output")


def _combine_additive_residual(
    y_full: np.ndarray,
    y_resid: np.ndarray,
    d_obs: int,
    *,
    allow_full_residual: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if y_resid.shape[0] == int(d_obs):
        obs_base, state_next = split_observation_state(
            y_full,
            d_obs,
            label="full_predict output",
        )
        return obs_base + y_resid, state_next
    if allow_full_residual and y_resid.shape[0] == y_full.shape[0]:
        return split_observation_state(
            y_full + y_resid,
            d_obs,
            label="residual-augmented output",
        )
    allow_msg = f" or {y_full.shape[0]}" if allow_full_residual else ""
    raise ValueError(
        f"Residual output size mismatch: got {y_resid.shape[0]}, expected {int(d_obs)}{allow_msg}."
    )


def predict_additive_residual(
    full_predict: Callable[[Any, Any, Any], Any],
    residual_predict: Callable[[Any, Any, Any], Any],
    state: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    d_obs: int,
    *,
    allow_full_residual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y_full = _call_full_predict(
        full_predict,
        state,
        shocks,
        theta,
        label="full_predict output",
    )
    y_resid = _call_full_predict(
        residual_predict,
        state,
        shocks,
        theta,
        label="residual_predict output",
    )
    return _combine_additive_residual(
        y_full,
        y_resid,
        d_obs,
        allow_full_residual=allow_full_residual,
    )


def _norm_stats_value(norm_stats: Any, keys: tuple[str, ...], *, label: str) -> np.ndarray:
    if isinstance(norm_stats, Mapping):
        for key in keys:
            if key in norm_stats:
                return _as_float_vector(norm_stats[key], label=label)
    else:
        for key in keys:
            if hasattr(norm_stats, key):
                return _as_float_vector(getattr(norm_stats, key), label=label)
        if isinstance(norm_stats, (tuple, list)) and len(norm_stats) >= 2:
            if label.endswith("mean"):
                return _as_float_vector(norm_stats[0], label=label)
            return _as_float_vector(norm_stats[1], label=label)
    raise ValueError(
        f"norm_stats is missing {label}; expected one of {', '.join(keys)}."
    )


def _input_norm_stats(norm_stats: Any) -> tuple[np.ndarray, np.ndarray]:
    mu = _norm_stats_value(
        norm_stats,
        ("μX", "muX", "mu_x", "x_mean", "input_mean", "mean", "mu"),
        label="input mean",
    )
    sigma = _norm_stats_value(
        norm_stats,
        ("σX", "sigmaX", "sigma_x", "x_std", "input_std", "std", "sigma"),
        label="input std",
    )
    if mu.shape != sigma.shape:
        raise ValueError(f"norm_stats mean/std shape mismatch: {mu.shape} vs {sigma.shape}.")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("norm_stats contains non-finite values.")
    if not np.all(sigma > 0):
        raise ValueError("norm_stats input std must be strictly positive.")
    return mu, sigma


def predict_additive_residual_ood(
    full_predict: Callable[[Any, Any, Any], Any],
    residual_predict: Callable[[Any, Any, Any], Any],
    state: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    d_obs: int,
    norm_stats: Any,
    *,
    z_threshold: float = 4.0,
    allow_full_residual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(z_threshold)
    if threshold <= 0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}.")
    state_vec = _as_float_vector(state, label="state")
    shock_vec = _as_float_vector(shocks, label="shocks")
    theta_vec = _as_float_vector(theta, label="theta")
    x_input = np.concatenate([state_vec, shock_vec, theta_vec], axis=0)
    mu, sigma = _input_norm_stats(norm_stats)
    if x_input.shape[0] != mu.shape[0]:
        raise ValueError(
            f"norm_stats input length mismatch: got {mu.shape[0]}, expected {x_input.shape[0]}."
        )

    y_full = _call_full_predict(
        full_predict,
        state,
        shocks,
        theta,
        label="full_predict output",
    )
    if float(np.max(np.abs((x_input - mu) / sigma))) > threshold:
        return split_observation_state(
            y_full,
            d_obs,
            label="full_predict output (OOD fallback)",
        )

    y_resid = _call_full_predict(
        residual_predict,
        state,
        shocks,
        theta,
        label="residual_predict output",
    )
    return _combine_additive_residual(
        y_full,
        y_resid,
        d_obs,
        allow_full_residual=allow_full_residual,
    )


def conditional_loglik_per_period(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    s0: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    obs_data: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    obs_sigma: Sequence[float] | np.ndarray | jax.Array,
) -> np.ndarray:
    shock_matrix = _as_float_matrix(shocks, label="shocks")
    observations = _as_float_matrix(obs_data, label="obs_data")
    sigma = _as_float_vector(obs_sigma, label="obs_sigma")
    if observations.shape[1] != shock_matrix.shape[1]:
        raise ValueError(
            "obs_data/shocks period mismatch: "
            f"{observations.shape[1]} vs {shock_matrix.shape[1]}."
        )
    if observations.shape[0] != sigma.shape[0]:
        raise ValueError(
            "obs_sigma length mismatch: "
            f"{sigma.shape[0]} vs obs_data rows ({observations.shape[0]})."
        )
    if not np.isfinite(sigma).all():
        raise ValueError("obs_sigma contains non-finite values.")
    if not np.all(sigma > 0):
        raise ValueError("obs_sigma must be strictly positive.")

    state = _as_float_vector(s0, label="s0")
    theta_vec = _as_float_vector(theta, label="theta")
    log_norm = np.log(2.0 * np.pi * sigma**2)
    ll = np.zeros((shock_matrix.shape[1],), dtype=np.float64)
    for period in range(shock_matrix.shape[1]):
        obs_pred, state_next = _call_predict(
            predict_fn,
            state,
            shock_matrix[:, period],
            theta_vec,
        )
        if obs_pred.shape[0] != observations.shape[0]:
            raise ValueError(
                "predict_fn observation length mismatch at t="
                f"{period + 1}: got {obs_pred.shape[0]}, expected {observations.shape[0]}."
            )
        resid = observations[:, period] - obs_pred
        ll[period] = -0.5 * np.sum((resid / sigma) ** 2 + log_norm)
        state = state_next
    return ll


def additive_residual_loglik_per_period(
    full_predict: Callable[[Any, Any, Any], Any],
    residual_predict: Callable[[Any, Any, Any], Any],
    s0: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    obs_data: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    obs_sigma: Sequence[float] | np.ndarray | jax.Array,
    *,
    d_obs: Optional[int] = None,
    allow_full_residual: bool = True,
) -> np.ndarray:
    observations = _as_float_matrix(obs_data, label="obs_data")
    obs_dim = observations.shape[0] if d_obs is None else int(d_obs)

    def predict_fn(state: Any, shock_t: Any, theta_local: Any) -> tuple[np.ndarray, np.ndarray]:
        return predict_additive_residual(
            full_predict,
            residual_predict,
            state,
            shock_t,
            theta_local,
            obs_dim,
            allow_full_residual=allow_full_residual,
        )

    return conditional_loglik_per_period(
        predict_fn,
        s0,
        shocks,
        theta,
        observations,
        obs_sigma,
    )


def rollout_observations(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    s0: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    *,
    check_finite: bool = False,
) -> np.ndarray:
    shock_matrix = _as_float_matrix(shocks, label="shocks")
    state = _as_float_vector(s0, label="s0")
    theta_vec = _as_float_vector(theta, label="theta")
    periods = shock_matrix.shape[1]
    if periods == 0:
        return np.zeros((0, 0), dtype=np.float64)

    obs_pred_0, state_next = _call_predict(
        predict_fn,
        state,
        shock_matrix[:, 0],
        theta_vec,
    )
    obs_dim = obs_pred_0.shape[0]
    if obs_dim == 0:
        raise ValueError("predict_fn must return a non-empty observation vector.")
    observations = np.zeros((obs_dim, periods), dtype=np.float64)
    observations[:, 0] = obs_pred_0
    if check_finite and not np.isfinite(observations[:, 0]).all():
        raise ValueError("predict_fn produced non-finite observation values at t=1.")
    if check_finite and not np.isfinite(state_next).all():
        raise ValueError("predict_fn produced non-finite state values at t=1.")
    state = state_next

    for period in range(1, periods):
        obs_pred_t, state_next = _call_predict(
            predict_fn,
            state,
            shock_matrix[:, period],
            theta_vec,
        )
        if obs_pred_t.shape[0] != obs_dim:
            raise ValueError(
                "predict_fn observation length mismatch at t="
                f"{period + 1}: got {obs_pred_t.shape[0]}, expected {obs_dim}."
            )
        observations[:, period] = obs_pred_t
        if check_finite and not np.isfinite(observations[:, period]).all():
            raise ValueError(
                f"predict_fn produced non-finite observation values at t={period + 1}."
            )
        if check_finite and not np.isfinite(state_next).all():
            raise ValueError(
                f"predict_fn produced non-finite state values at t={period + 1}."
            )
        state = state_next

    return observations


def advance_state(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    s0: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    steps: int,
) -> np.ndarray:
    steps_int = int(steps)
    if steps_int < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}.")
    shock_matrix = _as_float_matrix(shocks, label="shocks")
    if steps_int > shock_matrix.shape[1]:
        raise ValueError(
            f"steps ({steps_int}) exceeds available shock periods ({shock_matrix.shape[1]})."
        )
    state = _as_float_vector(s0, label="s0")
    theta_vec = _as_float_vector(theta, label="theta")
    for period in range(steps_int):
        _, state = _call_predict(
            predict_fn,
            state,
            shock_matrix[:, period],
            theta_vec,
        )
    return state


def _jacobian_wrt_structural_shocks(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: np.ndarray,
    theta: np.ndarray,
    d_eps: int,
    structural_zero_based: np.ndarray,
    eps_struct: np.ndarray,
) -> np.ndarray:
    def obs_from_eps(eps_s: jax.Array) -> jax.Array:
        eps_full = jnp.zeros((d_eps,), dtype=jnp.float64)
        eps_full = eps_full.at[structural_zero_based].set(eps_s)
        obs_pred, _ = _call_predict_jax(
            predict_fn,
            jnp.asarray(state, dtype=jnp.float64),
            eps_full,
            jnp.asarray(theta, dtype=jnp.float64),
        )
        return obs_pred

    try:
        jacobian = np.asarray(
            jax.jacobian(obs_from_eps)(jnp.asarray(eps_struct, dtype=jnp.float64)),
            dtype=np.float64,
        )
        if np.isfinite(jacobian).all():
            return jacobian
    except Exception:
        pass

    fd_step = 1e-6
    base = np.asarray(eps_struct, dtype=np.float64)
    base_obs = np.asarray(obs_from_eps(jnp.asarray(base, dtype=jnp.float64)), dtype=np.float64)
    jacobian = np.zeros((base_obs.shape[0], base.shape[0]), dtype=np.float64)
    for col in range(base.shape[0]):
        bumped = base.copy()
        bumped[col] += fd_step
        obs_bumped = np.asarray(
            obs_from_eps(jnp.asarray(bumped, dtype=jnp.float64)),
            dtype=np.float64,
        )
        jacobian[:, col] = (obs_bumped - base_obs) / fd_step
    return jacobian


def _full_shock_vector(d_eps: int, structural_zero_based: np.ndarray, eps_struct: np.ndarray) -> np.ndarray:
    eps_full = np.zeros((d_eps,), dtype=np.float64)
    eps_full[structural_zero_based] = eps_struct
    return eps_full


def _clamp_residual_vector(
    values: Sequence[float] | np.ndarray | jax.Array,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array],
    *,
    label: str,
) -> np.ndarray:
    vector = _as_float_vector(values, label=label)
    if correction_clamp is None:
        return vector
    clamp = np.asarray(correction_clamp, dtype=np.float64).reshape(-1)
    if clamp.size == 1:
        bound = np.full_like(vector, float(clamp[0]))
    else:
        bound = _as_float_vector(correction_clamp, label="correction_clamp")
        if bound.shape[0] != vector.shape[0]:
            raise ValueError(
                "correction_clamp length mismatch: "
                f"{bound.shape[0]} vs residual output length {vector.shape[0]}."
            )
    if not np.isfinite(bound).all():
        raise ValueError("correction_clamp contains non-finite values.")
    if not np.all(bound >= 0):
        raise ValueError("correction_clamp must be nonnegative.")
    return np.minimum(np.maximum(vector, -bound), bound)


def _evaluate_inversion_model(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: np.ndarray,
    eps_full: np.ndarray,
    theta: np.ndarray,
    d_obs: int,
    *,
    period_idx: int,
    eval_predict_fn: Optional[Callable[[Any, Any, Any], tuple[Any, Any]]] = None,
    single_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    gate_mask: Optional[np.ndarray] = None,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if single_eval_residual_fn is not None:
        obs_rom, state_rom_next = _call_predict(predict_fn, state, eps_full, theta)
        x_nn = np.concatenate([state, eps_full, theta], axis=0)
        y_nn = _clamp_residual_vector(
            single_eval_residual_fn(x_nn),
            correction_clamp,
            label="single_eval_residual_fn output",
        )
        if y_nn.shape[0] < int(d_obs):
            raise ValueError(
                "single_eval_residual_fn output length mismatch: "
                f"got {y_nn.shape[0]}, expected at least d_obs={int(d_obs)}."
            )
        obs_pred = obs_rom + y_nn[: int(d_obs)]
        if gate_mask is not None and bool(gate_mask[period_idx]) and y_nn.shape[0] > int(d_obs):
            state_resid = y_nn[int(d_obs) :]
            if state_resid.shape[0] == 0:
                state_next = state_rom_next
            elif state_resid.shape[0] == state_rom_next.shape[0]:
                state_next = state_rom_next + state_resid
            else:
                raise ValueError(
                    "single_eval_residual_fn state residual length mismatch: "
                    f"{state_resid.shape[0]} vs state length {state_rom_next.shape[0]}."
                )
        else:
            state_next = state_rom_next
        return obs_pred, state_next
    if eval_predict_fn is not None:
        return _call_predict(eval_predict_fn, state, eps_full, theta)
    return _call_predict(predict_fn, state, eps_full, theta)


def _period_inversion_objective(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: np.ndarray,
    eps_full: np.ndarray,
    theta: np.ndarray,
    y_obs: np.ndarray,
    obs_sigma: np.ndarray,
    shock_std: np.ndarray,
    structural_zero_based: np.ndarray,
    obs_log_norm_const: float,
    shock_log_norm_const: float,
    d_obs: int,
    *,
    period_idx: int,
    eval_predict_fn: Optional[Callable[[Any, Any, Any], tuple[Any, Any]]] = None,
    single_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    gate_mask: Optional[np.ndarray] = None,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    obs_pred, state_next = _evaluate_inversion_model(
        predict_fn,
        state,
        eps_full,
        theta,
        d_obs,
        period_idx=period_idx,
        eval_predict_fn=eval_predict_fn,
        single_eval_residual_fn=single_eval_residual_fn,
        gate_mask=gate_mask,
        correction_clamp=correction_clamp,
    )
    if not np.isfinite(obs_pred).all() or not np.isfinite(state_next).all():
        return -np.inf, obs_pred, state_next
    resid = (y_obs - obs_pred) / obs_sigma
    ll_t = -0.5 * (float(np.sum(resid**2)) + obs_log_norm_const)
    if structural_zero_based.size:
        eps_struct = eps_full[structural_zero_based]
        ll_t += -0.5 * (
            float(np.sum((eps_struct / shock_std) ** 2)) + shock_log_norm_const
        )
    return ll_t, obs_pred, state_next


def _finite_difference_eval_jacobian(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: np.ndarray,
    theta: np.ndarray,
    d_eps: int,
    structural_zero_based: np.ndarray,
    eps_struct: np.ndarray,
    d_obs: int,
    *,
    period_idx: int,
    eval_predict_fn: Optional[Callable[[Any, Any, Any], tuple[Any, Any]]] = None,
    single_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    gate_mask: Optional[np.ndarray] = None,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
) -> np.ndarray:
    fd_step = 1e-6
    base = np.asarray(eps_struct, dtype=np.float64)
    base_full = _full_shock_vector(d_eps, structural_zero_based, base)
    base_obs, _ = _evaluate_inversion_model(
        predict_fn,
        state,
        base_full,
        theta,
        d_obs,
        period_idx=period_idx,
        eval_predict_fn=eval_predict_fn,
        single_eval_residual_fn=single_eval_residual_fn,
        gate_mask=gate_mask,
        correction_clamp=correction_clamp,
    )
    jacobian = np.zeros((base_obs.shape[0], base.shape[0]), dtype=np.float64)
    for col in range(base.shape[0]):
        bumped = base.copy()
        bumped[col] += fd_step
        obs_bumped, _ = _evaluate_inversion_model(
            predict_fn,
            state,
            _full_shock_vector(d_eps, structural_zero_based, bumped),
            theta,
            d_obs,
            period_idx=period_idx,
            eval_predict_fn=eval_predict_fn,
            single_eval_residual_fn=single_eval_residual_fn,
            gate_mask=gate_mask,
            correction_clamp=correction_clamp,
        )
        jacobian[:, col] = (obs_bumped - base_obs) / fd_step
    return jacobian


def _as_residual_matrix(values: Any, *, label: str, periods: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if periods != 1:
            raise ValueError(
                f"{label} must be rank-2 for {periods} periods, got shape {array.shape}."
            )
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be rank-2, got shape {array.shape}.")
    if array.shape[1] != periods:
        raise ValueError(
            f"{label} period mismatch: got {array.shape[1]}, expected {periods}."
        )
    return array


def _clamp_residual_matrix(
    values: Any,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array],
    *,
    label: str,
    periods: int,
) -> np.ndarray:
    matrix = _as_residual_matrix(values, label=label, periods=periods)
    if correction_clamp is None:
        return matrix
    clamp = np.asarray(correction_clamp, dtype=np.float64).reshape(-1)
    if clamp.size == 1:
        bound = np.full((matrix.shape[0], 1), float(clamp[0]), dtype=np.float64)
    else:
        if clamp.shape[0] != matrix.shape[0]:
            raise ValueError(
                "correction_clamp length mismatch: "
                f"{clamp.shape[0]} vs residual output rows {matrix.shape[0]}."
            )
        bound = clamp.reshape(-1, 1)
    if not np.isfinite(bound).all():
        raise ValueError("correction_clamp contains non-finite values.")
    if not np.all(bound >= 0):
        raise ValueError("correction_clamp must be nonnegative.")
    return np.minimum(np.maximum(matrix, -bound), bound)


def _batch_residual_inversion_loglik(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    s0: np.ndarray,
    theta: np.ndarray,
    observations: np.ndarray,
    obs_sigma: np.ndarray,
    shock_sigmas: np.ndarray,
    shocks_out: np.ndarray,
    batch_eval_residual_fn: Callable[[Any], Any],
    structural_zero_based: np.ndarray,
    shock_log_norm_const: float,
    *,
    single_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    gate_mask: Optional[np.ndarray] = None,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
) -> np.ndarray:
    d_obs, periods = observations.shape
    d_state = s0.shape[0]
    theta_col = theta.reshape(-1, 1)

    if gate_mask is not None and single_eval_residual_fn is not None and bool(np.any(gate_mask)):
        obs_pred = np.zeros((d_obs, periods), dtype=np.float64)
        input_states = np.zeros((d_state, periods), dtype=np.float64)
        state_rom = s0.copy()
        for period in range(periods):
            input_states[:, period] = state_rom
            obs_t, state_next = _call_predict(
                predict_fn,
                state_rom,
                shocks_out[:, period],
                theta,
            )
            if bool(gate_mask[period]):
                x_nn = np.concatenate([state_rom, shocks_out[:, period], theta], axis=0)
                y_nn = _clamp_residual_vector(
                    single_eval_residual_fn(x_nn),
                    correction_clamp,
                    label="single_eval_residual_fn output",
                )
                if y_nn.shape[0] < d_obs:
                    raise ValueError(
                        "single_eval_residual_fn output length mismatch: "
                        f"got {y_nn.shape[0]}, expected at least d_obs={d_obs}."
                    )
                obs_pred[:, period] = obs_t + y_nn[:d_obs]
                state_resid = y_nn[d_obs:]
                if state_resid.shape[0] == state_next.shape[0]:
                    state_rom = state_next + state_resid
                elif state_resid.shape[0] == 0:
                    state_rom = state_next
                else:
                    raise ValueError(
                        "single_eval_residual_fn state residual length mismatch: "
                        f"{state_resid.shape[0]} vs state length {state_next.shape[0]}."
                    )
            else:
                obs_pred[:, period] = obs_t
                state_rom = state_next

        non_gate_idx = np.flatnonzero(~gate_mask)
        if non_gate_idx.size:
            x_nn_ng = np.vstack(
                [
                    input_states[:, non_gate_idx],
                    shocks_out[:, non_gate_idx],
                    np.repeat(theta_col, non_gate_idx.size, axis=1),
                ]
            )
            y_nn_ng = _clamp_residual_matrix(
                batch_eval_residual_fn(x_nn_ng),
                correction_clamp,
                label="batch_eval_residual_fn output",
                periods=non_gate_idx.size,
            )
            if y_nn_ng.shape[0] < d_obs:
                raise ValueError(
                    "batch_eval_residual_fn output row mismatch: "
                    f"got {y_nn_ng.shape[0]}, expected at least d_obs={d_obs}."
                )
            obs_pred[:, non_gate_idx] += y_nn_ng[:d_obs, :]
    else:
        input_states = np.zeros((d_state, periods), dtype=np.float64)
        rom_obs = np.zeros((d_obs, periods), dtype=np.float64)
        state_rom = s0.copy()
        for period in range(periods):
            input_states[:, period] = state_rom
            obs_t, state_rom = _call_predict(
                predict_fn,
                state_rom,
                shocks_out[:, period],
                theta,
            )
            if obs_t.shape[0] != d_obs:
                raise ValueError(
                    "predict_fn observation length mismatch during batch replay: "
                    f"{obs_t.shape[0]} vs {d_obs}."
                )
            rom_obs[:, period] = obs_t
        x_nn = np.vstack(
            [
                input_states,
                shocks_out,
                np.repeat(theta_col, periods, axis=1),
            ]
        )
        y_nn = _clamp_residual_matrix(
            batch_eval_residual_fn(x_nn),
            correction_clamp,
            label="batch_eval_residual_fn output",
            periods=periods,
        )
        if y_nn.shape[0] < d_obs:
            raise ValueError(
                "batch_eval_residual_fn output row mismatch: "
                f"got {y_nn.shape[0]}, expected at least d_obs={d_obs}."
            )
        obs_pred = rom_obs + y_nn[:d_obs, :]

    log_norm_sum = float(np.sum(np.log(2.0 * np.pi * obs_sigma**2)))
    resid_scaled = (observations - obs_pred) / obs_sigma[:, None]
    ll_eval = -0.5 * (np.sum(resid_scaled**2, axis=0) + log_norm_sum)
    if structural_zero_based.size:
        shock_std = shock_sigmas[structural_zero_based]
        shock_penalty = -0.5 * (
            np.sum((shocks_out[structural_zero_based, :] / shock_std[:, None]) ** 2, axis=0)
            + shock_log_norm_const
        )
        ll_eval = ll_eval + shock_penalty
    return np.asarray(ll_eval, dtype=np.float64)


def inversion_step(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    state: Sequence[float] | np.ndarray | jax.Array,
    y_obs: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    obs_sigma: Sequence[float] | np.ndarray | jax.Array,
    shock_sigmas: Sequence[float] | np.ndarray | jax.Array,
    structural_idx: Sequence[int] | np.ndarray,
    *,
    eps_init: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
    maxit: int = 10,
    tol: float = 1e-6,
    lambda_: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    maxit_int = int(maxit)
    tol_float = float(tol)
    lambda_float = float(lambda_)
    if maxit_int <= 0:
        raise ValueError(f"maxit must be positive, got {maxit}.")
    if tol_float <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if lambda_float < 0:
        raise ValueError(f"lambda must be nonnegative, got {lambda_}.")

    state_vec = _as_float_vector(state, label="state")
    y_obs_vec = _as_float_vector(y_obs, label="y_obs")
    theta_vec = _as_float_vector(theta, label="theta")
    obs_sigma_vec = _as_float_vector(obs_sigma, label="obs_sigma")
    shock_sigma_vec = _as_float_vector(shock_sigmas, label="shock_sigmas")
    if y_obs_vec.shape[0] != obs_sigma_vec.shape[0]:
        raise ValueError(
            "y_obs/obs_sigma length mismatch: "
            f"{y_obs_vec.shape[0]} vs {obs_sigma_vec.shape[0]}."
        )
    if not np.isfinite(obs_sigma_vec).all():
        raise ValueError("obs_sigma contains non-finite values.")
    if not np.all(obs_sigma_vec > 0):
        raise ValueError("obs_sigma must be strictly positive.")
    if not np.isfinite(shock_sigma_vec).all():
        raise ValueError("shock_sigmas contains non-finite values.")
    if not np.all(shock_sigma_vec >= 0):
        raise ValueError("shock_sigmas must be nonnegative.")

    d_eps = shock_sigma_vec.shape[0]
    structural = _normalize_julia_indices(
        structural_idx,
        upper=d_eps,
        label="structural_idx",
    )
    shock_std = shock_sigma_vec[structural]
    if structural.size and not np.all(shock_std > 0):
        raise ValueError("shock_sigmas at structural_idx must be > 0.")
    if eps_init is None:
        eps_struct = np.zeros((structural.shape[0],), dtype=np.float64)
    else:
        eps_struct = _as_float_vector(eps_init, label="eps_init").copy()
    if eps_struct.shape[0] != structural.shape[0]:
        raise ValueError(
            "eps_init length mismatch: "
            f"{eps_struct.shape[0]} vs structural shock count {structural.shape[0]}."
        )

    obs_log_norm_const = float(np.sum(np.log(2.0 * np.pi * obs_sigma_vec**2)))
    if structural.size == 0:
        eps_full = np.zeros((d_eps,), dtype=np.float64)
        obs_pred, state_next = _call_predict(
            predict_fn,
            state_vec,
            eps_full,
            theta_vec,
        )
        resid = (y_obs_vec - obs_pred) / obs_sigma_vec
        ll = -0.5 * (float(np.sum(resid**2)) + obs_log_norm_const)
        return eps_full, state_next, ll

    shock_log_norm_const = float(np.sum(np.log(2.0 * np.pi * shock_std**2)))
    lambda_eff = lambda_float
    for inv_iter in range(maxit_int):
        eps_full = _full_shock_vector(d_eps, structural, eps_struct)
        obs_pred, _ = _call_predict(
            predict_fn,
            state_vec,
            eps_full,
            theta_vec,
        )
        if not np.isfinite(obs_pred).all():
            break
        resid = (y_obs_vec - obs_pred) / obs_sigma_vec
        r = np.concatenate([resid, eps_struct / shock_std], axis=0)

        jacobian = _jacobian_wrt_structural_shocks(
            predict_fn,
            state_vec,
            theta_vec,
            d_eps,
            structural,
            eps_struct,
        )
        if not np.isfinite(jacobian).all():
            break
        j_obs = -(jacobian / obs_sigma_vec[:, None])
        j_prior = np.diag(1.0 / shock_std)
        j_aug = np.vstack([j_obs, j_prior])
        lhs = j_aug.T @ j_aug + lambda_eff * np.eye(structural.shape[0], dtype=np.float64)
        diag_vals = np.diag(lhs)
        kappa_approx = float(np.max(diag_vals) / max(float(np.min(diag_vals)), np.finfo(float).eps))
        if kappa_approx > 1e10:
            kappa = float(np.linalg.cond(lhs))
            if kappa > 1e14:
                lambda_eff = max(lambda_eff * 10.0, 1e-2)
                lhs = j_aug.T @ j_aug + lambda_eff * np.eye(
                    structural.shape[0],
                    dtype=np.float64,
                )
        elif kappa_approx < 1e4 and lambda_eff > lambda_float and inv_iter > 0:
            lambda_eff = max(lambda_eff / 2.0, lambda_float)
        rhs = -(j_aug.T @ r)
        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            step = np.full((structural.shape[0],), np.nan, dtype=np.float64)
        if not np.isfinite(step).all():
            break
        eps_struct = eps_struct + step
        if np.linalg.norm(step) <= tol_float * (1.0 + np.linalg.norm(eps_struct)):
            break

    eps_full = _full_shock_vector(d_eps, structural, eps_struct)
    obs_pred, state_next = _call_predict(
        predict_fn,
        state_vec,
        eps_full,
        theta_vec,
    )
    if not np.isfinite(obs_pred).all() or not np.isfinite(state_next).all():
        return eps_full, state_vec, -np.inf
    resid = (y_obs_vec - obs_pred) / obs_sigma_vec
    ll = -0.5 * (
        float(np.sum(resid**2))
        + float(np.sum((eps_struct / shock_std) ** 2))
        + obs_log_norm_const
        + shock_log_norm_const
    )
    return eps_full, state_next, ll


def inversion_loglik_per_period(
    predict_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    s0: Sequence[float] | np.ndarray | jax.Array,
    theta: Sequence[float] | np.ndarray | jax.Array,
    obs_data: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    obs_sigma: Sequence[float] | np.ndarray | jax.Array,
    shock_sigmas: Sequence[float] | np.ndarray | jax.Array,
    *,
    eval_predict_fn: Optional[Callable[[Any, Any, Any], tuple[Any, Any]]] = None,
    batch_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    maxit: int = 10,
    tol: float = 1e-6,
    lambda_: float = 1e-4,
    refine_maxit: int = 0,
    refine_tol: float = 1e-4,
    refine_max_step_std: float = 2.0,
    refine_min_alpha: float = 1e-3,
    refine_accept_tol: float = 1e-10,
    single_eval_residual_fn: Optional[Callable[[Any], Any]] = None,
    gate_mask: Optional[Sequence[bool] | np.ndarray] = None,
    correction_clamp: Optional[Sequence[float] | np.ndarray | jax.Array] = None,
) -> tuple[np.ndarray, np.ndarray]:
    maxit_int = int(maxit)
    tol_float = float(tol)
    lambda_float = float(lambda_)
    refine_maxit_int = int(refine_maxit)
    refine_tol_float = float(refine_tol)
    refine_max_step_std_float = float(refine_max_step_std)
    refine_min_alpha_float = float(refine_min_alpha)
    refine_accept_tol_float = float(refine_accept_tol)
    if maxit_int <= 0:
        raise ValueError(f"maxit must be positive, got {maxit}.")
    if tol_float <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if lambda_float < 0:
        raise ValueError(f"lambda must be nonnegative, got {lambda_}.")
    if refine_maxit_int < 0:
        raise ValueError(f"refine_maxit must be nonnegative, got {refine_maxit}.")
    if refine_tol_float <= 0:
        raise ValueError(f"refine_tol must be positive, got {refine_tol}.")
    if refine_max_step_std_float <= 0:
        raise ValueError(
            f"refine_max_step_std must be positive, got {refine_max_step_std}."
        )
    if not (0.0 < refine_min_alpha_float <= 1.0):
        raise ValueError(
            f"refine_min_alpha must be in (0, 1], got {refine_min_alpha}."
        )
    if refine_accept_tol_float < 0:
        raise ValueError(
            f"refine_accept_tol must be nonnegative, got {refine_accept_tol}."
        )

    observations = _as_float_matrix(obs_data, label="obs_data")
    obs_sigma_vec = _as_float_vector(obs_sigma, label="obs_sigma")
    shock_sigma_vec = _as_float_vector(shock_sigmas, label="shock_sigmas")
    if observations.shape[0] != obs_sigma_vec.shape[0]:
        raise ValueError(
            "obs_data/obs_sigma mismatch: obs rows="
            f"{observations.shape[0]} vs sigma={obs_sigma_vec.shape[0]}."
        )

    state = _as_float_vector(s0, label="s0")
    theta_vec = _as_float_vector(theta, label="theta")
    structural_idx = np.flatnonzero(shock_sigma_vec > 0.0) + 1
    structural_zero = structural_idx - 1
    shock_std = shock_sigma_vec[structural_zero] if structural_zero.size else np.zeros((0,), dtype=np.float64)
    obs_log_norm_const = float(np.sum(np.log(2.0 * np.pi * obs_sigma_vec**2)))
    shock_log_norm_const = (
        float(np.sum(np.log(2.0 * np.pi * shock_std**2))) if structural_zero.size else 0.0
    )
    gate_vec: Optional[np.ndarray]
    if gate_mask is None:
        gate_vec = None
    else:
        gate_vec = np.asarray(gate_mask, dtype=bool).reshape(-1)
        if gate_vec.shape[0] != observations.shape[1]:
            raise ValueError(
                f"gate_mask length mismatch: {gate_vec.shape[0]} vs {observations.shape[1]} periods."
            )
    eps_init = np.zeros((structural_idx.shape[0],), dtype=np.float64)
    ll = np.zeros((observations.shape[1],), dtype=np.float64)
    shocks_out = np.zeros((shock_sigma_vec.shape[0], observations.shape[1]), dtype=np.float64)
    do_refine = (
        refine_maxit_int > 0
        and structural_zero.size > 0
        and eval_predict_fn is not None
    )

    for period in range(observations.shape[1]):
        eps_full, state_next, ll_t = inversion_step(
            predict_fn,
            state,
            observations[:, period],
            theta_vec,
            obs_sigma_vec,
            shock_sigma_vec,
            structural_idx,
            eps_init=eps_init,
            maxit=maxit_int,
            tol=tol_float,
            lambda_=lambda_float,
        )
        if do_refine:
            current_obj, _, _ = _period_inversion_objective(
                predict_fn,
                state,
                eps_full,
                theta_vec,
                observations[:, period],
                obs_sigma_vec,
                shock_std,
                structural_zero,
                obs_log_norm_const,
                shock_log_norm_const,
                observations.shape[0],
                period_idx=period,
                eval_predict_fn=eval_predict_fn,
                single_eval_residual_fn=single_eval_residual_fn,
                gate_mask=gate_vec,
                correction_clamp=correction_clamp,
            )
            if not np.isfinite(current_obj):
                shocks_out[:, period] = eps_full
                ll[period] = -1e10
                continue

            for _ in range(refine_maxit_int):
                obs_nl, _ = _evaluate_inversion_model(
                    predict_fn,
                    state,
                    eps_full,
                    theta_vec,
                    observations.shape[0],
                    period_idx=period,
                    eval_predict_fn=eval_predict_fn,
                    single_eval_residual_fn=single_eval_residual_fn,
                    gate_mask=gate_vec,
                    correction_clamp=correction_clamp,
                )
                if not np.isfinite(obs_nl).all():
                    break
                r_obs = observations[:, period] - obs_nl
                if np.linalg.norm(r_obs / obs_sigma_vec) <= refine_tol_float:
                    break
                eps_struct = eps_full[structural_zero]
                jacobian = _finite_difference_eval_jacobian(
                    predict_fn,
                    state,
                    theta_vec,
                    shock_sigma_vec.shape[0],
                    structural_zero,
                    eps_struct,
                    observations.shape[0],
                    period_idx=period,
                    eval_predict_fn=eval_predict_fn,
                    single_eval_residual_fn=single_eval_residual_fn,
                    gate_mask=gate_vec,
                    correction_clamp=correction_clamp,
                )
                if not np.isfinite(jacobian).all():
                    break
                j_scaled = jacobian / obs_sigma_vec[:, None]
                j_prior = np.diag(1.0 / shock_std)
                j_aug = np.vstack([j_scaled, j_prior])
                r_aug = np.concatenate([r_obs / obs_sigma_vec, -eps_struct / shock_std], axis=0)
                lhs = j_aug.T @ j_aug + lambda_float * np.eye(
                    structural_zero.shape[0],
                    dtype=np.float64,
                )
                rhs = j_aug.T @ r_aug
                try:
                    delta = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError:
                    delta = np.full((structural_zero.shape[0],), np.nan, dtype=np.float64)
                if not np.isfinite(delta).all():
                    break

                max_scaled_step = float(np.max(np.abs(delta / shock_std)))
                if max_scaled_step > refine_max_step_std_float:
                    delta = delta * (refine_max_step_std_float / max_scaled_step)

                accepted = False
                accepted_alpha = 0.0
                alpha = 1.0
                while alpha >= refine_min_alpha_float:
                    eps_candidate = eps_full.copy()
                    eps_candidate[structural_zero] = eps_struct + alpha * delta
                    candidate_obj, _, _ = _period_inversion_objective(
                        predict_fn,
                        state,
                        eps_candidate,
                        theta_vec,
                        observations[:, period],
                        obs_sigma_vec,
                        shock_std,
                        structural_zero,
                        obs_log_norm_const,
                        shock_log_norm_const,
                        observations.shape[0],
                        period_idx=period,
                        eval_predict_fn=eval_predict_fn,
                        single_eval_residual_fn=single_eval_residual_fn,
                        gate_mask=gate_vec,
                        correction_clamp=correction_clamp,
                    )
                    if np.isfinite(candidate_obj) and candidate_obj >= current_obj - refine_accept_tol_float:
                        eps_full = eps_candidate
                        current_obj = candidate_obj
                        accepted = True
                        accepted_alpha = alpha
                        break
                    alpha *= 0.5
                if not accepted:
                    break
                if np.linalg.norm(accepted_alpha * delta) <= refine_tol_float * (
                    1.0 + np.linalg.norm(eps_full[structural_zero])
                ):
                    break

        ll_t, _, state_eval = _period_inversion_objective(
            predict_fn,
            state,
            eps_full,
            theta_vec,
            observations[:, period],
            obs_sigma_vec,
            shock_std,
            structural_zero,
            obs_log_norm_const,
            shock_log_norm_const,
            observations.shape[0],
            period_idx=period,
            eval_predict_fn=eval_predict_fn,
            single_eval_residual_fn=single_eval_residual_fn,
            gate_mask=gate_vec,
            correction_clamp=correction_clamp,
        )
        shocks_out[:, period] = eps_full
        if np.isfinite(ll_t) and np.isfinite(state_eval).all():
            ll[period] = ll_t
            state = state_eval
        else:
            ll[period] = -1e10
        eps_init = eps_full[structural_idx - 1]

    if batch_eval_residual_fn is not None:
        return _batch_residual_inversion_loglik(
            predict_fn,
            _as_float_vector(s0, label="s0"),
            theta_vec,
            observations,
            obs_sigma_vec,
            shock_sigma_vec,
            shocks_out,
            batch_eval_residual_fn,
            structural_zero,
            shock_log_norm_const,
            single_eval_residual_fn=single_eval_residual_fn,
            gate_mask=gate_vec,
            correction_clamp=correction_clamp,
        ), shocks_out

    return ll, shocks_out


def linear_reference_loglik_per_period(
    theta: Sequence[float] | np.ndarray | jax.Array,
    s0: Sequence[float] | np.ndarray | jax.Array,
    shocks: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    obs_data: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    obs_sigma: Sequence[float] | np.ndarray | jax.Array,
    shock_sigmas: Sequence[float] | np.ndarray | jax.Array,
    *,
    shock_filter: str,
    linear_filter: str,
    predict_linear: Optional[Callable[[Any, Any, Any], tuple[Any, Any]]] = None,
    kalman_linear_loglik: Optional[Callable[[np.ndarray], Sequence[float] | np.ndarray]] = None,
    inversion_maxit: int = 10,
    inversion_tol: float = 1e-6,
    inversion_lambda: float = 1e-4,
    inversion_refine_maxit: int = 0,
    inversion_refine_tol: float = 1e-4,
    inversion_refine_max_step_std: float = 2.0,
    inversion_refine_min_alpha: float = 1e-3,
    inversion_refine_accept_tol: float = 1e-10,
) -> np.ndarray:
    observations = _as_float_matrix(obs_data, label="obs_data")
    shock_matrix = _as_float_matrix(shocks, label="shocks")
    if shock_matrix.shape[1] != observations.shape[1]:
        raise ValueError(
            f"obs_data/shocks period mismatch: {observations.shape[1]} vs {shock_matrix.shape[1]}."
        )
    theta_vec = _as_float_vector(theta, label="theta")
    shock_filter_name = str(shock_filter)
    linear_filter_name = str(linear_filter)

    if shock_filter_name == "sampling":
        if predict_linear is None:
            raise ValueError("predict_linear is required when shock_filter='sampling'.")
        return conditional_loglik_per_period(
            predict_linear,
            s0,
            shock_matrix,
            theta_vec,
            observations,
            obs_sigma,
        )

    if linear_filter_name == "inversion":
        if predict_linear is None:
            raise ValueError("predict_linear is required when linear_filter='inversion'.")
        ll, _ = inversion_loglik_per_period(
            predict_linear,
            s0,
            theta_vec,
            observations,
            obs_sigma,
            shock_sigmas,
            maxit=inversion_maxit,
            tol=inversion_tol,
            lambda_=inversion_lambda,
            refine_maxit=inversion_refine_maxit,
            refine_tol=inversion_refine_tol,
            refine_max_step_std=inversion_refine_max_step_std,
            refine_min_alpha=inversion_refine_min_alpha,
            refine_accept_tol=inversion_refine_accept_tol,
        )
        return ll

    if linear_filter_name == "kalman":
        if kalman_linear_loglik is None:
            raise ValueError("kalman_linear_loglik is required when linear_filter='kalman'.")
        ll = _as_float_vector(kalman_linear_loglik(theta_vec), label="kalman_linear_loglik output")
        if ll.shape[0] != observations.shape[1]:
            raise ValueError(
                f"kalman_linear_loglik length mismatch: {ll.shape[0]} vs expected {observations.shape[1]}."
            )
        return ll

    raise ValueError(
        f"Unsupported linear_filter={linear_filter!r}. Use 'kalman' or 'inversion'."
    )


def build_shocks_from_eps(
    eps_mean: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    shock_sigmas: Sequence[float] | np.ndarray | jax.Array,
    shock_guided: Optional[Sequence[Sequence[float]] | np.ndarray | jax.Array],
    *,
    sample_idx: Optional[Sequence[int] | np.ndarray] = None,
    shocks_base: Optional[Sequence[Sequence[float]] | np.ndarray | jax.Array] = None,
    T_full: Optional[int] = None,
) -> np.ndarray:
    eps_matrix = _as_float_matrix(eps_mean, label="eps_mean")
    shock_sigma_vec = _as_float_vector(shock_sigmas, label="shock_sigmas")
    structural = np.flatnonzero(shock_sigma_vec > 0.0)
    if eps_matrix.shape[0] != structural.shape[0]:
        raise ValueError(
            f"eps_mean row mismatch: got {eps_matrix.shape[0]}, expected {structural.shape[0]} structural shocks."
        )

    if sample_idx is None:
        sample_idx_one_based = np.arange(1, eps_matrix.shape[1] + 1, dtype=np.int64)
    else:
        sample_idx_one_based = np.asarray(sample_idx, dtype=np.int64).reshape(-1)
    if eps_matrix.shape[1] != sample_idx_one_based.shape[0]:
        raise ValueError(
            f"eps_mean column mismatch: got {eps_matrix.shape[1]}, expected {sample_idx_one_based.shape[0]} sample periods."
        )
    if sample_idx_one_based.size and np.min(sample_idx_one_based) < 1:
        raise ValueError("sample_idx must be >= 1.")

    base = shocks_base if shocks_base is not None else shock_guided
    inferred_t = (
        int(np.max(sample_idx_one_based))
        if base is None and sample_idx_one_based.size
        else eps_matrix.shape[1]
        if base is None
        else _as_float_matrix(base, label="base shocks").shape[1]
    )
    total_periods = inferred_t if T_full is None else int(T_full)
    if total_periods < 0:
        raise ValueError(f"T_full must be nonnegative, got {T_full}.")
    if sample_idx_one_based.size and int(np.max(sample_idx_one_based)) > total_periods:
        raise ValueError(f"sample_idx exceeds target length {total_periods}.")

    if base is None:
        shocks = np.zeros((shock_sigma_vec.shape[0], total_periods), dtype=np.float64)
    else:
        shocks = _as_float_matrix(base, label="base shocks").astype(np.float64, copy=True)
        if shocks.shape[0] != shock_sigma_vec.shape[0]:
            raise ValueError(
                f"Base shock matrix row mismatch: got {shocks.shape[0]}, expected {shock_sigma_vec.shape[0]}."
            )
        if shocks.shape[1] != total_periods:
            raise ValueError(
                f"Base shock matrix length mismatch: got {shocks.shape[1]}, expected {total_periods}."
            )

    target_cols = sample_idx_one_based - 1
    for row, shock_index in enumerate(structural):
        shocks[shock_index, target_cols] = (
            shocks[shock_index, target_cols]
            + eps_matrix[row, :] * shock_sigma_vec[shock_index]
        )
    return shocks


def _unwrap_chain_payload(chain: Any) -> Any:
    if isinstance(chain, Mapping):
        if "chain" in chain:
            return chain["chain"]
        if "samples" in chain and isinstance(chain["samples"], Mapping):
            return chain["samples"]
    return chain


def _chain_samples_mapping(chain: Any) -> Mapping[str, Any]:
    payload = _unwrap_chain_payload(chain)
    if isinstance(payload, Mapping):
        return payload
    if hasattr(payload, "get_samples"):
        try:
            samples = payload.get_samples(group_by_chain=False)
        except TypeError:
            samples = payload.get_samples()
        if not isinstance(samples, Mapping):
            raise ValueError("chain.get_samples() must return a mapping of sample sites.")
        return samples
    raise TypeError(
        "chain must be a mapping of site arrays, a checkpoint-like payload containing "
        "`chain`/`samples`, or an object exposing `get_samples()`."
    )


def _flatten_chain_scalar_site(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1).astype(np.float64)
    if array.ndim == 1:
        return array.astype(np.float64, copy=False)
    if array.ndim == 2:
        return array.reshape(-1).astype(np.float64, copy=False)
    raise ValueError(
        f"{label} must be scalar-valued per draw, got array shape {array.shape}."
    )


def _mean_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.mean(array))


def _count_optional(value: Any) -> Optional[int]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return int(np.sum(array))


def _scalar_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    return float(array.reshape(-1)[-1])


def theta_draws(
    chain: Any,
    theta_names: Sequence[str],
) -> np.ndarray:
    samples = _chain_samples_mapping(chain)
    names = tuple(str(name) for name in theta_names)
    if not names:
        if not samples:
            return np.zeros((0, 0), dtype=np.float64)
        first = next(iter(samples.values()))
        n_draws = _flatten_chain_scalar_site(first, label="chain sample").shape[0]
        return np.zeros((n_draws, 0), dtype=np.float64)

    missing = tuple(name for name in names if name not in samples)
    if missing:
        raise ValueError("Theta names not found in chain samples: " + ", ".join(missing) + ".")

    columns: list[np.ndarray] = []
    n_draws: Optional[int] = None
    for name in names:
        values = _flatten_chain_scalar_site(samples[name], label=f"chain sample `{name}`")
        if n_draws is None:
            n_draws = int(values.shape[0])
        elif values.shape[0] != n_draws:
            raise ValueError(
                f"Chain sample length mismatch for `{name}`: got {values.shape[0]}, expected {n_draws}."
            )
        columns.append(values)
    return np.column_stack(columns)


def epsilon_means_from_chain(
    chain: Any,
    sample_idx: Optional[Sequence[int]] = None,
) -> Optional[np.ndarray]:
    samples = _chain_samples_mapping(chain)
    eps_meta: list[tuple[str, int, int, np.ndarray]] = []
    for sym, values in samples.items():
        name = str(sym).replace(" ", "")
        match = _EPSILON_SITE_RE.match(name)
        if match is None:
            continue
        i = int(match.group(1))
        t = int(match.group(2))
        flattened = _flatten_chain_scalar_site(values, label=f"chain sample `{sym}`")
        eps_meta.append((str(sym), i, t, flattened))
    if not eps_meta:
        return None

    max_i = max(item[1] for item in eps_meta)
    max_t = max(item[2] for item in eps_meta)
    if sample_idx is None:
        sample_len = max_t
    else:
        sample_len = len(tuple(sample_idx))
        if max_t != sample_len:
            warnings.warn(
                "epsilon index max_t does not match sample_idx length; truncating to the "
                "requested number of periods.",
                UserWarning,
                stacklevel=2,
            )

    eps_mean = np.full((max_i, sample_len), np.nan, dtype=np.float64)
    for _, i, t, values in eps_meta:
        if t <= sample_len:
            eps_mean[i - 1, t - 1] = float(np.mean(values))
    return eps_mean


def chunk_stats(chain: Any) -> tuple[Optional[float], Optional[int], Optional[float]]:
    payload = _unwrap_chain_payload(chain)

    if isinstance(payload, Mapping):
        info = payload.get("info")
        if isinstance(info, Mapping):
            internals = info.get("internals")
            if isinstance(internals, Mapping):
                acc = _mean_optional(internals.get("avg_acceptance_rate"))
                div = _count_optional(internals.get("count_divergences"))
                step = _scalar_optional(internals.get("step_size"))
                if acc is not None or div is not None or step is not None:
                    return acc, div, step
        extra_fields = payload.get("extra_fields")
        last_state = payload.get("last_state")
    else:
        extra_fields = (
            payload.get_extra_fields()
            if hasattr(payload, "get_extra_fields")
            else None
        )
        last_state = getattr(payload, "last_state", None)

    accept = None
    divergences = None
    step_size = None

    if isinstance(extra_fields, Mapping):
        accept = _mean_optional(
            extra_fields.get("accept_prob", extra_fields.get("avg_acceptance_rate"))
        )
        divergences = _count_optional(
            extra_fields.get("diverging", extra_fields.get("count_divergences"))
        )
        step_size = _scalar_optional(extra_fields.get("step_size"))

    if last_state is not None:
        if accept is None:
            accept = _mean_optional(
                getattr(
                    last_state,
                    "mean_accept_prob",
                    getattr(last_state, "accept_prob", None),
                )
            )
        if divergences is None:
            divergences = _count_optional(getattr(last_state, "diverging", None))
        if step_size is None:
            adapt_state = getattr(last_state, "adapt_state", None)
            step_size = _scalar_optional(
                getattr(adapt_state, "step_size", getattr(last_state, "step_size", None))
            )

    return accept, divergences, step_size


def run_chunked_sampling(
    total_samples: int,
    chunk_size: int,
    *,
    sample_chunk: Callable[[int, int, int], Any],
    concat_chunks: Callable[[Any, Any], Any],
    on_chunk: Optional[Callable[[int, int, int, Any, Any, float], None]] = None,
) -> Any:
    total = int(total_samples)
    chunk = int(chunk_size)
    if total <= 0:
        raise ValueError(f"total_samples must be positive, got {total_samples}.")
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    n_chunks = int(np.ceil(total / chunk))
    samples: Any = None
    start_time = time()
    for chunk_idx in range(1, n_chunks + 1):
        n_i = min(chunk, total - (chunk_idx - 1) * chunk)
        current = sample_chunk(n_i, chunk_idx, n_chunks)
        samples = current if samples is None else concat_chunks(samples, current)
        if on_chunk is not None:
            on_chunk(chunk_idx, n_chunks, n_i, current, samples, time() - start_time)
    return samples


def linear_model_loglik_per_period(
    model: Any,
    obs_data: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    theta: Sequence[float] | np.ndarray | jax.Array,
    theta_names: Sequence[str],
    *,
    observables: Optional[Sequence[str] | str] = None,
    model_parameter_names: Optional[Sequence[str]] = None,
    base_parameters: Optional[Sequence[float] | np.ndarray] = None,
    theta_idx: Optional[Sequence[int] | np.ndarray] = None,
    algorithm: str = "first_order",
    filter: str = "kalman",
    on_failure_loglikelihood: float = -1e12,
    presample_periods: int = 0,
    initial_covariance: str = "theoretical",
    verbose: bool = False,
    theta_label: str = "Theta names",
) -> np.ndarray:
    parameter_names = (
        tuple(str(name) for name in model_parameter_names)
        if model_parameter_names is not None
        else tuple(str(name) for name in getattr(model, "parameter_names"))
    )
    base_params = (
        _as_float_vector(base_parameters, label="base_parameters")
        if base_parameters is not None
        else _as_float_vector(getattr(model, "parameter_values"), label="model.parameter_values")
    )
    theta_vec = _as_float_vector(theta, label="theta")
    params = (
        override_named_parameters(
            base_params,
            parameter_names,
            tuple(str(name) for name in theta_names),
            theta_vec,
            label=theta_label,
        )
        if theta_idx is None
        else _override_named_parameters_with_index(
            base_params,
            theta_idx,
            theta_vec,
            theta_label=theta_label,
        )
    )

    filter_name = str(filter)
    if filter_name == "kalman":
        if str(algorithm) != "first_order":
            raise NotImplementedError(
                "Kalman linear-model loglikelihood is currently available only for the first-order path."
            )
        ll = model.kalman_loglikelihood_per_period(
            obs_data,
            observables=observables,
            parameter_values=params,
            initial_covariance_strategy=initial_covariance,
            presample_periods=int(presample_periods),
            on_failure_loglikelihood=float(on_failure_loglikelihood),
        )
    elif filter_name == "inversion":
        ll = model.inversion_loglikelihood_per_period(
            obs_data,
            observables=observables,
            parameter_values=params,
            algorithm=str(algorithm),
            presample_periods=int(presample_periods),
            on_failure_loglikelihood=float(on_failure_loglikelihood),
        )
    else:
        raise ValueError(f"Unsupported filter={filter!r}. Use 'kalman' or 'inversion'.")
    del verbose
    return _as_float_vector(ll, label="linear_model_loglik_per_period output")
