from __future__ import annotations

from time import time
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


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
    if mode == "synthetic":
        if theta_values is None:
            raise ValueError(f"{mode_label}='synthetic' requires theta values.")
        return override_named_parameters(
            base_params,
            model_parameter_names,
            theta_names,
            theta_values,
            label=theta_label,
        )
    raise ValueError(
        f"Unknown {mode_label}={theta_mode!r}. Use 'baseline' or 'synthetic'."
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
    for _ in range(maxit_int):
        eps_full = np.zeros((d_eps,), dtype=np.float64)
        eps_full[structural] = eps_struct
        obs_pred, _ = _call_predict(
            predict_fn,
            state_vec,
            eps_full,
            theta_vec,
        )
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
        j_obs = -(jacobian / obs_sigma_vec[:, None])
        j_prior = np.diag(1.0 / shock_std)
        j_aug = np.vstack([j_obs, j_prior])
        lhs = j_aug.T @ j_aug + lambda_float * np.eye(structural.shape[0], dtype=np.float64)
        rhs = -(j_aug.T @ r)
        step = np.linalg.solve(lhs, rhs)
        eps_struct = eps_struct + step
        if np.linalg.norm(step) <= tol_float * (1.0 + np.linalg.norm(eps_struct)):
            break

    eps_full = np.zeros((d_eps,), dtype=np.float64)
    eps_full[structural] = eps_struct
    obs_pred, state_next = _call_predict(
        predict_fn,
        state_vec,
        eps_full,
        theta_vec,
    )
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
    maxit: int = 10,
    tol: float = 1e-6,
    lambda_: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    maxit_int = int(maxit)
    tol_float = float(tol)
    lambda_float = float(lambda_)
    if maxit_int <= 0:
        raise ValueError(f"maxit must be positive, got {maxit}.")
    if tol_float <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if lambda_float < 0:
        raise ValueError(f"lambda must be nonnegative, got {lambda_}.")

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
    eps_init = np.zeros((structural_idx.shape[0],), dtype=np.float64)
    ll = np.zeros((observations.shape[1],), dtype=np.float64)
    shocks_out = np.zeros((shock_sigma_vec.shape[0], observations.shape[1]), dtype=np.float64)

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
        shocks_out[:, period] = eps_full
        ll[period] = ll_t
        state = state_next
        eps_init = eps_full[structural_idx - 1]

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
