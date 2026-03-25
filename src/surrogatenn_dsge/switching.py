from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


class SwitchingLikelihoodResult(NamedTuple):
    total: jax.Array
    per_period: jax.Array
    hard_mask: jax.Array
    gate_probs: jax.Array
    ll_rom: jax.Array
    ll_fom: jax.Array


class GateCalibrationResult(NamedTuple):
    quantile: float
    tau_eps: float
    tau_y: float
    achieved_share: float


class LinearGateStatsResult(NamedTuple):
    linear_observations: jax.Array
    shocks: jax.Array
    e_stat: jax.Array
    f_stat: jax.Array


@dataclass(frozen=True)
class RegimeSwitchConfig:
    gate_mode: str = "hard"
    tau_eps: float = 1.95
    tau_y: float = 1.95
    beta_eps: float = 1.0
    beta_y: float = 1.0
    bias: float = 0.0
    k_pre: int = 0
    k_post: int = 0
    min_len: int = 1
    use_eps: bool = True
    use_y: bool = True
    hard_threshold: float = 0.5
    prob_floor: float = 1e-4
    prob_ceiling: float = 1.0 - 1e-4
    soft_mixture: str = "logsumexp"


@dataclass(frozen=True)
class GateCalibrationConfig:
    target_share: float = 0.1
    tol: float = 1e-4
    maxiter: int = 50
    use_eps: bool = True
    use_y: bool = True


@dataclass(frozen=True)
class SwitchingLikelihoodConfig:
    gate_mode: str = "hard"
    hard_threshold: float = 0.5
    prob_floor: float = 1e-4
    prob_ceiling: float = 1.0 - 1e-4
    soft_mixture: str = "logsumexp"


def _validate_gate_series(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eps = np.asarray(e_stat, dtype=np.float64)
    err = np.asarray(f_stat, dtype=np.float64)
    if eps.ndim != 1 or err.ndim != 1:
        raise ValueError(
            f"Gate statistics must be rank-1, got {eps.shape} and {err.shape}."
        )
    if eps.shape != err.shape:
        raise ValueError(
            f"Gate statistics length mismatch: {eps.shape} and {err.shape}."
        )
    if not np.isfinite(eps).all() or not np.isfinite(err).all():
        raise ValueError("Gate statistics must contain only finite values.")
    return eps, err


def _gate_mask(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    tau_eps: float,
    tau_y: float,
    *,
    use_eps: bool = True,
    use_y: bool = True,
) -> np.ndarray:
    eps, err = _validate_gate_series(e_stat, f_stat)
    if not (use_eps or use_y):
        raise ValueError("At least one of use_eps/use_y must be true.")
    mask = np.zeros_like(eps, dtype=bool)
    if use_eps:
        mask |= eps > tau_eps
    if use_y:
        mask |= err > tau_y
    return mask


def gate_share(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    tau_eps: float,
    tau_y: float,
    *,
    use_eps: bool = True,
    use_y: bool = True,
) -> float:
    return float(
        np.mean(
            _gate_mask(
                e_stat,
                f_stat,
                tau_eps,
                tau_y,
                use_eps=use_eps,
                use_y=use_y,
            )
        )
    )


def _gate_norm(values: np.ndarray, mode: str) -> float:
    if mode == "l2":
        return float(np.linalg.norm(values, ord=2))
    if mode == "linf":
        return float(np.linalg.norm(values, ord=np.inf))
    raise ValueError(f"Unknown norm mode {mode!r}. Use 'l2' or 'linf'.")


def _gate_norm_jax(values: jax.Array, mode: str, *, axis: int) -> jax.Array:
    if mode == "l2":
        return jnp.linalg.norm(values, ord=2, axis=axis)
    if mode == "linf":
        return jnp.linalg.norm(values, ord=jnp.inf, axis=axis)
    raise ValueError(f"Unknown norm mode {mode!r}. Use 'l2' or 'linf'.")


def compute_gate_stat_series(
    obs_data: Sequence[Sequence[float]] | np.ndarray,
    lin_obs: Sequence[Sequence[float]] | np.ndarray,
    shocks: Sequence[Sequence[float]] | np.ndarray,
    obs_sigma: Sequence[float] | np.ndarray,
    shock_sigmas: Sequence[float] | np.ndarray,
    *,
    structural_idx: Optional[Sequence[int] | np.ndarray] = None,
    shock_norm: str = "l2",
    error_norm: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(obs_data, dtype=np.float64)
    linear_observations = np.asarray(lin_obs, dtype=np.float64)
    shock_matrix = np.asarray(shocks, dtype=np.float64)
    observation_sigma = np.asarray(obs_sigma, dtype=np.float64)
    shock_sigma = np.asarray(shock_sigmas, dtype=np.float64)

    if observations.shape != linear_observations.shape:
        raise ValueError(
            "obs_data and lin_obs size mismatch: "
            f"{observations.shape} vs {linear_observations.shape}."
        )
    if shock_matrix.ndim != 2 or observations.ndim != 2:
        raise ValueError("obs_data, lin_obs, and shocks must be rank-2.")
    if shock_matrix.shape[1] != observations.shape[1]:
        raise ValueError(
            "Shock matrix length mismatch: got "
            f"{shock_matrix.shape[1]} periods, expected {observations.shape[1]}."
        )
    if observation_sigma.shape != (observations.shape[0],):
        raise ValueError(
            "obs_sigma length mismatch: got "
            f"{observation_sigma.shape}, expected ({observations.shape[0]},)."
        )
    if shock_sigma.shape != (shock_matrix.shape[0],):
        raise ValueError(
            "shock_sigmas length mismatch: got "
            f"{shock_sigma.shape}, expected ({shock_matrix.shape[0]},)."
        )
    if not np.isfinite(observation_sigma).all() or not np.all(observation_sigma > 0):
        raise ValueError("obs_sigma must contain only positive finite values.")
    if not np.isfinite(shock_sigma).all():
        raise ValueError("shock_sigmas must contain only finite values.")

    if structural_idx is None:
        structural = np.flatnonzero(shock_sigma > 0.0)
    else:
        structural = np.asarray(structural_idx, dtype=np.int64)
        if structural.ndim != 1:
            raise ValueError("structural_idx must be rank-1 when provided.")
        if structural.size and (
            np.min(structural) < 0 or np.max(structural) >= shock_matrix.shape[0]
        ):
            raise ValueError("structural_idx contains an out-of-bounds index.")
    if structural.size and not np.all(shock_sigma[structural] > 0):
        raise ValueError("shock_sigmas at structural_idx must be strictly positive.")

    periods = observations.shape[1]
    e_stat = np.zeros((periods,), dtype=np.float64)
    f_stat = np.zeros((periods,), dtype=np.float64)
    for period in range(periods):
        f_stat[period] = _gate_norm(
            (observations[:, period] - linear_observations[:, period]) / observation_sigma,
            error_norm,
        )
        if structural.size == 0:
            e_stat[period] = 0.0
        else:
            e_stat[period] = _gate_norm(
                shock_matrix[structural, period] / shock_sigma[structural],
                shock_norm,
            )
    return e_stat, f_stat


def compute_gate_stat_series_jax(
    obs_data: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    lin_obs: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    shocks: Sequence[Sequence[float]] | jax.Array | np.ndarray,
    obs_sigma: Sequence[float] | jax.Array | np.ndarray,
    shock_sigmas: Sequence[float] | jax.Array | np.ndarray,
    *,
    structural_idx: Optional[Sequence[int] | np.ndarray] = None,
    shock_norm: str = "l2",
    error_norm: str = "l2",
) -> tuple[jax.Array, jax.Array]:
    observations = jnp.asarray(obs_data, dtype=jnp.float64)
    linear_observations = jnp.asarray(lin_obs, dtype=jnp.float64)
    shock_matrix = jnp.asarray(shocks, dtype=jnp.float64)
    observation_sigma = jnp.asarray(obs_sigma, dtype=jnp.float64)
    shock_sigma = jnp.asarray(shock_sigmas, dtype=jnp.float64)

    if observations.shape != linear_observations.shape:
        raise ValueError(
            "obs_data and lin_obs size mismatch: "
            f"{observations.shape} vs {linear_observations.shape}."
        )
    if shock_matrix.ndim != 2 or observations.ndim != 2:
        raise ValueError("obs_data, lin_obs, and shocks must be rank-2.")
    if shock_matrix.shape[1] != observations.shape[1]:
        raise ValueError(
            "Shock matrix length mismatch: got "
            f"{shock_matrix.shape[1]} periods, expected {observations.shape[1]}."
        )
    if observation_sigma.shape != (observations.shape[0],):
        raise ValueError(
            "obs_sigma length mismatch: got "
            f"{observation_sigma.shape}, expected ({observations.shape[0]},)."
        )
    if shock_sigma.shape != (shock_matrix.shape[0],):
        raise ValueError(
            "shock_sigmas length mismatch: got "
            f"{shock_sigma.shape}, expected ({shock_matrix.shape[0]},)."
        )

    if structural_idx is None:
        structural_mask = shock_sigma > 0.0
    else:
        structural = np.asarray(structural_idx, dtype=np.int64)
        if structural.ndim != 1:
            raise ValueError("structural_idx must be rank-1 when provided.")
        if structural.size and (
            np.min(structural) < 0 or np.max(structural) >= shock_matrix.shape[0]
        ):
            raise ValueError("structural_idx contains an out-of-bounds index.")
        mask = np.zeros((shock_matrix.shape[0],), dtype=bool)
        mask[structural] = True
        structural_mask = jnp.asarray(mask, dtype=jnp.bool_)

    standardized_errors = (
        observations - linear_observations
    ) / observation_sigma[:, None]
    f_stat = _gate_norm_jax(standardized_errors, error_norm, axis=0)

    safe_shock_sigma = jnp.where(structural_mask, shock_sigma, 1.0)
    standardized_shocks = jnp.where(
        structural_mask[:, None],
        shock_matrix / safe_shock_sigma[:, None],
        0.0,
    )
    e_stat = lax.cond(
        jnp.any(structural_mask),
        lambda values: _gate_norm_jax(values, shock_norm, axis=0),
        lambda values: jnp.zeros((values.shape[1],), dtype=values.dtype),
        standardized_shocks,
    )
    return e_stat, f_stat


def calibrate_gate(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    *,
    config: GateCalibrationConfig = GateCalibrationConfig(),
) -> GateCalibrationResult:
    eps, err = _validate_gate_series(e_stat, f_stat)
    if not (config.use_eps or config.use_y):
        raise ValueError("At least one of use_eps/use_y must be true.")
    if not (0.0 < config.target_share < 1.0):
        raise ValueError("target_share must lie strictly between 0 and 1.")

    lo = 0.0
    hi = 1.0
    tau_eps = float(np.quantile(eps, 0.5)) if config.use_eps else float("inf")
    tau_y = float(np.quantile(err, 0.5)) if config.use_y else float("inf")
    achieved_share = gate_share(
        eps,
        err,
        tau_eps,
        tau_y,
        use_eps=config.use_eps,
        use_y=config.use_y,
    )

    for _ in range(config.maxiter):
        quantile = 0.5 * (lo + hi)
        tau_eps = float(np.quantile(eps, quantile)) if config.use_eps else float("inf")
        tau_y = float(np.quantile(err, quantile)) if config.use_y else float("inf")
        achieved_share = gate_share(
            eps,
            err,
            tau_eps,
            tau_y,
            use_eps=config.use_eps,
            use_y=config.use_y,
        )
        if abs(achieved_share - config.target_share) < config.tol:
            return GateCalibrationResult(
                quantile=float(quantile),
                tau_eps=tau_eps,
                tau_y=tau_y,
                achieved_share=float(achieved_share),
            )
        if achieved_share > config.target_share:
            lo = quantile
        else:
            hi = quantile

    quantile = 0.5 * (lo + hi)
    tau_eps = float(np.quantile(eps, quantile)) if config.use_eps else float("inf")
    tau_y = float(np.quantile(err, quantile)) if config.use_y else float("inf")
    achieved_share = gate_share(
        eps,
        err,
        tau_eps,
        tau_y,
        use_eps=config.use_eps,
        use_y=config.use_y,
    )
    return GateCalibrationResult(
        quantile=float(quantile),
        tau_eps=tau_eps,
        tau_y=tau_y,
        achieved_share=float(achieved_share),
    )


def calibrate_tau_y(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    tau_eps: float,
    target_share: float,
    *,
    tol: float = 1e-4,
    maxiter: int = 60,
) -> tuple[float, float]:
    eps, err = _validate_gate_series(e_stat, f_stat)
    if not (0.0 < target_share < 1.0):
        raise ValueError("target_share must lie strictly between 0 and 1.")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")

    lo = float(np.min(err))
    hi = float(np.max(err))
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        share = gate_share(eps, err, tau_eps, mid)
        if abs(share - target_share) < tol:
            return float(mid), float(share)
        if share > target_share:
            lo = mid
        else:
            hi = mid
    tau_y = 0.5 * (lo + hi)
    return float(tau_y), float(gate_share(eps, err, tau_eps, tau_y))


def calibrate_tau_eps(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    tau_y: float,
    target_share: float,
    *,
    tol: float = 1e-4,
    maxiter: int = 60,
) -> tuple[float, float]:
    eps, err = _validate_gate_series(e_stat, f_stat)
    if not (0.0 < target_share < 1.0):
        raise ValueError("target_share must lie strictly between 0 and 1.")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")

    lo = float(np.min(eps))
    hi = float(np.max(eps))
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        share = gate_share(eps, err, mid, tau_y)
        if abs(share - target_share) < tol:
            return float(mid), float(share)
        if share > target_share:
            lo = mid
        else:
            hi = mid
    tau_eps = 0.5 * (lo + hi)
    return float(tau_eps), float(gate_share(eps, err, tau_eps, tau_y))


def apply_gate_padding(
    mask: Sequence[bool] | np.ndarray,
    k_pre: int,
    k_post: int,
    min_len: int,
) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    periods = int(base.size)
    expanded = np.zeros_like(base, dtype=bool)
    for period, active in enumerate(base):
        if not active:
            continue
        start = max(0, period - k_pre)
        stop = min(periods, period + k_post + 1)
        expanded[start:stop] = True

    if min_len <= 1:
        return expanded

    adjusted = np.zeros_like(expanded, dtype=bool)
    period = 0
    while period < periods:
        if not expanded[period]:
            period += 1
            continue
        start = period
        while period < periods and expanded[period]:
            period += 1
        stop = period - 1
        length = stop - start + 1
        if length < min_len:
            extra = min_len - length
            add_right = min(extra, periods - stop - 1)
            add_left = extra - add_right
            new_start = max(0, start - add_left)
            new_stop = min(periods - 1, stop + add_right)
            if new_stop - new_start + 1 < min_len:
                new_start = max(0, new_start - (min_len - (new_stop - new_start + 1)))
            start = new_start
            stop = new_stop
        adjusted[start : stop + 1] = True
    return adjusted


def apply_gate_padding_jax(
    mask: Sequence[bool] | jax.Array | np.ndarray,
    k_pre: int,
    k_post: int,
    min_len: int,
) -> jax.Array:
    base = jnp.asarray(mask, dtype=jnp.bool_).reshape(-1)
    periods = base.shape[0]
    if periods == 0:
        return base

    idx = jnp.arange(periods, dtype=jnp.int32)
    expanded = jnp.any(
        base[None, :]
        & (idx[:, None] >= (idx[None, :] - int(k_pre)))
        & (idx[:, None] <= (idx[None, :] + int(k_post))),
        axis=1,
    )
    if int(min_len) <= 1:
        return expanded

    prev_expanded = jnp.concatenate(
        [jnp.asarray([False], dtype=jnp.bool_), expanded[:-1]],
        axis=0,
    )
    next_expanded = jnp.concatenate(
        [expanded[1:], jnp.asarray([False], dtype=jnp.bool_)],
        axis=0,
    )
    start_flags = expanded & ~prev_expanded
    end_flags = expanded & ~next_expanded
    stop_candidates = jnp.where(
        (idx[None, :] >= idx[:, None]) & end_flags[None, :],
        idx[None, :],
        periods,
    )
    stop_idx = jnp.min(stop_candidates, axis=1)
    run_length = stop_idx - idx + 1
    extra = jnp.maximum(int(min_len) - run_length, 0)
    add_right = jnp.minimum(extra, periods - stop_idx - 1)
    add_left = extra - add_right
    new_start = jnp.maximum(0, idx - add_left)
    new_stop = jnp.minimum(periods - 1, stop_idx + add_right)
    current_len = new_stop - new_start + 1
    new_start = jnp.maximum(0, new_start - jnp.maximum(int(min_len) - current_len, 0))
    adjusted = jnp.any(
        start_flags[:, None]
        & (idx[None, :] >= new_start[:, None])
        & (idx[None, :] <= new_stop[:, None]),
        axis=0,
    )
    return adjusted


def assign_regimes(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    tau_eps_or_config: float | RegimeSwitchConfig,
    tau_y: Optional[float] = None,
    *,
    use_eps: bool = True,
    use_y: bool = True,
    k_pre: int = 0,
    k_post: int = 0,
    min_len: int = 1,
) -> np.ndarray:
    if isinstance(tau_eps_or_config, RegimeSwitchConfig):
        config = tau_eps_or_config
        return assign_regimes(
            e_stat,
            f_stat,
            config.tau_eps,
            config.tau_y,
            use_eps=config.use_eps,
            use_y=config.use_y,
            k_pre=config.k_pre,
            k_post=config.k_post,
            min_len=config.min_len,
        )
    if tau_y is None:
        raise ValueError("tau_y must be provided when using explicit thresholds.")
    base = _gate_mask(
        e_stat,
        f_stat,
        float(tau_eps_or_config),
        float(tau_y),
        use_eps=use_eps,
        use_y=use_y,
    )
    return apply_gate_padding(base, k_pre, k_post, min_len)


def assign_regimes_jax(
    e_stat: Sequence[float] | jax.Array | np.ndarray,
    f_stat: Sequence[float] | jax.Array | np.ndarray,
    tau_eps_or_config: float | RegimeSwitchConfig,
    tau_y: Optional[float] = None,
    *,
    use_eps: bool = True,
    use_y: bool = True,
    k_pre: int = 0,
    k_post: int = 0,
    min_len: int = 1,
) -> jax.Array:
    if isinstance(tau_eps_or_config, RegimeSwitchConfig):
        config = tau_eps_or_config
        return assign_regimes_jax(
            e_stat,
            f_stat,
            config.tau_eps,
            config.tau_y,
            use_eps=config.use_eps,
            use_y=config.use_y,
            k_pre=config.k_pre,
            k_post=config.k_post,
            min_len=config.min_len,
        )
    if tau_y is None:
        raise ValueError("tau_y must be provided when using explicit thresholds.")
    eps = jnp.asarray(e_stat, dtype=jnp.float64).reshape(-1)
    err = jnp.asarray(f_stat, dtype=jnp.float64).reshape(-1)
    if eps.shape != err.shape:
        raise ValueError(
            f"Gate statistics length mismatch: {eps.shape} and {err.shape}."
        )
    if not (use_eps or use_y):
        raise ValueError("At least one of use_eps/use_y must be true.")

    base = jnp.zeros_like(eps, dtype=jnp.bool_)
    if use_eps:
        base = base | (eps > float(tau_eps_or_config))
    if use_y:
        base = base | (err > float(tau_y))
    return apply_gate_padding_jax(base, k_pre, k_post, min_len)


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    if np.isscalar(x):
        return float(result)
    return result


def logistic_jax(x: float | jax.Array | np.ndarray) -> jax.Array:
    values = jnp.asarray(x, dtype=jnp.float64)
    positive = values >= 0.0
    return jnp.where(
        positive,
        1.0 / (1.0 + jnp.exp(-values)),
        jnp.exp(values) / (1.0 + jnp.exp(values)),
    )


def logit(p: float) -> float:
    probability = float(p)
    return float(np.log(probability / (1.0 - probability)))


def calibrate_gate_bias(
    scores: Sequence[float] | np.ndarray,
    target_share: float,
) -> float:
    if not (0.0 < target_share < 1.0):
        raise ValueError("target_share must lie strictly between 0 and 1.")
    values = np.asarray(scores, dtype=np.float64)
    lo = -20.0
    hi = 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        share = float(np.mean(logistic(mid + values)))
        if share > target_share:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def gate_probabilities(
    e_stat: Sequence[float] | np.ndarray,
    f_stat: Sequence[float] | np.ndarray,
    config: RegimeSwitchConfig,
) -> np.ndarray:
    eps, err = _validate_gate_series(e_stat, f_stat)
    if config.gate_mode == "hard":
        hard = assign_regimes(eps, err, config)
        probs = hard.astype(np.float64)
        return np.clip(probs, config.prob_floor, config.prob_ceiling)
    if config.gate_mode == "soft":
        scores = np.zeros_like(eps, dtype=np.float64)
        if config.use_eps:
            scores += config.beta_eps * (eps - config.tau_eps)
        if config.use_y:
            scores += config.beta_y * (err - config.tau_y)
        probs = np.asarray(logistic(config.bias + scores), dtype=np.float64)
        probs = np.clip(probs, config.prob_floor, config.prob_ceiling)
        hard = probs >= config.hard_threshold
        if config.k_pre > 0 or config.k_post > 0 or config.min_len > 1:
            _ = apply_gate_padding(hard, config.k_pre, config.k_post, config.min_len)
        return probs
    raise ValueError(
        f"Unknown gate_mode={config.gate_mode!r}. Use 'hard' or 'soft'."
    )


def gate_probabilities_jax(
    e_stat: Sequence[float] | jax.Array | np.ndarray,
    f_stat: Sequence[float] | jax.Array | np.ndarray,
    config: RegimeSwitchConfig,
) -> jax.Array:
    eps = jnp.asarray(e_stat, dtype=jnp.float64).reshape(-1)
    err = jnp.asarray(f_stat, dtype=jnp.float64).reshape(-1)
    if eps.shape != err.shape:
        raise ValueError(
            f"Gate statistics length mismatch: {eps.shape} and {err.shape}."
        )
    if config.gate_mode == "hard":
        hard = assign_regimes_jax(eps, err, config)
        probs = hard.astype(jnp.float64)
        return jnp.clip(probs, config.prob_floor, config.prob_ceiling)
    if config.gate_mode == "soft":
        scores = jnp.zeros_like(eps, dtype=jnp.float64)
        if config.use_eps:
            scores = scores + config.beta_eps * (eps - config.tau_eps)
        if config.use_y:
            scores = scores + config.beta_y * (err - config.tau_y)
        probs = logistic_jax(config.bias + scores)
        probs = jnp.clip(probs, config.prob_floor, config.prob_ceiling)
        return probs
    raise ValueError(
        f"Unknown gate_mode={config.gate_mode!r}. Use 'hard' or 'soft'."
    )


def compute_switching_loglikelihood(
    ll_rom: Sequence[float] | jax.Array | np.ndarray,
    ll_fom: Sequence[float] | jax.Array | np.ndarray,
    *,
    hard_mask: Optional[Sequence[bool] | jax.Array | np.ndarray] = None,
    gate_probs: Optional[Sequence[float] | jax.Array | np.ndarray] = None,
    config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> SwitchingLikelihoodResult:
    rom = jnp.asarray(ll_rom, dtype=jnp.float64)
    fom = jnp.asarray(ll_fom, dtype=jnp.float64)
    if rom.ndim != 1 or fom.ndim != 1:
        raise ValueError(
            f"ll_rom and ll_fom must be rank-1, got {rom.shape} and {fom.shape}."
        )
    if rom.shape != fom.shape:
        raise ValueError(
            f"ll_rom and ll_fom must have identical shapes, got {rom.shape} and {fom.shape}."
        )
    if hard_mask is None and gate_probs is None:
        raise ValueError("Provide either hard_mask or gate_probs.")

    if hard_mask is not None:
        mask = jnp.asarray(hard_mask, dtype=jnp.bool_)
        if mask.shape != rom.shape:
            raise ValueError(
                f"hard_mask must have shape {rom.shape}, got {mask.shape}."
            )
        probs = jnp.clip(
            mask.astype(jnp.float64),
            config.prob_floor,
            config.prob_ceiling,
        )
        per_period = jnp.where(mask, fom, rom)
        return SwitchingLikelihoodResult(
            total=jnp.sum(per_period),
            per_period=per_period,
            hard_mask=mask,
            gate_probs=probs,
            ll_rom=rom,
            ll_fom=fom,
        )

    probs = jnp.asarray(gate_probs, dtype=jnp.float64)
    if probs.shape != rom.shape:
        raise ValueError(
            f"gate_probs must have shape {rom.shape}, got {probs.shape}."
        )
    probs = jnp.clip(probs, config.prob_floor, config.prob_ceiling)
    mask = probs >= config.hard_threshold
    if config.soft_mixture == "linear":
        per_period = probs * fom + (1.0 - probs) * rom
    elif config.soft_mixture == "logsumexp":
        per_period = jnp.logaddexp(
            jnp.log(probs) + fom,
            jnp.log1p(-probs) + rom,
        )
    else:
        raise ValueError(
            f"Unknown soft_mixture={config.soft_mixture!r}. Use 'linear' or 'logsumexp'."
        )
    return SwitchingLikelihoodResult(
        total=jnp.sum(per_period),
        per_period=per_period,
        hard_mask=mask,
        gate_probs=probs,
        ll_rom=rom,
        ll_fom=fom,
    )


def mix_loglikelihood(
    ll_fom: Sequence[float] | jax.Array | np.ndarray,
    ll_rom: Sequence[float] | jax.Array | np.ndarray,
    gate_probs: Sequence[float] | jax.Array | np.ndarray,
    *,
    config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(
        gate_mode="soft",
        soft_mixture="logsumexp",
    ),
) -> jax.Array:
    return compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        gate_probs=gate_probs,
        config=config,
    ).total


def evaluate_switching_vs_fom(
    ll_switching: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    *,
    runtime_switching: Optional[float] = None,
    runtime_fom: Optional[float] = None,
) -> dict[str, float | int | None]:
    switching = np.asarray(ll_switching, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    if switching.shape != fom.shape:
        raise ValueError(
            "ll_switching and ll_fom must have identical shapes, got "
            f"{switching.shape} and {fom.shape}."
        )
    diffs = switching - fom
    abs_diffs = np.abs(diffs)
    denom = max(float(np.mean(np.abs(fom))), float(np.finfo(np.float64).eps))
    speedup = None
    if (
        runtime_switching is not None
        and runtime_fom is not None
        and float(runtime_switching) > 0.0
    ):
        speedup = float(runtime_fom) / float(runtime_switching)
    return {
        "n": int(switching.size),
        "switching_total": float(np.sum(switching)),
        "fom_total": float(np.sum(fom)),
        "total_diff": float(np.sum(switching) - np.sum(fom)),
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diffs**2))) if diffs.size else 0.0,
        "relative_mean_abs_diff": float(np.mean(abs_diffs) / denom),
        "runtime_switching_s": None
        if runtime_switching is None
        else float(runtime_switching),
        "runtime_fom_s": None if runtime_fom is None else float(runtime_fom),
        "speedup": speedup,
    }


def oracle_nonlinear_mask(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    *,
    gain_tol: float = 0.0,
) -> np.ndarray:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    if rom.shape != fom.shape:
        raise ValueError(
            "ll_rom and ll_fom must have identical shapes, got "
            f"{rom.shape} and {fom.shape}."
        )
    return np.asarray((fom - rom) > float(gain_tol), dtype=bool)


def optimal_nonlinear_mask_for_budget(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    nonlinear_periods: int,
) -> np.ndarray:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    if rom.shape != fom.shape:
        raise ValueError(
            "ll_rom and ll_fom must have identical shapes, got "
            f"{rom.shape} and {fom.shape}."
        )
    if nonlinear_periods < 0:
        raise ValueError(
            f"nonlinear_periods must be >= 0, got {nonlinear_periods}."
        )
    budget = min(int(nonlinear_periods), int(rom.size))
    mask = np.zeros((rom.size,), dtype=bool)
    if budget == 0 or rom.size == 0:
        return mask
    deltas = fom - rom
    ranked = np.argsort(-deltas, kind="mergesort")
    mask[ranked[:budget]] = True
    return mask


def _binary_classification_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    predicted_bool = np.asarray(predicted, dtype=bool).reshape(-1)
    target_bool = np.asarray(target, dtype=bool).reshape(-1)
    if predicted_bool.shape != target_bool.shape:
        raise ValueError(
            "predicted and target must have identical shapes, got "
            f"{predicted_bool.shape} and {target_bool.shape}."
        )
    tp = int(np.sum(predicted_bool & target_bool))
    tn = int(np.sum((~predicted_bool) & (~target_bool)))
    fp = int(np.sum(predicted_bool & (~target_bool)))
    fn = int(np.sum((~predicted_bool) & target_bool))
    positive = tp + fn
    negative = tn + fp
    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if positive == 0 else float(tp / positive)
    specificity = 0.0 if negative == 0 else float(tn / negative)
    accuracy = 0.0 if predicted_bool.size == 0 else float((tp + tn) / predicted_bool.size)
    f1 = 0.0 if (precision + recall) == 0.0 else float(
        2.0 * precision * recall / (precision + recall)
    )
    jaccard = 0.0 if (tp + fp + fn) == 0 else float(tp / (tp + fp + fn))
    balanced_accuracy = float(0.5 * (recall + specificity))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "jaccard": jaccard,
        "target_positives": positive,
        "predicted_positives": int(tp + fp),
    }


def _binary_auc(
    scores: np.ndarray,
    target: np.ndarray,
) -> Optional[float]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    target_bool = np.asarray(target, dtype=bool).reshape(-1)
    if score_array.shape != target_bool.shape:
        raise ValueError(
            "scores and target must have identical shapes, got "
            f"{score_array.shape} and {target_bool.shape}."
        )
    positive = int(np.sum(target_bool))
    negative = int(np.sum(~target_bool))
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score_array, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score_array.size + 1, dtype=np.float64)
    sorted_scores = score_array[order]
    start = 0
    while start < score_array.size:
        stop = start + 1
        while stop < score_array.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        if stop - start > 1:
            avg_rank = 0.5 * (start + 1 + stop)
            ranks[order[start:stop]] = avg_rank
        start = stop
    rank_sum_positive = float(np.sum(ranks[target_bool]))
    auc = (rank_sum_positive - positive * (positive + 1) / 2.0) / (positive * negative)
    return float(auc)


def _average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(value_array, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, value_array.size + 1, dtype=np.float64)
    sorted_values = value_array[order]
    start = 0
    while start < value_array.size:
        stop = start + 1
        while stop < value_array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            avg_rank = 0.5 * (start + 1 + stop)
            ranks[order[start:stop]] = avg_rank
        start = stop
    return ranks


def _pearson_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> Optional[float]:
    x_array = np.asarray(x, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError(
            f"x and y must have identical shapes, got {x_array.shape} and {y_array.shape}."
        )
    if x_array.size <= 1:
        return None
    centered_x = x_array - np.mean(x_array)
    centered_y = y_array - np.mean(y_array)
    denom = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    if denom <= float(np.finfo(np.float64).eps):
        return None
    return float(np.dot(centered_x, centered_y) / denom)


def evaluate_likelihood_surface_alignment(
    reference: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    top_share: float = 0.1,
) -> dict[str, float | int | bool | None]:
    reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "reference and candidate must have identical shapes, got "
            f"{reference_array.shape} and {candidate_array.shape}."
        )
    if not (0.0 < float(top_share) <= 1.0):
        raise ValueError("top_share must lie in (0, 1].")

    finite_mask = np.isfinite(reference_array) & np.isfinite(candidate_array)
    n_total = int(reference_array.size)
    n_finite = int(np.sum(finite_mask))
    if n_finite == 0:
        return {
            "n_total": n_total,
            "n_finite_pairs": 0,
            "top_share": float(top_share),
            "top_count": 0,
            "mean_error": None,
            "mean_abs_error": None,
            "rmse": None,
            "max_abs_error": None,
            "pearson_corr": None,
            "spearman_corr": None,
            "reference_std": None,
            "candidate_std": None,
            "std_ratio": None,
            "best_reference_index": None,
            "best_candidate_index": None,
            "best_draw_match": None,
            "top_overlap_count": None,
            "top_overlap_share": None,
        }

    reference_finite = reference_array[finite_mask]
    candidate_finite = candidate_array[finite_mask]
    finite_indices = np.flatnonzero(finite_mask)
    diffs = candidate_finite - reference_finite
    abs_diffs = np.abs(diffs)
    pearson_corr = _pearson_correlation(reference_finite, candidate_finite)
    spearman_corr = _pearson_correlation(
        _average_ranks(reference_finite),
        _average_ranks(candidate_finite),
    )
    reference_std = float(np.std(reference_finite))
    candidate_std = float(np.std(candidate_finite))
    std_ratio = (
        None
        if reference_std <= float(np.finfo(np.float64).eps)
        else float(candidate_std / reference_std)
    )
    best_reference_local = int(np.argmax(reference_finite))
    best_candidate_local = int(np.argmax(candidate_finite))
    best_reference_index = int(finite_indices[best_reference_local])
    best_candidate_index = int(finite_indices[best_candidate_local])
    top_count = max(1, int(np.ceil(float(top_share) * n_finite)))
    reference_order = np.argsort(-reference_finite, kind="mergesort")
    candidate_order = np.argsort(-candidate_finite, kind="mergesort")
    top_reference = set(finite_indices[reference_order[:top_count]].tolist())
    top_candidate = set(finite_indices[candidate_order[:top_count]].tolist())
    top_overlap_count = len(top_reference & top_candidate)

    return {
        "n_total": n_total,
        "n_finite_pairs": n_finite,
        "top_share": float(top_share),
        "top_count": int(top_count),
        "mean_error": float(np.mean(diffs)),
        "mean_abs_error": float(np.mean(abs_diffs)),
        "rmse": float(np.sqrt(np.mean(diffs**2))),
        "max_abs_error": float(np.max(abs_diffs)),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "reference_std": reference_std,
        "candidate_std": candidate_std,
        "std_ratio": std_ratio,
        "best_reference_index": best_reference_index,
        "best_candidate_index": best_candidate_index,
        "best_draw_match": bool(best_reference_index == best_candidate_index),
        "top_overlap_count": int(top_overlap_count),
        "top_overlap_share": float(top_overlap_count / top_count),
    }


def evaluate_switching_surface_alignment(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    ll_switching: Sequence[float] | np.ndarray,
    *,
    top_share: float = 0.1,
) -> dict[str, dict[str, float | int | bool | None] | float | None]:
    rom_vs_fom = evaluate_likelihood_surface_alignment(
        ll_fom,
        ll_rom,
        top_share=top_share,
    )
    switching_vs_fom = evaluate_likelihood_surface_alignment(
        ll_fom,
        ll_switching,
        top_share=top_share,
    )

    rom_mae = rom_vs_fom["mean_abs_error"]
    switching_mae = switching_vs_fom["mean_abs_error"]
    rom_rmse = rom_vs_fom["rmse"]
    switching_rmse = switching_vs_fom["rmse"]
    rom_corr = rom_vs_fom["pearson_corr"]
    switching_corr = switching_vs_fom["pearson_corr"]
    rom_rank = rom_vs_fom["spearman_corr"]
    switching_rank = switching_vs_fom["spearman_corr"]
    rom_top_overlap = rom_vs_fom["top_overlap_share"]
    switching_top_overlap = switching_vs_fom["top_overlap_share"]

    return {
        "rom_vs_fom": rom_vs_fom,
        "switching_vs_fom": switching_vs_fom,
        "mae_ratio_switching_over_rom": None
        if rom_mae in (None, 0.0) or switching_mae is None
        else float(switching_mae / rom_mae),
        "rmse_ratio_switching_over_rom": None
        if rom_rmse in (None, 0.0) or switching_rmse is None
        else float(switching_rmse / rom_rmse),
        "pearson_corr_gain": None
        if rom_corr is None or switching_corr is None
        else float(switching_corr - rom_corr),
        "spearman_corr_gain": None
        if rom_rank is None or switching_rank is None
        else float(switching_rank - rom_rank),
        "top_overlap_gain": None
        if rom_top_overlap is None or switching_top_overlap is None
        else float(switching_top_overlap - rom_top_overlap),
    }


def evaluate_gate_decisions(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
    *,
    gain_tol: float = 0.0,
) -> dict[str, float | int]:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    hard = np.asarray(mask, dtype=bool).reshape(-1)
    if rom.shape != fom.shape or rom.shape != hard.shape:
        raise ValueError(
            "ll_rom, ll_fom, and mask must have identical shapes, got "
            f"{rom.shape}, {fom.shape}, and {hard.shape}."
        )
    deltas = fom - rom
    oracle = oracle_nonlinear_mask(rom, fom, gain_tol=gain_tol)
    budget_oracle = optimal_nonlinear_mask_for_budget(rom, fom, int(np.sum(hard)))
    mixed = np.where(hard, fom, rom)
    oracle_mixed = np.where(oracle, fom, rom)
    budget_mixed = np.where(budget_oracle, fom, rom)
    positive_gain = np.maximum(deltas, 0.0)
    negative_cost = np.maximum(-deltas, 0.0)
    captured_gain = float(np.sum(positive_gain[hard]))
    total_positive_gain = float(np.sum(positive_gain))
    wasted_cost = float(np.sum(negative_cost[hard]))
    budget_metrics = _binary_classification_metrics(hard, budget_oracle)
    oracle_metrics = _binary_classification_metrics(hard, oracle)
    return {
        **{f"oracle_{key}": value for key, value in oracle_metrics.items()},
        **{f"budget_{key}": value for key, value in budget_metrics.items()},
        "periods_total": int(rom.size),
        "periods_nonlinear_selected": int(np.sum(hard)),
        "mixed_total": float(np.sum(mixed)),
        "oracle_total": float(np.sum(oracle_mixed)),
        "budget_oracle_total": float(np.sum(budget_mixed)),
        "regret_vs_oracle": float(np.sum(oracle_mixed) - np.sum(mixed)),
        "regret_vs_budget_oracle": float(np.sum(budget_mixed) - np.sum(mixed)),
        "total_positive_gain": total_positive_gain,
        "captured_positive_gain": captured_gain,
        "captured_gain_share": 0.0
        if total_positive_gain <= 0.0
        else float(captured_gain / total_positive_gain),
        "wasted_nonlinear_cost": wasted_cost,
    }


def evaluate_gate_probabilities(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    gate_probs: Sequence[float] | np.ndarray,
    *,
    gain_tol: float = 0.0,
    hard_threshold: float = 0.5,
) -> dict[str, float | int | None]:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    probs = np.asarray(gate_probs, dtype=np.float64).reshape(-1)
    if rom.shape != fom.shape or rom.shape != probs.shape:
        raise ValueError(
            "ll_rom, ll_fom, and gate_probs must have identical shapes, got "
            f"{rom.shape}, {fom.shape}, and {probs.shape}."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("gate_probs must contain only finite values.")
    oracle = oracle_nonlinear_mask(rom, fom, gain_tol=gain_tol)
    clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    oracle_float = oracle.astype(np.float64)
    brier = float(np.mean((probs - oracle_float) ** 2)) if probs.size else 0.0
    log_loss = float(
        -np.mean(
            oracle_float * np.log(clipped) + (1.0 - oracle_float) * np.log(1.0 - clipped)
        )
    ) if probs.size else 0.0
    auc = _binary_auc(probs, oracle)
    implied_mask = probs >= float(hard_threshold)
    decision = evaluate_gate_decisions(rom, fom, implied_mask, gain_tol=gain_tol)
    positive_probs = probs[oracle]
    negative_probs = probs[~oracle]
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "auc": auc,
        "mean_prob_oracle_positive": 0.0
        if positive_probs.size == 0
        else float(np.mean(positive_probs)),
        "mean_prob_oracle_negative": 0.0
        if negative_probs.size == 0
        else float(np.mean(negative_probs)),
        "hard_threshold": float(hard_threshold),
        **{f"hard_{key}": value for key, value in decision.items()},
    }


def _topk_mask_from_scores(
    scores: Sequence[float] | np.ndarray,
    budget: int,
) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.ndim != 1:
        raise ValueError("scores must be rank-1.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values.")
    if budget < 0 or budget > score_array.size:
        raise ValueError(
            f"budget must be between 0 and {score_array.size}, got {budget}."
        )
    mask = np.zeros(score_array.shape, dtype=bool)
    if budget == 0 or score_array.size == 0:
        return mask
    order = np.argsort(-score_array, kind="mergesort")
    mask[order[:budget]] = True
    return mask


def evaluate_gate_budget_frontier(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    budgets: Optional[Sequence[int] | np.ndarray] = None,
    gain_tol: float = 0.0,
) -> dict[str, float | np.ndarray]:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if rom.shape != fom.shape or rom.shape != score_array.shape:
        raise ValueError(
            "ll_rom, ll_fom, and scores must have identical shapes, got "
            f"{rom.shape}, {fom.shape}, and {score_array.shape}."
        )
    periods = int(rom.size)
    if budgets is None:
        budget_array = np.arange(periods + 1, dtype=np.int64)
    else:
        budget_array = np.asarray(budgets, dtype=np.int64).reshape(-1)
        if budget_array.size == 0:
            raise ValueError("budgets must contain at least one entry.")
        if np.any(budget_array < 0) or np.any(budget_array > periods):
            raise ValueError(
                f"budgets must be between 0 and {periods}, got {budget_array!r}."
            )
        budget_array = np.unique(np.sort(budget_array))

    selected_total = np.zeros(budget_array.shape, dtype=np.float64)
    oracle_total = np.zeros(budget_array.shape, dtype=np.float64)
    regret = np.zeros(budget_array.shape, dtype=np.float64)
    captured_gain_share = np.zeros(budget_array.shape, dtype=np.float64)
    selected_share = np.zeros(budget_array.shape, dtype=np.float64)

    for idx, budget in enumerate(budget_array):
        mask = _topk_mask_from_scores(score_array, int(budget))
        metrics = evaluate_gate_decisions(rom, fom, mask, gain_tol=gain_tol)
        selected_total[idx] = float(metrics["mixed_total"])
        oracle_total[idx] = float(metrics["budget_oracle_total"])
        regret[idx] = float(metrics["regret_vs_budget_oracle"])
        captured_gain_share[idx] = float(metrics["captured_gain_share"])
        selected_share[idx] = 0.0 if periods == 0 else float(budget) / float(periods)

    if budget_array.size <= 1:
        area_regret = float(regret[0]) if regret.size else 0.0
        area_capture = float(captured_gain_share[0]) if captured_gain_share.size else 0.0
    else:
        area_regret = float(np.trapezoid(regret, x=selected_share))
        area_capture = float(np.trapezoid(captured_gain_share, x=selected_share))

    return {
        "budgets": budget_array,
        "selected_share": selected_share,
        "selected_total": selected_total,
        "budget_oracle_total": oracle_total,
        "regret_vs_budget_oracle": regret,
        "captured_gain_share": captured_gain_share,
        "mean_regret_vs_budget_oracle": 0.0 if regret.size == 0 else float(np.mean(regret)),
        "max_regret_vs_budget_oracle": 0.0 if regret.size == 0 else float(np.max(regret)),
        "area_regret_vs_budget_oracle": area_regret,
        "area_captured_gain_share": area_capture,
    }


def _gate_segments(mask: Sequence[bool] | np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    segments: list[tuple[int, int]] = []
    period = 0
    while period < values.size:
        if not values[period]:
            period += 1
            continue
        start = period + 1
        while period < values.size and values[period]:
            period += 1
        segments.append((start, period))
    return tuple(segments)


def contiguous_true_runs(mask: Sequence[bool] | np.ndarray) -> tuple[range, ...]:
    return tuple(range(start, stop + 1) for start, stop in _gate_segments(mask))


def choose_gated_run(
    runs: Sequence[range],
    strategy: str,
) -> Optional[range]:
    if not runs:
        return None
    if strategy == "first":
        return runs[0]
    if strategy == "last":
        return runs[-1]
    if strategy == "longest":
        return max(runs, key=len)
    raise ValueError(
        f"Unsupported gated block strategy {strategy!r}. "
        "Use 'first', 'last', or 'longest'."
    )


def select_gated_block_periods(
    gate_mask: Sequence[bool] | np.ndarray,
    strategy: str,
    context_periods: int,
    max_eval_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    runs = contiguous_true_runs(gate_mask)
    if not runs:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty, "No gated periods found."
    run = choose_gated_run(runs, strategy)
    assert run is not None
    eval_periods = np.asarray(list(run), dtype=np.int64)
    if max_eval_periods > 0 and eval_periods.size > max_eval_periods:
        eval_periods = eval_periods[:max_eval_periods]
    context = np.asarray([], dtype=np.int64)
    if context_periods > 0 and eval_periods.size > 0 and eval_periods[0] > 1:
        start = max(1, int(eval_periods[0]) - int(context_periods))
        context = np.arange(start, int(eval_periods[0]), dtype=np.int64)
    selected = np.concatenate([context, eval_periods])
    note = f"Selected {strategy} block {run.start}:{run.stop - 1}"
    if context.size:
        note += f" with context {context[0]}:{context[-1]}"
    return selected, eval_periods, context, note


def compute_gate_stats(mask: Sequence[bool] | np.ndarray) -> dict[str, float | int]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    segments = _gate_segments(values)
    lengths = np.asarray([stop - start + 1 for start, stop in segments], dtype=np.int64)
    nonlinear = int(np.sum(values))
    total = int(values.size)
    return {
        "periods_total": total,
        "periods_nonlinear": nonlinear,
        "periods_linear": total - nonlinear,
        "share_nonlinear": 0.0 if total == 0 else float(nonlinear / total),
        "episodes": len(segments),
        "max_episode_len": int(np.max(lengths)) if lengths.size else 0,
        "min_episode_len": int(np.min(lengths)) if lengths.size else 0,
        "mean_episode_len": 0.0 if lengths.size == 0 else float(np.mean(lengths)),
    }


def episode_overlap(
    mask: Sequence[bool] | np.ndarray,
    window_start: int,
    window_end: int,
) -> dict[str, float | int]:
    if window_end < window_start:
        raise ValueError("window_end must be >= window_start.")
    values = np.asarray(mask, dtype=bool).reshape(-1)
    total = int(values.size)
    if total == 0:
        return {
            "window_start": int(window_start),
            "window_end": int(window_end),
            "window_periods": 0,
            "nonlinear_in_window": 0,
            "share_window_nonlinear": 0.0,
            "share_nonlinear_inside_window": 0.0,
        }
    lo = int(np.clip(window_start, 1, max(total, 1)))
    hi = int(np.clip(window_end, 1, max(total, 1)))
    if hi < lo:
        return {
            "window_start": lo,
            "window_end": hi,
            "window_periods": 0,
            "nonlinear_in_window": 0,
            "share_window_nonlinear": 0.0,
            "share_nonlinear_inside_window": 0.0,
        }
    in_window = values[lo - 1 : hi]
    nonlinear_in_window = int(np.sum(in_window))
    nonlinear_total = int(np.sum(values))
    window_periods = int(hi - lo + 1)
    return {
        "window_start": lo,
        "window_end": hi,
        "window_periods": window_periods,
        "nonlinear_in_window": nonlinear_in_window,
        "share_window_nonlinear": float(nonlinear_in_window / window_periods),
        "share_nonlinear_inside_window": 0.0
        if nonlinear_total == 0
        else float(nonlinear_in_window / nonlinear_total),
    }


def summarize_loglik_decomposition(
    ll_rom: Sequence[float] | np.ndarray,
    ll_fom: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
) -> dict[str, float | int]:
    rom = np.asarray(ll_rom, dtype=np.float64).reshape(-1)
    fom = np.asarray(ll_fom, dtype=np.float64).reshape(-1)
    hard = np.asarray(mask, dtype=bool).reshape(-1)
    if rom.shape != fom.shape or rom.shape != hard.shape:
        raise ValueError(
            "ll_rom, ll_fom, and mask must have identical shapes, got "
            f"{rom.shape}, {fom.shape}, and {hard.shape}."
        )
    mixed = np.where(hard, fom, rom)
    return {
        "ll_rom_total": float(np.sum(rom)),
        "ll_fom_total": float(np.sum(fom)),
        "ll_mixed_total": float(np.sum(mixed)),
        "ll_rom_linear_periods": float(np.sum(rom[~hard])),
        "ll_rom_nonlinear_periods": float(np.sum(rom[hard])),
        "ll_fom_linear_periods": float(np.sum(fom[~hard])),
        "ll_fom_nonlinear_periods": float(np.sum(fom[hard])),
        "periods_nonlinear": int(np.sum(hard)),
        "periods_total": int(hard.size),
    }


def summarize_runtime(
    *,
    runtime_switching_s: Optional[float] = None,
    runtime_fom_s: Optional[float] = None,
) -> dict[str, float | None]:
    speedup = None
    if (
        runtime_switching_s is not None
        and runtime_fom_s is not None
        and float(runtime_switching_s) > 0.0
    ):
        speedup = float(runtime_fom_s) / float(runtime_switching_s)
    return {
        "runtime_switching_s": None
        if runtime_switching_s is None
        else float(runtime_switching_s),
        "runtime_fom_s": None if runtime_fom_s is None else float(runtime_fom_s),
        "speedup": speedup,
    }
