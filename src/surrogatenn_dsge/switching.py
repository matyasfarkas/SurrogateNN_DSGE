from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import jax
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
