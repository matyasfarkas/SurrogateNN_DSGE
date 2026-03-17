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


@dataclass(frozen=True)
class SwitchingLikelihoodConfig:
    gate_mode: str = "hard"
    hard_threshold: float = 0.5
    prob_floor: float = 1e-4
    prob_ceiling: float = 1.0 - 1e-4
    soft_mixture: str = "logsumexp"


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
