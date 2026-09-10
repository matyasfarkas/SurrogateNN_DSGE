from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


class StaticHMCState(NamedTuple):
    position: Any
    log_prob: Any
    grad_log_prob: Any


class StaticHMCResult(NamedTuple):
    samples: Any
    log_prob: Any
    accept_prob: Any
    accepted: Any
    final_position: Any
    final_log_prob: Any
    final_grad_log_prob: Any
    step_size: Any
    warmup_accept_prob: Any
    warmup_accepted: Any
    warmup_step_size: Any


def unconstrained_to_bounded(unconstrained: Any, lower: Any, upper: Any) -> Any:
    """Map unconstrained real parameters into open bounded intervals."""
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    return lower + (upper - lower) * jax.nn.sigmoid(unconstrained)


def bounded_to_unconstrained(
    bounded: Any,
    lower: Any,
    upper: Any,
    *,
    eps: float = 1.0e-12,
) -> Any:
    """Map bounded parameters to unconstrained real coordinates."""
    bounded = jnp.asarray(bounded)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    unit = (bounded - lower) / (upper - lower)
    eps_array = jnp.maximum(
        jnp.asarray(eps, dtype=unit.dtype),
        jnp.asarray(jnp.finfo(unit.dtype).eps, dtype=unit.dtype),
    )
    unit = jnp.clip(unit, eps_array, 1.0 - eps_array)
    return jnp.log(unit) - jnp.log1p(-unit)


def bounded_log_abs_det_jacobian(unconstrained: Any, lower: Any, upper: Any) -> Any:
    """Log absolute Jacobian determinant for unconstrained_to_bounded."""
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    unconstrained = jnp.asarray(unconstrained)
    log_sigmoid = -jax.nn.softplus(-unconstrained)
    log_one_minus_sigmoid = -jax.nn.softplus(unconstrained)
    return jnp.sum(jnp.log(upper - lower) + log_sigmoid + log_one_minus_sigmoid)


def _kinetic_energy(momentum: Any, inverse_mass: Any) -> Any:
    return 0.5 * jnp.sum(momentum * momentum * inverse_mass, axis=-1)


def _momentum_like(
    key: Any,
    position: Any,
    inverse_mass: Any,
) -> Any:
    standard_normal = jax.random.normal(key, shape=position.shape, dtype=position.dtype)
    return standard_normal / jnp.sqrt(inverse_mass)


def _sanitize_accept_prob(log_accept_ratio: Any) -> Any:
    accept_prob = jnp.exp(jnp.minimum(log_accept_ratio, 0.0))
    return jnp.where(jnp.isfinite(accept_prob), accept_prob, 0.0)


def static_hmc_sample(
    log_prob_fn: Callable[[Any], Any],
    initial_position: Any,
    key: Any,
    *,
    num_warmup: int,
    num_samples: int,
    step_size: float,
    num_leapfrog_steps: int,
    inverse_mass: Any | None = None,
    target_accept_prob: float = 0.8,
    adapt_step_size: bool = True,
    adaptation_rate: float = 0.05,
    min_step_size: float = 1.0e-5,
    max_step_size: float = 1.0,
) -> StaticHMCResult:
    """Run fixed-shape HMC with chains vectorized in the leading dimension.

    The implementation intentionally avoids NumPyro's general MCMC machinery:
    chains are updated with `vmap`, warmup/sampling are `scan`s, and the
    leapfrog length is fixed. This makes the compiled graph more predictable
    for GPU benchmarking.
    """
    position = jnp.asarray(initial_position)
    if position.ndim != 2:
        raise ValueError("initial_position must have shape (num_chains, num_parameters)")
    if num_warmup < 0 or num_samples < 0:
        raise ValueError("num_warmup and num_samples must be non-negative")
    if num_leapfrog_steps <= 0:
        raise ValueError("num_leapfrog_steps must be positive")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if min_step_size <= 0.0 or max_step_size <= min_step_size:
        raise ValueError("step-size bounds must satisfy 0 < min < max")

    if inverse_mass is None:
        inverse_mass_array = jnp.ones((position.shape[-1],), dtype=position.dtype)
    else:
        inverse_mass_array = jnp.asarray(inverse_mass, dtype=position.dtype)
        if inverse_mass_array.shape != (position.shape[-1],):
            raise ValueError("inverse_mass must have shape (num_parameters,)")

    value_and_grad = jax.value_and_grad(log_prob_fn)
    batched_value_and_grad = jax.vmap(value_and_grad)
    initial_log_prob, initial_grad = batched_value_and_grad(position)
    state = StaticHMCState(position, initial_log_prob, initial_grad)
    log_min_step = jnp.log(jnp.asarray(min_step_size, dtype=position.dtype))
    log_max_step = jnp.log(jnp.asarray(max_step_size, dtype=position.dtype))
    initial_log_step = jnp.log(jnp.asarray(step_size, dtype=position.dtype))
    initial_log_step = jnp.clip(initial_log_step, log_min_step, log_max_step)

    def transition(
        current_state: StaticHMCState,
        transition_key: Any,
        transition_step_size: Any,
    ) -> tuple[StaticHMCState, tuple[Any, Any]]:
        momentum_key, accept_key = jax.random.split(transition_key)
        initial_momentum = _momentum_like(
            momentum_key,
            current_state.position,
            inverse_mass_array,
        )
        initial_hamiltonian = (
            -current_state.log_prob
            + _kinetic_energy(initial_momentum, inverse_mass_array)
        )
        momentum = initial_momentum + 0.5 * transition_step_size * current_state.grad_log_prob
        position_candidate = current_state.position

        def leapfrog_step(
            carry: tuple[Any, Any, Any, Any],
            step_index: Any,
        ) -> tuple[tuple[Any, Any, Any, Any], None]:
            candidate_position, candidate_momentum, _, _ = carry
            candidate_position = (
                candidate_position
                + transition_step_size * inverse_mass_array * candidate_momentum
            )
            candidate_log_prob, candidate_grad = batched_value_and_grad(candidate_position)
            full_step_momentum = candidate_momentum + transition_step_size * candidate_grad
            candidate_momentum = jnp.where(
                step_index < num_leapfrog_steps - 1,
                full_step_momentum,
                candidate_momentum,
            )
            return (
                candidate_position,
                candidate_momentum,
                candidate_log_prob,
                candidate_grad,
            ), None

        position_candidate, momentum, log_prob_candidate, grad_candidate = jax.lax.scan(
            leapfrog_step,
            (
                position_candidate,
                momentum,
                current_state.log_prob,
                current_state.grad_log_prob,
            ),
            jnp.arange(num_leapfrog_steps),
        )[0]
        momentum = momentum + 0.5 * transition_step_size * grad_candidate
        proposed_hamiltonian = -log_prob_candidate + _kinetic_energy(momentum, inverse_mass_array)
        log_accept_ratio = initial_hamiltonian - proposed_hamiltonian
        accept_prob = _sanitize_accept_prob(log_accept_ratio)
        accepted = jax.random.uniform(
            accept_key,
            shape=accept_prob.shape,
            dtype=position_candidate.dtype,
        ) < accept_prob
        next_position = jnp.where(accepted[:, None], position_candidate, current_state.position)
        next_log_prob = jnp.where(accepted, log_prob_candidate, current_state.log_prob)
        next_grad = jnp.where(accepted[:, None], grad_candidate, current_state.grad_log_prob)
        return StaticHMCState(next_position, next_log_prob, next_grad), (accept_prob, accepted)

    def warmup_step(
        carry: tuple[StaticHMCState, Any, Any],
        step_index: Any,
    ) -> tuple[tuple[StaticHMCState, Any, Any], tuple[Any, Any, Any]]:
        current_state, log_step_size, current_key = carry
        current_key, transition_key = jax.random.split(current_key)
        next_state, (accept_prob, accepted) = transition(
            current_state,
            transition_key,
            jnp.exp(log_step_size),
        )
        if adapt_step_size:
            rate = jnp.asarray(adaptation_rate, dtype=position.dtype) / jnp.sqrt(
                jnp.asarray(step_index + 1, dtype=position.dtype)
            )
            log_step_size = jnp.clip(
                log_step_size + rate * (jnp.mean(accept_prob) - target_accept_prob),
                log_min_step,
                log_max_step,
            )
        return (next_state, log_step_size, current_key), (
            accept_prob,
            accepted,
            jnp.exp(log_step_size),
        )

    warmup_key, sample_key = jax.random.split(key)
    if num_warmup:
        (state, final_log_step, _), warmup_info = jax.lax.scan(
            warmup_step,
            (state, initial_log_step, warmup_key),
            jnp.arange(num_warmup),
        )
        warmup_accept_prob, warmup_accepted, warmup_step_size = warmup_info
    else:
        final_log_step = initial_log_step
        warmup_accept_prob = jnp.zeros(
            (0, position.shape[0]),
            dtype=position.dtype,
        )
        warmup_accepted = jnp.zeros((0, position.shape[0]), dtype=bool)
        warmup_step_size = jnp.zeros((0,), dtype=position.dtype)

    final_step_size = jnp.exp(final_log_step)

    def sample_step(
        carry: tuple[StaticHMCState, Any],
        _: Any,
    ) -> tuple[tuple[StaticHMCState, Any], tuple[Any, Any, Any, Any]]:
        current_state, current_key = carry
        current_key, transition_key = jax.random.split(current_key)
        next_state, (accept_prob, accepted) = transition(
            current_state,
            transition_key,
            final_step_size,
        )
        return (next_state, current_key), (
            next_state.position,
            next_state.log_prob,
            accept_prob,
            accepted,
        )

    (final_state, _), sample_info = jax.lax.scan(
        sample_step,
        (state, sample_key),
        None,
        length=num_samples,
    )
    samples, log_prob, accept_prob, accepted = sample_info
    return StaticHMCResult(
        samples=samples,
        log_prob=log_prob,
        accept_prob=accept_prob,
        accepted=accepted,
        final_position=final_state.position,
        final_log_prob=final_state.log_prob,
        final_grad_log_prob=final_state.grad_log_prob,
        step_size=final_step_size,
        warmup_accept_prob=warmup_accept_prob,
        warmup_accepted=warmup_accepted,
        warmup_step_size=warmup_step_size,
    )
