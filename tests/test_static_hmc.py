from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    bounded_log_abs_det_jacobian,
    bounded_to_unconstrained,
    static_hmc_sample,
    unconstrained_to_bounded,
)


def test_bounded_transform_round_trips_and_matches_jacobian() -> None:
    lower = jnp.asarray([0.1, -2.0], dtype=jnp.float64)
    upper = jnp.asarray([1.1, 3.0], dtype=jnp.float64)
    bounded = jnp.asarray([0.4, 0.25], dtype=jnp.float64)

    unconstrained = bounded_to_unconstrained(bounded, lower, upper)
    recovered = unconstrained_to_bounded(unconstrained, lower, upper)

    np.testing.assert_allclose(
        np.asarray(recovered),
        np.asarray(bounded),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    jacobian = jax.jacobian(lambda z: unconstrained_to_bounded(z, lower, upper))(
        unconstrained
    )
    expected = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))
    actual = bounded_log_abs_det_jacobian(unconstrained, lower, upper)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_static_hmc_jits_and_updates_parallel_chains() -> None:
    def log_prob(position):
        return -0.5 * jnp.sum(position * position)

    initial = jnp.zeros((4, 2), dtype=jnp.float64)
    sampler = jax.jit(
        lambda key: static_hmc_sample(
            log_prob,
            initial,
            key,
            num_warmup=8,
            num_samples=12,
            step_size=0.1,
            num_leapfrog_steps=4,
            target_accept_prob=0.75,
        )
    )

    result = sampler(jax.random.PRNGKey(123))

    assert result.samples.shape == (12, 4, 2)
    assert result.log_prob.shape == (12, 4)
    assert result.accept_prob.shape == (12, 4)
    assert result.accepted.shape == (12, 4)
    assert result.warmup_accept_prob.shape == (8, 4)
    assert result.warmup_step_size.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(result.samples)))
    assert bool(jnp.all(jnp.isfinite(result.log_prob)))
    assert bool(jnp.all((result.accept_prob >= 0.0) & (result.accept_prob <= 1.0)))
    assert bool(jnp.any(result.accepted))
    assert float(result.step_size) > 0.0


def test_static_hmc_validates_static_inputs() -> None:
    def log_prob(position):
        return -0.5 * jnp.sum(position * position)

    with pytest.raises(ValueError, match="initial_position"):
        static_hmc_sample(
            log_prob,
            jnp.zeros((2,), dtype=jnp.float64),
            jax.random.PRNGKey(0),
            num_warmup=1,
            num_samples=1,
            step_size=0.1,
            num_leapfrog_steps=2,
        )

    with pytest.raises(ValueError, match="num_leapfrog_steps"):
        static_hmc_sample(
            log_prob,
            jnp.zeros((2, 1), dtype=jnp.float64),
            jax.random.PRNGKey(0),
            num_warmup=1,
            num_samples=1,
            step_size=0.1,
            num_leapfrog_steps=0,
        )
