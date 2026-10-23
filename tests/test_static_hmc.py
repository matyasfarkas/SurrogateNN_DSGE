from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    bounded_log_abs_det_jacobian,
    bounded_to_unconstrained,
    static_hmc_sample,
    surrogate_trajectory_hmc_sample,
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


def test_surrogate_trajectory_hmc_matches_static_hmc_when_densities_match() -> None:
    def log_prob(position):
        return -0.5 * jnp.sum(position * position)

    initial = jnp.asarray(
        [[0.1, -0.2], [0.3, 0.4]],
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(917)
    kwargs = {
        "num_warmup": 3,
        "num_samples": 5,
        "step_size": 0.08,
        "num_leapfrog_steps": 3,
        "adapt_step_size": False,
    }

    exact = jax.jit(lambda run_key: static_hmc_sample(log_prob, initial, run_key, **kwargs))(
        key
    )
    corrected = jax.jit(
        lambda run_key: surrogate_trajectory_hmc_sample(
            log_prob,
            log_prob,
            initial,
            run_key,
            **kwargs,
        )
    )(key)

    np.testing.assert_allclose(corrected.samples, exact.samples, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(corrected.log_prob, exact.log_prob, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        corrected.accept_prob,
        exact.accept_prob,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(corrected.accepted, exact.accepted)


def test_surrogate_trajectory_hmc_uses_exact_target_for_acceptance() -> None:
    def target_log_prob(position):
        return -0.5 * jnp.sum(position * position)

    def shifted_proposal_log_prob(position):
        return -0.5 * jnp.sum((position - 1.5) ** 2)

    initial = jnp.zeros((3, 1), dtype=jnp.float64)
    sampler = jax.jit(
        lambda run_key: surrogate_trajectory_hmc_sample(
            target_log_prob,
            shifted_proposal_log_prob,
            initial,
            run_key,
            num_warmup=4,
            num_samples=7,
            step_size=0.1,
            num_leapfrog_steps=2,
        )
    )

    result = sampler(jax.random.PRNGKey(22))

    assert result.samples.shape == (7, 3, 1)
    assert result.log_prob.shape == (7, 3)
    assert result.proposal_log_prob.shape == (7, 3)
    assert result.final_grad_proposal_log_prob.shape == (3, 1)
    assert bool(jnp.all(jnp.isfinite(result.samples)))
    np.testing.assert_allclose(
        result.log_prob,
        -0.5 * jnp.sum(result.samples**2, axis=-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
