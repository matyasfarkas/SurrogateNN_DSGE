from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    discrete_lyapunov_residual,
    solve_discrete_lyapunov,
    solve_discrete_lyapunov_direct,
    solve_discrete_lyapunov_doubling,
    solve_lyapunov_equation,
)


def test_scalar_closed_form_matches_both_algorithms() -> None:
    a = jnp.array([[0.6]])
    c = jnp.array([[2.0]])
    expected = c / (1.0 - a**2)

    direct = solve_discrete_lyapunov_direct(a, c)
    doubling = solve_discrete_lyapunov_doubling(a, c)

    np.testing.assert_allclose(direct.solution, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(doubling.solution, expected, rtol=1e-10, atol=1e-10)
    assert bool(np.asarray(direct.converged))
    assert bool(np.asarray(doubling.converged))


def test_doubling_solver_has_small_residual_and_preserves_symmetry() -> None:
    a = jnp.array([[0.2, -0.1], [0.05, 0.3]])
    c = jnp.array([[1.0, 0.2], [0.2, 0.5]])

    result = solve_discrete_lyapunov_doubling(a, c)

    assert bool(np.asarray(result.converged))
    assert float(np.asarray(result.relative_residual)) < 1e-12
    np.testing.assert_allclose(result.solution, result.solution.T, rtol=1e-12, atol=1e-12)


def test_wrapper_falls_back_to_direct_when_doubling_is_cut_short() -> None:
    a = jnp.array([[0.25, 0.1], [0.0, 0.35]])
    c = jnp.array([[0.8, 0.1], [0.1, 0.4]])

    outcome = solve_discrete_lyapunov(
        a,
        c,
        algorithm="doubling",
        max_iter=1,
        tol=1e-30,
        fallback_to_direct=True,
    )

    assert outcome.algorithm == "direct"
    assert outcome.fallback_used
    assert outcome.converged
    assert outcome.relative_residual < 1e-12


def test_direct_solver_is_autodiff_friendly() -> None:
    c = jnp.array([[1.5]])

    def objective(a_scalar: jax.Array) -> jax.Array:
        a = jnp.array([[a_scalar]])
        return solve_discrete_lyapunov_direct(a, c).solution[0, 0]

    grad_value = jax.grad(objective)(jnp.array(0.4))
    expected = (2.0 * 0.4 * 1.5) / (1.0 - 0.4**2) ** 2

    np.testing.assert_allclose(grad_value, expected, rtol=1e-10, atol=1e-10)


def test_doubling_kernel_is_jittable() -> None:
    a = jnp.array([[0.3, 0.1], [0.0, 0.25]])
    c = jnp.array([[0.5, 0.0], [0.0, 0.25]])

    compiled = jax.jit(solve_discrete_lyapunov_doubling)
    result = compiled(a, c)

    assert bool(np.asarray(result.converged))
    assert float(np.asarray(result.relative_residual)) < 1e-12


def test_public_alias_matches_wrapper() -> None:
    a = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    c = jnp.eye(2)

    direct = solve_discrete_lyapunov(a, c, algorithm="direct")
    alias = solve_lyapunov_equation(a, c, algorithm="direct")

    np.testing.assert_allclose(alias.solution, direct.solution, rtol=1e-12, atol=1e-12)
    assert alias.converged == direct.converged
    assert alias.algorithm == direct.algorithm


def test_residual_helper_matches_solver_output() -> None:
    a = jnp.array([[0.2]])
    c = jnp.array([[0.75]])
    result = solve_discrete_lyapunov_direct(a, c)

    residual = discrete_lyapunov_residual(a, result.solution, c)

    np.testing.assert_allclose(
        residual,
        result.relative_residual,
        rtol=1e-12,
        atol=1e-12,
    )
