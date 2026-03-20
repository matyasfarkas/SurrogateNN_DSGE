from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    discrete_sylvester_residual,
    solve_discrete_sylvester,
    solve_discrete_sylvester_direct,
    solve_discrete_sylvester_doubling,
    solve_sylvester_equation,
)


def test_scalar_closed_form_matches_both_algorithms() -> None:
    a = jnp.array([[0.4]])
    b = jnp.array([[0.25]])
    c = jnp.array([[2.5]])
    expected = c / (1.0 - a * b)

    direct = solve_discrete_sylvester_direct(a, b, c)
    doubling = solve_discrete_sylvester_doubling(a, b, c)

    np.testing.assert_allclose(direct.solution, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(doubling.solution, expected, rtol=1e-10, atol=1e-10)
    assert bool(np.asarray(direct.converged))
    assert bool(np.asarray(doubling.converged))


def test_doubling_solver_has_small_residual() -> None:
    a = jnp.array([[0.2, -0.05], [0.1, 0.25]])
    b = jnp.array([[0.15, 0.02], [0.0, 0.1]])
    c = jnp.array([[1.0, 0.3], [-0.2, 0.5]])

    result = solve_discrete_sylvester_doubling(a, b, c)

    assert bool(np.asarray(result.converged))
    assert float(np.asarray(result.relative_residual)) < 1e-10


def test_wrapper_reuses_exact_initial_guess() -> None:
    a = jnp.array([[0.2]])
    b = jnp.array([[0.3]])
    c = jnp.array([[0.9]])
    exact = c / (1.0 - a * b)

    outcome = solve_discrete_sylvester(
        a,
        b,
        c,
        initial_guess=exact,
        algorithm="doubling",
    )

    np.testing.assert_allclose(outcome.solution, exact, rtol=1e-12, atol=1e-12)
    assert outcome.converged
    assert outcome.iterations == 0
    assert not outcome.fallback_used


def test_wrapper_falls_back_to_direct_when_doubling_is_cut_short() -> None:
    a = jnp.array([[0.3, 0.1], [0.0, 0.2]])
    b = jnp.array([[0.2, 0.0], [0.05, 0.1]])
    c = jnp.array([[0.7, 0.1], [0.2, 0.4]])

    outcome = solve_discrete_sylvester(
        a,
        b,
        c,
        algorithm="doubling",
        max_iter=1,
        tol=1e-30,
        fallback_to_direct=True,
    )

    assert outcome.algorithm == "direct"
    assert outcome.fallback_used
    assert outcome.converged
    assert outcome.relative_residual < 1e-10


def test_direct_solver_is_autodiff_friendly() -> None:
    b = jnp.array([[0.5]])
    c = jnp.array([[1.2]])

    def objective(a_scalar: jax.Array) -> jax.Array:
        a = jnp.array([[a_scalar]])
        return solve_discrete_sylvester_direct(a, b, c).solution[0, 0]

    grad_value = jax.grad(objective)(jnp.array(0.4))
    expected = (1.2 * 0.5) / (1.0 - 0.4 * 0.5) ** 2

    np.testing.assert_allclose(grad_value, expected, rtol=1e-10, atol=1e-10)


def test_iterative_sylvester_algorithms_match_direct_solution() -> None:
    a = jnp.array([[0.2, 0.05], [0.0, 0.3]])
    b = jnp.array([[0.15, 0.02], [0.0, 0.1]])
    c = jnp.array([[0.8, 0.1], [0.2, 0.4]])
    direct = solve_discrete_sylvester_direct(a, b, c)

    for algorithm in ("bicgstab", "gmres", "dqgmres"):
        outcome = solve_discrete_sylvester(
            a,
            b,
            c,
            algorithm=algorithm,
            tol=1e-12,
            acceptance_tol=1e-10,
            max_iter=200,
        )
        assert outcome.algorithm == algorithm
        assert outcome.converged
        assert outcome.relative_residual < 1e-10
        np.testing.assert_allclose(
            outcome.solution,
            direct.solution,
            rtol=1e-9,
            atol=1e-9,
        )


def test_iterative_sylvester_falls_back_to_direct_when_cut_short() -> None:
    a = jnp.array([[0.3, 0.1], [0.0, 0.2]])
    b = jnp.array([[0.2, 0.0], [0.05, 0.1]])
    c = jnp.array([[0.7, 0.1], [0.2, 0.4]])

    outcome = solve_discrete_sylvester(
        a,
        b,
        c,
        algorithm="dqgmres",
        max_iter=1,
        tol=1e-30,
        fallback_to_direct=True,
    )

    assert outcome.algorithm == "direct"
    assert outcome.fallback_used
    assert outcome.converged
    assert outcome.relative_residual < 1e-10


def test_doubling_kernel_is_jittable() -> None:
    a = jnp.array([[0.2, 0.0], [0.1, 0.15]])
    b = jnp.array([[0.1, 0.05], [0.0, 0.2]])
    c = jnp.array([[0.5, 0.1], [0.0, 0.3]])

    compiled = jax.jit(solve_discrete_sylvester_doubling)
    result = compiled(a, b, c)

    assert bool(np.asarray(result.converged))
    assert float(np.asarray(result.relative_residual)) < 1e-10


def test_public_alias_matches_wrapper() -> None:
    a = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    b = jnp.array([[0.2, 0.0], [0.0, 0.1]])
    c = jnp.array([[1.0, 0.2], [0.1, 0.8]])

    direct = solve_discrete_sylvester(a, b, c, algorithm="direct")
    alias = solve_sylvester_equation(a, b, c, algorithm="direct")

    np.testing.assert_allclose(alias.solution, direct.solution, rtol=1e-12, atol=1e-12)
    assert alias.converged == direct.converged
    assert alias.algorithm == direct.algorithm


def test_residual_helper_matches_solver_output() -> None:
    a = jnp.array([[0.25]])
    b = jnp.array([[0.2]])
    c = jnp.array([[0.75]])
    result = solve_discrete_sylvester_direct(a, b, c)

    residual = discrete_sylvester_residual(a, result.solution, b, c)

    np.testing.assert_allclose(
        residual,
        result.relative_residual,
        rtol=1e-12,
        atol=1e-12,
    )
