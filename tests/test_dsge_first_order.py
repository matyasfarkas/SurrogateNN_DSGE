from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    DSGETimings,
    kalman_loglikelihood,
    linear_state_space_from_first_order_solution,
    simulate_linear_gaussian_state_space,
    solve_first_order_dsge_solution,
    solve_first_order_dsge_solution_jax,
    solve_quadratic_matrix_equation_doubling,
    solve_quadratic_matrix_equation_doubling_jax,
)


def _dense_from_julia_sparse(rows, cols, values, shape):
    matrix = np.zeros(shape, dtype=np.float64)
    for row, col, value in zip(rows, cols, values):
        matrix[row - 1, col - 1] = value
    return jnp.asarray(matrix)


def _rbc_cme_fixture():
    timings = DSGETimings.from_julia(
        present_only=["R", "y"],
        future_not_past=["Pi", "c"],
        past_not_future=["k", "z_delta"],
        mixed=["A"],
        future_not_past_and_mixed=["A", "Pi", "c"],
        past_not_future_and_mixed=["A", "k", "z_delta"],
        present_but_not_only=["A", "Pi", "c", "k", "z_delta"],
        mixed_in_past=["A"],
        not_mixed_in_past=["k", "z_delta"],
        mixed_in_future=["A"],
        exo=["delta_eps", "eps_z"],
        var=["A", "Pi", "R", "c", "k", "y", "z_delta"],
        aux=[],
        exo_present=[],
        nPresent_only=2,
        nMixed=1,
        nFuture_not_past_and_mixed=3,
        nPast_not_future_and_mixed=3,
        nPresent_but_not_only=5,
        nVars=7,
        nExo=2,
        present_only_idx=[3, 6],
        present_but_not_only_idx=[1, 2, 4, 5, 7],
        future_not_past_and_mixed_idx=[1, 2, 4],
        not_mixed_in_past_idx=[2, 3],
        past_not_future_and_mixed_idx=[1, 5, 7],
        mixed_in_past_idx=[1],
        mixed_in_future_idx=[1],
        past_not_future_idx=[5, 7],
        reorder=[5, 6, 1, 7, 3, 2, 4],
        dynamic_order=[3, 4, 5, 1, 2],
    )
    jacobian = _dense_from_julia_sparse(
        rows=[2, 3, 2, 3, 1, 5, 7, 4, 3, 4, 2, 3, 5, 2, 5, 1, 5, 6, 7, 1, 5, 6, 6, 7],
        cols=[1, 2, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15],
        values=[
            -0.01949762952125457,
            0.8249811251976374,
            0.6838672481452972,
            0.683867248150105,
            -1.42321160651834,
            1.42321160651834,
            1.0,
            -1.5000000000190756,
            -0.8241561440666456,
            0.999,
            -0.6838672481452971,
            -0.6838672481452971,
            -1.0,
            0.0017360837926088356,
            -1.0,
            1.0,
            -0.21396717122439846,
            1.0,
            -0.9,
            -0.023601001001000967,
            1.001001001001001,
            -0.9,
            -0.005,
            -0.0068,
        ],
        shape=(7, 15),
    )
    expected_solution = jnp.asarray(
        [
            [0.9, 5.41234e-16, -6.41848e-17, 0.0, 0.0068],
            [0.0223801, -0.00364902, 0.00121336, 6.7409e-6, 0.000169094],
            [0.0336038, -0.00547901, 0.00182186, 1.01215e-5, 0.000253895],
            [0.28748, 0.049647, -0.0660339, -0.000366855, 0.00217207],
            [0.99341, 0.951354, -0.126537, -0.000702981, 0.00750577],
            [1.28089, 0.023601, -9.12755e-17, -0.0, 0.00967784],
            [0.0, 0.0, 0.9, 0.005, -0.0],
        ],
        dtype=jnp.float64,
    )
    return timings, jacobian, expected_solution


def test_quadratic_matrix_equation_solver_converges_on_scalar_case() -> None:
    result = solve_quadratic_matrix_equation_doubling(
        jnp.array([[1.0]]),
        jnp.array([[-3.0]]),
        jnp.array([[2.0]]),
        initial_guess=jnp.array([[0.1]]),
    )

    assert result.converged
    np.testing.assert_allclose(result.solution, jnp.array([[1.0]]), rtol=1e-8, atol=1e-8)


def test_jax_quadratic_matrix_equation_solver_is_jittable() -> None:
    compiled = jax.jit(solve_quadratic_matrix_equation_doubling_jax)
    result = compiled(
        jnp.array([[1.0]]),
        jnp.array([[-3.0]]),
        jnp.array([[2.0]]),
        initial_guess=jnp.array([[0.1]]),
    )

    assert bool(np.asarray(result.converged))
    np.testing.assert_allclose(result.solution, jnp.array([[1.0]]), rtol=1e-8, atol=1e-8)


def test_first_order_solution_matches_julia_fixture() -> None:
    timings, jacobian, expected_solution = _rbc_cme_fixture()

    result = solve_first_order_dsge_solution(jacobian, timings)

    assert result.converged
    np.testing.assert_allclose(
        result.solution_matrix,
        expected_solution,
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_first_order_solution_matches_existing_solver() -> None:
    timings, jacobian, expected_solution = _rbc_cme_fixture()
    compiled = jax.jit(
        solve_first_order_dsge_solution_jax,
        static_argnames=("timings",),
    )

    result = compiled(jacobian, timings)

    assert bool(np.asarray(result.converged))
    np.testing.assert_allclose(
        result.solution_matrix,
        expected_solution,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.solution_matrix,
        solve_first_order_dsge_solution(jacobian, timings).solution_matrix,
        rtol=1e-8,
        atol=1e-8,
    )


def test_first_order_solution_feeds_state_space_layer() -> None:
    timings, jacobian, _ = _rbc_cme_fixture()
    result = solve_first_order_dsge_solution(jacobian, timings)
    state_space = linear_state_space_from_first_order_solution(
        result.solution_matrix,
        timings,
        observable_indices=[2, 5],
    )

    simulation = simulate_linear_gaussian_state_space(
        state_space,
        key=jnp.array([0, 123], dtype=jnp.uint32),
        num_periods=40,
    )
    ll = kalman_loglikelihood(state_space, simulation.observations)

    assert np.isfinite(ll)
