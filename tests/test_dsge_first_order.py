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
    solve_quadratic_matrix_equation_schur,
    solve_quadratic_matrix_equation_schur_jax,
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


def _indexin(values, reference):
    ref_map = {value: idx for idx, value in enumerate(reference)}
    return tuple(ref_map[value] for value in values)


def _rbc_cme_qme_fixture():
    timings, jacobian, _ = _rbc_cme_fixture()
    grad = jnp.asarray(jacobian, dtype=jnp.float64)
    dyn_index = np.arange(timings.nPresent_only, timings.nVars)
    comb = tuple(
        sorted(
            set(timings.future_not_past_and_mixed_idx)
            | set(timings.past_not_future_idx)
        )
    )
    future_in_comb = _indexin(timings.future_not_past_and_mixed_idx, comb)
    past_in_comb = _indexin(timings.past_not_future_and_mixed_idx, comb)
    selector = jnp.eye(len(comb), dtype=grad.dtype)

    n_future = timings.nFuture_not_past_and_mixed
    n_vars = timings.nVars
    n_past = timings.nPast_not_future_and_mixed

    grad_plus = grad[:, :n_future]
    grad_zero = grad[:, n_future : n_future + n_vars]
    grad_minus = grad[:, n_future + n_vars : n_future + n_vars + n_past]

    q_complete = jnp.linalg.qr(
        grad_zero[:, timings.present_only_idx],
        mode="complete",
    )[0]
    a_plus = q_complete.T @ grad_plus
    a_zero = q_complete.T @ grad_zero
    a_minus = q_complete.T @ grad_minus

    a_tilde_plus = a_plus[dyn_index] @ selector[list(future_in_comb), :]
    a_tilde_zero = a_zero[dyn_index][:, comb]
    a_tilde_minus = a_minus[dyn_index] @ selector[list(past_in_comb), :]
    return timings, a_tilde_plus, a_tilde_zero, a_tilde_minus


def _static_qme_timings():
    return DSGETimings.from_julia(
        present_only=[],
        future_not_past=[],
        past_not_future=[],
        mixed=[],
        future_not_past_and_mixed=[],
        past_not_future_and_mixed=[],
        present_but_not_only=[],
        mixed_in_past=[],
        not_mixed_in_past=[],
        mixed_in_future=[],
        exo=[],
        var=[],
        aux=[],
        exo_present=[],
        nPresent_only=0,
        nMixed=0,
        nFuture_not_past_and_mixed=0,
        nPast_not_future_and_mixed=0,
        nPresent_but_not_only=0,
        nVars=0,
        nExo=0,
        present_only_idx=[],
        present_but_not_only_idx=[],
        future_not_past_and_mixed_idx=[],
        not_mixed_in_past_idx=[],
        past_not_future_and_mixed_idx=[],
        mixed_in_past_idx=[],
        mixed_in_future_idx=[],
        past_not_future_idx=[],
        reorder=[],
        dynamic_order=[],
    )


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


def test_quadratic_matrix_equation_schur_matches_doubling_on_rbc_fixture() -> None:
    timings, a_tilde_plus, a_tilde_zero, a_tilde_minus = _rbc_cme_qme_fixture()

    schur_result = solve_quadratic_matrix_equation_schur(
        a_tilde_plus,
        a_tilde_zero,
        a_tilde_minus,
        timings,
    )
    doubling_result = solve_quadratic_matrix_equation_doubling(
        a_tilde_plus,
        a_tilde_zero,
        a_tilde_minus,
    )

    assert schur_result.converged
    np.testing.assert_allclose(
        schur_result.solution,
        doubling_result.solution,
        rtol=1e-8,
        atol=1e-8,
    )


def test_jax_schur_quadratic_matrix_equation_supports_reverse_mode_autodiff() -> None:
    timings, a_tilde_plus, a_tilde_zero, a_tilde_minus = _rbc_cme_qme_fixture()
    compiled_grad = jax.jit(
        jax.grad(
            lambda shift: jnp.sum(
                solve_quadratic_matrix_equation_schur_jax(
                    a_tilde_plus,
                    a_tilde_zero.at[0, 0].add(shift),
                    a_tilde_minus,
                    timings,
                ).solution
                ** 2
            )
        )
    )
    epsilon = 1e-6
    autodiff_grad = float(np.asarray(compiled_grad(0.0)))

    def objective(shift: float) -> float:
        result = solve_quadratic_matrix_equation_schur(
            a_tilde_plus,
            a_tilde_zero.at[0, 0].add(shift),
            a_tilde_minus,
            timings,
        )
        return float(np.asarray(jnp.sum(result.solution ** 2)))

    finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(autodiff_grad, finite_difference, rtol=5e-5, atol=5e-6)


def test_jax_schur_quadratic_matrix_equation_supports_vmap() -> None:
    timings, a_tilde_plus, a_tilde_zero, a_tilde_minus = _rbc_cme_qme_fixture()
    shifts = jnp.asarray([0.0, 1.0e-5, -1.0e-5], dtype=jnp.float64)

    def solve_shifted(shift: jax.Array) -> jax.Array:
        return solve_quadratic_matrix_equation_schur_jax(
            a_tilde_plus,
            a_tilde_zero.at[0, 0].add(shift),
            a_tilde_minus,
            timings,
        ).solution

    batched = jax.jit(jax.vmap(solve_shifted))(shifts)

    assert batched.shape == (3,) + a_tilde_plus.shape
    for idx, shift in enumerate(np.asarray(shifts)):
        expected = solve_quadratic_matrix_equation_schur(
            a_tilde_plus,
            a_tilde_zero.at[0, 0].add(float(shift)),
            a_tilde_minus,
            timings,
        )
        assert expected.converged
        np.testing.assert_allclose(
            batched[idx],
            expected.solution,
            rtol=1e-8,
            atol=1e-8,
        )


def test_quadratic_matrix_equation_schur_handles_empty_pencils_without_crashing() -> None:
    timings = _static_qme_timings()

    result = solve_quadratic_matrix_equation_schur(
        jnp.zeros((0, 0), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        jnp.zeros((0, 0), dtype=jnp.float64),
        timings,
    )

    assert result.converged
    assert result.solution.shape == (0, 0)
    np.testing.assert_allclose(
        np.asarray(result.relative_residual, dtype=np.float64),
        np.asarray(0.0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


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


def test_first_order_solution_defaults_to_schur() -> None:
    timings, jacobian, _ = _rbc_cme_fixture()

    default_result = solve_first_order_dsge_solution(jacobian, timings)
    schur_result = solve_first_order_dsge_solution(
        jacobian,
        timings,
        qme_algorithm="schur",
    )

    np.testing.assert_allclose(
        default_result.solution_matrix,
        schur_result.solution_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        default_result.qme_solution,
        schur_result.qme_solution,
        rtol=1e-10,
        atol=1e-10,
    )


def test_first_order_solution_schur_matches_julia_fixture() -> None:
    timings, jacobian, expected_solution = _rbc_cme_fixture()

    result = solve_first_order_dsge_solution(
        jacobian,
        timings,
        qme_algorithm="schur",
    )

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


def test_jax_first_order_solution_defaults_to_schur() -> None:
    timings, jacobian, _ = _rbc_cme_fixture()
    compiled = jax.jit(
        solve_first_order_dsge_solution_jax,
        static_argnames=("timings",),
    )

    default_result = compiled(jacobian, timings)
    schur_result = jax.jit(
        solve_first_order_dsge_solution_jax,
        static_argnames=("timings", "qme_algorithm"),
    )(
        jacobian,
        timings,
        qme_algorithm="schur",
    )

    np.testing.assert_allclose(
        default_result.solution_matrix,
        schur_result.solution_matrix,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_first_order_solution_schur_matches_existing_solver() -> None:
    timings, jacobian, expected_solution = _rbc_cme_fixture()
    compiled = jax.jit(
        solve_first_order_dsge_solution_jax,
        static_argnames=("timings", "qme_algorithm"),
    )

    result = compiled(
        jacobian,
        timings,
        qme_algorithm="schur",
    )

    assert bool(np.asarray(result.converged))
    np.testing.assert_allclose(
        result.solution_matrix,
        expected_solution,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.solution_matrix,
        solve_first_order_dsge_solution(
            jacobian,
            timings,
            qme_algorithm="schur",
        ).solution_matrix,
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
