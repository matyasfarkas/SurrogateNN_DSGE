from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    DSGETimings,
    create_second_order_auxiliary_matrices,
    parse_macro_model,
    pruned_second_order_state_update,
    second_order_state_update,
    solve_first_order_dsge_solution,
    solve_second_order_dsge_solution,
    solve_second_order_model,
    solve_second_order_stochastic_steady_state,
)


RBC_CME_SOURCE = """
@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME verbose = true begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end
"""


def _dense_from_julia_sparse(rows, cols, values, shape):
    matrix = np.zeros(shape, dtype=np.float64)
    for row, col, value in zip(rows, cols, values):
        matrix[row - 1, col - 1] = value
    return jnp.asarray(matrix, dtype=jnp.float64)


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
    hessian = _dense_from_julia_sparse(
        rows=[2, 2, 3, 3, 3, 2, 3, 2, 3, 3, 2, 1, 5, 4, 3, 3, 2, 3, 2, 2, 2, 5, 1, 5, 5, 1, 5],
        cols=[3, 8, 17, 18, 21, 31, 32, 33, 33, 36, 38, 57, 57, 65, 77, 78, 97, 97, 106, 108, 113, 147, 169, 169, 175, 177, 177],
        values=[
            0.016123811656420906,
            0.0017360837926088356,
            -1.6460086683698225,
            -0.6822285892902225,
            0.822181329844946,
            0.016123811656420906,
            -0.6822285892902225,
            -1.1310653631067793,
            -1.1310653631147312,
            0.6815463606961406,
            -0.0014356764785829514,
            -0.023601001001000967,
            0.023601001001000967,
            -0.748202876155088,
            0.822181329844946,
            0.6815463606961406,
            1.1310653631067793,
            1.1310653631067793,
            0.0017360837926088356,
            -0.0014356764785829514,
            -0.00033795378281254373,
            -0.0226,
            -0.023601001001000967,
            0.023601001001000967,
            -0.0226,
            0.0021014511165327685,
            -0.0021014511165327685,
        ],
        shape=(7, 225),
    )
    return timings, jacobian, hessian


def test_second_order_auxiliary_matrices_have_expected_shapes_and_roundtrip() -> None:
    timings, _, _ = _rbc_cme_fixture()
    auxiliary = create_second_order_auxiliary_matrices(timings)

    assert auxiliary.compression_matrix.shape == (36, 21)
    assert auxiliary.uncompression_matrix.shape == (21, 36)
    assert auxiliary.hessian_compression_matrix.shape == (225, 120)
    assert auxiliary.hessian_uncompression_matrix.shape == (120, 225)
    assert auxiliary.sigma.shape == (36, 36)

    compressed = jnp.arange(21, dtype=jnp.float64)[None, :]
    full = compressed @ auxiliary.uncompression_matrix
    np.testing.assert_allclose(full @ auxiliary.compression_matrix, compressed, rtol=0, atol=0)


def test_second_order_solution_matches_julia_scalar_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    auxiliary = create_second_order_auxiliary_matrices(timings)

    result = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        first_order,
        timings,
        auxiliary_matrices=auxiliary,
    )

    assert result.converged
    assert result.compressed_solution.shape == (7, 21)
    assert result.solution_matrix.shape == (7, 36)
    np.testing.assert_allclose(
        result.solution_matrix @ auxiliary.compression_matrix,
        result.compressed_solution,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.solution_matrix[1, 0],
        -0.006642814796744731,
        rtol=1e-7,
        atol=1e-9,
    )


def test_second_order_state_update_matches_julia_irf_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        first_order,
        timings,
    )

    zero_state = jnp.zeros(timings.nVars, dtype=jnp.float64)
    response_delta = second_order_state_update(
        zero_state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        first_order,
        second_order,
        timings,
    )
    response_eps = second_order_state_update(
        zero_state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        first_order,
        second_order,
        timings,
    )

    np.testing.assert_allclose(
        jnp.asarray([response_delta[3], response_eps[3]], dtype=jnp.float64),
        jnp.asarray(
            [-0.0004547347878067665, 0.0020831426377533636],
            dtype=jnp.float64,
        ),
        rtol=1e-6,
        atol=1e-9,
    )


def test_pruned_second_order_state_update_matches_julia_irf_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        first_order,
        timings,
    )

    initial_states = (
        jnp.zeros(timings.nVars, dtype=jnp.float64),
        jnp.zeros(timings.nVars, dtype=jnp.float64),
    )
    response_delta = pruned_second_order_state_update(
        initial_states,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        first_order,
        second_order,
        timings,
    )
    response_eps = pruned_second_order_state_update(
        initial_states,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        first_order,
        second_order,
        timings,
    )

    np.testing.assert_allclose(
        jnp.asarray(
            [
                (response_delta[0] + response_delta[1])[3],
                (response_eps[0] + response_eps[1])[3],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [-0.00045473478780675195, 0.002083142637753389],
            dtype=jnp.float64,
        ),
        rtol=1e-6,
        atol=1e-9,
    )


def test_second_order_stochastic_steady_state_is_fixed_point() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        first_order,
        timings,
    )

    stochastic_steady_state = solve_second_order_stochastic_steady_state(
        first_order,
        second_order,
        timings,
    )
    fixed_point = second_order_state_update(
        stochastic_steady_state.state_vector,
        jnp.zeros(timings.nExo, dtype=jnp.float64),
        first_order,
        second_order,
        timings,
    )

    assert stochastic_steady_state.converged
    np.testing.assert_allclose(
        fixed_point,
        stochastic_steady_state.state_vector,
        rtol=1e-9,
        atol=1e-9,
    )


def test_parsed_model_second_order_matches_low_level_result() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    low_level_first_order = solve_first_order_dsge_solution(jacobian, timings)
    low_level_second_order = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        low_level_first_order,
        timings,
    )

    model = parse_macro_model(RBC_CME_SOURCE)
    parsed = solve_second_order_model(model)

    assert parsed.first_order_solution.converged
    assert parsed.second_order_solution.converged
    assert parsed.stochastic_steady_state.converged
    np.testing.assert_allclose(
        parsed.second_order_solution.solution_matrix,
        low_level_second_order.solution_matrix,
        rtol=1e-6,
        atol=1e-6,
    )
