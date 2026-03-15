from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    create_third_order_auxiliary_matrices,
    parse_macro_model,
    pruned_third_order_state_update,
    solve_first_order_dsge_solution,
    solve_second_order_dsge_solution,
    solve_third_order_dsge_solution,
    solve_third_order_model,
    solve_third_order_stochastic_steady_state,
    third_order_state_update,
)
from test_dsge_second_order import RBC_CME_SOURCE, _rbc_cme_fixture
from test_model_parser import _expected_rbc_third_order


def test_third_order_auxiliary_matrices_have_expected_shapes_and_roundtrip() -> None:
    timings, _, _ = _rbc_cme_fixture()
    auxiliary = create_third_order_auxiliary_matrices(timings)

    assert auxiliary.compression_matrix.shape == (216, 56)
    assert auxiliary.uncompression_matrix.shape == (56, 216)
    assert auxiliary.permutation_matrix.shape == (216, 216)
    assert auxiliary.swap12_left.shape == (216, 216)
    assert auxiliary.swap12_right.shape == (216, 216)
    assert auxiliary.swap23_left_dynamic.shape == (3375, 3375)
    assert auxiliary.swap13_left_dynamic.shape == (3375, 3375)

    compressed = jnp.arange(56, dtype=jnp.float64)[None, :]
    full = compressed @ auxiliary.uncompression_matrix
    np.testing.assert_allclose(full @ auxiliary.compression_matrix, compressed, rtol=0, atol=0)


def test_third_order_solution_matches_julia_scalar_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    third_order_derivatives = _expected_rbc_third_order()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(jacobian, hessian, first_order, timings)
    auxiliary = create_third_order_auxiliary_matrices(timings)

    result = solve_third_order_dsge_solution(
        jacobian,
        hessian,
        third_order_derivatives,
        first_order,
        second_order,
        timings,
        third_order_auxiliary_matrices=auxiliary,
    )

    assert result.converged
    assert result.compressed_solution.shape == (7, 56)
    assert result.solution_matrix.shape == (7, 216)
    np.testing.assert_allclose(
        result.solution_matrix @ auxiliary.compression_matrix,
        result.compressed_solution,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.solution_matrix[1, 0],
        0.003453249193794699,
        rtol=1e-5,
        atol=1e-8,
    )


def test_third_order_state_update_matches_julia_irf_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    third_order_derivatives = _expected_rbc_third_order()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(jacobian, hessian, first_order, timings)
    third_order = solve_third_order_dsge_solution(
        jacobian,
        hessian,
        third_order_derivatives,
        first_order,
        second_order,
        timings,
    )

    zero_state = jnp.zeros(timings.nVars, dtype=jnp.float64)
    response_delta = third_order_state_update(
        zero_state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        first_order,
        second_order,
        third_order,
        timings,
    )
    response_eps = third_order_state_update(
        zero_state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        first_order,
        second_order,
        third_order,
        timings,
    )

    np.testing.assert_allclose(
        jnp.asarray([response_delta[3], response_eps[3]], dtype=jnp.float64),
        jnp.asarray(
            [-0.00045473149068020854, 0.002083198241302615],
            dtype=jnp.float64,
        ),
        rtol=1e-6,
        atol=1e-9,
    )


def test_pruned_third_order_state_update_matches_julia_irf_fixture() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    third_order_derivatives = _expected_rbc_third_order()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(jacobian, hessian, first_order, timings)
    third_order = solve_third_order_dsge_solution(
        jacobian,
        hessian,
        third_order_derivatives,
        first_order,
        second_order,
        timings,
    )

    initial_states = (
        jnp.zeros(timings.nVars, dtype=jnp.float64),
        jnp.zeros(timings.nVars, dtype=jnp.float64),
        jnp.zeros(timings.nVars, dtype=jnp.float64),
    )
    response_delta = pruned_third_order_state_update(
        initial_states,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        first_order,
        second_order,
        third_order,
        timings,
    )
    response_eps = pruned_third_order_state_update(
        initial_states,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        first_order,
        second_order,
        third_order,
        timings,
    )

    np.testing.assert_allclose(
        jnp.asarray(
            [
                (response_delta[0] + response_delta[1] + response_delta[2])[3],
                (response_eps[0] + response_eps[1] + response_eps[2])[3],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [-0.0004547315171573783, 0.0020831990353127696],
            dtype=jnp.float64,
        ),
        rtol=1e-6,
        atol=1e-9,
    )


def test_third_order_stochastic_steady_state_is_fixed_point() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    third_order_derivatives = _expected_rbc_third_order()
    first_order = solve_first_order_dsge_solution(jacobian, timings)
    second_order = solve_second_order_dsge_solution(jacobian, hessian, first_order, timings)
    third_order = solve_third_order_dsge_solution(
        jacobian,
        hessian,
        third_order_derivatives,
        first_order,
        second_order,
        timings,
    )

    stochastic_steady_state = solve_third_order_stochastic_steady_state(
        first_order,
        second_order,
        third_order,
        timings,
    )
    fixed_point = third_order_state_update(
        stochastic_steady_state.state_vector,
        jnp.zeros(timings.nExo, dtype=jnp.float64),
        first_order,
        second_order,
        third_order,
        timings,
    )

    assert stochastic_steady_state.converged
    np.testing.assert_allclose(
        fixed_point,
        stochastic_steady_state.state_vector,
        rtol=1e-9,
        atol=1e-9,
    )


def test_parsed_model_third_order_matches_low_level_result() -> None:
    timings, jacobian, hessian = _rbc_cme_fixture()
    third_order_derivatives = _expected_rbc_third_order()
    low_level_first_order = solve_first_order_dsge_solution(jacobian, timings)
    low_level_second_order = solve_second_order_dsge_solution(
        jacobian,
        hessian,
        low_level_first_order,
        timings,
    )
    low_level_third_order = solve_third_order_dsge_solution(
        jacobian,
        hessian,
        third_order_derivatives,
        low_level_first_order,
        low_level_second_order,
        timings,
    )

    model = parse_macro_model(RBC_CME_SOURCE)
    parsed = solve_third_order_model(model)

    assert parsed.first_order_solution.converged
    assert parsed.second_order_solution.converged
    assert parsed.third_order_solution.converged
    assert parsed.stochastic_steady_state.converged
    np.testing.assert_allclose(
        parsed.third_order_solution.solution_matrix,
        low_level_third_order.solution_matrix,
        rtol=1e-6,
        atol=1e-6,
    )
