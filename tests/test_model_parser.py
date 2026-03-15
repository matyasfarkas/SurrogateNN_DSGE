from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    DSGETimings,
    calculate_hessian,
    calculate_jacobian,
    calculate_third_order_derivatives,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
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
    return jnp.asarray(matrix)


def _expected_rbc_timings() -> DSGETimings:
    return DSGETimings.from_julia(
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


def _expected_rbc_jacobian() -> jnp.ndarray:
    return _dense_from_julia_sparse(
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


def _expected_rbc_hessian() -> jnp.ndarray:
    return _dense_from_julia_sparse(
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


def _expected_rbc_third_order() -> jnp.ndarray:
    return _dense_from_julia_sparse(
        rows=[2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 2, 2, 1, 5, 4, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 5, 1, 5, 1, 5],
        cols=[33, 38, 108, 113, 242, 243, 246, 257, 258, 261, 302, 303, 453, 458, 467, 468, 471, 481, 482, 483, 483, 486, 488, 527, 528, 556, 558, 563, 852, 852, 965, 1142, 1143, 1157, 1158, 1447, 1447, 1578, 1583, 1606, 1608, 1613, 1681, 1683, 1688, 2532, 2532, 2644, 2644, 2652, 2652],
        values=[
            -0.026667580492113735,
            -0.0014356764785829514,
            -0.0014356764785829514,
            -0.00033795378281254373,
            4.926193679339384,
            1.3611877138551502,
            -1.6404224952084816,
            1.3611877138551502,
            1.1283551437214383,
            -0.6799132630658674,
            -1.6404224952084818,
            -0.6799132630658674,
            -0.026667580492113735,
            -0.0014356764785829514,
            1.3611877138551502,
            1.1283551437214383,
            -0.6799132630658674,
            -0.026667580492113735,
            1.1283551437214383,
            2.806046478514929,
            2.8060464785346566,
            -1.1272267885697917,
            0.00237450169160211,
            -0.6799132630658674,
            -1.1272267885697917,
            -0.0014356764785829514,
            0.00237450169160211,
            0.0002794751606447495,
            0.0021014511165327685,
            -0.0021014511165327685,
            0.3732050292530844,
            -1.6404224952084816,
            -0.6799132630658674,
            -0.6799132630658674,
            -1.1272267885697917,
            -2.8060464785149284,
            -2.8060464785149284,
            -0.0014356764785829514,
            -0.00033795378281254373,
            -0.0014356764785829514,
            0.00237450169160211,
            0.0002794751606447495,
            -0.00033795378281254373,
            0.0002794751606447495,
            0.00010148350673731277,
            0.0021014511165327685,
            -0.0021014511165327685,
            0.0021014511165327685,
            -0.0021014511165327685,
            -0.0004090778090616675,
            0.0004090778090616675,
        ],
        shape=(7, 3375),
    )


def _expected_rbc_first_order_solution() -> jnp.ndarray:
    return jnp.asarray(
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


def test_parse_macro_model_matches_rbc_timings() -> None:
    model = parse_macro_model(RBC_CME_SOURCE)
    expected = _expected_rbc_timings()

    assert model.name == "RBC_CME"
    assert model.parameter_names == (
        "Pibar",
        "alpha",
        "beta",
        "delta",
        "phi_pi",
        "rho_z_delta",
        "rhoz",
        "std_eps",
        "std_z_delta",
    )
    assert model.timings == expected


def test_rbc_steady_state_matches_julia_fixture() -> None:
    model = parse_macro_model(RBC_CME_SOURCE)

    result = solve_steady_state(model)

    assert result.converged
    np.testing.assert_allclose(
        result.base_steady_state,
        jnp.asarray(
            [
                1.0,
                1.0024019205374952,
                1.003405325870413,
                1.2092444352939415,
                9.467573947982233,
                1.42321160651834,
                1.0,
            ],
            dtype=jnp.float64,
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(result.steady_state, result.base_steady_state, rtol=0, atol=0)


def test_rbc_symbolic_derivatives_match_julia_fixture() -> None:
    model = parse_macro_model(RBC_CME_SOURCE)
    steady_state = solve_steady_state(model).steady_state

    jacobian = calculate_jacobian(model, steady_state=steady_state)
    hessian = calculate_hessian(model, steady_state=steady_state)
    third_order = calculate_third_order_derivatives(model, steady_state=steady_state)

    np.testing.assert_allclose(jacobian, _expected_rbc_jacobian(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hessian, _expected_rbc_hessian(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        third_order,
        _expected_rbc_third_order(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_rbc_first_order_solution_from_parsed_source_matches_julia_fixture() -> None:
    model = parse_macro_model(RBC_CME_SOURCE)

    result = solve_first_order_model(model)

    assert result.solution.converged
    np.testing.assert_allclose(
        result.solution.solution_matrix,
        _expected_rbc_first_order_solution(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_parser_creates_auxiliary_variables_for_large_leads_and_lags() -> None:
    source = """
    @model aux_test begin
        x[0] = y[2] + z[-2] + eps[x+1] + eta[x-2]
        y[0] = rho * y[-1]
        z[0] = rho * z[-1]
    end

    @parameters aux_test begin
        rho = 0.9
    end
    """

    model = parse_macro_model(source)

    assert model.timings.aux == ("y__L1", "z__L-1")
    assert model.timings.exo == ("eps", "eta")
    assert model.timings.exo_present == ("eps", "eta", "eta__L-1")
