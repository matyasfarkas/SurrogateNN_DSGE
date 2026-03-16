from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    calculate_hessian,
    calculate_jacobian,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
)


ADDITIVE_LOOP_SOURCE = """
@model additive_loop begin
    y_sum[0] = for lag in -2:0 y[lag] end
    y[0] = rho * y[-1] + eps[x]
end

@parameters additive_loop begin
    rho = 0.9
end
"""


ADDITIVE_EXPLICIT_SOURCE = """
@model additive_explicit begin
    y_sum[0] = y[-2] + y[-1] + y[0]
    y[0] = rho * y[-1] + eps[x]
end

@parameters additive_explicit begin
    rho = 0.9
end
"""


PRODUCT_LOOP_SOURCE = """
@model product_loop begin
    gross_r_ann[0] = for operator = :*, lag in -1:0 (1 + r[lag]) end
    r[0] = rho * r[-1] + eps[x]
end

@parameters product_loop begin
    rho = 0.5
end
"""


PRODUCT_EXPLICIT_SOURCE = """
@model product_explicit begin
    gross_r_ann[0] = (1 + r[-1]) * (1 + r[0])
    r[0] = rho * r[-1] + eps[x]
end

@parameters product_explicit begin
    rho = 0.5
end
"""


MULTILINE_LOOP_SOURCE = """
@model multiline_loop begin
    z[0] =
        for lag in 0:1 beta^lag * y[lag] end +
        for lag in 1:2 -beta^lag * y[lag] end
    y[0] = rho * y[-1] + eps[x]
end

@parameters multiline_loop begin
    beta = 0.95
    rho = 0.9
end
"""


MULTILINE_EXPLICIT_SOURCE = """
@model multiline_explicit begin
    z[0] = beta^0 * y[0] + beta^1 * y[1] + -beta^1 * y[1] + -beta^2 * y[2]
    y[0] = rho * y[-1] + eps[x]
end

@parameters multiline_explicit begin
    beta = 0.95
    rho = 0.9
end
"""


def _assert_loop_model_matches_explicit(
    loop_source: str,
    explicit_source: str,
    *,
    initial_guess: dict[str, float],
    check_hessian: bool = False,
) -> None:
    loop_model = parse_macro_model(loop_source)
    explicit_model = parse_macro_model(explicit_source)

    assert loop_model.timings == explicit_model.timings

    loop_steady_state = solve_steady_state(loop_model, initial_guess=initial_guess)
    explicit_steady_state = solve_steady_state(explicit_model, initial_guess=initial_guess)

    assert loop_steady_state.converged
    assert explicit_steady_state.converged
    np.testing.assert_allclose(
        loop_steady_state.steady_state,
        explicit_steady_state.steady_state,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        loop_steady_state.parameter_values,
        explicit_steady_state.parameter_values,
        rtol=1e-10,
        atol=1e-10,
    )

    loop_jacobian = calculate_jacobian(loop_model, steady_state=loop_steady_state.steady_state)
    explicit_jacobian = calculate_jacobian(
        explicit_model,
        steady_state=explicit_steady_state.steady_state,
    )
    np.testing.assert_allclose(loop_jacobian, explicit_jacobian, rtol=1e-10, atol=1e-10)

    if check_hessian:
        loop_hessian = calculate_hessian(loop_model, steady_state=loop_steady_state.steady_state)
        explicit_hessian = calculate_hessian(
            explicit_model,
            steady_state=explicit_steady_state.steady_state,
        )
        np.testing.assert_allclose(loop_hessian, explicit_hessian, rtol=1e-10, atol=1e-10)

    loop_solution = solve_first_order_model(loop_model, steady_state=loop_steady_state.steady_state)
    explicit_solution = solve_first_order_model(
        explicit_model,
        steady_state=explicit_steady_state.steady_state,
    )
    np.testing.assert_allclose(
        loop_solution.solution.solution_matrix,
        explicit_solution.solution.solution_matrix,
        rtol=1e-10,
        atol=1e-10,
    )


def test_additive_time_for_loop_matches_explicit_model() -> None:
    _assert_loop_model_matches_explicit(
        ADDITIVE_LOOP_SOURCE,
        ADDITIVE_EXPLICIT_SOURCE,
        initial_guess={"y": 0.1, "y_sum": 0.0},
    )


def test_product_time_for_loop_matches_explicit_model() -> None:
    _assert_loop_model_matches_explicit(
        PRODUCT_LOOP_SOURCE,
        PRODUCT_EXPLICIT_SOURCE,
        initial_guess={"gross_r_ann": 1.0, "r": 0.1},
        check_hessian=True,
    )


def test_multiline_time_for_loops_match_explicit_model() -> None:
    _assert_loop_model_matches_explicit(
        MULTILINE_LOOP_SOURCE,
        MULTILINE_EXPLICIT_SOURCE,
        initial_guess={"y": 0.1, "z": 0.0},
    )


def test_top_level_model_for_loop_blocks_remain_explicitly_unsupported() -> None:
    source = """
    @model unsupported_loop_block begin
        for lag in -1:0
            y[lag] = rho * y[lag - 1]
        end
        y[1] = rho * y[0] + eps[x]
    end

    @parameters unsupported_loop_block begin
        rho = 0.9
    end
    """

    with pytest.raises(NotImplementedError, match="not ported yet"):
        parse_macro_model(source)
