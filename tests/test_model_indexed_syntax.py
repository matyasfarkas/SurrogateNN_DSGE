from __future__ import annotations

import numpy as np

from surrogatenn_dsge import (
    calculate_jacobian,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
)


INDEXED_BLOCK_SOURCE = """
@model indexed_block begin
    for co in [H, F]
        y{co}[0] = rho{co} * y{co}[-1] + u{co}[x]
    end
end

@parameters indexed_block begin
    rho{H} = 0.9
    rho{F} = 0.8
end
"""


INDEXED_BLOCK_EXPLICIT_SOURCE = """
@model indexed_block_explicit begin
    y{H}[0] = rho{H} * y{H}[-1] + u{H}[x]
    y{F}[0] = rho{F} * y{F}[-1] + u{F}[x]
end

@parameters indexed_block_explicit begin
    rho{H} = 0.9
    rho{F} = 0.8
end
"""


INDEXED_SUM_SOURCE = """
@model indexed_sum begin
    total[0] = for co in [H, F] y{co}[0] end
    y{H}[0] = rho{H} * y{H}[-1]
    y{F}[0] = rho{F} * y{F}[-1]
end

@parameters indexed_sum begin
    rho{H} = 0.9
    rho{F} = 0.8
end
"""


INDEXED_SUM_EXPLICIT_SOURCE = """
@model indexed_sum_explicit begin
    total[0] = y{H}[0] + y{F}[0]
    y{H}[0] = rho{H} * y{H}[-1]
    y{F}[0] = rho{F} * y{F}[-1]
end

@parameters indexed_sum_explicit begin
    rho{H} = 0.9
    rho{F} = 0.8
end
"""


NESTED_INDEX_PARAMETER_SOURCE = """
@model nested_index_parameter begin
    z[0] = rho{H}{F} + rho{F}{H}
end

@parameters nested_index_parameter begin
    rho{H}{F} = 0.5
    rho{F}{H} = rho{H}{F}
end
"""


def _assert_models_match(loop_source: str, explicit_source: str) -> None:
    loop_model = parse_macro_model(loop_source)
    explicit_model = parse_macro_model(explicit_source)

    assert loop_model.parameter_names == explicit_model.parameter_names
    assert loop_model.timings == explicit_model.timings

    loop_steady_state = solve_steady_state(loop_model)
    explicit_steady_state = solve_steady_state(explicit_model)

    assert loop_steady_state.converged
    assert explicit_steady_state.converged
    np.testing.assert_allclose(
        loop_steady_state.steady_state,
        explicit_steady_state.steady_state,
        rtol=1e-12,
        atol=1e-12,
    )

    loop_jacobian = calculate_jacobian(loop_model, steady_state=loop_steady_state.steady_state)
    explicit_jacobian = calculate_jacobian(
        explicit_model,
        steady_state=explicit_steady_state.steady_state,
    )
    np.testing.assert_allclose(loop_jacobian, explicit_jacobian, rtol=1e-12, atol=1e-12)

    loop_solution = solve_first_order_model(loop_model, steady_state=loop_steady_state.steady_state)
    explicit_solution = solve_first_order_model(
        explicit_model,
        steady_state=explicit_steady_state.steady_state,
    )
    np.testing.assert_allclose(
        loop_solution.solution.solution_matrix,
        explicit_solution.solution.solution_matrix,
        rtol=1e-12,
        atol=1e-12,
    )


def test_top_level_symbolic_for_block_matches_explicit_indexed_equations() -> None:
    _assert_models_match(INDEXED_BLOCK_SOURCE, INDEXED_BLOCK_EXPLICIT_SOURCE)


def test_inline_symbolic_for_sum_matches_explicit_indexed_equation() -> None:
    _assert_models_match(INDEXED_SUM_SOURCE, INDEXED_SUM_EXPLICIT_SOURCE)


def test_nested_index_parameter_names_resolve_and_preserve_original_syntax() -> None:
    model = parse_macro_model(NESTED_INDEX_PARAMETER_SOURCE)
    steady_state_result = solve_steady_state(model)

    assert model.parameter_names == ("rho{F}{H}", "rho{H}{F}")
    assert steady_state_result.converged
    np.testing.assert_allclose(
        steady_state_result.parameter_values,
        np.asarray([0.5, 0.5]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        np.asarray([1.0]),
        rtol=0,
        atol=1e-12,
    )
