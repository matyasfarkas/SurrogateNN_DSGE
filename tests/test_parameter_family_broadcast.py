from __future__ import annotations

import numpy as np

from surrogatenn_dsge import (
    calculate_jacobian,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
)


DIRECT_PARAMETER_FAMILY_SOURCE = """
@model direct_parameter_family begin
    y{H}[0] = alpha{H} * y{H}[-1] + c{H}
    y{F}[0] = alpha{F} * y{F}[-1] + c{F}
end

@parameters direct_parameter_family begin
    alpha = 0.5
    c{H} = 1.0
    c{F} = 2.0
end
"""


DIRECT_PARAMETER_FAMILY_EXPLICIT_SOURCE = """
@model direct_parameter_family_explicit begin
    y{H}[0] = alpha{H} * y{H}[-1] + c{H}
    y{F}[0] = alpha{F} * y{F}[-1] + c{F}
end

@parameters direct_parameter_family_explicit begin
    alpha{H} = 0.5
    alpha{F} = 0.5
    c{H} = 1.0
    c{F} = 2.0
end
"""


CALIBRATION_PARAMETER_FAMILY_SOURCE = """
@model calibration_parameter_family begin
    y{H}[0] = phi{H} * y{H}[-1] + beta{H}
    y{F}[0] = phi{F} * y{F}[-1] + beta{F}
end

@parameters calibration_parameter_family begin
    phi = 0.5
    target = 2.0
    y[ss] = target | beta
end
"""


CALIBRATION_PARAMETER_FAMILY_EXPLICIT_SOURCE = """
@model calibration_parameter_family_explicit begin
    y{H}[0] = phi{H} * y{H}[-1] + beta{H}
    y{F}[0] = phi{F} * y{F}[-1] + beta{F}
end

@parameters calibration_parameter_family_explicit begin
    phi{H} = 0.5
    phi{F} = 0.5
    target = 2.0
    y{H}[ss] = target | beta{H}
    y{F}[ss] = target | beta{F}
end
"""


def _parameter_mapping(model, values) -> dict[str, float]:
    return dict(zip(model.parameter_names, np.asarray(values, dtype=np.float64).tolist()))


def _assert_models_match(loop_source: str, explicit_source: str) -> None:
    loop_model = parse_macro_model(loop_source)
    explicit_model = parse_macro_model(explicit_source)

    assert loop_model.parameter_names == explicit_model.parameter_names
    assert loop_model.calibrated_parameter_names == explicit_model.calibrated_parameter_names
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
    np.testing.assert_allclose(
        loop_steady_state.parameter_values,
        explicit_steady_state.parameter_values,
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


def test_direct_parameter_family_broadcast_matches_explicit_definitions() -> None:
    model = parse_macro_model(DIRECT_PARAMETER_FAMILY_SOURCE)
    steady_state_result = solve_steady_state(model)

    assert model.parameter_names == ("alpha{F}", "alpha{H}", "c{F}", "c{H}")
    assert steady_state_result.converged
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        np.asarray([4.0, 2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        steady_state_result.parameter_values,
        np.asarray([0.5, 0.5, 2.0, 1.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )

    _assert_models_match(
        DIRECT_PARAMETER_FAMILY_SOURCE,
        DIRECT_PARAMETER_FAMILY_EXPLICIT_SOURCE,
    )


def test_calibration_parameter_family_broadcast_matches_explicit_equations() -> None:
    model = parse_macro_model(CALIBRATION_PARAMETER_FAMILY_SOURCE)
    steady_state_result = solve_steady_state(model)
    parameter_map = _parameter_mapping(model, steady_state_result.parameter_values)

    assert set(model.calibrated_parameter_names) == {"beta{H}", "beta{F}"}
    assert steady_state_result.converged
    np.testing.assert_allclose(parameter_map["phi{H}"], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(parameter_map["phi{F}"], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(parameter_map["beta{H}"], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(parameter_map["beta{F}"], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(parameter_map["target"], 2.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        np.asarray([2.0, 2.0], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )

    _assert_models_match(
        CALIBRATION_PARAMETER_FAMILY_SOURCE,
        CALIBRATION_PARAMETER_FAMILY_EXPLICIT_SOURCE,
    )
