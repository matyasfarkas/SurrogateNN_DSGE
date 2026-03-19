from __future__ import annotations

from pathlib import Path

import numpy as np

from surrogatenn_dsge import parse_macro_model, solve_steady_state


TAIL_IF_TRUE_SOURCE = """
@model conditional_model begin
    y[0] = x
end

@parameters conditional_model begin
    sigma = 1
    x = 1
    if sigma == 1
    x = 2
end
"""


TAIL_IF_ELSE_SOURCE = """
@model conditional_else_model begin
    y[0] = x
end

@parameters conditional_else_model begin
    sigma = 2
    x = 1
    if sigma == 1
    x = 2
    else
    x = 3
end
"""


def test_parameter_tail_if_overrides_previous_definition() -> None:
    model = parse_macro_model(TAIL_IF_TRUE_SOURCE)

    parameter_lookup = dict(
        zip(model.parameter_names, np.asarray(model.parameter_values, dtype=np.float64))
    )
    np.testing.assert_allclose(parameter_lookup["x"], 2.0, rtol=0.0, atol=1e-12)

    steady_state = solve_steady_state(model)
    assert steady_state.converged
    np.testing.assert_allclose(
        np.asarray(steady_state.steady_state, dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_parameter_tail_if_else_selects_active_branch() -> None:
    model = parse_macro_model(TAIL_IF_ELSE_SOURCE)

    parameter_lookup = dict(
        zip(model.parameter_names, np.asarray(model.parameter_values, dtype=np.float64))
    )
    np.testing.assert_allclose(parameter_lookup["x"], 3.0, rtol=0.0, atol=1e-12)


def test_upstream_qipf_minimum_tail_if_source_parses() -> None:
    path = Path(
        "/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/models/QIPF/testttfmodel_ttf_minimum.jl"
    )
    model = parse_macro_model(path.read_text())

    parameter_lookup = dict(
        zip(model.parameter_names, np.asarray(model.parameter_values, dtype=np.float64))
    )
    assert model.name == "qipf"
    assert parameter_lookup["sigma"] == 1.0
    assert np.isfinite(parameter_lookup["SS_U"])
    assert np.isfinite(parameter_lookup["SS_U_ST"])
