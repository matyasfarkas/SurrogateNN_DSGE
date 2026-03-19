from __future__ import annotations

from pathlib import Path

import numpy as np

from surrogatenn_dsge import (
    parse_macro_model,
    resolve_parameter_values,
    solve_first_order_model,
    solve_steady_state,
)


QMIPF_HELPER_SOURCE = """
@model m begin
    y[0] = a
end

@parameters m begin
    a = QMIPF_solve_SS(1.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
end
"""


def test_parameter_block_evaluates_qmipf_helper_function() -> None:
    model = parse_macro_model(QMIPF_HELPER_SOURCE)

    parameter_lookup = dict(
        zip(model.parameter_names, np.asarray(model.parameter_values, dtype=np.float64))
    )
    assert set(parameter_lookup) == {"a"}
    np.testing.assert_allclose(parameter_lookup["a"], 0.2, rtol=0.0, atol=1e-10)

    steady_state = solve_steady_state(model)
    assert steady_state.converged
    np.testing.assert_allclose(
        np.asarray(steady_state.steady_state, dtype=np.float64),
        np.asarray([0.2], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def test_qipf_source_evaluates_helper_backed_parameter_definition() -> None:
    path = Path(
        "/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/models/qipf.jl"
    )
    model = parse_macro_model(path.read_text())

    parameter_lookup = dict(
        zip(model.parameter_names, np.asarray(model.parameter_values, dtype=np.float64))
    )
    assert "SS_N_ST" in parameter_lookup
    assert np.isfinite(parameter_lookup["SS_N_ST"])
    assert parameter_lookup["SS_N_ST"] > 0.0
    assert not np.isclose(parameter_lookup["SS_N_ST"], 1.0)

    resolved = dict(
        zip(
            model.parameter_names,
            np.asarray(resolve_parameter_values(model), dtype=np.float64),
        )
    )
    np.testing.assert_allclose(
        resolved["SS_N_ST"],
        parameter_lookup["SS_N_ST"],
        rtol=0.0,
        atol=1e-10,
    )

    first_order = solve_first_order_model(model)
    assert first_order.solution is not None
