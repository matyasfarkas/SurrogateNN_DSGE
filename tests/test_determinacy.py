from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    analyze_first_order_dsge_determinacy,
    analyze_first_order_model_determinacy,
    parse_macro_model,
    solve_first_order_model,
    solve_steady_state,
)


UNIQUE_SOURCE = """
@model det begin
    k[0] = rho * k[-1]
    q[0] = beta * q[1]
end

@parameters det begin
    rho = 0.9
    beta = 0.95
end
"""


INDETERMINATE_SOURCE = """
@model det begin
    k[0] = rho * k[-1]
    q[1] = beta * q[0]
end

@parameters det begin
    rho = 0.9
    beta = 0.95
end
"""


NO_STABLE_SOURCE = """
@model det begin
    k[-1] = rho * k[0]
    q[0] = beta * q[1]
end

@parameters det begin
    rho = 0.9
    beta = 0.95
end
"""


def test_low_level_first_order_determinacy_reports_unique_stable_solution() -> None:
    model = parse_macro_model(UNIQUE_SOURCE)
    parameter_values = np.asarray(model.parameter_values, dtype=np.float64)
    steady_state_result = solve_steady_state(
        model,
        parameter_values=parameter_values,
        initial_guess={"k": 0.0, "q": 0.0},
    )
    jacobian = model.calculate_jacobian(
        parameter_values=parameter_values,
        steady_state=steady_state_result.steady_state,
    )

    determinacy = analyze_first_order_dsge_determinacy(jacobian, model.timings)

    assert determinacy.unique_stable_solution
    assert determinacy.classification == "unique_stable_solution"
    assert determinacy.solution.converged
    assert determinacy.qme_diagnostics.stable_count == 1
    assert determinacy.qme_diagnostics.expected_stable_count == 1
    assert determinacy.qme_diagnostics.decomposition_succeeded
    assert determinacy.qme_diagnostics.invariant_subspace_invertible
    assert determinacy.qme_diagnostics.solution_extracted
    assert determinacy.qme_diagnostics.relative_residual < 1e-8


@pytest.mark.parametrize(
    ("source", "classification", "stable_relation", "solver_converged"),
    [
        (UNIQUE_SOURCE, "unique_stable_solution", "equal", True),
        (INDETERMINATE_SOURCE, "indeterminate", "greater", False),
        (NO_STABLE_SOURCE, "no_stable_solution", "less", False),
    ],
)
def test_parsed_model_determinacy_classifies_schur_cases(
    source: str,
    classification: str,
    stable_relation: str,
    solver_converged: bool,
) -> None:
    model = parse_macro_model(source)
    parameter_values = np.asarray(model.parameter_values, dtype=np.float64)

    determinacy_result = analyze_first_order_model_determinacy(
        model,
        parameter_values=parameter_values,
        steady_state_initial_guess={"k": 0.0, "q": 0.0},
    )
    first_order_result = solve_first_order_model(
        model,
        parameter_values=parameter_values,
        steady_state_initial_guess={"k": 0.0, "q": 0.0},
    )

    diagnostics = determinacy_result.determinacy.qme_diagnostics
    assert determinacy_result.determinacy.classification == classification
    assert determinacy_result.determinacy.unique_stable_solution is (
        classification == "unique_stable_solution"
    )
    assert first_order_result.solution.converged is solver_converged
    assert determinacy_result.determinacy.solution.converged is solver_converged

    if stable_relation == "equal":
        assert diagnostics.stable_count == diagnostics.expected_stable_count
    elif stable_relation == "greater":
        assert diagnostics.stable_count > diagnostics.expected_stable_count
    elif stable_relation == "less":
        assert diagnostics.stable_count < diagnostics.expected_stable_count
    else:
        raise AssertionError(f"Unexpected stable_relation {stable_relation!r}.")
