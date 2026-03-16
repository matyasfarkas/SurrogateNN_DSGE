from __future__ import annotations

import numpy as np

import surrogatenn_dsge.model as model_module
from surrogatenn_dsge import (
    calculate_hessian,
    calculate_jacobian,
    calculate_third_order_derivatives,
    evaluate_dynamic_residual,
    parse_macro_model,
    solve_steady_state,
)


SIMPLE_SOURCE = """
@model lazy_model begin
    x[0] = rho * x[-1] + eps[x]
end

@parameters lazy_model begin
    rho = 0.9
end
"""


def _count_lambdify_calls(monkeypatch) -> list[int]:
    call_count = [0]
    original = model_module.sp.lambdify

    def counted_lambdify(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module.sp, "lambdify", counted_lambdify)
    return call_count


def test_parse_macro_model_defers_lambdify_until_first_use(monkeypatch) -> None:
    call_count = _count_lambdify_calls(monkeypatch)

    model = parse_macro_model(SIMPLE_SOURCE)

    assert call_count[0] == 0

    steady_state_result = solve_steady_state(model)

    assert steady_state_result.converged
    assert call_count[0] > 0


def test_lazy_symbolic_functions_are_cached_after_first_use(monkeypatch) -> None:
    call_count = _count_lambdify_calls(monkeypatch)
    model = parse_macro_model(SIMPLE_SOURCE)

    steady_state_result = solve_steady_state(model)
    after_steady_state = call_count[0]
    solve_steady_state(model)
    assert call_count[0] == after_steady_state

    steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
    calculate_jacobian(model, steady_state=steady_state)
    after_jacobian = call_count[0]
    calculate_jacobian(model, steady_state=steady_state)
    assert call_count[0] == after_jacobian

    calculate_hessian(model, steady_state=steady_state)
    after_hessian = call_count[0]
    calculate_hessian(model, steady_state=steady_state)
    assert call_count[0] == after_hessian

    calculate_third_order_derivatives(model, steady_state=steady_state)
    after_third_order = call_count[0]
    calculate_third_order_derivatives(model, steady_state=steady_state)
    assert call_count[0] == after_third_order

    zero_state = np.zeros((model.timings.nVars,), dtype=np.float64)
    evaluate_dynamic_residual(
        model,
        zero_state,
        zero_state,
        zero_state,
        shock=np.zeros((model.timings.nExo,), dtype=np.float64),
        steady_state=steady_state,
    )
    after_residual = call_count[0]
    evaluate_dynamic_residual(
        model,
        zero_state,
        zero_state,
        zero_state,
        shock=np.zeros((model.timings.nExo,), dtype=np.float64),
        steady_state=steady_state,
    )
    assert call_count[0] == after_residual
