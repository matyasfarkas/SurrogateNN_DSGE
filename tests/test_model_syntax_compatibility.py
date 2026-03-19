from __future__ import annotations

from surrogatenn_dsge import parse_macro_model, solve_first_order_model


def test_parser_accepts_residual_only_equations() -> None:
    source = """
@model residual_only begin
    y[0] - rho * y[-1] - eps[x]
end

@parameters residual_only begin
    rho = 0.7
end
"""
    model = parse_macro_model(source)
    result = solve_first_order_model(model, steady_state_initial_guess={"y": 0.0})

    assert result.solution.converged


def test_parser_accepts_multiline_obc_equations() -> None:
    source = """
@model multiline_obc begin
    r[0] = max(r_bar,
               rho * r[-1] +
               eps[x])
end

@parameters multiline_obc begin
    r_bar = 1.0
    rho = 0.8
end
"""
    model = parse_macro_model(source)

    assert model.has_obc
    assert len(model.equations) == 1


def test_parser_accepts_multiline_parameter_expressions_with_unknown_functions() -> None:
    source = """
@model multiline_parameter_function begin
    y[0] = alpha * y[-1]
end

@parameters multiline_parameter_function begin
    alpha = external_calibration(
        beta,
        gamma
    )
    beta = 0.9
    gamma = 0.1
end
"""
    model = parse_macro_model(source)

    assert set(model.parameter_names) == {"alpha", "beta", "gamma"}


def test_parser_accepts_unicode_superscript_identifiers() -> None:
    source = """
@model unicode_superscripts begin
    x[0] = A¹¹ * x[-1] + σᶻ * ϵᶻ[x]
end

@parameters unicode_superscripts begin
    A¹¹ = 0.9
    σᶻ = 0.05
end
"""
    model = parse_macro_model(source)
    result = solve_first_order_model(model, steady_state_initial_guess={"x": 0.0})

    assert result.solution.converged
