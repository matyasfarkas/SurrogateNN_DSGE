from __future__ import annotations

import numpy as np

import surrogatenn_dsge.model as model_module
from surrogatenn_dsge import parse_macro_model, solve_steady_state


def test_newton_solver_falls_back_to_regularized_normal_equations(
    monkeypatch,
) -> None:
    def residual_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                x[0] + x[1] - 1.0,
                x[0] + x[1] - 1.0,
            ],
            dtype=np.float64,
        )

    def jacobian_fn(_: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )

    original_lstsq = model_module.np.linalg.lstsq

    def failing_lstsq(*args, **kwargs):
        raise np.linalg.LinAlgError("forced lstsq failure")

    monkeypatch.setattr(model_module.np.linalg, "lstsq", failing_lstsq)
    try:
        solution, converged, iterations, residual_norm = model_module._solve_newton_system(
            np.asarray([0.0, 0.0], dtype=np.float64),
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            lower_bounds=None,
            upper_bounds=None,
            tol=1e-12,
            max_iter=10,
            line_search_min_step=2.0**-16,
            nonfinite_message="unexpected non-finite residual",
        )
    finally:
        monkeypatch.setattr(model_module.np.linalg, "lstsq", original_lstsq)

    np.testing.assert_allclose(solution.sum(), 1.0, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        residual_fn(solution),
        np.zeros(2, dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )
    assert converged
    assert iterations <= 2
    assert residual_norm <= 1e-8


def test_steady_state_solver_restarts_when_default_guess_is_nonfinite() -> None:
    source = """
@model sqrt_domain begin
    sqrt(n[0] - 2.0)
end

@parameters sqrt_domain begin
end
"""
    model = parse_macro_model(source)
    result = solve_steady_state(model)

    assert result.converged
    np.testing.assert_allclose(
        np.asarray(result.steady_state, dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )
