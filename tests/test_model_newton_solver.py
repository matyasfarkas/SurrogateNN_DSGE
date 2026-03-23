from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import surrogatenn_dsge.model as model_module
from surrogatenn_dsge import parse_macro_model, solve_steady_state, solve_steady_state_jax


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


def test_steady_state_solver_uses_unit_scale_restart_candidates_for_domain_failures() -> None:
    source = """
@model capital_unit_restart begin
    sqrt(1.0 - capital[0])
end

@parameters capital_unit_restart begin
end
"""
    model = parse_macro_model(source)
    result = solve_steady_state(model)

    assert result.converged
    np.testing.assert_allclose(
        np.asarray(result.steady_state, dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )


def test_steady_state_solver_uses_large_geometric_restart_candidates() -> None:
    source = """
@model asset_geometric_restart begin
    sqrt(asset[0] - 10.0)
end

@parameters asset_geometric_restart begin
end
"""
    model = parse_macro_model(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = solve_steady_state(model)

    assert result.converged
    np.testing.assert_allclose(
        np.asarray(result.steady_state, dtype=np.float64),
        np.asarray([10.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    )


_CACHE_SOURCE = """
@model cache_model begin
    x[0] = level
end

@parameters cache_model begin
    level = 2.0
end
"""


def test_steady_state_solver_uses_cached_guess_for_nearby_parameters(
    monkeypatch,
) -> None:
    model = parse_macro_model(_CACHE_SOURCE)
    baseline = solve_steady_state(model)
    np.testing.assert_allclose(
        np.asarray(baseline.base_steady_state, dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )

    captured: dict[str, np.ndarray] = {}

    def fake_solver(
        x0: np.ndarray,
        *,
        residual_fn,
        jacobian_fn,
        default_guess,
        lower_bounds,
        upper_bounds,
        tol,
        max_iter,
        line_search_min_step,
        nonfinite_message,
    ):
        del (
            residual_fn,
            jacobian_fn,
            default_guess,
            lower_bounds,
            upper_bounds,
            tol,
            max_iter,
            line_search_min_step,
            nonfinite_message,
        )
        captured["x0"] = np.asarray(x0, dtype=np.float64)
        return np.asarray(x0, dtype=np.float64), True, 0, 0.0

    monkeypatch.setattr(model_module, "_solve_newton_system_with_restarts", fake_solver)
    solve_steady_state(model, parameter_values=[2.1])

    np.testing.assert_allclose(
        captured["x0"],
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_jax_steady_state_solver_uses_cached_guess_for_nearby_parameters(
    monkeypatch,
) -> None:
    model = parse_macro_model(_CACHE_SOURCE)
    baseline = solve_steady_state(model)
    np.testing.assert_allclose(
        np.asarray(baseline.base_steady_state, dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )

    captured: dict[str, np.ndarray] = {}

    def fake_solver(
        x0,
        *,
        residual_fn,
        jacobian_fn,
        default_guess,
        lower_bounds,
        upper_bounds,
        tol,
        max_iter,
        line_search_min_step,
    ):
        del (
            residual_fn,
            jacobian_fn,
            default_guess,
            lower_bounds,
            upper_bounds,
            tol,
            max_iter,
            line_search_min_step,
        )
        captured["x0"] = np.asarray(x0, dtype=np.float64)
        return jnp.asarray(x0, dtype=jnp.float64), True, 0, 0.0

    monkeypatch.setattr(model_module, "_solve_newton_system_jax_with_restarts", fake_solver)
    solve_steady_state_jax(model, parameter_values=[2.1])

    np.testing.assert_allclose(
        captured["x0"],
        np.asarray([2.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
