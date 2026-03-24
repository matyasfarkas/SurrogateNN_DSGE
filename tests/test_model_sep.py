from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    SEPConfig,
    evaluate_dynamic_residual,
    homotopy_sep,
    parse_macro_model,
    solve_sep_at_noise_level,
    solve_stochastic_extended_path_model,
    solve_stochastic_extended_path_residual_expectation,
)
import surrogatenn_dsge.model as model_module


NONLINEAR_SEP_SOURCE = """
@model nonlinear_sep begin
    y[0] = rho * y[-1] + gamma * y[1]^2 + u[x]
end

@parameters nonlinear_sep begin
    gamma = 0.15
    rho = 0.25
end
"""


def test_evaluate_dynamic_residual_matches_manual_equation() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)

    residual = evaluate_dynamic_residual(
        model,
        lag_state=[0.5],
        current_state=[0.7],
        lead_state=[-0.2],
        shock=[0.1],
        steady_state=[0.0],
    )

    expected = 0.7 - (0.25 * 0.5 + 0.15 * (-0.2) ** 2 + 0.1)
    np.testing.assert_allclose(
        residual,
        jnp.asarray([expected], dtype=jnp.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parsed_model_sep_matches_manual_conditional_residual_solver() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    config = SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10)
    deterministic = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)

    def conditional_residual(
        y_prev: jnp.ndarray,
        y_curr: jnp.ndarray,
        y_next: jnp.ndarray,
        shock: jnp.ndarray,
        params: tuple[float, float],
    ) -> jnp.ndarray:
        rho, gamma = params
        return y_curr - (rho * y_prev + gamma * y_next**2 + shock)

    manual = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=config,
        deterministic_shocks=deterministic,
        params=(0.25, 0.15),
    )
    parsed = solve_stochastic_extended_path_model(
        model,
        config=config,
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )

    assert manual.converged
    assert parsed.solution.converged
    assert parsed.solution.jacobian_method == "autodiff"
    np.testing.assert_allclose(parsed.steady_state, 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        parsed.solution.stacked_states,
        manual.stacked_states,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        parsed.solution.mean_path,
        manual.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )


def test_parsed_model_sep_hmc_backend_runs() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    solution = solve_stochastic_extended_path_model(
        model,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=16,
            hmc_warmup=8,
            hmc_leapfrog_steps=6,
            hmc_step_size=0.07,
            hmc_seed=11,
            tol=1e-8,
        ),
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )

    assert solution.solution.converged
    assert np.all(np.isfinite(solution.solution.mean_path))


def test_parsed_model_sep_builds_linear_first_order_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    captured: dict[str, object] = {}
    original_solver = model_module.solve_stochastic_extended_path_residual_expectation

    def wrapped_solver(*args: object, **kwargs: object):
        captured["initial_guess"] = kwargs.get("initial_guess")
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(
        model_module,
        "solve_stochastic_extended_path_residual_expectation",
        wrapped_solver,
    )

    result = solve_stochastic_extended_path_model(
        model,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-10),
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )

    assert result.solution.converged
    assert captured["initial_guess"] is not None
    np.testing.assert_allclose(
        np.asarray(captured["initial_guess"], dtype=np.float64).reshape(3),
        np.asarray([0.2, 0.05, 0.0125], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_explicit_sep_initial_guess_overrides_linear_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    captured: dict[str, object] = {}
    original_solver = model_module.solve_stochastic_extended_path_residual_expectation
    explicit_guess = np.asarray([[0.7], [0.6], [0.5]], dtype=np.float64)

    def wrapped_solver(*args: object, **kwargs: object):
        captured["initial_guess"] = kwargs.get("initial_guess")
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(
        model_module,
        "solve_stochastic_extended_path_residual_expectation",
        wrapped_solver,
    )

    result = solve_stochastic_extended_path_model(
        model,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-10),
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
        initial_guess=explicit_guess,
    )

    assert result.solution.converged
    np.testing.assert_allclose(
        np.asarray(captured["initial_guess"], dtype=np.float64),
        explicit_guess,
        rtol=0.0,
        atol=0.0,
    )


def test_solve_sep_at_noise_level_scales_shocks_and_forces_deterministic_sigma_zero() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    base_config = SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10)

    sigma_zero = solve_sep_at_noise_level(
        model,
        sigma=0.0,
        config=base_config,
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )
    deterministic = solve_stochastic_extended_path_model(
        model,
        config=SEPConfig(periods=3, branching_order=0, nnodes=3, tol=1e-10),
        deterministic_shocks={"u": [0.0, 0.0, 0.0]},
    )
    sigma_half = solve_sep_at_noise_level(
        model,
        sigma=0.5,
        config=base_config,
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )
    half_scaled = solve_stochastic_extended_path_model(
        model,
        config=base_config,
        deterministic_shocks={"u": [0.1, 0.0, 0.0]},
    )

    assert sigma_zero.solution.accepted
    assert sigma_half.solution.accepted
    np.testing.assert_allclose(
        sigma_zero.solution.mean_path,
        deterministic.solution.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        sigma_half.solution.mean_path,
        half_scaled.solution.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )


def test_homotopy_sep_adapts_by_subdividing_failed_sigma_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)
    calls: list[float] = []

    def fake_solve_sep_at_noise_level(
        _model: object,
        *,
        sigma: float,
        **kwargs: object,
    ) -> model_module.ParsedModelSEPResult:
        calls.append(float(sigma))
        attempt_count = sum(abs(value - float(sigma)) < 1e-12 for value in calls)
        accepted = not (abs(float(sigma) - 1.0) < 1e-12 and attempt_count == 1)
        return model_module.ParsedModelSEPResult(
            steady_state=np.asarray([0.0], dtype=np.float64),
            parameter_values=np.asarray([0.25, 0.15], dtype=np.float64),
            solution=model_module.SEPSolution(
                stacked_states=jnp.asarray([float(sigma)], dtype=jnp.float64),
                mean_path=jnp.asarray(
                    [[0.0, float(sigma)]],
                    dtype=jnp.float64,
                ),
                residual_norm=1e-6 if accepted else 1.0,
                converged=accepted,
                accepted=accepted,
                iterations=1,
                group_counts=(1, 1),
                jacobian_method="autodiff",
            ),
        )

    monkeypatch.setattr(
        model_module,
        "solve_sep_at_noise_level",
        fake_solve_sep_at_noise_level,
    )

    result = homotopy_sep(
        model,
        n_steps=1,
        adaptive=True,
        max_retries=1,
        config=SEPConfig(periods=1, branching_order=1, tol=1e-10),
        deterministic_shocks={"u": [0.2]},
    )

    assert result.success
    assert result.sigma_path == (0.0, 0.5, 1.0)
    np.testing.assert_allclose(
        np.asarray(calls, dtype=np.float64),
        np.asarray([0.0, 1.0, 0.5, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_homotopy_sep_runs_on_parsed_nonlinear_model() -> None:
    model = parse_macro_model(NONLINEAR_SEP_SOURCE)

    result = homotopy_sep(
        model,
        n_steps=3,
        adaptive=True,
        max_retries=2,
        config=SEPConfig(periods=3, branching_order=1, nnodes=3, tol=1e-8),
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )

    assert result.success
    assert result.result.solution.accepted
    assert result.sigma_path[0] == 0.0
    assert abs(result.sigma_path[-1] - 1.0) < 1e-12
    assert np.all(np.isfinite(np.asarray(result.result.solution.mean_path, dtype=np.float64)))
