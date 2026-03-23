from __future__ import annotations

import jax
import numpy as np

from surrogatenn_dsge import (
    SEPConfig,
    build_linear_state_space_from_model,
    first_order_inversion_loglikelihood,
    get_sep_inversion_last_diagnostics,
    inversion_loglikelihood_from_model,
    inversion_loglikelihood_per_period_from_model,
    parse_macro_model,
    reset_sep_inversion_last_diagnostics,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
    solve_stochastic_extended_path_model,
)


LINEAR_INVERSION_SOURCE = """
@model inversion_linear begin
    y[0] = rho * y[-1] + eps[x]
end

@parameters inversion_linear begin
    0 < rho < 1
    rho = 0.7
end
"""

NONLINEAR_SEP_INVERSION_SOURCE = """
@model sep_inversion_nonlinear begin
    y[0] = rho * y[-1] + gamma * y[1]^2 + u[x]
end

@parameters sep_inversion_nonlinear begin
    gamma = 0.1
    rho = 0.35
end
"""


def _linear_inversion_fixture():
    model = parse_macro_model(LINEAR_INVERSION_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"y": 0.0},
    )
    state_space = build_linear_state_space_from_model(
        model,
        ("y",),
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
    )
    simulation = simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(0),
        num_periods=24,
    )
    levels = np.asarray(simulation.observations, dtype=np.float64)
    return model, first_order_result, simulation, levels


def _sep_inversion_fixture():
    model = parse_macro_model(NONLINEAR_SEP_INVERSION_SOURCE)
    config = SEPConfig(periods=3, branching_order=1, nnodes=3, tol=1e-10)
    solution = solve_stochastic_extended_path_model(
        model,
        config=config,
        deterministic_shocks={"u": [0.2, 0.0, 0.0]},
    )
    levels = np.asarray(solution.solution.mean_path[:, 1:], dtype=np.float64)
    return model, config, levels


def test_first_order_inversion_matches_low_level_and_is_jittable() -> None:
    model, first_order_result, simulation, levels = _linear_inversion_fixture()

    high_level = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
    )
    per_period = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
    )
    low_level = first_order_inversion_loglikelihood(
        first_order_result.solution.solution_matrix,
        model.timings,
        simulation.observations,
        (0,),
    )
    compiled = jax.jit(
        lambda obs: first_order_inversion_loglikelihood(
            first_order_result.solution.solution_matrix,
            model.timings,
            obs,
            (0,),
        )
    )

    np.testing.assert_allclose(high_level, low_level, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(high_level, np.sum(per_period), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        compiled(simulation.observations),
        low_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_first_order_inversion_accepts_warmup_and_failure_value() -> None:
    model, _, _, levels = _linear_inversion_fixture()
    warmup_value = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        warmup_iterations=2,
    )
    assert np.isfinite(warmup_value)

    bad_parameters = np.asarray(model.parameter_values, dtype=np.float64).copy()
    bad_parameters[model.parameter_names.index("rho")] = 1.2
    failure_value = -1e9
    failed_total = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        parameter_values=bad_parameters,
        steady_state_initial_guess={"y": 0.0},
        on_failure_loglikelihood=failure_value,
    )
    failed_per_period = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        parameter_values=bad_parameters,
        steady_state_initial_guess={"y": 0.0},
        on_failure_loglikelihood=failure_value,
    )

    np.testing.assert_allclose(failed_total, failure_value, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        failed_per_period,
        np.full((levels.shape[1],), failure_value),
        rtol=0.0,
        atol=0.0,
    )


def test_sep_inversion_runs_is_deterministic_and_records_diagnostics() -> None:
    model, config, levels = _sep_inversion_fixture()
    failure_value = -1e12

    reset_sep_inversion_last_diagnostics()
    value_1 = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        on_failure_loglikelihood=failure_value,
    )
    diagnostics = get_sep_inversion_last_diagnostics()
    value_2 = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        on_failure_loglikelihood=failure_value,
    )
    per_period = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        on_failure_loglikelihood=failure_value,
    )

    assert np.isfinite(value_1)
    assert value_1 > failure_value / 10.0
    np.testing.assert_allclose(value_1, value_2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(value_1, np.sum(per_period), rtol=1e-10, atol=1e-10)
    assert diagnostics is not None
    assert diagnostics["kind"] == "sep_inversion_filter"
    assert diagnostics["status"] == "ok"
    assert diagnostics["n_periods"] == levels.shape[1]
    assert diagnostics["sep_order"] == config.branching_order


def test_sep_inversion_accepts_runtime_overrides() -> None:
    model, _, levels = _sep_inversion_fixture()

    reset_sep_inversion_last_diagnostics()
    value = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        sep_periods=4,
        sep_order=1,
        sep_nnodes=3,
        sep_sparse_tree=True,
        sep_maxit=60,
        sep_tol=1e-7,
        sep_accept_tol=0.25,
        sep_inv_maxit=8,
        sep_inv_resid_tol=1e-6,
        sep_inv_step_tol=1e-6,
        sep_inv_lambda=1e-4,
        on_failure_loglikelihood=-1e12,
    )
    diagnostics = get_sep_inversion_last_diagnostics()

    assert np.isfinite(value)
    assert diagnostics is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["sep_periods"] == 4
    assert diagnostics["sep_order"] == 1
    assert diagnostics["sep_nnodes"] == 3
    assert diagnostics["sep_sparse_tree"] is True
    assert diagnostics["sep_inv_maxit"] == 8


def test_sep_inversion_honors_sparse_tree_in_config() -> None:
    model, _, levels = _sep_inversion_fixture()
    config = SEPConfig(
        periods=4,
        branching_order=1,
        nnodes=3,
        sparse_tree=True,
        tol=1e-7,
    )

    reset_sep_inversion_last_diagnostics()
    value = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        on_failure_loglikelihood=-1e12,
    )
    diagnostics = get_sep_inversion_last_diagnostics()

    assert np.isfinite(value)
    assert diagnostics is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["sep_sparse_tree"] is True
    assert diagnostics["sep_periods"] == 4


def test_sep_inversion_reuses_shifted_period_warm_starts() -> None:
    model, _, levels = _sep_inversion_fixture()
    config = SEPConfig(
        periods=4,
        branching_order=1,
        nnodes=3,
        sparse_tree=True,
        tol=1e-7,
    )

    reset_sep_inversion_last_diagnostics()
    value = inversion_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        on_failure_loglikelihood=-1e12,
    )
    diagnostics = get_sep_inversion_last_diagnostics()

    assert np.isfinite(value)
    assert diagnostics is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["sep_carry_warm_start_strategy"] == "shifted_tree"
    assert diagnostics["sep_period_carry_warm_start_used"] == [False, True, True]
    assert diagnostics["sep_total_predict_calls"] >= levels.shape[1]
    assert diagnostics["sep_period_predict_calls"][0] >= 1
    assert diagnostics["sep_period_predict_calls"][1] >= 1
    assert diagnostics["sep_period_predict_calls"][2] >= 1
    assert sum(diagnostics["sep_period_predict_calls"]) == diagnostics["sep_total_predict_calls"]
