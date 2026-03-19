from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    RegimeSwitchConfig,
    SwitchingLikelihoodConfig,
    compute_linear_gate_stats_from_filter,
    compute_linear_gate_stats_from_shocks,
    compute_linear_gate_stats_from_filter_model_jax,
    estimate_observed_shocks_matrix,
    estimate_observed_shocks_matrix_jax,
    estimate_observed_variables_matrix,
    estimate_observed_variables_matrix_jax,
    gate_probabilities,
    gate_probabilities_jax,
    linear_filter_full_state_initial,
    linear_filter_full_state_initial_jax,
    linear_filter_initial_state,
    linear_filter_initial_state_jax,
    parse_macro_model,
    rollout_first_order_solution,
    solve_first_order_model,
    switching_loglikelihood_from_model,
    switching_loglikelihood_from_model_jax,
)


LINEAR_FILTER_SOURCE = """
@model linear_filter_fixture begin
    y[0] = rho * y[-1] + eps[x]
end

@parameters linear_filter_fixture begin
    0 < rho < 1
    rho = 0.7
end
"""

_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_TEST_MODEL_DIR = _ROOT / "SurrogateNN_Estimation.jl" / "test" / "models"
_UPSTREAM_MODEL_DIR = _ROOT / "SurrogateNN_Estimation.jl" / "models"

_RBC_STEADY_STATE_GUESS = {
    "A": 1.0,
    "Pi": 1.0025,
    "R": 1.0035,
    "c": 1.2,
    "k": 9.4,
    "y": 1.42,
    "z_delta": 1.0,
}

_FILTER_SMOKE_MODELS = (
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR / "RBC_CME.jl",
        None,
        id="rbc_cme",
    ),
    pytest.param(
        _UPSTREAM_MODEL_DIR / "RBC_Dynare.jl",
        {
            "Capital": 10.0,
            "Consumption": 0.8,
            "Efficiency": 1.0,
            "Investment": 0.2,
            "Labour": 0.3,
            "Output": 1.0,
            "efficiency": 0.0,
        },
        id="rbc_dynare",
    ),
    pytest.param(
        _UPSTREAM_MODEL_DIR / "FS2000.jl",
        {
            "P": 1.0,
            "R": 1.0,
            "W": 1.0,
            "c": 0.8,
            "d": 0.0,
            "dA": 1.01,
            "e": 1.0,
            "gp_obs": 1.0,
            "gy_obs": 1.01,
            "k": 8.0,
            "l": 0.9,
            "log_gp_obs": 0.0,
            "log_gy_obs": 0.01,
            "m": 1.0,
            "n": 0.3,
            "y": 1.0,
        },
        id="fs2000",
    ),
    pytest.param(
        _UPSTREAM_MODEL_DIR / "RBC_baseline.jl",
        {
            "c": 0.55,
            "g": 0.20,
            "i": 0.25,
            "k": 10.4,
            "l": 1.0 / 3.0,
            "r": 0.128,
            "w": 2.0,
            "y": 1.0,
            "z": 1.0,
        },
        id="rbc_baseline",
    ),
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR / "Backus_Kehoe_Kydland_1992.jl",
        {
            "A{F}": 2.0 / 3.0,
            "A{H}": 2.0 / 3.0,
            "C{F}": 0.8,
            "C{H}": 0.8,
            "K{F}": 11.0,
            "K{H}": 11.0,
            "LAMBDA{F}": 1.0,
            "LAMBDA{H}": 1.0,
            "LGM": 0.5,
            "L{F}": 2.0 / 3.0,
            "L{H}": 2.0 / 3.0,
            "NX{F}": 0.0,
            "NX{H}": 0.0,
            "N{F}": 1.0 / 3.0,
            "N{H}": 1.0 / 3.0,
            "S{F}": 0.275,
            "S{H}": 0.275,
            "U{F}": 1.0,
            "U{H}": 1.0,
            "X{F}": 0.275,
            "X{H}": 0.275,
            "Y{F}": 1.1,
            "Y{H}": 1.1,
            "Z{F}": 1.0,
            "Z{H}": 1.0,
            "dLGM": 1.0,
            "dLGM_ann": 1.0,
        },
        id="backus_kehoe_kydland_1992",
    ),
)


def _linear_filter_fixture():
    model = parse_macro_model(LINEAR_FILTER_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"y": 0.0},
    )
    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    shock_name = model.timings.exo[0]
    shocks = np.asarray([[0.15, -0.05, 0.1, 0.0, -0.02]], dtype=np.float64)
    levels = np.asarray(
        rollout_first_order_solution(
            first_order_result.solution.solution_matrix,
            model.timings,
            shocks,
        ),
        dtype=np.float64,
    ) + steady_state[:, None]
    return model, first_order_result, shock_name, shocks, levels


def test_first_order_filter_helpers_match_exact_linear_inversion_recovery() -> None:
    model, first_order_result, shock_name, shocks, levels = _linear_filter_fixture()

    estimated_shocks = estimate_observed_shocks_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
        expected_rows=1,
        expected_cols=levels.shape[1],
    )
    estimated_variables, variable_names = estimate_observed_variables_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
        expected_rows=1,
        expected_cols=levels.shape[1],
    )
    terminal_state = linear_filter_initial_state(
        model,
        levels,
        ("y",),
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
    )
    initial_state = linear_filter_full_state_initial(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
    )

    np.testing.assert_allclose(estimated_shocks, shocks, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(estimated_variables, levels, rtol=1e-12, atol=1e-12)
    assert variable_names == ("y",)
    np.testing.assert_allclose(terminal_state, levels[:, -1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(initial_state, levels[:, 0], rtol=1e-12, atol=1e-12)

    from_filter = compute_linear_gate_stats_from_filter(
        model,
        levels,
        {"y": 0.1},
        {shock_name: 0.5},
        ("y",),
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
    )
    manual = compute_linear_gate_stats_from_shocks(
        model,
        levels,
        shocks,
        {"y": 0.1},
        {shock_name: 0.5},
        observables=("y",),
        first_order_result=first_order_result,
        initial_state=levels[:, 0],
    )

    np.testing.assert_allclose(
        from_filter.linear_observations,
        manual.linear_observations,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(from_filter.shocks, shocks, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_filter.e_stat, manual.e_stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_filter.f_stat, manual.f_stat, rtol=1e-12, atol=1e-12)


def test_first_order_filter_helpers_match_linear_kalman_recursion() -> None:
    model, first_order_result, _, shocks, levels = _linear_filter_fixture()
    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    rho = float(model.parameter_values[model.parameter_names.index("rho")])
    first_period_multiplier = 1.0 - rho**2

    estimated_shocks = estimate_observed_shocks_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="kalman",
        smooth=False,
    )
    estimated_variables, variable_names = estimate_observed_variables_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="kalman",
        smooth=False,
    )
    smoothed_shocks = estimate_observed_shocks_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter="kalman",
        smooth=True,
    )

    expected_filtered_shocks = shocks.copy()
    expected_filtered_shocks[:, 0] = first_period_multiplier * shocks[:, 0]
    expected_filtered_variables = rho * np.asarray(levels, dtype=np.float64)

    np.testing.assert_allclose(
        estimated_shocks,
        expected_filtered_shocks,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        estimated_variables,
        expected_filtered_variables,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        smoothed_shocks,
        expected_filtered_shocks,
        rtol=1e-6,
        atol=1e-6,
    )
    assert variable_names == ("y",)


def test_filter_helpers_accept_schur_qme_algorithm_without_explicit_first_order_result() -> None:
    model, _, shock_name, shocks, levels = _linear_filter_fixture()

    estimated_shocks = estimate_observed_shocks_matrix(
        model,
        levels,
        observables=("y",),
        steady_state_initial_guess={"y": 0.0},
        qme_algorithm="schur",
        filter="inversion",
    )
    estimated_variables, variable_names = estimate_observed_variables_matrix(
        model,
        levels,
        observables=("y",),
        steady_state_initial_guess={"y": 0.0},
        qme_algorithm="schur",
        filter="inversion",
    )
    gate_stats = compute_linear_gate_stats_from_filter(
        model,
        levels,
        {"y": 0.1},
        {shock_name: 0.5},
        ("y",),
        observables=("y",),
        steady_state_initial_guess={"y": 0.0},
        qme_algorithm="schur",
        filter="inversion",
    )

    np.testing.assert_allclose(estimated_shocks, shocks, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(estimated_variables, levels, rtol=1e-12, atol=1e-12)
    assert variable_names == ("y",)
    np.testing.assert_allclose(gate_stats.shocks, shocks, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("filter_name", "smooth", "rtol", "atol"),
    (
        pytest.param("inversion", False, 1e-12, 1e-12, id="inversion"),
        pytest.param("kalman", False, 1e-6, 1e-6, id="kalman"),
        pytest.param("kalman", True, 1e-6, 1e-6, id="kalman_smoothed"),
    ),
)
def test_jax_filter_helpers_match_concrete_paths(
    filter_name: str,
    smooth: bool,
    rtol: float,
    atol: float,
) -> None:
    model, first_order_result, shock_name, shocks, levels = _linear_filter_fixture()
    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)

    expected_shocks = estimate_observed_shocks_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter=filter_name,
        smooth=smooth,
    )
    expected_variables, _ = estimate_observed_variables_matrix(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter=filter_name,
        smooth=smooth,
    )
    expected_terminal = linear_filter_initial_state(
        model,
        levels,
        ("y",),
        observables=("y",),
        first_order_result=first_order_result,
        filter=filter_name,
        smooth=smooth,
    )
    expected_initial = linear_filter_full_state_initial(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        filter=filter_name,
        smooth=smooth,
    )
    expected_gate_stats = compute_linear_gate_stats_from_filter(
        model,
        levels,
        {"y": 0.1},
        {shock_name: 0.5},
        ("y",),
        observables=("y",),
        first_order_result=first_order_result,
        filter=filter_name,
        smooth=smooth,
    )

    compiled_shocks = jax.jit(
        lambda theta: estimate_observed_shocks_matrix_jax(
            model,
            levels,
            observables=("y",),
            parameter_values=theta,
            steady_state=steady_state,
            qme_algorithm="schur",
            filter=filter_name,
            smooth=smooth,
        )
    )
    compiled_variables = jax.jit(
        lambda theta: estimate_observed_variables_matrix_jax(
            model,
            levels,
            observables=("y",),
            parameter_values=theta,
            steady_state=steady_state,
            qme_algorithm="schur",
            filter=filter_name,
            smooth=smooth,
        )
    )
    compiled_terminal = jax.jit(
        lambda theta: linear_filter_initial_state_jax(
            model,
            levels,
            ("y",),
            observables=("y",),
            parameter_values=theta,
            steady_state=steady_state,
            qme_algorithm="schur",
            filter=filter_name,
            smooth=smooth,
        )
    )
    compiled_initial = jax.jit(
        lambda theta: linear_filter_full_state_initial_jax(
            model,
            levels,
            observables=("y",),
            parameter_values=theta,
            steady_state=steady_state,
            qme_algorithm="schur",
            filter=filter_name,
            smooth=smooth,
        )
    )
    compiled_gate_stats = jax.jit(
        lambda theta: compute_linear_gate_stats_from_filter_model_jax(
            model,
            levels,
            {"y": 0.1},
            {shock_name: 0.5},
            ("y",),
            observables=("y",),
            parameter_values=theta,
            steady_state=steady_state,
            qme_algorithm="schur",
            filter=filter_name,
            smooth=smooth,
        )
    )
    theta = jnp.asarray(model.parameter_values, dtype=jnp.float64)

    estimated_shocks = compiled_shocks(theta)
    estimated_variables = compiled_variables(theta)
    terminal_state = compiled_terminal(theta)
    initial_state = compiled_initial(theta)
    gate_stats = compiled_gate_stats(theta)

    np.testing.assert_allclose(estimated_shocks, expected_shocks, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        estimated_variables,
        expected_variables,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(terminal_state, expected_terminal, rtol=rtol, atol=atol)
    np.testing.assert_allclose(initial_state, expected_initial, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        gate_stats.linear_observations,
        expected_gate_stats.linear_observations,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(gate_stats.shocks, expected_gate_stats.shocks, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gate_stats.e_stat, expected_gate_stats.e_stat, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gate_stats.f_stat, expected_gate_stats.f_stat, rtol=rtol, atol=atol)


def test_jax_filter_gate_construction_composes_with_switching_likelihood() -> None:
    model, first_order_result, _, _, levels = _linear_filter_fixture()
    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    regime_config = RegimeSwitchConfig(
        gate_mode="soft",
        tau_eps=0.05,
        tau_y=0.05,
        beta_eps=2.0,
        beta_y=1.5,
        bias=-0.1,
        prob_floor=1e-4,
        prob_ceiling=1.0 - 1e-4,
    )
    switching_config = SwitchingLikelihoodConfig(soft_mixture="logsumexp")

    expected_stats = compute_linear_gate_stats_from_filter(
        model,
        levels,
        {"y": 0.1},
        {model.timings.exo[0]: 0.5},
        ("y",),
        observables=("y",),
        first_order_result=first_order_result,
        filter="inversion",
    )
    expected_gate_probs = gate_probabilities(
        np.asarray(expected_stats.e_stat, dtype=np.float64),
        np.asarray(expected_stats.f_stat, dtype=np.float64),
        regime_config,
    )
    expected_loglikelihood = switching_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        gate_probs=expected_gate_probs,
        fom_algorithm="first_order",
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
        switching_config=switching_config,
    ).total

    compiled = jax.jit(
        lambda theta: switching_loglikelihood_from_model_jax(
            model,
            levels,
            observables=("y",),
            gate_probs=gate_probabilities_jax(
                *compute_linear_gate_stats_from_filter_model_jax(
                    model,
                    levels,
                    {"y": 0.1},
                    {model.timings.exo[0]: 0.5},
                    ("y",),
                    observables=("y",),
                    parameter_values=theta,
                    steady_state=steady_state,
                    qme_algorithm="schur",
                    filter="inversion",
                )[2:4],
                regime_config,
            ),
            fom_algorithm="first_order",
            parameter_values=theta,
            steady_state=steady_state,
            measurement_error_scale=0.0,
            qme_algorithm="schur",
            switching_config=switching_config,
        )
    )
    loglikelihood = compiled(jnp.asarray(model.parameter_values, dtype=jnp.float64))

    np.testing.assert_allclose(
        loglikelihood,
        expected_loglikelihood,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("model_path", "steady_state_initial_guess"),
    _FILTER_SMOKE_MODELS,
)
def test_filter_helpers_compile_and_run_on_multiple_upstream_models(
    model_path: Path,
    steady_state_initial_guess: dict[str, float] | None,
) -> None:
    model = parse_macro_model(model_path.read_text())
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess=steady_state_initial_guess,
    )
    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    observable = model.steady_state_names[0]
    observable_index = model.timings.var.index(observable)
    levels = np.full((1, 4), steady_state[observable_index], dtype=np.float64)

    for filter_name in ("kalman", "inversion"):
        shocks = estimate_observed_shocks_matrix(
            model,
            levels,
            observables=(observable,),
            first_order_result=first_order_result,
            filter=filter_name,
        )
        variables, variable_names = estimate_observed_variables_matrix(
            model,
            levels,
            observables=(observable,),
            first_order_result=first_order_result,
            filter=filter_name,
        )
        initial_state = linear_filter_full_state_initial(
            model,
            levels,
            observables=(observable,),
            first_order_result=first_order_result,
            filter=filter_name,
        )

        assert shocks.shape == (model.timings.nExo, levels.shape[1])
        assert variables.shape == (model.timings.nVars, levels.shape[1])
        assert variable_names == tuple(model.timings.var)
        assert np.isfinite(np.asarray(shocks, dtype=np.float64)).all()
        assert np.isfinite(np.asarray(variables, dtype=np.float64)).all()
        assert np.isfinite(np.asarray(initial_state, dtype=np.float64)).all()
        np.testing.assert_allclose(shocks, 0.0, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(variables, dtype=np.float64)[observable_index],
            levels[0],
            rtol=0.0,
            atol=1e-7,
        )


def test_filter_helpers_reject_non_first_order_algorithm() -> None:
    model, first_order_result, _, _, levels = _linear_filter_fixture()

    with pytest.raises(ValueError, match="first-order filter helper path"):
        estimate_observed_shocks_matrix(
            model,
            levels,
            observables=("y",),
            first_order_result=first_order_result,
            algorithm="second_order",
        )
