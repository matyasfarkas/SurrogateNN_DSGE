from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    compute_gate_stat_series,
    compute_linear_gate_stats_from_shocks_model,
    compute_linear_gate_stats_from_shocks_model_jax,
    parse_macro_model,
    rollout_first_order_solution,
    solve_first_order_model,
)


_UPSTREAM_TEST_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "SurrogateNN_Estimation.jl" / "test" / "models"
)
_RBC_STEADY_STATE_GUESS = {
    "A": 1.0,
    "Pi": 1.0025,
    "R": 1.0035,
    "c": 1.2,
    "k": 9.4,
    "y": 1.42,
    "z_delta": 1.0,
}


def test_model_linear_gate_stats_from_shocks_matches_manual_first_order_rollout() -> None:
    model = parse_macro_model((_UPSTREAM_TEST_MODEL_DIR / "RBC_CME.jl").read_text())
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess=_RBC_STEADY_STATE_GUESS,
    )

    observable_names = ("y", "c")
    observable_indices = model.resolve_observable_indices(observable_names)
    shock_names = model.timings.exo
    shock_mapping = {
        shock_names[0]: np.asarray([0.02, -0.01, 0.0, 0.015], dtype=np.float64),
        shock_names[1]: np.asarray([0.05, 0.0, -0.02, 0.01], dtype=np.float64),
    }
    shock_matrix = np.vstack([shock_mapping[name] for name in shock_names])

    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    initial_state = steady_state.copy()
    initial_state[model.timings.past_not_future_and_mixed_idx[0]] += 0.03
    state_indices = np.asarray(model.timings.past_not_future_and_mixed_idx, dtype=np.int64)
    linear_deviations = np.asarray(
        rollout_first_order_solution(
            first_order_result.solution.solution_matrix,
            model.timings,
            shock_matrix,
            initial_reduced_state=initial_state[state_indices] - steady_state[state_indices],
        ),
        dtype=np.float64,
    )
    levels = linear_deviations[list(observable_indices), :] + steady_state[
        list(observable_indices)
    ][:, None]

    result = compute_linear_gate_stats_from_shocks_model(
        model,
        levels,
        shock_mapping,
        {"y": 0.1, "c": 0.2},
        {shock_names[0]: 0.25, shock_names[1]: 0.5},
        observables=observable_names,
        first_order_result=first_order_result,
        initial_state=initial_state,
    )

    expected_e, expected_f = compute_gate_stat_series(
        levels,
        levels,
        shock_matrix,
        np.asarray([0.1, 0.2], dtype=np.float64),
        np.asarray([0.25, 0.5], dtype=np.float64),
    )

    np.testing.assert_allclose(
        result.linear_observations,
        levels,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.shocks, shock_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.e_stat, expected_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.f_stat, expected_f, rtol=1e-12, atol=1e-12)


def test_jax_model_linear_gate_stats_from_shocks_matches_concrete_path() -> None:
    model = parse_macro_model((_UPSTREAM_TEST_MODEL_DIR / "RBC_CME.jl").read_text())
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess=_RBC_STEADY_STATE_GUESS,
    )

    observable_names = ("y", "c")
    shock_names = model.timings.exo
    shock_mapping = {
        shock_names[0]: np.asarray([0.015, -0.01, 0.0, 0.005], dtype=np.float64),
        shock_names[1]: np.asarray([0.02, 0.01, -0.015, 0.0], dtype=np.float64),
    }
    obs_sigma = {"y": 0.1, "c": 0.2}
    shock_sigmas = {shock_names[0]: 0.25, shock_names[1]: 0.5}

    steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
    initial_state = steady_state.copy()
    initial_state[model.timings.past_not_future_and_mixed_idx[0]] += 0.02

    shock_matrix = np.vstack([shock_mapping[name] for name in shock_names])
    state_indices = np.asarray(model.timings.past_not_future_and_mixed_idx, dtype=np.int64)
    linear_deviations = np.asarray(
        rollout_first_order_solution(
            first_order_result.solution.solution_matrix,
            model.timings,
            shock_matrix,
            initial_reduced_state=initial_state[state_indices] - steady_state[state_indices],
        ),
        dtype=np.float64,
    )
    observable_indices = model.resolve_observable_indices(observable_names)
    levels = linear_deviations[list(observable_indices), :] + steady_state[
        list(observable_indices)
    ][:, None]

    expected = compute_linear_gate_stats_from_shocks_model(
        model,
        levels,
        shock_mapping,
        obs_sigma,
        shock_sigmas,
        observables=observable_names,
        first_order_result=first_order_result,
        initial_state=initial_state,
    )
    compiled = jax.jit(
        lambda theta: compute_linear_gate_stats_from_shocks_model_jax(
            model,
            levels,
            shock_mapping,
            obs_sigma,
            shock_sigmas,
            observables=observable_names,
            parameter_values=theta,
            steady_state=steady_state,
            initial_state=initial_state,
            qme_algorithm="schur",
        )
    )
    result = compiled(jnp.asarray(model.parameter_values, dtype=jnp.float64))

    np.testing.assert_allclose(
        result.linear_observations,
        expected.linear_observations,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.shocks, expected.shocks, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.e_stat, expected.e_stat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.f_stat, expected.f_stat, rtol=1e-12, atol=1e-12)
