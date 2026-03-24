from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    SEPConfig,
    SwitchingLikelihoodConfig,
    RegimeSwitchConfig,
    apply_gate_padding,
    apply_gate_padding_jax,
    assign_regimes,
    assign_regimes_jax,
    compute_switching_loglikelihood,
    compute_gate_stat_series,
    compute_gate_stat_series_jax,
    evaluate_gate_decisions,
    evaluate_gate_budget_frontier,
    evaluate_gate_probabilities,
    gate_probabilities,
    gate_probabilities_jax,
    get_sep_inversion_last_diagnostics,
    inversion_loglikelihood_per_period_from_model,
    kalman_loglikelihood_per_period_from_model,
    mix_loglikelihood,
    parse_macro_model,
    reset_sep_inversion_last_diagnostics,
    solve_first_order_model,
    solve_stochastic_extended_path_model,
    switching_pipeline_report_from_model,
    optimal_nonlinear_mask_for_budget,
    oracle_nonlinear_mask,
    switching_loglikelihood_from_model,
)


SWITCHING_SOURCE = """
@model switching_linear begin
    y[0] = rho * y[-1] + eps[x]
end

@parameters switching_linear begin
    0 < rho < 1
    rho = 0.65
end
"""

NONLINEAR_SWITCHING_SOURCE = """
@model switching_sep_nonlinear begin
    y[0] = rho * y[-1] + gamma * y[1]^2 + u[x]
end

@parameters switching_sep_nonlinear begin
    gamma = 0.1
    rho = 0.35
end
"""


def _switching_fixture():
    model = parse_macro_model(SWITCHING_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"y": 0.0},
    )
    levels = np.asarray([[0.1, -0.05, 0.12, 0.03, -0.02]], dtype=np.float64)
    return model, first_order_result, levels


def _nonlinear_switching_fixture():
    model = parse_macro_model(NONLINEAR_SWITCHING_SOURCE)
    config = SEPConfig(
        periods=4,
        branching_order=1,
        nnodes=3,
        sparse_tree=True,
        tol=1e-10,
    )
    solution = solve_stochastic_extended_path_model(
        model,
        config=config,
        deterministic_shocks={"u": [0.2, -0.05, 0.0, 0.0]},
    )
    levels = np.asarray(solution.solution.mean_path[:, 1:], dtype=np.float64)
    return model, config, levels


def test_compute_switching_loglikelihood_matches_manual_formulas() -> None:
    ll_rom = jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64)
    ll_fom = jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64)
    gate_probs = jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64)

    linear = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="linear"),
    )
    logsumexp = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )
    hard = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        hard_mask=[False, True, True],
    )

    np.testing.assert_allclose(
        linear.per_period,
        gate_probs * ll_fom + (1.0 - gate_probs) * ll_rom,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        logsumexp.total,
        mix_loglikelihood(ll_fom, ll_rom, gate_probs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hard.per_period,
        np.asarray([-2.0, -2.5, -0.25], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_switching_loglikelihood_is_jittable() -> None:
    compiled = jax.jit(
        lambda rom, fom, probs: compute_switching_loglikelihood(
            rom,
            fom,
            gate_probs=probs,
            config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
        ).total
    )

    total = compiled(
        jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64),
        jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64),
        jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64),
    )

    np.testing.assert_allclose(
        total,
        mix_loglikelihood(
            jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64),
            jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64),
            jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64),
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_gate_stat_series_jax_matches_numpy_and_jits() -> None:
    observations = jnp.asarray(
        [[1.0, 0.8, 1.2], [0.4, 0.1, -0.2]],
        dtype=jnp.float64,
    )
    linear_observations = jnp.asarray(
        [[0.9, 0.7, 1.1], [0.2, 0.0, -0.1]],
        dtype=jnp.float64,
    )
    shocks = jnp.asarray(
        [[0.05, -0.03, 0.01], [0.0, 0.0, 0.0], [0.02, -0.01, 0.04]],
        dtype=jnp.float64,
    )
    observation_sigma = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    shock_sigma = jnp.asarray([0.1, 0.0, 0.2], dtype=jnp.float64)

    compiled = jax.jit(
        lambda obs, lin, eps: compute_gate_stat_series_jax(
            obs,
            lin,
            eps,
            observation_sigma,
            shock_sigma,
            shock_norm="linf",
            error_norm="l2",
        )
    )
    e_stat, f_stat = compiled(observations, linear_observations, shocks)
    expected_e, expected_f = compute_gate_stat_series(
        np.asarray(observations),
        np.asarray(linear_observations),
        np.asarray(shocks),
        np.asarray(observation_sigma),
        np.asarray(shock_sigma),
        shock_norm="linf",
        error_norm="l2",
    )

    np.testing.assert_allclose(e_stat, expected_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f_stat, expected_f, rtol=1e-12, atol=1e-12)


def test_gate_probability_helpers_match_numpy_and_jit() -> None:
    e_stat = jnp.asarray([0.1, 0.5, 1.0, 1.5], dtype=jnp.float64)
    f_stat = jnp.asarray([0.2, 0.4, 0.8, 1.2], dtype=jnp.float64)
    hard_cfg = RegimeSwitchConfig(
        gate_mode="hard",
        tau_eps=0.75,
        tau_y=0.75,
        prob_floor=0.05,
        prob_ceiling=0.95,
        k_pre=1,
        k_post=0,
        min_len=2,
    )
    soft_cfg = RegimeSwitchConfig(
        gate_mode="soft",
        tau_eps=0.75,
        tau_y=0.75,
        beta_eps=2.0,
        beta_y=1.5,
        bias=-0.1,
        prob_floor=0.05,
        prob_ceiling=0.95,
    )

    compiled = jax.jit(
        lambda eps, err: (
            apply_gate_padding_jax(eps > 0.75, 1, 0, 2),
            assign_regimes_jax(eps, err, hard_cfg),
            gate_probabilities_jax(eps, err, hard_cfg),
            gate_probabilities_jax(eps, err, soft_cfg),
        )
    )
    padded_jax, assigned_jax, hard_jax, soft_jax = compiled(e_stat, f_stat)

    padded_np = apply_gate_padding(np.asarray(e_stat > 0.75), 1, 0, 2)
    assigned_np = assign_regimes(np.asarray(e_stat), np.asarray(f_stat), hard_cfg)
    hard_np = gate_probabilities(np.asarray(e_stat), np.asarray(f_stat), hard_cfg)
    soft_np = gate_probabilities(np.asarray(e_stat), np.asarray(f_stat), soft_cfg)

    np.testing.assert_array_equal(padded_jax, padded_np)
    np.testing.assert_array_equal(assigned_jax, assigned_np)
    np.testing.assert_allclose(hard_jax, hard_np, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(soft_jax, soft_np, rtol=1e-12, atol=1e-12)


def test_model_switching_bridge_matches_manual_component_mix() -> None:
    model, first_order_result, levels = _switching_fixture()
    gate_probs = np.asarray([0.1, 0.35, 0.5, 0.7, 0.9], dtype=np.float64)

    rom = kalman_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
    )
    fom = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="first_order",
        first_order_result=first_order_result,
    )
    manual = compute_switching_loglikelihood(
        rom,
        fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )
    bridged = switching_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        gate_probs=gate_probs,
        fom_algorithm="first_order",
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
        switching_config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )

    np.testing.assert_allclose(bridged.total, manual.total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        bridged.per_period,
        manual.per_period,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bridged.gate_probs,
        gate_probs,
        rtol=1e-12,
        atol=1e-12,
    )


def test_model_switching_bridge_supports_sparse_tree_sep_fom() -> None:
    model, config, levels = _nonlinear_switching_fixture()
    gate_probs = np.asarray([0.15, 0.35, 0.65, 0.85], dtype=np.float64)
    switching_config = SwitchingLikelihoodConfig(soft_mixture="logsumexp")

    rom = kalman_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
    )
    reset_sep_inversion_last_diagnostics()
    fom = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="stochastic_extended_path",
        config=config,
        sep_sparse_tree=True,
        steady_state_initial_guess={"y": 0.0},
        on_failure_loglikelihood=-1e12,
    )
    manual = compute_switching_loglikelihood(
        rom,
        fom,
        gate_probs=gate_probs,
        config=switching_config,
    )

    reset_sep_inversion_last_diagnostics()
    bridged = switching_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        gate_probs=gate_probs,
        fom_algorithm="stochastic_extended_path",
        config=config,
        sep_sparse_tree=True,
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        on_failure_loglikelihood=-1e12,
        switching_config=switching_config,
    )
    diagnostics = get_sep_inversion_last_diagnostics()

    np.testing.assert_allclose(bridged.total, manual.total, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        bridged.per_period,
        manual.per_period,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        bridged.gate_probs,
        gate_probs,
        rtol=1e-12,
        atol=1e-12,
    )
    assert diagnostics is not None
    assert diagnostics["status"] == "ok"
    assert diagnostics["sep_sparse_tree"] is True
    assert diagnostics["sep_carry_warm_start_strategy"] == "shifted_tree"


def test_gate_decision_metrics_quantify_regret_and_budget_gap() -> None:
    ll_rom = np.asarray([-2.0, -1.2, -0.7, -1.5], dtype=np.float64)
    ll_fom = np.asarray([-1.0, -1.4, -0.3, -1.8], dtype=np.float64)
    hard_mask = np.asarray([False, True, True, False], dtype=bool)

    oracle = oracle_nonlinear_mask(ll_rom, ll_fom)
    budget_oracle = optimal_nonlinear_mask_for_budget(ll_rom, ll_fom, int(np.sum(hard_mask)))
    metrics = evaluate_gate_decisions(ll_rom, ll_fom, hard_mask)

    np.testing.assert_array_equal(
        oracle,
        np.asarray([True, False, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        budget_oracle,
        np.asarray([True, False, True, False], dtype=bool),
    )
    assert metrics["oracle_tp"] == 1
    assert metrics["oracle_fp"] == 1
    assert metrics["oracle_fn"] == 1
    np.testing.assert_allclose(metrics["captured_positive_gain"], 0.4, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["total_positive_gain"], 1.4, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["captured_gain_share"], 0.4 / 1.4, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["wasted_nonlinear_cost"], 0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["oracle_total"], -4.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["mixed_total"], -5.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["regret_vs_oracle"], 1.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["budget_oracle_total"], -4.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["regret_vs_budget_oracle"], 1.2, rtol=0.0, atol=1e-12)


def test_gate_probability_metrics_match_oracle_classification() -> None:
    ll_rom = np.asarray([-2.0, -1.2, -0.7, -1.5], dtype=np.float64)
    ll_fom = np.asarray([-1.0, -1.4, -0.3, -1.8], dtype=np.float64)
    gate_probs = np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float64)

    metrics = evaluate_gate_probabilities(
        ll_rom,
        ll_fom,
        gate_probs,
        hard_threshold=0.5,
    )

    np.testing.assert_allclose(metrics["brier_score"], 0.025, rtol=0.0, atol=1e-12)
    assert metrics["auc"] == 1.0
    np.testing.assert_allclose(
        metrics["mean_prob_oracle_positive"],
        0.85,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        metrics["mean_prob_oracle_negative"],
        0.15,
        rtol=0.0,
        atol=1e-12,
    )
    assert metrics["hard_oracle_f1"] == 1.0
    assert metrics["hard_regret_vs_oracle"] == 0.0


def test_gate_budget_frontier_quantifies_budget_ranking_quality() -> None:
    ll_rom = np.asarray([-2.0, -1.5, -1.1, -0.8], dtype=np.float64)
    ll_fom = np.asarray([-1.0, -1.6, -0.4, -0.7], dtype=np.float64)
    gate_scores = np.asarray([0.9, 0.1, 0.8, 0.3], dtype=np.float64)

    frontier = evaluate_gate_budget_frontier(
        ll_rom,
        ll_fom,
        gate_scores,
        budgets=[0, 1, 2, 4],
    )

    np.testing.assert_array_equal(frontier["budgets"], np.asarray([0, 1, 2, 4]))
    np.testing.assert_allclose(
        frontier["regret_vs_budget_oracle"],
        np.zeros((4,), dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        frontier["captured_gain_share"],
        np.asarray([0.0, 1.0 / 1.8, 1.7 / 1.8, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        frontier["selected_share"],
        np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert frontier["area_regret_vs_budget_oracle"] == 0.0
    assert frontier["max_regret_vs_budget_oracle"] == 0.0


def test_switching_pipeline_report_collects_sparse_tree_sep_comparison() -> None:
    model, config, levels = _nonlinear_switching_fixture()
    gate_probs = np.asarray([0.15, 0.35, 0.65, 0.85], dtype=np.float64)

    report = switching_pipeline_report_from_model(
        model,
        levels,
        observables=("y",),
        gate_probs=gate_probs,
        fom_algorithm="stochastic_extended_path",
        config=config,
        sep_sparse_tree=True,
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        on_failure_loglikelihood=-1e12,
        switching_config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
        budget_frontier_budgets=[0, 1, 2, 4],
        benchmark_reps=1,
    )

    assert report["ll_rom"].shape == (levels.shape[1],)
    assert report["ll_fom"].shape == (levels.shape[1],)
    assert report["ll_switching"].shape == (levels.shape[1],)
    np.testing.assert_allclose(
        np.sum(report["ll_switching"]),
        report["switching_total"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert report["comparison"]["n"] == levels.shape[1]
    assert report["gate_stats"]["periods_total"] == levels.shape[1]
    assert report["gate_stats"]["periods_nonlinear"] == 2
    assert report["decision_quality"]["periods_total"] == levels.shape[1]
    assert report["probability_quality"]["hard_threshold"] == 0.5
    np.testing.assert_array_equal(
        report["budget_frontier"]["budgets"],
        np.asarray([0, 1, 2, 4], dtype=np.int64),
    )
    assert report["budget_frontier"]["mean_regret_vs_budget_oracle"] >= 0.0
    np.testing.assert_allclose(
        report["decomposition"]["ll_mixed_total"],
        np.sum(np.where(report["hard_mask"], report["ll_fom"], report["ll_rom"])),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        report["decision_quality"]["mixed_total"],
        report["decomposition"]["ll_mixed_total"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        report["decision_quality"]["regret_vs_oracle"],
        report["decision_quality"]["oracle_total"] - report["decomposition"]["ll_mixed_total"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert report["runtime"]["runtime_fom_s"] is not None
    assert report["runtime"]["runtime_switching_s"] is not None
    assert report["fom_sep_diagnostics"] is not None
    assert report["switching_sep_diagnostics"] is not None
    assert report["fom_sep_diagnostics"]["sep_sparse_tree"] is True
    assert report["switching_sep_diagnostics"]["sep_sparse_tree"] is True
    assert report["fom_sep_diagnostics"]["sep_carry_warm_start_strategy"] == "shifted_tree"
    assert (
        report["switching_sep_diagnostics"]["sep_carry_warm_start_strategy"]
        == "shifted_tree"
    )
