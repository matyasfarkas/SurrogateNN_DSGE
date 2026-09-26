from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_profile_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "profile_surrogate_pipeline_gpu.py"
    spec = importlib.util.spec_from_file_location("profile_surrogate_pipeline_gpu", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_hlt_dataset_has_expected_shapes_and_residual_signal() -> None:
    mod = _load_profile_module()
    shape = mod.SyntheticHLTShape(state_dim=5, shock_dim=2, theta_dim=3, obs_dim=2)
    dataset = mod.make_synthetic_hlt_surrogate_dataset(
        samples=24,
        theta_draws=4,
        shape=shape,
        seed=123,
    )

    assert dataset.X.shape == (10, 24)
    assert dataset.Y.shape == (7, 24)
    assert dataset.Y_rom.shape == (7, 24)
    assert dataset.theta.shape == (3, 4)
    assert dataset.target_mode == "fom_full"
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.Y).all()
    assert float(np.sqrt(np.mean((dataset.Y - dataset.Y_rom) ** 2))) > 0.0
    np.testing.assert_array_equal(np.unique(dataset.theta_ids), np.arange(4))


def test_memory_estimate_counts_core_arrays() -> None:
    mod = _load_profile_module()
    shape = mod.SyntheticHLTShape(state_dim=5, shock_dim=2, theta_dim=3, obs_dim=2)
    memory = mod.estimate_dataset_memory_bytes(shape=shape, samples=11, dtype=np.float64)

    assert memory["X"] == 10 * 11 * 8
    assert memory["Y"] == 7 * 11 * 8
    assert memory["Y_rom"] == 7 * 11 * 8
    assert memory["theta_ids"] == 11 * 8
    assert memory["total_core_arrays"] == memory["X"] + 2 * memory["Y"] + memory["theta_ids"]


def test_training_profile_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "calibration",
            "--device",
            "cpu",
            "--samples",
            "32",
            "--theta-draws",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--hidden",
            "8",
            "--blocks",
            "0",
            "--state-dim",
            "4",
            "--shock-dim",
            "2",
            "--theta-dim",
            "3",
            "--obs-dim",
            "2",
        ]
    )
    shape = mod.SyntheticHLTShape(state_dim=4, shock_dim=2, theta_dim=3, obs_dim=2)
    result = mod.run_training_profile(args, shape)

    assert result["status"] == "ok"
    assert result["samples"] == 32
    assert result["train_size"] > 0
    assert result["train_s"] >= 0.0


def test_synthetic_batched_hlt_rollout_arrays_include_masked_failures() -> None:
    mod = _load_profile_module()
    shape = mod.SyntheticHLTShape(state_dim=4, shock_dim=2, theta_dim=3, obs_dim=2)
    arrays = mod.make_synthetic_hlt_batched_rollout_arrays(
        samples=20,
        theta_draws=4,
        shape=shape,
        seed=456,
        mask_fraction=0.25,
    )

    assert arrays.X.shape == (9, 20)
    assert arrays.Y.shape == (6, 20)
    assert arrays.Y_rom.shape == (6, 20)
    assert arrays.theta.shape == (3, 4)
    assert int(np.count_nonzero(np.asarray(arrays.sample_mask, dtype=bool))) < arrays.X.shape[1]
    assert int(np.count_nonzero(np.asarray(arrays.theta_success, dtype=bool))) == 3


def test_batched_training_profile_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "batched-training",
            "--device",
            "cpu",
            "--samples",
            "16",
            "--theta-draws",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden",
            "8",
            "--blocks",
            "0",
            "--state-dim",
            "4",
            "--shock-dim",
            "2",
            "--theta-dim",
            "3",
            "--obs-dim",
            "2",
            "--batched-mask-fraction",
            "0.25",
        ]
    )
    shape = mod.SyntheticHLTShape(state_dim=4, shock_dim=2, theta_dim=3, obs_dim=2)
    result = mod.run_batched_training_profile(args, shape)

    assert result["status"] == "ok"
    assert result["kind"] == "synthetic_hlt_fixed_shape_batched_training"
    assert result["actual_samples"] == 16
    assert result["train_size"] > 0
    assert result["sample_mask_false_count"] > 0
    assert result["masked_sample_count"] == result["sample_mask_false_count"]
    assert result["train_s"] >= 0.0


def test_batched_sep_micro_profile_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "batched-sep-micro",
            "--device",
            "cpu",
            "--sep-batch-size",
            "3",
            "--sep-state-dim",
            "2",
            "--sep-shock-dim",
            "1",
            "--sep-periods",
            "3",
            "--sep-order",
            "1",
            "--sep-nnodes",
            "3",
            "--sep-max-iter",
            "8",
            "--sep-reps",
            "1",
        ]
    )
    result = mod.run_batched_sep_micro_profile(args)

    assert result["status"] == "ok"
    assert result["kind"] == "batched_sep_sparse_tree_microbenchmark"
    assert result["batch_size"] == 3
    assert result["accepted_count"] == 3
    assert result["converged_count"] == 3
    assert result["median_s"] >= 0.0


def test_batched_sep_training_profile_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "batched-sep-training",
            "--device",
            "cpu",
            "--sep-batch-size",
            "3",
            "--sep-state-dim",
            "2",
            "--sep-shock-dim",
            "1",
            "--sep-periods",
            "3",
            "--sep-order",
            "1",
            "--sep-nnodes",
            "3",
            "--sep-max-iter",
            "10",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden",
            "8",
            "--blocks",
            "0",
            "--theta-dim",
            "3",
            "--obs-dim",
            "1",
        ]
    )
    shape = mod.SyntheticHLTShape(state_dim=4, shock_dim=2, theta_dim=3, obs_dim=1)
    result = mod.run_batched_sep_training_profile(args, shape)

    assert result["status"] == "ok"
    assert result["kind"] == "synthetic_batched_sep_target_training"
    assert result["batch_size"] == 3
    assert result["actual_samples"] == 9
    assert result["sep_accepted_count"] == 3
    assert result["sep_converged_count"] == 3
    assert result["train_size"] > 0
    assert result["jax_likelihood_status"] == "ok"
    assert result["jax_likelihood_grad_finite"]
    assert result["end_to_end_s"] >= 0.0


def test_batched_sep_training_profile_can_skip_likelihood_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "batched-sep-training",
            "--device",
            "cpu",
            "--sep-batch-size",
            "3",
            "--sep-state-dim",
            "2",
            "--sep-shock-dim",
            "1",
            "--sep-periods",
            "3",
            "--sep-order",
            "1",
            "--sep-nnodes",
            "3",
            "--sep-max-iter",
            "10",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden",
            "8",
            "--blocks",
            "0",
            "--theta-dim",
            "3",
            "--obs-dim",
            "1",
            "--skip-batched-sep-likelihood",
        ]
    )
    shape = mod.SyntheticHLTShape(state_dim=4, shock_dim=2, theta_dim=3, obs_dim=1)
    result = mod.run_batched_sep_training_profile(args, shape)

    assert result["status"] == "ok"
    assert result["sep_accepted_count"] == 3
    assert result["sep_converged_count"] == 3
    assert result["jax_likelihood_status"] == "skipped"
    assert result["jax_likelihood_value"] is None
    assert result["jax_likelihood_grad_norm"] is None
    assert not result["jax_likelihood_grad_finite"]


def test_parsed_batched_sep_training_profile_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "parsed-batched-sep-training",
            "--device",
            "cpu",
            "--sep-batch-size",
            "3",
            "--sep-periods",
            "3",
            "--sep-order",
            "1",
            "--sep-nnodes",
            "3",
            "--sep-max-iter",
            "12",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden",
            "8",
            "--blocks",
            "0",
            "--obs-dim",
            "2",
        ]
    )
    result = mod.run_parsed_batched_sep_training_profile(args)

    assert result["status"] == "ok"
    assert result["kind"] == "parsed_batched_sep_target_training"
    assert result["batch_size"] == 3
    assert result["actual_samples"] == 9
    assert result["sep_accepted_count"] == 3
    assert result["sep_converged_count"] == 3
    assert result["train_size"] > 0
    assert result["end_to_end_s"] >= 0.0


def test_hlt_surrogate_hmc_prior_intervals_keep_bounded_parameters_inside_support() -> None:
    mod = _load_profile_module()

    lower, upper = mod._hlt_uniform_prior_arrays(
        ("calfa", "crhob", "crhoms", "cry"),
        np.asarray([0.24, 0.95, 0.0, -0.1]),
        width_scale=0.2,
        width_floor=1.0e-4,
    )

    assert lower.shape == (4,)
    assert upper.shape == (4,)
    assert 0.0 < lower[0] < 0.24 < upper[0] < 1.0
    assert 0.0 < lower[1] < 0.95 < upper[1] < 1.0
    assert lower[2] < 0.0 < upper[2]
    assert lower[3] < -0.1 < upper[3]


def test_hlt_parameter_set_selector_supports_payload_safe_and_all() -> None:
    mod = _load_profile_module()

    class DummyModel:
        parameter_names = (
            *mod.SW07_SAFE_27_PARAMETERS,
            "cprobp",
            "cindp",
            "curvp",
            "extra_parameter",
        )

    case = {"parameter_subset": ["cprobp", "cindp", "curvp"]}

    assert mod._select_hlt_parameter_subset(DummyModel, case, "payload") == (
        "cprobp",
        "cindp",
        "curvp",
    )
    assert mod._select_hlt_parameter_subset(DummyModel, case, "sw07_safe_15") == mod.SW07_SAFE_15_PARAMETERS
    assert mod._select_hlt_parameter_subset(DummyModel, case, "sw07_safe_27") == mod.SW07_SAFE_27_PARAMETERS
    assert mod._select_hlt_parameter_subset(DummyModel, case, "all") == DummyModel.parameter_names
    assert mod._select_hlt_parameter_subset(DummyModel, case, "calfa, crhob") == ("calfa", "crhob")


def test_static_hmc_on_bounded_surrogate_log_density_tiny_cpu_smoke() -> None:
    mod = _load_profile_module()
    center = mod.jnp.asarray([0.25, 0.75], dtype=mod.jnp.float64)

    def log_density(theta):
        return -0.5 * mod.jnp.sum((theta - center) ** 2)

    result = mod.run_static_hmc_on_bounded_surrogate_log_density(
        log_density_fn=log_density,
        center=center,
        parameter_names=("calfa", "crhob"),
        lower=mod.jnp.asarray([0.1, 0.5], dtype=mod.jnp.float64),
        upper=mod.jnp.asarray([0.4, 0.95], dtype=mod.jnp.float64),
        chains=2,
        warmup=1,
        samples=2,
        leapfrog_steps=2,
        step_size=0.01,
        target_accept_prob=0.8,
        adapt_step_size=True,
        initial_jitter=0.01,
        seed=123,
    )

    assert result["status"] == "ok"
    assert result["kind"] == "fixed_rom_surrogate_static_hmc"
    assert result["samples_shape"] == [2, 2, 2]
    assert result["samples_finite"]
    assert result["post_warmup_draws"] == 4
    assert result["accepted_share"] is not None
    assert set(result["parameter_summary"]) == {"calfa", "crhob"}


def test_scale_aware_parity_metrics_allow_large_loglikelihood_tiny_relative_error() -> None:
    mod = _load_profile_module()

    metrics = mod._parity_metrics(
        value=-14844.412480000916,
        reference=-14844.412476276484,
        atol=1.0e-7,
        rtol=1.0e-9,
    )

    assert metrics["abs_diff"] > metrics["atol"]
    assert metrics["effective_tol"] > metrics["abs_diff"]
    assert metrics["rel_diff"] < metrics["rtol"]
    assert metrics["ok"]


def test_static_hmc_retries_low_acceptance_with_smaller_step_size() -> None:
    mod = _load_profile_module()
    center = mod.jnp.asarray([0.25], dtype=mod.jnp.float64)

    def log_density(theta):
        return -0.5 * mod.jnp.sum(((theta - center) / 0.001) ** 2)

    result = mod.run_static_hmc_on_bounded_surrogate_log_density(
        log_density_fn=log_density,
        center=center,
        parameter_names=("x",),
        lower=mod.jnp.asarray([0.1], dtype=mod.jnp.float64),
        upper=mod.jnp.asarray([0.4], dtype=mod.jnp.float64),
        chains=2,
        warmup=0,
        samples=2,
        leapfrog_steps=4,
        step_size=1.0,
        target_accept_prob=0.8,
        adapt_step_size=False,
        initial_jitter=0.0,
        seed=1,
        min_accepted_share=0.01,
        max_retries=3,
        retry_step_size_factor=1.0e-3,
    )

    assert result["status"] == "ok"
    assert result["retry_count"] >= 1
    assert result["retry_history"][0]["accepted_share"] == 0.0
    assert result["initial_step_size"] < result["requested_initial_step_size"]
    assert result["accepted_share"] >= 0.01


def test_hlt_adaptive_sep_attempt_specs_auto_ladder() -> None:
    mod = _load_profile_module()
    args = mod.parse_args(
        [
            "--mode",
            "hlt-fixed-ss-smoke",
            "--sep-order",
            "1",
            "--sep-periods",
            "2",
            "--sep-max-iter",
            "7",
            "--hlt-sep-shock-scale-ladder",
            "1.0,0.5",
        ]
    )

    specs = mod._hlt_sep_attempt_specs(args)

    assert [spec.index for spec in specs] == list(range(len(specs)))
    assert [
        (spec.config.branching_order, spec.config.periods, spec.shock_scale, spec.config.max_iter)
        for spec in specs
    ] == [
        (1, 2, 1.0, 7),
        (1, 2, 0.5, 7),
        (1, 1, 1.0, 7),
        (1, 1, 0.5, 7),
        (0, 2, 1.0, 7),
        (0, 2, 0.5, 7),
        (0, 1, 1.0, 7),
        (0, 1, 0.5, 7),
    ]


def test_adaptive_hlt_sep_dataset_keeps_multi_theta_fallback_targets() -> None:
    mod = _load_profile_module()
    theta = np.asarray([[1.0, 2.0]], dtype=np.float64)
    initial_states = np.asarray([[0.0, 0.5]], dtype=np.float64)
    shocks = np.asarray(
        [
            [[0.1, -0.2]],
            [[0.05, 0.0]],
        ],
        dtype=np.float64,
    )
    specs = (
        mod.HLTSEPAttemptSpec(
            index=0,
            shock_scale=1.0,
            config=mod.SEPConfig(periods=1, branching_order=1, nnodes=3, max_iter=2),
        ),
        mod.HLTSEPAttemptSpec(
            index=1,
            shock_scale=1.0,
            config=mod.SEPConfig(periods=1, branching_order=0, nnodes=1, max_iter=2),
        ),
    )

    def rom_predict(state, shock, theta_t):
        next_state = np.asarray(state, dtype=np.float64) + 0.5 * np.asarray(shock, dtype=np.float64)
        return next_state.copy(), next_state

    def sep_predict(state, shock, theta_t, config):
        if int(config.branching_order) == 1:
            raise RuntimeError("strict sparse-tree SEP failed")
        next_state = np.asarray(state, dtype=np.float64) + np.asarray(shock, dtype=np.float64) + 0.01 * theta_t[0]
        return next_state.copy(), next_state, {"residual_norm": 1.0e-8, "accepted": True}

    dataset, diagnostics = mod._build_adaptive_hlt_sep_dataset(
        rom_predict=rom_predict,
        sep_predict=sep_predict,
        initial_states=initial_states,
        shocks=shocks,
        theta=theta,
        attempt_specs=specs,
        target_mode="fom_full",
        min_stable_periods=2,
        input_names=("x", "eps", "theta"),
        output_names=("obs", "x[1]"),
        max_logged_failures=10,
    )

    assert dataset.n_samples == 4
    np.testing.assert_array_equal(dataset.theta_ids, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(dataset.theta_success, np.asarray([True, True]))
    np.testing.assert_array_equal(dataset.theta_stable_periods, np.asarray([2, 2]))
    assert diagnostics["status"] == "ok"
    assert diagnostics["fallback_samples"] == 4
    assert diagnostics["fallback_share"] == 1.0
    assert diagnostics["accepted_by_branching_order"] == {"0": 4}
    assert len(diagnostics["failure_log"]) == 4


def test_batched_hlt_sep_dataset_matches_single_config_adaptive_builder() -> None:
    mod = _load_profile_module()
    model = mod.parse_macro_model(
        """
        @model nonlinear_sep begin
            y[0] = rho * y[-1] + gamma * y[1]^2 + u[x]
        end

        @parameters nonlinear_sep begin
            gamma = 0.15
            rho = 0.25
        end
        """
    )
    parameter_names = tuple(model.parameter_names)
    theta = np.asarray(
        [
            [0.12, 0.18],
            [0.22, 0.30],
        ],
        dtype=np.float64,
    )
    initial_states = np.zeros((1, theta.shape[1]), dtype=np.float64)
    steady_states = np.zeros((theta.shape[1], 1), dtype=np.float64)
    shocks = np.asarray([[[0.04, -0.01]], [[-0.03, 0.02]]], dtype=np.float64)
    config = mod.SEPConfig(periods=2, branching_order=1, nnodes=3, tol=1e-10, accept_tol=1e-8, max_iter=30)
    runtimes = []
    for theta_idx in range(theta.shape[1]):
        params = theta[:, theta_idx]
        first_order = model.solve_first_order(parameter_values=params, steady_state=[0.0])
        assert first_order.solution.converged
        runtimes.append(
            {
                "parameter_values": params,
                "steady_state": np.zeros((1,), dtype=np.float64),
                "state_transition": np.asarray(first_order.solution.state_transition, dtype=np.float64),
                "shock_impact": np.asarray(first_order.solution.shock_impact, dtype=np.float64),
            }
        )

    def runtime_for_theta(theta_t):
        theta_arr = np.asarray(theta_t, dtype=np.float64).reshape(-1)
        idx = int(np.argmin(np.linalg.norm(theta.T - theta_arr[None, :], axis=1)))
        return runtimes[idx]

    def rom_predict(state, shock, theta_t):
        runtime = runtime_for_theta(theta_t)
        state_arr = np.asarray(state, dtype=np.float64)
        shock_arr = np.asarray(shock, dtype=np.float64)
        next_state = (
            runtime["steady_state"]
            + runtime["state_transition"] @ (state_arr - runtime["steady_state"])
            + runtime["shock_impact"] @ shock_arr
        )
        return next_state.copy(), next_state

    def sep_predict(state, shock, theta_t, sep_config):
        runtime = runtime_for_theta(theta_t)
        deterministic = np.zeros((sep_config.periods, 1), dtype=np.float64)
        deterministic[0, :] = np.asarray(shock, dtype=np.float64)
        sep_result = model.solve_stochastic_extended_path(
            parameter_values=runtime["parameter_values"],
            steady_state=runtime["steady_state"],
            initial_state=np.asarray(state, dtype=np.float64),
            terminal_state=runtime["steady_state"],
            deterministic_shocks=deterministic,
            config=sep_config,
        )
        assert sep_result.solution.accepted
        next_state = np.asarray(sep_result.solution.mean_path, dtype=np.float64)[:, 1]
        return next_state.copy(), next_state, {"residual_norm": sep_result.solution.residual_norm}

    adaptive, adaptive_diag = mod._build_adaptive_hlt_sep_dataset(
        rom_predict=rom_predict,
        sep_predict=sep_predict,
        initial_states=initial_states,
        shocks=shocks,
        theta=theta,
        attempt_specs=(mod.HLTSEPAttemptSpec(index=0, shock_scale=1.0, config=config),),
        target_mode="fom_full",
        min_stable_periods=2,
        input_names=("y", "u", *parameter_names),
        output_names=("y_obs", "y[1]"),
        max_logged_failures=10,
    )
    batched, batched_diag = mod._build_batched_hlt_sep_dataset(
        model=model,
        parameter_values_by_theta=theta.T,
        theta_features_by_theta=theta.T,
        steady_states_by_theta=steady_states,
        state_transition_by_theta=np.stack([runtime["state_transition"] for runtime in runtimes], axis=0),
        shock_impact_by_theta=np.stack([runtime["shock_impact"] for runtime in runtimes], axis=0),
        initial_states=initial_states,
        shocks=shocks,
        config=config,
        target_mode="fom_full",
        min_stable_periods=2,
        observable_idx=(0,),
        state_idx=(0,),
        input_names=("y", "u", *parameter_names),
        output_names=("y_obs", "y[1]"),
        target_device=None,
        max_logged_failures=10,
    )

    def sorted_columns(dataset):
        order = np.lexsort((dataset.period_ids, dataset.theta_ids))
        return dataset.X[:, order], dataset.Y[:, order], dataset.Y_rom[:, order]

    assert adaptive_diag["status"] == "ok"
    assert batched_diag["status"] == "ok"
    assert batched_diag["builder"] == "batched_sep"
    assert batched_diag["batched_calls"] == shocks.shape[2]
    np.testing.assert_array_equal(adaptive.theta_ids, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(batched.theta_ids, np.asarray([0, 0, 1, 1]))
    for left, right in zip(sorted_columns(adaptive), sorted_columns(batched)):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-10)
