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
        ("calfa", "crhob", "cry"),
        np.asarray([0.24, 0.95, -0.1]),
        width_scale=0.2,
        width_floor=1.0e-4,
    )

    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert 0.0 < lower[0] < 0.24 < upper[0] < 1.0
    assert 0.0 < lower[1] < 0.95 < upper[1] < 1.0
    assert lower[2] < -0.1 < upper[2]


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
