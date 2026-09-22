from __future__ import annotations

import jax
import numpy as np
import pytest

from surrogatenn_dsge import (
    FrozenResNet,
    SurrogateDataset,
    build_surrogate_residual_arrays_jax,
    fit_surrogate_pipeline,
    load_surrogate_bundle,
    predict_frozen_batch,
    resolve_jax_device,
    save_surrogate_bundle,
    split_surrogate_dataset,
    surrogate_sample_weights_from_residuals,
    train_surrogate_from_batched_arrays_jax,
    train_surrogate_from_dataset,
)


def _array_platform(value) -> str:
    devices = value.devices() if hasattr(value, "devices") else {value.device()}
    return str(next(iter(devices)).platform)


def _supervised_dataset() -> SurrogateDataset:
    rng = np.random.default_rng(123)
    theta = np.asarray(
        [
            [-0.8, -0.2, 0.4, 0.9],
            [0.5, 0.8, 1.1, 1.4],
        ],
        dtype=np.float64,
    )
    X_cols: list[np.ndarray] = []
    Y_cols: list[np.ndarray] = []
    Y_rom_cols: list[np.ndarray] = []
    theta_ids: list[int] = []
    period_ids: list[int] = []
    for theta_id in range(theta.shape[1]):
        theta_t = theta[:, theta_id]
        for period in range(12):
            state_shock = rng.normal(size=3)
            rom = np.asarray(
                [
                    0.30 * state_shock[0] + 0.10 * state_shock[1],
                    -0.20 * state_shock[2],
                ],
                dtype=np.float64,
            )
            residual = np.asarray(
                [
                    0.45 * theta_t[0] + 0.20 * state_shock[0],
                    -0.35 * theta_t[1] + 0.15 * state_shock[1],
                ],
                dtype=np.float64,
            )
            X_cols.append(np.concatenate([state_shock, theta_t]))
            Y_rom_cols.append(rom)
            Y_cols.append(rom + residual)
            theta_ids.append(theta_id)
            period_ids.append(period)
    return SurrogateDataset(
        X=np.column_stack(X_cols),
        Y=np.column_stack(Y_cols),
        Y_rom=np.column_stack(Y_rom_cols),
        theta=theta,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=np.asarray([True, True, True, True]),
        theta_stable_periods=np.full((theta.shape[1],), 12, dtype=np.int64),
        target_mode="fom_obs",
        theta_names=("rho", "scale"),
    )


def test_split_surrogate_dataset_can_hold_out_whole_theta_groups() -> None:
    dataset = _supervised_dataset()
    split = split_surrogate_dataset(dataset, validation_fraction=0.25, split_by_theta=True, seed=4)

    assert split.train_size == 36
    assert split.val_size == 12
    assert split.train_theta_ids is not None
    assert split.val_theta_ids is not None
    assert not np.intersect1d(split.train_theta_ids, split.val_theta_ids).size
    assert set(dataset.theta_ids[split.val_idx]) == set(split.val_theta_ids)


def test_surrogate_sample_weights_from_residuals_are_inverse_and_mean_one() -> None:
    weights = surrogate_sample_weights_from_residuals([0.1, 1.0, np.inf, -1.0])

    assert np.isfinite(weights).all()
    np.testing.assert_allclose(np.mean(weights), 1.0, rtol=1e-12, atol=1e-12)
    assert weights[0] > weights[1]
    np.testing.assert_allclose(weights[1], weights[2], rtol=1e-7, atol=1e-12)
    assert weights[2] == weights[3]


def test_train_surrogate_from_dataset_mlp_rom_residual_is_device_placed() -> None:
    dataset = _supervised_dataset()
    cpu = jax.devices("cpu")[0]
    result = train_surrogate_from_dataset(
        dataset,
        architecture="mlp",
        rom_residual=True,
        validation_fraction=0.25,
        seed=7,
        d_hidden=24,
        d_hidden2=None,
        nepoch=160,
        eta_init=4e-3,
        batch_size=16,
        device=cpu,
    )

    assert result.validation_rmse is not None
    assert result.validation_rmse_rom is not None
    assert result.validation_improvement is not None
    assert float(np.nanmean(result.validation_improvement)) > 0.25
    assert result.metadata["jax_device_platform"] == "cpu"
    assert _array_platform(result.frozen.W1) == "cpu"


def test_train_surrogate_from_dataset_resnet_dispatch_accepts_device_selector() -> None:
    dataset = _supervised_dataset()
    result = train_surrogate_from_dataset(
        dataset,
        architecture="resnet",
        rom_residual=True,
        validation_fraction=0.0,
        seed=3,
        d_hidden=8,
        n_blocks=1,
        nepoch=4,
        batch_size=16,
        device="cpu",
    )

    assert isinstance(result.frozen, FrozenResNet)
    assert result.val_size == 0
    assert result.metadata["jax_device_platform"] == "cpu"
    assert _array_platform(result.frozen.W_embed) == "cpu"


def test_train_surrogate_from_batched_arrays_uses_masked_jax_rollouts() -> None:
    theta = np.asarray([[0.1, 0.2], [1.0, 1.2]], dtype=np.float64)
    states = np.asarray(
        [
            [[0.0, 0.1], [0.2, 0.0], [0.3, -0.1], [0.4, 0.2]],
            [[-0.1, 0.0], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        ],
        dtype=np.float64,
    )
    shocks = np.asarray([[[0.05], [0.10], [-0.02], [0.03]], [[0.02], [-0.01], [0.04], [0.06]]], dtype=np.float64)
    rom_obs = 0.4 * states[:, :, :1] + shocks
    rom_state_next = states + np.concatenate([shocks, -shocks], axis=2)
    fom_obs = rom_obs + 0.1 * states[:, :, :1] ** 2 + theta.T[:, None, :1]
    fom_state_next = rom_state_next + 0.05 * np.concatenate([shocks**2, shocks**2], axis=2)
    fom_obs[1, 2, 0] = np.nan

    arrays = build_surrogate_residual_arrays_jax(
        states,
        shocks,
        theta,
        rom_obs,
        rom_state_next,
        fom_obs,
        fom_state_next,
        target_mode="fom_obs",
        min_stable_periods=2,
    )
    result = train_surrogate_from_batched_arrays_jax(
        arrays,
        architecture="mlp",
        rom_residual=True,
        d_hidden=8,
        d_hidden2=None,
        nepoch=4,
        batch_size=1,
        seed=14,
        device="cpu",
    )

    assert result.train_size == 6
    assert result.val_size == 0
    assert result.metadata["masked_sample_count"] == 2
    assert result.metadata["jax_device_platform"] == "cpu"
    assert _array_platform(result.frozen.W1) == "cpu"
    prediction = np.asarray(predict_frozen_batch(result.frozen, np.asarray(arrays.X)[:, :2]), dtype=np.float64)
    assert np.isfinite(prediction).all()


def test_save_and_load_surrogate_training_result_bundle_round_trips_predictions(tmp_path) -> None:
    dataset = _supervised_dataset()
    result = train_surrogate_from_dataset(
        dataset,
        architecture="mlp",
        rom_residual=True,
        validation_fraction=0.25,
        seed=9,
        d_hidden=16,
        d_hidden2=None,
        nepoch=120,
        eta_init=4e-3,
        batch_size=16,
        device="cpu",
    )
    path = save_surrogate_bundle(tmp_path / "surrogate_bundle.snn", result, metadata={"experiment": "unit-test"})

    loaded = load_surrogate_bundle(path, device="cpu")
    X_probe = dataset.X[:, :5]
    np.testing.assert_allclose(
        predict_frozen_batch(loaded.frozen, X_probe),
        predict_frozen_batch(result.frozen, X_probe),
        rtol=1e-12,
        atol=1e-12,
    )
    assert loaded.metadata["experiment"] == "unit-test"
    assert loaded.metadata["rom_residual"] is True
    np.testing.assert_array_equal(loaded.train_idx, result.split.train_idx)
    np.testing.assert_array_equal(loaded.val_idx, result.split.val_idx)
    np.testing.assert_allclose(loaded.validation_rmse, result.validation_rmse, rtol=0, atol=0)
    assert _array_platform(loaded.frozen.W1) == "cpu"


def test_save_and_load_raw_resnet_bundle_round_trips_predictions(tmp_path) -> None:
    dataset = _supervised_dataset()
    result = train_surrogate_from_dataset(
        dataset,
        architecture="resnet",
        rom_residual=True,
        validation_fraction=0.0,
        seed=5,
        d_hidden=8,
        n_blocks=1,
        nepoch=4,
        batch_size=16,
        device="cpu",
    )
    path = save_surrogate_bundle(
        tmp_path / "raw_resnet_bundle.snn",
        result.frozen,
        metadata={"architecture": "resnet", "raw": True},
    )

    loaded = load_surrogate_bundle(path, device="cpu")
    assert isinstance(loaded.frozen, FrozenResNet)
    X_probe = dataset.X[:, :4]
    np.testing.assert_allclose(
        predict_frozen_batch(loaded.frozen, X_probe),
        predict_frozen_batch(result.frozen, X_probe),
        rtol=1e-12,
        atol=1e-12,
    )
    assert loaded.metadata["raw"] is True
    assert loaded.validation_rmse is None


def test_fit_surrogate_pipeline_builds_trains_and_saves_bundle(tmp_path) -> None:
    theta = np.asarray(
        [
            [0.1, 0.3, 0.5],
            [0.8, 1.0, 1.2],
        ],
        dtype=np.float64,
    )
    shocks = np.asarray(
        [
            [0.10, -0.20, 0.15, -0.05, 0.08],
            [0.05, 0.12, -0.08, 0.02, -0.04],
        ],
        dtype=np.float64,
    )

    def rom_predict(state, shock, theta_t):
        state = np.asarray(state, dtype=np.float64)
        shock = np.asarray(shock, dtype=np.float64)
        theta_t = np.asarray(theta_t, dtype=np.float64)
        obs = np.asarray([0.4 * state[0] + shock[0] + 0.1 * theta_t[0]], dtype=np.float64)
        next_state = np.asarray(
            [
                0.65 * state[0] + 0.10 * state[1] + shock[0],
                0.20 * state[0] + 0.55 * state[1] + theta_t[1] * shock[1],
            ],
            dtype=np.float64,
        )
        return obs, next_state

    def fom_predict(state, shock, theta_t):
        obs_rom, next_rom = rom_predict(state, shock, theta_t)
        state = np.asarray(state, dtype=np.float64)
        shock = np.asarray(shock, dtype=np.float64)
        obs_resid = np.asarray([0.2 * state[0] * theta_t[0] + 0.05 * shock[0] ** 2], dtype=np.float64)
        state_resid = np.asarray([0.02 * shock[0] ** 2, -0.03 * state[1] * shock[1]], dtype=np.float64)
        return obs_rom + obs_resid, next_rom + state_resid

    result = fit_surrogate_pipeline(
        rom_predict,
        fom_predict,
        initial_state=np.asarray([0.2, -0.1], dtype=np.float64),
        shocks=shocks,
        theta_design=theta,
        target_mode="fom_obs",
        architecture="mlp",
        rom_residual=True,
        validation_fraction=0.25,
        train_seed=11,
        d_hidden=10,
        d_hidden2=None,
        nepoch=12,
        eta_init=2e-3,
        batch_size=8,
        device="cpu",
        bundle_path=tmp_path / "pipeline_surrogate.snn",
        bundle_metadata={"experiment": "pipeline-unit"},
    )

    assert result.dataset.n_samples == theta.shape[1] * shocks.shape[1]
    assert result.dataset_summary["n_samples"] == result.dataset.n_samples
    assert result.training.metadata["architecture"] == "mlp"
    assert result.training.metadata["jax_device_platform"] == "cpu"
    assert result.bundle_path is not None
    assert result.bundle_path.is_file()

    loaded = load_surrogate_bundle(result.bundle_path, device="cpu")
    assert loaded.metadata["pipeline"] == "fit_surrogate_pipeline"
    assert loaded.metadata["experiment"] == "pipeline-unit"
    assert loaded.metadata["dataset_summary"]["n_samples"] == result.dataset.n_samples
    X_probe = result.dataset.X[:, :4]
    np.testing.assert_allclose(
        predict_frozen_batch(loaded.frozen, X_probe),
        predict_frozen_batch(result.training.frozen, X_probe),
        rtol=1e-12,
        atol=1e-12,
    )


def test_resolve_jax_device_requires_requested_gpu_backend() -> None:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if gpu_devices:
        assert resolve_jax_device("gpu").platform == "gpu"
    else:
        with pytest.raises(ValueError, match="gpu"):
            resolve_jax_device("gpu")
