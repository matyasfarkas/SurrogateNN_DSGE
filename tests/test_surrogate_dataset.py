from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    ParameterDesign,
    build_surrogate_residual_dataset,
    summarize_surrogate_dataset,
)


def _theta_design() -> ParameterDesign:
    theta = np.asarray(
        [
            [0.1, 0.2],
            [1.0, 1.5],
        ],
        dtype=np.float64,
    )
    return ParameterDesign(
        theta=theta,
        names=("rho", "scale"),
        lower=np.asarray([0.0, 0.0], dtype=np.float64),
        upper=np.asarray([1.0, 2.0], dtype=np.float64),
        method="test",
    )


def _rom_predict(state, shock, theta):
    state = np.asarray(state, dtype=np.float64)
    shock = np.asarray(shock, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    obs = np.asarray([state[0] + shock[0] + theta[0]], dtype=np.float64)
    next_state = np.asarray(
        [
            0.8 * state[0] + 0.2 * state[1] + shock[0],
            0.3 * state[0] + 0.5 * state[1] + theta[1] * shock[0],
        ],
        dtype=np.float64,
    )
    return obs, next_state


def _fom_predict(state, shock, theta):
    obs_rom, state_rom = _rom_predict(state, shock, theta)
    state = np.asarray(state, dtype=np.float64)
    shock = np.asarray(shock, dtype=np.float64)
    obs_delta = np.asarray([0.1 * state[0] ** 2 + 0.05 * theta[1]], dtype=np.float64)
    state_delta = np.asarray([0.2 * shock[0] ** 2, -0.1 * state[1] * shock[0]], dtype=np.float64)
    return obs_rom + obs_delta, state_rom + state_delta


def test_residual_full_dataset_uses_current_fom_state_and_next_residual_target() -> None:
    shocks = np.asarray([[0.3, -0.2, 0.1]], dtype=np.float64)
    dataset = build_surrogate_residual_dataset(
        _rom_predict,
        _fom_predict,
        initial_state=np.asarray([1.0, -0.5], dtype=np.float64),
        shocks=shocks,
        theta_design=_theta_design(),
        target_mode="residual_full",
    )

    assert dataset.X.shape == (5, 6)
    assert dataset.Y.shape == (3, 6)
    assert dataset.target_mode == "residual_full"
    np.testing.assert_array_equal(dataset.theta_ids, np.asarray([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(dataset.period_ids, np.asarray([0, 1, 2, 0, 1, 2]))

    theta0 = _theta_design().theta[:, 0]
    state0 = np.asarray([1.0, -0.5], dtype=np.float64)
    shock0 = np.asarray([0.3], dtype=np.float64)
    rom_obs, rom_state = _rom_predict(state0, shock0, theta0)
    fom_obs, fom_state = _fom_predict(state0, shock0, theta0)
    np.testing.assert_allclose(dataset.X[:, 0], np.concatenate([state0, shock0, theta0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        dataset.Y[:, 0],
        np.concatenate([fom_obs - rom_obs, fom_state - rom_state]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(dataset.Y_rom[:, 0], np.zeros((3,), dtype=np.float64), rtol=0, atol=0)

    # The next input state is the FOM state, not the ROM state.
    np.testing.assert_allclose(dataset.X[:2, 1], fom_state, rtol=0, atol=1e-12)


def test_fom_obs_target_stores_rom_observation_as_baseline() -> None:
    shocks = np.asarray([[0.3, -0.2]], dtype=np.float64)
    dataset = build_surrogate_residual_dataset(
        _rom_predict,
        _fom_predict,
        initial_state=[1.0, -0.5],
        shocks=shocks,
        theta_design=_theta_design(),
        target_mode="fom_obs",
    )

    theta0 = _theta_design().theta[:, 0]
    state0 = np.asarray([1.0, -0.5], dtype=np.float64)
    shock0 = np.asarray([0.3], dtype=np.float64)
    rom_obs, _ = _rom_predict(state0, shock0, theta0)
    fom_obs, _ = _fom_predict(state0, shock0, theta0)
    assert dataset.Y.shape == (1, 4)
    np.testing.assert_allclose(dataset.Y[:, 0], fom_obs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dataset.Y_rom[:, 0], rom_obs, rtol=0, atol=1e-12)


def test_samples_per_theta_uses_julia_like_sampling_with_replacement() -> None:
    shocks = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    dataset = build_surrogate_residual_dataset(
        _rom_predict,
        _fom_predict,
        initial_state=[0.0, 0.0],
        shocks=shocks,
        theta_design=_theta_design(),
        target_mode="residual_obs",
        samples_per_theta=2,
        seed=123,
    )

    rng = np.random.default_rng(123)
    expected_periods = np.concatenate(
        [
            rng.integers(0, 4, size=2, endpoint=False, dtype=np.int64),
            rng.integers(0, 4, size=2, endpoint=False, dtype=np.int64),
        ]
    )
    assert dataset.X.shape[1] == 4
    np.testing.assert_array_equal(dataset.period_ids, expected_periods)
    np.testing.assert_array_equal(dataset.theta_ids, np.asarray([0, 0, 1, 1]))


def test_stable_prefix_is_kept_but_theta_success_marks_incomplete_path() -> None:
    def failing_fom(state, shock, theta):
        if float(np.asarray(shock).reshape(-1)[0]) > 0.25:
            raise RuntimeError("synthetic SEP failure")
        return _fom_predict(state, shock, theta)

    shocks = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    dataset = build_surrogate_residual_dataset(
        _rom_predict,
        failing_fom,
        initial_state=[0.0, 0.0],
        shocks=shocks,
        theta_design=_theta_design(),
        target_mode="residual_obs",
        min_stable_periods=2,
    )

    np.testing.assert_array_equal(dataset.theta_stable_periods, np.asarray([2, 2]))
    np.testing.assert_array_equal(dataset.theta_success, np.asarray([False, False]))
    np.testing.assert_array_equal(dataset.period_ids, np.asarray([0, 1, 0, 1]))


def test_surrogate_dataset_summary_and_empty_failure() -> None:
    shocks = np.asarray([[0.1, 0.2]], dtype=np.float64)
    dataset = build_surrogate_residual_dataset(
        _rom_predict,
        _fom_predict,
        initial_state=[0.0, 0.0],
        shocks=shocks,
        theta_design=_theta_design(),
        target_mode="residual_obs",
    )
    summary = summarize_surrogate_dataset(dataset)
    assert summary["n_samples"] == 4
    assert summary["input_dim"] == 5
    assert summary["output_dim"] == 1
    assert summary["theta_success_rate"] == 1.0

    with pytest.raises(ValueError, match="No stable surrogate-dataset samples"):
        build_surrogate_residual_dataset(
            _rom_predict,
            lambda state, shock, theta: (_ for _ in ()).throw(RuntimeError("fail")),
            initial_state=[0.0, 0.0],
            shocks=shocks,
            theta_design=_theta_design(),
            min_stable_periods=1,
        )
