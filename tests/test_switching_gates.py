from __future__ import annotations

import numpy as np

from surrogatenn_dsge import (
    GateCalibrationConfig,
    RegimeSwitchConfig,
    apply_gate_padding,
    assign_regimes,
    calibrate_gate,
    calibrate_gate_bias,
    compute_gate_stat_series,
    gate_probabilities,
    gate_share,
    logistic,
    logit,
)


def test_compute_gate_stat_series_matches_manual_l2_norms() -> None:
    obs_data = np.asarray([[1.0, 1.2, 0.9], [0.5, 0.4, 0.3]], dtype=np.float64)
    lin_obs = np.asarray([[0.9, 1.1, 1.0], [0.45, 0.45, 0.35]], dtype=np.float64)
    shocks = np.asarray([[0.1, -0.2, 0.05], [0.0, 0.1, -0.1]], dtype=np.float64)
    obs_sigma = np.asarray([0.1, 0.2], dtype=np.float64)
    shock_sigma = np.asarray([0.5, 2.0], dtype=np.float64)

    e_stat, f_stat = compute_gate_stat_series(
        obs_data,
        lin_obs,
        shocks,
        obs_sigma,
        shock_sigma,
    )

    expected_e = np.asarray(
        [
            np.linalg.norm(np.asarray([0.1 / 0.5, 0.0 / 2.0]), ord=2),
            np.linalg.norm(np.asarray([-0.2 / 0.5, 0.1 / 2.0]), ord=2),
            np.linalg.norm(np.asarray([0.05 / 0.5, -0.1 / 2.0]), ord=2),
        ],
        dtype=np.float64,
    )
    expected_f = np.asarray(
        [
            np.linalg.norm(np.asarray([(1.0 - 0.9) / 0.1, (0.5 - 0.45) / 0.2]), ord=2),
            np.linalg.norm(np.asarray([(1.2 - 1.1) / 0.1, (0.4 - 0.45) / 0.2]), ord=2),
            np.linalg.norm(np.asarray([(0.9 - 1.0) / 0.1, (0.3 - 0.35) / 0.2]), ord=2),
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(e_stat, expected_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f_stat, expected_f, rtol=1e-12, atol=1e-12)


def test_gate_calibration_and_padding_match_target_share() -> None:
    e_stat = np.asarray([0.2, 0.8, 1.4, 0.5, 1.2, 0.3], dtype=np.float64)
    f_stat = np.asarray([0.1, 0.9, 1.1, 0.4, 1.3, 0.2], dtype=np.float64)

    calibration = calibrate_gate(
        e_stat,
        f_stat,
        config=GateCalibrationConfig(target_share=1.0 / 3.0, tol=5e-2),
    )
    share = gate_share(e_stat, f_stat, calibration.tau_eps, calibration.tau_y)
    padded = apply_gate_padding([False, True, False, False, True, False], 1, 0, 2)
    assigned = assign_regimes(e_stat, f_stat, calibration.tau_eps, calibration.tau_y)

    assert abs(share - 1.0 / 3.0) < 5e-2
    assert calibration.achieved_share == share
    np.testing.assert_array_equal(
        padded,
        np.asarray([True, True, False, True, True, False], dtype=bool),
    )
    assert assigned.dtype == bool
    assert assigned.shape == e_stat.shape


def test_gate_probabilities_support_hard_and_soft_modes() -> None:
    e_stat = np.asarray([0.1, 0.5, 1.0, 1.5], dtype=np.float64)
    f_stat = np.asarray([0.2, 0.4, 0.8, 1.2], dtype=np.float64)

    hard_cfg = RegimeSwitchConfig(
        gate_mode="hard",
        tau_eps=0.75,
        tau_y=0.75,
        prob_floor=0.05,
        prob_ceiling=0.95,
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

    hard = gate_probabilities(e_stat, f_stat, hard_cfg)
    soft = gate_probabilities(e_stat, f_stat, soft_cfg)
    bias = calibrate_gate_bias(np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), 0.4)

    np.testing.assert_array_equal(
        hard,
        np.asarray([0.05, 0.05, 0.95, 0.95], dtype=np.float64),
    )
    assert np.all(soft >= soft_cfg.prob_floor)
    assert np.all(soft <= soft_cfg.prob_ceiling)
    assert soft[0] < soft[-1]
    np.testing.assert_allclose(logit(0.4), np.log(0.4 / 0.6), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        logistic(logit(0.4)),
        0.4,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isfinite(bias)
