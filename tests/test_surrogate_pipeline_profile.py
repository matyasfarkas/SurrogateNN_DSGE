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
