from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "static_hmc_sampling_speed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "static_hmc_sampling_speed_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_samples_by_chain_transposes_sample_chain_layout() -> None:
    module = _load_module()
    samples = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float64,
    )

    by_chain = module._samples_by_chain(samples, ("alpha", "beta"))

    np.testing.assert_allclose(
        by_chain["alpha"],
        np.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        by_chain["beta"],
        np.asarray([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_samples_by_chain_rejects_bad_parameter_count() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="parameter_names"):
        module._samples_by_chain(np.zeros((2, 3, 2)), ("alpha",))


def test_parse_args_accepts_static_hmc_gpu_shape() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--parameters",
            "sw07_safe_15",
            "--chains",
            "64",
            "--warmup",
            "32",
            "--samples",
            "128",
            "--leapfrog-steps",
            "10",
            "--step-size",
            "0.05",
            "--steady-reps",
            "1",
            "--platform",
            "gpu",
            "--force-gpu",
        ]
    )

    assert args.parameters == "sw07_safe_15"
    assert args.chains == 64
    assert args.warmup == 32
    assert args.samples == 128
    assert args.leapfrog_steps == 10
    assert args.step_size == 0.05
    assert args.steady_reps == 1
    assert args.platform == "gpu"
    assert args.force_gpu is True
