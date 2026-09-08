from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "posterior_sampling_speed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "posterior_sampling_speed_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_summary_filters_deterministic_sites_from_ess() -> None:
    module = _load_module()
    samples = {
        "rho": np.asarray(
            [
                [0.80, 0.81, 0.79, 0.82, 0.80, 0.81],
                [0.78, 0.79, 0.80, 0.81, 0.82, 0.83],
            ],
            dtype=np.float64,
        ),
        "loglikelihood": np.asarray(
            [
                [-10.0, -10.1, -10.2, -10.1, -10.0, -10.2],
                [-10.3, -10.1, -10.2, -10.0, -10.1, -10.3],
            ],
            dtype=np.float64,
        ),
    }

    summary = module._sample_summary(samples, parameter_names=("rho",))

    assert tuple(summary["parameters"]) == ("rho",)
    assert summary["parameters"]["rho"]["shape"] == [2, 6]
    assert summary["min_ess"] is not None
    assert summary["min_ess"] > 0.0


def test_sw07_safe_presets_reference_known_parameter_names() -> None:
    module = _load_module()
    model = SimpleNamespace(parameter_names=module.SW07_SAFE_27_PARAMETERS)

    selected_15 = module._select_parameter_names(model, "sw07_safe_15")
    selected_27 = module._select_parameter_names(model, "sw07_safe_27")

    assert selected_15 == module.SW07_SAFE_15_PARAMETERS
    assert selected_27 == module.SW07_SAFE_27_PARAMETERS
    assert set(selected_15).issubset(selected_27)


def test_prior_interval_keeps_unit_root_like_parameters_inside_unit_bounds() -> None:
    module = _load_module()

    lower, upper = module._prior_interval("crhoa", 0.9977, 0.01, 1.0e-4)

    assert 0.0 < lower < 0.9977 < upper < 1.0
