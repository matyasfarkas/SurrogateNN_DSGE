from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "profile_batched_likelihood.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "profile_batched_likelihood_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_batch_sizes_rejects_non_positive_values() -> None:
    module = _load_module()

    assert module._parse_batch_sizes("1,4,16") == (1, 4, 16)

    with pytest.raises(Exception):
        module._parse_batch_sizes("1,0")


def test_make_centered_draws_are_reproducible_and_inside_support() -> None:
    module = _load_module()
    center = np.asarray([0.5, 2.0], dtype=np.float64)
    lower = np.asarray([0.0, 1.0], dtype=np.float64)
    upper = np.asarray([1.0, 3.0], dtype=np.float64)

    first = module._make_centered_draws(
        center=center,
        lower=lower,
        upper=upper,
        batch_size=8,
        draw_scale=0.35,
        dtype=np.float32,
    )
    second = module._make_centered_draws(
        center=center,
        lower=lower,
        upper=upper,
        batch_size=8,
        draw_scale=0.35,
        dtype=np.float32,
    )

    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[0], center, rtol=0.0, atol=1.0e-7)
    assert np.all(first > lower)
    assert np.all(first < upper)
