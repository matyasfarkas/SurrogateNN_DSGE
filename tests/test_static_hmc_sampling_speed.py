from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
            "--posterior-draws-output",
            "draws.npz",
            "--no-jit",
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
    assert args.posterior_draws_output == Path("draws.npz")
    assert args.no_jit is True


def test_parse_args_accepts_schur_support_audit_options() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--qme-algorithm",
            "doubling",
            "--support-audit-draws",
            "7",
            "--support-audit-schur-acceptance-tol",
            "1e-7",
        ]
    )

    assert args.qme_algorithm == "doubling"
    assert args.support_audit_draws == 7
    assert args.support_audit_schur_acceptance_tol == 1.0e-7


def test_parse_step_size_grid_accepts_commas_and_spaces() -> None:
    module = _load_module()

    assert module._parse_step_size_grid("0.05, 0.2 1.0", 0.1) == [
        0.05,
        0.2,
        1.0,
    ]
    assert module._parse_step_size_grid(None, 0.25) == [0.25]


def test_parse_step_size_grid_rejects_nonpositive_values() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="positive"):
        module._parse_step_size_grid("0.1,0", 0.1)


def test_schur_support_audit_skips_when_disabled() -> None:
    module = _load_module()

    audit = module._schur_support_audit(
        args=SimpleNamespace(support_audit_draws=0),
        context={},
        constrained_samples=np.zeros((1, 1, 1), dtype=np.float64),
    )

    assert audit is None


def test_schur_support_audit_delegates_with_chain_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    seen = {}

    def fake_support_audit(**kwargs):
        seen.update(kwargs)
        return {
            "audited_draws": 2,
            "schur_classification_counts": {"unique": 1, "indeterminate": 1},
            "doubling_accepts_non_unique_count": 1,
            "doubling_accepts_non_unique_share": 0.5,
            "both_accept_count": 1,
            "schur_reject_count": 1,
            "doubling_reject_count": 0,
            "examples": [],
        }

    monkeypatch.setattr(module.posterior_speed, "_support_audit", fake_support_audit)
    constrained_samples = np.asarray(
        [
            [[0.1, 0.2]],
            [[0.3, 0.4]],
        ],
        dtype=np.float64,
    )
    context = {
        "model": SimpleNamespace(parameter_names=("alpha", "beta")),
        "observations": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "observables": ("y",),
        "steady_state": np.asarray([0.0], dtype=np.float64),
        "parameter_values": np.asarray([0.9, 0.8], dtype=np.float64),
        "parameter_names": ("alpha", "beta"),
        "measurement_error_scale": 1.0e-9,
        "jitter": 1.0e-8,
    }

    audit = module._schur_support_audit(
        args=SimpleNamespace(
            support_audit_draws=2,
            support_audit_schur_acceptance_tol=1.0e-7,
            failure_value=-1.0e12,
            qme_algorithm="doubling",
        ),
        context=context,
        constrained_samples=constrained_samples,
    )

    np.testing.assert_allclose(
        seen["samples_by_chain"]["alpha"],
        np.asarray([[0.1, 0.3]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        seen["samples_by_chain"]["beta"],
        np.asarray([[0.2, 0.4]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert seen["max_draws"] == 2
    assert seen["schur_acceptance_tol"] == 1.0e-7
    assert seen["failure_value"] == -1.0e12
    assert audit["audited_from"] == "static_hmc_post_warmup_draws"
    assert audit["inner_qme_algorithm"] == "doubling"
    assert audit["fast_algorithm_under_test"] == "doubling"
    assert audit["doubling_accepts_non_unique_count"] == 1
