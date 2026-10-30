from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "validate_sw07_paper_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_sw07_paper_results_for_tests", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_parity_accepts_tight_value_and_gradient_match() -> None:
    module = _load_module()
    benchmark = {
        "preset": "sw07_paper",
        "parameter_names": [f"p{idx}" for idx in range(module.EXPECTED_PARAMETER_COUNT)],
        "steady_state_mode": "solve",
        "dtype": "float64",
        "periods": 180,
        "presample_periods": 6,
        "initial_covariance_strategy": "theoretical",
        "measurement_error_scale": 0.0,
        "jitter": 0.0,
        "data_provenance": {
            "sha256": module.EXPECTED_DATA_SHA256,
            "model_source_sha256": "model-sha256",
            "sample_start_index_zero_based": 50,
        },
    }
    gradient = [float(idx) for idx in range(module.EXPECTED_PARAMETER_COUNT)]
    cpu = {
        "benchmark": {**benchmark, "qme_algorithm": "schur"},
        "runtime": {"jax_default_backend": "cpu"},
        "preflight": {
            "loglikelihood": module.EXPECTED_DYNARE_LOGLIKELIHOOD,
            "gradient": gradient,
        },
    }
    gpu = {
        "benchmark": {**benchmark, "qme_algorithm": "schur_gpu"},
        "runtime": {"jax_default_backend": "gpu"},
        "preflight": {
            "loglikelihood": module.EXPECTED_DYNARE_LOGLIKELIHOOD + 1.0e-9,
            "gradient": [value + 1.0e-7 for value in gradient],
        }
    }

    parity = module._preflight_parity(cpu, gpu)

    assert parity["passed"] is True


def test_basin_search_reports_separated_posterior_means() -> None:
    module = _load_module()
    names = [f"p{idx}" for idx in range(module.EXPECTED_PARAMETER_COUNT)]
    run_a = {
        "posterior_diagnostics": {
            "parameters": {name: {"mean": 0.0, "std": 0.1} for name in names}
        }
    }
    run_b = {
        "posterior_diagnostics": {
            "parameters": {name: {"mean": 0.3, "std": 0.1} for name in names}
        }
    }

    basin = module._basin_separation(run_a, run_b)

    assert basin["assessed"] is True
    assert basin["distinct_basin_detected"] is True


def _raw_draw_result(module):
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    rng = np.random.default_rng(20261029)
    raw = {
        name: rng.normal(size=(4, 32)) + idx / 100.0
        for idx, name in enumerate(module.EXPECTED_PARAMETER_NAMES)
    }
    parameters = {}
    ess_values = []
    r_hat_values = []
    for name, values in raw.items():
        flat = values.reshape(-1)
        ess = float(np.asarray(effective_sample_size(values)))
        r_hat = float(np.asarray(split_gelman_rubin(values)))
        ess_values.append(ess)
        r_hat_values.append(r_hat)
        parameters[name] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.quantile(flat, 0.50)),
            "q95": float(np.quantile(flat, 0.95)),
            "n_eff": ess,
            "r_hat": r_hat,
            "shape": [4, 32],
        }
    return {
        "benchmark": {"chains": 4, "samples": 32},
        "posterior_draws": {name: values.tolist() for name, values in raw.items()},
        "posterior_diagnostics": {
            "parameters": parameters,
            "min_ess": min(ess_values),
            "max_r_hat": max(r_hat_values),
        },
    }


def test_raw_draw_audit_recomputes_summaries_and_diagnostics() -> None:
    module = _load_module()
    result = _raw_draw_result(module)

    audit = module._raw_draw_audit(result)

    assert audit["passed"] is True
    assert audit["invalid_parameters"] == []
    assert audit["summary_max_absolute_error"] == 0.0


def test_raw_draw_audit_fails_closed_on_tampered_summary() -> None:
    module = _load_module()
    result = _raw_draw_result(module)
    first = module.EXPECTED_PARAMETER_NAMES[0]
    result["posterior_diagnostics"]["parameters"][first]["mean"] += 0.01

    audit = module._raw_draw_audit(result)

    assert audit["passed"] is False
    assert audit["summary_max_absolute_error"] >= 0.01 - 1.0e-12
