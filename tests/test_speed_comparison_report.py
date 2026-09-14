from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "speed_comparison_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "speed_comparison_report_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_hmc_step_grid_rows_distinguish_cold_and_post_compile(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "static.json"
    payload = {
        "benchmark": {
            "chains": 8,
            "warmup": 2,
            "samples": 3,
            "periods": 40,
            "parameter_count": 15,
        },
        "runtime": {"platform": "Linux", "jax_default_backend": "gpu"},
        "step_size_runs": [
            {
                "initial_step_size": 0.2,
                "first_run": {
                    "timing_s": 10.0,
                    "draws_per_second": 2.4,
                    "post_warmup_draws": 24,
                    "min_ess": 4.0,
                    "mean_ess": 5.0,
                    "seconds_per_min_ess": 2.5,
                    "posterior_diagnostics": {"max_r_hat": 1.2},
                    "acceptance": {"accept_prob": {"mean": 0.99}},
                },
            },
            {
                "initial_step_size": 1.0,
                "first_run": {
                    "timing_s": 4.0,
                    "draws_per_second": 6.0,
                    "post_warmup_draws": 24,
                    "min_ess": 8.0,
                    "mean_ess": 9.0,
                    "seconds_per_min_ess": 0.5,
                    "posterior_diagnostics": {"max_r_hat": 1.01},
                    "acceptance": {"accept_prob": {"mean": 0.88}},
                },
            },
        ],
    }
    path.write_text(json.dumps(payload))

    rows = module._posterior_rows([path])

    assert [row["timing_scope"] for row in rows] == [
        "cold compile+run",
        "post-compile run",
    ]
    assert rows[1]["step_size"] == 1.0
    assert rows[1]["seconds_per_min_ess"] == 0.5


def test_posterior_sampler_row_extracts_numpyro_metrics(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "numpyro.json"
    payload = {
        "benchmark": {
            "kernel": "nuts",
            "chains": 4,
            "warmup": 5,
            "samples": 6,
            "periods": 40,
            "parameter_count": 15,
        },
        "runtime": {"platform": "macOS", "jax_default_backend": "cpu"},
        "throughput": {
            "sampling_wall_s": 12.0,
            "post_warmup_draws": 24,
            "draws_per_second": 2.0,
            "min_ess": 10.0,
            "mean_ess": 12.0,
            "seconds_per_min_ess": 1.2,
        },
        "posterior_diagnostics": {"max_r_hat": 1.03},
        "extra_fields": {"accept_prob": {"mean": 0.9}},
    }
    path.write_text(json.dumps(payload))

    rows = module._posterior_rows([path])

    assert rows == [
        {
            "source": str(path),
            "environment": "macOS / JAX cpu",
            "sampler": "NumPyro NUTS",
            "chains": 4,
            "warmup": 5,
            "samples": 6,
            "post_warmup_draws": 24,
            "periods": 40,
            "parameters": 15,
            "wall_s": 12.0,
            "draws_per_second": 2.0,
            "min_ess": 10.0,
            "mean_ess": 12.0,
            "seconds_per_min_ess": 1.2,
            "max_r_hat": 1.03,
            "accept_prob_mean": 0.9,
            "step_size": None,
            "timing_scope": "full MCMC run",
        }
    ]


def test_render_markdown_includes_ratio() -> None:
    module = _load_module()
    report = {
        "validation_dir": "/tmp/validation",
        "stage_rows": [],
        "batched_likelihood_rows": [],
        "posterior_rows": [
            {
                "environment": "macOS / JAX cpu",
                "sampler": "NumPyro HMC",
                "chains": 8,
                "seconds_per_min_ess": 1.0,
                "draws_per_second": 2.0,
            },
            {
                "environment": "Linux / JAX gpu",
                "sampler": "JAX static HMC",
                "chains": 32,
                "seconds_per_min_ess": 0.25,
                "draws_per_second": 8.0,
            },
        ],
        "best_gpu_static_hmc": {
            "seconds_per_min_ess": 0.25,
            "draws_per_second": 8.0,
        },
        "best_m4_numpyro": {
            "seconds_per_min_ess": 1.0,
            "draws_per_second": 2.0,
        },
    }

    markdown = module.render_markdown(report)

    assert "4.00x faster" in markdown
    assert "Raw posterior draws/sec ratio is 4.00x" in markdown
