from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gpuhub_hlt_resnn_runner.py"


def _load_module():
    scripts_dir = str(_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("gpuhub_hlt_resnn_runner_for_tests", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_sequence_runs_smoke_before_full() -> None:
    module = _load_module()

    assert module.stage_sequence("setup") == ()
    assert module.stage_sequence("smoke_then_full") == ("smoke", "full")
    assert module.stage_sequence("smoke_pilot_full") == ("smoke", "pilot", "full")


def test_build_hlt_stage_environment_sets_gpu_and_runtime_modes(tmp_path: Path) -> None:
    module = _load_module()

    env = module.build_hlt_stage_environment(
        stage="full",
        python="/opt/venv/bin/python",
        result_root=tmp_path,
        allow_cpu=False,
        hlt_target_builder="batched-sep",
        hlt_parameter_set="all",
        steady_state_mode="solve-or-reference",
        likelihood_runtime_mode="full-jax",
        likelihood_qme_algorithm="schur_gpu",
        likelihood_static_rows_mode="reference",
        jax_log_density_gradient=False,
        differentiate_shocks=True,
        extra_env={"HMC_CHAINS": "64"},
    )

    assert env["MODE"] == "full"
    assert env["PYTHON"] == "/opt/venv/bin/python"
    assert env["DEVICE"] == "gpu"
    assert env["REQUIRE_GPU"] == "1"
    assert env["HLT_TARGET_BUILDER"] == "batched-sep"
    assert env["HLT_PARAMETER_SET"] == "all"
    assert env["HLT_STEADY_STATE_MODE"] == "solve-or-reference"
    assert env["LIKELIHOOD_RUNTIME_MODE"] == "full-jax"
    assert env["LIKELIHOOD_QME_ALGORITHM"] == "schur_gpu"
    assert env["JAX_LOG_DENSITY_GRADIENT"] == "0"
    assert env["DIFFERENTIATE_SHOCKS"] == "1"
    assert env["HMC_CHAINS"] == "64"


def test_summarize_hlt_result_extracts_release_metrics(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "hlt_full_surrogate_estimation.json"
    output.write_text(
        json.dumps(
            {
                "results": {
                    "hlt_fixed_ss_smoke": {
                        "status": "ok",
                        "backend": "gpu",
                        "parameter_subset": ["a", "b"],
                        "hlt_parameter_set": "all",
                        "theta_draws": 8,
                        "train_size": 64,
                        "val_size": 8,
                        "pipeline_s": 12.5,
                        "target_diagnostics": {
                            "builder": "adaptive_sep",
                            "status": "ok",
                            "accepted_samples": 64,
                            "theta_full_success_count": 8,
                            "fallback_share": 0.25,
                        },
                        "surrogate_inversion_likelihood": {
                            "status": "ok",
                            "total_loglikelihood": -123.0,
                        },
                        "jax_surrogate_log_density": {
                            "status": "ok",
                            "parity_ok": True,
                            "runtime_mode": "fixed-reference",
                        },
                        "surrogate_hmc": {
                            "status": "ok",
                            "post_warmup_draws": 1024,
                            "elapsed_s": 30.0,
                            "draws_per_second": 34.13,
                            "accepted_share": 0.92,
                            "retry_count": 0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = module.summarize_hlt_result(output)

    assert summary["status"] == "ok"
    assert summary["backend"] == "gpu"
    assert summary["parameter_count"] == 2
    assert summary["accepted_samples"] == 64
    assert summary["jax_log_density_parity_ok"] is True
    assert summary["hmc_draws_per_second"] == 34.13
