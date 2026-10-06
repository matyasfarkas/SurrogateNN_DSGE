from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
    assert module.stage_sequence("calibration") == ("calibration",)
    assert module.stage_sequence("smoke_then_calibration") == ("smoke", "calibration")
    assert module.stage_sequence("smoke_then_full") == ("smoke", "full")
    assert module.stage_sequence("smoke_pilot_full") == ("smoke", "pilot", "full")
    assert module.stage_sequence("estimation_pilot") == ("estimation_pilot",)
    assert module.stage_sequence("full_hlt") == ("full_hlt",)
    assert module.stage_sequence("smoke_then_estimation_pilot") == ("smoke", "estimation_pilot")
    assert module.stage_sequence("smoke_then_final_nonlinear") == ("smoke", "final_nonlinear")
    assert module.stage_sequence("smoke_then_full_hlt") == ("smoke", "full_hlt")
    assert module.stage_sequence("smoke_estimation_pilot_full") == (
        "smoke",
        "estimation_pilot",
        "full",
    )
    assert module.stage_sequence("smoke_estimation_pilot_final") == (
        "smoke",
        "estimation_pilot",
        "final_nonlinear",
    )
    assert module.stage_sequence("smoke_estimation_pilot_full_hlt") == (
        "smoke",
        "estimation_pilot",
        "full_hlt",
    )


def test_hlt_estimation_pilot_dry_run_resolves_parallel_defaults(tmp_path: Path) -> None:
    script = _ROOT / "benchmarks" / "run_hlt_gpu_estimation.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "MODE": "estimation_pilot",
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path),
        },
    )

    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert lines["HLT_GPU_ESTIMATION_DRY_RUN"] == "1"
    assert lines["MODE"] == "estimation_pilot"
    assert lines["HLT_TARGET_BUILDER"] == "batched-sep"
    assert lines["HLT_PARAMETER_SET"] == "sw07_safe_27"
    assert lines["HLT_THETA_DRAWS"] == "128"
    assert lines["HLT_PERIODS"] == "4"
    assert lines["SEP_PERIODS"] == "4"
    assert lines["HLT_SEP_BATCH_CHUNK_SIZE"] == "8"
    assert lines["HLT_SURROGATE_HMC_DRAWS_PATH"] == str(tmp_path / "hlt_estimation_pilot_surrogate_hmc_draws.npz")
    assert lines["EPOCHS"] == "200"
    assert lines["HIDDEN"] == "192"
    assert lines["BLOCKS"] == "4"
    assert lines["TRAIN_BATCH_SIZE"] == "2048"
    assert lines["TRAIN_DTYPE"] == "float32"
    assert lines["JAX_LOG_DENSITY_BATCH_SIZE"] == "2048"
    assert lines["HMC_CHAINS"] == "64"
    assert lines["HMC_SAMPLES"] == "512"
    assert lines["RUN_ROM1_COMPARISON"] == "0"


def test_hlt_dry_run_grid_batched_targets_can_enable_rom1_comparison(tmp_path: Path) -> None:
    script = _ROOT / "benchmarks" / "run_hlt_gpu_estimation.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "MODE": "estimation_pilot",
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path),
            "HLT_TARGET_BUILDER": "grid-batched-sep",
            "RUN_ROM1_COMPARISON": "1",
        },
    )

    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert lines["HLT_TARGET_BUILDER"] == "grid-batched-sep"
    assert lines["HLT_SEP_BATCH_CHUNK_SIZE"] == "8"
    assert lines["RUN_ROM1_COMPARISON"] == "1"
    assert lines["ROM1_HMC_DRAWS_PATH"] == str(tmp_path / "hlt_estimation_pilot_rom1_hmc_draws.npz")
    assert lines["POSTERIOR_COMPARISON_PATH"] == str(tmp_path / "hlt_estimation_pilot_posterior_comparison.json")


def test_hlt_final_nonlinear_dry_run_enables_correctness_gates(tmp_path: Path) -> None:
    script = _ROOT / "benchmarks" / "run_hlt_gpu_estimation.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "MODE": "final_nonlinear",
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path),
        },
    )

    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert lines["MODE"] == "final_nonlinear"
    assert lines["HLT_TARGET_BUILDER"] == "batched-sep"
    assert lines["HLT_SEP_BATCH_CHUNK_SIZE"] == "16"
    assert lines["HLT_PARAMETER_SET"] == "phase1_18params_narrow"
    assert lines["HLT_THETA_DESIGN"] == "prior"
    assert lines["HLT_THETA_DESIGN_SET"] == "phase1_18params_narrow"
    assert lines["HLT_THETA_INCLUDE_REFERENCE"] == "1"
    assert lines["HLT_DROP_RUNTIME_FAILURES"] == "1"
    assert lines["HLT_RUNTIME_PREFLIGHT_ONLY"] == "0"
    assert lines["HLT_MIN_RUNTIME_SUCCESSFUL_THETA"] == "64"
    assert lines["HLT_THETA_DRAWS"] == "192"
    assert lines["HLT_MIN_ACCEPTED_SAMPLES"] == "256"
    assert lines["HLT_SURROGATE_BUNDLE_PATH"] == str(tmp_path / "hlt_final_nonlinear_surrogate_bundle.snn.npz")
    assert lines["HLT_REUSE_SURROGATE_BUNDLE"] == "0"
    assert lines["HLT_REQUIRE_FULL_TARGET_SUCCESS"] == "1"
    assert lines["HLT_REQUIRE_JAX_PARITY"] == "1"
    assert lines["HLT_REQUIRE_HMC"] == "1"
    assert lines["HLT_REQUIRE_SOLVED_STEADY_STATE"] == "1"
    assert lines["HLT_STEADY_STATE_MAX_ITER"] == "200"
    assert lines["LIKELIHOOD_RUNTIME_MODE"] == "full-jax"
    assert lines["LIKELIHOOD_QME_ALGORITHM"] == "schur_gpu"
    assert lines["JAX_LOG_DENSITY_BATCH_SIZE"] == "0"
    assert lines["JAX_LOG_DENSITY_BATCH_REPEAT_EVALS"] == "0"
    assert lines["TRAIN_DTYPE"] == "float32"
    assert lines["FAIL_ON_QUALITY_GATE"] == "1"


def test_hlt_final_nonlinear_dry_run_can_enable_runtime_preflight_only(tmp_path: Path) -> None:
    script = _ROOT / "benchmarks" / "run_hlt_gpu_estimation.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "MODE": "final_nonlinear",
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path),
            "HLT_RUNTIME_PREFLIGHT_ONLY": "1",
        },
    )

    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert lines["MODE"] == "final_nonlinear"
    assert lines["HLT_RUNTIME_PREFLIGHT_ONLY"] == "1"
    assert lines["HLT_THETA_DRAWS"] == "192"
    assert lines["HLT_MIN_RUNTIME_SUCCESSFUL_THETA"] == "64"


def test_hlt_full_hlt_dry_run_enables_all_parameter_strict_profile(tmp_path: Path) -> None:
    script = _ROOT / "benchmarks" / "run_hlt_gpu_estimation.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "MODE": "full_hlt",
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path),
        },
    )

    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert lines["MODE"] == "full_hlt"
    assert lines["HLT_TARGET_BUILDER"] == "batched-sep"
    assert lines["HLT_SEP_BATCH_CHUNK_SIZE"] == "2"
    assert lines["HLT_PARAMETER_SET"] == "all"
    assert lines["HLT_THETA_DESIGN"] == "perturbation"
    assert lines["HLT_DROP_RUNTIME_FAILURES"] == "1"
    assert lines["HLT_MIN_RUNTIME_SUCCESSFUL_THETA"] == "128"
    assert lines["HLT_THETA_DRAWS"] == "288"
    assert lines["HLT_PERIODS"] == "8"
    assert lines["SEP_PERIODS"] == "8"
    assert lines["HLT_SURROGATE_BUNDLE_PATH"] == str(tmp_path / "hlt_full_hlt_surrogate_bundle.snn.npz")
    assert lines["HLT_REUSE_SURROGATE_BUNDLE"] == "0"
    assert lines["HLT_MIN_ACCEPTED_SAMPLES"] == "1024"
    assert lines["HLT_REQUIRE_FULL_TARGET_SUCCESS"] == "1"
    assert lines["HLT_REQUIRE_JAX_PARITY"] == "1"
    assert lines["HLT_REQUIRE_HMC"] == "1"
    assert lines["HLT_REQUIRE_SOLVED_STEADY_STATE"] == "1"
    assert lines["HLT_STEADY_STATE_MAX_ITER"] == "200"
    assert lines["LIKELIHOOD_RUNTIME_MODE"] == "full-jax"
    assert lines["LIKELIHOOD_QME_ALGORITHM"] == "schur_gpu"
    assert lines["JAX_LOG_DENSITY_BATCH_SIZE"] == "0"
    assert lines["JAX_LOG_DENSITY_BATCH_REPEAT_EVALS"] == "0"
    assert lines["HMC_CHAINS"] == "64"
    assert lines["HMC_SAMPLES"] == "1000"
    assert lines["TRAIN_DTYPE"] == "float32"
    assert lines["FAIL_ON_QUALITY_GATE"] == "1"


def test_build_hlt_stage_environment_sets_gpu_and_runtime_modes(tmp_path: Path) -> None:
    module = _load_module()

    env = module.build_hlt_stage_environment(
        stage="full",
        python="/opt/venv/bin/python",
        result_root=tmp_path,
        allow_cpu=False,
        hlt_target_builder="grid-batched-sep",
        hlt_parameter_set="all",
        steady_state_mode="solve-or-reference",
        likelihood_runtime_mode="full-jax",
        likelihood_qme_algorithm="schur_gpu",
        likelihood_static_rows_mode="reference",
        jax_log_density_gradient=False,
        differentiate_shocks=True,
        extra_env={"HMC_CHAINS": "64", "RUN_ROM1_COMPARISON": "1"},
    )

    assert env["MODE"] == "full"
    assert env["PYTHON"] == "/opt/venv/bin/python"
    assert env["DEVICE"] == "gpu"
    assert env["REQUIRE_GPU"] == "1"
    assert env["HLT_TARGET_BUILDER"] == "grid-batched-sep"
    assert env["HLT_PARAMETER_SET"] == "all"
    assert env["HLT_STEADY_STATE_MODE"] == "solve-or-reference"
    assert env["LIKELIHOOD_RUNTIME_MODE"] == "full-jax"
    assert env["LIKELIHOOD_QME_ALGORITHM"] == "schur_gpu"
    assert env["JAX_LOG_DENSITY_GRADIENT"] == "0"
    assert env["DIFFERENTIATE_SHOCKS"] == "1"
    assert env["HMC_CHAINS"] == "64"
    assert env["RUN_ROM1_COMPARISON"] == "1"


def test_build_hlt_stage_environment_can_preserve_stage_defaults(tmp_path: Path) -> None:
    module = _load_module()

    env = module.build_hlt_stage_environment(
        stage="full_hlt",
        python="/opt/venv/bin/python",
        result_root=tmp_path,
        allow_cpu=False,
        hlt_target_builder=None,
        hlt_parameter_set=None,
        steady_state_mode=None,
        likelihood_runtime_mode=None,
        likelihood_qme_algorithm=None,
        likelihood_static_rows_mode=None,
        jax_log_density_gradient=True,
        differentiate_shocks=False,
        extra_env={},
    )

    assert env["MODE"] == "full_hlt"
    assert "HLT_STEADY_STATE_MODE" not in env
    assert "LIKELIHOOD_RUNTIME_MODE" not in env
    assert "LIKELIHOOD_QME_ALGORITHM" not in env
    assert "LIKELIHOOD_STATIC_ROWS_MODE" not in env


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
                        "surrogate_bundle_path": "/tmp/hlt_bundle.snn.npz",
                        "surrogate_bundle_reused": True,
                        "steady_state_solved_count": 8,
                        "steady_state_fallback_count": 0,
                        "steady_state_attempted_solved_count": 8,
                        "steady_state_attempted_fallback_count": 2,
                        "strict_solved_steady_state_preflight": True,
                        "target_diagnostics": {
                            "builder": "adaptive_sep",
                            "status": "ok",
                            "accepted_samples": 64,
                            "runtime_prepared_theta_draws": 8,
                            "runtime_dropped_theta_count": 2,
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
                            "posterior_draws": {
                                "path": "/tmp/surrogate_draws.npz",
                                "post_warmup_draws": 1024,
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hlt_full_posterior_comparison.json").write_text(
        json.dumps(
            {
                "max_abs_mean_diff": 0.01,
                "max_ks_distance": 0.25,
                "mean_quantile_wasserstein": 0.005,
                "common_parameters": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )

    summary = module.summarize_hlt_result(output)

    assert summary["status"] == "ok"
    assert summary["backend"] == "gpu"
    assert summary["parameter_count"] == 2
    assert summary["accepted_samples"] == 64
    assert summary["runtime_prepared_theta_draws"] == 8
    assert summary["runtime_dropped_theta_count"] == 2
    assert summary["surrogate_bundle_path"] == "/tmp/hlt_bundle.snn.npz"
    assert summary["surrogate_bundle_reused"] is True
    assert summary["steady_state_solved_count"] == 8
    assert summary["steady_state_fallback_count"] == 0
    assert summary["steady_state_attempted_solved_count"] == 8
    assert summary["steady_state_attempted_fallback_count"] == 2
    assert summary["strict_solved_steady_state_preflight"] is True
    assert summary["jax_log_density_parity_ok"] is True
    assert summary["hmc_draws_per_second"] == 34.13
    assert summary["surrogate_hmc_draws"]["path"] == "/tmp/surrogate_draws.npz"
    assert summary["posterior_comparison_path"] == str(tmp_path / "hlt_full_posterior_comparison.json")
    assert summary["posterior_comparison"]["max_ks_distance"] == 0.25
