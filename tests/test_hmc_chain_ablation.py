from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "hmc_chain_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "hmc_chain_ablation_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        methods="static_hmc,numpyro_hmc,numpyro_nuts",
        chains="1,4",
        label="test",
        results_dir=tmp_path,
        preset="sw07_hlt",
        payload_path=tmp_path / "payloads.json",
        case="medium_sw07_hlt",
        periods=40,
        parameters="sw07_safe_15",
        prior_width_scale=0.0025,
        prior_width_floor=1.0e-4,
        warmup=2,
        samples=3,
        chain_method="vectorized",
        target_accept_prob=0.8,
        max_tree_depth=5,
        numpyro_hmc_num_steps=8,
        numpyro_hmc_step_size=0.1,
        static_leapfrog_steps=8,
        static_step_size=1.0,
        static_max_step_size=2.0,
        steady_reps=0,
        dtype="float64",
        platform="gpu",
        force_gpu=True,
        qme_algorithm="schur_gpu",
        parameters_are_resolved=True,
        skip_parameter_bounds=True,
        failure_value=-1.0e12,
        verbose=True,
    )


def test_parse_int_list_accepts_commas_and_spaces() -> None:
    module = _load_module()

    assert module.parse_int_list("1, 2 4") == (1, 2, 4)


def test_parse_int_list_rejects_nonpositive_values() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="positive"):
        module.parse_int_list("1,0")


def test_build_jobs_emits_all_methods_and_chain_counts(tmp_path: Path) -> None:
    module = _load_module()

    jobs = module.build_jobs(_args(tmp_path))

    assert [(job.method, job.chains) for job in jobs] == [
        ("static_hmc", 1),
        ("static_hmc", 4),
        ("numpyro_hmc", 1),
        ("numpyro_hmc", 4),
        ("numpyro_nuts", 1),
        ("numpyro_nuts", 4),
    ]
    static_command = jobs[0].command
    assert "static_hmc_sampling_speed.py" in " ".join(static_command)
    assert "--force-gpu" in static_command
    assert "--parameters-are-resolved" in static_command
    assert "--skip-parameter-bounds" in static_command
    assert str(jobs[0].output_path).endswith(
        "test_static_hmc_sw07_hlt_medium_sw07_hlt_sw07_safe_15_40p_1chains_float64.json"
    )


def test_dry_run_records_commands_without_execution(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    args.methods = "static_hmc"
    args.chains = "2"
    args.dry_run = True
    args.skip_existing = False
    args.summary_output = tmp_path / "summary.json"

    summary = module.run_ablation(args)

    assert summary["jobs"][0]["status"] == "dry_run"
    assert (tmp_path / "summary.json").exists()
