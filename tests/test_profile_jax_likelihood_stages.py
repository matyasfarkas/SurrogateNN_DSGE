from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "profile_jax_likelihood_stages.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "profile_jax_likelihood_stages_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_keeps_gradient_profile_opt_in() -> None:
    module = _load_module()

    default_args = module._parse_args([])
    gradient_args = module._parse_args(["--include-gradient", "--gradient-reps", "2"])

    assert default_args.include_gradient is False
    assert default_args.gradient_reps == 0
    assert gradient_args.include_gradient is True
    assert gradient_args.gradient_reps == 2


def test_toy_stage_profile_cli_writes_model_level_timings(tmp_path: Path) -> None:
    output_path = tmp_path / "toy_stage_profile.json"
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--preset",
            "toy_ar2",
            "--periods",
            "4",
            "--parameters",
            "rho_a",
            "--reps",
            "0",
            "--dtype",
            "float64",
            "--platform",
            "cpu",
            "--qme-algorithm",
            "schur_gpu",
            "--output",
            str(output_path),
        ],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.loads(output_path.read_text())

    assert result["benchmark"]["preset"] == "toy_ar2"
    assert result["values"]["first_order_converged"] is True
    assert "qme_solution_max_abs" in result["values"]
    assert "first_order_relative_qme_residual" not in result["values"]
    assert result["values"]["loglikelihood"] == result["values"]["full_loglikelihood"]
    assert {
        "resolve_parameters",
        "dynamic_jacobian",
        "first_order_solution",
        "state_space_and_initial_covariance",
        "kalman_loglikelihood_only",
        "full_loglikelihood",
    }.issubset(result["timings"])
