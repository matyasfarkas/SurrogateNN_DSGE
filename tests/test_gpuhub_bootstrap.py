from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gpuhub_bootstrap.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gpuhub_bootstrap_for_tests", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_nvidia_smi_extracts_driver_cuda_and_memory() -> None:
    module = _load_module()
    text = """
    | NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0 |
    |   0  NVIDIA GeForce RTX 5090      On | 00000000:01:00.0 Off |                  N/A |
    |      1024MiB / 32768MiB |
    """

    parsed = module.parse_nvidia_smi(text)

    assert parsed["driver_version"] == "580.65.06"
    assert parsed["cuda_version"] == "13.0"
    assert parsed["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert parsed["memory_total_mib"] == 32768


def test_choose_jax_requirement_prefers_cuda13_when_driver_allows() -> None:
    module = _load_module()

    requirement = module.choose_jax_requirement(
        "auto",
        cuda_version="13.0",
        driver_version="580.65.06",
    )

    assert requirement == "jax[cuda13]>=0.6"


def test_choose_jax_requirement_falls_back_to_cuda12_for_driver_550() -> None:
    module = _load_module()

    requirement = module.choose_jax_requirement(
        "auto",
        cuda_version="12.8",
        driver_version="550.144.03",
    )

    assert requirement == "jax[cuda12]>=0.6"


def test_choose_jax_requirement_rejects_legacy_cuda11_stack() -> None:
    module = _load_module()

    try:
        module.choose_jax_requirement(
            "auto",
            cuda_version="11.1",
            driver_version="455.32",
        )
    except RuntimeError as exc:
        assert "JAX 0.3.10 / CUDA 11.1" in str(exc)
    else:
        raise AssertionError("Expected legacy CUDA 11.1 stack to be rejected.")


def test_build_benchmark_command_uses_force_gpu_and_ess_output() -> None:
    module = _load_module()

    cmd, output = module.build_benchmark_command(
        python="/usr/bin/python",
        root=Path("/root/gpuhub-tmp/SurrogateNN_DSGE"),
        mode="proper_5090",
        dtype="float32",
        qme_algorithm="doubling",
        force_gpu=True,
        preflight_reps=1,
        progress_bar=False,
        verbose=True,
        heartbeat_seconds=15.0,
    )

    assert "benchmarks/posterior_sampling_speed.py" in cmd
    assert "--force-gpu" in cmd
    assert "--verbose" in cmd
    assert "--heartbeat-seconds" in cmd
    assert "15.0" in cmd
    assert "--schur-support-draws" in cmd
    assert "sw07_safe_15" in cmd
    assert output.name == "gpuhub_sw07_posterior_ess_proper_5090_float32_doubling.json"


def test_sanitize_jax_runtime_environment_clears_ld_library_path_for_cuda_wheels() -> None:
    module = _load_module()

    updates = module.sanitize_jax_runtime_environment("jax[cuda13]>=0.6")

    assert updates["LD_LIBRARY_PATH"] is None
    assert updates["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
