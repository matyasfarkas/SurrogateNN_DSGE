from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from compile_sw07_profile_comparison import (  # noqa: E402
    compile_profile,
    render_latex,
    render_markdown,
)


def _result(tmp_path: Path, backend: str, qme: str, scale: float) -> tuple[dict, Path]:
    path = tmp_path / f"{backend}.json"
    result = {
        "runtime": {
            "jax_default_backend": backend,
            "jax_devices": ["TFRT_CPU_0" if backend == "cpu" else "cuda:0"],
            "jax_enable_x64": True,
            "jax_version": "test",
            "numpyro_version": "test",
            "platform": backend,
            "processor": backend,
            "python": "test",
            "nvidia_smi": "NVIDIA GeForce RTX TEST |" if backend == "gpu" else "",
        },
        "benchmark": {
            "preset": "sw07_paper",
            "model_name": "Smets_Wouters_2007_linear",
            "parameter_names": [f"p{i}" for i in range(36)],
            "parameter_count": 36,
            "steady_state_mode": "solve",
            "dtype": "float64",
            "periods": 180,
            "presample_periods": 6,
            "initial_covariance_strategy": "theoretical",
            "measurement_error_scale": 0.0,
            "jitter": 0.0,
            "qme_algorithm": qme,
            "n_vars": 40,
            "n_exo": 7,
            "n_observables": 7,
            "data_provenance": {
                "sha256": "35e09d915973e89442e50584f790f4859393219818c03ec8f2eead2b8a3c2050",
                "model_source_sha256": "model",
                "sample_start_index_zero_based": 50,
            },
        },
        "preflight": {
            "loglikelihood": -1045.8945641261721,
            "gradient": [float(i) for i in range(36)],
            "value_first_call_s": 10.0 * scale,
            "gradient_first_call_s": 20.0 * scale,
            "value_steady": {"reps": 3, "median_s": 2.0 * scale},
            "gradient_steady": {"reps": 3, "median_s": 4.0 * scale},
        },
    }
    path.write_text(json.dumps(result))
    return result, path


def test_compile_profile_reports_speedups_and_parity(tmp_path: Path) -> None:
    cpu, cpu_path = _result(tmp_path, "cpu", "schur", 1.0)
    gpu, gpu_path = _result(tmp_path, "gpu", "schur_gpu", 0.25)
    profile = compile_profile(cpu, gpu, cpu_path=cpu_path, gpu_path=gpu_path)
    assert profile["parity"]["passed"] is True
    assert profile["comparison"]["value_steady_speedup_cpu_over_gpu"] == 4.0
    assert profile["profiles"]["gpu"]["gradient"]["steady_evaluations_per_hour"] == 3600.0
    assert "CPU/GPU likelihood absolute error" in render_markdown(profile)
    assert "controlled hardware-only ablation" in render_markdown(profile)
    assert "Matched SW07 CPU--GPU" in render_latex(profile)
    assert "not a controlled hardware-only ablation" in render_latex(profile)


def test_compile_profile_fails_closed_on_numerical_mismatch(tmp_path: Path) -> None:
    cpu, cpu_path = _result(tmp_path, "cpu", "schur", 1.0)
    gpu, gpu_path = _result(tmp_path, "gpu", "schur_gpu", 0.25)
    gpu["preflight"]["gradient"][0] += 1.0
    with pytest.raises(ValueError, match="parity gate failed"):
        compile_profile(cpu, gpu, cpu_path=cpu_path, gpu_path=gpu_path)
