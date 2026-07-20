from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_COLAB_NOTEBOOK = _ROOT / "notebooks" / "colab_jax_numpyro_gemini_profile.ipynb"
_SW07_LONG_NOTEBOOK = _ROOT / "notebooks" / "colab_sw07_long_profile.ipynb"


def _notebook_code(path: Path) -> tuple[dict[str, object], str]:
    notebook = json.loads(path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return notebook, code


def _assert_clean_notebook(notebook: dict[str, object]) -> None:
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_colab_profile_notebook_is_clean_and_gpu_ready() -> None:
    notebook, code = _notebook_code(_COLAB_NOTEBOOK)

    _assert_clean_notebook(notebook)
    assert r"CUDA Version:\s*([0-9]+)" in code
    assert r"Driver Version:\s*([0-9.]+)" in code
    assert r"CUDA Version:\\s*" not in code
    assert "JAX_ENABLE_X64" in code
    assert "numpyro>=0.20" in code
    assert "RUN_FULL_JULIA_BENCHMARK = False" in code
    assert "Likelihood did not change across rho_a/rho_y values" in code


def test_sw07_long_profile_notebook_is_dedicated_large_model_runner() -> None:
    notebook, code = _notebook_code(_SW07_LONG_NOTEBOOK)

    _assert_clean_notebook(notebook)
    assert "PROFILE_MODE = \"calibration\"" in code
    assert "SW07_MODEL_SOURCE_PATH" in code
    assert "benchmarks/model_sources/Smets_Wouters_2007_HLT.jl" in code
    assert "Smets_Wouters_2007_HLT" in code
    assert "test_payloads.json" in code
    assert "solve_first_order_model" in code
    assert "kalman_loglikelihood_from_model_jax" in code
    assert "evaluate_numpyro_kalman_log_density_jax" in code
    assert "RUN_NUTS_SMOKE = False" in code
    assert "COLAB_SETUP_VERSION" in code
    assert "Colab wheels changed. Restarting the runtime once" in code
    assert "numpy>=2.1,<2.3" in code
    assert "NumPyro log density did not change under SW07 parameter perturbations" in code
    assert "JULIA_REPO_URL" not in code
    assert "JULIA_ROOT" not in code
    assert "SurrogateNN_Estimation.jl" not in code
    assert "profile_validation.py" not in code
