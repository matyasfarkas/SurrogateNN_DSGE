from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_COLAB_NOTEBOOK = _ROOT / "notebooks" / "colab_jax_numpyro_gemini_profile.ipynb"


def test_colab_profile_notebook_is_clean_and_gpu_ready() -> None:
    notebook = json.loads(_COLAB_NOTEBOOK.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

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
    assert r"CUDA Version:\s*([0-9]+)" in code
    assert r"Driver Version:\s*([0-9.]+)" in code
    assert r"CUDA Version:\\s*" not in code
    assert "JAX_ENABLE_X64" in code
    assert "numpyro>=0.20" in code
    assert "RUN_FULL_JULIA_BENCHMARK = False" in code
    assert "Likelihood did not change across rho_a/rho_y values" in code
