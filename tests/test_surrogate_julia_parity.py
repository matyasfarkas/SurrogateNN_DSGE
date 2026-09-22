from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


def _load_nonlinear_likelihood_parity_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "benchmarks" / "validate_nonlinear_likelihood_julia_parity.py"
    spec = importlib.util.spec_from_file_location(
        "validate_nonlinear_likelihood_julia_parity",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_surrogate_predictions_match_julia_reference_when_enabled() -> None:
    if os.environ.get("SURROGATENN_RUN_JULIA_PARITY") != "1":
        pytest.skip("Set SURROGATENN_RUN_JULIA_PARITY=1 to run Julia surrogate parity.")
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "validate_surrogate_julia_parity.py"
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)


def test_stored_nonlinear_switching_likelihood_matches_julia_reference() -> None:
    module = _load_nonlinear_likelihood_parity_module()

    checks = module.validate_stored_nonlinear_likelihood_parity()

    assert {check.case for check in checks} == {"medium_sw07_hlt", "small_fs2000"}
    assert all(check.switching_abs_diff <= 1.0e-2 for check in checks)
    assert all(check.gate_max_abs_diff <= 4.0e-5 for check in checks)


def test_stored_nonlinear_likelihood_validator_reports_missing_sep_fields(
    tmp_path: Path,
) -> None:
    module = _load_nonlinear_likelihood_parity_module()
    stages = {
        "switching_value": {"status": "ok", "result": {"value": 1.0}},
        "gate_stats": {
            "status": "ok",
            "result": {
                "e_stat": [0.1],
                "f_stat": [0.2],
                "linear_observations": [[0.3]],
                "shocks": [[0.4]],
            },
        },
        "sep_inversion": {"status": "ok", "result": {}},
    }
    (tmp_path / "julia_results.json").write_text(
        json.dumps({"cases": {"toy": {"stages": stages}}}),
        encoding="utf-8",
    )
    stages["sep_inversion"] = {"status": "ok", "result": {"value": 1.0}}
    (tmp_path / "python_results.json").write_text(
        json.dumps({"cases": {"toy": {"stages": stages}}}),
        encoding="utf-8",
    )

    with pytest.raises(module.PayloadFieldError) as exc_info:
        module.validate_stored_nonlinear_likelihood_parity(tmp_path, cases=("toy",))

    assert "julia_results.json" in str(exc_info.value)
    assert "cases.toy.stages.sep_inversion.result.value" in str(exc_info.value)
