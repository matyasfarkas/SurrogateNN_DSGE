from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]
RAW_REPORT = ROOT / "paper" / "generated" / "gali_sep_likelihood_raw.json"


def _load_module() -> ModuleType:
    path = ROOT / "benchmarks" / "build_gali_sep_likelihood_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "build_gali_sep_likelihood_evidence", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw() -> dict:
    return json.loads(RAW_REPORT.read_text(encoding="utf-8"))


def test_build_evidence_accepts_archived_scoped_pass() -> None:
    module = _load_module()
    evidence = module.build_evidence(RAW_REPORT)

    assert evidence["publication_ready"] is False
    assert evidence["model"]["variables"] == 23
    assert evidence["metrics"]["recovered_shock_max_abs"] < 1.0e-9
    assert evidence["metrics"]["jacobian_max_abs"] < 1.0e-6
    assert evidence["metrics"]["loglikelihood_abs"] < 1.0e-4
    assert "multi-period nonlinear likelihood recursion" in evidence["claims"][
        "not_supported"
    ]


def test_rejects_dirty_source_repository() -> None:
    module = _load_module()
    report = _raw()
    report["provenance"]["repository"]["dirty"] = True

    with pytest.raises(ValueError, match="was not clean"):
        module.validate_report(report)


def test_rejects_relaxed_likelihood_threshold() -> None:
    module = _load_module()
    report = _raw()
    report["thresholds"]["loglikelihood_abs"] = 1.0e-2

    with pytest.raises(ValueError, match="Unexpected thresholds.loglikelihood_abs"):
        module.validate_report(report)


def test_recomputes_and_rejects_hidden_jacobian_mismatch() -> None:
    module = _load_module()
    report = _raw()
    report["julia_export"]["result"]["jacobian"][0][0] += 2.0e-5

    with pytest.raises(ValueError, match="Recomputed numerical gates failed: Jacobian"):
        module.validate_report(report)


def test_rejects_dropped_numerical_gate() -> None:
    module = _load_module()
    report = _raw()
    report["gates"].pop("loglikelihood_parity")

    with pytest.raises(ValueError, match="Every declared"):
        module.validate_report(report)
