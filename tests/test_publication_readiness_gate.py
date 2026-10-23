from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "publication_readiness_gate.py"
    spec = importlib.util.spec_from_file_location("publication_readiness_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assign(payload: dict[str, Any], path: str, value: Any) -> None:
    node = payload
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _passing_payload(module: ModuleType, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for gate in module.GATES:
        threshold = gate.thresholds[mode]
        if threshold is None:
            continue
        if gate.operator in {
            "min",
            "max",
            "min_int",
            "max_int",
            "min_fraction",
            "max_fraction",
            "eq",
        }:
            value = threshold
        elif gate.operator == "nonempty":
            value = "nonlinear_gali_toy"
        elif gate.operator == "nonempty_list":
            value = [f"theta_{idx}" for idx in range(6)]
        elif gate.operator == "git_sha":
            value = "a" * 40
        elif gate.operator == "sha256":
            value = "b" * 64
        else:  # pragma: no cover - catches a gate added without updating this fixture
            raise AssertionError(gate.operator)
        _assign(payload, gate.path, value)
    return payload


def test_publication_payload_passes_and_is_scoped_to_reduced_experiment() -> None:
    module = _load_module()
    payload = _passing_payload(module, "publication")

    report = module.evaluate_payload(payload, mode="publication")

    assert report["status"] == "pass"
    assert report["computational_evidence_gate_passed"] is True
    assert report["experiment_scope"] == "reduced_experiment"
    assert "does not validate a full-scale Galí estimation" in report["interpretation"]
    assert set(report["categories"]) == {
        "scope",
        "reproducibility",
        "numerical_solution",
        "qz_sep_accuracy",
        "surrogate_holdout",
        "likelihood_parity",
        "hmc",
        "timing",
    }


def test_smoke_pass_does_not_claim_publication_readiness() -> None:
    module = _load_module()

    report = module.evaluate_payload(_passing_payload(module, "smoke"), mode="smoke")

    assert report["status"] == "pass"
    assert report["computational_evidence_gate_passed"] is True
    assert report["summary"]["not_required"] == 10
    cross_checks = [
        check
        for check in report["categories"]["qz_sep_accuracy"]["checks"]
        if check["path"].startswith("qz_sep_accuracy.cross_implementation")
    ]
    assert {check["status"] for check in cross_checks} == {"not_required"}


def test_missing_and_nonfinite_metrics_fail_their_categories() -> None:
    module = _load_module()
    payload = _passing_payload(module, "publication")
    del payload["likelihood_parity"]["gradient_cosine_min"]
    payload["numerical_solution"]["runtime_success_share"] = float("nan")

    report = module.evaluate_payload(payload, mode="publication")

    assert report["status"] == "fail"
    assert report["computational_evidence_gate_passed"] is False
    assert report["categories"]["likelihood_parity"]["status"] == "fail"
    assert report["categories"]["numerical_solution"]["status"] == "fail"
    checks = {
        check["path"]: check
        for category in report["categories"].values()
        for check in category["checks"]
    }
    assert checks["likelihood_parity.gradient_cosine_min"]["missing"] is True
    assert checks["numerical_solution.runtime_success_share"]["detail"] == (
        "not a finite number"
    )


def test_scope_guard_rejects_full_model_claim() -> None:
    module = _load_module()
    payload = _passing_payload(module, "publication")
    payload["metadata"]["claimed_scope"] = "full_estimation"

    report = module.evaluate_payload(payload, mode="publication")

    assert report["status"] == "fail"
    failed = [
        check["path"]
        for check in report["categories"]["scope"]["checks"]
        if check["status"] == "fail"
    ]
    assert failed == ["metadata.claimed_scope"]


def test_out_of_range_fraction_and_parameter_count_mismatch_fail() -> None:
    module = _load_module()
    payload = _passing_payload(module, "publication")
    payload["numerical_solution"]["runtime_success_share"] = 1.5
    payload["metadata"]["reduction"]["parameter_names"] = ["phi_pi", "phi_y"]

    report = module.evaluate_payload(payload, mode="publication")

    assert report["status"] == "fail"
    checks = {
        check["path"]: check
        for category in report["categories"].values()
        for check in category["checks"]
    }
    assert checks["numerical_solution.runtime_success_share"]["detail"] == (
        "not a fraction in [0, 1]"
    )
    assert checks[
        "metadata.reduction.estimated_parameter_count_vs_parameter_names"
    ]["status"] == "fail"


def test_cli_writes_json_and_markdown_and_returns_two_on_failure(tmp_path: Path) -> None:
    module = _load_module()
    payload = _passing_payload(module, "evidence")
    payload["hmc"]["max_r_hat"] = 1.2
    input_path = tmp_path / "evidence.json"
    output_path = tmp_path / "gate.json"
    markdown_path = tmp_path / "gate.md"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    returncode = module.main(
        [
            str(input_path),
            "--mode",
            "evidence",
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert returncode == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert len(report["input_sha256"]) == 64
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "**Status:** **FAIL**" in markdown
    assert "`hmc.max_r_hat`" in markdown


def test_publication_thresholds_are_not_weaker_than_smoke_thresholds() -> None:
    module = _load_module()
    for gate in module.GATES:
        smoke = gate.thresholds["smoke"]
        publication = gate.thresholds["publication"]
        if gate.operator in {"min", "min_int", "min_fraction"} and isinstance(smoke, (int, float)):
            assert publication >= smoke, gate.path
        if gate.operator in {"max", "max_int", "max_fraction"} and isinstance(smoke, (int, float)):
            assert publication <= smoke, gate.path
