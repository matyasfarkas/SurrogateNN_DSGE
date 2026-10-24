from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "build_stochastic_pilot_evidence.py"
    spec = importlib.util.spec_from_file_location("build_stochastic_pilot_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path, *, model: str = "Gali_2015_chapter_3_nonlinear") -> None:
    root.mkdir()
    run: dict[str, Any] = {
        "model_name": model,
        "equation_count": 23,
        "parameter_count": 16,
        "parameter_subset": ["phi_pi", "phi_y"],
        "original_theta_draws": 8,
        "periods": 10,
        "pipeline_s": 7178.95,
        "steady_state_solved_count": 8,
        "steady_state_fallback_count": 0,
        "validation_rmse_mean": 3.8e-4,
        "validation_improvement_mean": 0.92,
        "dataset_summary": {"theta_success_rate": 0.375},
        "quality_gate": {
            "status": "failed",
            "issues": ["Full-path theta share below gate: 0.375 < 1."],
        },
        "target_diagnostics": {
            "theta_full_success_count": 3,
            "accepted_samples": 30,
            "residual_norm_max": 9.4e-9,
            "sep_config": {"branching_order": 1, "nnodes": 3},
            "memory_estimate": {
                "stacked_unknowns": 1886,
                "rough_chunk_workspace_gib": 2.54,
            },
            "shifted_path_warm_start": {"supported": False},
            "failure_log": [
                {"theta_index": index, "period": 0, "accepted": False}
                for index in range(5)
            ],
        },
        "jax_surrogate_log_density": {"parity_abs_diff": 3.1e-12},
        "surrogate_hmc": {
            "chains": 1,
            "warmup": 25,
            "post_warmup_draws": 50,
            "accepted_share": 1.0,
            "draws_per_second": 7.24,
        },
        "training_metadata": {"residual_shrinkage_calibrated": True},
    }
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps({"results": {"hlt_fixed_ss_smoke": run}}), encoding="utf-8"
    )
    for name in ("run.log", "posterior_draws.npz", "surrogate_bundle.snn.npz"):
        (root / name).write_bytes(name.encode("utf-8"))

    artifacts = []
    for name in ("result.json", "run.log", "posterior_draws.npz", "surrogate_bundle.snn.npz"):
        path = root / name
        artifacts.append(
            {
                "path": f"benchmarks/results/example/{name}",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "surrogatenn-dsge-publication-run-manifest/v1",
        "status": "failed",
        "exit_code": 1,
        "prepared_at_utc": "2026-10-23T21:55:04+00:00",
        "completed_at_utc": "2026-10-23T23:55:14+00:00",
        "repository": {"commit": "a" * 40, "dirty_worktree": False},
        "repository_after": {"commit": "a" * 40, "dirty_worktree": False},
        "environment": {
            "nvidia_runtime": {"rows": ["NVIDIA RTX, uuid, driver, 32760"]}
        },
        "artifacts": artifacts,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_failed_pilot_is_preserved_as_negative_result(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    _write_fixture(run_dir)

    evidence = module.build_evidence(run_dir)

    assert evidence["publication_ready"] is False
    assert evidence["evidence_level"] == "engineering_pilot_negative_result"
    assert evidence["solver"]["full_paths"] == 3
    assert evidence["solver"]["theta_draws"] == 8
    assert evidence["solver"]["first_period_failures"] == 5
    assert evidence["solver"]["reference_path_failed"] is True
    assert evidence["quality_gate"]["status"] == "failed"
    assert evidence["diagnostics"]["validation_is_untouched"] is False
    assert "publication-ready estimation evidence" in evidence["interpretation"][
        "not_supported"
    ]


def test_cli_writes_machine_and_manuscript_outputs(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    _write_fixture(run_dir)
    output = tmp_path / "generated"

    assert module.main([str(run_dir), "--output-dir", str(output)]) == 0

    markdown = (output / "stochastic_pilot_evidence.md").read_text(encoding="utf-8")
    latex = (output / "stochastic_pilot_results.tex").read_text(encoding="utf-8")
    assert "**Publication ready:** no" in markdown
    assert "3/8" in markdown
    assert "Strict quality gate & failed" in latex
    assert "applied shock scales twice" in latex


def test_builder_rejects_tampered_artifact(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    _write_fixture(run_dir)
    (run_dir / "run.log").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        module.build_evidence(run_dir)


def test_builder_rejects_wrong_model(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    _write_fixture(run_dir, model="toy")

    with pytest.raises(ValueError, match="Unexpected model"):
        module.build_evidence(run_dir)
