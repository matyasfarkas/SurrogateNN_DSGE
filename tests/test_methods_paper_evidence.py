from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "build_methods_paper_evidence.py"
    spec = importlib.util.spec_from_file_location("build_methods_paper_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary_run(*, full_jax: bool, corrected: bool) -> dict[str, Any]:
    return {
        "accepted_targets": 16,
        "exact_endpoint_correction": corrected,
        "full_paths": 4,
        "hmc_accepted_share": 1.0,
        "hmc_chains": 2 if corrected else 1,
        "hmc_draws_per_second": 0.03 if corrected else 1.0,
        "hmc_samples": 2,
        "hmc_warmup": 1,
        "jax_parity_abs_diff": 0.0,
        "likelihood_runtime_mode": "full-jax" if full_jax else "fixed-reference",
        "max_sep_residual": 1.0e-9,
        "periods": 4,
        "pipeline_s": 10.0,
        "quality_gate": "ok",
        "status": "ok",
        "target_gradient_evaluated": False if corrected else True,
        "theta_draws": 4,
        "validation_improvement_mean": 0.5,
        "validation_rmse_mean": 1.0e-4,
    }


def _write_suite(root: Path, *, model_name: str = "Gali_2015_chapter_3_nonlinear") -> None:
    summaries = {
        "quick": _summary_run(full_jax=False, corrected=False),
        "corrected_full_jax_resource": _summary_run(full_jax=True, corrected=True),
        "default": _summary_run(full_jax=False, corrected=False),
    }
    summary_dir = root / "suite"
    summary_dir.mkdir(parents=True)
    (summary_dir / "scientific_summary.json").write_text(
        json.dumps({"model": model_name, "runs": summaries}), encoding="utf-8"
    )
    directories = {
        "quick": "gali_quick_1",
        "corrected_full_jax_resource": "gali_full_jax_resource_1",
        "default": "gali_default_1",
    }
    for label, dirname in directories.items():
        run_dir = root / dirname
        run_dir.mkdir()
        summary = summaries[label]
        run = {
            "model_name": model_name,
            "equation_count": 23,
            "parameter_count": 16,
            "target_diagnostics": {"builder": "batched_sep"},
            "quality_gate": {"status": "ok"},
            "jax_surrogate_log_density": {
                "caveat": "surrogate likelihood",
                "backend": "gpu",
                "qme_algorithm": "schur",
            },
            "surrogate_hmc": {
                "kind": "corrected" if summary["exact_endpoint_correction"] else "smoke"
            },
        }
        run["target_diagnostics"]["sep_config"] = {
            "branching_order": 0,
            "nnodes": 1,
        }
        (run_dir / "result.json").write_text(
            json.dumps({"results": {"hlt_fixed_ss_smoke": run}}), encoding="utf-8"
        )


def test_builder_preserves_claim_boundary(tmp_path: Path) -> None:
    module = _load_module()
    _write_suite(tmp_path)

    evidence = module.build_evidence(tmp_path)

    claims = {claim["id"]: claim for claim in evidence["claim_ledger"]}
    assert evidence["publication_ready"] is False
    assert claims["target_corrected_sampler_executes"]["status"] == (
        "resource_feasibility_only"
    )
    assert claims["exact_full_nonlinear_posterior"]["status"] == "not_supported"
    corrected = evidence["runs"]["corrected_full_jax_resource"]
    assert corrected["likelihood_kind"] == "surrogate_profile_inversion_objective"
    assert corrected["full_nonlinear_likelihood_in_hmc"] is False
    assert corrected["sep_branching_order"] == 0
    assert corrected["sep_nodes"] == 1
    assert corrected["jax_backend"] == "gpu"
    assert corrected["qme_algorithm"] == "schur"
    assert corrected["gpu_native_qme"] is False


def test_builder_rejects_wrong_model(tmp_path: Path) -> None:
    module = _load_module()
    _write_suite(tmp_path, model_name="Gali_3eq_linear")

    with pytest.raises(ValueError, match="Unexpected suite model"):
        module.build_evidence(tmp_path)


def test_cli_writes_machine_and_manuscript_outputs(tmp_path: Path) -> None:
    module = _load_module()
    suite = tmp_path / "input"
    suite.mkdir()
    _write_suite(suite)
    output = tmp_path / "generated"

    assert module.main([str(suite), "--output-dir", str(output)]) == 0

    evidence = json.loads((output / "pilot_evidence.json").read_text(encoding="utf-8"))
    assert evidence["evidence_level"] == "engineering_pilot"
    markdown = (output / "pilot_evidence.md").read_text(encoding="utf-8")
    latex = (output / "pilot_results.tex").read_text(encoding="utf-8")
    assert "**Publication ready:** no" in markdown
    assert "Correction therefore targets the declared surrogate likelihood" in latex
