from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


def _load_harness() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "benchmarks" / "validate_hlt_pipeline_parity.py"
    spec = importlib.util.spec_from_file_location("validate_hlt_pipeline_parity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_payload(*, perturb: float = 0.0) -> dict[str, object]:
    return {
        "cases": {
            "toy_hlt": {
                "artifacts": {
                    "rom": {
                        "states": [[0.0, 0.2, 0.3], [0.1, 0.1, 0.4]],
                        "observations": [[1.0, 1.1, 1.3], [0.4, 0.5, 0.7]],
                        "shocks": [[0.0, 0.1, -0.1]],
                    },
                    "gate": {
                        "e_stat": [0.0, 2.0, 0.1],
                        "f_stat": [0.1, 0.3, 1.1],
                        "gate_probs": [0.0001, 0.95, 0.7],
                        "hard_mask": [False, True, True],
                    },
                    "support": {
                        "features": [
                            [0.0, 0.2],
                            [0.1, 0.4],
                            [0.0, -0.1],
                            [1.5, 1.5],
                        ],
                        "selected_indices": [1, 2],
                    },
                    "sep": {
                        "targets": [[1.11, 1.29 + perturb], [0.52, 0.69]],
                        "rom_targets": [[1.10, 1.30], [0.50, 0.70]],
                        "residual_labels": [[0.01, -0.01 + perturb], [0.02, -0.01]],
                    },
                    "surrogate": {
                        "predictions": [[0.010001, -0.010001], [0.020001, -0.010001]],
                    },
                    "likelihood": {
                        "switching_per_period": [-1.0, -2.0, -1.5],
                    },
                    "posterior": {
                        "log_density": -12.5 + perturb,
                    },
                }
            }
        }
    }


def test_hlt_pipeline_parity_validator_accepts_full_matching_artifacts() -> None:
    module = _load_harness()
    checks = module.validate_hlt_pipeline_parity(
        julia_payload=_artifact_payload(),
        python_payload=_artifact_payload(perturb=1.0e-8),
    )

    assert len(checks) == 1
    assert checks[0].case == "toy_hlt"
    assert checks[0].passed
    assert {stage.stage for stage in checks[0].stage_checks} == {
        "rom_state_path",
        "rom_observation_path",
        "rom_shock_path",
        "gate_e_stat",
        "gate_f_stat",
        "gate_probs",
        "gate_mask",
        "support_features",
        "support_selected_indices",
        "sep_targets",
        "rom_targets",
        "residual_labels",
        "surrogate_predictions",
        "switching_loglik_per_period",
        "posterior_log_density",
    }


def test_hlt_pipeline_parity_validator_rejects_gate_mask_mismatch() -> None:
    module = _load_harness()
    python_payload = _artifact_payload()
    python_payload["cases"]["toy_hlt"]["artifacts"]["gate"]["hard_mask"] = [False, False, True]

    with pytest.raises(AssertionError, match="gate_mask"):
        module.validate_hlt_pipeline_parity(
            julia_payload=_artifact_payload(),
            python_payload=python_payload,
        )


def test_hlt_pipeline_parity_validator_rejects_missing_required_stage() -> None:
    module = _load_harness()
    python_payload = _artifact_payload()
    del python_payload["cases"]["toy_hlt"]["artifacts"]["sep"]["residual_labels"]

    with pytest.raises(module.PipelineParityPayloadError) as exc_info:
        module.validate_hlt_pipeline_parity(
            julia_payload=_artifact_payload(),
            python_payload=python_payload,
        )

    assert "residual_labels" in str(exc_info.value)


def test_hlt_pipeline_parity_cli_writes_json_and_report(tmp_path: Path) -> None:
    module = _load_harness()
    (tmp_path / "julia_pipeline_artifacts.json").write_text(
        json.dumps(_artifact_payload()),
        encoding="utf-8",
    )
    (tmp_path / "python_pipeline_artifacts.json").write_text(
        json.dumps(_artifact_payload(perturb=1.0e-8)),
        encoding="utf-8",
    )

    checks = module.validate_hlt_pipeline_parity_files(
        julia_file=tmp_path / "julia_pipeline_artifacts.json",
        python_file=tmp_path / "python_pipeline_artifacts.json",
    )
    report = module.checks_to_markdown(checks)

    assert "HLT Pipeline Parity Report" in report
    assert "toy_hlt" in report
    assert module.checks_to_jsonable(checks)[0]["passed"]


def test_hlt_pipeline_parity_file_validator_can_restrict_stages(tmp_path: Path) -> None:
    module = _load_harness()
    payload = {
        "cases": {
            "toy_hlt": {
                "artifacts": {
                    "support": {
                        "features": [[0.0, 0.2], [0.1, 0.4]],
                        "selected_indices": [1, 2],
                    },
                    "sep": {
                        "targets": [[1.11, 1.29], [0.52, 0.69]],
                        "rom_targets": [[1.10, 1.30], [0.50, 0.70]],
                        "residual_labels": [[0.01, -0.01], [0.02, -0.01]],
                    },
                }
            }
        }
    }
    julia_file = tmp_path / "julia_pipeline_artifacts.json"
    python_file = tmp_path / "python_pipeline_artifacts.json"
    julia_file.write_text(json.dumps(payload), encoding="utf-8")
    python_file.write_text(json.dumps(payload), encoding="utf-8")

    checks = module.validate_hlt_pipeline_parity_files(
        julia_file=julia_file,
        python_file=python_file,
        stages=("support_features", "support_selected_indices", "sep_targets", "rom_targets", "residual_labels"),
    )

    assert checks[0].passed
    assert {stage.stage for stage in checks[0].stage_checks} == {
        "support_features",
        "support_selected_indices",
        "sep_targets",
        "rom_targets",
        "residual_labels",
    }


def test_julia_hlt_artifact_exporter_writes_canonical_dataset_payload(tmp_path: Path) -> None:
    julia = shutil.which("julia")
    if julia is None:
        pytest.skip("Julia executable not available.")
    root = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "toy_hlt_dataset.jls"
    output_path = tmp_path / "julia_pipeline_artifacts.json"
    make_dataset = tmp_path / "make_dataset.jl"
    make_dataset.write_text(
        """
using Serialization
X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
Y = [10.0 20.0; 30.0 40.0]
Y_rom1 = [9.0 18.0; 31.0 43.0]
meta = Dict(
    "source" => "toy",
    "selected_global_periods" => [7, 8],
    "theta_names" => ["rho"],
)
serialize(ARGS[1], Dict(
    "X" => X,
    "Y" => Y,
    "Y_rom1" => Y_rom1,
    "sep_residuals" => [0.1, 0.2],
    "meta" => meta,
))
""",
        encoding="utf-8",
    )
    subprocess.run(
        [julia, "--startup-file=no", str(make_dataset), str(dataset_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            julia,
            "--startup-file=no",
            str(root / "benchmarks" / "export_julia_hlt_pipeline_artifacts.jl"),
            "--dataset",
            str(dataset_path),
            "--out",
            str(output_path),
            "--case",
            "toy_hlt",
            "--max-columns",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    artifacts = payload["cases"]["toy_hlt"]["artifacts"]
    assert artifacts["support"]["features"] == [[1.0], [3.0], [5.0]]
    assert artifacts["support"]["selected_indices"] == [7]
    assert artifacts["sep"]["targets"] == [[10.0], [30.0]]
    assert artifacts["sep"]["rom_targets"] == [[9.0], [31.0]]
    assert artifacts["sep"]["residual_labels"] == [[1.0], [-1.0]]
    assert artifacts["diagnostics"]["sep_residuals"] == [0.1]
