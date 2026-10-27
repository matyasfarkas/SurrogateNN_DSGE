from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "build_stochastic_horizon_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_stochastic_horizon_evidence", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preflight(root: Path, *, commit: str) -> None:
    root.mkdir()
    run = {
        "status": "ok",
        "model_name": "Gali_2015_chapter_3_nonlinear",
        "equation_count": 23,
        "parameter_count": 16,
        "pipeline_s": 10.5,
        "steady_state_solved_count": 8,
        "steady_state_fallback_count": 0,
        "sep_config": {
            "periods": 5,
            "branching_order": 1,
            "nnodes": 3,
            "sparse_tree": True,
            "shock_scale": 1.0,
            "linear_solver": "qr",
            "line_search_norm": "inf",
            "positive_state_indices": [],
            "tol": 1e-8,
            "accept_tol": 1e-8,
        },
        "quality_gate": {"status": "ok", "issues": []},
        "target_diagnostics": {
            "status": "ok",
            "failure_log": [],
            "theta_full_success_count": 8,
            "accepted_samples": 8,
            "attempted_residual_summary": {"max": 2e-10},
        },
    }
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps({"results": {"hlt_fixed_ss_smoke": run}}), encoding="utf-8"
    )
    log_path = root / "run.log"
    log_path.write_text("ok\n", encoding="utf-8")
    artifacts = [
        {
            "path": f"benchmarks/results/example/{path.name}",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (result_path, log_path)
    ]
    manifest = {
        "schema_version": "surrogatenn-dsge-publication-run-manifest/v1",
        "status": "completed",
        "exit_code": 0,
        "prepared_at_utc": "2026-10-27T09:00:00+00:00",
        "completed_at_utc": "2026-10-27T09:01:00+00:00",
        "repository": {
            "branch": "codex/test",
            "commit": commit,
            "dirty_worktree": False,
        },
        "repository_after": {"commit": commit, "dirty_worktree": False},
        "command": {
            "argv": [
                "python",
                "benchmarks/profile_surrogate_pipeline_gpu.py",
                "--hlt-target-builder",
                "batched-sep",
                "--hlt-theta-draws",
                "8",
                "--hlt-parameter-perturbation",
                "0.12",
                "--hlt-periods",
                "1",
                "--hlt-shock-scale",
                "0.12",
                "--hlt-require-full-target-success",
                "--sep-periods",
                "5",
                "--sep-order",
                "1",
                "--sep-nnodes",
                "3",
                "--sep-tol",
                "1e-8",
                "--sep-accept-tol",
                "1e-8",
                "--sep-line-search-norm",
                "inf",
                "--hlt-likelihood-periods",
                "0",
                "--hlt-surrogate-hmc-samples",
                "0",
            ]
        },
        "artifacts": artifacts,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _diagnostic_payload(
    *,
    commit: str,
    periods: int,
    solver: str,
    theta_indices: list[int],
) -> dict[str, Any]:
    draws = []
    for theta_index in theta_indices:
        first_step = {
            "log_y": 0.1 * theta_index + periods * 1e-3,
            "pi_ann": 0.2 * theta_index + periods * 2e-3,
            "i_ann": 0.3 * theta_index + periods * 3e-3,
            "other": 0.4 * theta_index + periods * 4e-3,
        }
        if periods == 6:
            draws.append(
                {
                    "theta_index": theta_index,
                    "theta": [1.32, 0.14],
                    "sequential": {
                        "accepted": False,
                        "residual_norm": 1.02e-4,
                        "iterations": 6,
                    },
                    "batched": {
                        "accepted": [False],
                        "converged": [False],
                        "residual_norm": [1.03e-4],
                        "iterations": [41],
                        "termination": ["line_search_failed"],
                        "mean_path_first_step": first_step,
                    },
                }
            )
        else:
            draws.append(
                {
                    "theta_index": theta_index,
                    "batched": {
                        "accepted": [True],
                        "converged": [True],
                        "residual_norm": [1e-10 + theta_index * 1e-12],
                        "iterations": [5 + theta_index % 2],
                        "mean_path_first_step": first_step,
                    },
                }
            )
    return {
        "status": "complete",
        "git": {"commit": commit, "dirty": False, "status_short": []},
        "command": {
            "argv": ["python", "benchmarks/diagnose_gali_sep_backend_parity.py"],
            "shell_escaped": "python benchmarks/diagnose_gali_sep_backend_parity.py",
            "working_directory": "/repo",
        },
        "inputs": {
            "model_source": {"sha256": "a" * 64},
            "payload": {"sha256": "b" * 64},
        },
        "model": {
            "name": "Gali_2015_chapter_3_nonlinear",
            "variables": 23,
            "parameters": 16,
        },
        "config": {
            "periods": periods,
            "branching_order": 1,
            "nnodes": 3,
            "sparse_tree": True,
            "shock_scale": 1.0,
            "linear_solver": "qr",
            "line_search_norm": "inf",
            "positive_state_indices": [],
            "tol": 1e-8,
            "accept_tol": 1e-8,
            "solver": solver,
        },
        "theta_indices": theta_indices,
        "draws": draws,
    }


def _write_sources(root: Path) -> tuple[Path, list[Path], Path]:
    commit = "c" * 40
    preflight = root / "preflight"
    _write_preflight(preflight, commit=commit)
    horizons = []
    for periods in (3, 4, 5):
        path = root / f"h{periods}.json"
        path.write_text(
            json.dumps(
                _diagnostic_payload(
                    commit=commit,
                    periods=periods,
                    solver="batched",
                    theta_indices=list(range(8)),
                )
            ),
            encoding="utf-8",
        )
        horizons.append(path)
    horizon_six = root / "h6.json"
    horizon_six.write_text(
        json.dumps(
            _diagnostic_payload(
                commit=commit,
                periods=6,
                solver="both",
                theta_indices=[1],
            )
        ),
        encoding="utf-8",
    )
    return preflight, horizons, horizon_six


def test_builder_preserves_reduced_horizon_claim_boundary(tmp_path: Path) -> None:
    module = _load_module()
    preflight, horizons, horizon_six = _write_sources(tmp_path)

    evidence = module.build_evidence(
        preflight,
        horizon_paths=horizons,
        horizon_six_path=horizon_six,
    )

    assert evidence["publication_ready"] is False
    assert evidence["target_preflight"]["accepted_targets"] == 8
    assert [row["periods"] for row in evidence["horizon_results"]] == [3, 4, 5]
    assert all(row["accepted"] == 8 for row in evidence["horizon_results"])
    comparison = evidence["first_step_sensitivity"][1]
    assert comparison["from_periods"] == 4
    assert comparison["to_periods"] == 5
    assert comparison["variables"]["log_y"]["max_abs"] == pytest.approx(1e-3)
    assert evidence["horizon_six_hard_case"]["sequential"]["accepted"] is False
    assert "does not establish terminal-horizon convergence" in evidence[
        "interpretation"
    ]["not_supported"]


def test_cli_writes_json_markdown_and_latex(tmp_path: Path) -> None:
    module = _load_module()
    preflight, horizons, horizon_six = _write_sources(tmp_path)
    output = tmp_path / "generated"
    argv = [str(preflight)]
    for path in horizons:
        argv.extend(["--horizon-result", str(path)])
    argv.extend(
        [
            "--horizon-six-result",
            str(horizon_six),
            "--output-dir",
            str(output),
        ]
    )

    assert module.main(argv) == 0

    evidence = json.loads(
        (output / "stochastic_horizon_evidence.json").read_text(encoding="utf-8")
    )
    markdown = (output / "stochastic_horizon_evidence.md").read_text(
        encoding="utf-8"
    )
    latex = (output / "stochastic_horizon_results.tex").read_text(encoding="utf-8")
    assert evidence["publication_ready"] is False
    assert "**Publication ready:** no" in markdown
    assert "Horizon-six boundary" in markdown
    assert "Reduced-horizon stochastic SEP evidence" in latex
    assert "not terminal-horizon convergence" in latex


def test_builder_rejects_tampered_preflight_artifact(tmp_path: Path) -> None:
    module = _load_module()
    preflight, horizons, horizon_six = _write_sources(tmp_path)
    (preflight / "run.log").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Hash mismatch"):
        module.build_evidence(
            preflight,
            horizon_paths=horizons,
            horizon_six_path=horizon_six,
        )


def test_builder_rejects_dirty_or_cross_revision_diagnostic(tmp_path: Path) -> None:
    module = _load_module()
    preflight, horizons, horizon_six = _write_sources(tmp_path)
    payload = json.loads(horizons[0].read_text(encoding="utf-8"))
    payload["git"]["dirty"] = True
    horizons[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="clean revision"):
        module.build_evidence(
            preflight,
            horizon_paths=horizons,
            horizon_six_path=horizon_six,
        )
