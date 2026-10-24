from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "publication_run_manifest.py"
    spec = importlib.util.spec_from_file_location("publication_run_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "manifest-test@example.invalid")
    _git(repo, "config", "user.name", "Manifest Test")
    (repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    (repo / "protocol.json").write_text('{"seed": 17}\n', encoding="utf-8")
    _git(repo, "add", ".gitignore", "protocol.json")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_manifest_records_clean_revision_command_inputs_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repo = _clean_repo(tmp_path)
    monkeypatch.setattr(
        module,
        "_collect_environment",
        lambda: {"sha256": "e" * 64, "float64_enabled": True},
    )

    prepared = module.build_pre_manifest(
        repo_root=repo,
        command=["python", "benchmark.py", "--seed", "17"],
        inputs=[Path("protocol.json")],
        seeds={"design": 17},
        allow_dirty=False,
    )

    assert prepared["status"] == "prepared"
    assert prepared["repository"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert prepared["repository"]["dirty_worktree"] is False
    assert prepared["command"]["argv"] == ["python", "benchmark.py", "--seed", "17"]
    assert prepared["seeds"] == {"design": 17}
    assert prepared["inputs"][0]["path"] == "protocol.json"
    assert len(prepared["input_sha256"]) == 64

    result = repo / "results" / "result.json"
    result.parent.mkdir()
    result.write_text('{"status": "ok"}\n', encoding="utf-8")
    finalized = module.finalize_manifest(
        prepared,
        repo_root=repo,
        exit_code=0,
        artifacts=[Path("results/result.json"), Path("results/missing.npz")],
    )

    assert finalized["status"] == "completed"
    assert finalized["exit_code"] == 0
    assert finalized["artifacts"][0]["sha256"] == module._sha256_file(result)
    assert finalized["artifacts"][1]["kind"] == "missing"
    assert len(finalized["artifact_sha256"]) == 64
    assert len(finalized["canonical_payload_sha256"]) == 64


def test_manifest_refuses_dirty_preparation_and_repository_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    repo = _clean_repo(tmp_path)
    monkeypatch.setattr(module, "_collect_environment", lambda: {"sha256": "e" * 64})

    (repo / "protocol.json").write_text('{"seed": 18}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean worktree"):
        module.build_pre_manifest(
            repo_root=repo,
            command=["python", "benchmark.py"],
            inputs=[Path("protocol.json")],
            seeds={},
            allow_dirty=False,
        )

    _git(repo, "checkout", "--", "protocol.json")
    prepared = module.build_pre_manifest(
        repo_root=repo,
        command=["python", "benchmark.py"],
        inputs=[Path("protocol.json")],
        seeds={},
        allow_dirty=False,
    )
    (repo / "protocol.json").write_text('{"seed": 19}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty state changed"):
        module.finalize_manifest(
            prepared,
            repo_root=repo,
            exit_code=1,
            artifacts=[],
        )


def test_manifest_cli_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    repo = _clean_repo(tmp_path)
    monkeypatch.setattr(module, "_collect_environment", lambda: {"sha256": "e" * 64})
    manifest_path = repo / "results" / "run_manifest.json"

    assert module.main(
        [
            "pre",
            "--output",
            str(manifest_path),
            "--repo-root",
            str(repo),
            "--input",
            "protocol.json",
            "--seed",
            "design=23",
            "--command",
            "python",
            "benchmark.py",
        ]
    ) == 0
    assert module.main(
        [
            "finalize",
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(repo),
            "--exit-code",
            "3",
            "--artifact",
            "results/result.json",
        ]
    ) == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 3
    assert payload["seeds"] == {"design": 23}


def test_duplicate_seed_labels_are_rejected() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Duplicate seed"):
        module._parse_seeds(["design=1", "design=2"])


def test_gali_pilot_dry_run_uses_stochastic_sep_and_explicit_seeds(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "DRY_RUN": "1",
            "RESULT_ROOT": str(tmp_path / "results"),
            "PYTHON": ".venv/bin/python",
        }
    )

    completed = subprocess.run(
        ["bash", "benchmarks/run_nonlinear_gali_pilot.sh"],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sep-order 1" in completed.stdout
    assert "--sep-nnodes 3" in completed.stdout
    assert "--sep-periods 15" in completed.stdout
    assert completed.stdout.startswith("JAX_ENABLE_X64=1 ")
    assert "--no-calibrate-residual-shrinkage" in completed.stdout
    assert "--hlt-surrogate-hmc-seed 20261023" in completed.stdout
    assert "--seed 20261019" in completed.stdout
