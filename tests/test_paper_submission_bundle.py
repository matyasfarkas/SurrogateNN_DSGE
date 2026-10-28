from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
import zipfile

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "paper" / "build_submission_bundle.py"
    spec = importlib.util.spec_from_file_location("build_submission_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_paper(root: Path) -> None:
    files = {
        "LICENSE": "license\n",
        "docs/submission_readiness_plan.md": "plan\n",
        "paper/README.md": "readme\n",
        "paper/SUBMISSION_CHECKLIST.md": "checklist\n",
        "paper/build_submission_bundle.py": "builder\n",
        "paper/main.pdf": "%PDF-test\n",
        "paper/main.tex": "\\documentclass{article}\n",
        "paper/references.bib": "@misc{x}\n",
        "paper/generated/evidence.json": "{}\n",
        "paper/generated/results.tex": "table\n",
        "paper/tables/submission_gates.tex": "gates\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_bundle_is_deterministic_and_hash_verified(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_paper(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    manifest = module.build_bundle(
        tmp_path,
        first,
        require_clean=False,
        git_revision="a" * 40,
    )
    module.build_bundle(
        tmp_path,
        second,
        require_clean=False,
        git_revision="a" * 40,
    )

    assert first.read_bytes() == second.read_bytes()
    assert module.verify_bundle(first) == manifest
    assert manifest["worktree_clean"] is True
    assert manifest["claim_scope"].startswith("validation architecture")
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist()[0] == "submission_manifest.json"
        stored = json.loads(archive.read("submission_manifest.json"))
        assert stored == manifest


def test_verify_rejects_unmanifested_archive_member(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_paper(tmp_path)
    bundle = tmp_path / "bundle.zip"
    module.build_bundle(
        tmp_path,
        bundle,
        require_clean=False,
        git_revision="b" * 40,
    )
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("unexpected.txt", "unexpected")

    with pytest.raises(ValueError, match="membership"):
        module.verify_bundle(bundle)


def test_bundle_requires_all_core_submission_files(tmp_path: Path) -> None:
    module = _load_module()
    _write_minimal_paper(tmp_path)
    (tmp_path / "paper" / "main.pdf").unlink()

    with pytest.raises(ValueError, match="paper/main.pdf"):
        module.build_bundle(
            tmp_path,
            tmp_path / "bundle.zip",
            require_clean=False,
            git_revision="c" * 40,
        )
