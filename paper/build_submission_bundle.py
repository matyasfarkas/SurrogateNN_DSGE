#!/usr/bin/env python3
"""Build and verify the conference-neutral paper submission bundle.

The archive contains the compiled paper, its self-contained LaTeX sources, and
the machine-readable evidence that generates every numerical table.  It is a
submission bundle, not a replacement for the repository-level replication
package or for conference-specific author and template checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence
import zipfile


SCHEMA_VERSION = "surrogatenn-dsge-paper-submission/v1"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED_FILES = (
    "LICENSE",
    "docs/end_to_end_pipeline_validation.md",
    "docs/submission_readiness_plan.md",
    "paper/README.md",
    "paper/SUBMISSION_CHECKLIST.md",
    "paper/build_submission_bundle.py",
    "paper/main.pdf",
    "paper/main.tex",
    "paper/references.bib",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status.strip())


def _collect_files(repo_root: Path) -> list[Path]:
    paths = [repo_root / relative for relative in REQUIRED_FILES]
    paths.extend(sorted((repo_root / "paper" / "generated").glob("*")))
    paths.extend(sorted((repo_root / "paper" / "tables").glob("*.tex")))
    paths = sorted({path.resolve() for path in paths if path.is_file()})

    missing = [relative for relative in REQUIRED_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise ValueError(f"Missing required submission files: {', '.join(missing)}")
    if not any(path.parent.name == "generated" for path in paths):
        raise ValueError("The submission bundle has no generated evidence files.")
    if not any(path.parent.name == "tables" for path in paths):
        raise ValueError("The submission bundle has no paper table sources.")
    return paths


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(
    repo_root: Path,
    output: Path,
    *,
    require_clean: bool = True,
    git_revision: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic archive and return its verified manifest."""

    repo_root = repo_root.resolve()
    output = output.resolve()
    if git_revision is None:
        git_revision, clean = _git_state(repo_root)
    else:
        clean = True
    if require_clean and not clean:
        raise ValueError("Refusing to build a final submission bundle from a dirty worktree.")

    entries: list[tuple[str, bytes]] = []
    for path in _collect_files(repo_root):
        name = path.relative_to(repo_root).as_posix()
        entries.append((name, path.read_bytes()))

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "conference_neutral_source_and_pdf_bundle",
        "title": "Auditable Residual Surrogates for Nonlinear DSGE Estimation",
        "git_revision": git_revision,
        "worktree_clean": clean,
        "claim_scope": (
            "validation architecture and reduced-horizon algorithmic parity; "
            "not production posterior fidelity or end-to-end GPU speedup"
        ),
        "files": [
            {"path": name, "size_bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in entries
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(_zip_info("submission_manifest.json"), manifest_bytes)
        for name, data in entries:
            archive.writestr(_zip_info(name), data)

    verified = verify_bundle(output)
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(
        f"{_sha256_bytes(output.read_bytes())}  {output.name}\n", encoding="utf-8"
    )
    return verified


def verify_bundle(path: Path) -> dict[str, Any]:
    """Verify archive membership, file sizes, and all recorded hashes."""

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Submission bundle contains duplicate paths.")
        if "submission_manifest.json" not in names:
            raise ValueError("Submission bundle has no manifest.")
        manifest = json.loads(archive.read("submission_manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Submission manifest has an unsupported schema version.")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("Submission manifest has no file records.")
        expected = {"submission_manifest.json"}
        for record in records:
            name = record.get("path")
            if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path in submission manifest: {name!r}")
            data = archive.read(name)
            expected.add(name)
            if len(data) != record.get("size_bytes"):
                raise ValueError(f"Size mismatch for {name}.")
            if _sha256_bytes(data) != record.get("sha256"):
                raise ValueError(f"SHA-256 mismatch for {name}.")
        if set(names) != expected:
            raise ValueError("Archive membership does not match the submission manifest.")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/auditable_residual_surrogates_submission.zip"),
        help="Output ZIP path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a diagnostic bundle from a dirty worktree.",
    )
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    manifest = build_bundle(
        args.repo_root,
        output,
        require_clean=not args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "sha256_file": str(output.resolve().with_suffix(output.suffix + ".sha256")),
                "git_revision": manifest["git_revision"],
                "file_count": len(manifest["files"]),
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
