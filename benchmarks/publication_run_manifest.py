#!/usr/bin/env python3
"""Create and finalize a content-addressed manifest for confirmatory runs.

The ``pre`` command records the clean Git revision, exact argv vector, declared
seeds, hashed inputs, and the numerical environment before a run starts.  The
``finalize`` command adds the exit code and hashes every declared artifact while
verifying that the repository did not change during execution.

Examples
--------
``python benchmarks/publication_run_manifest.py pre --output results/run_manifest.json \
  --input benchmarks/model.jl --seed design=123 --command python benchmark.py``

``python benchmarks/publication_run_manifest.py finalize \
  --manifest results/run_manifest.json --exit-code 0 --artifact results/result.json``
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-publication-run-manifest/v1"
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "NVIDIA_VISIBLE_DEVICES",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)
PACKAGE_NAMES = ("jax", "jaxlib", "numpy", "numpyro", "scipy", "sympy")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _path_record(path: Path, repo_root: Path, *, require_exists: bool) -> dict[str, Any]:
    unresolved = path if path.is_absolute() else repo_root / path
    if unresolved.is_symlink():
        raise ValueError(f"Symlink inputs and artifacts are not accepted: {unresolved}")
    resolved = unresolved.resolve()
    display = _display_path(resolved, repo_root)
    if not resolved.exists():
        if require_exists:
            raise FileNotFoundError(f"Declared input does not exist: {display}")
        return {"path": display, "kind": "missing", "exists": False}
    if resolved.is_file():
        return {
            "path": display,
            "kind": "file",
            "exists": True,
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    if not resolved.is_dir():
        raise ValueError(f"Unsupported path type: {display}")

    files: list[dict[str, Any]] = []
    total_size = 0
    for member in sorted(resolved.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"Symlink inside declared directory: {member}")
        if not member.is_file():
            continue
        size = member.stat().st_size
        total_size += size
        files.append(
            {
                "path": member.relative_to(resolved).as_posix(),
                "size_bytes": size,
                "sha256": _sha256_file(member),
            }
        )
    return {
        "path": display,
        "kind": "directory",
        "exists": True,
        "file_count": len(files),
        "size_bytes": total_size,
        "tree_sha256": _canonical_sha256({"files": files}),
        "files": files,
    }


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _collect_repository(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "root": str(repo_root),
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "dirty_worktree": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_distributions() -> list[dict[str, str]]:
    """Return a credential-free package inventory (names and versions only)."""
    rows: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        rows.append({"name": str(name), "version": str(distribution.version)})
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"]))


def _collect_jax_runtime() -> dict[str, Any]:
    try:
        import jax

        return {
            "available": True,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "float64_enabled": bool(jax.config.read("jax_enable_x64")),
        }
    except Exception as exc:  # pragma: no cover - depends on host accelerator setup
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _collect_nvidia_runtime() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "query": command[1:],
        "rows": [line for line in completed.stdout.splitlines() if line.strip()],
        "stderr": completed.stderr.strip(),
    }


def _collect_environment() -> dict[str, Any]:
    environment = {
        key: os.environ[key]
        for key in SAFE_ENVIRONMENT_KEYS
        if key in os.environ
    }
    payload = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "installed_distributions": _installed_distributions(),
        "jax_runtime": _collect_jax_runtime(),
        "nvidia_runtime": _collect_nvidia_runtime(),
        "allowlisted_environment": environment,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _parse_seeds(values: Sequence[str]) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for value in values:
        name, separator, raw_seed = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Seed must have NAME=INTEGER form, got {value!r}")
        name = name.strip()
        if name in seeds:
            raise ValueError(f"Duplicate seed label {name!r}")
        seeds[name] = int(raw_seed)
    return seeds


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_pre_manifest(
    *,
    repo_root: Path,
    command: Sequence[str],
    inputs: Sequence[Path],
    seeds: Mapping[str, int],
    allow_dirty: bool,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    repository = _collect_repository(repo_root)
    if repository["dirty_worktree"] and not allow_dirty:
        changed = ", ".join(repository["status_porcelain"][:5])
        raise RuntimeError(
            "Confirmatory runs require a clean worktree; commit or remove changes first"
            + (f" ({changed})" if changed else "")
        )
    if not command:
        raise ValueError("The exact command argv must be recorded")
    input_records = [
        _path_record(path, repo_root, require_exists=True)
        for path in inputs
    ]
    input_digest = _canonical_sha256({"inputs": input_records})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at_utc": _utc_now(),
        "repository": repository,
        "command": {
            "argv": list(command),
            "shell_escaped": shlex.join(command),
            "working_directory": str(repo_root),
        },
        "seeds": dict(seeds),
        "inputs": input_records,
        "input_sha256": input_digest,
        "environment": _collect_environment(),
    }


def finalize_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    exit_code: int,
    artifacts: Sequence[Path],
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing publication-run manifest schema")
    current_repository = _collect_repository(repo_root.resolve())
    prepared_repository = manifest.get("repository")
    if not isinstance(prepared_repository, Mapping):
        raise ValueError("Prepared manifest is missing repository metadata")
    if current_repository["commit"] != prepared_repository.get("commit"):
        raise RuntimeError("Repository commit changed during the run")
    if current_repository["dirty_worktree"] != prepared_repository.get("dirty_worktree"):
        raise RuntimeError("Repository dirty state changed during the run")
    if current_repository["status_porcelain"] != prepared_repository.get("status_porcelain"):
        raise RuntimeError("Repository file status changed during the run")

    payload = dict(manifest)
    artifact_records = [
        _path_record(path, repo_root.resolve(), require_exists=False)
        for path in artifacts
    ]
    payload.update(
        {
            "status": "completed" if exit_code == 0 else "failed",
            "completed_at_utc": _utc_now(),
            "exit_code": int(exit_code),
            "repository_after": current_repository,
            "artifacts": artifact_records,
            "artifact_sha256": _canonical_sha256({"artifacts": artifact_records}),
        }
    )
    canonical_payload = dict(payload)
    canonical_payload.pop("canonical_payload_sha256", None)
    payload["canonical_payload_sha256"] = _canonical_sha256(canonical_payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    pre = subparsers.add_parser("pre", help="Capture pre-run provenance.")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    pre.add_argument("--input", type=Path, action="append", default=[])
    pre.add_argument("--seed", action="append", default=[])
    pre.add_argument("--allow-dirty", action="store_true")
    pre.add_argument("--command", nargs=argparse.REMAINDER, required=True)

    final = subparsers.add_parser("finalize", help="Hash artifacts and close a manifest.")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    final.add_argument("--exit-code", type=int, required=True)
    final.add_argument("--artifact", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.action == "pre":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        payload = build_pre_manifest(
            repo_root=args.repo_root,
            command=command,
            inputs=args.input,
            seeds=_parse_seeds(args.seed),
            allow_dirty=bool(args.allow_dirty),
        )
        _write_json_atomic(args.output, payload)
        print(f"PREPARED manifest={args.output} commit={payload['repository']['commit']}")
        return 0

    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Manifest root must be an object")
    payload = finalize_manifest(
        raw,
        repo_root=args.repo_root,
        exit_code=args.exit_code,
        artifacts=args.artifact,
    )
    _write_json_atomic(args.manifest, payload)
    print(
        f"{payload['status'].upper()} manifest={args.manifest} "
        f"artifacts={len(payload['artifacts'])}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
