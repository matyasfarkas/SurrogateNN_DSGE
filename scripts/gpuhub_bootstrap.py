#!/usr/bin/env python3
"""Bootstrap and run the GPUHub SW07 posterior ESS benchmark.

This script is intentionally conservative: it removes stale JAX installs,
selects a CUDA wheel from the driver reported by ``nvidia-smi``, verifies that
JAX really sees a GPU, then runs the posterior ESS benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPO_URL = "https://github.com/matyasfarkas/SurrogateNN_DSGE.git"
DEFAULT_BRANCH = "codex/colab-jax-gemini-profile"
DEFAULT_DATA_ROOT = (
    Path(os.environ["GPUHUB_DATA_DIR"])
    if "GPUHUB_DATA_DIR" in os.environ
    else Path("/root/autodl-tmp")
    if Path("/root/autodl-tmp").exists()
    else Path("/root/gpuhub-tmp")
)
DEFAULT_ROOT = DEFAULT_DATA_ROOT / "SurrogateNN_DSGE"


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print("$", " ".join(map(str, cmd)), flush=True)
    if stream:
        process = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
        )
        stdout_parts: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            stdout_parts.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        stdout = "".join(stdout_parts)
        completed = subprocess.CompletedProcess(
            args=list(map(str, cmd)),
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )
        if check and returncode:
            raise subprocess.CalledProcessError(
                returncode,
                completed.args,
                output=stdout,
                stderr="",
            )
        return completed
    completed = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=True,
        env=full_env,
    )
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    return completed


def parse_nvidia_smi(text: str) -> dict[str, Any]:
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    memory_match = re.search(r"([0-9]+)MiB\s*/\s*([0-9]+)MiB", text)
    gpu_match = re.search(r"\|\s+\d+\s+([^|]+?)\s{2,}", text)
    return {
        "driver_version": driver_match.group(1) if driver_match else None,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
        "gpu_name": gpu_match.group(1).strip() if gpu_match else None,
        "memory_total_mib": int(memory_match.group(2)) if memory_match else None,
    }


def _version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    parts = []
    for part in version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def choose_jax_requirement(
    requested: str,
    *,
    cuda_version: str | None,
    driver_version: str | None,
) -> str:
    if requested == "cpu":
        return "jax>=0.6"
    if requested == "cuda12":
        return "jax[cuda12]>=0.6"
    if requested == "cuda13":
        return "jax[cuda13]>=0.6"
    if requested != "auto":
        raise ValueError(f"Unknown JAX backend request: {requested!r}.")

    cuda = _version_tuple(cuda_version)
    driver = _version_tuple(driver_version)
    if cuda and cuda[0] >= 13 and driver >= (580,):
        return "jax[cuda13]>=0.6"
    if cuda and cuda[0] >= 12 and driver >= (525,):
        return "jax[cuda12]>=0.6"
    raise RuntimeError(
        "Could not select a modern CUDA JAX wheel from nvidia-smi. "
        f"Driver={driver_version!r}, CUDA={cuda_version!r}. For this benchmark, "
        "RTX 5090 should expose a recent driver with CUDA 12 or CUDA 13 support. "
        "A legacy JAX 0.3.10 / CUDA 11.1 stack is not usable for this repo."
    )


def benchmark_settings(mode: str) -> dict[str, Any]:
    if mode == "calibration":
        return {
            "periods": 80,
            "parameters": "calfa,cg,cgy,crdy,crhob,crpi",
            "warmup": 16,
            "samples": 16,
            "chains": 2,
            "schur_support_draws": 16,
            "output_suffix": "calibration",
        }
    if mode == "proper_5090":
        return {
            "periods": 160,
            "parameters": "sw07_safe_15",
            "warmup": 256,
            "samples": 256,
            "chains": 4,
            "schur_support_draws": 64,
            "output_suffix": "proper_5090",
        }
    raise ValueError(f"Unsupported benchmark mode {mode!r}.")


def build_benchmark_command(
    *,
    python: str,
    root: Path,
    mode: str,
    dtype: str,
    qme_algorithm: str,
    force_gpu: bool,
    preflight_reps: int,
    progress_bar: bool,
    verbose: bool,
    heartbeat_seconds: float,
) -> tuple[list[str], Path]:
    settings = benchmark_settings(mode)
    output = (
        root
        / "benchmarks"
        / "results"
        / f"gpuhub_sw07_posterior_ess_{settings['output_suffix']}_{dtype}_{qme_algorithm}.json"
    )
    cmd = [
        python,
        "benchmarks/posterior_sampling_speed.py",
        "--preset",
        "sw07_hlt",
        "--periods",
        str(settings["periods"]),
        "--parameters",
        settings["parameters"],
        "--warmup",
        str(settings["warmup"]),
        "--samples",
        str(settings["samples"]),
        "--chains",
        str(settings["chains"]),
        "--chain-method",
        "vectorized",
        "--dtype",
        dtype,
        "--qme-algorithm",
        qme_algorithm,
        "--target-accept-prob",
        "0.8",
        "--max-tree-depth",
        "8",
        "--prior-width-scale",
        "0.0025",
        "--prior-width-floor",
        "0.0001",
        "--preflight",
        "--preflight-reps",
        str(preflight_reps),
        "--schur-support-draws",
        str(settings["schur_support_draws"]),
        "--output",
        str(output),
    ]
    if verbose:
        cmd.append("--verbose")
        cmd.extend(["--heartbeat-seconds", str(heartbeat_seconds)])
    if force_gpu:
        cmd.append("--force-gpu")
    if progress_bar:
        cmd.append("--progress-bar")
    return cmd, output


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Python 3.10+ is required for the modern JAX/NumPyro GPU stack. "
            f"Current interpreter is {sys.version.split()[0]}. Choose a newer "
            "GPUHub image/kernel or create a Python 3.10+ conda environment."
        )


def sync_repo(root: Path, repo_url: str, branch: str) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        run(["git", "fetch", "origin", branch], cwd=root)
        run(["git", "checkout", branch], cwd=root)
        run(["git", "reset", "--hard", f"origin/{branch}"], cwd=root)
    else:
        run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(root)])


def install_stack(root: Path, jax_requirement: str, *, force_reinstall_jax: bool) -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    if force_reinstall_jax:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "jax",
                "jaxlib",
                "jax-cuda12-plugin",
                "jax-cuda12-pjrt",
                "jax-cuda13-plugin",
                "jax-cuda13-pjrt",
            ],
            check=False,
        )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "numpy>=2.1,<2.3",
            jax_requirement,
            "numpyro>=0.20",
            "scipy>=1.14,<2",
            "sympy>=1.13,<2",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps"])
    run(
        [sys.executable, "-m", "pip", "show", "jax", "jaxlib", "numpyro", "numpy", "scipy"],
        check=False,
    )


def sanitize_jax_runtime_environment(jax_requirement: str) -> dict[str, str | None]:
    """Remove CUDA-library overrides when using JAX's pip-bundled CUDA wheels."""

    updates: dict[str, str | None] = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "false",
        ),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "0.85",
        ),
    }
    if "cuda" in jax_requirement:
        updates["LD_LIBRARY_PATH"] = None
    return updates


def apply_environment_updates(updates: dict[str, str | None]) -> None:
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def probe_gpu(*, force_gpu: bool, dtype: str) -> None:
    probe = f"""
import jax
import jax.numpy as jnp
import numpyro
jax.config.update("jax_enable_x64", {dtype == "float64"!r})
print("JAX", jax.__version__)
print("NumPyro", numpyro.__version__)
print("x64", jax.config.read("jax_enable_x64"))
print("backend", jax.default_backend())
print("devices", jax.devices())
x = (jnp.ones((1024, 1024), dtype=jnp.float32) @ jnp.ones((1024, 1024), dtype=jnp.float32)).block_until_ready()
print("matmul dtype", x.dtype, "value", float(x[0, 0]))
if {force_gpu!r} and jax.default_backend() != "gpu":
    raise RuntimeError("Expected GPU backend, but JAX default backend is " + jax.default_backend())
"""
    run([sys.executable, "-c", probe])


def support_audit_ok(output: Path) -> bool:
    payload = json.loads(output.read_text())
    audit = payload.get("schur_support_audit") or {}
    return int(audit.get("doubling_accepts_non_unique_count") or 0) == 0


def run_benchmark(
    root: Path,
    *,
    mode: str,
    dtype: str,
    qme_algorithm: str,
    force_gpu: bool,
    preflight_reps: int,
    progress_bar: bool,
    verbose: bool,
    heartbeat_seconds: float,
) -> Path:
    cmd, output = build_benchmark_command(
        python=sys.executable,
        root=root,
        mode=mode,
        dtype=dtype,
        qme_algorithm=qme_algorithm,
        force_gpu=force_gpu,
        preflight_reps=preflight_reps,
        progress_bar=progress_bar,
        verbose=verbose,
        heartbeat_seconds=heartbeat_seconds,
    )
    started = time.perf_counter()
    run(cmd, cwd=root, stream=True)
    print(f"{mode} wall seconds: {time.perf_counter() - started:.3f}", flush=True)
    print(f"Output: {output}", flush=True)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("setup", "calibration", "proper_5090", "both"),
        default="setup",
    )
    parser.add_argument("--skip-repo-sync", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--jax-extra",
        choices=("auto", "cuda13", "cuda12", "cpu"),
        default="auto",
        help="Use auto unless debugging a specific JAX wheel family.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--qme-algorithm", choices=("doubling", "schur"), default="doubling")
    parser.add_argument("--preflight-reps", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--quiet-benchmark",
        action="store_false",
        dest="verbose_benchmark",
        help="Disable verbose posterior_sampling_speed.py stage logs.",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-force-reinstall-jax", action="store_true")
    parser.add_argument("--no-strict-support", action="store_true")
    parser.set_defaults(verbose_benchmark=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_python_version()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    smi = run(["nvidia-smi"], check=False)
    smi_info = parse_nvidia_smi((smi.stdout or "") + (smi.stderr or ""))
    print("nvidia-smi parsed:", json.dumps(smi_info, indent=2), flush=True)
    jax_requirement = choose_jax_requirement(
        "cpu" if args.allow_cpu else args.jax_extra,
        cuda_version=smi_info.get("cuda_version"),
        driver_version=smi_info.get("driver_version"),
    )
    print(f"Selected JAX requirement: {jax_requirement}", flush=True)
    apply_environment_updates(sanitize_jax_runtime_environment(jax_requirement))

    if not args.skip_repo_sync:
        sync_repo(args.root, args.repo_url, args.branch)
    if not args.skip_install:
        install_stack(
            args.root,
            jax_requirement,
            force_reinstall_jax=not args.no_force_reinstall_jax,
        )
    probe_gpu(force_gpu=not args.allow_cpu, dtype=args.dtype)

    if args.mode == "setup":
        return
    if args.mode in {"calibration", "proper_5090"}:
        run_benchmark(
            args.root,
            mode=args.mode,
            dtype=args.dtype,
            qme_algorithm=args.qme_algorithm,
            force_gpu=not args.allow_cpu,
            preflight_reps=args.preflight_reps,
            progress_bar=args.progress_bar,
            verbose=bool(args.verbose_benchmark),
            heartbeat_seconds=float(args.heartbeat_seconds),
        )
        return

    calibration = run_benchmark(
        args.root,
        mode="calibration",
        dtype=args.dtype,
        qme_algorithm=args.qme_algorithm,
        force_gpu=not args.allow_cpu,
        preflight_reps=args.preflight_reps,
        progress_bar=args.progress_bar,
        verbose=bool(args.verbose_benchmark),
        heartbeat_seconds=float(args.heartbeat_seconds),
    )
    if not args.no_strict_support and not support_audit_ok(calibration):
        raise RuntimeError(
            "Calibration completed, but doubling accepted posterior draws outside "
            "Schur unique-stable support. Not starting the proper run."
        )
    run_benchmark(
        args.root,
        mode="proper_5090",
        dtype=args.dtype,
        qme_algorithm=args.qme_algorithm,
        force_gpu=not args.allow_cpu,
        preflight_reps=args.preflight_reps,
        progress_bar=args.progress_bar,
        verbose=bool(args.verbose_benchmark),
        heartbeat_seconds=float(args.heartbeat_seconds),
    )


if __name__ == "__main__":
    main()
