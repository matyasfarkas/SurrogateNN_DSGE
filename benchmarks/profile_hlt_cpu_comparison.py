#!/usr/bin/env python3
"""Run local CPU profiles for the HLT SEP/ResNN pipeline in Python and Julia.

The Python stage uses the integrated JAX benchmark harness.  The Julia stage
uses the real HLT surrogate dataset-generation and training scripts from the
companion Julia repository.  These are overlapping heavy stages, not a claim of
one-for-one posterior wrapper parity: the Julia repository currently exposes
the HLT SEP target generation and ResNN training as separate scripts, while the
Python harness also runs the fixed-reference surrogate likelihood/HMC wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JULIA_REPO = ROOT.parent / "SurrogateNN_Estimation.jl"
RESULTS_ROOT = ROOT / "benchmarks" / "results"

def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def progress(message: str) -> None:
    print(f"[cpu-profile] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_key_value_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty override key in {item!r}.")
        overrides[key] = value
    return overrides


def host_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    for key, cmd in {
        "mac_cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "mac_memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "mac_performance_cores": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
        "mac_efficiency_cores": ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
    }.items():
        try:
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        except OSError:
            continue
        if proc.returncode == 0:
            value = proc.stdout.strip()
            if value:
                payload[key] = value
    return payload


def run_stage(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(cmd)
    progress(f"stage={name} cwd={cwd}")
    progress(f"stage={name} command={command_text}")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "cwd": str(cwd),
        "command": cmd,
        "log_path": str(log_path),
        "start_utc": utc_stamp(),
    }
    if dry_run:
        result.update({"status": "dry_run", "returncode": 0, "elapsed_s": 0.0})
        log_path.write_text(command_text + "\n", encoding="utf-8")
        return result

    merged_env = os.environ.copy()
    merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_text}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = proc.wait()

    elapsed = time.perf_counter() - started
    status = "ok" if returncode == 0 else "failed"
    progress(f"stage={name} status={status} elapsed_s={elapsed:.3f} returncode={returncode}")
    result.update(
        {
            "status": status,
            "returncode": int(returncode),
            "elapsed_s": float(elapsed),
            "end_utc": utc_stamp(),
        }
    )
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return result


def read_python_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("results", {}).get("hlt_fixed_ss_smoke", {})
    hmc = result.get("surrogate_hmc", {})
    target = result.get("target_diagnostics", {})
    return {
        "status": result.get("status"),
        "backend": result.get("backend"),
        "parameter_count": len(result.get("parameter_subset", [])),
        "theta_draws": result.get("theta_draws"),
        "train_size": result.get("train_size"),
        "val_size": result.get("val_size"),
        "pipeline_s": result.get("pipeline_s"),
        "target_builder": target.get("builder"),
        "accepted_samples": target.get("accepted_samples"),
        "theta_full_success_count": target.get("theta_full_success_count"),
        "target_fallback_share": target.get("fallback_share"),
        "hmc_status": hmc.get("status"),
        "hmc_elapsed_s": hmc.get("elapsed_s"),
        "hmc_draws": hmc.get("post_warmup_draws"),
        "hmc_draws_per_second": hmc.get("draws_per_second"),
        "path": str(path),
    }


def parse_julia_log_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, Any] = {"path": str(path)}
    saved_dataset = re.findall(r"Saved dataset:\s*(.+)", text)
    saved_surrogate = re.findall(r"Saved surrogate:\s*(.+)", text)
    kept_match = re.findall(r"Filtering to fully successful samples:\s*kept\s+(\d+)\s*/\s*(\d+)", text)
    validation_match = re.findall(r"Validation RMSE \(per output dim\):", text)
    total_time_match = re.findall(r"total_time_sec[\"=>:\s]+([0-9.eE+-]+)", text)
    if saved_dataset:
        metrics["dataset_path"] = saved_dataset[-1].strip()
    if saved_surrogate:
        metrics["surrogate_path"] = saved_surrogate[-1].strip()
    if kept_match:
        kept, total = kept_match[-1]
        metrics["fully_successful_samples_kept"] = int(kept)
        metrics["fully_successful_samples_total"] = int(total)
    if validation_match:
        metrics["validation_reported"] = True
    if total_time_match:
        metrics["reported_total_time_sec"] = float(total_time_match[-1])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", default=f"hlt_cpu_comparison_{utc_stamp()}")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--julia-repo", type=Path, default=DEFAULT_JULIA_REPO)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--julia", default=shutil.which("julia") or "julia")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 10))
    parser.add_argument("--theta-draws", type=int, default=128)
    parser.add_argument("--periods", type=int, default=4)
    parser.add_argument("--sep-max-iter", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--python-sep-chunk-size", type=int, default=4)
    parser.add_argument("--julia-param-set", default="phase1_18params_narrow")
    parser.add_argument("--julia-theta-sampling", choices=("prior", "lhs", "grid"), default="prior")
    parser.add_argument("--julia-shock-scale", type=float, default=0.05)
    parser.add_argument("--julia-sep-max-iter", type=int, default=20)
    parser.add_argument("--julia-sep-accept-tol", type=float, default=1.0e-4)
    parser.add_argument("--julia-theta-attempts-per-theta", type=int, default=10)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-julia", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-env", action="append", default=[], help="Extra KEY=VALUE env for Python stage.")
    parser.add_argument("--julia-env", action="append", default=[], help="Extra KEY=VALUE env for Julia stages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = (args.results_root / args.run_label).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exe = args.python
    if not python_exe.is_file():
        python_exe = Path(sys.executable)

    summary: dict[str, Any] = {
        "run_label": args.run_label,
        "run_dir": str(run_dir),
        "host": host_info(),
        "notes": [
            "Python stage is the integrated JAX fixed-reference HLT SEP/ResNN/likelihood/HMC harness.",
            "Julia stage profiles the companion Julia real-HLT SEP target-generation and ResNN training scripts.",
            "Julia does not currently expose the identical fixed-ROM HMC wrapper used by the Python stage.",
            "Python defaults to sw07_safe_27 because the perturbed phase1_18params draw can fail first-order convergence before target generation.",
            "Julia defaults to phase1_18params_narrow with prior sampling, retry attempts, and a smaller shock scale because endpoint LHS draws can produce zero SEP-success samples in small CPU pilots.",
        ],
        "config": {
            "theta_draws": args.theta_draws,
            "periods": args.periods,
            "sep_max_iter": args.sep_max_iter,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "blocks": args.blocks,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "python_sep_chunk_size": args.python_sep_chunk_size,
            "python_parameter_set": "sw07_safe_27",
            "julia_parameter_set": args.julia_param_set,
            "julia_theta_sampling": args.julia_theta_sampling,
            "julia_shock_scale": args.julia_shock_scale,
            "julia_sep_max_iter": args.julia_sep_max_iter,
            "julia_sep_accept_tol": args.julia_sep_accept_tol,
            "julia_theta_attempts_per_theta": args.julia_theta_attempts_per_theta,
        },
        "stages": [],
    }

    summary_path = run_dir / "cpu_profile_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    python_overrides = parse_key_value_overrides(args.python_env)
    julia_overrides = parse_key_value_overrides(args.julia_env)

    if not args.skip_python:
        python_dir = run_dir / "python_jax_cpu"
        python_env = {
            "MODE": "estimation_pilot",
            "DEVICE": "cpu",
            "REQUIRE_GPU": "0",
            "INSTALL_DEPS": "0",
            "RESULT_ROOT": str(python_dir),
            "PYTHON": str(python_exe),
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "HLT_TARGET_BUILDER": "batched-sep",
            "HLT_SEP_BATCH_CHUNK_SIZE": str(args.python_sep_chunk_size),
            "HLT_PARAMETER_SET": "sw07_safe_27",
            "HLT_PARAMETER_PERTURBATION": "0",
            "HLT_THETA_DRAWS": str(args.theta_draws),
            "HLT_PERIODS": str(args.periods),
            "SEP_PERIODS": str(args.periods),
            "SEP_ORDER": "1",
            "SEP_NNODES": "3",
            "SEP_MAX_ITER": str(args.sep_max_iter),
            "EPOCHS": str(args.epochs),
            "HIDDEN": str(args.hidden),
            "BLOCKS": str(args.blocks),
            "TRAIN_BATCH_SIZE": str(args.batch_size),
            "HMC_CHAINS": "64",
            "HMC_WARMUP": "256",
            "HMC_SAMPLES": "512",
            "HMC_LEAPFROG_STEPS": "6",
            "HMC_STEP_SIZE": "0.0005",
            "OMP_NUM_THREADS": str(args.threads),
            "PROFILE_SUPPRESS_JSON_STDOUT": "1",
        }
        python_env.update(python_overrides)
        stage = run_stage(
            name="python_jax_cpu",
            cmd=["bash", "benchmarks/run_hlt_gpu_estimation.sh"],
            cwd=ROOT,
            env=python_env,
            log_path=run_dir / "python_jax_cpu.log",
            dry_run=args.dry_run,
        )
        stage["metrics"] = read_python_metrics(python_dir / "hlt_estimation_pilot_surrogate_estimation.json")
        summary["stages"].append(stage)
        summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.skip_julia:
        julia_repo = args.julia_repo.resolve()
        dataset_dir = run_dir / "julia_hlt_phase1_18_narrow"
        dataset_path = dataset_dir / "hlt_sep_surrogate_dataset.jls"
        trained_path = dataset_dir / "hlt_sep_surrogate_trained.jls"
        common_julia_env = {
            "JULIA_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
            "OMP_NUM_THREADS": str(args.threads),
        }
        common_julia_env.update(julia_overrides)
        dataset_cmd = [
            str(args.julia),
            f"--project={julia_repo}",
            str(julia_repo / "scripts" / "hlt_sep_surrogate_dataset_generate.jl"),
            f"--param-set={args.julia_param_set}",
            f"--theta-sampling={args.julia_theta_sampling}",
            f"--theta-samples={args.theta_draws}",
            f"--theta-attempts-per-theta={args.julia_theta_attempts_per_theta}",
            f"--theta-max-attempts={max(args.theta_draws * args.julia_theta_attempts_per_theta, args.theta_draws)}",
            f"--samples-per-theta={args.periods}",
            "--burn-in=0",
            "--sample-start=1",
            f"--sample-length={args.periods}",
            f"--shock-scale={args.julia_shock_scale:g}",
            f"--sep-horizon={args.periods}",
            "--sep-order=1",
            "--sep-nnodes=3",
            f"--sep-maxit={args.julia_sep_max_iter}",
            "--sep-tol=1e-8",
            f"--sep-accept-tol={args.julia_sep_accept_tol:g}",
            "--sep-linear-solver=qr",
            "--sep-line-search=true",
            "--stable-prefix",
            "--stable-min-periods=1",
            "--rom-orders=1",
            "--rom-mode=baseline",
            f"--output-dir={dataset_dir}",
            "--progress-every=1",
            "--checkpoint-every=10",
            "--timing",
        ]
        stage = run_stage(
            name="julia_hlt_dataset_cpu",
            cmd=dataset_cmd,
            cwd=julia_repo,
            env=common_julia_env,
            log_path=run_dir / "julia_hlt_dataset_cpu.log",
            dry_run=args.dry_run,
        )
        stage["expected_dataset_path"] = str(dataset_path)
        stage["metrics"] = parse_julia_log_metrics(run_dir / "julia_hlt_dataset_cpu.log")
        if dataset_path.is_file():
            stage["dataset_bytes"] = dataset_path.stat().st_size
        summary["stages"].append(stage)
        summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        train_cmd = [
            str(args.julia),
            f"--project={julia_repo}",
            str(julia_repo / "scripts" / "hlt_sep_surrogate_train.jl"),
            str(dataset_path),
            "--arch=resnet",
            "--rom-residual=1",
            "--obs-only",
            "--only-full-success",
            f"--epochs={args.epochs}",
            f"--hidden={args.hidden}",
            f"--n-blocks={args.blocks}",
            f"--batch={args.batch_size}",
            "--lr=1e-3",
            f"--out={trained_path}",
        ]
        if args.theta_draws >= 4:
            train_cmd.insert(9, "--split-by-theta")
        stage = run_stage(
            name="julia_hlt_resnn_train_cpu",
            cmd=train_cmd,
            cwd=julia_repo,
            env=common_julia_env,
            log_path=run_dir / "julia_hlt_resnn_train_cpu.log",
            dry_run=args.dry_run,
        )
        stage["expected_surrogate_path"] = str(trained_path)
        stage["metrics"] = parse_julia_log_metrics(run_dir / "julia_hlt_resnn_train_cpu.log")
        if trained_path.is_file():
            stage["surrogate_bytes"] = trained_path.stat().st_size
        summary["stages"].append(stage)
        summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary["end_utc"] = utc_stamp()
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress(f"wrote summary {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
