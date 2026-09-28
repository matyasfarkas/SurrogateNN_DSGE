#!/usr/bin/env python3
"""Bootstrap and run the GPUHub HLT nonlinear SEP/ResNN pipeline.

This runner is deliberately staged. Use ``smoke_then_full`` only after the
small smoke stage passes on the selected GPU. The production estimator currently
uses the fixed-reference steady-state/fixed-ROM surrogate likelihood unless
``--likelihood-runtime-mode full-jax`` is requested explicitly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpuhub_bootstrap import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_REPO_URL,
    apply_environment_updates,
    choose_jax_requirement,
    ensure_python_version,
    install_stack,
    parse_nvidia_smi,
    probe_gpu,
    run,
    sanitize_jax_runtime_environment,
    sync_repo,
)


DEFAULT_BRANCH = "codex/nonlinear-sep-surrogate-port"
DEFAULT_ROOT = DEFAULT_DATA_ROOT / "SurrogateNN_DSGE"


def _utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_env_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Environment override must have KEY=VALUE form, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Environment override has an empty key: {item!r}.")
        overrides[key] = value
    return overrides


def stage_sequence(mode: str) -> tuple[str, ...]:
    if mode == "setup":
        return ()
    if mode in {"smoke", "pilot", "full"}:
        return (mode,)
    if mode == "smoke_then_pilot":
        return ("smoke", "pilot")
    if mode == "smoke_then_full":
        return ("smoke", "full")
    if mode == "smoke_pilot_full":
        return ("smoke", "pilot", "full")
    raise ValueError(f"Unknown HLT run mode {mode!r}.")


def stage_output_path(result_root: Path, stage: str) -> Path:
    return result_root / f"hlt_{stage}_surrogate_estimation.json"


def build_hlt_stage_environment(
    *,
    stage: str,
    python: str,
    result_root: Path,
    allow_cpu: bool,
    hlt_target_builder: str | None,
    hlt_parameter_set: str | None,
    steady_state_mode: str,
    likelihood_runtime_mode: str,
    likelihood_qme_algorithm: str,
    likelihood_static_rows_mode: str,
    jax_log_density_gradient: bool,
    differentiate_shocks: bool,
    extra_env: dict[str, str],
) -> dict[str, str]:
    env = {
        "MODE": stage,
        "PYTHON": python,
        "DEVICE": "cpu" if allow_cpu else "gpu",
        "REQUIRE_GPU": "0" if allow_cpu else "1",
        "INSTALL_DEPS": "0",
        "RESULT_ROOT": str(result_root),
        "HLT_STEADY_STATE_MODE": steady_state_mode,
        "LIKELIHOOD_RUNTIME_MODE": likelihood_runtime_mode,
        "LIKELIHOOD_QME_ALGORITHM": likelihood_qme_algorithm,
        "LIKELIHOOD_STATIC_ROWS_MODE": likelihood_static_rows_mode,
        "JAX_LOG_DENSITY_GRADIENT": "1" if jax_log_density_gradient else "0",
        "DIFFERENTIATE_SHOCKS": "1" if differentiate_shocks else "0",
    }
    if hlt_target_builder is not None:
        env["HLT_TARGET_BUILDER"] = hlt_target_builder
    if hlt_parameter_set is not None:
        env["HLT_PARAMETER_SET"] = hlt_parameter_set
    env.update(extra_env)
    return env


def summarize_hlt_result(output: Path) -> dict[str, Any]:
    if not output.exists():
        return {"status": "missing", "output": str(output)}
    payload = json.loads(output.read_text(encoding="utf-8"))
    result = payload.get("results", {}).get("hlt_fixed_ss_smoke", {})
    target = result.get("target_diagnostics", {}) or {}
    log_density = result.get("jax_surrogate_log_density", {}) or {}
    hmc = result.get("surrogate_hmc", {}) or {}
    likelihood = result.get("surrogate_inversion_likelihood", {}) or {}
    return {
        "status": result.get("status"),
        "output": str(output),
        "backend": result.get("backend"),
        "parameter_count": len(result.get("parameter_subset") or []),
        "parameter_set": result.get("hlt_parameter_set"),
        "theta_draws": result.get("theta_draws"),
        "train_size": result.get("train_size"),
        "val_size": result.get("val_size"),
        "pipeline_s": result.get("pipeline_s"),
        "target_builder": target.get("builder"),
        "target_status": target.get("status"),
        "accepted_samples": target.get("accepted_samples"),
        "theta_full_success_count": target.get("theta_full_success_count"),
        "fallback_share": target.get("fallback_share"),
        "likelihood_status": likelihood.get("status"),
        "likelihood": likelihood.get("total_loglikelihood"),
        "jax_log_density_status": log_density.get("status"),
        "jax_log_density_parity_ok": log_density.get("parity_ok"),
        "jax_log_density_runtime_mode": log_density.get("runtime_mode"),
        "hmc_status": hmc.get("status"),
        "hmc_draws": hmc.get("post_warmup_draws"),
        "hmc_elapsed_s": hmc.get("elapsed_s"),
        "hmc_draws_per_second": hmc.get("draws_per_second"),
        "hmc_accepted_share": hmc.get("accepted_share"),
        "hmc_retry_count": hmc.get("retry_count"),
    }


def write_master_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote master summary: {path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--mode",
        choices=(
            "setup",
            "smoke",
            "pilot",
            "full",
            "smoke_then_pilot",
            "smoke_then_full",
            "smoke_pilot_full",
        ),
        default="setup",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--skip-repo-sync", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-gpu-probe", action="store_true")
    parser.add_argument("--jax-extra", choices=("auto", "cuda13", "cuda12", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-force-reinstall-jax", action="store_true")
    parser.add_argument("--preserve-ld-library-path", action="store_true")
    parser.add_argument("--hlt-target-builder", choices=("adaptive-sep", "batched-sep", "callback"), default=None)
    parser.add_argument("--hlt-parameter-set", default=None)
    parser.add_argument(
        "--steady-state-mode",
        choices=("fixed-reference", "solve", "solve-or-reference"),
        default="fixed-reference",
    )
    parser.add_argument(
        "--likelihood-runtime-mode",
        choices=("fixed-reference", "full-jax"),
        default="fixed-reference",
    )
    parser.add_argument(
        "--likelihood-qme-algorithm",
        choices=("schur", "schur_gpu", "doubling"),
        default="schur",
    )
    parser.add_argument(
        "--likelihood-static-rows-mode",
        choices=("reference", "none"),
        default="reference",
    )
    parser.add_argument("--no-jax-log-density-gradient", action="store_true")
    parser.add_argument("--differentiate-shocks", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override passed to benchmarks/run_hlt_gpu_estimation.sh.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not (args.allow_cpu and args.skip_install):
        ensure_python_version()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    run_label = str(args.run_label or _utc_label())
    results_base = args.results_dir or (
        args.root / "benchmarks" / "results" / f"gpuhub_hlt_resnn_{run_label}"
    )
    stages = stage_sequence(args.mode)
    extra_env = _parse_env_overrides(args.env)

    try:
        smi = run(["nvidia-smi"], check=False)
        smi_text = (smi.stdout or "") + (smi.stderr or "")
    except FileNotFoundError:
        if not args.allow_cpu:
            raise
        smi_text = ""
    smi_info = parse_nvidia_smi(smi_text)
    print("nvidia-smi parsed:", json.dumps(smi_info, indent=2), flush=True)
    jax_requirement = choose_jax_requirement(
        "cpu" if args.allow_cpu else args.jax_extra,
        cuda_version=smi_info.get("cuda_version"),
        driver_version=smi_info.get("driver_version"),
    )
    print(f"Selected JAX requirement: {jax_requirement}", flush=True)
    apply_environment_updates(
        sanitize_jax_runtime_environment(
            jax_requirement,
            preserve_ld_library_path=bool(args.preserve_ld_library_path),
        )
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "branch": args.branch,
                    "results_base": str(results_base),
                    "stages": stages,
                    "jax_requirement": jax_requirement,
                    "extra_env": extra_env,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    if not args.skip_repo_sync:
        sync_repo(args.root, args.repo_url, args.branch)
    if not args.skip_install:
        install_stack(
            args.root,
            jax_requirement,
            force_reinstall_jax=not args.no_force_reinstall_jax,
        )
    if not args.skip_gpu_probe:
        probe_gpu(force_gpu=not args.allow_cpu, dtype="float64")

    stage_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for stage in stages:
        result_root = results_base / stage
        result_root.mkdir(parents=True, exist_ok=True)
        env = build_hlt_stage_environment(
            stage=stage,
            python=sys.executable,
            result_root=result_root,
            allow_cpu=bool(args.allow_cpu),
            hlt_target_builder=args.hlt_target_builder,
            hlt_parameter_set=args.hlt_parameter_set,
            steady_state_mode=str(args.steady_state_mode),
            likelihood_runtime_mode=str(args.likelihood_runtime_mode),
            likelihood_qme_algorithm=str(args.likelihood_qme_algorithm),
            likelihood_static_rows_mode=str(args.likelihood_static_rows_mode),
            jax_log_density_gradient=not bool(args.no_jax_log_density_gradient),
            differentiate_shocks=bool(args.differentiate_shocks),
            extra_env=extra_env,
        )
        print(f"Starting HLT stage={stage} result_root={result_root}", flush=True)
        stage_started = time.perf_counter()
        run(["bash", "benchmarks/run_hlt_gpu_estimation.sh"], cwd=args.root, env=env, stream=True)
        summary = summarize_hlt_result(stage_output_path(result_root, stage))
        summary["stage"] = stage
        summary["stage_wall_s"] = time.perf_counter() - stage_started
        stage_summaries.append(summary)
        write_master_summary(
            results_base / "master_summary.json",
            {
                "status": "running" if stage != stages[-1] else "ok",
                "mode": args.mode,
                "run_label": run_label,
                "branch": args.branch,
                "root": str(args.root),
                "results_base": str(results_base),
                "jax_requirement": jax_requirement,
                "nvidia_smi": smi_info,
                "elapsed_s": time.perf_counter() - started,
                "stages": stage_summaries,
            },
        )

    if not stages:
        write_master_summary(
            results_base / "master_summary.json",
            {
                "status": "setup_ok",
                "mode": args.mode,
                "run_label": run_label,
                "branch": args.branch,
                "root": str(args.root),
                "results_base": str(results_base),
                "jax_requirement": jax_requirement,
                "nvidia_smi": smi_info,
            },
        )


if __name__ == "__main__":
    main()
