from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_PATH = ROOT / "benchmarks" / "results" / "20260712T083357" / "validation_payloads.json"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "hmc_chain_ablation"


@dataclass(frozen=True)
class AblationJob:
    method: str
    chains: int
    command: tuple[str, ...]
    output_path: Path


def parse_int_list(value: str) -> tuple[int, ...]:
    parts = [part for part in value.replace(",", " ").split() if part]
    if not parts:
        raise ValueError("chain list must contain at least one positive integer")
    numbers = tuple(int(part) for part in parts)
    if any(number <= 0 for number in numbers):
        raise ValueError("chain counts must be positive")
    return numbers


def parse_method_list(value: str) -> tuple[str, ...]:
    allowed = {"numpyro_nuts", "numpyro_hmc", "static_hmc"}
    methods = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    unknown = tuple(method for method in methods if method not in allowed)
    if unknown:
        raise ValueError(f"unknown ablation methods: {', '.join(unknown)}")
    if not methods:
        raise ValueError("method list must contain at least one method")
    return methods


def _common_dataset_args(args: argparse.Namespace) -> list[str]:
    return [
        "--preset",
        args.preset,
        "--payload-path",
        str(args.payload_path),
        "--case",
        args.case,
        "--periods",
        str(args.periods),
        "--parameters",
        args.parameters,
        "--prior-width-scale",
        str(args.prior_width_scale),
        "--prior-width-floor",
        str(args.prior_width_floor),
        "--dtype",
        args.dtype,
        "--qme-algorithm",
        args.qme_algorithm,
        "--failure-value",
        str(args.failure_value),
    ]


def _runtime_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    if args.platform is not None:
        result.extend(["--platform", args.platform])
    if args.force_gpu:
        result.append("--force-gpu")
    if args.parameters_are_resolved:
        result.append("--parameters-are-resolved")
    if args.skip_parameter_bounds:
        result.append("--skip-parameter-bounds")
    if args.verbose:
        result.append("--verbose")
    return result


def _output_path(args: argparse.Namespace, method: str, chains: int) -> Path:
    stem = (
        f"{args.label}_{method}_{args.preset}_{args.case}_"
        f"{args.parameters}_{args.periods}p_{chains}chains_{args.dtype}.json"
    )
    return args.results_dir / stem


def build_jobs(args: argparse.Namespace) -> list[AblationJob]:
    methods = parse_method_list(args.methods)
    chains = parse_int_list(args.chains)
    jobs: list[AblationJob] = []
    common = _common_dataset_args(args)
    runtime = _runtime_args(args)
    python = sys.executable
    for method in methods:
        for chain_count in chains:
            output_path = _output_path(args, method, chain_count)
            if method == "static_hmc":
                command = [
                    python,
                    str(ROOT / "benchmarks" / "static_hmc_sampling_speed.py"),
                    *common,
                    "--chains",
                    str(chain_count),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--leapfrog-steps",
                    str(args.static_leapfrog_steps),
                    "--step-size",
                    str(args.static_step_size),
                    "--max-step-size",
                    str(args.static_max_step_size),
                    "--steady-reps",
                    str(args.steady_reps),
                    *runtime,
                    "--output",
                    str(output_path),
                ]
            elif method == "numpyro_hmc":
                command = [
                    python,
                    str(ROOT / "benchmarks" / "posterior_sampling_speed.py"),
                    *common,
                    "--chains",
                    str(chain_count),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--chain-method",
                    args.chain_method,
                    "--kernel",
                    "hmc",
                    "--hmc-num-steps",
                    str(args.numpyro_hmc_num_steps),
                    "--hmc-step-size",
                    str(args.numpyro_hmc_step_size),
                    "--target-accept-prob",
                    str(args.target_accept_prob),
                    "--max-tree-depth",
                    str(args.max_tree_depth),
                    *runtime,
                    "--output",
                    str(output_path),
                ]
            else:
                command = [
                    python,
                    str(ROOT / "benchmarks" / "posterior_sampling_speed.py"),
                    *common,
                    "--chains",
                    str(chain_count),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--chain-method",
                    args.chain_method,
                    "--kernel",
                    "nuts",
                    "--target-accept-prob",
                    str(args.target_accept_prob),
                    "--max-tree-depth",
                    str(args.max_tree_depth),
                    *runtime,
                    "--output",
                    str(output_path),
                ]
            jobs.append(
                AblationJob(
                    method=method,
                    chains=chain_count,
                    command=tuple(command),
                    output_path=output_path,
                )
            )
    return jobs


def _run_job(job: AblationJob, *, dry_run: bool, skip_existing: bool) -> dict[str, Any]:
    if skip_existing and job.output_path.exists():
        return {
            "method": job.method,
            "chains": job.chains,
            "output_path": str(job.output_path),
            "status": "skipped_existing",
            "command": list(job.command),
        }
    if dry_run:
        return {
            "method": job.method,
            "chains": job.chains,
            "output_path": str(job.output_path),
            "status": "dry_run",
            "command": list(job.command),
        }
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(job.command),
        cwd=ROOT,
        check=False,
        text=True,
    )
    return {
        "method": job.method,
        "chains": job.chains,
        "output_path": str(job.output_path),
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "command": list(job.command),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    jobs = build_jobs(args)
    results = [
        _run_job(job, dry_run=bool(args.dry_run), skip_existing=bool(args.skip_existing))
        for job in jobs
    ]
    summary = {
        "benchmark": {
            "label": args.label,
            "methods": list(parse_method_list(args.methods)),
            "chains": list(parse_int_list(args.chains)),
            "preset": args.preset,
            "case": args.case,
            "periods": int(args.periods),
            "parameters": args.parameters,
            "warmup": int(args.warmup),
            "samples": int(args.samples),
            "platform": args.platform,
            "dtype": args.dtype,
            "qme_algorithm": args.qme_algorithm,
        },
        "jobs": results,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chain-count ablations for JAX/NumPyro DSGE HMC benchmarks."
    )
    parser.add_argument("--methods", default="static_hmc,numpyro_hmc,numpyro_nuts")
    parser.add_argument("--chains", default="1,2,4,8,16,32")
    parser.add_argument("--label", default="sw07_safe15")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "chain_ablation_summary.json",
    )
    parser.add_argument("--preset", choices=("toy_ar2", "sw07_hlt"), default="sw07_hlt")
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--parameters", default="sw07_safe_15")
    parser.add_argument("--prior-width-scale", type=float, default=0.0025)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--chain-method", choices=("parallel", "vectorized", "sequential"), default="vectorized")
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=5)
    parser.add_argument("--numpyro-hmc-num-steps", type=int, default=8)
    parser.add_argument("--numpyro-hmc-step-size", type=float, default=0.1)
    parser.add_argument("--static-leapfrog-steps", type=int, default=8)
    parser.add_argument("--static-step-size", type=float, default=1.0)
    parser.add_argument("--static-max-step-size", type=float, default=2.0)
    parser.add_argument("--steady-reps", type=int, default=0)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--qme-algorithm",
        choices=("schur", "schur_gpu", "doubling"),
        default="schur_gpu",
    )
    parser.add_argument("--parameters-are-resolved", action="store_true")
    parser.add_argument("--skip-parameter-bounds", action="store_true")
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_ablation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
