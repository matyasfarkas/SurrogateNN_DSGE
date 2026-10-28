#!/usr/bin/env python3
"""Run a command while recording publication-grade NVIDIA GPU telemetry.

The profiler samples device-wide ``nvidia-smi`` counters throughout the child
process.  It writes the raw time series to CSV and a provenance-rich summary to
JSON.  Device-wide counters are appropriate only on a dedicated GPU; they do
not attribute utilization to an individual process.

Example
-------

.. code-block:: bash

   python benchmarks/profile_gpu_run.py \
     --output-dir benchmarks/results/example/gpu_profile \
     --require-complete -- python benchmarks/my_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-gpu-profile/v1"
CSV_NAME = "gpu_profile_samples.csv"
SUMMARY_NAME = "gpu_profile_summary.json"
NVIDIA_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "pstate",
)
CSV_FIELDS = (
    "sample_index",
    "elapsed_s",
    "timestamp_utc",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "driver_version",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "temperature_c",
    "sm_clock_mhz",
    "pstate",
)


@dataclass(frozen=True)
class GPUSample:
    sample_index: int
    elapsed_s: float
    timestamp_utc: str
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    driver_version: str
    gpu_utilization_percent: float | None
    memory_utilization_percent: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    temperature_c: float | None
    sm_clock_mhz: float | None
    pstate: str


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_nvidia_smi_csv(
    text: str,
    *,
    sample_index: int,
    elapsed_s: float,
    timestamp_utc: str,
) -> list[GPUSample]:
    """Parse one header-free ``nvidia-smi --query-gpu`` response."""

    samples: list[GPUSample] = []
    for row_number, row in enumerate(csv.reader(text.splitlines()), start=1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(NVIDIA_FIELDS):
            raise ValueError(
                f"nvidia-smi row {row_number} has {len(row)} fields; "
                f"expected {len(NVIDIA_FIELDS)}: {row!r}"
            )
        values = [cell.strip() for cell in row]
        try:
            gpu_index = int(values[0])
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index in row {row_number}: {values[0]!r}") from exc
        samples.append(
            GPUSample(
                sample_index=int(sample_index),
                elapsed_s=float(elapsed_s),
                timestamp_utc=str(timestamp_utc),
                gpu_index=gpu_index,
                gpu_uuid=values[1],
                gpu_name=values[2],
                driver_version=values[3],
                gpu_utilization_percent=_optional_float(values[4]),
                memory_utilization_percent=_optional_float(values[5]),
                memory_used_mib=_optional_float(values[6]),
                memory_total_mib=_optional_float(values[7]),
                power_draw_w=_optional_float(values[8]),
                power_limit_w=_optional_float(values[9]),
                temperature_c=_optional_float(values[10]),
                sm_clock_mhz=_optional_float(values[11]),
                pstate=values[12],
            )
        )
    if not samples:
        raise ValueError("nvidia-smi returned no GPU rows.")
    return samples


def _quantile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _metric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p50": _quantile(finite, 0.50),
        "p95": _quantile(finite, 0.95),
        "max": max(finite),
    }


def _trapezoid_energy_wh(samples: Sequence[GPUSample]) -> tuple[float | None, float]:
    usable = sorted(
        (sample for sample in samples if sample.power_draw_w is not None),
        key=lambda sample: sample.elapsed_s,
    )
    if len(usable) < 2:
        return None, 0.0
    watt_seconds = 0.0
    covered_s = 0.0
    for left, right in zip(usable, usable[1:]):
        delta_s = max(0.0, float(right.elapsed_s - left.elapsed_s))
        watt_seconds += 0.5 * float(left.power_draw_w + right.power_draw_w) * delta_s
        covered_s += delta_s
    return watt_seconds / 3600.0, covered_s


def summarize_samples(samples: Sequence[GPUSample]) -> list[dict[str, Any]]:
    """Summarize a raw telemetry series separately for every GPU UUID."""

    grouped: dict[str, list[GPUSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.gpu_uuid, []).append(sample)
    summaries: list[dict[str, Any]] = []
    for uuid in sorted(grouped):
        group = sorted(grouped[uuid], key=lambda sample: sample.elapsed_s)
        energy_wh, energy_window_s = _trapezoid_energy_wh(group)
        memory = _metric_summary(sample.memory_used_mib for sample in group)
        gpu_util = _metric_summary(sample.gpu_utilization_percent for sample in group)
        finite_util = [
            float(sample.gpu_utilization_percent)
            for sample in group
            if sample.gpu_utilization_percent is not None
        ]
        first_memory = group[0].memory_used_mib
        peak_memory = memory["max"]
        peak_delta = (
            float(peak_memory) - float(first_memory)
            if peak_memory is not None and first_memory is not None
            else None
        )
        summaries.append(
            {
                "gpu_index": group[0].gpu_index,
                "gpu_uuid": uuid,
                "gpu_name": group[0].gpu_name,
                "driver_version": group[0].driver_version,
                "sample_count": len(group),
                "sample_window_s": max(0.0, group[-1].elapsed_s - group[0].elapsed_s),
                "gpu_utilization_percent": gpu_util,
                "memory_utilization_percent": _metric_summary(
                    sample.memory_utilization_percent for sample in group
                ),
                "memory_used_mib": memory,
                "memory_total_mib": group[0].memory_total_mib,
                "memory_initial_mib": first_memory,
                "memory_peak_minus_initial_mib": peak_delta,
                "power_draw_w": _metric_summary(sample.power_draw_w for sample in group),
                "power_limit_w": group[0].power_limit_w,
                "temperature_c": _metric_summary(sample.temperature_c for sample in group),
                "sm_clock_mhz": _metric_summary(sample.sm_clock_mhz for sample in group),
                "active_share_above_10_percent": (
                    sum(value > 10.0 for value in finite_util) / len(finite_util)
                    if finite_util
                    else None
                ),
                "estimated_energy_wh": energy_wh,
                "energy_integration_window_s": energy_window_s,
            }
        )
    return summaries


def _git_provenance(cwd: Path) -> dict[str, Any]:
    def run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args), cwd=cwd, text=True, capture_output=True, check=False
        )

    revision = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_error": status.stderr.strip() or None if status.returncode != 0 else None,
    }


def _write_samples(path: Path, samples: Sequence[GPUSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def _query_once(nvidia_smi: str, *, sample_index: int, start: float) -> list[GPUSample]:
    elapsed_s = time.monotonic() - start
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    command = [
        nvidia_smi,
        "--query-gpu=" + ",".join(NVIDIA_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"nvidia-smi query failed with code {process.returncode}: {detail}")
    return parse_nvidia_smi_csv(
        process.stdout,
        sample_index=sample_index,
        elapsed_s=elapsed_s,
        timestamp_utc=timestamp_utc,
    )


def run_profile(
    command: Sequence[str],
    *,
    output_dir: Path,
    sample_interval_s: float,
    nvidia_smi: str = "nvidia-smi",
    require_complete: bool = False,
    cwd: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run ``command`` and return its effective exit code and profile summary."""

    if not command:
        raise ValueError("A child command is required.")
    if sample_interval_s <= 0.0:
        raise ValueError("sample_interval_s must be positive.")
    working_dir = (cwd or Path.cwd()).resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_smi = shutil.which(nvidia_smi) if not Path(nvidia_smi).is_absolute() else nvidia_smi
    if not resolved_smi or not Path(resolved_smi).exists():
        raise FileNotFoundError(f"Cannot find nvidia-smi executable: {nvidia_smi!r}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_monotonic = time.monotonic()
    initial = _query_once(str(resolved_smi), sample_index=0, start=started_monotonic)
    samples: list[GPUSample] = list(initial)
    errors: list[str] = []
    lock = threading.Lock()
    stop = threading.Event()
    sample_index = 1

    def sampler() -> None:
        nonlocal sample_index
        while not stop.wait(sample_interval_s):
            try:
                batch = _query_once(
                    str(resolved_smi), sample_index=sample_index, start=started_monotonic
                )
            except Exception as exc:  # Keep the workload alive; make loss explicit in JSON.
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
            else:
                with lock:
                    samples.extend(batch)
            sample_index += 1

    thread = threading.Thread(target=sampler, name="gpu-telemetry", daemon=True)
    thread.start()
    child: subprocess.Popen[Any] | None = None
    child_pid: int | None = None
    child_elapsed_s: float | None = None
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    try:
        child_started_monotonic = time.monotonic()
        child = subprocess.Popen(list(command), cwd=working_dir)
        child_pid = int(child.pid)
        child_exit_code = int(child.wait())
        child_elapsed_s = time.monotonic() - child_started_monotonic
    finally:
        stop.set()
        thread.join(timeout=max(1.0, 2.0 * sample_interval_s))
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    try:
        final = _query_once(
            str(resolved_smi), sample_index=sample_index, start=started_monotonic
        )
    except Exception as exc:
        errors.append(f"final {type(exc).__name__}: {exc}")
    else:
        samples.extend(final)

    ended_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elapsed_s = time.monotonic() - started_monotonic
    sample_groups = {sample.sample_index for sample in samples}
    # Initial and final snapshots alone are not a continuous profile. Require at
    # least one successful in-workload sample before the evidence is complete.
    minimum_sample_groups = 3
    complete = len(sample_groups) >= minimum_sample_groups and not errors
    effective_exit_code = child_exit_code
    if require_complete and child_exit_code == 0 and not complete:
        effective_exit_code = 3

    sample_path = output_dir / CSV_NAME
    summary_path = output_dir / SUMMARY_NAME
    _write_samples(sample_path, samples)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if complete else "incomplete",
        "measurement_scope": "system_wide_device_counters",
        "dedicated_gpu_required_for_attribution": True,
        "speedup_claim_supported": False,
        "speedup_claim_note": (
            "This profile characterizes one workload. A speedup claim additionally requires "
            "repeated same-work, same-precision baseline and treatment runs."
        ),
        "sampling_overhead_not_subtracted": True,
        "sampling_overhead_note": (
            "Each sample launches an nvidia-smi query. Apply the same sampling interval "
            "to all comparison arms and report unprofiled sensitivity timings."
        ),
        "energy_estimate_scope": (
            "Trapezoidal integration from the first to last successful device sample; "
            "the window includes small pre-launch and post-exit boundaries."
        ),
        "command": list(command),
        "cwd": str(working_dir),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "elapsed_s": elapsed_s,
        "sample_interval_s": sample_interval_s,
        "sample_group_count": len(sample_groups),
        "minimum_sample_groups_for_complete": minimum_sample_groups,
        "sample_row_count": len(samples),
        "telemetry_errors": errors,
        "child_exit_code": child_exit_code,
        "child_pid": child_pid,
        "child_elapsed_s": child_elapsed_s,
        "effective_exit_code": effective_exit_code,
        "require_complete": require_complete,
        "git": _git_provenance(working_dir),
        "raw_samples_csv": CSV_NAME,
        "gpus": summarize_samples(samples),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return effective_exit_code, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after '--'")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        exit_code, summary = run_profile(
            args.command,
            output_dir=args.output_dir,
            sample_interval_s=args.sample_interval_s,
            nvidia_smi=args.nvidia_smi,
            require_complete=args.require_complete,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"GPU profiling failed before workload launch: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(args.output_dir / SUMMARY_NAME),
                "exit_code": exit_code,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
