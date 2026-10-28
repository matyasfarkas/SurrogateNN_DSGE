from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "profile_gpu_run.py"
    spec = importlib.util.spec_from_file_location("profile_gpu_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(
    module: ModuleType,
    *,
    index: int,
    elapsed: float,
    utilization: float,
    memory: float,
    power: float,
) -> object:
    return module.GPUSample(
        sample_index=index,
        elapsed_s=elapsed,
        timestamp_utc=f"2026-01-01T00:00:0{index}Z",
        gpu_index=0,
        gpu_uuid="GPU-test",
        gpu_name="Test GPU",
        driver_version="1.2.3",
        gpu_utilization_percent=utilization,
        memory_utilization_percent=utilization / 2.0,
        memory_used_mib=memory,
        memory_total_mib=1000.0,
        power_draw_w=power,
        power_limit_w=300.0,
        temperature_c=50.0 + index,
        sm_clock_mhz=1500.0,
        pstate="P2",
    )


def test_parse_nvidia_smi_csv_handles_units_removed_and_na() -> None:
    module = _load_module()
    text = (
        "0, GPU-abc, NVIDIA RTX Test, 595.1, 87, 42, 1234, 32760, "
        "250.5, 320, 71, 2400, P2\n"
        "1, GPU-def, NVIDIA RTX Other, 595.1, N/A, 0, 10, 32760, "
        "[Not Supported], 320, 30, 210, P8\n"
    )

    samples = module.parse_nvidia_smi_csv(
        text,
        sample_index=4,
        elapsed_s=2.5,
        timestamp_utc="2026-01-01T00:00:00Z",
    )

    assert len(samples) == 2
    assert samples[0].gpu_utilization_percent == 87.0
    assert samples[0].memory_total_mib == 32760.0
    assert samples[1].gpu_utilization_percent is None
    assert samples[1].power_draw_w is None


def test_summary_reports_peak_utilization_memory_and_energy() -> None:
    module = _load_module()
    samples = [
        _sample(module, index=0, elapsed=0.0, utilization=0.0, memory=100.0, power=100.0),
        _sample(module, index=1, elapsed=1.0, utilization=50.0, memory=400.0, power=200.0),
        _sample(module, index=2, elapsed=2.0, utilization=100.0, memory=300.0, power=300.0),
    ]

    summary = module.summarize_samples(samples)[0]

    assert summary["sample_count"] == 3
    assert summary["gpu_utilization_percent"]["mean"] == 50.0
    assert summary["gpu_utilization_percent"]["max"] == 100.0
    assert summary["memory_used_mib"]["max"] == 400.0
    assert summary["memory_peak_minus_initial_mib"] == 300.0
    assert summary["active_share_above_10_percent"] == pytest.approx(2.0 / 3.0)
    assert summary["estimated_energy_wh"] == pytest.approx(400.0 / 3600.0)
    assert summary["energy_integration_window_s"] == 2.0


def test_run_profile_writes_raw_and_summary_files(tmp_path: Path) -> None:
    module = _load_module()
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '0, GPU-test, Test GPU, 595.1, 75, 20, 500, 1000, 200, 300, 60, 1500, P2'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IXUSR)
    output = tmp_path / "profile"

    exit_code, summary = module.run_profile(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        output_dir=output,
        sample_interval_s=0.01,
        nvidia_smi=str(fake_smi),
        require_complete=True,
        cwd=Path(__file__).parents[1],
    )

    assert exit_code == 0
    assert summary["status"] == "ok"
    assert summary["measurement_scope"] == "system_wide_device_counters"
    assert summary["speedup_claim_supported"] is False
    assert summary["sample_group_count"] >= 2
    assert summary["gpus"][0]["gpu_utilization_percent"]["mean"] == 75.0
    assert (output / module.CSV_NAME).is_file()
    stored = json.loads((output / module.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert stored["child_exit_code"] == 0
    assert stored["child_pid"] > 0
    assert stored["child_elapsed_s"] > 0.0
    assert stored["sampling_overhead_not_subtracted"] is True
    assert stored["gpus"][0]["gpu_uuid"] == "GPU-test"


def test_require_complete_rejects_only_initial_and_final_samples(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '0, GPU-test, Test GPU, 595.1, 0, 0, 10, 1000, 10, 300, 30, 200, P8'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IXUSR)

    exit_code, summary = module.run_profile(
        [sys.executable, "-c", "pass"],
        output_dir=tmp_path / "profile",
        sample_interval_s=10.0,
        nvidia_smi=str(fake_smi),
        require_complete=True,
        cwd=Path(__file__).parents[1],
    )

    # Two point snapshots do not constitute a continuous workload profile.
    assert exit_code == 3
    assert summary["status"] == "incomplete"
    assert summary["sample_group_count"] == 2
