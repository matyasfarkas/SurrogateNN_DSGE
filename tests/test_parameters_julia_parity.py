from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_parameter_config_matches_julia_reference_when_enabled() -> None:
    if os.environ.get("SURROGATENN_RUN_JULIA_PARITY") != "1":
        pytest.skip("Set SURROGATENN_RUN_JULIA_PARITY=1 to run Julia parameter-config parity.")
    root = Path(__file__).resolve().parents[1]
    script = root / "benchmarks" / "validate_parameter_config_julia_parity.py"
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)
