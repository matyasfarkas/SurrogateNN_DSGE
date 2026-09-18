from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JULIA_REPO = ROOT.parent / "SurrogateNN_Estimation.jl"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from surrogatenn_dsge import (  # noqa: E402
    get_parameter_specs,
    get_phase1_18param_baseline,
)

PARAMETER_SETS = (
    "legacy_3params",
    "phase1_18params",
    "phase1_18params_narrow",
    "investment_4p",
    "investment_4p_supported",
    "investment_curvature_5p",
)


def _julia_script(julia_repo: Path) -> str:
    repo_literal = json.dumps(str(julia_repo))
    sets_literal = "[" + ", ".join(json.dumps(name) for name in PARAMETER_SETS) + "]"
    return f"""
include(joinpath({repo_literal}, "scripts", "hlt_surrogate", "parameter_config.jl"))

function normalize_prior_key(key)
    s = string(key)
    if s == "\\u03b1"
        return "alpha"
    elseif s == "\\u03b2"
        return "beta"
    elseif s == "\\u03bc"
        return "mu"
    elseif s == "\\u03c3"
        return "sigma"
    elseif s == "\\u03b8"
        return "theta"
    else
        return s
    end
end

function params_string(nt)
    parts = String[]
    for key in keys(nt)
        value = Float64(getproperty(nt, key))
        push!(parts, string(normalize_prior_key(key), "=", value))
    end
    return join(parts, ",")
end

for set_name in {sets_literal}
    specs = get_parameter_specs(Symbol(set_name))
    println("SET|", set_name, "|", length(specs))
    for (i, spec) in enumerate(specs)
        lower, upper = spec.bounds
        println(
            "SPEC|", set_name, "|", i, "|", string(spec.name), "|",
            string(spec.prior_type), "|", params_string(spec.prior_params), "|",
            lower, "|", upper, "|", spec.description,
        )
    end
end

baseline = get_phase1_18param_baseline()
for key in sort(collect(keys(baseline)), by=string)
    println("BASE|", string(key), "|", baseline[key])
end
"""


def _parse_params(text: str) -> dict[str, float]:
    if not text:
        return {}
    params: dict[str, float] = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        params[key] = float(value)
    return params


def _parse_output(stdout: str) -> tuple[dict[str, int], dict[str, list[dict[str, object]]], dict[str, float]]:
    counts: dict[str, int] = {}
    specs: dict[str, list[dict[str, object]]] = {}
    baseline: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split("|")
        if not parts:
            continue
        if parts[0] == "SET":
            counts[parts[1]] = int(parts[2])
        elif parts[0] == "SPEC":
            set_name = parts[1]
            specs.setdefault(set_name, []).append(
                {
                    "index": int(parts[2]),
                    "name": parts[3],
                    "prior_type": parts[4],
                    "prior_params": _parse_params(parts[5]),
                    "bounds": (float(parts[6]), float(parts[7])),
                    "description": parts[8],
                }
            )
        elif parts[0] == "BASE":
            baseline[parts[1]] = float(parts[2])
    return counts, specs, baseline


def run_parity(julia_repo: Path, julia_exe: str = "julia") -> None:
    if shutil.which(julia_exe) is None:
        raise RuntimeError(f"Julia executable not found: {julia_exe}")
    if not julia_repo.exists():
        raise RuntimeError(f"Julia repo not found: {julia_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "parameter_config_parity.jl"
        script_path.write_text(_julia_script(julia_repo), encoding="utf-8")
        proc = subprocess.run(
            [julia_exe, f"--project={julia_repo}", str(script_path)],
            text=True,
            capture_output=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Julia parameter parity failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    counts, observed_specs, observed_baseline = _parse_output(proc.stdout)
    for set_name in PARAMETER_SETS:
        expected_specs = get_parameter_specs(set_name)
        if counts.get(set_name) != len(expected_specs):
            raise AssertionError(f"{set_name}: Julia count {counts.get(set_name)} != Python {len(expected_specs)}")
        if len(observed_specs.get(set_name, [])) != len(expected_specs):
            raise AssertionError(f"{set_name}: missing Julia specs")
        for index, (observed, expected) in enumerate(zip(observed_specs[set_name], expected_specs), start=1):
            assert observed["index"] == index
            assert observed["name"] == expected.name
            assert observed["prior_type"] == expected.prior_type
            assert observed["description"] == expected.description
            assert observed["prior_params"] == expected.prior_params
            np.testing.assert_allclose(observed["bounds"], expected.bounds, rtol=0, atol=0, err_msg=set_name)

    expected_baseline = get_phase1_18param_baseline()
    if set(observed_baseline) != set(expected_baseline):
        raise AssertionError(f"Baseline key mismatch: {sorted(set(observed_baseline) ^ set(expected_baseline))}")
    for key, expected in expected_baseline.items():
        np.testing.assert_allclose(observed_baseline[key], expected, rtol=0, atol=0, err_msg=key)

    print("Parameter-config Julia parity passed for:", ", ".join(PARAMETER_SETS))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Python HLT parameter metadata against Julia reference.")
    parser.add_argument("--julia-repo", type=Path, default=DEFAULT_JULIA_REPO)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    run_parity(args.julia_repo, args.julia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
