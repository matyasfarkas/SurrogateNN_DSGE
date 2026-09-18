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

from surrogatenn_dsge import lhs_to_bounds, parameter_grid  # noqa: E402


def _julia_script(julia_repo: Path) -> str:
    repo_literal = json.dumps(str(julia_repo))
    return f"""
include(joinpath({repo_literal}, "scripts", "hlt_surrogate", "parameter_config.jl"))

function print_vec(label, x)
    println(label * "=" * join(string.(vec(x)), ","))
end

function lhs_to_bounds(lhs_sample::Matrix{{Float64}},
                       bounds_dict::Dict{{Symbol, Tuple{{Float64, Float64}}}},
                       param_names::Vector{{Symbol}})
    d, n = size(lhs_sample)
    transformed = similar(lhs_sample)

    for (i, name) in enumerate(param_names)
        lb, ub = bounds_dict[name]
        transformed[i, :] = lb .+ (ub - lb) .* lhs_sample[i, :]
    end

    return transformed
end

unit = [0.0 0.5 1.0;
        0.25 0.75 0.5;
        1.0 0.0 0.5]
legacy_bounds = get_parameter_bounds(:legacy_3params)
legacy_names = get_parameter_names(:legacy_3params)
print_vec("legacy_lhs_to_bounds", lhs_to_bounds(unit, legacy_bounds, legacy_names))

cprobp_grid = range(0.5, 0.95, length = 5)
cindp_grid = range(0.01, 0.99, length = 5)
curvp_grid = range(20.0, 150.0, length = 5)
theta_grid = Vector{{Vector{{Float64}}}}()
for cprobp in cprobp_grid, cindp in cindp_grid, curvp in curvp_grid
    push!(theta_grid, [cprobp, cindp, curvp])
end
grid_matrix = hcat(theta_grid...)
print_vec("legacy_grid", grid_matrix)
"""


def _parse_output(stdout: str) -> dict[str, np.ndarray]:
    parsed: dict[str, np.ndarray] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"legacy_lhs_to_bounds", "legacy_grid"}:
            parsed[key] = np.asarray([float(x) for x in value.split(",") if x], dtype=np.float64)
    return parsed


def run_parity(julia_repo: Path, julia_exe: str = "julia") -> None:
    if shutil.which(julia_exe) is None:
        raise RuntimeError(f"Julia executable not found: {julia_exe}")
    if not julia_repo.exists():
        raise RuntimeError(f"Julia repo not found: {julia_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "parameter_sampling_parity.jl"
        script_path.write_text(_julia_script(julia_repo), encoding="utf-8")
        proc = subprocess.run(
            [julia_exe, f"--project={julia_repo}", str(script_path)],
            text=True,
            capture_output=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Julia sampling parity failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    observed = _parse_output(proc.stdout)
    missing = {"legacy_lhs_to_bounds", "legacy_grid"} - set(observed)
    if missing:
        raise RuntimeError(f"Julia sampling parity did not emit {sorted(missing)}:\n{proc.stdout}")

    unit = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.5],
            [1.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    expected_lhs = lhs_to_bounds(unit, "legacy_3params").reshape(-1, order="F")
    expected_grid = parameter_grid("legacy_3params", points_per_dim=5).theta.reshape(-1, order="F")
    np.testing.assert_allclose(observed["legacy_lhs_to_bounds"], expected_lhs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(observed["legacy_grid"], expected_grid, rtol=0, atol=1e-12)
    print("Parameter-sampling Julia parity passed for lhs_to_bounds and legacy grid.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Python parameter sampling helpers against Julia reference.")
    parser.add_argument("--julia-repo", type=Path, default=DEFAULT_JULIA_REPO)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    run_parity(args.julia_repo, args.julia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
