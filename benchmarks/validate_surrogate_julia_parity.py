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
    FrozenMLP,
    FrozenResNet,
    NormStats,
    ResBlock,
    predict_frozen,
    predict_frozen_batch,
)


def _python_objects() -> tuple[FrozenMLP, FrozenResNet, np.ndarray, np.ndarray]:
    mlp_norm = NormStats([1.0, -2.0, 0.5], [2.0, 0.5, 4.0], [-0.25, 1.5], [0.7, 2.0])
    mlp = FrozenMLP(
        W1=np.asarray(
            [
                [0.10, -0.20, 0.30],
                [0.40, 0.05, -0.10],
                [-0.30, 0.20, 0.25],
                [0.15, -0.35, 0.05],
            ],
            dtype=np.float64,
        ),
        b1=np.asarray([0.01, -0.02, 0.03, 0.04], dtype=np.float64),
        W2=np.asarray(
            [
                [0.30, -0.10, 0.20, 0.05],
                [-0.25, 0.15, 0.10, -0.20],
            ],
            dtype=np.float64,
        ),
        b2=np.asarray([0.07, -0.04], dtype=np.float64),
        W3=None,
        b3=None,
        norm=mlp_norm,
        d_in=3,
        d_out=2,
        activation="silu",
    )

    res_norm = NormStats(np.zeros(5), np.ones(5), np.zeros(2), np.ones(2))
    block = ResBlock(
        W1=0.05 * np.eye(4),
        b1=np.asarray([0.01, -0.02, 0.03, -0.04]),
        W2=0.03 * np.eye(4),
        b2=np.asarray([0.02, 0.01, -0.01, -0.02]),
    )
    resnet = FrozenResNet(
        W_embed=np.asarray(
            [
                [0.2, -0.1, 0.05],
                [0.0, 0.3, -0.2],
                [0.1, 0.1, 0.1],
                [-0.2, 0.05, 0.25],
            ],
            dtype=np.float64,
        ),
        b_embed=np.asarray([0.01, -0.01, 0.02, -0.02]),
        d_theta=2,
        W_gamma=np.asarray([[0.01, 0.02], [-0.03, 0.01], [0.02, -0.02], [0.01, 0.0]]),
        b_gamma=np.ones(4),
        W_beta=np.asarray([[0.02, -0.01], [0.01, 0.03], [0.0, -0.02], [-0.01, 0.02]]),
        b_beta=np.zeros(4),
        blocks=(block,),
        W_out=np.asarray([[0.2, -0.1, 0.05, 0.01], [-0.05, 0.03, 0.2, -0.1]]),
        b_out=np.asarray([0.01, -0.02]),
        norm=res_norm,
        d_in=5,
        d_out=2,
    )
    x_mlp = np.asarray([0.2, -1.75, 2.5], dtype=np.float64)
    X_resnet = np.asarray(
        [
            [0.1, 0.4, -0.2],
            [0.0, -0.3, 0.5],
            [0.2, 0.1, -0.1],
            [0.6, 0.7, 0.8],
            [-0.4, -0.5, -0.6],
        ],
        dtype=np.float64,
    )
    return mlp, resnet, x_mlp, X_resnet


def _julia_script(julia_repo: Path) -> str:
    repo_literal = json.dumps(str(julia_repo))
    return f"""
include(joinpath({repo_literal}, "scripts", "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

function print_vec(label, x)
    println(label * "=" * join(string.(vec(x)), ","))
end

W1 = [0.10 -0.20 0.30;
      0.40 0.05 -0.10;
      -0.30 0.20 0.25;
      0.15 -0.35 0.05]
b1 = [0.01, -0.02, 0.03, 0.04]
W2 = [0.30 -0.10 0.20 0.05;
      -0.25 0.15 0.10 -0.20]
b2 = [0.07, -0.04]
norm = NormStats([1.0, -2.0, 0.5], [2.0, 0.5, 4.0], [-0.25, 1.5], [0.7, 2.0])
f = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, 3, 2, :silu)
x = [0.2, -1.75, 2.5]
X = hcat(x, x .+ [0.1, -0.2, 0.3])
print_vec("mlp_single", predict_frozen(f, x))
print_vec("mlp_batch", predict_frozen_batch(f, X))

blk = ResBlock(0.05 .* Matrix(I, 4, 4), [0.01, -0.02, 0.03, -0.04],
               0.03 .* Matrix(I, 4, 4), [0.02, 0.01, -0.01, -0.02])
res = FrozenResNet(
    [0.2 -0.1 0.05;
     0.0 0.3 -0.2;
     0.1 0.1 0.1;
     -0.2 0.05 0.25],
    [0.01, -0.01, 0.02, -0.02],
    2,
    [0.01 0.02; -0.03 0.01; 0.02 -0.02; 0.01 0.0],
    ones(4),
    [0.02 -0.01; 0.01 0.03; 0.0 -0.02; -0.01 0.02],
    zeros(4),
    [blk],
    [0.2 -0.1 0.05 0.01; -0.05 0.03 0.2 -0.1],
    [0.01, -0.02],
    NormStats(zeros(5), ones(5), zeros(2), ones(2)),
    5,
    2,
)
Xr = [0.1 0.4 -0.2;
      0.0 -0.3 0.5;
      0.2 0.1 -0.1;
      0.6 0.7 0.8;
      -0.4 -0.5 -0.6]
print_vec("resnet_batch", predict_frozen_batch(res, Xr))
"""


def _parse_output(stdout: str) -> dict[str, np.ndarray]:
    parsed: dict[str, np.ndarray] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"mlp_single", "mlp_batch", "resnet_batch"}:
            parsed[key] = np.asarray([float(x) for x in value.split(",") if x], dtype=np.float64)
    return parsed


def run_parity(julia_repo: Path, julia_exe: str = "julia") -> None:
    if shutil.which(julia_exe) is None:
        raise RuntimeError(f"Julia executable not found: {julia_exe}")
    if not julia_repo.exists():
        raise RuntimeError(f"Julia repo not found: {julia_repo}")

    mlp, resnet, x_mlp, X_resnet = _python_objects()
    X_mlp = np.column_stack([x_mlp, x_mlp + np.asarray([0.1, -0.2, 0.3])])
    expected = {
        "mlp_single": np.asarray(predict_frozen(mlp, x_mlp), dtype=np.float64).reshape(-1),
        "mlp_batch": np.asarray(predict_frozen_batch(mlp, X_mlp), dtype=np.float64).reshape(-1, order="F"),
        "resnet_batch": np.asarray(predict_frozen_batch(resnet, X_resnet), dtype=np.float64).reshape(-1, order="F"),
    }

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "surrogate_parity.jl"
        script_path.write_text(_julia_script(julia_repo), encoding="utf-8")
        proc = subprocess.run(
            [julia_exe, f"--project={julia_repo}", str(script_path)],
            text=True,
            capture_output=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Julia parity script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    observed = _parse_output(proc.stdout)
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RuntimeError(f"Julia parity script did not emit: {missing}\nSTDOUT:\n{proc.stdout}")
    for key, exp in expected.items():
        np.testing.assert_allclose(observed[key], exp, rtol=1e-11, atol=1e-11, err_msg=key)
    print("Surrogate Julia parity passed for:", ", ".join(sorted(expected)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Python/JAX surrogate predictions against Julia utilities.")
    parser.add_argument("--julia-repo", type=Path, default=DEFAULT_JULIA_REPO)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    run_parity(args.julia_repo, args.julia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
