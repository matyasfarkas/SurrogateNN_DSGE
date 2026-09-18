from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax

from surrogatenn_dsge import (
    SurrogateDataset,
    load_surrogate_bundle,
    predict_frozen_batch,
    save_surrogate_bundle,
    split_surrogate_dataset,
    train_surrogate_from_dataset,
)


JULIA_SOURCE_DEFAULT = Path("/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl")
JULIA_NN_UTILS = JULIA_SOURCE_DEFAULT / "scripts" / "hlt_surrogate" / "hlt_sep_surrogate_nn_utils.jl"
RESULT_SENTINEL = "__SURROGATE_TRAINING_SMOKE_JSON__"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_dataset(*, theta_draws: int, periods: int, seed: int) -> SurrogateDataset:
    rng = np.random.default_rng(seed)
    d_state_shock = 5
    d_theta = 3
    d_out = 3
    theta = rng.uniform(
        low=np.asarray([-0.7, 0.2, 0.8])[:, None],
        high=np.asarray([0.7, 1.4, 1.8])[:, None],
        size=(d_theta, theta_draws),
    )
    X_cols: list[np.ndarray] = []
    Y_cols: list[np.ndarray] = []
    Y_rom_cols: list[np.ndarray] = []
    theta_ids: list[int] = []
    period_ids: list[int] = []
    for theta_id in range(theta_draws):
        theta_t = theta[:, theta_id]
        state = rng.normal(size=3)
        for period in range(periods):
            shock = rng.normal(scale=0.6, size=2)
            state_shock = np.concatenate([state, shock])
            rom = np.asarray(
                [
                    0.30 * state[0] + 0.08 * state[1] + shock[0],
                    -0.20 * state[2] + 0.15 * shock[1],
                    0.10 * state[0] - 0.15 * state[1] + 0.05 * shock[0],
                ],
                dtype=np.float64,
            )
            residual = np.asarray(
                [
                    0.35 * theta_t[0] + 0.18 * state[0] * theta_t[1] + 0.06 * shock[0] ** 2,
                    -0.22 * theta_t[1] + 0.12 * state[1] * theta_t[0] + 0.04 * shock[1] ** 2,
                    0.16 * theta_t[2] + 0.10 * state[2] * theta_t[0] - 0.03 * shock[0] * shock[1],
                ],
                dtype=np.float64,
            )
            X_cols.append(np.concatenate([state_shock, theta_t]))
            Y_rom_cols.append(rom)
            Y_cols.append(rom + residual)
            theta_ids.append(theta_id)
            period_ids.append(period)
            state = np.asarray(
                [
                    0.65 * state[0] + 0.10 * state[1] + shock[0],
                    0.20 * state[0] + 0.55 * state[1] + 0.3 * shock[1],
                    0.70 * state[2] + 0.10 * shock[0] - 0.05 * shock[1],
                ],
                dtype=np.float64,
            )
    return SurrogateDataset(
        X=np.column_stack(X_cols),
        Y=np.column_stack(Y_cols),
        Y_rom=np.column_stack(Y_rom_cols),
        theta=theta,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=np.ones((theta_draws,), dtype=bool),
        theta_stable_periods=np.full((theta_draws,), periods, dtype=np.int64),
        target_mode="fom_obs",
        theta_names=("theta_0", "theta_1", "theta_2"),
    )


def _block_frozen(frozen: Any) -> None:
    if hasattr(frozen, "W1"):
        jax.block_until_ready(frozen.W1)
    elif hasattr(frozen, "W_embed"):
        jax.block_until_ready(frozen.W_embed)


def run_python_case(dataset: SurrogateDataset, args: argparse.Namespace, architecture: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="surrogatenn_smoke_py_") as tmp:
        tmpdir = Path(tmp)
        start = time.perf_counter()
        result = train_surrogate_from_dataset(
            dataset,
            architecture=architecture,
            rom_residual=True,
            validation_fraction=args.validation_fraction,
            split_by_theta=args.split_by_theta,
            seed=args.seed,
            d_hidden=args.hidden,
            d_hidden2=None,
            n_blocks=args.blocks,
            nepoch=args.epochs,
            eta_init=args.learning_rate,
            batch_size=args.batch_size,
            device=None if args.device == "auto" else args.device,
        )
        _block_frozen(result.frozen)
        train_s = time.perf_counter() - start
        X_probe = dataset.X[:, : min(8, dataset.n_samples)]
        pred_before = np.asarray(predict_frozen_batch(result.frozen, X_probe), dtype=np.float64)
        bundle_path = save_surrogate_bundle(tmpdir / f"{architecture}_bundle.snn", result)
        loaded = load_surrogate_bundle(bundle_path, device=None if args.device == "auto" else args.device)
        pred_after = np.asarray(predict_frozen_batch(loaded.frozen, X_probe), dtype=np.float64)
        roundtrip_max_abs = float(np.max(np.abs(pred_before - pred_after)))
        return {
            "status": "ok",
            "architecture": architecture,
            "device_request": args.device,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "train_s": train_s,
            "train_size": result.train_size,
            "val_size": result.val_size,
            "validation_rmse": None if result.validation_rmse is None else result.validation_rmse,
            "validation_rmse_rom": None if result.validation_rmse_rom is None else result.validation_rmse_rom,
            "validation_improvement": None if result.validation_improvement is None else result.validation_improvement,
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_roundtrip_max_abs": roundtrip_max_abs,
        }


def _write_csv(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",")


def _julia_literal(path: Path) -> str:
    return json.dumps(str(path))


def _julia_script(
    *,
    nn_utils_path: Path,
    x_train_path: Path,
    y_train_path: Path,
    x_probe_path: Path,
    architecture: str,
    hidden: int,
    blocks: int,
    d_theta: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> str:
    if architecture == "mlp":
        train_call = (
            "f = train_mlp!(copy(X), copy(Y); "
            f"d_hidden={hidden}, d_hidden2=nothing, nepoch={epochs}, batch_size={batch_size}, "
            f"seed={seed}, verbose=false, activation=:silu)"
        )
    else:
        train_call = (
            "f = train_resnet!(copy(X), copy(Y); "
            f"d_hidden={hidden}, n_blocks={blocks}, d_theta={d_theta}, nepoch={epochs}, "
            f"batch_size={batch_size}, seed={seed}, verbose=false)"
        )
    return f"""
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
include({_julia_literal(nn_utils_path)})
X = Matrix{{Float64}}(readdlm({_julia_literal(x_train_path)}, ',', Float64))
Y = Matrix{{Float64}}(readdlm({_julia_literal(y_train_path)}, ',', Float64))
X_probe = Matrix{{Float64}}(readdlm({_julia_literal(x_probe_path)}, ',', Float64))
Random.seed!({seed})
t0 = time()
{train_call}
train_s = time() - t0
pred_train = predict_frozen_batch(f, X)
pred_probe = predict_frozen_batch(f, X_probe)
rmse = sqrt(mean((pred_train .- Y) .^ 2))
probe_l2 = norm(vec(pred_probe))
println("{RESULT_SENTINEL}" * "{{\\"status\\":\\"ok\\",\\"architecture\\":\\"{architecture}\\",\\"train_s\\":" * string(train_s) * ",\\"train_rmse\\":" * string(rmse) * ",\\"probe_l2\\":" * string(probe_l2) * "}}")
"""


def run_julia_case(dataset: SurrogateDataset, args: argparse.Namespace, architecture: str) -> dict[str, Any]:
    source_root = Path(args.julia_source)
    nn_utils = source_root / "scripts" / "hlt_surrogate" / "hlt_sep_surrogate_nn_utils.jl"
    if not nn_utils.is_file():
        return {"status": "skipped", "reason": f"Julia NN utility not found: {nn_utils}", "architecture": architecture}
    split = split_surrogate_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        split_by_theta=args.split_by_theta,
        seed=args.seed,
    )
    y_target = dataset.Y - dataset.Y_rom
    with tempfile.TemporaryDirectory(prefix="surrogatenn_smoke_julia_") as tmp:
        tmpdir = Path(tmp)
        x_train_path = tmpdir / "X_train.csv"
        y_train_path = tmpdir / "Y_train.csv"
        x_probe_path = tmpdir / "X_probe.csv"
        script_path = tmpdir / "run_julia_training_smoke.jl"
        _write_csv(x_train_path, dataset.X[:, split.train_idx])
        _write_csv(y_train_path, y_target[:, split.train_idx])
        _write_csv(x_probe_path, dataset.X[:, : min(8, dataset.n_samples)])
        script_path.write_text(
            _julia_script(
                nn_utils_path=nn_utils,
                x_train_path=x_train_path,
                y_train_path=y_train_path,
                x_probe_path=x_probe_path,
                architecture=architecture,
                hidden=args.hidden,
                blocks=args.blocks,
                d_theta=dataset.theta.shape[0],
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            ),
            encoding="utf-8",
        )
        cmd = ["julia", f"--project={source_root}", str(script_path)]
        start = time.perf_counter()
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        wall_s = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "status": "error",
            "architecture": architecture,
            "returncode": proc.returncode,
            "wall_s": wall_s,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            result_line = line[len(RESULT_SENTINEL) :]
    if result_line is None:
        return {
            "status": "error",
            "architecture": architecture,
            "wall_s": wall_s,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "reason": "Julia result sentinel not found.",
        }
    result = json.loads(result_line)
    result["wall_s"] = wall_s
    result["train_size"] = int(split.train_size)
    result["val_size"] = int(split.val_size)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-batch surrogate NN training smoke benchmark.")
    parser.add_argument("--mode", choices=("python", "julia", "both"), default="both")
    parser.add_argument("--architecture", choices=("mlp", "resnet", "both"), default="both")
    parser.add_argument("--device", default="auto", help="Python/JAX device selector: auto, cpu, gpu, or a JAX device.")
    parser.add_argument("--theta-draws", type=int, default=4)
    parser.add_argument("--periods", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--split-by-theta", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--julia-source", default=str(JULIA_SOURCE_DEFAULT))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.theta_draws < 2:
        raise ValueError("--theta-draws must be at least 2.")
    if args.periods < 2:
        raise ValueError("--periods must be at least 2.")
    dataset = make_dataset(theta_draws=args.theta_draws, periods=args.periods, seed=args.seed)
    architectures = ["mlp", "resnet"] if args.architecture == "both" else [args.architecture]
    output: dict[str, Any] = {
        "benchmark": {
            "name": "surrogate_training_smoke",
            "mode": args.mode,
            "architectures": architectures,
            "theta_draws": args.theta_draws,
            "periods": args.periods,
            "samples": dataset.n_samples,
            "input_dim": int(dataset.X.shape[0]),
            "output_dim": int(dataset.Y.shape[0]),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "python": {},
        "julia": {},
    }
    for architecture in architectures:
        if args.mode in {"python", "both"}:
            try:
                output["python"][architecture] = run_python_case(dataset, args, architecture)
            except Exception as exc:
                output["python"][architecture] = {"status": "error", "architecture": architecture, "error": repr(exc)}
        if args.mode in {"julia", "both"}:
            output["julia"][architecture] = run_julia_case(dataset, args, architecture)
    output = _jsonable(output)
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    failed = False
    for section in ("python", "julia"):
        for result in output.get(section, {}).values():
            failed = failed or result.get("status") == "error"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
