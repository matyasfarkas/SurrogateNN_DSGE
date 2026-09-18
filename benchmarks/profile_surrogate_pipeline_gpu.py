#!/usr/bin/env python3
"""Profile the surrogate-SEP pipeline pieces that can run on JAX devices.

This is a staging profiler, not a claim that the full HLT nonlinear pipeline is
already GPU-native. It separates three costs:

1. HLT-shaped supervised ResNet training on JAX.
2. Current callback-based ROM/FOM dataset orchestration.
3. A small sparse-tree SEP Newton solve microbenchmark.

The first item should be GPU accelerated today. The second and third items are
included to expose the remaining bottleneck before spending GPUHub time on a
large run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp

from surrogatenn_dsge import (
    SEPConfig,
    SurrogateDataset,
    build_surrogate_residual_dataset,
    predict_frozen_batch,
    resolve_jax_device,
    solve_stochastic_extended_path_residual_expectation,
    summarize_surrogate_dataset,
    train_surrogate_from_dataset,
)


DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "surrogate_pipeline_gpu"


@dataclass(frozen=True)
class SyntheticHLTShape:
    state_dim: int = 40
    shock_dim: int = 7
    theta_dim: int = 18
    obs_dim: int = 7

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.shock_dim + self.theta_dim

    @property
    def output_dim(self) -> int:
        return self.obs_dim + self.state_dim


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _run_text(cmd: Sequence[str], *, check: bool = False) -> str:
    try:
        proc = subprocess.run(
            list(map(str, cmd)),
            check=check,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def environment_report() -> dict[str, Any]:
    smi_text = _run_text(["nvidia-smi"], check=False)
    return {
        "python": sys.version,
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "nvidia_smi": smi_text[-6000:] if smi_text else None,
    }


def estimate_dataset_memory_bytes(
    *,
    shape: SyntheticHLTShape,
    samples: int,
    dtype: np.dtype | type[np.floating[Any]] = np.float64,
) -> dict[str, int]:
    dtype_np = np.dtype(dtype)
    x_bytes = shape.input_dim * int(samples) * dtype_np.itemsize
    y_bytes = shape.output_dim * int(samples) * dtype_np.itemsize
    theta_ids_bytes = int(samples) * np.dtype(np.int64).itemsize
    return {
        "X": x_bytes,
        "Y": y_bytes,
        "Y_rom": y_bytes,
        "theta_ids": theta_ids_bytes,
        "total_core_arrays": x_bytes + 2 * y_bytes + theta_ids_bytes,
    }


def _stable_transition_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.normal(scale=0.12, size=(dim, dim))
    raw += 0.72 * np.eye(dim)
    radius = float(max(abs(np.linalg.eigvals(raw)))) if dim else 0.0
    if radius > 0.92:
        raw *= 0.92 / radius
    return raw.astype(np.float64)


def _mm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->ik", left, right, optimize=True)


def _mv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", left, right, optimize=True)


def make_synthetic_hlt_surrogate_dataset(
    *,
    samples: int,
    theta_draws: int,
    shape: SyntheticHLTShape = SyntheticHLTShape(),
    seed: int = 1234,
) -> SurrogateDataset:
    """Build a deterministic HLT-shaped ROM/FOM residual dataset.

    The map is intentionally synthetic; it profiles JAX training shape and memory
    pressure without pretending to be an economic validation of HLT dynamics.
    """

    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples}.")
    if theta_draws < 2:
        raise ValueError(f"theta_draws must be at least 2, got {theta_draws}.")
    if samples < theta_draws:
        raise ValueError("samples must be at least theta_draws so every theta appears.")

    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(low=-0.8, high=0.8, size=(shape.theta_dim, theta_draws))
    theta_ids = np.arange(samples, dtype=np.int64) % theta_draws
    period_ids = np.arange(samples, dtype=np.int64) // theta_draws
    theta_by_sample = theta[:, theta_ids]
    states = rng.normal(scale=0.75, size=(shape.state_dim, samples))
    shocks = rng.normal(scale=0.45, size=(shape.shock_dim, samples))

    obs_state = rng.normal(scale=0.08, size=(shape.obs_dim, shape.state_dim))
    obs_shock = rng.normal(scale=0.15, size=(shape.obs_dim, shape.shock_dim))
    state_transition = _stable_transition_matrix(rng, shape.state_dim)
    state_shock = rng.normal(scale=0.08, size=(shape.state_dim, shape.shock_dim))
    obs_theta = rng.normal(scale=0.04, size=(shape.obs_dim, shape.theta_dim))
    state_theta = rng.normal(scale=0.02, size=(shape.state_dim, shape.theta_dim))

    obs_rom = _mm(obs_state, states) + _mm(obs_shock, shocks)
    state_rom = _mm(state_transition, states) + _mm(state_shock, shocks)

    theta_signal = theta_by_sample[: min(6, shape.theta_dim), :]
    state_signal = states[: min(6, shape.state_dim), :]
    shock_signal = shocks[: min(6, shape.shock_dim), :]
    shared_nl = np.tanh(
        0.25 * np.sum(theta_signal, axis=0, keepdims=True)
        + 0.15 * np.sum(state_signal, axis=0, keepdims=True)
        + 0.20 * np.sum(shock_signal, axis=0, keepdims=True)
    )
    obs_resid = (
        0.05 * np.tanh(_mm(obs_theta, theta_by_sample))
        + 0.03 * shared_nl[:1, :]
        + 0.015 * np.sin(obs_rom)
    )
    state_resid = (
        0.04 * np.tanh(_mm(state_theta, theta_by_sample))
        + 0.02 * shared_nl[:1, :]
        + 0.01 * np.sin(state_rom)
    )
    y_rom = np.vstack([obs_rom, state_rom])
    y = np.vstack([obs_rom + obs_resid, state_rom + state_resid])
    x = np.vstack([states, shocks, theta_by_sample])

    counts = np.bincount(theta_ids, minlength=theta_draws)
    return SurrogateDataset(
        X=x,
        Y=y,
        Y_rom=y_rom,
        theta=theta,
        theta_ids=theta_ids,
        period_ids=period_ids,
        theta_success=np.ones((theta_draws,), dtype=bool),
        theta_stable_periods=counts.astype(np.int64),
        target_mode="fom_full",
        input_names=tuple(
            [f"s{i}" for i in range(shape.state_dim)]
            + [f"eps{i}" for i in range(shape.shock_dim)]
            + [f"theta{i}" for i in range(shape.theta_dim)]
        ),
        output_names=tuple(
            [f"obs{i}" for i in range(shape.obs_dim)]
            + [f"s_next{i}" for i in range(shape.state_dim)]
        ),
    )


def _block_until_ready_tree(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_training_profile(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    started = time.perf_counter()
    dataset = make_synthetic_hlt_surrogate_dataset(
        samples=args.samples,
        theta_draws=args.theta_draws,
        shape=shape,
        seed=args.seed,
    )
    dataset_s = time.perf_counter() - started
    train_started = time.perf_counter()
    result = train_surrogate_from_dataset(
        dataset,
        architecture="resnet",
        rom_residual=True,
        validation_fraction=args.validation_fraction,
        split_by_theta=args.split_by_theta,
        seed=args.seed,
        d_hidden=args.hidden,
        n_blocks=args.blocks,
        nepoch=args.epochs,
        eta_init=args.learning_rate,
        batch_size=args.batch_size,
        device=target_device,
    )
    _block_until_ready_tree(result.frozen)
    train_s = time.perf_counter() - train_started
    touched_samples = int(result.train_size) * int(args.epochs)
    return {
        "status": "ok",
        "kind": "synthetic_hlt_resnet_training",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "shape": shape.__dict__,
        "samples": int(args.samples),
        "theta_draws": int(args.theta_draws),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "blocks": int(args.blocks),
        "batch_size": int(args.batch_size),
        "dataset_build_s": dataset_s,
        "train_s": train_s,
        "train_size": int(result.train_size),
        "val_size": int(result.val_size),
        "train_sample_updates_per_s": touched_samples / train_s if train_s > 0 else math.inf,
        "validation_rmse_mean": None
        if result.validation_rmse is None
        else float(np.mean(result.validation_rmse)),
        "validation_rom_rmse_mean": None
        if result.validation_rmse_rom is None
        else float(np.mean(result.validation_rmse_rom)),
        "validation_improvement_mean": None
        if result.validation_improvement is None
        else float(np.nanmean(result.validation_improvement)),
        "dataset_summary": summarize_surrogate_dataset(dataset),
        "memory_estimate_bytes": estimate_dataset_memory_bytes(
            shape=shape,
            samples=args.samples,
            dtype=np.float64,
        ),
    }


def run_prediction_profile(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    dataset = make_synthetic_hlt_surrogate_dataset(
        samples=max(args.batch_size * 2, min(args.samples, 4096)),
        theta_draws=min(args.theta_draws, max(2, args.batch_size // 2)),
        shape=shape,
        seed=args.seed + 17,
    )
    result = train_surrogate_from_dataset(
        dataset,
        architecture="resnet",
        rom_residual=True,
        validation_fraction=0.0,
        seed=args.seed,
        d_hidden=args.hidden,
        n_blocks=args.blocks,
        nepoch=max(1, min(args.epochs, 2)),
        eta_init=args.learning_rate,
        batch_size=args.batch_size,
        device=target_device,
    )
    probe = dataset.X[:, : args.batch_size]
    X_probe = jnp.asarray(probe, dtype=jnp.float64)
    if target_device is not None:
        X_probe = jax.device_put(X_probe, target_device)

    @jax.jit
    def predict_once(x_batch: jax.Array) -> jax.Array:
        return predict_frozen_batch(result.frozen, x_batch)

    t0 = time.perf_counter()
    y0 = predict_once(X_probe)
    y0.block_until_ready()
    first_s = time.perf_counter() - t0
    reps = max(1, int(args.predict_reps))
    steady: list[float] = []
    for _ in range(reps):
        t1 = time.perf_counter()
        y = predict_once(X_probe)
        y.block_until_ready()
        steady.append(time.perf_counter() - t1)
    median_s = float(statistics.median(steady))
    return {
        "status": "ok",
        "kind": "synthetic_hlt_resnet_prediction",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "batch_size": int(args.batch_size),
        "first_call_s": first_s,
        "steady_median_s": median_s,
        "steady_samples_per_s": args.batch_size / median_s if median_s > 0 else math.inf,
        "output_norm": float(np.linalg.norm(np.asarray(y0))),
    }


def run_callback_dataset_profile(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    theta_draws = max(2, min(int(args.callback_theta_draws), int(args.theta_draws)))
    periods = max(1, int(args.callback_periods))
    rng = np.random.default_rng(int(args.seed) + 31)
    theta = rng.uniform(-0.5, 0.5, size=(shape.theta_dim, theta_draws))
    shocks = rng.normal(scale=0.4, size=(theta_draws, shape.shock_dim, periods))
    initial_state = rng.normal(scale=0.2, size=(shape.state_dim,))
    obs_state = rng.normal(scale=0.05, size=(shape.obs_dim, shape.state_dim))
    obs_shock = rng.normal(scale=0.10, size=(shape.obs_dim, shape.shock_dim))
    state_transition = _stable_transition_matrix(rng, shape.state_dim)
    state_shock = rng.normal(scale=0.05, size=(shape.state_dim, shape.shock_dim))
    obs_theta = rng.normal(scale=0.03, size=(shape.obs_dim, shape.theta_dim))

    def rom_predict(state: Any, shock_t: Any, _theta: Any) -> tuple[np.ndarray, np.ndarray]:
        state_arr = np.asarray(state, dtype=np.float64)
        shock_arr = np.asarray(shock_t, dtype=np.float64)
        obs = _mv(obs_state, state_arr) + _mv(obs_shock, shock_arr)
        next_state = _mv(state_transition, state_arr) + _mv(state_shock, shock_arr)
        return obs, next_state

    def fom_predict(state: Any, shock_t: Any, theta_t: Any) -> tuple[np.ndarray, np.ndarray]:
        obs, next_state = rom_predict(state, shock_t, theta_t)
        theta_arr = np.asarray(theta_t, dtype=np.float64)
        signal = float(np.tanh(0.2 * np.sum(theta_arr[: min(6, theta_arr.size)])))
        obs_resid = 0.04 * np.tanh(_mv(obs_theta, theta_arr)) + 0.02 * signal
        state_resid = 0.01 * np.sin(next_state) + 0.01 * signal
        return obs + obs_resid, next_state + state_resid

    started = time.perf_counter()
    dataset = build_surrogate_residual_dataset(
        rom_predict,
        fom_predict,
        initial_state=initial_state,
        shocks=shocks,
        theta_design=theta,
        target_mode="fom_full",
        min_stable_periods=periods,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "kind": "callback_dataset_orchestration",
        "theta_draws": theta_draws,
        "periods": periods,
        "samples": int(dataset.n_samples),
        "elapsed_s": elapsed,
        "samples_per_s": dataset.n_samples / elapsed if elapsed > 0 else math.inf,
        "dataset_summary": summarize_surrogate_dataset(dataset),
    }


def run_sep_micro_profile(args: argparse.Namespace) -> dict[str, Any]:
    state_dim = int(args.sep_state_dim)
    shock_dim = int(args.sep_shock_dim)
    rng = np.random.default_rng(int(args.seed) + 53)
    B = jnp.asarray(rng.normal(scale=0.06, size=(state_dim, shock_dim)), dtype=jnp.float64)
    rho = jnp.asarray(0.84, dtype=jnp.float64)
    gamma = jnp.asarray(0.05, dtype=jnp.float64)

    def conditional_residual(
        prev_state: jax.Array,
        current_state: jax.Array,
        next_state: jax.Array,
        current_shock: jax.Array,
        _params: object,
    ) -> jax.Array:
        target = rho * prev_state + gamma * jnp.tanh(next_state) + B @ current_shock
        return current_state - target

    config = SEPConfig(
        periods=int(args.sep_periods),
        branching_order=int(args.sep_order),
        nnodes=int(args.sep_nnodes),
        sparse_tree=bool(args.sep_sparse_tree),
        max_iter=int(args.sep_max_iter),
        tol=float(args.sep_tol),
        accept_tol=float(args.sep_accept_tol),
        line_search=True,
    )
    initial_state = np.zeros((state_dim,), dtype=np.float64)
    terminal_state = np.zeros((state_dim,), dtype=np.float64)
    deterministic = np.zeros((config.periods, shock_dim), dtype=np.float64)
    times: list[float] = []
    last_solution: Any = None
    reps = max(1, int(args.sep_reps))
    for _ in range(reps):
        started = time.perf_counter()
        last_solution = solve_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=initial_state,
            terminal_state=terminal_state,
            shock_dim=shock_dim,
            deterministic_shocks=deterministic,
            config=config,
            params=None,
        )
        _block_until_ready_tree(last_solution)
        times.append(time.perf_counter() - started)
    assert last_solution is not None
    return {
        "status": "ok",
        "kind": "sep_sparse_tree_microbenchmark",
        "backend": jax.default_backend(),
        "state_dim": state_dim,
        "shock_dim": shock_dim,
        "periods": int(config.periods),
        "branching_order": int(config.branching_order),
        "nnodes": int(config.nnodes),
        "sparse_tree": bool(config.sparse_tree),
        "reps": reps,
        "first_s": float(times[0]),
        "median_s": float(statistics.median(times)),
        "residual_norm": float(last_solution.residual_norm),
        "accepted": bool(last_solution.accepted),
        "converged": bool(last_solution.converged),
        "iterations": int(last_solution.iterations),
        "group_counts": [int(x) for x in last_solution.group_counts],
        "jacobian_method": str(last_solution.jacobian_method),
    }


def scenario_defaults(mode: str) -> dict[str, Any]:
    if mode == "calibration":
        return {
            "samples": 4096,
            "theta_draws": 64,
            "epochs": 3,
            "batch_size": 512,
            "hidden": 64,
            "blocks": 2,
            "predict_reps": 5,
        }
    if mode == "training-scale":
        return {
            "samples": 200_000,
            "theta_draws": 1_000,
            "epochs": 20,
            "batch_size": 4096,
            "hidden": 192,
            "blocks": 4,
            "predict_reps": 20,
        }
    if mode == "full-scout":
        return {
            "samples": 1_000_000,
            "theta_draws": 5_000,
            "epochs": 40,
            "batch_size": 8192,
            "hidden": 256,
            "blocks": 5,
            "predict_reps": 20,
        }
    raise ValueError(f"Unknown scenario {mode!r}.")


def _apply_scenario_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = scenario_defaults(args.mode) if args.mode in {"calibration", "training-scale", "full-scout"} else {}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.samples is None:
        args.samples = 4096
    if args.theta_draws is None:
        args.theta_draws = 64
    if args.epochs is None:
        args.epochs = 3
    if args.batch_size is None:
        args.batch_size = 512
    if args.hidden is None:
        args.hidden = 64
    if args.blocks is None:
        args.blocks = 2
    if args.predict_reps is None:
        args.predict_reps = 5
    return args


def build_plan(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    return {
        "status": "plan",
        "shape": shape.__dict__,
        "requested_mode": args.mode,
        "samples": int(args.samples),
        "theta_draws": int(args.theta_draws),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden": int(args.hidden),
        "blocks": int(args.blocks),
        "memory_estimate_bytes_float64": estimate_dataset_memory_bytes(
            shape=shape,
            samples=args.samples,
            dtype=np.float64,
        ),
        "memory_estimate_bytes_float32": estimate_dataset_memory_bytes(
            shape=shape,
            samples=args.samples,
            dtype=np.float32,
        ),
        "caveats": [
            "The current surrogate training implementation stores and trains in float64.",
            "The current SEP dataset-generation API is callback/Python-loop based, not a batched JAX SEP kernel.",
            "This profiler can validate GPU training throughput now; it cannot certify full HLT SEP generation speedup yet.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "calibration", "training-scale", "full-scout", "sep-micro", "callback-dataset"),
        default="calibration",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--theta-draws", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-by-theta", action="store_true")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--predict-reps", type=int)
    parser.add_argument("--state-dim", type=int, default=40)
    parser.add_argument("--shock-dim", type=int, default=7)
    parser.add_argument("--theta-dim", type=int, default=18)
    parser.add_argument("--obs-dim", type=int, default=7)
    parser.add_argument("--callback-theta-draws", type=int, default=16)
    parser.add_argument("--callback-periods", type=int, default=32)
    parser.add_argument("--sep-state-dim", type=int, default=8)
    parser.add_argument("--sep-shock-dim", type=int, default=3)
    parser.add_argument("--sep-periods", type=int, default=4)
    parser.add_argument("--sep-order", type=int, default=1)
    parser.add_argument("--sep-nnodes", type=int, default=3)
    parser.add_argument("--sep-sparse-tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sep-max-iter", type=int, default=8)
    parser.add_argument("--sep-tol", type=float, default=1e-8)
    parser.add_argument("--sep-accept-tol", type=float, default=1e-5)
    parser.add_argument("--sep-reps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return _apply_scenario_defaults(parser.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU was required, but JAX default backend is {jax.default_backend()!r}.")

    shape = SyntheticHLTShape(
        state_dim=int(args.state_dim),
        shock_dim=int(args.shock_dim),
        theta_dim=int(args.theta_dim),
        obs_dim=int(args.obs_dim),
    )
    payload: dict[str, Any] = {
        "environment": environment_report(),
        "plan": build_plan(args, shape),
        "results": {},
    }
    if args.mode == "plan":
        pass
    elif args.mode in {"calibration", "training-scale", "full-scout"}:
        payload["results"]["training"] = run_training_profile(args, shape)
        payload["results"]["prediction"] = run_prediction_profile(args, shape)
        payload["results"]["callback_dataset"] = run_callback_dataset_profile(args, shape)
        if args.mode == "calibration":
            payload["results"]["sep_micro"] = run_sep_micro_profile(args)
    elif args.mode == "sep-micro":
        payload["results"]["sep_micro"] = run_sep_micro_profile(args)
    elif args.mode == "callback-dataset":
        payload["results"]["callback_dataset"] = run_callback_dataset_profile(args, shape)
    else:
        raise ValueError(f"Unsupported mode {args.mode!r}.")

    out_path = args.output
    if out_path is None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out_path = DEFAULT_RESULTS_DIR / f"{args.mode}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
