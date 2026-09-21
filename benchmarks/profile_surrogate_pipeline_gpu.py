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
    fit_surrogate_pipeline,
    parse_macro_model,
    predict_frozen_batch,
    resolve_jax_device,
    solve_stochastic_extended_path_residual_expectation,
    summarize_surrogate_dataset,
    surrogate_inversion_loglik_per_period,
    surrogate_inversion_loglikelihood_jax,
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


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _load_hlt_payload_case(args: argparse.Namespace) -> dict[str, Any]:
    payload_path = Path(args.hlt_payload)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    for case in cases:
        if case.get("name") == args.hlt_case_name:
            return dict(case)
    raise ValueError(f"Could not find case {args.hlt_case_name!r} in {payload_path}.")


def _make_hlt_theta_design(
    *,
    base_parameters: np.ndarray,
    parameter_names: Sequence[str],
    subset_names: Sequence[str],
    draws: int,
    perturbation: float,
) -> tuple[np.ndarray, list[int]]:
    if draws < 1:
        raise ValueError(f"hlt_theta_draws must be >= 1, got {draws}.")
    subset_idx = [tuple(parameter_names).index(name) for name in subset_names]
    base_subset = np.asarray(base_parameters[subset_idx], dtype=np.float64)
    theta = np.repeat(base_subset[:, None], int(draws), axis=1)
    if draws > 1 and perturbation != 0.0:
        grid = np.linspace(-1.0, 1.0, int(draws), dtype=np.float64)
        signs = np.where(np.arange(base_subset.size) % 2 == 0, 1.0, -1.0)
        theta *= 1.0 + float(perturbation) * signs[:, None] * grid[None, :]
    return theta, subset_idx


def run_hlt_fixed_steady_state_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Run the actual HLT model through a tiny ROM/FOM surrogate path.

    This is intentionally a smoke/profile mode. It verifies that the parsed HLT
    model, first-order ROM, SEP FOM target generation, JAX surrogate training,
    and trained-surrogate likelihood evaluation compose on the selected device.
    Use ``--hlt-steady-state-mode solve`` to require parameter-specific steady
    states; the default fixed-reference mode is a conservative stability check.
    """

    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    case = _load_hlt_payload_case(args)
    model_source = Path(args.hlt_model_source)
    started = time.perf_counter()
    model = parse_macro_model(model_source.read_text(encoding="utf-8"))
    parse_s = time.perf_counter() - started

    reference_steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    base_parameters = np.asarray(model.parameter_values, dtype=np.float64)
    parameter_subset = [str(name) for name in case["parameter_subset"]]
    theta, subset_idx = _make_hlt_theta_design(
        base_parameters=base_parameters,
        parameter_names=model.parameter_names,
        subset_names=parameter_subset,
        draws=int(args.hlt_theta_draws),
        perturbation=float(args.hlt_parameter_perturbation),
    )

    periods = int(args.hlt_periods)
    shock_dim = int(model.timings.nExo)
    shocks = np.zeros((theta.shape[1], shock_dim, periods), dtype=np.float64)
    if shock_dim > 0 and periods > 0:
        shocks[:, 0, 0] = float(args.hlt_shock_scale)
    if shock_dim > 1 and periods > 1:
        shocks[:, 1, 1] = -0.5 * float(args.hlt_shock_scale)

    observables = [str(name) for name in case["observables"]]
    observable_idx = [model.timings.var.index(name) for name in observables]
    state_idx = np.asarray(model.timings.past_not_future_and_mixed_idx, dtype=np.int64)
    sep_config = SEPConfig(
        periods=int(args.sep_periods),
        branching_order=int(args.sep_order),
        nnodes=int(args.sep_nnodes),
        sparse_tree=bool(args.sep_sparse_tree),
        max_iter=int(args.sep_max_iter),
        tol=float(args.sep_tol),
        accept_tol=float(args.sep_accept_tol),
    )
    steady_state_mode = str(args.hlt_steady_state_mode).strip().lower()
    if steady_state_mode not in {"fixed-reference", "solve", "solve-or-reference"}:
        raise ValueError(
            "hlt_steady_state_mode must be 'fixed-reference', 'solve', or "
            f"'solve-or-reference', got {args.hlt_steady_state_mode!r}."
        )

    def full_parameters(theta_t: Any) -> np.ndarray:
        values = base_parameters.copy()
        values[subset_idx] = np.asarray(theta_t, dtype=np.float64)
        return values

    first_order_s = 0.0
    steady_state_s = 0.0
    runtime_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    steady_state_diagnostics: list[dict[str, Any]] = []

    def theta_key(theta_t: Any) -> tuple[float, ...]:
        theta_arr = np.asarray(theta_t, dtype=np.float64).reshape(-1)
        return tuple(float(x) for x in np.round(theta_arr, 14))

    def runtime_for_theta(theta_t: Any, *, theta_index: int | None = None) -> dict[str, Any]:
        nonlocal first_order_s, steady_state_s
        theta_arr = np.asarray(theta_t, dtype=np.float64).reshape(-1)
        key = theta_key(theta_arr)
        cached = runtime_cache.get(key)
        if cached is not None:
            return cached

        parameter_values = full_parameters(theta_arr)
        steady_state = reference_steady_state.copy()
        ss_status = "fixed_reference"
        ss_converged: bool | None = None
        ss_iterations: int | None = None
        ss_residual_norm: float | None = None
        ss_error: str | None = None
        ss_elapsed = 0.0

        if steady_state_mode != "fixed-reference":
            ss_started = time.perf_counter()
            try:
                steady_state_result = model.solve_steady_state(
                    parameter_values=parameter_values,
                    initial_guess=reference_steady_state,
                    tol=float(args.hlt_steady_state_tol),
                    max_iter=int(args.hlt_steady_state_max_iter),
                )
                ss_elapsed = time.perf_counter() - ss_started
                ss_converged = bool(steady_state_result.converged)
                ss_iterations = int(steady_state_result.iterations)
                ss_residual_norm = _finite_float_or_none(steady_state_result.residual_norm)
                candidate = np.asarray(steady_state_result.steady_state, dtype=np.float64)
                if ss_converged and np.isfinite(candidate).all():
                    steady_state = candidate
                    ss_status = "solved"
                elif steady_state_mode == "solve":
                    raise RuntimeError(
                        "HLT steady-state solve failed "
                        f"(converged={ss_converged}, residual={steady_state_result.residual_norm})."
                    )
                else:
                    ss_status = "fallback_reference"
            except Exception as exc:
                ss_elapsed = time.perf_counter() - ss_started
                ss_error = repr(exc)
                if steady_state_mode == "solve":
                    raise
                ss_status = "fallback_reference_error"
        steady_state_s += ss_elapsed

        first_order_started = time.perf_counter()
        first_order = model.solve_first_order(
            parameter_values=parameter_values,
            steady_state=steady_state,
        )
        first_order_elapsed = time.perf_counter() - first_order_started
        first_order_s += first_order_elapsed
        state_transition = np.asarray(first_order.solution.state_transition, dtype=np.float64)
        shock_impact = np.asarray(first_order.solution.shock_impact, dtype=np.float64)
        if not first_order.solution.converged:
            raise RuntimeError("HLT first-order ROM did not converge.")
        if not np.isfinite(state_transition).all() or not np.isfinite(shock_impact).all():
            raise RuntimeError("HLT first-order ROM contains non-finite matrices.")

        runtime = {
            "parameter_values": parameter_values,
            "steady_state": steady_state,
            "state_transition": state_transition,
            "shock_impact": shock_impact,
        }
        runtime_cache[key] = runtime
        steady_state_diagnostics.append(
            {
                "theta_index": theta_index,
                "steady_state_status": ss_status,
                "steady_state_converged": ss_converged,
                "steady_state_iterations": ss_iterations,
                "steady_state_residual_norm": ss_residual_norm,
                "steady_state_s": ss_elapsed,
                "steady_state_error": ss_error,
                "first_order_s": first_order_elapsed,
                "theta": theta_arr.tolist(),
            }
        )
        return runtime

    runtime_prepare_started = time.perf_counter()
    initial_states = np.column_stack(
        [
            runtime_for_theta(theta[:, theta_idx], theta_index=theta_idx)["steady_state"]
            for theta_idx in range(theta.shape[1])
        ]
    )
    runtime_prepare_s = time.perf_counter() - runtime_prepare_started

    def rom_predict(state: Any, shock_t: Any, theta_t: Any) -> tuple[np.ndarray, np.ndarray]:
        runtime = runtime_for_theta(theta_t)
        state_arr = np.asarray(state, dtype=np.float64)
        shock_arr = np.asarray(shock_t, dtype=np.float64)
        steady_state = runtime["steady_state"]
        state_transition = runtime["state_transition"]
        shock_impact = runtime["shock_impact"]
        state_dev = state_arr[state_idx] - steady_state[state_idx]
        next_state = steady_state + _mv(state_transition, state_dev) + _mv(shock_impact, shock_arr)
        if not np.isfinite(next_state).all():
            raise RuntimeError("HLT ROM produced a non-finite next state.")
        return next_state[observable_idx], next_state

    def fom_predict(state: Any, shock_t: Any, theta_t: Any) -> tuple[np.ndarray, np.ndarray]:
        runtime = runtime_for_theta(theta_t)
        steady_state = runtime["steady_state"]
        deterministic = np.zeros((sep_config.periods, shock_dim), dtype=np.float64)
        if sep_config.periods > 0:
            deterministic[0, :] = np.asarray(shock_t, dtype=np.float64)
        sep_result = model.solve_stochastic_extended_path(
            parameter_values=runtime["parameter_values"],
            steady_state=steady_state,
            initial_state=np.asarray(state, dtype=np.float64),
            terminal_state=steady_state,
            deterministic_shocks=deterministic,
            config=sep_config,
        )
        if not sep_result.solution.accepted:
            raise RuntimeError(f"HLT SEP failed with residual {sep_result.solution.residual_norm}.")
        next_state = np.asarray(sep_result.solution.mean_path, dtype=np.float64)[:, 1]
        if not np.isfinite(next_state).all():
            raise RuntimeError("HLT SEP produced a non-finite next state.")
        return next_state[observable_idx], next_state

    pipeline_started = time.perf_counter()
    result = fit_surrogate_pipeline(
        rom_predict,
        fom_predict,
        initial_state=initial_states,
        shocks=shocks,
        theta_design=theta,
        target_mode="fom_full",
        min_stable_periods=periods,
        input_names=tuple(list(model.timings.var) + list(model.timings.exo) + parameter_subset),
        output_names=tuple(observables + [f"{name}[1]" for name in model.timings.var]),
        architecture="resnet",
        rom_residual=True,
        validation_fraction=float(args.validation_fraction),
        split_by_theta=bool(args.split_by_theta),
        d_hidden=int(args.hidden),
        n_blocks=int(args.blocks),
        nepoch=int(args.epochs),
        eta_init=float(args.learning_rate),
        batch_size=int(args.batch_size),
        device=target_device,
    )
    pipeline_s = time.perf_counter() - pipeline_started

    likelihood_started = time.perf_counter()
    likelihood_result: dict[str, Any]
    jax_log_density_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "hlt_jax_log_density_smoke disabled or likelihood skipped",
    }
    likelihood_periods = int(args.hlt_likelihood_periods)
    if likelihood_periods <= 0:
        likelihood_result = {"status": "skipped", "reason": "hlt_likelihood_periods <= 0"}
    else:
        try:
            observations = np.asarray(case["observations"], dtype=np.float64)
            if observations.ndim != 2 or observations.shape[0] != len(observables):
                raise ValueError(
                    f"HLT observations must have shape ({len(observables)}, T), got {observations.shape}."
                )
            likelihood_periods = min(likelihood_periods, observations.shape[1])
            obs_data = observations[:, :likelihood_periods]
            obs_sigma_map = {str(name): value for name, value in dict(case["obs_sigma"]).items()}
            shock_sigma_map = {str(name): value for name, value in dict(case["shock_sigmas"]).items()}
            shock_names = [str(name) for name in case.get("shock_names", model.timings.exo)]
            obs_sigma = np.asarray([obs_sigma_map[name] for name in observables], dtype=np.float64)
            shock_sigmas = np.asarray([shock_sigma_map[name] for name in shock_names], dtype=np.float64)
            if shock_sigmas.shape[0] != shock_dim:
                raise ValueError(f"Expected {shock_dim} shock sigmas, got {shock_sigmas.shape[0]}.")
            theta0 = theta[:, 0]
            runtime0 = runtime_for_theta(theta0, theta_index=0)
            loglik_per_period, inferred_shocks = surrogate_inversion_loglik_per_period(
                rom_predict,
                result.training.frozen,
                runtime0["steady_state"],
                theta0,
                obs_data,
                obs_sigma,
                shock_sigmas,
                maxit=int(args.hlt_surrogate_inversion_maxit),
                tol=float(args.hlt_surrogate_inversion_tol),
                lambda_=float(args.hlt_surrogate_inversion_lambda),
            )
            loglik_per_period = np.asarray(loglik_per_period, dtype=np.float64)
            inferred_shocks = np.asarray(inferred_shocks, dtype=np.float64)
            likelihood_result = {
                "status": "ok",
                "periods": likelihood_periods,
                "elapsed_s": time.perf_counter() - likelihood_started,
                "total_loglikelihood": float(np.sum(loglik_per_period)),
                "per_period_loglikelihood": loglik_per_period.tolist(),
                "inferred_shocks_shape": list(inferred_shocks.shape),
                "inferred_shocks_finite": bool(np.isfinite(inferred_shocks).all()),
                "inferred_shocks_max_abs": float(np.max(np.abs(inferred_shocks))) if inferred_shocks.size else 0.0,
            }
            if bool(args.hlt_jax_log_density_smoke):
                jax_started = time.perf_counter()

                def device_array(values: Any, *, dtype: Any = jnp.float64) -> jax.Array:
                    array = jnp.asarray(values, dtype=dtype)
                    return array if target_device is None else jax.device_put(array, target_device)

                state_idx_jax = device_array(state_idx.astype(np.int64), dtype=jnp.int64)
                observable_idx_jax = device_array(np.asarray(observable_idx, dtype=np.int64), dtype=jnp.int64)
                steady_state_jax = device_array(runtime0["steady_state"])
                state_transition_jax = device_array(runtime0["state_transition"])
                shock_impact_jax = device_array(runtime0["shock_impact"])
                obs_data_jax = device_array(obs_data)
                obs_sigma_jax = device_array(obs_sigma)
                theta0_jax = device_array(theta0)

                def rom_predict_jax(state: Any, shock_t: Any, _theta_t: Any) -> tuple[jax.Array, jax.Array]:
                    state_arr = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
                    shock_arr = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
                    state_dev = state_arr[state_idx_jax] - steady_state_jax[state_idx_jax]
                    next_state = steady_state_jax + state_transition_jax @ state_dev + shock_impact_jax @ shock_arr
                    return next_state[observable_idx_jax], next_state

                def log_density(theta_local: jax.Array) -> jax.Array:
                    return surrogate_inversion_loglikelihood_jax(
                        rom_predict_jax,
                        result.training.frozen,
                        steady_state_jax,
                        theta_local,
                        obs_data_jax,
                        obs_sigma_jax,
                        shock_sigmas,
                        maxit=int(args.hlt_surrogate_inversion_maxit),
                        tol=float(args.hlt_surrogate_inversion_tol),
                        lambda_=float(args.hlt_surrogate_inversion_lambda),
                        shock_solver=str(args.hlt_jax_shock_solver),
                        batch_replay=bool(args.hlt_jax_batch_replay),
                        differentiate_shocks=bool(args.hlt_jax_differentiate_shocks),
                    )

                value_and_grad = jax.jit(jax.value_and_grad(log_density))
                value, grad = value_and_grad(theta0_jax)
                _block_until_ready_tree((value, grad))
                jax_elapsed = time.perf_counter() - jax_started
                grad_np = np.asarray(grad, dtype=np.float64)
                value_float = float(np.asarray(value))
                python_total = _finite_float_or_none(likelihood_result.get("total_loglikelihood"))
                value_minus_python = None if python_total is None else value_float - python_total
                parity_abs_diff = None if value_minus_python is None else abs(value_minus_python)
                parity_tol = float(args.hlt_jax_python_parity_tol)
                jax_log_density_result = {
                    "status": "ok",
                    "elapsed_s": jax_elapsed,
                    "value": value_float,
                    "python_surrogate_total_loglikelihood": python_total,
                    "value_minus_python": value_minus_python,
                    "parity_abs_diff": parity_abs_diff,
                    "parity_tol": parity_tol,
                    "parity_ok": None if parity_abs_diff is None else bool(parity_abs_diff <= parity_tol),
                    "gradient": grad_np.tolist(),
                    "gradient_finite": bool(np.isfinite(grad_np).all()),
                    "gradient_norm": float(np.linalg.norm(grad_np)),
                    "backend": jax.default_backend(),
                    "target_device": None if target_device is None else str(target_device),
                    "shock_solver": str(args.hlt_jax_shock_solver),
                    "batch_replay": bool(args.hlt_jax_batch_replay),
                    "differentiate_shocks": bool(args.hlt_jax_differentiate_shocks),
                    "caveat": (
                        "Differentiates the fixed-ROM surrogate likelihood through theta and inferred shocks; "
                        "steady-state and first-order matrices are held fixed in this smoke check."
                        if bool(args.hlt_jax_differentiate_shocks)
                        else "Differentiates the fixed-ROM surrogate likelihood through theta with inferred shocks treated as stop-gradient replay inputs; "
                        "steady-state and first-order matrices are held fixed in this smoke check."
                    ),
                }
        except Exception as exc:
            likelihood_result = {
                "status": "error",
                "elapsed_s": time.perf_counter() - likelihood_started,
                "error": repr(exc),
            }
            jax_log_density_result = {
                "status": "skipped",
                "reason": "Python surrogate likelihood failed before JAX log-density smoke.",
            }

    steady_statuses = [str(row["steady_state_status"]) for row in steady_state_diagnostics]
    fallback_count = sum(status.startswith("fallback") for status in steady_statuses)
    solved_count = sum(status == "solved" for status in steady_statuses)
    caveats = [
        "SEP target generation is still callback/Python-loop based; ResNet training is the GPU-native part.",
        "The likelihood block evaluates a trained-surrogate inversion likelihood; it is not a full HMC posterior run.",
    ]
    if steady_state_mode == "fixed-reference":
        caveats.append(
            "Uses a Julia-exported reference steady state for all draws; this is a fixed-SS smoke/stress test."
        )
    elif steady_state_mode == "solve-or-reference":
        caveats.append(
            "Attempts parameter-specific steady states and falls back to the Julia reference if a solve fails; inspect fallback_count."
        )
    else:
        caveats.append("Requires parameter-specific steady states; any failed solve aborts the run.")

    return {
        "status": "ok",
        "kind": "actual_hlt_surrogate_pipeline",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "model_source": str(model_source),
        "payload_case": str(case["name"]),
        "caveats": caveats,
        "parse_s": parse_s,
        "runtime_prepare_s": runtime_prepare_s,
        "steady_state_s": steady_state_s,
        "first_order_s": first_order_s,
        "pipeline_s": pipeline_s,
        "n_vars": int(model.timings.nVars),
        "n_exo": int(model.timings.nExo),
        "parameter_subset": parameter_subset,
        "theta_draws": int(theta.shape[1]),
        "steady_state_mode": steady_state_mode,
        "steady_state_solved_count": int(solved_count),
        "steady_state_fallback_count": int(fallback_count),
        "steady_state_diagnostics": steady_state_diagnostics,
        "periods": periods,
        "sep_config": {
            "periods": int(sep_config.periods),
            "branching_order": int(sep_config.branching_order),
            "nnodes": int(sep_config.nnodes),
            "sparse_tree": bool(sep_config.sparse_tree),
            "max_iter": int(sep_config.max_iter),
            "tol": float(sep_config.tol),
            "accept_tol": float(sep_config.accept_tol),
        },
        "dataset_summary": result.dataset_summary,
        "train_size": int(result.training.train_size),
        "val_size": int(result.training.val_size),
        "validation_rmse_mean": None
        if result.training.validation_rmse is None
        else float(np.mean(result.training.validation_rmse)),
        "validation_improvement_mean": None
        if result.training.validation_improvement is None
        else float(np.nanmean(result.training.validation_improvement)),
        "surrogate_inversion_likelihood": likelihood_result,
        "jax_surrogate_log_density": jax_log_density_result,
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
        choices=(
            "plan",
            "calibration",
            "training-scale",
            "full-scout",
            "sep-micro",
            "callback-dataset",
            "hlt-fixed-ss-smoke",
        ),
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
    parser.add_argument(
        "--hlt-model-source",
        type=Path,
        default=ROOT / "benchmarks" / "model_sources" / "Smets_Wouters_2007_HLT.jl",
    )
    parser.add_argument(
        "--hlt-payload",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "test_payloads.json",
    )
    parser.add_argument("--hlt-case-name", default="medium_sw07_hlt")
    parser.add_argument("--hlt-periods", type=int, default=2)
    parser.add_argument("--hlt-theta-draws", type=int, default=2)
    parser.add_argument("--hlt-shock-scale", type=float, default=0.02)
    parser.add_argument("--hlt-parameter-perturbation", type=float, default=1e-6)
    parser.add_argument(
        "--hlt-steady-state-mode",
        choices=("fixed-reference", "solve", "solve-or-reference"),
        default="fixed-reference",
        help=(
            "How the actual HLT smoke runner handles parameter-dependent steady states. "
            "'fixed-reference' preserves the Julia payload steady state; 'solve' requires "
            "a successful Python steady-state solve for each theta draw; "
            "'solve-or-reference' records failures and falls back to the reference."
        ),
    )
    parser.add_argument("--hlt-steady-state-tol", type=float, default=1e-10)
    parser.add_argument("--hlt-steady-state-max-iter", type=int, default=100)
    parser.add_argument("--hlt-likelihood-periods", type=int, default=1)
    parser.add_argument("--hlt-surrogate-inversion-maxit", type=int, default=4)
    parser.add_argument("--hlt-surrogate-inversion-tol", type=float, default=1e-5)
    parser.add_argument("--hlt-surrogate-inversion-lambda", type=float, default=1e-4)
    parser.add_argument("--hlt-jax-log-density-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hlt-jax-shock-solver", choices=("rom", "surrogate"), default="rom")
    parser.add_argument("--hlt-jax-batch-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hlt-jax-differentiate-shocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hlt-jax-python-parity-tol", type=float, default=1e-7)
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
    elif args.mode == "hlt-fixed-ss-smoke":
        payload["results"]["hlt_fixed_ss_smoke"] = run_hlt_fixed_steady_state_profile(args)
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
