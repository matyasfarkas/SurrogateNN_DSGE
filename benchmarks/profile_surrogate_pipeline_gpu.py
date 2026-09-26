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
    SurrogatePipelineResult,
    build_surrogate_residual_arrays_from_batched_sep_jax,
    build_surrogate_residual_arrays_jax,
    build_surrogate_residual_dataset,
    bounded_log_abs_det_jacobian,
    bounded_to_unconstrained,
    fit_surrogate_pipeline_from_batched_sep_jax,
    fit_surrogate_pipeline,
    parse_macro_model,
    predict_frozen_batch,
    resolve_jax_device,
    solve_batched_stochastic_extended_path_model,
    solve_batched_stochastic_extended_path_residual_expectation,
    solve_first_order_model_jax,
    solve_stochastic_extended_path_residual_expectation,
    static_hmc_sample,
    summarize_surrogate_dataset,
    surrogate_inversion_loglik_per_period,
    surrogate_inversion_loglikelihood_jax,
    train_surrogate_from_batched_arrays_jax,
    train_surrogate_from_dataset,
    unconstrained_to_bounded,
)


DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "surrogate_pipeline_gpu"

SW07_SAFE_15_PARAMETERS = (
    "calfa",
    "cg",
    "cgy",
    "cindw",
    "crdy",
    "crhob",
    "crhoqs",
    "crpi",
    "crr",
    "cry",
    "csigl",
    "z_ea",
    "z_eg",
    "z_em",
    "z_ew",
)

SW07_SAFE_27_PARAMETERS = (
    "calfa",
    "cg",
    "cgy",
    "cindw",
    "cmap",
    "cmaw",
    "constelab",
    "crdy",
    "crhoa",
    "crhob",
    "crhog",
    "crhoms",
    "crhopinf",
    "crhoqs",
    "crhow",
    "crpi",
    "crr",
    "cry",
    "csigl",
    "ctou",
    "z_ea",
    "z_eb",
    "z_eg",
    "z_em",
    "z_epinf",
    "z_eqs",
    "z_ew",
)


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


def _parity_metrics(
    *,
    value: float,
    reference: float | None,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return scale-aware parity diagnostics for scalar likelihood checks."""

    atol_float = float(atol)
    rtol_float = float(rtol)
    if atol_float < 0.0 or rtol_float < 0.0:
        raise ValueError(f"Parity tolerances must be nonnegative, got atol={atol}, rtol={rtol}.")
    if reference is None:
        return {
            "value_minus_reference": None,
            "abs_diff": None,
            "rel_diff": None,
            "scale": None,
            "atol": atol_float,
            "rtol": rtol_float,
            "effective_tol": None,
            "ok": None,
        }
    value_float = float(value)
    reference_float = float(reference)
    diff = value_float - reference_float
    abs_diff = abs(diff)
    scale = max(abs(reference_float), 1.0)
    effective_tol = atol_float + rtol_float * scale
    return {
        "value_minus_reference": diff,
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / scale,
        "scale": scale,
        "atol": atol_float,
        "rtol": rtol_float,
        "effective_tol": effective_tol,
        "ok": bool(abs_diff <= effective_tol),
    }


def _progress(message: str) -> None:
    print(f"[profile] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def _unique_preserve_order(values: Sequence[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def _parse_int_ladder(spec: str, *, base: int, include_zero: bool = False) -> tuple[int, ...]:
    normalized = str(spec).strip().lower()
    if normalized in {"", "auto"}:
        values: list[int] = [int(base)]
        if include_zero:
            values.append(0)
        return tuple(int(x) for x in _unique_preserve_order(values))
    values = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError("integer ladder must contain at least one value.")
    if include_zero and 0 not in values:
        values.append(0)
    return tuple(int(x) for x in _unique_preserve_order(values))


def _parse_float_ladder(spec: str) -> tuple[float, ...]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError("float ladder must contain at least one value.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"float ladder contains non-finite values: {values}.")
    return tuple(float(x) for x in _unique_preserve_order(values))


def _hlt_uniform_prior_interval(
    name: str,
    center: float,
    *,
    width_scale: float,
    width_floor: float,
) -> tuple[float, float]:
    """Build a conservative bounded interval around an HLT reference value."""

    if width_scale <= 0.0:
        raise ValueError(f"width_scale must be positive, got {width_scale}.")
    if width_floor <= 0.0:
        raise ValueError(f"width_floor must be positive, got {width_floor}.")
    width = max(abs(float(center)) * float(width_scale), float(width_floor))
    lower = float(center) - width
    upper = float(center) + width
    bounded_unit_parameter = name.startswith(("crho", "cprob", "cind")) or name in {"calfa"}
    if bounded_unit_parameter and center > 0.0:
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if center > 0.0 and lower <= 0.0 and name not in {"cry"}:
        lower = max(center * 0.5, np.finfo(float).tiny)
    if not lower < center < upper:
        raise ValueError(
            f"Invalid HLT prior interval for {name}: center={center}, lower={lower}, upper={upper}."
        )
    return lower, upper


def _select_hlt_parameter_subset(
    model: Any,
    case: dict[str, Any],
    spec: str,
) -> tuple[str, ...]:
    """Resolve HLT parameter-set aliases used by GPU estimation profiles."""

    normalized = str(spec).strip()
    if normalized == "payload":
        names = tuple(str(name) for name in case["parameter_subset"])
    elif normalized == "sw07_safe_15":
        names = SW07_SAFE_15_PARAMETERS
    elif normalized == "sw07_safe_27":
        names = SW07_SAFE_27_PARAMETERS
    elif normalized == "all":
        names = tuple(str(name) for name in model.parameter_names)
    else:
        names = tuple(part.strip() for part in normalized.split(",") if part.strip())
        if not names:
            raise ValueError("hlt_parameter_set must not be empty.")
    unknown = tuple(name for name in names if name not in model.parameter_names)
    if unknown:
        raise ValueError("Unknown HLT parameter names: " + ", ".join(unknown))
    return tuple(names)


def _hlt_uniform_prior_arrays(
    parameter_names: Sequence[str],
    center: Any,
    *,
    width_scale: float,
    width_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(center, dtype=np.float64).reshape(-1)
    if center_array.shape[0] != len(parameter_names):
        raise ValueError("center length must match parameter_names.")
    lower = np.empty_like(center_array)
    upper = np.empty_like(center_array)
    for idx, name in enumerate(parameter_names):
        lower[idx], upper[idx] = _hlt_uniform_prior_interval(
            str(name),
            float(center_array[idx]),
            width_scale=width_scale,
            width_floor=width_floor,
        )
    return lower, upper


def _summarize_static_hmc_result(
    *,
    result: Any,
    constrained_samples: Any,
    parameter_names: Sequence[str],
    elapsed_s: float,
) -> dict[str, Any]:
    samples = np.asarray(constrained_samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("constrained_samples must have shape (samples, chains, parameters).")
    if samples.shape[-1] != len(parameter_names):
        raise ValueError("parameter_names length must match the HMC sample parameter dimension.")
    accepted = np.asarray(result.accepted, dtype=bool)
    accept_prob = np.asarray(result.accept_prob, dtype=np.float64)
    flat = samples.reshape((-1, samples.shape[-1]))
    parameter_summary = {
        str(name): {
            "mean": float(np.mean(flat[:, idx])),
            "std": float(np.std(flat[:, idx])),
            "min": float(np.min(flat[:, idx])),
            "max": float(np.max(flat[:, idx])),
        }
        for idx, name in enumerate(parameter_names)
    }
    draws = int(np.prod(samples.shape[:2]))
    return {
        "elapsed_s": float(elapsed_s),
        "samples_shape": list(samples.shape),
        "post_warmup_draws": draws,
        "draws_per_second": float(draws / elapsed_s) if elapsed_s > 0 else math.inf,
        "accepted_share": float(np.mean(accepted)) if accepted.size else None,
        "accept_prob_mean": float(np.mean(accept_prob)) if accept_prob.size else None,
        "accept_prob_min": float(np.min(accept_prob)) if accept_prob.size else None,
        "accept_prob_max": float(np.max(accept_prob)) if accept_prob.size else None,
        "final_step_size": _finite_float_or_none(np.asarray(result.step_size, dtype=np.float64)),
        "final_log_prob_mean": _finite_float_or_none(np.mean(np.asarray(result.final_log_prob, dtype=np.float64))),
        "samples_finite": bool(np.isfinite(samples).all()),
        "parameter_summary": parameter_summary,
    }


def run_static_hmc_on_bounded_surrogate_log_density(
    *,
    log_density_fn: Any,
    center: Any,
    parameter_names: Sequence[str],
    lower: Any,
    upper: Any,
    chains: int,
    warmup: int,
    samples: int,
    leapfrog_steps: int,
    step_size: float,
    target_accept_prob: float,
    adapt_step_size: bool,
    initial_jitter: float,
    seed: int,
    min_accepted_share: float = 0.0,
    max_retries: int = 0,
    retry_step_size_factor: float = 0.25,
) -> dict[str, Any]:
    """Run vectorized static HMC over a bounded surrogate log likelihood."""

    if chains < 1:
        raise ValueError(f"chains must be positive, got {chains}.")
    if warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {warmup}.")
    if samples < 1:
        raise ValueError(f"samples must be positive, got {samples}.")
    if leapfrog_steps < 1:
        raise ValueError(f"leapfrog_steps must be positive, got {leapfrog_steps}.")
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}.")
    min_accept = float(min_accepted_share)
    if not 0.0 <= min_accept <= 1.0:
        raise ValueError(f"min_accepted_share must be in [0, 1], got {min_accepted_share}.")
    retries = int(max_retries)
    if retries < 0:
        raise ValueError(f"max_retries must be nonnegative, got {max_retries}.")
    retry_factor = float(retry_step_size_factor)
    if not 0.0 < retry_factor < 1.0:
        raise ValueError(f"retry_step_size_factor must be in (0, 1), got {retry_step_size_factor}.")
    center_jax = jnp.asarray(center, dtype=jnp.float64)
    lower_jax = jnp.asarray(lower, dtype=jnp.float64)
    upper_jax = jnp.asarray(upper, dtype=jnp.float64)
    if center_jax.ndim != 1:
        raise ValueError("center must be a one-dimensional parameter vector.")
    if lower_jax.shape != center_jax.shape or upper_jax.shape != center_jax.shape:
        raise ValueError("lower, upper, and center must have the same shape.")
    if center_jax.shape[0] != len(parameter_names):
        raise ValueError("parameter_names length must match center length.")
    prior_log_const = -jnp.sum(jnp.log(upper_jax - lower_jax))

    def log_posterior_unconstrained(unconstrained: jax.Array) -> jax.Array:
        theta_local = unconstrained_to_bounded(unconstrained, lower_jax, upper_jax)
        return (
            log_density_fn(theta_local)
            + prior_log_const
            + bounded_log_abs_det_jacobian(unconstrained, lower_jax, upper_jax)
        )

    key = jax.random.PRNGKey(int(seed))
    init_key, sample_key = jax.random.split(key)
    initial_center = bounded_to_unconstrained(center_jax, lower_jax, upper_jax)
    initial_position = initial_center[None, :] + float(initial_jitter) * jax.random.normal(
        init_key,
        shape=(int(chains), int(center_jax.shape[0])),
        dtype=center_jax.dtype,
    )
    def run_once(attempt_step_size: float) -> dict[str, Any]:
        compiled_sampler = jax.jit(
            lambda run_key, position: static_hmc_sample(
                log_posterior_unconstrained,
                position,
                run_key,
                num_warmup=int(warmup),
                num_samples=int(samples),
                step_size=float(attempt_step_size),
                num_leapfrog_steps=int(leapfrog_steps),
                target_accept_prob=float(target_accept_prob),
                adapt_step_size=bool(adapt_step_size),
            )
        )
        started = time.perf_counter()
        result = compiled_sampler(sample_key, initial_position)
        _block_until_ready_tree(result)
        elapsed = time.perf_counter() - started
        constrained_samples = unconstrained_to_bounded(result.samples, lower_jax, upper_jax)
        _block_until_ready_tree(constrained_samples)
        summary_once = _summarize_static_hmc_result(
            result=result,
            constrained_samples=constrained_samples,
            parameter_names=parameter_names,
            elapsed_s=elapsed,
        )
        summary_once["_raw_result"] = result
        summary_once["_constrained_samples"] = constrained_samples
        return summary_once

    requested_step_size = float(step_size)
    attempt_summaries: list[dict[str, Any]] = []
    selected_attempt = 0
    selected_summary: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        attempt_step_size = requested_step_size * (retry_factor**attempt)
        attempt_summary = run_once(attempt_step_size)
        attempt_summary["attempt"] = int(attempt)
        attempt_summary["attempt_step_size"] = float(attempt_step_size)
        attempt_summaries.append(attempt_summary)
        selected_attempt = attempt
        selected_summary = attempt_summary
        accepted_share = attempt_summary.get("accepted_share")
        if accepted_share is None or float(accepted_share) >= min_accept:
            break

    assert selected_summary is not None
    summary = dict(selected_summary)
    summary.pop("_raw_result", None)
    summary.pop("_constrained_samples", None)
    selected_step_size = float(summary["attempt_step_size"])
    summary.update(
        {
            "status": "ok",
            "kind": "fixed_rom_surrogate_static_hmc",
            "parameter_names": [str(name) for name in parameter_names],
            "prior_lower": np.asarray(lower_jax, dtype=np.float64).tolist(),
            "prior_upper": np.asarray(upper_jax, dtype=np.float64).tolist(),
            "chains": int(chains),
            "warmup": int(warmup),
            "samples": int(samples),
            "leapfrog_steps": int(leapfrog_steps),
            "initial_step_size": selected_step_size,
            "requested_initial_step_size": requested_step_size,
            "target_accept_prob": float(target_accept_prob),
            "adapt_step_size": bool(adapt_step_size),
            "initial_jitter": float(initial_jitter),
            "seed": int(seed),
            "retry_attempt": int(selected_attempt),
            "retry_count": int(selected_attempt),
            "retry_max_retries": retries,
            "retry_step_size_factor": retry_factor,
            "retry_min_accepted_share": min_accept,
            "retry_history": [
                {
                    "attempt": int(item["attempt"]),
                    "attempt_step_size": float(item["attempt_step_size"]),
                    "elapsed_s": float(item["elapsed_s"]),
                    "accepted_share": item.get("accepted_share"),
                    "accept_prob_mean": item.get("accept_prob_mean"),
                    "draws_per_second": item.get("draws_per_second"),
                    "final_step_size": item.get("final_step_size"),
                }
                for item in attempt_summaries
            ],
            "backend": jax.default_backend(),
            "caveat": (
                "Samples the trained-surrogate inversion likelihood with fixed reference steady state "
                "and fixed first-order ROM matrices. This is not yet a parameter-specific full HLT "
                "steady-state/ROM recomputation inside the HMC transition."
            ),
        }
    )
    return summary


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


def make_synthetic_hlt_batched_rollout_arrays(
    *,
    samples: int,
    theta_draws: int,
    shape: SyntheticHLTShape = SyntheticHLTShape(),
    seed: int = 1234,
    mask_fraction: float = 0.0,
    device: Any = None,
):
    """Build fixed-shape HLT-like rollout tensors and assemble JAX arrays.

    This synthetic path profiles the GPU-compatible layout expected from a future
    batched SEP FOM generator: arrays are shaped ``(theta_draw, period, dim)``
    and failed branches are represented by non-finite suffixes plus a mask.
    """

    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples}.")
    if theta_draws < 2:
        raise ValueError(f"theta_draws must be at least 2, got {theta_draws}.")
    if mask_fraction < 0.0 or mask_fraction >= 1.0:
        raise ValueError(f"mask_fraction must be in [0, 1), got {mask_fraction}.")

    periods = max(1, int(math.ceil(int(samples) / int(theta_draws))))
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(low=-0.8, high=0.8, size=(shape.theta_dim, theta_draws))
    states = rng.normal(scale=0.75, size=(theta_draws, periods, shape.state_dim))
    shocks = rng.normal(scale=0.45, size=(theta_draws, periods, shape.shock_dim))

    obs_state = rng.normal(scale=0.08, size=(shape.obs_dim, shape.state_dim))
    obs_shock = rng.normal(scale=0.15, size=(shape.obs_dim, shape.shock_dim))
    state_transition = _stable_transition_matrix(rng, shape.state_dim)
    state_shock = rng.normal(scale=0.08, size=(shape.state_dim, shape.shock_dim))
    obs_theta = rng.normal(scale=0.04, size=(shape.obs_dim, shape.theta_dim))
    state_theta = rng.normal(scale=0.02, size=(shape.state_dim, shape.theta_dim))

    theta_by_draw = theta.T[:, None, :]
    theta_by_period = np.broadcast_to(theta_by_draw, (theta_draws, periods, shape.theta_dim))
    rom_obs = np.einsum("os,tps->tpo", obs_state, states, optimize=True) + np.einsum(
        "oe,tpe->tpo", obs_shock, shocks, optimize=True
    )
    rom_state_next = np.einsum("ij,tpj->tpi", state_transition, states, optimize=True) + np.einsum(
        "ie,tpe->tpi", state_shock, shocks, optimize=True
    )
    shared_nl = np.tanh(
        0.25 * np.sum(theta_by_draw[:, :, : min(6, shape.theta_dim)], axis=2, keepdims=True)
        + 0.15 * np.sum(states[:, :, : min(6, shape.state_dim)], axis=2, keepdims=True)
        + 0.20 * np.sum(shocks[:, :, : min(6, shape.shock_dim)], axis=2, keepdims=True)
    )
    obs_resid = (
        0.05 * np.tanh(
            np.einsum(
                "oq,tpq->tpo",
                obs_theta,
                theta_by_period,
                optimize=True,
            )
        )
        + 0.03 * shared_nl
        + 0.015 * np.sin(rom_obs)
    )
    state_resid = (
        0.04 * np.tanh(
            np.einsum(
                "iq,tpq->tpi",
                state_theta,
                theta_by_period,
                optimize=True,
            )
        )
        + 0.02 * shared_nl
        + 0.01 * np.sin(rom_state_next)
    )
    fom_obs = rom_obs + obs_resid
    fom_state_next = rom_state_next + state_resid

    if mask_fraction > 0.0 and periods > 1:
        n_fail = min(theta_draws - 1, max(1, int(round(theta_draws * float(mask_fraction)))))
        fail_ids = np.arange(theta_draws - n_fail, theta_draws, dtype=np.int64)
        fail_from = max(1, periods // 2)
        fom_obs[fail_ids, fail_from:, 0] = np.nan

    def maybe_put(values: Any) -> Any:
        array = jnp.asarray(values, dtype=jnp.float64)
        return array if device is None else jax.device_put(array, device)

    return build_surrogate_residual_arrays_jax(
        maybe_put(states),
        maybe_put(shocks),
        maybe_put(theta),
        maybe_put(rom_obs),
        maybe_put(rom_state_next),
        maybe_put(fom_obs),
        maybe_put(fom_state_next),
        target_mode="fom_full",
        min_stable_periods=1,
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


def run_batched_training_profile(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    started = time.perf_counter()
    arrays = make_synthetic_hlt_batched_rollout_arrays(
        samples=args.samples,
        theta_draws=args.theta_draws,
        shape=shape,
        seed=args.seed,
        mask_fraction=float(args.batched_mask_fraction),
        device=target_device,
    )
    _block_until_ready_tree(arrays)
    dataset_s = time.perf_counter() - started

    train_started = time.perf_counter()
    result = train_surrogate_from_batched_arrays_jax(
        arrays,
        architecture="resnet",
        rom_residual=True,
        only_full_success=bool(args.only_full_success),
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

    probe_count = min(int(args.batch_size), int(arrays.X.shape[1]))
    probe = arrays.X[:, :probe_count]

    @jax.jit
    def predict_once(x_batch: jax.Array) -> jax.Array:
        return predict_frozen_batch(result.frozen, x_batch)

    predict_started = time.perf_counter()
    y0 = predict_once(probe)
    y0.block_until_ready()
    predict_first_s = time.perf_counter() - predict_started
    mask = np.asarray(arrays.sample_mask, dtype=bool)
    touched_samples = int(result.train_size) * int(args.epochs)
    return {
        "status": "ok",
        "kind": "synthetic_hlt_fixed_shape_batched_training",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "shape": shape.__dict__,
        "requested_samples": int(args.samples),
        "actual_samples": int(arrays.X.shape[1]),
        "theta_draws": int(args.theta_draws),
        "periods": int(arrays.X.shape[1] // args.theta_draws),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "blocks": int(args.blocks),
        "batch_size": int(args.batch_size),
        "mask_fraction_requested": float(args.batched_mask_fraction),
        "sample_mask_true_count": int(np.count_nonzero(mask)),
        "sample_mask_false_count": int(mask.size - np.count_nonzero(mask)),
        "theta_success_count": int(np.count_nonzero(np.asarray(arrays.theta_success, dtype=bool))),
        "dataset_build_s": dataset_s,
        "train_s": train_s,
        "predict_first_s": predict_first_s,
        "train_size": int(result.train_size),
        "val_size": int(result.val_size),
        "train_sample_updates_per_s": touched_samples / train_s if train_s > 0 else math.inf,
        "masked_sample_count": int(result.metadata["masked_sample_count"]),
        "memory_estimate_bytes": estimate_dataset_memory_bytes(
            shape=shape,
            samples=int(arrays.X.shape[1]),
            dtype=np.float64,
        ),
        "caveat": (
            "Synthetic fixed-shape rollout tensors profile the GPU-compatible training path. "
            "They do not replace actual HLT SEP target-generation parity tests."
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


def run_batched_sep_micro_profile(args: argparse.Namespace) -> dict[str, Any]:
    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    state_dim = int(args.sep_state_dim)
    shock_dim = int(args.sep_shock_dim)
    batch_size = int(args.sep_batch_size)
    if batch_size < 1:
        raise ValueError(f"sep_batch_size must be positive, got {batch_size}.")
    rng = np.random.default_rng(int(args.seed) + 71)
    B_np = rng.normal(scale=0.06, size=(state_dim, shock_dim))
    initial_np = rng.normal(scale=0.04, size=(batch_size, state_dim))
    terminal_np = np.zeros((state_dim,), dtype=np.float64)
    deterministic_np = rng.normal(scale=0.02, size=(batch_size, int(args.sep_periods), shock_dim))
    if int(args.sep_periods) > 1:
        deterministic_np[:, 1:, :] *= 0.25
    rho_np = np.linspace(0.80, 0.88, batch_size, dtype=np.float64)[:, None]

    def put(values: Any) -> jax.Array:
        array = jnp.asarray(values, dtype=jnp.float64)
        return array if target_device is None else jax.device_put(array, target_device)

    B = put(B_np)
    initial = put(initial_np)
    terminal = put(terminal_np)
    deterministic = put(deterministic_np)
    rho = put(rho_np)
    gamma = jnp.asarray(0.05, dtype=jnp.float64)

    def conditional_residual(
        prev_state: jax.Array,
        current_state: jax.Array,
        next_state: jax.Array,
        current_shock: jax.Array,
        params: jax.Array,
    ) -> jax.Array:
        target = params[0] * prev_state + gamma * jnp.tanh(next_state) + B @ current_shock
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
        line_search_batch=True,
        jit=True,
        vectorize_residual=True,
    )
    reps = max(1, int(args.sep_reps))
    times: list[float] = []
    last_solution: Any = None
    for _ in range(reps):
        started = time.perf_counter()
        last_solution = solve_batched_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=initial,
            terminal_state=terminal,
            shock_dim=shock_dim,
            deterministic_shocks=deterministic,
            config=config,
            params=rho,
        )
        _block_until_ready_tree(last_solution)
        times.append(time.perf_counter() - started)
    assert last_solution is not None
    accepted_count = int(np.count_nonzero(np.asarray(last_solution.accepted, dtype=bool)))
    converged_count = int(np.count_nonzero(np.asarray(last_solution.converged, dtype=bool)))
    median_s = float(statistics.median(times))
    return {
        "status": "ok",
        "kind": "batched_sep_sparse_tree_microbenchmark",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "batch_size": batch_size,
        "state_dim": state_dim,
        "shock_dim": shock_dim,
        "periods": int(config.periods),
        "branching_order": int(config.branching_order),
        "nnodes": int(config.nnodes),
        "sparse_tree": bool(config.sparse_tree),
        "reps": reps,
        "first_s": float(times[0]),
        "median_s": median_s,
        "solves_per_s_median": batch_size / median_s if median_s > 0 else math.inf,
        "accepted_count": accepted_count,
        "converged_count": converged_count,
        "max_residual_norm": float(np.max(np.asarray(last_solution.residual_norm))),
        "mean_path_shape": list(last_solution.mean_path.shape),
        "stacked_states_shape": list(last_solution.stacked_states.shape),
        "group_counts": [int(x) for x in last_solution.group_counts],
        "jacobian_method": str(last_solution.jacobian_method),
        "caveat": (
            "Synthetic batched conditional-residual SEP benchmark. It validates fixed-shape GPU SEP mechanics, "
            "not parsed HLT dynamic equations or OBC enforcement."
        ),
    }


def run_batched_sep_training_profile(args: argparse.Namespace, shape: SyntheticHLTShape) -> dict[str, Any]:
    """Profile the GPU-native SEP-solve -> target-array -> ResNet path.

    This remains a synthetic conditional-residual benchmark. It is the fixed
    shape path needed for GPU target generation, but it does not yet represent
    parsed HLT equations or OBC enforcement.
    """

    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    state_dim = int(args.sep_state_dim)
    shock_dim = int(args.sep_shock_dim)
    batch_size = int(args.sep_batch_size)
    theta_dim = int(shape.theta_dim)
    obs_dim = min(int(shape.obs_dim), state_dim)
    if state_dim < 1:
        raise ValueError(f"sep_state_dim must be positive, got {state_dim}.")
    if shock_dim < 1:
        raise ValueError(f"sep_shock_dim must be positive, got {shock_dim}.")
    if batch_size < 1:
        raise ValueError(f"sep_batch_size must be positive, got {batch_size}.")
    if theta_dim < 1:
        raise ValueError(f"theta_dim must be positive, got {theta_dim}.")
    if obs_dim < 1:
        raise ValueError("obs_dim must be positive and no larger than sep_state_dim for this profile.")

    rng = np.random.default_rng(int(args.seed) + 97)
    theta_np = rng.normal(scale=0.20, size=(theta_dim, batch_size))
    initial_np = rng.normal(scale=0.04, size=(batch_size, state_dim))
    terminal_np = np.zeros((state_dim,), dtype=np.float64)
    deterministic_np = rng.normal(scale=0.02, size=(batch_size, int(args.sep_periods), shock_dim))
    if int(args.sep_periods) > 1:
        deterministic_np[:, 1:, :] *= 0.25
    sep_shock_np = rng.normal(scale=0.06, size=(state_dim, shock_dim))
    theta_state_np = rng.normal(scale=0.015, size=(state_dim, theta_dim))
    rom_transition_np = _stable_transition_matrix(rng, state_dim)
    rom_shock_np = rng.normal(scale=0.05, size=(state_dim, shock_dim))
    rom_theta_np = rng.normal(scale=0.01, size=(state_dim, theta_dim))
    observable_idx = np.arange(obs_dim, dtype=np.int64)

    def put(values: Any) -> jax.Array:
        array = jnp.asarray(values, dtype=jnp.float64)
        return array if target_device is None else jax.device_put(array, target_device)

    theta = put(theta_np)
    params = jnp.swapaxes(theta, 0, 1)
    initial = put(initial_np)
    terminal = put(terminal_np)
    deterministic = put(deterministic_np)
    sep_shock = put(sep_shock_np)
    theta_state = put(theta_state_np)
    rom_transition = put(rom_transition_np)
    rom_shock = put(rom_shock_np)
    rom_theta = put(rom_theta_np)
    gamma = jnp.asarray(0.04, dtype=jnp.float64)
    observable_idx_jax = jnp.asarray(observable_idx, dtype=jnp.int32)

    def conditional_residual(
        prev_state: jax.Array,
        current_state: jax.Array,
        next_state: jax.Array,
        current_shock: jax.Array,
        theta_one: jax.Array,
    ) -> jax.Array:
        rho = 0.76 + 0.08 * jax.nn.sigmoid(theta_one[0])
        theta_push = theta_state @ theta_one
        target = rho * prev_state + gamma * jnp.tanh(next_state) + sep_shock @ current_shock
        return current_state - (target + jnp.tanh(theta_push))

    def rom_predict_jax(
        state: jax.Array,
        shock_t: jax.Array,
        theta_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        state_vec = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
        shock_vec = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
        theta_vec = jnp.asarray(theta_local, dtype=jnp.float64).reshape(-1)
        state_next = rom_transition @ state_vec + rom_shock @ shock_vec + rom_theta @ theta_vec
        return jnp.take(state_next, observable_idx_jax, axis=0), state_next

    config = SEPConfig(
        periods=int(args.sep_periods),
        branching_order=int(args.sep_order),
        nnodes=int(args.sep_nnodes),
        sparse_tree=bool(args.sep_sparse_tree),
        max_iter=int(args.sep_max_iter),
        tol=float(args.sep_tol),
        accept_tol=float(args.sep_accept_tol),
        line_search=True,
        line_search_batch=True,
        jit=True,
        vectorize_residual=True,
    )

    solve_started = time.perf_counter()
    sep_solution = solve_batched_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=initial,
        terminal_state=terminal,
        shock_dim=shock_dim,
        deterministic_shocks=deterministic,
        config=config,
        params=params,
    )
    _block_until_ready_tree(sep_solution)
    solve_s = time.perf_counter() - solve_started

    assemble_started = time.perf_counter()
    states = jnp.swapaxes(sep_solution.mean_path[:, :, :-1], 1, 2)
    theta_effect = jnp.einsum("ij,bj->bi", rom_theta, params)
    rom_state_next = (
        jnp.einsum("ij,bpj->bpi", rom_transition, states)
        + jnp.einsum("ij,bpj->bpi", rom_shock, deterministic)
        + theta_effect[:, None, :]
    )
    rom_obs = jnp.take(rom_state_next, jnp.asarray(observable_idx, dtype=jnp.int32), axis=2)
    arrays = build_surrogate_residual_arrays_from_batched_sep_jax(
        states,
        deterministic,
        theta,
        rom_obs,
        rom_state_next,
        sep_solution,
        observable_indices=observable_idx,
        target_mode="fom_full",
        min_stable_periods=1,
    )
    _block_until_ready_tree(arrays)
    assemble_s = time.perf_counter() - assemble_started

    train_started = time.perf_counter()
    result = train_surrogate_from_batched_arrays_jax(
        arrays,
        architecture="resnet",
        rom_residual=True,
        only_full_success=bool(args.only_full_success),
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

    predict_count = min(int(args.batch_size), int(arrays.X.shape[1]))
    predict_probe = arrays.X[:, :predict_count]

    @jax.jit
    def predict_once(x_batch: jax.Array) -> jax.Array:
        return predict_frozen_batch(result.frozen, x_batch)

    predict_started = time.perf_counter()
    y_probe = predict_once(predict_probe)
    y_probe.block_until_ready()
    predict_first_s = time.perf_counter() - predict_started

    if bool(args.skip_batched_sep_likelihood):
        likelihood_s = 0.0
        loglikelihood_value = jnp.asarray(np.nan, dtype=jnp.float64)
        loglikelihood_grad_np = np.full((theta_dim,), np.nan, dtype=np.float64)
        likelihood_status = "skipped"
    else:
        likelihood_started = time.perf_counter()
        fom_state_next = jnp.swapaxes(sep_solution.mean_path[:, :, 1 : int(config.periods) + 1], 1, 2)
        obs_data = jnp.take(fom_state_next[0], observable_idx_jax, axis=1).T
        obs_sigma = jnp.full((obs_dim,), 0.10, dtype=jnp.float64)
        shock_sigmas = jnp.full((shock_dim,), 0.15, dtype=jnp.float64)

        def loglikelihood(theta_local: jax.Array) -> jax.Array:
            return surrogate_inversion_loglikelihood_jax(
                rom_predict_jax,
                result.frozen,
                initial[0],
                theta_local,
                obs_data,
                obs_sigma,
                shock_sigmas,
                maxit=int(args.hlt_surrogate_inversion_maxit),
                tol=float(args.hlt_surrogate_inversion_tol),
                lambda_=float(args.hlt_surrogate_inversion_lambda),
                shock_solver=str(args.hlt_jax_shock_solver),
                batch_replay=bool(args.hlt_jax_batch_replay),
                differentiate_shocks=bool(args.hlt_jax_differentiate_shocks),
            )

        value_and_grad = jax.jit(jax.value_and_grad(loglikelihood))
        loglikelihood_value, loglikelihood_grad = value_and_grad(theta[:, 0])
        _block_until_ready_tree((loglikelihood_value, loglikelihood_grad))
        likelihood_s = time.perf_counter() - likelihood_started
        loglikelihood_grad_np = np.asarray(loglikelihood_grad, dtype=np.float64)
        likelihood_status = (
            "ok"
            if bool(np.isfinite(float(loglikelihood_value))) and bool(np.isfinite(loglikelihood_grad_np).all())
            else "nonfinite"
        )

    mask = np.asarray(arrays.sample_mask, dtype=bool)
    accepted = np.asarray(sep_solution.accepted, dtype=bool)
    converged = np.asarray(sep_solution.converged, dtype=bool)
    touched_samples = int(result.train_size) * int(args.epochs)
    return {
        "status": "ok",
        "kind": "synthetic_batched_sep_target_training",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "batch_size": batch_size,
        "actual_samples": int(arrays.X.shape[1]),
        "state_dim": state_dim,
        "shock_dim": shock_dim,
        "theta_dim": theta_dim,
        "obs_dim": obs_dim,
        "periods": int(config.periods),
        "branching_order": int(config.branching_order),
        "nnodes": int(config.nnodes),
        "sparse_tree": bool(config.sparse_tree),
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "blocks": int(args.blocks),
        "training_batch_size": int(args.batch_size),
        "sep_solve_s": solve_s,
        "target_assemble_s": assemble_s,
        "train_s": train_s,
        "predict_first_s": predict_first_s,
        "jax_likelihood_first_s": likelihood_s,
        "end_to_end_s": solve_s + assemble_s + train_s + predict_first_s + likelihood_s,
        "sep_accepted_count": int(np.count_nonzero(accepted)),
        "sep_converged_count": int(np.count_nonzero(converged)),
        "sample_mask_true_count": int(np.count_nonzero(mask)),
        "sample_mask_false_count": int(mask.size - np.count_nonzero(mask)),
        "train_size": int(result.train_size),
        "val_size": int(result.val_size),
        "train_sample_updates_per_s": touched_samples / train_s if train_s > 0 else math.inf,
        "target_samples_per_s": int(arrays.X.shape[1]) / (solve_s + assemble_s)
        if solve_s + assemble_s > 0
        else math.inf,
        "max_residual_norm": float(np.max(np.asarray(sep_solution.residual_norm))),
        "masked_sample_count": int(result.metadata["masked_sample_count"]),
        "jax_likelihood_status": likelihood_status,
        "jax_likelihood_value": _finite_float_or_none(loglikelihood_value),
        "jax_likelihood_grad_norm": _finite_float_or_none(np.linalg.norm(loglikelihood_grad_np)),
        "jax_likelihood_grad_finite": bool(np.isfinite(loglikelihood_grad_np).all()),
        "prediction_output_norm": float(np.linalg.norm(np.asarray(y_probe))),
        "caveat": (
            "Synthetic batched conditional-residual SEP target-generation and training. "
            "This exercises the GPU-native shape and masking path, not parsed HLT equations or OBC enforcement."
        ),
    }


def run_parsed_batched_sep_training_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Profile parser-backed batched SEP target generation plus training."""

    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    model = parse_macro_model(
        """
        @model parsed_batched_sep_profile begin
            y[0] = rho_y * y[-1] + gamma_y * y[1]^2 + u_y[x]
            z[0] = rho_z * z[-1] + gamma_z * z[1]^2 + cross * y[0] + u_z[x]
        end

        @parameters parsed_batched_sep_profile begin
            rho_y = 0.28
            rho_z = 0.22
            gamma_y = 0.06
            gamma_z = 0.04
            cross = 0.03
        end
        """
    )
    batch_size = int(args.sep_batch_size)
    if batch_size < 1:
        raise ValueError(f"sep_batch_size must be positive, got {batch_size}.")
    rng = np.random.default_rng(int(args.seed) + 113)
    base_params = np.asarray(model.parameter_values, dtype=np.float64)
    parameter_names = tuple(model.parameter_names)
    name_to_idx = {name: idx for idx, name in enumerate(parameter_names)}
    draws = np.broadcast_to(base_params[None, :], (batch_size, base_params.size)).copy()
    grid = np.linspace(-1.0, 1.0, batch_size, dtype=np.float64)
    draws[:, name_to_idx["rho_y"]] = np.clip(base_params[name_to_idx["rho_y"]] + 0.06 * grid, 0.05, 0.80)
    draws[:, name_to_idx["rho_z"]] = np.clip(base_params[name_to_idx["rho_z"]] - 0.04 * grid, 0.05, 0.80)
    draws[:, name_to_idx["gamma_y"]] = base_params[name_to_idx["gamma_y"]] * (1.0 + 0.20 * grid)
    draws[:, name_to_idx["gamma_z"]] = base_params[name_to_idx["gamma_z"]] * (1.0 - 0.15 * grid)
    draws[:, name_to_idx["cross"]] = base_params[name_to_idx["cross"]] * (1.0 + 0.10 * np.sin(grid))
    steady_states = np.zeros((batch_size, model.timings.nVars), dtype=np.float64)
    deterministic = rng.normal(scale=0.03, size=(batch_size, int(args.sep_periods), model.timings.nExo))
    if int(args.sep_periods) > 1:
        deterministic[:, 1:, :] *= 0.35

    def put(values: Any) -> jax.Array:
        array = jnp.asarray(values, dtype=jnp.float64)
        return array if target_device is None else jax.device_put(array, target_device)

    config = SEPConfig(
        periods=int(args.sep_periods),
        branching_order=int(args.sep_order),
        nnodes=int(args.sep_nnodes),
        sparse_tree=bool(args.sep_sparse_tree),
        max_iter=int(args.sep_max_iter),
        tol=float(args.sep_tol),
        accept_tol=float(args.sep_accept_tol),
        line_search=True,
        line_search_batch=True,
        jit=True,
        vectorize_residual=True,
    )
    solve_started = time.perf_counter()
    parsed_sep = solve_batched_stochastic_extended_path_model(
        model,
        parameter_values=put(draws),
        steady_state=put(steady_states),
        initial_state=put(steady_states),
        terminal_state=put(steady_states),
        config=config,
        deterministic_shocks=put(deterministic),
    )
    _block_until_ready_tree(parsed_sep)
    solve_s = time.perf_counter() - solve_started

    assemble_train_started = time.perf_counter()
    theta = jnp.swapaxes(parsed_sep.parameter_values, 0, 1)
    states = jnp.swapaxes(parsed_sep.solution.mean_path[:, :, :-1], 1, 2)
    shocks = put(deterministic)
    rho_y = parsed_sep.parameter_values[:, name_to_idx["rho_y"]]
    rho_z = parsed_sep.parameter_values[:, name_to_idx["rho_z"]]
    cross = parsed_sep.parameter_values[:, name_to_idx["cross"]]
    rom_y = rho_y[:, None] * states[:, :, 0] + shocks[:, :, 0]
    rom_z = rho_z[:, None] * states[:, :, 1] + cross[:, None] * rom_y + shocks[:, :, 1]
    rom_state_next = jnp.stack([rom_y, rom_z], axis=2)
    obs_dim = min(max(1, int(args.obs_dim)), model.timings.nVars)
    observable_indices = np.arange(obs_dim, dtype=np.int64)
    rom_obs = jnp.take(rom_state_next, jnp.asarray(observable_indices, dtype=jnp.int32), axis=2)
    pipeline = fit_surrogate_pipeline_from_batched_sep_jax(
        states,
        shocks,
        theta,
        rom_obs,
        rom_state_next,
        parsed_sep.solution,
        observable_indices=observable_indices,
        architecture="resnet",
        rom_residual=True,
        d_hidden=int(args.hidden),
        n_blocks=int(args.blocks),
        nepoch=int(args.epochs),
        eta_init=float(args.learning_rate),
        batch_size=int(args.batch_size),
        train_seed=int(args.seed),
        device=target_device,
    )
    _block_until_ready_tree(pipeline.training.frozen)
    assemble_train_s = time.perf_counter() - assemble_train_started

    probe_count = min(int(args.batch_size), int(pipeline.arrays.X.shape[1]))

    @jax.jit
    def predict_once(x_batch: jax.Array) -> jax.Array:
        return predict_frozen_batch(pipeline.training.frozen, x_batch)

    predict_started = time.perf_counter()
    y_probe = predict_once(pipeline.arrays.X[:, :probe_count])
    y_probe.block_until_ready()
    predict_first_s = time.perf_counter() - predict_started
    accepted = np.asarray(parsed_sep.solution.accepted, dtype=bool)
    converged = np.asarray(parsed_sep.solution.converged, dtype=bool)
    return {
        "status": "ok",
        "kind": "parsed_batched_sep_target_training",
        "backend": jax.default_backend(),
        "target_device": None if target_device is None else str(target_device),
        "batch_size": batch_size,
        "actual_samples": int(pipeline.arrays.X.shape[1]),
        "n_vars": int(model.timings.nVars),
        "n_exo": int(model.timings.nExo),
        "n_parameters": int(len(parameter_names)),
        "periods": int(config.periods),
        "branching_order": int(config.branching_order),
        "nnodes": int(config.nnodes),
        "sparse_tree": bool(config.sparse_tree),
        "sep_solve_s": solve_s,
        "assemble_train_s": assemble_train_s,
        "predict_first_s": predict_first_s,
        "end_to_end_s": solve_s + assemble_train_s + predict_first_s,
        "sep_accepted_count": int(np.count_nonzero(accepted)),
        "sep_converged_count": int(np.count_nonzero(converged)),
        "array_summary": pipeline.array_summary,
        "train_size": int(pipeline.training.train_size),
        "masked_sample_count": int(pipeline.training.metadata["masked_sample_count"]),
        "prediction_output_norm": float(np.linalg.norm(np.asarray(y_probe))),
        "max_residual_norm": float(np.max(np.asarray(parsed_sep.solution.residual_norm))),
        "caveat": (
            "Parser-backed batched nonlinear SEP profile with parameter draws and fixed explicit steady states. "
            "It does not include parameter-specific steady-state solving or auxiliary OBC shock reinjection."
        ),
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


@dataclass(frozen=True)
class HLTSEPAttemptSpec:
    index: int
    shock_scale: float
    config: SEPConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "shock_scale": float(self.shock_scale),
            "periods": int(self.config.periods),
            "branching_order": int(self.config.branching_order),
            "nnodes": int(self.config.nnodes),
            "sparse_tree": bool(self.config.sparse_tree),
            "max_iter": int(self.config.max_iter),
            "tol": float(self.config.tol),
            "accept_tol": None if self.config.accept_tol is None else float(self.config.accept_tol),
            "linear_solver": str(self.config.linear_solver),
        }


def _hlt_sep_attempt_specs(args: argparse.Namespace) -> tuple[HLTSEPAttemptSpec, ...]:
    order_ladder = _parse_int_ladder(
        str(args.hlt_sep_order_ladder),
        base=int(args.sep_order),
        include_zero=bool(args.hlt_adaptive_include_order_zero),
    )
    period_spec = str(args.hlt_sep_periods_ladder).strip().lower()
    if period_spec in {"", "auto"}:
        period_ladder = tuple(
            int(x)
            for x in _unique_preserve_order(
                [int(args.sep_periods), 1] if int(args.sep_periods) != 1 else [1]
            )
        )
    else:
        period_ladder = _parse_int_ladder(period_spec, base=int(args.sep_periods))
    max_iter_ladder = _parse_int_ladder(
        str(args.hlt_sep_max_iter_ladder),
        base=int(args.sep_max_iter),
    )
    shock_scale_ladder = _parse_float_ladder(str(args.hlt_sep_shock_scale_ladder))
    specs: list[HLTSEPAttemptSpec] = []
    for order in order_ladder:
        if order < 0:
            raise ValueError(f"SEP order ladder values must be nonnegative, got {order}.")
        for periods in period_ladder:
            if periods < 1:
                raise ValueError(f"SEP period ladder values must be positive, got {periods}.")
            for shock_scale in shock_scale_ladder:
                if shock_scale < 0.0:
                    raise ValueError(f"SEP shock-scale ladder values must be nonnegative, got {shock_scale}.")
                for max_iter in max_iter_ladder:
                    if max_iter < 1:
                        raise ValueError(f"SEP max-iter ladder values must be positive, got {max_iter}.")
                    specs.append(
                        HLTSEPAttemptSpec(
                            index=len(specs),
                            shock_scale=float(shock_scale),
                            config=SEPConfig(
                                periods=int(periods),
                                branching_order=int(order),
                                nnodes=int(args.sep_nnodes),
                                sparse_tree=bool(args.sep_sparse_tree),
                                max_iter=int(max_iter),
                                tol=float(args.sep_tol),
                                accept_tol=float(args.sep_accept_tol),
                            ),
                        )
                    )
    if not specs:
        raise ValueError("At least one HLT SEP attempt spec is required.")
    return tuple(specs)


def _hlt_surrogate_target_vector(
    target_mode: str,
    fom_obs: np.ndarray,
    fom_state_next: np.ndarray,
    rom_obs: np.ndarray,
    rom_state_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(target_mode).strip().lower()
    if mode == "fom_obs":
        return fom_obs, rom_obs
    if mode == "fom_full":
        return np.concatenate([fom_obs, fom_state_next]), np.concatenate([rom_obs, rom_state_next])
    if mode == "residual_obs":
        return fom_obs - rom_obs, np.zeros_like(fom_obs)
    if mode == "residual_full":
        y_fom = np.concatenate([fom_obs, fom_state_next])
        y_rom = np.concatenate([rom_obs, rom_state_next])
        return y_fom - y_rom, np.zeros_like(y_fom)
    raise ValueError(f"Unsupported HLT target_mode {target_mode!r}.")


def _build_adaptive_hlt_sep_dataset(
    *,
    rom_predict: Any,
    sep_predict: Any,
    initial_states: np.ndarray,
    shocks: np.ndarray,
    theta: np.ndarray,
    attempt_specs: Sequence[HLTSEPAttemptSpec],
    target_mode: str,
    min_stable_periods: int,
    input_names: Sequence[str],
    output_names: Sequence[str],
    max_logged_failures: int,
) -> tuple[SurrogateDataset, dict[str, Any]]:
    """Build multi-theta HLT targets by accepting only successful SEP solves.

    The builder tries an ordered ladder of SEP attempt specs for each theta-period
    pair. If all attempts fail for a period, that theta path stops and its stable
    prefix is kept if it meets `min_stable_periods`.
    """

    theta_array = np.asarray(theta, dtype=np.float64)
    if theta_array.ndim != 2 or theta_array.shape[1] < 1:
        raise ValueError("theta must have shape (parameters, theta_draws).")
    initial_array = np.asarray(initial_states, dtype=np.float64)
    if initial_array.ndim != 2 or initial_array.shape[1] != theta_array.shape[1]:
        raise ValueError("initial_states must have shape (state_dim, theta_draws).")
    shock_array = np.asarray(shocks, dtype=np.float64)
    if shock_array.ndim != 3 or shock_array.shape[0] != theta_array.shape[1]:
        raise ValueError("shocks must have shape (theta_draws, shock_dim, periods).")
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")
    if not attempt_specs:
        raise ValueError("attempt_specs must contain at least one candidate.")

    target_mode_norm = str(target_mode).strip().lower()
    n_theta = int(theta_array.shape[1])
    periods = int(shock_array.shape[2])
    X_columns: list[np.ndarray] = []
    Y_columns: list[np.ndarray] = []
    Y_rom_columns: list[np.ndarray] = []
    theta_ids: list[int] = []
    period_ids: list[int] = []
    theta_success = np.zeros((n_theta,), dtype=bool)
    theta_stable_periods = np.zeros((n_theta,), dtype=np.int64)
    accepted_attempt_indices: list[int] = []
    accepted_orders: list[int] = []
    accepted_shock_scales: list[float] = []
    accepted_residual_norms: list[float] = []
    failure_log: list[dict[str, Any]] = []
    attempted_count = 0

    def log_failure(row: dict[str, Any]) -> None:
        if len(failure_log) < int(max_logged_failures):
            failure_log.append(row)

    for theta_idx in range(n_theta):
        theta_t = theta_array[:, theta_idx]
        state = initial_array[:, theta_idx].copy()
        period_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]] = []
        for period in range(periods):
            base_shock = shock_array[theta_idx, :, period]
            accepted_record: tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                int,
                float,
                float,
                np.ndarray,
            ] | None = None
            last_error: str | None = None
            for spec in attempt_specs:
                attempted_count += 1
                shock_t = np.asarray(base_shock * float(spec.shock_scale), dtype=np.float64)
                try:
                    rom_obs, rom_state_next = rom_predict(state, shock_t, theta_t)
                    rom_obs = np.asarray(rom_obs, dtype=np.float64).reshape(-1)
                    rom_state_next = np.asarray(rom_state_next, dtype=np.float64).reshape(-1)
                    fom_obs, fom_state_next, sep_diag = sep_predict(state, shock_t, theta_t, spec.config)
                    fom_obs = np.asarray(fom_obs, dtype=np.float64).reshape(-1)
                    fom_state_next = np.asarray(fom_state_next, dtype=np.float64).reshape(-1)
                    if not (
                        np.isfinite(rom_obs).all()
                        and np.isfinite(rom_state_next).all()
                        and np.isfinite(fom_obs).all()
                        and np.isfinite(fom_state_next).all()
                    ):
                        raise RuntimeError("non-finite ROM/FOM target arrays")
                    if rom_obs.shape != fom_obs.shape:
                        raise RuntimeError(f"ROM/FOM observation shape mismatch: {rom_obs.shape} vs {fom_obs.shape}")
                    if rom_state_next.shape != fom_state_next.shape or fom_state_next.shape != state.shape:
                        raise RuntimeError(
                            "ROM/FOM state shape mismatch: "
                            f"rom={rom_state_next.shape} fom={fom_state_next.shape} state={state.shape}"
                        )
                    residual_norm = _finite_float_or_none(dict(sep_diag).get("residual_norm"))
                    residual_value = math.nan if residual_norm is None else float(residual_norm)
                    accepted_record = (
                        shock_t,
                        rom_obs,
                        rom_state_next,
                        fom_obs,
                        int(spec.index),
                        float(spec.shock_scale),
                        residual_value,
                        fom_state_next,
                    )
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    log_failure(
                        {
                            "theta_index": int(theta_idx),
                            "period": int(period),
                            "candidate_index": int(spec.index),
                            "branching_order": int(spec.config.branching_order),
                            "shock_scale": float(spec.shock_scale),
                            "max_iter": int(spec.config.max_iter),
                            "error": last_error,
                        }
                    )
            if accepted_record is None:
                _progress(
                    f"theta {theta_idx}/{n_theta - 1} stopped at period {period}/{periods - 1}; "
                    f"last_error={last_error}"
                )
                break

            shock_t, rom_obs, rom_state_next, fom_obs, spec_index, used_shock_scale, residual_value, fom_state_next = accepted_record
            y, y_rom = _hlt_surrogate_target_vector(
                target_mode_norm,
                fom_obs,
                fom_state_next,
                rom_obs,
                rom_state_next,
            )
            x = np.concatenate([state, shock_t, theta_t])
            period_records.append((x, y, y_rom, int(period), spec_index, used_shock_scale, residual_value))
            state = np.asarray(fom_state_next, dtype=np.float64)

        stable = len(period_records)
        theta_stable_periods[theta_idx] = stable
        theta_success[theta_idx] = stable >= int(min_stable_periods) and stable == periods
        if stable < int(min_stable_periods):
            continue
        for x, y, y_rom, period, spec_index, used_shock_scale, residual_value in period_records:
            X_columns.append(x)
            Y_columns.append(y)
            Y_rom_columns.append(y_rom)
            theta_ids.append(theta_idx)
            period_ids.append(period)
            accepted_attempt_indices.append(spec_index)
            accepted_shock_scales.append(used_shock_scale)
            accepted_residual_norms.append(residual_value)
            accepted_orders.append(int(attempt_specs[spec_index].config.branching_order))
        _progress(
            f"theta {theta_idx}/{n_theta - 1} accepted stable_periods={stable}/{periods} "
            f"full_success={bool(theta_success[theta_idx])}"
        )

    if not X_columns:
        diagnostics = {
            "builder": "adaptive_sep",
            "status": "error",
            "attempted_count": int(attempted_count),
            "failure_log": failure_log,
            "candidate_specs": [spec.as_dict() for spec in attempt_specs],
        }
        raise ValueError("No stable surrogate-dataset samples were generated. Diagnostics: " + json.dumps(_jsonable(diagnostics)))

    dataset = SurrogateDataset(
        X=np.column_stack(X_columns),
        Y=np.column_stack(Y_columns),
        Y_rom=np.column_stack(Y_rom_columns),
        theta=theta_array,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=theta_success,
        theta_stable_periods=theta_stable_periods,
        target_mode=target_mode_norm,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        theta_names=tuple(input_names[-theta_array.shape[0] :]) if len(input_names) >= theta_array.shape[0] else (),
    )
    accepted_attempt_array = np.asarray(accepted_attempt_indices, dtype=np.int64)
    accepted_order_array = np.asarray(accepted_orders, dtype=np.int64)
    accepted_scale_array = np.asarray(accepted_shock_scales, dtype=np.float64)
    residual_array = np.asarray(accepted_residual_norms, dtype=np.float64)
    diagnostics = {
        "builder": "adaptive_sep",
        "status": "ok",
        "candidate_specs": [spec.as_dict() for spec in attempt_specs],
        "attempted_count": int(attempted_count),
        "accepted_samples": int(dataset.n_samples),
        "fallback_samples": int(np.count_nonzero(accepted_attempt_array != 0)),
        "fallback_share": float(np.mean(accepted_attempt_array != 0)) if accepted_attempt_array.size else 0.0,
        "theta_full_success_count": int(np.count_nonzero(theta_success)),
        "theta_with_any_sample_count": int(np.count_nonzero(theta_stable_periods >= int(min_stable_periods))),
        "theta_stable_periods": theta_stable_periods.tolist(),
        "theta_success": theta_success.tolist(),
        "accepted_by_candidate": {
            str(int(idx)): int(np.count_nonzero(accepted_attempt_array == int(idx)))
            for idx in np.unique(accepted_attempt_array)
        },
        "accepted_by_branching_order": {
            str(int(order)): int(np.count_nonzero(accepted_order_array == int(order)))
            for order in np.unique(accepted_order_array)
        },
        "accepted_by_shock_scale": {
            f"{float(scale):.12g}": int(np.count_nonzero(np.isclose(accepted_scale_array, float(scale))))
            for scale in np.unique(accepted_scale_array)
        },
        "residual_norm_mean": _finite_float_or_none(np.nanmean(residual_array)) if residual_array.size else None,
        "residual_norm_max": _finite_float_or_none(np.nanmax(residual_array)) if residual_array.size else None,
        "failure_log": failure_log,
    }
    return dataset, diagnostics


def run_hlt_fixed_steady_state_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Run the actual HLT model through a tiny ROM/FOM surrogate path.

    This is intentionally a smoke/profile mode. It verifies that the parsed HLT
    model, first-order ROM, SEP FOM target generation, JAX surrogate training,
    and trained-surrogate likelihood evaluation compose on the selected device.
    Use ``--hlt-steady-state-mode solve`` to require parameter-specific steady
    states; the default fixed-reference mode is a conservative stability check.
    """

    _progress(
        "starting actual-HLT fixed-SS profile "
        f"device={args.device} theta_draws={args.hlt_theta_draws} "
        f"parameter_set={args.hlt_parameter_set}"
    )
    target_device = None if args.device == "auto" else resolve_jax_device(args.device)
    case = _load_hlt_payload_case(args)
    model_source = Path(args.hlt_model_source)
    started = time.perf_counter()
    model = parse_macro_model(model_source.read_text(encoding="utf-8"))
    parse_s = time.perf_counter() - started
    _progress(f"parsed HLT model in {parse_s:.3f}s")

    reference_steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    base_parameters = np.asarray(model.parameter_values, dtype=np.float64)
    parameter_subset = list(_select_hlt_parameter_subset(model, case, str(args.hlt_parameter_set)))
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
        theta_label = "unknown" if theta_index is None else str(theta_index)
        _progress(f"preparing runtime for theta {theta_label}/{theta.shape[1] - 1}")

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
        if steady_state_mode != "fixed-reference":
            _progress(
                f"steady-state theta {theta_label} status={ss_status} "
                f"elapsed={ss_elapsed:.3f}s residual={ss_residual_norm}"
            )

        first_order_started = time.perf_counter()
        first_order = model.solve_first_order(
            parameter_values=parameter_values,
            steady_state=steady_state,
            qme_algorithm=str(args.hlt_first_order_qme_algorithm),
        )
        first_order_elapsed = time.perf_counter() - first_order_started
        first_order_s += first_order_elapsed
        state_transition = np.asarray(first_order.solution.state_transition, dtype=np.float64)
        shock_impact = np.asarray(first_order.solution.shock_impact, dtype=np.float64)
        if not first_order.solution.converged:
            raise RuntimeError("HLT first-order ROM did not converge.")
        if not np.isfinite(state_transition).all() or not np.isfinite(shock_impact).all():
            raise RuntimeError("HLT first-order ROM contains non-finite matrices.")
        _progress(
            f"first-order theta {theta_label} elapsed={first_order_elapsed:.3f}s "
            f"cumulative_first_order={first_order_s:.3f}s"
        )

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
    _progress(
        f"prepared {theta.shape[1]} theta runtimes in {runtime_prepare_s:.3f}s; "
        f"starting ROM/FOM surrogate target generation and training"
    )

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

    def sep_predict(state: Any, shock_t: Any, theta_t: Any, config: SEPConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        runtime = runtime_for_theta(theta_t)
        steady_state = runtime["steady_state"]
        deterministic = np.zeros((config.periods, shock_dim), dtype=np.float64)
        if config.periods > 0:
            deterministic[0, :] = np.asarray(shock_t, dtype=np.float64)
        sep_result = model.solve_stochastic_extended_path(
            parameter_values=runtime["parameter_values"],
            steady_state=steady_state,
            initial_state=np.asarray(state, dtype=np.float64),
            terminal_state=steady_state,
            deterministic_shocks=deterministic,
            config=config,
        )
        if not sep_result.solution.accepted:
            raise RuntimeError(f"HLT SEP failed with residual {sep_result.solution.residual_norm}.")
        next_state = np.asarray(sep_result.solution.mean_path, dtype=np.float64)[:, 1]
        if not np.isfinite(next_state).all():
            raise RuntimeError("HLT SEP produced a non-finite next state.")
        diagnostics = {
            "residual_norm": _finite_float_or_none(sep_result.solution.residual_norm),
            "iterations": int(sep_result.solution.iterations),
            "converged": bool(sep_result.solution.converged),
            "accepted": bool(sep_result.solution.accepted),
            "branching_order": int(config.branching_order),
            "max_iter": int(config.max_iter),
        }
        return next_state[observable_idx], next_state, diagnostics

    def fom_predict(state: Any, shock_t: Any, theta_t: Any) -> tuple[np.ndarray, np.ndarray]:
        obs, next_state, _ = sep_predict(state, shock_t, theta_t, sep_config)
        return obs, next_state

    pipeline_started = time.perf_counter()
    target_diagnostics: dict[str, Any]
    target_min_stable_periods = (
        int(periods)
        if int(args.hlt_target_min_stable_periods) < 0
        else int(args.hlt_target_min_stable_periods)
    )
    input_names = tuple(list(model.timings.var) + list(model.timings.exo) + parameter_subset)
    output_names = tuple(observables + [f"{name}[1]" for name in model.timings.var])
    target_builder = str(args.hlt_target_builder).strip().lower()
    if target_builder == "callback":
        result = fit_surrogate_pipeline(
            rom_predict,
            fom_predict,
            initial_state=initial_states,
            shocks=shocks,
            theta_design=theta,
            target_mode="fom_full",
            min_stable_periods=target_min_stable_periods,
            input_names=input_names,
            output_names=output_names,
            architecture="resnet",
            rom_residual=True,
            validation_fraction=float(args.validation_fraction),
            split_by_theta=bool(args.split_by_theta),
            only_full_success=bool(args.only_full_success),
            d_hidden=int(args.hidden),
            n_blocks=int(args.blocks),
            nepoch=int(args.epochs),
            eta_init=float(args.learning_rate),
            batch_size=int(args.batch_size),
            device=target_device,
        )
        target_diagnostics = {
            "builder": "callback",
            "status": "ok",
            "target_min_stable_periods": int(target_min_stable_periods),
            "sep_config": {
                "periods": int(sep_config.periods),
                "branching_order": int(sep_config.branching_order),
                "nnodes": int(sep_config.nnodes),
                "sparse_tree": bool(sep_config.sparse_tree),
                "max_iter": int(sep_config.max_iter),
                "tol": float(sep_config.tol),
                "accept_tol": None if sep_config.accept_tol is None else float(sep_config.accept_tol),
            },
        }
    elif target_builder == "adaptive-sep":
        attempt_specs = _hlt_sep_attempt_specs(args)
        _progress(
            "adaptive HLT target generation "
            f"min_stable_periods={target_min_stable_periods} "
            f"attempt_specs={len(attempt_specs)}"
        )
        dataset, target_diagnostics = _build_adaptive_hlt_sep_dataset(
            rom_predict=rom_predict,
            sep_predict=sep_predict,
            initial_states=initial_states,
            shocks=shocks,
            theta=theta,
            attempt_specs=attempt_specs,
            target_mode="fom_full",
            min_stable_periods=target_min_stable_periods,
            input_names=input_names,
            output_names=output_names,
            max_logged_failures=int(args.hlt_target_max_logged_failures),
        )
        dataset_summary = summarize_surrogate_dataset(dataset)
        successful_groups = np.unique(dataset.theta_ids).size
        validation_fraction = float(args.validation_fraction)
        split_by_theta = bool(args.split_by_theta)
        if split_by_theta and successful_groups < 2 and validation_fraction > 0.0:
            _progress(
                "disabling held-out-theta validation because fewer than two theta groups "
                f"produced targets (groups={successful_groups})"
            )
            validation_fraction = 0.0
        training = train_surrogate_from_dataset(
            dataset,
            architecture="resnet",
            rom_residual=True,
            validation_fraction=validation_fraction,
            split_by_theta=split_by_theta,
            only_full_success=bool(args.only_full_success),
            d_hidden=int(args.hidden),
            n_blocks=int(args.blocks),
            nepoch=int(args.epochs),
            eta_init=float(args.learning_rate),
            batch_size=int(args.batch_size),
            device=target_device,
        )
        result = SurrogatePipelineResult(
            dataset=dataset,
            dataset_summary=dataset_summary,
            training=training,
        )
        target_diagnostics["target_min_stable_periods"] = int(target_min_stable_periods)
        target_diagnostics["effective_validation_fraction"] = float(validation_fraction)
    else:
        raise ValueError("hlt_target_builder must be 'adaptive-sep' or 'callback'.")
    pipeline_s = time.perf_counter() - pipeline_started
    _progress(
        f"finished surrogate pipeline in {pipeline_s:.3f}s "
        f"train_size={result.training.split.train_size} val_size={result.training.split.val_size}"
    )

    likelihood_started = time.perf_counter()
    likelihood_result: dict[str, Any]
    jax_log_density_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "hlt_jax_log_density_smoke disabled or likelihood skipped",
    }
    surrogate_hmc_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "hlt_surrogate_hmc_samples <= 0 or likelihood skipped",
    }
    likelihood_periods = int(args.hlt_likelihood_periods)
    if likelihood_periods <= 0:
        likelihood_result = {"status": "skipped", "reason": "hlt_likelihood_periods <= 0"}
    else:
        python_likelihood_completed = False
        try:
            _progress(f"starting surrogate inversion likelihood periods={likelihood_periods}")
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
            python_likelihood_completed = True
            _progress(
                "finished Python surrogate inversion likelihood "
                f"elapsed={likelihood_result['elapsed_s']:.3f}s "
                f"loglik={likelihood_result['total_loglikelihood']:.6g}"
            )
            if bool(args.hlt_jax_log_density_smoke) or int(args.hlt_surrogate_hmc_samples) > 0:
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
                shock_sigmas_jax = device_array(shock_sigmas)
                theta0_jax = device_array(theta0)
                base_parameters_jax = device_array(base_parameters)
                subset_idx_jax = device_array(np.asarray(subset_idx, dtype=np.int64), dtype=jnp.int64)
                reference_steady_state_guess = model._extract_base_steady_state(reference_steady_state)
                failure_loglikelihood_jax = device_array(float(args.hlt_likelihood_on_failure))
                likelihood_runtime_mode = str(args.hlt_likelihood_runtime_mode).strip().lower()
                likelihood_qme_algorithm = str(args.hlt_likelihood_qme_algorithm).strip().lower()
                likelihood_static_rows_mode = str(args.hlt_likelihood_static_rows_mode).strip().lower()
                if likelihood_runtime_mode not in {"fixed-reference", "full-jax"}:
                    raise ValueError(
                        "hlt_likelihood_runtime_mode must be 'fixed-reference' or "
                        f"'full-jax', got {args.hlt_likelihood_runtime_mode!r}."
                    )
                if likelihood_static_rows_mode not in {"reference", "none"}:
                    raise ValueError(
                        "hlt_likelihood_static_rows_mode must be 'reference' or "
                        f"'none', got {args.hlt_likelihood_static_rows_mode!r}."
                    )
                likelihood_static_rows = None
                if (
                    likelihood_runtime_mode == "full-jax"
                    and likelihood_static_rows_mode == "reference"
                    and model.timings.nPresent_only > 0
                ):
                    likelihood_static_rows = model._first_order_static_equation_rows_for_values(
                        steady_state=runtime0["steady_state"],
                        parameter_values=runtime0["parameter_values"],
                    )

                def full_parameter_vector_jax(theta_local: jax.Array) -> jax.Array:
                    theta_arr = jnp.asarray(theta_local, dtype=jnp.float64).reshape(-1)
                    return base_parameters_jax.at[subset_idx_jax].set(theta_arr)

                def rom_predict_jax(state: Any, shock_t: Any, _theta_t: Any) -> tuple[jax.Array, jax.Array]:
                    state_arr = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
                    shock_arr = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
                    state_dev = state_arr[state_idx_jax] - steady_state_jax[state_idx_jax]
                    next_state = steady_state_jax + state_transition_jax @ state_dev + shock_impact_jax @ shock_arr
                    return next_state[observable_idx_jax], next_state

                def fixed_reference_log_density(theta_local: jax.Array) -> jax.Array:
                    return surrogate_inversion_loglikelihood_jax(
                        rom_predict_jax,
                        result.training.frozen,
                        steady_state_jax,
                        theta_local,
                        obs_data_jax,
                        obs_sigma_jax,
                        shock_sigmas_jax,
                        maxit=int(args.hlt_surrogate_inversion_maxit),
                        tol=float(args.hlt_surrogate_inversion_tol),
                        lambda_=float(args.hlt_surrogate_inversion_lambda),
                        shock_solver=str(args.hlt_jax_shock_solver),
                        batch_replay=bool(args.hlt_jax_batch_replay),
                        differentiate_shocks=bool(args.hlt_jax_differentiate_shocks),
                    )

                def full_jax_log_density(theta_local: jax.Array) -> jax.Array:
                    parameter_vector = full_parameter_vector_jax(theta_local)
                    runtime = solve_first_order_model_jax(
                        model,
                        parameter_values=parameter_vector,
                        steady_state_initial_guess=reference_steady_state_guess,
                        steady_state_tol=float(args.hlt_steady_state_tol),
                        steady_state_max_iter=int(args.hlt_steady_state_max_iter),
                        qme_algorithm=likelihood_qme_algorithm,
                        static_equation_rows=likelihood_static_rows,
                        check_parameter_bounds=True,
                    )

                    def runtime_rom_predict(
                        state: Any,
                        shock_t: Any,
                        _theta_t: Any,
                    ) -> tuple[jax.Array, jax.Array]:
                        state_arr = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
                        shock_arr = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
                        state_dev = state_arr[state_idx_jax] - runtime.steady_state[state_idx_jax]
                        next_state = (
                            runtime.steady_state
                            + runtime.state_transition @ state_dev
                            + runtime.shock_impact @ shock_arr
                        )
                        return next_state[observable_idx_jax], next_state

                    runtime_ok = (
                        runtime.converged
                        & jnp.all(jnp.isfinite(runtime.steady_state))
                        & jnp.all(jnp.isfinite(runtime.state_transition))
                        & jnp.all(jnp.isfinite(runtime.shock_impact))
                    )

                    return jax.lax.cond(
                        runtime_ok,
                        lambda _: surrogate_inversion_loglikelihood_jax(
                            runtime_rom_predict,
                            result.training.frozen,
                            runtime.steady_state,
                            theta_local,
                            obs_data_jax,
                            obs_sigma_jax,
                            shock_sigmas_jax,
                            maxit=int(args.hlt_surrogate_inversion_maxit),
                            tol=float(args.hlt_surrogate_inversion_tol),
                            lambda_=float(args.hlt_surrogate_inversion_lambda),
                            shock_solver=str(args.hlt_jax_shock_solver),
                            batch_replay=bool(args.hlt_jax_batch_replay),
                            differentiate_shocks=bool(args.hlt_jax_differentiate_shocks),
                        ),
                        lambda _: failure_loglikelihood_jax,
                        operand=None,
                    )

                log_density = (
                    fixed_reference_log_density
                    if likelihood_runtime_mode == "fixed-reference"
                    else full_jax_log_density
                )

                if bool(args.hlt_jax_log_density_smoke):
                    _progress(
                        "starting JAX log-density smoke "
                        f"runtime_mode={likelihood_runtime_mode} "
                        f"qme={likelihood_qme_algorithm} "
                        f"gradient={bool(args.hlt_jax_log_density_gradient)}"
                    )
                    jax_started = time.perf_counter()

                    grad_np: np.ndarray | None
                    repeat_timings: list[float] = []
                    repeat_values: list[float] = []
                    repeat_gradient_norms: list[float] = []
                    batched_first_s: float | None = None
                    batched_repeat_timings: list[float] = []
                    batched_values: list[list[float]] = []
                    batched_gradient_norms: list[list[float]] = []
                    if bool(args.hlt_jax_log_density_gradient):
                        value_and_grad = jax.jit(jax.value_and_grad(log_density))
                        value, grad = value_and_grad(theta0_jax)
                        _block_until_ready_tree((value, grad))
                        grad_np = np.asarray(grad, dtype=np.float64)
                        repeat_fn = value_and_grad
                    else:
                        value_fn = jax.jit(log_density)
                        value = value_fn(theta0_jax)
                        _block_until_ready_tree(value)
                        grad_np = None
                        repeat_fn = value_fn
                    jax_elapsed = time.perf_counter() - jax_started

                    repeat_count = max(0, int(args.hlt_jax_log_density_repeat_evals))
                    repeat_perturbation = float(args.hlt_jax_log_density_repeat_perturbation)
                    if repeat_count > 0:
                        _progress(
                            "starting JAX log-density repeat timings "
                            f"count={repeat_count} perturbation={repeat_perturbation:g}"
                        )
                    for repeat_idx in range(repeat_count):
                        if repeat_perturbation == 0.0:
                            theta_eval = theta0_jax
                        else:
                            direction = jnp.where(
                                (jnp.arange(theta0_jax.shape[0]) + repeat_idx) % 2 == 0,
                                jnp.asarray(1.0, dtype=jnp.float64),
                                jnp.asarray(-1.0, dtype=jnp.float64),
                            )
                            theta_eval = theta0_jax + repeat_perturbation * direction
                        repeat_started = time.perf_counter()
                        if bool(args.hlt_jax_log_density_gradient):
                            repeat_value, repeat_grad = repeat_fn(theta_eval)
                            _block_until_ready_tree((repeat_value, repeat_grad))
                            repeat_gradient_norms.append(
                                float(np.linalg.norm(np.asarray(repeat_grad, dtype=np.float64)))
                            )
                        else:
                            repeat_value = repeat_fn(theta_eval)
                            _block_until_ready_tree(repeat_value)
                        repeat_timings.append(time.perf_counter() - repeat_started)
                        repeat_values.append(float(np.asarray(repeat_value)))

                    batch_size = max(0, int(args.hlt_jax_log_density_batch_size))
                    batch_repeat_count = max(
                        0,
                        int(args.hlt_jax_log_density_batch_repeat_evals),
                    )
                    batch_perturbation = float(args.hlt_jax_log_density_batch_perturbation)
                    if batch_size > 0:
                        base_direction = jnp.where(
                            jnp.arange(theta0_jax.shape[0]) % 2 == 0,
                            jnp.asarray(1.0, dtype=jnp.float64),
                            jnp.asarray(-1.0, dtype=jnp.float64),
                        )
                        theta_batch = jnp.stack(
                            [
                                theta0_jax
                                + (
                                    batch_perturbation
                                    * jnp.asarray(i, dtype=jnp.float64)
                                    * base_direction
                                )
                                for i in range(batch_size)
                            ],
                            axis=0,
                        )
                        _progress(
                            "starting batched JAX log-density timing "
                            f"batch_size={batch_size} repeats={batch_repeat_count} "
                            f"perturbation={batch_perturbation:g}"
                        )
                        if bool(args.hlt_jax_log_density_gradient):
                            batched_fn = jax.jit(jax.vmap(jax.value_and_grad(log_density)))
                        else:
                            batched_fn = jax.jit(jax.vmap(log_density))
                        batch_started = time.perf_counter()
                        batched_result = batched_fn(theta_batch)
                        _block_until_ready_tree(batched_result)
                        batched_first_s = time.perf_counter() - batch_started

                        if bool(args.hlt_jax_log_density_gradient):
                            batch_values_arr, batch_grad_arr = batched_result
                            batched_values.append(
                                np.asarray(batch_values_arr, dtype=np.float64).tolist()
                            )
                            batched_gradient_norms.append(
                                np.linalg.norm(
                                    np.asarray(batch_grad_arr, dtype=np.float64),
                                    axis=1,
                                ).tolist()
                            )
                        else:
                            batched_values.append(
                                np.asarray(batched_result, dtype=np.float64).tolist()
                            )

                        for batch_repeat_idx in range(batch_repeat_count):
                            if batch_perturbation == 0.0:
                                theta_eval_batch = theta_batch
                            else:
                                theta_eval_batch = theta_batch + (
                                    batch_perturbation
                                    * jnp.asarray(batch_repeat_idx + 1, dtype=jnp.float64)
                                    * base_direction[None, :]
                                )
                            batch_repeat_started = time.perf_counter()
                            batched_repeat_result = batched_fn(theta_eval_batch)
                            _block_until_ready_tree(batched_repeat_result)
                            batched_repeat_timings.append(
                                time.perf_counter() - batch_repeat_started
                            )
                            if bool(args.hlt_jax_log_density_gradient):
                                batch_values_arr, batch_grad_arr = batched_repeat_result
                                batched_values.append(
                                    np.asarray(batch_values_arr, dtype=np.float64).tolist()
                                )
                                batched_gradient_norms.append(
                                    np.linalg.norm(
                                        np.asarray(batch_grad_arr, dtype=np.float64),
                                        axis=1,
                                    ).tolist()
                                )
                            else:
                                batched_values.append(
                                    np.asarray(batched_repeat_result, dtype=np.float64).tolist()
                                )

                    value_float = float(np.asarray(value))
                    python_total = _finite_float_or_none(likelihood_result.get("total_loglikelihood"))
                    parity = _parity_metrics(
                        value=value_float,
                        reference=python_total,
                        atol=float(args.hlt_jax_python_parity_tol),
                        rtol=float(args.hlt_jax_python_parity_rtol),
                    )
                    jax_log_density_result = {
                        "status": "ok",
                        "elapsed_s": jax_elapsed,
                        "repeat_eval_count": repeat_count,
                        "repeat_eval_timings_s": repeat_timings,
                        "repeat_eval_median_s": None
                        if not repeat_timings
                        else float(statistics.median(repeat_timings)),
                        "repeat_eval_min_s": None if not repeat_timings else float(min(repeat_timings)),
                        "repeat_eval_max_s": None if not repeat_timings else float(max(repeat_timings)),
                        "repeat_eval_values": repeat_values,
                        "repeat_eval_gradient_norms": repeat_gradient_norms,
                        "repeat_eval_perturbation": repeat_perturbation,
                        "batched_eval_batch_size": batch_size,
                        "batched_eval_first_s": batched_first_s,
                        "batched_eval_first_per_theta_s": None
                        if batched_first_s is None or batch_size <= 0
                        else float(batched_first_s / batch_size),
                        "batched_eval_repeat_count": batch_repeat_count,
                        "batched_eval_repeat_timings_s": batched_repeat_timings,
                        "batched_eval_repeat_median_s": None
                        if not batched_repeat_timings
                        else float(statistics.median(batched_repeat_timings)),
                        "batched_eval_repeat_median_per_theta_s": None
                        if not batched_repeat_timings or batch_size <= 0
                        else float(statistics.median(batched_repeat_timings) / batch_size),
                        "batched_eval_repeat_min_s": None
                        if not batched_repeat_timings
                        else float(min(batched_repeat_timings)),
                        "batched_eval_repeat_max_s": None
                        if not batched_repeat_timings
                        else float(max(batched_repeat_timings)),
                        "batched_eval_values": batched_values,
                        "batched_eval_gradient_norms": batched_gradient_norms,
                        "batched_eval_perturbation": batch_perturbation,
                        "value": value_float,
                        "python_surrogate_total_loglikelihood": python_total,
                        "value_minus_python": parity["value_minus_reference"],
                        "parity_abs_diff": parity["abs_diff"],
                        "parity_rel_diff": parity["rel_diff"],
                        "parity_scale": parity["scale"],
                        "parity_tol": parity["atol"],
                        "parity_rtol": parity["rtol"],
                        "parity_effective_tol": parity["effective_tol"],
                        "parity_ok": parity["ok"],
                        "gradient_evaluated": bool(args.hlt_jax_log_density_gradient),
                        "gradient": None if grad_np is None else grad_np.tolist(),
                        "gradient_finite": None if grad_np is None else bool(np.isfinite(grad_np).all()),
                        "gradient_norm": None if grad_np is None else float(np.linalg.norm(grad_np)),
                        "backend": jax.default_backend(),
                        "target_device": None if target_device is None else str(target_device),
                        "runtime_mode": likelihood_runtime_mode,
                        "qme_algorithm": likelihood_qme_algorithm,
                        "static_rows_mode": likelihood_static_rows_mode,
                        "static_equation_rows": None
                        if likelihood_static_rows is None
                        else [int(row) for row in likelihood_static_rows],
                        "failure_loglikelihood": float(args.hlt_likelihood_on_failure),
                        "shock_solver": str(args.hlt_jax_shock_solver),
                        "batch_replay": bool(args.hlt_jax_batch_replay),
                        "differentiate_shocks": bool(args.hlt_jax_differentiate_shocks),
                        "caveat": (
                            (
                                "Differentiates the full JAX surrogate likelihood through parameter-dependent "
                                "steady-state and first-order ROM solves."
                                if bool(args.hlt_jax_log_density_gradient)
                                else "Evaluates the full JAX surrogate likelihood with parameter-dependent "
                                "steady-state and first-order ROM solves; gradient evaluation was disabled."
                            )
                            if likelihood_runtime_mode == "full-jax"
                            else (
                                "Differentiates the fixed-ROM surrogate likelihood through theta and inferred shocks; "
                                "steady-state and first-order matrices are held fixed in this smoke check."
                                if bool(args.hlt_jax_differentiate_shocks)
                                else "Differentiates the fixed-ROM surrogate likelihood through theta with inferred shocks treated as stop-gradient replay inputs; "
                                "steady-state and first-order matrices are held fixed in this smoke check."
                            )
                        ),
                    }
                    _progress(
                        "finished JAX log-density smoke "
                        f"elapsed={jax_elapsed:.3f}s parity_ok={jax_log_density_result['parity_ok']} "
                        f"grad_norm={jax_log_density_result['gradient_norm']} "
                        f"repeat_median={jax_log_density_result['repeat_eval_median_s']}"
                    )

                if int(args.hlt_surrogate_hmc_samples) > 0:
                    _progress(
                        "starting surrogate HMC "
                        f"runtime_mode={likelihood_runtime_mode} "
                        f"chains={args.hlt_surrogate_hmc_chains} "
                        f"warmup={args.hlt_surrogate_hmc_warmup} "
                        f"samples={args.hlt_surrogate_hmc_samples}"
                    )
                    lower, upper = _hlt_uniform_prior_arrays(
                        parameter_subset,
                        theta0,
                        width_scale=float(args.hlt_surrogate_hmc_prior_width_scale),
                        width_floor=float(args.hlt_surrogate_hmc_prior_width_floor),
                    )
                    surrogate_hmc_result = run_static_hmc_on_bounded_surrogate_log_density(
                        log_density_fn=log_density,
                        center=theta0_jax,
                        parameter_names=parameter_subset,
                        lower=device_array(lower),
                        upper=device_array(upper),
                        chains=int(args.hlt_surrogate_hmc_chains),
                        warmup=int(args.hlt_surrogate_hmc_warmup),
                        samples=int(args.hlt_surrogate_hmc_samples),
                        leapfrog_steps=int(args.hlt_surrogate_hmc_leapfrog_steps),
                        step_size=float(args.hlt_surrogate_hmc_step_size),
                        target_accept_prob=float(args.hlt_surrogate_hmc_target_accept_prob),
                        adapt_step_size=not bool(args.hlt_surrogate_hmc_no_adapt_step_size),
                        initial_jitter=float(args.hlt_surrogate_hmc_initial_jitter),
                        seed=int(args.hlt_surrogate_hmc_seed),
                        min_accepted_share=float(args.hlt_surrogate_hmc_min_accepted_share),
                        max_retries=int(args.hlt_surrogate_hmc_max_retries),
                        retry_step_size_factor=float(args.hlt_surrogate_hmc_retry_step_size_factor),
                    )
                    surrogate_hmc_result["runtime_mode"] = likelihood_runtime_mode
                    surrogate_hmc_result["qme_algorithm"] = likelihood_qme_algorithm
                    surrogate_hmc_result["static_rows_mode"] = likelihood_static_rows_mode
                    _progress(
                        "finished static surrogate HMC "
                        f"elapsed={surrogate_hmc_result.get('elapsed_s')} "
                        f"draws_per_second={surrogate_hmc_result.get('draws_per_second')}"
                    )
        except Exception as exc:
            _progress(f"likelihood/HMC stage failed: {exc!r}")
            if python_likelihood_completed:
                jax_log_density_result = {
                    "status": "error",
                    "elapsed_s": time.perf_counter() - likelihood_started,
                    "error": repr(exc),
                }
                surrogate_hmc_result = {
                    "status": "skipped",
                    "reason": "JAX log-density/HMC stage failed after Python likelihood completed.",
                }
            else:
                likelihood_result = {
                    "status": "error",
                    "elapsed_s": time.perf_counter() - likelihood_started,
                    "error": repr(exc),
                }
                jax_log_density_result = {
                    "status": "skipped",
                    "reason": "Python surrogate likelihood failed before JAX log-density smoke.",
                }
                surrogate_hmc_result = {
                    "status": "skipped",
                    "reason": "Python surrogate likelihood failed before surrogate HMC.",
                }

    steady_statuses = [str(row["steady_state_status"]) for row in steady_state_diagnostics]
    fallback_count = sum(status.startswith("fallback") for status in steady_statuses)
    solved_count = sum(status == "solved" for status in steady_statuses)
    caveats = [
        "SEP target generation is still callback/Python-loop based; ResNet training is the GPU-native part.",
        "The likelihood block evaluates a trained-surrogate inversion likelihood.",
    ]
    likelihood_runtime_mode_summary = str(args.hlt_likelihood_runtime_mode).strip().lower()
    if likelihood_runtime_mode_summary == "full-jax":
        caveats.append(
            "Optional surrogate HMC recomputes steady states and first-order ROM matrices inside the JAX log density."
        )
    else:
        caveats.append(
            "Optional surrogate HMC samples a fixed-reference steady-state/fixed-ROM likelihood."
        )
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
        "hlt_parameter_set": str(args.hlt_parameter_set),
        "theta_draws": int(theta.shape[1]),
        "steady_state_mode": steady_state_mode,
        "first_order_qme_algorithm": str(args.hlt_first_order_qme_algorithm),
        "likelihood_runtime_mode": likelihood_runtime_mode_summary,
        "likelihood_qme_algorithm": str(args.hlt_likelihood_qme_algorithm),
        "likelihood_static_rows_mode": str(args.hlt_likelihood_static_rows_mode),
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
        "target_diagnostics": target_diagnostics,
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
        "surrogate_hmc": surrogate_hmc_result,
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
            "Synthetic fixed-shape batched rollout training can be profiled with --mode batched-training.",
            "Synthetic batched SEP target generation plus training can be profiled with --mode batched-sep-training.",
            "Actual parsed HLT SEP target generation is still callback/Python-loop based, not a fully batched JAX kernel.",
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
            "batched-training",
            "sep-micro",
            "batched-sep-micro",
            "batched-sep-training",
            "parsed-batched-sep-training",
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
    parser.add_argument("--batched-mask-fraction", type=float, default=0.0)
    parser.add_argument("--only-full-success", action="store_true")
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
    parser.add_argument("--sep-batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-batched-sep-likelihood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip the JAX likelihood-gradient smoke in --mode batched-sep-training. "
            "Use this for large target-generation throughput profiles where shock "
            "inversion/AD compilation would otherwise dominate the timing."
        ),
    )
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
    parser.add_argument(
        "--hlt-parameter-set",
        default="payload",
        help=(
            "HLT parameter subset to estimate: 'payload', 'sw07_safe_15', "
            "'sw07_safe_27', 'all', or a comma-separated list of parameter names."
        ),
    )
    parser.add_argument("--hlt-periods", type=int, default=2)
    parser.add_argument("--hlt-theta-draws", type=int, default=2)
    parser.add_argument("--hlt-shock-scale", type=float, default=0.02)
    parser.add_argument("--hlt-parameter-perturbation", type=float, default=1e-6)
    parser.add_argument(
        "--hlt-target-builder",
        choices=("adaptive-sep", "callback"),
        default="adaptive-sep",
        help=(
            "How to build HLT surrogate targets. 'adaptive-sep' tries an ordered "
            "ladder of accepted SEP solves per theta-period and records diagnostics; "
            "'callback' preserves the original one-config callback path."
        ),
    )
    parser.add_argument(
        "--hlt-target-min-stable-periods",
        type=int,
        default=-1,
        help=(
            "Minimum stable prefix length needed to keep a theta draw. "
            "Use -1 to require the requested HLT period count."
        ),
    )
    parser.add_argument(
        "--hlt-sep-order-ladder",
        default="auto",
        help=(
            "Comma-separated SEP branching-order candidates for adaptive HLT targets. "
            "'auto' starts with --sep-order and optionally appends 0."
        ),
    )
    parser.add_argument(
        "--hlt-sep-periods-ladder",
        default="auto",
        help=(
            "Comma-separated SEP horizon candidates for adaptive HLT targets. "
            "'auto' tries --sep-periods first and then 1 if different."
        ),
    )
    parser.add_argument(
        "--hlt-adaptive-include-order-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the adaptive HLT target ladder may fall back to deterministic order-0 SEP.",
    )
    parser.add_argument(
        "--hlt-sep-max-iter-ladder",
        default="auto",
        help="Comma-separated max-iteration candidates for adaptive HLT SEP targets; 'auto' uses --sep-max-iter.",
    )
    parser.add_argument(
        "--hlt-sep-shock-scale-ladder",
        default="1.0,0.5,0.25,0.1,0.0",
        help="Comma-separated deterministic-shock multipliers for adaptive HLT SEP target attempts.",
    )
    parser.add_argument("--hlt-target-max-logged-failures", type=int, default=20)
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
    parser.add_argument(
        "--hlt-first-order-qme-algorithm",
        choices=("doubling", "schur", "schur_gpu"),
        default="schur",
        help="QME algorithm used by the Python first-order runtime for HLT target generation.",
    )
    parser.add_argument("--hlt-likelihood-periods", type=int, default=1)
    parser.add_argument(
        "--hlt-likelihood-runtime-mode",
        choices=("fixed-reference", "full-jax"),
        default="fixed-reference",
        help=(
            "'fixed-reference' keeps the old HMC smoke path with frozen steady state and ROM; "
            "'full-jax' recomputes steady state and first-order matrices inside the JAX log density."
        ),
    )
    parser.add_argument(
        "--hlt-likelihood-qme-algorithm",
        choices=("doubling", "schur", "schur_gpu"),
        default="schur",
        help="QME algorithm used by the full-JAX likelihood runtime.",
    )
    parser.add_argument(
        "--hlt-likelihood-static-rows-mode",
        choices=("reference", "none"),
        default="reference",
        help=(
            "Static present-only equation rows passed to the full-JAX first-order solve. "
            "'reference' avoids differentiating complete QR; 'none' uses generic QR and is mainly diagnostic."
        ),
    )
    parser.add_argument("--hlt-likelihood-on-failure", type=float, default=-1e12)
    parser.add_argument("--hlt-surrogate-inversion-maxit", type=int, default=4)
    parser.add_argument("--hlt-surrogate-inversion-tol", type=float, default=1e-5)
    parser.add_argument("--hlt-surrogate-inversion-lambda", type=float, default=1e-4)
    parser.add_argument("--hlt-jax-log-density-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hlt-jax-log-density-gradient", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hlt-jax-log-density-repeat-evals",
        type=int,
        default=0,
        help="Number of post-compile repeated JAX log-density evaluations to time.",
    )
    parser.add_argument(
        "--hlt-jax-log-density-repeat-perturbation",
        type=float,
        default=0.0,
        help="Add alternating +/- perturbations of this size to theta during repeat timing.",
    )
    parser.add_argument(
        "--hlt-jax-log-density-batch-size",
        type=int,
        default=0,
        help="Batch size for vmapped post-smoke JAX log-density timing.",
    )
    parser.add_argument(
        "--hlt-jax-log-density-batch-repeat-evals",
        type=int,
        default=0,
        help="Number of post-compile vmapped JAX log-density batches to time.",
    )
    parser.add_argument(
        "--hlt-jax-log-density-batch-perturbation",
        type=float,
        default=0.0,
        help="Theta perturbation step used across vmapped log-density batch entries.",
    )
    parser.add_argument("--hlt-jax-shock-solver", choices=("rom", "surrogate"), default="rom")
    parser.add_argument("--hlt-jax-batch-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hlt-jax-differentiate-shocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hlt-jax-python-parity-tol", type=float, default=1e-7)
    parser.add_argument(
        "--hlt-jax-python-parity-rtol",
        type=float,
        default=1e-9,
        help=(
            "Relative tolerance for Python-vs-JAX log-density parity. The effective "
            "tolerance is atol + rtol * max(abs(Python value), 1)."
        ),
    )
    parser.add_argument("--hlt-surrogate-hmc-warmup", type=int, default=0)
    parser.add_argument("--hlt-surrogate-hmc-samples", type=int, default=0)
    parser.add_argument("--hlt-surrogate-hmc-chains", type=int, default=1)
    parser.add_argument("--hlt-surrogate-hmc-leapfrog-steps", type=int, default=4)
    parser.add_argument("--hlt-surrogate-hmc-step-size", type=float, default=0.05)
    parser.add_argument("--hlt-surrogate-hmc-target-accept-prob", type=float, default=0.8)
    parser.add_argument("--hlt-surrogate-hmc-initial-jitter", type=float, default=0.02)
    parser.add_argument("--hlt-surrogate-hmc-prior-width-scale", type=float, default=0.01)
    parser.add_argument("--hlt-surrogate-hmc-prior-width-floor", type=float, default=1e-4)
    parser.add_argument("--hlt-surrogate-hmc-no-adapt-step-size", action="store_true")
    parser.add_argument(
        "--hlt-surrogate-hmc-min-accepted-share",
        type=float,
        default=0.01,
        help="Automatically retry with a smaller initial step size if HMC acceptance is below this share.",
    )
    parser.add_argument(
        "--hlt-surrogate-hmc-max-retries",
        type=int,
        default=3,
        help="Maximum number of smaller-step HMC retries after a low-acceptance run.",
    )
    parser.add_argument(
        "--hlt-surrogate-hmc-retry-step-size-factor",
        type=float,
        default=0.25,
        help="Multiplicative initial-step-size shrinkage applied on each HMC retry.",
    )
    parser.add_argument("--hlt-surrogate-hmc-seed", type=int, default=20260923)
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
    elif args.mode == "batched-training":
        payload["results"]["batched_training"] = run_batched_training_profile(args, shape)
    elif args.mode == "sep-micro":
        payload["results"]["sep_micro"] = run_sep_micro_profile(args)
    elif args.mode == "batched-sep-micro":
        payload["results"]["batched_sep_micro"] = run_batched_sep_micro_profile(args)
    elif args.mode == "batched-sep-training":
        payload["results"]["batched_sep_training"] = run_batched_sep_training_profile(args, shape)
    elif args.mode == "parsed-batched-sep-training":
        payload["results"]["parsed_batched_sep_training"] = run_parsed_batched_sep_training_profile(args)
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
