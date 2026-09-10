from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import posterior_sampling_speed as posterior_speed  # noqa: E402
import surrogatenn_dsge as sdsge  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "results" / "batched_likelihood_profile.json"


def _block_tree(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            _block_tree(item)
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_tree(item)
        return value
    return value


def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    _block_tree(value)
    return value, time.perf_counter() - start


def _timing_stats(times: Sequence[float]) -> dict[str, Any]:
    if not times:
        return {"reps": 0}
    return {
        "reps": len(times),
        "mean_s": float(statistics.mean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def _make_centered_draws(
    *,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    batch_size: int,
    draw_scale: float,
    dtype: np.dtype[Any],
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if center.ndim != 1:
        raise ValueError("center must be one-dimensional")
    if lower.shape != center.shape or upper.shape != center.shape:
        raise ValueError("lower/upper bounds must match center")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.all(lower < center) or not np.all(center < upper):
        raise ValueError("center must be strictly inside all parameter intervals")
    safe_radius = np.minimum(center - lower, upper - center)
    phase = np.arange(batch_size, dtype=np.float64)[:, None] + 1.0
    frequency = np.arange(center.size, dtype=np.float64)[None, :] + 1.0
    perturbation = np.sin(phase * frequency * 1.618033988749895)
    draws = center[None, :] + float(draw_scale) * safe_radius[None, :] * perturbation
    draws = np.clip(draws, lower[None, :], upper[None, :])
    if batch_size > 0:
        draws[0, :] = center
    return draws.astype(dtype, copy=False)


def _make_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        preset=args.preset,
        payload_path=args.payload_path,
        case=args.case,
        model_source=args.model_source,
        periods=args.periods,
        synthetic_seed=args.synthetic_seed,
        parameters=args.parameters,
        dtype=args.dtype,
        platform=args.platform,
        force_gpu=args.force_gpu,
        host_device_count=None,
        suppress_dtype_warnings=True,
    )


def _make_likelihood_context(args: argparse.Namespace) -> dict[str, Any]:
    dataset_args = _make_dataset_args(args)
    jax, jnp, numpyro, dist, _ = posterior_speed._configure_runtime(dataset_args)
    data = posterior_speed._build_dataset(dataset_args, sdsge, jax)
    model = data["model"]
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    steady_state = np.asarray(data["steady_state"], dtype=np_dtype)
    observations = np.asarray(data["observations"], dtype=np_dtype)
    observables = tuple(data["observables"])
    parameter_names = tuple(data["parameter_names"])
    raw_parameter_values = np.asarray(model.parameter_values, dtype=np.float64)
    resolved_parameter_values = np.asarray(
        model.resolve_parameter_values(
            parameter_values=raw_parameter_values,
            steady_state=np.asarray(data["steady_state"], dtype=np.float64),
        ),
        dtype=np.float64,
    )
    parameter_values = (
        resolved_parameter_values if args.parameters_are_resolved else raw_parameter_values
    ).astype(np_dtype, copy=False)
    priors, _, intervals = posterior_speed._make_centered_uniform_priors(
        dist,
        model,
        parameter_names,
        parameter_values=parameter_values,
        width_scale=float(args.prior_width_scale),
        width_floor=float(args.prior_width_floor),
    )
    del priors
    parameter_index = np.asarray(
        [model.parameter_names.index(name) for name in parameter_names],
        dtype=np.int32,
    )
    lower = np.asarray([intervals[name][0] for name in parameter_names], dtype=np.float64)
    upper = np.asarray([intervals[name][1] for name in parameter_names], dtype=np.float64)
    center = np.asarray(
        [parameter_values[idx] for idx in parameter_index],
        dtype=np.float64,
    )
    static_equation_rows = (
        model._first_order_static_equation_rows_for_values(
            steady_state=steady_state,
            parameter_values=parameter_values,
        )
        if (args.qme_algorithm == "schur_gpu" and not model.has_obc)
        else None
    )
    return {
        "jax": jax,
        "jnp": jnp,
        "numpyro": numpyro,
        "model": model,
        "observations": observations,
        "observables": observables,
        "steady_state": steady_state,
        "parameter_values": parameter_values,
        "parameter_names": parameter_names,
        "parameter_index": parameter_index,
        "center": center,
        "lower": lower,
        "upper": upper,
        "static_equation_rows": static_equation_rows,
        "measurement_error_scale": float(data["measurement_error_scale"]),
        "jitter": float(data["jitter"]),
        "resolved_parameter_max_abs_diff": float(
            np.max(np.abs(resolved_parameter_values - raw_parameter_values))
        )
        if raw_parameter_values.size
        else 0.0,
    }


def run_batched_profile(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()

    def log(message: str) -> None:
        if args.verbose:
            elapsed = time.perf_counter() - started_at
            print(f"[batched-profile +{elapsed:9.2f}s] {message}", flush=True)

    log("configure runtime and build dataset")
    context = _make_likelihood_context(args)
    jax = context["jax"]
    jnp = context["jnp"]
    model = context["model"]
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    theta0 = jnp.asarray(context["parameter_values"])
    index = jnp.asarray(context["parameter_index"], dtype=jnp.int32)
    obs = jnp.asarray(context["observations"])
    steady = jnp.asarray(context["steady_state"])

    def inject(x):
        return theta0.at[index].set(x)

    def loglikelihood_from_subset(x):
        return sdsge.kalman_loglikelihood_from_model_jax(
            model,
            obs,
            observables=context["observables"],
            parameter_values=inject(x),
            steady_state=steady,
            measurement_error_scale=context["measurement_error_scale"],
            jitter=context["jitter"],
            qme_algorithm=args.qme_algorithm,
            on_failure_loglikelihood=float(args.failure_value),
            static_equation_rows=context["static_equation_rows"],
            parameters_are_resolved=bool(args.parameters_are_resolved),
            check_parameter_bounds=not bool(args.skip_parameter_bounds),
        )

    value_batch = jax.jit(jax.vmap(loglikelihood_from_subset))
    value_grad_batch = jax.jit(jax.vmap(jax.value_and_grad(loglikelihood_from_subset)))
    max_batch = max(args.batch_sizes)
    log(
        "dataset ready: "
        f"vars={model.timings.nVars}, exo={model.timings.nExo}, "
        f"observations={context['observations'].shape}, "
        f"parameters={context['parameter_names']}, max_batch={max_batch}",
    )
    draws = _make_centered_draws(
        center=context["center"],
        lower=context["lower"],
        upper=context["upper"],
        batch_size=max_batch,
        draw_scale=float(args.draw_scale),
        dtype=np_dtype,
    )
    results: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        log(f"START batch {batch_size} value first call/JIT")
        batch = jnp.asarray(draws[:batch_size])
        value, first_value_s = _timed_call(lambda: value_batch(batch))
        log(f"END batch {batch_size} value first call/JIT in {first_value_s:.3f}s")
        value_times: list[float] = []
        for rep in range(max(args.reps, 0)):
            log(f"START batch {batch_size} value steady rep {rep + 1}/{args.reps}")
            _, elapsed = _timed_call(lambda: value_batch(batch))
            value_times.append(elapsed)
            log(
                f"END batch {batch_size} value steady rep {rep + 1}/{args.reps} "
                f"in {elapsed:.3f}s"
            )
        value_array = np.asarray(value, dtype=np.float64)
        batch_result: dict[str, Any] = {
            "value_first_call_s": float(first_value_s),
            "value_steady": _timing_stats(value_times),
            "value_per_draw_steady_median_s": (
                float(statistics.median(value_times) / batch_size) if value_times else None
            ),
            "value_draws_per_second_median": (
                float(batch_size / statistics.median(value_times)) if value_times else None
            ),
            "loglikelihood_min": float(np.min(value_array)),
            "loglikelihood_max": float(np.max(value_array)),
            "failure_count": int(np.sum(value_array <= float(args.failure_value) / 10.0)),
        }
        if args.include_gradient:
            log(f"START batch {batch_size} gradient first call/JIT")
            grad_value, first_grad_s = _timed_call(lambda: value_grad_batch(batch))
            log(f"END batch {batch_size} gradient first call/JIT in {first_grad_s:.3f}s")
            grad_times: list[float] = []
            for rep in range(max(args.gradient_reps, 0)):
                log(
                    f"START batch {batch_size} gradient steady rep "
                    f"{rep + 1}/{args.gradient_reps}"
                )
                _, elapsed = _timed_call(lambda: value_grad_batch(batch))
                grad_times.append(elapsed)
                log(
                    f"END batch {batch_size} gradient steady rep "
                    f"{rep + 1}/{args.gradient_reps} in {elapsed:.3f}s"
                )
            grad_values, gradients = grad_value
            grad_value_array = np.asarray(grad_values, dtype=np.float64)
            grad_array = np.asarray(gradients, dtype=np.float64)
            batch_result.update(
                {
                    "gradient_first_call_s": float(first_grad_s),
                    "gradient_steady": _timing_stats(grad_times),
                    "gradient_per_draw_steady_median_s": (
                        float(statistics.median(grad_times) / batch_size)
                        if grad_times
                        else None
                    ),
                    "gradient_draws_per_second_median": (
                        float(batch_size / statistics.median(grad_times))
                        if grad_times
                        else None
                    ),
                    "gradient_loglikelihood_min": float(np.min(grad_value_array)),
                    "gradient_loglikelihood_max": float(np.max(grad_value_array)),
                    "gradient_max_abs": float(np.max(np.abs(grad_array))),
                    "gradient_failure_count": int(
                        np.sum(grad_value_array <= float(args.failure_value) / 10.0)
                    ),
                }
            )
        results[str(batch_size)] = batch_result
        log(f"END batch {batch_size}")
    return {
        "benchmark": {
            "preset": args.preset,
            "case": args.case if args.preset == "sw07_hlt" else None,
            "periods": int(context["observations"].shape[1]),
            "n_observables": int(context["observations"].shape[0]),
            "n_vars": int(model.timings.nVars),
            "n_exo": int(model.timings.nExo),
            "parameter_count": len(context["parameter_names"]),
            "parameter_names": list(context["parameter_names"]),
            "qme_algorithm": args.qme_algorithm,
            "dtype": args.dtype,
            "batch_sizes": list(args.batch_sizes),
            "parameters_are_resolved": bool(args.parameters_are_resolved),
            "check_parameter_bounds": not bool(args.skip_parameter_bounds),
            "resolved_parameter_max_abs_diff": context["resolved_parameter_max_abs_diff"],
            "static_equation_rows": (
                list(context["static_equation_rows"])
                if context["static_equation_rows"] is not None
                else None
            ),
        },
        "runtime": posterior_speed._runtime_info(jax, context["numpyro"]),
        "batches": results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile batched JAX DSGE Kalman likelihood and gradient throughput "
            "using jit(vmap(...))."
        )
    )
    parser.add_argument("--preset", choices=("toy_ar2", "sw07_hlt"), default="sw07_hlt")
    parser.add_argument(
        "--payload-path",
        type=Path,
        default=posterior_speed.DEFAULT_PAYLOAD_PATH,
    )
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument(
        "--model-source",
        type=Path,
        default=posterior_speed.DEFAULT_SW07_MODEL_SOURCE_PATH,
    )
    parser.add_argument("--parameters", default="sw07_safe_15")
    parser.add_argument("--periods", type=int, default=80)
    parser.add_argument("--synthetic-seed", type=int, default=20260712)
    parser.add_argument("--prior-width-scale", type=float, default=0.0025)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--draw-scale", type=float, default=0.35)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 4, 16, 64))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--include-gradient", action="store_true")
    parser.add_argument("--gradient-reps", type=int, default=3)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--qme-algorithm",
        choices=("schur", "schur_gpu", "doubling"),
        default="schur_gpu",
    )
    parser.add_argument(
        "--parameters-are-resolved",
        action="store_true",
        help=(
            "Use a parameter vector pre-resolved against calibration equations. "
            "Only use this when sampled parameters do not require dependent "
            "calibration values to be recomputed."
        ),
    )
    parser.add_argument(
        "--skip-parameter-bounds",
        action="store_true",
        help="Skip model bound checks when generated draws are already in support.",
    )
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_batched_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    print(json.dumps(result["batches"], indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
