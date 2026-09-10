from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import posterior_sampling_speed as posterior_speed  # noqa: E402
import profile_batched_likelihood as batched_profile  # noqa: E402
import surrogatenn_dsge as sdsge  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "results" / "static_hmc_sampling_speed.json"


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


def _timed_call(fn):
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


def _samples_by_chain(
    constrained_samples: Any,
    parameter_names: Sequence[str],
) -> dict[str, np.ndarray]:
    array = np.asarray(constrained_samples, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("constrained_samples must have shape (samples, chains, parameters)")
    if array.shape[-1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample parameter dimension")
    return {
        name: np.swapaxes(array[:, :, idx], 0, 1)
        for idx, name in enumerate(parameter_names)
    }


def _acceptance_summary(result: Any, num_leapfrog_steps: int) -> dict[str, Any]:
    accept_prob = np.asarray(result.accept_prob, dtype=np.float64)
    accepted = np.asarray(result.accepted, dtype=bool)
    warmup_accept_prob = np.asarray(result.warmup_accept_prob, dtype=np.float64)
    return {
        "accept_prob": {
            "mean": float(np.mean(accept_prob)) if accept_prob.size else None,
            "median": float(np.median(accept_prob)) if accept_prob.size else None,
            "min": float(np.min(accept_prob)) if accept_prob.size else None,
            "max": float(np.max(accept_prob)) if accept_prob.size else None,
        },
        "accepted_share": float(np.mean(accepted)) if accepted.size else None,
        "warmup_accept_prob": {
            "mean": float(np.mean(warmup_accept_prob)) if warmup_accept_prob.size else None,
            "median": float(np.median(warmup_accept_prob)) if warmup_accept_prob.size else None,
            "min": float(np.min(warmup_accept_prob)) if warmup_accept_prob.size else None,
            "max": float(np.max(warmup_accept_prob)) if warmup_accept_prob.size else None,
        },
        "num_steps": {
            "mean": float(num_leapfrog_steps),
            "median": float(num_leapfrog_steps),
            "min": float(num_leapfrog_steps),
            "max": float(num_leapfrog_steps),
        },
    }


def _diagnostics(
    *,
    result: Any,
    constrained_samples: Any,
    parameter_names: Sequence[str],
    elapsed_s: float,
    num_leapfrog_steps: int,
) -> dict[str, Any]:
    samples_by_chain = _samples_by_chain(constrained_samples, parameter_names)
    sample_summary = posterior_speed._sample_summary(samples_by_chain, parameter_names)
    post_warmup_draws = int(np.prod(np.asarray(result.accepted).shape))
    min_ess = sample_summary["min_ess"]
    mean_ess = sample_summary["mean_ess"]
    return {
        "timing_s": float(elapsed_s),
        "post_warmup_draws": post_warmup_draws,
        "draws_per_second": float(post_warmup_draws / elapsed_s) if elapsed_s > 0 else None,
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "seconds_per_min_ess": (
            float(elapsed_s / min_ess) if isinstance(min_ess, float) and min_ess > 0 else None
        ),
        "seconds_per_mean_ess": (
            float(elapsed_s / mean_ess)
            if isinstance(mean_ess, float) and mean_ess > 0
            else None
        ),
        "posterior_diagnostics": sample_summary,
        "acceptance": _acceptance_summary(result, num_leapfrog_steps),
        "final_step_size": float(np.asarray(result.step_size, dtype=np.float64)),
        "final_log_prob_mean": float(np.mean(np.asarray(result.final_log_prob, dtype=np.float64))),
    }


def run_static_hmc_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()

    def log(message: str) -> None:
        if args.verbose:
            elapsed = time.perf_counter() - started_at
            print(f"[static-hmc +{elapsed:9.2f}s] {message}", flush=True)

    log("configure runtime and build likelihood context")
    context = batched_profile._make_likelihood_context(args)
    jax = context["jax"]
    jnp = context["jnp"]
    model = context["model"]
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    theta0 = jnp.asarray(context["parameter_values"])
    index = jnp.asarray(context["parameter_index"], dtype=jnp.int32)
    observations = jnp.asarray(context["observations"])
    steady_state = jnp.asarray(context["steady_state"])
    lower = jnp.asarray(context["lower"], dtype=theta0.dtype)
    upper = jnp.asarray(context["upper"], dtype=theta0.dtype)
    center = jnp.asarray(context["center"], dtype=theta0.dtype)
    prior_log_const = -jnp.sum(jnp.log(upper - lower))

    def log_posterior_unconstrained(unconstrained_subset):
        constrained_subset = sdsge.unconstrained_to_bounded(
            unconstrained_subset,
            lower,
            upper,
        )
        theta = theta0.at[index].set(constrained_subset)
        loglikelihood = sdsge.kalman_loglikelihood_from_model_jax(
            model,
            observations,
            observables=context["observables"],
            parameter_values=theta,
            steady_state=steady_state,
            measurement_error_scale=context["measurement_error_scale"],
            jitter=context["jitter"],
            qme_algorithm=args.qme_algorithm,
            on_failure_loglikelihood=float(args.failure_value),
            static_equation_rows=context["static_equation_rows"],
            parameters_are_resolved=bool(args.parameters_are_resolved),
            check_parameter_bounds=not bool(args.skip_parameter_bounds),
        )
        return (
            loglikelihood
            + prior_log_const
            + sdsge.bounded_log_abs_det_jacobian(unconstrained_subset, lower, upper)
        )

    initial_center = sdsge.bounded_to_unconstrained(center, lower, upper)
    init_key, run_key = jax.random.split(jax.random.PRNGKey(int(args.seed)))
    initial_position = initial_center[None, :] + float(args.initial_jitter) * jax.random.normal(
        init_key,
        shape=(int(args.chains), int(initial_center.shape[0])),
        dtype=jnp.float32 if args.dtype == "float32" else jnp.float64,
    )
    initial_position = jnp.asarray(initial_position, dtype=theta0.dtype)

    log(
        "context ready: "
        f"vars={model.timings.nVars}, exo={model.timings.nExo}, "
        f"observations={context['observations'].shape}, "
        f"parameters={context['parameter_names']}, chains={args.chains}",
    )

    def sample_once(key):
        return sdsge.static_hmc_sample(
            log_posterior_unconstrained,
            initial_position,
            key,
            num_warmup=int(args.warmup),
            num_samples=int(args.samples),
            step_size=float(args.step_size),
            num_leapfrog_steps=int(args.leapfrog_steps),
            target_accept_prob=float(args.target_accept_prob),
            adapt_step_size=not bool(args.no_adapt_step_size),
            adaptation_rate=float(args.adaptation_rate),
            min_step_size=float(args.min_step_size),
            max_step_size=float(args.max_step_size),
        )

    compiled_sampler = jax.jit(sample_once)
    log("START cold static-HMC run")
    first_key, *steady_keys = jax.random.split(run_key, int(args.steady_reps) + 1)
    first_result, first_s = _timed_call(lambda: compiled_sampler(first_key))
    log(f"END cold static-HMC run in {first_s:.3f}s")
    constrained_first = sdsge.unconstrained_to_bounded(first_result.samples, lower, upper)
    first_diagnostics = _diagnostics(
        result=first_result,
        constrained_samples=constrained_first,
        parameter_names=context["parameter_names"],
        elapsed_s=first_s,
        num_leapfrog_steps=int(args.leapfrog_steps),
    )
    steady_times: list[float] = []
    steady_diagnostics = None
    for rep, key in enumerate(steady_keys, start=1):
        log(f"START steady static-HMC run {rep}/{args.steady_reps}")
        result, elapsed = _timed_call(lambda key=key: compiled_sampler(key))
        steady_times.append(elapsed)
        log(f"END steady static-HMC run {rep}/{args.steady_reps} in {elapsed:.3f}s")
        constrained = sdsge.unconstrained_to_bounded(result.samples, lower, upper)
        steady_diagnostics = _diagnostics(
            result=result,
            constrained_samples=constrained,
            parameter_names=context["parameter_names"],
            elapsed_s=elapsed,
            num_leapfrog_steps=int(args.leapfrog_steps),
        )

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
            "chains": int(args.chains),
            "warmup": int(args.warmup),
            "samples": int(args.samples),
            "leapfrog_steps": int(args.leapfrog_steps),
            "initial_step_size": float(args.step_size),
            "target_accept_prob": float(args.target_accept_prob),
            "adapt_step_size": not bool(args.no_adapt_step_size),
            "adaptation_rate": float(args.adaptation_rate),
            "initial_jitter": float(args.initial_jitter),
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
        "cold_run": first_diagnostics,
        "steady_timing": _timing_stats(steady_times),
        "steady_last_run": steady_diagnostics,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lean static JAX HMC benchmark over the DSGE Kalman posterior."
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
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--prior-width-scale", type=float, default=0.0025)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--leapfrog-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--adaptation-rate", type=float, default=0.05)
    parser.add_argument("--min-step-size", type=float, default=1.0e-5)
    parser.add_argument("--max-step-size", type=float, default=1.0)
    parser.add_argument("--initial-jitter", type=float, default=0.05)
    parser.add_argument("--steady-reps", type=int, default=0)
    parser.add_argument("--no-adapt-step-size", action="store_true")
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
        help="Skip model bound checks because the transform keeps draws inside priors.",
    )
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_static_hmc_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    print(json.dumps(result["cold_run"], indent=2, sort_keys=True, default=_json_default))
    if result["steady_last_run"] is not None:
        print(json.dumps(result["steady_last_run"], indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
