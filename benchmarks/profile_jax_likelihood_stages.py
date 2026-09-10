from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import posterior_sampling_speed as posterior_speed  # noqa: E402
import surrogatenn_dsge as sdsge  # noqa: E402
from surrogatenn_dsge.inference import _linear_state_space_from_first_order_solution_jax  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "results" / "jax_likelihood_stage_profile.json"


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


def _make_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
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


def _time_stage(
    *,
    jax: Any,
    name: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    reps: int,
    verbose: bool,
) -> tuple[Any, dict[str, Any]]:
    if verbose:
        print(f"[stage-profile] START {name} first call/JIT", flush=True)
    compiled = jax.jit(fn)
    value, first_s = _timed_call(lambda: compiled(*args))
    if verbose:
        print(f"[stage-profile] END {name} first call/JIT in {first_s:.6f}s", flush=True)
    steady_times: list[float] = []
    for rep in range(max(0, reps)):
        if verbose:
            print(f"[stage-profile] START {name} steady rep {rep + 1}/{reps}", flush=True)
        _, elapsed = _timed_call(lambda: compiled(*args))
        steady_times.append(elapsed)
        if verbose:
            print(f"[stage-profile] END {name} steady rep {rep + 1}/{reps} in {elapsed:.6f}s", flush=True)
    return value, {
        "first_call_s": float(first_s),
        "steady": _timing_stats(steady_times),
    }


def run_stage_profile(args: argparse.Namespace) -> dict[str, Any]:
    dataset_args = _make_dataset_args(args)
    jax, jnp, numpyro, _, _ = posterior_speed._configure_runtime(dataset_args)
    data = posterior_speed._build_dataset(dataset_args, sdsge, jax)
    model = data["model"]
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    observables = tuple(data["observables"])
    observable_names, observation_data = model._coerce_observations(
        data["observations"],
        observables=observables,
    )
    observations = np.asarray(observation_data, dtype=np_dtype)
    steady_state = np.asarray(data["steady_state"], dtype=np_dtype)
    parameter_values = np.asarray(model.parameter_values, dtype=np_dtype)
    observables = observable_names
    observable_indices = model.resolve_observable_indices(observables)
    observable_index_array = jnp.asarray(observable_indices, dtype=jnp.int32)
    theta = jnp.asarray(parameter_values)
    steady = jnp.asarray(steady_state)
    obs = jnp.asarray(observations)
    shocks0 = jnp.zeros((model.timings.nExo,), dtype=theta.dtype)
    demeaned_observations = obs - steady[observable_index_array, None]
    static_equation_rows = (
        model._first_order_static_equation_rows
        if (args.qme_algorithm == "schur_gpu" and not model.has_obc)
        else None
    )

    def resolve_parameters(current_theta):
        return model.resolve_parameter_values_jax(
            parameter_values=current_theta,
            steady_state=steady,
        )

    def dynamic_jacobian(resolved_parameters):
        steady_reference_values = model._steady_reference_values_jax(steady)
        if model.has_obc:
            future_index_array = jnp.asarray(
                model.timings.future_not_past_and_mixed_idx,
                dtype=jnp.int32,
            )
            past_index_array = jnp.asarray(
                model.timings.past_not_future_and_mixed_idx,
                dtype=jnp.int32,
            )
            dynamic_point = jnp.concatenate(
                [
                    steady[future_index_array],
                    steady,
                    steady[past_index_array],
                    shocks0,
                ]
            )

            def residual_from_dynamic_vector(dynamic_vector):
                lead_state = steady.at[future_index_array].set(
                    dynamic_vector[: model.timings.nFuture_not_past_and_mixed]
                )
                current_start = model.timings.nFuture_not_past_and_mixed
                current_end = current_start + model.timings.nVars
                current_state = dynamic_vector[current_start:current_end]
                lag_state = steady.at[past_index_array].set(
                    dynamic_vector[
                        current_end : current_end + model.timings.nPast_not_future_and_mixed
                    ]
                )
                shock = dynamic_vector[
                    current_end + model.timings.nPast_not_future_and_mixed :
                ]
                return model._evaluate_dynamic_residual_with_context(
                    lag_state,
                    current_state,
                    lead_state,
                    shock,
                    parameter_values=resolved_parameters,
                    steady_reference_values=steady_reference_values,
                )

            return jax.jacrev(residual_from_dynamic_vector)(dynamic_point)
        return model._evaluate_dynamic_jacobian_with_context_jax(
            steady,
            steady,
            steady,
            shocks0,
            parameter_values=resolved_parameters,
            steady_reference_values=steady_reference_values,
        )

    def first_order_solution(jacobian):
        return sdsge.solve_first_order_dsge_solution_jax(
            jacobian,
            model.timings,
            qme_algorithm=args.qme_algorithm,
            static_equation_rows=static_equation_rows,
        )

    def state_space_from_solution(solution_matrix):
        return _linear_state_space_from_first_order_solution_jax(
            solution_matrix,
            model,
            observable_indices,
            measurement_error_scale=float(data["measurement_error_scale"]),
        )

    def kalman_loglikelihood_from_state_space(state_space):
        return sdsge.kalman_loglikelihood(
            state_space,
            demeaned_observations,
            jitter=float(data["jitter"]),
        )

    def full_loglikelihood_fn(current_theta):
        return sdsge.kalman_loglikelihood_from_model_jax(
            model,
            obs,
            observables=observables,
            parameter_values=current_theta,
            steady_state=steady,
            measurement_error_scale=float(data["measurement_error_scale"]),
            jitter=float(data["jitter"]),
            qme_algorithm=args.qme_algorithm,
            on_failure_loglikelihood=float(args.failure_value),
        )

    def full_loglikelihood_gradient(current_theta):
        return jax.grad(full_loglikelihood_fn)(current_theta)

    trace_context = (
        jax.profiler.trace(str(args.trace_dir), create_perfetto_link=False)
        if args.trace_dir is not None
        else nullcontext()
    )
    with trace_context:
        resolved, resolve_timing = _time_stage(
            jax=jax,
            name="resolve_parameters",
            fn=resolve_parameters,
            args=(theta,),
            reps=args.reps,
            verbose=args.verbose,
        )
        jacobian, jacobian_timing = _time_stage(
            jax=jax,
            name="dynamic_jacobian",
            fn=dynamic_jacobian,
            args=(resolved,),
            reps=args.reps,
            verbose=args.verbose,
        )
        first_order, first_order_timing = _time_stage(
            jax=jax,
            name="first_order_solution",
            fn=first_order_solution,
            args=(jacobian,),
            reps=args.reps,
            verbose=args.verbose,
        )
        state_space, state_space_timing = _time_stage(
            jax=jax,
            name="state_space_and_initial_covariance",
            fn=state_space_from_solution,
            args=(first_order.solution_matrix,),
            reps=args.reps,
            verbose=args.verbose,
        )
        loglikelihood, kalman_timing = _time_stage(
            jax=jax,
            name="kalman_loglikelihood_only",
            fn=kalman_loglikelihood_from_state_space,
            args=(state_space,),
            reps=args.reps,
            verbose=args.verbose,
        )
        full_loglikelihood_value, full_timing = _time_stage(
            jax=jax,
            name="full_loglikelihood",
            fn=full_loglikelihood_fn,
            args=(theta,),
            reps=args.reps,
            verbose=args.verbose,
        )
        gradient_value = None
        gradient_timing = None
        if args.include_gradient:
            gradient_value, gradient_timing = _time_stage(
                jax=jax,
                name="full_loglikelihood_gradient",
                fn=full_loglikelihood_gradient,
                args=(theta,),
                reps=args.gradient_reps,
                verbose=args.verbose,
            )

    result = {
        "benchmark": {
            "preset": args.preset,
            "case": args.case if args.preset == "sw07_hlt" else None,
            "periods": int(observations.shape[1]),
            "n_observables": int(observations.shape[0]),
            "n_vars": int(model.timings.nVars),
            "n_exo": int(model.timings.nExo),
            "qme_algorithm": args.qme_algorithm,
            "dtype": args.dtype,
            "static_equation_rows": (
                list(static_equation_rows) if static_equation_rows is not None else None
            ),
        },
        "runtime": posterior_speed._runtime_info(jax, numpyro),
        "values": {
            "first_order_converged": bool(np.asarray(first_order.converged)),
            "qme_solution_max_abs": float(
                np.max(np.abs(np.asarray(first_order.qme_solution)))
            )
            if first_order.qme_solution.size
            else 0.0,
            "loglikelihood": float(np.asarray(loglikelihood, dtype=np.float64)),
            "full_loglikelihood": float(
                np.asarray(full_loglikelihood_value, dtype=np.float64)
            ),
        },
        "timings": {
            "resolve_parameters": resolve_timing,
            "dynamic_jacobian": jacobian_timing,
            "first_order_solution": first_order_timing,
            "state_space_and_initial_covariance": state_space_timing,
            "kalman_loglikelihood_only": kalman_timing,
            "full_loglikelihood": full_timing,
        },
    }
    if gradient_timing is not None:
        result["values"]["full_loglikelihood_gradient_max_abs"] = float(
            np.max(np.abs(np.asarray(gradient_value)))
        )
        result["timings"]["full_loglikelihood_gradient"] = gradient_timing
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile staged JAX components of the DSGE Kalman likelihood."
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
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--qme-algorithm",
        choices=("schur", "schur_gpu", "doubling"),
        default="schur_gpu",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--include-gradient",
        action="store_true",
        help=(
            "Also profile reverse-mode gradient of the full likelihood. "
            "This can compile for minutes on medium DSGE models."
        ),
    )
    parser.add_argument("--gradient-reps", type=int, default=0)
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_stage_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    print(json.dumps(result["timings"], indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
