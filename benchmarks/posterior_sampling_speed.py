from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_PATH = ROOT / "benchmarks" / "results" / "test_payloads.json"
DEFAULT_SW07_MODEL_SOURCE_PATH = ROOT / "benchmarks" / "model_sources" / "Smets_Wouters_2007_HLT.jl"
DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "results" / "posterior_sampling_speed.json"

TOY_AR2_SOURCE = """
@model posterior_speed_toy begin
    a[0] = rho_a * a[-1] + (1 - rho_a) * a_bar + eps_a[x]
    y[0] = rho_y * y[-1] + (1 - rho_y) * y_bar + alpha * (a[0] - a_bar) + eps_y[x]
end

@parameters posterior_speed_toy begin
    0 < rho_a < 1
    0 < rho_y < 1
    alpha = 0.4
    a_bar = 1.5
    y_bar = 2.0
    rho_a = 0.8
    rho_y = 0.6
end
"""

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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _run_text(cmd: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(cmd),
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


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


def _log(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "verbose", False):
        elapsed = time.perf_counter() - getattr(args, "_benchmark_started_at", time.perf_counter())
        print(f"[posterior_sampling_speed +{elapsed:9.2f}s] {message}", flush=True)


@contextmanager
def _logged_stage(args: argparse.Namespace, name: str):
    start = time.perf_counter()
    _log(args, f"START {name}")
    stop = threading.Event()
    heartbeat_seconds = float(getattr(args, "heartbeat_seconds", 0.0) or 0.0)
    thread: threading.Thread | None = None
    if getattr(args, "verbose", False) and heartbeat_seconds > 0.0:

        def heartbeat() -> None:
            while not stop.wait(heartbeat_seconds):
                elapsed = time.perf_counter() - start
                _log(args, f"HEARTBEAT {name} still running after {elapsed:.1f}s")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        _log(args, f"FAIL {name} after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = time.perf_counter() - start
        _log(args, f"END {name} after {elapsed:.2f}s")
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=0.1)


def _configure_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = args.platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    import surrogatenn_dsge as sdsge

    # The package currently enables x64 at import time for parity. Reset it here
    # so FP32 GPU runs are measured as true FP32 runs.
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    if args.dtype == "float32" and args.suppress_dtype_warnings:
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype .*float64.*",
            category=UserWarning,
        )
    if args.host_device_count is not None:
        numpyro.set_host_device_count(int(args.host_device_count))
    if args.force_gpu and not any(device.platform in {"gpu", "cuda"} for device in jax.devices()):
        raise RuntimeError(
            "--force-gpu was set, but JAX did not report a CUDA/GPU device. "
            f"Detected devices: {jax.devices()}"
        )
    return jax, jnp, numpyro, dist, sdsge


def _runtime_info(jax: Any, numpyro: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpyro_version": getattr(numpyro, "__version__", "unknown"),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "nvidia_smi": _run_text(["nvidia-smi"]).strip(),
    }


def _load_payload_case(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for case in payload["cases"]:
        if case["name"] == name:
            return case
    raise ValueError(f"Case {name!r} not found in {path}.")


def _select_parameter_names(model: Any, spec: str) -> tuple[str, ...]:
    spec = spec.strip()
    if spec == "sw07_safe_15":
        names = SW07_SAFE_15_PARAMETERS
    elif spec == "sw07_safe_27":
        names = SW07_SAFE_27_PARAMETERS
    elif spec == "payload":
        raise ValueError("The 'payload' parameter selector must be expanded by the caller.")
    elif spec == "all":
        names = tuple(model.parameter_names)
    else:
        names = tuple(name.strip() for name in spec.split(",") if name.strip())
    unknown = tuple(name for name in names if name not in model.parameter_names)
    if unknown:
        raise ValueError("Unknown parameter names: " + ", ".join(unknown))
    return names


def _prior_interval(name: str, center: float, scale: float, floor: float) -> tuple[float, float]:
    width = max(abs(center) * scale, floor)
    lower = center - width
    upper = center + width
    bounded_unit_prefixes = ("crho", "cprob", "cind")
    if name.startswith(bounded_unit_prefixes):
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if name in {"calfa"}:
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if center > 0.0 and lower <= 0.0 and name not in {"cry"}:
        lower = max(center * 0.5, np.finfo(float).tiny)
    if not lower < center < upper:
        raise ValueError(
            f"Invalid prior interval for {name}: center={center}, lower={lower}, upper={upper}."
        )
    return float(lower), float(upper)


def _make_centered_uniform_priors(
    dist: Any,
    model: Any,
    parameter_names: Sequence[str],
    *,
    width_scale: float,
    width_floor: float,
) -> tuple[dict[str, Any], dict[str, float], dict[str, tuple[float, float]]]:
    index = {name: idx for idx, name in enumerate(model.parameter_names)}
    priors: dict[str, Any] = {}
    initial_values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    values = np.asarray(model.parameter_values, dtype=np.float64)
    for name in parameter_names:
        center = float(values[index[name]])
        lower, upper = _prior_interval(name, center, width_scale, width_floor)
        priors[name] = dist.Uniform(lower, upper)
        initial_values[name] = center
        intervals[name] = (lower, upper)
    return priors, initial_values, intervals


def _toy_payload(sdsge: Any, jax: Any, periods: int, seed: int) -> dict[str, Any]:
    model = sdsge.parse_macro_model(TOY_AR2_SOURCE)
    first_order = sdsge.solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    observables = ("y", "a")
    state_space = sdsge.build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order,
        measurement_error_scale=1.0e-8,
    )
    simulation = sdsge.simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(seed),
        num_periods=periods,
    )
    steady_lookup = dict(zip(model.timings.var, np.asarray(first_order.steady_state)))
    observations = np.asarray(simulation.observations, dtype=np.float64) + np.asarray(
        [[steady_lookup[name]] for name in observables],
        dtype=np.float64,
    )
    return {
        "model": model,
        "steady_state": np.asarray(first_order.steady_state, dtype=np.float64),
        "observations": observations,
        "observables": observables,
        "parameter_names": ("rho_a", "rho_y"),
        "measurement_error_scale": 1.0e-8,
        "jitter": 1.0e-8,
    }


def _sw07_payload(args: argparse.Namespace, sdsge: Any) -> dict[str, Any]:
    case = _load_payload_case(args.payload_path, args.case)
    model_source = args.model_source.read_text()
    model = sdsge.parse_macro_model(model_source)
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    observables = tuple(case["observables"])
    periods = int(args.periods)
    if periods <= len(case["observations"][0]):
        observations = np.asarray(case["observations"], dtype=np.float64)[:, :periods]
    else:
        first_order = sdsge.solve_first_order_model(
            model,
            steady_state=steady_state,
            qme_algorithm="schur",
        )
        if not first_order.solution.converged:
            raise RuntimeError("Could not generate synthetic SW07 observations: Schur solve failed.")
        shock_names = tuple(case["shock_names"])
        shock_sigmas = np.asarray(
            [float(case["shock_sigmas"][name]) for name in shock_names],
            dtype=np.float64,
        )
        rng = np.random.default_rng(int(args.synthetic_seed))
        shock_matrix = shock_sigmas[:, None] * rng.standard_normal((len(shock_names), periods))
        full_levels = (
            np.asarray(
                sdsge.rollout_first_order_solution(
                    first_order.solution.solution_matrix,
                    model.timings,
                    shock_matrix,
                ),
                dtype=np.float64,
            )
            + steady_state[:, None]
        )
        observation_indices = [model.timings.var.index(name) for name in observables]
        observations = full_levels[observation_indices, :]
    parameter_spec = args.parameters
    parameter_names = (
        tuple(case["parameter_subset"])
        if parameter_spec == "payload"
        else _select_parameter_names(model, parameter_spec)
    )
    return {
        "model": model,
        "steady_state": steady_state,
        "observations": observations,
        "observables": observables,
        "parameter_names": parameter_names,
        "measurement_error_scale": float(case.get("measurement_error_scale", 1.0e-8)),
        "jitter": float(case.get("jitter", 1.0e-9)),
    }


def _build_dataset(args: argparse.Namespace, sdsge: Any, jax: Any) -> dict[str, Any]:
    if args.preset == "toy_ar2":
        return _toy_payload(sdsge, jax, int(args.periods), int(args.synthetic_seed))
    if args.preset == "sw07_hlt":
        return _sw07_payload(args, sdsge)
    raise ValueError(f"Unsupported preset {args.preset!r}.")


def _sample_summary(
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    selected_names = (
        tuple(parameter_names)
        if parameter_names is not None
        else tuple(samples_by_chain)
    )
    np_samples = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in samples_by_chain.items()
        if name in selected_names
    }
    parameter_metrics: dict[str, dict[str, float]] = {}
    ess_values: list[float] = []
    rhat_values: list[float] = []
    for name, values in np_samples.items():
        flat = values.reshape(-1)
        try:
            ess = float(np.asarray(effective_sample_size(values), dtype=np.float64))
        except Exception:
            ess = float("nan")
        try:
            rhat = float(np.asarray(split_gelman_rubin(values), dtype=np.float64))
        except Exception:
            rhat = float("nan")
        parameter_metrics[name] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "n_eff": ess,
            "r_hat": rhat,
            "shape": list(values.shape),
        }
        if np.isfinite(ess) and ess > 0.0:
            ess_values.append(ess)
        if np.isfinite(rhat):
            rhat_values.append(rhat)
    return {
        "parameters": parameter_metrics,
        "min_ess": float(min(ess_values)) if ess_values else None,
        "mean_ess": float(statistics.mean(ess_values)) if ess_values else None,
        "max_r_hat": float(max(rhat_values)) if rhat_values else None,
    }


def _extra_field_summary(extra_fields: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("accept_prob", "diverging", "num_steps"):
        if name not in extra_fields:
            result[name] = None
            continue
        values = np.asarray(extra_fields[name])
        if values.size == 0:
            result[name] = None
            continue
        if name == "diverging":
            result[name] = {
                "count": int(np.asarray(values, dtype=bool).sum()),
                "share": float(np.asarray(values, dtype=bool).mean()),
            }
        else:
            finite = np.asarray(values, dtype=np.float64)
            result[name] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
    return result


def _flatten_posterior_draws(
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str],
    *,
    max_draws: int,
) -> list[dict[str, float]]:
    if max_draws <= 0:
        return []
    arrays = {
        name: np.asarray(samples_by_chain[name], dtype=np.float64).reshape(-1)
        for name in parameter_names
    }
    n_draws = min(max_draws, *(array.shape[0] for array in arrays.values()))
    return [
        {name: float(arrays[name][draw_idx]) for name in parameter_names}
        for draw_idx in range(n_draws)
    ]


def _support_audit(
    *,
    sdsge: Any,
    model: Any,
    observations: np.ndarray,
    observables: Sequence[str],
    steady_state: np.ndarray,
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str],
    base_parameter_values: np.ndarray,
    measurement_error_scale: float,
    jitter: float,
    max_draws: int,
    schur_acceptance_tol: float,
    failure_value: float,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    draws = _flatten_posterior_draws(
        samples_by_chain,
        parameter_names,
        max_draws=max_draws,
    )
    classifications: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    doubling_accepts_non_unique = 0
    both_accept = 0
    schur_rejects = 0
    doubling_rejects = 0
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for draw_idx, draw in enumerate(draws):
        if log_fn is not None:
            log_fn(f"support audit draw {draw_idx + 1}/{len(draws)}")
        theta = np.asarray(base_parameter_values, dtype=np.float64).copy()
        for name, value in draw.items():
            theta[parameter_index[name]] = value
        try:
            schur = sdsge.analyze_first_order_model_determinacy(
                model,
                parameter_values=theta,
                steady_state=steady_state,
                qme_acceptance_tol=schur_acceptance_tol,
            )
            classification = str(schur.determinacy.classification)
            schur_unique = bool(schur.determinacy.unique_stable_solution)
        except Exception as exc:
            classification = f"schur_error:{type(exc).__name__}"
            schur_unique = False
        classifications[classification] = classifications.get(classification, 0) + 1
        try:
            doubling_ll = sdsge.kalman_loglikelihood_from_model(
                model,
                observations,
                observables=observables,
                parameter_values=theta,
                steady_state=steady_state,
                measurement_error_scale=measurement_error_scale,
                jitter=jitter,
                qme_algorithm="doubling",
                on_failure_loglikelihood=failure_value,
            )
            doubling_value = float(np.asarray(doubling_ll, dtype=np.float64))
            doubling_accept = np.isfinite(doubling_value) and doubling_value > failure_value / 10.0
        except Exception as exc:
            doubling_value = float("nan")
            doubling_accept = False
            if len(examples) < 5:
                examples.append(
                    {
                        "draw_index": draw_idx,
                        "classification": classification,
                        "doubling_error": f"{type(exc).__name__}: {exc}",
                        "parameters": draw,
                    }
                )
        if schur_unique and doubling_accept:
            both_accept += 1
        if not schur_unique:
            schur_rejects += 1
        if not doubling_accept:
            doubling_rejects += 1
        if doubling_accept and not schur_unique:
            doubling_accepts_non_unique += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "draw_index": draw_idx,
                        "classification": classification,
                        "doubling_loglikelihood": doubling_value,
                        "parameters": draw,
                    }
                )
        if log_fn is not None:
            log_fn(
                "support audit draw "
                f"{draw_idx + 1}/{len(draws)} classified {classification}; "
                f"schur_unique={schur_unique}; doubling_accept={doubling_accept}"
            )
    audited = len(draws)
    return {
        "audited_draws": audited,
        "schur_classification_counts": classifications,
        "both_accept_count": both_accept,
        "schur_reject_count": schur_rejects,
        "doubling_reject_count": doubling_rejects,
        "doubling_accepts_non_unique_count": doubling_accepts_non_unique,
        "doubling_accepts_non_unique_share": (
            float(doubling_accepts_non_unique / audited) if audited else None
        ),
        "examples": examples,
    }


def _preflight_metrics(
    *,
    jax: Any,
    jnp: Any,
    sdsge: Any,
    model: Any,
    observations: np.ndarray,
    observables: Sequence[str],
    steady_state: np.ndarray,
    parameter_values: np.ndarray,
    parameter_names: Sequence[str],
    measurement_error_scale: float,
    jitter: float,
    qme_algorithm: str,
    reps: int,
    failure_value: float,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    theta = jnp.asarray(parameter_values)
    obs = jnp.asarray(observations)
    steady = jnp.asarray(steady_state)
    value_fn = jax.jit(
        lambda current_theta: sdsge.kalman_loglikelihood_from_model_jax(
            model,
            obs,
            observables=observables,
            parameter_values=current_theta,
            steady_state=steady,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            qme_algorithm=qme_algorithm,
            on_failure_loglikelihood=failure_value,
        )
    )
    index = jnp.asarray(
        [model.parameter_names.index(name) for name in parameter_names],
        dtype=jnp.int32,
    )
    x0 = theta[index]

    def inject(x):
        return theta.at[index].set(x)

    grad_fn = jax.jit(jax.value_and_grad(lambda x: value_fn(inject(x))))
    if log_fn is not None:
        log_fn("preflight value first call/JIT start")
    value, first_value_s = _timed_call(lambda: value_fn(theta))
    if log_fn is not None:
        log_fn(f"preflight value first call/JIT done in {first_value_s:.3f}s")
    value_times = []
    for rep in range(max(reps, 0)):
        if log_fn is not None:
            log_fn(f"preflight value steady rep {rep + 1}/{reps} start")
        _, elapsed = _timed_call(lambda: value_fn(theta))
        value_times.append(elapsed)
        if log_fn is not None:
            log_fn(f"preflight value steady rep {rep + 1}/{reps} done in {elapsed:.3f}s")
    if log_fn is not None:
        log_fn("preflight gradient first call/JIT start")
    grad_value, first_grad_s = _timed_call(lambda: grad_fn(x0))
    if log_fn is not None:
        log_fn(f"preflight gradient first call/JIT done in {first_grad_s:.3f}s")
    grad_times = []
    for rep in range(max(reps, 0)):
        if log_fn is not None:
            log_fn(f"preflight gradient steady rep {rep + 1}/{reps} start")
        _, elapsed = _timed_call(lambda: grad_fn(x0))
        grad_times.append(elapsed)
        if log_fn is not None:
            log_fn(f"preflight gradient steady rep {rep + 1}/{reps} done in {elapsed:.3f}s")
    return {
        "loglikelihood": float(np.asarray(value, dtype=np.float64)),
        "loglikelihood_dtype": str(getattr(value, "dtype", "")),
        "value_first_call_s": float(first_value_s),
        "value_steady": _timing_stats(value_times),
        "gradient_value": float(np.asarray(grad_value[0], dtype=np.float64)),
        "gradient": np.asarray(grad_value[1], dtype=np.float64).tolist(),
        "gradient_dtype": str(getattr(grad_value[1], "dtype", "")),
        "gradient_first_call_s": float(first_grad_s),
        "gradient_steady": _timing_stats(grad_times),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args._benchmark_started_at = time.perf_counter()
    _log(args, "benchmark process started")
    with _logged_stage(args, "configure JAX/NumPyro runtime"):
        jax, jnp, numpyro, dist, sdsge = _configure_runtime(args)
        _log(
            args,
            "runtime configured: "
            f"jax={getattr(jax, '__version__', 'unknown')}, "
            f"numpyro={getattr(numpyro, '__version__', 'unknown')}, "
            f"backend={jax.default_backend()}, devices={jax.devices()}",
        )
    if args.dtype == "float32":
        np_dtype = np.float32
    else:
        np_dtype = np.float64
    with _logged_stage(args, "build benchmark dataset"):
        data = _build_dataset(args, sdsge, jax)
    model = data["model"]
    steady_state = np.asarray(data["steady_state"], dtype=np_dtype)
    observations = np.asarray(data["observations"], dtype=np_dtype)
    observables = tuple(data["observables"])
    parameter_names = tuple(data["parameter_names"])
    parameter_values = np.asarray(model.parameter_values, dtype=np_dtype)
    _log(
        args,
        "dataset ready: "
        f"preset={args.preset}, vars={model.timings.nVars}, exo={model.timings.nExo}, "
        f"observations={observations.shape}, parameters={parameter_names}",
    )
    with _logged_stage(args, "build priors"):
        priors, initial_values, prior_intervals = _make_centered_uniform_priors(
            dist,
            model,
            parameter_names,
            width_scale=float(args.prior_width_scale),
            width_floor=float(args.prior_width_floor),
        )
    from numpyro.infer import MCMC, NUTS, init_to_value

    with _logged_stage(args, "build NumPyro model and NUTS kernel"):
        numpyro_model = sdsge.build_numpyro_kalman_model_jax(
            model,
            observations,
            priors,
            observables=observables,
            base_parameter_values=parameter_values,
            steady_state=steady_state,
            measurement_error_scale=float(data["measurement_error_scale"]),
            jitter=float(data["jitter"]),
            on_failure_loglikelihood=float(args.failure_value),
            qme_algorithm=args.qme_algorithm,
        )
        kernel = NUTS(
            numpyro_model,
            dense_mass=bool(args.dense_mass),
            target_accept_prob=float(args.target_accept_prob),
            max_tree_depth=int(args.max_tree_depth),
            init_strategy=init_to_value(values=initial_values),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(args.warmup),
            num_samples=int(args.samples),
            num_chains=int(args.chains),
            chain_method=args.chain_method,
            progress_bar=bool(args.progress_bar),
        )

    benchmark_info = {
        "preset": args.preset,
        "case": args.case if args.preset == "sw07_hlt" else None,
        "periods": int(observations.shape[1]),
        "n_observables": int(observations.shape[0]),
        "n_vars": int(model.timings.nVars),
        "n_exo": int(model.timings.nExo),
        "parameter_count": len(parameter_names),
        "parameter_names": list(parameter_names),
        "qme_algorithm": args.qme_algorithm,
        "dtype": args.dtype,
        "warmup": int(args.warmup),
        "samples": int(args.samples),
        "chains": int(args.chains),
        "chain_method": args.chain_method,
        "dense_mass": bool(args.dense_mass),
        "target_accept_prob": float(args.target_accept_prob),
        "max_tree_depth": int(args.max_tree_depth),
        "prior_intervals": prior_intervals,
    }

    preflight = None
    if args.preflight:
        with _logged_stage(args, "preflight likelihood and gradient"):
            preflight = _preflight_metrics(
                jax=jax,
                jnp=jnp,
                sdsge=sdsge,
                model=model,
                observations=observations,
                observables=observables,
                steady_state=steady_state,
                parameter_values=parameter_values,
                parameter_names=parameter_names,
                measurement_error_scale=float(data["measurement_error_scale"]),
                jitter=float(data["jitter"]),
                qme_algorithm=args.qme_algorithm,
                reps=int(args.preflight_reps),
                failure_value=float(args.failure_value),
                log_fn=lambda message: _log(args, message),
            )
            _log(
                args,
                "preflight summary: "
                f"loglikelihood={preflight['loglikelihood']}, "
                f"value_steady_median={preflight['value_steady'].get('median_s')}, "
                f"gradient_steady_median={preflight['gradient_steady'].get('median_s')}",
            )

    if args.preflight_only:
        _log(args, "preflight-only requested; skipping MCMC")
        return {
            "benchmark": benchmark_info,
            "runtime": _runtime_info(jax, numpyro),
            "preflight": preflight,
            "throughput": None,
            "posterior_diagnostics": None,
            "extra_fields": None,
            "schur_support_audit": None,
        }

    _log(
        args,
        "starting MCMC: "
        f"warmup={args.warmup}, samples={args.samples}, chains={args.chains}, "
        f"chain_method={args.chain_method}, dense_mass={args.dense_mass}",
    )
    start = time.perf_counter()
    with _logged_stage(args, "NumPyro MCMC run"):
        mcmc.run(
            jax.random.PRNGKey(int(args.seed)),
            extra_fields=("accept_prob", "diverging", "num_steps"),
        )
    with _logged_stage(args, "block and collect samples"):
        _block_tree(mcmc.get_samples(group_by_chain=True))
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    sampling_wall_s = time.perf_counter() - start
    with _logged_stage(args, "posterior diagnostics"):
        sample_summary = _sample_summary(samples_by_chain, parameter_names)
    post_warmup_draws = int(args.samples) * int(args.chains)
    min_ess = sample_summary["min_ess"]
    mean_ess = sample_summary["mean_ess"]
    throughput = {
        "sampling_wall_s": float(sampling_wall_s),
        "post_warmup_draws": post_warmup_draws,
        "draws_per_second": float(post_warmup_draws / sampling_wall_s),
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "seconds_per_min_ess": (
            float(sampling_wall_s / min_ess) if isinstance(min_ess, float) and min_ess > 0 else None
        ),
        "seconds_per_mean_ess": (
            float(sampling_wall_s / mean_ess) if isinstance(mean_ess, float) and mean_ess > 0 else None
        ),
        "min_ess_per_second": (
            float(min_ess / sampling_wall_s) if isinstance(min_ess, float) and sampling_wall_s > 0 else None
        ),
        "mean_ess_per_second": (
            float(mean_ess / sampling_wall_s) if isinstance(mean_ess, float) and sampling_wall_s > 0 else None
        ),
    }

    support_audit = None
    if args.schur_support_draws > 0:
        with _logged_stage(args, "Schur support audit"):
            support_audit = _support_audit(
                sdsge=sdsge,
                model=model,
                observations=np.asarray(observations, dtype=np.float64),
                observables=observables,
                steady_state=np.asarray(steady_state, dtype=np.float64),
                samples_by_chain=samples_by_chain,
                parameter_names=parameter_names,
                base_parameter_values=np.asarray(model.parameter_values, dtype=np.float64),
                measurement_error_scale=float(data["measurement_error_scale"]),
                jitter=float(data["jitter"]),
                max_draws=int(args.schur_support_draws),
                schur_acceptance_tol=float(args.schur_acceptance_tol),
                failure_value=float(args.failure_value),
                log_fn=lambda message: _log(args, message),
            )
            _log(
                args,
                "support audit summary: "
                f"audited={support_audit['audited_draws']}, "
                f"doubling_accepts_non_unique="
                f"{support_audit['doubling_accepts_non_unique_count']}",
            )

    _log(args, f"throughput summary: {json.dumps(throughput, sort_keys=True, default=_json_default)}")

    return {
        "benchmark": benchmark_info,
        "runtime": _runtime_info(jax, numpyro),
        "preflight": preflight,
        "throughput": throughput,
        "posterior_diagnostics": sample_summary,
        "extra_fields": _extra_field_summary(extra_fields),
        "schur_support_audit": support_audit,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure posterior sampling speed as seconds per effective sample "
            "for NumPyro/JAX DSGE estimation benchmarks."
        )
    )
    parser.add_argument("--preset", choices=("toy_ar2", "sw07_hlt"), default="sw07_hlt")
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--model-source", type=Path, default=DEFAULT_SW07_MODEL_SOURCE_PATH)
    parser.add_argument("--parameters", default="sw07_safe_15")
    parser.add_argument("--periods", type=int, default=80)
    parser.add_argument("--synthetic-seed", type=int, default=20260712)
    parser.add_argument("--prior-width-scale", type=float, default=0.0025)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "vectorized", "sequential"),
        default="vectorized",
    )
    parser.add_argument("--host-device-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--qme-algorithm", choices=("schur", "doubling"), default="doubling")
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Compile and time the likelihood/gradient, then exit before MCMC.",
    )
    parser.add_argument("--preflight-reps", type=int, default=0)
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--schur-support-draws", type=int, default=0)
    parser.add_argument("--schur-acceptance-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage timing and heartbeat logs while compiling/sampling.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval for verbose long-running stages; set 0 to disable.",
    )
    parser.add_argument(
        "--show-dtype-warnings",
        action="store_false",
        dest="suppress_dtype_warnings",
        help="Show JAX dtype truncation warnings during true-FP32 runs.",
    )
    parser.set_defaults(suppress_dtype_warnings=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.preflight_only:
        args.preflight = True
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    print(json.dumps(result["throughput"], indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
