from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from surrogatenn_dsge import (
    RegimeSwitchConfig,
    SEPConfig,
    SwitchingLikelihoodConfig,
    build_numpyro_kalman_model_jax,
    compute_linear_gate_stats_from_filter,
    evaluate_numpyro_kalman_log_density_jax,
    evaluate_numpyro_switching_log_density_jax,
    estimate_observed_shocks_matrix,
    estimate_observed_variables_matrix,
    kalman_loglikelihood_from_model_jax,
    kalman_loglikelihood_per_period_from_model,
    inversion_loglikelihood_from_model,
    parse_macro_model,
    solve_first_order_model,
    switching_loglikelihood_from_model_jax,
    switching_loglikelihood_from_model_filter_gates_jax,
)


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, tuple):
        for item in value:
            _block(item)
        return value
    if isinstance(value, list):
        for item in value:
            _block(item)
        return value
    return value


def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start_ns = time.perf_counter_ns()
    value = fn()
    _block(value)
    elapsed_s = (time.perf_counter_ns() - start_ns) * 1e-9
    return value, elapsed_s


def _steady_stats(times: list[float]) -> dict[str, Any]:
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


def _measure_stage(
    fn: Callable[[], Any],
    *,
    steady_reps: int,
    serializer: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    try:
        first_value, first_s = _timed_call(fn)
        steady_times: list[float] = []
        last_value = first_value
        for _ in range(steady_reps):
            last_value, elapsed_s = _timed_call(fn)
            steady_times.append(elapsed_s)
        return {
            "status": "ok",
            "first_call_s": float(first_s),
            "steady": _steady_stats(steady_times),
            "result": serializer(last_value),
        }
    except Exception as exc:  # pragma: no cover - benchmark harness
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _scalar_result(value: Any) -> dict[str, Any]:
    if hasattr(value, "solution") and hasattr(value.solution, "converged"):
        return {
            "converged": bool(value.solution.converged),
        }
    if isinstance(value, tuple):
        primary = value[0]
        grad = value[1]
        return {
            "value": float(np.asarray(primary, dtype=np.float64)),
            "grad_l2": float(np.linalg.norm(np.asarray(grad, dtype=np.float64))),
        }
    return {
        "value": float(np.asarray(value, dtype=np.float64)),
    }


def _first_order_result(value: Any) -> dict[str, Any]:
    solution_matrix = np.asarray(value.solution.solution_matrix, dtype=np.float64)
    steady_state = np.asarray(value.steady_state, dtype=np.float64)
    stacked = np.column_stack([steady_state, solution_matrix]).T
    return {
        "converged": bool(value.solution.converged),
        "solution_matrix": stacked.tolist(),
        "shape": list(stacked.shape),
    }


def _gradient_result(value: Any) -> dict[str, Any]:
    primary, grad = value
    grad_array = np.asarray(grad, dtype=np.float64)
    return {
        "value": float(np.asarray(primary, dtype=np.float64)),
        "grad": grad_array.tolist(),
        "grad_l2": float(np.linalg.norm(grad_array)),
    }


def _paths_result(value: Any) -> dict[str, Any]:
    return {
        key: np.asarray(array, dtype=np.float64).tolist()
        for key, array in value.items()
    }


def _gate_stats_result(value: Any) -> dict[str, Any]:
    return {
        "linear_observations": np.asarray(
            value.linear_observations,
            dtype=np.float64,
        ).tolist(),
        "shocks": np.asarray(value.shocks, dtype=np.float64).tolist(),
        "e_stat": np.asarray(value.e_stat, dtype=np.float64).tolist(),
        "f_stat": np.asarray(value.f_stat, dtype=np.float64).tolist(),
    }


def _samples_result(value: Any) -> dict[str, Any]:
    samples = {
        name: np.asarray(array, dtype=np.float64)
        for name, array in value.items()
    }
    return {
        "sample_count": int(next(iter(samples.values())).shape[0]) if samples else 0,
        "parameter_names": sorted(samples),
        "means": {
            name: float(np.mean(array))
            for name, array in samples.items()
        },
        "stds": {
            name: float(np.std(array))
            for name, array in samples.items()
        },
        "shapes": {
            name: list(array.shape)
            for name, array in samples.items()
        },
    }


def _skipped_stage(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
    }


def _subset_indices(all_names: tuple[str, ...], subset_names: list[str]) -> np.ndarray:
    return np.asarray([all_names.index(name) for name in subset_names], dtype=np.int64)


def _inject_subset(base_theta: np.ndarray, subset_idx: np.ndarray, x: Any) -> np.ndarray:
    theta = jnp.asarray(base_theta, dtype=jnp.float64)
    theta = theta.at[jnp.asarray(subset_idx, dtype=jnp.int32)].set(
        jnp.asarray(x, dtype=jnp.float64)
    )
    return theta


def _numpyro_priors_and_samples(
    model_parameter_names: tuple[str, ...],
    parameter_values: np.ndarray,
    parameter_names: list[str],
    *,
    width_scale: float,
    width_floor: float,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    if width_scale <= 0:
        raise ValueError(f"numpyro_prior_width_scale must be positive, got {width_scale}.")
    if width_floor <= 0:
        raise ValueError(f"numpyro_prior_width_floor must be positive, got {width_floor}.")
    priors: dict[str, Any] = {}
    samples: dict[str, jax.Array] = {}
    for name in parameter_names:
        if name not in model_parameter_names:
            raise ValueError(f"Unknown NumPyro benchmark parameter {name!r}.")
        idx = model_parameter_names.index(name)
        center = float(parameter_values[idx])
        width = max(abs(center) * width_scale, width_floor)
        lower = center - width
        upper = center + width
        if center > 0.0 and lower <= 0.0:
            lower = max(center * 0.5, np.finfo(float).tiny)
        if not lower < center < upper:
            raise ValueError(
                f"Invalid NumPyro benchmark prior interval for {name}: "
                f"lower={lower}, center={center}, upper={upper}."
            )
        priors[name] = dist.Uniform(lower, upper)
        samples[name] = jnp.asarray(center, dtype=jnp.float64)
    return priors, samples


def _case_result(case: dict[str, Any]) -> dict[str, Any]:
    model_source = Path(case["model_path"]).read_text()
    load_stage = _measure_stage(
        lambda: parse_macro_model(model_source),
        steady_reps=0,
        serializer=lambda loaded_model: {
            "n_vars": len(loaded_model.timings.var),
            "n_exo": len(loaded_model.timings.exo),
            "parameter_count": len(loaded_model.parameter_names),
        },
    )
    model = parse_macro_model(model_source)
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    observations = np.asarray(case["observations"], dtype=np.float64)
    observation_names = tuple(case["observables"])
    state_names = tuple(case["state_names"])
    shared_gate_probs = np.asarray(case["shared_gate_probs"], dtype=np.float64)
    obs_sigma = {name: float(case["obs_sigma"][name]) for name in observation_names}
    shock_sigmas = {name: float(case["shock_sigmas"][name]) for name in case["shock_names"]}
    parameter_values = np.asarray(model.parameter_values, dtype=np.float64)
    subset_idx = _subset_indices(model.parameter_names, list(case["parameter_subset"]))
    x0 = parameter_values[subset_idx]
    measurement_error_scale = float(case["measurement_error_scale"])
    jitter = float(case["jitter"])
    numpyro_parameter_names = list(
        case.get("numpyro_parameter_subset", case["parameter_subset"])
    )
    numpyro_priors, numpyro_samples = _numpyro_priors_and_samples(
        model.parameter_names,
        parameter_values,
        numpyro_parameter_names,
        width_scale=float(case.get("numpyro_prior_width_scale", 0.01)),
        width_floor=float(case.get("numpyro_prior_width_floor", 1e-4)),
    )

    solve_fn = lambda: solve_first_order_model(
        model,
        steady_state=steady_state,
        qme_algorithm="schur",
    )

    kalman_objective = lambda theta: kalman_loglikelihood_from_model_jax(
        model,
        observations,
        observables=observation_names,
        parameter_values=theta,
        steady_state=steady_state,
        measurement_error_scale=measurement_error_scale,
        jitter=jitter,
        on_failure_loglikelihood=-1.0e12,
        qme_algorithm="schur",
    )
    kalman_objective_grad = lambda theta: kalman_loglikelihood_from_model_jax(
        model,
        observations,
        observables=observation_names,
        parameter_values=theta,
        steady_state=steady_state,
        measurement_error_scale=measurement_error_scale,
        jitter=jitter,
        on_failure_loglikelihood=-1.0e12,
        qme_algorithm="doubling",
    )
    kalman_value_fn = jax.jit(kalman_objective)
    kalman_per_period_fn = lambda: kalman_loglikelihood_per_period_from_model(
        model,
        observations,
        observables=observation_names,
        parameter_values=parameter_values,
        steady_state=steady_state,
        measurement_error_scale=measurement_error_scale,
        jitter=jitter,
        on_failure_loglikelihood=-1.0e12,
        qme_algorithm="schur",
    )
    kalman_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda x: kalman_objective_grad(
                _inject_subset(parameter_values, subset_idx, x)
            )
        )
    )
    kalman_paths_fn = lambda: {
        "filtered_variables": estimate_observed_variables_matrix(
            model,
            observations,
            observables=observation_names,
            parameter_values=parameter_values,
            steady_state=steady_state,
            jitter=jitter,
            smooth=False,
            filter="kalman",
            qme_algorithm="schur",
        )[0],
        "smoothed_variables": estimate_observed_variables_matrix(
            model,
            observations,
            observables=observation_names,
            parameter_values=parameter_values,
            steady_state=steady_state,
            jitter=jitter,
            smooth=True,
            filter="kalman",
            qme_algorithm="schur",
        )[0],
        "filtered_shocks": estimate_observed_shocks_matrix(
            model,
            observations,
            observables=observation_names,
            parameter_values=parameter_values,
            steady_state=steady_state,
            jitter=jitter,
            smooth=False,
            filter="kalman",
            qme_algorithm="schur",
        ),
        "smoothed_shocks": estimate_observed_shocks_matrix(
            model,
            observations,
            observables=observation_names,
            parameter_values=parameter_values,
            steady_state=steady_state,
            jitter=jitter,
            smooth=True,
            filter="kalman",
            qme_algorithm="schur",
        ),
    }

    regime_switch_config = RegimeSwitchConfig(
        gate_mode=str(case["gate_mode"]),
        tau_eps=float(case["regime_switch_config"]["tau_eps"]),
        tau_y=float(case["regime_switch_config"]["tau_y"]),
        beta_eps=float(case["regime_switch_config"]["beta_eps"]),
        beta_y=float(case["regime_switch_config"]["beta_y"]),
        hard_threshold=float(case["gate_hard_threshold"]),
        prob_floor=float(case["gate_prob_floor"]),
        prob_ceiling=float(case["gate_prob_ceiling"]),
        soft_mixture=str(case["soft_mixture"]),
    )
    switching_config = SwitchingLikelihoodConfig(
        gate_mode=str(case["gate_mode"]),
        hard_threshold=float(case["gate_hard_threshold"]),
        prob_floor=float(case["gate_prob_floor"]),
        prob_ceiling=float(case["gate_prob_ceiling"]),
        soft_mixture=str(case["soft_mixture"]),
    )
    gate_stats_fn = lambda: compute_linear_gate_stats_from_filter(
        model,
        observations,
        obs_sigma,
        shock_sigmas,
        state_names,
        observables=observation_names,
        parameter_values=parameter_values,
        steady_state=steady_state,
        qme_algorithm="schur",
        filter="kalman",
        algorithm="first_order",
        smooth=False,
    )
    switching_fixed_fn = jax.jit(
        lambda theta: switching_loglikelihood_from_model_jax(
            model,
            observations,
            gate_probs=shared_gate_probs,
            observables=observation_names,
            parameter_values=theta,
            steady_state=steady_state,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            qme_algorithm="schur",
            on_failure_loglikelihood=-1.0e12,
        )
    )
    switching_value_fn = jax.jit(
        lambda theta: switching_loglikelihood_from_model_filter_gates_jax(
            model,
            observations,
            obs_sigma,
            shock_sigmas,
            regime_switch_config=regime_switch_config,
            switching_config=switching_config,
            observables=observation_names,
            state_names=state_names,
            parameter_values=theta,
            steady_state=steady_state,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            qme_algorithm="schur",
            on_failure_loglikelihood=-1.0e12,
        )
    )
    numpyro_kalman_log_density_fn = jax.jit(
        lambda: evaluate_numpyro_kalman_log_density_jax(
            model,
            observations,
            numpyro_priors,
            numpyro_samples,
            observables=observation_names,
            steady_state=steady_state,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            on_failure_loglikelihood=-1.0e12,
            qme_algorithm="schur",
        )
    )
    numpyro_switching_log_density_fn = jax.jit(
        lambda: evaluate_numpyro_switching_log_density_jax(
            model,
            observations,
            numpyro_priors,
            numpyro_samples,
            gate_probs=shared_gate_probs,
            observables=observation_names,
            base_parameter_values=parameter_values,
            steady_state=steady_state,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            qme_algorithm="schur",
            on_failure_loglikelihood=-1.0e12,
        )
    )

    def numpyro_nuts_fn() -> dict[str, jax.Array]:
        numpyro_model = build_numpyro_kalman_model_jax(
            model,
            observations,
            numpyro_priors,
            observables=observation_names,
            base_parameter_values=parameter_values,
            steady_state=steady_state,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            on_failure_loglikelihood=-1.0e12,
            qme_algorithm="schur",
        )
        kernel = NUTS(
            numpyro_model,
            dense_mass=False,
            target_accept_prob=float(case.get("numpyro_target_accept_prob", 0.8)),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(case.get("numpyro_nuts_warmup", 2)),
            num_samples=int(case.get("numpyro_nuts_samples", 2)),
            num_chains=int(case.get("numpyro_nuts_chains", 1)),
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(int(case.get("numpyro_seed", 0))))
        return mcmc.get_samples()

    sep_config = SEPConfig(
        periods=int(case["sep_periods"]),
        branching_order=int(case["sep_branching_order"]),
        nnodes=int(case["sep_nnodes"]),
        sparse_tree=bool(case["sep_sparse_tree"]),
        max_iter=int(case["sep_maxit"]),
        tol=float(case["sep_tol"]),
    )
    sep_observations = observations[:, : int(case["sep_eval_periods"])]
    sep_fn = lambda: inversion_loglikelihood_from_model(
        model,
        sep_observations,
        observables=observation_names,
        algorithm="stochastic_extended_path",
        parameter_values=parameter_values,
        steady_state=steady_state,
        config=sep_config,
        sep_accept_tol=float(case["sep_accept_tol"]),
        sep_inv_maxit=int(case["sep_inv_maxit"]),
        sep_inv_step_tol=float(case["sep_inv_step_tol"]),
        sep_inv_resid_tol=float(case["sep_inv_resid_tol"]),
        sep_inv_lambda=float(case["sep_inv_lambda"]),
        on_failure_loglikelihood=-1.0e12,
        qme_algorithm="schur",
    )

    return {
        "model_info": {
            "n_vars": len(model.timings.var),
            "n_exo": len(model.timings.exo),
            "parameter_count": len(model.parameter_names),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "stages": {
            "model_load": load_stage,
            "first_order_solve": _measure_stage(
                solve_fn,
                steady_reps=int(case["solve_reps"]),
                serializer=_first_order_result,
            ),
            "kalman_value": _measure_stage(
                lambda: kalman_value_fn(parameter_values),
                steady_reps=int(case["kalman_value_reps"]),
                serializer=_scalar_result,
            ),
            "kalman_per_period": _measure_stage(
                kalman_per_period_fn,
                steady_reps=int(case["kalman_per_period_reps"]),
                serializer=lambda value: {
                    "per_period": np.asarray(value, dtype=np.float64).tolist(),
                },
            ),
            "kalman_paths": _measure_stage(
                kalman_paths_fn,
                steady_reps=int(case["kalman_paths_reps"]),
                serializer=_paths_result,
            ),
            "kalman_grad": _measure_stage(
                lambda: kalman_grad_fn(x0),
                steady_reps=int(case["kalman_grad_reps"]),
                serializer=_gradient_result,
            ),
            "gate_stats": _measure_stage(
                gate_stats_fn,
                steady_reps=int(case["gate_stats_reps"]),
                serializer=_gate_stats_result,
            ),
            "switching_fixed": _measure_stage(
                lambda: switching_fixed_fn(parameter_values),
                steady_reps=int(case["switching_fixed_reps"]),
                serializer=_scalar_result,
            ),
            "switching_value": _measure_stage(
                lambda: switching_value_fn(parameter_values),
                steady_reps=int(case["switching_reps"]),
                serializer=_scalar_result,
            ),
            "numpyro_kalman_log_density": _measure_stage(
                numpyro_kalman_log_density_fn,
                steady_reps=int(case.get("numpyro_log_density_reps", 0)),
                serializer=_scalar_result,
            ),
            "numpyro_switching_log_density": _measure_stage(
                numpyro_switching_log_density_fn,
                steady_reps=int(case.get("numpyro_switching_log_density_reps", 0)),
                serializer=_scalar_result,
            ),
            "numpyro_nuts_smoke": (
                _measure_stage(
                    numpyro_nuts_fn,
                    steady_reps=0,
                    serializer=_samples_result,
                )
                if int(case.get("numpyro_nuts_samples", 0)) > 0
                else _skipped_stage("numpyro_nuts_samples=0")
            ),
            "sep_inversion": _measure_stage(
                sep_fn,
                steady_reps=int(case["sep_reps"]),
                serializer=_scalar_result,
            ),
        },
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: profile_validation_python.py <payload.json> <output.json>"
        )
    payload_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(payload_path.read_text())

    results = {
        "language": "python",
        "python_version": sys.version,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "cases": {},
    }
    for case in payload["cases"]:
        results["cases"][case["name"]] = _case_result(case)

    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
