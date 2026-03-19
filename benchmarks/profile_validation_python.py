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

from surrogatenn_dsge import (
    RegimeSwitchConfig,
    SEPConfig,
    SwitchingLikelihoodConfig,
    kalman_loglikelihood_from_model_jax,
    inversion_loglikelihood_from_model,
    parse_macro_model,
    solve_first_order_model,
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


def _subset_indices(all_names: tuple[str, ...], subset_names: list[str]) -> np.ndarray:
    return np.asarray([all_names.index(name) for name in subset_names], dtype=np.int64)


def _inject_subset(base_theta: np.ndarray, subset_idx: np.ndarray, x: Any) -> np.ndarray:
    theta = jnp.asarray(base_theta, dtype=jnp.float64)
    theta = theta.at[jnp.asarray(subset_idx, dtype=jnp.int32)].set(
        jnp.asarray(x, dtype=jnp.float64)
    )
    return theta


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
    obs_sigma = {name: float(case["obs_sigma"][name]) for name in observation_names}
    shock_sigmas = {name: float(case["shock_sigmas"][name]) for name in case["shock_names"]}
    parameter_values = np.asarray(model.parameter_values, dtype=np.float64)
    subset_idx = _subset_indices(model.parameter_names, list(case["parameter_subset"]))
    x0 = parameter_values[subset_idx]
    measurement_error_scale = float(case["measurement_error_scale"])
    jitter = float(case["jitter"])

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
    kalman_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda x: kalman_objective_grad(
                _inject_subset(parameter_values, subset_idx, x)
            )
        )
    )

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
                serializer=_scalar_result,
            ),
            "kalman_value": _measure_stage(
                lambda: kalman_value_fn(parameter_values),
                steady_reps=int(case["kalman_value_reps"]),
                serializer=_scalar_result,
            ),
            "kalman_grad": _measure_stage(
                lambda: kalman_grad_fn(x0),
                steady_reps=int(case["kalman_grad_reps"]),
                serializer=_scalar_result,
            ),
            "switching_value": _measure_stage(
                lambda: switching_value_fn(parameter_values),
                steady_reps=int(case["switching_reps"]),
                serializer=_scalar_result,
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
