from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
from jax import core as jax_core
import jax.numpy as jnp
import numpy as np

from .model import MacroModel, kalman_loglikelihood_from_model


def _require_numpyro() -> tuple[Any, Any, Any]:
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer.util import log_density
    except ImportError as exc:
        raise ImportError(
            "NumPyro integration requires the optional `numpyro` dependency. "
            "Install the `inference` extra or add `numpyro` to the environment."
        ) from exc
    return numpyro, dist, log_density


def _coerce_base_parameter_vector(
    model: MacroModel,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]],
) -> jax.Array:
    if base_parameter_values is None:
        return jnp.asarray(model.parameter_values, dtype=jnp.float64)
    if isinstance(base_parameter_values, Mapping):
        unknown = tuple(
            sorted(set(base_parameter_values).difference(model.parameter_names))
        )
        if unknown:
            raise ValueError(
                "Unknown parameter names in `base_parameter_values`: "
                + ", ".join(unknown)
                + "."
            )
        base = np.asarray(model.parameter_values, dtype=np.float64).copy()
        index_lookup = {name: idx for idx, name in enumerate(model.parameter_names)}
        for name, value in base_parameter_values.items():
            base[index_lookup[name]] = float(value)
        return jnp.asarray(base, dtype=jnp.float64)
    base = jnp.asarray(base_parameter_values, dtype=jnp.float64)
    expected_shape = (len(model.parameter_names),)
    if base.shape != expected_shape:
        raise ValueError(
            "base_parameter_values must have shape "
            f"{expected_shape}, got {base.shape}."
        )
    return base


def assemble_parameter_vector(
    model: MacroModel,
    updated_parameter_values: Mapping[str, Any],
    *,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
) -> jax.Array:
    unknown = tuple(
        sorted(set(updated_parameter_values).difference(model.parameter_names))
    )
    if unknown:
        raise ValueError(
            "Unknown parameter names in `updated_parameter_values`: "
            + ", ".join(unknown)
            + "."
        )

    parameter_vector = _coerce_base_parameter_vector(model, base_parameter_values)
    index_lookup = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value in updated_parameter_values.items():
        parameter_vector = parameter_vector.at[index_lookup[name]].set(
            jnp.asarray(value, dtype=jnp.float64)
        )
    return parameter_vector


def build_numpyro_kalman_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    *,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
):
    numpyro, _, _ = _require_numpyro()

    prior_names = tuple(priors)
    if not prior_names:
        raise ValueError("priors must contain at least one parameter prior.")
    unknown = tuple(sorted(set(prior_names).difference(model.parameter_names)))
    if unknown:
        raise ValueError(
            "Unknown parameter names in `priors`: "
            + ", ".join(unknown)
            + "."
        )
    base_parameters = _coerce_base_parameter_vector(model, base_parameter_values)

    def numpyro_model() -> None:
        sampled_values = {
            name: numpyro.sample(name, priors[name])
            for name in prior_names
        }
        if any(isinstance(value, jax_core.Tracer) for value in sampled_values.values()):
            raise NotImplementedError(
                "Parsed-model NumPyro estimation is not yet JAX-traceable enough "
                "for compiled kernels like NUTS. The current wrapper supports "
                "concrete log-density evaluation and explicit parameter-vector "
                "assembly, but the steady-state / symbolic derivative path still "
                "needs a pure-JAX port."
            )

        parameter_vector = assemble_parameter_vector(
            model,
            sampled_values,
            base_parameter_values=base_parameters,
        )
        loglikelihood = kalman_loglikelihood_from_model(
            model,
            observations,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )
        numpyro.deterministic("parameter_vector", parameter_vector)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("kalman_loglikelihood", loglikelihood)

    return numpyro_model


def evaluate_numpyro_kalman_log_density(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    *,
    observables: Optional[Sequence[str] | str] = None,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    _, _, log_density = _require_numpyro()
    numpyro_model = build_numpyro_kalman_model(
        model,
        observations,
        priors,
        observables=observables,
        base_parameter_values=base_parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)
