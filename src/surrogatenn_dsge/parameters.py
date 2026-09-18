from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    prior_type: str
    prior_params: Mapping[str, float]
    bounds: tuple[float, float]
    description: str

    def __post_init__(self) -> None:
        name = str(self.name)
        prior_type = str(self.prior_type)
        params = {str(k): float(v) for k, v in self.prior_params.items()}
        lower, upper = (float(self.bounds[0]), float(self.bounds[1]))
        if not name:
            raise ValueError("ParameterSpec.name must be non-empty.")
        if prior_type not in {"Beta", "Normal", "InvGamma", "Uniform"}:
            raise ValueError(f"Unsupported prior_type={prior_type!r}.")
        if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
            raise ValueError(f"Invalid bounds for {name}: {self.bounds}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prior_type", prior_type)
        object.__setattr__(self, "prior_params", params)
        object.__setattr__(self, "bounds", (lower, upper))
        object.__setattr__(self, "description", str(self.description))


def _spec(
    name: str,
    prior_type: str,
    prior_params: Mapping[str, float],
    bounds: tuple[float, float],
    description: str,
) -> ParameterSpec:
    return ParameterSpec(name, prior_type, prior_params, bounds, description)


def get_phase1_18param_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("crhoa", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "TFP shock persistence"),
        _spec("crhob", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Risk premium shock persistence"),
        _spec("crhog", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Government spending shock persistence"),
        _spec("crhoqs", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Investment shock persistence"),
        _spec("crhopinf", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Price markup shock persistence"),
        _spec("crhow", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Wage markup shock persistence"),
        _spec("crhoms", "Beta", {"alpha": 0.5, "beta": 0.2}, (0.01, 0.99), "Monetary policy shock persistence"),
        _spec("z_ea", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "TFP shock volatility"),
        _spec("z_eb", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Risk premium shock volatility"),
        _spec("z_eg", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Government spending shock volatility"),
        _spec("z_eqs", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Investment shock volatility"),
        _spec("z_epinf", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Price markup shock volatility"),
        _spec("z_ew", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Wage markup shock volatility"),
        _spec("z_em", "InvGamma", {"alpha": 2.0, "theta": 0.1}, (0.001, 5.0), "Monetary policy shock volatility"),
        _spec("cprobp", "Beta", {"alpha": 0.5, "beta": 0.1}, (0.5, 0.95), "Calvo price stickiness (ξ_p)"),
        _spec("cindp", "Beta", {"alpha": 0.5, "beta": 0.15}, (0.01, 0.99), "Price indexation (ι_p)"),
        _spec("curvp", "Normal", {"mu": 64.5, "sigma": 25.0}, (2.0, 150.0), "Kimball price curvature (ε_p)"),
        _spec("cprobw", "Beta", {"alpha": 0.5, "beta": 0.1}, (0.5, 0.95), "Calvo wage stickiness (ξ_w)"),
    )


def get_phase1_18param_narrow_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("crhoa", "Normal", {"mu": 0.95, "sigma": 0.05}, (0.85, 0.995), "TFP shock persistence"),
        _spec("crhob", "Normal", {"mu": 0.60, "sigma": 0.10}, (0.45, 0.75), "Risk premium shock persistence"),
        _spec("crhog", "Normal", {"mu": 0.95, "sigma": 0.05}, (0.85, 0.995), "Government spending shock persistence"),
        _spec("crhoqs", "Normal", {"mu": 0.72, "sigma": 0.08}, (0.60, 0.85), "Investment shock persistence"),
        _spec("crhopinf", "Normal", {"mu": 0.10, "sigma": 0.08}, (0.0, 0.30), "Price markup shock persistence"),
        _spec("crhow", "Normal", {"mu": 0.10, "sigma": 0.08}, (0.0, 0.30), "Wage markup shock persistence"),
        _spec("crhoms", "Normal", {"mu": 0.10, "sigma": 0.08}, (0.0, 0.30), "Monetary policy shock persistence"),
        _spec("z_ea", "Normal", {"mu": 0.46, "sigma": 0.15}, (0.20, 0.80), "TFP shock volatility"),
        _spec("z_eb", "Normal", {"mu": 1.85, "sigma": 0.50}, (1.0, 3.0), "Risk premium shock volatility"),
        _spec("z_eg", "Normal", {"mu": 0.61, "sigma": 0.18}, (0.30, 1.00), "Government spending shock volatility"),
        _spec("z_eqs", "Normal", {"mu": 0.60, "sigma": 0.18}, (0.30, 1.00), "Investment shock volatility"),
        _spec("z_epinf", "Normal", {"mu": 0.15, "sigma": 0.06}, (0.05, 0.30), "Price markup shock volatility"),
        _spec("z_ew", "Normal", {"mu": 0.21, "sigma": 0.08}, (0.10, 0.40), "Wage markup shock volatility"),
        _spec("z_em", "Normal", {"mu": 0.24, "sigma": 0.08}, (0.10, 0.40), "Monetary policy shock volatility"),
        _spec("cprobp", "Normal", {"mu": 0.60, "sigma": 0.08}, (0.50, 0.75), "Calvo price stickiness (ξ_p)"),
        _spec("cindp", "Normal", {"mu": 0.47, "sigma": 0.10}, (0.30, 0.65), "Price indexation (ι_p)"),
        _spec("curvp", "Normal", {"mu": 10.0, "sigma": 4.0}, (5.0, 20.0), "Kimball price curvature (ε_p)"),
        _spec("cprobw", "Normal", {"mu": 0.81, "sigma": 0.06}, (0.70, 0.90), "Calvo wage stickiness (ξ_w)"),
    )


def get_legacy_3param_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("cprobp", "Beta", {"alpha": 0.5, "beta": 0.1}, (0.5, 0.95), "Calvo price stickiness (ξ_p)"),
        _spec("cindp", "Beta", {"alpha": 0.5, "beta": 0.15}, (0.01, 0.99), "Price indexation (ι_p)"),
        _spec("curvp", "Normal", {"mu": 75.0, "sigma": 25.0}, (20.0, 150.0), "Kimball price curvature (ε_p)"),
    )


def get_investment_4p_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("crhob", "Normal", {"mu": 0.5799, "sigma": 0.06}, (0.45, 0.75), "Risk premium shock persistence"),
        _spec("crhoqs", "Normal", {"mu": 0.7165, "sigma": 0.06}, (0.60, 0.85), "Investment-specific shock persistence"),
        _spec("z_eb", "Normal", {"mu": 1.8513, "sigma": 0.25}, (1.20, 2.50), "Risk premium shock volatility"),
        _spec("z_eqs", "Normal", {"mu": 0.6017, "sigma": 0.12}, (0.35, 0.90), "Investment-specific shock volatility"),
    )


def get_investment_4p_supported_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("crhob", "Normal", {"mu": 0.5799, "sigma": 0.06}, (0.45, 0.75), "Risk premium shock persistence"),
        _spec("crhoqs", "Normal", {"mu": 0.7165, "sigma": 0.06}, (0.60, 0.85), "Investment-specific shock persistence"),
        _spec("z_eb", "Normal", {"mu": 1.8513, "sigma": 0.20}, (1.20, 1.85), "Risk premium shock volatility, trimmed to mapped SEP support"),
        _spec("z_eqs", "Normal", {"mu": 0.6017, "sigma": 0.12}, (0.35, 0.90), "Investment-specific shock volatility"),
    )


def get_investment_curvature_5p_specs() -> tuple[ParameterSpec, ...]:
    return (
        _spec("csadjcost", "Normal", {"mu": 6.0144, "sigma": 1.0}, (4.0, 8.5), "Investment adjustment-cost curvature"),
        *get_investment_4p_specs(),
    )


def _normalize_parameter_set(parameter_set: str) -> str:
    name = str(parameter_set).strip()
    if name.startswith(":"):
        name = name[1:]
    return name


def get_parameter_specs(parameter_set: str) -> tuple[ParameterSpec, ...]:
    name = _normalize_parameter_set(parameter_set)
    if name == "legacy_3params":
        return get_legacy_3param_specs()
    if name == "phase1_18params":
        return get_phase1_18param_specs()
    if name == "phase1_18params_narrow":
        return get_phase1_18param_narrow_specs()
    if name == "investment_4p":
        return get_investment_4p_specs()
    if name == "investment_4p_supported":
        return get_investment_4p_supported_specs()
    if name == "investment_curvature_5p":
        return get_investment_curvature_5p_specs()
    raise ValueError(
        "Unknown parameter set: "
        f"{parameter_set!r}. Valid options: legacy_3params, phase1_18params, "
        "phase1_18params_narrow, investment_4p, investment_4p_supported, "
        "investment_curvature_5p."
    )


def get_parameter_names(parameter_set: str) -> tuple[str, ...]:
    return tuple(spec.name for spec in get_parameter_specs(parameter_set))


def get_parameter_bounds(parameter_set: str) -> dict[str, tuple[float, float]]:
    return {spec.name: spec.bounds for spec in get_parameter_specs(parameter_set)}


def parameter_bounds_array(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
) -> tuple[np.ndarray, np.ndarray]:
    specs = (
        get_parameter_specs(parameter_set_or_specs)
        if isinstance(parameter_set_or_specs, str)
        else tuple(parameter_set_or_specs)
    )
    lower = np.asarray([spec.bounds[0] for spec in specs], dtype=np.float64)
    upper = np.asarray([spec.bounds[1] for spec in specs], dtype=np.float64)
    return lower, upper


def get_phase1_18param_baseline() -> dict[str, float]:
    return {
        "csadjcost": 6.0144,
        "crhoa": 0.9977,
        "crhob": 0.5799,
        "crhog": 0.9957,
        "crhoqs": 0.7165,
        "crhopinf": 0.0,
        "crhow": 0.0,
        "crhoms": 0.0,
        "z_ea": 0.4618,
        "z_eb": 1.8513,
        "z_eg": 0.6090,
        "z_eqs": 0.6017,
        "z_epinf": 0.1455,
        "z_ew": 0.2089,
        "z_em": 0.2397,
        "cprobp": 0.6,
        "cindp": 0.47,
        "curvp": 10.0,
        "cprobw": 0.8087,
    }


def baseline_parameter_vector(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    *,
    baseline: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    specs = (
        get_parameter_specs(parameter_set_or_specs)
        if isinstance(parameter_set_or_specs, str)
        else tuple(parameter_set_or_specs)
    )
    values = get_phase1_18param_baseline() if baseline is None else dict(baseline)
    missing = [spec.name for spec in specs if spec.name not in values]
    if missing:
        raise ValueError(f"Baseline is missing values for: {', '.join(missing)}.")
    return np.asarray([float(values[spec.name]) for spec in specs], dtype=np.float64)


def theta_within_bounds(
    theta: Any,
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
) -> bool:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    lower, upper = parameter_bounds_array(parameter_set_or_specs)
    if values.shape != lower.shape:
        raise ValueError(f"theta length mismatch: {values.shape[0]} vs {lower.shape[0]}.")
    return bool(np.all(np.isfinite(values)) and np.all(values >= lower) and np.all(values <= upper))


def theta_within_bounds_jax(
    theta: Any,
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
) -> jnp.ndarray:
    values = jnp.asarray(theta, dtype=jnp.float64).reshape(-1)
    lower_np, upper_np = parameter_bounds_array(parameter_set_or_specs)
    lower = jnp.asarray(lower_np, dtype=jnp.float64)
    upper = jnp.asarray(upper_np, dtype=jnp.float64)
    if values.shape != lower.shape:
        raise ValueError(f"theta length mismatch: {values.shape[0]} vs {lower.shape[0]}.")
    return jnp.all(jnp.isfinite(values) & (values >= lower) & (values <= upper))


def _mask_numpyro_log_prob_to_bounds(base_dist: Any, lower: float, upper: float, dist: Any) -> Any:
    class _BoundedDistribution(dist.Distribution):
        arg_constraints: dict[str, Any] = {}
        pytree_data_fields = ("base_dist",)
        pytree_aux_fields = ("lower", "upper")

        def __init__(self, bounded_base: Any, lower_bound: float, upper_bound: float):
            self.base_dist = bounded_base
            self.lower = float(lower_bound)
            self.upper = float(upper_bound)
            super().__init__(bounded_base.batch_shape, bounded_base.event_shape)

        @property
        def has_rsample(self) -> bool:
            return bool(getattr(self.base_dist, "has_rsample", False))

        @property
        def support(self) -> Any:
            return dist.constraints.interval(self.lower, self.upper)

        @property
        def mean(self) -> Any:
            return self.base_dist.mean

        @property
        def variance(self) -> Any:
            return self.base_dist.variance

        def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
            return self.base_dist.sample(key, sample_shape=sample_shape)

        def rsample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
            return self.base_dist.rsample(key, sample_shape=sample_shape)

        def log_prob(self, value: Any) -> Any:
            value_arr = jnp.asarray(value)
            inside = (value_arr >= self.lower) & (value_arr <= self.upper)
            return jnp.where(inside, self.base_dist.log_prob(value), -jnp.inf)

    return _BoundedDistribution(base_dist, lower, upper)


def make_numpyro_priors(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    *,
    truncate: bool = True,
) -> dict[str, Any]:
    try:
        import numpyro.distributions as dist
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra.
        raise ImportError("make_numpyro_priors requires numpyro to be installed.") from exc

    specs = (
        get_parameter_specs(parameter_set_or_specs)
        if isinstance(parameter_set_or_specs, str)
        else tuple(parameter_set_or_specs)
    )
    priors: dict[str, Any] = {}
    for spec in specs:
        lower, upper = spec.bounds
        if spec.prior_type == "Beta":
            base = dist.Beta(spec.prior_params["alpha"], spec.prior_params["beta"])
            prior = dist.TransformedDistribution(
                base,
                dist.transforms.AffineTransform(lower, upper - lower),
            )
        elif spec.prior_type == "Normal":
            base = dist.Normal(spec.prior_params["mu"], spec.prior_params["sigma"])
            prior = (
                _mask_numpyro_log_prob_to_bounds(
                    dist.TruncatedDistribution(base, low=lower, high=upper),
                    lower,
                    upper,
                    dist,
                )
                if truncate
                else base
            )
        elif spec.prior_type == "InvGamma":
            base = dist.InverseGamma(spec.prior_params["alpha"], spec.prior_params["theta"])
            prior = (
                _mask_numpyro_log_prob_to_bounds(
                    dist.TruncatedDistribution(base, low=lower, high=upper),
                    lower,
                    upper,
                    dist,
                )
                if truncate
                else base
            )
        elif spec.prior_type == "Uniform":
            prior = dist.Uniform(lower, upper)
        else:  # pragma: no cover - constructor prevents this.
            raise ValueError(f"Unsupported prior_type={spec.prior_type!r}.")
        priors[spec.name] = prior
    return priors


def get_parameter_priors(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    *,
    truncate: bool = True,
) -> dict[str, Any]:
    return make_numpyro_priors(parameter_set_or_specs, truncate=truncate)


def format_parameter_summary(parameter_set: str) -> str:
    specs = get_parameter_specs(parameter_set)
    lines = [
        "=" * 80,
        f"Parameter Set: {_normalize_parameter_set(parameter_set)}",
        f"Number of parameters: {len(specs)}",
        "=" * 80,
        "",
    ]
    for i, spec in enumerate(specs, start=1):
        lines.extend(
            [
                f"[{i}] {spec.name}",
                f"    Description: {spec.description}",
                f"    Prior: {spec.prior_type}{dict(spec.prior_params)}",
                f"    Bounds: {spec.bounds}",
                "",
            ]
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def print_parameter_summary(parameter_set: str) -> None:
    print(format_parameter_summary(parameter_set))
