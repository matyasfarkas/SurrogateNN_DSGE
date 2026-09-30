from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from .parameters import (
    ParameterSpec,
    get_parameter_specs,
    parameter_bounds_array,
)


@dataclass(frozen=True)
class ParameterDesign:
    theta: np.ndarray
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    method: str
    unit_sample: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        names = tuple(str(name) for name in self.names)
        if theta.ndim != 2:
            raise ValueError(f"ParameterDesign.theta must be 2D, got shape {theta.shape}.")
        if theta.shape[0] != len(names):
            raise ValueError(f"theta row count {theta.shape[0]} does not match {len(names)} names.")
        if lower.shape != upper.shape or lower.shape != (len(names),):
            raise ValueError("lower/upper bounds must match the parameter-name dimension.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("ParameterDesign.theta must be finite.")
        if not np.all(lower < upper):
            raise ValueError("Parameter bounds must satisfy lower < upper.")
        if self.unit_sample is not None:
            unit_sample = np.asarray(self.unit_sample, dtype=np.float64)
            if unit_sample.shape != theta.shape:
                raise ValueError(
                    f"unit_sample shape {unit_sample.shape} does not match theta shape {theta.shape}."
                )
            object.__setattr__(self, "unit_sample", unit_sample)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "method", str(self.method))


def _coerce_specs(parameter_set_or_specs: Union[str, Sequence[ParameterSpec]]) -> tuple[ParameterSpec, ...]:
    return (
        get_parameter_specs(parameter_set_or_specs)
        if isinstance(parameter_set_or_specs, str)
        else tuple(parameter_set_or_specs)
    )


def latin_hypercube_unit(
    n_samples: int,
    n_dim: int,
    *,
    seed: Optional[int] = None,
    centered: bool = False,
) -> np.ndarray:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    if n_dim < 1:
        raise ValueError(f"n_dim must be >= 1, got {n_dim}.")

    rng = np.random.default_rng(seed)
    sample = np.empty((int(n_dim), int(n_samples)), dtype=np.float64)
    for dim in range(int(n_dim)):
        order = rng.permutation(int(n_samples))
        offset = 0.5 if centered else rng.random(int(n_samples))
        sample[dim, :] = (order + offset) / float(n_samples)
    return sample


def lhs_to_bounds(
    unit_sample: np.ndarray,
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
) -> np.ndarray:
    sample = np.asarray(unit_sample, dtype=np.float64)
    if sample.ndim != 2:
        raise ValueError(f"unit_sample must be 2D with shape (n_parameters, n_samples), got {sample.shape}.")
    specs = _coerce_specs(parameter_set_or_specs)
    lower, upper = parameter_bounds_array(specs)
    if sample.shape[0] != len(specs):
        raise ValueError(f"unit_sample has {sample.shape[0]} rows but parameter set has {len(specs)} parameters.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("unit_sample must be finite.")
    if np.any(sample < 0.0) or np.any(sample > 1.0):
        raise ValueError("unit_sample entries must lie in [0, 1].")
    return lower[:, None] + (upper - lower)[:, None] * sample


def sample_lhs_parameters(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    n_samples: int,
    *,
    seed: Optional[int] = None,
    centered: bool = False,
) -> ParameterDesign:
    specs = _coerce_specs(parameter_set_or_specs)
    unit_sample = latin_hypercube_unit(n_samples, len(specs), seed=seed, centered=centered)
    theta = lhs_to_bounds(unit_sample, specs)
    lower, upper = parameter_bounds_array(specs)
    return ParameterDesign(
        theta=theta,
        names=tuple(spec.name for spec in specs),
        lower=lower,
        upper=upper,
        method="lhs",
        unit_sample=unit_sample,
    )


def _sample_truncated_by_rejection(
    draw_fn: Callable[[int], np.ndarray],
    *,
    lower: float,
    upper: float,
    n_samples: int,
    max_rounds: int = 128,
) -> np.ndarray:
    values = np.empty((int(n_samples),), dtype=np.float64)
    filled = 0
    rounds = 0
    while filled < int(n_samples) and rounds < int(max_rounds):
        rounds += 1
        remaining = int(n_samples) - filled
        raw = np.asarray(draw_fn(max(remaining * 2, 16)), dtype=np.float64).reshape(-1)
        accepted = raw[np.isfinite(raw) & (raw >= float(lower)) & (raw <= float(upper))]
        take = min(remaining, accepted.size)
        if take:
            values[filled : filled + take] = accepted[:take]
            filled += take
    if filled < int(n_samples):
        raise RuntimeError(
            "Could not draw enough bounded prior samples after "
            f"{max_rounds} rejection rounds for bounds ({lower}, {upper})."
        )
    return values


def sample_prior_parameters(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    n_samples: int,
    *,
    seed: Optional[int] = None,
) -> ParameterDesign:
    """Draw an independent bounded prior design using the local ParameterSpec table."""

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    specs = _coerce_specs(parameter_set_or_specs)
    rng = np.random.default_rng(seed)
    theta = np.empty((len(specs), int(n_samples)), dtype=np.float64)
    lower, upper = parameter_bounds_array(specs)

    for idx, spec in enumerate(specs):
        lo = float(lower[idx])
        hi = float(upper[idx])
        params = spec.prior_params
        if spec.prior_type == "Beta":
            alpha = float(params["alpha"])
            beta = float(params["beta"])
            unit = rng.beta(alpha, beta, size=int(n_samples))
            theta[idx, :] = lo + (hi - lo) * unit
        elif spec.prior_type == "Normal":
            mu = float(params["mu"])
            sigma = float(params["sigma"])
            theta[idx, :] = _sample_truncated_by_rejection(
                lambda size, mu=mu, sigma=sigma: rng.normal(mu, sigma, size=size),
                lower=lo,
                upper=hi,
                n_samples=int(n_samples),
            )
        elif spec.prior_type == "InvGamma":
            alpha = float(params["alpha"])
            theta_rate = float(params["theta"])
            theta[idx, :] = _sample_truncated_by_rejection(
                lambda size, alpha=alpha, theta_rate=theta_rate: theta_rate
                / rng.gamma(alpha, 1.0, size=size),
                lower=lo,
                upper=hi,
                n_samples=int(n_samples),
            )
        elif spec.prior_type == "Uniform":
            theta[idx, :] = rng.uniform(lo, hi, size=int(n_samples))
        else:  # pragma: no cover - ParameterSpec validates this.
            raise ValueError(f"Unsupported prior_type={spec.prior_type!r}.")

    return ParameterDesign(
        theta=theta,
        names=tuple(spec.name for spec in specs),
        lower=lower,
        upper=upper,
        method="prior",
    )


def parameter_grid(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    *,
    points_per_dim: Union[int, Mapping[str, int], Sequence[int]] = 5,
    max_points: Optional[int] = 1_000_000,
) -> ParameterDesign:
    specs = _coerce_specs(parameter_set_or_specs)
    lower, upper = parameter_bounds_array(specs)
    names = tuple(spec.name for spec in specs)

    if isinstance(points_per_dim, Mapping):
        counts = tuple(int(points_per_dim.get(name, 5)) for name in names)
    elif isinstance(points_per_dim, int):
        counts = (int(points_per_dim),) * len(specs)
    else:
        counts = tuple(int(value) for value in points_per_dim)
        if len(counts) != len(specs):
            raise ValueError(f"points_per_dim has length {len(counts)} but expected {len(specs)}.")
    if any(count < 1 for count in counts):
        raise ValueError("All grid point counts must be >= 1.")

    total = int(np.prod(np.asarray(counts, dtype=np.int64)))
    if max_points is not None and total > int(max_points):
        raise ValueError(
            f"Parameter grid would contain {total} points, exceeding max_points={max_points}. "
            "Use LHS for high-dimensional parameter sets or raise max_points explicitly."
        )

    axes = [
        np.asarray([(lo + hi) / 2.0], dtype=np.float64) if count == 1 else np.linspace(lo, hi, count)
        for lo, hi, count in zip(lower, upper, counts)
    ]
    columns = itertools.product(*axes)
    theta = np.asarray(list(columns), dtype=np.float64).T
    return ParameterDesign(theta=theta, names=names, lower=lower, upper=upper, method="grid")


def summarize_parameter_design(design: ParameterDesign) -> dict[str, object]:
    theta = np.asarray(design.theta, dtype=np.float64)
    ranges = design.upper - design.lower
    observed_ranges = np.max(theta, axis=1) - np.min(theta, axis=1)
    coverage = np.divide(observed_ranges, ranges, out=np.zeros_like(observed_ranges), where=ranges > 0.0)
    if theta.shape[1] > 1 and theta.shape[0] > 1:
        corr = np.corrcoef(theta)
        off_diag = corr[np.triu_indices(theta.shape[0], k=1)]
        max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
        mean_abs_corr = float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0
    else:
        max_abs_corr = 0.0
        mean_abs_corr = 0.0
    return {
        "method": design.method,
        "n_parameters": len(design.names),
        "n_samples": int(theta.shape[1]),
        "names": design.names,
        "min": dict(zip(design.names, np.min(theta, axis=1))),
        "max": dict(zip(design.names, np.max(theta, axis=1))),
        "coverage": dict(zip(design.names, coverage)),
        "min_coverage": float(np.min(coverage)) if coverage.size else 0.0,
        "max_abs_corr": max_abs_corr,
        "mean_abs_corr": mean_abs_corr,
    }


def sample_parameter_design(
    parameter_set_or_specs: Union[str, Sequence[ParameterSpec]],
    *,
    method: str = "lhs",
    n_samples: Optional[int] = None,
    points_per_dim: Union[int, Mapping[str, int], Sequence[int]] = 5,
    seed: Optional[int] = None,
    centered: bool = False,
    max_points: Optional[int] = 1_000_000,
) -> ParameterDesign:
    normalized = str(method).strip().lower()
    if normalized == "lhs":
        if n_samples is None:
            raise ValueError("n_samples is required for method='lhs'.")
        return sample_lhs_parameters(parameter_set_or_specs, n_samples, seed=seed, centered=centered)
    if normalized == "prior":
        if n_samples is None:
            raise ValueError("n_samples is required for method='prior'.")
        return sample_prior_parameters(parameter_set_or_specs, n_samples, seed=seed)
    if normalized == "grid":
        return parameter_grid(parameter_set_or_specs, points_per_dim=points_per_dim, max_points=max_points)
    raise ValueError(f"Unknown parameter-design method {method!r}. Supported methods: 'lhs', 'prior', 'grid'.")
