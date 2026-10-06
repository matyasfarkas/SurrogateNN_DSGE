from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _validate_draw_array(samples: Any, parameter_names: Sequence[str]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            "posterior draws must have shape (samples, chains, parameters); "
            f"got {array.shape}."
        )
    if array.shape[-1] != len(parameter_names):
        raise ValueError(
            "parameter_names length must match posterior draw parameter dimension: "
            f"{len(parameter_names)} vs {array.shape[-1]}."
        )
    if not np.isfinite(array).all():
        raise ValueError("posterior draws must be finite.")
    return array


def save_posterior_draws_npz(
    path: str | Path,
    samples: Any,
    parameter_names: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist post-warmup posterior draws in a compact, comparable format.

    The canonical layout is ``(samples, chains, parameters)``. This matches the
    static HMC sampler layout and keeps chain information for diagnostics while
    making per-parameter flattening unambiguous.
    """

    names = tuple(str(name) for name in parameter_names)
    array = _validate_draw_array(samples, names)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata or {}), sort_keys=True, default=_json_default)
    np.savez_compressed(
        output,
        samples=array,
        parameter_names=np.asarray(names, dtype=object),
        metadata_json=np.asarray(metadata_json, dtype=object),
    )
    return {
        "path": str(output),
        "sample_shape": list(array.shape),
        "post_warmup_draws": int(array.shape[0] * array.shape[1]),
        "parameter_names": list(names),
    }


def load_posterior_draws_npz(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=True) as payload:
        samples = np.asarray(payload["samples"], dtype=np.float64)
        parameter_names = tuple(str(name) for name in payload["parameter_names"].tolist())
        metadata_raw = payload.get("metadata_json")
        metadata_json = str(metadata_raw.tolist()) if metadata_raw is not None else "{}"
    _validate_draw_array(samples, parameter_names)
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        metadata = {"raw_metadata_json": metadata_json}
    return {
        "path": str(source),
        "samples": samples,
        "parameter_names": parameter_names,
        "metadata": metadata,
    }


def _flat_by_parameter(samples: np.ndarray, parameter_names: Sequence[str]) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(samples[:, :, idx], dtype=np.float64).reshape(-1)
        for idx, name in enumerate(parameter_names)
    }


def summarize_posterior_draws(
    samples: Any,
    parameter_names: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    names = tuple(str(name) for name in parameter_names)
    array = _validate_draw_array(samples, names)
    flat = _flat_by_parameter(array, names)
    q = tuple(float(value) for value in quantiles)
    parameters = {}
    for name, values in flat.items():
        q_values = np.quantile(values, q)
        parameters[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": {str(level): float(value) for level, value in zip(q, q_values)},
        }
    return {
        "samples_shape": list(array.shape),
        "post_warmup_draws": int(array.shape[0] * array.shape[1]),
        "parameter_names": list(names),
        "parameters": parameters,
    }


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    x = np.sort(np.concatenate([left, right]))
    if x.size == 0:
        return float("nan")
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, x, side="right") / float(left_sorted.size)
    right_cdf = np.searchsorted(right_sorted, x, side="right") / float(right_sorted.size)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _quantile_wasserstein(left: np.ndarray, right: np.ndarray, *, grid_size: int = 1001) -> float:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    grid = np.linspace(0.0, 1.0, int(grid_size))
    left_q = np.quantile(left, grid)
    right_q = np.quantile(right, grid)
    return float(np.mean(np.abs(left_q - right_q)))


def compare_posterior_draws(
    left_samples: Any,
    left_parameter_names: Sequence[str],
    right_samples: Any,
    right_parameter_names: Sequence[str],
    *,
    left_label: str = "left",
    right_label: str = "right",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    left_names = tuple(str(name) for name in left_parameter_names)
    right_names = tuple(str(name) for name in right_parameter_names)
    left_array = _validate_draw_array(left_samples, left_names)
    right_array = _validate_draw_array(right_samples, right_names)
    common = tuple(name for name in left_names if name in set(right_names))
    if not common:
        raise ValueError("posterior draw files do not share any parameter names.")
    left_flat = _flat_by_parameter(left_array, left_names)
    right_flat = _flat_by_parameter(right_array, right_names)
    q = tuple(float(value) for value in quantiles)
    parameters = {}
    max_abs_mean_diff = 0.0
    max_ks_distance = 0.0
    wasserstein_values = []
    for name in common:
        left_values = left_flat[name]
        right_values = right_flat[name]
        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        left_std = float(np.std(left_values))
        right_std = float(np.std(right_values))
        pooled_std = float(np.sqrt(0.5 * (left_std**2 + right_std**2)))
        mean_diff = right_mean - left_mean
        q_left = np.quantile(left_values, q)
        q_right = np.quantile(right_values, q)
        quantile_diff = {
            str(level): float(candidate - baseline)
            for level, baseline, candidate in zip(q, q_left, q_right)
        }
        ks = _ks_distance(left_values, right_values)
        wasserstein = _quantile_wasserstein(left_values, right_values)
        max_abs_mean_diff = max(max_abs_mean_diff, abs(mean_diff))
        max_ks_distance = max(max_ks_distance, ks)
        wasserstein_values.append(wasserstein)
        parameters[name] = {
            "left_mean": left_mean,
            "right_mean": right_mean,
            "mean_diff": float(mean_diff),
            "standardized_mean_diff": (
                float(mean_diff / pooled_std) if pooled_std > 0.0 else None
            ),
            "left_std": left_std,
            "right_std": right_std,
            "std_ratio": float(right_std / left_std) if left_std > 0.0 else None,
            "quantile_diff": quantile_diff,
            "ks_distance": ks,
            "quantile_wasserstein": wasserstein,
            "quantile_wasserstein_over_left_std": (
                float(wasserstein / left_std) if left_std > 0.0 else None
            ),
        }
    return {
        "left_label": str(left_label),
        "right_label": str(right_label),
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
        "common_parameters": list(common),
        "missing_from_left": [name for name in right_names if name not in left_names],
        "missing_from_right": [name for name in left_names if name not in right_names],
        "max_abs_mean_diff": float(max_abs_mean_diff),
        "max_ks_distance": float(max_ks_distance),
        "mean_quantile_wasserstein": float(np.mean(wasserstein_values)),
        "parameters": parameters,
    }


__all__ = [
    "compare_posterior_draws",
    "load_posterior_draws_npz",
    "save_posterior_draws_npz",
    "summarize_posterior_draws",
]
