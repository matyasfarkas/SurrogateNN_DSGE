from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .surrogate import FrozenMLP, FrozenResNet, predict_frozen_batch
from .surrogate_dataset import BatchedSurrogateRolloutArrays, SurrogateDataset


class HLTPipelineArtifactError(ValueError):
    """Raised when canonical HLT parity artifacts cannot be constructed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_float_array(value: Any, *, label: str, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 and not allow_empty:
        raise HLTPipelineArtifactError(f"{label} must be non-empty.")
    if not np.isfinite(array).all():
        raise HLTPipelineArtifactError(f"{label} contains non-finite values.")
    return array


def _as_float_matrix(value: Any, *, label: str) -> np.ndarray:
    array = _as_float_array(value, label=label)
    if array.ndim != 2:
        raise HLTPipelineArtifactError(f"{label} must be a rank-2 matrix, got shape {array.shape}.")
    return array


def _as_bool_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool).reshape(-1)
    if array.size == 0:
        raise HLTPipelineArtifactError(f"{label} must be non-empty.")
    return array


def _as_int_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise HLTPipelineArtifactError(f"{label} must be non-empty.")
    int_array = np.asarray(array, dtype=np.int64).reshape(-1)
    if not np.array_equal(array.reshape(-1), int_array):
        raise HLTPipelineArtifactError(f"{label} must contain integer values.")
    return int_array


def _lookup_mapping(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _require_mapping_value(mapping: Mapping[str, Any], *keys: str, label: str) -> Any:
    value = _lookup_mapping(mapping, *keys)
    if value is None:
        raise HLTPipelineArtifactError(f"Mapping dataset must contain {label}. Accepted keys: {', '.join(keys)}.")
    return value


def _sample_mask_from_source(source: Any, n_samples: int) -> np.ndarray:
    if isinstance(source, BatchedSurrogateRolloutArrays):
        mask = np.asarray(source.sample_mask, dtype=bool).reshape(-1)
    elif isinstance(source, Mapping) and _lookup_mapping(source, "sample_mask", "mask") is not None:
        mask = np.asarray(_lookup_mapping(source, "sample_mask", "mask"), dtype=bool).reshape(-1)
    else:
        mask = np.ones((n_samples,), dtype=bool)
    if mask.shape[0] != n_samples:
        raise HLTPipelineArtifactError(f"sample_mask length mismatch: {mask.shape[0]} vs {n_samples}.")
    return mask


def _dataset_arrays(
    source: SurrogateDataset | BatchedSurrogateRolloutArrays | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    if isinstance(source, SurrogateDataset):
        if source.Y_rom is None:
            raise HLTPipelineArtifactError("SurrogateDataset must include Y_rom to export HLT residual labels.")
        diagnostics = {
            "target_mode": source.target_mode,
            "input_names": source.input_names,
            "output_names": source.output_names,
            "theta_names": source.theta_names,
            "theta": source.theta,
            "theta_ids": source.theta_ids,
            "period_ids": source.period_ids,
            "theta_success": source.theta_success,
            "theta_stable_periods": source.theta_stable_periods,
        }
        return source.X, source.Y, source.Y_rom, diagnostics, np.ones((source.n_samples,), dtype=bool)

    if isinstance(source, BatchedSurrogateRolloutArrays):
        X = _as_float_matrix(source.X, label="arrays.X")
        Y = _as_float_matrix(source.Y, label="arrays.Y")
        Y_rom = _as_float_matrix(source.Y_rom, label="arrays.Y_rom")
        diagnostics = {
            "theta": _as_float_array(source.theta, label="arrays.theta"),
            "theta_ids": _as_int_vector(source.theta_ids, label="arrays.theta_ids"),
            "period_ids": _as_int_vector(source.period_ids, label="arrays.period_ids"),
            "theta_success": _as_bool_vector(source.theta_success, label="arrays.theta_success"),
            "theta_stable_periods": _as_int_vector(
                source.theta_stable_periods,
                label="arrays.theta_stable_periods",
            ),
            "fixed_shape_total_columns": int(X.shape[1]),
        }
        return X, Y, Y_rom, diagnostics, _sample_mask_from_source(source, X.shape[1])

    if isinstance(source, Mapping):
        X = _as_float_matrix(
            _require_mapping_value(source, "X", "features", "support_features", label="X"),
            label="X",
        )
        Y = _as_float_matrix(
            _require_mapping_value(source, "Y", "targets", "sep_targets", label="Y"),
            label="Y",
        )
        y_rom_value = _lookup_mapping(source, "Y_rom", "Y_rom1", "rom_targets")
        if y_rom_value is None:
            raise HLTPipelineArtifactError("Mapping dataset must contain Y_rom, Y_rom1, or rom_targets.")
        Y_rom = _as_float_matrix(y_rom_value, label="Y_rom")
        diagnostics = {}
        for key in (
            "target_mode",
            "input_names",
            "output_names",
            "theta_names",
            "theta",
            "theta_ids",
            "period_ids",
            "theta_success",
            "theta_stable_periods",
        ):
            if key in source:
                diagnostics[key] = source[key]
        return X, Y, Y_rom, diagnostics, _sample_mask_from_source(source, X.shape[1])

    raise TypeError(f"Unsupported HLT artifact source type: {type(source)!r}.")


def _selected_columns(n_samples: int, sample_mask: np.ndarray, *, valid_only: bool) -> np.ndarray:
    if bool(valid_only):
        selected = np.flatnonzero(sample_mask)
    else:
        selected = np.arange(n_samples, dtype=np.int64)
    if selected.size == 0:
        raise HLTPipelineArtifactError("No samples selected for artifact export.")
    return selected.astype(np.int64)


def _coerce_selected_indices(
    selected_indices: Optional[Any],
    selected_columns: np.ndarray,
    *,
    n_samples: int,
) -> np.ndarray:
    if selected_indices is None:
        # Julia's exporter writes 1-based column ids by default.
        return selected_columns.astype(np.int64) + 1
    values = _as_int_vector(selected_indices, label="support_selected_indices")
    if values.shape[0] == n_samples:
        return values[selected_columns]
    if values.shape[0] == selected_columns.shape[0]:
        return values
    raise HLTPipelineArtifactError(
        "support_selected_indices must have one entry per original sample or one entry per exported sample; "
        f"got {values.shape[0]} for {n_samples} original / {selected_columns.shape[0]} exported."
    )


def _maybe_add_array(section: dict[str, Any], name: str, value: Optional[Any], *, bool_array: bool = False) -> None:
    if value is None:
        return
    if bool_array:
        section[name] = _as_bool_vector(value, label=name)
    else:
        section[name] = _as_float_array(value, label=name)


def _surrogate_frozen(surrogate: Optional[Any]) -> Optional[FrozenMLP | FrozenResNet]:
    if surrogate is None:
        return None
    if isinstance(surrogate, (FrozenMLP, FrozenResNet)):
        return surrogate
    frozen = getattr(surrogate, "frozen", None)
    if isinstance(frozen, (FrozenMLP, FrozenResNet)):
        return frozen
    raise TypeError("surrogate must be a FrozenMLP, FrozenResNet, or bundle-like object with a `frozen` field.")


def build_hlt_pipeline_artifact_payload(
    *,
    case: str,
    dataset: SurrogateDataset | BatchedSurrogateRolloutArrays | Mapping[str, Any],
    surrogate: Optional[Any] = None,
    surrogate_predictions: Optional[Any] = None,
    support_selected_indices: Optional[Any] = None,
    valid_only: bool = True,
    rom_states: Optional[Any] = None,
    rom_observations: Optional[Any] = None,
    rom_shocks: Optional[Any] = None,
    gate_e_stat: Optional[Any] = None,
    gate_f_stat: Optional[Any] = None,
    gate_probs: Optional[Any] = None,
    gate_mask: Optional[Any] = None,
    switching_loglik_per_period: Optional[Any] = None,
    posterior_log_density: Optional[float] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the canonical HLT parity JSON payload for Python pipeline outputs.

    The emitted schema matches `benchmarks/export_julia_hlt_pipeline_artifacts.jl`
    and `benchmarks/validate_hlt_pipeline_parity.py`: support features, SEP/FOM
    targets, ROM targets, residual labels, optional frozen-surrogate residual
    predictions, optional ROM/gate/likelihood arrays, and optional posterior log
    density.
    """

    X_all, Y_all, Y_rom_all, source_diagnostics, sample_mask = _dataset_arrays(dataset)
    if X_all.shape[1] != Y_all.shape[1] or Y_rom_all.shape != Y_all.shape:
        raise HLTPipelineArtifactError(
            f"Dataset shape mismatch: X={X_all.shape}, Y={Y_all.shape}, Y_rom={Y_rom_all.shape}."
        )
    selected = _selected_columns(X_all.shape[1], sample_mask, valid_only=valid_only)
    X = X_all[:, selected]
    Y = Y_all[:, selected]
    Y_rom = Y_rom_all[:, selected]
    residual_labels = Y - Y_rom

    support = {
        "features": X,
        "selected_indices": _coerce_selected_indices(
            support_selected_indices,
            selected,
            n_samples=X_all.shape[1],
        ),
    }
    sep = {
        "targets": Y,
        "rom_targets": Y_rom,
        "residual_labels": residual_labels,
    }
    artifact_diagnostics: dict[str, Any] = {
        "source": "python",
        "total_columns": int(X_all.shape[1]),
        "exported_columns": int(selected.size),
        "selected_columns_zero_based": selected,
        "valid_only": bool(valid_only),
        **source_diagnostics,
    }
    if diagnostics:
        artifact_diagnostics.update(dict(diagnostics))

    artifacts: dict[str, Any] = {
        "support": support,
        "sep": sep,
        "diagnostics": artifact_diagnostics,
    }

    frozen = _surrogate_frozen(surrogate)
    if surrogate_predictions is not None and frozen is not None:
        raise HLTPipelineArtifactError("Provide either surrogate or surrogate_predictions, not both.")
    if frozen is not None:
        predicted = np.asarray(predict_frozen_batch(frozen, X), dtype=np.float64)
    elif surrogate_predictions is not None:
        predicted_all = _as_float_matrix(surrogate_predictions, label="surrogate_predictions")
        if predicted_all.shape[1] == X_all.shape[1]:
            predicted = predicted_all[:, selected]
        elif predicted_all.shape[1] == selected.shape[0]:
            predicted = predicted_all
        else:
            raise HLTPipelineArtifactError(
                "surrogate_predictions must have one column per original sample or exported sample; "
                f"got {predicted_all.shape[1]} for {X_all.shape[1]} original / {selected.shape[0]} exported."
            )
    else:
        predicted = None
    if predicted is not None:
        if predicted.ndim != 2 or predicted.shape[1] != X.shape[1]:
            raise HLTPipelineArtifactError(
                f"surrogate_predictions shape mismatch: expected (*, {X.shape[1]}), got {predicted.shape}."
            )
        artifacts["surrogate"] = {"predictions": predicted}

    rom: dict[str, Any] = {}
    _maybe_add_array(rom, "states", rom_states)
    _maybe_add_array(rom, "observations", rom_observations)
    _maybe_add_array(rom, "shocks", rom_shocks)
    if rom:
        artifacts["rom"] = rom

    gate: dict[str, Any] = {}
    _maybe_add_array(gate, "e_stat", gate_e_stat)
    _maybe_add_array(gate, "f_stat", gate_f_stat)
    _maybe_add_array(gate, "gate_probs", gate_probs)
    _maybe_add_array(gate, "hard_mask", gate_mask, bool_array=True)
    if gate:
        artifacts["gate"] = gate

    if switching_loglik_per_period is not None:
        artifacts["likelihood"] = {
            "switching_per_period": _as_float_array(
                switching_loglik_per_period,
                label="switching_loglik_per_period",
            )
        }
    if posterior_log_density is not None:
        log_density = float(posterior_log_density)
        if not np.isfinite(log_density):
            raise HLTPipelineArtifactError("posterior_log_density must be finite.")
        artifacts["posterior"] = {"log_density": log_density}

    return {"cases": {str(case): {"artifacts": _jsonable(artifacts)}}}


def save_hlt_pipeline_artifact_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_hlt_pipeline_artifact_payload(path: str | Path) -> dict[str, Any]:
    in_path = Path(path)
    with in_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise HLTPipelineArtifactError(f"{in_path} must contain a JSON object.")
    return payload
