from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from .surrogate import (
    FrozenMLP,
    FrozenResNet,
    NormStats,
    ResBlock,
    SurrogateValidationResult,
    predict_frozen_batch,
    resolve_jax_device,
    train_mlp,
    train_resnet,
    validate_surrogate,
)
from .surrogate_dataset import (
    BatchedSurrogateRolloutArrays,
    PredictTupleFn,
    SurrogateDataset,
    build_surrogate_residual_dataset,
    summarize_surrogate_dataset,
)


SURROGATE_BUNDLE_VERSION = 1


@dataclass(frozen=True)
class SurrogateTrainValidationSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    split_by_theta: bool
    validation_fraction: float
    only_full_success: bool
    train_theta_ids: Optional[np.ndarray] = None
    val_theta_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        train_idx = np.asarray(self.train_idx, dtype=np.int64).reshape(-1)
        val_idx = np.asarray(self.val_idx, dtype=np.int64).reshape(-1)
        if train_idx.size < 1:
            raise ValueError("train_idx must contain at least one sample.")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError("train_idx and val_idx must be disjoint.")
        object.__setattr__(self, "train_idx", train_idx)
        object.__setattr__(self, "val_idx", val_idx)
        object.__setattr__(self, "split_by_theta", bool(self.split_by_theta))
        object.__setattr__(self, "validation_fraction", float(self.validation_fraction))
        object.__setattr__(self, "only_full_success", bool(self.only_full_success))
        if self.train_theta_ids is not None:
            object.__setattr__(self, "train_theta_ids", np.asarray(self.train_theta_ids, dtype=np.int64).reshape(-1))
        if self.val_theta_ids is not None:
            object.__setattr__(self, "val_theta_ids", np.asarray(self.val_theta_ids, dtype=np.int64).reshape(-1))

    @property
    def train_size(self) -> int:
        return int(self.train_idx.size)

    @property
    def val_size(self) -> int:
        return int(self.val_idx.size)


@dataclass(frozen=True)
class SurrogateTrainingResult:
    frozen: FrozenMLP | FrozenResNet
    architecture: str
    split: SurrogateTrainValidationSplit
    validation: Optional[SurrogateValidationResult]
    validation_rmse: Optional[np.ndarray]
    validation_rmse_residual: Optional[np.ndarray]
    validation_rmse_rom: Optional[np.ndarray]
    validation_improvement: Optional[np.ndarray]
    target_is_residual: bool
    output_indices: Optional[np.ndarray]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", str(self.architecture).lower())
        object.__setattr__(self, "target_is_residual", bool(self.target_is_residual))
        for field in ("validation_rmse", "validation_rmse_residual", "validation_rmse_rom", "validation_improvement"):
            values = getattr(self, field)
            if values is not None:
                array = np.asarray(values, dtype=np.float64).reshape(-1)
                object.__setattr__(self, field, array)
        if self.output_indices is not None:
            object.__setattr__(self, "output_indices", np.asarray(self.output_indices, dtype=np.int64).reshape(-1))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def train_size(self) -> int:
        return self.split.train_size

    @property
    def val_size(self) -> int:
        return self.split.val_size


@dataclass(frozen=True)
class SurrogateBundle:
    path: Optional[str]
    frozen: FrozenMLP | FrozenResNet
    metadata: dict[str, object]
    validation_rmse: Optional[np.ndarray] = None
    validation_rmse_residual: Optional[np.ndarray] = None
    validation_rmse_rom: Optional[np.ndarray] = None
    validation_improvement: Optional[np.ndarray] = None
    train_idx: Optional[np.ndarray] = None
    val_idx: Optional[np.ndarray] = None
    train_theta_ids: Optional[np.ndarray] = None
    val_theta_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        for field in (
            "validation_rmse",
            "validation_rmse_residual",
            "validation_rmse_rom",
            "validation_improvement",
            "train_idx",
            "val_idx",
            "train_theta_ids",
            "val_theta_ids",
        ):
            values = getattr(self, field)
            if values is None:
                continue
            dtype = np.int64 if field in {"train_idx", "val_idx", "train_theta_ids", "val_theta_ids"} else np.float64
            object.__setattr__(self, field, np.asarray(values, dtype=dtype).reshape(-1))


@dataclass(frozen=True)
class SurrogatePipelineResult:
    dataset: SurrogateDataset
    dataset_summary: dict[str, object]
    training: SurrogateTrainingResult
    bundle_path: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_summary", dict(self.dataset_summary))
        if self.bundle_path is not None:
            object.__setattr__(self, "bundle_path", Path(self.bundle_path))


def split_surrogate_dataset(
    dataset: SurrogateDataset,
    *,
    validation_fraction: float = 0.10,
    split_by_theta: bool = False,
    only_full_success: bool = False,
    seed: int = 1,
) -> SurrogateTrainValidationSplit:
    """Split a surrogate residual dataset into train/validation columns.

    `split_by_theta=True` mirrors the Julia training script's held-out-parameter
    validation mode: all samples from selected parameter draws are held out.
    """

    val_frac = float(validation_fraction)
    if val_frac < 0.0 or val_frac >= 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}.")

    sample_idx = np.arange(dataset.n_samples, dtype=np.int64)
    if only_full_success:
        keep_mask = dataset.theta_success[dataset.theta_ids]
        sample_idx = sample_idx[keep_mask]
        if sample_idx.size == 0:
            raise ValueError("No samples remain after filtering to fully successful theta paths.")

    rng = np.random.default_rng(int(seed))
    sample_theta_ids = dataset.theta_ids[sample_idx]
    if split_by_theta:
        groups = np.unique(sample_theta_ids)
        if val_frac <= 0.0 or sample_idx.size == 1:
            return SurrogateTrainValidationSplit(
                train_idx=sample_idx,
                val_idx=np.zeros((0,), dtype=np.int64),
                split_by_theta=True,
                validation_fraction=val_frac,
                only_full_success=only_full_success,
                train_theta_ids=groups,
                val_theta_ids=np.zeros((0,), dtype=np.int64),
            )
        if groups.size < 2:
            raise ValueError("split_by_theta requires at least two theta groups when validation_fraction > 0.")
        shuffled_groups = rng.permutation(groups)
        n_val_groups = int(np.rint(val_frac * groups.size))
        n_val_groups = min(max(n_val_groups, 1), groups.size - 1)
        val_theta_ids = np.asarray(shuffled_groups[:n_val_groups], dtype=np.int64)
        train_theta_ids = np.asarray(shuffled_groups[n_val_groups:], dtype=np.int64)
        val_mask = np.isin(sample_theta_ids, val_theta_ids)
        train_idx = sample_idx[~val_mask]
        val_idx = sample_idx[val_mask]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Theta-group split produced an empty train or validation set.")
        return SurrogateTrainValidationSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            split_by_theta=True,
            validation_fraction=val_frac,
            only_full_success=only_full_success,
            train_theta_ids=train_theta_ids,
            val_theta_ids=val_theta_ids,
        )

    if val_frac <= 0.0 or sample_idx.size == 1:
        train_idx = sample_idx
        val_idx = np.zeros((0,), dtype=np.int64)
    else:
        perm = rng.permutation(sample_idx)
        n_train = int(np.floor((1.0 - val_frac) * sample_idx.size))
        n_train = min(max(n_train, 1), sample_idx.size - 1)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
    return SurrogateTrainValidationSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        split_by_theta=False,
        validation_fraction=val_frac,
        only_full_success=only_full_success,
        train_theta_ids=np.unique(dataset.theta_ids[train_idx]),
        val_theta_ids=np.unique(dataset.theta_ids[val_idx]) if val_idx.size else np.zeros((0,), dtype=np.int64),
    )


def surrogate_sample_weights_from_residuals(
    sep_residuals: Any,
    *,
    epsilon_floor: float = 1e-8,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> np.ndarray:
    """Compute Julia-style inverse-SEP-residual sample weights.

    Lower SEP residuals receive larger weights. Invalid residuals fall back to
    unit raw weight, and the final vector is normalized to mean one.
    """

    residuals = np.asarray(sep_residuals, dtype=np.float64).reshape(-1)
    if residuals.size < 1:
        raise ValueError("sep_residuals must contain at least one value.")
    if epsilon_floor <= 0.0:
        raise ValueError(f"epsilon_floor must be positive, got {epsilon_floor}.")
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError(f"Invalid clip bounds: clip_min={clip_min}, clip_max={clip_max}.")
    valid = np.isfinite(residuals) & (residuals > 0.0)
    if not np.any(valid):
        return np.ones_like(residuals, dtype=np.float64)
    raw = np.ones_like(residuals, dtype=np.float64)
    raw[valid] = 1.0 / (residuals[valid] + float(epsilon_floor))
    raw /= float(np.mean(raw))
    raw = np.clip(raw, float(clip_min), float(clip_max))
    raw /= float(np.mean(raw))
    return raw


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _array_or_none(value: Optional[Any], *, dtype: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype).reshape(-1)


def _put_loaded_array(npz: Any, key: str, device: Optional[jax.Device]) -> jax.Array:
    value = jnp.asarray(np.asarray(npz[key]), dtype=jnp.float64)
    return value if device is None else jax.device_put(value, device)


def _norm_from_npz(npz: Any, device: Optional[jax.Device]) -> NormStats:
    return NormStats(
        mu_x=_put_loaded_array(npz, "norm_mu_x", device),
        sigma_x=_put_loaded_array(npz, "norm_sigma_x", device),
        mu_y=_put_loaded_array(npz, "norm_mu_y", device),
        sigma_y=_put_loaded_array(npz, "norm_sigma_y", device),
    )


def _frozen_arrays_and_meta(frozen: FrozenMLP | FrozenResNet) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {
        "norm_mu_x": np.asarray(frozen.norm.mu_x, dtype=np.float64),
        "norm_sigma_x": np.asarray(frozen.norm.sigma_x, dtype=np.float64),
        "norm_mu_y": np.asarray(frozen.norm.mu_y, dtype=np.float64),
        "norm_sigma_y": np.asarray(frozen.norm.sigma_y, dtype=np.float64),
    }
    if isinstance(frozen, FrozenMLP):
        arrays.update(
            {
                "mlp_W1": np.asarray(frozen.W1, dtype=np.float64),
                "mlp_b1": np.asarray(frozen.b1, dtype=np.float64),
                "mlp_W2": np.asarray(frozen.W2, dtype=np.float64),
                "mlp_b2": np.asarray(frozen.b2, dtype=np.float64),
            }
        )
        has_second_hidden = frozen.W3 is not None
        if has_second_hidden:
            assert frozen.W3 is not None and frozen.b3 is not None
            arrays["mlp_W3"] = np.asarray(frozen.W3, dtype=np.float64)
            arrays["mlp_b3"] = np.asarray(frozen.b3, dtype=np.float64)
        return arrays, {
            "frozen_type": "mlp",
            "d_in": frozen.d_in,
            "d_out": frozen.d_out,
            "activation": frozen.activation,
            "has_second_hidden": has_second_hidden,
        }

    arrays.update(
        {
            "resnet_W_embed": np.asarray(frozen.W_embed, dtype=np.float64),
            "resnet_b_embed": np.asarray(frozen.b_embed, dtype=np.float64),
            "resnet_W_gamma": np.asarray(frozen.W_gamma, dtype=np.float64),
            "resnet_b_gamma": np.asarray(frozen.b_gamma, dtype=np.float64),
            "resnet_W_beta": np.asarray(frozen.W_beta, dtype=np.float64),
            "resnet_b_beta": np.asarray(frozen.b_beta, dtype=np.float64),
            "resnet_W_out": np.asarray(frozen.W_out, dtype=np.float64),
            "resnet_b_out": np.asarray(frozen.b_out, dtype=np.float64),
        }
    )
    for idx, block in enumerate(frozen.blocks):
        arrays[f"resnet_block_{idx}_W1"] = np.asarray(block.W1, dtype=np.float64)
        arrays[f"resnet_block_{idx}_b1"] = np.asarray(block.b1, dtype=np.float64)
        arrays[f"resnet_block_{idx}_W2"] = np.asarray(block.W2, dtype=np.float64)
        arrays[f"resnet_block_{idx}_b2"] = np.asarray(block.b2, dtype=np.float64)
    return arrays, {
        "frozen_type": "resnet",
        "d_in": frozen.d_in,
        "d_out": frozen.d_out,
        "d_theta": frozen.d_theta,
        "n_blocks": len(frozen.blocks),
    }


def _frozen_from_npz(npz: Any, metadata: dict[str, object], device: Optional[jax.Device]) -> FrozenMLP | FrozenResNet:
    frozen_meta = metadata.get("frozen", {})
    if not isinstance(frozen_meta, dict):
        raise ValueError("Surrogate bundle metadata field 'frozen' must be a mapping.")
    frozen_type = str(frozen_meta.get("frozen_type", "")).lower()
    norm = _norm_from_npz(npz, device)
    if frozen_type == "mlp":
        has_second_hidden = bool(frozen_meta.get("has_second_hidden", False))
        return FrozenMLP(
            W1=_put_loaded_array(npz, "mlp_W1", device),
            b1=_put_loaded_array(npz, "mlp_b1", device),
            W2=_put_loaded_array(npz, "mlp_W2", device),
            b2=_put_loaded_array(npz, "mlp_b2", device),
            W3=_put_loaded_array(npz, "mlp_W3", device) if has_second_hidden else None,
            b3=_put_loaded_array(npz, "mlp_b3", device) if has_second_hidden else None,
            norm=norm,
            d_in=int(frozen_meta["d_in"]),
            d_out=int(frozen_meta["d_out"]),
            activation=str(frozen_meta.get("activation", "tanh")),
        )
    if frozen_type == "resnet":
        n_blocks = int(frozen_meta["n_blocks"])
        blocks = tuple(
            ResBlock(
                W1=_put_loaded_array(npz, f"resnet_block_{idx}_W1", device),
                b1=_put_loaded_array(npz, f"resnet_block_{idx}_b1", device),
                W2=_put_loaded_array(npz, f"resnet_block_{idx}_W2", device),
                b2=_put_loaded_array(npz, f"resnet_block_{idx}_b2", device),
            )
            for idx in range(n_blocks)
        )
        return FrozenResNet(
            W_embed=_put_loaded_array(npz, "resnet_W_embed", device),
            b_embed=_put_loaded_array(npz, "resnet_b_embed", device),
            d_theta=int(frozen_meta["d_theta"]),
            W_gamma=_put_loaded_array(npz, "resnet_W_gamma", device),
            b_gamma=_put_loaded_array(npz, "resnet_b_gamma", device),
            W_beta=_put_loaded_array(npz, "resnet_W_beta", device),
            b_beta=_put_loaded_array(npz, "resnet_b_beta", device),
            blocks=blocks,
            W_out=_put_loaded_array(npz, "resnet_W_out", device),
            b_out=_put_loaded_array(npz, "resnet_b_out", device),
            norm=norm,
            d_in=int(frozen_meta["d_in"]),
            d_out=int(frozen_meta["d_out"]),
        )
    raise ValueError(f"Unsupported frozen surrogate type in bundle: {frozen_type!r}.")


def save_surrogate_bundle(
    path: str | Path,
    result_or_frozen: SurrogateTrainingResult | FrozenMLP | FrozenResNet,
    *,
    metadata: Optional[dict[str, object]] = None,
    validation_rmse: Optional[Any] = None,
    validation_rmse_residual: Optional[Any] = None,
    validation_rmse_rom: Optional[Any] = None,
    validation_improvement: Optional[Any] = None,
    train_idx: Optional[Any] = None,
    val_idx: Optional[Any] = None,
    train_theta_ids: Optional[Any] = None,
    val_theta_ids: Optional[Any] = None,
) -> Path:
    """Save a frozen surrogate bundle in a portable Python-native NPZ format."""

    if isinstance(result_or_frozen, SurrogateTrainingResult):
        result = result_or_frozen
        frozen = result.frozen
        bundle_metadata = dict(result.metadata)
        if metadata is not None:
            bundle_metadata.update(metadata)
        validation_rmse = result.validation_rmse if validation_rmse is None else validation_rmse
        validation_rmse_residual = (
            result.validation_rmse_residual if validation_rmse_residual is None else validation_rmse_residual
        )
        validation_rmse_rom = result.validation_rmse_rom if validation_rmse_rom is None else validation_rmse_rom
        validation_improvement = result.validation_improvement if validation_improvement is None else validation_improvement
        train_idx = result.split.train_idx if train_idx is None else train_idx
        val_idx = result.split.val_idx if val_idx is None else val_idx
        train_theta_ids = result.split.train_theta_ids if train_theta_ids is None else train_theta_ids
        val_theta_ids = result.split.val_theta_ids if val_theta_ids is None else val_theta_ids
    else:
        frozen = result_or_frozen
        if not isinstance(frozen, (FrozenMLP, FrozenResNet)):
            raise TypeError("result_or_frozen must be a SurrogateTrainingResult, FrozenMLP, or FrozenResNet.")
        bundle_metadata = {} if metadata is None else dict(metadata)

    arrays, frozen_meta = _frozen_arrays_and_meta(frozen)
    optional_arrays = {
        "validation_rmse": _array_or_none(validation_rmse, dtype=np.float64),
        "validation_rmse_residual": _array_or_none(validation_rmse_residual, dtype=np.float64),
        "validation_rmse_rom": _array_or_none(validation_rmse_rom, dtype=np.float64),
        "validation_improvement": _array_or_none(validation_improvement, dtype=np.float64),
        "train_idx": _array_or_none(train_idx, dtype=np.int64),
        "val_idx": _array_or_none(val_idx, dtype=np.int64),
        "train_theta_ids": _array_or_none(train_theta_ids, dtype=np.int64),
        "val_theta_ids": _array_or_none(val_theta_ids, dtype=np.int64),
    }
    present_optional = [name for name, values in optional_arrays.items() if values is not None]
    arrays.update({name: values for name, values in optional_arrays.items() if values is not None})
    metadata_payload = {
        "bundle_version": SURROGATE_BUNDLE_VERSION,
        "format": "surrogatenn_dsge_surrogate_bundle_npz",
        "frozen": frozen_meta,
        "metadata": _json_safe(bundle_metadata),
        "optional_arrays": present_optional,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata_payload, sort_keys=True))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(out_path)
    return out_path


def load_surrogate_bundle(path: str | Path, *, device: Optional[Any] = None) -> SurrogateBundle:
    """Load a Python-native surrogate bundle saved by `save_surrogate_bundle`."""

    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Surrogate bundle not found: {bundle_path}")
    target_device = resolve_jax_device(device)
    with np.load(bundle_path, allow_pickle=False) as npz:
        if "metadata_json" not in npz:
            raise ValueError("Surrogate bundle is missing metadata_json.")
        metadata_payload = json.loads(str(np.asarray(npz["metadata_json"]).item()))
        if int(metadata_payload.get("bundle_version", 0)) > SURROGATE_BUNDLE_VERSION:
            raise ValueError(
                "Surrogate bundle was saved by a newer format version "
                f"{metadata_payload.get('bundle_version')}."
            )
        optional_names = set(metadata_payload.get("optional_arrays", []))
        frozen = _frozen_from_npz(npz, metadata_payload, target_device)

        def optional_array(name: str, dtype: Any) -> Optional[np.ndarray]:
            if name not in optional_names:
                return None
            return np.asarray(npz[name], dtype=dtype).reshape(-1)

        return SurrogateBundle(
            path=str(bundle_path),
            frozen=frozen,
            metadata=dict(metadata_payload.get("metadata", {})),
            validation_rmse=optional_array("validation_rmse", np.float64),
            validation_rmse_residual=optional_array("validation_rmse_residual", np.float64),
            validation_rmse_rom=optional_array("validation_rmse_rom", np.float64),
            validation_improvement=optional_array("validation_improvement", np.float64),
            train_idx=optional_array("train_idx", np.int64),
            val_idx=optional_array("val_idx", np.int64),
            train_theta_ids=optional_array("train_theta_ids", np.int64),
            val_theta_ids=optional_array("val_theta_ids", np.int64),
        )


def _output_index_array(output_indices: Optional[Sequence[int] | np.ndarray], d_out: int) -> Optional[np.ndarray]:
    if output_indices is None:
        return None
    idx = np.asarray(output_indices, dtype=np.int64).reshape(-1)
    if idx.size < 1:
        raise ValueError("output_indices must contain at least one output row when provided.")
    if np.unique(idx).size != idx.size:
        raise ValueError("output_indices must not contain duplicates.")
    if np.any(idx < 0) or np.any(idx >= d_out):
        raise ValueError(f"output_indices must be zero-based rows in [0, {d_out}), got {idx}.")
    return idx


def _validate_sample_weights(sample_weights: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_samples:
        raise ValueError(f"sample_weights length mismatch: {weights.shape[0]} vs {n_samples}.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or not np.sum(weights) > 0.0:
        raise ValueError("sample_weights must be finite, nonnegative, and have positive sum.")
    return weights


def _rmse_per_dim(error: np.ndarray) -> np.ndarray:
    if error.shape[1] == 0:
        return np.zeros((error.shape[0],), dtype=np.float64)
    return np.sqrt(np.mean(error**2, axis=1))


def _improvement_vs_baseline(model_rmse: np.ndarray, baseline_rmse: np.ndarray) -> np.ndarray:
    improvement = np.full(model_rmse.shape, np.nan, dtype=np.float64)
    nonzero = baseline_rmse > np.sqrt(np.finfo(np.float64).eps)
    improvement[nonzero] = 1.0 - model_rmse[nonzero] / baseline_rmse[nonzero]
    return improvement


def train_surrogate_from_dataset(
    dataset: SurrogateDataset,
    *,
    architecture: str = "resnet",
    rom_residual: bool = False,
    output_indices: Optional[Sequence[int] | np.ndarray] = None,
    validation_fraction: float = 0.10,
    split_by_theta: bool = False,
    only_full_success: bool = False,
    seed: int = 1,
    sample_weights: Optional[Any] = None,
    d_hidden: int = 128,
    d_hidden2: Optional[int] = 64,
    n_blocks: int = 3,
    nepoch: Optional[int] = None,
    eta_init: float = 1e-3,
    batch_size: Optional[int] = None,
    weight_decay: float = 1e-5,
    clip_norm: float = 5.0,
    activation: str = "silu",
    device: Optional[Any] = None,
) -> SurrogateTrainingResult:
    """Train an MLP or ResNet surrogate from a `SurrogateDataset`.

    If `rom_residual=True`, training targets are `dataset.Y - dataset.Y_rom` and
    validation reports both residual RMSE and reconstructed full-output RMSE.
    """

    arch = str(architecture).strip().lower()
    if arch not in {"mlp", "resnet"}:
        raise ValueError(f"architecture must be 'mlp' or 'resnet', got {architecture!r}.")
    if rom_residual and dataset.Y_rom is None:
        raise ValueError("rom_residual=True requires dataset.Y_rom.")

    out_idx = _output_index_array(output_indices, dataset.Y.shape[0])
    Y_full = dataset.Y if out_idx is None else dataset.Y[out_idx, :]
    Y_rom = None if dataset.Y_rom is None else (dataset.Y_rom if out_idx is None else dataset.Y_rom[out_idx, :])
    Y_target = Y_full - Y_rom if rom_residual else Y_full
    if not np.isfinite(Y_target).all():
        raise ValueError("Training targets contain non-finite values after preprocessing.")

    split = split_surrogate_dataset(
        dataset,
        validation_fraction=validation_fraction,
        split_by_theta=split_by_theta,
        only_full_success=only_full_success,
        seed=seed,
    )
    weights = _validate_sample_weights(sample_weights, dataset.n_samples)
    train_weights = None if weights is None else weights[split.train_idx]
    epochs = int(nepoch) if nepoch is not None else (600 if arch == "resnet" else 400)
    target_device = resolve_jax_device(device)

    X_train = dataset.X[:, split.train_idx]
    Y_train = Y_target[:, split.train_idx]
    if arch == "resnet":
        d_theta = int(dataset.theta.shape[0])
        if d_theta <= 0:
            raise ValueError("ResNet architecture requires at least one theta parameter.")
        frozen: FrozenMLP | FrozenResNet = train_resnet(
            X_train,
            Y_train,
            d_theta=d_theta,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            nepoch=epochs,
            eta_init=eta_init,
            batch_size=batch_size,
            seed=seed,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            sample_weights=train_weights,
            device=target_device,
        )
    else:
        frozen = train_mlp(
            X_train,
            Y_train,
            d_hidden=d_hidden,
            d_hidden2=d_hidden2,
            nepoch=epochs,
            eta_init=eta_init,
            batch_size=batch_size,
            seed=seed,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            activation=activation,
            sample_weights=train_weights,
            device=target_device,
        )

    validation: Optional[SurrogateValidationResult] = None
    validation_rmse: Optional[np.ndarray] = None
    validation_rmse_residual: Optional[np.ndarray] = None
    validation_rmse_rom: Optional[np.ndarray] = None
    validation_improvement: Optional[np.ndarray] = None
    if split.val_idx.size:
        X_val = dataset.X[:, split.val_idx]
        Y_val_target = Y_target[:, split.val_idx]
        Y_pred_target = np.asarray(predict_frozen_batch(frozen, X_val), dtype=np.float64)
        validation_rmse_residual = _rmse_per_dim(Y_pred_target - Y_val_target)
        if rom_residual:
            assert Y_rom is not None
            Y_rom_val = Y_rom[:, split.val_idx]
            Y_full_val = Y_full[:, split.val_idx]
            validation_rmse = _rmse_per_dim(Y_pred_target + Y_rom_val - Y_full_val)
            validation_rmse_rom = _rmse_per_dim(Y_rom_val - Y_full_val)
            validation_improvement = _improvement_vs_baseline(validation_rmse, validation_rmse_rom)
            validation = validate_surrogate(frozen, X_val, Y_val_target, Y_rom=np.zeros_like(Y_val_target))
        else:
            validation_rmse = validation_rmse_residual
            if Y_rom is None:
                validation = validate_surrogate(frozen, X_val, Y_val_target)
            else:
                Y_rom_val = Y_rom[:, split.val_idx]
                Y_full_val = Y_full[:, split.val_idx]
                validation_rmse_rom = _rmse_per_dim(Y_rom_val - Y_full_val)
                validation_improvement = _improvement_vs_baseline(validation_rmse, validation_rmse_rom)
                validation = validate_surrogate(frozen, X_val, Y_val_target, Y_rom=Y_rom_val)

    metadata: dict[str, object] = {
        "architecture": arch,
        "target_mode": dataset.target_mode,
        "rom_residual": bool(rom_residual),
        "validation_split": "held-out parameter vectors" if split_by_theta else "random samples",
        "train_size": split.train_size,
        "val_size": split.val_size,
        "jax_backend": jax.default_backend(),
        "jax_device": None if target_device is None else str(target_device),
        "jax_device_platform": None if target_device is None else str(target_device.platform),
        "theta_names": dataset.theta_names,
        "output_indices": None if out_idx is None else out_idx.copy(),
    }
    if arch == "resnet":
        metadata["d_theta"] = int(dataset.theta.shape[0])
        metadata["n_blocks"] = int(n_blocks)
    return SurrogateTrainingResult(
        frozen=frozen,
        architecture=arch,
        split=split,
        validation=validation,
        validation_rmse=validation_rmse,
        validation_rmse_residual=validation_rmse_residual,
        validation_rmse_rom=validation_rmse_rom,
        validation_improvement=validation_improvement,
        target_is_residual=bool(rom_residual),
        output_indices=out_idx,
        metadata=metadata,
    )


def train_surrogate_from_batched_arrays_jax(
    arrays: BatchedSurrogateRolloutArrays,
    *,
    architecture: str = "resnet",
    rom_residual: bool = False,
    output_indices: Optional[Sequence[int] | np.ndarray] = None,
    only_full_success: bool = False,
    seed: int = 1,
    sample_weights: Optional[Any] = None,
    d_hidden: int = 128,
    d_hidden2: Optional[int] = 64,
    n_blocks: int = 3,
    nepoch: Optional[int] = None,
    eta_init: float = 1e-3,
    batch_size: Optional[int] = None,
    weight_decay: float = 1e-5,
    clip_norm: float = 5.0,
    activation: str = "silu",
    device: Optional[Any] = None,
) -> SurrogateTrainingResult:
    """Train directly from fixed-shape JAX rollout arrays.

    Failed SEP branches are represented by ``arrays.sample_mask=False`` and are
    assigned zero weight. This avoids variable-size compaction before training
    while preserving the same MLP/ResNet trainer and device-placement controls.
    Validation is intentionally disabled for this fixed-shape path; compact to
    ``SurrogateDataset`` first when held-out validation diagnostics are needed.
    """

    arch = str(architecture).strip().lower()
    if arch not in {"mlp", "resnet"}:
        raise ValueError(f"architecture must be 'mlp' or 'resnet', got {architecture!r}.")

    n_samples_total = int(arrays.X.shape[1])
    if int(arrays.Y.shape[1]) != n_samples_total or int(arrays.Y_rom.shape[1]) != n_samples_total:
        raise ValueError("Batched rollout X/Y/Y_rom arrays must have the same sample count.")
    if int(arrays.sample_mask.shape[0]) != n_samples_total:
        raise ValueError("Batched rollout sample_mask must have one entry per sample.")
    if int(arrays.theta_ids.shape[0]) != n_samples_total or int(arrays.period_ids.shape[0]) != n_samples_total:
        raise ValueError("Batched rollout theta_ids and period_ids must have one entry per sample.")

    out_idx = _output_index_array(output_indices, int(arrays.Y.shape[0]))
    y_full = arrays.Y if out_idx is None else jnp.take(arrays.Y, jnp.asarray(out_idx, dtype=jnp.int32), axis=0)
    y_rom = arrays.Y_rom if out_idx is None else jnp.take(arrays.Y_rom, jnp.asarray(out_idx, dtype=jnp.int32), axis=0)
    y_target = y_full - y_rom if rom_residual else y_full

    mask = np.asarray(arrays.sample_mask, dtype=bool).reshape(-1)
    theta_ids = np.asarray(arrays.theta_ids, dtype=np.int64).reshape(-1)
    if only_full_success:
        theta_success = np.asarray(arrays.theta_success, dtype=bool).reshape(-1)
        if theta_success.shape[0] != int(arrays.theta.shape[1]):
            raise ValueError("Batched rollout theta_success must have one entry per theta draw.")
        mask &= theta_success[theta_ids]
    weights = mask.astype(np.float64)
    if sample_weights is not None:
        weights *= _validate_sample_weights(sample_weights, n_samples_total)
    if not np.sum(weights) > 0.0:
        raise ValueError("No positive-weight samples remain after applying the batched rollout mask.")

    epochs = int(nepoch) if nepoch is not None else (600 if arch == "resnet" else 400)
    target_device = resolve_jax_device(device)
    if arch == "resnet":
        d_theta = int(arrays.theta.shape[0])
        if d_theta <= 0:
            raise ValueError("ResNet architecture requires at least one theta parameter.")
        frozen: FrozenMLP | FrozenResNet = train_resnet(
            arrays.X,
            y_target,
            d_theta=d_theta,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            nepoch=epochs,
            eta_init=eta_init,
            batch_size=batch_size,
            seed=seed,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            sample_weights=weights,
            device=target_device,
        )
    else:
        frozen = train_mlp(
            arrays.X,
            y_target,
            d_hidden=d_hidden,
            d_hidden2=d_hidden2,
            nepoch=epochs,
            eta_init=eta_init,
            batch_size=batch_size,
            seed=seed,
            weight_decay=weight_decay,
            clip_norm=clip_norm,
            activation=activation,
            sample_weights=weights,
            device=target_device,
        )

    train_idx = np.flatnonzero(weights > 0.0).astype(np.int64)
    split = SurrogateTrainValidationSplit(
        train_idx=train_idx,
        val_idx=np.zeros((0,), dtype=np.int64),
        split_by_theta=False,
        validation_fraction=0.0,
        only_full_success=only_full_success,
        train_theta_ids=np.unique(theta_ids[train_idx]),
        val_theta_ids=np.zeros((0,), dtype=np.int64),
    )
    metadata: dict[str, object] = {
        "architecture": arch,
        "target_mode": "batched_jax",
        "rom_residual": bool(rom_residual),
        "validation_split": "disabled for fixed-shape batched arrays",
        "train_size": split.train_size,
        "val_size": 0,
        "n_samples_total": n_samples_total,
        "masked_sample_count": int(np.count_nonzero(weights <= 0.0)),
        "jax_backend": jax.default_backend(),
        "jax_device": None if target_device is None else str(target_device),
        "jax_device_platform": None if target_device is None else str(target_device.platform),
        "output_indices": None if out_idx is None else out_idx.copy(),
    }
    if arch == "resnet":
        metadata["d_theta"] = int(arrays.theta.shape[0])
        metadata["n_blocks"] = int(n_blocks)

    return SurrogateTrainingResult(
        frozen=frozen,
        architecture=arch,
        split=split,
        validation=None,
        validation_rmse=None,
        validation_rmse_residual=None,
        validation_rmse_rom=None,
        validation_improvement=None,
        target_is_residual=bool(rom_residual),
        output_indices=out_idx,
        metadata=metadata,
    )


def fit_surrogate_pipeline(
    rom_predict: PredictTupleFn,
    fom_predict: PredictTupleFn,
    initial_state: Any,
    shocks: Any,
    theta_design: Any,
    *,
    target_mode: str = "fom_obs",
    samples_per_theta: Optional[int] = None,
    sample_replace: bool = True,
    dataset_seed: int = 0,
    min_stable_periods: int = 1,
    input_names: Sequence[str] = (),
    output_names: Sequence[str] = (),
    architecture: str = "resnet",
    rom_residual: bool = True,
    output_indices: Optional[Sequence[int] | np.ndarray] = None,
    validation_fraction: float = 0.10,
    split_by_theta: bool = False,
    only_full_success: bool = False,
    train_seed: int = 1,
    sample_weights: Optional[Any] = None,
    d_hidden: int = 128,
    d_hidden2: Optional[int] = 64,
    n_blocks: int = 3,
    nepoch: Optional[int] = None,
    eta_init: float = 1e-3,
    batch_size: Optional[int] = None,
    weight_decay: float = 1e-5,
    clip_norm: float = 5.0,
    activation: str = "silu",
    device: Optional[Any] = None,
    bundle_path: Optional[str | Path] = None,
    bundle_metadata: Optional[dict[str, object]] = None,
) -> SurrogatePipelineResult:
    """Build a ROM/FOM dataset, train a surrogate, and optionally save it.

    This is the supervised-learning counterpart to the SEP/switching pipeline:
    the expensive FOM callback supplies targets, the ROM callback supplies the
    baseline path, and the resulting frozen JAX surrogate can be placed on CPU
    or GPU through `device`.
    """

    dataset = build_surrogate_residual_dataset(
        rom_predict,
        fom_predict,
        initial_state,
        shocks,
        theta_design,
        target_mode=target_mode,
        samples_per_theta=samples_per_theta,
        sample_replace=sample_replace,
        seed=dataset_seed,
        min_stable_periods=min_stable_periods,
        input_names=input_names,
        output_names=output_names,
    )
    dataset_summary = summarize_surrogate_dataset(dataset)
    training = train_surrogate_from_dataset(
        dataset,
        architecture=architecture,
        rom_residual=rom_residual,
        output_indices=output_indices,
        validation_fraction=validation_fraction,
        split_by_theta=split_by_theta,
        only_full_success=only_full_success,
        seed=train_seed,
        sample_weights=sample_weights,
        d_hidden=d_hidden,
        d_hidden2=d_hidden2,
        n_blocks=n_blocks,
        nepoch=nepoch,
        eta_init=eta_init,
        batch_size=batch_size,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        activation=activation,
        device=device,
    )
    saved_path: Optional[Path] = None
    if bundle_path is not None:
        metadata = {
            "pipeline": "fit_surrogate_pipeline",
            "dataset_summary": dataset_summary,
        }
        if bundle_metadata is not None:
            metadata.update(bundle_metadata)
        saved_path = save_surrogate_bundle(bundle_path, training, metadata=metadata)
    return SurrogatePipelineResult(
        dataset=dataset,
        dataset_summary=dataset_summary,
        training=training,
        bundle_path=saved_path,
    )
