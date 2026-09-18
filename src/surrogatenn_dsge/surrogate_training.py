from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import jax

from .surrogate import (
    FrozenMLP,
    FrozenResNet,
    SurrogateValidationResult,
    predict_frozen_batch,
    resolve_jax_device,
    train_mlp,
    train_resnet,
    validate_surrogate,
)
from .surrogate_dataset import SurrogateDataset


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
