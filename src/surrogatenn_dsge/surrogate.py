from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .regime_switching_api import (
    additive_residual_loglik_per_period,
    inversion_loglik_per_period,
    predict_additive_residual,
    predict_additive_residual_ood,
)


ArrayLike = Union[Sequence[float], np.ndarray, jax.Array]


def _as_jax_array(values: Any, *, label: str, ndim: Optional[int] = None) -> jax.Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}, got shape {array.shape}.")
    if array.size and not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{label} contains non-finite values.")
    return array


def _as_numpy_matrix(values: Any, *, label: str, copy: bool) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=copy)
    if array.ndim != 2:
        raise ValueError(f"{label} must be rank-2 with shape (dim, samples), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def _safe_std(std_values: np.ndarray) -> np.ndarray:
    out = np.asarray(std_values, dtype=np.float64).copy()
    bad = (~np.isfinite(out)) | (out <= math.sqrt(np.finfo(np.float64).eps))
    out[bad] = 1.0
    return out


def _linear(weight: jax.Array, bias: jax.Array, x: jax.Array) -> jax.Array:
    value = weight @ x
    if value.ndim == 1:
        return value + bias
    return value + bias[:, None]


def silu(x: Any) -> jax.Array:
    values = jnp.asarray(x, dtype=jnp.float64)
    return values / (1.0 + jnp.exp(-values))


def _activation_fn(name: str) -> Callable[[jax.Array], jax.Array]:
    activation = str(name).lower()
    if activation == "tanh":
        return jnp.tanh
    if activation == "silu":
        return silu
    raise ValueError(f"Unsupported activation {name!r}. Use 'tanh' or 'silu'.")


@dataclass(frozen=True)
class NormStats:
    """Training-set normalization statistics for surrogate inputs and outputs."""

    mu_x: jax.Array
    sigma_x: jax.Array
    mu_y: jax.Array
    sigma_y: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu_x", _as_jax_array(self.mu_x, label="mu_x", ndim=1))
        object.__setattr__(self, "sigma_x", _as_jax_array(self.sigma_x, label="sigma_x", ndim=1))
        object.__setattr__(self, "mu_y", _as_jax_array(self.mu_y, label="mu_y", ndim=1))
        object.__setattr__(self, "sigma_y", _as_jax_array(self.sigma_y, label="sigma_y", ndim=1))
        if self.mu_x.shape != self.sigma_x.shape:
            raise ValueError(f"Input mean/std shape mismatch: {self.mu_x.shape} vs {self.sigma_x.shape}.")
        if self.mu_y.shape != self.sigma_y.shape:
            raise ValueError(f"Output mean/std shape mismatch: {self.mu_y.shape} vs {self.sigma_y.shape}.")
        if bool(jnp.any(self.sigma_x <= 0.0)) or bool(jnp.any(self.sigma_y <= 0.0)):
            raise ValueError("Normalization standard deviations must be strictly positive.")

    @property
    def input_dim(self) -> int:
        return int(self.mu_x.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.mu_y.shape[0])

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NormStats":
        return cls(
            mu_x=_mapping_value(payload, ("mu_x", "muX", "x_mean", "input_mean", "mean", "mu", "μX")),
            sigma_x=_mapping_value(payload, ("sigma_x", "sigmaX", "x_std", "input_std", "std", "sigma", "σX")),
            mu_y=_mapping_value(payload, ("mu_y", "muY", "y_mean", "output_mean", "μY")),
            sigma_y=_mapping_value(payload, ("sigma_y", "sigmaY", "y_std", "output_std", "σY")),
        )

    def as_julia_dict(self) -> dict[str, np.ndarray]:
        return {
            "muX": np.asarray(self.mu_x),
            "sigmaX": np.asarray(self.sigma_x),
            "muY": np.asarray(self.mu_y),
            "sigmaY": np.asarray(self.sigma_y),
            "μX": np.asarray(self.mu_x),
            "σX": np.asarray(self.sigma_x),
            "μY": np.asarray(self.mu_y),
            "σY": np.asarray(self.sigma_y),
        }


def _mapping_value(payload: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise ValueError(f"Mapping is missing one of: {', '.join(names)}.")


@dataclass(frozen=True)
class FrozenMLP:
    """Frozen Julia-compatible MLP surrogate.

    The shape convention follows `hlt_sep_surrogate_nn_utils.jl`: inputs and
    batched inputs are column-major feature arrays, with shape `(d_in,)` or
    `(d_in, batch)`.
    """

    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    W3: Optional[jax.Array]
    b3: Optional[jax.Array]
    norm: NormStats
    d_in: int
    d_out: int
    activation: str = "tanh"

    def __post_init__(self) -> None:
        object.__setattr__(self, "W1", _as_jax_array(self.W1, label="W1", ndim=2))
        object.__setattr__(self, "b1", _as_jax_array(self.b1, label="b1", ndim=1))
        object.__setattr__(self, "W2", _as_jax_array(self.W2, label="W2", ndim=2))
        object.__setattr__(self, "b2", _as_jax_array(self.b2, label="b2", ndim=1))
        if self.W3 is None:
            if self.b3 is not None:
                raise ValueError("b3 must be None when W3 is None.")
        else:
            if self.b3 is None:
                raise ValueError("b3 must be provided when W3 is provided.")
            object.__setattr__(self, "W3", _as_jax_array(self.W3, label="W3", ndim=2))
            object.__setattr__(self, "b3", _as_jax_array(self.b3, label="b3", ndim=1))
        object.__setattr__(self, "d_in", int(self.d_in))
        object.__setattr__(self, "d_out", int(self.d_out))
        activation = str(self.activation).lower()
        _activation_fn(activation)
        object.__setattr__(self, "activation", activation)
        if self.norm.input_dim != self.d_in:
            raise ValueError(f"NormStats input dimension {self.norm.input_dim} does not match d_in={self.d_in}.")
        if self.norm.output_dim != self.d_out:
            raise ValueError(f"NormStats output dimension {self.norm.output_dim} does not match d_out={self.d_out}.")
        if self.W1.shape[1] != self.d_in or self.W1.shape[0] != self.b1.shape[0]:
            raise ValueError("W1/b1 dimensions are inconsistent with d_in.")
        if self.W3 is None:
            if self.W2.shape != (self.d_out, self.W1.shape[0]) or self.b2.shape[0] != self.d_out:
                raise ValueError("One-hidden-layer W2/b2 dimensions are inconsistent.")
        else:
            assert self.W3 is not None and self.b3 is not None
            if self.W2.shape[1] != self.W1.shape[0] or self.W2.shape[0] != self.b2.shape[0]:
                raise ValueError("Two-hidden-layer W2/b2 dimensions are inconsistent.")
            if self.W3.shape != (self.d_out, self.W2.shape[0]) or self.b3.shape[0] != self.d_out:
                raise ValueError("Two-hidden-layer W3/b3 dimensions are inconsistent.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenMLP":
        norm_payload = _mapping_value(payload, ("norm", "normalization", "norm_stats"))
        norm = norm_payload if isinstance(norm_payload, NormStats) else NormStats.from_mapping(norm_payload)
        return cls(
            W1=_mapping_value(payload, ("W1",)),
            b1=_mapping_value(payload, ("b1",)),
            W2=_mapping_value(payload, ("W2",)),
            b2=_mapping_value(payload, ("b2",)),
            W3=payload.get("W3"),
            b3=payload.get("b3"),
            norm=norm,
            d_in=int(payload.get("d_in", norm.input_dim)),
            d_out=int(payload.get("d_out", norm.output_dim)),
            activation=str(payload.get("activation", "tanh")),
        )


@dataclass(frozen=True)
class ResBlock:
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "W1", _as_jax_array(self.W1, label="block.W1", ndim=2))
        object.__setattr__(self, "b1", _as_jax_array(self.b1, label="block.b1", ndim=1))
        object.__setattr__(self, "W2", _as_jax_array(self.W2, label="block.W2", ndim=2))
        object.__setattr__(self, "b2", _as_jax_array(self.b2, label="block.b2", ndim=1))
        if self.W1.shape[0] != self.b1.shape[0] or self.W1.shape[1] != self.W2.shape[1]:
            raise ValueError("Residual block W1/W2 dimensions are inconsistent.")
        if self.W2.shape[0] != self.b2.shape[0] or self.W2.shape[0] != self.W1.shape[1]:
            raise ValueError("Residual block output dimensions are inconsistent.")


@dataclass(frozen=True)
class FrozenResNet:
    W_embed: jax.Array
    b_embed: jax.Array
    d_theta: int
    W_gamma: jax.Array
    b_gamma: jax.Array
    W_beta: jax.Array
    b_beta: jax.Array
    blocks: tuple[ResBlock, ...]
    W_out: jax.Array
    b_out: jax.Array
    norm: NormStats
    d_in: int
    d_out: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "W_embed", _as_jax_array(self.W_embed, label="W_embed", ndim=2))
        object.__setattr__(self, "b_embed", _as_jax_array(self.b_embed, label="b_embed", ndim=1))
        object.__setattr__(self, "W_gamma", _as_jax_array(self.W_gamma, label="W_gamma", ndim=2))
        object.__setattr__(self, "b_gamma", _as_jax_array(self.b_gamma, label="b_gamma", ndim=1))
        object.__setattr__(self, "W_beta", _as_jax_array(self.W_beta, label="W_beta", ndim=2))
        object.__setattr__(self, "b_beta", _as_jax_array(self.b_beta, label="b_beta", ndim=1))
        object.__setattr__(self, "W_out", _as_jax_array(self.W_out, label="W_out", ndim=2))
        object.__setattr__(self, "b_out", _as_jax_array(self.b_out, label="b_out", ndim=1))
        object.__setattr__(self, "d_theta", int(self.d_theta))
        object.__setattr__(self, "d_in", int(self.d_in))
        object.__setattr__(self, "d_out", int(self.d_out))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.d_theta <= 0 or self.d_theta >= self.d_in:
            raise ValueError("FrozenResNet requires 0 < d_theta < d_in.")
        d_state_shock = self.d_in - self.d_theta
        hidden = self.W_embed.shape[0]
        if self.W_embed.shape[1] != d_state_shock or self.b_embed.shape[0] != hidden:
            raise ValueError("Embedding dimensions are inconsistent.")
        expected_film = (hidden, self.d_theta)
        if self.W_gamma.shape != expected_film or self.W_beta.shape != expected_film:
            raise ValueError("FiLM gamma/beta matrices have inconsistent dimensions.")
        if self.b_gamma.shape[0] != hidden or self.b_beta.shape[0] != hidden:
            raise ValueError("FiLM gamma/beta biases have inconsistent dimensions.")
        for block in self.blocks:
            if block.W1.shape != (hidden, hidden) or block.W2.shape != (hidden, hidden):
                raise ValueError("All residual blocks must be square hidden-dimensional blocks.")
        if self.W_out.shape != (self.d_out, hidden) or self.b_out.shape[0] != self.d_out:
            raise ValueError("Output projection dimensions are inconsistent.")
        if self.norm.input_dim != self.d_in or self.norm.output_dim != self.d_out:
            raise ValueError("NormStats dimensions are inconsistent with FrozenResNet.")


@dataclass(frozen=True)
class SurrogateValidationResult:
    rmse_per_dim: np.ndarray
    rmse_total: float
    max_abs_error: float
    ood_fraction: float
    improvement_vs_rom: Optional[np.ndarray]
    n_samples: int


def standardize_xy(
    X: Any,
    Y: Any,
    *,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray, NormStats]:
    X_arr = _as_numpy_matrix(X, label="X", copy=copy)
    Y_arr = _as_numpy_matrix(Y, label="Y", copy=copy)
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(f"X/Y sample count mismatch: {X_arr.shape[1]} vs {Y_arr.shape[1]}.")
    mu_x = np.mean(X_arr, axis=1)
    sigma_x = _safe_std(np.std(X_arr, axis=1, ddof=0))
    X_arr[...] = (X_arr - mu_x[:, None]) / sigma_x[:, None]
    mu_y = np.mean(Y_arr, axis=1)
    sigma_y = _safe_std(np.std(Y_arr, axis=1, ddof=0))
    Y_arr[...] = (Y_arr - mu_y[:, None]) / sigma_y[:, None]
    return X_arr, Y_arr, NormStats(mu_x, sigma_x, mu_y, sigma_y)


def mlp_forward(
    W1: ArrayLike,
    b1: ArrayLike,
    W2: ArrayLike,
    b2: ArrayLike,
    W3: Optional[ArrayLike],
    b3: Optional[ArrayLike],
    x: ArrayLike,
    *,
    activation: str = "tanh",
) -> jax.Array:
    act = _activation_fn(activation)
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    h1 = act(_linear(jnp.asarray(W1, dtype=jnp.float64), jnp.asarray(b1, dtype=jnp.float64), x_arr))
    if W3 is None:
        return _linear(jnp.asarray(W2, dtype=jnp.float64), jnp.asarray(b2, dtype=jnp.float64), h1)
    if b3 is None:
        raise ValueError("b3 must be provided when W3 is provided.")
    h2 = act(_linear(jnp.asarray(W2, dtype=jnp.float64), jnp.asarray(b2, dtype=jnp.float64), h1))
    return _linear(jnp.asarray(W3, dtype=jnp.float64), jnp.asarray(b3, dtype=jnp.float64), h2)


def resnet_forward(net: FrozenResNet, x: ArrayLike) -> jax.Array:
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    d_state_shock = net.d_in - net.d_theta
    if x_arr.ndim == 1:
        x_state_shock = x_arr[:d_state_shock]
        x_theta = x_arr[d_state_shock:]
    elif x_arr.ndim == 2:
        x_state_shock = x_arr[:d_state_shock, :]
        x_theta = x_arr[d_state_shock:, :]
    else:
        raise ValueError(f"x must be rank-1 or rank-2, got shape {x_arr.shape}.")
    z = silu(_linear(net.W_embed, net.b_embed, x_state_shock))
    gamma = _linear(net.W_gamma, net.b_gamma, x_theta)
    beta = _linear(net.W_beta, net.b_beta, x_theta)
    z = gamma * z + beta
    for block in net.blocks:
        z = z + _linear(block.W2, block.b2, silu(_linear(block.W1, block.b1, z)))
    return _linear(net.W_out, net.b_out, z)


def predict_frozen(frozen: FrozenMLP | FrozenResNet, x: ArrayLike) -> jax.Array:
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    if x_arr.shape[0] != frozen.d_in:
        raise ValueError(f"Input length mismatch: got {x_arr.shape[0]}, expected {frozen.d_in}.")
    x_norm = (x_arr - frozen.norm.mu_x) / frozen.norm.sigma_x
    if isinstance(frozen, FrozenMLP):
        y_norm = mlp_forward(
            frozen.W1,
            frozen.b1,
            frozen.W2,
            frozen.b2,
            frozen.W3,
            frozen.b3,
            x_norm,
            activation=frozen.activation,
        )
    else:
        y_norm = resnet_forward(frozen, x_norm)
    return frozen.norm.mu_y + frozen.norm.sigma_y * y_norm


def predict_frozen_batch(frozen: FrozenMLP | FrozenResNet, X: ArrayLike) -> jax.Array:
    X_arr = jnp.asarray(X, dtype=jnp.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be rank-2 with shape (d_in, batch), got {X_arr.shape}.")
    if X_arr.shape[0] != frozen.d_in:
        raise ValueError(f"Input row mismatch: got {X_arr.shape[0]}, expected {frozen.d_in}.")
    X_norm = (X_arr - frozen.norm.mu_x[:, None]) / frozen.norm.sigma_x[:, None]
    if isinstance(frozen, FrozenMLP):
        Y_norm = mlp_forward(
            frozen.W1,
            frozen.b1,
            frozen.W2,
            frozen.b2,
            frozen.W3,
            frozen.b3,
            X_norm,
            activation=frozen.activation,
        )
    else:
        Y_norm = resnet_forward(frozen, X_norm)
    return frozen.norm.mu_y[:, None] + frozen.norm.sigma_y[:, None] * Y_norm


def predict_frozen_safe(
    frozen: FrozenMLP | FrozenResNet,
    x: ArrayLike,
    *,
    z_threshold: float = 4.0,
) -> tuple[jax.Array, bool, float]:
    threshold = float(z_threshold)
    if threshold <= 0.0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}.")
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    z_scores = jnp.abs((x_arr - frozen.norm.mu_x) / frozen.norm.sigma_x)
    max_z = float(np.asarray(jnp.max(z_scores)))
    return predict_frozen(frozen, x_arr), max_z > threshold, max_z


def compute_ood_flag(
    norm: NormStats,
    state: ArrayLike,
    shock: ArrayLike,
    theta: ArrayLike,
    *,
    z_threshold: float = 4.0,
) -> bool:
    threshold = float(z_threshold)
    if threshold <= 0.0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}.")
    x = jnp.concatenate(
        [
            jnp.asarray(state, dtype=jnp.float64).reshape(-1),
            jnp.asarray(shock, dtype=jnp.float64).reshape(-1),
            jnp.asarray(theta, dtype=jnp.float64).reshape(-1),
        ],
        axis=0,
    )
    if x.shape[0] != norm.input_dim:
        raise ValueError(f"Input length mismatch: got {x.shape[0]}, expected {norm.input_dim}.")
    max_z = float(np.asarray(jnp.max(jnp.abs((x - norm.mu_x) / norm.sigma_x))))
    return max_z > threshold


def weighted_mse(y_hat: ArrayLike, y: ArrayLike, weights: ArrayLike) -> jax.Array:
    y_hat_arr = jnp.asarray(y_hat, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)
    if y_hat_arr.shape != y_arr.shape:
        raise ValueError(f"y_hat/y shape mismatch: {y_hat_arr.shape} vs {y_arr.shape}.")
    if y_hat_arr.ndim != 2:
        raise ValueError(f"y_hat and y must be rank-2, got {y_hat_arr.shape}.")
    if w.shape[0] != y_hat_arr.shape[1]:
        raise ValueError(f"weights length mismatch: {w.shape[0]} vs samples {y_hat_arr.shape[1]}.")
    if bool(jnp.any(w < 0.0)):
        raise ValueError("weights must be nonnegative.")
    total_w = jnp.sum(w)
    if not bool(total_w > 0.0):
        raise ValueError("weights must have positive sum.")
    diff_sq = jnp.sum((y_hat_arr - y_arr) ** 2, axis=0)
    return jnp.sum(diff_sq * w) / total_w


def validate_surrogate(
    frozen: FrozenMLP | FrozenResNet,
    X_val: ArrayLike,
    Y_val: ArrayLike,
    *,
    Y_rom: Optional[ArrayLike] = None,
    z_threshold: float = 4.0,
) -> SurrogateValidationResult:
    X = np.asarray(X_val, dtype=np.float64)
    Y = np.asarray(Y_val, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X_val and Y_val must be rank-2 matrices.")
    if X.shape[1] == 0:
        raise ValueError("Empty validation set.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X_val/Y_val sample mismatch: {X.shape[1]} vs {Y.shape[1]}.")
    Y_pred = np.asarray(predict_frozen_batch(frozen, X), dtype=np.float64)
    err = Y_pred - Y
    rmse_per_dim = np.sqrt(np.mean(err**2, axis=1))
    rmse_total = float(np.sqrt(np.mean(err**2)))
    max_abs_error = float(np.max(np.abs(err)))
    z = np.abs((X - np.asarray(frozen.norm.mu_x)[:, None]) / np.asarray(frozen.norm.sigma_x)[:, None])
    ood_fraction = float(np.mean(np.max(z, axis=0) > float(z_threshold)))

    improvement: Optional[np.ndarray] = None
    if Y_rom is not None:
        rom = np.asarray(Y_rom, dtype=np.float64)
        if rom.shape != Y.shape:
            raise ValueError(f"Y_rom shape mismatch: {rom.shape} vs {Y.shape}.")
        rom_rmse = np.sqrt(np.mean((rom - Y) ** 2, axis=1))
        improvement = np.full((Y.shape[0],), np.nan, dtype=np.float64)
        nonzero = rom_rmse > math.sqrt(np.finfo(np.float64).eps)
        improvement[nonzero] = 1.0 - rmse_per_dim[nonzero] / rom_rmse[nonzero]

    return SurrogateValidationResult(
        rmse_per_dim=rmse_per_dim,
        rmse_total=rmse_total,
        max_abs_error=max_abs_error,
        ood_fraction=ood_fraction,
        improvement_vs_rom=improvement,
        n_samples=int(X.shape[1]),
    )


def _cosine_schedule_with_warmup(epoch: int, nepoch: int, lr_init: float, warmup_frac: float = 0.1) -> float:
    warmup_epochs = max(1, int(math.floor(nepoch * warmup_frac)))
    if epoch <= warmup_epochs:
        return lr_init * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, nepoch - warmup_epochs)
    return lr_init * 0.5 * (1.0 + math.cos(math.pi * progress))


def _he_init(key: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def resolve_jax_device(device: Optional[Any] = None) -> Optional[jax.Device]:
    """Resolve an optional JAX device selector.

    `None` keeps JAX's default placement. Strings such as `"gpu"` or `"cpu"`
    require that backend to be available; this avoids accidental CPU fallback
    in GPU training runs.
    """

    if device is None:
        return None
    if isinstance(device, str):
        selector = device.strip().lower()
        backend = "gpu" if selector in {"gpu", "cuda"} else selector
        if backend in {"cpu", "gpu", "tpu"}:
            try:
                devices = jax.devices(backend)
            except RuntimeError as exc:
                raise ValueError(f"Requested JAX device backend {device!r} is not available.") from exc
            if not devices:
                raise ValueError(f"Requested JAX device backend {device!r} is not available.")
            return devices[0]
        for available in jax.devices():
            if selector in {str(available).lower(), f"{available.platform}:{available.id}".lower()}:
                return available
        raise ValueError(f"Could not resolve JAX device selector {device!r}.")
    return device


def _device_put(value: Any, device: Optional[jax.Device]) -> Any:
    if device is None:
        return value
    return jax.device_put(value, device)


def _norm_on_device(norm: NormStats, device: Optional[jax.Device]) -> NormStats:
    if device is None:
        return norm
    return NormStats(
        mu_x=_device_put(norm.mu_x, device),
        sigma_x=_device_put(norm.sigma_x, device),
        mu_y=_device_put(norm.mu_y, device),
        sigma_y=_device_put(norm.sigma_y, device),
    )


def train_mlp(
    X: Any,
    Y: Any,
    *,
    d_hidden: int = 128,
    d_hidden2: Optional[int] = 64,
    nepoch: int = 400,
    eta_init: float = 1e-3,
    batch_size: Optional[int] = None,
    seed: int = 1,
    weight_decay: float = 1e-5,
    clip_norm: float = 5.0,
    activation: str = "silu",
    sample_weights: Optional[ArrayLike] = None,
    device: Optional[Any] = None,
) -> FrozenMLP:
    """Train a Julia-compatible frozen MLP with JAX AdamW.

    This intentionally mirrors the Julia utility rather than introducing a
    heavyweight NN dependency. Arrays are shaped `(features, samples)`.
    """

    hidden1 = int(d_hidden)
    hidden2 = None if d_hidden2 is None else int(d_hidden2)
    epochs = int(nepoch)
    if hidden1 <= 0:
        raise ValueError(f"d_hidden must be positive, got {d_hidden}.")
    if hidden2 is not None and hidden2 <= 0:
        raise ValueError(f"d_hidden2 must be positive or None, got {d_hidden2}.")
    if epochs <= 0:
        raise ValueError(f"nepoch must be positive, got {nepoch}.")
    if eta_init <= 0.0:
        raise ValueError(f"eta_init must be positive, got {eta_init}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}.")
    if clip_norm < 0.0:
        raise ValueError(f"clip_norm must be nonnegative, got {clip_norm}.")
    activation = str(activation).lower()
    _activation_fn(activation)

    X_std, Y_std, norm = standardize_xy(X, Y, copy=True)
    d_in, n_samples = X_std.shape
    d_out = Y_std.shape[0]
    if batch_size is None:
        batch = min(512, max(32, n_samples // 20))
    else:
        batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    batch = min(batch, n_samples)

    if sample_weights is None:
        weights_np = np.ones((n_samples,), dtype=np.float64)
    else:
        weights_np = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights_np.shape[0] != n_samples:
            raise ValueError(f"sample_weights length mismatch: {weights_np.shape[0]} vs {n_samples}.")
        if not np.isfinite(weights_np).all() or np.any(weights_np < 0.0) or not np.sum(weights_np) > 0.0:
            raise ValueError("sample_weights must be finite, nonnegative, and have positive sum.")

    target_device = resolve_jax_device(device)
    key = _device_put(jax.random.PRNGKey(int(seed)), target_device)
    keys = jax.random.split(key, 3 if hidden2 is None else 4)
    params: dict[str, jax.Array] = {
        "W1": _he_init(keys[0], (hidden1, d_in), 0.1),
        "b1": jnp.zeros((hidden1,), dtype=jnp.float64),
    }
    if hidden2 is None:
        params["W2"] = _he_init(keys[1], (d_out, hidden1), 0.1)
        params["b2"] = jnp.zeros((d_out,), dtype=jnp.float64)
    else:
        params["W2"] = _he_init(keys[1], (hidden2, hidden1), 0.1)
        params["b2"] = jnp.zeros((hidden2,), dtype=jnp.float64)
        params["W3"] = _he_init(keys[2], (d_out, hidden2), 0.1)
        params["b3"] = jnp.zeros((d_out,), dtype=jnp.float64)

    params = _device_put(params, target_device)
    opt_m = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_v = jax.tree_util.tree_map(jnp.zeros_like, params)
    X_jax = _device_put(jnp.asarray(X_std, dtype=jnp.float64), target_device)
    Y_jax = _device_put(jnp.asarray(Y_std, dtype=jnp.float64), target_device)
    weights_jax = _device_put(jnp.asarray(weights_np, dtype=jnp.float64), target_device)
    act = _activation_fn(activation)

    def forward(params_local: Mapping[str, jax.Array], X_batch: jax.Array) -> jax.Array:
        h1 = act(_linear(params_local["W1"], params_local["b1"], X_batch))
        if "W3" not in params_local:
            return _linear(params_local["W2"], params_local["b2"], h1)
        h2 = act(_linear(params_local["W2"], params_local["b2"], h1))
        return _linear(params_local["W3"], params_local["b3"], h2)

    def loss_fn(params_local: Mapping[str, jax.Array], X_batch: jax.Array, Y_batch: jax.Array, w_batch: jax.Array) -> jax.Array:
        pred = forward(params_local, X_batch)
        diff_sq = jnp.sum((pred - Y_batch) ** 2, axis=0)
        return jnp.sum(diff_sq * w_batch) / jnp.sum(w_batch)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update_step(
        params_local: Mapping[str, jax.Array],
        m_local: Mapping[str, jax.Array],
        v_local: Mapping[str, jax.Array],
        grads: Mapping[str, jax.Array],
        lr: jax.Array,
        step: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
        global_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
        scale = jnp.where((clip_norm > 0.0) & (global_norm > clip_norm), clip_norm / (global_norm + 1e-12), 1.0)
        grads_scaled = jax.tree_util.tree_map(lambda g: g * scale, grads)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_next = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, m_local, grads_scaled)
        v_next = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), v_local, grads_scaled)
        m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1**step), m_next)
        v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2**step), v_next)
        params_next = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
            params_local,
            m_hat,
            v_hat,
        )
        return params_next, m_next, v_next

    rng = np.random.default_rng(int(seed))
    step_count = 0
    for epoch in range(1, epochs + 1):
        lr_value = _cosine_schedule_with_warmup(epoch, epochs, float(eta_init))
        if batch < n_samples:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        for start in range(0, n_samples, batch):
            batch_idx = indices[start : start + batch]
            batch_idx_jax = _device_put(jnp.asarray(batch_idx, dtype=jnp.int64), target_device)
            X_batch = jnp.take(X_jax, batch_idx_jax, axis=1)
            Y_batch = jnp.take(Y_jax, batch_idx_jax, axis=1)
            w_batch = jnp.take(weights_jax, batch_idx_jax, axis=0)
            _, grads = value_and_grad(params, X_batch, Y_batch, w_batch)
            step_count += 1
            params, opt_m, opt_v = update_step(
                params,
                opt_m,
                opt_v,
                grads,
                _device_put(jnp.asarray(lr_value, dtype=jnp.float64), target_device),
                _device_put(jnp.asarray(step_count, dtype=jnp.float64), target_device),
            )

    W3 = params.get("W3")
    b3 = params.get("b3")
    return FrozenMLP(
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        W3=W3,
        b3=b3,
        norm=_norm_on_device(norm, target_device),
        d_in=d_in,
        d_out=d_out,
        activation=activation,
    )


def train_resnet(
    X: Any,
    Y: Any,
    *,
    d_theta: int,
    d_hidden: int = 128,
    n_blocks: int = 3,
    nepoch: int = 600,
    eta_init: float = 1e-3,
    batch_size: Optional[int] = None,
    seed: int = 1,
    weight_decay: float = 1e-5,
    clip_norm: float = 5.0,
    sample_weights: Optional[ArrayLike] = None,
    device: Optional[Any] = None,
) -> FrozenResNet:
    """Train the Julia-style FiLM residual surrogate with JAX AdamW.

    Inputs are shaped `(features, samples)` and must end with `d_theta`
    parameter rows, matching `FrozenResNet` and the HLT residual dataset
    convention `X = [state_t; shock_t; theta]`.
    """

    theta_dim = int(d_theta)
    hidden = int(d_hidden)
    block_count = int(n_blocks)
    epochs = int(nepoch)
    if theta_dim <= 0:
        raise ValueError(f"d_theta must be positive, got {d_theta}.")
    if hidden <= 0:
        raise ValueError(f"d_hidden must be positive, got {d_hidden}.")
    if block_count < 0:
        raise ValueError(f"n_blocks must be nonnegative, got {n_blocks}.")
    if epochs <= 0:
        raise ValueError(f"nepoch must be positive, got {nepoch}.")
    if eta_init <= 0.0:
        raise ValueError(f"eta_init must be positive, got {eta_init}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}.")
    if clip_norm < 0.0:
        raise ValueError(f"clip_norm must be nonnegative, got {clip_norm}.")

    X_std, Y_std, norm = standardize_xy(X, Y, copy=True)
    d_in, n_samples = X_std.shape
    d_out = Y_std.shape[0]
    if theta_dim >= d_in:
        raise ValueError(f"d_theta must be smaller than input dimension {d_in}, got {d_theta}.")
    d_state_shock = d_in - theta_dim

    if batch_size is None:
        batch = min(512, max(32, n_samples // 20))
    else:
        batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    batch = min(batch, n_samples)

    if sample_weights is None:
        weights_np = np.ones((n_samples,), dtype=np.float64)
    else:
        weights_np = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights_np.shape[0] != n_samples:
            raise ValueError(f"sample_weights length mismatch: {weights_np.shape[0]} vs {n_samples}.")
        if not np.isfinite(weights_np).all() or np.any(weights_np < 0.0) or not np.sum(weights_np) > 0.0:
            raise ValueError("sample_weights must be finite, nonnegative, and have positive sum.")

    target_device = resolve_jax_device(device)
    key = _device_put(jax.random.PRNGKey(int(seed)), target_device)
    key_iter = iter(jax.random.split(key, 4 + 2 * block_count))
    params: dict[str, Any] = {
        "W_embed": _he_init(next(key_iter), (hidden, d_state_shock), math.sqrt(2.0 / max(1, d_state_shock))),
        "b_embed": jnp.zeros((hidden,), dtype=jnp.float64),
        "W_gamma": 0.02 * jax.random.normal(next(key_iter), (hidden, theta_dim), dtype=jnp.float64),
        "b_gamma": jnp.ones((hidden,), dtype=jnp.float64),
        "W_beta": 0.02 * jax.random.normal(next(key_iter), (hidden, theta_dim), dtype=jnp.float64),
        "b_beta": jnp.zeros((hidden,), dtype=jnp.float64),
    }
    blocks: list[dict[str, jax.Array]] = []
    hidden_scale = math.sqrt(2.0 / max(1, hidden))
    for _ in range(block_count):
        blocks.append(
            {
                "W1": _he_init(next(key_iter), (hidden, hidden), hidden_scale),
                "b1": jnp.zeros((hidden,), dtype=jnp.float64),
                "W2": 0.01 * jax.random.normal(next(key_iter), (hidden, hidden), dtype=jnp.float64),
                "b2": jnp.zeros((hidden,), dtype=jnp.float64),
            }
        )
    params["blocks"] = tuple(blocks)
    params["W_out"] = 0.01 * jax.random.normal(next(key_iter), (d_out, hidden), dtype=jnp.float64)
    params["b_out"] = jnp.zeros((d_out,), dtype=jnp.float64)

    params = _device_put(params, target_device)
    opt_m = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt_v = jax.tree_util.tree_map(jnp.zeros_like, params)
    X_jax = _device_put(jnp.asarray(X_std, dtype=jnp.float64), target_device)
    Y_jax = _device_put(jnp.asarray(Y_std, dtype=jnp.float64), target_device)
    weights_jax = _device_put(jnp.asarray(weights_np, dtype=jnp.float64), target_device)

    def forward(params_local: Mapping[str, Any], X_batch: jax.Array) -> jax.Array:
        x_state_shock = X_batch[:d_state_shock, :]
        x_theta = X_batch[d_state_shock:, :]
        z = silu(_linear(params_local["W_embed"], params_local["b_embed"], x_state_shock))
        gamma = _linear(params_local["W_gamma"], params_local["b_gamma"], x_theta)
        beta = _linear(params_local["W_beta"], params_local["b_beta"], x_theta)
        z = gamma * z + beta
        for block in params_local["blocks"]:
            z = z + _linear(block["W2"], block["b2"], silu(_linear(block["W1"], block["b1"], z)))
        return _linear(params_local["W_out"], params_local["b_out"], z)

    def loss_fn(params_local: Mapping[str, Any], X_batch: jax.Array, Y_batch: jax.Array, w_batch: jax.Array) -> jax.Array:
        pred = forward(params_local, X_batch)
        diff_sq = jnp.sum((pred - Y_batch) ** 2, axis=0)
        return jnp.sum(diff_sq * w_batch) / jnp.sum(w_batch)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update_step(
        params_local: Mapping[str, Any],
        m_local: Mapping[str, Any],
        v_local: Mapping[str, Any],
        grads: Mapping[str, Any],
        lr: jax.Array,
        step: jax.Array,
    ) -> tuple[Any, Any, Any]:
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
        global_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
        scale = jnp.where((clip_norm > 0.0) & (global_norm > clip_norm), clip_norm / (global_norm + 1e-12), 1.0)
        grads_scaled = jax.tree_util.tree_map(lambda g: g * scale, grads)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_next = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, m_local, grads_scaled)
        v_next = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), v_local, grads_scaled)
        m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1**step), m_next)
        v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2**step), v_next)
        params_next = jax.tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
            params_local,
            m_hat,
            v_hat,
        )
        return params_next, m_next, v_next

    rng = np.random.default_rng(int(seed))
    step_count = 0
    for epoch in range(1, epochs + 1):
        lr_value = _cosine_schedule_with_warmup(epoch, epochs, float(eta_init))
        if batch < n_samples:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        for start in range(0, n_samples, batch):
            batch_idx = indices[start : start + batch]
            batch_idx_jax = _device_put(jnp.asarray(batch_idx, dtype=jnp.int64), target_device)
            X_batch = jnp.take(X_jax, batch_idx_jax, axis=1)
            Y_batch = jnp.take(Y_jax, batch_idx_jax, axis=1)
            w_batch = jnp.take(weights_jax, batch_idx_jax, axis=0)
            _, grads = value_and_grad(params, X_batch, Y_batch, w_batch)
            step_count += 1
            params, opt_m, opt_v = update_step(
                params,
                opt_m,
                opt_v,
                grads,
                _device_put(jnp.asarray(lr_value, dtype=jnp.float64), target_device),
                _device_put(jnp.asarray(step_count, dtype=jnp.float64), target_device),
            )

    trained_blocks = tuple(
        ResBlock(W1=block["W1"], b1=block["b1"], W2=block["W2"], b2=block["b2"])
        for block in params["blocks"]
    )
    return FrozenResNet(
        W_embed=params["W_embed"],
        b_embed=params["b_embed"],
        d_theta=theta_dim,
        W_gamma=params["W_gamma"],
        b_gamma=params["b_gamma"],
        W_beta=params["W_beta"],
        b_beta=params["b_beta"],
        blocks=trained_blocks,
        W_out=params["W_out"],
        b_out=params["b_out"],
        norm=_norm_on_device(norm, target_device),
        d_in=d_in,
        d_out=d_out,
    )


def make_surrogate_residual_predictor(
    frozen: FrozenMLP | FrozenResNet,
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], jax.Array]:
    def residual_predict(state: ArrayLike, shock_t: ArrayLike, theta: ArrayLike) -> jax.Array:
        x = jnp.concatenate(
            [
                jnp.asarray(state, dtype=jnp.float64).reshape(-1),
                jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1),
                jnp.asarray(theta, dtype=jnp.float64).reshape(-1),
            ],
            axis=0,
        )
        return predict_frozen(frozen, x)

    return residual_predict


def make_batched_surrogate_residual_predictor(
    frozen: FrozenMLP | FrozenResNet,
) -> Callable[[ArrayLike], jax.Array]:
    def batch_residual_predict(X: ArrayLike) -> jax.Array:
        return predict_frozen_batch(frozen, X)

    return batch_residual_predict


def surrogate_additive_residual_loglik_per_period(
    full_predict: Callable[[Any, Any, Any], Any],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    shocks: ArrayLike,
    theta: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    *,
    d_obs: Optional[int] = None,
    allow_full_residual: bool = True,
    z_threshold: Optional[float] = None,
) -> np.ndarray:
    residual_predict = make_surrogate_residual_predictor(frozen)
    if z_threshold is None:
        return additive_residual_loglik_per_period(
            full_predict,
            residual_predict,
            s0,
            shocks,
            theta,
            obs_data,
            obs_sigma,
            d_obs=d_obs,
            allow_full_residual=allow_full_residual,
        )
    observations = np.asarray(obs_data, dtype=np.float64)
    obs_dim = observations.shape[0] if d_obs is None else int(d_obs)

    def predict_fn(state: Any, shock_t: Any, theta_local: Any) -> tuple[np.ndarray, np.ndarray]:
        return predict_additive_residual_ood(
            full_predict,
            residual_predict,
            state,
            shock_t,
            theta_local,
            obs_dim,
            frozen.norm.as_julia_dict(),
            z_threshold=z_threshold,
            allow_full_residual=allow_full_residual,
        )

    from .regime_switching_api import conditional_loglik_per_period

    return conditional_loglik_per_period(
        predict_fn,
        s0,
        shocks,
        theta,
        observations,
        obs_sigma,
    )


def surrogate_inversion_loglik_per_period(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    theta: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    shock_sigmas: ArrayLike,
    *,
    gate_mask: Optional[Sequence[bool] | np.ndarray] = None,
    correction_clamp: Optional[ArrayLike] = None,
    maxit: int = 10,
    tol: float = 1e-6,
    lambda_: float = 1e-4,
    refine_maxit: int = 0,
    refine_tol: float = 1e-4,
    refine_max_step_std: float = 2.0,
    refine_min_alpha: float = 1e-3,
    refine_accept_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    residual_predict = make_surrogate_residual_predictor(frozen)
    batch_residual_predict = make_batched_surrogate_residual_predictor(frozen)
    d_obs = np.asarray(obs_data, dtype=np.float64).shape[0]

    def eval_predict(state: Any, shock_t: Any, theta_local: Any) -> tuple[np.ndarray, np.ndarray]:
        return predict_additive_residual(
            lambda s, e, th: np.concatenate(rom_predict(s, e, th), axis=0),
            residual_predict,
            state,
            shock_t,
            theta_local,
            d_obs,
            allow_full_residual=True,
        )

    return inversion_loglik_per_period(
        rom_predict,
        s0,
        theta,
        obs_data,
        obs_sigma,
        shock_sigmas,
        eval_predict_fn=eval_predict,
        batch_eval_residual_fn=batch_residual_predict,
        single_eval_residual_fn=lambda x: residual_predict(
            np.asarray(x)[: np.asarray(s0).reshape(-1).shape[0]],
            np.asarray(x)[np.asarray(s0).reshape(-1).shape[0] : np.asarray(s0).reshape(-1).shape[0] + np.asarray(shock_sigmas).reshape(-1).shape[0]],
            np.asarray(x)[np.asarray(s0).reshape(-1).shape[0] + np.asarray(shock_sigmas).reshape(-1).shape[0] :],
        ),
        gate_mask=gate_mask,
        correction_clamp=correction_clamp,
        maxit=maxit,
        tol=tol,
        lambda_=lambda_,
        refine_maxit=refine_maxit,
        refine_tol=refine_tol,
        refine_max_step_std=refine_max_step_std,
        refine_min_alpha=refine_min_alpha,
        refine_accept_tol=refine_accept_tol,
    )


def surrogate_predict_additive_jax(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    state: ArrayLike,
    shock_t: ArrayLike,
    theta: ArrayLike,
    *,
    d_obs: int,
    allow_full_residual: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """JAX-native additive ROM+surrogate predictor.

    The frozen surrogate is interpreted as a residual model trained on
    ``[state, shock, theta]``. If the residual has at least
    ``d_obs + d_state`` outputs, the trailing ``d_state`` entries correct the
    next-state prediction; otherwise only the observations are corrected.
    """

    obs_rom, state_rom = rom_predict(state, shock_t, theta)
    obs_rom_arr = jnp.asarray(obs_rom, dtype=jnp.float64).reshape(-1)
    state_rom_arr = jnp.asarray(state_rom, dtype=jnp.float64).reshape(-1)
    state_arr = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
    shock_arr = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
    theta_arr = jnp.asarray(theta, dtype=jnp.float64).reshape(-1)
    d_obs_int = int(d_obs)
    if d_obs_int <= 0:
        raise ValueError(f"d_obs must be positive, got {d_obs}.")
    if obs_rom_arr.shape[0] != d_obs_int:
        raise ValueError(f"rom_predict returned {obs_rom_arr.shape[0]} observations, expected {d_obs_int}.")
    x = jnp.concatenate([state_arr, shock_arr, theta_arr], axis=0)
    residual = predict_frozen(frozen, x).reshape(-1)
    if residual.shape[0] < d_obs_int:
        raise ValueError(
            f"Surrogate output dimension {residual.shape[0]} is smaller than d_obs={d_obs_int}."
        )
    obs = obs_rom_arr + residual[:d_obs_int]
    if allow_full_residual and residual.shape[0] >= d_obs_int + state_rom_arr.shape[0]:
        state_next = state_rom_arr + residual[d_obs_int : d_obs_int + state_rom_arr.shape[0]]
    else:
        state_next = state_rom_arr
    return obs, state_next


def surrogate_inversion_loglik_per_period_jax(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    theta: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    shock_sigmas: ArrayLike,
    *,
    maxit: int = 8,
    lambda_: float = 1e-4,
    allow_full_residual: bool = True,
    active_shock_indices: Optional[Sequence[int]] = None,
) -> tuple[jax.Array, jax.Array]:
    """JAX-native surrogate inversion likelihood.

    This mirrors ``surrogate_inversion_loglik_per_period`` but uses an unrolled
    Gauss-Newton shock inversion with fixed loop count. The fixed loop is
    intentionally HMC-friendly: it JIT-compiles, differentiates through the
    unrolled solver, and avoids data-dependent shapes. The ROM matrices and
    frozen surrogate are treated as fixed objects; parameter dependence enters
    through the supplied ``theta`` vector and through ``rom_predict`` if that
    closure uses it with JAX operations.
    """

    maxit_int = int(maxit)
    if maxit_int <= 0:
        raise ValueError(f"maxit must be positive, got {maxit}.")
    lambda_float = float(lambda_)
    if lambda_float < 0.0:
        raise ValueError(f"lambda_ must be nonnegative, got {lambda_}.")

    observations = jnp.asarray(obs_data, dtype=jnp.float64)
    if observations.ndim != 2:
        raise ValueError(f"obs_data must be rank-2 with shape (d_obs, periods), got {observations.shape}.")
    obs_sigma_vec = jnp.asarray(obs_sigma, dtype=jnp.float64).reshape(-1)
    shock_sigma_vec = jnp.asarray(shock_sigmas, dtype=jnp.float64).reshape(-1)
    if obs_sigma_vec.shape[0] != observations.shape[0]:
        raise ValueError(
            f"obs_data/obs_sigma mismatch: {observations.shape[0]} vs {obs_sigma_vec.shape[0]}."
        )
    if active_shock_indices is None:
        shock_sigma_host = np.asarray(shock_sigmas, dtype=np.float64).reshape(-1)
        active_tuple = tuple(int(i) for i in np.flatnonzero(shock_sigma_host > 0.0))
    else:
        active_tuple = tuple(int(i) for i in active_shock_indices)
    if any(i < 0 or i >= int(shock_sigma_vec.shape[0]) for i in active_tuple):
        raise ValueError(f"active_shock_indices out of bounds for {shock_sigma_vec.shape[0]} shocks.")
    active_idx = jnp.asarray(active_tuple, dtype=jnp.int64)
    n_active = len(active_tuple)
    shock_std = shock_sigma_vec[active_idx] if n_active else jnp.zeros((0,), dtype=jnp.float64)
    state0 = jnp.asarray(s0, dtype=jnp.float64).reshape(-1)
    theta_vec = jnp.asarray(theta, dtype=jnp.float64).reshape(-1)
    d_obs = int(observations.shape[0])
    d_shock = int(shock_sigma_vec.shape[0])
    eye_active = jnp.eye(n_active, dtype=jnp.float64)
    obs_log_norm_const = jnp.sum(jnp.log(2.0 * jnp.pi * obs_sigma_vec**2))
    shock_log_norm_const = (
        jnp.sum(jnp.log(2.0 * jnp.pi * shock_std**2))
        if n_active
        else jnp.asarray(0.0, dtype=jnp.float64)
    )

    def full_shock(eps_struct: jax.Array) -> jax.Array:
        eps_full = jnp.zeros((d_shock,), dtype=jnp.float64)
        if n_active:
            eps_full = eps_full.at[active_idx].set(eps_struct)
        return eps_full

    def predict_period(state: jax.Array, eps_struct: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        eps_full = full_shock(eps_struct)
        obs_pred, state_next = surrogate_predict_additive_jax(
            rom_predict,
            frozen,
            state,
            eps_full,
            theta_vec,
            d_obs=d_obs,
            allow_full_residual=allow_full_residual,
        )
        return eps_full, obs_pred, state_next

    def one_period(state: jax.Array, y_obs: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        if not n_active:
            eps_empty = jnp.zeros((0,), dtype=jnp.float64)
            eps_full, obs_pred, state_next = predict_period(state, eps_empty)
            resid_obs = (y_obs - obs_pred) / obs_sigma_vec
            ll_t = -0.5 * (jnp.sum(resid_obs**2) + obs_log_norm_const)
            return state_next, (ll_t, eps_full)

        def residual_aug(eps_struct: jax.Array) -> jax.Array:
            _, obs_pred, _ = predict_period(state, eps_struct)
            resid_obs = (y_obs - obs_pred) / obs_sigma_vec
            resid_prior = eps_struct / shock_std
            return jnp.concatenate([resid_obs, resid_prior], axis=0)

        def gn_body(_i: int, eps_struct: jax.Array) -> jax.Array:
            resid = residual_aug(eps_struct)
            jac = jax.jacfwd(residual_aug)(eps_struct)
            lhs = jac.T @ jac + lambda_float * eye_active
            rhs = -(jac.T @ resid)
            step = jnp.linalg.solve(lhs, rhs)
            return eps_struct + step

        eps0 = jnp.zeros((n_active,), dtype=jnp.float64)
        eps_hat = jax.lax.fori_loop(0, maxit_int, gn_body, eps0)
        eps_full, obs_pred, state_next = predict_period(state, eps_hat)
        resid_obs = (y_obs - obs_pred) / obs_sigma_vec
        resid_prior = eps_hat / shock_std
        ll_t = -0.5 * (
            jnp.sum(resid_obs**2)
            + jnp.sum(resid_prior**2)
            + obs_log_norm_const
            + shock_log_norm_const
        )
        return state_next, (ll_t, eps_full)

    _, (ll, shocks_t) = jax.lax.scan(one_period, state0, observations.T)
    return ll, shocks_t.T


def surrogate_inversion_loglikelihood_jax(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    theta: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    shock_sigmas: ArrayLike,
    *,
    maxit: int = 8,
    lambda_: float = 1e-4,
    allow_full_residual: bool = True,
    active_shock_indices: Optional[Sequence[int]] = None,
) -> jax.Array:
    ll, _ = surrogate_inversion_loglik_per_period_jax(
        rom_predict,
        frozen,
        s0,
        theta,
        obs_data,
        obs_sigma,
        shock_sigmas,
        maxit=maxit,
        lambda_=lambda_,
        allow_full_residual=allow_full_residual,
        active_shock_indices=active_shock_indices,
    )
    return jnp.sum(ll)


def _require_numpyro_surrogate() -> tuple[Any, Any]:
    try:
        import numpyro
        from numpyro.infer.util import log_density
    except ImportError as exc:
        raise ImportError(
            "NumPyro surrogate integration requires the optional `numpyro` dependency."
        ) from exc
    return numpyro, log_density


def build_numpyro_surrogate_inversion_model_jax(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    shock_sigmas: ArrayLike,
    priors: Mapping[str, Any],
    *,
    parameter_names: Optional[Sequence[str]] = None,
    theta_transform: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    maxit: int = 8,
    lambda_: float = 1e-4,
    allow_full_residual: bool = True,
    active_shock_indices: Optional[Sequence[int]] = None,
):
    """Build a NumPyro model using the JAX surrogate inversion likelihood.

    By default, the theta vector is assembled from ``priors`` in insertion
    order. Pass ``parameter_names`` to make that order explicit, or pass
    ``theta_transform`` to map sampled values into an arbitrary theta vector.
    """

    numpyro, _ = _require_numpyro_surrogate()
    prior_names = tuple(str(name) for name in priors)
    if not prior_names:
        raise ValueError("priors must contain at least one parameter prior.")
    if theta_transform is None:
        theta_names = tuple(str(name) for name in (parameter_names or prior_names))
        missing = tuple(name for name in theta_names if name not in priors)
        if missing:
            raise ValueError("parameter_names contains names missing from priors: " + ", ".join(missing))
    else:
        theta_names = tuple(str(name) for name in (parameter_names or prior_names))

    def numpyro_model() -> None:
        sampled_values = {
            name: numpyro.sample(name, priors[name])
            for name in prior_names
        }
        if theta_transform is None:
            theta = jnp.asarray([sampled_values[name] for name in theta_names], dtype=jnp.float64)
        else:
            theta = jnp.asarray(theta_transform(sampled_values), dtype=jnp.float64).reshape(-1)
        loglikelihood = surrogate_inversion_loglikelihood_jax(
            rom_predict,
            frozen,
            s0,
            theta,
            obs_data,
            obs_sigma,
            shock_sigmas,
            maxit=maxit,
            lambda_=lambda_,
            allow_full_residual=allow_full_residual,
            active_shock_indices=active_shock_indices,
        )
        numpyro.deterministic("theta_vector", theta)
        numpyro.deterministic("loglikelihood", loglikelihood)
        numpyro.factor("surrogate_inversion_loglikelihood", loglikelihood)

    return numpyro_model


def evaluate_numpyro_surrogate_log_density_jax(
    rom_predict: Callable[[Any, Any, Any], tuple[Any, Any]],
    frozen: FrozenMLP | FrozenResNet,
    s0: ArrayLike,
    obs_data: ArrayLike,
    obs_sigma: ArrayLike,
    shock_sigmas: ArrayLike,
    priors: Mapping[str, Any],
    parameter_samples: Mapping[str, Any],
    *,
    parameter_names: Optional[Sequence[str]] = None,
    theta_transform: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    maxit: int = 8,
    lambda_: float = 1e-4,
    allow_full_residual: bool = True,
    active_shock_indices: Optional[Sequence[int]] = None,
) -> jax.Array:
    _, log_density = _require_numpyro_surrogate()
    numpyro_model = build_numpyro_surrogate_inversion_model_jax(
        rom_predict,
        frozen,
        s0,
        obs_data,
        obs_sigma,
        shock_sigmas,
        priors,
        parameter_names=parameter_names,
        theta_transform=theta_transform,
        maxit=maxit,
        lambda_=lambda_,
        allow_full_residual=allow_full_residual,
        active_shock_indices=active_shock_indices,
    )
    log_joint, _ = log_density(numpyro_model, (), {}, parameter_samples)
    return jnp.asarray(log_joint, dtype=jnp.float64)
