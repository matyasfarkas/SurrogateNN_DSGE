from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .parameter_sampling import ParameterDesign
from .sep import (
    BatchedSEPSolution,
    SEPConditionalResidualFn,
    SEPConfig,
    solve_batched_stochastic_extended_path_residual_expectation,
)


PredictTupleFn = Callable[[Any, Any, Any], tuple[Any, Any]]
PredictBatchTupleFn = Callable[[Any, Any, Any], tuple[Any, Any]]
_TARGET_MODES = ("residual_obs", "residual_full", "fom_obs", "fom_full")


class BatchedSurrogateRolloutArrays(NamedTuple):
    X: jax.Array
    Y: jax.Array
    Y_rom: jax.Array
    theta: jax.Array
    theta_ids: jax.Array
    period_ids: jax.Array
    sample_mask: jax.Array
    theta_success: jax.Array
    theta_stable_periods: jax.Array

    @property
    def n_samples_total(self) -> int:
        return int(self.X.shape[1])


@dataclass(frozen=True)
class SurrogateDataset:
    X: np.ndarray
    Y: np.ndarray
    theta: np.ndarray
    theta_ids: np.ndarray
    period_ids: np.ndarray
    theta_success: np.ndarray
    theta_stable_periods: np.ndarray
    target_mode: str
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    theta_names: tuple[str, ...] = ()
    Y_rom: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        Y = np.asarray(self.Y, dtype=np.float64)
        theta = np.asarray(self.theta, dtype=np.float64)
        theta_ids = np.asarray(self.theta_ids, dtype=np.int64).reshape(-1)
        period_ids = np.asarray(self.period_ids, dtype=np.int64).reshape(-1)
        theta_success = np.asarray(self.theta_success, dtype=bool).reshape(-1)
        theta_stable_periods = np.asarray(self.theta_stable_periods, dtype=np.int64).reshape(-1)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be rank-2 matrices with shape (features, samples).")
        if theta.ndim != 2:
            raise ValueError("theta must be a rank-2 matrix with shape (parameters, theta_draws).")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"X/Y sample mismatch: {X.shape[1]} vs {Y.shape[1]}.")
        if theta_ids.shape[0] != X.shape[1] or period_ids.shape[0] != X.shape[1]:
            raise ValueError("theta_ids and period_ids must have one entry per sample.")
        if theta_success.shape[0] != theta.shape[1] or theta_stable_periods.shape[0] != theta.shape[1]:
            raise ValueError("theta_success and theta_stable_periods must have one entry per theta draw.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all() or not np.isfinite(theta).all():
            raise ValueError("SurrogateDataset arrays must be finite.")
        if self.Y_rom is not None:
            Y_rom = np.asarray(self.Y_rom, dtype=np.float64)
            if Y_rom.shape != Y.shape:
                raise ValueError(f"Y_rom shape mismatch: {Y_rom.shape} vs {Y.shape}.")
            if not np.isfinite(Y_rom).all():
                raise ValueError("Y_rom must be finite.")
            object.__setattr__(self, "Y_rom", Y_rom)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "theta_ids", theta_ids)
        object.__setattr__(self, "period_ids", period_ids)
        object.__setattr__(self, "theta_success", theta_success)
        object.__setattr__(self, "theta_stable_periods", theta_stable_periods)
        object.__setattr__(self, "target_mode", str(self.target_mode))
        object.__setattr__(self, "input_names", tuple(str(name) for name in self.input_names))
        object.__setattr__(self, "output_names", tuple(str(name) for name in self.output_names))
        object.__setattr__(self, "theta_names", tuple(str(name) for name in self.theta_names))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_theta(self) -> int:
        return int(self.theta.shape[1])


def _as_vector(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def _call_predict_tuple(
    predict_fn: PredictTupleFn,
    state: np.ndarray,
    shock: np.ndarray,
    theta: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    output = predict_fn(state, shock, theta)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError(f"{label} must return `(obs, next_state)`.")
    obs = _as_vector(output[0], label=f"{label} observation")
    next_state = _as_vector(output[1], label=f"{label} next_state")
    return obs, next_state


def _theta_matrix(theta_design: ParameterDesign | np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(theta_design, ParameterDesign):
        return np.asarray(theta_design.theta, dtype=np.float64), theta_design.names
    theta = np.asarray(theta_design, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"theta_design must be rank-2, got shape {theta.shape}.")
    return theta, tuple(f"theta_{i}" for i in range(theta.shape[0]))


def _theta_array_jax(theta_design: ParameterDesign | jax.Array | np.ndarray) -> jax.Array:
    theta_source = theta_design.theta if isinstance(theta_design, ParameterDesign) else theta_design
    theta = jnp.asarray(theta_source, dtype=jnp.float64)
    if theta.ndim != 2:
        raise ValueError(f"theta_design must be rank-2, got shape {theta.shape}.")
    return theta


def _coerce_shocks(shocks: Any, *, n_theta: int) -> np.ndarray:
    array = np.asarray(shocks, dtype=np.float64)
    if array.ndim == 2:
        if not np.isfinite(array).all():
            raise ValueError("shocks contains non-finite values.")
        return np.broadcast_to(array[None, :, :], (n_theta, array.shape[0], array.shape[1])).copy()
    if array.ndim != 3:
        raise ValueError("shocks must have shape (d_shock, periods), (n_theta, d_shock, periods), or (d_shock, periods, n_theta).")
    if array.shape[0] == n_theta:
        out = array
    elif array.shape[2] == n_theta:
        out = np.moveaxis(array, 2, 0)
    else:
        raise ValueError(
            "Rank-3 shocks must put theta draws on axis 0 or axis 2; "
            f"got shape {array.shape} for n_theta={n_theta}."
        )
    if not np.isfinite(out).all():
        raise ValueError("shocks contains non-finite values.")
    return np.asarray(out, dtype=np.float64)


def _coerce_initial_states(initial_state: Any, *, n_theta: int) -> np.ndarray:
    array = np.asarray(initial_state, dtype=np.float64)
    if array.ndim == 1:
        state = _as_vector(array, label="initial_state")
        return np.broadcast_to(state[None, :], (int(n_theta), state.shape[0])).copy()
    if array.ndim != 2:
        raise ValueError(
            "initial_state must have shape (d_state,), (n_theta, d_state), "
            f"or (d_state, n_theta); got shape {array.shape}."
        )
    if array.shape[0] == int(n_theta):
        out = array
    elif array.shape[1] == int(n_theta):
        out = array.T
    else:
        raise ValueError(
            "Rank-2 initial_state must put theta draws on axis 0 or axis 1; "
            f"got shape {array.shape} for n_theta={n_theta}."
        )
    if not np.isfinite(out).all():
        raise ValueError("initial_state contains non-finite values.")
    return np.asarray(out, dtype=np.float64).copy()


def _normalize_target_mode(target_mode: str) -> str:
    target_mode_norm = str(target_mode).strip().lower()
    if target_mode_norm not in _TARGET_MODES:
        raise ValueError(
            "target_mode must be one of 'residual_obs', 'residual_full', "
            f"'fom_obs', or 'fom_full', got {target_mode!r}."
        )
    return target_mode_norm


def _as_batched_rollout_tensor(
    values: Any,
    *,
    label: str,
    n_theta: int,
    periods: Optional[int] = None,
    dim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{label} must have shape (n_theta, periods, dim), got {array.shape}.")
    if array.shape[0] != int(n_theta):
        raise ValueError(
            f"{label} theta-axis mismatch: expected {int(n_theta)} theta draws on axis 0, "
            f"got shape {array.shape}."
        )
    if periods is not None and array.shape[1] != int(periods):
        raise ValueError(
            f"{label} period-axis mismatch: expected {int(periods)} periods on axis 1, "
            f"got shape {array.shape}."
        )
    if dim is not None and array.shape[2] != int(dim):
        raise ValueError(
            f"{label} feature-axis mismatch: expected dimension {int(dim)} on axis 2, "
            f"got shape {array.shape}."
        )
    return array


def _as_batched_rollout_array_jax(
    values: Any,
    *,
    label: str,
    n_theta: int,
    periods: Optional[int] = None,
    dim: Optional[int] = None,
) -> jax.Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    if array.ndim != 3:
        raise ValueError(f"{label} must have shape (n_theta, periods, dim), got {array.shape}.")
    if array.shape[0] != int(n_theta):
        raise ValueError(
            f"{label} theta-axis mismatch: expected {int(n_theta)} theta draws on axis 0, "
            f"got shape {array.shape}."
        )
    if periods is not None and array.shape[1] != int(periods):
        raise ValueError(
            f"{label} period-axis mismatch: expected {int(periods)} periods on axis 1, "
            f"got shape {array.shape}."
        )
    if dim is not None and array.shape[2] != int(dim):
        raise ValueError(
            f"{label} feature-axis mismatch: expected dimension {int(dim)} on axis 2, "
            f"got shape {array.shape}."
        )
    return array


def _finite_period_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("At least one rollout tensor is required.")
    mask = np.ones(arrays[0].shape[:2], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array).all(axis=2)
    return mask


def _stable_prefix_lengths(finite_by_period: np.ndarray) -> np.ndarray:
    stable_periods = np.zeros((finite_by_period.shape[0],), dtype=np.int64)
    total_periods = int(finite_by_period.shape[1])
    for theta_idx, theta_finite in enumerate(finite_by_period):
        first_bad = np.flatnonzero(~theta_finite)
        stable_periods[theta_idx] = total_periods if first_bad.size == 0 else int(first_bad[0])
    return stable_periods


def _finite_period_mask_jax(*arrays: jax.Array) -> jax.Array:
    if not arrays:
        raise ValueError("At least one rollout tensor is required.")
    mask = jnp.ones(arrays[0].shape[:2], dtype=bool)
    for array in arrays:
        mask = mask & jnp.all(jnp.isfinite(array), axis=2)
    return mask


def _stable_prefix_lengths_jax(finite_by_period: jax.Array) -> jax.Array:
    finite_int = finite_by_period.astype(jnp.int32)
    return jnp.sum(jnp.cumprod(finite_int, axis=1), axis=1).astype(jnp.int32)


def _target_vector(
    target_mode: str,
    fom_obs: np.ndarray,
    fom_state_next: np.ndarray,
    rom_obs: np.ndarray,
    rom_state_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if target_mode == "residual_obs":
        y = fom_obs - rom_obs
        y_rom = np.zeros_like(y)
    elif target_mode == "residual_full":
        y = np.concatenate([fom_obs - rom_obs, fom_state_next - rom_state_next])
        y_rom = np.zeros_like(y)
    elif target_mode == "fom_obs":
        y = fom_obs
        y_rom = rom_obs
    elif target_mode == "fom_full":
        y = np.concatenate([fom_obs, fom_state_next])
        y_rom = np.concatenate([rom_obs, rom_state_next])
    else:
        _normalize_target_mode(target_mode)
        raise AssertionError("unreachable target_mode branch")
    return y, y_rom


def _target_arrays_jax(
    target_mode: str,
    fom_obs: jax.Array,
    fom_state_next: jax.Array,
    rom_obs: jax.Array,
    rom_state_next: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if target_mode == "residual_obs":
        y = fom_obs - rom_obs
        y_rom = jnp.zeros_like(y)
    elif target_mode == "residual_full":
        y = jnp.concatenate([fom_obs - rom_obs, fom_state_next - rom_state_next], axis=2)
        y_rom = jnp.zeros_like(y)
    elif target_mode == "fom_obs":
        y = fom_obs
        y_rom = rom_obs
    elif target_mode == "fom_full":
        y = jnp.concatenate([fom_obs, fom_state_next], axis=2)
        y_rom = jnp.concatenate([rom_obs, rom_state_next], axis=2)
    else:
        _normalize_target_mode(target_mode)
        raise AssertionError("unreachable target_mode branch")
    return y, y_rom


def _concatenate_batched_surrogate_arrays(
    chunks: Sequence[BatchedSurrogateRolloutArrays],
) -> BatchedSurrogateRolloutArrays:
    if not chunks:
        raise ValueError("At least one batched surrogate chunk is required.")
    if len(chunks) == 1:
        return chunks[0]

    x_dim = int(chunks[0].X.shape[0])
    y_dim = int(chunks[0].Y.shape[0])
    y_rom_dim = int(chunks[0].Y_rom.shape[0])
    theta_dim = int(chunks[0].theta.shape[0])
    theta_offset = 0
    theta_ids = []
    for chunk in chunks:
        if int(chunk.X.shape[0]) != x_dim:
            raise ValueError("Cannot concatenate batched surrogate chunks with different X dimensions.")
        if int(chunk.Y.shape[0]) != y_dim or int(chunk.Y_rom.shape[0]) != y_rom_dim:
            raise ValueError("Cannot concatenate batched surrogate chunks with different Y dimensions.")
        if int(chunk.theta.shape[0]) != theta_dim:
            raise ValueError("Cannot concatenate batched surrogate chunks with different theta dimensions.")
        theta_ids.append(jnp.asarray(chunk.theta_ids, dtype=jnp.int32) + int(theta_offset))
        theta_offset += int(chunk.theta.shape[1])

    return BatchedSurrogateRolloutArrays(
        X=jnp.concatenate([chunk.X for chunk in chunks], axis=1),
        Y=jnp.concatenate([chunk.Y for chunk in chunks], axis=1),
        Y_rom=jnp.concatenate([chunk.Y_rom for chunk in chunks], axis=1),
        theta=jnp.concatenate([chunk.theta for chunk in chunks], axis=1),
        theta_ids=jnp.concatenate(theta_ids, axis=0),
        period_ids=jnp.concatenate([chunk.period_ids for chunk in chunks], axis=0),
        sample_mask=jnp.concatenate([chunk.sample_mask for chunk in chunks], axis=0),
        theta_success=jnp.concatenate([chunk.theta_success for chunk in chunks], axis=0),
        theta_stable_periods=jnp.concatenate([chunk.theta_stable_periods for chunk in chunks], axis=0),
    )


def _evaluate_rom_batch(
    rom_predict: PredictTupleFn,
    states: np.ndarray,
    shocks: np.ndarray,
    theta_samples: np.ndarray,
    *,
    rom_predict_batch: Optional[PredictBatchTupleFn] = None,
    rom_predict_is_jax: bool = False,
) -> tuple[jax.Array, jax.Array]:
    if rom_predict_batch is not None:
        obs, next_state = rom_predict_batch(
            jnp.asarray(states, dtype=jnp.float64),
            jnp.asarray(shocks, dtype=jnp.float64),
            jnp.asarray(theta_samples, dtype=jnp.float64),
        )
        obs_arr = jnp.asarray(obs, dtype=jnp.float64)
        next_arr = jnp.asarray(next_state, dtype=jnp.float64)
    elif bool(rom_predict_is_jax):
        obs_arr, next_arr = jax.vmap(rom_predict)(
            jnp.asarray(states, dtype=jnp.float64),
            jnp.asarray(shocks, dtype=jnp.float64),
            jnp.asarray(theta_samples, dtype=jnp.float64),
        )
        obs_arr = jnp.asarray(obs_arr, dtype=jnp.float64)
        next_arr = jnp.asarray(next_arr, dtype=jnp.float64)
    else:
        obs_rows = []
        next_rows = []
        for sample_idx in range(states.shape[0]):
            obs, next_state = _call_predict_tuple(
                rom_predict,
                states[sample_idx],
                shocks[sample_idx],
                theta_samples[sample_idx],
                label="rom_predict",
            )
            obs_rows.append(obs)
            next_rows.append(next_state)
        obs_arr = jnp.asarray(np.vstack(obs_rows), dtype=jnp.float64)
        next_arr = jnp.asarray(np.vstack(next_rows), dtype=jnp.float64)

    if obs_arr.ndim != 2:
        raise ValueError(f"ROM observations must have shape (batch, obs_dim), got {obs_arr.shape}.")
    if next_arr.shape != (states.shape[0], states.shape[1]):
        raise ValueError(
            "ROM next-state batch must have shape "
            f"({states.shape[0]}, {states.shape[1]}), got {next_arr.shape}."
        )
    return obs_arr, next_arr


def _sample_periods(
    available: int,
    samples_per_theta: Optional[int],
    rng: np.random.Generator,
    *,
    replace: bool,
) -> np.ndarray:
    if available < 1:
        return np.zeros((0,), dtype=np.int64)
    if samples_per_theta is None or int(samples_per_theta) >= available:
        return np.arange(available, dtype=np.int64)
    n_samples = int(samples_per_theta)
    if n_samples < 1:
        raise ValueError(f"samples_per_theta must be positive when provided, got {samples_per_theta}.")
    if replace:
        return rng.integers(0, available, size=n_samples, endpoint=False, dtype=np.int64)
    return np.sort(rng.choice(available, size=n_samples, replace=False)).astype(np.int64)


def build_surrogate_residual_arrays_jax(
    states: Any,
    shocks: Any,
    theta_design: ParameterDesign | jax.Array | np.ndarray,
    rom_obs: Any,
    rom_state_next: Any,
    fom_obs: Any,
    fom_state_next: Any,
    *,
    target_mode: str = "residual_full",
    min_stable_periods: int = 1,
) -> BatchedSurrogateRolloutArrays:
    """Assemble fixed-shape surrogate arrays on a JAX device.

    Inputs use shape ``(n_theta, periods, dim)``. The returned matrices keep all
    theta-period columns and provide ``sample_mask`` for valid stable-prefix
    samples, avoiding host-side compaction in GPU profiling/training pipelines.
    """

    theta = _theta_array_jax(theta_design)
    if theta.shape[1] < 1:
        raise ValueError("theta_design must contain at least one theta draw.")
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")
    n_theta = int(theta.shape[1])

    states_array = _as_batched_rollout_array_jax(states, label="states", n_theta=n_theta)
    periods = int(states_array.shape[1])
    state_dim = int(states_array.shape[2])
    shocks_array = _as_batched_rollout_array_jax(shocks, label="shocks", n_theta=n_theta, periods=periods)
    rom_obs_array = _as_batched_rollout_array_jax(rom_obs, label="rom_obs", n_theta=n_theta, periods=periods)
    fom_obs_array = _as_batched_rollout_array_jax(
        fom_obs,
        label="fom_obs",
        n_theta=n_theta,
        periods=periods,
        dim=int(rom_obs_array.shape[2]),
    )
    rom_state_next_array = _as_batched_rollout_array_jax(
        rom_state_next,
        label="rom_state_next",
        n_theta=n_theta,
        periods=periods,
        dim=state_dim,
    )
    fom_state_next_array = _as_batched_rollout_array_jax(
        fom_state_next,
        label="fom_state_next",
        n_theta=n_theta,
        periods=periods,
        dim=state_dim,
    )

    target_mode_norm = _normalize_target_mode(target_mode)
    finite_by_period = _finite_period_mask_jax(
        states_array,
        shocks_array,
        rom_obs_array,
        rom_state_next_array,
        fom_obs_array,
        fom_state_next_array,
    )
    theta_stable_periods = _stable_prefix_lengths_jax(finite_by_period)
    period_grid = jnp.broadcast_to(
        jnp.arange(periods, dtype=jnp.int32)[None, :],
        (n_theta, periods),
    )
    sample_mask = (period_grid < theta_stable_periods[:, None]) & (
        theta_stable_periods[:, None] >= int(min_stable_periods)
    )
    theta_success = (theta_stable_periods >= int(min_stable_periods)) & (
        theta_stable_periods == periods
    )

    theta_by_period = jnp.broadcast_to(
        theta.T[:, None, :],
        (n_theta, periods, int(theta.shape[0])),
    )
    x = jnp.concatenate([states_array, shocks_array, theta_by_period], axis=2)
    y, y_rom = _target_arrays_jax(
        target_mode_norm,
        fom_obs_array,
        fom_state_next_array,
        rom_obs_array,
        rom_state_next_array,
    )
    theta_ids = jnp.broadcast_to(
        jnp.arange(n_theta, dtype=jnp.int32)[:, None],
        (n_theta, periods),
    )
    period_ids = period_grid

    return BatchedSurrogateRolloutArrays(
        X=jnp.reshape(x, (n_theta * periods, x.shape[2])).T,
        Y=jnp.reshape(y, (n_theta * periods, y.shape[2])).T,
        Y_rom=jnp.reshape(y_rom, (n_theta * periods, y_rom.shape[2])).T,
        theta=theta,
        theta_ids=jnp.reshape(theta_ids, (n_theta * periods,)),
        period_ids=jnp.reshape(period_ids, (n_theta * periods,)),
        sample_mask=jnp.reshape(sample_mask, (n_theta * periods,)),
        theta_success=theta_success,
        theta_stable_periods=theta_stable_periods,
    )


def build_surrogate_residual_arrays_from_batched_sep_jax(
    states: Any,
    shocks: Any,
    theta_design: ParameterDesign | jax.Array | np.ndarray,
    rom_obs: Any,
    rom_state_next: Any,
    sep_solution: BatchedSEPSolution,
    observable_indices: Sequence[int],
    *,
    target_mode: str = "residual_full",
    min_stable_periods: int = 1,
    require_accepted: bool = True,
) -> BatchedSurrogateRolloutArrays:
    """Assemble GPU-friendly surrogate targets from a batched SEP solve.

    ``sep_solution.mean_path`` is expected to have shape
    ``(n_theta, state_dim, periods + 1)``. The FOM next-state target for period
    ``t`` is the SEP mean state at ``t + 1``; FOM observables are selected from
    that next-state path using ``observable_indices``. Failed SEP draws are
    masked out by default instead of being silently used for training.
    """

    theta = _theta_array_jax(theta_design)
    if theta.shape[1] < 1:
        raise ValueError("theta_design must contain at least one theta draw.")
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")
    n_theta = int(theta.shape[1])

    states_array = _as_batched_rollout_array_jax(states, label="states", n_theta=n_theta)
    periods = int(states_array.shape[1])
    state_dim = int(states_array.shape[2])
    shocks_array = _as_batched_rollout_array_jax(shocks, label="shocks", n_theta=n_theta, periods=periods)
    rom_obs_array = _as_batched_rollout_array_jax(rom_obs, label="rom_obs", n_theta=n_theta, periods=periods)
    rom_state_next_array = _as_batched_rollout_array_jax(
        rom_state_next,
        label="rom_state_next",
        n_theta=n_theta,
        periods=periods,
        dim=state_dim,
    )

    mean_path = jnp.asarray(sep_solution.mean_path, dtype=jnp.float64)
    if mean_path.ndim != 3:
        raise ValueError(
            "sep_solution.mean_path must have shape (n_theta, state_dim, periods + 1), "
            f"got {mean_path.shape}."
        )
    if mean_path.shape[0] != n_theta:
        raise ValueError(
            f"sep_solution.mean_path theta-axis mismatch: expected {n_theta}, got {mean_path.shape[0]}."
        )
    if mean_path.shape[1] != state_dim:
        raise ValueError(
            "sep_solution.mean_path state-axis mismatch: expected "
            f"{state_dim}, got {mean_path.shape[1]}."
        )
    if mean_path.shape[2] < periods + 1:
        raise ValueError(
            "sep_solution.mean_path must contain at least periods + 1 states; "
            f"expected {periods + 1}, got {mean_path.shape[2]}."
        )

    obs_idx_np = np.asarray(observable_indices, dtype=np.int64).reshape(-1)
    if obs_idx_np.size < 1:
        raise ValueError("observable_indices must contain at least one state index.")
    if np.any(obs_idx_np < 0) or np.any(obs_idx_np >= state_dim):
        raise ValueError(
            "observable_indices out of bounds for SEP state dimension "
            f"{state_dim}: {obs_idx_np.tolist()}."
        )

    fom_state_next = jnp.swapaxes(mean_path[:, :, 1 : periods + 1], 1, 2)
    fom_obs = jnp.take(fom_state_next, jnp.asarray(obs_idx_np, dtype=jnp.int32), axis=2)

    if bool(require_accepted):
        accepted = jnp.asarray(sep_solution.accepted, dtype=bool).reshape(-1)
        if accepted.shape[0] != n_theta:
            raise ValueError(
                "sep_solution.accepted must have one entry per theta draw; "
                f"expected {n_theta}, got {accepted.shape[0]}."
            )
        fom_state_next = jnp.where(accepted[:, None, None], fom_state_next, jnp.nan)
        fom_obs = jnp.where(accepted[:, None, None], fom_obs, jnp.nan)

    return build_surrogate_residual_arrays_jax(
        states_array,
        shocks_array,
        theta,
        rom_obs_array,
        rom_state_next_array,
        fom_obs,
        fom_state_next,
        target_mode=target_mode,
        min_stable_periods=min_stable_periods,
    )


def build_surrogate_residual_arrays_from_batched_sep_feature_grid_jax(
    rom_predict: PredictTupleFn,
    conditional_residual_fn: SEPConditionalResidualFn,
    feature_grid: Any,
    *,
    state_dim: int,
    shock_dim: int,
    terminal_state: Any,
    config: SEPConfig,
    observable_indices: Sequence[int],
    target_mode: str = "residual_full",
    min_successful_samples: int = 1,
    min_stable_periods: int = 1,
    params: object = None,
    params_batched: Optional[bool] = None,
    initial_guess: Optional[Any] = None,
    rom_obs: Optional[Any] = None,
    rom_state_next: Optional[Any] = None,
    rom_predict_batch: Optional[PredictBatchTupleFn] = None,
    rom_predict_is_jax: bool = False,
    chunk_size: Optional[int] = None,
) -> tuple[BatchedSurrogateRolloutArrays, dict[str, object]]:
    """Generate SEP FOM targets for an adaptive feature grid with batched JAX solves.

    ``feature_grid`` columns must be ``[state; shock; theta]``. Each selected
    column is treated as an independent current state/shock/parameter support
    point. The expensive FOM label is produced by one batched SEP solve per
    chunk, while ROM labels are evaluated either through a vectorized JAX
    ``rom_predict_batch``/``rom_predict`` or a scalar Python fallback.

    This is the GPU-native companion to
    :func:`build_surrogate_residual_dataset_from_feature_grid`: instead of
    spending one sequential FOM/SEP call per grid point, it maps the grid into
    a fixed-shape ``BatchedSurrogateRolloutArrays`` object suitable for direct
    JAX/ResNN training. Only the first SEP transition is emitted as a supervised
    target for each grid column; longer ``config.periods`` values still improve
    the nonlinear SEP look-ahead used to compute that first transition.
    """

    grid = np.asarray(feature_grid, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"feature_grid must have shape (features, samples), got {grid.shape}.")
    state_count = int(state_dim)
    shock_count = int(shock_dim)
    if state_count < 1:
        raise ValueError(f"state_dim must be positive, got {state_dim}.")
    if shock_count < 0:
        raise ValueError(f"shock_dim must be nonnegative, got {shock_dim}.")
    theta_dim = int(grid.shape[0]) - state_count - shock_count
    if theta_dim < 1:
        raise ValueError(
            "feature_grid must contain at least one theta row after state/shock rows; "
            f"got feature dimension {grid.shape[0]}, state_dim={state_dim}, shock_dim={shock_dim}."
        )
    attempted = int(grid.shape[1])
    if attempted < 1:
        raise ValueError("feature_grid must contain at least one candidate column.")
    if not np.isfinite(grid).all():
        raise ValueError("feature_grid must be finite.")
    if int(min_successful_samples) < 1:
        raise ValueError(f"min_successful_samples must be >= 1, got {min_successful_samples}.")
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")

    target_mode_norm = _normalize_target_mode(target_mode)
    effective_chunk_size = attempted if chunk_size is None or int(chunk_size) <= 0 else int(chunk_size)
    if effective_chunk_size < 1:
        raise ValueError(f"chunk_size must be positive when provided, got {chunk_size}.")

    states_all = grid[:state_count].T.copy()
    shocks_all = grid[state_count : state_count + shock_count].T.copy()
    theta_all = grid[state_count + shock_count :].T.copy()
    rom_obs_all = None if rom_obs is None else np.asarray(rom_obs, dtype=np.float64)
    rom_state_next_all = None if rom_state_next is None else np.asarray(rom_state_next, dtype=np.float64)
    if (rom_obs_all is None) != (rom_state_next_all is None):
        raise ValueError("rom_obs and rom_state_next must either both be provided or both be omitted.")
    if rom_obs_all is not None:
        if rom_obs_all.ndim != 2 or int(rom_obs_all.shape[0]) != attempted:
            raise ValueError(
                "rom_obs must have shape (samples, obs_dim) when provided, "
                f"got {rom_obs_all.shape}."
            )
        if rom_state_next_all is None:
            raise AssertionError("rom_state_next_all unexpectedly missing.")
        if rom_state_next_all.shape != (attempted, state_count):
            raise ValueError(
                "rom_state_next must have shape "
                f"({attempted}, {state_count}) when provided, got {rom_state_next_all.shape}."
            )
    terminal_array = np.asarray(terminal_state, dtype=np.float64)
    if terminal_array.ndim == 1 and terminal_array.shape[0] != state_count:
        raise ValueError(
            f"terminal_state must have length state_dim={state_count}, got {terminal_array.shape}."
        )
    if terminal_array.ndim == 2 and terminal_array.shape not in {
        (attempted, state_count),
        (state_count, attempted),
    }:
        raise ValueError(
            "Batched terminal_state must have shape "
            f"({attempted}, {state_count}) or ({state_count}, {attempted}), got {terminal_array.shape}."
        )

    initial_guess_array = None if initial_guess is None else np.asarray(initial_guess, dtype=np.float64)

    def infer_params_batched(values: object) -> bool:
        leaves = jax.tree_util.tree_leaves(values)
        array_leaves = [np.asarray(leaf) for leaf in leaves if leaf is not None]
        if not array_leaves:
            return False
        return all(array.ndim >= 1 and int(array.shape[0]) == attempted for array in array_leaves)

    explicit_params_are_batched = (
        False if params is None else infer_params_batched(params) if params_batched is None else bool(params_batched)
    )

    def params_for_chunk(start_idx: int, end_idx: int) -> object:
        if params is None:
            return None
        if not explicit_params_are_batched:
            return params

        def slice_leaf(leaf: Any) -> Any:
            array = jnp.asarray(leaf, dtype=jnp.float64)
            if array.ndim < 1 or int(array.shape[0]) != attempted:
                raise ValueError(
                    "params_batched=True requires every params leaf to have leading "
                    f"dimension {attempted}; got leaf shape {array.shape}."
                )
            return array[start_idx:end_idx]

        return jax.tree_util.tree_map(slice_leaf, params)
    chunks: list[BatchedSurrogateRolloutArrays] = []
    residual_norm_parts: list[np.ndarray] = []
    accepted_parts: list[np.ndarray] = []
    iteration_parts: list[np.ndarray] = []

    for start in range(0, attempted, effective_chunk_size):
        end = min(start + effective_chunk_size, attempted)
        states = states_all[start:end]
        shocks = shocks_all[start:end]
        theta_samples = theta_all[start:end]
        batch = int(end - start)

        deterministic = np.zeros((batch, int(config.periods), shock_count), dtype=np.float64)
        if shock_count:
            deterministic[:, 0, :] = shocks

        if terminal_array.ndim == 2:
            terminal_chunk = terminal_array[start:end] if terminal_array.shape[0] == attempted else terminal_array[:, start:end].T
        else:
            terminal_chunk = terminal_array

        if initial_guess_array is None:
            initial_guess_chunk = None
        elif initial_guess_array.shape[0] == attempted:
            initial_guess_chunk = initial_guess_array[start:end]
        else:
            initial_guess_chunk = initial_guess_array

        params_chunk = params_for_chunk(start, end)
        if params_chunk is None:
            params_chunk = jnp.asarray(theta_samples, dtype=jnp.float64)

        sep_solution = solve_batched_stochastic_extended_path_residual_expectation(
            conditional_residual_fn,
            initial_state=states,
            terminal_state=terminal_chunk,
            shock_dim=shock_count,
            deterministic_shocks=deterministic,
            config=config,
            params=params_chunk,
            initial_guess=initial_guess_chunk,
        )
        if rom_obs_all is not None and rom_state_next_all is not None:
            rom_obs_chunk = jnp.asarray(rom_obs_all[start:end], dtype=jnp.float64)
            rom_state_next_chunk = jnp.asarray(rom_state_next_all[start:end], dtype=jnp.float64)
        else:
            rom_obs_chunk, rom_state_next_chunk = _evaluate_rom_batch(
                rom_predict,
                states,
                shocks,
                theta_samples,
                rom_predict_batch=rom_predict_batch,
                rom_predict_is_jax=rom_predict_is_jax,
            )
        arrays = build_surrogate_residual_arrays_from_batched_sep_jax(
            jnp.asarray(states[:, None, :], dtype=jnp.float64),
            jnp.asarray(shocks[:, None, :], dtype=jnp.float64),
            jnp.asarray(theta_samples.T, dtype=jnp.float64),
            rom_obs_chunk[:, None, :],
            rom_state_next_chunk[:, None, :],
            sep_solution,
            observable_indices=observable_indices,
            target_mode=target_mode_norm,
            min_stable_periods=min_stable_periods,
            require_accepted=True,
        )
        chunks.append(arrays)
        residual_norm_parts.append(np.asarray(sep_solution.residual_norm, dtype=np.float64).reshape(-1))
        accepted_parts.append(np.asarray(sep_solution.accepted, dtype=bool).reshape(-1))
        iteration_parts.append(np.asarray(sep_solution.iterations, dtype=np.int64).reshape(-1))

    result = _concatenate_batched_surrogate_arrays(chunks)
    sample_mask = np.asarray(result.sample_mask, dtype=bool).reshape(-1)
    accepted = int(np.count_nonzero(sample_mask))
    residual_norm = np.concatenate(residual_norm_parts) if residual_norm_parts else np.zeros((0,), dtype=np.float64)
    accepted_sep = np.concatenate(accepted_parts) if accepted_parts else np.zeros((0,), dtype=bool)
    iterations = np.concatenate(iteration_parts) if iteration_parts else np.zeros((0,), dtype=np.int64)
    finite_residuals = residual_norm[np.isfinite(residual_norm)]
    diagnostics = {
        "builder": "batched_sep_feature_grid",
        "status": "ok" if accepted >= int(min_successful_samples) else "error",
        "attempted_count": attempted,
        "accepted_samples": accepted,
        "masked_samples": int(sample_mask.size - accepted),
        "sep_accepted_count": int(np.count_nonzero(accepted_sep)),
        "chunk_count": int(len(chunks)),
        "chunk_size": int(effective_chunk_size),
        "params_batched": bool(explicit_params_are_batched),
        "target_mode": target_mode_norm,
        "residual_p50": float(np.percentile(finite_residuals, 50)) if finite_residuals.size else None,
        "residual_p90": float(np.percentile(finite_residuals, 90)) if finite_residuals.size else None,
        "residual_max": float(np.max(finite_residuals)) if finite_residuals.size else None,
        "iterations_max": int(np.max(iterations)) if iterations.size else 0,
    }
    if accepted < int(min_successful_samples):
        raise ValueError(
            "Batched SEP feature-grid target generation produced fewer than "
            f"{int(min_successful_samples)} successful samples. Diagnostics: {diagnostics}"
        )
    return result, diagnostics


def build_surrogate_residual_dataset(
    rom_predict: PredictTupleFn,
    fom_predict: PredictTupleFn,
    initial_state: Any,
    shocks: Any,
    theta_design: ParameterDesign | np.ndarray,
    *,
    target_mode: str = "residual_full",
    samples_per_theta: Optional[int] = None,
    sample_replace: bool = True,
    seed: int = 0,
    min_stable_periods: int = 1,
    input_names: Sequence[str] = (),
    output_names: Sequence[str] = (),
) -> SurrogateDataset:
    theta, theta_names = _theta_matrix(theta_design)
    if theta.shape[1] < 1:
        raise ValueError("theta_design must contain at least one theta draw.")
    shock_cube = _coerce_shocks(shocks, n_theta=theta.shape[1])
    initial_states = _coerce_initial_states(initial_state, n_theta=theta.shape[1])
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")

    X_columns: list[np.ndarray] = []
    Y_columns: list[np.ndarray] = []
    Y_rom_columns: list[np.ndarray] = []
    theta_ids: list[int] = []
    period_ids: list[int] = []
    theta_success = np.zeros((theta.shape[1],), dtype=bool)
    theta_stable_periods = np.zeros((theta.shape[1],), dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    target_mode_norm = _normalize_target_mode(target_mode)
    for theta_idx in range(theta.shape[1]):
        theta_t = theta[:, theta_idx]
        shock_matrix = shock_cube[theta_idx]
        state = initial_states[theta_idx].copy()
        period_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for period in range(shock_matrix.shape[1]):
            shock_t = shock_matrix[:, period]
            try:
                rom_obs, rom_state_next = _call_predict_tuple(
                    rom_predict,
                    state,
                    shock_t,
                    theta_t,
                    label="rom_predict",
                )
                fom_obs, fom_state_next = _call_predict_tuple(
                    fom_predict,
                    state,
                    shock_t,
                    theta_t,
                    label="fom_predict",
                )
            except Exception:
                break
            if rom_obs.shape != fom_obs.shape:
                raise ValueError(f"ROM/FOM observation shape mismatch: {rom_obs.shape} vs {fom_obs.shape}.")
            if rom_state_next.shape != fom_state_next.shape:
                raise ValueError(f"ROM/FOM next-state shape mismatch: {rom_state_next.shape} vs {fom_state_next.shape}.")
            if fom_state_next.shape != state.shape:
                raise ValueError(f"FOM next-state shape {fom_state_next.shape} does not match state shape {state.shape}.")
            y, y_rom = _target_vector(target_mode_norm, fom_obs, fom_state_next, rom_obs, rom_state_next)
            x = np.concatenate([state, shock_t, theta_t])
            period_records.append((x, y, y_rom, period))
            state = fom_state_next

        stable = len(period_records)
        theta_stable_periods[theta_idx] = stable
        theta_success[theta_idx] = stable >= int(min_stable_periods) and stable == shock_matrix.shape[1]
        if stable < int(min_stable_periods):
            continue
        selected = _sample_periods(stable, samples_per_theta, rng, replace=bool(sample_replace))
        for local_idx in selected:
            x, y, y_rom, period = period_records[int(local_idx)]
            X_columns.append(x)
            Y_columns.append(y)
            Y_rom_columns.append(y_rom)
            theta_ids.append(theta_idx)
            period_ids.append(period)

    if not X_columns:
        raise ValueError("No stable surrogate-dataset samples were generated.")
    X = np.column_stack(X_columns)
    Y = np.column_stack(Y_columns)
    Y_rom = np.column_stack(Y_rom_columns)
    return SurrogateDataset(
        X=X,
        Y=Y,
        Y_rom=Y_rom,
        theta=theta,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=theta_success,
        theta_stable_periods=theta_stable_periods,
        target_mode=target_mode_norm,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        theta_names=theta_names,
    )


def build_surrogate_residual_dataset_from_feature_grid(
    rom_predict: PredictTupleFn,
    fom_predict: PredictTupleFn,
    feature_grid: Any,
    *,
    state_dim: int,
    shock_dim: int,
    target_mode: str = "residual_full",
    min_successful_samples: int = 1,
    input_names: Sequence[str] = (),
    output_names: Sequence[str] = (),
    theta_names: Sequence[str] = (),
    max_logged_failures: int = 20,
) -> tuple[SurrogateDataset, dict[str, object]]:
    """Evaluate ROM/FOM targets only at selected feature-grid columns.

    ``feature_grid`` must have columns ``[state; shock; theta]``. This is the
    companion to an endogenous adaptive grid: a cheap scorer chooses relevant
    state/shock/parameter combinations, and this function spends expensive FOM
    or SEP calls only at those columns.
    """

    grid = np.asarray(feature_grid, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"feature_grid must have shape (features, samples), got {grid.shape}.")
    state_count = int(state_dim)
    shock_count = int(shock_dim)
    if state_count < 1:
        raise ValueError(f"state_dim must be positive, got {state_dim}.")
    if shock_count < 0:
        raise ValueError(f"shock_dim must be nonnegative, got {shock_dim}.")
    theta_dim = int(grid.shape[0]) - state_count - shock_count
    if theta_dim < 1:
        raise ValueError(
            "feature_grid must contain at least one theta row after state/shock rows; "
            f"got feature dimension {grid.shape[0]}, state_dim={state_dim}, shock_dim={shock_dim}."
        )
    if grid.shape[1] < 1:
        raise ValueError("feature_grid must contain at least one candidate column.")
    if not np.isfinite(grid).all():
        raise ValueError("feature_grid must be finite.")
    if int(min_successful_samples) < 1:
        raise ValueError(f"min_successful_samples must be >= 1, got {min_successful_samples}.")

    target_mode_norm = _normalize_target_mode(target_mode)
    X_columns: list[np.ndarray] = []
    Y_columns: list[np.ndarray] = []
    Y_rom_columns: list[np.ndarray] = []
    theta_columns: list[np.ndarray] = []
    theta_lookup: dict[tuple[float, ...], int] = {}
    theta_ids: list[int] = []
    period_ids: list[int] = []
    failures: list[dict[str, object]] = []
    attempted = int(grid.shape[1])

    for sample_idx in range(attempted):
        x = grid[:, sample_idx]
        state = x[:state_count]
        shock_t = x[state_count : state_count + shock_count]
        theta_t = x[state_count + shock_count :]
        try:
            rom_obs, rom_state_next = _call_predict_tuple(
                rom_predict,
                state,
                shock_t,
                theta_t,
                label="rom_predict",
            )
            fom_obs, fom_state_next = _call_predict_tuple(
                fom_predict,
                state,
                shock_t,
                theta_t,
                label="fom_predict",
            )
            if rom_obs.shape != fom_obs.shape:
                raise ValueError(f"ROM/FOM observation shape mismatch: {rom_obs.shape} vs {fom_obs.shape}.")
            if rom_state_next.shape != fom_state_next.shape:
                raise ValueError(
                    f"ROM/FOM next-state shape mismatch: {rom_state_next.shape} vs {fom_state_next.shape}."
                )
            if fom_state_next.shape[0] != state_count:
                raise ValueError(
                    f"FOM next-state length {fom_state_next.shape[0]} does not match state_dim={state_count}."
                )
            y, y_rom = _target_vector(target_mode_norm, fom_obs, fom_state_next, rom_obs, rom_state_next)
        except Exception as exc:
            if len(failures) < int(max_logged_failures):
                failures.append({"sample_index": int(sample_idx), "error": repr(exc)})
            continue
        X_columns.append(x.copy())
        Y_columns.append(y)
        Y_rom_columns.append(y_rom)
        theta_key = tuple(float(value) for value in theta_t)
        theta_id = theta_lookup.get(theta_key)
        if theta_id is None:
            theta_id = len(theta_columns)
            theta_lookup[theta_key] = theta_id
            theta_columns.append(theta_t.copy())
        theta_ids.append(theta_id)
        period_ids.append(0)

    if len(X_columns) < int(min_successful_samples):
        diagnostics = {
            "builder": "feature_grid",
            "status": "error",
            "attempted_count": attempted,
            "accepted_samples": int(len(X_columns)),
            "failure_log": failures,
        }
        raise ValueError(
            "Feature-grid target generation produced fewer than "
            f"{int(min_successful_samples)} successful samples. Diagnostics: {diagnostics}"
        )

    theta_array = np.column_stack(theta_columns)
    accepted = int(len(X_columns))
    n_theta = int(theta_array.shape[1])
    theta_name_tuple = (
        tuple(str(name) for name in theta_names)
        if theta_names
        else tuple(f"theta_{idx}" for idx in range(theta_dim))
    )
    dataset = SurrogateDataset(
        X=np.column_stack(X_columns),
        Y=np.column_stack(Y_columns),
        Y_rom=np.column_stack(Y_rom_columns),
        theta=theta_array,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=np.ones((n_theta,), dtype=bool),
        theta_stable_periods=np.ones((n_theta,), dtype=np.int64),
        target_mode=target_mode_norm,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        theta_names=theta_name_tuple,
    )
    diagnostics = {
        "builder": "feature_grid",
        "status": "ok",
        "attempted_count": attempted,
        "accepted_samples": accepted,
        "unique_theta_count": n_theta,
        "failure_count": int(attempted - accepted),
        "failure_log": failures,
    }
    return dataset, diagnostics


def build_surrogate_residual_dataset_from_batched_rollouts(
    states: Any,
    shocks: Any,
    theta_design: ParameterDesign | np.ndarray,
    rom_obs: Any,
    rom_state_next: Any,
    fom_obs: Any,
    fom_state_next: Any,
    *,
    target_mode: str = "residual_full",
    samples_per_theta: Optional[int] = None,
    sample_replace: bool = True,
    seed: int = 0,
    min_stable_periods: int = 1,
    input_names: Sequence[str] = (),
    output_names: Sequence[str] = (),
) -> SurrogateDataset:
    """Build a surrogate dataset from batched rollout tensors.

    All rollout tensors must use explicit shape ``(n_theta, periods, dim)``.
    This matches JAX ``vmap`` over theta draws plus ``lax.scan`` over time
    after moving the scan axis behind the theta axis. Non-finite rollout rows
    terminate only that theta draw's stable prefix, mirroring the sequential
    builder's exception-driven prefix behavior without per-period Python calls.
    """

    theta, theta_names = _theta_matrix(theta_design)
    if theta.shape[1] < 1:
        raise ValueError("theta_design must contain at least one theta draw.")
    if not np.isfinite(theta).all():
        raise ValueError("theta_design contains non-finite values.")
    if int(min_stable_periods) < 0:
        raise ValueError(f"min_stable_periods must be nonnegative, got {min_stable_periods}.")

    n_theta = int(theta.shape[1])
    states_array = _as_batched_rollout_tensor(states, label="states", n_theta=n_theta)
    periods = int(states_array.shape[1])
    state_dim = int(states_array.shape[2])
    shocks_array = _as_batched_rollout_tensor(shocks, label="shocks", n_theta=n_theta, periods=periods)
    rom_obs_array = _as_batched_rollout_tensor(rom_obs, label="rom_obs", n_theta=n_theta, periods=periods)
    fom_obs_array = _as_batched_rollout_tensor(
        fom_obs,
        label="fom_obs",
        n_theta=n_theta,
        periods=periods,
        dim=int(rom_obs_array.shape[2]),
    )
    rom_state_next_array = _as_batched_rollout_tensor(
        rom_state_next,
        label="rom_state_next",
        n_theta=n_theta,
        periods=periods,
        dim=state_dim,
    )
    fom_state_next_array = _as_batched_rollout_tensor(
        fom_state_next,
        label="fom_state_next",
        n_theta=n_theta,
        periods=periods,
        dim=state_dim,
    )

    target_mode_norm = _normalize_target_mode(target_mode)
    finite_by_period = _finite_period_mask(
        states_array,
        shocks_array,
        rom_obs_array,
        rom_state_next_array,
        fom_obs_array,
        fom_state_next_array,
    )
    theta_stable_periods = _stable_prefix_lengths(finite_by_period)
    theta_success = (theta_stable_periods >= int(min_stable_periods)) & (theta_stable_periods == periods)

    X_columns: list[np.ndarray] = []
    Y_columns: list[np.ndarray] = []
    Y_rom_columns: list[np.ndarray] = []
    theta_ids: list[int] = []
    period_ids: list[int] = []
    rng = np.random.default_rng(int(seed))

    for theta_idx in range(n_theta):
        stable = int(theta_stable_periods[theta_idx])
        if stable < int(min_stable_periods):
            continue
        selected = _sample_periods(stable, samples_per_theta, rng, replace=bool(sample_replace))
        theta_t = theta[:, theta_idx]
        for period in selected:
            period_idx = int(period)
            y, y_rom = _target_vector(
                target_mode_norm,
                fom_obs_array[theta_idx, period_idx],
                fom_state_next_array[theta_idx, period_idx],
                rom_obs_array[theta_idx, period_idx],
                rom_state_next_array[theta_idx, period_idx],
            )
            x = np.concatenate([states_array[theta_idx, period_idx], shocks_array[theta_idx, period_idx], theta_t])
            X_columns.append(x)
            Y_columns.append(y)
            Y_rom_columns.append(y_rom)
            theta_ids.append(theta_idx)
            period_ids.append(period_idx)

    if not X_columns:
        raise ValueError("No stable surrogate-dataset samples were generated.")
    return SurrogateDataset(
        X=np.column_stack(X_columns),
        Y=np.column_stack(Y_columns),
        Y_rom=np.column_stack(Y_rom_columns),
        theta=theta,
        theta_ids=np.asarray(theta_ids, dtype=np.int64),
        period_ids=np.asarray(period_ids, dtype=np.int64),
        theta_success=theta_success,
        theta_stable_periods=theta_stable_periods,
        target_mode=target_mode_norm,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        theta_names=theta_names,
    )


def summarize_surrogate_dataset(dataset: SurrogateDataset) -> dict[str, object]:
    theta_counts = np.bincount(dataset.theta_ids, minlength=dataset.n_theta)
    return {
        "n_samples": dataset.n_samples,
        "n_theta": dataset.n_theta,
        "input_dim": int(dataset.X.shape[0]),
        "output_dim": int(dataset.Y.shape[0]),
        "target_mode": dataset.target_mode,
        "theta_success_rate": float(np.mean(dataset.theta_success)) if dataset.theta_success.size else 0.0,
        "min_stable_periods": int(np.min(dataset.theta_stable_periods)) if dataset.theta_stable_periods.size else 0,
        "max_stable_periods": int(np.max(dataset.theta_stable_periods)) if dataset.theta_stable_periods.size else 0,
        "samples_per_theta": dict(zip(range(dataset.n_theta), theta_counts.astype(int))),
        "Y_rmse_vs_rom": float(np.sqrt(np.mean((dataset.Y - dataset.Y_rom) ** 2))) if dataset.Y_rom is not None else None,
    }


def summarize_batched_surrogate_arrays(arrays: BatchedSurrogateRolloutArrays) -> dict[str, object]:
    """Summarize fixed-shape JAX rollout arrays without compacting samples."""

    mask = np.asarray(arrays.sample_mask, dtype=bool).reshape(-1)
    theta_success = np.asarray(arrays.theta_success, dtype=bool).reshape(-1)
    theta_stable_periods = np.asarray(arrays.theta_stable_periods, dtype=np.int64).reshape(-1)
    theta_ids = np.asarray(arrays.theta_ids, dtype=np.int64).reshape(-1)
    n_theta = int(arrays.theta.shape[1])
    theta_counts = np.bincount(theta_ids[mask], minlength=n_theta)
    return {
        "n_samples_total": int(arrays.X.shape[1]),
        "n_samples_valid": int(np.count_nonzero(mask)),
        "n_samples_masked": int(mask.size - np.count_nonzero(mask)),
        "n_theta": n_theta,
        "input_dim": int(arrays.X.shape[0]),
        "output_dim": int(arrays.Y.shape[0]),
        "theta_success_rate": float(np.mean(theta_success)) if theta_success.size else 0.0,
        "min_stable_periods": int(np.min(theta_stable_periods)) if theta_stable_periods.size else 0,
        "max_stable_periods": int(np.max(theta_stable_periods)) if theta_stable_periods.size else 0,
        "valid_samples_per_theta": dict(zip(range(n_theta), theta_counts.astype(int))),
        "Y_rmse_vs_rom_valid": float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(arrays.Y[:, mask], dtype=np.float64)
                        - np.asarray(arrays.Y_rom[:, mask], dtype=np.float64)
                    )
                    ** 2
                )
            )
        )
        if np.count_nonzero(mask)
        else None,
    }
