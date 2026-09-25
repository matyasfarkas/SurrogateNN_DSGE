from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .parameter_sampling import ParameterDesign
from .sep import BatchedSEPSolution


PredictTupleFn = Callable[[Any, Any, Any], tuple[Any, Any]]
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
