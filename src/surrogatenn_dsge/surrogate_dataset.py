from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .parameter_sampling import ParameterDesign


PredictTupleFn = Callable[[Any, Any, Any], tuple[Any, Any]]


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
        raise ValueError(
            "target_mode must be one of 'residual_obs', 'residual_full', "
            f"'fom_obs', or 'fom_full', got {target_mode!r}."
        )
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
    state0 = _as_vector(initial_state, label="initial_state")
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

    target_mode_norm = str(target_mode).strip().lower()
    for theta_idx in range(theta.shape[1]):
        theta_t = theta[:, theta_idx]
        shock_matrix = shock_cube[theta_idx]
        state = state0.copy()
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
