from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np


ScoreFn = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class AdaptiveGridConfig:
    """Controls endogenous adaptive sampling in a bounded feature space."""

    n_initial: int = 64
    n_rounds: int = 2
    candidates_per_round: int = 128
    keep: int = 32
    final_size: Optional[int] = None
    local_scale: float = 0.20
    scale_decay: float = 0.55
    exploration_fraction: float = 0.10
    deduplicate_tol: float = 1e-10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.n_initial) < 1:
            raise ValueError(f"n_initial must be >= 1, got {self.n_initial}.")
        if int(self.n_rounds) < 0:
            raise ValueError(f"n_rounds must be >= 0, got {self.n_rounds}.")
        if int(self.candidates_per_round) < 1:
            raise ValueError(
                f"candidates_per_round must be >= 1, got {self.candidates_per_round}."
            )
        if int(self.keep) < 1:
            raise ValueError(f"keep must be >= 1, got {self.keep}.")
        if self.final_size is not None and int(self.final_size) < 1:
            raise ValueError(f"final_size must be >= 1 when provided, got {self.final_size}.")
        if not np.isfinite(float(self.local_scale)) or float(self.local_scale) <= 0.0:
            raise ValueError(f"local_scale must be positive, got {self.local_scale}.")
        if not np.isfinite(float(self.scale_decay)) or float(self.scale_decay) <= 0.0:
            raise ValueError(f"scale_decay must be positive, got {self.scale_decay}.")
        if (
            not np.isfinite(float(self.exploration_fraction))
            or float(self.exploration_fraction) < 0.0
            or float(self.exploration_fraction) > 1.0
        ):
            raise ValueError(
                "exploration_fraction must lie in [0, 1], "
                f"got {self.exploration_fraction}."
            )
        if not np.isfinite(float(self.deduplicate_tol)) or float(self.deduplicate_tol) < 0.0:
            raise ValueError(f"deduplicate_tol must be nonnegative, got {self.deduplicate_tol}.")


@dataclass(frozen=True)
class AdaptiveGridResult:
    points: np.ndarray
    scores: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...]
    method: str
    history: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        lower = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        names = tuple(str(name) for name in self.names)
        if points.ndim != 2:
            raise ValueError(f"points must be rank-2 with shape (dim, samples), got {points.shape}.")
        if scores.shape[0] != points.shape[1]:
            raise ValueError(f"scores length {scores.shape[0]} does not match point count {points.shape[1]}.")
        if lower.shape != upper.shape or lower.shape[0] != points.shape[0]:
            raise ValueError("lower/upper bounds must match the point feature dimension.")
        if len(names) != points.shape[0]:
            raise ValueError(f"names length {len(names)} does not match point dimension {points.shape[0]}.")
        if not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise ValueError("AdaptiveGridResult points and scores must be finite.")
        if not np.all(lower < upper):
            raise ValueError("AdaptiveGridResult bounds must satisfy lower < upper.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "history", tuple(dict(item) for item in self.history))

    @property
    def n_samples(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_dim(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class EndogenousSupportSelectionConfig:
    """Select sampler-visited features for expensive SEP relabeling."""

    selection: str = "gate_ood"
    max_points: Optional[int] = None
    repeat_active: int = 1
    ood_z_threshold: float = 4.0

    def __post_init__(self) -> None:
        mode = str(self.selection)
        if mode not in {"gate", "ood", "gate_ood", "all", "worst"}:
            raise ValueError(
                "selection must be one of 'gate', 'ood', 'gate_ood', 'all', or 'worst'."
            )
        if self.max_points is not None and int(self.max_points) < 1:
            raise ValueError(f"max_points must be >= 1 when provided, got {self.max_points}.")
        if int(self.repeat_active) < 1:
            raise ValueError(f"repeat_active must be >= 1, got {self.repeat_active}.")
        if not np.isfinite(float(self.ood_z_threshold)) or float(self.ood_z_threshold) <= 0.0:
            raise ValueError(f"ood_z_threshold must be positive, got {self.ood_z_threshold}.")


@dataclass(frozen=True)
class EndogenousSupportSelectionResult:
    """Gate/OOD-selected feature columns to label with SEP."""

    points: np.ndarray
    selected_indices: np.ndarray
    base_selected_indices: np.ndarray
    gate_mask: np.ndarray
    ood_mask: np.ndarray
    max_z: np.ndarray
    scores: Optional[np.ndarray]
    selection: str
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        selected = np.asarray(self.selected_indices, dtype=np.int64).reshape(-1)
        base = np.asarray(self.base_selected_indices, dtype=np.int64).reshape(-1)
        gate = np.asarray(self.gate_mask, dtype=bool).reshape(-1)
        ood = np.asarray(self.ood_mask, dtype=bool).reshape(-1)
        max_z = np.asarray(self.max_z, dtype=np.float64).reshape(-1)
        scores = None if self.scores is None else np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if points.ndim != 2:
            raise ValueError(f"points must be rank-2 with shape (dim, samples), got {points.shape}.")
        if selected.shape[0] != points.shape[1]:
            raise ValueError("selected_indices length must match selected point count.")
        n_candidates = gate.shape[0]
        if ood.shape[0] != n_candidates or max_z.shape[0] != n_candidates:
            raise ValueError("gate_mask, ood_mask, and max_z must have the same length.")
        if scores is not None and scores.shape[0] != n_candidates:
            raise ValueError("scores length must match gate_mask length.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(self, "base_selected_indices", base)
        object.__setattr__(self, "gate_mask", gate)
        object.__setattr__(self, "ood_mask", ood)
        object.__setattr__(self, "max_z", max_z)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "selection", str(self.selection))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def n_samples(self) -> int:
        return int(self.points.shape[1])


def _coerce_bounds(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
    lower_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
    if lower_arr.shape != upper_arr.shape:
        raise ValueError(f"lower/upper shape mismatch: {lower_arr.shape} vs {upper_arr.shape}.")
    if lower_arr.size < 1:
        raise ValueError("At least one feature bound is required.")
    if not np.isfinite(lower_arr).all() or not np.isfinite(upper_arr).all():
        raise ValueError("Adaptive grid bounds must be finite.")
    if not np.all(lower_arr < upper_arr):
        raise ValueError("Adaptive grid bounds must satisfy lower < upper.")
    return lower_arr, upper_arr


def _coerce_points(points: Any, *, dim: int, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(dim, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must have shape (dim, samples), got {array.shape}.")
    if array.shape[0] != int(dim):
        raise ValueError(f"{label} row count {array.shape[0]} does not match dim={dim}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array


def _coerce_feature_matrix(features: Any, *, label: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{label} must have shape (dim, samples), got {array.shape}.")
    if array.shape[0] < 1:
        raise ValueError(f"{label} must contain at least one feature row.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array


def _input_norm_stats(norm_stats: Any, dim: int) -> tuple[np.ndarray, np.ndarray]:
    def read(keys: tuple[str, ...], *, label: str) -> np.ndarray:
        if isinstance(norm_stats, dict):
            for key in keys:
                if key in norm_stats:
                    return np.asarray(norm_stats[key], dtype=np.float64).reshape(-1)
        else:
            for key in keys:
                if hasattr(norm_stats, key):
                    return np.asarray(getattr(norm_stats, key), dtype=np.float64).reshape(-1)
        raise ValueError(f"norm_stats is missing {label}.")

    mu = read(("mu_x", "muX", "x_mean", "input_mean", "mean", "mu", "μX"), label="input mean")
    sigma = read(("sigma_x", "sigmaX", "x_std", "input_std", "std", "sigma", "σX"), label="input std")
    if mu.shape != (dim,) or sigma.shape != (dim,):
        raise ValueError(f"norm_stats input dimension mismatch: got {mu.shape}/{sigma.shape}, expected ({dim},).")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("norm_stats input mean/std must be finite and std must be strictly positive.")
    return mu, sigma


def _rank_selected(
    indices: np.ndarray,
    *,
    scores: Optional[np.ndarray],
    max_z: np.ndarray,
    max_points: Optional[int],
    selection: str,
) -> np.ndarray:
    if indices.size == 0:
        return indices
    if scores is not None:
        rank_values = scores[indices]
    elif selection == "worst":
        rank_values = max_z[indices]
    elif np.isfinite(max_z[indices]).any():
        rank_values = max_z[indices]
    else:
        rank_values = -np.arange(indices.size, dtype=np.float64)
    order = np.argsort(rank_values)[::-1]
    ordered = indices[order]
    if max_points is not None:
        ordered = ordered[: min(int(max_points), ordered.size)]
    return ordered


def select_endogenous_support_points(
    features: Any,
    *,
    gate_mask: Optional[Sequence[bool] | np.ndarray] = None,
    norm_stats: Optional[Any] = None,
    max_z: Optional[Sequence[float] | np.ndarray] = None,
    scores: Optional[Sequence[float] | np.ndarray] = None,
    config: EndogenousSupportSelectionConfig = EndogenousSupportSelectionConfig(),
) -> EndogenousSupportSelectionResult:
    """Select active support points for posterior-guided SEP target generation.

    This mirrors the Julia active-support modes used by the HLT workflow:
    select by switching gate, by NN-support OOD diagnostics, their union, all
    points, or the worst-scored/OOD points. Repeating active points is a simple
    way to weight nonlinear periods more heavily during residual training.
    """

    X = _coerce_feature_matrix(features)
    n_samples = int(X.shape[1])
    if gate_mask is None:
        gate = np.zeros((n_samples,), dtype=bool)
    else:
        gate = np.asarray(gate_mask, dtype=bool).reshape(-1)
        if gate.shape[0] != n_samples:
            raise ValueError(f"gate_mask length mismatch: {gate.shape[0]} vs {n_samples} samples.")

    if max_z is None:
        if norm_stats is None:
            max_z_values = np.zeros((n_samples,), dtype=np.float64)
            ood = np.zeros((n_samples,), dtype=bool)
        else:
            mu, sigma = _input_norm_stats(norm_stats, X.shape[0])
            z = np.abs((X - mu[:, None]) / sigma[:, None])
            max_z_values = np.max(z, axis=0)
            ood = max_z_values > float(config.ood_z_threshold)
    else:
        max_z_values = np.asarray(max_z, dtype=np.float64).reshape(-1)
        if max_z_values.shape[0] != n_samples:
            raise ValueError(f"max_z length mismatch: {max_z_values.shape[0]} vs {n_samples} samples.")
        if not np.isfinite(max_z_values).all():
            raise ValueError("max_z must be finite.")
        ood = max_z_values > float(config.ood_z_threshold)

    score_values: Optional[np.ndarray]
    if scores is None:
        score_values = None
    else:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_values.shape[0] != n_samples:
            raise ValueError(f"scores length mismatch: {score_values.shape[0]} vs {n_samples} samples.")
        if not np.isfinite(score_values).all():
            raise ValueError("scores must be finite.")

    mode = str(config.selection)
    if mode == "gate":
        selected_mask = gate
    elif mode == "ood":
        selected_mask = ood
    elif mode == "gate_ood":
        selected_mask = gate | ood
    elif mode == "all":
        selected_mask = np.ones((n_samples,), dtype=bool)
    else:
        selected_mask = np.ones((n_samples,), dtype=bool)

    base_indices = np.flatnonzero(selected_mask)
    if mode == "worst" and base_indices.size == 0:
        base_indices = np.arange(n_samples, dtype=np.int64)
    base_indices = _rank_selected(
        base_indices.astype(np.int64, copy=False),
        scores=score_values,
        max_z=max_z_values,
        max_points=config.max_points,
        selection=mode,
    )
    repeated_indices = np.repeat(base_indices, int(config.repeat_active))
    points = X[:, repeated_indices] if repeated_indices.size else np.zeros((X.shape[0], 0), dtype=np.float64)
    diagnostics = {
        "selection": mode,
        "candidate_count": n_samples,
        "gate_count": int(np.count_nonzero(gate)),
        "ood_count": int(np.count_nonzero(ood)),
        "base_selected_count": int(base_indices.size),
        "selected_count": int(repeated_indices.size),
        "repeat_active": int(config.repeat_active),
        "ood_z_threshold": float(config.ood_z_threshold),
        "max_points": None if config.max_points is None else int(config.max_points),
        "max_z_selected_max": (
            None if base_indices.size == 0 else float(np.max(max_z_values[base_indices]))
        ),
    }
    return EndogenousSupportSelectionResult(
        points=points,
        selected_indices=repeated_indices,
        base_selected_indices=base_indices,
        gate_mask=gate,
        ood_mask=ood,
        max_z=max_z_values,
        scores=score_values,
        selection=mode,
        diagnostics=diagnostics,
    )


def _lhs_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    sample = np.empty((int(n_dim), int(n_samples)), dtype=np.float64)
    for dim in range(int(n_dim)):
        order = rng.permutation(int(n_samples))
        sample[dim, :] = (order + rng.random(int(n_samples))) / float(n_samples)
    return sample


def _to_unit(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (points - lower[:, None]) / (upper - lower)[:, None]


def _from_unit(unit_points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower[:, None] + (upper - lower)[:, None] * unit_points


def _deduplicate_against(
    candidate_points: np.ndarray,
    existing_points: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    if candidate_points.shape[1] == 0:
        return candidate_points
    if tol <= 0.0:
        return candidate_points
    existing_keys = {
        tuple(row)
        for row in np.round(existing_points.T / float(tol)).astype(np.int64)
    }
    keep: list[int] = []
    local_keys: set[tuple[int, ...]] = set()
    candidate_keys = np.round(candidate_points.T / float(tol)).astype(np.int64)
    for idx, key_array in enumerate(candidate_keys):
        key = tuple(int(value) for value in key_array)
        if key in existing_keys or key in local_keys:
            continue
        keep.append(idx)
        local_keys.add(key)
    if not keep:
        return np.zeros((candidate_points.shape[0], 0), dtype=np.float64)
    return candidate_points[:, np.asarray(keep, dtype=np.int64)]


def _evaluate_scores(score_fn: ScoreFn, points: np.ndarray) -> tuple[np.ndarray, int]:
    raw = np.asarray(score_fn(points), dtype=np.float64).reshape(-1)
    if raw.shape[0] != points.shape[1]:
        raise ValueError(
            f"score_fn returned {raw.shape[0]} scores for {points.shape[1]} candidate points."
        )
    nonfinite = ~np.isfinite(raw)
    scores = raw.copy()
    scores[nonfinite] = -np.inf
    return scores, int(np.count_nonzero(nonfinite))


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        raise ValueError("Adaptive grid score_fn produced no finite scores.")
    ordered = finite[np.argsort(scores[finite])[::-1]]
    return ordered[: min(int(count), ordered.size)]


def endogenous_adaptive_grid(
    score_fn: ScoreFn,
    lower: Any,
    upper: Any,
    *,
    names: Optional[Sequence[str]] = None,
    initial_points: Optional[Any] = None,
    config: AdaptiveGridConfig = AdaptiveGridConfig(),
) -> AdaptiveGridResult:
    """Build an endogenous adaptive grid in a bounded feature space.

    ``score_fn`` should be cheap relative to SEP/FOM evaluation. Typical scores
    are ROM posterior density, linear-filter state likelihood, switching-gate
    probability, or a pilot residual/error indicator. The returned points are
    the high-score locations where expensive nonlinear SEP targets should be
    evaluated.
    """

    lower_arr, upper_arr = _coerce_bounds(lower, upper)
    dim = int(lower_arr.shape[0])
    feature_names = tuple(f"x{i}" for i in range(dim)) if names is None else tuple(str(name) for name in names)
    if len(feature_names) != dim:
        raise ValueError(f"names length {len(feature_names)} does not match feature dimension {dim}.")
    rng = np.random.default_rng(config.seed)

    if initial_points is None:
        unit_initial = _lhs_unit(int(config.n_initial), dim, rng)
        points = _from_unit(unit_initial, lower_arr, upper_arr)
    else:
        anchor_points = _coerce_points(initial_points, dim=dim, label="initial_points")
        if np.any(anchor_points < lower_arr[:, None]) or np.any(anchor_points > upper_arr[:, None]):
            raise ValueError("initial_points must lie inside adaptive grid bounds.")
        if anchor_points.shape[1] >= int(config.n_initial):
            points = anchor_points
        else:
            missing = int(config.n_initial) - int(anchor_points.shape[1])
            global_points = _from_unit(_lhs_unit(missing, dim, rng), lower_arr, upper_arr)
            points = np.column_stack([anchor_points, global_points])

    scores, nonfinite_count = _evaluate_scores(score_fn, points)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("Adaptive grid score_fn produced no finite scores.")
    history: list[dict[str, object]] = [
        {
            "round": 0,
            "stage": "initial",
            "candidate_count": int(points.shape[1]),
            "accepted_count": int(points.shape[1]),
            "nonfinite_score_count": int(nonfinite_count),
            "best_score": float(np.max(finite_scores)),
            "median_score": float(np.median(finite_scores)),
        }
    ]

    for round_idx in range(1, int(config.n_rounds) + 1):
        top = _top_indices(scores, int(config.keep))
        round_candidates = int(config.candidates_per_round)
        global_count = int(round(round_candidates * float(config.exploration_fraction)))
        global_count = min(max(global_count, 0), round_candidates)
        local_count = round_candidates - global_count
        radius = float(config.local_scale) * (float(config.scale_decay) ** float(round_idx - 1))

        candidate_parts: list[np.ndarray] = []
        if local_count > 0:
            parent_ranks = np.arange(top.size, 0, -1, dtype=np.float64)
            parent_prob = parent_ranks / np.sum(parent_ranks)
            parent_ids = rng.choice(top, size=local_count, replace=True, p=parent_prob)
            parent_unit = _to_unit(points[:, parent_ids], lower_arr, upper_arr)
            jitter = rng.normal(loc=0.0, scale=radius, size=(dim, local_count))
            local_unit = np.clip(parent_unit + jitter, 0.0, 1.0)
            candidate_parts.append(_from_unit(local_unit, lower_arr, upper_arr))
        if global_count > 0:
            candidate_parts.append(_from_unit(_lhs_unit(global_count, dim, rng), lower_arr, upper_arr))
        candidates = np.column_stack(candidate_parts) if candidate_parts else np.zeros((dim, 0), dtype=np.float64)
        candidates = _deduplicate_against(candidates, points, tol=float(config.deduplicate_tol))
        if candidates.shape[1] == 0:
            history.append(
                {
                    "round": int(round_idx),
                    "stage": "adaptive",
                    "candidate_count": int(round_candidates),
                    "accepted_count": 0,
                    "local_candidate_count": int(local_count),
                    "global_candidate_count": int(global_count),
                    "radius": float(radius),
                    "nonfinite_score_count": 0,
                    "best_score": float(np.max(scores[np.isfinite(scores)])),
                    "median_score": float(np.median(scores[np.isfinite(scores)])),
                }
            )
            continue

        candidate_scores, candidate_nonfinite = _evaluate_scores(score_fn, candidates)
        points = np.column_stack([points, candidates])
        scores = np.concatenate([scores, candidate_scores])
        finite_scores = scores[np.isfinite(scores)]
        history.append(
            {
                "round": int(round_idx),
                "stage": "adaptive",
                "candidate_count": int(round_candidates),
                "accepted_count": int(candidates.shape[1]),
                "local_candidate_count": int(local_count),
                "global_candidate_count": int(global_count),
                "radius": float(radius),
                "nonfinite_score_count": int(candidate_nonfinite),
                "best_score": float(np.max(finite_scores)),
                "median_score": float(np.median(finite_scores)),
            }
        )

    final_count = int(config.keep if config.final_size is None else config.final_size)
    selected = _top_indices(scores, final_count)
    selected_scores = scores[selected]
    order = np.argsort(selected_scores)[::-1]
    selected = selected[order]
    return AdaptiveGridResult(
        points=points[:, selected],
        scores=scores[selected],
        lower=lower_arr,
        upper=upper_arr,
        names=feature_names,
        method="endogenous_adaptive_grid",
        history=tuple(history),
    )


def summarize_adaptive_grid(result: AdaptiveGridResult) -> dict[str, object]:
    points = np.asarray(result.points, dtype=np.float64)
    return {
        "method": result.method,
        "n_dim": result.n_dim,
        "n_samples": result.n_samples,
        "names": result.names,
        "best_score": float(np.max(result.scores)) if result.scores.size else None,
        "worst_selected_score": float(np.min(result.scores)) if result.scores.size else None,
        "mean_selected_score": float(np.mean(result.scores)) if result.scores.size else None,
        "min": dict(zip(result.names, np.min(points, axis=1))),
        "max": dict(zip(result.names, np.max(points, axis=1))),
        "history": result.history,
    }
