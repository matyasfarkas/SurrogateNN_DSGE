from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    AdaptiveGridConfig,
    endogenous_adaptive_grid,
    summarize_adaptive_grid,
)


def test_endogenous_adaptive_grid_concentrates_near_relevant_region() -> None:
    center = np.asarray([0.82, 0.18], dtype=np.float64)

    def score(points: np.ndarray) -> np.ndarray:
        return -np.sum((points - center[:, None]) ** 2, axis=0)

    initial = np.asarray(
        [
            [0.10, 0.45, 0.68],
            [0.90, 0.45, 0.32],
        ],
        dtype=np.float64,
    )
    initial_best = float(np.max(score(initial)))
    result = endogenous_adaptive_grid(
        score,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        names=("state_gap", "rho"),
        initial_points=initial,
        config=AdaptiveGridConfig(
            n_initial=8,
            n_rounds=3,
            candidates_per_round=96,
            keep=8,
            final_size=6,
            local_scale=0.18,
            scale_decay=0.50,
            exploration_fraction=0.05,
            seed=123,
        ),
    )

    assert result.points.shape == (2, 6)
    assert result.names == ("state_gap", "rho")
    assert float(result.scores[0]) >= initial_best
    assert float(np.linalg.norm(result.points[:, 0] - center)) < 0.06
    assert all(entry["stage"] in {"initial", "adaptive"} for entry in result.history)

    summary = summarize_adaptive_grid(result)
    assert summary["n_samples"] == 6
    assert summary["best_score"] == pytest.approx(float(result.scores[0]))


def test_endogenous_adaptive_grid_rejects_unscorable_design() -> None:
    with pytest.raises(ValueError, match="no finite scores"):
        endogenous_adaptive_grid(
            lambda points: np.full((points.shape[1],), np.nan),
            lower=[0.0],
            upper=[1.0],
            config=AdaptiveGridConfig(n_initial=4),
        )


def test_endogenous_adaptive_grid_validates_initial_points_bounds() -> None:
    with pytest.raises(ValueError, match="inside adaptive grid bounds"):
        endogenous_adaptive_grid(
            lambda points: np.zeros((points.shape[1],)),
            lower=[0.0, 0.0],
            upper=[1.0, 1.0],
            initial_points=np.asarray([[1.2], [0.5]], dtype=np.float64),
        )
