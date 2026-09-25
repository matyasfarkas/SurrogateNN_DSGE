from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    AdaptiveGridConfig,
    EndogenousSupportSelectionConfig,
    NormStats,
    endogenous_adaptive_grid,
    select_endogenous_support_points,
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


def test_select_endogenous_support_points_uses_gate_ood_union_and_repeat_weighting() -> None:
    features = np.asarray(
        [
            [0.0, 1.0, 5.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    norm = NormStats(
        mu_x=np.zeros((2,), dtype=np.float64),
        sigma_x=np.ones((2,), dtype=np.float64),
        mu_y=np.zeros((1,), dtype=np.float64),
        sigma_y=np.ones((1,), dtype=np.float64),
    )

    result = select_endogenous_support_points(
        features,
        gate_mask=[False, True, False, False],
        norm_stats=norm,
        config=EndogenousSupportSelectionConfig(
            selection="gate_ood",
            repeat_active=2,
            ood_z_threshold=4.0,
        ),
    )

    assert set(result.base_selected_indices.tolist()) == {1, 2}
    assert result.selected_indices.tolist().count(1) == 2
    assert result.selected_indices.tolist().count(2) == 2
    assert result.points.shape == (2, 4)
    assert result.diagnostics["gate_count"] == 1
    assert result.diagnostics["ood_count"] == 1
    assert result.diagnostics["selected_count"] == 4


def test_select_endogenous_support_points_worst_mode_uses_scores_budget() -> None:
    features = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    result = select_endogenous_support_points(
        features,
        scores=[0.0, 4.0, 2.0, 9.0],
        config=EndogenousSupportSelectionConfig(
            selection="worst",
            max_points=2,
        ),
    )

    np.testing.assert_array_equal(result.base_selected_indices, np.asarray([3, 1], dtype=np.int64))
    np.testing.assert_allclose(result.points, features[:, [3, 1]], rtol=0, atol=0)
