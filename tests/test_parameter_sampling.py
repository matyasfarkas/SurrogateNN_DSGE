from __future__ import annotations

import numpy as np
import pytest

from surrogatenn_dsge import (
    latin_hypercube_unit,
    lhs_to_bounds,
    parameter_grid,
    sample_lhs_parameters,
    sample_parameter_design,
    summarize_parameter_design,
)


def test_lhs_to_bounds_uses_julia_column_sample_orientation() -> None:
    unit = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.5],
            [1.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )

    theta = lhs_to_bounds(unit, "legacy_3params")

    np.testing.assert_allclose(
        theta,
        np.asarray(
            [
                [0.5, 0.725, 0.95],
                [0.255, 0.745, 0.5],
                [150.0, 20.0, 85.0],
            ],
            dtype=np.float64,
        ),
        rtol=0,
        atol=1e-12,
    )


def test_latin_hypercube_unit_has_one_draw_per_stratum_per_dimension() -> None:
    n_samples = 32
    n_dim = 18
    unit = latin_hypercube_unit(n_samples, n_dim, seed=123)

    assert unit.shape == (n_dim, n_samples)
    assert np.all(unit >= 0.0)
    assert np.all(unit < 1.0)
    for row in unit:
        strata = np.floor(row * n_samples).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(n_samples))


def test_sample_lhs_parameters_bounds_and_summary_for_phase1() -> None:
    design = sample_lhs_parameters("phase1_18params", 100, seed=7)

    assert design.theta.shape == (18, 100)
    assert design.unit_sample is not None
    assert np.all(design.theta >= design.lower[:, None])
    assert np.all(design.theta <= design.upper[:, None])

    summary = summarize_parameter_design(design)
    assert summary["n_parameters"] == 18
    assert summary["n_samples"] == 100
    assert summary["min_coverage"] > 0.90
    assert summary["mean_abs_corr"] < 0.15


def test_parameter_grid_matches_julia_legacy_nested_loop_order() -> None:
    design = parameter_grid("legacy_3params", points_per_dim=5)

    assert design.theta.shape == (3, 125)
    np.testing.assert_allclose(design.theta[:, 0], np.asarray([0.5, 0.01, 20.0]), rtol=0, atol=0)
    np.testing.assert_allclose(design.theta[:, 1], np.asarray([0.5, 0.01, 52.5]), rtol=0, atol=0)
    np.testing.assert_allclose(design.theta[:, 5], np.asarray([0.5, 0.255, 20.0]), rtol=0, atol=0)
    np.testing.assert_allclose(design.theta[:, -1], np.asarray([0.95, 0.99, 150.0]), rtol=0, atol=0)


def test_parameter_grid_refuses_accidental_high_dimensional_explosion() -> None:
    with pytest.raises(ValueError, match="exceeding max_points"):
        parameter_grid("phase1_18params", points_per_dim=5)


def test_sample_parameter_design_dispatch_and_validation() -> None:
    lhs_design = sample_parameter_design("legacy_3params", method="lhs", n_samples=4, seed=1, centered=True)
    grid_design = sample_parameter_design("legacy_3params", method="grid", points_per_dim=2)

    assert lhs_design.theta.shape == (3, 4)
    assert grid_design.theta.shape == (3, 8)
    with pytest.raises(ValueError, match="n_samples is required"):
        sample_parameter_design("legacy_3params", method="lhs")
    with pytest.raises(ValueError, match="Unknown parameter-design method"):
        sample_parameter_design("legacy_3params", method="prior", n_samples=4)
