from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    SEPConfig,
    gauss_hermite_rule,
    solve_stochastic_extended_path,
    solve_stochastic_extended_path_residual_expectation,
)


def test_gauss_hermite_rule_has_unit_total_weight() -> None:
    rule = gauss_hermite_rule(3, 2)
    np.testing.assert_allclose(jnp.sum(rule.weights), 1.0, rtol=1e-12, atol=1e-12)


def test_sep_zero_shock_linear_model_returns_zero_mean_path() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.5 * y_prev + 0.3 * expected_next + shock)

    solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=5, branching_order=1, nnodes=3, tol=1e-10),
    )

    assert solution.converged
    np.testing.assert_allclose(solution.mean_path, 0.0, rtol=1e-10, atol=1e-10)


def test_sep_mean_path_matches_deterministic_path_for_linear_zero_mean_shocks() -> None:
    deterministic_shocks = jnp.array([[1.0], [0.2], [0.0], [0.0]])

    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    stochastic_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=4, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )
    deterministic_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=4, branching_order=0, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )

    assert stochastic_solution.converged
    assert deterministic_solution.converged
    np.testing.assert_allclose(
        stochastic_solution.mean_path,
        deterministic_solution.mean_path,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sep_handles_nonlinear_expectational_equation() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (
            0.3 * y_prev
            + 0.4 * expected_next
            - 0.05 * y_curr**2
            + shock
        )

    solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=1, nnodes=3, tol=1e-8),
        deterministic_shocks=[[0.5], [0.0], [0.0]],
    )

    assert solution.converged
    assert solution.residual_norm < 1e-8
    assert np.all(np.isfinite(solution.mean_path))


def test_sep_conditional_residual_mode_matches_expectation_api_when_equivalent() -> None:
    deterministic_shocks = jnp.array([[0.2], [0.0], [0.0]])

    def residual(y_prev, y_curr, expected_square, shock, params):
        return y_curr - (0.25 * y_prev + 0.15 * expected_square + shock)

    def expectation(next_state, next_shock, params):
        return next_state**2

    def conditional_residual(y_prev, y_curr, next_state, shock, params):
        return y_curr - (0.25 * y_prev + 0.15 * next_state**2 + shock)

    expectation_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
        expectation_fn=expectation,
    )
    conditional_solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )

    assert expectation_solution.converged
    assert conditional_solution.converged
    np.testing.assert_allclose(
        conditional_solution.stacked_states,
        expectation_solution.stacked_states,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        conditional_solution.mean_path,
        expectation_solution.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sep_sparse_tree_uses_fishbone_group_counts() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.3 * y_prev + 0.2 * expected_next + jnp.sum(shock))

    solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=2,
        config=SEPConfig(
            periods=4,
            branching_order=2,
            nnodes=3,
            sparse_tree=True,
            tol=1e-10,
        ),
    )

    assert solution.converged
    assert solution.group_counts == (1, 1, 6, 11, 11)


def test_sep_sparse_tree_matches_full_tree_mean_path_for_linear_zero_mean_shocks() -> None:
    deterministic_shocks = jnp.asarray(
        [[0.4, -0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=jnp.float64,
    )

    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (
            0.35 * y_prev + 0.15 * expected_next + shock[0] - 0.5 * shock[1]
        )

    full_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=2,
        config=SEPConfig(periods=4, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )
    sparse_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=2,
        config=SEPConfig(
            periods=4,
            branching_order=2,
            nnodes=3,
            sparse_tree=True,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )

    assert full_solution.converged
    assert sparse_solution.converged
    assert sparse_solution.group_counts[1] == 1
    assert full_solution.group_counts[1] == 9
    np.testing.assert_allclose(
        sparse_solution.mean_path,
        full_solution.mean_path,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sep_warm_start_accepts_previous_solution_and_finishes_immediately() -> None:
    deterministic_shocks = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)

    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (
            0.25 * y_prev + 0.15 * expected_next - 0.05 * y_curr**2 + shock
        )

    cold = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
    )
    warm = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(periods=3, branching_order=2, nnodes=3, tol=1e-10),
        deterministic_shocks=deterministic_shocks,
        initial_guess=cold.stacked_states,
    )

    assert cold.converged
    assert warm.converged
    assert warm.iterations == 0
    np.testing.assert_allclose(
        warm.stacked_states,
        cold.stacked_states,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sep_rejects_invalid_warm_start_shape() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    with np.testing.assert_raises_regex(ValueError, "initial_guess must flatten"):
        solve_stochastic_extended_path(
            residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(periods=3, branching_order=1, nnodes=3),
            initial_guess=jnp.zeros((5,), dtype=jnp.float64),
        )


def test_sep_validates_sparse_tree_requires_odd_nnodes() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    with np.testing.assert_raises_regex(ValueError, "odd nnodes"):
        solve_stochastic_extended_path(
            residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(periods=3, branching_order=1, nnodes=2, sparse_tree=True),
        )


def test_sep_validates_bad_config_values() -> None:
    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    with np.testing.assert_raises_regex(ValueError, "line_search_factor"):
        solve_stochastic_extended_path(
            residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(
                periods=3,
                branching_order=1,
                nnodes=3,
                line_search_factor=1.0,
            ),
        )


def test_sep_hmc_residual_expectation_is_deterministic_for_fixed_seed() -> None:
    deterministic_shocks = jnp.asarray([[0.1], [0.0], [0.0]], dtype=jnp.float64)

    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.3 * y_prev + 0.25 * y_next + shock)

    hmc_1 = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=16,
            hmc_warmup=8,
            hmc_leapfrog_steps=6,
            hmc_step_size=0.07,
            hmc_seed=7,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )
    hmc_2 = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=16,
            hmc_warmup=8,
            hmc_leapfrog_steps=6,
            hmc_step_size=0.07,
            hmc_seed=7,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )

    assert hmc_1.converged
    assert hmc_2.converged
    assert hmc_1.group_counts == (1, 1, 1, 1)
    np.testing.assert_allclose(
        hmc_1.mean_path,
        hmc_2.mean_path,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sep_hmc_parallel_tempering_runs() -> None:
    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.2 * y_prev + 0.15 * y_next**2 + shock)

    solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=12,
            hmc_warmup=6,
            hmc_leapfrog_steps=5,
            hmc_step_size=0.06,
            hmc_use_tempering=True,
            hmc_temperatures=(1.0, 0.5, 0.25),
            hmc_swap_interval=4,
            hmc_seed=3,
            tol=1e-8,
        ),
        deterministic_shocks=[[0.15], [0.0], [0.0]],
    )

    assert solution.converged
    assert np.all(np.isfinite(solution.mean_path))


def test_sep_hmc_legacy_expectation_api_runs_deterministically() -> None:
    deterministic_shocks = jnp.asarray([[0.15], [0.0], [0.0]], dtype=jnp.float64)

    def residual(y_prev, y_curr, expected_next, shock, params):
        return y_curr - (0.4 * y_prev + 0.2 * expected_next + shock)

    def expectation(next_state, next_shock, params):
        del next_state, next_shock, params
        return jnp.asarray([0.5], dtype=jnp.float64)

    gh_solution = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            nnodes=3,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
        expectation_fn=expectation,
    )
    hmc_1 = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=16,
            hmc_warmup=8,
            hmc_leapfrog_steps=6,
            hmc_step_size=0.07,
            hmc_seed=19,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
        expectation_fn=expectation,
    )
    hmc_2 = solve_stochastic_extended_path(
        residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            expectation_method="hmc",
            hmc_samples=16,
            hmc_warmup=8,
            hmc_leapfrog_steps=6,
            hmc_step_size=0.07,
            hmc_seed=19,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
        expectation_fn=expectation,
    )

    assert gh_solution.converged
    assert hmc_1.converged
    assert hmc_2.converged
    assert hmc_1.group_counts == (1, 1, 1, 1)
    np.testing.assert_allclose(hmc_1.mean_path, hmc_2.mean_path, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hmc_1.mean_path, gh_solution.mean_path, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        hmc_1.stacked_states,
        np.asarray(hmc_1.mean_path[:, 1:], dtype=np.float64).reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )
