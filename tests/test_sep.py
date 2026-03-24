from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import surrogatenn_dsge.sep as sep_module

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


def test_sep_sparse_tree_precomputes_tree_metadata_consistently() -> None:
    periods = 4
    branching_order = 2
    shock_dim = 1
    rule = sep_module._gauss_hermite_sparse_rule(3, shock_dim, 1.0)
    counts = sep_module._group_counts(
        periods,
        branching_order,
        int(rule.weights.shape[0]),
        sparse_tree=True,
    )
    deterministic = jnp.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float64)
    metadata = sep_module._precompute_sep_tree_metadata(
        rule=rule,
        deterministic=deterministic,
        counts=counts,
        periods=periods,
        branching_order=branching_order,
        num_nodes=int(rule.weights.shape[0]),
        shock_dim=shock_dim,
        sparse_tree=True,
        use_hmc=False,
    )

    assert metadata.parent_indices[0] is None
    np.testing.assert_array_equal(
        metadata.parent_indices[1],
        np.asarray([0, 0, 0], dtype=np.int64),
    )
    for t in range(1, periods + 1):
        period_shocks = np.asarray(metadata.current_shocks[t - 1], dtype=np.float64)
        assert period_shocks.shape == (counts[t], shock_dim)
        for g in range(counts[t]):
            expected_current = np.asarray(
                deterministic[t - 1]
                + sep_module._group_shock_at_time(
                    rule,
                    g,
                    t,
                    branching_order,
                    int(rule.weights.shape[0]),
                    sparse_tree=True,
                ),
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                period_shocks[g],
                expected_current,
                rtol=0.0,
                atol=1e-12,
            )
        if t == periods:
            assert metadata.child_groups[t - 1] is None
            assert metadata.child_shocks[t - 1] is None
            continue
        for g in range(counts[t]):
            expected_groups = sep_module._child_groups(
                g,
                t,
                branching_order,
                int(rule.weights.shape[0]),
                sparse_tree=True,
            )
            assert metadata.child_groups[t - 1][g] == expected_groups


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
    with np.testing.assert_raises_regex(ValueError, "linear_solver"):
        solve_stochastic_extended_path(
            residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(
                periods=3,
                branching_order=1,
                nnodes=3,
                linear_solver="bad_solver",
            ),
        )
    with np.testing.assert_raises_regex(ValueError, "line_search_maxit"):
        solve_stochastic_extended_path(
            residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(
                periods=3,
                branching_order=1,
                nnodes=3,
                line_search_maxit=0,
            ),
        )


def test_sep_qr_linear_solver_matches_normal_equations_on_smooth_model() -> None:
    deterministic_shocks = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)

    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.2 * y_prev + 0.15 * y_next**2 + shock)

    normal_solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            linear_solver="normal_equations",
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )
    qr_solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            linear_solver="qr",
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )

    assert normal_solution.converged
    assert qr_solution.converged
    np.testing.assert_allclose(
        qr_solution.mean_path,
        normal_solution.mean_path,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sep_switches_to_fallback_solver_after_stall(monkeypatch) -> None:
    deterministic_shocks = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)
    original_solver = sep_module._solve_sep_newton_direction
    calls: list[str] = []

    def recording_solver(jacobian, residual, *, lambda_value, solver):
        calls.append(str(solver))
        return original_solver(
            jacobian,
            residual,
            lambda_value=lambda_value,
            solver=solver,
        )

    monkeypatch.setattr(sep_module, "_solve_sep_newton_direction", recording_solver)

    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (
            0.2 * y_prev + 0.15 * y_next**2 - 0.05 * y_curr**2 + shock
        )

    solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            linear_solver="normal_equations",
            fallback_solver="qr",
            stall_iters=1,
            stall_rel_tol=1.0,
            stall_abs_tol=1.0,
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )

    assert solution.converged
    assert calls
    assert calls[0] == "normal_equations"
    assert "qr" in calls[1:]


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


def test_sep_finite_difference_jacobian_matches_autodiff_on_smooth_model() -> None:
    deterministic_shocks = jnp.asarray([[0.2], [0.0], [0.0]], dtype=jnp.float64)

    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.2 * y_prev + 0.15 * y_next**2 + shock)

    autodiff_solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            jacobian_method="autodiff",
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )
    finite_difference_solution = solve_stochastic_extended_path_residual_expectation(
        conditional_residual,
        initial_state=[0.0],
        terminal_state=[0.0],
        shock_dim=1,
        config=SEPConfig(
            periods=3,
            branching_order=1,
            jacobian_method="finite_difference",
            tol=1e-10,
        ),
        deterministic_shocks=deterministic_shocks,
    )

    assert autodiff_solution.converged
    assert finite_difference_solution.converged
    assert autodiff_solution.jacobian_method == "autodiff"
    assert finite_difference_solution.jacobian_method == "finite_difference"
    np.testing.assert_allclose(
        finite_difference_solution.mean_path,
        autodiff_solution.mean_path,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sep_subgradient_requires_custom_jacobian() -> None:
    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.2 * y_prev + 0.15 * y_next**2 + shock)

    with np.testing.assert_raises_regex(ValueError, "requires a jacobian_fn"):
        solve_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(
                periods=3,
                branching_order=1,
                jacobian_method="subgradient",
            ),
        )


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


def test_sep_rejects_autodiff_jacobian_for_hmc() -> None:
    def conditional_residual(y_prev, y_curr, y_next, shock, params):
        return y_curr - (0.2 * y_prev + 0.15 * y_next**2 + shock)

    with np.testing.assert_raises_regex(
        ValueError,
        "must not be 'autodiff' or 'subgradient' when expectation_method='hmc'",
    ):
        solve_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=[0.0],
            terminal_state=[0.0],
            shock_dim=1,
            config=SEPConfig(
                periods=3,
                branching_order=1,
                expectation_method="hmc",
                jacobian_method="autodiff",
            ),
        )
