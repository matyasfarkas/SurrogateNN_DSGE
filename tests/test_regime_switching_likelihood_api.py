from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    additive_residual_loglik_per_period,
    advance_state,
    build_shocks_from_eps,
    conditional_loglik_per_period,
    extract_named_parameters,
    inversion_loglik_per_period,
    inversion_step,
    linear_model_loglik_per_period,
    linear_reference_loglik_per_period,
    override_named_parameters,
    parameters_with_theta_mode,
    parse_macro_model,
    predict_additive_residual,
    predict_from_full,
    rollout_observations,
    run_chunked_sampling,
    solve_first_order_model,
    split_observation_state,
)


LINEAR_MODEL_SOURCE = """
@model regime_switch_linear begin
    y[0] = rho * y[-1] + sigma * eps[x]
end

@parameters regime_switch_linear begin
    rho = 0.7
    sigma = 0.1
end
"""


def _predict_full(state, shock_t, theta_local):
    state_arr = jnp.asarray(state, dtype=jnp.float64)
    shock_arr = jnp.asarray(shock_t, dtype=jnp.float64)
    theta_arr = jnp.asarray(theta_local, dtype=jnp.float64)
    return jnp.asarray(
        [
            state_arr[0] + shock_arr[0] + theta_arr[0],
            state_arr[0] + shock_arr[0],
        ],
        dtype=jnp.float64,
    )


def _predict_split(state, shock_t, theta_local):
    full = _predict_full(state, shock_t, theta_local)
    return full[:1], full[1:]


def _predict_toy(state, shock_t, theta_local):
    state_arr = jnp.asarray(state, dtype=jnp.float64)
    shock_arr = jnp.asarray(shock_t, dtype=jnp.float64)
    theta_arr = jnp.asarray(theta_local, dtype=jnp.float64)
    obs_pred = jnp.asarray([state_arr[0] + shock_arr[0] + theta_arr[0]], dtype=jnp.float64)
    state_next = jnp.asarray([0.8 * state_arr[0] + 0.2 * shock_arr[0]], dtype=jnp.float64)
    return obs_pred, state_next


def test_named_parameter_helpers_match_julia_behavior() -> None:
    base_params = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    model_params = ("p1", "p2", "p3")
    theta_names = ("p3", "p1")
    theta_vals = np.asarray([3.5, 1.5], dtype=np.float64)

    np.testing.assert_allclose(
        extract_named_parameters(base_params, model_params, theta_names),
        np.asarray([30.0, 10.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        override_named_parameters(base_params, model_params, theta_names, theta_vals),
        np.asarray([1.5, 20.0, 3.5], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        parameters_with_theta_mode(
            base_params,
            model_params,
            theta_names,
            theta_vals,
            theta_mode="synthetic",
        ),
        np.asarray([1.5, 20.0, 3.5], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        parameters_with_theta_mode(
            base_params,
            model_params,
            theta_names,
            theta_vals,
            theta_mode="baseline",
        ),
        base_params,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="length mismatch"):
        override_named_parameters(base_params, model_params, theta_names, [1.0])
    with pytest.raises(ValueError, match="requires theta values"):
        parameters_with_theta_mode(
            base_params,
            model_params,
            theta_names,
            None,
            theta_mode="synthetic",
        )
    with pytest.raises(ValueError, match="not found"):
        extract_named_parameters(base_params, model_params, ("missing",))


def test_linear_model_loglik_helper_matches_direct_model_call() -> None:
    model = parse_macro_model(LINEAR_MODEL_SOURCE)
    solve_first_order_model(model, steady_state_initial_guess={"y": 0.0})

    observations = np.zeros((1, 4), dtype=np.float64)
    theta_names = ("sigma",)
    theta = np.asarray([0.2], dtype=np.float64)
    sigma_idx = model.parameter_names.index("sigma") + 1
    expected_params = override_named_parameters(
        np.asarray(model.parameter_values, dtype=np.float64),
        model.parameter_names,
        theta_names,
        theta,
    )

    direct = np.asarray(
        model.kalman_loglikelihood_per_period(
            observations,
            observables=("y",),
            parameter_values=expected_params,
            steady_state_initial_guess={"y": 0.0},
            on_failure_loglikelihood=-99.0,
        ),
        dtype=np.float64,
    )
    helper = linear_model_loglik_per_period(
        model,
        observations,
        theta,
        theta_names,
        observables=("y",),
        on_failure_loglikelihood=-99.0,
    )
    helper_idx = linear_model_loglik_per_period(
        model,
        observations,
        theta,
        theta_names,
        observables=("y",),
        theta_idx=[sigma_idx],
        on_failure_loglikelihood=-99.0,
    )

    np.testing.assert_allclose(helper, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(helper_idx, direct, rtol=1e-12, atol=1e-12)

    with pytest.raises(ValueError, match="index > number of model parameters"):
        linear_model_loglik_per_period(
            model,
            observations,
            theta,
            theta_names,
            observables=("y",),
            theta_idx=[len(model.parameter_names) + 1],
        )


def test_callback_based_loglik_helpers_match_julia_toy_cases() -> None:
    ll_cond = conditional_loglik_per_period(
        _predict_split,
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [0.0],
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        [1.0],
    )
    np.testing.assert_allclose(
        ll_cond,
        np.full((2,), -0.5 * np.log(2.0 * np.pi), dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )

    obs_split, state_split = split_observation_state([1.0, 2.0, 3.0], 2)
    np.testing.assert_allclose(obs_split, np.asarray([1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(state_split, np.asarray([3.0], dtype=np.float64))
    with pytest.raises(ValueError, match="length mismatch"):
        split_observation_state([1.0], 2)

    obs_full, state_full = predict_from_full(_predict_full, [0.0], [1.0], [0.0], 1)
    np.testing.assert_allclose(obs_full, np.asarray([1.0], dtype=np.float64))
    np.testing.assert_allclose(state_full, np.asarray([1.0], dtype=np.float64))

    obs_resid, state_resid = predict_additive_residual(
        _predict_full,
        lambda state, shock_t, theta_local: jnp.asarray([0.5], dtype=jnp.float64),
        [0.0],
        [1.0],
        [0.0],
        1,
        allow_full_residual=True,
    )
    np.testing.assert_allclose(obs_resid, np.asarray([1.5], dtype=np.float64))
    np.testing.assert_allclose(state_resid, np.asarray([1.0], dtype=np.float64))

    obs_resid_full, state_resid_full = predict_additive_residual(
        _predict_full,
        lambda state, shock_t, theta_local: jnp.asarray([0.25, -0.5], dtype=jnp.float64),
        [0.0],
        [1.0],
        [0.0],
        1,
        allow_full_residual=True,
    )
    np.testing.assert_allclose(obs_resid_full, np.asarray([1.25], dtype=np.float64))
    np.testing.assert_allclose(state_resid_full, np.asarray([0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="Residual output size mismatch"):
        predict_additive_residual(
            _predict_full,
            lambda state, shock_t, theta_local: jnp.asarray([0.25, -0.5], dtype=jnp.float64),
            [0.0],
            [1.0],
            [0.0],
            1,
            allow_full_residual=False,
        )
    with pytest.raises(ValueError, match="Residual output size mismatch"):
        predict_additive_residual(
            _predict_full,
            lambda state, shock_t, theta_local: jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64),
            [0.0],
            [1.0],
            [0.0],
            1,
            allow_full_residual=True,
        )

    ll_additive = additive_residual_loglik_per_period(
        _predict_full,
        lambda state, shock_t, theta_local: jnp.asarray([0.5], dtype=jnp.float64),
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [0.0],
        np.asarray([[1.5, 1.5]], dtype=np.float64),
        [1.0],
        d_obs=1,
        allow_full_residual=True,
    )
    np.testing.assert_allclose(
        ll_additive,
        np.full((2,), -0.5 * np.log(2.0 * np.pi), dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )


def test_rollout_advance_and_inversion_helpers_match_toy_behavior() -> None:
    obs_roll = rollout_observations(
        _predict_split,
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [0.0],
    )
    np.testing.assert_allclose(
        obs_roll,
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="length mismatch"):
        rollout_observations(
            lambda state, shock_t, theta_local: (
                jnp.asarray([state[0]], dtype=jnp.float64),
                jnp.asarray([state[0]], dtype=jnp.float64),
            )
            if shock_t[0] > 0.5
            else (
                jnp.asarray([state[0], 0.0], dtype=jnp.float64),
                jnp.asarray([state[0]], dtype=jnp.float64),
            ),
            [0.0],
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            [0.0],
        )
    with pytest.raises(ValueError, match="non-finite observation values"):
        rollout_observations(
            lambda state, shock_t, theta_local: (
                jnp.asarray([jnp.nan], dtype=jnp.float64),
                jnp.asarray([state[0]], dtype=jnp.float64),
            ),
            [0.0],
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            [0.0],
            check_finite=True,
        )

    state_adv = advance_state(
        _predict_split,
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [0.0],
        2,
    )
    np.testing.assert_allclose(state_adv, np.asarray([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="exceeds available shock periods"):
        advance_state(
            _predict_split,
            [0.0],
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            [0.0],
            3,
        )

    eps_step, state_next_step, ll_step = inversion_step(
        _predict_toy,
        [0.0],
        [1.0],
        [0.0],
        [0.1],
        [0.5, 0.0],
        [1],
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    assert eps_step.shape == (2,)
    assert eps_step[1] == 0.0
    assert np.isfinite(ll_step)
    assert np.isfinite(state_next_step[0])

    ll_inv, shocks_inv = inversion_loglik_per_period(
        _predict_toy,
        [0.0],
        [0.0],
        np.asarray([[1.0, 0.5]], dtype=np.float64),
        [0.1],
        [0.5, 0.0],
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    assert ll_inv.shape == (2,)
    assert shocks_inv.shape == (2, 2)
    assert np.isfinite(ll_inv).all()
    assert np.isfinite(shocks_inv).all()

    ll_lin_sampling = linear_reference_loglik_per_period(
        [0.0],
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        [1.0],
        [0.5, 0.0],
        shock_filter="sampling",
        linear_filter="kalman",
        predict_linear=_predict_split,
        kalman_linear_loglik=lambda theta_local: np.full((2,), -99.0, dtype=np.float64),
    )
    np.testing.assert_allclose(ll_lin_sampling, ll_cond := conditional_loglik_per_period(
        _predict_split,
        [0.0],
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        [0.0],
        np.asarray([[1.0, 1.0]], dtype=np.float64),
        [1.0],
    ), rtol=1e-12, atol=1e-12)

    called = {"kalman": False}
    ll_lin_kalman = linear_reference_loglik_per_period(
        [0.0],
        [0.0],
        np.zeros((2, 2), dtype=np.float64),
        np.asarray([[1.0, 0.5]], dtype=np.float64),
        [0.1],
        [0.5, 0.0],
        shock_filter="inversion",
        linear_filter="kalman",
        predict_linear=_predict_toy,
        kalman_linear_loglik=lambda theta_local: (
            called.__setitem__("kalman", True)
            or np.asarray([-3.0, -4.0], dtype=np.float64)
        ),
    )
    assert called["kalman"]
    np.testing.assert_allclose(
        ll_lin_kalman,
        np.asarray([-3.0, -4.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    ll_lin_inversion = linear_reference_loglik_per_period(
        [0.0],
        [0.0],
        np.zeros((2, 2), dtype=np.float64),
        np.asarray([[1.0, 0.5]], dtype=np.float64),
        [0.1],
        [0.5, 0.0],
        shock_filter="inversion",
        linear_filter="inversion",
        predict_linear=_predict_toy,
        kalman_linear_loglik=lambda theta_local: np.full((2,), -99.0, dtype=np.float64),
        inversion_maxit=12,
        inversion_tol=1e-8,
        inversion_lambda=1e-6,
    )
    np.testing.assert_allclose(ll_lin_inversion, ll_inv, rtol=1e-8, atol=1e-8)
    with pytest.raises(ValueError, match="Unsupported linear_filter"):
        linear_reference_loglik_per_period(
            [0.0],
            [0.0],
            np.zeros((2, 2), dtype=np.float64),
            np.asarray([[1.0, 0.5]], dtype=np.float64),
            [0.1],
            [0.5, 0.0],
            shock_filter="inversion",
            linear_filter="unknown",
            predict_linear=_predict_toy,
        )


def test_build_shocks_from_eps_and_chunk_runner_match_expected_behavior() -> None:
    eps_mean = np.asarray([[1.0, -1.0], [0.25, 0.5]], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0, 2.0], dtype=np.float64)
    shocks = build_shocks_from_eps(eps_mean, shock_sigmas, None)
    np.testing.assert_allclose(
        shocks,
        np.asarray([[0.5, -0.5], [0.0, 0.0], [0.5, 1.0]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    guided_base = np.ones((3, 5), dtype=np.float64)
    shocks_sub = build_shocks_from_eps(
        eps_mean,
        shock_sigmas,
        guided_base,
        sample_idx=[2, 4],
        T_full=5,
    )
    np.testing.assert_allclose(shocks_sub[:, 0], np.ones((3,), dtype=np.float64))
    np.testing.assert_allclose(shocks_sub[[0, 2], 1], np.asarray([1.5, 1.5], dtype=np.float64))
    np.testing.assert_allclose(shocks_sub[[0, 2], 3], np.asarray([0.5, 2.0], dtype=np.float64))
    np.testing.assert_allclose(shocks_sub[:, 4], np.ones((3,), dtype=np.float64))

    with pytest.raises(ValueError, match="row mismatch"):
        build_shocks_from_eps(np.random.randn(3, 2), shock_sigmas, None)
    with pytest.raises(ValueError, match="exceeds target length"):
        build_shocks_from_eps(
            eps_mean,
            shock_sigmas,
            guided_base,
            sample_idx=[2, 6],
            T_full=5,
        )

    seen_chunks: list[tuple[int, int, int]] = []
    out = run_chunked_sampling(
        5,
        2,
        sample_chunk=lambda n_i, i, n_chunks: np.full((n_i,), i, dtype=np.int64),
        concat_chunks=lambda a, b: np.concatenate([a, b], axis=0),
        on_chunk=lambda i, n_chunks, n_i, chunk, samps, elapsed: seen_chunks.append((i, n_chunks, n_i)),
    )
    np.testing.assert_array_equal(out, np.asarray([1, 1, 2, 2, 3], dtype=np.int64))
    assert seen_chunks == [(1, 3, 2), (2, 3, 2), (3, 3, 1)]
