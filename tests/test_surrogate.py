from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    FrozenMLP,
    FrozenResNet,
    NormStats,
    ResBlock,
    build_numpyro_surrogate_inversion_model_jax,
    additive_residual_loglik_per_period,
    compute_ood_flag,
    evaluate_numpyro_surrogate_log_density_jax,
    inversion_loglik_per_period,
    predict_frozen,
    predict_frozen_batch,
    predict_frozen_safe,
    scale_frozen_output,
    standardize_xy,
    surrogate_additive_residual_loglik_per_period,
    surrogate_inversion_loglik_per_period,
    surrogate_inversion_loglik_per_period_jax,
    surrogate_inversion_loglikelihood_jax,
    train_mlp,
    train_resnet,
    validate_surrogate,
    weighted_mse,
)


def _toy_full_predict(state, shock_t, theta):
    state = np.asarray(state, dtype=np.float64)
    shock_t = np.asarray(shock_t, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return np.asarray(
        [
            state[0] + shock_t[0] + theta[0],
            state[0] + shock_t[0],
        ],
        dtype=np.float64,
    )


def _toy_split_predict(state, shock_t, theta):
    y = _toy_full_predict(state, shock_t, theta)
    return y[:1], y[1:]


def _toy_split_predict_jax(state, shock_t, theta):
    state = jnp.asarray(state, dtype=jnp.float64).reshape(-1)
    shock_t = jnp.asarray(shock_t, dtype=jnp.float64).reshape(-1)
    theta = jnp.asarray(theta, dtype=jnp.float64).reshape(-1)
    obs = jnp.asarray([state[0] + shock_t[0] + theta[0]], dtype=jnp.float64)
    next_state = jnp.asarray([state[0] + shock_t[0]], dtype=jnp.float64)
    return obs, next_state


def _constant_residual_mlp(d_in: int, d_out: int, value: float = 0.5) -> FrozenMLP:
    return FrozenMLP(
        W1=np.zeros((1, d_in), dtype=np.float64),
        b1=np.zeros((1,), dtype=np.float64),
        W2=np.zeros((d_out, 1), dtype=np.float64),
        b2=np.full((d_out,), value, dtype=np.float64),
        W3=None,
        b3=None,
        norm=NormStats(
            np.zeros((d_in,), dtype=np.float64),
            np.ones((d_in,), dtype=np.float64),
            np.zeros((d_out,), dtype=np.float64),
            np.ones((d_out,), dtype=np.float64),
        ),
        d_in=d_in,
        d_out=d_out,
        activation="tanh",
    )


def test_scale_frozen_output_shrinks_physical_residual_exactly() -> None:
    frozen = _constant_residual_mlp(d_in=3, d_out=2, value=0.5)
    x = np.asarray([1.0, -0.5, 0.2], dtype=np.float64)

    scaled = scale_frozen_output(frozen, [0.25, 0.0])

    np.testing.assert_allclose(
        predict_frozen(scaled, x),
        np.asarray([0.125, 0.0], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_frozen_mlp_prediction_matches_julia_formula() -> None:
    W1 = np.asarray(
        [
            [0.10, -0.20, 0.30],
            [0.40, 0.05, -0.10],
            [-0.30, 0.20, 0.25],
            [0.15, -0.35, 0.05],
        ],
        dtype=np.float64,
    )
    b1 = np.asarray([0.01, -0.02, 0.03, 0.04], dtype=np.float64)
    W2 = np.asarray(
        [
            [0.30, -0.10, 0.20, 0.05],
            [-0.25, 0.15, 0.10, -0.20],
        ],
        dtype=np.float64,
    )
    b2 = np.asarray([0.07, -0.04], dtype=np.float64)
    norm = NormStats(
        [1.0, -2.0, 0.5],
        [2.0, 0.5, 4.0],
        [-0.25, 1.5],
        [0.7, 2.0],
    )
    frozen = FrozenMLP(W1, b1, W2, b2, None, None, norm, 3, 2, "silu")
    x = np.asarray([0.2, -1.75, 2.5], dtype=np.float64)

    xnorm = (x - np.asarray(norm.mu_x)) / np.asarray(norm.sigma_x)
    hidden = (W1 @ xnorm + b1) / (1.0 + np.exp(-(W1 @ xnorm + b1)))
    expected = np.asarray(norm.mu_y) + np.asarray(norm.sigma_y) * (W2 @ hidden + b2)

    np.testing.assert_allclose(predict_frozen(frozen, x), expected, rtol=1e-12, atol=1e-12)

    X = np.column_stack([x, x + np.asarray([0.1, -0.2, 0.3])])
    Y_batch = np.asarray(predict_frozen_batch(frozen, X), dtype=np.float64)
    np.testing.assert_allclose(Y_batch[:, 0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Y_batch[:, 1], predict_frozen(frozen, X[:, 1]), rtol=1e-12, atol=1e-12)


def test_frozen_resnet_batch_matches_single_prediction() -> None:
    norm = NormStats(np.zeros(5), np.ones(5), np.zeros(2), np.ones(2))
    block = ResBlock(
        W1=0.05 * np.eye(4),
        b1=np.asarray([0.01, -0.02, 0.03, -0.04]),
        W2=0.03 * np.eye(4),
        b2=np.asarray([0.02, 0.01, -0.01, -0.02]),
    )
    net = FrozenResNet(
        W_embed=np.asarray(
            [
                [0.2, -0.1, 0.05],
                [0.0, 0.3, -0.2],
                [0.1, 0.1, 0.1],
                [-0.2, 0.05, 0.25],
            ],
            dtype=np.float64,
        ),
        b_embed=np.asarray([0.01, -0.01, 0.02, -0.02]),
        d_theta=2,
        W_gamma=np.asarray([[0.01, 0.02], [-0.03, 0.01], [0.02, -0.02], [0.01, 0.0]]),
        b_gamma=np.ones(4),
        W_beta=np.asarray([[0.02, -0.01], [0.01, 0.03], [0.0, -0.02], [-0.01, 0.02]]),
        b_beta=np.zeros(4),
        blocks=(block,),
        W_out=np.asarray([[0.2, -0.1, 0.05, 0.01], [-0.05, 0.03, 0.2, -0.1]]),
        b_out=np.asarray([0.01, -0.02]),
        norm=norm,
        d_in=5,
        d_out=2,
    )
    X = np.asarray(
        [
            [0.1, 0.4, -0.2],
            [0.0, -0.3, 0.5],
            [0.2, 0.1, -0.1],
            [0.6, 0.7, 0.8],
            [-0.4, -0.5, -0.6],
        ],
        dtype=np.float64,
    )
    Y_batch = np.asarray(predict_frozen_batch(net, X), dtype=np.float64)
    for col in range(X.shape[1]):
        np.testing.assert_allclose(Y_batch[:, col], predict_frozen(net, X[:, col]), rtol=1e-12, atol=1e-12)


def test_ood_weighted_mse_and_validation_diagnostics() -> None:
    frozen = _constant_residual_mlp(d_in=4, d_out=2, value=0.25)
    x_id = np.asarray([1.0, -1.0, 0.5, 0.0])
    pred, is_ood, max_z = predict_frozen_safe(frozen, x_id, z_threshold=4.0)
    assert not is_ood
    assert max_z < 4.0
    np.testing.assert_allclose(pred, np.full((2,), 0.25))
    assert compute_ood_flag(frozen.norm, [10.0], [0.0, 0.0], [0.0], z_threshold=4.0)

    y_hat = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.asarray([[1.1, 2.2, 3.3], [4.1, 5.2, 6.3]])
    w = np.asarray([10.0, 1.0, 1.0])
    expected = np.sum(np.sum((y_hat - y) ** 2, axis=0) * w) / np.sum(w)
    np.testing.assert_allclose(weighted_mse(y_hat, y, w), expected, rtol=1e-12, atol=1e-12)

    X_val = np.zeros((4, 5), dtype=np.float64)
    Y_val = np.full((2, 5), 0.25, dtype=np.float64)
    result = validate_surrogate(frozen, X_val, Y_val, Y_rom=np.zeros_like(Y_val))
    assert result.n_samples == 5
    assert result.rmse_total < 1e-12
    assert result.max_abs_error < 1e-12
    assert result.ood_fraction == 0.0
    assert result.improvement_vs_rom is not None
    np.testing.assert_allclose(result.improvement_vs_rom, np.ones((2,)), rtol=1e-12, atol=1e-12)


def test_weighted_standardization_ignores_zero_weight_nonfinite_samples() -> None:
    X = np.asarray(
        [
            [1.0, 3.0, np.nan],
            [2.0, 4.0, np.inf],
        ],
        dtype=np.float64,
    )
    Y = np.asarray([[2.0, 6.0, np.nan]], dtype=np.float64)
    weights = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)

    X_std, Y_std, norm = standardize_xy(X, Y, sample_weights=weights)

    np.testing.assert_allclose(np.asarray(norm.mu_x), np.asarray([2.0, 3.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(norm.sigma_x), np.asarray([1.0, 1.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(norm.mu_y), np.asarray([4.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(norm.sigma_y), np.asarray([2.0]), rtol=0, atol=1e-12)
    assert np.isfinite(X_std).all()
    assert np.isfinite(Y_std).all()
    np.testing.assert_allclose(X_std[:, 2], np.zeros((2,)), rtol=0, atol=0)
    np.testing.assert_allclose(Y_std[:, 2], np.zeros((1,)), rtol=0, atol=0)

    with pytest.raises(ValueError, match="Positive-weight samples must be finite"):
        standardize_xy(X, Y, sample_weights=np.asarray([1.0, 0.0, 1.0], dtype=np.float64))


def test_train_mlp_learns_simple_map_with_jax_backend() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(3, 96))
    A = np.asarray([[0.8, -0.4, 0.2], [-0.1, 0.3, 0.7]], dtype=np.float64)
    Y = A @ X

    frozen = train_mlp(
        X,
        Y,
        d_hidden=24,
        d_hidden2=None,
        nepoch=160,
        eta_init=5e-3,
        batch_size=32,
        seed=7,
        weight_decay=1e-6,
        activation="silu",
    )
    Y_pred = np.asarray(predict_frozen_batch(frozen, X), dtype=np.float64)
    rmse = float(np.sqrt(np.mean((Y_pred - Y) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((Y - Y.mean(axis=1, keepdims=True)) ** 2)))
    assert rmse < 0.35 * baseline_rmse


def test_train_mlp_accepts_zero_weight_masked_nonfinite_columns() -> None:
    rng = np.random.default_rng(124)
    X_good = rng.normal(size=(3, 24))
    A = np.asarray([[0.5, -0.2, 0.1]], dtype=np.float64)
    Y_good = A @ X_good
    X = np.column_stack([X_good, np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64)])
    Y = np.column_stack([Y_good, np.asarray([np.nan], dtype=np.float64)])
    weights = np.concatenate([np.ones((X_good.shape[1],), dtype=np.float64), np.zeros((1,), dtype=np.float64)])

    frozen = train_mlp(
        X,
        Y,
        d_hidden=8,
        d_hidden2=None,
        nepoch=4,
        eta_init=1e-3,
        batch_size=1,
        seed=8,
        sample_weights=weights,
    )

    Y_pred = np.asarray(predict_frozen_batch(frozen, X_good[:, :3]), dtype=np.float64)
    assert np.isfinite(Y_pred).all()


def test_train_resnet_accepts_zero_weight_masked_nonfinite_columns() -> None:
    rng = np.random.default_rng(125)
    X_good = rng.normal(size=(5, 24))
    Y_good = np.vstack(
        [
            0.2 * X_good[0, :] - 0.1 * X_good[1, :] + 0.3 * X_good[3, :],
            -0.4 * X_good[2, :] + 0.2 * X_good[4, :],
        ]
    )
    X = np.column_stack([X_good, np.full((5,), np.nan, dtype=np.float64)])
    Y = np.column_stack([Y_good, np.asarray([np.nan, np.inf], dtype=np.float64)])
    weights = np.concatenate([np.ones((X_good.shape[1],), dtype=np.float64), np.zeros((1,), dtype=np.float64)])

    frozen = train_resnet(
        X,
        Y,
        d_theta=2,
        d_hidden=8,
        n_blocks=1,
        nepoch=4,
        eta_init=1e-3,
        batch_size=1,
        seed=9,
        sample_weights=weights,
    )

    Y_pred = np.asarray(predict_frozen_batch(frozen, X_good[:, :3]), dtype=np.float64)
    assert np.isfinite(Y_pred).all()


def test_train_resnet_learns_theta_conditioned_residual_map_with_jax_backend() -> None:
    rng = np.random.default_rng(321)
    X = rng.normal(size=(5, 128))
    state_shock = X[:3, :]
    theta = X[3:, :]
    Y = np.vstack(
        [
            0.45 * state_shock[0, :] - 0.20 * state_shock[1, :] + 0.75 * theta[0, :]
            + 0.18 * state_shock[2, :] * theta[1, :],
            -0.10 * state_shock[0, :] + 0.35 * state_shock[2, :] - 0.55 * theta[1, :]
            + 0.15 * state_shock[1, :] * theta[0, :],
        ]
    )

    frozen = train_resnet(
        X,
        Y,
        d_theta=2,
        d_hidden=28,
        n_blocks=1,
        nepoch=220,
        eta_init=3e-3,
        batch_size=32,
        seed=11,
        weight_decay=1e-6,
    )
    Y_pred = np.asarray(predict_frozen_batch(frozen, X), dtype=np.float64)
    rmse = float(np.sqrt(np.mean((Y_pred - Y) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((Y - Y.mean(axis=1, keepdims=True)) ** 2)))
    assert rmse < 0.40 * baseline_rmse


def test_train_resnet_validates_theta_dimension() -> None:
    X = np.zeros((3, 8), dtype=np.float64)
    Y = np.zeros((1, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="d_theta"):
        train_resnet(X, Y, d_theta=0, nepoch=1)
    with pytest.raises(ValueError, match="smaller"):
        train_resnet(X, Y, d_theta=3, nepoch=1)


def test_surrogate_additive_likelihood_matches_existing_callback_path() -> None:
    frozen = _constant_residual_mlp(d_in=3, d_out=1, value=0.5)
    shocks = np.asarray([[1.0, 0.0]], dtype=np.float64)
    obs = np.asarray([[1.5, 1.5]], dtype=np.float64)

    helper = surrogate_additive_residual_loglik_per_period(
        _toy_full_predict,
        frozen,
        [0.0],
        shocks,
        [0.0],
        obs,
        [1.0],
        d_obs=1,
    )
    direct = additive_residual_loglik_per_period(
        _toy_full_predict,
        lambda state, shock_t, theta: np.asarray(predict_frozen(frozen, np.r_[state, shock_t, theta])),
        [0.0],
        shocks,
        [0.0],
        obs,
        [1.0],
        d_obs=1,
    )
    np.testing.assert_allclose(helper, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(helper, np.full((2,), -0.5 * np.log(2.0 * np.pi)), rtol=1e-12, atol=1e-12)


def test_economic_smell_ood_guard_suppresses_pathological_correction() -> None:
    frozen_large = _constant_residual_mlp(d_in=3, d_out=1, value=100.0)
    frozen_large = FrozenMLP(
        frozen_large.W1,
        frozen_large.b1,
        frozen_large.W2,
        frozen_large.b2,
        frozen_large.W3,
        frozen_large.b3,
        NormStats(np.zeros(3), np.full(3, 0.1), np.zeros(1), np.ones(1)),
        3,
        1,
        "tanh",
    )
    shocks = np.asarray([[1.0, 0.0]], dtype=np.float64)
    baseline_obs = np.asarray([[1.0, 1.0]], dtype=np.float64)
    obs_sigma = np.asarray([1.0], dtype=np.float64)

    no_guard = surrogate_additive_residual_loglik_per_period(
        _toy_full_predict,
        frozen_large,
        [0.0],
        shocks,
        [0.0],
        baseline_obs,
        obs_sigma,
        d_obs=1,
        z_threshold=None,
    )
    guarded = surrogate_additive_residual_loglik_per_period(
        _toy_full_predict,
        frozen_large,
        [0.0],
        shocks,
        [0.0],
        baseline_obs,
        obs_sigma,
        d_obs=1,
        z_threshold=4.0,
    )

    baseline = additive_residual_loglik_per_period(
        _toy_full_predict,
        lambda state, shock_t, theta: np.zeros(1, dtype=np.float64),
        [0.0],
        shocks,
        [0.0],
        baseline_obs,
        obs_sigma,
        d_obs=1,
    )
    assert float(np.sum(no_guard)) < float(np.sum(baseline)) - 1_000.0
    np.testing.assert_allclose(guarded, baseline, rtol=1e-12, atol=1e-12)


def test_surrogate_inversion_likelihood_zero_residual_matches_rom() -> None:
    frozen_zero = _constant_residual_mlp(d_in=4, d_out=1, value=0.0)
    obs = np.asarray([[1.0, 0.5]], dtype=np.float64)
    obs_sigma = np.asarray([0.1], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0], dtype=np.float64)

    ll_rom, shocks_rom = inversion_loglik_per_period(
        _toy_split_predict,
        [0.0],
        [0.0],
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    ll_sur, shocks_sur = surrogate_inversion_loglik_per_period(
        _toy_split_predict,
        frozen_zero,
        [0.0],
        [0.0],
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    np.testing.assert_allclose(shocks_sur, shocks_rom, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ll_sur, ll_rom, rtol=1e-10, atol=1e-10)


def test_surrogate_inversion_likelihood_jax_matches_numpy_and_differentiates() -> None:
    frozen_zero = _constant_residual_mlp(d_in=4, d_out=1, value=0.0)
    obs = np.asarray([[1.0, 0.5]], dtype=np.float64)
    obs_sigma = np.asarray([0.1], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0], dtype=np.float64)
    theta = np.asarray([0.1], dtype=np.float64)

    ll_py, shocks_py = surrogate_inversion_loglik_per_period(
        _toy_split_predict,
        frozen_zero,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    ll_jax, shocks_jax = surrogate_inversion_loglik_per_period_jax(
        _toy_split_predict_jax,
        frozen_zero,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        lambda_=1e-6,
    )

    np.testing.assert_allclose(np.asarray(shocks_jax), shocks_py, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ll_jax), ll_py, rtol=1e-9, atol=1e-9)

    compiled = jax.jit(
        lambda theta_local: surrogate_inversion_loglikelihood_jax(
            _toy_split_predict_jax,
            frozen_zero,
            jnp.asarray([0.0], dtype=jnp.float64),
            theta_local,
            jnp.asarray(obs, dtype=jnp.float64),
            jnp.asarray(obs_sigma, dtype=jnp.float64),
            shock_sigmas,
            maxit=8,
            lambda_=1e-6,
        )
    )
    value = compiled(jnp.asarray(theta, dtype=jnp.float64))
    grad = jax.grad(lambda x: compiled(jnp.asarray([x], dtype=jnp.float64)))(jnp.asarray(0.1, dtype=jnp.float64))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(grad))

    full_unrolled = jax.jit(
        lambda theta_local: surrogate_inversion_loglikelihood_jax(
            _toy_split_predict_jax,
            frozen_zero,
            jnp.asarray([0.0], dtype=jnp.float64),
            theta_local,
            jnp.asarray(obs, dtype=jnp.float64),
            jnp.asarray(obs_sigma, dtype=jnp.float64),
            shock_sigmas,
            maxit=8,
            lambda_=1e-6,
            differentiate_shocks=True,
        )
    )
    full_grad = jax.grad(lambda x: full_unrolled(jnp.asarray([x], dtype=jnp.float64)))(jnp.asarray(0.1, dtype=jnp.float64))
    assert bool(jnp.isfinite(full_grad))


def test_surrogate_inversion_likelihood_jax_matches_python_residual_replay() -> None:
    frozen_residual = _constant_residual_mlp(d_in=4, d_out=1, value=0.2)
    obs = np.asarray([[1.0, 0.5]], dtype=np.float64)
    obs_sigma = np.asarray([0.1], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0], dtype=np.float64)
    theta = np.asarray([0.1], dtype=np.float64)

    ll_py, shocks_py = surrogate_inversion_loglik_per_period(
        _toy_split_predict,
        frozen_residual,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    ll_jax, shocks_jax = surrogate_inversion_loglik_per_period_jax(
        _toy_split_predict_jax,
        frozen_residual,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        lambda_=1e-6,
        shock_solver="rom",
        batch_replay=True,
    )

    np.testing.assert_allclose(np.asarray(shocks_jax), shocks_py, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ll_jax), ll_py, rtol=1e-9, atol=1e-9)


def test_surrogate_inversion_jax_matches_python_with_state_residual_carry() -> None:
    frozen_full_residual = _constant_residual_mlp(d_in=4, d_out=2, value=0.15)
    obs = np.asarray([[1.0, 0.65, 0.25]], dtype=np.float64)
    obs_sigma = np.asarray([0.1], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0], dtype=np.float64)
    theta = np.asarray([0.1], dtype=np.float64)

    ll_py, shocks_py = surrogate_inversion_loglik_per_period(
        _toy_split_predict,
        frozen_full_residual,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        tol=1e-8,
        lambda_=1e-6,
    )
    ll_jax, shocks_jax = surrogate_inversion_loglik_per_period_jax(
        _toy_split_predict_jax,
        frozen_full_residual,
        [0.0],
        theta,
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=12,
        lambda_=1e-6,
        shock_solver="rom",
        batch_replay=True,
    )

    np.testing.assert_allclose(np.asarray(shocks_jax), shocks_py, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ll_jax), ll_py, rtol=1e-9, atol=1e-9)


def test_numpyro_surrogate_log_density_wraps_jax_likelihood() -> None:
    numpyro = pytest.importorskip("numpyro")
    dist = pytest.importorskip("numpyro.distributions")
    frozen_zero = _constant_residual_mlp(d_in=4, d_out=1, value=0.0)
    obs = np.asarray([[1.0, 0.5]], dtype=np.float64)
    obs_sigma = np.asarray([0.1], dtype=np.float64)
    shock_sigmas = np.asarray([0.5, 0.0], dtype=np.float64)
    priors = {"theta": dist.Normal(0.0, 1.0)}
    samples = {"theta": jnp.asarray(0.1, dtype=jnp.float64)}

    model = build_numpyro_surrogate_inversion_model_jax(
        _toy_split_predict_jax,
        frozen_zero,
        [0.0],
        obs,
        obs_sigma,
        shock_sigmas,
        priors,
        parameter_names=("theta",),
        maxit=8,
        lambda_=1e-6,
    )
    assert callable(model)

    log_joint = evaluate_numpyro_surrogate_log_density_jax(
        _toy_split_predict_jax,
        frozen_zero,
        [0.0],
        obs,
        obs_sigma,
        shock_sigmas,
        priors,
        samples,
        parameter_names=("theta",),
        maxit=8,
        lambda_=1e-6,
    )
    likelihood = surrogate_inversion_loglikelihood_jax(
        _toy_split_predict_jax,
        frozen_zero,
        [0.0],
        jnp.asarray([0.1], dtype=jnp.float64),
        obs,
        obs_sigma,
        shock_sigmas,
        maxit=8,
        lambda_=1e-6,
    )
    expected = likelihood + priors["theta"].log_prob(samples["theta"])
    np.testing.assert_allclose(np.asarray(log_joint), np.asarray(expected), rtol=1e-10, atol=1e-10)
    assert numpyro.__version__
