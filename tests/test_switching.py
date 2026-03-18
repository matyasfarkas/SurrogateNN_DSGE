from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    SwitchingLikelihoodConfig,
    compute_switching_loglikelihood,
    inversion_loglikelihood_per_period_from_model,
    kalman_loglikelihood_per_period_from_model,
    mix_loglikelihood,
    parse_macro_model,
    solve_first_order_model,
    switching_loglikelihood_from_model,
)


SWITCHING_SOURCE = """
@model switching_linear begin
    y[0] = rho * y[-1] + eps[x]
end

@parameters switching_linear begin
    0 < rho < 1
    rho = 0.65
end
"""


def _switching_fixture():
    model = parse_macro_model(SWITCHING_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"y": 0.0},
    )
    levels = np.asarray([[0.1, -0.05, 0.12, 0.03, -0.02]], dtype=np.float64)
    return model, first_order_result, levels


def test_compute_switching_loglikelihood_matches_manual_formulas() -> None:
    ll_rom = jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64)
    ll_fom = jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64)
    gate_probs = jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64)

    linear = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="linear"),
    )
    logsumexp = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )
    hard = compute_switching_loglikelihood(
        ll_rom,
        ll_fom,
        hard_mask=[False, True, True],
    )

    np.testing.assert_allclose(
        linear.per_period,
        gate_probs * ll_fom + (1.0 - gate_probs) * ll_rom,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        logsumexp.total,
        mix_loglikelihood(ll_fom, ll_rom, gate_probs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hard.per_period,
        np.asarray([-2.0, -2.5, -0.25], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_switching_loglikelihood_is_jittable() -> None:
    compiled = jax.jit(
        lambda rom, fom, probs: compute_switching_loglikelihood(
            rom,
            fom,
            gate_probs=probs,
            config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
        ).total
    )

    total = compiled(
        jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64),
        jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64),
        jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64),
    )

    np.testing.assert_allclose(
        total,
        mix_loglikelihood(
            jnp.asarray([-1.0, -2.5, -0.25], dtype=jnp.float64),
            jnp.asarray([-2.0, -1.5, -0.5], dtype=jnp.float64),
            jnp.asarray([0.2, 0.7, 0.5], dtype=jnp.float64),
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_model_switching_bridge_matches_manual_component_mix() -> None:
    model, first_order_result, levels = _switching_fixture()
    gate_probs = np.asarray([0.1, 0.35, 0.5, 0.7, 0.9], dtype=np.float64)

    rom = kalman_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
    )
    fom = inversion_loglikelihood_per_period_from_model(
        model,
        levels,
        observables=("y",),
        algorithm="first_order",
        first_order_result=first_order_result,
    )
    manual = compute_switching_loglikelihood(
        rom,
        fom,
        gate_probs=gate_probs,
        config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )
    bridged = switching_loglikelihood_from_model(
        model,
        levels,
        observables=("y",),
        gate_probs=gate_probs,
        fom_algorithm="first_order",
        first_order_result=first_order_result,
        measurement_error_scale=0.0,
        switching_config=SwitchingLikelihoodConfig(soft_mixture="logsumexp"),
    )

    np.testing.assert_allclose(bridged.total, manual.total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        bridged.per_period,
        manual.per_period,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bridged.gate_probs,
        gate_probs,
        rtol=1e-12,
        atol=1e-12,
    )
