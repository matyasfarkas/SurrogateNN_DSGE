from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SA
import pytest

from surrogatenn_dsge import (
    RegimeSwitchConfig,
    assemble_parameter_vector,
    build_linear_state_space_from_model,
    build_numpyro_kalman_model,
    build_numpyro_kalman_model_jax,
    build_numpyro_switching_filter_model_jax,
    build_numpyro_switching_model_jax,
    compute_linear_gate_stats_from_filter,
    evaluate_numpyro_switching_filter_log_density_jax,
    evaluate_numpyro_kalman_log_density,
    evaluate_numpyro_kalman_log_density_jax,
    evaluate_numpyro_switching_log_density_jax,
    gate_probabilities,
    kalman_loglikelihood_from_model,
    kalman_loglikelihood_from_model_jax,
    parse_macro_model,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
    switching_loglikelihood_from_model,
    switching_loglikelihood_from_model_filter_gates_jax,
    switching_loglikelihood_from_model_jax,
)


LIKELIHOOD_SOURCE = """
@model parsed_loglikelihood begin
    a[0] = rho_a * a[-1] + (1 - rho_a) * a_bar + eps_a[x]
    y[0] = rho_y * y[-1] + (1 - rho_y) * y_bar + alpha * (a[0] - a_bar) + eps_y[x]
end

@parameters parsed_loglikelihood begin
    0 < rho_a < 1
    0 < rho_y < 1
    alpha = 0.4
    a_bar = 1.5
    y_bar = 2.0
    rho_a = 0.8
    rho_y = 0.6
end
"""


CALIBRATED_LIKELIHOOD_SOURCE = """
@model calibrated_loglikelihood begin
    x[0] = rho * x[-1] + (1 - rho) * mu + eps_x[x]
end

@parameters calibrated_loglikelihood begin
    target = theta + 1
    x[ss] = target | mu
    theta = 2
    0 < rho < 1
    rho = 0.8
end
"""


SWITCHING_SOURCE = """
@model switching_linear begin
    y[0] = rho * y[-1] + eps[x]
end

@parameters switching_linear begin
    0 < rho < 1
    rho = 0.65
end
"""


def _numpyro_fixture():
    model = parse_macro_model(LIKELIHOOD_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )
    observables = ("y", "a")
    state_space = build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order_result,
    )
    simulation = simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(0),
        num_periods=12,
    )
    steady_lookup = dict(zip(model.timings.var, np.asarray(first_order_result.steady_state)))
    levels = simulation.observations + np.asarray(
        [[steady_lookup[name]] for name in observables],
        dtype=np.float64,
    )
    priors = {
        "rho_a": dist.Uniform(0.05, 0.95),
        "rho_y": dist.Uniform(0.05, 0.95),
    }
    return model, first_order_result, observables, levels, priors


def _calibrated_numpyro_fixture():
    model = parse_macro_model(CALIBRATED_LIKELIHOOD_SOURCE)
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"x": 3.0},
    )
    observables = ("x",)
    state_space = build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order_result,
    )
    simulation = simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(3),
        num_periods=12,
    )
    steady_lookup = dict(zip(model.timings.var, np.asarray(first_order_result.steady_state)))
    levels = simulation.observations + np.asarray(
        [[steady_lookup["x"]]],
        dtype=np.float64,
    )
    priors = {
        "rho": dist.Uniform(0.05, 0.95),
    }
    return model, first_order_result, observables, levels, priors


def _switching_numpyro_fixture():
    model = parse_macro_model(SWITCHING_SOURCE)
    observables = ("y",)
    levels = np.asarray([[0.1, -0.05, 0.12, 0.03, -0.02]], dtype=np.float64)
    priors = {
        "rho": dist.Uniform(0.05, 0.95),
    }
    gate_probs = np.asarray([0.1, 0.35, 0.5, 0.7, 0.9], dtype=np.float64)
    return model, observables, levels, priors, gate_probs


def _switching_filter_numpyro_fixture():
    model = parse_macro_model(SWITCHING_SOURCE)
    observables = ("y",)
    levels = np.asarray([[0.1, -0.05, 0.12, 0.03, -0.02]], dtype=np.float64)
    priors = {
        "rho": dist.Uniform(0.05, 0.95),
    }
    shock_name = model.timings.exo[0]
    regime_config = RegimeSwitchConfig(
        gate_mode="soft",
        tau_eps=0.05,
        tau_y=0.05,
        beta_eps=2.0,
        beta_y=1.5,
        bias=-0.1,
        prob_floor=1e-4,
        prob_ceiling=1.0 - 1e-4,
    )
    obs_sigma = {"y": 0.1}
    shock_sigmas = {shock_name: 0.5}
    return model, observables, levels, priors, obs_sigma, shock_sigmas, regime_config


def test_assemble_parameter_vector_overrides_subset() -> None:
    model, _, _, _, _ = _numpyro_fixture()

    parameter_vector = assemble_parameter_vector(
        model,
        {"rho_y": jnp.asarray(0.7, dtype=jnp.float64)},
        base_parameter_values={"rho_a": 0.75},
    )

    expected = np.asarray(model.parameter_values).copy()
    expected[model.parameter_names.index("rho_a")] = 0.75
    expected[model.parameter_names.index("rho_y")] = 0.7
    np.testing.assert_allclose(parameter_vector, expected, rtol=1e-10, atol=1e-10)


def test_explicit_parameter_values_override_direct_defaults_in_first_order_solution() -> None:
    model, _, _, _, _ = _numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {
            "rho_a": jnp.asarray(0.33, dtype=jnp.float64),
            "rho_y": jnp.asarray(0.89, dtype=jnp.float64),
        },
    )

    resolved = model.resolve_parameter_values(parameter_values=parameter_vector)
    np.testing.assert_allclose(resolved, parameter_vector, rtol=0.0, atol=1e-12)

    first_order_result = solve_first_order_model(
        model,
        parameter_values=parameter_vector,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )

    np.testing.assert_allclose(
        first_order_result.parameter_values,
        parameter_vector,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_order_result.jacobian[0, 2],
        -0.33,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_order_result.jacobian[1, 3],
        -0.89,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_order_result.solution.qme_solution,
        jnp.asarray([[0.33, 0.0], [0.132, 0.89]], dtype=jnp.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_numpyro_log_density_matches_manual_prior_plus_likelihood() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    parameter_samples = {
        "rho_a": jnp.asarray(0.8, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.6, dtype=jnp.float64),
    }

    log_density = evaluate_numpyro_kalman_log_density(
        model,
        levels,
        priors,
        parameter_samples,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )
    parameter_vector = assemble_parameter_vector(model, parameter_samples)
    manual_log_density = (
        priors["rho_a"].log_prob(parameter_samples["rho_a"])
        + priors["rho_y"].log_prob(parameter_samples["rho_y"])
        + kalman_loglikelihood_from_model(
            model,
            levels,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state_initial_guess={"a": 1.5, "y": 2.0},
        )
    )

    np.testing.assert_allclose(
        log_density,
        manual_log_density,
        rtol=1e-10,
        atol=1e-10,
    )


def test_numpyro_log_density_accepts_schur_qme_algorithm() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    parameter_samples = {
        "rho_a": jnp.asarray(0.8, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.6, dtype=jnp.float64),
    }

    log_density = evaluate_numpyro_kalman_log_density(
        model,
        levels,
        priors,
        parameter_samples,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    parameter_vector = assemble_parameter_vector(model, parameter_samples)
    manual_log_density = (
        priors["rho_a"].log_prob(parameter_samples["rho_a"])
        + priors["rho_y"].log_prob(parameter_samples["rho_y"])
        + kalman_loglikelihood_from_model(
            model,
            levels,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state_initial_guess={"a": 1.5, "y": 2.0},
            qme_algorithm="schur",
        )
    )

    np.testing.assert_allclose(
        log_density,
        manual_log_density,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_fixed_steady_state_loglikelihood_matches_high_level_path() -> None:
    model, first_order_result, observables, levels, _ = _numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho_a": jnp.asarray(0.8, dtype=jnp.float64), "rho_y": jnp.asarray(0.6, dtype=jnp.float64)},
    )

    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
        )
    )
    jax_loglikelihood = compiled(parameter_vector)
    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        parameter_values=parameter_vector,
        steady_state=first_order_result.steady_state,
    )

    np.testing.assert_allclose(
        jax_loglikelihood,
        high_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_fixed_steady_state_loglikelihood_defaults_to_schur() -> None:
    model, first_order_result, observables, levels, _ = _numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho_a": jnp.asarray(0.8, dtype=jnp.float64), "rho_y": jnp.asarray(0.6, dtype=jnp.float64)},
    )

    default_value = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
        )
    )(parameter_vector)
    schur_value = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
            qme_algorithm="schur",
        )
    )(parameter_vector)

    np.testing.assert_allclose(default_value, schur_value, rtol=1e-10, atol=1e-10)


def test_jax_schur_fixed_steady_state_loglikelihood_matches_high_level_path() -> None:
    model, _, observables, levels, _ = _numpyro_fixture()
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho_a": jnp.asarray(0.8, dtype=jnp.float64), "rho_y": jnp.asarray(0.6, dtype=jnp.float64)},
    )

    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
            qme_algorithm="schur",
        )
    )
    jax_loglikelihood = compiled(parameter_vector)
    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_vector,
        steady_state=first_order_result.steady_state,
    )

    np.testing.assert_allclose(
        jax_loglikelihood,
        high_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_auto_steady_state_loglikelihood_matches_high_level_path() -> None:
    model, _, observables, levels, _ = _numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho_a": jnp.asarray(0.8, dtype=jnp.float64), "rho_y": jnp.asarray(0.6, dtype=jnp.float64)},
    )

    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state_initial_guess={"a": 1.5, "y": 2.0},
        )
    )
    jax_loglikelihood = compiled(parameter_vector)
    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        parameter_values=parameter_vector,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )

    np.testing.assert_allclose(
        jax_loglikelihood,
        high_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_numpyro_log_density_matches_manual_prior_plus_likelihood() -> None:
    model, first_order_result, observables, levels, priors = _numpyro_fixture()
    parameter_samples = {
        "rho_a": jnp.asarray(0.8, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.6, dtype=jnp.float64),
    }

    log_density = evaluate_numpyro_kalman_log_density_jax(
        model,
        levels,
        priors,
        parameter_samples,
        observables=observables,
        steady_state=first_order_result.steady_state,
    )
    parameter_vector = assemble_parameter_vector(model, parameter_samples)
    manual_log_density = (
        priors["rho_a"].log_prob(parameter_samples["rho_a"])
        + priors["rho_y"].log_prob(parameter_samples["rho_y"])
        + kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=parameter_vector,
            steady_state=first_order_result.steady_state,
        )
    )

    np.testing.assert_allclose(
        log_density,
        manual_log_density,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_loglikelihood_changes_when_sampled_parameters_change() -> None:
    model, first_order_result, observables, levels, priors = _numpyro_fixture()
    baseline_samples = {
        "rho_a": jnp.asarray(0.8, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.6, dtype=jnp.float64),
    }
    alternative_samples = {
        "rho_a": jnp.asarray(0.33, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.89, dtype=jnp.float64),
    }

    baseline_vector = assemble_parameter_vector(model, baseline_samples)
    alternative_vector = assemble_parameter_vector(model, alternative_samples)
    baseline_likelihood = kalman_loglikelihood_from_model_jax(
        model,
        levels,
        observables=observables,
        parameter_values=baseline_vector,
        steady_state=first_order_result.steady_state,
        qme_algorithm="schur",
    )
    alternative_likelihood = kalman_loglikelihood_from_model_jax(
        model,
        levels,
        observables=observables,
        parameter_values=alternative_vector,
        steady_state=first_order_result.steady_state,
        qme_algorithm="schur",
    )
    baseline_density = evaluate_numpyro_kalman_log_density_jax(
        model,
        levels,
        priors,
        baseline_samples,
        observables=observables,
        steady_state=first_order_result.steady_state,
        qme_algorithm="schur",
    )
    alternative_density = evaluate_numpyro_kalman_log_density_jax(
        model,
        levels,
        priors,
        alternative_samples,
        observables=observables,
        steady_state=first_order_result.steady_state,
        qme_algorithm="schur",
    )

    assert not np.isclose(
        float(baseline_likelihood),
        float(alternative_likelihood),
        rtol=0.0,
        atol=1e-8,
    )
    assert not np.isclose(
        float(baseline_density),
        float(alternative_density),
        rtol=0.0,
        atol=1e-8,
    )


def test_numpyro_model_records_parameter_vector_and_loglikelihood() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    parameter_samples = {
        "rho_a": jnp.asarray(0.82, dtype=jnp.float64),
        "rho_y": jnp.asarray(0.58, dtype=jnp.float64),
    }
    numpyro_model = build_numpyro_kalman_model(
        model,
        levels,
        priors,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )

    traced = handlers.trace(
        handlers.seed(
            handlers.substitute(numpyro_model, data=parameter_samples),
            rng_seed=0,
        )
    ).get_trace()

    assert "parameter_vector" in traced
    assert "loglikelihood" in traced
    assert "kalman_loglikelihood" in traced
    np.testing.assert_allclose(
        traced["parameter_vector"]["value"],
        assemble_parameter_vector(model, parameter_samples),
        rtol=1e-10,
        atol=1e-10,
    )


def test_numpyro_wrapper_fails_fast_for_compiled_structural_kernels() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    numpyro_model = build_numpyro_kalman_model(
        model,
        levels,
        priors,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )
    kernel = SA(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=1, num_samples=1, num_chains=1, progress_bar=False)

    with pytest.raises(NotImplementedError, match="not yet JAX-traceable"):
        mcmc.run(jax.random.PRNGKey(1))


def test_jax_wrapper_runs_nuts_with_auto_steady_state() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    numpyro_model = build_numpyro_kalman_model_jax(
        model,
        levels,
        priors,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
    )
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=4, num_chains=1, progress_bar=False)

    mcmc.run(jax.random.PRNGKey(2))
    samples = mcmc.get_samples()

    assert samples["rho_a"].shape == (4,)
    assert samples["rho_y"].shape == (4,)


def test_jax_wrapper_runs_nuts_with_auto_steady_state_and_schur() -> None:
    model, _, observables, levels, priors = _numpyro_fixture()
    numpyro_model = build_numpyro_kalman_model_jax(
        model,
        levels,
        priors,
        observables=observables,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=4, num_chains=1, progress_bar=False)

    mcmc.run(jax.random.PRNGKey(22))
    samples = mcmc.get_samples()

    assert samples["rho_a"].shape == (4,)
    assert samples["rho_y"].shape == (4,)


def test_jax_calibrated_auto_steady_state_loglikelihood_matches_high_level_path() -> None:
    model, _, observables, levels, _ = _calibrated_numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho": jnp.asarray(0.8, dtype=jnp.float64)},
    )

    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            parameter_values=theta,
            steady_state_initial_guess={"x": 3.0},
        )
    )
    jax_loglikelihood = compiled(parameter_vector)
    high_level = kalman_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        parameter_values=parameter_vector,
        steady_state_initial_guess={"x": 3.0},
    )

    np.testing.assert_allclose(
        jax_loglikelihood,
        high_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_calibrated_wrapper_runs_nuts() -> None:
    model, _, observables, levels, priors = _calibrated_numpyro_fixture()
    numpyro_model = build_numpyro_kalman_model_jax(
        model,
        levels,
        priors,
        observables=observables,
        steady_state_initial_guess={"x": 3.0},
    )
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=4, num_chains=1, progress_bar=False)

    mcmc.run(jax.random.PRNGKey(4))
    samples = mcmc.get_samples()

    assert samples["rho"].shape == (4,)


def test_jax_switching_loglikelihood_matches_high_level_path() -> None:
    model, observables, levels, _, gate_probs = _switching_numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho": jnp.asarray(0.65, dtype=jnp.float64)},
    )

    compiled = jax.jit(
        lambda theta: switching_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            gate_probs=gate_probs,
            fom_algorithm="first_order",
            parameter_values=theta,
            steady_state_initial_guess={"y": 0.0},
            measurement_error_scale=0.0,
            qme_algorithm="schur",
        )
    )
    jax_loglikelihood = compiled(parameter_vector)
    high_level = switching_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        gate_probs=gate_probs,
        fom_algorithm="first_order",
        parameter_values=parameter_vector,
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    ).total

    np.testing.assert_allclose(
        jax_loglikelihood,
        high_level,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_switching_log_density_matches_manual_prior_plus_likelihood() -> None:
    model, observables, levels, priors, gate_probs = _switching_numpyro_fixture()
    parameter_samples = {
        "rho": jnp.asarray(0.65, dtype=jnp.float64),
    }

    log_density = evaluate_numpyro_switching_log_density_jax(
        model,
        levels,
        priors,
        parameter_samples,
        observables=observables,
        gate_probs=gate_probs,
        fom_algorithm="first_order",
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    )
    parameter_vector = assemble_parameter_vector(model, parameter_samples)
    manual_log_density = (
        priors["rho"].log_prob(parameter_samples["rho"])
        + switching_loglikelihood_from_model_jax(
            model,
            levels,
            observables=observables,
            gate_probs=gate_probs,
            fom_algorithm="first_order",
            parameter_values=parameter_vector,
            steady_state_initial_guess={"y": 0.0},
            measurement_error_scale=0.0,
            qme_algorithm="schur",
        )
    )

    np.testing.assert_allclose(
        log_density,
        manual_log_density,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_switching_wrapper_runs_nuts_with_schur() -> None:
    model, observables, levels, priors, gate_probs = _switching_numpyro_fixture()
    numpyro_model = build_numpyro_switching_model_jax(
        model,
        levels,
        priors,
        observables=observables,
        gate_probs=gate_probs,
        fom_algorithm="first_order",
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    )
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=4, num_chains=1, progress_bar=False)

    mcmc.run(jax.random.PRNGKey(24))
    samples = mcmc.get_samples()

    assert samples["rho"].shape == (4,)


def test_jax_switching_filter_loglikelihood_matches_composed_high_level_path() -> None:
    (
        model,
        observables,
        levels,
        _,
        obs_sigma,
        shock_sigmas,
        regime_config,
    ) = _switching_filter_numpyro_fixture()
    parameter_vector = assemble_parameter_vector(
        model,
        {"rho": jnp.asarray(0.65, dtype=jnp.float64)},
    )

    expected_gate_stats = compute_linear_gate_stats_from_filter(
        model,
        levels,
        obs_sigma,
        shock_sigmas,
        ("y",),
        observables=observables,
        parameter_values=parameter_vector,
        steady_state_initial_guess={"y": 0.0},
        qme_algorithm="schur",
        filter="kalman",
    )
    expected_gate_probs = gate_probabilities(
        np.asarray(expected_gate_stats.e_stat, dtype=np.float64),
        np.asarray(expected_gate_stats.f_stat, dtype=np.float64),
        regime_config,
    )
    expected_loglikelihood = switching_loglikelihood_from_model(
        model,
        levels,
        observables=observables,
        gate_probs=expected_gate_probs,
        fom_algorithm="first_order",
        parameter_values=parameter_vector,
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    ).total

    compiled = jax.jit(
        lambda theta: switching_loglikelihood_from_model_filter_gates_jax(
            model,
            levels,
            obs_sigma,
            shock_sigmas,
            regime_switch_config=regime_config,
            state_names=("y",),
            observables=observables,
            gate_filter="kalman",
            fom_algorithm="first_order",
            parameter_values=theta,
            steady_state_initial_guess={"y": 0.0},
            measurement_error_scale=0.0,
            qme_algorithm="schur",
        )
    )
    loglikelihood = compiled(parameter_vector)

    np.testing.assert_allclose(
        loglikelihood,
        expected_loglikelihood,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_switching_filter_log_density_matches_manual_prior_plus_likelihood() -> None:
    (
        model,
        observables,
        levels,
        priors,
        obs_sigma,
        shock_sigmas,
        regime_config,
    ) = _switching_filter_numpyro_fixture()
    parameter_samples = {
        "rho": jnp.asarray(0.65, dtype=jnp.float64),
    }

    log_density = evaluate_numpyro_switching_filter_log_density_jax(
        model,
        levels,
        priors,
        parameter_samples,
        obs_sigma,
        shock_sigmas,
        regime_switch_config=regime_config,
        state_names=("y",),
        observables=observables,
        gate_filter="kalman",
        fom_algorithm="first_order",
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    )
    parameter_vector = assemble_parameter_vector(model, parameter_samples)
    manual_log_density = (
        priors["rho"].log_prob(parameter_samples["rho"])
        + switching_loglikelihood_from_model_filter_gates_jax(
            model,
            levels,
            obs_sigma,
            shock_sigmas,
            regime_switch_config=regime_config,
            state_names=("y",),
            observables=observables,
            gate_filter="kalman",
            fom_algorithm="first_order",
            parameter_values=parameter_vector,
            steady_state_initial_guess={"y": 0.0},
            measurement_error_scale=0.0,
            qme_algorithm="schur",
        )
    )

    np.testing.assert_allclose(
        log_density,
        manual_log_density,
        rtol=1e-10,
        atol=1e-10,
    )


def test_jax_switching_filter_wrapper_runs_nuts_with_schur() -> None:
    (
        model,
        observables,
        levels,
        priors,
        obs_sigma,
        shock_sigmas,
        regime_config,
    ) = _switching_filter_numpyro_fixture()
    numpyro_model = build_numpyro_switching_filter_model_jax(
        model,
        levels,
        priors,
        obs_sigma,
        shock_sigmas,
        regime_switch_config=regime_config,
        state_names=("y",),
        observables=observables,
        gate_filter="kalman",
        fom_algorithm="first_order",
        steady_state_initial_guess={"y": 0.0},
        measurement_error_scale=0.0,
        qme_algorithm="schur",
    )
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=4, num_chains=1, progress_bar=False)

    mcmc.run(jax.random.PRNGKey(25))
    samples = mcmc.get_samples()

    assert samples["rho"].shape == (4,)
