from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surrogatenn_dsge import (
    ParameterSpec,
    baseline_parameter_vector,
    format_parameter_summary,
    get_parameter_bounds,
    get_parameter_names,
    get_parameter_priors,
    get_parameter_specs,
    get_phase1_18param_baseline,
    make_numpyro_priors,
    parameter_bounds_array,
    theta_within_bounds,
    theta_within_bounds_jax,
)


EXPECTED_NAMES = {
    "legacy_3params": ("cprobp", "cindp", "curvp"),
    "phase1_18params": (
        "crhoa",
        "crhob",
        "crhog",
        "crhoqs",
        "crhopinf",
        "crhow",
        "crhoms",
        "z_ea",
        "z_eb",
        "z_eg",
        "z_eqs",
        "z_epinf",
        "z_ew",
        "z_em",
        "cprobp",
        "cindp",
        "curvp",
        "cprobw",
    ),
    "phase1_18params_narrow": (
        "crhoa",
        "crhob",
        "crhog",
        "crhoqs",
        "crhopinf",
        "crhow",
        "crhoms",
        "z_ea",
        "z_eb",
        "z_eg",
        "z_eqs",
        "z_epinf",
        "z_ew",
        "z_em",
        "cprobp",
        "cindp",
        "curvp",
        "cprobw",
    ),
    "investment_4p": ("crhob", "crhoqs", "z_eb", "z_eqs"),
    "investment_4p_supported": ("crhob", "crhoqs", "z_eb", "z_eqs"),
    "investment_curvature_5p": ("csadjcost", "crhob", "crhoqs", "z_eb", "z_eqs"),
}


@pytest.mark.parametrize("parameter_set, expected_names", EXPECTED_NAMES.items())
def test_parameter_sets_expose_expected_names_and_bounds(parameter_set: str, expected_names: tuple[str, ...]) -> None:
    specs = get_parameter_specs(parameter_set)

    assert get_parameter_names(":" + parameter_set) == expected_names
    assert tuple(spec.name for spec in specs) == expected_names
    assert len(get_parameter_bounds(parameter_set)) == len(specs)

    lower, upper = parameter_bounds_array(specs)
    assert lower.shape == upper.shape == (len(specs),)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(lower < upper)


def test_phase1_baseline_and_vectors_match_hlt_calibration() -> None:
    baseline = get_phase1_18param_baseline()

    assert baseline["csadjcost"] == 6.0144
    assert baseline["crhoa"] == 0.9977
    assert baseline["crhoqs"] == 0.7165
    assert baseline["crhopinf"] == 0.0
    assert baseline["z_eb"] == 1.8513
    assert baseline["cprobw"] == 0.8087

    np.testing.assert_allclose(
        baseline_parameter_vector("investment_curvature_5p"),
        np.asarray([6.0144, 0.5799, 0.7165, 1.8513, 0.6017], dtype=np.float64),
        rtol=0,
        atol=0,
    )
    assert theta_within_bounds(baseline_parameter_vector("investment_4p"), "investment_4p")


def test_supported_investment_bounds_reject_unmapped_risk_premium_volatility() -> None:
    wide_bounds = get_parameter_bounds("investment_4p")
    supported_bounds = get_parameter_bounds("investment_4p_supported")

    assert wide_bounds["z_eb"] == (1.20, 2.50)
    assert supported_bounds["z_eb"] == (1.20, 1.85)

    assert theta_within_bounds([0.5799, 0.7165, 1.85, 0.6017], "investment_4p_supported")
    assert not theta_within_bounds([0.5799, 0.7165, 1.8513, 0.6017], "investment_4p_supported")
    assert not theta_within_bounds([0.5799, 0.7165, 2.50, 0.6017], "investment_4p_supported")


def test_bounds_check_rejects_bad_shape_and_nonfinite_values() -> None:
    assert not theta_within_bounds([0.6, math.nan, 75.0], "legacy_3params")
    with pytest.raises(ValueError, match="theta length mismatch"):
        theta_within_bounds([0.6, 0.5], "legacy_3params")


def test_jax_bounds_check_is_jittable() -> None:
    specs = get_parameter_specs("legacy_3params")
    check = jax.jit(lambda theta: theta_within_bounds_jax(theta, specs))

    assert bool(check(jnp.asarray([0.6, 0.5, 75.0], dtype=jnp.float64)))
    assert not bool(check(jnp.asarray([0.4, 0.5, 75.0], dtype=jnp.float64)))


def test_numpyro_priors_are_finite_inside_support_and_alias_matches() -> None:
    numpyro = pytest.importorskip("numpyro")
    log_density = pytest.importorskip("numpyro.infer.util").log_density

    priors = make_numpyro_priors("legacy_3params")
    alias_priors = get_parameter_priors("legacy_3params")

    assert set(priors) == {"cprobp", "cindp", "curvp"}
    assert set(alias_priors) == set(priors)
    for name, value in {"cprobp": 0.7, "cindp": 0.5, "curvp": 75.0}.items():
        assert bool(jnp.isfinite(priors[name].log_prob(jnp.asarray(value, dtype=jnp.float64))))
        assert bool(jnp.isfinite(alias_priors[name].log_prob(jnp.asarray(value, dtype=jnp.float64))))

    assert not bool(jnp.isfinite(priors["cprobp"].log_prob(jnp.asarray(0.49, dtype=jnp.float64))))
    assert not bool(jnp.isfinite(priors["curvp"].log_prob(jnp.asarray(151.0, dtype=jnp.float64))))

    def model() -> None:
        for name, prior in priors.items():
            numpyro.sample(name, prior)

    inside_params = {
        "cprobp": jnp.asarray(0.7, dtype=jnp.float64),
        "cindp": jnp.asarray(0.5, dtype=jnp.float64),
        "curvp": jnp.asarray(75.0, dtype=jnp.float64),
    }
    outside_params = dict(inside_params)
    outside_params["curvp"] = jnp.asarray(151.0, dtype=jnp.float64)
    assert bool(jnp.isfinite(log_density(model, (), {}, inside_params)[0]))
    assert not bool(jnp.isfinite(log_density(model, (), {}, outside_params)[0]))


def test_parameter_summary_and_spec_validation() -> None:
    summary = format_parameter_summary("legacy_3params")

    assert "Parameter Set: legacy_3params" in summary
    assert "Number of parameters: 3" in summary
    assert "cprobp" in summary

    with pytest.raises(ValueError, match="Unsupported prior_type"):
        ParameterSpec("bad", "Gamma", {"shape": 1.0}, (0.0, 1.0), "bad")
    with pytest.raises(ValueError, match="Invalid bounds"):
        ParameterSpec("bad", "Normal", {"mu": 0.0, "sigma": 1.0}, (1.0, 1.0), "bad")
