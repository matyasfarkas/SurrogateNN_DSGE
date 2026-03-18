from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surrogatenn_dsge import (
    SEPConfig,
    get_irf,
    parse_macro_model,
    simulate_model,
)


LINEAR_IRF_SOURCE = """
@model linear_irf begin
    y[0] = rho * y[-1] + eps_y[x]
end

@parameters linear_irf begin
    rho = 0.5
end
"""


OBC_MAX_SOURCE = """
@model obc_linear begin
    r[0] = max(r_star[0], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_linear begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_MODEL_DIR = _ROOT / "SurrogateNN_Estimation.jl" / "models"

_RUNTIME_SMOKE_MODELS = (
    pytest.param(
        _UPSTREAM_MODEL_DIR / "RBC_Dynare.jl",
        {
            "Capital": 10.0,
            "Consumption": 0.8,
            "Efficiency": 1.0,
            "Investment": 0.2,
            "Labour": 0.3,
            "Output": 1.0,
            "efficiency": 0.0,
        },
        id="rbc_dynare",
    ),
    pytest.param(
        _UPSTREAM_MODEL_DIR / "FS2000.jl",
        {
            "P": 1.0,
            "R": 1.0,
            "W": 1.0,
            "c": 0.8,
            "d": 0.0,
            "dA": 1.01,
            "e": 1.0,
            "gp_obs": 1.0,
            "gy_obs": 1.01,
            "k": 8.0,
            "l": 0.9,
            "log_gp_obs": 0.0,
            "log_gy_obs": 0.01,
            "m": 1.0,
            "n": 0.3,
            "y": 1.0,
        },
        id="fs2000",
    ),
    pytest.param(
        _UPSTREAM_MODEL_DIR / "RBC_baseline.jl",
        {
            "c": 0.55,
            "g": 0.20,
            "i": 0.25,
            "k": 10.4,
            "l": 1.0 / 3.0,
            "r": 0.128,
            "w": 2.0,
            "y": 1.0,
            "z": 1.0,
        },
        id="rbc_baseline",
    ),
)


def test_get_irf_first_order_matches_closed_form_linear_response() -> None:
    model = parse_macro_model(LINEAR_IRF_SOURCE)

    result = get_irf(
        model,
        periods=4,
        variables=("y",),
        shocks="eps_y",
        shock_size=2.0,
    )

    expected = np.asarray([2.0, 1.0, 0.5, 0.25], dtype=np.float64).reshape(1, 4, 1)
    assert result.algorithm_used == "first_order"
    assert result.variables == ("y",)
    assert result.shock_names == ("eps_y",)
    np.testing.assert_allclose(
        np.asarray(result.responses, dtype=np.float64),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_simulate_model_matches_linear_first_order_rollout() -> None:
    model = parse_macro_model(LINEAR_IRF_SOURCE)

    result = simulate_model(
        model,
        periods=4,
        variables=("y",),
        shocks={"eps_y": [2.0, 0.0, 0.0, 0.0]},
    )

    expected = np.asarray([2.0, 1.0, 0.5, 0.25], dtype=np.float64).reshape(1, 4)
    assert result.algorithm_used == "first_order"
    np.testing.assert_allclose(
        np.asarray(result.data, dtype=np.float64),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_get_irf_routes_obc_models_through_sep_unless_ignored() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)

    ignored = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        ignore_obc=True,
    )
    enforced = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert ignored.algorithm_used == "first_order"
    assert enforced.algorithm_used == "stochastic_extended_path"
    assert np.min(np.asarray(ignored.responses, dtype=np.float64)) < 1.0 - 1e-3
    assert np.all(np.asarray(enforced.responses, dtype=np.float64) >= 1.0 - 1e-8)


@pytest.mark.parametrize(
    ("model_path", "steady_state_initial_guess"),
    _RUNTIME_SMOKE_MODELS,
)
def test_upstream_models_support_irf_and_simulation_runtime_helpers(
    model_path: Path,
    steady_state_initial_guess: dict[str, float],
) -> None:
    model = parse_macro_model(model_path.read_text())
    variable = (model.steady_state_names[0],)

    irf = get_irf(
        model,
        periods=3,
        variables=variable,
        shocks="all",
        steady_state_initial_guess=steady_state_initial_guess,
    )
    sim = simulate_model(
        model,
        periods=3,
        variables=variable,
        shocks=np.zeros((model.timings.nExo, 3), dtype=np.float64),
        steady_state_initial_guess=steady_state_initial_guess,
    )

    assert irf.responses.shape[0] == 1
    assert irf.responses.shape[1] == 3
    assert irf.responses.shape[2] >= 1
    assert sim.data.shape == (1, 3)
    assert np.all(np.isfinite(np.asarray(irf.responses, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(sim.data, dtype=np.float64)))
