from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import surrogatenn_dsge.model as model_module

from surrogatenn_dsge import (
    SEPConfig,
    get_irf,
    homotopy_chained_trajectory,
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


OBC_AUX_SHOCK_SOURCE = """
@model obc_aux_shock begin
    r[0] = max(r_star[0] + eps_zlbᵒᵇᶜ[x], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_aux_shock begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


OBC_LOG_SOURCE = """
@model obc_log begin
    log(r[0]) = max(log(r_star[0]), log(zlb))
    log(r_star[0]) = rho * log(r_star[-1]) + (1-rho) * log(mu) + eps_r[x]
end

@parameters obc_log begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


OBC_LOG_AUX_SHOCK_SOURCE = """
@model obc_log_aux_shock begin
    log(r[0]) = max(log(r_star[0]) + eps_zlbᵒᵇᶜ[x], log(zlb))
    log(r_star[0]) = rho * log(r_star[-1]) + (1-rho) * log(mu) + eps_r[x]
end

@parameters obc_log_aux_shock begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


OBC_HORIZON_AUX_SHOCK_SOURCE = """
@model obc_horizon_aux_shock max_obc_horizon = 2 begin
    r[0] = max(r_star[0] + eps_zlbᵒᵇᶜ[x], zlb)
    r_star[0] = rho * r_star[-1] + (1-rho) * mu + eps_r[x]
end

@parameters obc_horizon_aux_shock begin
    rho = 0.8
    mu = 1.2
    zlb = 1.0
end
"""


OBC_MIN_COMPLEMENTARITY_SOURCE = """
@model obc_min_complementarity begin
    0 = min(bnot[0] - b[0], lm[0])
    bnot[0] = b_bar
    b[0] = b_bar + eps_b[x]
end

@parameters obc_min_complementarity begin
    b_bar = 1.0
end
"""


SIMULATE_TOKEN_SOURCE = """
@model simulate_token begin
    y[0] = rho * y[-1] + eps_y[x] + eps_auxᵒᵇᶜ[x]
end

@parameters simulate_token begin
    rho = 0.4
end
"""


AUX_SELECTION_SOURCE = """
@model aux_runtime begin
    x[0] = rho * x[-2] + eps[x+1]
end

@parameters aux_runtime begin
    rho = 0.5
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


_GALI_OBC_PATH = _UPSTREAM_MODEL_DIR / "Gali_2015_chapter_3_obc.jl"
_GALI_OBC_STEADY_STATE_GUESS = {
    "R": 1.01,
    "Q": 0.99,
    "realinterest": 1.0,
    "Pi": 1.0,
    "Pi_star": 1.0,
    "C": 1.0,
    "Y": 1.0,
    "N": 0.3,
    "S": 1.0,
    "A": 1.0,
    "Z": 1.0,
    "nu": 0.0,
    "MC": 1.0,
    "W_real": 1.0,
    "x_aux_1": 1.0,
    "x_aux_2": 1.0,
    "log_y": 0.0,
    "log_W_real": 0.0,
    "log_N": -1.2,
    "pi_ann": 0.0,
    "i_ann": 0.0,
    "r_real_ann": 0.0,
    "M_real": 1.0,
}


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


def test_get_irf_supports_simulate_token_with_deterministic_seed() -> None:
    model = parse_macro_model(SIMULATE_TOKEN_SOURCE)

    first = get_irf(
        model,
        periods=5,
        variables=("y",),
        shocks="simulate",
        shock_size=0.5,
        random_seed=7,
    )
    second = get_irf(
        model,
        periods=5,
        variables=("y",),
        shocks=":simulate",
        shock_size=0.5,
        random_seed=7,
    )

    assert first.shock_names == ("simulate",)
    np.testing.assert_allclose(
        np.asarray(first.responses, dtype=np.float64),
        np.asarray(second.responses, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(first.shocks, dtype=np.float64),
        np.asarray(second.shocks, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(first.shocks, dtype=np.float64)[model.timings.exo.index("eps_auxᵒᵇᶜ"), :, 0],
        np.zeros(5, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    shocks = np.asarray(first.shocks, dtype=np.float64)[model.timings.exo.index("eps_y"), :, 0]
    expected = np.zeros(5, dtype=np.float64)
    state = 0.0
    for t, shock in enumerate(shocks):
        state = 0.4 * state + shock
        expected[t] = state
    np.testing.assert_allclose(
        np.asarray(first.responses, dtype=np.float64)[0, :, 0],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_runtime_helpers_support_colon_prefixed_names_and_selector_tokens() -> None:
    model = parse_macro_model(SIMULATE_TOKEN_SOURCE)

    direct = get_irf(
        model,
        periods=4,
        variables="y",
        shocks="eps_y",
    )
    colon = get_irf(
        model,
        periods=4,
        variables=":y",
        shocks=":eps_y",
    )
    excluding_obc = get_irf(
        model,
        periods=4,
        variables=":all_excluding_obc",
        shocks=":all_excluding_obc",
    )
    all_shocks = get_irf(
        model,
        periods=4,
        variables=":all",
        shocks=":all",
    )

    np.testing.assert_allclose(
        np.asarray(direct.responses, dtype=np.float64),
        np.asarray(colon.responses, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert excluding_obc.variables == tuple(model.timings.var)
    assert excluding_obc.shock_names == ("eps_y",)
    assert all_shocks.shock_names == ("eps_auxᵒᵇᶜ", "eps_y")


def test_runtime_variable_selectors_can_exclude_auxiliary_variables() -> None:
    model = parse_macro_model(AUX_SELECTION_SOURCE)

    all_result = get_irf(
        model,
        periods=3,
        variables=":all",
        shocks=":eps",
    )
    filtered = get_irf(
        model,
        periods=3,
        variables=":all_excluding_auxiliary_and_obc",
        shocks=":eps",
    )
    excluding_obc = get_irf(
        model,
        periods=3,
        variables=":all_excluding_obc",
        shocks=":eps",
    )

    auxiliary_name = model.timings.aux[0]
    assert auxiliary_name in all_result.variables
    assert "eps" in all_result.variables
    assert auxiliary_name in excluding_obc.variables
    assert "eps" in excluding_obc.variables
    assert filtered.variables == ("x",)


def test_runtime_helpers_accept_grouped_variable_and_shock_name_inputs() -> None:
    model = parse_macro_model(SIMULATE_TOKEN_SOURCE)

    grouped_irf = get_irf(
        model,
        periods=4,
        variables=[["y"], ("y",)],
        shocks=[["eps_auxᵒᵇᶜ"], (":eps_y",)],
    )
    explicit_irf = get_irf(
        model,
        periods=4,
        variables=("y",),
        shocks=("eps_auxᵒᵇᶜ", "eps_y"),
    )
    grouped_sim = simulate_model(
        model,
        periods=4,
        variables=[["y"], (":y",)],
        shocks={"eps_y": [1.0, 0.0, 0.0, 0.0]},
    )
    explicit_sim = simulate_model(
        model,
        periods=4,
        variables=("y",),
        shocks={"eps_y": [1.0, 0.0, 0.0, 0.0]},
    )

    assert grouped_irf.variables == ("y",)
    assert grouped_irf.shock_names == ("eps_auxᵒᵇᶜ", "eps_y")
    np.testing.assert_allclose(
        np.asarray(grouped_irf.responses, dtype=np.float64),
        np.asarray(explicit_irf.responses, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grouped_sim.data, dtype=np.float64),
        np.asarray(explicit_sim.data, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_simulate_model_supports_simulate_token_and_matches_irf_path() -> None:
    model = parse_macro_model(SIMULATE_TOKEN_SOURCE)

    simulation = simulate_model(
        model,
        periods=6,
        variables=("y",),
        shocks="simulate",
        shock_size=0.25,
        random_seed=11,
    )
    irf = get_irf(
        model,
        periods=6,
        variables=("y",),
        shocks="simulate",
        shock_size=0.25,
        random_seed=11,
        levels=True,
    )

    assert simulation.algorithm_used == "first_order"
    np.testing.assert_allclose(
        np.asarray(simulation.data, dtype=np.float64),
        np.asarray(irf.responses, dtype=np.float64)[:, :, 0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(simulation.shocks, dtype=np.float64),
        np.asarray(irf.shocks, dtype=np.float64)[:, :, 0],
        rtol=0.0,
        atol=1e-12,
    )

    colon_simulation = simulate_model(
        model,
        periods=6,
        variables=":y",
        shocks=":simulate",
        shock_size=0.25,
        random_seed=11,
    )
    np.testing.assert_allclose(
        np.asarray(simulation.data, dtype=np.float64),
        np.asarray(colon_simulation.data, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_get_irf_enforces_supported_obc_models_with_dedicated_first_order_path() -> None:
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
    sep = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        algorithm="sep",
        ignore_obc=False,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert ignored.algorithm_used == "first_order"
    assert enforced.algorithm_used == "first_order"
    assert sep.algorithm_used == "stochastic_extended_path"
    assert np.min(np.asarray(ignored.responses, dtype=np.float64)) < 1.0 - 1e-3
    assert np.all(np.asarray(enforced.responses, dtype=np.float64) >= 1.0 - 1e-8)
    np.testing.assert_allclose(
        np.asarray(enforced.responses, dtype=np.float64),
        np.asarray(sep.responses, dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )


def test_get_irf_obc_requests_with_terminal_state_still_fall_back_to_sep() -> None:
    model = parse_macro_model(OBC_MAX_SOURCE)

    result = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
        terminal_state=[1.2, 1.2],
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert result.algorithm_used == "stochastic_extended_path"
    assert np.all(np.asarray(result.responses, dtype=np.float64) >= 1.0 - 1e-8)


def test_get_irf_ignore_obc_is_overridden_when_obc_shocks_are_selected() -> None:
    model = parse_macro_model(OBC_AUX_SHOCK_SOURCE)

    ignored = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_zlbᵒᵇᶜ",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        ignore_obc=True,
    )
    enforced = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_zlbᵒᵇᶜ",
        shock_size=2.0,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
    )

    assert ignored.algorithm_used == "first_order"
    np.testing.assert_allclose(
        np.asarray(ignored.responses, dtype=np.float64),
        np.asarray(enforced.responses, dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.all(np.asarray(ignored.responses, dtype=np.float64) >= 1.0 - 1e-8)


def test_get_irf_supported_obc_path_recovers_implied_obc_shocks() -> None:
    model = parse_macro_model(OBC_AUX_SHOCK_SOURCE)

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
    )

    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    assert enforced.algorithm_used == "first_order"
    assert np.all(np.asarray(enforced.responses, dtype=np.float64) >= 1.0 - 1e-8)
    assert np.allclose(
        np.asarray(ignored.shocks, dtype=np.float64)[obc_index, :, 0],
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(np.asarray(enforced.shocks, dtype=np.float64)[obc_index, :, 0])) > 1e-6
    np.testing.assert_allclose(
        np.asarray(enforced.shocks, dtype=np.float64)[obc_index, 0, 0],
        1.8,
        rtol=0.0,
        atol=1e-10,
    )


def test_get_irf_supports_log_obc_constraints_with_first_order_path() -> None:
    model = parse_macro_model(OBC_LOG_SOURCE)

    enforced = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=0.3,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
    )
    sep = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=0.3,
        negative_shock=True,
        levels=True,
        algorithm="sep",
        ignore_obc=False,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert enforced.algorithm_used == "first_order"
    assert np.all(np.asarray(enforced.responses, dtype=np.float64) >= 1.0 - 1e-8)
    np.testing.assert_allclose(
        np.asarray(enforced.responses, dtype=np.float64),
        np.asarray(sep.responses, dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )


def test_get_irf_recovers_implied_obc_shocks_for_log_constraints() -> None:
    model = parse_macro_model(OBC_LOG_AUX_SHOCK_SOURCE)

    enforced = get_irf(
        model,
        periods=3,
        variables=("r",),
        shocks="eps_r",
        shock_size=0.3,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
    )

    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    assert enforced.algorithm_used == "first_order"
    assert np.all(np.asarray(enforced.responses, dtype=np.float64) >= 1.0 - 1e-8)
    np.testing.assert_allclose(
        np.asarray(enforced.shocks, dtype=np.float64)[obc_index, 0, 0],
        2.0 / 15.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_simulate_model_uses_horizon_obc_shock_optimization(monkeypatch) -> None:
    model = parse_macro_model(OBC_HORIZON_AUX_SHOCK_SOURCE)
    calls: list[np.ndarray] = []
    expected_obc = np.asarray([0.1, 0.04, 0.0], dtype=np.float64)

    def fake_minimize(fun, x0, jac=None, method=None, constraints=(), options=None):
        assert method == "SLSQP"
        calls.append(np.asarray(x0, dtype=np.float64).copy())
        candidate = expected_obc.copy()
        assert jac is not None
        np.testing.assert_allclose(
            np.asarray(jac(candidate), dtype=np.float64),
            2.0 * candidate,
            rtol=0.0,
            atol=0.0,
        )
        assert constraints
        constraint_values = np.asarray(constraints[0]["fun"](candidate), dtype=np.float64)
        assert np.min(constraint_values) >= -1e-10
        return SimpleNamespace(x=candidate, success=True)

    monkeypatch.setattr(model_module.scipy_optimize, "minimize", fake_minimize)

    result = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_r": [-0.3, 0.0, 0.0]},
        levels=True,
        ignore_obc=False,
    )

    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    assert result.algorithm_used == "first_order"
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.zeros(3, dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(result.shocks, dtype=np.float64)[obc_index],
        expected_obc,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result.data, dtype=np.float64),
        np.asarray([[1.0, 1.0, 1.008]], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def test_simulate_model_supports_min_complementarity_obc_with_first_order_path() -> None:
    model = parse_macro_model(OBC_MIN_COMPLEMENTARITY_SOURCE)
    steady_state = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    fake_solution = np.asarray(
        [
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=np.float64,
    )
    fake_first_order_result = SimpleNamespace(
        parameter_values=np.asarray(model.parameter_values, dtype=np.float64),
        solution=SimpleNamespace(solution_matrix=fake_solution),
    )

    def fake_prepare_first_order_solution_for_likelihood(self, **kwargs):
        del kwargs
        return fake_first_order_result, steady_state

    original_prepare = type(model)._prepare_first_order_solution_for_likelihood
    setattr(
        type(model),
        "_prepare_first_order_solution_for_likelihood",
        fake_prepare_first_order_solution_for_likelihood,
    )
    try:
        ignored = simulate_model(
            model,
            periods=1,
            variables=("bnot", "b", "lm"),
            shocks={"eps_b": [0.5]},
            levels=True,
            ignore_obc=True,
            qme_algorithm="schur",
        )
        enforced = simulate_model(
            model,
            periods=1,
            variables=("bnot", "b", "lm"),
            shocks={"eps_b": [0.5]},
            levels=True,
            ignore_obc=False,
            qme_algorithm="schur",
        )
    finally:
        setattr(
            type(model),
            "_prepare_first_order_solution_for_likelihood",
            original_prepare,
        )

    assert ignored.algorithm_used == "first_order"
    assert enforced.algorithm_used == "first_order"
    np.testing.assert_allclose(
        np.asarray(ignored.data, dtype=np.float64),
        np.asarray([[1.0], [1.5], [0.0]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(enforced.data, dtype=np.float64),
        np.asarray([[1.0], [1.0], [0.0]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_simulate_model_obc_horizon_optimization_falls_back_to_local_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(OBC_HORIZON_AUX_SHOCK_SOURCE)
    calls = 0

    def failing_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("forced failure")

    monkeypatch.setattr(model_module.scipy_optimize, "minimize", failing_minimize)

    result = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_r": [-0.3, 0.0, 0.0]},
        levels=True,
        ignore_obc=False,
    )

    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    assert result.algorithm_used == "first_order"
    assert calls >= 1
    assert np.all(np.asarray(result.data, dtype=np.float64) >= 1.0 - 1e-10)
    np.testing.assert_allclose(
        np.asarray(result.shocks, dtype=np.float64)[obc_index],
        np.asarray([0.1, 0.04, 0.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def test_simulate_model_ignore_obc_is_overridden_when_obc_shocks_are_present() -> None:
    model = parse_macro_model(OBC_AUX_SHOCK_SOURCE)

    ignored = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_zlbᵒᵇᶜ": [-2.0, 0.0, 0.0]},
        levels=True,
        ignore_obc=True,
    )
    enforced = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_zlbᵒᵇᶜ": [-2.0, 0.0, 0.0]},
        levels=True,
        ignore_obc=False,
    )

    assert ignored.algorithm_used == "first_order"
    np.testing.assert_allclose(
        np.asarray(ignored.data, dtype=np.float64),
        np.asarray(enforced.data, dtype=np.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.all(np.asarray(ignored.data, dtype=np.float64) >= 1.0 - 1e-8)


def test_simulate_model_supported_obc_path_recovers_implied_obc_shocks() -> None:
    model = parse_macro_model(OBC_AUX_SHOCK_SOURCE)

    ignored = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_r": [-2.0, 0.0, 0.0]},
        levels=True,
        ignore_obc=True,
    )
    enforced = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_r": [-2.0, 0.0, 0.0]},
        levels=True,
        ignore_obc=False,
    )

    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    assert enforced.algorithm_used == "first_order"
    assert np.all(np.asarray(enforced.data, dtype=np.float64) >= 1.0 - 1e-8)
    assert np.allclose(
        np.asarray(ignored.shocks, dtype=np.float64)[obc_index],
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(np.asarray(enforced.shocks, dtype=np.float64)[obc_index])) > 1e-6
    np.testing.assert_allclose(
        np.asarray(enforced.shocks, dtype=np.float64)[obc_index, 0],
        1.8,
        rtol=0.0,
        atol=1e-10,
    )


def test_sep_runtime_reports_reinjected_obc_shocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(OBC_AUX_SHOCK_SOURCE)
    obc_index = model.timings.exo.index("eps_zlbᵒᵇᶜ")
    calls: list[np.ndarray] = []

    def fake_sep_core(
        self,
        *,
        full_steady_state: np.ndarray,
        parameter_values: np.ndarray,
        initial_state: np.ndarray,
        terminal_state: np.ndarray,
        config: SEPConfig,
        deterministic_shocks: np.ndarray | None,
        initial_guess: object = None,
    ) -> model_module.ParsedModelSEPResult:
        shock_matrix = (
            np.zeros((config.periods, self.timings.nExo), dtype=np.float64)
            if deterministic_shocks is None
            else np.asarray(deterministic_shocks, dtype=np.float64)
        )
        calls.append(shock_matrix.copy())
        if len(calls) == 1:
            mean_path = np.asarray(
                [
                    [1.2, 0.4, 0.72, 0.976],
                    [1.2, 0.4, 0.72, 0.976],
                ],
                dtype=np.float64,
            )
        else:
            assert float(np.max(calls[-1][:, obc_index])) > 0.0
            implied_r = np.asarray(
                [
                    0.4 + shock_matrix[0, obc_index],
                    0.72 + shock_matrix[1, obc_index],
                    0.976 + shock_matrix[2, obc_index],
                ],
                dtype=np.float64,
            )
            mean_path = np.asarray(
                [
                    [1.2, *implied_r],
                    [1.2, 0.4, 0.72, 0.976],
                ],
                dtype=np.float64,
            )
        return model_module.ParsedModelSEPResult(
            steady_state=np.asarray(full_steady_state, dtype=np.float64),
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            solution=model_module.SEPSolution(
                stacked_states=jnp.asarray(mean_path[:, 1:].T.reshape(-1), dtype=jnp.float64),
                mean_path=jnp.asarray(mean_path, dtype=jnp.float64),
                residual_norm=1e-8,
                converged=True,
                accepted=True,
                iterations=2,
                group_counts=(1, 1, 1, 1),
                jacobian_method="subgradient",
            ),
        )

    monkeypatch.setattr(
        model_module.MacroModel,
        "_solve_stochastic_extended_path_core",
        fake_sep_core,
    )

    result = simulate_model(
        model,
        periods=3,
        variables=("r",),
        shocks={"eps_r": [-2.0, 0.0, 0.0]},
        algorithm="sep",
        levels=True,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert len(calls) >= 2
    assert result.algorithm_used == "stochastic_extended_path"
    assert np.max(np.asarray(result.shocks, dtype=np.float64)[obc_index]) > 0.0
    assert np.all(np.asarray(result.data, dtype=np.float64).reshape(-1) >= 1.0)


def test_upstream_models_support_random_simulation_token() -> None:
    model_path = _ROOT / "SurrogateNN_Estimation.jl" / "test" / "models" / "RBC_CME.jl"
    model = parse_macro_model(model_path.read_text())

    sim = simulate_model(
        model,
        periods=4,
        variables=(model.steady_state_names[0],),
        shocks="simulate",
        random_seed=3,
        steady_state_initial_guess={"k": 10.0, "c": 0.8, "y": 1.0, "l": 0.3, "z": 1.0},
    )

    assert sim.data.shape == (1, 4)
    assert sim.shocks.shape[1] == 4
    assert np.all(np.isfinite(np.asarray(sim.data, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(sim.shocks, dtype=np.float64)))


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
    random_sim = simulate_model(
        model,
        periods=3,
        variables=variable,
        shocks="simulate",
        random_seed=0,
        steady_state_initial_guess=steady_state_initial_guess,
    )

    assert irf.responses.shape[0] == 1
    assert irf.responses.shape[1] == 3
    assert irf.responses.shape[2] >= 1
    assert sim.data.shape == (1, 3)
    assert random_sim.data.shape == (1, 3)
    assert np.all(np.isfinite(np.asarray(irf.responses, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(sim.data, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(random_sim.data, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(random_sim.shocks, dtype=np.float64)))


def test_upstream_gali_obc_uses_dedicated_first_order_runtime_path() -> None:
    model = parse_macro_model(_GALI_OBC_PATH.read_text())

    result = get_irf(
        model,
        periods=3,
        variables=("R",),
        shocks="eps_nu",
        shock_size=0.01,
        negative_shock=True,
        levels=True,
        ignore_obc=False,
        steady_state_initial_guess=_GALI_OBC_STEADY_STATE_GUESS,
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8),
    )

    assert result.algorithm_used == "first_order"
    assert np.all(np.isfinite(np.asarray(result.responses, dtype=np.float64)))


def test_homotopy_chained_trajectory_matches_manual_period_by_period_homotopy() -> None:
    model = parse_macro_model(LINEAR_IRF_SOURCE)
    shocks = np.asarray([[0.2, -0.1, 0.05]], dtype=np.float64)
    config = SEPConfig(periods=3, branching_order=0, tol=1e-10)

    chained = homotopy_chained_trajectory(
        model,
        periods=3,
        shocks=shocks,
        config=config,
    )
    manual = np.zeros((1, 4), dtype=np.float64)
    for period_idx in range(3):
        period_shocks = np.zeros((1, config.periods), dtype=np.float64)
        period_shocks[:, 0] = shocks[:, period_idx]
        step = model_module.homotopy_sep(
            model,
            n_steps=10,
            adaptive=True,
            max_retries=3,
            initial_state=manual[:, period_idx],
            config=config,
            deterministic_shocks=period_shocks,
        )
        assert step.success
        manual[:, period_idx + 1] = np.asarray(
            step.result.solution.mean_path[:, 1],
            dtype=np.float64,
        )

    assert chained.success
    assert chained.periods_completed == 3
    np.testing.assert_allclose(
        np.asarray(chained.trajectory, dtype=np.float64),
        manual,
        rtol=1e-10,
        atol=1e-10,
    )


def test_homotopy_chained_trajectory_reports_failure_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(LINEAR_IRF_SOURCE)
    calls: list[int] = []

    def fake_homotopy_sep(*args: object, **kwargs: object) -> model_module.HomotopySEPResult:
        period_index = len(calls)
        calls.append(period_index)
        success = period_index == 0
        mean_path = jnp.asarray([[0.0, 0.1 + period_index]], dtype=jnp.float64)
        return model_module.HomotopySEPResult(
            success=success,
            result=model_module.ParsedModelSEPResult(
                steady_state=jnp.asarray([0.0], dtype=jnp.float64),
                parameter_values=jnp.asarray([0.5], dtype=jnp.float64),
                solution=model_module.SEPSolution(
                    stacked_states=jnp.zeros((1,), dtype=jnp.float64),
                    mean_path=mean_path,
                    residual_norm=1e-6 if success else 1.0,
                    converged=success,
                    accepted=success,
                    iterations=1,
                    group_counts=(1, 1),
                    jacobian_method="autodiff",
                ),
            ),
            sigma_path=(0.0, 1.0 if success else 0.5),
        )

    monkeypatch.setattr(model_module, "homotopy_sep", fake_homotopy_sep)

    result = homotopy_chained_trajectory(
        model,
        periods=3,
        shocks=np.zeros((1, 3), dtype=np.float64),
        config=SEPConfig(periods=2, branching_order=0, tol=1e-10),
    )

    assert not result.success
    assert result.periods_completed == 1
    np.testing.assert_allclose(
        np.asarray(result.trajectory[:, :2], dtype=np.float64),
        np.asarray([[0.0, 0.1]], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert result.sigma_paths == ((0.0, 1.0), (0.0, 0.5))


def test_sep_runtime_path_accepts_solution_within_configured_accept_tol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = parse_macro_model(LINEAR_IRF_SOURCE)
    expected_path = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)

    def fake_sep_solver(self, **kwargs):
        return (
            model_module.ParsedModelSEPResult(
                steady_state=np.asarray([0.0], dtype=np.float64),
                parameter_values=np.asarray([0.5], dtype=np.float64),
                solution=model_module.SEPSolution(
                    stacked_states=jnp.zeros((3,), dtype=jnp.float64),
                    mean_path=jnp.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=jnp.float64),
                    residual_norm=1e-4,
                    converged=False,
                    accepted=True,
                    iterations=2,
                    group_counts=(1, 1, 1, 1),
                    jacobian_method="autodiff",
                ),
            ),
            np.zeros((3, model.timings.nExo), dtype=np.float64),
        )

    monkeypatch.setattr(
        model_module.MacroModel,
        "_solve_stochastic_extended_path_with_obc_enforcement",
        fake_sep_solver,
    )

    state_path = model._simulate_sep_path(
        np.zeros((model.timings.nExo, 3), dtype=np.float64),
        parameter_values=np.asarray([0.5], dtype=np.float64),
        steady_state=np.asarray([0.0], dtype=np.float64),
        config=SEPConfig(periods=3, branching_order=0, tol=1e-8, accept_tol=1e-3),
    )

    np.testing.assert_allclose(state_path, expected_path, rtol=0.0, atol=0.0)
