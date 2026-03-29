from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import surrogatenn_dsge.model as model_module

from surrogatenn_dsge import (
    SEPConfig,
    parse_macro_model,
    solve_steady_state,
    solve_steady_state_jax,
)


_OPTION_SOURCE = """
@model option_model max_obc_horizon = 5 begin
    r[0] = max(r_bar, rho * r[-1] + eps[x])
end

@parameters option_model simplify = false verbose = true guess = Dict(:r => 1.1) begin
    r_bar = 1.0
    rho = 0.8
end
"""

_SYMBOLIC_OPTION_SOURCE = """
@model symbolic_option_model begin
    k[0] = beta
    c[0] = k[0] + alpha
    y[0] = c[0] + 1
end

@parameters symbolic_option_model silent = true symbolic = true perturbation_order = 1 begin
    alpha = 0.25
    beta = 2.0
end
"""

_PRECOMPILE_OPTION_SOURCE = """
@model precompile_option_model precompile = true begin
    k[0] = beta
    c[0] = alpha * k[0]
    y[0] = c[0] + 1
end

@parameters precompile_option_model precompile = true symbolic = true perturbation_order = 2 begin
    alpha = 0.25
    beta = 2.0
end
"""


def test_parser_records_model_and_parameter_options() -> None:
    model = parse_macro_model(_OPTION_SOURCE)

    assert model.max_obc_horizon == 5
    assert model.model_options["max_obc_horizon"] == 5
    assert model.parameter_options["simplify"] is False
    assert model.parameter_options["verbose"] is True
    assert model.default_initial_guess["r"] == 1.1


def test_parser_records_remaining_parameter_directives() -> None:
    model = parse_macro_model(_SYMBOLIC_OPTION_SOURCE)

    assert model.parameter_options["silent"] is True
    assert model.parameter_options["symbolic"] is True
    assert model.parameter_options["perturbation_order"] == 1


def test_parser_precompile_option_eagerly_builds_cached_symbolic_objects() -> None:
    model = parse_macro_model(_PRECOMPILE_OPTION_SOURCE)

    assert model.model_options["precompile"] is True
    assert model.parameter_options["precompile"] is True
    assert "_steady_state_fn" in model.__dict__
    assert "_steady_state_jacobian_fn" in model.__dict__
    assert "_steady_state_residual_jax_fn" in model.__dict__
    assert "_parameter_equation_fn" in model.__dict__
    assert "_parameter_equation_residual_jax_fn" in model.__dict__
    assert "_dynamic_residual_fn" in model.__dict__
    assert "_dynamic_jacobian_fn" in model.__dict__
    assert "_dynamic_hessian_fn" in model.__dict__
    assert "_dynamic_third_order_fn" not in model.__dict__
    assert "_symbolic_steady_state_seed_fn" in model.__dict__
    assert "_symbolic_steady_state_seed_jax_fn" in model.__dict__


def test_symbolic_parameter_option_seeds_exact_numpy_steady_state() -> None:
    model = parse_macro_model(_SYMBOLIC_OPTION_SOURCE)

    result = solve_steady_state(model)
    values = dict(
        zip(
            model.steady_state_names,
            np.asarray(result.base_steady_state, dtype=np.float64),
        )
    )

    assert values["k"] == 2.0
    assert values["c"] == 2.25
    assert values["y"] == 3.25
    assert bool(result.converged)
    assert int(result.iterations) == 0


def test_symbolic_parameter_option_seeds_exact_jax_steady_state() -> None:
    model = parse_macro_model(_SYMBOLIC_OPTION_SOURCE)

    result = solve_steady_state_jax(model)
    values = dict(
        zip(
            model.steady_state_names,
            np.asarray(result.base_steady_state, dtype=np.float64),
        )
    )

    assert values["k"] == 2.0
    assert values["c"] == 2.25
    assert values["y"] == 3.25
    assert bool(result.converged)
    assert int(np.asarray(result.iterations)) == 0


def test_first_order_obc_sep_fallback_uses_model_max_obc_horizon(monkeypatch) -> None:
    model = parse_macro_model(_OPTION_SOURCE)
    captured: list[tuple[int, int, int]] = []

    def fake_simulate_sep_path(
        self,
        shocks: np.ndarray,
        *,
        parameter_values: np.ndarray,
        steady_state: np.ndarray,
        initial_state,
        terminal_state,
        config,
    ) -> model_module._SEPPathSimulationResult:
        del parameter_values, initial_state, terminal_state
        captured.append((shocks.shape[1], int(config.periods), int(config.branching_order)))
        state_path = np.repeat(
            np.asarray(steady_state, dtype=np.float64)[:, None],
            shocks.shape[1],
            axis=1,
        )
        return model_module._SEPPathSimulationResult(
            state_path=state_path,
            shocks=np.asarray(shocks, dtype=np.float64),
            sep_result=model_module.ParsedModelSEPResult(
                steady_state=np.asarray(steady_state, dtype=np.float64),
                parameter_values=np.asarray([0.5], dtype=np.float64),
                solution=model_module.SEPSolution(
                    stacked_states=jnp.asarray(state_path.T.reshape(-1), dtype=jnp.float64),
                    mean_path=jnp.asarray(
                        np.concatenate(
                            [np.asarray(steady_state, dtype=np.float64)[:, None], state_path],
                            axis=1,
                        ),
                        dtype=jnp.float64,
                    ),
                    residual_norm=0.0,
                    converged=True,
                    accepted=True,
                    iterations=1,
                    group_counts=(1,) * (shocks.shape[1] + 1),
                    jacobian_method="autodiff",
                ),
            ),
        )

    monkeypatch.setattr(type(model), "_simulate_sep_path_with_shocks", fake_simulate_sep_path)
    result = model.simulate(
        periods=2,
        shocks={"eps": [0.1, 0.0]},
        algorithm="first_order",
        terminal_state=[1.0],
        config=SEPConfig(periods=1),
    )

    assert result.algorithm_used == "stochastic_extended_path"
    assert captured == [(2, 5, 0)]
