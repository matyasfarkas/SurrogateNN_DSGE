from __future__ import annotations

import numpy as np

from surrogatenn_dsge import SEPConfig, parse_macro_model


_OPTION_SOURCE = """
@model option_model max_obc_horizon = 5 begin
    r[0] = max(r_bar, rho * r[-1] + eps[x])
end

@parameters option_model simplify = false verbose = true guess = Dict(:r => 1.1) begin
    r_bar = 1.0
    rho = 0.8
end
"""


def test_parser_records_model_and_parameter_options() -> None:
    model = parse_macro_model(_OPTION_SOURCE)

    assert model.max_obc_horizon == 5
    assert model.model_options["max_obc_horizon"] == 5
    assert model.parameter_options["simplify"] is False
    assert model.parameter_options["verbose"] is True
    assert model.default_initial_guess["r"] == 1.1


def test_first_order_obc_runtime_uses_model_max_obc_horizon(monkeypatch) -> None:
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
    ) -> np.ndarray:
        del parameter_values, initial_state, terminal_state
        captured.append((shocks.shape[1], int(config.periods), int(config.branching_order)))
        return np.repeat(np.asarray(steady_state, dtype=np.float64)[:, None], shocks.shape[1], axis=1)

    monkeypatch.setattr(type(model), "_simulate_sep_path", fake_simulate_sep_path)
    result = model.simulate(
        periods=2,
        shocks={"eps": [0.1, 0.0]},
        algorithm="first_order",
        config=SEPConfig(periods=1),
    )

    assert result.algorithm_used == "stochastic_extended_path"
    assert captured == [(2, 5, 0)]
