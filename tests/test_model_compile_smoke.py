from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import jax
import numpy as np
import pytest

from surrogatenn_dsge import (
    estimate_observed_shocks_matrix,
    estimate_observed_variables_matrix,
    kalman_loglikelihood_from_model,
    kalman_loglikelihood_from_model_jax,
    kalman_loglikelihood_per_period_from_model,
    parse_macro_model,
    solve_first_order_model,
)


_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_TEST_MODEL_DIR = _ROOT / "SurrogateNN_Estimation.jl" / "test" / "models"
_UPSTREAM_MODEL_DIR = _ROOT / "SurrogateNN_Estimation.jl" / "models"
_BENCHMARK_PAYLOAD_PATH = _ROOT / "SurrogateNN_DSGE" / "benchmarks" / "results" / "test_payloads.json"

_RBC_STEADY_STATE_GUESS = {
    "A": 1.0,
    "Pi": 1.0025,
    "R": 1.0035,
    "c": 1.2,
    "k": 9.4,
    "y": 1.42,
    "z_delta": 1.0,
}

_COMPILE_SMOKE_MODELS = (
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR / "RBC_CME.jl",
        None,
        id="rbc_cme",
    ),
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR / "RBC_CME_calibration_equations.jl",
        None,
        id="rbc_cme_calibration",
    ),
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR
        / "RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl",
        {
            **_RBC_STEADY_STATE_GUESS,
            "ZZ_avg": 1.0,
            "ZZ_avg_fut": 1.0,
            "log_ZZ_avg": 0.0,
            "c_normlogpdf": -1.2,
            "c_norminvcdf": -0.8,
            "c_erfcinv": 1.0,
            "c_erfinv": 0.3,
        },
        id="rbc_cme_specfuns",
    ),
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR
        / "RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl",
        {
            **_RBC_STEADY_STATE_GUESS,
            "ZZ_avg": 1.0,
            "ZZ_avg_fut": 1.0,
            "log_ZZ_avg": 0.0,
            "c_normlogpdf": -1.2,
            "c_norminvcdf": -0.8,
        },
        id="rbc_cme_lead_lags",
    ),
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
    pytest.param(
        _UPSTREAM_TEST_MODEL_DIR / "Backus_Kehoe_Kydland_1992.jl",
        {
            "A{F}": 2.0 / 3.0,
            "A{H}": 2.0 / 3.0,
            "C{F}": 0.8,
            "C{H}": 0.8,
            "K{F}": 11.0,
            "K{H}": 11.0,
            "LAMBDA{F}": 1.0,
            "LAMBDA{H}": 1.0,
            "LGM": 0.5,
            "L{F}": 2.0 / 3.0,
            "L{H}": 2.0 / 3.0,
            "NX{F}": 0.0,
            "NX{H}": 0.0,
            "N{F}": 1.0 / 3.0,
            "N{H}": 1.0 / 3.0,
            "S{F}": 0.275,
            "S{H}": 0.275,
            "U{F}": 1.0,
            "U{H}": 1.0,
            "X{F}": 0.275,
            "X{H}": 0.275,
            "Y{F}": 1.1,
            "Y{H}": 1.1,
            "Z{F}": 1.0,
            "Z{H}": 1.0,
            "dLGM": 1.0,
            "dLGM_ann": 1.0,
        },
        id="backus_kehoe_kydland_1992",
    ),
)

_SCHUR_COMPILE_SMOKE_MODELS = (
    _COMPILE_SMOKE_MODELS[0],
    _COMPILE_SMOKE_MODELS[4],
    _COMPILE_SMOKE_MODELS[5],
    _COMPILE_SMOKE_MODELS[7],
)


def _hlt_case() -> dict[str, object]:
    payload = json.loads(_BENCHMARK_PAYLOAD_PATH.read_text())
    return next(entry for entry in payload["cases"] if entry["name"] == "medium_sw07_hlt")


@lru_cache(maxsize=1)
def _hlt_julia_kalman_reference() -> dict[str, object]:
    julia = shutil.which("julia")
    if julia is None:
        pytest.skip("Julia is not available for MacroModelling parity checks.")

    project = _ROOT / "SurrogateNN_Estimation.jl"
    if not project.exists():
        pytest.skip("Upstream Julia reference repo is not available.")

    code = f"""
using AxisKeys
using JSON
using MacroModelling

quiet(f::Function) = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        f()
    end
end

payload = JSON.parsefile(raw\"{_BENCHMARK_PAYLOAD_PATH}\")
case = only(filter(c -> c[\"name\"] == \"medium_sw07_hlt\", payload[\"cases\"]))
quiet() do
    Base.include(Main, case[\"model_path\"])
end
model = Base.invokelatest(() -> getfield(Main, Symbol(case[\"model_symbol\"])))
params = Float64.(model.parameter_values)
observables = Symbol.(case[\"observables\"])
rows = [Float64.(row) for row in case[\"observations\"]]
matrix = reduce(vcat, [reshape(row, 1, :) for row in rows])
data = KeyedArray(matrix; Variable = observables, Time = collect(1:size(matrix, 2)))
per_period = quiet() do
    MacroModelling.get_loglikelihood_per_period(
        model,
        data,
        params;
        algorithm = :first_order,
        filter = :kalman,
        quadratic_matrix_equation_algorithm = :schur,
        initial_covariance = :theoretical,
        presample_periods = 0,
        verbose = false,
    )
end
sorted_observables = sort(observables)
row_lookup = Dict(name => idx for (idx, name) in enumerate(observables))
sorted_matrix = reduce(
    vcat,
    [reshape(matrix[row_lookup[name], :], 1, :) for name in sorted_observables],
)
steady_state = quiet() do
    MacroModelling.get_steady_state(
        model;
        parameters = params,
        algorithm = :first_order,
        derivatives = false,
        verbose = false,
    )
end
steady_lookup = Dict(axiskeys(steady_state, 1) .=> Array(steady_state))
steady_observables = [steady_lookup[name] for name in sorted_observables]
opts = MacroModelling.merge_calculation_options(
    quadratic_matrix_equation_algorithm = :schur,
    verbose = false,
)
smoothed_variables, _, smoothed_shocks, _, filtered_variables, _, filtered_shocks, _ = quiet() do
    MacroModelling.filter_and_smooth(
        model,
        sorted_matrix .- reshape(steady_observables, :, 1),
        collect(sorted_observables);
        opts = opts,
    )
end
println(
    JSON.json(
        Dict(
            \"per_period\" => per_period,
            \"filtered_variables\" => filtered_variables,
            \"smoothed_variables\" => smoothed_variables,
            \"filtered_shocks\" => filtered_shocks,
            \"smoothed_shocks\" => smoothed_shocks,
        ),
    ),
)
"""

    completed = subprocess.run(
        [julia, f"--project={project}", "-e", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("model_path", "steady_state_initial_guess"),
    _COMPILE_SMOKE_MODELS,
)
def test_upstream_fixture_solves_first_order_and_compiled_kalman(
    model_path: Path,
    steady_state_initial_guess: dict[str, float] | None,
) -> None:
    model = parse_macro_model(model_path.read_text())
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess=steady_state_initial_guess,
    )

    assert first_order_result.solution.converged

    observable = model.steady_state_names[0]
    steady_lookup = dict(
        zip(
            model.timings.var,
            np.asarray(first_order_result.steady_state, dtype=np.float64).tolist(),
        )
    )
    levels = np.asarray([[steady_lookup[observable]] * 4], dtype=np.float64)
    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=(observable,),
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
            measurement_error_scale=1e-8,
            on_failure_loglikelihood=-1e12,
        )
    )

    value = compiled(np.asarray(first_order_result.parameter_values, dtype=np.float64))
    assert np.isfinite(value)


@pytest.mark.parametrize(
    ("model_path", "steady_state_initial_guess"),
    _SCHUR_COMPILE_SMOKE_MODELS,
)
def test_upstream_fixture_solves_first_order_and_compiled_kalman_with_schur(
    model_path: Path,
    steady_state_initial_guess: dict[str, float] | None,
) -> None:
    model = parse_macro_model(model_path.read_text())
    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess=steady_state_initial_guess,
        qme_algorithm="schur",
    )

    assert first_order_result.solution.converged

    observable = model.steady_state_names[0]
    steady_lookup = dict(
        zip(
            model.timings.var,
            np.asarray(first_order_result.steady_state, dtype=np.float64).tolist(),
        )
    )
    levels = np.asarray([[steady_lookup[observable]] * 4], dtype=np.float64)
    compiled = jax.jit(
        lambda theta: kalman_loglikelihood_from_model_jax(
            model,
            levels,
            observables=(observable,),
            parameter_values=theta,
            steady_state=first_order_result.steady_state,
            measurement_error_scale=1e-8,
            on_failure_loglikelihood=-1e12,
            qme_algorithm="schur",
        )
    )

    value = compiled(np.asarray(first_order_result.parameter_values, dtype=np.float64))
    assert np.isfinite(value)


def test_hlt_kalman_loglikelihood_matches_julia_reference() -> None:
    case = _hlt_case()
    model = parse_macro_model(Path(case["model_path"]).read_text())

    expected = -600.6439319278583
    high_level = float(
        kalman_loglikelihood_from_model(
            model,
            case["observations"],
            observables=case["observables"],
            steady_state=case["reference_steady_state"],
            measurement_error_scale=0.0,
            jitter=0.0,
            on_failure_loglikelihood=-1e12,
            qme_algorithm="schur",
        )
    )
    compiled = float(
        kalman_loglikelihood_from_model_jax(
            model,
            case["observations"],
            observables=case["observables"],
            steady_state=case["reference_steady_state"],
            measurement_error_scale=0.0,
            jitter=0.0,
            on_failure_loglikelihood=-1e12,
            qme_algorithm="schur",
        )
    )

    np.testing.assert_allclose(high_level, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(compiled, expected, rtol=1e-10, atol=1e-10)


def test_hlt_kalman_filter_paths_match_julia_reference() -> None:
    case = _hlt_case()
    reference = _hlt_julia_kalman_reference()
    model = parse_macro_model(Path(case["model_path"]).read_text())

    observations = np.asarray(case["observations"], dtype=np.float64)
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    per_period = kalman_loglikelihood_per_period_from_model(
        model,
        observations,
        observables=case["observables"],
        steady_state=steady_state,
        measurement_error_scale=0.0,
        jitter=0.0,
        on_failure_loglikelihood=-1e12,
        qme_algorithm="schur",
    )
    filtered_variables, variable_names = estimate_observed_variables_matrix(
        model,
        observations,
        observables=case["observables"],
        steady_state=steady_state,
        qme_algorithm="schur",
        filter="kalman",
        smooth=False,
    )
    smoothed_variables, _ = estimate_observed_variables_matrix(
        model,
        observations,
        observables=case["observables"],
        steady_state=steady_state,
        qme_algorithm="schur",
        filter="kalman",
        smooth=True,
    )
    filtered_shocks = estimate_observed_shocks_matrix(
        model,
        observations,
        observables=case["observables"],
        steady_state=steady_state,
        qme_algorithm="schur",
        filter="kalman",
        smooth=False,
    )
    smoothed_shocks = estimate_observed_shocks_matrix(
        model,
        observations,
        observables=case["observables"],
        steady_state=steady_state,
        qme_algorithm="schur",
        filter="kalman",
        smooth=True,
    )

    assert variable_names == tuple(model.timings.var)
    np.testing.assert_allclose(
        per_period,
        np.asarray(reference["per_period"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        filtered_variables,
        np.asarray(reference["filtered_variables"], dtype=np.float64).T + steady_state[:, None],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        smoothed_variables,
        np.asarray(reference["smoothed_variables"], dtype=np.float64).T + steady_state[:, None],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        filtered_shocks,
        np.asarray(reference["filtered_shocks"], dtype=np.float64).T,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        smoothed_shocks,
        np.asarray(reference["smoothed_shocks"], dtype=np.float64).T,
        rtol=1e-6,
        atol=1e-6,
    )
