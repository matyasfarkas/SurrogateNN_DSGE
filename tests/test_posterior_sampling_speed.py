from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "benchmarks" / "posterior_sampling_speed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "posterior_sampling_speed_for_tests",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_summary_filters_deterministic_sites_from_ess() -> None:
    module = _load_module()
    samples = {
        "rho": np.asarray(
            [
                [0.80, 0.81, 0.79, 0.82, 0.80, 0.81],
                [0.78, 0.79, 0.80, 0.81, 0.82, 0.83],
            ],
            dtype=np.float64,
        ),
        "loglikelihood": np.asarray(
            [
                [-10.0, -10.1, -10.2, -10.1, -10.0, -10.2],
                [-10.3, -10.1, -10.2, -10.0, -10.1, -10.3],
            ],
            dtype=np.float64,
        ),
    }

    summary = module._sample_summary(samples, parameter_names=("rho",))

    assert tuple(summary["parameters"]) == ("rho",)
    assert summary["parameters"]["rho"]["shape"] == [2, 6]
    assert summary["min_ess"] is not None
    assert summary["min_ess"] > 0.0


def test_sw07_safe_presets_reference_known_parameter_names() -> None:
    module = _load_module()
    model = SimpleNamespace(parameter_names=module.SW07_SAFE_27_PARAMETERS)

    selected_15 = module._select_parameter_names(model, "sw07_safe_15")
    selected_27 = module._select_parameter_names(model, "sw07_safe_27")

    assert selected_15 == module.SW07_SAFE_15_PARAMETERS
    assert selected_27 == module.SW07_SAFE_27_PARAMETERS
    assert set(selected_15).issubset(selected_27)


def test_package_parameter_sets_are_accepted_by_sw07_selector() -> None:
    module = _load_module()
    phase1 = (
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
    )
    model = SimpleNamespace(parameter_names=phase1)

    selected = module._select_parameter_names(model, "phase1_18params_narrow")

    assert selected == phase1


def test_gali3_presets_reference_known_parameter_names() -> None:
    module = _load_module()
    model = SimpleNamespace(parameter_names=module.GALI3_ALL_STABLE_PARAMETERS)

    selected_4 = module._select_parameter_names(model, "gali3_policy_4")
    selected_7 = module._select_parameter_names(model, "gali3_policy_7")
    selected_all = module._select_parameter_names(model, "gali3_all_stable")

    assert selected_4 == module.GALI3_POLICY_4_PARAMETERS
    assert selected_7 == module.GALI3_POLICY_7_PARAMETERS
    assert selected_all == module.GALI3_ALL_STABLE_PARAMETERS
    assert set(selected_4).issubset(selected_7)
    assert set(selected_7).issubset(selected_all)


def test_prior_interval_keeps_unit_root_like_parameters_inside_unit_bounds() -> None:
    module = _load_module()

    lower, upper = module._prior_interval("crhoa", 0.9977, 0.01, 1.0e-4)

    assert 0.0 < lower < 0.9977 < upper < 1.0


def test_prior_interval_allows_zero_centered_bounded_parameters() -> None:
    module = _load_module()

    lower, upper = module._prior_interval("crhopinf", 0.0, 0.01, 1.0e-4)

    assert lower < 0.0 < upper
    assert upper - lower > 0.0


def test_prior_interval_keeps_gali3_policy_parameters_in_safe_support() -> None:
    module = _load_module()

    phi_lower, phi_upper = module._prior_interval("phi_pi", 1.5, 0.75, 1.0e-4)
    rho_lower, rho_upper = module._prior_interval("rho_i", 0.7, 0.75, 1.0e-4)

    assert 1.0 < phi_lower < 1.5 < phi_upper
    assert 0.0 < rho_lower < 0.7 < rho_upper < 1.0


def test_parse_args_preflight_only_enables_preflight() -> None:
    module = _load_module()

    args = module._parse_args(["--preflight-only"])

    assert args.preflight_only is True
    assert args.preflight is True


def test_parse_args_accepts_gali3_preset() -> None:
    module = _load_module()

    args = module._parse_args(["--preset", "gali3_nk", "--parameters", "gali3_policy_7"])

    assert args.preset == "gali3_nk"
    assert args.parameters == "gali3_policy_7"


def test_parse_args_configures_paper_matched_sw07_defaults() -> None:
    module = _load_module()

    args = module._parse_args(
        ["--preset", "sw07_paper", "--sw07-data-mat", "usmodel_data.mat"]
    )

    assert args.parameters == "sw07_paper_36"
    assert args.sw07_sample_start_index == 50
    assert args.sw07_sample_periods == 180
    assert args.presample_periods == 6
    assert args.store_posterior_draws is False


def test_parse_args_enables_raw_posterior_draw_storage() -> None:
    module = _load_module()

    args = module._parse_args(["--store-posterior-draws"])

    assert args.store_posterior_draws is True


def test_sw07_paper_spec_has_reference_target_for_every_estimand() -> None:
    module = _load_module()

    assert len(module.SW07_PAPER_PARAMETER_NAMES) == 36
    assert set(module.SW07_PAPER_PARAMETER_NAMES) == set(
        module.SW07_PAPER_REFERENCE_POSTERIOR
    )


def test_sw07_paper_model_recomputes_dynare_parameter_dependencies() -> None:
    module = _load_module()
    import jax
    import surrogatenn_dsge as sdsge

    jax.config.update("jax_enable_x64", True)
    model = sdsge.parse_macro_model(
        module.DEFAULT_SW07_PAPER_MODEL_SOURCE_PATH.read_text()
    )
    parameters = np.asarray(model.parameter_values, dtype=np.float64).copy()
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value, *_ in module.SW07_PAPER_PRIOR_SPECS:
        parameters[parameter_index[name]] = value

    numpy_result = sdsge.solve_steady_state(model, parameter_values=parameters)
    numpy_parameters = dict(
        zip(model.parameter_names, np.asarray(numpy_result.parameter_values))
    )
    truncated_result = sdsge.solve_steady_state_jax(
        model,
        parameter_values=parameters,
        initial_guess=numpy_result.steady_state,
        max_iter=2,
    )
    jax_result = sdsge.solve_steady_state_jax(
        model,
        parameter_values=parameters,
        initial_guess=numpy_result.steady_state,
        max_iter=20,
    )

    assert numpy_result.converged
    assert bool(truncated_result.converged) is False
    assert bool(jax_result.converged) is True
    np.testing.assert_allclose(
        np.asarray(jax_result.parameter_values),
        np.asarray(numpy_result.parameter_values),
        rtol=0,
        atol=5.0e-12,
    )
    np.testing.assert_allclose(
        numpy_parameters["cZcap"],
        numpy_parameters["czcap"] / (1.0 - numpy_parameters["czcap"]),
        rtol=0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        numpy_parameters["cbetabar"],
        numpy_parameters["cbeta"]
        * numpy_parameters["cgamma"] ** (-numpy_parameters["csigma"]),
        rtol=0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("family", ["normal", "beta", "gamma", "inverse_gamma"])
def test_dynare_prior_moment_conversion_recovers_declared_moments(family: str) -> None:
    module = _load_module()
    import numpyro.distributions as dist

    mean, std = (0.5, 0.1) if family == "beta" else (0.6, 0.2)
    parameters = module._distribution_parameters_from_moments(family, mean, std)
    if family == "normal":
        distribution = dist.Normal(parameters["loc"], parameters["scale"])
    elif family == "beta":
        distribution = dist.Beta(
            parameters["concentration1"], parameters["concentration0"]
        )
    elif family == "gamma":
        distribution = dist.Gamma(parameters["concentration"], parameters["rate"])
    else:
        distribution = dist.InverseGamma(
            parameters["concentration"], parameters["rate"]
        )

    np.testing.assert_allclose(float(distribution.mean), mean, rtol=1e-6)
    np.testing.assert_allclose(float(np.sqrt(distribution.variance)), std, rtol=1e-6)


def test_sw07_paper_data_loader_selects_and_maps_requested_sample(tmp_path: Path) -> None:
    module = _load_module()
    from scipy.io import savemat

    data_path = tmp_path / "usmodel_data.mat"
    raw = {
        key: np.arange(230, dtype=np.float64) + idx
        for idx, key in enumerate(module.SW07_PAPER_DATA_KEYS)
    }
    savemat(data_path, raw)

    observations, provenance = module._load_sw07_paper_observations(
        data_path,
        sample_start_index=50,
        sample_periods=180,
        expected_sha256="",
    )

    assert observations.shape == (7, 180)
    np.testing.assert_allclose(observations[:, 0], np.arange(7) + 50)
    assert provenance["declared_sample"] == "1960Q1--2004Q4"
    assert provenance["observable_mapping"]["dw"] == "dwobs"


def test_sw07_reference_comparison_is_exact_for_reference_summary() -> None:
    module = _load_module()
    summary = {
        "parameters": {
            name: {"mean": mean, "q05": q05, "q95": q95}
            for name, (mean, q05, q95) in module.SW07_PAPER_REFERENCE_POSTERIOR.items()
        }
    }

    comparison = module._sw07_reference_comparison(summary)

    assert comparison["complete"] is True
    assert comparison["max_normalized_mean_error"] == 0.0
    assert comparison["posterior_means_inside_reference_90pct_count"] == 36
    assert comparison["reference_means_inside_posterior_90pct_count"] == 36


def test_parse_args_accepts_gpu_schur_qme_algorithm() -> None:
    module = _load_module()

    args = module._parse_args(["--qme-algorithm", "schur_gpu"])

    assert args.qme_algorithm == "schur_gpu"


def test_gpu_posterior_defaults_to_determinacy_aware_qme_solver() -> None:
    module = _load_module()

    args = module._parse_args([])

    assert args.qme_algorithm == "schur_gpu"


def test_parse_args_accepts_cuda_platform_aliases() -> None:
    module = _load_module()

    gpu_args = module._parse_args(["--platform", "gpu"])
    cuda_args = module._parse_args(["--platform", "cuda"])

    assert module._jax_platform_name(gpu_args.platform) == "cuda"
    assert module._jax_platform_name(cuda_args.platform) == "cuda"


def test_parse_args_accepts_fixed_step_hmc_kernel() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--kernel",
            "hmc",
            "--hmc-num-steps",
            "12",
            "--hmc-step-size",
            "0.25",
            "--no-adapt-step-size",
            "--no-adapt-mass-matrix",
        ]
    )

    assert args.kernel == "hmc"
    assert args.hmc_num_steps == 12
    assert args.hmc_step_size == 0.25
    assert args.no_adapt_step_size is True
    assert args.no_adapt_mass_matrix is True


def test_gali3_payload_solves_and_matches_qme_likelihoods() -> None:
    module = _load_module()
    import jax
    import surrogatenn_dsge as sdsge

    args = SimpleNamespace(
        model_source=module.DEFAULT_SW07_MODEL_SOURCE_PATH,
        parameters="sw07_safe_15",
        periods=8,
        synthetic_seed=20260927,
    )

    payload = module._gali3_payload(args, sdsge, jax)
    model = payload["model"]
    observations = payload["observations"]
    observables = payload["observables"]
    steady_state = payload["steady_state"]

    assert model.name == "Gali_3eq_linear"
    assert payload["parameter_names"] == module.GALI3_POLICY_4_PARAMETERS
    assert observations.shape == (3, 8)

    likelihoods = [
        float(
            sdsge.kalman_loglikelihood_from_model(
                model,
                observations,
                observables=observables,
                steady_state=steady_state,
                qme_algorithm=algorithm,
                on_failure_loglikelihood=-1.0e12,
            )
        )
        for algorithm in ("schur", "schur_gpu", "doubling")
    ]
    assert max(likelihoods) - min(likelihoods) < 1.0e-8


def test_gali3_doubling_preflight_gradient_compiles_with_static_rows() -> None:
    module = _load_module()
    import jax
    import jax.numpy as jnp
    import surrogatenn_dsge as sdsge

    args = SimpleNamespace(
        model_source=module.DEFAULT_SW07_MODEL_SOURCE_PATH,
        parameters="gali3_policy_4",
        periods=8,
        synthetic_seed=20260927,
    )
    payload = module._gali3_payload(args, sdsge, jax)
    model = payload["model"]
    steady_state = np.asarray(payload["steady_state"], dtype=np.float64)
    parameter_values = np.asarray(model.parameter_values, dtype=np.float64)
    static_equation_rows = model._first_order_static_equation_rows_for_values(
        steady_state=steady_state,
        parameter_values=parameter_values,
    )

    preflight = module._preflight_metrics(
        jax=jax,
        jnp=jnp,
        sdsge=sdsge,
        model=model,
        observations=np.asarray(payload["observations"], dtype=np.float64),
        observables=payload["observables"],
        steady_state=steady_state,
        parameter_values=parameter_values,
        parameter_names=payload["parameter_names"],
        measurement_error_scale=float(payload["measurement_error_scale"]),
        jitter=float(payload["jitter"]),
        qme_algorithm="doubling",
        static_equation_rows=static_equation_rows,
        reps=0,
        failure_value=-1.0e12,
        parameters_are_resolved=False,
        check_parameter_bounds=True,
    )

    assert np.isfinite(preflight["loglikelihood"])
    assert np.isfinite(preflight["gradient_value"])
    assert len(preflight["gradient"]) == len(payload["parameter_names"])
