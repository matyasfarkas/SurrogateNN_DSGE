from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from surrogatenn_dsge import SEPConfig


def _load_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "generate_gali_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("generate_gali_synthetic_data", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    name = "Gali_2015_chapter_3_nonlinear"
    parameter_names = ("std_a", "std_nu", "std_z", "ϕᵖⁱ", "ϕʸ")
    parameter_values = np.asarray([0.01, 0.0025, 0.05, 1.5, 0.125], dtype=np.float64)
    timings = SimpleNamespace(
        var=("log_y", "pi_ann", "i_ann"),
        exo=("eps_a", "eps_nu", "eps_z"),
        nExo=3,
    )

    def solve_steady_state(self, **_kwargs):
        return SimpleNamespace(
            steady_state=np.asarray([0.0, 0.0, 0.0]),
            parameter_values=self.parameter_values.copy(),
            converged=True,
            iterations=2,
            residual_norm=1.0e-14,
        )

    def solve_stochastic_extended_path(self, **kwargs):
        initial = np.asarray(kwargs["initial_state"], dtype=np.float64)
        shock = np.asarray(kwargs["deterministic_shocks"], dtype=np.float64)[0]
        next_state = 0.5 * initial + shock
        mean_path = np.column_stack([initial, next_state])
        solution = SimpleNamespace(
            mean_path=mean_path,
            residual_norm=1.0e-10,
            accepted=True,
            converged=True,
            iterations=3,
        )
        return SimpleNamespace(solution=solution)


def _template_case():
    return {
        "name": "template",
        "model_symbol": "Gali_2015_chapter_3_nonlinear",
        "observables": ["log_y", "pi_ann", "i_ann"],
        "observations": [[0.0], [0.0], [0.0]],
        "obs_sigma": {"log_y": 0.01, "pi_ann": 0.02, "i_ann": 0.03},
        "shock_names": ["eps_a", "eps_nu", "eps_z"],
        "shock_sigmas": {"eps_a": 1.0, "eps_nu": 1.0, "eps_z": 1.0},
        "shock_scale_convention": "standard_normal_innovations_scaled_in_model_equations",
        "model_innovation_scale_parameters": {
            "eps_a": "std_a",
            "eps_nu": "std_nu",
            "eps_z": "std_z",
        },
        "parameter_subset": ["ϕᵖⁱ", "ϕʸ"],
        "reference_steady_state": [0.0, 0.0, 0.0],
    }


def test_generator_is_seed_reproducible_and_records_dgp() -> None:
    mod = _load_module()
    config = SEPConfig(periods=2, branching_order=1, nnodes=3, sparse_tree=True)
    first = mod.generate_case(
        _FakeModel(),
        _template_case(),
        case_name="frozen",
        periods=4,
        burn_in=2,
        shock_seed=7,
        measurement_seed=11,
        sep_config=config,
    )
    second = mod.generate_case(
        _FakeModel(),
        _template_case(),
        case_name="frozen",
        periods=4,
        burn_in=2,
        shock_seed=7,
        measurement_seed=11,
        sep_config=config,
    )

    assert first == second
    assert np.asarray(first["observations"]).shape == (3, 4)
    assert np.asarray(first["dgp"]["arrays"]["structural_innovations"]).shape == (3, 6)
    assert np.asarray(first["dgp"]["arrays"]["latent_state_path"]).shape == (3, 7)
    assert len(first["dgp"]["arrays_sha256"]) == 64
    assert first["dgp"]["burn_in"] == 2
    assert first["dgp"]["sep"]["branching_order"] == 1


def test_generator_rejects_double_scaled_innovations() -> None:
    mod = _load_module()
    invalid = _template_case()
    invalid["shock_sigmas"] = {**invalid["shock_sigmas"], "eps_z": 0.05}

    with pytest.raises(ValueError, match="shock_sigmas must equal 1.0"):
        mod.generate_case(
            _FakeModel(),
            invalid,
            case_name="invalid",
            periods=2,
            burn_in=0,
            shock_seed=1,
            measurement_seed=2,
            sep_config=SEPConfig(periods=2),
        )
