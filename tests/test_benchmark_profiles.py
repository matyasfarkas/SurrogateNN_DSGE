from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        import toml as tomllib  # type: ignore[no-redef]


_ROOT = Path(__file__).resolve().parents[1]
_LONG_PROFILE_PATH = _ROOT / "benchmarks" / "sw07_long_profile.toml"


def _load_toml(path: Path) -> dict[str, object]:
    if tomllib.__name__ == "toml":
        return tomllib.load(str(path))
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_profile_validation_module() -> ModuleType:
    module_path = _ROOT / "benchmarks" / "profile_validation.py"
    spec = importlib.util.spec_from_file_location("profile_validation_for_tests", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sw07_long_profile_is_opt_in_hlt_stress_case() -> None:
    config = _load_toml(_LONG_PROFILE_PATH)
    cases = config["case"]  # type: ignore[index]

    assert len(cases) == 1
    case = cases[0]
    assert case["name"] == "large_sw07_hlt_switching_order_3h"
    assert case["model_symbol"] == "Smets_Wouters_2007_HLT"
    assert Path(case["model_path"]).name == "Smets_Wouters_2007_HLT.jl"
    assert case["observables"] == [
        "dy",
        "dc",
        "dinve",
        "labobs",
        "pinfobs",
        "dwobs",
        "robs",
    ]
    assert case["periods"] >= 240
    assert case["switching_reps"] >= 1_000
    assert case["kalman_value_reps"] >= 500
    assert case["numpyro_log_density_reps"] > 0
    assert case["numpyro_switching_log_density_reps"] > 0
    assert case["numpyro_nuts_samples"] == 0
    assert case["sep_sparse_tree"] is True
    assert case["sep_branching_order"] == 1
    assert case["sep_nnodes"] % 2 == 1


def test_profile_validation_accepts_cli_config_override(
    monkeypatch,
) -> None:
    module = _load_profile_validation_module()
    monkeypatch.delenv(module.CONFIG_ENV, raising=False)
    monkeypatch.chdir(_ROOT)

    resolved = module._resolve_config_path(["benchmarks/sw07_long_profile.toml"])

    assert resolved == _LONG_PROFILE_PATH.resolve()


def test_profile_validation_accepts_env_config_override(
    monkeypatch,
) -> None:
    module = _load_profile_validation_module()
    monkeypatch.chdir(_ROOT)
    monkeypatch.setenv(module.CONFIG_ENV, "benchmarks/sw07_long_profile.toml")

    resolved = module._resolve_config_path([])

    assert resolved == _LONG_PROFILE_PATH.resolve()


def test_report_text_handles_single_case_profile() -> None:
    module = _load_profile_validation_module()
    config = {"global": {"thread_count": 8, "jax_platform_name": "auto"}}
    payload = {
        "cases": [
            {
                "name": "large_sw07_hlt_switching_order_3h",
                "model_symbol": "Smets_Wouters_2007_HLT",
            }
        ]
    }
    paths_result = {
        "filtered_variables": [[0.0]],
        "smoothed_variables": [[0.0]],
        "filtered_shocks": [[0.0]],
        "smoothed_shocks": [[0.0]],
    }
    gate_result = {
        "linear_observations": [[0.0]],
        "shocks": [[0.0]],
        "e_stat": [0.0],
        "f_stat": [0.0],
    }
    case_result = {
        "model_info": {"jax_devices": ["TFRT_CPU_0"]},
        "stages": {
            "first_order_solve": {
                "status": "ok",
                "result": {"solution_matrix": [[1.0, 0.0]]},
            },
            "kalman_value": {"status": "ok", "result": {"value": -1.0}},
            "kalman_paths": {"status": "ok", "result": paths_result},
            "gate_stats": {"status": "ok", "result": gate_result},
            "switching_fixed": {"status": "ok", "result": {"value": -1.0}},
            "switching_value": {"status": "ok", "result": {"value": -1.0}},
            "numpyro_kalman_log_density": {
                "status": "ok",
                "result": {"value": -1.0},
            },
            "numpyro_switching_log_density": {
                "status": "ok",
                "result": {"value": -1.0},
            },
            "numpyro_nuts_smoke": {
                "status": "skipped",
                "reason": "numpyro_nuts_samples=0",
            },
        },
    }
    results = {
        "cases": {"large_sw07_hlt_switching_order_3h": case_result},
    }
    report = module._report_text(
        config,
        payload,
        results,
        {"julia_version": "1.12.6", **results},
        {"real_s": 10.0, "user_s": 20.0, "sys_s": 1.0},
        {"real_s": 12.0, "user_s": 18.0, "sys_s": 1.5},
        {
            "cpu_brand": "test-cpu",
            "logical_cpu": "10",
            "perf_cores": "4",
            "eff_cores": "6",
            "mem_bytes": "17179869184",
        },
    )

    assert "`large_sw07_hlt_switching_order_3h`" in report
    assert "Cases:" in report
    assert "Medium model" not in report
