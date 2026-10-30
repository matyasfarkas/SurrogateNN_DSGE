from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_DATA_SHA256 = "35e09d915973e89442e50584f790f4859393219818c03ec8f2eead2b8a3c2050"
EXPECTED_PARAMETER_COUNT = 36
EXPECTED_DYNARE_LOGLIKELIHOOD = -1045.8945641261721
EXPECTED_PARAMETER_NAMES = (
    "z_ea",
    "z_eb",
    "z_eg",
    "z_eqs",
    "z_em",
    "z_epinf",
    "z_ew",
    "crhoa",
    "crhob",
    "crhog",
    "crhoqs",
    "crhoms",
    "crhopinf",
    "crhow",
    "cmap",
    "cmaw",
    "csadjcost",
    "csigma",
    "chabb",
    "cprobw",
    "csigl",
    "cprobp",
    "cindw",
    "cindp",
    "czcap",
    "cfc",
    "crpi",
    "crr",
    "cry",
    "crdy",
    "constepinf",
    "constebeta",
    "constelab",
    "ctrend",
    "cgy",
    "calfa",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _gate(pass_: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {"passed": bool(pass_), "observed": observed, "requirement": requirement}


def _raw_draw_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = result.get("benchmark") or {}
    diagnostics = result.get("posterior_diagnostics") or {}
    recorded_parameters = diagnostics.get("parameters") or {}
    raw = result.get("posterior_draws") or {}
    chains = int(benchmark.get("chains") or 0)
    samples = int(benchmark.get("samples") or 0)
    expected_shape = (chains, samples)
    if set(raw) != set(EXPECTED_PARAMETER_NAMES):
        return {
            "passed": False,
            "reason": "raw parameter names do not match the declared 36-parameter block",
            "observed_parameter_names": list(raw),
            "expected_shape": list(expected_shape),
        }

    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    ess_values: list[float] = []
    r_hat_values: list[float] = []
    summary_max_abs_error = 0.0
    invalid: list[str] = []
    summary_fields = ("mean", "std", "q05", "median", "q95")
    for name in EXPECTED_PARAMETER_NAMES:
        values = np.asarray(raw[name], dtype=np.float64)
        recorded = recorded_parameters.get(name) or {}
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            invalid.append(name)
            continue
        flat = values.reshape(-1)
        recomputed = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.quantile(flat, 0.50)),
            "q95": float(np.quantile(flat, 0.95)),
        }
        if list(recorded.get("shape") or []) != list(expected_shape):
            invalid.append(name)
            continue
        try:
            summary_max_abs_error = max(
                summary_max_abs_error,
                *(abs(float(recorded[field]) - recomputed[field]) for field in summary_fields),
            )
            ess = float(np.asarray(effective_sample_size(values), dtype=np.float64))
            r_hat = float(np.asarray(split_gelman_rubin(values), dtype=np.float64))
        except (KeyError, TypeError, ValueError):
            invalid.append(name)
            continue
        if not np.isfinite(ess) or not np.isfinite(r_hat):
            invalid.append(name)
            continue
        ess_values.append(ess)
        r_hat_values.append(r_hat)

    recomputed_min_ess = float(min(ess_values)) if ess_values else None
    recomputed_max_r_hat = float(max(r_hat_values)) if r_hat_values else None
    recorded_min_ess = diagnostics.get("min_ess")
    recorded_max_r_hat = diagnostics.get("max_r_hat")
    diagnostics_match = bool(
        recomputed_min_ess is not None
        and recomputed_max_r_hat is not None
        and recorded_min_ess is not None
        and recorded_max_r_hat is not None
        and np.isclose(recomputed_min_ess, float(recorded_min_ess), rtol=1.0e-10, atol=1.0e-10)
        and np.isclose(
            recomputed_max_r_hat,
            float(recorded_max_r_hat),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    )
    passed = bool(
        not invalid
        and len(ess_values) == EXPECTED_PARAMETER_COUNT
        and summary_max_abs_error <= 1.0e-10
        and diagnostics_match
    )
    return {
        "passed": passed,
        "expected_shape": list(expected_shape),
        "invalid_parameters": invalid,
        "summary_max_absolute_error": summary_max_abs_error,
        "recomputed_min_ess": recomputed_min_ess,
        "recorded_min_ess": recorded_min_ess,
        "recomputed_max_r_hat": recomputed_max_r_hat,
        "recorded_max_r_hat": recorded_max_r_hat,
    }


def _run_gates(
    result: Mapping[str, Any],
    *,
    require_reference_match: bool,
    expected_initialization: str,
) -> dict[str, Any]:
    benchmark = result.get("benchmark") or {}
    runtime = result.get("runtime") or {}
    diagnostics = result.get("posterior_diagnostics") or {}
    extra = result.get("extra_fields") or {}
    provenance = benchmark.get("data_provenance") or {}
    divergences = ((extra.get("diverging") or {}).get("count"))
    num_steps_max = ((extra.get("num_steps") or {}).get("max"))
    max_tree_depth = benchmark.get("max_tree_depth")
    max_steps = (2 ** int(max_tree_depth) - 1) if max_tree_depth is not None else None
    raw_audit = _raw_draw_audit(result)
    gates = {
        "paper_preset": _gate(
            benchmark.get("preset") == "sw07_paper",
            benchmark.get("preset"),
            "sw07_paper",
        ),
        "data_fingerprint": _gate(
            provenance.get("sha256") == EXPECTED_DATA_SHA256,
            provenance.get("sha256"),
            EXPECTED_DATA_SHA256,
        ),
        "sample": _gate(
            benchmark.get("periods") == 180
            and provenance.get("sample_start_index_zero_based") == 50,
            {
                "periods": benchmark.get("periods"),
                "start": provenance.get("sample_start_index_zero_based"),
            },
            "180 quarters starting at archived row 50 (1960Q1--2004Q4)",
        ),
        "estimands": _gate(
            benchmark.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and tuple(benchmark.get("parameter_names") or ()) == EXPECTED_PARAMETER_NAMES,
            {
                "count": benchmark.get("parameter_count"),
                "names": benchmark.get("parameter_names"),
            },
            "the ordered 36-parameter published block",
        ),
        "dynamic_steady_state": _gate(
            benchmark.get("steady_state_mode") == "solve",
            benchmark.get("steady_state_mode"),
            "solve",
        ),
        "matched_likelihood_configuration": _gate(
            benchmark.get("dtype") == "float64"
            and benchmark.get("qme_algorithm") == "schur_gpu"
            and benchmark.get("initial_covariance_strategy") == "theoretical"
            and benchmark.get("presample_periods") == 6
            and float(benchmark.get("measurement_error_scale", np.nan)) == 0.0
            and float(benchmark.get("jitter", np.nan)) == 0.0
            and runtime.get("jax_default_backend") in {"gpu", "cuda"},
            {
                "dtype": benchmark.get("dtype"),
                "qme_algorithm": benchmark.get("qme_algorithm"),
                "initial_covariance_strategy": benchmark.get("initial_covariance_strategy"),
                "presample_periods": benchmark.get("presample_periods"),
                "measurement_error_scale": benchmark.get("measurement_error_scale"),
                "jitter": benchmark.get("jitter"),
                "backend": runtime.get("jax_default_backend"),
            },
            "float64 GPU Schur likelihood, theoretical covariance, six presample periods, zero measurement error/jitter",
        ),
        "declared_initialization": _gate(
            benchmark.get("sw07_initialization") == expected_initialization,
            benchmark.get("sw07_initialization"),
            expected_initialization,
        ),
        "sampler_configuration": _gate(
            benchmark.get("kernel") == "nuts"
            and benchmark.get("dense_mass") is True
            and float(benchmark.get("target_accept_prob") or 0.0) >= 0.9
            and int(benchmark.get("max_tree_depth") or 0) >= 10,
            {
                "kernel": benchmark.get("kernel"),
                "dense_mass": benchmark.get("dense_mass"),
                "target_accept_prob": benchmark.get("target_accept_prob"),
                "max_tree_depth": benchmark.get("max_tree_depth"),
            },
            "dense-mass NUTS, target acceptance >= 0.9, max tree depth >= 10",
        ),
        "sampler_length": _gate(
            int(benchmark.get("warmup") or 0) >= 500
            and int(benchmark.get("samples") or 0) >= 1000
            and int(benchmark.get("chains") or 0) >= 4,
            {
                "warmup": benchmark.get("warmup"),
                "samples": benchmark.get("samples"),
                "chains": benchmark.get("chains"),
            },
            "at least 500 warmup, 1000 retained draws, and 4 chains",
        ),
        "zero_divergences": _gate(divergences == 0, divergences, "0"),
        "r_hat": _gate(
            diagnostics.get("max_r_hat") is not None
            and float(diagnostics["max_r_hat"]) <= 1.01,
            diagnostics.get("max_r_hat"),
            "maximum <= 1.01",
        ),
        "effective_sample_size": _gate(
            diagnostics.get("min_ess") is not None
            and float(diagnostics["min_ess"]) >= 400.0,
            diagnostics.get("min_ess"),
            "minimum >= 400",
        ),
        "tree_depth": _gate(
            num_steps_max is not None and max_steps is not None and float(num_steps_max) < max_steps,
            num_steps_max,
            f"maximum leapfrog steps < {max_steps}",
        ),
        "raw_draws_recompute": _gate(
            raw_audit["passed"],
            raw_audit,
            "finite chain-by-draw arrays for all 36 parameters with reproducible summaries, ESS, and R-hat",
        ),
    }
    if require_reference_match:
        comparison = result.get("sw07_reference_comparison") or {}
        gates.update(
            {
                "reference_complete": _gate(
                    comparison.get("complete") is True,
                    comparison.get("missing_parameters"),
                    "all 36 parameters compared",
                ),
                "reference_median_mean_error": _gate(
                    comparison.get("median_normalized_mean_error") is not None
                    and float(comparison["median_normalized_mean_error"]) <= 0.25,
                    comparison.get("median_normalized_mean_error"),
                    "median absolute mean error <= 0.25 reference interval widths",
                ),
                "reference_max_mean_error": _gate(
                    comparison.get("max_normalized_mean_error") is not None
                    and float(comparison["max_normalized_mean_error"]) <= 1.0,
                    comparison.get("max_normalized_mean_error"),
                    "maximum absolute mean error <= 1 reference interval width",
                ),
                "reference_interval_membership": _gate(
                    int(comparison.get("posterior_means_inside_reference_90pct_count") or 0)
                    >= 32,
                    comparison.get("posterior_means_inside_reference_90pct_count"),
                    "at least 32/36 posterior means inside published 90% intervals",
                ),
            }
        )
    return gates


def _preflight_parity(cpu: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    cpu_preflight = cpu.get("preflight") or {}
    gpu_preflight = gpu.get("preflight") or {}
    cpu_value = float(cpu_preflight.get("loglikelihood", np.nan))
    gpu_value = float(gpu_preflight.get("loglikelihood", np.nan))
    cpu_gradient = np.asarray(cpu_preflight.get("gradient", []), dtype=np.float64)
    gpu_gradient = np.asarray(gpu_preflight.get("gradient", []), dtype=np.float64)
    value_abs_error = abs(cpu_value - gpu_value)
    gradient_max_abs_error = (
        float(np.max(np.abs(cpu_gradient - gpu_gradient)))
        if cpu_gradient.shape == gpu_gradient.shape and cpu_gradient.size
        else float("inf")
    )
    cpu_benchmark = cpu.get("benchmark") or {}
    gpu_benchmark = gpu.get("benchmark") or {}
    cpu_runtime = cpu.get("runtime") or {}
    gpu_runtime = gpu.get("runtime") or {}
    cpu_provenance = cpu_benchmark.get("data_provenance") or {}
    gpu_provenance = gpu_benchmark.get("data_provenance") or {}
    gradients_are_complete_and_finite = bool(
        cpu_gradient.shape == gpu_gradient.shape == (EXPECTED_PARAMETER_COUNT,)
        and np.all(np.isfinite(cpu_gradient))
        and np.all(np.isfinite(gpu_gradient))
    )
    metadata_match = bool(
        cpu_benchmark.get("preset") == gpu_benchmark.get("preset") == "sw07_paper"
        and cpu_benchmark.get("parameter_names") == gpu_benchmark.get("parameter_names")
        and cpu_benchmark.get("steady_state_mode") == gpu_benchmark.get("steady_state_mode") == "solve"
        and cpu_benchmark.get("dtype") == gpu_benchmark.get("dtype") == "float64"
        and cpu_benchmark.get("periods") == gpu_benchmark.get("periods") == 180
        and cpu_benchmark.get("presample_periods")
        == gpu_benchmark.get("presample_periods")
        == 6
        and cpu_benchmark.get("initial_covariance_strategy")
        == gpu_benchmark.get("initial_covariance_strategy")
        == "theoretical"
        and float(cpu_benchmark.get("measurement_error_scale", np.nan)) == 0.0
        and float(gpu_benchmark.get("measurement_error_scale", np.nan)) == 0.0
        and float(cpu_benchmark.get("jitter", np.nan)) == 0.0
        and float(gpu_benchmark.get("jitter", np.nan)) == 0.0
        and cpu_provenance.get("sha256") == gpu_provenance.get("sha256") == EXPECTED_DATA_SHA256
        and cpu_provenance.get("model_source_sha256")
        == gpu_provenance.get("model_source_sha256")
        and cpu_provenance.get("sample_start_index_zero_based")
        == gpu_provenance.get("sample_start_index_zero_based")
        == 50
    )
    backend_match = bool(
        cpu_runtime.get("jax_default_backend") == "cpu"
        and gpu_runtime.get("jax_default_backend") in {"gpu", "cuda"}
        and cpu_benchmark.get("qme_algorithm") == "schur"
        and gpu_benchmark.get("qme_algorithm") == "schur_gpu"
    )
    return {
        "passed": bool(
            np.isfinite(cpu_value)
            and np.isfinite(gpu_value)
            and abs(cpu_value - EXPECTED_DYNARE_LOGLIKELIHOOD) <= 1.0e-8
            and abs(gpu_value - EXPECTED_DYNARE_LOGLIKELIHOOD) <= 1.0e-8
            and value_abs_error <= 1.0e-6
            and gradient_max_abs_error <= 1.0e-5
            and gradients_are_complete_and_finite
            and metadata_match
            and backend_match
        ),
        "cpu_loglikelihood": cpu_value,
        "gpu_loglikelihood": gpu_value,
        "loglikelihood_absolute_error": value_abs_error,
        "gradient_max_absolute_error": gradient_max_abs_error,
        "dynare_reference_loglikelihood": EXPECTED_DYNARE_LOGLIKELIHOOD,
        "cpu_dynare_absolute_error": abs(cpu_value - EXPECTED_DYNARE_LOGLIKELIHOOD),
        "gpu_dynare_absolute_error": abs(gpu_value - EXPECTED_DYNARE_LOGLIKELIHOOD),
        "gradients_complete_and_finite": gradients_are_complete_and_finite,
        "metadata_match": metadata_match,
        "backend_and_solver_match": backend_match,
        "requirements": {
            "loglikelihood_absolute_error": "<= 1e-6",
            "gradient_max_absolute_error": "<= 1e-5",
            "backend_to_dynare_loglikelihood_absolute_error": "<= 1e-8",
        },
    }


def _basin_separation(
    dynare_run: Mapping[str, Any],
    prior_mean_run: Mapping[str, Any],
) -> dict[str, Any]:
    dynare_metrics = (dynare_run.get("posterior_diagnostics") or {}).get("parameters") or {}
    prior_metrics = (prior_mean_run.get("posterior_diagnostics") or {}).get("parameters") or {}
    common = tuple(sorted(set(dynare_metrics).intersection(prior_metrics)))
    normalized_differences: list[float] = []
    for name in common:
        pooled_scale = max(
            float(dynare_metrics[name].get("std") or 0.0),
            float(prior_metrics[name].get("std") or 0.0),
            1.0e-12,
        )
        normalized_differences.append(
            abs(float(dynare_metrics[name]["mean"]) - float(prior_metrics[name]["mean"]))
            / pooled_scale
        )
    max_distance = float(max(normalized_differences)) if normalized_differences else None
    return {
        "assessed": len(common) == EXPECTED_PARAMETER_COUNT,
        "parameter_count": len(common),
        "max_mean_distance_in_pooled_sd": max_distance,
        "distinct_basin_detected": bool(max_distance is not None and max_distance >= 2.0),
        "interpretation": (
            "Distinct modes/basins detected from dispersed initializations."
            if max_distance is not None and max_distance >= 2.0
            else "No separated basin detected; this does not prove unimodality."
        ),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    dynare_run = _load(args.dynare_run)
    prior_mean_run = _load(args.prior_mean_run)
    cpu_preflight = _load(args.cpu_preflight)
    gpu_preflight = _load(args.gpu_preflight)
    dynare_gates = _run_gates(
        dynare_run,
        require_reference_match=True,
        expected_initialization="dynare",
    )
    prior_mean_gates = _run_gates(
        prior_mean_run,
        require_reference_match=False,
        expected_initialization="prior_mean",
    )
    parity = _preflight_parity(cpu_preflight, gpu_preflight)
    basins = _basin_separation(dynare_run, prior_mean_run)
    model_hashes = [
        ((artifact.get("benchmark") or {}).get("data_provenance") or {}).get(
            "model_source_sha256"
        )
        for artifact in (dynare_run, prior_mean_run, cpu_preflight, gpu_preflight)
    ]
    cross_artifact_gates = {
        "model_source_fingerprint": _gate(
            all(model_hashes) and len(set(model_hashes)) == 1,
            model_hashes,
            "identical nonempty model-source SHA256 across both posterior runs and both preflights",
        )
    }
    required = [
        *(gate["passed"] for gate in dynare_gates.values()),
        *(gate["passed"] for gate in prior_mean_gates.values()),
        parity["passed"],
        basins["assessed"],
        *(gate["passed"] for gate in cross_artifact_gates.values()),
    ]
    return {
        "publication_ready": bool(all(required)),
        "dynare_initialization_gates": dynare_gates,
        "prior_mean_initialization_gates": prior_mean_gates,
        "cpu_gpu_preflight_parity": parity,
        "basin_search": basins,
        "cross_artifact_gates": cross_artifact_gates,
        "scope": (
            "Linearized empirical SW07 posterior replication only; nonlinear HLT/ResNN "
            "validation remains a separate evidence layer."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper-matched SW07 GPU evidence.")
    parser.add_argument("--dynare-run", type=Path, required=True)
    parser.add_argument("--prior-mean-run", type=Path, required=True)
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"publication_ready": report["publication_ready"]}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
