#!/usr/bin/env python3
"""Evaluate computational evidence for a reduced full-equation Galí experiment.

The input is a JSON object with the sections used in ``GATES`` below.  The
script does not run an economic model.  It turns benchmark output into an
auditable pass/fail report and exits with status 2 when any required check
fails.  The model must be the full equation set in
``Gali_2015_chapter_3_nonlinear``.  Only the parameter subset, horizon, and
sampling budget may be reduced.  A pass in ``publication`` mode means only
that this reduced experiment meets the declared thresholds; it does not
validate a full-scale estimation design or a different DSGE model.

Example
-------
``python benchmarks/publication_readiness_gate.py evidence.json --mode publication``

The three modes have distinct purposes:

``smoke``
    Detect broken execution and non-finite output.  It is not paper evidence.
``evidence``
    Require a moderate independent validation exercise suitable for
    development and internal review.
``publication``
    Require the minimum numerical coverage, posterior diagnostics, exact
    cross-checks, reproducibility metadata, and timing protocol used for a
    reduced Galí experiment in a paper. A pass is a computational evidence
    result, not an assertion that the manuscript itself is publication-ready.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence


SCHEMA_VERSION = "surrogatenn-dsge-publication-gate/v2"
MODES = ("smoke", "evidence", "publication")
GALI_MODEL_SOURCE_SHA256 = "53f909976abc962494693c8ba5d30223653cee6164fc9f183064e3776240fb55"
_MISSING = object()


class Gate(NamedTuple):
    """One required observation and its mode-specific acceptance rule."""

    category: str
    path: str
    operator: str
    thresholds: Mapping[str, Any]
    rationale: str


def _gate(
    category: str,
    path: str,
    operator: str,
    smoke: Any,
    evidence: Any,
    publication: Any,
    rationale: str,
) -> Gate:
    return Gate(
        category=category,
        path=path,
        operator=operator,
        thresholds={"smoke": smoke, "evidence": evidence, "publication": publication},
        rationale=rationale,
    )


# Thresholds are deliberately absolute and are serialized into every report.
# Accuracy tolerances assume float64 reference calculations.  Counts concern
# held-out parameter draws, not repeated evaluations at a single calibration.
GATES: tuple[Gate, ...] = (
    _gate("scope", "metadata.model_name", "eq",
          "Gali_2015_chapter_3_nonlinear", "Gali_2015_chapter_3_nonlinear",
          "Gali_2015_chapter_3_nonlinear",
          "Require the named full nonlinear MacroModelling Galí model."),
    _gate("scope", "metadata.nonlinear_model", "eq", True, True, True,
          "The benchmark must exercise the nonlinear, not the three-equation linear, case."),
    _gate("scope", "metadata.full_equation_set", "eq", True, True, True,
          "Reduction applies to the experiment, not to the model equations."),
    _gate("scope", "metadata.equation_count", "eq", 23, 23, 23,
          "The canonical nonlinear Galí source contains 23 model equations."),
    _gate("scope", "metadata.parameter_count", "eq", 16, 16, 16,
          "The canonical nonlinear Galí source declares 16 parameters."),
    _gate("scope", "metadata.shock_count", "eq", 3, 3, 3,
          "The canonical nonlinear Galí source contains three shocks."),
    _gate("scope", "metadata.model_source_sha256", "eq",
          GALI_MODEL_SOURCE_SHA256, GALI_MODEL_SOURCE_SHA256, GALI_MODEL_SOURCE_SHA256,
          "The source must match the canonical MacroModelling model byte for byte."),
    _gate("scope", "metadata.experiment_scope", "eq", "reduced_experiment",
          "reduced_experiment", "reduced_experiment",
          "The gate is calibrated for a reduced design using the full equation set."),
    _gate("scope", "metadata.claimed_scope", "eq", "reduced_experiment",
          "reduced_experiment", "reduced_experiment",
          "Reduced evidence must not be presented as full-scale validation."),
    _gate("scope", "metadata.reduction.parameter_subset_declared", "eq", True, True, True,
          "The estimated parameter subset must be reported explicitly."),
    _gate("scope", "metadata.reduction.horizon_declared", "eq", True, True, True,
          "The reduced simulation and likelihood horizon must be reported."),
    _gate("scope", "metadata.reduction.sampling_budget_declared", "eq", True, True, True,
          "Training, validation, and posterior sampling budgets must be reported."),
    _gate("scope", "metadata.reduction.estimated_parameter_count", "min_int", 1, 1, 1,
          "The reduced experiment must estimate at least one parameter."),
    _gate("scope", "metadata.reduction.estimated_parameter_count", "max_int", 6, 6, 6,
          "At most six of the 16 parameters are estimated in the reduced experiment."),
    _gate("scope", "metadata.reduction.parameter_names", "nonempty_list", True, True, True,
          "The names of the estimated parameters must be recorded."),
    _gate("scope", "metadata.reduction.likelihood_periods", "min_int", 10, 50, 100,
          "The reduced horizon remains long enough to expose accumulated likelihood error."),
    _gate("scope", "metadata.reduction.training_samples", "min_int", 16, 500, 2000,
          "The declared training budget must support the corresponding validation mode."),

    _gate("reproducibility", "reproducibility.seed_count", "min_int", 1, 2, 3,
          "Multiple seeds reveal initialization- or simulation-specific conclusions."),
    _gate("reproducibility", "reproducibility.repeat_count", "min_int", 1, 3, 5,
          "Repeated executions are needed to measure numerical repeatability."),
    _gate("reproducibility", "reproducibility.same_seed_max_abs_diff", "max",
          1.0e-8, 1.0e-10, 1.0e-12,
          "A fixed seed should reproduce float64 benchmark outputs to roundoff."),
    _gate("reproducibility", "reproducibility.git_commit", "git_sha", True, True, True,
          "The exact source revision must be recorded."),
    _gate("reproducibility", "reproducibility.dirty_worktree", "eq", False, False, False,
          "Runs must use a clean, recoverable revision."),
    _gate("reproducibility", "reproducibility.environment_sha256", "sha256",
          True, True, True,
          "Package, accelerator, precision, and platform versions must be recoverable."),
    _gate("reproducibility", "reproducibility.float64_enabled", "eq", True, True, True,
          "Accuracy claims and reference tolerances assume float64 arithmetic."),
    _gate("reproducibility", "reproducibility.input_sha256", "sha256",
          True, True, True,
          "The benchmark input and synthetic data must be content-addressed."),

    _gate("numerical_solution", "numerical_solution.parameter_draws", "min_int", 8, 100, 1000,
          "Coverage must extend beyond a few favorable calibrations."),
    _gate("numerical_solution", "numerical_solution.runtime_success_count", "min_int", 0, 0, 0,
          "The runtime-success share must be backed by an explicit numerator."),
    _gate("numerical_solution", "numerical_solution.full_path_success_count", "min_int", 0, 0, 0,
          "The full-path success share must be backed by an explicit numerator."),
    _gate("numerical_solution", "numerical_solution.steady_state_fallback_count", "min_int", 0, 0, 0,
          "The fallback share must be backed by an explicit numerator."),
    _gate("numerical_solution", "numerical_solution.runtime_success_share", "min_fraction",
          0.80, 0.95, 0.99,
          "Steady-state and first-order preparation must work over the declared support."),
    _gate("numerical_solution", "numerical_solution.full_path_success_share", "min_fraction",
          0.75, 0.90, 0.99,
          "The complete nonlinear path, not only its stable prefix, must solve."),
    _gate("numerical_solution", "numerical_solution.steady_state_fallback_share", "max_fraction",
          0.20, 0.05, 0.01,
          "Reference-steady-state fallback cannot substitute for parameter-specific solves."),
    _gate("numerical_solution", "numerical_solution.nonfinite_count", "max_int", 0, 0, 0,
          "Non-finite solutions invalidate downstream accuracy comparisons."),
    _gate("numerical_solution", "numerical_solution.boundary_stress_draws", "min_int",
          0, 20, 100,
          "Near-boundary draws must be audited explicitly rather than filtered from the design."),
    _gate("numerical_solution", "numerical_solution.ambiguity_band_declared", "eq",
          None, True, True,
          "Evidence runs must prespecify the unit-circle ambiguity band."),

    _gate("qz_sep_accuracy", "qz_sep_accuracy.reference_draws", "min_int", 8, 100, 1000,
          "Real-QZ classifications require broad comparison with an independent reference."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.determinacy_agreement_share", "min_fraction", 1.0, 1.0, 1.0,
          "Unique-stable-solution classification must agree on every audited draw."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.max_solution_relative_error", "max",
          1.0e-6, 1.0e-8, 1.0e-9,
          "Accepted real-QZ solutions must match the reference solution matrix."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.max_invariant_subspace_residual", "max",
          1.0e-6, 1.0e-8, 1.0e-10,
          "A small pencil residual checks the computed invariant subspace directly."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.sep_paths", "min_int", 8, 100, 500,
          "SEP coverage must include independent parameter-shock paths."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.sep_branching_order", "min_int", 0, 1, 1,
          "Evidence runs must exercise stochastic expectations rather than order-zero extended path."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.sep_nodes", "min_int", 1, 3, 3,
          "Positive-order SEP requires multiple quadrature nodes."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.all_shocks_nonzero_coverage", "eq",
          None, True, True,
          "Every declared structural shock must vary in the stochastic SEP validation set."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.sep_success_share", "min_fraction", 0.75, 0.95, 0.99,
          "The nonlinear reference solver must be reliable over the benchmark support."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.sep_residual_p99", "max",
          1.0e-4, 1.0e-6, 1.0e-8,
          "The 99th percentile guards against a good mean hiding poor nonlinear solves."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.reference_independent_of_surrogate", "eq",
          True, True, True,
          "The nonlinear target must be computed without using the fitted surrogate."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.reference_implementation", "nonempty",
          True, True, True,
          "The nonlinear reference implementation must be identified."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.cross_implementation_reference", "eq",
          None, "MacroModelling.jl", "MacroModelling.jl",
          "Evidence runs require a Julia MacroModelling cross-implementation check."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.cross_implementation_sep_paths", "min_int",
          None, 20, 100,
          "A separate implementation reduces the risk of common-code agreement."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.cross_implementation_success_share", "min_fraction",
          None, 0.95, 0.99,
          "Python and Julia SEP must both solve almost all audited paths."),
    _gate("qz_sep_accuracy", "qz_sep_accuracy.cross_implementation_max_relative_error", "max",
          None, 1.0e-5, 1.0e-6,
          "Successful Python and Julia nonlinear trajectories must agree numerically."),

    _gate("surrogate_holdout", "surrogate_holdout.independent_theta_split", "eq",
          True, True, True,
          "Holding out rows from seen parameters is not an out-of-parameter validation."),
    _gate("surrogate_holdout", "surrogate_holdout.theta_draws", "min_int", 4, 50, 200,
          "Holdout accuracy must cover distinct parameter values."),
    _gate("surrogate_holdout", "surrogate_holdout.samples", "min_int", 16, 500, 2000,
          "The validation set must cover parameter-shock-state combinations."),
    _gate("surrogate_holdout", "surrogate_holdout.normalized_rmse", "max",
          0.25, 0.10, 0.05,
          "RMSE is normalized by held-out target standard deviations."),
    _gate("surrogate_holdout", "surrogate_holdout.normalized_error_p99", "max",
          0.75, 0.25, 0.10,
          "Tail error matters because posterior trajectories can visit rare states."),
    _gate("surrogate_holdout", "surrogate_holdout.improvement_over_rom", "min",
          0.0, 0.25, 0.50,
          "The surrogate should reduce nonlinear-state error relative to the ROM baseline."),

    _gate("likelihood_parity", "likelihood_parity.theta_draws", "min_int", 4, 50, 200,
          "Likelihood parity must hold away from one calibration."),
    _gate("likelihood_parity", "likelihood_parity.parameter_dependent_runtime", "eq",
          None, True, True,
          "Evidence likelihoods must recompute parameter-specific steady states and ROM objects."),
    _gate("likelihood_parity", "likelihood_parity.observations_per_draw", "min_int", 10, 50, 100,
          "Short likelihoods can conceal accumulated approximation error."),
    _gate("likelihood_parity", "likelihood_parity.mean_abs_error_per_observation", "max",
          0.10, 0.02, 0.01,
          "Per-observation scaling makes likelihood errors comparable across samples."),
    _gate("likelihood_parity", "likelihood_parity.max_abs_error_per_observation", "max",
          0.50, 0.10, 0.05,
          "A worst-case bound prevents local posterior distortion from being averaged away."),
    _gate("likelihood_parity", "likelihood_parity.correlation", "min",
          0.95, 0.99, 0.995,
          "Likelihood rankings across parameters drive posterior geometry."),
    _gate("likelihood_parity", "likelihood_parity.gradient_cosine_min", "min",
          0.90, 0.99, 0.999,
          "HMC depends on the direction of the log-posterior gradient."),

    _gate("pipeline_validation", "pipeline_validation.reference_generated_dgp", "eq",
          True, True, True,
          "Known-truth data must be generated by the declared nonlinear reference."),
    _gate("pipeline_validation", "pipeline_validation.common_data", "eq",
          True, True, True,
          "Reference and surrogate arms must use byte-identical observations."),
    _gate("pipeline_validation", "pipeline_validation.common_priors_transforms", "eq",
          True, True, True,
          "Prior and parameter-transform differences cannot be confounded with the surrogate."),
    _gate("pipeline_validation", "pipeline_validation.common_objective_terms", "eq",
          True, True, True,
          "The two arms must differ only in the nonlinear transition evaluator."),
    _gate("pipeline_validation", "pipeline_validation.common_chain_seeds", "eq",
          True, True, True,
          "Matched seeds reduce Monte Carlo noise in direct-versus-surrogate comparisons."),
    _gate("pipeline_validation", "pipeline_validation.heldout_residual_layer_passed", "eq",
          True, True, True,
          "The residual layer must pass before it is embedded in inversion or HMC."),
    _gate("pipeline_validation", "pipeline_validation.inversion_grid_layer_passed", "eq",
          True, True, True,
          "A common-shock inversion grid isolates transition error from shock recovery."),
    _gate("pipeline_validation", "pipeline_validation.matched_hmc_layer_passed", "eq",
          True, True, True,
          "The complete estimator must pass a matched direct-reference versus surrogate HMC check."),
    _gate("pipeline_validation", "pipeline_validation.attempted_cases_preserved", "eq",
          True, True, True,
          "Failed direct or surrogate evaluations must remain in the denominator."),
    _gate("pipeline_validation", "pipeline_validation.grid_finite_share", "min_fraction",
          0.80, 0.95, 1.00,
          "The posterior-surface comparison must be finite on the prespecified grid."),
    _gate("pipeline_validation", "pipeline_validation.grid_resolution_converged", "eq",
          None, True, True,
          "Posterior summaries must be stable when the reference grid is refined."),
    _gate("pipeline_validation", "pipeline_validation.interval_overlap_share", "min_fraction",
          0.50, 1.00, 1.00,
          "Direct-reference and surrogate intervals must overlap for every estimated parameter."),
    _gate("pipeline_validation", "pipeline_validation.true_value_coverage_share", "min_fraction",
          0.00, 0.50, 1.00,
          "A known-truth experiment must report whether both arms cover every identified truth."),
    _gate("pipeline_validation", "pipeline_validation.max_abs_mean_diff_combined_mcse", "max",
          2.00, 1.00, 0.50,
          "Posterior-mean differences are judged relative to combined Monte Carlo error."),
    _gate("pipeline_validation", "pipeline_validation.direct_numerical_failure_share", "max_fraction",
          0.10, 0.01, 0.005,
          "Direct-reference failures must be rare and reported rather than discarded."),
    _gate("pipeline_validation", "pipeline_validation.surrogate_numerical_failure_share", "max_fraction",
          0.10, 0.01, 0.001,
          "Surrogate numerical failures must be rare and reported rather than discarded."),
    _gate("pipeline_validation", "pipeline_validation.draws_per_objective", "min_int",
          20, 500, 4000,
          "The matched comparison needs enough retained draws to resolve posterior differences."),
    _gate("pipeline_validation", "pipeline_validation.identification_failures_preserved", "eq",
          None, True, True,
          "Agreement between two poorly identified targets is recorded as an identification failure."),

    _gate("hmc", "hmc.chains", "min_int", 1, 4, 4,
          "Independent chains are required for convergence diagnostics."),
    _gate("hmc", "hmc.warmup_per_chain", "min_int", 25, 500, 1000,
          "Adaptation must be long enough to tune the sampler."),
    _gate("hmc", "hmc.samples_per_chain", "min_int", 50, 500, 1000,
          "Posterior diagnostics need enough post-warmup draws."),
    _gate("hmc", "hmc.max_r_hat", "max", None, 1.05, 1.01,
          "Split R-hat checks mixing across chains."),
    _gate("hmc", "hmc.min_bulk_ess", "min", None, 200, 400,
          "Bulk ESS controls accuracy of central posterior summaries."),
    _gate("hmc", "hmc.min_tail_ess", "min", None, 200, 400,
          "Tail ESS controls accuracy of intervals and tail probabilities."),
    _gate("hmc", "hmc.accepted_share", "min_fraction", 0.05, 0.50, 0.60,
          "Static HMC needs enough accepted proposals to move through the posterior."),
    _gate("hmc", "hmc.accepted_share", "max_fraction", 0.999, 0.98, 0.95,
          "Near-unit acceptance indicates an inefficiently small static-HMC step size."),
    _gate("hmc", "hmc.numerical_failure_share", "max_fraction", 0.10, 0.01, 0.001,
          "Static-HMC transitions with numerical failures cannot support posterior claims."),
    _gate("hmc", "hmc.nonfinite_log_prob_count", "max_int", 0, 0, 0,
          "The recorded static-HMC log density must remain finite on every transition."),
    _gate("hmc", "hmc.exact_reference_draws", "min_int", 50, 500, 1000,
          "Surrogate HMC must be compared with an exact-likelihood reference chain."),
    _gate("hmc", "hmc.max_standardized_mean_error", "max", 0.50, 0.20, 0.10,
          "Posterior mean differences are scaled by the exact posterior standard deviation."),
    _gate("hmc", "hmc.max_sd_ratio_error", "max", 0.50, 0.20, 0.10,
          "Posterior uncertainty must agree, not only posterior means."),

    _gate("timing", "timing.warmup_runs", "min_int", 1, 3, 5,
          "Compilation and initialization must be separated from steady-state timing."),
    _gate("timing", "timing.measured_repeats", "min_int", 1, 10, 30,
          "Repeated measurements support medians and dispersion estimates."),
    _gate("timing", "timing.device_synchronized", "eq", True, True, True,
          "Asynchronous accelerator work must finish inside the timed region."),
    _gate("timing", "timing.reports_distribution", "eq", True, True, True,
          "Median and dispersion are more informative than one favorable timing."),
    _gate("timing", "timing.end_to_end_reported", "eq", True, True, True,
          "Kernel timing alone omits transfers, compilation, and orchestration costs."),
    _gate("timing", "timing.exact_baseline_reported", "eq", True, True, True,
          "Speed claims require the same-work exact baseline."),
    _gate("timing", "timing.hardware_software_recorded", "eq", True, True, True,
          "Timing results are interpretable only with hardware and software versions."),
    _gate("timing", "timing.exact_likelihood_median_s", "min", 1.0e-12, 1.0e-12, 1.0e-12,
          "The exact-likelihood baseline timing must be finite and positive."),
    _gate("timing", "timing.surrogate_likelihood_median_s", "min", 1.0e-12, 1.0e-12, 1.0e-12,
          "The surrogate-likelihood timing must be finite and positive."),
    _gate("timing", "timing.hmc_draws_per_second_median", "min", 1.0e-12, 1.0e-12, 1.0e-12,
          "Sampler throughput must be reported as a finite positive rate."),
    _gate("timing", "timing.end_to_end_median_s", "min", 1.0e-12, 1.0e-12, 1.0e-12,
          "End-to-end runtime must be reported as a finite positive duration."),
)


def _lookup(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _evaluate(value: Any, operator: str, threshold: Any) -> tuple[bool, str]:
    if value is _MISSING:
        return False, "missing"
    if value is None:
        return False, "null"
    if operator in {"min", "max", "min_int", "max_int", "min_fraction", "max_fraction"}:
        if operator.endswith("_int") and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            return False, "not a non-negative integer"
        if not _finite_number(value):
            return False, "not a finite number"
        if operator.endswith("_fraction") and not 0.0 <= float(value) <= 1.0:
            return False, "not a fraction in [0, 1]"
        if not _finite_number(threshold):
            raise TypeError(f"Invalid threshold {threshold!r} for {operator}.")
        if operator in {"min", "min_int", "min_fraction"}:
            return value >= threshold, f"expected >= {threshold!r}"
        return value <= threshold, f"expected <= {threshold!r}"
    if operator == "eq":
        return value == threshold, f"expected == {threshold!r}"
    if operator == "nonempty":
        passed = isinstance(value, str) and bool(value.strip())
        return passed, "expected a non-empty string"
    if operator == "nonempty_list":
        passed = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, str) and bool(item.strip()) for item in value)
            and len(value) == len(set(value))
        )
        return passed, "expected a non-empty list of unique, non-empty strings"
    if operator == "git_sha":
        passed = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None
        return passed, "expected a 40-character Git commit SHA"
    if operator == "sha256":
        passed = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        return passed, "expected a 64-character SHA-256 digest"
    raise ValueError(f"Unknown gate operator {operator!r}.")


def _numerical_solution_counts_are_consistent(
    payload: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], str]:
    """Check that reported numerical-solution shares use one declared denominator."""

    keys = (
        "parameter_draws",
        "runtime_success_count",
        "runtime_success_share",
        "full_path_success_count",
        "full_path_success_share",
        "steady_state_fallback_count",
        "steady_state_fallback_share",
    )
    observed = {
        key: _lookup(payload, f"numerical_solution.{key}")
        for key in keys
    }
    serializable = {
        key: None if value is _MISSING else value
        for key, value in observed.items()
    }
    draws = observed["parameter_draws"]
    count_keys = (
        "runtime_success_count",
        "full_path_success_count",
        "steady_state_fallback_count",
    )
    if (
        not isinstance(draws, int)
        or isinstance(draws, bool)
        or draws <= 0
        or any(
            not isinstance(observed[key], int)
            or isinstance(observed[key], bool)
            or not 0 <= observed[key] <= draws
            for key in count_keys
        )
    ):
        return (
            False,
            serializable,
            "counts must be non-negative integers no larger than parameter_draws",
        )
    if observed["full_path_success_count"] > observed["runtime_success_count"]:
        return (
            False,
            serializable,
            "full_path_success_count cannot exceed runtime_success_count",
        )

    count_share_pairs = (
        ("runtime_success_count", "runtime_success_share"),
        ("full_path_success_count", "full_path_success_share"),
        ("steady_state_fallback_count", "steady_state_fallback_share"),
    )
    for count_key, share_key in count_share_pairs:
        share = observed[share_key]
        expected = observed[count_key] / draws
        if not _finite_number(share) or not math.isclose(
            float(share), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        ):
            return (
                False,
                serializable,
                f"{share_key} must equal {count_key} / parameter_draws",
            )
    return True, serializable, "counts and shares use parameter_draws as their denominator"


def evaluate_payload(
    payload: Mapping[str, Any],
    *,
    mode: str,
    source: str | None = None,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a complete, machine-readable gate report."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")

    categories: dict[str, dict[str, Any]] = {}
    failed = 0
    evaluated = 0
    for gate in GATES:
        threshold = gate.thresholds[mode]
        observed = _lookup(payload, gate.path)
        category = categories.setdefault(
            gate.category,
            {"status": "pass", "passed": 0, "failed": 0, "not_required": 0, "checks": []},
        )
        if threshold is None:
            category["not_required"] += 1
            category["checks"].append(
                {
                    "path": gate.path,
                    "status": "not_required",
                    "observed": None if observed is _MISSING else observed,
                    "missing": observed is _MISSING,
                    "operator": gate.operator,
                    "threshold": None,
                    "detail": f"not required in {mode} mode",
                    "rationale": gate.rationale,
                }
            )
            continue
        evaluated += 1
        passed, detail = _evaluate(observed, gate.operator, threshold)
        if not passed:
            failed += 1
        category["passed" if passed else "failed"] += 1
        if not passed:
            category["status"] = "fail"
        category["checks"].append(
            {
                "path": gate.path,
                "status": "pass" if passed else "fail",
                "observed": None if observed is _MISSING else observed,
                "missing": observed is _MISSING,
                "operator": gate.operator,
                "threshold": threshold,
                "detail": detail,
                "rationale": gate.rationale,
            }
        )

    count = _lookup(payload, "metadata.reduction.estimated_parameter_count")
    names = _lookup(payload, "metadata.reduction.parameter_names")
    count_matches_names = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and isinstance(names, list)
        and count == len(names)
    )
    evaluated += 1
    scope = categories["scope"]
    scope["passed" if count_matches_names else "failed"] += 1
    if not count_matches_names:
        failed += 1
        scope["status"] = "fail"
    scope["checks"].append(
        {
            "path": "metadata.reduction.estimated_parameter_count_vs_parameter_names",
            "status": "pass" if count_matches_names else "fail",
            "observed": {
                "estimated_parameter_count": None if count is _MISSING else count,
                "parameter_names_length": (
                    len(names) if isinstance(names, list) else None
                ),
            },
            "missing": count is _MISSING or names is _MISSING,
            "operator": "consistent_length",
            "threshold": "equal",
            "detail": "parameter count must equal the number of parameter names",
            "rationale": "The declared parameter dimension and parameter list must agree.",
        }
    )

    numerical_counts_consistent, numerical_counts, numerical_counts_detail = (
        _numerical_solution_counts_are_consistent(payload)
    )
    evaluated += 1
    numerical = categories["numerical_solution"]
    numerical["passed" if numerical_counts_consistent else "failed"] += 1
    if not numerical_counts_consistent:
        failed += 1
        numerical["status"] = "fail"
    numerical["checks"].append(
        {
            "path": "numerical_solution.count_share_consistency",
            "status": "pass" if numerical_counts_consistent else "fail",
            "observed": numerical_counts,
            "missing": any(value is None for value in numerical_counts.values()),
            "operator": "consistent_counts_and_shares",
            "threshold": "shares equal counts / parameter_draws",
            "detail": numerical_counts_detail,
            "rationale": (
                "A common denominator prevents failed or dropped parameter draws from "
                "being removed when solution coverage is reported."
            ),
        }
    )

    total = evaluated
    model_name = _lookup(payload, "metadata.model_name")
    computational_passed = failed == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "status": "pass" if computational_passed else "fail",
        "computational_evidence_gate_passed": computational_passed,
        "model_name": None if model_name is _MISSING else model_name,
        "experiment_scope": "reduced_experiment",
        "source": source,
        "input_sha256": input_sha256,
        "summary": {
            "total": total,
            "passed": total - failed,
            "failed": failed,
            "not_required": int(
                sum(category["not_required"] for category in categories.values())
            ),
        },
        "categories": categories,
        "interpretation": (
            "A pass means that the normalized computational evidence meets the declared "
            "thresholds for this reduced experiment using the full "
            "Gali_2015_chapter_3_nonlinear equation set. It does not by itself make a "
            "manuscript publication-ready, and it does not validate a full-scale Galí "
            "estimation, HLT, SW07, or another model."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the JSON report."""

    status = str(report["status"]).upper()
    summary = report["summary"]
    lines = [
        "# Reduced Full-Equation Nonlinear Galí Computational Evidence Gate",
        "",
        f"**Mode:** `{report['mode']}`  ",
        f"**Status:** **{status}**  ",
        f"**Checks:** {summary['passed']}/{summary['total']} passed",
        "",
        str(report["interpretation"]),
        "",
        "| Category | Status | Passed | Failed | Not required |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, category in report["categories"].items():
        lines.append(
            f"| `{name}` | {str(category['status']).upper()} | "
            f"{category['passed']} | {category['failed']} | {category['not_required']} |"
        )
    failures = [
        check
        for category in report["categories"].values()
        for check in category["checks"]
        if check["status"] == "fail"
    ]
    if failures:
        lines.extend(["", "## Failed checks", ""])
        for check in failures:
            lines.append(
                f"- `{check['path']}`: observed `{check['observed']}`; "
                f"{check['detail']}."
            )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Normalized benchmark evidence JSON.")
    parser.add_argument("--mode", choices=MODES, default="publication")
    parser.add_argument("--output", type=Path, help="JSON report path; defaults beside input.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError("The evidence JSON root must be an object.")
    report = evaluate_payload(
        payload,
        mode=args.mode,
        source=str(args.input),
        input_sha256=hashlib.sha256(raw).hexdigest(),
    )
    output = args.output or args.input.with_name(
        f"{args.input.stem}.{args.mode}.readiness.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"{report['status'].upper()} mode={args.mode} "
        f"checks={report['summary']['passed']}/{report['summary']['total']} "
        f"report={output}"
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
