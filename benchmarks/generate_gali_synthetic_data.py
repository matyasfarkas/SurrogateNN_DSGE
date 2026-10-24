#!/usr/bin/env python3
"""Generate a frozen synthetic-data payload for the nonlinear Galí experiment.

The model equations already multiply standardized ``eps_*`` innovations by
``std_a``, ``std_nu``, and ``std_z``.  This generator therefore draws unit
normal structural innovations, advances the complete nonlinear model with
positive-order stochastic extended path, and adds a separately seeded
measurement error.  The emitted JSON is directly consumable through
``GALI_PAYLOAD=... benchmarks/run_nonlinear_gali_pilot.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogatenn_dsge import SEPConfig, parse_macro_model


SCHEMA_VERSION = "surrogatenn-dsge-gali-synthetic-data/v1"
EXPECTED_MODEL = "Gali_2015_chapter_3_nonlinear"
EXPECTED_SHOCK_CONVENTION = "standard_normal_innovations_scaled_in_model_equations"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "dirty_worktree": bool(status.strip()),
        "dirty_entries": [line for line in status.splitlines() if line.strip()],
    }


def _select_case(payload: Mapping[str, Any], case_name: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Template payload must contain a cases list.")
    for case in cases:
        if isinstance(case, Mapping) and str(case.get("name")) == case_name:
            return dict(case)
    raise ValueError(f"No template case named {case_name!r}.")


def _validate_template(model: Any, case: Mapping[str, Any]) -> None:
    if str(model.name) != EXPECTED_MODEL:
        raise ValueError(f"Expected model {EXPECTED_MODEL!r}, got {model.name!r}.")
    if str(case.get("model_symbol")) != EXPECTED_MODEL:
        raise ValueError("Template model_symbol does not match the parsed Galí model.")
    shock_names = tuple(str(name) for name in case.get("shock_names", ()))
    model_shocks = tuple(str(name) for name in model.timings.exo)
    if shock_names != model_shocks:
        raise ValueError(
            f"Template shock order does not match the model: {shock_names} vs {model_shocks}."
        )
    if str(case.get("shock_scale_convention", "")) != EXPECTED_SHOCK_CONVENTION:
        raise ValueError(
            "Template must declare standardized innovations scaled in the model equations."
        )
    innovation_sigmas = {
        str(name): float(value) for name, value in dict(case.get("shock_sigmas", {})).items()
    }
    if any(
        name not in innovation_sigmas
        or not math.isclose(innovation_sigmas[name], 1.0, rel_tol=0.0, abs_tol=0.0)
        for name in shock_names
    ):
        raise ValueError("Galí shock_sigmas must equal 1.0 for standardized innovations.")
    scale_parameters = {
        str(shock): str(parameter)
        for shock, parameter in dict(case.get("model_innovation_scale_parameters", {})).items()
    }
    if set(scale_parameters) != set(shock_names):
        raise ValueError("Template must map every innovation to its equation-level scale parameter.")
    missing = sorted(set(scale_parameters.values()) - set(model.parameter_names))
    if missing:
        raise ValueError("Unknown innovation-scale parameters: " + ", ".join(missing))


def generate_case(
    model: Any,
    template_case: Mapping[str, Any],
    *,
    case_name: str,
    periods: int,
    burn_in: int,
    shock_seed: int,
    measurement_seed: int,
    sep_config: SEPConfig,
    progress: bool = False,
) -> dict[str, Any]:
    """Return one deterministic, fully recorded synthetic-data case."""

    _validate_template(model, template_case)
    if periods < 1:
        raise ValueError("periods must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    observables = tuple(str(name) for name in template_case["observables"])
    observable_indices = tuple(model.timings.var.index(name) for name in observables)
    shock_names = tuple(str(name) for name in template_case["shock_names"])
    obs_sigma = np.asarray(
        [float(template_case["obs_sigma"][name]) for name in observables],
        dtype=np.float64,
    )
    if not np.isfinite(obs_sigma).all() or np.any(obs_sigma <= 0.0):
        raise ValueError("Measurement standard deviations must be finite and positive.")

    raw_parameters = np.asarray(model.parameter_values, dtype=np.float64)
    reference_guess = np.asarray(template_case["reference_steady_state"], dtype=np.float64)
    steady_result = model.solve_steady_state(
        parameter_values=raw_parameters,
        initial_guess=reference_guess,
        tol=1.0e-12,
        max_iter=200,
    )
    steady_state = np.asarray(steady_result.steady_state, dtype=np.float64)
    resolved_parameters = np.asarray(steady_result.parameter_values, dtype=np.float64)
    if not bool(steady_result.converged) or not np.isfinite(steady_state).all():
        raise RuntimeError(
            "Data-generating steady state did not converge: "
            f"residual={steady_result.residual_norm!r}."
        )

    total_periods = int(burn_in) + int(periods)
    shock_rng = np.random.Generator(np.random.PCG64(int(shock_seed)))
    measurement_rng = np.random.Generator(np.random.PCG64(int(measurement_seed)))
    structural_shocks = shock_rng.standard_normal((len(shock_names), total_periods))
    measurement_errors = measurement_rng.normal(
        loc=0.0,
        scale=obs_sigma[:, None],
        size=(len(observables), periods),
    )

    state = steady_state.copy()
    state_path = [state.copy()]
    latent_observations = np.empty((len(observables), periods), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    for period_index in range(total_periods):
        deterministic = np.zeros(
            (int(sep_config.periods), int(model.timings.nExo)),
            dtype=np.float64,
        )
        deterministic[0, :] = structural_shocks[:, period_index]
        result = model.solve_stochastic_extended_path(
            parameter_values=resolved_parameters,
            steady_state=steady_state,
            initial_state=state,
            terminal_state=steady_state,
            deterministic_shocks=deterministic,
            config=sep_config,
        )
        solution = result.solution
        next_state = np.asarray(solution.mean_path, dtype=np.float64)[:, 1]
        residual = float(solution.residual_norm)
        accepted = bool(solution.accepted)
        if not accepted or not math.isfinite(residual) or not np.isfinite(next_state).all():
            raise RuntimeError(
                "Synthetic-data SEP failed at absolute period "
                f"{period_index}: accepted={accepted}, residual={residual!r}."
            )
        state = next_state
        state_path.append(state.copy())
        diagnostics.append(
            {
                "absolute_period": int(period_index),
                "retained_period": (
                    None if period_index < burn_in else int(period_index - burn_in)
                ),
                "accepted": accepted,
                "converged": bool(solution.converged),
                "iterations": int(solution.iterations),
                "residual_norm": residual,
            }
        )
        if period_index >= burn_in:
            latent_observations[:, period_index - burn_in] = state[list(observable_indices)]
        if progress:
            print(
                "[dgp] "
                f"period={period_index + 1}/{total_periods} "
                f"retained={max(0, period_index + 1 - burn_in)}/{periods} "
                f"residual={residual:.3e} iterations={int(solution.iterations)}",
                flush=True,
            )

    observations = latent_observations + measurement_errors
    arrays = {
        "structural_innovations": structural_shocks.tolist(),
        "measurement_errors": measurement_errors.tolist(),
        "latent_observations": latent_observations.tolist(),
        "latent_state_path": np.column_stack(state_path).tolist(),
        "observations": observations.tolist(),
    }
    if len(model.parameter_names) != resolved_parameters.size:
        raise ValueError("Resolved parameter vector does not match model.parameter_names.")
    parameter_lookup = {
        str(name): float(value)
        for name, value in zip(model.parameter_names, resolved_parameters)
    }
    raw_parameter_lookup = {
        str(name): float(value)
        for name, value in zip(model.parameter_names, raw_parameters)
    }
    output_case = dict(template_case)
    output_case.update(
        {
            "name": str(case_name),
            "observations": observations.tolist(),
            "reference_steady_state": steady_state.tolist(),
            "dgp": {
                "schema_version": SCHEMA_VERSION,
                "model_name": EXPECTED_MODEL,
                "periods": int(periods),
                "burn_in": int(burn_in),
                "initial_condition": "solved_deterministic_steady_state",
                "initial_state": steady_state.tolist(),
                "shock_seed": int(shock_seed),
                "measurement_seed": int(measurement_seed),
                "random_generator": "numpy.random.PCG64",
                "parameter_values_raw": raw_parameter_lookup,
                "parameter_values_resolved": parameter_lookup,
                "steady_state": {
                    "converged": bool(steady_result.converged),
                    "iterations": int(steady_result.iterations),
                    "residual_norm": float(steady_result.residual_norm),
                },
                "sep": {
                    "periods": int(sep_config.periods),
                    "branching_order": int(sep_config.branching_order),
                    "nnodes": int(sep_config.nnodes),
                    "shock_scale": float(sep_config.shock_scale),
                    "sparse_tree": bool(sep_config.sparse_tree),
                    "max_iter": int(sep_config.max_iter),
                    "tol": float(sep_config.tol),
                    "accept_tol": (
                        None if sep_config.accept_tol is None else float(sep_config.accept_tol)
                    ),
                    "linear_solver": str(sep_config.linear_solver),
                    "line_search": bool(sep_config.line_search),
                },
                "solver_diagnostics": diagnostics,
                "arrays": arrays,
                "arrays_sha256": _canonical_sha256(arrays),
            },
        }
    )
    return output_case


def build_payload(
    *,
    model_source: Path,
    template_payload: Path,
    template_case_name: str,
    output_case_name: str,
    periods: int,
    burn_in: int,
    shock_seed: int,
    measurement_seed: int,
    sep_config: SEPConfig,
    progress: bool = False,
) -> dict[str, Any]:
    model_source_bytes = model_source.read_bytes()
    template_bytes = template_payload.read_bytes()
    model = parse_macro_model(model_source_bytes.decode("utf-8"))
    template_root = json.loads(template_bytes)
    template_case = _select_case(template_root, template_case_name)
    case = generate_case(
        model,
        template_case,
        case_name=output_case_name,
        periods=periods,
        burn_in=burn_in,
        shock_seed=shock_seed,
        measurement_seed=measurement_seed,
        sep_config=sep_config,
        progress=progress,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_hashes": {
            "generator_sha256": _sha256_file(Path(__file__).resolve()),
            "model_source_sha256": _sha256_bytes(model_source_bytes),
            "template_payload_sha256": _sha256_bytes(template_bytes),
        },
        "cases": [case],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-source",
        type=Path,
        default=ROOT / "benchmarks" / "model_sources" / "Gali_2015_chapter_3_nonlinear.jl",
    )
    parser.add_argument(
        "--template-payload",
        type=Path,
        default=ROOT / "benchmarks" / "gali_nonlinear_pilot_payload.json",
    )
    parser.add_argument("--template-case-name", default="gali_full_nonlinear_reduced_pilot")
    parser.add_argument("--output-case-name", default="gali_full_nonlinear_confirmatory_v1")
    parser.add_argument("--periods", type=int, default=100)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--shock-seed", type=int, default=20261101)
    parser.add_argument("--measurement-seed", type=int, default=20261102)
    parser.add_argument("--sep-periods", type=int, default=10)
    parser.add_argument("--sep-order", type=int, default=1)
    parser.add_argument("--sep-nnodes", type=int, default=3)
    parser.add_argument("--sep-shock-scale", type=float, default=1.0)
    parser.add_argument("--sep-sparse-tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sep-max-iter", type=int, default=50)
    parser.add_argument("--sep-tol", type=float, default=1.0e-8)
    parser.add_argument("--sep-accept-tol", type=float, default=1.0e-5)
    parser.add_argument("--sep-linear-solver", choices=("qr", "normal_equations"), default="qr")
    parser.add_argument("--sep-line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a debugging DGP from a dirty worktree; never use for paper evidence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    git_metadata = _git_metadata(ROOT)
    if git_metadata["dirty_worktree"] and not bool(args.allow_dirty):
        raise RuntimeError(
            "Refusing to generate confirmatory data from a dirty worktree. "
            "Commit the protocol first or use --allow-dirty for non-evidence debugging."
        )
    config = SEPConfig(
        periods=int(args.sep_periods),
        branching_order=int(args.sep_order),
        nnodes=int(args.sep_nnodes),
        shock_scale=float(args.sep_shock_scale),
        sparse_tree=bool(args.sep_sparse_tree),
        max_iter=int(args.sep_max_iter),
        tol=float(args.sep_tol),
        accept_tol=float(args.sep_accept_tol),
        linear_solver=str(args.sep_linear_solver),
        line_search=bool(args.sep_line_search),
    )
    payload = build_payload(
        model_source=args.model_source.resolve(),
        template_payload=args.template_payload.resolve(),
        template_case_name=str(args.template_case_name),
        output_case_name=str(args.output_case_name),
        periods=int(args.periods),
        burn_in=int(args.burn_in),
        shock_seed=int(args.shock_seed),
        measurement_seed=int(args.measurement_seed),
        sep_config=config,
        progress=True,
    )
    payload["reproducibility"] = {
        **git_metadata,
        "command_argv": [str(Path(__file__).resolve()), *list(sys.argv[1:] if argv is None else argv)],
        "numpy_version": str(np.__version__),
        "float64": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"WROTE path={args.output} sha256={_sha256_bytes(rendered.encode('utf-8'))} "
        f"case={args.output_case_name} periods={args.periods} burn_in={args.burn_in}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
