#!/usr/bin/env python3
"""Streaming large-draw HLT support audit for QZ/Cayley determinacy screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import jax
import surrogatenn_dsge as sdsge

import posterior_sampling_speed as posterior_speed
from qz_cayley_support_audit import (
    _audit_draw,
    _centered_uniform_intervals,
    _example_bucket_names,
    _jsonable,
    _load_hlt_case,
    _parameter_names,
)
from surrogatenn_dsge.parameters import get_parameter_specs, parameter_bounds_array


BOOLEAN_COUNT_KEYS = (
    "qz_unique",
    "real_qz_unique",
    "cayley_success",
    "cayley_realpart_accept",
    "cayley_relaxed_accept",
    "cayley_safe_accept",
    "doubling_success",
    "qz_real_qz_disagree",
    "real_qz_accepts_complex_qz_reject",
    "real_qz_rejects_complex_qz_unique",
    "real_qz_cayley_disagree",
    "real_qz_cayley_realpart_disagree",
    "real_qz_cayley_relaxed_disagree",
    "real_qz_cayley_safe_disagree",
    "qz_cayley_disagree",
    "qz_cayley_safe_disagree",
    "qz_doubling_disagree",
    "doubling_accepts_non_unique",
    "doubling_rejects_qz_unique",
    "doubling_accepts_real_qz_non_unique",
    "cayley_accepts_non_unique",
    "cayley_accepts_real_qz_non_unique",
    "cayley_realpart_accepts_real_qz_non_unique",
    "cayley_relaxed_accepts_real_qz_non_unique",
    "cayley_safe_accepts_non_unique",
    "cayley_safe_accepts_real_qz_non_unique",
    "cayley_rejects_qz_unique",
    "cayley_rejects_real_qz_unique",
    "cayley_realpart_rejects_real_qz_unique",
    "cayley_relaxed_rejects_real_qz_unique",
    "cayley_safe_rejects_qz_unique",
    "cayley_safe_rejects_real_qz_unique",
)

METRIC_KEYS = (
    "real_qz_relative_residual",
    "real_qz_invariant_subspace_rcond",
    "cayley_relative_residual",
    "cayley_realpart_relative_residual",
    "cayley_solution_imag_max_abs",
    "cayley_invariant_subspace_rcond",
    "cayley_spectral_margin",
    "max_abs_real_qz_cayley_solution_diff",
)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "p01": None, "median": None, "p99": None, "max": None}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "median": float(np.median(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _add_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, default=posterior_speed.DEFAULT_SW07_MODEL_SOURCE_PATH)
    parser.add_argument("--payload-path", type=Path, default=posterior_speed.DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--parameters", default="phase1_18params")
    parser.add_argument("--draw-mode", choices=("package-prior", "centered-uniform"), default="package-prior")
    parser.add_argument("--draws", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--prior-width-scale", type=float, default=0.01)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument("--qme-acceptance-tol", type=float, default=1.0e-8)
    parser.add_argument("--unit-root-tol", type=float, default=1.0e-6)
    parser.add_argument("--cayley-safe-min-spectral-margin", type=float, default=1.0e-6)
    parser.add_argument("--cayley-safe-min-rcond", type=float, default=2.0e-4)
    parser.add_argument("--cayley-safe-imag-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--max-examples-per-type", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "qz_real_cayley_hlt_loose_1m.json",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    case = _load_hlt_case(args.payload_path, args.case)
    model = sdsge.parse_macro_model(args.model_source.read_text())
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    parameter_names = _parameter_names(model, case, args.parameters)
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    base_values = np.asarray(model.parameter_values, dtype=np.float64)
    base_subset = np.asarray(
        [base_values[parameter_index[name]] for name in parameter_names],
        dtype=np.float64,
    )
    if args.draw_mode == "package-prior":
        specs = get_parameter_specs(args.parameters)
        expected_names = tuple(spec.name for spec in specs)
        if expected_names != tuple(parameter_names):
            raise ValueError(
                "Package-prior parameter names do not match selected model parameters: "
                f"{expected_names} vs {tuple(parameter_names)}."
            )
        lower, upper = parameter_bounds_array(specs)
        support_label = args.parameters
    else:
        lower, upper = _centered_uniform_intervals(
            parameter_names,
            base_subset,
            width_scale=float(args.prior_width_scale),
            width_floor=float(args.prior_width_floor),
        )
        support_label = (
            f"centered_uniform(scale={float(args.prior_width_scale)}, "
            f"floor={float(args.prior_width_floor)})"
        )

    n_total = int(args.draws)
    n_random = n_total - (1 if bool(args.include_reference) else 0)
    if n_random < 0:
        raise ValueError("--draws must be at least 1 when --include-reference is set.")
    chunk_size = max(1, int(args.chunk_size))
    rng = np.random.default_rng(int(args.seed))
    print(
        f"starting audit draws={n_total} parameters={args.parameters} support={support_label} "
        f"backend={jax.default_backend()} chunk_size={chunk_size}",
        flush=True,
    )
    bool_counts = {key: 0 for key in BOOLEAN_COUNT_KEYS}
    qz_counts: dict[str, int] = {}
    real_qz_counts: dict[str, int] = {}
    metric_values = {key: np.full((n_total,), np.nan, dtype=np.float64) for key in METRIC_KEYS}
    examples: list[dict[str, Any]] = []
    examples_by_type: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    audited = 0
    example_cap = int(args.max_examples_per_type)

    emitted_reference = False
    random_done = 0
    draw_idx = 0
    chunk_index = 0
    while draw_idx < n_total:
        chunk_columns: list[np.ndarray] = []
        if bool(args.include_reference) and not emitted_reference:
            chunk_columns.append(base_subset[:, None])
            emitted_reference = True
        take_random = min(chunk_size - sum(chunk.shape[1] for chunk in chunk_columns), n_random - random_done)
        if take_random > 0:
            if args.draw_mode == "package-prior":
                design = sdsge.sample_prior_parameters(
                    args.parameters,
                    int(take_random),
                    seed=int(args.seed) + chunk_index,
                )
                theta_random = np.asarray(design.theta, dtype=np.float64)
            else:
                unit = rng.random((len(parameter_names), int(take_random)))
                theta_random = lower[:, None] + (upper - lower)[:, None] * unit
            chunk_columns.append(theta_random)
            random_done += int(take_random)
        if not chunk_columns:
            break
        theta_chunk = np.concatenate(chunk_columns, axis=1)
        chunk_index += 1
        for local_idx in range(theta_chunk.shape[1]):
            theta_column = theta_chunk[:, local_idx]
            current_draw_idx = draw_idx
            draw_idx += 1
            if draw_idx > n_total:
                break
            try:
                row = _audit_draw(
                    model=model,
                    steady_state=steady_state,
                    parameter_names=parameter_names,
                    theta_column=theta_column,
                    qme_acceptance_tol=float(args.qme_acceptance_tol),
                    unit_root_tol=float(args.unit_root_tol),
                    cayley_safe_min_spectral_margin=float(args.cayley_safe_min_spectral_margin),
                    cayley_safe_min_rcond=float(args.cayley_safe_min_rcond),
                    cayley_safe_imag_tol=float(args.cayley_safe_imag_tol),
                )
            except Exception as exc:
                if len(errors) < int(args.max_examples):
                    errors.append(
                        {
                            "draw_index": current_draw_idx,
                            "error": f"{type(exc).__name__}: {exc}",
                            "theta": {
                                name: float(value)
                                for name, value in zip(parameter_names, theta_column)
                            },
                        }
                    )
                continue

            audited += 1
            qz_cls = str(row["qz_classification"])
            real_qz_cls = str(row["real_qz_classification"])
            row["qz_cayley_safe_disagree"] = bool(row["qz_unique"]) != bool(
                row["cayley_safe_accept"]
            )
            _add_count(qz_counts, qz_cls)
            _add_count(real_qz_counts, real_qz_cls)
            for key in BOOLEAN_COUNT_KEYS:
                bool_counts[key] += int(bool(row[key]))
            for key in METRIC_KEYS:
                metric_values[key][audited - 1] = float(row[key])

            bucket_names = _example_bucket_names(row)
            if bucket_names:
                row_with_context = dict(row)
                row_with_context["draw_index"] = current_draw_idx
                row_with_context["theta"] = {
                    name: float(value) for name, value in zip(parameter_names, theta_column)
                }
                for bucket_name in bucket_names:
                    bucket = examples_by_type.setdefault(bucket_name, [])
                    if len(bucket) < example_cap:
                        bucket.append(row_with_context)
                if len(examples) < int(args.max_examples):
                    examples.append(row_with_context)

            if args.progress_every > 0 and draw_idx % int(args.progress_every) == 0:
                elapsed = time.perf_counter() - started
                rate = draw_idx / elapsed if elapsed > 0 else float("nan")
                remaining = (n_total - draw_idx) / rate if rate > 0 else float("nan")
                print(
                    f"progress {draw_idx}/{n_total} elapsed_s={elapsed:.1f} "
                    f"rate_draws_s={rate:.2f} eta_s={remaining:.1f}",
                    flush=True,
                )

    summaries = {
        f"{key}_summary": _summary(values[:audited])
        for key, values in metric_values.items()
    }
    payload: dict[str, Any] = {
        "status": "ok" if not errors else "partial",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "elapsed_s": float(time.perf_counter() - started),
        "case": args.case,
        "model_source": args.model_source,
        "payload_path": args.payload_path,
        "parameters": tuple(parameter_names),
        "support_label": support_label,
        "draw_mode": args.draw_mode,
        "qme_acceptance_tol": float(args.qme_acceptance_tol),
        "unit_root_tol": float(args.unit_root_tol),
        "cayley_safe_min_spectral_margin": float(args.cayley_safe_min_spectral_margin),
        "cayley_safe_min_rcond": float(args.cayley_safe_min_rcond),
        "cayley_safe_imag_tol": float(args.cayley_safe_imag_tol),
        "requested_draws": int(args.draws),
        "generated_draws": n_total,
        "audited_draws": audited,
        "error_count": n_total - audited,
        "qz_classification_counts": qz_counts,
        "real_qz_classification_counts": real_qz_counts,
        "lower": lower,
        "upper": upper,
        "examples": examples,
        "examples_by_type": examples_by_type,
        "errors": errors,
    }
    payload.update({f"{key}_count": int(value) for key, value in bool_counts.items()})
    payload.update(summaries)

    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
