from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "20260712T083357"
DEFAULT_CASES = ("medium_sw07_hlt", "small_fs2000")

SWITCHING_STAGE = "switching_value"
GATE_STAGE = "gate_stats"
SEP_STAGE = "sep_inversion"
GATE_ARRAY_FIELDS = ("e_stat", "f_stat", "linear_observations", "shocks")


class PayloadFieldError(AssertionError):
    """Raised when a stored Julia/Python parity payload is missing required data."""


@dataclass(frozen=True)
class NonlinearLikelihoodParityCheck:
    case: str
    switching_abs_diff: float
    switching_rel_diff: float
    gate_max_abs_diff: float
    sep_gap: str | None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PayloadFieldError(f"Missing required payload file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PayloadFieldError(f"{path.name}: expected a JSON object at the root.")
    return payload


def _format_path(parts: Iterable[object]) -> str:
    return ".".join(str(part) for part in parts)


def _require(payload: dict[str, Any], parts: tuple[object, ...], *, source: str) -> Any:
    current: Any = payload
    traversed: list[object] = []
    for part in parts:
        traversed.append(part)
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise PayloadFieldError(
            f"{source}: missing required field `{_format_path(traversed)}`."
        )
    return current


def _require_status_ok(
    payload: dict[str, Any],
    source: str,
    case: str,
    stage: str,
) -> None:
    status = _require(payload, ("cases", case, "stages", stage, "status"), source=source)
    if status != "ok":
        raise AssertionError(f"{source}: `{case}.{stage}` status is {status!r}, expected 'ok'.")


def _require_finite_float(
    payload: dict[str, Any],
    parts: tuple[object, ...],
    *,
    source: str,
) -> float:
    value = _require(payload, parts, source=source)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise AssertionError(
            f"{source}: `{_format_path(parts)}` must be a finite scalar, got {value!r}."
        )
    return float(value)


def _require_float_array(
    payload: dict[str, Any],
    parts: tuple[object, ...],
    *,
    source: str,
) -> np.ndarray:
    value = _require(payload, parts, source=source)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{source}: `{_format_path(parts)}` is not numeric.") from exc
    if array.size == 0 or not np.isfinite(array).all():
        raise AssertionError(
            f"{source}: `{_format_path(parts)}` must be non-empty and finite."
        )
    return array


def _assert_close(
    *,
    case: str,
    label: str,
    python_value: float,
    julia_value: float,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    abs_diff = abs(python_value - julia_value)
    rel_diff = abs_diff / max(abs(julia_value), 1.0)
    tolerance = atol + rtol * abs(julia_value)
    if abs_diff > tolerance:
        raise AssertionError(
            f"{case}: {label} abs diff {abs_diff:.6g} exceeds {tolerance:.6g} "
            f"(Julia={julia_value:.17g}, Python={python_value:.17g})."
        )
    return abs_diff, rel_diff


def _sep_gap(julia_value: Any, python_value: Any) -> str | None:
    if julia_value is None:
        return "Julia SEP inversion value is null in the stored fixture."
    if python_value is None:
        return "Python SEP inversion value is null in the stored fixture."
    if isinstance(python_value, (int, float)) and float(python_value) <= -1.0e11:
        return "Python SEP inversion stored the configured failure sentinel."
    if isinstance(julia_value, (int, float)) and float(julia_value) <= -1.0e11:
        return "Julia SEP inversion stored the configured failure sentinel."
    return None


def validate_stored_nonlinear_likelihood_parity(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    cases: tuple[str, ...] = DEFAULT_CASES,
    switching_atol: float = 1.0e-5,
    switching_rtol: float = 2.0e-5,
    gate_atol: float = 4.0e-5,
    gate_rtol: float = 2.0e-5,
) -> list[NonlinearLikelihoodParityCheck]:
    """Validate tracked Julia/Python nonlinear likelihood fixture payloads.

    The current tracked fixtures expose a nonlinear switching likelihood and
    gate diagnostic arrays. Direct SEP inversion is still stored as scalar
    stage output, so this validator requires that field to exist and records
    why it was not numerically compared when either side stored null/failure.
    """

    results_dir = Path(results_dir)
    julia_results = _load_json(results_dir / "julia_results.json")
    python_results = _load_json(results_dir / "python_results.json")
    checks: list[NonlinearLikelihoodParityCheck] = []

    for case in cases:
        for source, payload in (
            ("julia_results.json", julia_results),
            ("python_results.json", python_results),
        ):
            _require(payload, ("cases", case), source=source)
            _require_status_ok(payload, source, case, SWITCHING_STAGE)
            _require_status_ok(payload, source, case, GATE_STAGE)
            _require_status_ok(payload, source, case, SEP_STAGE)

        julia_switching = _require_finite_float(
            julia_results,
            ("cases", case, "stages", SWITCHING_STAGE, "result", "value"),
            source="julia_results.json",
        )
        python_switching = _require_finite_float(
            python_results,
            ("cases", case, "stages", SWITCHING_STAGE, "result", "value"),
            source="python_results.json",
        )
        switching_abs_diff, switching_rel_diff = _assert_close(
            case=case,
            label="nonlinear switching likelihood",
            python_value=python_switching,
            julia_value=julia_switching,
            atol=switching_atol,
            rtol=switching_rtol,
        )

        gate_max_abs_diff = 0.0
        for field in GATE_ARRAY_FIELDS:
            parts = ("cases", case, "stages", GATE_STAGE, "result", field)
            julia_array = _require_float_array(
                julia_results,
                parts,
                source="julia_results.json",
            )
            python_array = _require_float_array(
                python_results,
                parts,
                source="python_results.json",
            )
            if python_array.shape != julia_array.shape:
                raise AssertionError(
                    f"{case}: gate_stats.{field} shape mismatch "
                    f"(Julia={julia_array.shape}, Python={python_array.shape})."
                )
            diff = np.abs(python_array - julia_array)
            max_abs_diff = float(np.max(diff))
            gate_max_abs_diff = max(gate_max_abs_diff, max_abs_diff)
            tolerance = gate_atol + gate_rtol * max(
                float(np.max(np.abs(julia_array))),
                1.0,
            )
            if max_abs_diff > tolerance:
                raise AssertionError(
                    f"{case}: gate_stats.{field} max abs diff {max_abs_diff:.6g} "
                    f"exceeds {tolerance:.6g}."
                )

        sep_parts = ("cases", case, "stages", SEP_STAGE, "result", "value")
        julia_sep_value = _require(julia_results, sep_parts, source="julia_results.json")
        python_sep_value = _require(
            python_results,
            sep_parts,
            source="python_results.json",
        )
        checks.append(
            NonlinearLikelihoodParityCheck(
                case=case,
                switching_abs_diff=switching_abs_diff,
                switching_rel_diff=switching_rel_diff,
                gate_max_abs_diff=gate_max_abs_diff,
                sep_gap=_sep_gap(julia_sep_value, python_sep_value),
            )
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate stored Julia-vs-Python nonlinear likelihood parity payloads."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--switching-atol", type=float, default=1.0e-5)
    parser.add_argument("--switching-rtol", type=float, default=2.0e-5)
    parser.add_argument("--gate-atol", type=float, default=4.0e-5)
    parser.add_argument("--gate-rtol", type=float, default=2.0e-5)
    args = parser.parse_args()

    cases = tuple(args.cases) if args.cases else DEFAULT_CASES
    checks = validate_stored_nonlinear_likelihood_parity(
        args.results_dir,
        cases=cases,
        switching_atol=args.switching_atol,
        switching_rtol=args.switching_rtol,
        gate_atol=args.gate_atol,
        gate_rtol=args.gate_rtol,
    )
    print(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
