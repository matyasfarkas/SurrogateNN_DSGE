from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results" / "hlt_pipeline_parity"
DEFAULT_JULIA_FILE = "julia_pipeline_artifacts.json"
DEFAULT_PYTHON_FILE = "python_pipeline_artifacts.json"

StageKind = Literal["float_array", "float_scalar", "bool_array", "int_array"]


class PipelineParityPayloadError(AssertionError):
    """Raised when an HLT pipeline parity artifact is missing required data."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: StageKind
    aliases: tuple[tuple[str, ...], ...]
    atol: float
    rtol: float
    required: bool = True


@dataclass(frozen=True)
class PipelineParityStageCheck:
    case: str
    stage: str
    kind: StageKind
    shape: tuple[int, ...]
    max_abs_diff: float
    rel_diff: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class PipelineParityCaseCheck:
    case: str
    passed: bool
    stage_checks: tuple[PipelineParityStageCheck, ...]


PIPELINE_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        "rom_state_path",
        "float_array",
        (
            ("artifacts", "rom", "states"),
            ("rom", "states"),
            ("rom_state_path",),
            ("stages", "rom_path", "result", "states"),
            ("stages", "kalman_paths", "result", "filtered_variables"),
        ),
        atol=1.0e-8,
        rtol=1.0e-7,
    ),
    StageSpec(
        "rom_observation_path",
        "float_array",
        (
            ("artifacts", "rom", "observations"),
            ("rom", "observations"),
            ("rom_observation_path",),
            ("stages", "gate_stats", "result", "linear_observations"),
        ),
        atol=1.0e-8,
        rtol=1.0e-7,
    ),
    StageSpec(
        "rom_shock_path",
        "float_array",
        (
            ("artifacts", "rom", "shocks"),
            ("rom", "shocks"),
            ("rom_shock_path",),
            ("stages", "gate_stats", "result", "shocks"),
            ("stages", "kalman_paths", "result", "filtered_shocks"),
        ),
        atol=1.0e-8,
        rtol=1.0e-7,
    ),
    StageSpec(
        "gate_e_stat",
        "float_array",
        (
            ("artifacts", "gate", "e_stat"),
            ("gate", "e_stat"),
            ("gate_e_stat",),
            ("stages", "gate_stats", "result", "e_stat"),
        ),
        atol=4.0e-5,
        rtol=2.0e-5,
    ),
    StageSpec(
        "gate_f_stat",
        "float_array",
        (
            ("artifacts", "gate", "f_stat"),
            ("gate", "f_stat"),
            ("gate_f_stat",),
            ("stages", "gate_stats", "result", "f_stat"),
        ),
        atol=4.0e-5,
        rtol=2.0e-5,
    ),
    StageSpec(
        "gate_probs",
        "float_array",
        (
            ("artifacts", "gate", "probs"),
            ("artifacts", "gate", "gate_probs"),
            ("gate", "probs"),
            ("gate", "gate_probs"),
            ("gate_probs",),
            ("shared_gate_probs",),
        ),
        atol=4.0e-5,
        rtol=2.0e-5,
    ),
    StageSpec(
        "gate_mask",
        "bool_array",
        (
            ("artifacts", "gate", "mask"),
            ("artifacts", "gate", "hard_mask"),
            ("gate", "mask"),
            ("gate", "hard_mask"),
            ("gate_mask",),
            ("hard_mask",),
        ),
        atol=0.0,
        rtol=0.0,
    ),
    StageSpec(
        "support_features",
        "float_array",
        (
            ("artifacts", "support", "features"),
            ("support", "features"),
            ("support_features",),
            ("X",),
        ),
        atol=1.0e-9,
        rtol=1.0e-8,
    ),
    StageSpec(
        "support_selected_indices",
        "int_array",
        (
            ("artifacts", "support", "selected_indices"),
            ("support", "selected_indices"),
            ("selected_indices",),
            ("sample_idx",),
        ),
        atol=0.0,
        rtol=0.0,
    ),
    StageSpec(
        "sep_targets",
        "float_array",
        (
            ("artifacts", "sep", "targets"),
            ("sep", "targets"),
            ("sep_targets",),
            ("Y",),
        ),
        atol=1.0e-7,
        rtol=1.0e-6,
    ),
    StageSpec(
        "rom_targets",
        "float_array",
        (
            ("artifacts", "sep", "rom_targets"),
            ("sep", "rom_targets"),
            ("rom_targets",),
            ("Y_rom1",),
            ("Y_rom",),
        ),
        atol=1.0e-8,
        rtol=1.0e-7,
    ),
    StageSpec(
        "residual_labels",
        "float_array",
        (
            ("artifacts", "sep", "residual_labels"),
            ("sep", "residual_labels"),
            ("residual_labels",),
            ("sep_residual_labels",),
        ),
        atol=1.0e-7,
        rtol=1.0e-6,
    ),
    StageSpec(
        "surrogate_predictions",
        "float_array",
        (
            ("artifacts", "surrogate", "predictions"),
            ("surrogate", "predictions"),
            ("surrogate_predictions",),
            ("R_pred",),
        ),
        atol=1.0e-7,
        rtol=1.0e-6,
    ),
    StageSpec(
        "switching_loglik_per_period",
        "float_array",
        (
            ("artifacts", "likelihood", "switching_per_period"),
            ("artifacts", "likelihood", "per_period"),
            ("likelihood", "switching_per_period"),
            ("likelihood", "per_period"),
            ("switching_loglik_per_period",),
            ("ll_switching",),
        ),
        atol=1.0e-5,
        rtol=2.0e-5,
    ),
    StageSpec(
        "posterior_log_density",
        "float_scalar",
        (
            ("artifacts", "posterior", "log_density"),
            ("posterior", "log_density"),
            ("posterior_log_density",),
            ("log_density",),
        ),
        atol=1.0e-5,
        rtol=2.0e-5,
    ),
)
PIPELINE_STAGE_SPEC_BY_NAME = {spec.name: spec for spec in PIPELINE_STAGE_SPECS}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineParityPayloadError(f"Missing required artifact file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PipelineParityPayloadError(f"{path.name}: root must be a JSON object.")
    return payload


def _path_text(parts: Iterable[str]) -> str:
    return ".".join(str(part) for part in parts)


def _lookup(payload: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(_path_text(path))
    return current


def _case_names(payload: dict[str, Any]) -> tuple[str, ...]:
    cases = payload.get("cases")
    if isinstance(cases, dict):
        return tuple(str(name) for name in cases)
    if isinstance(cases, list):
        names: list[str] = []
        for idx, case in enumerate(cases):
            if not isinstance(case, dict) or "name" not in case:
                raise PipelineParityPayloadError(f"cases[{idx}] must be an object with a name.")
            names.append(str(case["name"]))
        return tuple(names)
    if "name" in payload:
        return (str(payload["name"]),)
    raise PipelineParityPayloadError("Artifact payload must contain `cases` or a top-level `name`.")


def _case_payload(payload: dict[str, Any], case: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if isinstance(cases, dict):
        if case not in cases:
            raise PipelineParityPayloadError(f"Missing case `{case}`.")
        case_payload = cases[case]
    elif isinstance(cases, list):
        matched = [entry for entry in cases if isinstance(entry, dict) and str(entry.get("name")) == case]
        if not matched:
            raise PipelineParityPayloadError(f"Missing case `{case}`.")
        case_payload = matched[0]
    else:
        case_payload = payload if str(payload.get("name")) == case else None
    if not isinstance(case_payload, dict):
        raise PipelineParityPayloadError(f"Case `{case}` must be a JSON object.")
    return case_payload


def _extract_stage(case_payload: dict[str, Any], spec: StageSpec, *, source: str, case: str) -> Any:
    for path in spec.aliases:
        try:
            return _lookup(case_payload, path)
        except KeyError:
            continue
    aliases = ", ".join(_path_text(path) for path in spec.aliases)
    if spec.required:
        raise PipelineParityPayloadError(
            f"{source}: case `{case}` is missing required stage `{spec.name}`. "
            f"Accepted paths: {aliases}."
        )
    return None


def _as_float_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise AssertionError(f"{label} must be non-empty.")
    if not np.isfinite(array).all():
        raise AssertionError(f"{label} must contain only finite values.")
    return array


def _as_bool_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise AssertionError(f"{label} must be non-empty.")
    return array.astype(bool, copy=False)


def _as_int_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise AssertionError(f"{label} must be non-empty.")
    int_array = np.asarray(array, dtype=np.int64)
    if not np.array_equal(array, int_array):
        raise AssertionError(f"{label} must contain integer values.")
    return int_array


def _as_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in {(), (1,)}:
        raise AssertionError(f"{label} must be scalar, got shape {array.shape}.")
    scalar = float(array.reshape(()))
    if not np.isfinite(scalar):
        raise AssertionError(f"{label} must be finite.")
    return scalar


def _max_abs_scale(array: np.ndarray) -> float:
    if array.size == 0:
        return 1.0
    return max(float(np.max(np.abs(array.astype(np.float64)))), 1.0)


def _compare_exact(
    *,
    case: str,
    spec: StageSpec,
    python: np.ndarray,
    julia: np.ndarray,
) -> PipelineParityStageCheck:
    if python.shape != julia.shape:
        raise AssertionError(
            f"{case}: {spec.name} shape mismatch: Python {python.shape} vs Julia {julia.shape}."
        )
    mismatches = int(np.count_nonzero(python != julia))
    if mismatches:
        raise AssertionError(f"{case}: {spec.name} has {mismatches} exact mismatches.")
    return PipelineParityStageCheck(
        case=case,
        stage=spec.name,
        kind=spec.kind,
        shape=tuple(int(x) for x in python.shape),
        max_abs_diff=0.0,
        rel_diff=0.0,
        tolerance=0.0,
        passed=True,
    )


def _compare_float_arrays(
    *,
    case: str,
    spec: StageSpec,
    python: np.ndarray,
    julia: np.ndarray,
) -> PipelineParityStageCheck:
    if python.shape != julia.shape:
        raise AssertionError(
            f"{case}: {spec.name} shape mismatch: Python {python.shape} vs Julia {julia.shape}."
        )
    diff = np.abs(python - julia)
    max_abs = float(np.max(diff))
    scale = _max_abs_scale(julia)
    tolerance = float(spec.atol + spec.rtol * scale)
    rel_diff = max_abs / scale
    if max_abs > tolerance:
        raise AssertionError(
            f"{case}: {spec.name} max abs diff {max_abs:.6g} exceeds {tolerance:.6g}."
        )
    return PipelineParityStageCheck(
        case=case,
        stage=spec.name,
        kind=spec.kind,
        shape=tuple(int(x) for x in python.shape),
        max_abs_diff=max_abs,
        rel_diff=float(rel_diff),
        tolerance=tolerance,
        passed=True,
    )


def _compare_float_scalars(
    *,
    case: str,
    spec: StageSpec,
    python: float,
    julia: float,
) -> PipelineParityStageCheck:
    max_abs = abs(float(python) - float(julia))
    scale = max(abs(float(julia)), 1.0)
    tolerance = float(spec.atol + spec.rtol * scale)
    rel_diff = max_abs / scale
    if max_abs > tolerance:
        raise AssertionError(
            f"{case}: {spec.name} abs diff {max_abs:.6g} exceeds {tolerance:.6g} "
            f"(Julia={julia:.17g}, Python={python:.17g})."
        )
    return PipelineParityStageCheck(
        case=case,
        stage=spec.name,
        kind=spec.kind,
        shape=(),
        max_abs_diff=float(max_abs),
        rel_diff=float(rel_diff),
        tolerance=tolerance,
        passed=True,
    )


def _compare_stage(
    *,
    case: str,
    spec: StageSpec,
    python_value: Any,
    julia_value: Any,
) -> PipelineParityStageCheck:
    label = f"{case}.{spec.name}"
    if spec.kind == "float_array":
        return _compare_float_arrays(
            case=case,
            spec=spec,
            python=_as_float_array(python_value, label=f"Python {label}"),
            julia=_as_float_array(julia_value, label=f"Julia {label}"),
        )
    if spec.kind == "float_scalar":
        return _compare_float_scalars(
            case=case,
            spec=spec,
            python=_as_scalar(python_value, label=f"Python {label}"),
            julia=_as_scalar(julia_value, label=f"Julia {label}"),
        )
    if spec.kind == "bool_array":
        return _compare_exact(
            case=case,
            spec=spec,
            python=_as_bool_array(python_value, label=f"Python {label}"),
            julia=_as_bool_array(julia_value, label=f"Julia {label}"),
        )
    if spec.kind == "int_array":
        return _compare_exact(
            case=case,
            spec=spec,
            python=_as_int_array(python_value, label=f"Python {label}"),
            julia=_as_int_array(julia_value, label=f"Julia {label}"),
        )
    raise AssertionError(f"Unsupported stage kind {spec.kind!r}.")


def validate_hlt_pipeline_parity(
    *,
    julia_payload: dict[str, Any],
    python_payload: dict[str, Any],
    cases: Sequence[str] | None = None,
    stage_specs: Sequence[StageSpec] = PIPELINE_STAGE_SPECS,
) -> tuple[PipelineParityCaseCheck, ...]:
    julia_cases = set(_case_names(julia_payload))
    python_cases = set(_case_names(python_payload))
    if cases is None:
        selected = tuple(sorted(julia_cases & python_cases))
    else:
        selected = tuple(str(case) for case in cases)
    if not selected:
        raise PipelineParityPayloadError("No shared cases to validate.")
    missing_julia = sorted(set(selected) - julia_cases)
    missing_python = sorted(set(selected) - python_cases)
    if missing_julia or missing_python:
        raise PipelineParityPayloadError(
            f"Missing cases. Julia missing={missing_julia}, Python missing={missing_python}."
        )

    case_checks: list[PipelineParityCaseCheck] = []
    for case in selected:
        julia_case = _case_payload(julia_payload, case)
        python_case = _case_payload(python_payload, case)
        stage_checks: list[PipelineParityStageCheck] = []
        for spec in stage_specs:
            julia_value = _extract_stage(julia_case, spec, source="Julia", case=case)
            python_value = _extract_stage(python_case, spec, source="Python", case=case)
            if julia_value is None or python_value is None:
                continue
            stage_checks.append(
                _compare_stage(
                    case=case,
                    spec=spec,
                    python_value=python_value,
                    julia_value=julia_value,
                )
            )
        case_checks.append(
            PipelineParityCaseCheck(
                case=case,
                passed=all(check.passed for check in stage_checks),
                stage_checks=tuple(stage_checks),
            )
        )
    return tuple(case_checks)


def validate_hlt_pipeline_parity_files(
    *,
    julia_file: Path,
    python_file: Path,
    cases: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
) -> tuple[PipelineParityCaseCheck, ...]:
    stage_specs = PIPELINE_STAGE_SPECS
    if stages is not None:
        unknown = sorted(set(stages) - set(PIPELINE_STAGE_SPEC_BY_NAME))
        if unknown:
            raise PipelineParityPayloadError(f"Unknown pipeline parity stages: {unknown}.")
        stage_specs = tuple(PIPELINE_STAGE_SPEC_BY_NAME[str(stage)] for stage in stages)
    return validate_hlt_pipeline_parity(
        julia_payload=_load_json(julia_file),
        python_payload=_load_json(python_file),
        cases=cases,
        stage_specs=stage_specs,
    )


def checks_to_jsonable(checks: Sequence[PipelineParityCaseCheck]) -> list[dict[str, Any]]:
    return [
        {
            "case": check.case,
            "passed": check.passed,
            "stage_checks": [asdict(stage) for stage in check.stage_checks],
        }
        for check in checks
    ]


def checks_to_markdown(checks: Sequence[PipelineParityCaseCheck]) -> str:
    lines = [
        "# HLT Pipeline Parity Report",
        "",
        "| Case | Stage | Shape | Max abs diff | Rel diff | Tolerance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for check in checks:
        for stage in check.stage_checks:
            shape = "scalar" if not stage.shape else "x".join(str(x) for x in stage.shape)
            lines.append(
                f"| {check.case} | {stage.stage} | {shape} | "
                f"{stage.max_abs_diff:.6g} | {stage.rel_diff:.6g} | {stage.tolerance:.6g} |"
            )
    lines.append("")
    lines.append(
        "All listed stages passed the configured tolerance. Missing stages are not omitted silently: "
        "the validator raises before this report is written."
    )
    return "\n".join(lines) + "\n"


def _resolve_files(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.results_dir is not None:
        base = Path(args.results_dir)
        julia_file = base / str(args.julia_file)
        python_file = base / str(args.python_file)
    else:
        julia_file = Path(args.julia_file)
        python_file = Path(args.python_file)
    return julia_file, python_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Julia-vs-Python HLT full-pipeline artifacts: ROM path, gate, "
            "support selection, SEP labels, residual surrogate predictions, likelihood, "
            "and posterior log density."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--julia-file", default=DEFAULT_JULIA_FILE)
    parser.add_argument("--python-file", default=DEFAULT_PYTHON_FILE)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        help=(
            "Restrict validation to a named pipeline stage. May be repeated. "
            "By default every canonical stage is required."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    julia_file, python_file = _resolve_files(args)
    checks = validate_hlt_pipeline_parity_files(
        julia_file=julia_file,
        python_file=python_file,
        cases=tuple(args.cases) if args.cases else None,
        stages=tuple(args.stages) if args.stages else None,
    )
    payload = checks_to_jsonable(checks)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(checks_to_markdown(checks), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
