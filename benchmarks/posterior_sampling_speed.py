from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_PATH = ROOT / "benchmarks" / "results" / "test_payloads.json"
DEFAULT_SW07_MODEL_SOURCE_PATH = ROOT / "benchmarks" / "model_sources" / "Smets_Wouters_2007_HLT.jl"
DEFAULT_SW07_PAPER_MODEL_SOURCE_PATH = ROOT / "benchmarks" / "model_sources" / "Smets_Wouters_2007_linear.jl"
DEFAULT_GALI3_MODEL_SOURCE_PATH = ROOT / "benchmarks" / "model_sources" / "Gali_3eq_linear.jl"
DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "results" / "posterior_sampling_speed.json"

TOY_AR2_SOURCE = """
@model posterior_speed_toy begin
    a[0] = rho_a * a[-1] + (1 - rho_a) * a_bar + eps_a[x]
    y[0] = rho_y * y[-1] + (1 - rho_y) * y_bar + alpha * (a[0] - a_bar) + eps_y[x]
end

@parameters posterior_speed_toy begin
    0 < rho_a < 1
    0 < rho_y < 1
    alpha = 0.4
    a_bar = 1.5
    y_bar = 2.0
    rho_a = 0.8
    rho_y = 0.6
end
"""

SW07_SAFE_15_PARAMETERS = (
    "calfa",
    "cg",
    "cgy",
    "cindw",
    "crdy",
    "crhob",
    "crhoqs",
    "crpi",
    "crr",
    "cry",
    "csigl",
    "z_ea",
    "z_eg",
    "z_em",
    "z_ew",
)
SW07_SAFE_27_PARAMETERS = (
    "calfa",
    "cg",
    "cgy",
    "cindw",
    "cmap",
    "cmaw",
    "constelab",
    "crdy",
    "crhoa",
    "crhob",
    "crhog",
    "crhoms",
    "crhopinf",
    "crhoqs",
    "crhow",
    "crpi",
    "crr",
    "cry",
    "csigl",
    "ctou",
    "z_ea",
    "z_eb",
    "z_eg",
    "z_em",
    "z_epinf",
    "z_eqs",
    "z_ew",
)

# Exact estimand/prior block used by the archived Dynare SW07 specification
# associated with Farkas and Tatar (2023).  Bounds are hard support restrictions;
# the distribution parameters are stated as means and standard deviations, as in
# Dynare's estimated_params block.  Keeping this specification executable avoids
# silently substituting the narrow centered uniforms used by the throughput-only
# SW07 benchmark.
SW07_PAPER_PRIOR_SPECS: tuple[tuple[str, float, float, float, str, float, float], ...] = (
    ("z_ea", 0.4618, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("z_eb", 0.1818513, 0.025, 5.0, "inverse_gamma", 0.1, 2.0),
    ("z_eg", 0.6090, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("z_eqs", 0.46017, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("z_em", 0.2397, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("z_epinf", 0.1455, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("z_ew", 0.2089, 0.01, 3.0, "inverse_gamma", 0.1, 2.0),
    ("crhoa", 0.9676, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhob", 0.2703, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhog", 0.9930, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhoqs", 0.5724, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhoms", 0.3, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhopinf", 0.8692, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("crhow", 0.9546, 0.001, 0.9999, "beta", 0.5, 0.20),
    ("cmap", 0.7652, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("cmaw", 0.8936, 0.01, 0.9999, "beta", 0.5, 0.20),
    ("csadjcost", 6.3325, 2.0, 15.0, "normal", 4.0, 1.5),
    ("csigma", 1.2312, 0.25, 3.0, "normal", 1.50, 0.375),
    ("chabb", 0.7205, 0.001, 0.99, "beta", 0.7, 0.1),
    ("cprobw", 0.7937, 0.3, 0.95, "beta", 0.5, 0.1),
    ("csigl", 2.8401, 0.25, 10.0, "normal", 2.0, 0.75),
    ("cprobp", 0.7813, 0.5, 0.95, "beta", 0.5, 0.10),
    ("cindw", 0.4425, 0.01, 0.99, "beta", 0.5, 0.15),
    ("cindp", 0.3291, 0.01, 0.99, "beta", 0.5, 0.15),
    ("czcap", 0.2648, 0.01, 1.0, "beta", 0.5, 0.15),
    ("cfc", 1.4672, 1.0, 3.0, "normal", 1.25, 0.125),
    ("crpi", 1.7985, 1.0, 3.0, "normal", 1.5, 0.25),
    ("crr", 0.8258, 0.5, 0.975, "beta", 0.75, 0.10),
    ("cry", 0.0893, 0.001, 0.5, "normal", 0.125, 0.05),
    ("crdy", 0.2239, 0.001, 0.5, "normal", 0.125, 0.05),
    ("constepinf", 0.7, 0.1, 2.0, "gamma", 0.625, 0.1),
    ("constebeta", 0.7420, 0.01, 2.0, "gamma", 0.25, 0.1),
    ("constelab", 1.2918, -10.0, 10.0, "normal", 0.0, 2.0),
    ("ctrend", 0.3982, 0.1, 0.8, "normal", 0.4, 0.10),
    ("cgy", 0.05, 0.01, 2.0, "normal", 0.5, 0.25),
    ("calfa", 0.24, 0.01, 1.0, "normal", 0.3, 0.05),
)
SW07_PAPER_PARAMETER_NAMES = tuple(spec[0] for spec in SW07_PAPER_PRIOR_SPECS)
SW07_PAPER_DATA_KEYS = ("dy", "dc", "dinve", "labobs", "pinfobs", "dw", "robs")
SW07_PAPER_OBSERVABLES = ("dy", "dc", "dinve", "labobs", "pinfobs", "dwobs", "robs")
SW07_PAPER_DATA_SHA256 = "35e09d915973e89442e50584f790f4859393219818c03ec8f2eead2b8a3c2050"

# Table 5 of Farkas and Tatar (2023): posterior mean and 5/95 percentiles for
# the original SW07 mode.  These are validation targets, not hard-coded results.
SW07_PAPER_REFERENCE_POSTERIOR: dict[str, tuple[float, float, float]] = {
    "csadjcost": (5.93, 4.38, 7.68),
    "csigma": (1.41, 1.20, 1.66),
    "chabb": (0.73, 0.65, 0.80),
    "cprobw": (0.75, 0.67, 0.84),
    "csigl": (2.10, 1.21, 3.02),
    "cprobp": (0.65, 0.56, 0.73),
    "cindw": (0.56, 0.35, 0.77),
    "cindp": (0.24, 0.11, 0.38),
    "czcap": (0.46, 0.30, 0.65),
    "cfc": (1.65, 1.53, 1.79),
    "crpi": (2.04, 1.77, 2.33),
    "crr": (0.82, 0.77, 0.85),
    "cry": (0.10, 0.07, 0.14),
    "crdy": (0.21, 0.17, 0.25),
    "constepinf": (0.67, 0.50, 0.86),
    "constebeta": (0.13, 0.07, 0.22),
    "constelab": (0.88, -0.68, 2.45),
    "ctrend": (0.47, 0.43, 0.49),
    "calfa": (0.21, 0.18, 0.23),
    "z_ea": (0.48, 0.43, 0.52),
    "z_eb": (0.24, 0.19, 0.28),
    "z_eg": (0.52, 0.47, 0.57),
    "z_eqs": (0.46, 0.38, 0.54),
    "z_em": (0.23, 0.21, 0.25),
    "z_epinf": (0.13, 0.10, 0.16),
    "z_ew": (0.25, 0.21, 0.28),
    "crhoa": (0.98, 0.97, 0.99),
    "crhob": (0.27, 0.12, 0.48),
    "crhog": (0.97, 0.96, 0.99),
    "crhoqs": (0.69, 0.59, 0.78),
    "crhoms": (0.17, 0.07, 0.28),
    "crhopinf": (0.96, 0.92, 0.99),
    "crhow": (0.97, 0.94, 0.99),
    "cmap": (0.80, 0.66, 0.90),
    "cmaw": (0.89, 0.82, 0.94),
    "cgy": (0.57, 0.44, 0.70),
}

GALI3_POLICY_4_PARAMETERS = (
    "phi_pi",
    "phi_x",
    "rho_i",
    "rho_u",
)
GALI3_POLICY_7_PARAMETERS = (
    "kappa",
    "phi_pi",
    "phi_x",
    "rho_i",
    "rho_rn",
    "rho_u",
    "sigma",
)
GALI3_ALL_STABLE_PARAMETERS = (
    "beta",
    "kappa",
    "phi_pi",
    "phi_x",
    "rho_i",
    "rho_rn",
    "rho_u",
    "sigma",
    "std_i",
    "std_rn",
    "std_u",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _run_text(cmd: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(cmd),
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def _block_tree(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            _block_tree(item)
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_tree(item)
        return value
    return value


def _timed_call(fn):
    start = time.perf_counter()
    value = fn()
    _block_tree(value)
    return value, time.perf_counter() - start


def _timing_stats(times: Sequence[float]) -> dict[str, Any]:
    if not times:
        return {"reps": 0}
    return {
        "reps": len(times),
        "mean_s": float(statistics.mean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
    }


def _log(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "verbose", False):
        elapsed = time.perf_counter() - getattr(args, "_benchmark_started_at", time.perf_counter())
        print(f"[posterior_sampling_speed +{elapsed:9.2f}s] {message}", flush=True)


@contextmanager
def _logged_stage(args: argparse.Namespace, name: str):
    start = time.perf_counter()
    _log(args, f"START {name}")
    stop = threading.Event()
    heartbeat_seconds = float(getattr(args, "heartbeat_seconds", 0.0) or 0.0)
    thread: threading.Thread | None = None
    if getattr(args, "verbose", False) and heartbeat_seconds > 0.0:

        def heartbeat() -> None:
            while not stop.wait(heartbeat_seconds):
                elapsed = time.perf_counter() - start
                _log(args, f"HEARTBEAT {name} still running after {elapsed:.1f}s")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        _log(args, f"FAIL {name} after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = time.perf_counter() - start
        _log(args, f"END {name} after {elapsed:.2f}s")
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=0.1)


def _jax_platform_name(platform: str | None) -> str | None:
    """Map user-facing backend aliases to JAX backend names."""
    if platform == "gpu":
        return "cuda"
    return platform


def _configure_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    if args.platform:
        os.environ["JAX_PLATFORM_NAME"] = _jax_platform_name(args.platform)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    import surrogatenn_dsge as sdsge

    # The package currently enables x64 at import time for parity. Reset it here
    # so FP32 GPU runs are measured as true FP32 runs.
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    if args.dtype == "float32" and args.suppress_dtype_warnings:
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype .*float64.*",
            category=UserWarning,
        )
    if args.host_device_count is not None:
        numpyro.set_host_device_count(int(args.host_device_count))
    if args.force_gpu and not any(device.platform in {"gpu", "cuda"} for device in jax.devices()):
        raise RuntimeError(
            "--force-gpu was set, but JAX did not report a CUDA/GPU device. "
            f"Detected devices: {jax.devices()}"
        )
    return jax, jnp, numpyro, dist, sdsge


def _runtime_info(jax: Any, numpyro: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "numpyro_version": getattr(numpyro, "__version__", "unknown"),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "nvidia_smi": _run_text(["nvidia-smi"]).strip(),
    }


def _load_payload_case(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for case in payload["cases"]:
        if case["name"] == name:
            return case
    raise ValueError(f"Case {name!r} not found in {path}.")


def _select_parameter_names(model: Any, spec: str) -> tuple[str, ...]:
    spec = spec.strip()
    if spec == "sw07_safe_15":
        names = SW07_SAFE_15_PARAMETERS
    elif spec == "sw07_safe_27":
        names = SW07_SAFE_27_PARAMETERS
    elif spec == "sw07_paper_36":
        names = SW07_PAPER_PARAMETER_NAMES
    elif spec == "gali3_policy_4":
        names = GALI3_POLICY_4_PARAMETERS
    elif spec == "gali3_policy_7":
        names = GALI3_POLICY_7_PARAMETERS
    elif spec == "gali3_all_stable":
        names = GALI3_ALL_STABLE_PARAMETERS
    elif spec == "payload":
        raise ValueError("The 'payload' parameter selector must be expanded by the caller.")
    elif spec == "all":
        names = tuple(model.parameter_names)
    elif "," not in spec:
        try:
            import surrogatenn_dsge as sdsge

            names = tuple(sdsge.get_parameter_names(spec))
        except Exception:
            names = (spec,)
    else:
        names = tuple(name.strip() for name in spec.split(",") if name.strip())
    unknown = tuple(name for name in names if name not in model.parameter_names)
    if unknown:
        raise ValueError("Unknown parameter names: " + ", ".join(unknown))
    return names


def _prior_interval(name: str, center: float, scale: float, floor: float) -> tuple[float, float]:
    width = max(abs(center) * scale, floor)
    lower = center - width
    upper = center + width
    bounded_unit_prefixes = ("crho", "cprob", "cind")
    bounded_unit_parameter = name.startswith(bounded_unit_prefixes) or name in {"rho_i", "rho_rn", "rho_u"}
    if bounded_unit_parameter and center > 0.0:
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if name in {"calfa"}:
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if name == "beta":
        lower = max(1.0e-4, lower)
        upper = min(0.9999, upper)
    if name == "phi_pi":
        lower = max(1.0001, lower)
    if center > 0.0 and lower <= 0.0 and name not in {"cry"}:
        lower = max(center * 0.5, np.finfo(float).tiny)
    if not lower < center < upper:
        raise ValueError(
            f"Invalid prior interval for {name}: center={center}, lower={lower}, upper={upper}."
        )
    return float(lower), float(upper)


def _make_centered_uniform_priors(
    dist: Any,
    model: Any,
    parameter_names: Sequence[str],
    *,
    parameter_values: np.ndarray | None = None,
    width_scale: float,
    width_floor: float,
) -> tuple[dict[str, Any], dict[str, float], dict[str, tuple[float, float]]]:
    index = {name: idx for idx, name in enumerate(model.parameter_names)}
    priors: dict[str, Any] = {}
    initial_values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    values = (
        np.asarray(model.parameter_values, dtype=np.float64)
        if parameter_values is None
        else np.asarray(parameter_values, dtype=np.float64)
    )
    for name in parameter_names:
        center = float(values[index[name]])
        lower, upper = _prior_interval(name, center, width_scale, width_floor)
        priors[name] = dist.Uniform(lower, upper)
        initial_values[name] = center
        intervals[name] = (lower, upper)
    return priors, initial_values, intervals


def _distribution_parameters_from_moments(
    family: str,
    mean: float,
    std: float,
) -> dict[str, float]:
    """Convert Dynare mean/sd prior declarations to NumPyro parameters."""
    if std <= 0.0:
        raise ValueError(f"Prior standard deviation must be positive, got {std}.")
    if family == "normal":
        return {"loc": float(mean), "scale": float(std)}
    if family == "beta":
        concentration = mean * (1.0 - mean) / (std * std) - 1.0
        if not 0.0 < mean < 1.0 or concentration <= 0.0:
            raise ValueError(f"Invalid beta moments mean={mean}, std={std}.")
        return {
            "concentration1": float(mean * concentration),
            "concentration0": float((1.0 - mean) * concentration),
        }
    if family == "gamma":
        if mean <= 0.0:
            raise ValueError(f"Gamma mean must be positive, got {mean}.")
        return {
            "concentration": float((mean / std) ** 2),
            "rate": float(mean / (std * std)),
        }
    if family == "inverse_gamma":
        if mean <= 0.0:
            raise ValueError(f"Inverse-gamma mean must be positive, got {mean}.")
        concentration = 2.0 + (mean / std) ** 2
        return {
            "concentration": float(concentration),
            "rate": float(mean * (concentration - 1.0)),
        }
    raise ValueError(f"Unsupported prior family {family!r}.")


def _bounded_prior_distribution(
    dist: Any,
    base_distribution: Any,
    lower: float,
    upper: float,
) -> Any:
    """Return a hard-bounded prior with the base density inside its support.

    NumPyro does not provide a single truncation wrapper for every distribution
    family used by Dynare.  The missing truncation normalizer is constant in the
    sampled parameter, so this distribution defines exactly the same posterior
    target while exposing interval support to HMC's unconstraining transform.
    """

    class _BoundedPrior(dist.Distribution):
        arg_constraints: dict[str, Any] = {}
        reparametrized_params: list[str] = []

        def __init__(self) -> None:
            self.base_distribution = base_distribution
            self.lower = lower
            self.upper = upper
            super().__init__(
                batch_shape=base_distribution.batch_shape,
                event_shape=base_distribution.event_shape,
            )

        @property
        def support(self) -> Any:
            return dist.constraints.interval(self.lower, self.upper)

        def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
            return dist.Uniform(self.lower, self.upper).sample(key, sample_shape)

        def log_prob(self, value: Any) -> Any:
            return self.base_distribution.log_prob(value)

    return _BoundedPrior()


def _make_sw07_paper_priors(
    dist: Any,
    *,
    initialization: str,
) -> tuple[dict[str, Any], dict[str, float], dict[str, tuple[float, float]], dict[str, Any]]:
    if initialization not in {"dynare", "prior_mean"}:
        raise ValueError("SW07 initialization must be 'dynare' or 'prior_mean'.")
    priors: dict[str, Any] = {}
    initial_values: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    metadata: dict[str, Any] = {}
    for name, dynare_initial, lower, upper, family, mean, std in SW07_PAPER_PRIOR_SPECS:
        parameters = _distribution_parameters_from_moments(family, mean, std)
        if family == "normal":
            base = dist.Normal(parameters["loc"], parameters["scale"])
        elif family == "beta":
            base = dist.Beta(parameters["concentration1"], parameters["concentration0"])
        elif family == "gamma":
            base = dist.Gamma(parameters["concentration"], parameters["rate"])
        elif family == "inverse_gamma":
            base = dist.InverseGamma(parameters["concentration"], parameters["rate"])
        else:  # pragma: no cover - protected by the conversion helper
            raise AssertionError(family)
        priors[name] = _bounded_prior_distribution(dist, base, lower, upper)
        candidate = dynare_initial if initialization == "dynare" else mean
        if not lower < candidate < upper:
            raise ValueError(
                f"Initialization {candidate} for {name} is outside ({lower}, {upper})."
            )
        initial_values[name] = float(candidate)
        intervals[name] = (float(lower), float(upper))
        metadata[name] = {
            "family": family,
            "mean": float(mean),
            "std": float(std),
            "lower": float(lower),
            "upper": float(upper),
            "dynare_initial": float(dynare_initial),
            "distribution_parameters": parameters,
        }
    return priors, initial_values, intervals, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sw07_paper_observations(
    path: Path,
    *,
    sample_start_index: int,
    sample_periods: int,
    expected_sha256: str = SW07_PAPER_DATA_SHA256,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"SW07 data file not found: {path}. Pass the archived usmodel_data.mat "
            "with --sw07-data-mat; the data are intentionally not bundled."
        )
    actual_sha256 = _sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "SW07 data fingerprint mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    if sample_start_index < 0 or sample_periods <= 0:
        raise ValueError("SW07 sample_start_index must be nonnegative and sample_periods positive.")
    from scipy.io import loadmat

    raw = loadmat(path, squeeze_me=True)
    series: list[np.ndarray] = []
    full_lengths: dict[str, int] = {}
    stop = sample_start_index + sample_periods
    for key in SW07_PAPER_DATA_KEYS:
        if key not in raw:
            raise ValueError(f"SW07 data file is missing required series {key!r}.")
        values = np.asarray(raw[key], dtype=np.float64).reshape(-1)
        full_lengths[key] = int(values.size)
        if values.size < stop:
            raise ValueError(
                f"SW07 series {key!r} has {values.size} rows; need at least {stop}."
            )
        selected = values[sample_start_index:stop]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"SW07 series {key!r} contains non-finite selected observations.")
        series.append(selected)
    if len(set(full_lengths.values())) != 1:
        raise ValueError(f"SW07 series lengths disagree: {full_lengths}.")
    observations = np.stack(series, axis=0)
    provenance = {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "data_keys": list(SW07_PAPER_DATA_KEYS),
        "observable_mapping": dict(zip(SW07_PAPER_DATA_KEYS, SW07_PAPER_OBSERVABLES)),
        "full_lengths": full_lengths,
        "sample_start_index_zero_based": int(sample_start_index),
        "sample_periods": int(sample_periods),
        "declared_sample": "1960Q1--2004Q4"
        if sample_start_index == 50 and sample_periods == 180
        else "custom",
    }
    return observations, provenance


def _toy_payload(sdsge: Any, jax: Any, periods: int, seed: int) -> dict[str, Any]:
    model = sdsge.parse_macro_model(TOY_AR2_SOURCE)
    first_order = sdsge.solve_first_order_model(
        model,
        steady_state_initial_guess={"a": 1.5, "y": 2.0},
        qme_algorithm="schur",
    )
    observables = ("y", "a")
    state_space = sdsge.build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order,
        measurement_error_scale=1.0e-8,
    )
    simulation = sdsge.simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(seed),
        num_periods=periods,
    )
    steady_lookup = dict(zip(model.timings.var, np.asarray(first_order.steady_state)))
    observations = np.asarray(simulation.observations, dtype=np.float64) + np.asarray(
        [[steady_lookup[name]] for name in observables],
        dtype=np.float64,
    )
    return {
        "model": model,
        "steady_state": np.asarray(first_order.steady_state, dtype=np.float64),
        "observations": observations,
        "observables": observables,
        "parameter_names": ("rho_a", "rho_y"),
        "measurement_error_scale": 1.0e-8,
        "jitter": 1.0e-8,
    }


def _gali3_payload(args: argparse.Namespace, sdsge: Any, jax: Any) -> dict[str, Any]:
    model_source_path = (
        DEFAULT_GALI3_MODEL_SOURCE_PATH
        if Path(args.model_source) == DEFAULT_SW07_MODEL_SOURCE_PATH
        else Path(args.model_source)
    )
    model = sdsge.parse_macro_model(model_source_path.read_text())
    steady_state_result = sdsge.solve_steady_state(model)
    steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
    first_order = sdsge.solve_first_order_model(
        model,
        steady_state=steady_state,
        qme_algorithm="schur",
    )
    if not first_order.solution.converged:
        raise RuntimeError("Could not generate synthetic Gali observations: Schur solve failed.")
    observables = ("y_obs", "pi_obs", "i_obs")
    state_space = sdsge.build_linear_state_space_from_model(
        model,
        observables,
        first_order_result=first_order,
        measurement_error_scale=1.0e-8,
    )
    simulation = sdsge.simulate_linear_gaussian_state_space(
        state_space,
        key=jax.random.PRNGKey(int(args.synthetic_seed)),
        num_periods=int(args.periods),
    )
    steady_lookup = dict(zip(model.timings.var, steady_state))
    observations = np.asarray(simulation.observations, dtype=np.float64) + np.asarray(
        [[steady_lookup[name]] for name in observables],
        dtype=np.float64,
    )
    parameter_spec = (
        "gali3_policy_4" if args.parameters == "sw07_safe_15" else args.parameters
    )
    return {
        "model": model,
        "steady_state": steady_state,
        "observations": observations,
        "observables": observables,
        "parameter_names": _select_parameter_names(model, parameter_spec),
        "measurement_error_scale": 1.0e-8,
        "jitter": 1.0e-8,
    }


def _sw07_payload(args: argparse.Namespace, sdsge: Any) -> dict[str, Any]:
    case = _load_payload_case(args.payload_path, args.case)
    model_source = args.model_source.read_text()
    model = sdsge.parse_macro_model(model_source)
    steady_state = np.asarray(case["reference_steady_state"], dtype=np.float64)
    observables = tuple(case["observables"])
    periods = int(args.periods)
    if periods <= len(case["observations"][0]):
        observations = np.asarray(case["observations"], dtype=np.float64)[:, :periods]
    else:
        first_order = sdsge.solve_first_order_model(
            model,
            steady_state=steady_state,
            qme_algorithm="schur",
        )
        if not first_order.solution.converged:
            raise RuntimeError("Could not generate synthetic SW07 observations: Schur solve failed.")
        shock_names = tuple(case["shock_names"])
        shock_sigmas = np.asarray(
            [float(case["shock_sigmas"][name]) for name in shock_names],
            dtype=np.float64,
        )
        rng = np.random.default_rng(int(args.synthetic_seed))
        shock_matrix = shock_sigmas[:, None] * rng.standard_normal((len(shock_names), periods))
        full_levels = (
            np.asarray(
                sdsge.rollout_first_order_solution(
                    first_order.solution.solution_matrix,
                    model.timings,
                    shock_matrix,
                ),
                dtype=np.float64,
            )
            + steady_state[:, None]
        )
        observation_indices = [model.timings.var.index(name) for name in observables]
        observations = full_levels[observation_indices, :]
    parameter_spec = args.parameters
    parameter_names = (
        tuple(case["parameter_subset"])
        if parameter_spec == "payload"
        else _select_parameter_names(model, parameter_spec)
    )
    return {
        "model": model,
        "steady_state": steady_state,
        "observations": observations,
        "observables": observables,
        "parameter_names": parameter_names,
        "measurement_error_scale": float(case.get("measurement_error_scale", 1.0e-8)),
        "jitter": float(case.get("jitter", 1.0e-9)),
    }


def _sw07_paper_payload(args: argparse.Namespace, sdsge: Any) -> dict[str, Any]:
    model_source_path = (
        DEFAULT_SW07_PAPER_MODEL_SOURCE_PATH
        if Path(args.model_source) == DEFAULT_SW07_MODEL_SOURCE_PATH
        else Path(args.model_source)
    )
    model = sdsge.parse_macro_model(model_source_path.read_text())
    unknown = tuple(name for name in SW07_PAPER_PARAMETER_NAMES if name not in model.parameter_names)
    if unknown:
        raise ValueError("SW07 paper model is missing estimands: " + ", ".join(unknown))
    base_parameter_overrides = {
        name: float(initial)
        for name, initial, _lower, _upper, _family, _mean, _std in SW07_PAPER_PRIOR_SPECS
    }
    theta = np.asarray(model.parameter_values, dtype=np.float64).copy()
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value in base_parameter_overrides.items():
        theta[parameter_index[name]] = value
    steady_state_result = sdsge.solve_steady_state(model, parameter_values=theta)
    if not steady_state_result.converged:
        raise RuntimeError(
            "The paper-matched linear SW07 model did not solve at the archived initialization."
        )
    steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
    if args.sw07_data_mat is None:
        raise ValueError("--preset sw07_paper requires --sw07-data-mat.")
    observations, provenance = _load_sw07_paper_observations(
        Path(args.sw07_data_mat),
        sample_start_index=int(args.sw07_sample_start_index),
        sample_periods=int(args.sw07_sample_periods),
    )
    provenance["model_source_path"] = str(model_source_path.resolve())
    provenance["model_source_sha256"] = _sha256(model_source_path)
    return {
        "model": model,
        "steady_state": steady_state,
        "observations": observations,
        "observables": SW07_PAPER_OBSERVABLES,
        "parameter_names": SW07_PAPER_PARAMETER_NAMES,
        "base_parameter_overrides": base_parameter_overrides,
        # The archived Dynare specification has neither measurement error nor
        # an innovation-covariance ridge.  Keep both at exactly zero so the
        # same-point likelihood is a genuine cross-implementation parity test.
        "measurement_error_scale": 0.0,
        "jitter": 0.0,
        "data_provenance": provenance,
        "prior_specification": "Farkas-Tatar-2023 archived Dynare estimated_params",
    }


def _build_dataset(args: argparse.Namespace, sdsge: Any, jax: Any) -> dict[str, Any]:
    if args.preset == "toy_ar2":
        return _toy_payload(sdsge, jax, int(args.periods), int(args.synthetic_seed))
    if args.preset == "gali3_nk":
        return _gali3_payload(args, sdsge, jax)
    if args.preset == "sw07_hlt":
        return _sw07_payload(args, sdsge)
    if args.preset == "sw07_paper":
        return _sw07_paper_payload(args, sdsge)
    raise ValueError(f"Unsupported preset {args.preset!r}.")


def _sample_summary(
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    selected_names = (
        tuple(parameter_names)
        if parameter_names is not None
        else tuple(samples_by_chain)
    )
    np_samples = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in samples_by_chain.items()
        if name in selected_names
    }
    parameter_metrics: dict[str, dict[str, float]] = {}
    ess_values: list[float] = []
    rhat_values: list[float] = []
    for name, values in np_samples.items():
        flat = values.reshape(-1)
        try:
            ess = float(np.asarray(effective_sample_size(values), dtype=np.float64))
        except Exception:
            ess = float("nan")
        try:
            rhat = float(np.asarray(split_gelman_rubin(values), dtype=np.float64))
        except Exception:
            rhat = float("nan")
        parameter_metrics[name] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.quantile(flat, 0.50)),
            "q95": float(np.quantile(flat, 0.95)),
            "n_eff": ess,
            "r_hat": rhat,
            "shape": list(values.shape),
        }
        if np.isfinite(ess) and ess > 0.0:
            ess_values.append(ess)
        if np.isfinite(rhat):
            rhat_values.append(rhat)
    return {
        "parameters": parameter_metrics,
        "min_ess": float(min(ess_values)) if ess_values else None,
        "mean_ess": float(statistics.mean(ess_values)) if ess_values else None,
        "max_r_hat": float(max(rhat_values)) if rhat_values else None,
    }


def _sw07_reference_comparison(sample_summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = sample_summary.get("parameters", {})
    missing = tuple(name for name in SW07_PAPER_PARAMETER_NAMES if name not in metrics)
    if missing:
        return {
            "complete": False,
            "missing_parameters": list(missing),
            "parameters": {},
        }
    parameter_comparison: dict[str, Any] = {}
    normalized_mean_errors: list[float] = []
    posterior_means_inside: list[bool] = []
    reference_means_inside: list[bool] = []
    overlap_shares: list[float] = []
    for name in SW07_PAPER_PARAMETER_NAMES:
        reference_mean, reference_q05, reference_q95 = SW07_PAPER_REFERENCE_POSTERIOR[name]
        posterior_mean = float(metrics[name]["mean"])
        posterior_q05 = float(metrics[name]["q05"])
        posterior_q95 = float(metrics[name]["q95"])
        reference_width = reference_q95 - reference_q05
        normalized_error = abs(posterior_mean - reference_mean) / reference_width
        overlap = max(0.0, min(reference_q95, posterior_q95) - max(reference_q05, posterior_q05))
        overlap_share = overlap / reference_width
        posterior_mean_inside = reference_q05 <= posterior_mean <= reference_q95
        reference_mean_inside = posterior_q05 <= reference_mean <= posterior_q95
        normalized_mean_errors.append(normalized_error)
        overlap_shares.append(overlap_share)
        posterior_means_inside.append(posterior_mean_inside)
        reference_means_inside.append(reference_mean_inside)
        parameter_comparison[name] = {
            "posterior_mean": posterior_mean,
            "posterior_q05": posterior_q05,
            "posterior_q95": posterior_q95,
            "reference_mean": reference_mean,
            "reference_q05": reference_q05,
            "reference_q95": reference_q95,
            "normalized_mean_error": float(normalized_error),
            "reference_interval_overlap_share": float(overlap_share),
            "posterior_mean_inside_reference_90pct": posterior_mean_inside,
            "reference_mean_inside_posterior_90pct": reference_mean_inside,
        }
    return {
        "complete": True,
        "missing_parameters": [],
        "parameters": parameter_comparison,
        "median_normalized_mean_error": float(np.median(normalized_mean_errors)),
        "max_normalized_mean_error": float(np.max(normalized_mean_errors)),
        "mean_reference_interval_overlap_share": float(np.mean(overlap_shares)),
        "posterior_means_inside_reference_90pct_count": int(sum(posterior_means_inside)),
        "reference_means_inside_posterior_90pct_count": int(sum(reference_means_inside)),
        "parameter_count": len(SW07_PAPER_PARAMETER_NAMES),
    }


def _extra_field_summary(extra_fields: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("accept_prob", "diverging", "num_steps"):
        if name not in extra_fields:
            result[name] = None
            continue
        values = np.asarray(extra_fields[name])
        if values.size == 0:
            result[name] = None
            continue
        if name == "diverging":
            result[name] = {
                "count": int(np.asarray(values, dtype=bool).sum()),
                "share": float(np.asarray(values, dtype=bool).mean()),
            }
        else:
            finite = np.asarray(values, dtype=np.float64)
            result[name] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
    return result


def _flatten_posterior_draws(
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str],
    *,
    max_draws: int,
) -> list[dict[str, float]]:
    if max_draws <= 0:
        return []
    arrays = {
        name: np.asarray(samples_by_chain[name], dtype=np.float64).reshape(-1)
        for name in parameter_names
    }
    n_draws = min(max_draws, *(array.shape[0] for array in arrays.values()))
    return [
        {name: float(arrays[name][draw_idx]) for name in parameter_names}
        for draw_idx in range(n_draws)
    ]


def _support_audit(
    *,
    sdsge: Any,
    model: Any,
    observations: np.ndarray,
    observables: Sequence[str],
    steady_state: np.ndarray,
    samples_by_chain: Mapping[str, Any],
    parameter_names: Sequence[str],
    base_parameter_values: np.ndarray,
    measurement_error_scale: float,
    jitter: float,
    max_draws: int,
    schur_acceptance_tol: float,
    failure_value: float,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    draws = _flatten_posterior_draws(
        samples_by_chain,
        parameter_names,
        max_draws=max_draws,
    )
    classifications: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    doubling_accepts_non_unique = 0
    both_accept = 0
    schur_rejects = 0
    doubling_rejects = 0
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for draw_idx, draw in enumerate(draws):
        if log_fn is not None:
            log_fn(f"support audit draw {draw_idx + 1}/{len(draws)}")
        theta = np.asarray(base_parameter_values, dtype=np.float64).copy()
        for name, value in draw.items():
            theta[parameter_index[name]] = value
        try:
            schur = sdsge.analyze_first_order_model_determinacy(
                model,
                parameter_values=theta,
                steady_state=steady_state,
                qme_acceptance_tol=schur_acceptance_tol,
            )
            classification = str(schur.determinacy.classification)
            schur_unique = bool(schur.determinacy.unique_stable_solution)
        except Exception as exc:
            classification = f"schur_error:{type(exc).__name__}"
            schur_unique = False
        classifications[classification] = classifications.get(classification, 0) + 1
        try:
            doubling_ll = sdsge.kalman_loglikelihood_from_model(
                model,
                observations,
                observables=observables,
                parameter_values=theta,
                steady_state=steady_state,
                measurement_error_scale=measurement_error_scale,
                jitter=jitter,
                qme_algorithm="doubling",
                on_failure_loglikelihood=failure_value,
            )
            doubling_value = float(np.asarray(doubling_ll, dtype=np.float64))
            doubling_accept = np.isfinite(doubling_value) and doubling_value > failure_value / 10.0
        except Exception as exc:
            doubling_value = float("nan")
            doubling_accept = False
            if len(examples) < 5:
                examples.append(
                    {
                        "draw_index": draw_idx,
                        "classification": classification,
                        "doubling_error": f"{type(exc).__name__}: {exc}",
                        "parameters": draw,
                    }
                )
        if schur_unique and doubling_accept:
            both_accept += 1
        if not schur_unique:
            schur_rejects += 1
        if not doubling_accept:
            doubling_rejects += 1
        if doubling_accept and not schur_unique:
            doubling_accepts_non_unique += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "draw_index": draw_idx,
                        "classification": classification,
                        "doubling_loglikelihood": doubling_value,
                        "parameters": draw,
                    }
                )
        if log_fn is not None:
            log_fn(
                "support audit draw "
                f"{draw_idx + 1}/{len(draws)} classified {classification}; "
                f"schur_unique={schur_unique}; doubling_accept={doubling_accept}"
            )
    audited = len(draws)
    return {
        "audited_draws": audited,
        "schur_classification_counts": classifications,
        "both_accept_count": both_accept,
        "schur_reject_count": schur_rejects,
        "doubling_reject_count": doubling_rejects,
        "doubling_accepts_non_unique_count": doubling_accepts_non_unique,
        "doubling_accepts_non_unique_share": (
            float(doubling_accepts_non_unique / audited) if audited else None
        ),
        "examples": examples,
    }


def _preflight_metrics(
    *,
    jax: Any,
    jnp: Any,
    sdsge: Any,
    model: Any,
    observations: np.ndarray,
    observables: Sequence[str],
    steady_state: np.ndarray | None,
    parameter_values: np.ndarray,
    parameter_names: Sequence[str],
    measurement_error_scale: float,
    jitter: float,
    qme_algorithm: str,
    static_equation_rows: Sequence[int] | None,
    reps: int,
    failure_value: float,
    parameters_are_resolved: bool,
    check_parameter_bounds: bool,
    steady_state_initial_guess: np.ndarray | None = None,
    presample_periods: int = 0,
    initial_covariance_strategy: str = "theoretical",
    steady_state_tol: float = 1.0e-12,
    steady_state_max_iter: int = 200,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    theta = jnp.asarray(parameter_values)
    obs = jnp.asarray(observations)
    steady = None if steady_state is None else jnp.asarray(steady_state)
    steady_guess = (
        None if steady_state_initial_guess is None else jnp.asarray(steady_state_initial_guess)
    )
    value_fn = jax.jit(
        lambda current_theta: sdsge.kalman_loglikelihood_from_model_jax(
            model,
            obs,
            observables=observables,
            parameter_values=current_theta,
            steady_state=steady,
            steady_state_initial_guess=steady_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            initial_covariance_strategy=initial_covariance_strategy,
            presample_periods=presample_periods,
            measurement_error_scale=measurement_error_scale,
            jitter=jitter,
            qme_algorithm=qme_algorithm,
            on_failure_loglikelihood=failure_value,
            static_equation_rows=static_equation_rows,
            parameters_are_resolved=parameters_are_resolved,
            check_parameter_bounds=check_parameter_bounds,
        )
    )
    index = jnp.asarray(
        [model.parameter_names.index(name) for name in parameter_names],
        dtype=jnp.int32,
    )
    x0 = theta[index]

    def inject(x):
        return theta.at[index].set(x)

    grad_fn = jax.jit(jax.value_and_grad(lambda x: value_fn(inject(x))))
    if log_fn is not None:
        log_fn("preflight value first call/JIT start")
    value, first_value_s = _timed_call(lambda: value_fn(theta))
    if log_fn is not None:
        log_fn(f"preflight value first call/JIT done in {first_value_s:.3f}s")
    value_times = []
    for rep in range(max(reps, 0)):
        if log_fn is not None:
            log_fn(f"preflight value steady rep {rep + 1}/{reps} start")
        _, elapsed = _timed_call(lambda: value_fn(theta))
        value_times.append(elapsed)
        if log_fn is not None:
            log_fn(f"preflight value steady rep {rep + 1}/{reps} done in {elapsed:.3f}s")
    if log_fn is not None:
        log_fn("preflight gradient first call/JIT start")
    grad_value, first_grad_s = _timed_call(lambda: grad_fn(x0))
    if log_fn is not None:
        log_fn(f"preflight gradient first call/JIT done in {first_grad_s:.3f}s")
    grad_times = []
    for rep in range(max(reps, 0)):
        if log_fn is not None:
            log_fn(f"preflight gradient steady rep {rep + 1}/{reps} start")
        _, elapsed = _timed_call(lambda: grad_fn(x0))
        grad_times.append(elapsed)
        if log_fn is not None:
            log_fn(f"preflight gradient steady rep {rep + 1}/{reps} done in {elapsed:.3f}s")
    return {
        "loglikelihood": float(np.asarray(value, dtype=np.float64)),
        "loglikelihood_dtype": str(getattr(value, "dtype", "")),
        "value_first_call_s": float(first_value_s),
        "value_steady": _timing_stats(value_times),
        "gradient_value": float(np.asarray(grad_value[0], dtype=np.float64)),
        "gradient": np.asarray(grad_value[1], dtype=np.float64).tolist(),
        "gradient_dtype": str(getattr(grad_value[1], "dtype", "")),
        "gradient_first_call_s": float(first_grad_s),
        "gradient_steady": _timing_stats(grad_times),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    args._benchmark_started_at = time.perf_counter()
    _log(args, "benchmark process started")
    with _logged_stage(args, "configure JAX/NumPyro runtime"):
        jax, jnp, numpyro, dist, sdsge = _configure_runtime(args)
        _log(
            args,
            "runtime configured: "
            f"jax={getattr(jax, '__version__', 'unknown')}, "
            f"numpyro={getattr(numpyro, '__version__', 'unknown')}, "
            f"backend={jax.default_backend()}, devices={jax.devices()}",
        )
    if args.dtype == "float32":
        np_dtype = np.float32
    else:
        np_dtype = np.float64
    with _logged_stage(args, "build benchmark dataset"):
        data = _build_dataset(args, sdsge, jax)
    model = data["model"]
    reference_steady_state = np.asarray(data["steady_state"], dtype=np_dtype)
    observations = np.asarray(data["observations"], dtype=np_dtype)
    observables = tuple(data["observables"])
    parameter_names = tuple(data["parameter_names"])
    raw_parameter_values = np.asarray(model.parameter_values, dtype=np.float64).copy()
    parameter_index = {name: idx for idx, name in enumerate(model.parameter_names)}
    for name, value in data.get("base_parameter_overrides", {}).items():
        raw_parameter_values[parameter_index[name]] = float(value)
    resolved_parameter_values = np.asarray(
        model.resolve_parameter_values(
            parameter_values=raw_parameter_values,
            steady_state=np.asarray(reference_steady_state, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    parameter_values = (
        resolved_parameter_values if args.parameters_are_resolved else raw_parameter_values
    ).astype(np_dtype, copy=False)
    likelihood_steady_state = (
        reference_steady_state if args.steady_state_mode == "fixed" else None
    )
    steady_state_initial_guess = (
        None if args.steady_state_mode == "fixed" else reference_steady_state
    )
    static_equation_rows = (
        model._first_order_static_equation_rows_for_values(
            steady_state=reference_steady_state,
            parameter_values=parameter_values,
        )
        if not model.has_obc
        else None
    )
    _log(
        args,
        "dataset ready: "
        f"preset={args.preset}, vars={model.timings.nVars}, exo={model.timings.nExo}, "
        f"observations={observations.shape}, parameters={parameter_names}",
    )
    with _logged_stage(args, "build priors"):
        prior_metadata = None
        if args.preset == "sw07_paper":
            priors, initial_values, prior_intervals, prior_metadata = _make_sw07_paper_priors(
                dist,
                initialization=args.sw07_initialization,
            )
        else:
            priors, initial_values, prior_intervals = _make_centered_uniform_priors(
                dist,
                model,
                parameter_names,
                parameter_values=parameter_values,
                width_scale=float(args.prior_width_scale),
                width_floor=float(args.prior_width_floor),
            )
    from numpyro.infer import HMC, MCMC, NUTS, init_to_value

    with _logged_stage(args, "build NumPyro model and MCMC kernel"):
        numpyro_model = sdsge.build_numpyro_kalman_model_jax(
            model,
            observations,
            priors,
            observables=observables,
            base_parameter_values=parameter_values,
            steady_state=likelihood_steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=float(args.steady_state_tol),
            steady_state_max_iter=int(args.steady_state_max_iter),
            initial_covariance_strategy=args.initial_covariance_strategy,
            presample_periods=int(args.presample_periods),
            measurement_error_scale=float(data["measurement_error_scale"]),
            jitter=float(data["jitter"]),
            on_failure_loglikelihood=float(args.failure_value),
            qme_algorithm=args.qme_algorithm,
            static_equation_rows=static_equation_rows,
            parameters_are_resolved=bool(args.parameters_are_resolved),
            check_parameter_bounds=not bool(args.skip_parameter_bounds),
        )
        init_strategy = init_to_value(values=initial_values)
        if args.kernel == "nuts":
            kernel = NUTS(
                numpyro_model,
                dense_mass=bool(args.dense_mass),
                target_accept_prob=float(args.target_accept_prob),
                max_tree_depth=int(args.max_tree_depth),
                init_strategy=init_strategy,
            )
        else:
            kernel = HMC(
                numpyro_model,
                dense_mass=bool(args.dense_mass),
                target_accept_prob=float(args.target_accept_prob),
                num_steps=int(args.hmc_num_steps),
                trajectory_length=None,
                step_size=float(args.hmc_step_size),
                adapt_step_size=not bool(args.no_adapt_step_size),
                adapt_mass_matrix=not bool(args.no_adapt_mass_matrix),
                init_strategy=init_strategy,
            )
        mcmc = MCMC(
            kernel,
            num_warmup=int(args.warmup),
            num_samples=int(args.samples),
            num_chains=int(args.chains),
            chain_method=args.chain_method,
            progress_bar=bool(args.progress_bar),
        )

    benchmark_info = {
        "preset": args.preset,
        "case": args.case if args.preset in {"sw07_hlt", "sw07_paper"} else None,
        "model_name": getattr(model, "name", None),
        "periods": int(observations.shape[1]),
        "n_observables": int(observations.shape[0]),
        "n_vars": int(model.timings.nVars),
        "n_exo": int(model.timings.nExo),
        "parameter_count": len(parameter_names),
        "parameter_names": list(parameter_names),
        "qme_algorithm": args.qme_algorithm,
        "steady_state_mode": args.steady_state_mode,
        "steady_state_tol": float(args.steady_state_tol),
        "steady_state_max_iter": int(args.steady_state_max_iter),
        "initial_covariance_strategy": args.initial_covariance_strategy,
        "presample_periods": int(args.presample_periods),
        "measurement_error_scale": float(data["measurement_error_scale"]),
        "jitter": float(data["jitter"]),
        "dtype": args.dtype,
        "parameters_are_resolved": bool(args.parameters_are_resolved),
        "check_parameter_bounds": not bool(args.skip_parameter_bounds),
        "static_equation_rows": (
            list(static_equation_rows) if static_equation_rows is not None else None
        ),
        "resolved_parameter_max_abs_diff": float(
            np.max(np.abs(resolved_parameter_values - raw_parameter_values))
        )
        if raw_parameter_values.size
        else 0.0,
        "warmup": int(args.warmup),
        "samples": int(args.samples),
        "chains": int(args.chains),
        "chain_method": args.chain_method,
        "kernel": args.kernel,
        "hmc_num_steps": int(args.hmc_num_steps) if args.kernel == "hmc" else None,
        "hmc_step_size": float(args.hmc_step_size) if args.kernel == "hmc" else None,
        "adapt_step_size": not bool(args.no_adapt_step_size),
        "adapt_mass_matrix": not bool(args.no_adapt_mass_matrix),
        "dense_mass": bool(args.dense_mass),
        "target_accept_prob": float(args.target_accept_prob),
        "max_tree_depth": int(args.max_tree_depth),
        "prior_intervals": prior_intervals,
        "prior_metadata": prior_metadata,
        "sw07_initialization": args.sw07_initialization if args.preset == "sw07_paper" else None,
        "data_provenance": data.get("data_provenance"),
        "prior_specification": data.get("prior_specification", "centered uniform stress prior"),
        "store_posterior_draws": bool(args.store_posterior_draws),
    }

    preflight = None
    if args.preflight:
        with _logged_stage(args, "preflight likelihood and gradient"):
            preflight = _preflight_metrics(
                jax=jax,
                jnp=jnp,
                sdsge=sdsge,
                model=model,
                observations=observations,
                observables=observables,
                steady_state=likelihood_steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                parameter_values=parameter_values,
                parameter_names=parameter_names,
                measurement_error_scale=float(data["measurement_error_scale"]),
                jitter=float(data["jitter"]),
                qme_algorithm=args.qme_algorithm,
                static_equation_rows=static_equation_rows,
                reps=int(args.preflight_reps),
                failure_value=float(args.failure_value),
                parameters_are_resolved=bool(args.parameters_are_resolved),
                check_parameter_bounds=not bool(args.skip_parameter_bounds),
                presample_periods=int(args.presample_periods),
                initial_covariance_strategy=args.initial_covariance_strategy,
                steady_state_tol=float(args.steady_state_tol),
                steady_state_max_iter=int(args.steady_state_max_iter),
                log_fn=lambda message: _log(args, message),
            )
            _log(
                args,
                "preflight summary: "
                f"loglikelihood={preflight['loglikelihood']}, "
                f"value_steady_median={preflight['value_steady'].get('median_s')}, "
                f"gradient_steady_median={preflight['gradient_steady'].get('median_s')}",
            )

    if args.preflight_only:
        _log(args, "preflight-only requested; skipping MCMC")
        return {
            "benchmark": benchmark_info,
            "runtime": _runtime_info(jax, numpyro),
            "preflight": preflight,
            "throughput": None,
            "posterior_diagnostics": None,
            "extra_fields": None,
            "schur_support_audit": None,
            "sw07_reference_comparison": None,
            "posterior_draws": None,
        }

    _log(
        args,
        "starting MCMC: "
        f"warmup={args.warmup}, samples={args.samples}, chains={args.chains}, "
        f"chain_method={args.chain_method}, kernel={args.kernel}, "
        f"dense_mass={args.dense_mass}",
    )
    start = time.perf_counter()
    with _logged_stage(args, "NumPyro MCMC run"):
        mcmc.run(
            jax.random.PRNGKey(int(args.seed)),
            extra_fields=("accept_prob", "diverging", "num_steps"),
        )
    with _logged_stage(args, "block and collect samples"):
        _block_tree(mcmc.get_samples(group_by_chain=True))
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    sampling_wall_s = time.perf_counter() - start
    with _logged_stage(args, "posterior diagnostics"):
        sample_summary = _sample_summary(samples_by_chain, parameter_names)
    posterior_draws = (
        {
            name: np.asarray(samples_by_chain[name], dtype=np.float64).tolist()
            for name in parameter_names
        }
        if args.store_posterior_draws
        else None
    )
    post_warmup_draws = int(args.samples) * int(args.chains)
    min_ess = sample_summary["min_ess"]
    mean_ess = sample_summary["mean_ess"]
    throughput = {
        "sampling_wall_s": float(sampling_wall_s),
        "post_warmup_draws": post_warmup_draws,
        "draws_per_second": float(post_warmup_draws / sampling_wall_s),
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "seconds_per_min_ess": (
            float(sampling_wall_s / min_ess) if isinstance(min_ess, float) and min_ess > 0 else None
        ),
        "seconds_per_mean_ess": (
            float(sampling_wall_s / mean_ess) if isinstance(mean_ess, float) and mean_ess > 0 else None
        ),
        "min_ess_per_second": (
            float(min_ess / sampling_wall_s) if isinstance(min_ess, float) and sampling_wall_s > 0 else None
        ),
        "mean_ess_per_second": (
            float(mean_ess / sampling_wall_s) if isinstance(mean_ess, float) and sampling_wall_s > 0 else None
        ),
    }

    support_audit = None
    if args.schur_support_draws > 0:
        with _logged_stage(args, "Schur support audit"):
            support_audit = _support_audit(
                sdsge=sdsge,
                model=model,
                observations=np.asarray(observations, dtype=np.float64),
                observables=observables,
                steady_state=np.asarray(reference_steady_state, dtype=np.float64),
                samples_by_chain=samples_by_chain,
                parameter_names=parameter_names,
                base_parameter_values=np.asarray(parameter_values, dtype=np.float64),
                measurement_error_scale=float(data["measurement_error_scale"]),
                jitter=float(data["jitter"]),
                max_draws=int(args.schur_support_draws),
                schur_acceptance_tol=float(args.schur_acceptance_tol),
                failure_value=float(args.failure_value),
                log_fn=lambda message: _log(args, message),
            )
            _log(
                args,
                "support audit summary: "
                f"audited={support_audit['audited_draws']}, "
                f"doubling_accepts_non_unique="
                f"{support_audit['doubling_accepts_non_unique_count']}",
            )

    reference_comparison = (
        _sw07_reference_comparison(sample_summary)
        if args.preset == "sw07_paper"
        else None
    )

    _log(args, f"throughput summary: {json.dumps(throughput, sort_keys=True, default=_json_default)}")

    return {
        "benchmark": benchmark_info,
        "runtime": _runtime_info(jax, numpyro),
        "preflight": preflight,
        "throughput": throughput,
        "posterior_diagnostics": sample_summary,
        "extra_fields": _extra_field_summary(extra_fields),
        "schur_support_audit": support_audit,
        "sw07_reference_comparison": reference_comparison,
        "posterior_draws": posterior_draws,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure posterior sampling speed as seconds per effective sample "
            "for NumPyro/JAX DSGE estimation benchmarks."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("toy_ar2", "gali3_nk", "sw07_hlt", "sw07_paper"),
        default="sw07_hlt",
    )
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--case", default="medium_sw07_hlt")
    parser.add_argument("--model-source", type=Path, default=DEFAULT_SW07_MODEL_SOURCE_PATH)
    parser.add_argument(
        "--parameters",
        default="sw07_safe_15",
        help=(
            "Parameter subset. Supports comma-separated names, all, sw07_safe_15, "
            "sw07_safe_27, sw07_paper_36, gali3_policy_4, gali3_policy_7, gali3_all_stable, "
            "and named package parameter sets such as phase1_18params_narrow. "
            "For --preset gali3_nk, the default sw07_safe_15 is interpreted as "
            "gali3_policy_4 for convenience."
        ),
    )
    parser.add_argument("--periods", type=int, default=80)
    parser.add_argument("--synthetic-seed", type=int, default=20260712)
    parser.add_argument("--prior-width-scale", type=float, default=0.0025)
    parser.add_argument("--prior-width-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--sw07-data-mat",
        type=Path,
        default=None,
        help="Archived Dynare usmodel_data.mat required by --preset sw07_paper.",
    )
    parser.add_argument(
        "--sw07-sample-start-index",
        type=int,
        default=50,
        help="Zero-based first row; 50 selects 1960Q1 from the 1947Q3 archive.",
    )
    parser.add_argument("--sw07-sample-periods", type=int, default=180)
    parser.add_argument(
        "--sw07-initialization",
        choices=("dynare", "prior_mean"),
        default="dynare",
        help="Initialization basin for the empirical SW07 run.",
    )
    parser.add_argument(
        "--steady-state-mode",
        choices=("fixed", "solve"),
        default="fixed",
        help="Use the reference steady state or solve it at every parameter draw.",
    )
    parser.add_argument("--steady-state-tol", type=float, default=1.0e-12)
    parser.add_argument("--steady-state-max-iter", type=int, default=200)
    parser.add_argument(
        "--initial-covariance-strategy",
        choices=("theoretical", "diagonal"),
        default="theoretical",
    )
    parser.add_argument(
        "--presample-periods",
        type=int,
        default=None,
        help="Likelihood presample periods; defaults to 6 for sw07_paper and 0 otherwise.",
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "vectorized", "sequential"),
        default="vectorized",
    )
    parser.add_argument("--kernel", choices=("nuts", "hmc"), default="nuts")
    parser.add_argument(
        "--hmc-num-steps",
        type=int,
        default=8,
        help="Fixed leapfrog step count used when --kernel hmc.",
    )
    parser.add_argument(
        "--hmc-step-size",
        type=float,
        default=1.0,
        help="Initial HMC step size used when --kernel hmc.",
    )
    parser.add_argument(
        "--no-adapt-step-size",
        action="store_true",
        help="Disable HMC/NUTS step-size adaptation during warmup.",
    )
    parser.add_argument(
        "--no-adapt-mass-matrix",
        action="store_true",
        help="Disable HMC/NUTS mass-matrix adaptation during warmup.",
    )
    parser.add_argument("--host-device-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--platform", choices=("cpu", "gpu", "cuda"), default=None)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument(
        "--qme-algorithm",
        choices=("schur", "schur_gpu", "doubling"),
        default="schur_gpu",
    )
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--dense-mass", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--store-posterior-draws",
        action="store_true",
        help=(
            "Store chain-by-draw parameter arrays in the JSON artifact so diagnostics "
            "and posterior summaries can be independently recomputed."
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Compile and time the likelihood/gradient, then exit before MCMC.",
    )
    parser.add_argument("--preflight-reps", type=int, default=0)
    parser.add_argument(
        "--parameters-are-resolved",
        action="store_true",
        help=(
            "Use a parameter vector that is pre-resolved against active calibration "
            "equations and skip calibration resolution inside the JAX likelihood."
        ),
    )
    parser.add_argument(
        "--skip-parameter-bounds",
        action="store_true",
        help=(
            "Skip the outer model bounds branch. Use only when NumPyro priors or "
            "parameter transforms already enforce support."
        ),
    )
    parser.add_argument("--failure-value", type=float, default=-1.0e12)
    parser.add_argument("--schur-support-draws", type=int, default=0)
    parser.add_argument("--schur-acceptance-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stage timing and heartbeat logs while compiling/sampling.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval for verbose long-running stages; set 0 to disable.",
    )
    parser.add_argument(
        "--show-dtype-warnings",
        action="store_false",
        dest="suppress_dtype_warnings",
        help="Show JAX dtype truncation warnings during true-FP32 runs.",
    )
    parser.set_defaults(suppress_dtype_warnings=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.presample_periods is None:
        args.presample_periods = 6 if args.preset == "sw07_paper" else 0
    if args.preset == "sw07_paper" and args.parameters not in {
        "sw07_safe_15",
        "sw07_paper_36",
    }:
        raise ValueError(
            "--preset sw07_paper always estimates the published 36-parameter block; "
            "do not combine it with a custom --parameters selection."
        )
    if args.preset == "sw07_paper":
        args.parameters = "sw07_paper_36"
    if args.preflight_only:
        args.preflight = True
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    print(json.dumps(result["throughput"], indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
