from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any, Sequence

from . import __version__


MappingLike = dict[str, Any]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _device_payload(device: Any) -> dict[str, Any]:
    return {
        "id": getattr(device, "id", None),
        "platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
        "process_index": getattr(device, "process_index", None),
    }


def runtime_info() -> dict[str, Any]:
    import jax
    import numpy as np
    import scipy
    import sympy

    try:
        import numpyro
    except Exception:  # pragma: no cover - optional dependency
        numpyro_version = None
    else:
        numpyro_version = getattr(numpyro, "__version__", "unknown")

    devices = [_device_payload(device) for device in jax.devices()]
    return {
        "package": "surrogatenn-dsge",
        "version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": devices,
        "numpy_version": getattr(np, "__version__", "unknown"),
        "numpyro_version": numpyro_version,
        "scipy_version": getattr(scipy, "__version__", "unknown"),
        "sympy_version": getattr(sympy, "__version__", "unknown"),
    }


def _print_payload(payload: MappingLike, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


def run_smoke(*, dtype: str, require_gpu: bool) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from .statespace import build_linear_gaussian_state_space, kalman_loglikelihood

    devices = jax.devices()
    has_gpu = any(getattr(device, "platform", None) == "gpu" for device in devices)
    if require_gpu and not has_gpu:
        raise RuntimeError("No JAX GPU device is visible.")

    dtype_obj = jnp.float32 if dtype == "float32" else jnp.float64
    state_space = build_linear_gaussian_state_space(
        jnp.asarray([[0.8]], dtype=dtype_obj),
        jnp.asarray([[0.04]], dtype=dtype_obj),
        jnp.asarray([[1.0]], dtype=dtype_obj),
        jnp.asarray([[0.01]], dtype=dtype_obj),
        initial_mean=jnp.asarray([0.0], dtype=dtype_obj),
        initial_covariance=jnp.asarray([[0.2]], dtype=dtype_obj),
    )
    observations = jnp.zeros((1, 8), dtype=dtype_obj)
    value_fn = jax.jit(lambda y: kalman_loglikelihood(state_space, y))
    loglikelihood = value_fn(observations)
    loglikelihood.block_until_ready()
    return {
        "status": "ok",
        "dtype": dtype,
        "default_backend": jax.default_backend(),
        "has_gpu": has_gpu,
        "loglikelihood": float(loglikelihood),
        "devices": [_device_payload(device) for device in devices],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogatenn-dsge",
        description="Package diagnostics for the SurrogateNN DSGE Python/JAX port.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info = subparsers.add_parser("info", help="Print package and JAX runtime information.")
    info.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    smoke = subparsers.add_parser("smoke", help="Run a tiny JIT-compiled Kalman smoke test.")
    smoke.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    smoke.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    smoke.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless at least one JAX GPU device is visible.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["info"]
    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.command == "info":
        _print_payload(runtime_info(), as_json=bool(args.json))
        return 0
    if args.command == "smoke":
        _print_payload(
            run_smoke(dtype=str(args.dtype), require_gpu=bool(args.require_gpu)),
            as_json=bool(args.json),
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
