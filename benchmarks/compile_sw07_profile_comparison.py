from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from validate_sw07_paper_results import _preflight_parity


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _timing(preflight: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    first_call_s = _positive_float(preflight.get(f"{prefix}_first_call_s"))
    steady = preflight.get(f"{prefix}_steady") or {}
    median_s = _positive_float(steady.get("median_s"))
    return {
        "first_call_s": first_call_s,
        "steady_median_s": median_s,
        "steady_reps": int(steady.get("reps") or 0),
        "steady_evaluations_per_second": None if median_s is None else 1.0 / median_s,
        "steady_evaluations_per_hour": None if median_s is None else 3600.0 / median_s,
    }


def _gpu_name(runtime: Mapping[str, Any]) -> str | None:
    match = re.search(r"NVIDIA\s+[^|\n]+", str(runtime.get("nvidia_smi") or ""))
    return match.group(0).strip() if match else None


def _backend_profile(result: Mapping[str, Any]) -> dict[str, Any]:
    runtime = result.get("runtime") or {}
    preflight = result.get("preflight") or {}
    return {
        "backend": runtime.get("jax_default_backend"),
        "devices": runtime.get("jax_devices") or [],
        "gpu_name": _gpu_name(runtime),
        "platform": runtime.get("platform"),
        "processor": runtime.get("processor"),
        "python": runtime.get("python"),
        "jax_version": runtime.get("jax_version"),
        "numpyro_version": runtime.get("numpyro_version"),
        "jax_enable_x64": runtime.get("jax_enable_x64"),
        "loglikelihood": preflight.get("loglikelihood"),
        "value": _timing(preflight, "value"),
        "gradient": _timing(preflight, "gradient"),
    }


def _speedup(cpu_s: Any, gpu_s: Any) -> float | None:
    cpu = _positive_float(cpu_s)
    gpu = _positive_float(gpu_s)
    return None if cpu is None or gpu is None else cpu / gpu


def compile_profile(
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    *,
    cpu_path: Path,
    gpu_path: Path,
) -> dict[str, Any]:
    parity = _preflight_parity(cpu, gpu)
    if not parity["passed"]:
        raise ValueError(f"SW07 CPU/GPU parity gate failed: {parity}")

    cpu_profile = _backend_profile(cpu)
    gpu_profile = _backend_profile(gpu)
    benchmark = gpu.get("benchmark") or {}
    provenance = benchmark.get("data_provenance") or {}
    comparison = {
        "value_first_call_speedup_cpu_over_gpu": _speedup(
            cpu_profile["value"]["first_call_s"], gpu_profile["value"]["first_call_s"]
        ),
        "value_steady_speedup_cpu_over_gpu": _speedup(
            cpu_profile["value"]["steady_median_s"], gpu_profile["value"]["steady_median_s"]
        ),
        "gradient_first_call_speedup_cpu_over_gpu": _speedup(
            cpu_profile["gradient"]["first_call_s"],
            gpu_profile["gradient"]["first_call_s"],
        ),
        "gradient_steady_speedup_cpu_over_gpu": _speedup(
            cpu_profile["gradient"]["steady_median_s"],
            gpu_profile["gradient"]["steady_median_s"],
        ),
    }
    return {
        "schema_version": 1,
        "scope": "paper-matched linearized Smets-Wouters (2007) preflight",
        "model": {
            "name": benchmark.get("model_name"),
            "preset": benchmark.get("preset"),
            "variables": benchmark.get("n_vars"),
            "shocks": benchmark.get("n_exo"),
            "observables": benchmark.get("n_observables"),
            "estimated_parameters": benchmark.get("parameter_count"),
            "sample_periods": benchmark.get("periods"),
            "presample_periods": benchmark.get("presample_periods"),
            "dtype": benchmark.get("dtype"),
            "steady_state_mode": benchmark.get("steady_state_mode"),
            "initial_covariance_strategy": benchmark.get("initial_covariance_strategy"),
            "data_sha256": provenance.get("sha256"),
            "model_source_sha256": provenance.get("model_source_sha256"),
        },
        "parity": parity,
        "profiles": {"cpu": cpu_profile, "gpu": gpu_profile},
        "comparison": comparison,
        "artifacts": {
            "cpu_path": str(cpu_path),
            "cpu_sha256": _sha256(cpu_path),
            "gpu_path": str(gpu_path),
            "gpu_sha256": _sha256(gpu_path),
        },
        "interpretation": (
            "First-call timings include JAX/XLA compilation. Steady timings exclude "
            "compilation. Speedup is CPU seconds divided by GPU seconds; values above "
            "one favor the GPU. These are exact likelihood/gradient microprofiles, not "
            "posterior effective-sample throughput."
        ),
    }


def _format(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def render_markdown(profile: Mapping[str, Any]) -> str:
    cpu = profile["profiles"]["cpu"]
    gpu = profile["profiles"]["gpu"]
    comparison = profile["comparison"]
    parity = profile["parity"]
    rows = [
        (
            "Likelihood first call (s)",
            cpu["value"]["first_call_s"],
            gpu["value"]["first_call_s"],
            comparison["value_first_call_speedup_cpu_over_gpu"],
        ),
        (
            "Likelihood steady median (s)",
            cpu["value"]["steady_median_s"],
            gpu["value"]["steady_median_s"],
            comparison["value_steady_speedup_cpu_over_gpu"],
        ),
        (
            "Gradient first call (s)",
            cpu["gradient"]["first_call_s"],
            gpu["gradient"]["first_call_s"],
            comparison["gradient_first_call_speedup_cpu_over_gpu"],
        ),
        (
            "Gradient steady median (s)",
            cpu["gradient"]["steady_median_s"],
            gpu["gradient"]["steady_median_s"],
            comparison["gradient_steady_speedup_cpu_over_gpu"],
        ),
    ]
    table = "\n".join(
        f"| {label} | {_format(cpu_value)} | {_format(gpu_value)} | {_format(speedup, 2)}× |"
        for label, cpu_value, gpu_value, speedup in rows
    )
    model = profile["model"]
    return f"""# SW07 matched CPU/GPU profile

This comparison uses the same `{model['dtype']}` likelihood, {model['sample_periods']}
observations, {model['estimated_parameters']} estimated parameters, solved steady
states, and theoretical Kalman initialization on both backends.

| Metric | CPU | GPU | CPU/GPU speedup |
|---|---:|---:|---:|
{table}

## Numerical gate

- CPU/GPU likelihood absolute error: `{parity['loglikelihood_absolute_error']:.3e}`
- Maximum gradient-component absolute error: `{parity['gradient_max_absolute_error']:.3e}`
- GPU/Dynare likelihood absolute error: `{parity['gpu_dynare_absolute_error']:.3e}`
- All 36 gradients finite and metadata matched: `{str(parity['passed']).lower()}`

## Runtime provenance

- CPU: `{cpu['platform']}`, JAX `{cpu['jax_version']}`, NumPyro `{cpu['numpyro_version']}`
- GPU: `{gpu['gpu_name'] or gpu['devices']}`, JAX `{gpu['jax_version']}`, NumPyro `{gpu['numpyro_version']}`

First-call timings include compilation. Steady timings exclude compilation. This
is an exact likelihood/gradient microprofile; posterior ESS per second is reported
separately after the sampling runs pass convergence gates. The CPU and GPU runs
use the recorded software stacks above, so their timing ratio is an observational
deployment comparison rather than a controlled hardware-only ablation.
"""


def render_latex(profile: Mapping[str, Any]) -> str:
    cpu = profile["profiles"]["cpu"]
    gpu = profile["profiles"]["gpu"]
    comparison = profile["comparison"]
    parity = profile["parity"]
    rows = [
        (
            "Likelihood first call",
            cpu["value"]["first_call_s"],
            gpu["value"]["first_call_s"],
            comparison["value_first_call_speedup_cpu_over_gpu"],
        ),
        (
            "Likelihood steady median",
            cpu["value"]["steady_median_s"],
            gpu["value"]["steady_median_s"],
            comparison["value_steady_speedup_cpu_over_gpu"],
        ),
        (
            "Gradient first call",
            cpu["gradient"]["first_call_s"],
            gpu["gradient"]["first_call_s"],
            comparison["gradient_first_call_speedup_cpu_over_gpu"],
        ),
        (
            "Gradient steady median",
            cpu["gradient"]["steady_median_s"],
            gpu["gradient"]["steady_median_s"],
            comparison["gradient_steady_speedup_cpu_over_gpu"],
        ),
    ]
    body = "\n".join(
        f"{label} & {_format(cpu_value)} & {_format(gpu_value)} & {_format(speedup, 2)} \\\\"
        for label, cpu_value, gpu_value, speedup in rows
    )
    likelihood_error = _latex_scientific(parity["loglikelihood_absolute_error"])
    gradient_error = _latex_scientific(parity["gradient_max_absolute_error"])
    return f"""% Generated by benchmarks/compile_sw07_profile_comparison.py. Do not edit.
\\begin{{table}}[t]
\\centering
\\caption{{Matched SW07 CPU--GPU likelihood and gradient profile}}
\\label{{tab:sw07-cpu-gpu-profile}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Metric (seconds) & CPU & GPU & CPU/GPU \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.4em}}
\\parbox{{0.96\\textwidth}}{{\\footnotesize\\textit{{Notes:}} Both backends evaluate the same float64, 36-parameter likelihood on the 180-quarter sample, solve the steady state at every draw, and use theoretical Kalman initialization. First-call times include JAX/XLA compilation; steady medians exclude compilation. CPU/GPU is the ratio of elapsed times, so values above one favor the GPU. The likelihood discrepancy is {likelihood_error}; the maximum gradient-component discrepancy is {gradient_error}. The CPU environment is macOS arm64 with JAX {cpu['jax_version']} and NumPyro {cpu['numpyro_version']}; the GPU environment is Linux x86\\_64 on an RTX 5090 D with JAX {gpu['jax_version']} and NumPyro {gpu['numpyro_version']}. The timing ratio is therefore an observational deployment comparison, not a controlled hardware-only ablation. These microprofiles do not measure effective samples per second.}}
\\end{{table}}
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a matched SW07 CPU/GPU profile.")
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-latex", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    profile = compile_profile(_load(args.cpu), _load(args.gpu), cpu_path=args.cpu, gpu_path=args.gpu)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_latex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(render_markdown(profile))
    args.output_latex.write_text(render_latex(profile))
    print(json.dumps(profile["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
