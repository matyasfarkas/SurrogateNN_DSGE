#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from surrogatenn_dsge import (  # noqa: E402
    compare_posterior_draws,
    load_posterior_draws_npz,
    summarize_posterior_draws,
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in report["parameters"].items():
        rows.append(
            {
                "parameter": name,
                "left_mean": metrics["left_mean"],
                "right_mean": metrics["right_mean"],
                "mean_diff": metrics["mean_diff"],
                "standardized_mean_diff": metrics["standardized_mean_diff"],
                "left_std": metrics["left_std"],
                "right_std": metrics["right_std"],
                "std_ratio": metrics["std_ratio"],
                "ks_distance": metrics["ks_distance"],
                "quantile_wasserstein": metrics["quantile_wasserstein"],
                "quantile_wasserstein_over_left_std": metrics[
                    "quantile_wasserstein_over_left_std"
                ],
                "q05_diff": metrics["quantile_diff"].get("0.05"),
                "q50_diff": metrics["quantile_diff"].get("0.5"),
                "q95_diff": metrics["quantile_diff"].get("0.95"),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_files(args: argparse.Namespace) -> dict[str, Any]:
    left = load_posterior_draws_npz(args.left)
    right = load_posterior_draws_npz(args.right)
    report = compare_posterior_draws(
        left["samples"],
        left["parameter_names"],
        right["samples"],
        right["parameter_names"],
        left_label=args.left_label,
        right_label=args.right_label,
    )
    report["left"] = {
        "path": left["path"],
        "metadata": left["metadata"],
        "summary": summarize_posterior_draws(left["samples"], left["parameter_names"]),
    }
    report["right"] = {
        "path": right["path"],
        "metadata": right["metadata"],
        "summary": summarize_posterior_draws(right["samples"], right["parameter_names"]),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two posterior draw archives saved by the DSGE benchmarks."
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--left-label", default="linear_rom1")
    parser.add_argument("--right-label", default="surrogate_sep_resnn")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = compare_files(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    rows = _comparison_rows(report)
    if args.csv_output is not None:
        _write_csv(args.csv_output, rows)
    print(
        json.dumps(
            {
                "left_label": report["left_label"],
                "right_label": report["right_label"],
                "common_parameters": report["common_parameters"],
                "max_abs_mean_diff": report["max_abs_mean_diff"],
                "max_ks_distance": report["max_ks_distance"],
                "mean_quantile_wasserstein": report["mean_quantile_wasserstein"],
                "output": str(args.output),
                "csv_output": None if args.csv_output is None else str(args.csv_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
