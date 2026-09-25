#!/usr/bin/env python3
"""Export Python HLT pipeline artifacts to the canonical parity JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from surrogatenn_dsge import (  # noqa: E402
    build_hlt_pipeline_artifact_payload,
    load_surrogate_bundle,
    save_hlt_pipeline_artifact_payload,
)


def _load_npz_dataset(path: Path, *, first_column: int, max_columns: int) -> dict[str, Any]:
    if first_column < 1:
        raise ValueError("--first-column must be >= 1.")
    if max_columns < 0:
        raise ValueError("--max-columns must be >= 0.")
    with np.load(path, allow_pickle=False) as npz:
        payload = {key: np.asarray(npz[key]) for key in npz.files}
    if "X" not in payload or "Y" not in payload:
        raise ValueError("Input NPZ must contain at least X and Y arrays.")
    if "Y_rom" not in payload and "Y_rom1" not in payload and "rom_targets" not in payload:
        raise ValueError("Input NPZ must contain Y_rom, Y_rom1, or rom_targets.")
    n_samples = int(np.asarray(payload["X"]).shape[1])
    first = int(first_column) - 1
    if first >= n_samples:
        raise ValueError(f"--first-column={first_column} exceeds sample count {n_samples}.")
    last = n_samples if max_columns == 0 else min(n_samples, first + int(max_columns))
    cols = np.arange(first, last, dtype=np.int64)
    for key in ("X", "Y", "Y_rom", "Y_rom1", "rom_targets", "surrogate_predictions"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 2 and arr.shape[1] == n_samples:
                payload[key] = arr[:, cols]
    for key in ("theta_ids", "period_ids", "sample_mask", "selected_indices", "support_selected_indices"):
        if key in payload:
            arr = np.asarray(payload[key]).reshape(-1)
            if arr.shape[0] == n_samples:
                payload[key] = arr[cols]
    payload["selected_columns_one_based"] = cols + 1
    return payload


def _optional_npz_array(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _load_overlay(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Overlay JSON must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path, help="NPZ with X, Y, and Y_rom/Y_rom1 arrays.")
    parser.add_argument("--out", required=True, type=Path, help="Output canonical JSON artifact.")
    parser.add_argument("--case", default="hlt_pipeline")
    parser.add_argument("--surrogate-bundle", type=Path, help="Optional Python .npz surrogate bundle.")
    parser.add_argument("--overlay-json", type=Path, help="Optional JSON with rom/gate/likelihood/posterior arrays.")
    parser.add_argument("--first-column", type=int, default=1)
    parser.add_argument("--max-columns", type=int, default=0)
    parser.add_argument("--include-invalid", action="store_true", help="Do not filter by sample_mask when present.")
    args = parser.parse_args(argv)

    dataset = _load_npz_dataset(args.npz, first_column=args.first_column, max_columns=args.max_columns)
    overlay = _load_overlay(args.overlay_json)
    surrogate = load_surrogate_bundle(args.surrogate_bundle) if args.surrogate_bundle is not None else None
    payload = build_hlt_pipeline_artifact_payload(
        case=args.case,
        dataset=dataset,
        surrogate=surrogate,
        surrogate_predictions=_optional_npz_array(dataset, "surrogate_predictions", "R_pred"),
        support_selected_indices=_optional_npz_array(dataset, "support_selected_indices", "selected_indices", "selected_columns_one_based"),
        valid_only=not bool(args.include_invalid),
        rom_states=_optional_npz_array(overlay, "rom_states", "rom_state_path"),
        rom_observations=_optional_npz_array(overlay, "rom_observations", "rom_observation_path"),
        rom_shocks=_optional_npz_array(overlay, "rom_shocks", "rom_shock_path"),
        gate_e_stat=_optional_npz_array(overlay, "gate_e_stat", "e_stat"),
        gate_f_stat=_optional_npz_array(overlay, "gate_f_stat", "f_stat"),
        gate_probs=_optional_npz_array(overlay, "gate_probs"),
        gate_mask=_optional_npz_array(overlay, "gate_mask", "hard_mask"),
        switching_loglik_per_period=_optional_npz_array(overlay, "switching_loglik_per_period", "ll_switching"),
        posterior_log_density=overlay.get("posterior_log_density", overlay.get("log_density")),
        diagnostics={"npz_path": str(args.npz), "overlay_json": None if args.overlay_json is None else str(args.overlay_json)},
    )
    save_hlt_pipeline_artifact_payload(args.out, payload)
    print(f"Wrote Python HLT pipeline parity artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
