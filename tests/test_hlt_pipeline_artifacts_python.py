from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np

from surrogatenn_dsge import (
    FrozenMLP,
    NormStats,
    SurrogateDataset,
    build_hlt_pipeline_artifact_payload,
    build_surrogate_residual_arrays_jax,
    load_hlt_pipeline_artifact_payload,
    predict_frozen_batch,
    save_hlt_pipeline_artifact_payload,
)


def _load_validator() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "benchmarks" / "validate_hlt_pipeline_parity.py"
    spec = importlib.util.spec_from_file_location("validate_hlt_pipeline_parity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_dataset() -> SurrogateDataset:
    X = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.1, 0.3, 0.5],
            [1.5, 1.5, 1.5],
        ],
        dtype=np.float64,
    )
    Y_rom = np.asarray(
        [
            [1.0, 1.1, 1.2],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    residual = np.asarray(
        [
            [0.01, 0.02, 0.03],
            [-0.02, -0.01, 0.00],
        ],
        dtype=np.float64,
    )
    return SurrogateDataset(
        X=X,
        Y=Y_rom + residual,
        Y_rom=Y_rom,
        theta=np.asarray([[1.5]], dtype=np.float64),
        theta_ids=np.asarray([0, 0, 0], dtype=np.int64),
        period_ids=np.asarray([0, 1, 2], dtype=np.int64),
        theta_success=np.asarray([True]),
        theta_stable_periods=np.asarray([3], dtype=np.int64),
        target_mode="fom_obs",
        input_names=("state", "shock", "theta"),
        output_names=("y", "pi"),
        theta_names=("rho",),
    )


def _toy_mlp() -> FrozenMLP:
    return FrozenMLP(
        W1=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        b1=np.asarray([0.0, 0.0], dtype=np.float64),
        W2=np.asarray([[0.1, 0.2], [-0.2, 0.3]], dtype=np.float64),
        b2=np.asarray([0.01, -0.02], dtype=np.float64),
        W3=None,
        b3=None,
        norm=NormStats(
            mu_x=np.zeros((3,), dtype=np.float64),
            sigma_x=np.ones((3,), dtype=np.float64),
            mu_y=np.zeros((2,), dtype=np.float64),
            sigma_y=np.ones((2,), dtype=np.float64),
        ),
        d_in=3,
        d_out=2,
        activation="tanh",
    )


def test_python_hlt_artifact_payload_matches_validator_schema(tmp_path: Path) -> None:
    dataset = _toy_dataset()
    frozen = _toy_mlp()
    payload = build_hlt_pipeline_artifact_payload(
        case="toy_hlt",
        dataset=dataset,
        surrogate=frozen,
        support_selected_indices=[11, 12, 13],
        rom_states=np.asarray([[0.0, 0.1, 0.2]]),
        rom_observations=dataset.Y_rom,
        rom_shocks=np.asarray([[0.0, 0.1, -0.1]]),
        gate_e_stat=[0.0, 2.0, 0.1],
        gate_f_stat=[0.1, 0.3, 1.1],
        gate_probs=[0.0001, 0.95, 0.7],
        gate_mask=[False, True, True],
        switching_loglik_per_period=[-1.0, -2.0, -1.5],
        posterior_log_density=-12.5,
    )

    artifacts = payload["cases"]["toy_hlt"]["artifacts"]
    np.testing.assert_allclose(artifacts["sep"]["residual_labels"], dataset.Y - dataset.Y_rom)
    np.testing.assert_allclose(artifacts["surrogate"]["predictions"], predict_frozen_batch(frozen, dataset.X))
    assert artifacts["support"]["selected_indices"] == [11, 12, 13]

    path = save_hlt_pipeline_artifact_payload(tmp_path / "python_pipeline_artifacts.json", payload)
    loaded = load_hlt_pipeline_artifact_payload(path)
    validator = _load_validator()
    checks = validator.validate_hlt_pipeline_parity(
        julia_payload=loaded,
        python_payload=loaded,
    )
    assert checks[0].passed


def test_python_hlt_artifact_payload_compacts_batched_arrays_by_sample_mask() -> None:
    theta = np.asarray([[1.5, 1.6]], dtype=np.float64)
    states = np.asarray(
        [
            [[0.0, 0.1], [0.2, 0.3]],
            [[1.0, 1.1], [1.2, 1.3]],
        ],
        dtype=np.float64,
    )
    shocks = np.asarray(
        [
            [[0.01], [0.02]],
            [[0.03], [0.04]],
        ],
        dtype=np.float64,
    )
    rom_obs = states[:, :, :1]
    rom_next = states + 0.1
    fom_obs = rom_obs + 0.01
    fom_next = rom_next + 0.02
    arrays = build_surrogate_residual_arrays_jax(
        states,
        shocks,
        theta,
        rom_obs,
        rom_next,
        fom_obs,
        fom_next,
        target_mode="fom_full",
    )
    arrays = arrays._replace(sample_mask=np.asarray([True, False, True, False]))

    payload = build_hlt_pipeline_artifact_payload(case="batched_hlt", dataset=arrays)
    artifacts = payload["cases"]["batched_hlt"]["artifacts"]

    assert np.asarray(artifacts["support"]["features"]).shape == (4, 2)
    assert artifacts["support"]["selected_indices"] == [1, 3]
    np.testing.assert_allclose(
        artifacts["sep"]["residual_labels"],
        np.asarray(artifacts["sep"]["targets"]) - np.asarray(artifacts["sep"]["rom_targets"]),
    )


def test_python_hlt_artifact_export_cli_roundtrip(tmp_path: Path) -> None:
    dataset = _toy_dataset()
    npz_path = tmp_path / "dataset.npz"
    out_path = tmp_path / "python_pipeline_artifacts.json"
    np.savez(
        npz_path,
        X=dataset.X,
        Y=dataset.Y,
        Y_rom=dataset.Y_rom,
        theta_ids=dataset.theta_ids,
        period_ids=dataset.period_ids,
        sample_mask=np.asarray([True, False, True]),
    )

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "export_python_hlt_pipeline_artifacts.py"),
            "--npz",
            str(npz_path),
            "--out",
            str(out_path),
            "--case",
            "toy_hlt",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    artifacts = payload["cases"]["toy_hlt"]["artifacts"]
    assert artifacts["support"]["selected_indices"] == [1, 3]
    assert np.asarray(artifacts["support"]["features"]).shape == (3, 2)
