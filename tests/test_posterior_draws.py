from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from surrogatenn_dsge import (
    compare_posterior_draws,
    load_posterior_draws_npz,
    save_posterior_draws_npz,
    summarize_posterior_draws,
)


_ROOT = Path(__file__).resolve().parents[1]
_COMPARE_SCRIPT = _ROOT / "benchmarks" / "compare_posterior_draws.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_posterior_draws_for_tests",
        _COMPARE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_COMPARE_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_posterior_draw_archive_roundtrips_metadata(tmp_path: Path) -> None:
    samples = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "draws.npz"

    saved = save_posterior_draws_npz(
        path,
        samples,
        ("alpha", "beta"),
        metadata={"source": "test", "chains": 2},
    )
    loaded = load_posterior_draws_npz(path)

    assert saved["post_warmup_draws"] == 4
    assert loaded["parameter_names"] == ("alpha", "beta")
    assert loaded["metadata"]["source"] == "test"
    np.testing.assert_allclose(loaded["samples"], samples)


def test_posterior_draw_summary_and_comparison_are_parameter_aligned() -> None:
    left = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float64,
    )
    right = left + np.asarray([[[0.5, -5.0]]], dtype=np.float64)

    summary = summarize_posterior_draws(left, ("alpha", "beta"))
    report = compare_posterior_draws(
        left,
        ("alpha", "beta"),
        right,
        ("alpha", "beta"),
        left_label="rom",
        right_label="surrogate",
    )

    assert summary["post_warmup_draws"] == 4
    assert summary["parameters"]["alpha"]["mean"] == 2.5
    assert report["left_label"] == "rom"
    assert report["right_label"] == "surrogate"
    assert report["common_parameters"] == ["alpha", "beta"]
    assert report["parameters"]["alpha"]["mean_diff"] == 0.5
    assert report["parameters"]["beta"]["mean_diff"] == -5.0
    assert report["max_ks_distance"] >= 0.0


def test_compare_script_writes_json_and_csv(tmp_path: Path) -> None:
    module = _load_compare_module()
    left = np.zeros((2, 2, 1), dtype=np.float64)
    right = np.ones((2, 2, 1), dtype=np.float64)
    left_path = tmp_path / "left.npz"
    right_path = tmp_path / "right.npz"
    output = tmp_path / "comparison.json"
    csv_output = tmp_path / "comparison.csv"
    save_posterior_draws_npz(left_path, left, ("theta",))
    save_posterior_draws_npz(right_path, right, ("theta",))

    module.main(
        [
            "--left",
            str(left_path),
            "--right",
            str(right_path),
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
        ]
    )

    assert output.exists()
    assert csv_output.exists()
    assert "theta" in csv_output.read_text(encoding="utf-8")
