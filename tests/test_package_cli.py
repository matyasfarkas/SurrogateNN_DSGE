from __future__ import annotations

import json

import surrogatenn_dsge as sdsge
from surrogatenn_dsge import cli


def test_package_exports_version() -> None:
    assert isinstance(sdsge.__version__, str)
    assert sdsge.__version__


def test_cli_info_json_reports_runtime(capsys) -> None:
    exit_code = cli.main(["info", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["package"] == "surrogatenn-dsge"
    assert payload["version"] == sdsge.__version__
    assert payload["jax_devices"]


def test_cli_smoke_json_runs_jitted_kalman(capsys) -> None:
    exit_code = cli.main(["smoke", "--dtype", "float64", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["dtype"] == "float64"
    assert isinstance(payload["loglikelihood"], float)
