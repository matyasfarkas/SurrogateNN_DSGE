from __future__ import annotations

from pathlib import Path

import pytest

from surrogatenn_dsge import parse_macro_model


_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _ROOT / "SurrogateNN_Estimation.jl"
_UPSTREAM_MODEL_FILES = tuple(
    sorted((_UPSTREAM_ROOT / "models").glob("*.jl"))
    + sorted((_UPSTREAM_ROOT / "test" / "models").glob("*.jl"))
)
_KNOWN_BROKEN_TESTQIPF = _UPSTREAM_ROOT / "models" / "testqipf.jl"
_PARSE_COMPAT_MODELS = tuple(
    path for path in _UPSTREAM_MODEL_FILES if path != _KNOWN_BROKEN_TESTQIPF
)


@pytest.mark.parametrize("model_path", _PARSE_COMPAT_MODELS, ids=lambda path: path.stem)
def test_upstream_model_source_parses_with_macro_modelling_nomenclature(
    model_path: Path,
) -> None:
    model = parse_macro_model(model_path.read_text())
    assert model.name
    assert len(model.equations) > 0


def test_upstream_testqipf_fixture_is_flagged_for_gamm_typo() -> None:
    with pytest.raises(ValueError, match="GAMM"):
        parse_macro_model(_KNOWN_BROKEN_TESTQIPF.read_text())
