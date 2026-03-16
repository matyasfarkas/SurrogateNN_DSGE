from __future__ import annotations

import math
import numpy as np
import scipy.special as scipy_special

from surrogatenn_dsge import parse_macro_model, solve_steady_state


SPECIAL_FUNCTION_SOURCE = """
@model special_functions begin
    xcdf[0] = normcdf(x[0])
    xpdf[0] = normpdf(x[0])
    xpnorm[0] = pnorm(x[0])
    xdnorm[0] = dnorm(x[0])
    xlogpdf[0] = normlogpdf(x[0])
    xinv[0] = erfcinv(x[0]) * gamma + eta
    x[0] = level
end

@parameters special_functions begin
    gamma = 0.99
    eta = 0.01
    level = 0.25
end
"""


def test_special_function_aliases_parse_and_evaluate_in_steady_state() -> None:
    model = parse_macro_model(SPECIAL_FUNCTION_SOURCE)
    result = solve_steady_state(model)
    values = dict(
        zip(
            model.steady_state_names,
            np.asarray(result.base_steady_state, dtype=np.float64).tolist(),
        )
    )

    assert model.parameter_names == ("eta", "gamma", "level")
    assert result.converged
    np.testing.assert_allclose(values["x"], 0.25, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        values["xcdf"],
        0.5 * (1.0 + math.erf(0.25 / np.sqrt(2.0))),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        values["xpdf"],
        np.exp(-(0.25**2) / 2.0) / np.sqrt(2.0 * np.pi),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(values["xpnorm"], values["xcdf"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(values["xdnorm"], values["xpdf"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        values["xlogpdf"],
        -(0.25**2) / 2.0 - np.log(np.sqrt(2.0 * np.pi)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        values["xinv"],
        float(scipy_special.erfcinv(np.asarray(0.25, dtype=np.float64))) * 0.99 + 0.01,
        rtol=1e-12,
        atol=1e-12,
    )
