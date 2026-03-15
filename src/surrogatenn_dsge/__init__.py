from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .linalg import (
    LyapunovOutcome,
    LyapunovResult,
    SylvesterOutcome,
    SylvesterResult,
    discrete_lyapunov_residual,
    discrete_sylvester_residual,
    solve_discrete_lyapunov,
    solve_discrete_lyapunov_direct,
    solve_discrete_lyapunov_doubling,
    solve_discrete_sylvester,
    solve_discrete_sylvester_direct,
    solve_discrete_sylvester_doubling,
    solve_lyapunov_equation,
    solve_sylvester_equation,
)

__all__ = [
    "LyapunovOutcome",
    "LyapunovResult",
    "SylvesterOutcome",
    "SylvesterResult",
    "discrete_lyapunov_residual",
    "discrete_sylvester_residual",
    "solve_discrete_lyapunov",
    "solve_discrete_lyapunov_direct",
    "solve_discrete_lyapunov_doubling",
    "solve_discrete_sylvester",
    "solve_discrete_sylvester_direct",
    "solve_discrete_sylvester_doubling",
    "solve_lyapunov_equation",
    "solve_sylvester_equation",
]
