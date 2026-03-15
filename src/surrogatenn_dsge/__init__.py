from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .linalg import (
    LyapunovOutcome,
    LyapunovResult,
    discrete_lyapunov_residual,
    solve_discrete_lyapunov,
    solve_discrete_lyapunov_direct,
    solve_discrete_lyapunov_doubling,
    solve_lyapunov_equation,
)

__all__ = [
    "LyapunovOutcome",
    "LyapunovResult",
    "discrete_lyapunov_residual",
    "solve_discrete_lyapunov",
    "solve_discrete_lyapunov_direct",
    "solve_discrete_lyapunov_doubling",
    "solve_lyapunov_equation",
]

