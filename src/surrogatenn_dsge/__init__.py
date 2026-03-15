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
from .statespace import (
    KalmanFilterResult,
    KalmanSmootherResult,
    LinearGaussianStateSpace,
    StateSpaceSimulation,
    build_linear_gaussian_state_space,
    kalman_filter,
    kalman_loglikelihood,
    kalman_loglikelihood_per_period,
    kalman_smoother,
    simulate_linear_gaussian_state_space,
)

__all__ = [
    "KalmanFilterResult",
    "KalmanSmootherResult",
    "LinearGaussianStateSpace",
    "LyapunovOutcome",
    "LyapunovResult",
    "StateSpaceSimulation",
    "build_linear_gaussian_state_space",
    "SylvesterOutcome",
    "SylvesterResult",
    "discrete_lyapunov_residual",
    "discrete_sylvester_residual",
    "kalman_filter",
    "kalman_loglikelihood",
    "kalman_loglikelihood_per_period",
    "kalman_smoother",
    "simulate_linear_gaussian_state_space",
    "solve_discrete_lyapunov",
    "solve_discrete_lyapunov_direct",
    "solve_discrete_lyapunov_doubling",
    "solve_discrete_sylvester",
    "solve_discrete_sylvester_direct",
    "solve_discrete_sylvester_doubling",
    "solve_lyapunov_equation",
    "solve_sylvester_equation",
]
