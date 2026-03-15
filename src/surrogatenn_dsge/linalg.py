from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

LyapunovAlgorithm = Literal["doubling", "direct"]


class LyapunovResult(NamedTuple):
    solution: jax.Array
    converged: jax.Array
    iterations: jax.Array
    relative_residual: jax.Array


@dataclass(frozen=True)
class LyapunovOutcome:
    solution: jax.Array
    converged: bool
    iterations: int
    relative_residual: float
    algorithm: LyapunovAlgorithm
    fallback_used: bool


class _DoublingState(NamedTuple):
    current: jax.Array
    a_power: jax.Array
    iterations: jax.Array
    rel_change: jax.Array
    done: jax.Array


def _cast_matrix_inputs(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
) -> tuple[jax.Array, jax.Array]:
    a_arr = jnp.asarray(a)
    c_arr = jnp.asarray(c)
    if a_arr.ndim != 2 or c_arr.ndim != 2:
        raise ValueError("A and C must both be rank-2 matrices.")
    if a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError(f"A must be square, got shape {a_arr.shape}.")
    if c_arr.shape != a_arr.shape:
        raise ValueError(
            f"C must have the same shape as A, got A={a_arr.shape} and C={c_arr.shape}."
        )
    dtype = jnp.result_type(a_arr, c_arr, jnp.float64)
    return jnp.asarray(a_arr, dtype=dtype), jnp.asarray(c_arr, dtype=dtype)


def _validate_finite_inputs(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
) -> None:
    a_np = np.asarray(a)
    c_np = np.asarray(c)
    if not np.isfinite(a_np).all() or not np.isfinite(c_np).all():
        raise ValueError("A and C must contain only finite values.")


def discrete_lyapunov_residual(a: jax.Array, x: jax.Array, c: jax.Array) -> jax.Array:
    residual = a @ x @ a.T + c - x
    denom = jnp.maximum(jnp.linalg.norm(x), jnp.finfo(x.dtype).eps)
    return jnp.linalg.norm(residual) / denom


def solve_discrete_lyapunov_direct(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    acceptance_tol: float = 1e-12,
) -> LyapunovResult:
    a_arr, c_arr = _cast_matrix_inputs(a, c)
    n = a_arr.shape[0]
    system = jnp.eye(n * n, dtype=a_arr.dtype) - jnp.kron(a_arr, a_arr)
    rhs = jnp.ravel(c_arr.T)
    solution_vec = jnp.linalg.solve(system, rhs)
    solution = jnp.reshape(solution_vec, (n, n)).T
    rel_residual = discrete_lyapunov_residual(a_arr, solution, c_arr)
    converged = rel_residual < acceptance_tol
    return LyapunovResult(solution, converged, jnp.asarray(0), rel_residual)


def solve_discrete_lyapunov_doubling(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    tol: float = 1e-14,
    acceptance_tol: float = 1e-12,
    max_iter: int = 500,
) -> LyapunovResult:
    a_arr, c_arr = _cast_matrix_inputs(a, c)
    tol_arr = jnp.asarray(tol, dtype=a_arr.dtype)

    def body(i: int, state: _DoublingState) -> _DoublingState:
        next_current = state.a_power @ state.current @ state.a_power.T + state.current
        next_a_power = state.a_power @ state.a_power

        check_now = (i + 1) % 2 == 0
        norm_current = jnp.linalg.norm(state.current)
        norm_next = jnp.linalg.norm(next_current)
        denom = jnp.maximum(norm_current, norm_next)
        normdiff = jnp.linalg.norm(next_current - state.current)
        rel_change = jnp.where(denom > 0, normdiff / denom, normdiff)
        stop_now = jnp.logical_and(
            jnp.asarray(check_now),
            jnp.logical_or(~jnp.isfinite(normdiff), rel_change < tol_arr),
        )

        freeze = jnp.logical_or(state.done, stop_now)
        current = jnp.where(freeze, state.current, next_current)
        a_power = jnp.where(freeze, state.a_power, next_a_power)
        iterations = jnp.where(
            state.done,
            state.iterations,
            jnp.where(stop_now, jnp.asarray(i + 1), state.iterations),
        )
        rel_change_out = jnp.where(
            state.done,
            state.rel_change,
            jnp.where(jnp.asarray(check_now), rel_change, state.rel_change),
        )
        return _DoublingState(
            current=current,
            a_power=a_power,
            iterations=iterations,
            rel_change=rel_change_out,
            done=freeze,
        )

    init = _DoublingState(
        current=c_arr,
        a_power=a_arr,
        iterations=jnp.asarray(max_iter),
        rel_change=jnp.asarray(jnp.inf, dtype=a_arr.dtype),
        done=jnp.asarray(False),
    )
    final = lax.fori_loop(0, max_iter, body, init)
    rel_residual = discrete_lyapunov_residual(a_arr, final.current, c_arr)
    converged = rel_residual < acceptance_tol
    return LyapunovResult(final.current, converged, final.iterations, rel_residual)


def solve_discrete_lyapunov(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    algorithm: LyapunovAlgorithm = "doubling",
    tol: float = 1e-14,
    acceptance_tol: float = 1e-12,
    max_iter: int = 500,
    fallback_to_direct: bool = True,
) -> LyapunovOutcome:
    if algorithm not in ("doubling", "direct"):
        raise ValueError(
            f"Unsupported Lyapunov algorithm {algorithm!r}. "
            "Only 'doubling' and 'direct' are implemented in this port."
        )
    _validate_finite_inputs(a, c)

    if algorithm == "doubling":
        result = solve_discrete_lyapunov_doubling(
            a,
            c,
            tol=tol,
            acceptance_tol=acceptance_tol,
            max_iter=max_iter,
        )
    else:
        result = solve_discrete_lyapunov_direct(
            a,
            c,
            acceptance_tol=acceptance_tol,
        )

    algorithm_used: LyapunovAlgorithm = algorithm
    fallback_used = False

    if fallback_to_direct and algorithm == "doubling" and not bool(np.asarray(result.converged)):
        direct_result = solve_discrete_lyapunov_direct(
            a,
            c,
            acceptance_tol=acceptance_tol,
        )
        if float(np.asarray(direct_result.relative_residual)) < float(
            np.asarray(result.relative_residual)
        ):
            result = direct_result
            algorithm_used = "direct"
            fallback_used = True

    return LyapunovOutcome(
        solution=result.solution,
        converged=bool(np.asarray(result.converged)),
        iterations=int(np.asarray(result.iterations)),
        relative_residual=float(np.asarray(result.relative_residual)),
        algorithm=algorithm_used,
        fallback_used=fallback_used,
    )


def solve_lyapunov_equation(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    **kwargs: object,
) -> LyapunovOutcome:
    return solve_discrete_lyapunov(a, c, **kwargs)
