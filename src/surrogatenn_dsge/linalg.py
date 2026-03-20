from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse.linalg as scipy_sparse_linalg
from jax import lax

LyapunovAlgorithm = Literal[
    "doubling",
    "direct",
    "bartels_stewart",
    "bicgstab",
    "gmres",
    "dqgmres",
]
SylvesterAlgorithm = Literal["doubling", "direct", "bicgstab", "gmres", "dqgmres"]


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


class SylvesterResult(NamedTuple):
    solution: jax.Array
    converged: jax.Array
    iterations: jax.Array
    relative_residual: jax.Array


@dataclass(frozen=True)
class SylvesterOutcome:
    solution: jax.Array
    converged: bool
    iterations: int
    relative_residual: float
    algorithm: SylvesterAlgorithm
    fallback_used: bool


class _SylvesterDoublingState(NamedTuple):
    current: jax.Array
    a_power: jax.Array
    b_power: jax.Array
    iterations: jax.Array
    rel_change: jax.Array
    done: jax.Array


def _flatten_sylvester_matrix(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).T.reshape(-1)


def _unflatten_sylvester_matrix(
    values: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    rows, cols = shape
    return np.asarray(values, dtype=np.float64).reshape((cols, rows)).T


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


def _cast_sylvester_inputs(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    c_arr = jnp.asarray(c)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or c_arr.ndim != 2:
        raise ValueError("A, B, and C must all be rank-2 matrices.")
    if a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError(f"A must be square, got shape {a_arr.shape}.")
    if b_arr.shape[0] != b_arr.shape[1]:
        raise ValueError(f"B must be square, got shape {b_arr.shape}.")
    if c_arr.shape != (a_arr.shape[0], b_arr.shape[0]):
        raise ValueError(
            "C must have shape (A.shape[0], B.shape[0]), "
            f"got A={a_arr.shape}, B={b_arr.shape}, C={c_arr.shape}."
        )
    dtype = jnp.result_type(a_arr, b_arr, c_arr, jnp.float64)
    return (
        jnp.asarray(a_arr, dtype=dtype),
        jnp.asarray(b_arr, dtype=dtype),
        jnp.asarray(c_arr, dtype=dtype),
    )


def _validate_finite_sylvester_inputs(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
) -> None:
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    c_np = np.asarray(c)
    if not np.isfinite(a_np).all() or not np.isfinite(b_np).all() or not np.isfinite(c_np).all():
        raise ValueError("A, B, and C must contain only finite values.")


def discrete_lyapunov_residual(a: jax.Array, x: jax.Array, c: jax.Array) -> jax.Array:
    residual = a @ x @ a.T + c - x
    denom = jnp.maximum(jnp.linalg.norm(x), jnp.finfo(x.dtype).eps)
    return jnp.linalg.norm(residual) / denom


def discrete_sylvester_residual(
    a: jax.Array,
    x: jax.Array,
    b: jax.Array,
    c: jax.Array,
) -> jax.Array:
    residual = a @ x @ b + c - x
    denom = jnp.maximum(jnp.maximum(jnp.linalg.norm(x), jnp.linalg.norm(c)), jnp.finfo(x.dtype).eps)
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


def solve_discrete_lyapunov_bartels_stewart(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    acceptance_tol: float = 1e-12,
) -> LyapunovResult:
    a_arr, c_arr = _cast_matrix_inputs(a, c)
    solution_np = scipy_linalg.solve_discrete_lyapunov(
        np.asarray(a_arr, dtype=np.float64),
        np.asarray(c_arr, dtype=np.float64),
    )
    solution = jnp.asarray(solution_np, dtype=a_arr.dtype)
    rel_residual = discrete_lyapunov_residual(a_arr, solution, c_arr)
    converged = rel_residual < acceptance_tol
    return LyapunovResult(solution, converged, jnp.asarray(0), rel_residual)


def _solve_discrete_lyapunov_iterative(
    a: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    algorithm: Literal["bicgstab", "gmres", "dqgmres"],
    tol: float = 1e-14,
    acceptance_tol: float = 1e-12,
    max_iter: int = 500,
) -> LyapunovResult:
    a_arr, c_arr = _cast_matrix_inputs(a, c)
    a_np = np.asarray(a_arr, dtype=np.float64)
    c_np = np.asarray(c_arr, dtype=np.float64)
    n = a_np.shape[0]
    system = np.eye(n * n, dtype=np.float64) - np.kron(a_np, a_np)
    rhs = c_np.T.reshape(-1)
    iterations = 0

    def _callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    if algorithm == "bicgstab":
        solution_vec, info = scipy_sparse_linalg.bicgstab(
            system,
            rhs,
            rtol=tol,
            atol=0.0,
            maxiter=max_iter,
            callback=_callback,
        )
    else:
        solution_vec, info = scipy_sparse_linalg.gmres(
            system,
            rhs,
            rtol=tol,
            atol=0.0,
            restart=min(max_iter, max(20, n * n)),
            maxiter=max_iter,
            callback=_callback,
            callback_type="legacy",
        )

    solution = jnp.asarray(
        np.asarray(solution_vec, dtype=np.float64).reshape((n, n)).T,
        dtype=a_arr.dtype,
    )
    rel_residual = discrete_lyapunov_residual(a_arr, solution, c_arr)
    converged = (info == 0) & (rel_residual < acceptance_tol)
    iteration_count = iterations if iterations > 0 else max(int(info), 0)
    return LyapunovResult(
        solution,
        converged,
        jnp.asarray(iteration_count),
        rel_residual,
    )


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
    if algorithm not in (
        "doubling",
        "direct",
        "bartels_stewart",
        "bicgstab",
        "gmres",
        "dqgmres",
    ):
        raise ValueError(
            f"Unsupported Lyapunov algorithm {algorithm!r}. "
            "Supported algorithms are 'doubling', 'direct', 'bartels_stewart', "
            "'bicgstab', 'gmres', and 'dqgmres'."
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
    elif algorithm == "bartels_stewart":
        result = solve_discrete_lyapunov_bartels_stewart(
            a,
            c,
            acceptance_tol=acceptance_tol,
        )
    elif algorithm in ("bicgstab", "gmres", "dqgmres"):
        result = _solve_discrete_lyapunov_iterative(
            a,
            c,
            algorithm=algorithm,
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

    if (
        fallback_to_direct
        and algorithm in ("doubling", "bicgstab", "gmres", "dqgmres")
        and not bool(np.asarray(result.converged))
    ):
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


def solve_discrete_sylvester_direct(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    acceptance_tol: float = 1e-10,
) -> SylvesterResult:
    a_arr, b_arr, c_arr = _cast_sylvester_inputs(a, b, c)
    if initial_guess is None:
        guess = jnp.zeros_like(c_arr)
    else:
        guess = jnp.asarray(initial_guess, dtype=c_arr.dtype)
        if guess.shape != c_arr.shape:
            raise ValueError(
                f"initial_guess must match C's shape, got {guess.shape} and {c_arr.shape}."
            )

    residual_rhs = a_arr @ guess @ b_arr + c_arr - guess
    n, m = c_arr.shape
    system = jnp.eye(n * m, dtype=a_arr.dtype) - jnp.kron(b_arr.T, a_arr)
    rhs = jnp.ravel(residual_rhs.T)
    solution_vec = jnp.linalg.solve(system, rhs)
    delta = jnp.reshape(solution_vec, (m, n)).T
    solution = delta + guess
    rel_residual = discrete_sylvester_residual(a_arr, solution, b_arr, c_arr)
    converged = rel_residual < acceptance_tol
    return SylvesterResult(solution, converged, jnp.asarray(0), rel_residual)


def solve_discrete_sylvester_doubling(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    acceptance_tol: float = 1e-10,
    max_iter: int = 500,
) -> SylvesterResult:
    a_arr, b_arr, c_arr = _cast_sylvester_inputs(a, b, c)
    if initial_guess is None:
        guess = jnp.zeros_like(c_arr)
    else:
        guess = jnp.asarray(initial_guess, dtype=c_arr.dtype)
        if guess.shape != c_arr.shape:
            raise ValueError(
                f"initial_guess must match C's shape, got {guess.shape} and {c_arr.shape}."
            )

    initial_residual = a_arr @ guess @ b_arr + c_arr - guess
    tol_arr = jnp.asarray(tol, dtype=a_arr.dtype)

    def body(i: int, state: _SylvesterDoublingState) -> _SylvesterDoublingState:
        next_current = state.a_power @ state.current @ state.b_power + state.current
        next_a_power = state.a_power @ state.a_power
        next_b_power = state.b_power @ state.b_power

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
        b_power = jnp.where(freeze, state.b_power, next_b_power)
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
        return _SylvesterDoublingState(
            current=current,
            a_power=a_power,
            b_power=b_power,
            iterations=iterations,
            rel_change=rel_change_out,
            done=freeze,
        )

    init = _SylvesterDoublingState(
        current=initial_residual,
        a_power=a_arr,
        b_power=b_arr,
        iterations=jnp.asarray(max_iter),
        rel_change=jnp.asarray(jnp.inf, dtype=a_arr.dtype),
        done=jnp.asarray(False),
    )
    final = lax.fori_loop(0, max_iter, body, init)
    solution = final.current + guess
    rel_residual = discrete_sylvester_residual(a_arr, solution, b_arr, c_arr)
    converged = rel_residual < acceptance_tol
    return SylvesterResult(solution, converged, final.iterations, rel_residual)


def _solve_discrete_sylvester_iterative(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    algorithm: Literal["bicgstab", "gmres", "dqgmres"],
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    acceptance_tol: float = 1e-10,
    max_iter: int = 500,
) -> SylvesterResult:
    a_arr, b_arr, c_arr = _cast_sylvester_inputs(a, b, c)
    a_np = np.asarray(a_arr, dtype=np.float64)
    b_np = np.asarray(b_arr, dtype=np.float64)
    c_np = np.asarray(c_arr, dtype=np.float64)
    n, m = c_np.shape

    if initial_guess is None:
        guess_np = np.zeros_like(c_np)
    else:
        guess_np = np.asarray(initial_guess, dtype=np.float64)
        if guess_np.shape != c_np.shape:
            raise ValueError(
                f"initial_guess must match C's shape, got {guess_np.shape} and {c_np.shape}."
            )

    residual_rhs = a_np @ guess_np @ b_np + c_np - guess_np
    rhs = _flatten_sylvester_matrix(residual_rhs)

    def _matvec(vec: np.ndarray) -> np.ndarray:
        x = _unflatten_sylvester_matrix(vec, (n, m))
        return _flatten_sylvester_matrix(x - a_np @ x @ b_np)

    operator = scipy_sparse_linalg.LinearOperator(
        (n * m, n * m),
        matvec=_matvec,
        dtype=np.float64,
    )
    x0 = np.zeros(n * m, dtype=np.float64)
    iterations = 0

    def _callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    if algorithm == "bicgstab":
        delta_vec, info = scipy_sparse_linalg.bicgstab(
            operator,
            rhs,
            x0=x0,
            rtol=tol,
            atol=0.0,
            maxiter=max_iter,
            callback=_callback,
        )
    else:
        delta_vec, info = scipy_sparse_linalg.gmres(
            operator,
            rhs,
            x0=x0,
            rtol=tol,
            atol=0.0,
            restart=min(max_iter, max(20, n * m)),
            maxiter=max_iter,
            callback=_callback,
            callback_type="legacy",
        )

    delta = _unflatten_sylvester_matrix(np.asarray(delta_vec, dtype=np.float64), (n, m))
    solution = jnp.asarray(delta + guess_np, dtype=a_arr.dtype)
    rel_residual = discrete_sylvester_residual(a_arr, solution, b_arr, c_arr)
    converged = (info == 0) & (rel_residual < acceptance_tol)
    iteration_count = iterations if iterations > 0 else max(int(info), 0)
    return SylvesterResult(
        solution,
        converged,
        jnp.asarray(iteration_count),
        rel_residual,
    )


def solve_discrete_sylvester(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    algorithm: SylvesterAlgorithm = "doubling",
    tol: float = 1e-14,
    acceptance_tol: float = 1e-10,
    max_iter: int = 500,
    fallback_to_direct: bool = True,
) -> SylvesterOutcome:
    if algorithm not in ("doubling", "direct", "bicgstab", "gmres", "dqgmres"):
        raise ValueError(
            f"Unsupported Sylvester algorithm {algorithm!r}. "
            "Supported algorithms are 'doubling', 'direct', 'bicgstab', "
            "'gmres', and 'dqgmres'."
        )
    a_arr, b_arr, c_arr = _cast_sylvester_inputs(a, b, c)
    _validate_finite_sylvester_inputs(a_arr, b_arr, c_arr)

    if initial_guess is not None:
        guess_arr = jnp.asarray(initial_guess, dtype=c_arr.dtype)
        if guess_arr.shape != c_arr.shape:
            raise ValueError(
                f"initial_guess must match C's shape, got {guess_arr.shape} and {c_arr.shape}."
            )
        guess_residual = float(
            np.asarray(
                discrete_sylvester_residual(
                    a_arr,
                    guess_arr,
                    b_arr,
                    c_arr,
                )
            )
        )
        if guess_residual < acceptance_tol:
            return SylvesterOutcome(
                solution=guess_arr,
                converged=True,
                iterations=0,
                relative_residual=guess_residual,
                algorithm=algorithm,
                fallback_used=False,
            )

    if algorithm == "doubling":
        result = solve_discrete_sylvester_doubling(
            a_arr,
            b_arr,
            c_arr,
            initial_guess=initial_guess,
            tol=tol,
            acceptance_tol=acceptance_tol,
            max_iter=max_iter,
        )
    elif algorithm in ("bicgstab", "gmres", "dqgmres"):
        result = _solve_discrete_sylvester_iterative(
            a_arr,
            b_arr,
            c_arr,
            algorithm=algorithm,
            initial_guess=initial_guess,
            tol=tol,
            acceptance_tol=acceptance_tol,
            max_iter=max_iter,
        )
    else:
        result = solve_discrete_sylvester_direct(
            a_arr,
            b_arr,
            c_arr,
            initial_guess=initial_guess,
            acceptance_tol=acceptance_tol,
        )

    algorithm_used: SylvesterAlgorithm = algorithm
    fallback_used = False

    if (
        fallback_to_direct
        and algorithm in ("doubling", "bicgstab", "gmres", "dqgmres")
        and not bool(np.asarray(result.converged))
    ):
        direct_result = solve_discrete_sylvester_direct(
            a_arr,
            b_arr,
            c_arr,
            initial_guess=initial_guess,
            acceptance_tol=acceptance_tol,
        )
        if float(np.asarray(direct_result.relative_residual)) < float(
            np.asarray(result.relative_residual)
        ):
            result = direct_result
            algorithm_used = "direct"
            fallback_used = True

    return SylvesterOutcome(
        solution=result.solution,
        converged=bool(np.asarray(result.converged)),
        iterations=int(np.asarray(result.iterations)),
        relative_residual=float(np.asarray(result.relative_residual)),
        algorithm=algorithm_used,
        fallback_used=fallback_used,
    )


def solve_sylvester_equation(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    **kwargs: object,
) -> SylvesterOutcome:
    return solve_discrete_sylvester(a, b, c, **kwargs)
