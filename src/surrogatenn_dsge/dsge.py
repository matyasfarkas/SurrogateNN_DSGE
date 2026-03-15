from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .linalg import solve_discrete_sylvester
from .statespace import LinearGaussianStateSpace, build_linear_gaussian_state_space


class QuadraticMatrixEquationResult(NamedTuple):
    solution: jax.Array
    converged: bool
    iterations: int
    relative_residual: float


class FirstOrderDSGEResult(NamedTuple):
    solution_matrix: jax.Array
    qme_solution: jax.Array
    converged: bool
    state_transition: jax.Array
    shock_impact: jax.Array


class SecondOrderAuxiliaryMatrices(NamedTuple):
    sigma: jax.Array
    compression_matrix: jax.Array
    uncompression_matrix: jax.Array
    hessian_compression_matrix: jax.Array
    hessian_uncompression_matrix: jax.Array


class SecondOrderDSGEResult(NamedTuple):
    compressed_solution: jax.Array
    solution_matrix: jax.Array
    converged: bool
    iterations: int
    relative_residual: float
    algorithm: str
    fallback_used: bool


class SecondOrderStochasticSteadyStateResult(NamedTuple):
    state_vector: jax.Array
    reduced_state: jax.Array
    converged: bool
    iterations: int


@dataclass(frozen=True)
class DSGETimings:
    present_only: tuple[str, ...]
    future_not_past: tuple[str, ...]
    past_not_future: tuple[str, ...]
    mixed: tuple[str, ...]
    future_not_past_and_mixed: tuple[str, ...]
    past_not_future_and_mixed: tuple[str, ...]
    present_but_not_only: tuple[str, ...]
    mixed_in_past: tuple[str, ...]
    not_mixed_in_past: tuple[str, ...]
    mixed_in_future: tuple[str, ...]
    exo: tuple[str, ...]
    var: tuple[str, ...]
    aux: tuple[str, ...]
    exo_present: tuple[str, ...]
    nPresent_only: int
    nMixed: int
    nFuture_not_past_and_mixed: int
    nPast_not_future_and_mixed: int
    nPresent_but_not_only: int
    nVars: int
    nExo: int
    present_only_idx: tuple[int, ...]
    present_but_not_only_idx: tuple[int, ...]
    future_not_past_and_mixed_idx: tuple[int, ...]
    not_mixed_in_past_idx: tuple[int, ...]
    past_not_future_and_mixed_idx: tuple[int, ...]
    mixed_in_past_idx: tuple[int, ...]
    mixed_in_future_idx: tuple[int, ...]
    past_not_future_idx: tuple[int, ...]
    reorder: tuple[int, ...]
    dynamic_order: tuple[int, ...]

    @staticmethod
    def _to_tuple(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(str(v) for v in values)

    @staticmethod
    def _to_zero_based(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(int(v) - 1 for v in values)

    @classmethod
    def from_julia(
        cls,
        present_only: Sequence[str],
        future_not_past: Sequence[str],
        past_not_future: Sequence[str],
        mixed: Sequence[str],
        future_not_past_and_mixed: Sequence[str],
        past_not_future_and_mixed: Sequence[str],
        present_but_not_only: Sequence[str],
        mixed_in_past: Sequence[str],
        not_mixed_in_past: Sequence[str],
        mixed_in_future: Sequence[str],
        exo: Sequence[str],
        var: Sequence[str],
        aux: Sequence[str],
        exo_present: Sequence[str],
        nPresent_only: int,
        nMixed: int,
        nFuture_not_past_and_mixed: int,
        nPast_not_future_and_mixed: int,
        nPresent_but_not_only: int,
        nVars: int,
        nExo: int,
        present_only_idx: Sequence[int],
        present_but_not_only_idx: Sequence[int],
        future_not_past_and_mixed_idx: Sequence[int],
        not_mixed_in_past_idx: Sequence[int],
        past_not_future_and_mixed_idx: Sequence[int],
        mixed_in_past_idx: Sequence[int],
        mixed_in_future_idx: Sequence[int],
        past_not_future_idx: Sequence[int],
        reorder: Sequence[int],
        dynamic_order: Sequence[int],
    ) -> "DSGETimings":
        return cls(
            present_only=cls._to_tuple(present_only),
            future_not_past=cls._to_tuple(future_not_past),
            past_not_future=cls._to_tuple(past_not_future),
            mixed=cls._to_tuple(mixed),
            future_not_past_and_mixed=cls._to_tuple(future_not_past_and_mixed),
            past_not_future_and_mixed=cls._to_tuple(past_not_future_and_mixed),
            present_but_not_only=cls._to_tuple(present_but_not_only),
            mixed_in_past=cls._to_tuple(mixed_in_past),
            not_mixed_in_past=cls._to_tuple(not_mixed_in_past),
            mixed_in_future=cls._to_tuple(mixed_in_future),
            exo=cls._to_tuple(exo),
            var=cls._to_tuple(var),
            aux=cls._to_tuple(aux),
            exo_present=cls._to_tuple(exo_present),
            nPresent_only=int(nPresent_only),
            nMixed=int(nMixed),
            nFuture_not_past_and_mixed=int(nFuture_not_past_and_mixed),
            nPast_not_future_and_mixed=int(nPast_not_future_and_mixed),
            nPresent_but_not_only=int(nPresent_but_not_only),
            nVars=int(nVars),
            nExo=int(nExo),
            present_only_idx=cls._to_zero_based(present_only_idx),
            present_but_not_only_idx=cls._to_zero_based(present_but_not_only_idx),
            future_not_past_and_mixed_idx=cls._to_zero_based(
                future_not_past_and_mixed_idx
            ),
            not_mixed_in_past_idx=cls._to_zero_based(not_mixed_in_past_idx),
            past_not_future_and_mixed_idx=cls._to_zero_based(
                past_not_future_and_mixed_idx
            ),
            mixed_in_past_idx=cls._to_zero_based(mixed_in_past_idx),
            mixed_in_future_idx=cls._to_zero_based(mixed_in_future_idx),
            past_not_future_idx=cls._to_zero_based(past_not_future_idx),
            reorder=cls._to_zero_based(reorder),
            dynamic_order=cls._to_zero_based(dynamic_order),
        )


def _indexin(values: Sequence[int], reference: Sequence[int]) -> tuple[int, ...]:
    ref_map = {value: idx for idx, value in enumerate(reference)}
    return tuple(ref_map[value] for value in values)


def _compression_matrices(size: int, dtype: jnp.dtype = jnp.float64) -> tuple[jax.Array, jax.Array]:
    compressed_positions: list[int] = []
    compressed_lookup: dict[int, int] = {}
    for col in range(size):
        for row in range(col + 1):
            flat_index = row + size * col
            compressed_lookup[flat_index] = len(compressed_positions)
            compressed_positions.append(flat_index)

    compressed_dim = len(compressed_positions)
    compression = np.zeros((size * size, compressed_dim), dtype=np.float64)
    for col, flat_index in enumerate(compressed_positions):
        compression[flat_index, col] = 1.0

    uncompression = np.zeros((compressed_dim, size * size), dtype=np.float64)
    full_col = 0
    for col in range(size):
        for row in range(size):
            sorted_flat_index = min(row, col) + size * max(row, col)
            compressed_col = compressed_lookup[sorted_flat_index]
            uncompression[compressed_col, full_col] = 1.0
            full_col += 1

    return (
        jnp.asarray(compression, dtype=dtype),
        jnp.asarray(uncompression, dtype=dtype),
    )


def _coerce_first_order_solution_matrix(
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
) -> jax.Array:
    if isinstance(first_order_solution, FirstOrderDSGEResult):
        return jnp.asarray(first_order_solution.solution_matrix, dtype=jnp.float64)
    return jnp.asarray(first_order_solution, dtype=jnp.float64)


def _coerce_second_order_solution_matrix(
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
) -> jax.Array:
    if isinstance(second_order_solution, SecondOrderDSGEResult):
        return jnp.asarray(second_order_solution.solution_matrix, dtype=jnp.float64)
    solution = jnp.asarray(second_order_solution, dtype=jnp.float64)
    full_cols = (timings.nPast_not_future_and_mixed + 1 + timings.nExo) ** 2
    if solution.shape[1] == full_cols:
        return solution
    if auxiliary_matrices is None:
        auxiliary_matrices = create_second_order_auxiliary_matrices(timings)
    compressed_cols = auxiliary_matrices.compression_matrix.shape[1]
    if solution.shape[1] != compressed_cols:
        raise ValueError(
            "second_order_solution must have either the compressed or full Julia-compatible "
            f"shape, got {solution.shape}."
        )
    return solution @ auxiliary_matrices.uncompression_matrix


def _augmented_first_order_solution(
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
) -> jax.Array:
    solution = _coerce_first_order_solution_matrix(first_order_solution)
    expected_shape = (timings.nVars, timings.nPast_not_future_and_mixed + timings.nExo)
    if solution.shape != expected_shape:
        raise ValueError(
            "first_order_solution must have shape "
            f"{expected_shape}, got {solution.shape}."
        )
    zero_column = jnp.zeros((timings.nVars, 1), dtype=solution.dtype)
    return jnp.concatenate(
        [
            solution[:, : timings.nPast_not_future_and_mixed],
            zero_column,
            solution[:, timings.nPast_not_future_and_mixed :],
        ],
        axis=1,
    )


def create_second_order_auxiliary_matrices(
    timings: DSGETimings,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> SecondOrderAuxiliaryMatrices:
    dynamic_dim = (
        timings.nPast_not_future_and_mixed
        + timings.nVars
        + timings.nFuture_not_past_and_mixed
        + timings.nExo
    )
    hessian_compression_matrix, hessian_uncompression_matrix = _compression_matrices(
        dynamic_dim,
        dtype=dtype,
    )

    state_dim = timings.nPast_not_future_and_mixed + 1 + timings.nExo
    compression_matrix, uncompression_matrix = _compression_matrices(
        state_dim,
        dtype=dtype,
    )

    sigma = np.zeros((state_dim * state_dim, state_dim * state_dim), dtype=np.float64)
    volatility_column = timings.nPast_not_future_and_mixed * (state_dim + 1)
    for shock_idx in range(state_dim - timings.nExo, state_dim):
        sigma[shock_idx + state_dim * shock_idx, volatility_column] = 1.0

    return SecondOrderAuxiliaryMatrices(
        sigma=jnp.asarray(sigma, dtype=dtype),
        compression_matrix=compression_matrix,
        uncompression_matrix=uncompression_matrix,
        hessian_compression_matrix=hessian_compression_matrix,
        hessian_uncompression_matrix=hessian_uncompression_matrix,
    )


def quadratic_matrix_equation_residual(
    a: jax.Array,
    x: jax.Array,
    b: jax.Array,
    c: jax.Array,
) -> jax.Array:
    residual = a @ x @ x + b @ x + c
    denom = jnp.maximum(
        jnp.maximum(jnp.linalg.norm(a @ x @ x), jnp.linalg.norm(c)),
        jnp.finfo(x.dtype).eps,
    )
    return jnp.linalg.norm(residual) / denom


def solve_quadratic_matrix_equation_doubling(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    acceptance_tol: float = 1e-8,
    max_iter: int = 100,
) -> QuadraticMatrixEquationResult:
    a_arr = jnp.asarray(a, dtype=jnp.float64)
    b_arr = jnp.asarray(b, dtype=jnp.float64)
    c_arr = jnp.asarray(c, dtype=jnp.float64)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or c_arr.ndim != 2:
        raise ValueError("A, B, and C must be rank-2 matrices.")
    if a_arr.shape[0] != a_arr.shape[1] or b_arr.shape[0] != b_arr.shape[1]:
        raise ValueError("A and B must be square.")
    if a_arr.shape != b_arr.shape or a_arr.shape != c_arr.shape:
        raise ValueError("A, B, and C must have identical shapes.")

    if initial_guess is None:
        guess = jnp.zeros_like(a_arr)
        guess_provided = False
    else:
        guess = jnp.asarray(initial_guess, dtype=a_arr.dtype)
        if guess.shape != a_arr.shape:
            raise ValueError(
                f"initial_guess must match A's shape, got {guess.shape} and {a_arr.shape}."
            )
        guess_provided = True

    b_bar = a_arr @ guess + b_arr
    try:
        e_mat = jnp.linalg.solve(b_bar, c_arr)
        f_mat = jnp.linalg.solve(b_bar, a_arr)
    except Exception:
        return QuadraticMatrixEquationResult(a_arr, False, 0, 1.0)

    x_mat = -e_mat - guess
    y_mat = -f_mat
    x_new = x_mat
    converged = False
    iterations = max_iter
    identity = jnp.eye(a_arr.shape[0], dtype=a_arr.dtype)

    for iteration in range(1, max_iter + 1):
        ei = identity - y_mat @ x_mat
        fi = identity - x_mat @ y_mat
        try:
            e_new = e_mat @ jnp.linalg.solve(ei, e_mat)
            f_new = f_mat @ jnp.linalg.solve(fi, f_mat)
            x_increment = f_mat @ jnp.linalg.solve(fi, x_mat @ e_mat)
            y_increment = e_mat @ jnp.linalg.solve(ei, y_mat @ f_mat)
        except Exception:
            return QuadraticMatrixEquationResult(a_arr, False, iteration, 1.0)

        if not np.isfinite(np.asarray(x_increment)).all():
            return QuadraticMatrixEquationResult(a_arr, False, iteration, 1.0)

        x_tol = float(np.asarray(jnp.linalg.norm(x_increment)))
        x_new = x_mat + x_increment
        y_new = y_mat + y_increment

        if (iteration > 5 or guess_provided) and x_tol < tol:
            converged = True
            iterations = iteration
            x_mat = x_new
            break

        x_mat = x_new
        y_mat = y_new
        e_mat = e_new
        f_mat = f_new

    solution = x_mat + guess
    residual = float(np.asarray(quadratic_matrix_equation_residual(a_arr, solution, b_arr, c_arr)))
    return QuadraticMatrixEquationResult(
        solution=solution,
        converged=converged or residual < acceptance_tol,
        iterations=iterations,
        relative_residual=residual,
    )


def solve_first_order_dsge_solution(
    jacobian: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    qme_initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    qme_tol: float = 1e-14,
    qme_acceptance_tol: float = 1e-8,
) -> FirstOrderDSGEResult:
    grad = jnp.asarray(jacobian, dtype=jnp.float64)
    if grad.ndim != 2:
        raise ValueError(f"jacobian must be rank-2, got shape {grad.shape}.")

    dyn_index = np.arange(timings.nPresent_only, timings.nVars)
    reverse_dynamic_order = _indexin(
        timings.past_not_future_idx + timings.future_not_past_and_mixed_idx,
        timings.present_but_not_only_idx,
    )
    comb = tuple(
        sorted(
            set(timings.future_not_past_and_mixed_idx)
            | set(timings.past_not_future_idx)
        )
    )
    future_in_comb = _indexin(timings.future_not_past_and_mixed_idx, comb)
    past_in_comb = _indexin(timings.past_not_future_and_mixed_idx, comb)
    selector = jnp.eye(len(comb), dtype=grad.dtype)

    n_future = timings.nFuture_not_past_and_mixed
    n_vars = timings.nVars
    n_past = timings.nPast_not_future_and_mixed

    grad_plus = grad[:, :n_future]
    grad_zero = grad[:, n_future : n_future + n_vars]
    grad_minus = grad[:, n_future + n_vars : n_future + n_vars + n_past]
    grad_exo = grad[:, n_future + n_vars + n_past :]

    q_complete = jnp.linalg.qr(
        grad_zero[:, timings.present_only_idx],
        mode="complete",
    )[0]
    a_plus = q_complete.T @ grad_plus
    a_zero = q_complete.T @ grad_zero
    a_minus = q_complete.T @ grad_minus

    a_tilde_plus = a_plus[dyn_index] @ selector[list(future_in_comb), :]
    a_tilde_zero = a_zero[dyn_index][:, comb]
    a_tilde_minus = a_minus[dyn_index] @ selector[list(past_in_comb), :]

    qme_result = solve_quadratic_matrix_equation_doubling(
        a_tilde_plus,
        a_tilde_zero,
        a_tilde_minus,
        initial_guess=qme_initial_guess,
        tol=qme_tol,
        acceptance_tol=qme_acceptance_tol,
    )
    if not qme_result.converged:
        empty = jnp.zeros((timings.nVars, timings.nPast_not_future_and_mixed + timings.nExo))
        return FirstOrderDSGEResult(
            solution_matrix=empty,
            qme_solution=qme_result.solution,
            converged=False,
            state_transition=empty[:, : timings.nPast_not_future_and_mixed],
            shock_impact=empty[:, timings.nPast_not_future_and_mixed :],
        )

    sol = qme_result.solution
    reverse_dynamic_order_idx = jnp.asarray(reverse_dynamic_order, dtype=jnp.int32)
    past_in_comb_idx = jnp.asarray(past_in_comb, dtype=jnp.int32)
    past_not_future_and_mixed_idx = jnp.asarray(
        _indexin(timings.past_not_future_and_mixed_idx, timings.present_but_not_only_idx),
        dtype=jnp.int32,
    )
    reorder_idx = jnp.asarray(timings.reorder, dtype=jnp.int32)
    future_idx = jnp.asarray(timings.future_not_past_and_mixed_idx, dtype=jnp.int32)
    state_idx = jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32)

    sol_compact = sol[reverse_dynamic_order_idx][:, past_in_comb_idx]

    if timings.nFuture_not_past_and_mixed > 0:
        d_block = sol_compact[-timings.nFuture_not_past_and_mixed :, :]
    else:
        d_block = jnp.zeros((0, sol_compact.shape[1]), dtype=sol_compact.dtype)

    l_block = sol[past_not_future_and_mixed_idx][:, past_in_comb_idx]

    a_bar_zero_u = a_zero[: timings.nPresent_only][:, timings.present_only_idx]
    a_plus_u = a_plus[: timings.nPresent_only]
    a_tilde_zero_u = a_zero[: timings.nPresent_only][:, timings.present_but_not_only_idx]
    a_minus_u = a_minus[: timings.nPresent_only]

    if timings.nPresent_only > 0:
        rhs = a_tilde_zero_u @ sol[:, past_in_comb_idx]
        if timings.nFuture_not_past_and_mixed > 0:
            rhs = rhs + a_plus_u @ d_block @ l_block
        rhs = rhs + a_minus_u
        upper_block = -jnp.linalg.solve(a_bar_zero_u, rhs)
    else:
        upper_block = jnp.zeros((0, n_past), dtype=sol.dtype)

    transition = jnp.vstack([upper_block, sol_compact])[reorder_idx]

    selector = jnp.eye(timings.nVars, dtype=transition.dtype)[
        state_idx, :
    ]
    if timings.nFuture_not_past_and_mixed > 0:
        m_matrix = transition[future_idx, :] @ selector
    else:
        m_matrix = jnp.zeros((0, timings.nVars), dtype=transition.dtype)

    current_block = grad_zero + grad_plus @ m_matrix
    exogenous = -jnp.linalg.solve(current_block, grad_exo)
    solution_matrix = jnp.concatenate([transition, exogenous], axis=1)

    return FirstOrderDSGEResult(
        solution_matrix=solution_matrix,
        qme_solution=qme_result.solution,
        converged=True,
        state_transition=transition,
        shock_impact=exogenous,
    )


def solve_second_order_dsge_solution(
    jacobian: Union[jax.Array, np.ndarray],
    hessian: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    sylvester_algorithm: str = "doubling",
    sylvester_tol: float = 1e-14,
    sylvester_acceptance_tol: float = 1e-10,
    sylvester_max_iter: int = 500,
) -> SecondOrderDSGEResult:
    grad1 = jnp.asarray(jacobian, dtype=jnp.float64)
    grad2 = jnp.asarray(hessian, dtype=jnp.float64)
    first_order = _coerce_first_order_solution_matrix(first_order_solution)
    if auxiliary_matrices is None:
        auxiliary_matrices = create_second_order_auxiliary_matrices(
            timings,
            dtype=grad1.dtype,
        )

    dynamic_dim = (
        timings.nFuture_not_past_and_mixed
        + timings.nVars
        + timings.nPast_not_future_and_mixed
        + timings.nExo
    )
    expected_first_order_shape = (
        timings.nVars,
        timings.nPast_not_future_and_mixed + timings.nExo,
    )
    if grad1.shape != (timings.nVars, dynamic_dim):
        raise ValueError(
            "jacobian must have shape "
            f"({timings.nVars}, {dynamic_dim}), got {grad1.shape}."
        )
    if grad2.shape != (timings.nVars, dynamic_dim * dynamic_dim):
        raise ValueError(
            "hessian must have shape "
            f"({timings.nVars}, {dynamic_dim * dynamic_dim}), got {grad2.shape}."
        )
    if first_order.shape != expected_first_order_shape:
        raise ValueError(
            "first_order_solution must have shape "
            f"{expected_first_order_shape}, got {first_order.shape}."
        )

    n = timings.nVars
    n_minus = timings.nPast_not_future_and_mixed
    n_plus = timings.nFuture_not_past_and_mixed
    n_exo = timings.nExo
    state_dim = n_minus + 1 + n_exo
    i_plus = np.asarray(timings.future_not_past_and_mixed_idx, dtype=np.int32)
    i_minus = np.asarray(timings.past_not_future_and_mixed_idx, dtype=np.int32)

    first_order_augmented = _augmented_first_order_solution(first_order, timings)
    top_state_transition = first_order_augmented[i_minus, :]
    bottom_state_transition = jnp.concatenate(
        [
            jnp.zeros((n_exo + 1, n_minus), dtype=grad1.dtype),
            jnp.eye(n_exo + 1, dtype=grad1.dtype)[:, :1],
            jnp.zeros((n_exo + 1, n_exo), dtype=grad1.dtype),
        ],
        axis=1,
    )
    state_transition = jnp.concatenate(
        [top_state_transition, bottom_state_transition],
        axis=0,
    )

    state_and_shock_rows = np.asarray(
        tuple(range(n_minus)) + tuple(range(n_minus + 1, state_dim)),
        dtype=np.int32,
    )
    stacked = jnp.concatenate(
        [
            (first_order_augmented @ state_transition)[i_plus, :],
            first_order_augmented,
            jnp.eye(state_dim, dtype=grad1.dtype)[state_and_shock_rows, :],
        ],
        axis=0,
    )
    first_order_plus_zero = jnp.concatenate(
        [
            first_order_augmented[i_plus, :],
            jnp.zeros((n_minus + n + n_exo, state_dim), dtype=grad1.dtype),
        ],
        axis=0,
    )

    identity_n = jnp.eye(n, dtype=grad1.dtype)
    m_matrix = (
        -grad1[:, :n_plus]
        @ first_order_augmented[i_plus, :n_minus]
        @ identity_n[i_minus, :]
        - grad1[:, n_plus : n_plus + n]
    )
    rhs_a = grad1[:, :n_plus] @ identity_n[i_plus, :]
    rhs_c = (
        grad2
        @ jnp.kron(stacked, stacked)
        @ auxiliary_matrices.compression_matrix
        + grad2
        @ jnp.kron(first_order_plus_zero, first_order_plus_zero)
        @ auxiliary_matrices.sigma
        @ auxiliary_matrices.compression_matrix
    )
    b_matrix = (
        auxiliary_matrices.uncompression_matrix
        @ jnp.kron(state_transition, state_transition)
        @ auxiliary_matrices.compression_matrix
        + auxiliary_matrices.uncompression_matrix
        @ auxiliary_matrices.sigma
        @ auxiliary_matrices.compression_matrix
    )

    try:
        a_matrix = jnp.linalg.solve(m_matrix, rhs_a)
        c_matrix = jnp.linalg.solve(m_matrix, rhs_c)
    except Exception as exc:
        raise ValueError("Second-order solve failed while inverting the linearized system.") from exc

    compressed_guess = None
    if initial_guess is not None:
        guess = jnp.asarray(initial_guess, dtype=grad1.dtype)
        compressed_cols = auxiliary_matrices.compression_matrix.shape[1]
        full_cols = state_dim * state_dim
        if guess.shape == (n, full_cols):
            compressed_guess = guess @ auxiliary_matrices.compression_matrix
        elif guess.shape == (n, compressed_cols):
            compressed_guess = guess
        else:
            raise ValueError(
                "initial_guess must match either the compressed or full second-order shape, "
                f"got {guess.shape}."
            )

    sylvester_outcome = solve_discrete_sylvester(
        a_matrix,
        b_matrix,
        c_matrix,
        initial_guess=compressed_guess,
        algorithm=sylvester_algorithm,
        tol=sylvester_tol,
        acceptance_tol=sylvester_acceptance_tol,
        max_iter=sylvester_max_iter,
    )
    solution_matrix = sylvester_outcome.solution @ auxiliary_matrices.uncompression_matrix
    return SecondOrderDSGEResult(
        compressed_solution=sylvester_outcome.solution,
        solution_matrix=solution_matrix,
        converged=sylvester_outcome.converged,
        iterations=sylvester_outcome.iterations,
        relative_residual=sylvester_outcome.relative_residual,
        algorithm=sylvester_outcome.algorithm,
        fallback_used=sylvester_outcome.fallback_used,
    )


def solve_second_order_stochastic_steady_state(
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    pruning: bool = False,
    auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    max_iter: int = 100,
) -> SecondOrderStochasticSteadyStateResult:
    first_order_raw = _coerce_first_order_solution_matrix(first_order_solution)
    first_order_augmented = _augmented_first_order_solution(first_order_raw, timings)
    second_order_full = _coerce_second_order_solution_matrix(
        second_order_solution,
        timings,
        auxiliary_matrices=auxiliary_matrices,
    )

    n = timings.nVars
    n_minus = timings.nPast_not_future_and_mixed
    n_exo = timings.nExo
    i_minus = np.asarray(timings.past_not_future_and_mixed_idx, dtype=np.int32)
    dtype = first_order_augmented.dtype

    if second_order_full.shape != (n, (n_minus + 1 + n_exo) ** 2):
        raise ValueError(
            "second_order_solution must have shape "
            f"({n}, {(n_minus + 1 + n_exo) ** 2}), got {second_order_full.shape}."
        )

    aug_state = jnp.concatenate(
        [
            jnp.zeros(n_minus, dtype=dtype),
            jnp.ones(1, dtype=dtype),
            jnp.zeros(n_exo, dtype=dtype),
        ]
    )
    initial_linear_system = (
        jnp.eye(n_minus, dtype=dtype)
        - first_order_augmented[i_minus, :n_minus]
    )
    initial_rhs = (
        second_order_full @ jnp.kron(aug_state, aug_state) / 2.0
    )[i_minus]
    try:
        default_initial_state = jnp.linalg.solve(initial_linear_system, initial_rhs)
    except Exception as exc:
        raise ValueError(
            "Second-order stochastic steady-state solve failed while forming the initial guess."
        ) from exc

    if initial_guess is None:
        reduced_state = default_initial_state
    else:
        guess = jnp.asarray(initial_guess, dtype=dtype)
        if guess.shape == (n_minus,):
            reduced_state = guess
        elif guess.shape == (n,):
            reduced_state = guess[i_minus]
        else:
            raise ValueError(
                "initial_guess must have shape "
                f"({n_minus},) or ({n},), got {guess.shape}."
            )

    zero_shock = jnp.zeros(n_exo, dtype=dtype)
    if pruning:
        state_vector = (
            first_order_augmented[:, :n_minus] @ reduced_state
            + second_order_full @ jnp.kron(aug_state, aug_state) / 2.0
        )
        return SecondOrderStochasticSteadyStateResult(
            state_vector=state_vector,
            reduced_state=reduced_state,
            converged=True,
            iterations=0,
        )

    state_and_constant_mask = np.kron(
        np.concatenate(
            [
                np.ones(n_minus + 1, dtype=bool),
                np.zeros(n_exo, dtype=bool),
            ]
        ),
        np.concatenate(
            [
                np.ones(n_minus + 1, dtype=bool),
                np.zeros(n_exo, dtype=bool),
            ]
        ),
    )
    state_only_mask = np.kron(
        np.concatenate(
            [
                np.ones(n_minus + 1, dtype=bool),
                np.zeros(n_exo, dtype=bool),
            ]
        ),
        np.concatenate(
            [
                np.ones(n_minus, dtype=bool),
                np.zeros(n_exo + 1, dtype=bool),
            ]
        ),
    )
    state_and_constant_idx = np.flatnonzero(state_and_constant_mask)
    state_only_idx = np.flatnonzero(state_only_mask)

    reduced_transition = first_order_augmented[i_minus, :n_minus]
    reduced_second_order_jacobian = second_order_full[i_minus][:, state_only_idx]
    reduced_second_order_constant = second_order_full[i_minus][:, state_and_constant_idx]

    converged = False
    iterations = max_iter
    identity_reduced = jnp.eye(n_minus, dtype=dtype)

    for iteration in range(1, max_iter + 1):
        augmented_state = jnp.concatenate([reduced_state, jnp.ones(1, dtype=dtype)])
        kron_state_identity = jnp.kron(augmented_state[:, None], identity_reduced)
        jacobian = (
            reduced_transition
            + reduced_second_order_jacobian
            @ kron_state_identity
            - identity_reduced
        )
        mapped_state = (
            reduced_transition @ reduced_state
            + reduced_second_order_constant
            @ jnp.kron(augmented_state, augmented_state)
            / 2.0
        )
        try:
            delta = jnp.linalg.solve(jacobian, mapped_state - reduced_state)
        except Exception as exc:
            raise ValueError(
                "Second-order stochastic steady-state Newton step failed."
            ) from exc

        if iteration > 3 and bool(
            np.asarray(jnp.allclose(mapped_state, reduced_state, rtol=tol, atol=0.0))
        ):
            converged = True
            iterations = iteration
            break

        reduced_state = reduced_state - delta

    augmented_state = jnp.concatenate([reduced_state, jnp.ones(1, dtype=dtype)])
    full_second_order_constant = second_order_full[:, state_and_constant_idx]
    state_vector = (
        first_order_augmented[:, :n_minus] @ reduced_state
        + full_second_order_constant @ jnp.kron(augmented_state, augmented_state) / 2.0
    )
    fixed_point = second_order_state_update(
        state_vector,
        zero_shock,
        first_order_raw,
        second_order_full,
        timings,
    )
    converged = converged or bool(
        np.asarray(jnp.allclose(fixed_point, state_vector, rtol=tol, atol=0.0))
    )
    return SecondOrderStochasticSteadyStateResult(
        state_vector=state_vector,
        reduced_state=reduced_state,
        converged=converged,
        iterations=iterations,
    )


def second_order_state_update(
    state: Union[jax.Array, np.ndarray],
    shock: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
) -> jax.Array:
    state_arr = jnp.asarray(state, dtype=jnp.float64)
    shock_arr = jnp.asarray(shock, dtype=jnp.float64)
    if state_arr.shape != (timings.nVars,):
        raise ValueError(
            f"state must have shape ({timings.nVars},), got {state_arr.shape}."
        )
    if shock_arr.shape != (timings.nExo,):
        raise ValueError(
            f"shock must have shape ({timings.nExo},), got {shock_arr.shape}."
        )

    first_order_augmented = _augmented_first_order_solution(first_order_solution, timings)
    second_order_full = _coerce_second_order_solution_matrix(
        second_order_solution,
        timings,
        auxiliary_matrices=auxiliary_matrices,
    )
    reduced_state = jnp.take(
        state_arr,
        jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32),
    )
    augmented_state = jnp.concatenate(
        [reduced_state, jnp.ones(1, dtype=state_arr.dtype), shock_arr]
    )
    return (
        first_order_augmented @ augmented_state
        + second_order_full @ jnp.kron(augmented_state, augmented_state) / 2.0
    )


def pruned_second_order_state_update(
    pruned_states: Sequence[Union[jax.Array, np.ndarray]],
    shock: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
) -> tuple[jax.Array, jax.Array]:
    if len(pruned_states) != 2:
        raise ValueError(
            f"pruned_states must contain exactly 2 state vectors, got {len(pruned_states)}."
        )
    linear_state = jnp.asarray(pruned_states[0], dtype=jnp.float64)
    quadratic_state = jnp.asarray(pruned_states[1], dtype=jnp.float64)
    shock_arr = jnp.asarray(shock, dtype=jnp.float64)
    if linear_state.shape != (timings.nVars,) or quadratic_state.shape != (timings.nVars,):
        raise ValueError(
            "Each entry in pruned_states must have shape "
            f"({timings.nVars},)."
        )
    if shock_arr.shape != (timings.nExo,):
        raise ValueError(
            f"shock must have shape ({timings.nExo},), got {shock_arr.shape}."
        )

    first_order_augmented = _augmented_first_order_solution(first_order_solution, timings)
    second_order_full = _coerce_second_order_solution_matrix(
        second_order_solution,
        timings,
        auxiliary_matrices=auxiliary_matrices,
    )
    state_idx = jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32)
    augmented_linear_state = jnp.concatenate(
        [
            jnp.take(linear_state, state_idx),
            jnp.ones(1, dtype=linear_state.dtype),
            shock_arr,
        ]
    )
    augmented_quadratic_state = jnp.concatenate(
        [
            jnp.take(quadratic_state, state_idx),
            jnp.zeros(1, dtype=quadratic_state.dtype),
            jnp.zeros_like(shock_arr),
        ]
    )
    linear_component = first_order_augmented @ augmented_linear_state
    quadratic_component = (
        first_order_augmented @ augmented_quadratic_state
        + second_order_full
        @ jnp.kron(augmented_linear_state, augmented_linear_state)
        / 2.0
    )
    return linear_component, quadratic_component


def linear_state_space_from_first_order_solution(
    solution_matrix: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    observable_indices: Sequence[int],
    *,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
) -> LinearGaussianStateSpace:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    obs_zero = tuple(int(i) for i in observable_indices)
    observables_and_states = tuple(
        sorted(set(timings.past_not_future_and_mixed_idx) | set(obs_zero))
    )
    selector = jnp.eye(len(observables_and_states), dtype=solution.dtype)[
        list(_indexin(timings.past_not_future_and_mixed_idx, observables_and_states)), :
    ]
    transition = solution[list(observables_and_states), : timings.nPast_not_future_and_mixed] @ selector
    shock_impact = solution[list(observables_and_states), timings.nPast_not_future_and_mixed :]
    observation = jnp.eye(len(observables_and_states), dtype=solution.dtype)[
        list(_indexin(tuple(sorted(obs_zero)), observables_and_states)), :
    ]
    observation_noise = measurement_error_scale * jnp.eye(
        observation.shape[0],
        dtype=solution.dtype,
    )
    return build_linear_gaussian_state_space(
        transition_matrix=transition,
        process_noise_covariance=shock_impact @ shock_impact.T,
        observation_matrix=observation,
        observation_noise_covariance=observation_noise,
        initial_covariance_strategy=initial_covariance_strategy,
    )
