from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
from jax import lax

from .linalg import solve_discrete_sylvester
from .statespace import LinearGaussianStateSpace, build_linear_gaussian_state_space

QuadraticMatrixEquationAlgorithm = Literal["doubling", "schur"]


class QuadraticMatrixEquationResult(NamedTuple):
    solution: jax.Array
    converged: bool
    iterations: int
    relative_residual: float


class SchurQZDeterminacyDiagnostics(NamedTuple):
    companion_roots: jax.Array
    stable_mask: jax.Array
    stable_count: int
    expected_stable_count: int
    decomposition_succeeded: bool
    invariant_subspace_invertible: bool
    solution_extracted: bool
    relative_residual: float
    classification: str
    unique_stable_solution: bool


class SchurQZDeterminacyResult(NamedTuple):
    solution: jax.Array
    diagnostics: SchurQZDeterminacyDiagnostics


class _QuadraticMatrixEquationDoublingState(NamedTuple):
    e_matrix: jax.Array
    f_matrix: jax.Array
    x_matrix: jax.Array
    y_matrix: jax.Array
    converged: jax.Array
    failed: jax.Array
    iterations: jax.Array


class FirstOrderDSGEResult(NamedTuple):
    solution_matrix: jax.Array
    qme_solution: jax.Array
    converged: bool
    state_transition: jax.Array
    shock_impact: jax.Array


class FirstOrderDeterminacyResult(NamedTuple):
    solution: FirstOrderDSGEResult
    qme_diagnostics: SchurQZDeterminacyDiagnostics
    classification: str
    unique_stable_solution: bool


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


class ThirdOrderAuxiliaryMatrices(NamedTuple):
    compression_matrix: jax.Array
    uncompression_matrix: jax.Array
    permutation_matrix: jax.Array
    swap12_left: jax.Array
    swap12_right: jax.Array
    swap23_left_dynamic: jax.Array
    swap13_left_dynamic: jax.Array
    swap23_left_state: jax.Array
    swap13_left_state: jax.Array
    swap23_right_state: jax.Array
    swap13_right_state: jax.Array


class ThirdOrderDSGEResult(NamedTuple):
    compressed_solution: jax.Array
    solution_matrix: jax.Array
    converged: bool
    iterations: int
    relative_residual: float
    algorithm: str
    fallback_used: bool


class ThirdOrderStochasticSteadyStateResult(NamedTuple):
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


def _compression_matrices_order3(
    size: int,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[jax.Array, jax.Array]:
    compressed_positions: list[int] = []
    compressed_lookup: dict[int, int] = {}
    for outer in range(size):
        for middle in range(outer + 1):
            for inner in range(middle + 1):
                flat_index = size * size * outer + size * middle + inner
                compressed_lookup[flat_index] = len(compressed_positions)
                compressed_positions.append(flat_index)

    compressed_dim = len(compressed_positions)
    compression = np.zeros((size**3, compressed_dim), dtype=np.float64)
    for col, flat_index in enumerate(compressed_positions):
        compression[flat_index, col] = 1.0

    uncompression = np.zeros((compressed_dim, size**3), dtype=np.float64)
    full_col = 0
    for outer in range(size):
        for middle in range(size):
            for inner in range(size):
                sorted_ids = sorted((outer, middle, inner))
                sorted_flat_index = (
                    size * size * sorted_ids[2]
                    + size * sorted_ids[1]
                    + sorted_ids[0]
                )
                compressed_col = compressed_lookup[sorted_flat_index]
                uncompression[compressed_col, full_col] = 1.0
                full_col += 1

    return (
        jnp.asarray(compression, dtype=dtype),
        jnp.asarray(uncompression, dtype=dtype),
    )


def _order3_permutation_vector(size: int, axes: tuple[int, int, int]) -> np.ndarray:
    tensor = np.arange(size**3, dtype=np.int32).reshape((size, size, size), order="F")
    return np.transpose(tensor, axes=tuple(axis - 1 for axis in axes)).reshape(-1, order="F")


def _order3_right_permutation_matrix(
    size: int,
    axes: tuple[int, int, int],
    dtype: jnp.dtype = jnp.float64,
) -> jax.Array:
    permutation = _order3_permutation_vector(size, axes)
    return jnp.asarray(np.eye(size**3, dtype=np.float64)[:, permutation], dtype=dtype)


def _order3_left_permutation_matrix(
    size: int,
    axes: tuple[int, int, int],
    dtype: jnp.dtype = jnp.float64,
) -> jax.Array:
    permutation = _order3_permutation_vector(size, axes)
    return jnp.asarray(np.eye(size**3, dtype=np.float64)[permutation, :], dtype=dtype)


def _order3_summed_permutation_matrix(
    size: int,
    axes_list: Sequence[tuple[int, int, int]],
    dtype: jnp.dtype = jnp.float64,
) -> jax.Array:
    matrix = np.zeros((size**3, size**3), dtype=np.float64)
    identity = np.eye(size**3, dtype=np.float64)
    for axes in axes_list:
        permutation = _order3_permutation_vector(size, axes)
        matrix += identity[:, permutation]
    return jnp.asarray(matrix, dtype=dtype)


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


def _coerce_third_order_solution_matrix(
    third_order_solution: Union[ThirdOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    auxiliary_matrices: Optional[ThirdOrderAuxiliaryMatrices] = None,
) -> jax.Array:
    if isinstance(third_order_solution, ThirdOrderDSGEResult):
        return jnp.asarray(third_order_solution.solution_matrix, dtype=jnp.float64)
    solution = jnp.asarray(third_order_solution, dtype=jnp.float64)
    full_cols = (timings.nPast_not_future_and_mixed + 1 + timings.nExo) ** 3
    if solution.shape[1] == full_cols:
        return solution
    if auxiliary_matrices is None:
        auxiliary_matrices = create_third_order_auxiliary_matrices(timings)
    compressed_cols = auxiliary_matrices.compression_matrix.shape[1]
    if solution.shape[1] != compressed_cols:
        raise ValueError(
            "third_order_solution must have either the compressed or full Julia-compatible "
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


def create_third_order_auxiliary_matrices(
    timings: DSGETimings,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> ThirdOrderAuxiliaryMatrices:
    state_dim = timings.nPast_not_future_and_mixed + 1 + timings.nExo
    dynamic_dim = (
        timings.nPast_not_future_and_mixed
        + timings.nVars
        + timings.nFuture_not_past_and_mixed
        + timings.nExo
    )
    compression_matrix, uncompression_matrix = _compression_matrices_order3(
        state_dim,
        dtype=dtype,
    )
    return ThirdOrderAuxiliaryMatrices(
        compression_matrix=compression_matrix,
        uncompression_matrix=uncompression_matrix,
        permutation_matrix=_order3_summed_permutation_matrix(
            state_dim,
            ((3, 1, 2), (1, 3, 2), (1, 2, 3)),
            dtype=dtype,
        ),
        swap12_left=_order3_left_permutation_matrix(state_dim, (2, 1, 3), dtype=dtype),
        swap12_right=_order3_right_permutation_matrix(state_dim, (2, 1, 3), dtype=dtype),
        swap23_left_dynamic=_order3_left_permutation_matrix(dynamic_dim, (1, 3, 2), dtype=dtype),
        swap13_left_dynamic=_order3_left_permutation_matrix(dynamic_dim, (3, 1, 2), dtype=dtype),
        swap23_left_state=_order3_left_permutation_matrix(state_dim, (1, 3, 2), dtype=dtype),
        swap13_left_state=_order3_left_permutation_matrix(state_dim, (3, 1, 2), dtype=dtype),
        swap23_right_state=_order3_right_permutation_matrix(state_dim, (1, 3, 2), dtype=dtype),
        swap13_right_state=_order3_right_permutation_matrix(state_dim, (3, 1, 2), dtype=dtype),
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


def _cast_quadratic_matrix_equation_inputs(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_arr = jnp.asarray(a, dtype=jnp.float64)
    b_arr = jnp.asarray(b, dtype=jnp.float64)
    c_arr = jnp.asarray(c, dtype=jnp.float64)
    if a_arr.ndim != 2 or b_arr.ndim != 2 or c_arr.ndim != 2:
        raise ValueError("A, B, and C must be rank-2 matrices.")
    if a_arr.shape[0] != a_arr.shape[1] or b_arr.shape[0] != b_arr.shape[1]:
        raise ValueError("A and B must be square.")
    if a_arr.shape != b_arr.shape or a_arr.shape != c_arr.shape:
        raise ValueError("A, B, and C must have identical shapes.")
    return a_arr, b_arr, c_arr


def _schur_stable_selection(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(beta / alpha)
    return np.isfinite(ratio) & (ratio < 1.0)


def _classify_schur_determinacy(
    *,
    expected_stable_count: int,
    stable_count: int,
    decomposition_succeeded: bool,
    invariant_subspace_invertible: bool,
    solution_extracted: bool,
    relative_residual: float,
    acceptance_tol: float,
) -> str:
    if not decomposition_succeeded:
        return "decomposition_failed"
    if stable_count < expected_stable_count:
        return "no_stable_solution"
    if stable_count > expected_stable_count:
        return "indeterminate"
    if not invariant_subspace_invertible:
        return "singular_invariant_subspace"
    if not solution_extracted:
        return "complex_solution"
    if relative_residual >= acceptance_tol:
        return "residual_too_large"
    return "unique_stable_solution"


def _analyze_quadratic_matrix_equation_schur_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    timings: DSGETimings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    empty_roots = np.asarray([], dtype=np.complex128)
    empty_mask = np.asarray([], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        comb = tuple(
            sorted(
                set(timings.future_not_past_and_mixed_idx)
                | set(timings.past_not_future_idx)
            )
        )
        future_in_comb = _indexin(timings.future_not_past_and_mixed_idx, comb)
        past_not_future_and_mixed_in_comb = _indexin(
            timings.past_not_future_and_mixed_idx,
            comb,
        )
        past_not_future_in_comb = _indexin(timings.past_not_future_idx, comb)

        a_tilde_plus = a_arr[:, list(future_in_comb)]
        a_tilde_minus = c_arr[:, list(past_not_future_and_mixed_in_comb)]
        a_tilde_zero_plus = b_arr[:, list(future_in_comb)]
        a_tilde_zero_minus = b_arr[:, list(past_not_future_in_comb)] @ np.eye(
            timings.nPast_not_future_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.not_mixed_in_past_idx), :]

        z_plus = np.zeros(
            (timings.nMixed, timings.nFuture_not_past_and_mixed),
            dtype=a_arr.dtype,
        )
        i_plus = np.eye(
            timings.nFuture_not_past_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.mixed_in_future_idx), :]
        z_minus = np.zeros(
            (timings.nMixed, timings.nPast_not_future_and_mixed),
            dtype=a_arr.dtype,
        )
        i_minus = np.eye(
            timings.nPast_not_future_and_mixed,
            dtype=a_arr.dtype,
        )[list(timings.mixed_in_past_idx), :]

        d_pencil = np.block(
            [
                [a_tilde_zero_minus, a_tilde_plus],
                [i_minus, z_plus],
            ]
        )
        e_pencil = np.block(
            [
                [-a_tilde_minus, -a_tilde_zero_plus],
                [z_minus, i_plus],
            ]
        )

        try:
            s_matrix, t_matrix, alpha, beta, _, z_matrix = scipy_linalg.ordqz(
                d_pencil,
                e_pencil,
                sort=_schur_stable_selection,
                output="complex",
                check_finite=False,
            )
        except Exception:
            return np.array(a_arr, copy=True), empty_roots, empty_mask, False, False, False

        with np.errstate(divide="ignore", invalid="ignore"):
            companion_roots = beta / alpha
        stable_mask = _schur_stable_selection(alpha, beta)
        if int(np.count_nonzero(stable_mask)) != timings.nPast_not_future_and_mixed:
            return (
                np.array(a_arr, copy=True),
                np.asarray(companion_roots, dtype=np.complex128),
                np.asarray(stable_mask, dtype=bool),
                True,
                False,
                False,
            )

        n_past = timings.nPast_not_future_and_mixed
        z21 = z_matrix[n_past:, :n_past]
        z11 = z_matrix[:n_past, :n_past]
        s11 = s_matrix[:n_past, :n_past]
        t11 = t_matrix[:n_past, :n_past]

        try:
            d_block = scipy_linalg.solve(z11.T, z21.T, check_finite=False).T
            l_core = scipy_linalg.solve(s11, t11, check_finite=False)
            l_block = scipy_linalg.solve(
                z11.T,
                (z11 @ l_core).T,
                check_finite=False,
            ).T
        except Exception:
            return (
                np.array(a_arr, copy=True),
                np.asarray(companion_roots, dtype=np.complex128),
                np.asarray(stable_mask, dtype=bool),
                True,
                False,
                False,
            )

        sol = np.vstack([l_block[list(timings.not_mixed_in_past_idx), :], d_block])
        selection = np.eye(len(comb), dtype=a_arr.dtype)[
            list(past_not_future_and_mixed_in_comb),
            :,
        ]
        x_matrix = sol[list(timings.dynamic_order), :] @ selection
        if np.max(np.abs(np.imag(x_matrix))) > 1e-9:
            return (
                np.array(a_arr, copy=True),
                np.asarray(companion_roots, dtype=np.complex128),
                np.asarray(stable_mask, dtype=bool),
                True,
                True,
                False,
            )
        return (
            np.asarray(np.real(x_matrix), dtype=np.float64),
            np.asarray(companion_roots, dtype=np.complex128),
            np.asarray(stable_mask, dtype=bool),
            True,
            True,
            True,
        )


def _solve_quadratic_matrix_equation_schur_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    timings: DSGETimings,
) -> tuple[np.ndarray, bool]:
    solution, _, _, decomposition_succeeded, _, solution_extracted = (
        _analyze_quadratic_matrix_equation_schur_numpy(
            a,
            b,
            c,
            timings,
        )
    )
    return solution, decomposition_succeeded and solution_extracted


def analyze_quadratic_matrix_equation_schur(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    acceptance_tol: float = 1e-8,
) -> SchurQZDeterminacyResult:
    a_arr, b_arr, c_arr = _cast_quadratic_matrix_equation_inputs(a, b, c)
    (
        solution_np,
        companion_roots_np,
        stable_mask_np,
        decomposition_succeeded,
        invariant_subspace_invertible,
        solution_extracted,
    ) = _analyze_quadratic_matrix_equation_schur_numpy(
        np.asarray(a_arr),
        np.asarray(b_arr),
        np.asarray(c_arr),
        timings,
    )
    solution = jnp.asarray(solution_np, dtype=a_arr.dtype)
    if solution_extracted:
        relative_residual = float(
            np.asarray(quadratic_matrix_equation_residual(a_arr, solution, b_arr, c_arr))
        )
    else:
        relative_residual = float("inf")
    stable_count = int(np.count_nonzero(stable_mask_np))
    expected_stable_count = int(timings.nPast_not_future_and_mixed)
    classification = _classify_schur_determinacy(
        expected_stable_count=expected_stable_count,
        stable_count=stable_count,
        decomposition_succeeded=decomposition_succeeded,
        invariant_subspace_invertible=invariant_subspace_invertible,
        solution_extracted=solution_extracted,
        relative_residual=relative_residual,
        acceptance_tol=acceptance_tol,
    )
    diagnostics = SchurQZDeterminacyDiagnostics(
        companion_roots=jnp.asarray(companion_roots_np, dtype=jnp.complex128),
        stable_mask=jnp.asarray(stable_mask_np, dtype=bool),
        stable_count=stable_count,
        expected_stable_count=expected_stable_count,
        decomposition_succeeded=decomposition_succeeded,
        invariant_subspace_invertible=invariant_subspace_invertible,
        solution_extracted=solution_extracted,
        relative_residual=relative_residual,
        classification=classification,
        unique_stable_solution=(classification == "unique_stable_solution"),
    )
    return SchurQZDeterminacyResult(
        solution=solution,
        diagnostics=diagnostics,
    )


def _solve_quadratic_matrix_equation_schur_callback(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    timings: DSGETimings,
) -> tuple[jax.Array, jax.Array]:
    result_shape = (
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )

    def _host_callback(
        a_host: np.ndarray,
        b_host: np.ndarray,
        c_host: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        solution, success = _solve_quadratic_matrix_equation_schur_numpy(
            np.asarray(a_host),
            np.asarray(b_host),
            np.asarray(c_host),
            timings,
        )
        return solution, np.asarray(success, dtype=np.bool_)

    return jax.pure_callback(
        _host_callback,
        result_shape,
        a,
        b,
        c,
    )


def _solve_qme_schur_adjoint(
    a: jax.Array,
    b: jax.Array,
    solution: jax.Array,
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = solution.shape[0]
    left = (a @ solution + b).T
    system = jnp.kron(jnp.eye(n, dtype=solution.dtype), left) + jnp.kron(
        solution,
        a.T,
    )
    rhs = jnp.ravel(cotangent.T)
    y_vec = jnp.linalg.solve(system, rhs)
    y_matrix = jnp.reshape(y_vec, solution.shape).T
    return (
        -(y_matrix @ (solution @ solution).T),
        -(y_matrix @ solution.T),
        -y_matrix,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _solve_quadratic_matrix_equation_schur_jax_core(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    timings: DSGETimings,
) -> tuple[jax.Array, jax.Array]:
    return _solve_quadratic_matrix_equation_schur_callback(a, b, c, timings)


def _solve_quadratic_matrix_equation_schur_jax_core_fwd(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    timings: DSGETimings,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    solution, success = _solve_quadratic_matrix_equation_schur_callback(a, b, c, timings)
    return (solution, success), (a, b, c, solution, success)


def _solve_quadratic_matrix_equation_schur_jax_core_bwd(
    timings: DSGETimings,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    a, b, _, solution, success = residuals
    solution_bar, _ = cotangents
    zero_grads = (
        jnp.zeros_like(a),
        jnp.zeros_like(b),
        jnp.zeros_like(solution),
    )
    return lax.cond(
        success,
        lambda args: _solve_qme_schur_adjoint(*args),
        lambda _: zero_grads,
        (a, b, solution, solution_bar),
    )


_solve_quadratic_matrix_equation_schur_jax_core.defvjp(
    _solve_quadratic_matrix_equation_schur_jax_core_fwd,
    _solve_quadratic_matrix_equation_schur_jax_core_bwd,
)


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
    a_arr, b_arr, c_arr = _cast_quadratic_matrix_equation_inputs(a, b, c)

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


def solve_quadratic_matrix_equation_doubling_jax(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    acceptance_tol: float = 1e-8,
    max_iter: int = 100,
) -> QuadraticMatrixEquationResult:
    a_arr, b_arr, c_arr = _cast_quadratic_matrix_equation_inputs(a, b, c)

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

    tol_arr = jnp.asarray(tol, dtype=a_arr.dtype)
    acceptance_tol_arr = jnp.asarray(acceptance_tol, dtype=a_arr.dtype)
    guess_provided_arr = jnp.asarray(guess_provided)
    identity = jnp.eye(a_arr.shape[0], dtype=a_arr.dtype)

    b_bar = a_arr @ guess + b_arr
    e_matrix = jnp.linalg.solve(b_bar, c_arr)
    f_matrix = jnp.linalg.solve(b_bar, a_arr)
    x_matrix = -e_matrix - guess
    y_matrix = -f_matrix
    initial_failed = ~(
        jnp.all(jnp.isfinite(e_matrix))
        & jnp.all(jnp.isfinite(f_matrix))
        & jnp.all(jnp.isfinite(x_matrix))
        & jnp.all(jnp.isfinite(y_matrix))
    )

    def step(
        iteration: int,
        state: _QuadraticMatrixEquationDoublingState,
    ) -> _QuadraticMatrixEquationDoublingState:
        def _active(current_state: _QuadraticMatrixEquationDoublingState) -> _QuadraticMatrixEquationDoublingState:
            ei = identity - current_state.y_matrix @ current_state.x_matrix
            fi = identity - current_state.x_matrix @ current_state.y_matrix
            e_new = current_state.e_matrix @ jnp.linalg.solve(ei, current_state.e_matrix)
            f_new = current_state.f_matrix @ jnp.linalg.solve(fi, current_state.f_matrix)
            x_increment = current_state.f_matrix @ jnp.linalg.solve(
                fi,
                current_state.x_matrix @ current_state.e_matrix,
            )
            y_increment = current_state.e_matrix @ jnp.linalg.solve(
                ei,
                current_state.y_matrix @ current_state.f_matrix,
            )
            x_new = current_state.x_matrix + x_increment
            y_new = current_state.y_matrix + y_increment
            finite_step = (
                jnp.all(jnp.isfinite(e_new))
                & jnp.all(jnp.isfinite(f_new))
                & jnp.all(jnp.isfinite(x_increment))
                & jnp.all(jnp.isfinite(y_increment))
                & jnp.all(jnp.isfinite(x_new))
                & jnp.all(jnp.isfinite(y_new))
            )
            converged_now = (
                ((iteration + 1 > 5) | guess_provided_arr)
                & (jnp.linalg.norm(x_increment) < tol_arr)
                & finite_step
            )
            return _QuadraticMatrixEquationDoublingState(
                e_matrix=jnp.where(finite_step, e_new, current_state.e_matrix),
                f_matrix=jnp.where(finite_step, f_new, current_state.f_matrix),
                x_matrix=jnp.where(finite_step, x_new, current_state.x_matrix),
                y_matrix=jnp.where(finite_step, y_new, current_state.y_matrix),
                converged=converged_now,
                failed=current_state.failed | (~finite_step),
                iterations=jnp.asarray(iteration + 1),
            )

        return lax.cond(
            state.converged | state.failed,
            lambda current_state: current_state,
            _active,
            state,
        )

    initial_state = _QuadraticMatrixEquationDoublingState(
        e_matrix=e_matrix,
        f_matrix=f_matrix,
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        converged=jnp.asarray(False),
        failed=initial_failed,
        iterations=jnp.asarray(0),
    )
    final_state = lax.fori_loop(0, max_iter, step, initial_state)
    solution = final_state.x_matrix + guess
    relative_residual = quadratic_matrix_equation_residual(
        a_arr,
        solution,
        b_arr,
        c_arr,
    )
    converged = (~final_state.failed) & (
        final_state.converged | (relative_residual < acceptance_tol_arr)
    )
    return QuadraticMatrixEquationResult(
        solution=solution,
        converged=converged,
        iterations=final_state.iterations,
        relative_residual=relative_residual,
    )


def solve_quadratic_matrix_equation_schur(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    acceptance_tol: float = 1e-8,
) -> QuadraticMatrixEquationResult:
    a_arr, b_arr, c_arr = _cast_quadratic_matrix_equation_inputs(a, b, c)
    if initial_guess is not None:
        guess = jnp.asarray(initial_guess, dtype=a_arr.dtype)
        if guess.shape != a_arr.shape:
            raise ValueError(
                f"initial_guess must match A's shape, got {guess.shape} and {a_arr.shape}."
            )
        guess_residual = float(
            np.asarray(
                quadratic_matrix_equation_residual(
                    a_arr,
                    guess,
                    b_arr,
                    c_arr,
                )
            )
        )
        if guess_residual < acceptance_tol:
            return QuadraticMatrixEquationResult(
                solution=guess,
                converged=True,
                iterations=0,
                relative_residual=guess_residual,
            )

    analysis = analyze_quadratic_matrix_equation_schur(
        a_arr,
        b_arr,
        c_arr,
        timings,
        acceptance_tol=acceptance_tol,
    )
    return QuadraticMatrixEquationResult(
        solution=analysis.solution,
        converged=analysis.diagnostics.unique_stable_solution,
        iterations=0,
        relative_residual=analysis.diagnostics.relative_residual,
    )


def solve_quadratic_matrix_equation_schur_jax(
    a: Union[jax.Array, np.ndarray],
    b: Union[jax.Array, np.ndarray],
    c: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    acceptance_tol: float = 1e-8,
) -> QuadraticMatrixEquationResult:
    a_arr, b_arr, c_arr = _cast_quadratic_matrix_equation_inputs(a, b, c)
    if initial_guess is not None:
        guess = jnp.asarray(initial_guess, dtype=a_arr.dtype)
        if guess.shape != a_arr.shape:
            raise ValueError(
                f"initial_guess must match A's shape, got {guess.shape} and {a_arr.shape}."
            )

    solution, success = _solve_quadratic_matrix_equation_schur_jax_core(
        a_arr,
        b_arr,
        c_arr,
        timings,
    )
    acceptance_tol_arr = jnp.asarray(acceptance_tol, dtype=a_arr.dtype)
    residual = quadratic_matrix_equation_residual(a_arr, solution, b_arr, c_arr)
    relative_residual = jnp.where(success, residual, jnp.asarray(1.0, dtype=a_arr.dtype))
    converged = success & (relative_residual < acceptance_tol_arr)
    return QuadraticMatrixEquationResult(
        solution=solution,
        converged=converged,
        iterations=jnp.asarray(0),
        relative_residual=relative_residual,
    )


def _prepare_first_order_qme_system(
    jacobian: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    grad = jnp.asarray(jacobian, dtype=jnp.float64)
    if grad.ndim != 2:
        raise ValueError(f"jacobian must be rank-2, got shape {grad.shape}.")

    dyn_index = np.arange(timings.nPresent_only, timings.nVars)
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

    q_complete = jnp.linalg.qr(
        grad_zero[:, timings.present_only_idx],
        mode="complete",
    )[0]
    a_plus = q_complete.T @ grad_plus
    a_zero = q_complete.T @ grad_zero
    a_minus = q_complete.T @ grad_minus

    return (
        grad,
        a_plus[dyn_index] @ selector[list(future_in_comb), :],
        a_zero[dyn_index][:, comb],
        a_minus[dyn_index] @ selector[list(past_in_comb), :],
    )


def solve_first_order_dsge_solution(
    jacobian: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    qme_algorithm: QuadraticMatrixEquationAlgorithm = "schur",
    qme_initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    qme_tol: float = 1e-14,
    qme_acceptance_tol: float = 1e-8,
) -> FirstOrderDSGEResult:
    grad, a_tilde_plus, a_tilde_zero, a_tilde_minus = _prepare_first_order_qme_system(
        jacobian,
        timings,
    )

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

    if qme_algorithm == "doubling":
        qme_result = solve_quadratic_matrix_equation_doubling(
            a_tilde_plus,
            a_tilde_zero,
            a_tilde_minus,
            initial_guess=qme_initial_guess,
            tol=qme_tol,
            acceptance_tol=qme_acceptance_tol,
        )
    elif qme_algorithm == "schur":
        qme_result = solve_quadratic_matrix_equation_schur(
            a_tilde_plus,
            a_tilde_zero,
            a_tilde_minus,
            timings,
            initial_guess=qme_initial_guess,
            acceptance_tol=qme_acceptance_tol,
        )
    else:
        raise ValueError(
            f"Unsupported quadratic matrix equation algorithm {qme_algorithm!r}. "
            "Only 'doubling' and 'schur' are implemented."
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


def analyze_first_order_dsge_determinacy(
    jacobian: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    qme_acceptance_tol: float = 1e-8,
) -> FirstOrderDeterminacyResult:
    _, a_tilde_plus, a_tilde_zero, a_tilde_minus = _prepare_first_order_qme_system(
        jacobian,
        timings,
    )
    qme_analysis = analyze_quadratic_matrix_equation_schur(
        a_tilde_plus,
        a_tilde_zero,
        a_tilde_minus,
        timings,
        acceptance_tol=qme_acceptance_tol,
    )
    solution = solve_first_order_dsge_solution(
        jacobian,
        timings,
        qme_algorithm="schur",
        qme_acceptance_tol=qme_acceptance_tol,
    )
    return FirstOrderDeterminacyResult(
        solution=solution,
        qme_diagnostics=qme_analysis.diagnostics,
        classification=qme_analysis.diagnostics.classification,
        unique_stable_solution=qme_analysis.diagnostics.unique_stable_solution,
    )


def solve_first_order_dsge_solution_jax(
    jacobian: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    qme_algorithm: QuadraticMatrixEquationAlgorithm = "schur",
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

    if qme_algorithm == "doubling":
        qme_result = solve_quadratic_matrix_equation_doubling_jax(
            a_tilde_plus,
            a_tilde_zero,
            a_tilde_minus,
            initial_guess=qme_initial_guess,
            tol=qme_tol,
            acceptance_tol=qme_acceptance_tol,
        )
    elif qme_algorithm == "schur":
        qme_result = solve_quadratic_matrix_equation_schur_jax(
            a_tilde_plus,
            a_tilde_zero,
            a_tilde_minus,
            timings,
            initial_guess=qme_initial_guess,
            acceptance_tol=qme_acceptance_tol,
        )
    else:
        raise ValueError(
            f"Unsupported quadratic matrix equation algorithm {qme_algorithm!r}. "
            "Only 'doubling' and 'schur' are implemented."
        )

    reverse_dynamic_order_idx = jnp.asarray(reverse_dynamic_order, dtype=jnp.int32)
    past_in_comb_idx = jnp.asarray(past_in_comb, dtype=jnp.int32)
    past_not_future_and_mixed_idx = jnp.asarray(
        _indexin(timings.past_not_future_and_mixed_idx, timings.present_but_not_only_idx),
        dtype=jnp.int32,
    )
    reorder_idx = jnp.asarray(timings.reorder, dtype=jnp.int32)
    future_idx = jnp.asarray(timings.future_not_past_and_mixed_idx, dtype=jnp.int32)
    state_idx = jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32)
    empty = jnp.zeros(
        (timings.nVars, timings.nPast_not_future_and_mixed + timings.nExo),
        dtype=grad.dtype,
    )

    def _failed_result(qme_solution: jax.Array) -> FirstOrderDSGEResult:
        return FirstOrderDSGEResult(
            solution_matrix=empty,
            qme_solution=qme_solution,
            converged=jnp.asarray(False),
            state_transition=empty[:, : timings.nPast_not_future_and_mixed],
            shock_impact=empty[:, timings.nPast_not_future_and_mixed :],
        )

    def _successful_result(qme_solution: jax.Array) -> FirstOrderDSGEResult:
        sol_compact = qme_solution[reverse_dynamic_order_idx][:, past_in_comb_idx]

        if timings.nFuture_not_past_and_mixed > 0:
            d_block = sol_compact[-timings.nFuture_not_past_and_mixed :, :]
        else:
            d_block = jnp.zeros((0, sol_compact.shape[1]), dtype=sol_compact.dtype)

        l_block = qme_solution[past_not_future_and_mixed_idx][:, past_in_comb_idx]

        a_bar_zero_u = a_zero[: timings.nPresent_only][:, timings.present_only_idx]
        a_plus_u = a_plus[: timings.nPresent_only]
        a_tilde_zero_u = a_zero[: timings.nPresent_only][:, timings.present_but_not_only_idx]
        a_minus_u = a_minus[: timings.nPresent_only]

        if timings.nPresent_only > 0:
            rhs = a_tilde_zero_u @ qme_solution[:, past_in_comb_idx]
            if timings.nFuture_not_past_and_mixed > 0:
                rhs = rhs + a_plus_u @ d_block @ l_block
            rhs = rhs + a_minus_u
            upper_block = -jnp.linalg.solve(a_bar_zero_u, rhs)
        else:
            upper_block = jnp.zeros((0, n_past), dtype=qme_solution.dtype)

        transition = jnp.vstack([upper_block, sol_compact])[reorder_idx]
        selector_matrix = jnp.eye(timings.nVars, dtype=transition.dtype)[state_idx, :]
        if timings.nFuture_not_past_and_mixed > 0:
            m_matrix = transition[future_idx, :] @ selector_matrix
        else:
            m_matrix = jnp.zeros((0, timings.nVars), dtype=transition.dtype)

        current_block = grad_zero + grad_plus @ m_matrix
        exogenous = -jnp.linalg.solve(current_block, grad_exo)
        solution_matrix = jnp.concatenate([transition, exogenous], axis=1)
        return FirstOrderDSGEResult(
            solution_matrix=solution_matrix,
            qme_solution=qme_solution,
            converged=jnp.asarray(True),
            state_transition=transition,
            shock_impact=exogenous,
        )

    return lax.cond(
        qme_result.converged,
        _successful_result,
        _failed_result,
        qme_result.solution,
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


def solve_third_order_dsge_solution(
    jacobian: Union[jax.Array, np.ndarray],
    hessian: Union[jax.Array, np.ndarray],
    third_order_derivatives: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    second_order_auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    third_order_auxiliary_matrices: Optional[ThirdOrderAuxiliaryMatrices] = None,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    sylvester_algorithm: str = "doubling",
    sylvester_tol: float = 1e-14,
    sylvester_acceptance_tol: float = 1e-10,
    sylvester_max_iter: int = 500,
) -> ThirdOrderDSGEResult:
    grad1 = jnp.asarray(jacobian, dtype=jnp.float64)
    grad2 = jnp.asarray(hessian, dtype=jnp.float64)
    grad3 = jnp.asarray(third_order_derivatives, dtype=jnp.float64)
    first_order_raw = _coerce_first_order_solution_matrix(first_order_solution)
    if second_order_auxiliary_matrices is None:
        second_order_auxiliary_matrices = create_second_order_auxiliary_matrices(
            timings,
            dtype=grad1.dtype,
        )
    if third_order_auxiliary_matrices is None:
        third_order_auxiliary_matrices = create_third_order_auxiliary_matrices(
            timings,
            dtype=grad1.dtype,
        )
    second_order_full = _coerce_second_order_solution_matrix(
        second_order_solution,
        timings,
        auxiliary_matrices=second_order_auxiliary_matrices,
    )

    dynamic_dim = (
        timings.nFuture_not_past_and_mixed
        + timings.nVars
        + timings.nPast_not_future_and_mixed
        + timings.nExo
    )
    n = timings.nVars
    n_minus = timings.nPast_not_future_and_mixed
    n_plus = timings.nFuture_not_past_and_mixed
    n_exo = timings.nExo
    state_dim = n_minus + 1 + n_exo
    expected_first_order_shape = (n, n_minus + n_exo)
    if grad1.shape != (n, dynamic_dim):
        raise ValueError(
            f"jacobian must have shape ({n}, {dynamic_dim}), got {grad1.shape}."
        )
    if grad2.shape != (n, dynamic_dim * dynamic_dim):
        raise ValueError(
            f"hessian must have shape ({n}, {dynamic_dim * dynamic_dim}), got {grad2.shape}."
        )
    if grad3.shape != (n, dynamic_dim**3):
        raise ValueError(
            f"third_order_derivatives must have shape ({n}, {dynamic_dim ** 3}), got {grad3.shape}."
        )
    if first_order_raw.shape != expected_first_order_shape:
        raise ValueError(
            "first_order_solution must have shape "
            f"{expected_first_order_shape}, got {first_order_raw.shape}."
        )
    if second_order_full.shape != (n, state_dim**2):
        raise ValueError(
            f"second_order_solution must have shape ({n}, {state_dim ** 2}), got {second_order_full.shape}."
        )

    i_plus = np.asarray(timings.future_not_past_and_mixed_idx, dtype=np.int32)
    i_minus = np.asarray(timings.past_not_future_and_mixed_idx, dtype=np.int32)
    identity_n = jnp.eye(n, dtype=grad1.dtype)
    first_order_augmented = _augmented_first_order_solution(first_order_raw, timings)
    state_transition = jnp.concatenate(
        [
            first_order_augmented[i_minus, :],
            jnp.concatenate(
                [
                    jnp.zeros((n_exo + 1, n_minus), dtype=grad1.dtype),
                    jnp.eye(n_exo + 1, dtype=grad1.dtype)[:, :1],
                    jnp.zeros((n_exo + 1, n_exo), dtype=grad1.dtype),
                ],
                axis=1,
            ),
        ],
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
    reduced_second_order = jnp.concatenate(
        [
            second_order_full[i_minus, :],
            jnp.zeros((n_exo + 1, state_dim**2), dtype=grad1.dtype),
        ],
        axis=0,
    )
    second_order_mixed = jnp.concatenate(
        [
            (
                second_order_full @ jnp.kron(state_transition, state_transition)
                + first_order_augmented @ reduced_second_order
            )[i_plus, :],
            second_order_full,
            jnp.zeros((n_minus + n_exo, state_dim**2), dtype=grad1.dtype),
        ],
        axis=0,
    )
    second_order_plus_zero = jnp.concatenate(
        [
            second_order_full[i_plus, :],
            jnp.zeros((n_minus + n + n_exo, state_dim**2), dtype=grad1.dtype),
        ],
        axis=0,
    )

    m_matrix = (
        -grad1[:, :n_plus]
        @ first_order_augmented[i_plus, :n_minus]
        @ identity_n[i_minus, :]
        - grad1[:, n_plus : n_plus + n]
    )
    rhs_a = grad1[:, :n_plus] @ identity_n[i_plus, :]

    try:
        a_matrix = jnp.linalg.solve(m_matrix, rhs_a)
    except Exception as exc:
        raise ValueError("Third-order solve failed while inverting the linearized system.") from exc

    tmpkron_b = jnp.kron(state_transition, second_order_auxiliary_matrices.sigma)
    b_full = (
        tmpkron_b
        + third_order_auxiliary_matrices.swap23_left_state
        @ tmpkron_b
        @ third_order_auxiliary_matrices.swap23_right_state
        + third_order_auxiliary_matrices.swap13_left_state
        @ tmpkron_b
        @ third_order_auxiliary_matrices.swap13_right_state
    )
    b_matrix = (
        third_order_auxiliary_matrices.uncompression_matrix
        @ b_full
        @ third_order_auxiliary_matrices.compression_matrix
        + third_order_auxiliary_matrices.uncompression_matrix
        @ jnp.kron(state_transition, jnp.kron(state_transition, state_transition))
        @ third_order_auxiliary_matrices.compression_matrix
    )

    sigma_kron = jnp.kron(first_order_plus_zero, first_order_plus_zero) @ second_order_auxiliary_matrices.sigma
    grad3_sigma = jnp.kron(stacked, sigma_kron)
    x3 = (
        grad3 @ grad3_sigma
        + grad3
        @ third_order_auxiliary_matrices.swap23_left_dynamic
        @ grad3_sigma
        @ third_order_auxiliary_matrices.swap23_right_state
        + grad3
        @ third_order_auxiliary_matrices.swap13_left_dynamic
        @ grad3_sigma
        @ third_order_auxiliary_matrices.swap13_right_state
    )

    tmpkron1 = jnp.kron(first_order_plus_zero, second_order_plus_zero)
    tmpkron2 = jnp.kron(second_order_auxiliary_matrices.sigma, state_transition)
    out2 = grad2 @ tmpkron1 @ tmpkron2
    out2 = (
        out2
        + grad2
        @ tmpkron1
        @ third_order_auxiliary_matrices.swap12_left
        @ tmpkron2
        @ third_order_auxiliary_matrices.swap12_right
        + grad2 @ jnp.kron(stacked, second_order_mixed)
        + grad2 @ jnp.kron(stacked, second_order_plus_zero @ second_order_auxiliary_matrices.sigma)
        + (grad1[:, :n_plus] @ identity_n[i_plus, :])
        @ second_order_full
        @ jnp.kron(state_transition, reduced_second_order)
    )
    x3 = (x3 + out2 @ third_order_auxiliary_matrices.permutation_matrix) @ third_order_auxiliary_matrices.compression_matrix
    x3 = x3 + grad3 @ jnp.kron(stacked, jnp.kron(stacked, stacked)) @ third_order_auxiliary_matrices.compression_matrix

    try:
        c_matrix = jnp.linalg.solve(m_matrix, x3)
    except Exception as exc:
        raise ValueError("Third-order solve failed while assembling the Sylvester right-hand side.") from exc

    compressed_guess = None
    if initial_guess is not None:
        guess = jnp.asarray(initial_guess, dtype=grad1.dtype)
        compressed_cols = third_order_auxiliary_matrices.compression_matrix.shape[1]
        full_cols = state_dim**3
        if guess.shape == (n, full_cols):
            compressed_guess = guess @ third_order_auxiliary_matrices.compression_matrix
        elif guess.shape == (n, compressed_cols):
            compressed_guess = guess
        else:
            raise ValueError(
                "initial_guess must match either the compressed or full third-order shape, "
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
    solution_matrix = (
        sylvester_outcome.solution @ third_order_auxiliary_matrices.uncompression_matrix
    )
    return ThirdOrderDSGEResult(
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


def solve_third_order_stochastic_steady_state(
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    third_order_solution: Union[ThirdOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    pruning: bool = False,
    second_order_auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    third_order_auxiliary_matrices: Optional[ThirdOrderAuxiliaryMatrices] = None,
    initial_guess: Optional[Union[jax.Array, np.ndarray]] = None,
    tol: float = 1e-14,
    max_iter: int = 100,
) -> ThirdOrderStochasticSteadyStateResult:
    first_order_raw = _coerce_first_order_solution_matrix(first_order_solution)
    first_order_augmented = _augmented_first_order_solution(first_order_raw, timings)
    if second_order_auxiliary_matrices is None:
        second_order_auxiliary_matrices = create_second_order_auxiliary_matrices(
            timings,
            dtype=first_order_augmented.dtype,
        )
    if third_order_auxiliary_matrices is None:
        third_order_auxiliary_matrices = create_third_order_auxiliary_matrices(
            timings,
            dtype=first_order_augmented.dtype,
        )
    second_order_full = _coerce_second_order_solution_matrix(
        second_order_solution,
        timings,
        auxiliary_matrices=second_order_auxiliary_matrices,
    )
    third_order_full = _coerce_third_order_solution_matrix(
        third_order_solution,
        timings,
        auxiliary_matrices=third_order_auxiliary_matrices,
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
    if third_order_full.shape != (n, (n_minus + 1 + n_exo) ** 3):
        raise ValueError(
            "third_order_solution must have shape "
            f"({n}, {(n_minus + 1 + n_exo) ** 3}), got {third_order_full.shape}."
        )

    aug_state = jnp.concatenate(
        [
            jnp.zeros(n_minus, dtype=dtype),
            jnp.ones(1, dtype=dtype),
            jnp.zeros(n_exo, dtype=dtype),
        ]
    )
    initial_linear_system = jnp.eye(n_minus, dtype=dtype) - first_order_augmented[i_minus, :n_minus]
    initial_rhs = (second_order_full @ jnp.kron(aug_state, aug_state) / 2.0)[i_minus]
    try:
        default_initial_state = jnp.linalg.solve(initial_linear_system, initial_rhs)
    except Exception as exc:
        raise ValueError(
            "Third-order stochastic steady-state solve failed while forming the initial guess."
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
        return ThirdOrderStochasticSteadyStateResult(
            state_vector=state_vector,
            reduced_state=reduced_state,
            converged=True,
            iterations=0,
        )

    state_plus_mask = np.concatenate(
        [np.ones(n_minus + 1, dtype=bool), np.zeros(n_exo, dtype=bool)]
    )
    state_only_mask = np.concatenate(
        [np.ones(n_minus, dtype=bool), np.zeros(n_exo + 1, dtype=bool)]
    )
    kron_splus_s = np.kron(state_plus_mask, state_only_mask)
    kron_splus_splus = np.kron(state_plus_mask, state_plus_mask)
    kron_splus_splus_splus = np.kron(state_plus_mask, kron_splus_splus)
    kron_s_splus_splus = np.kron(kron_splus_splus, state_only_mask)

    reduced_transition = first_order_augmented[i_minus, :n_minus]
    reduced_second_order_jacobian = second_order_full[i_minus][:, np.flatnonzero(kron_splus_s)]
    reduced_second_order_constant = second_order_full[i_minus][:, np.flatnonzero(kron_splus_splus)]
    reduced_third_order_jacobian = third_order_full[i_minus][:, np.flatnonzero(kron_s_splus_splus)]
    reduced_third_order_constant = third_order_full[i_minus][:, np.flatnonzero(kron_splus_splus_splus)]

    converged = False
    iterations = max_iter
    identity_reduced = jnp.eye(n_minus, dtype=dtype)

    for iteration in range(1, max_iter + 1):
        augmented_state = jnp.concatenate([reduced_state, jnp.ones(1, dtype=dtype)])
        kron_state_identity = jnp.kron(augmented_state[:, None], identity_reduced)
        kron_pair = jnp.kron(augmented_state, augmented_state)
        kron_pair_identity = jnp.kron(kron_pair[:, None], identity_reduced)
        mapped_state = (
            reduced_transition @ reduced_state
            + reduced_second_order_constant @ kron_pair / 2.0
            + reduced_third_order_constant
            @ jnp.kron(augmented_state, kron_pair)
            / 6.0
        )
        jacobian = (
            reduced_transition
            + reduced_second_order_jacobian @ kron_state_identity
            + reduced_third_order_jacobian @ kron_pair_identity / 2.0
            - identity_reduced
        )
        try:
            delta = jnp.linalg.solve(jacobian, mapped_state - reduced_state)
        except Exception as exc:
            raise ValueError("Third-order stochastic steady-state Newton step failed.") from exc

        if iteration > 5 and bool(
            np.asarray(jnp.allclose(mapped_state, reduced_state, rtol=tol, atol=0.0))
        ):
            converged = True
            iterations = iteration
            break

        reduced_state = reduced_state - delta

    augmented_state = jnp.concatenate([reduced_state, jnp.ones(1, dtype=dtype)])
    kron_pair = jnp.kron(augmented_state, augmented_state)
    full_second_order_constant = second_order_full[:, np.flatnonzero(kron_splus_splus)]
    full_third_order_constant = third_order_full[:, np.flatnonzero(kron_splus_splus_splus)]
    state_vector = (
        first_order_augmented[:, :n_minus] @ reduced_state
        + full_second_order_constant @ kron_pair / 2.0
        + full_third_order_constant @ jnp.kron(augmented_state, kron_pair) / 6.0
    )
    fixed_point = third_order_state_update(
        state_vector,
        zero_shock,
        first_order_raw,
        second_order_full,
        third_order_full,
        timings,
    )
    converged = converged or bool(
        np.asarray(jnp.allclose(fixed_point, state_vector, rtol=tol, atol=0.0))
    )
    return ThirdOrderStochasticSteadyStateResult(
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


def third_order_state_update(
    state: Union[jax.Array, np.ndarray],
    shock: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    third_order_solution: Union[ThirdOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    second_order_auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    third_order_auxiliary_matrices: Optional[ThirdOrderAuxiliaryMatrices] = None,
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
        auxiliary_matrices=second_order_auxiliary_matrices,
    )
    third_order_full = _coerce_third_order_solution_matrix(
        third_order_solution,
        timings,
        auxiliary_matrices=third_order_auxiliary_matrices,
    )
    reduced_state = jnp.take(
        state_arr,
        jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32),
    )
    augmented_state = jnp.concatenate(
        [reduced_state, jnp.ones(1, dtype=state_arr.dtype), shock_arr]
    )
    kron_pair = jnp.kron(augmented_state, augmented_state)
    return (
        first_order_augmented @ augmented_state
        + second_order_full @ kron_pair / 2.0
        + third_order_full @ jnp.kron(kron_pair, augmented_state) / 6.0
    )


def pruned_third_order_state_update(
    pruned_states: Sequence[Union[jax.Array, np.ndarray]],
    shock: Union[jax.Array, np.ndarray],
    first_order_solution: Union[FirstOrderDSGEResult, jax.Array, np.ndarray],
    second_order_solution: Union[SecondOrderDSGEResult, jax.Array, np.ndarray],
    third_order_solution: Union[ThirdOrderDSGEResult, jax.Array, np.ndarray],
    timings: DSGETimings,
    *,
    second_order_auxiliary_matrices: Optional[SecondOrderAuxiliaryMatrices] = None,
    third_order_auxiliary_matrices: Optional[ThirdOrderAuxiliaryMatrices] = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if len(pruned_states) != 3:
        raise ValueError(
            f"pruned_states must contain exactly 3 state vectors, got {len(pruned_states)}."
        )
    linear_state = jnp.asarray(pruned_states[0], dtype=jnp.float64)
    quadratic_state = jnp.asarray(pruned_states[1], dtype=jnp.float64)
    cubic_state = jnp.asarray(pruned_states[2], dtype=jnp.float64)
    shock_arr = jnp.asarray(shock, dtype=jnp.float64)
    if (
        linear_state.shape != (timings.nVars,)
        or quadratic_state.shape != (timings.nVars,)
        or cubic_state.shape != (timings.nVars,)
    ):
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
        auxiliary_matrices=second_order_auxiliary_matrices,
    )
    third_order_full = _coerce_third_order_solution_matrix(
        third_order_solution,
        timings,
        auxiliary_matrices=third_order_auxiliary_matrices,
    )
    state_idx = jnp.asarray(timings.past_not_future_and_mixed_idx, dtype=jnp.int32)
    augmented_linear_state = jnp.concatenate(
        [
            jnp.take(linear_state, state_idx),
            jnp.ones(1, dtype=linear_state.dtype),
            shock_arr,
        ]
    )
    augmented_linear_no_constant = jnp.concatenate(
        [
            jnp.take(linear_state, state_idx),
            jnp.zeros(1, dtype=linear_state.dtype),
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
    augmented_cubic_state = jnp.concatenate(
        [
            jnp.take(cubic_state, state_idx),
            jnp.zeros(1, dtype=cubic_state.dtype),
            jnp.zeros_like(shock_arr),
        ]
    )
    kron_linear = jnp.kron(augmented_linear_state, augmented_linear_state)
    linear_component = first_order_augmented @ augmented_linear_state
    quadratic_component = (
        first_order_augmented @ augmented_quadratic_state
        + second_order_full @ kron_linear / 2.0
    )
    cubic_component = (
        first_order_augmented @ augmented_cubic_state
        + second_order_full
        @ jnp.kron(augmented_linear_no_constant, augmented_quadratic_state)
        + third_order_full
        @ jnp.kron(kron_linear, augmented_linear_state)
        / 6.0
    )
    return linear_component, quadratic_component, cubic_component


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


def first_order_state_update(
    solution_matrix: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    reduced_state: Union[jax.Array, np.ndarray],
    shocks: Union[jax.Array, np.ndarray],
) -> jax.Array:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    reduced = jnp.asarray(reduced_state, dtype=solution.dtype)
    shock = jnp.asarray(shocks, dtype=solution.dtype)
    expected_solution_shape = (
        timings.nVars,
        timings.nPast_not_future_and_mixed + timings.nExo,
    )
    if solution.shape != expected_solution_shape:
        raise ValueError(
            "solution_matrix must have shape "
            f"{expected_solution_shape}, got {solution.shape}."
        )
    if reduced.shape != (timings.nPast_not_future_and_mixed,):
        raise ValueError(
            "reduced_state must have shape "
            f"({timings.nPast_not_future_and_mixed},), got {reduced.shape}."
        )
    if shock.shape != (timings.nExo,):
        raise ValueError(
            f"shocks must have shape ({timings.nExo},), got {shock.shape}."
        )
    return solution @ jnp.concatenate([reduced, shock], axis=0)


def rollout_first_order_solution(
    solution_matrix: Union[jax.Array, np.ndarray],
    timings: DSGETimings,
    shocks: Union[jax.Array, np.ndarray],
    *,
    initial_reduced_state: Optional[Union[jax.Array, np.ndarray]] = None,
) -> jax.Array:
    solution = jnp.asarray(solution_matrix, dtype=jnp.float64)
    shock_matrix = jnp.asarray(shocks, dtype=solution.dtype)
    expected_solution_shape = (
        timings.nVars,
        timings.nPast_not_future_and_mixed + timings.nExo,
    )
    if solution.shape != expected_solution_shape:
        raise ValueError(
            "solution_matrix must have shape "
            f"{expected_solution_shape}, got {solution.shape}."
        )
    if shock_matrix.ndim != 2:
        raise ValueError(f"shocks must be rank-2, got shape {shock_matrix.shape}.")
    if shock_matrix.shape[0] != timings.nExo:
        raise ValueError(
            f"shocks must have {timings.nExo} rows, got {shock_matrix.shape[0]}."
        )

    if initial_reduced_state is None:
        reduced_state = jnp.zeros(
            (timings.nPast_not_future_and_mixed,),
            dtype=solution.dtype,
        )
    else:
        reduced_state = jnp.asarray(initial_reduced_state, dtype=solution.dtype)
        if reduced_state.shape != (timings.nPast_not_future_and_mixed,):
            raise ValueError(
                "initial_reduced_state must have shape "
                f"({timings.nPast_not_future_and_mixed},), got {reduced_state.shape}."
            )

    state_indices = jnp.asarray(
        timings.past_not_future_and_mixed_idx,
        dtype=jnp.int32,
    )

    def step(
        current_reduced_state: jax.Array,
        shock_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        next_full_state = first_order_state_update(
            solution,
            timings,
            current_reduced_state,
            shock_t,
        )
        next_reduced_state = next_full_state[state_indices]
        return next_reduced_state, next_full_state

    _, full_states_t = lax.scan(step, reduced_state, shock_matrix.T)
    return full_states_t.T
