from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

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
