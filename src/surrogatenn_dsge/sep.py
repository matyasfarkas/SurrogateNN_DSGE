from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


SEPResidualFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, object], jax.Array]
SEPExpectationFn = Callable[[jax.Array, jax.Array, object], jax.Array]
SEPConditionalResidualFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, object],
    jax.Array,
]


class GaussHermiteRule(NamedTuple):
    nodes: jax.Array
    weights: jax.Array


class SEPSolution(NamedTuple):
    stacked_states: jax.Array
    mean_path: jax.Array
    residual_norm: float
    converged: bool
    iterations: int
    group_counts: tuple[int, ...]


@dataclass(frozen=True)
class SEPConfig:
    periods: int = 20
    branching_order: int = 1
    nnodes: int = 3
    shock_scale: float = 1.0
    sparse_tree: bool = False
    max_iter: int = 80
    tol: float = 1e-7
    line_search_factor: float = 0.5
    line_search_min_alpha: float = 1e-4
    newton_regularization: float = 1e-8


def gauss_hermite_rule(nnodes: int, shock_dim: int, shock_scale: float = 1.0) -> GaussHermiteRule:
    if shock_dim < 0:
        raise ValueError("shock_dim must be non-negative.")
    if shock_dim == 0:
        return GaussHermiteRule(
            nodes=jnp.zeros((1, 0), dtype=jnp.float64),
            weights=jnp.ones((1,), dtype=jnp.float64),
        )

    one_d_nodes, one_d_weights = np.polynomial.hermite.hermgauss(nnodes)
    one_d_nodes = np.sqrt(2.0) * one_d_nodes * shock_scale
    one_d_weights = one_d_weights / np.sqrt(np.pi)

    meshes = np.meshgrid(*([one_d_nodes] * shock_dim), indexing="ij")
    weight_meshes = np.meshgrid(*([one_d_weights] * shock_dim), indexing="ij")
    nodes = np.stack([mesh.reshape(-1) for mesh in meshes], axis=1)
    weights = np.prod(np.stack([mesh.reshape(-1) for mesh in weight_meshes], axis=1), axis=1)
    return GaussHermiteRule(
        nodes=jnp.asarray(nodes, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
    )


def _gauss_hermite_sparse_rule(
    nnodes: int,
    shock_dim: int,
    shock_scale: float = 1.0,
) -> GaussHermiteRule:
    if shock_dim < 0:
        raise ValueError("shock_dim must be non-negative.")
    if shock_dim == 0:
        return GaussHermiteRule(
            nodes=jnp.zeros((1, 0), dtype=jnp.float64),
            weights=jnp.ones((1,), dtype=jnp.float64),
        )

    one_d_nodes, one_d_weights = np.polynomial.hermite.hermgauss(nnodes)
    one_d_nodes = np.sqrt(2.0) * one_d_nodes * shock_scale
    one_d_weights = one_d_weights / np.sqrt(np.pi)
    zero_idx = int(np.argmin(np.abs(one_d_nodes)))
    if zero_idx != 0:
        one_d_nodes[[0, zero_idx]] = one_d_nodes[[zero_idx, 0]]
        one_d_weights[[0, zero_idx]] = one_d_weights[[zero_idx, 0]]

    num_nodes = shock_dim * nnodes
    nodes = np.zeros((num_nodes, shock_dim), dtype=np.float64)
    weights = np.zeros((num_nodes,), dtype=np.float64)
    offset = 0
    for dim in range(shock_dim):
        for local_idx in range(nnodes):
            nodes[offset, dim] = one_d_nodes[local_idx]
            weights[offset] = one_d_weights[local_idx] / shock_dim
            offset += 1
    weights /= np.sum(weights)
    return GaussHermiteRule(
        nodes=jnp.asarray(nodes, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
    )


def _group_counts(
    periods: int,
    branching_order: int,
    num_nodes: int,
    *,
    sparse_tree: bool,
) -> tuple[int, ...]:
    counts = [1]
    for t in range(1, periods + 1):
        if sparse_tree:
            if branching_order <= 0 or num_nodes <= 1:
                counts.append(1)
            else:
                branch_levels = min(max(t - 1, 0), branching_order)
                counts.append(1 + (num_nodes - 1) * branch_levels)
        elif t <= branching_order:
            counts.append(num_nodes**t)
        else:
            counts.append(num_nodes**branching_order)
    return tuple(counts)


def _group_probabilities(
    rule: GaussHermiteRule,
    periods: int,
    branching_order: int,
    *,
    sparse_tree: bool,
) -> tuple[jax.Array, ...]:
    num_nodes = int(rule.weights.shape[0])
    if sparse_tree:
        probs = [jnp.ones((1,), dtype=jnp.float64)]
        if branching_order <= 0 or num_nodes <= 1:
            probs.extend(
                [jnp.ones((1,), dtype=jnp.float64) for _ in range(periods)]
            )
            return tuple(probs)
        trunk_prob = jnp.asarray(1.0, dtype=jnp.float64)
        side_probs = jnp.zeros((0,), dtype=jnp.float64)
        for t in range(1, periods + 1):
            if t <= 1:
                probs.append(jnp.asarray([trunk_prob], dtype=jnp.float64))
                continue
            if t <= branching_order + 1:
                branch_parent_prob = trunk_prob
                trunk_prob = branch_parent_prob * rule.weights[0]
                side_probs = jnp.concatenate(
                    [side_probs, branch_parent_prob * rule.weights[1:]],
                    axis=0,
                )
            probs.append(
                jnp.concatenate(
                    [jnp.asarray([trunk_prob], dtype=jnp.float64), side_probs],
                    axis=0,
                )
            )
        return tuple(probs)

    probs = [jnp.ones((1,), dtype=jnp.float64)]
    for t in range(1, periods + 1):
        if t <= branching_order:
            probs.append(jnp.kron(probs[-1], rule.weights))
        else:
            probs.append(probs[-1])
    return tuple(probs)


def _sparse_branch_info(group: int, num_nodes: int) -> tuple[int, int] | None:
    if group == 0 or num_nodes <= 1:
        return None
    idx = group - 1
    branch_time = idx // (num_nodes - 1) + 2
    node_index = idx % (num_nodes - 1) + 1
    return branch_time, node_index


def _sparse_branch_group_index(branch_time: int, node_index: int, num_nodes: int) -> int:
    return 1 + (branch_time - 2) * (num_nodes - 1) + (node_index - 1)


def _group_shock_at_time(
    rule: GaussHermiteRule,
    group: int,
    time: int,
    branching_order: int,
    num_nodes: int,
    *,
    sparse_tree: bool,
) -> jax.Array:
    if rule.nodes.shape[1] == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if not sparse_tree:
        return rule.nodes[group % num_nodes]
    if group == 0 or num_nodes <= 1:
        return rule.nodes[0]
    info = _sparse_branch_info(group, num_nodes)
    if info is None:
        return rule.nodes[0]
    branch_time, node_index = info
    if time == branch_time and branch_time <= branching_order + 1:
        return rule.nodes[node_index]
    return rule.nodes[0]


def _parent_group(
    group: int,
    time: int,
    branching_order: int,
    num_nodes: int,
    *,
    sparse_tree: bool,
) -> int:
    if time <= 1:
        return 0
    if not sparse_tree:
        if time > branching_order:
            return group
        return (group // num_nodes) if num_nodes > 0 else 0
    if group == 0 or num_nodes <= 1:
        return 0
    info = _sparse_branch_info(group, num_nodes)
    if info is None:
        return group
    branch_time, _ = info
    return 0 if time == branch_time else group


def _child_groups(
    group: int,
    time: int,
    branching_order: int,
    num_nodes: int,
    *,
    sparse_tree: bool,
) -> tuple[int, ...]:
    if sparse_tree:
        if time > branching_order or num_nodes <= 1:
            return (group,)
        if group != 0:
            return (group,)
        return tuple(
            [0]
            + [
                _sparse_branch_group_index(time + 1, node_index, num_nodes)
                for node_index in range(1, num_nodes)
            ]
        )
    if time < branching_order:
        start = group * num_nodes
        return tuple(start + k for k in range(num_nodes))
    return (group,)


def solve_stochastic_extended_path(
    residual_fn: SEPResidualFn,
    *,
    initial_state: Sequence[float],
    terminal_state: Sequence[float],
    shock_dim: int,
    config: SEPConfig = SEPConfig(),
    deterministic_shocks: Optional[Sequence[Sequence[float]]] = None,
    params: object = None,
    expectation_fn: Optional[SEPExpectationFn] = None,
    initial_guess: Optional[Sequence[Sequence[float]]] = None,
) -> SEPSolution:
    return _solve_stochastic_extended_path_impl(
        initial_state=initial_state,
        terminal_state=terminal_state,
        shock_dim=shock_dim,
        config=config,
        deterministic_shocks=deterministic_shocks,
        params=params,
        expectation_fn=expectation_fn,
        initial_guess=initial_guess,
        residual_fn=residual_fn,
        conditional_residual_fn=None,
    )


def solve_stochastic_extended_path_residual_expectation(
    conditional_residual_fn: SEPConditionalResidualFn,
    *,
    initial_state: Sequence[float],
    terminal_state: Sequence[float],
    shock_dim: int,
    config: SEPConfig = SEPConfig(),
    deterministic_shocks: Optional[Sequence[Sequence[float]]] = None,
    params: object = None,
    initial_guess: Optional[Sequence[Sequence[float]]] = None,
) -> SEPSolution:
    return _solve_stochastic_extended_path_impl(
        initial_state=initial_state,
        terminal_state=terminal_state,
        shock_dim=shock_dim,
        config=config,
        deterministic_shocks=deterministic_shocks,
        params=params,
        expectation_fn=None,
        initial_guess=initial_guess,
        residual_fn=None,
        conditional_residual_fn=conditional_residual_fn,
    )


def _solve_stochastic_extended_path_impl(
    *,
    initial_state: Sequence[float],
    terminal_state: Sequence[float],
    shock_dim: int,
    config: SEPConfig,
    deterministic_shocks: Optional[Sequence[Sequence[float]]],
    params: object,
    expectation_fn: Optional[SEPExpectationFn],
    initial_guess: Optional[Sequence[Sequence[float]]],
    residual_fn: Optional[SEPResidualFn],
    conditional_residual_fn: Optional[SEPConditionalResidualFn],
) -> SEPSolution:
    initial_state_arr = jnp.asarray(initial_state, dtype=jnp.float64)
    terminal_state_arr = jnp.asarray(terminal_state, dtype=jnp.float64)
    state_dim = int(initial_state_arr.shape[0])
    if terminal_state_arr.shape != initial_state_arr.shape:
        raise ValueError("initial_state and terminal_state must have identical shapes.")
    if residual_fn is None and conditional_residual_fn is None:
        raise ValueError(
            "Either `residual_fn` or `conditional_residual_fn` must be provided."
        )

    if deterministic_shocks is None:
        deterministic = jnp.zeros((config.periods, shock_dim), dtype=jnp.float64)
    else:
        deterministic = jnp.asarray(deterministic_shocks, dtype=jnp.float64)
        if deterministic.shape != (config.periods, shock_dim):
            raise ValueError(
                "deterministic_shocks must have shape "
                f"({config.periods}, {shock_dim}), got {deterministic.shape}."
            )

    rule = (
        _gauss_hermite_sparse_rule(config.nnodes, shock_dim, config.shock_scale)
        if config.sparse_tree
        else gauss_hermite_rule(config.nnodes, shock_dim, config.shock_scale)
    )
    num_nodes = int(rule.weights.shape[0])
    counts = _group_counts(
        config.periods,
        config.branching_order,
        num_nodes,
        sparse_tree=config.sparse_tree,
    )
    probabilities = _group_probabilities(
        rule,
        config.periods,
        config.branching_order,
        sparse_tree=config.sparse_tree,
    )

    if expectation_fn is None and conditional_residual_fn is None:
        def expectation_fn(next_state: jax.Array, next_shock: jax.Array, _params: object) -> jax.Array:
            return next_state

    time_offsets = [0]
    for t in range(1, config.periods + 1):
        time_offsets.append(time_offsets[-1] + counts[t] * state_dim)

    def unflatten(stacked: jax.Array) -> tuple[jax.Array, ...]:
        values = []
        for t in range(1, config.periods + 1):
            start = time_offsets[t - 1]
            end = time_offsets[t]
            values.append(jnp.reshape(stacked[start:end], (counts[t], state_dim)))
        return tuple(values)

    if initial_guess is None:
        guess = jnp.concatenate(
            [
                jnp.tile(terminal_state_arr, (counts[t], 1)).reshape(-1)
                for t in range(1, config.periods + 1)
            ],
            axis=0,
        )
    else:
        guess_arr = jnp.asarray(initial_guess, dtype=jnp.float64)
        guess = guess_arr.reshape(-1)

    def residual_vector(stacked: jax.Array) -> jax.Array:
        states_by_time = unflatten(stacked)
        residuals = []
        zero_shock = jnp.zeros((shock_dim,), dtype=jnp.float64)
        terminal_expectation = None
        if conditional_residual_fn is None:
            terminal_expectation = expectation_fn(terminal_state_arr, zero_shock, params)
        for t in range(1, config.periods + 1):
            current_states = states_by_time[t - 1]
            prev_states = (
                jnp.expand_dims(initial_state_arr, axis=0)
                if t == 1
                else states_by_time[t - 2]
            )
            next_states = None if t == config.periods else states_by_time[t]
            for g in range(counts[t]):
                if t == 1:
                    prev_state = prev_states[0]
                else:
                    parent = _parent_group(
                        g,
                        t,
                        config.branching_order,
                        num_nodes,
                        sparse_tree=config.sparse_tree,
                    )
                    prev_state = prev_states[parent]

                deterministic_shock = deterministic[t - 1]
                stochastic_time_limit = (
                    config.branching_order + 1
                    if config.sparse_tree
                    else config.branching_order
                )
                stochastic_shock = (
                    _group_shock_at_time(
                        rule,
                        g,
                        t,
                        config.branching_order,
                        num_nodes,
                        sparse_tree=config.sparse_tree,
                    )
                    if t <= stochastic_time_limit and shock_dim > 0
                    else zero_shock
                )
                current_shock = deterministic_shock + stochastic_shock

                if conditional_residual_fn is None:
                    if t == config.periods:
                        expected_term = terminal_expectation
                    else:
                        child_groups = _child_groups(
                            g,
                            t,
                            config.branching_order,
                            num_nodes,
                            sparse_tree=config.sparse_tree,
                        )
                        if len(child_groups) == 1:
                            child_shock = (
                                deterministic[t]
                                + (
                                    _group_shock_at_time(
                                        rule,
                                        child_groups[0],
                                        t + 1,
                                        config.branching_order,
                                        num_nodes,
                                        sparse_tree=config.sparse_tree,
                                    )
                                    if t + 1 <= stochastic_time_limit
                                    else zero_shock
                                )
                            )
                            expected_term = expectation_fn(
                                next_states[child_groups[0]],
                                child_shock,
                                params,
                            )
                        else:
                            child_terms = []
                            for local_idx, child in enumerate(child_groups):
                                child_shock = deterministic[t] + _group_shock_at_time(
                                    rule,
                                    child,
                                    t + 1,
                                    config.branching_order,
                                    num_nodes,
                                    sparse_tree=config.sparse_tree,
                                )
                                child_terms.append(
                                    rule.weights[local_idx]
                                    * expectation_fn(next_states[child], child_shock, params)
                                )
                            expected_term = jnp.sum(
                                jnp.stack(child_terms, axis=0),
                                axis=0,
                            )

                    residuals.append(
                        residual_fn(
                            prev_state,
                            current_states[g],
                            expected_term,
                            current_shock,
                            params,
                        )
                    )
                    continue

                if t == config.periods:
                    residuals.append(
                        conditional_residual_fn(
                            prev_state,
                            current_states[g],
                            terminal_state_arr,
                            current_shock,
                            params,
                        )
                    )
                    continue

                child_groups = _child_groups(
                    g,
                    t,
                    config.branching_order,
                    num_nodes,
                    sparse_tree=config.sparse_tree,
                )
                if len(child_groups) == 1:
                    residuals.append(
                        conditional_residual_fn(
                            prev_state,
                            current_states[g],
                            next_states[child_groups[0]],
                            current_shock,
                            params,
                        )
                    )
                    continue

                child_terms = []
                for local_idx, child in enumerate(child_groups):
                    child_terms.append(
                        rule.weights[local_idx]
                        * conditional_residual_fn(
                            prev_state,
                            current_states[g],
                            next_states[child],
                            current_shock,
                            params,
                        )
                    )
                residuals.append(jnp.sum(jnp.stack(child_terms, axis=0), axis=0))
        return jnp.concatenate(residuals, axis=0)

    residual_norm = float(np.asarray(jnp.linalg.norm(residual_vector(guess), ord=jnp.inf)))
    converged = residual_norm < config.tol
    iterations = 0
    current = guess

    for iteration in range(1, config.max_iter + 1):
        residual = residual_vector(current)
        residual_norm = float(np.asarray(jnp.linalg.norm(residual, ord=jnp.inf)))
        if residual_norm < config.tol:
            converged = True
            iterations = iteration - 1
            break

        jacobian = jax.jacobian(residual_vector)(current)
        regularized = jacobian + config.newton_regularization * jnp.eye(
            jacobian.shape[0],
            dtype=jacobian.dtype,
        )
        step = jnp.linalg.solve(regularized, -residual)

        alpha = 1.0
        candidate = current + alpha * step
        candidate_norm = float(np.asarray(jnp.linalg.norm(residual_vector(candidate), ord=jnp.inf)))
        while candidate_norm >= residual_norm and alpha > config.line_search_min_alpha:
            alpha *= config.line_search_factor
            candidate = current + alpha * step
            candidate_norm = float(np.asarray(jnp.linalg.norm(residual_vector(candidate), ord=jnp.inf)))

        current = candidate
        iterations = iteration

    final_residual = residual_vector(current)
    residual_norm = float(np.asarray(jnp.linalg.norm(final_residual, ord=jnp.inf)))
    converged = residual_norm < config.tol

    states_by_time = unflatten(current)
    mean_path = [initial_state_arr]
    for t in range(1, config.periods + 1):
        weighted = probabilities[t][:, None] * states_by_time[t - 1]
        mean_path.append(jnp.sum(weighted, axis=0))

    return SEPSolution(
        stacked_states=current,
        mean_path=jnp.stack(mean_path, axis=1),
        residual_norm=residual_norm,
        converged=converged,
        iterations=iterations,
        group_counts=counts,
    )
