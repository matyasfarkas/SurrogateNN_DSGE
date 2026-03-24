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
SEPJacobianFn = Callable[[jax.Array], jax.Array]


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
    jacobian_method: str


class _SEPTreeMetadata(NamedTuple):
    parent_indices: tuple[Optional[np.ndarray], ...]
    current_shocks: tuple[jax.Array, ...]
    child_groups: tuple[Optional[tuple[tuple[int, ...], ...]], ...]
    child_shocks: tuple[Optional[tuple[jax.Array, ...]], ...]


@dataclass(frozen=True)
class SEPConfig:
    periods: int = 20
    branching_order: int = 1
    nnodes: int = 3
    shock_scale: float = 1.0
    sparse_tree: bool = False
    expectation_method: str = "gauss_hermite"
    jacobian_method: str = "auto"
    hmc_samples: int = 100
    hmc_warmup: int = 50
    hmc_leapfrog_steps: int = 10
    hmc_step_size: float = 0.1
    hmc_use_tempering: bool = False
    hmc_temperatures: tuple[float, ...] = (1.0, 0.5, 0.25)
    hmc_swap_interval: int = 10
    hmc_seed: int = 0
    max_iter: int = 80
    tol: float = 1e-7
    linear_solver: str = "qr"
    fallback_solver: Optional[str] = None
    stall_iters: int = 25
    stall_rel_tol: float = 1e-4
    stall_abs_tol: float = 1e-10
    line_search: bool = True
    line_search_maxit: int = 6
    line_search_factor: float = 0.5
    line_search_min_alpha: float = 1e-4
    newton_regularization: float = 1e-8
    lm_lambda_scale: float = 10.0
    lm_lambda_min: float = 1e-12
    lm_lambda_max: float = 1e4


def _validate_sep_config(config: SEPConfig, *, shock_dim: int) -> None:
    if config.periods < 1:
        raise ValueError(f"SEPConfig.periods must be >= 1, got {config.periods}.")
    if config.branching_order < 0:
        raise ValueError(
            f"SEPConfig.branching_order must be >= 0, got {config.branching_order}."
        )
    if config.nnodes < 1:
        raise ValueError(f"SEPConfig.nnodes must be >= 1, got {config.nnodes}.")
    if config.shock_scale <= 0.0:
        raise ValueError(
            f"SEPConfig.shock_scale must be > 0, got {config.shock_scale}."
        )
    if config.expectation_method not in {"gauss_hermite", "hmc"}:
        raise ValueError(
            "SEPConfig.expectation_method must be 'gauss_hermite' or 'hmc', "
            f"got {config.expectation_method!r}."
        )
    if config.jacobian_method not in {
        "auto",
        "autodiff",
        "finite_difference",
        "subgradient",
    }:
        raise ValueError(
            "SEPConfig.jacobian_method must be 'auto', 'autodiff', "
            f"'finite_difference', or 'subgradient', got {config.jacobian_method!r}."
        )
    if config.expectation_method == "hmc" and config.jacobian_method in {
        "autodiff",
        "subgradient",
    }:
        raise ValueError(
            "SEPConfig.jacobian_method must not be 'autodiff' or 'subgradient' "
            "when expectation_method='hmc'; use 'auto' or 'finite_difference'."
        )
    if config.hmc_samples < 1:
        raise ValueError(
            f"SEPConfig.hmc_samples must be >= 1, got {config.hmc_samples}."
        )
    if config.hmc_warmup < 0:
        raise ValueError(
            f"SEPConfig.hmc_warmup must be >= 0, got {config.hmc_warmup}."
        )
    if config.hmc_leapfrog_steps < 1:
        raise ValueError(
            "SEPConfig.hmc_leapfrog_steps must be >= 1, "
            f"got {config.hmc_leapfrog_steps}."
        )
    if config.hmc_step_size <= 0.0:
        raise ValueError(
            f"SEPConfig.hmc_step_size must be > 0, got {config.hmc_step_size}."
        )
    if config.hmc_swap_interval < 1:
        raise ValueError(
            f"SEPConfig.hmc_swap_interval must be >= 1, got {config.hmc_swap_interval}."
        )
    if any(temp <= 0.0 for temp in config.hmc_temperatures):
        raise ValueError("SEPConfig.hmc_temperatures must be strictly positive.")
    if config.max_iter < 1:
        raise ValueError(f"SEPConfig.max_iter must be >= 1, got {config.max_iter}.")
    if config.tol <= 0.0:
        raise ValueError(f"SEPConfig.tol must be > 0, got {config.tol}.")
    if config.linear_solver not in {"normal_equations", "qr"}:
        raise ValueError(
            "SEPConfig.linear_solver must be 'normal_equations' or 'qr', "
            f"got {config.linear_solver!r}."
        )
    if config.fallback_solver is not None and config.fallback_solver not in {
        "normal_equations",
        "qr",
    }:
        raise ValueError(
            "SEPConfig.fallback_solver must be None, 'normal_equations', or 'qr', "
            f"got {config.fallback_solver!r}."
        )
    if config.stall_iters < 1:
        raise ValueError(
            f"SEPConfig.stall_iters must be >= 1, got {config.stall_iters}."
        )
    if config.stall_rel_tol < 0.0:
        raise ValueError(
            f"SEPConfig.stall_rel_tol must be >= 0, got {config.stall_rel_tol}."
        )
    if config.stall_abs_tol < 0.0:
        raise ValueError(
            f"SEPConfig.stall_abs_tol must be >= 0, got {config.stall_abs_tol}."
        )
    if config.line_search_maxit < 1:
        raise ValueError(
            f"SEPConfig.line_search_maxit must be >= 1, got {config.line_search_maxit}."
        )
    if not 0.0 < config.line_search_factor < 1.0:
        raise ValueError(
            "SEPConfig.line_search_factor must be in (0, 1), "
            f"got {config.line_search_factor}."
        )
    if not 0.0 < config.line_search_min_alpha <= 1.0:
        raise ValueError(
            "SEPConfig.line_search_min_alpha must be in (0, 1], "
            f"got {config.line_search_min_alpha}."
        )
    if config.newton_regularization <= 0.0:
        raise ValueError(
            "SEPConfig.newton_regularization must be > 0, "
            f"got {config.newton_regularization}."
        )
    if config.lm_lambda_scale <= 1.0:
        raise ValueError(
            f"SEPConfig.lm_lambda_scale must be > 1, got {config.lm_lambda_scale}."
        )
    if config.lm_lambda_min <= 0.0:
        raise ValueError(
            f"SEPConfig.lm_lambda_min must be > 0, got {config.lm_lambda_min}."
        )
    if config.lm_lambda_max < config.newton_regularization:
        raise ValueError(
            "SEPConfig.lm_lambda_max must be >= SEPConfig.newton_regularization, "
            f"got {config.lm_lambda_max} < {config.newton_regularization}."
        )
    if config.sparse_tree and shock_dim > 0 and config.nnodes % 2 == 0:
        raise ValueError(
            "Sparse-tree SEP requires an odd nnodes value so the trunk can use the "
            f"zero Gauss-Hermite node, got nnodes={config.nnodes}."
        )


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


def _precompute_sep_tree_metadata(
    *,
    rule: GaussHermiteRule,
    deterministic: jax.Array,
    counts: tuple[int, ...],
    periods: int,
    branching_order: int,
    num_nodes: int,
    shock_dim: int,
    sparse_tree: bool,
    use_hmc: bool,
) -> _SEPTreeMetadata:
    zero_shock = jnp.zeros((shock_dim,), dtype=jnp.float64)
    stochastic_time_limit = branching_order + 1 if sparse_tree else branching_order
    if use_hmc:
        stochastic_time_limit = branching_order

    parent_indices: list[Optional[np.ndarray]] = []
    current_shocks: list[jax.Array] = []
    child_groups_by_time: list[Optional[tuple[tuple[int, ...], ...]]] = []
    child_shocks_by_time: list[Optional[tuple[jax.Array, ...]]] = []

    for t in range(1, periods + 1):
        if t == 1:
            parent_indices.append(None)
        else:
            parent_indices.append(
                np.asarray(
                    [
                        _parent_group(
                            g,
                            t,
                            branching_order,
                            num_nodes,
                            sparse_tree=sparse_tree,
                        )
                        for g in range(counts[t])
                    ],
                    dtype=np.int64,
                )
            )

        period_current_shocks = []
        for g in range(counts[t]):
            stochastic_shock = (
                _group_shock_at_time(
                    rule,
                    g,
                    t,
                    branching_order,
                    num_nodes,
                    sparse_tree=sparse_tree,
                )
                if (not use_hmc) and t <= stochastic_time_limit and shock_dim > 0
                else zero_shock
            )
            period_current_shocks.append(deterministic[t - 1] + stochastic_shock)
        current_shocks.append(jnp.stack(period_current_shocks, axis=0))

        if t == periods:
            child_groups_by_time.append(None)
            child_shocks_by_time.append(None)
            continue

        period_child_groups: list[tuple[int, ...]] = []
        period_child_shocks: list[jax.Array] = []
        for g in range(counts[t]):
            groups = _child_groups(
                g,
                t,
                branching_order,
                num_nodes,
                sparse_tree=sparse_tree,
            )
            period_child_groups.append(groups)
            next_shocks = []
            for child in groups:
                stochastic_shock = (
                    _group_shock_at_time(
                        rule,
                        child,
                        t + 1,
                        branching_order,
                        num_nodes,
                        sparse_tree=sparse_tree,
                    )
                    if (not use_hmc) and t + 1 <= stochastic_time_limit and shock_dim > 0
                    else zero_shock
                )
                next_shocks.append(deterministic[t] + stochastic_shock)
            period_child_shocks.append(jnp.stack(next_shocks, axis=0))
        child_groups_by_time.append(tuple(period_child_groups))
        child_shocks_by_time.append(tuple(period_child_shocks))

    return _SEPTreeMetadata(
        parent_indices=tuple(parent_indices),
        current_shocks=tuple(current_shocks),
        child_groups=tuple(child_groups_by_time),
        child_shocks=tuple(child_shocks_by_time),
    )


def _hmc_step(
    epsilon: jax.Array,
    *,
    key: jax.Array,
    energy_fn: Callable[[jax.Array], jax.Array],
    variance: float,
    leapfrog_steps: int,
    step_size: float,
    temperature: float = 1.0,
) -> tuple[jax.Array, bool]:
    key_momentum, key_accept = jax.random.split(key)
    scaled_variance = variance * temperature
    inv_scaled_variance = 1.0 / scaled_variance
    momentum = jax.random.normal(
        key_momentum,
        shape=epsilon.shape,
        dtype=jnp.float64,
    ) * jnp.sqrt(inv_scaled_variance)

    def tempered_energy(position: jax.Array) -> jax.Array:
        return energy_fn(position) / temperature

    energy_and_grad = jax.value_and_grad(tempered_energy)
    old_energy = tempered_energy(epsilon) + 0.5 * scaled_variance * jnp.vdot(momentum, momentum)

    eps_new = epsilon
    mom_new = momentum
    energy_value, grad_value = energy_and_grad(eps_new)
    if not bool(jnp.isfinite(energy_value)) or not bool(jnp.all(jnp.isfinite(grad_value))):
        return epsilon, False
    mom_new = mom_new - 0.5 * step_size * grad_value

    for _ in range(max(leapfrog_steps - 1, 0)):
        eps_new = eps_new + step_size * scaled_variance * mom_new
        energy_value, grad_value = energy_and_grad(eps_new)
        if not bool(jnp.isfinite(energy_value)) or not bool(jnp.all(jnp.isfinite(grad_value))):
            return epsilon, False
        mom_new = mom_new - step_size * grad_value

    eps_new = eps_new + step_size * scaled_variance * mom_new
    energy_value, grad_value = energy_and_grad(eps_new)
    if not bool(jnp.isfinite(energy_value)) or not bool(jnp.all(jnp.isfinite(grad_value))):
        return epsilon, False
    mom_new = mom_new - 0.5 * step_size * grad_value

    new_energy = tempered_energy(eps_new) + 0.5 * scaled_variance * jnp.vdot(mom_new, mom_new)
    log_accept_ratio = old_energy - new_energy
    accept_probability = jnp.where(
        jnp.isfinite(log_accept_ratio),
        jnp.exp(jnp.minimum(log_accept_ratio, 0.0)),
        0.0,
    )
    accepted = bool(
        np.asarray(
            jax.random.uniform(key_accept, (), dtype=jnp.float64) < accept_probability
        )
    )
    return (eps_new if accepted else epsilon), accepted


def _hmc_expectation_mean(
    sample_fn: Callable[[jax.Array], jax.Array],
    *,
    shock_dim: int,
    config: SEPConfig,
    key: jax.Array,
) -> jax.Array:
    if shock_dim == 0:
        return sample_fn(jnp.zeros((0,), dtype=jnp.float64))

    variance = float(config.shock_scale**2)

    def energy_fn(epsilon: jax.Array) -> jax.Array:
        sample = sample_fn(epsilon)
        if not bool(jnp.all(jnp.isfinite(sample))):
            return jnp.asarray(jnp.inf, dtype=jnp.float64)
        return 0.5 * jnp.vdot(sample, sample)

    total_draws = config.hmc_samples + config.hmc_warmup

    if not config.hmc_use_tempering:
        chain = jnp.zeros((shock_dim,), dtype=jnp.float64)
        samples: list[jax.Array] = []
        for draw_index in range(total_draws):
            draw_key = jax.random.fold_in(key, draw_index)
            chain, _ = _hmc_step(
                chain,
                key=draw_key,
                energy_fn=energy_fn,
                variance=variance,
                leapfrog_steps=config.hmc_leapfrog_steps,
                step_size=config.hmc_step_size,
            )
            if draw_index >= config.hmc_warmup:
                samples.append(sample_fn(chain))
        return jnp.mean(jnp.stack(samples, axis=0), axis=0)

    temperatures = tuple(float(temp) for temp in config.hmc_temperatures)
    chains = [
        jnp.zeros((shock_dim,), dtype=jnp.float64)
        for _ in range(len(temperatures))
    ]
    cold_samples: list[jax.Array] = []
    for draw_index in range(total_draws):
        for chain_index, temperature in enumerate(temperatures):
            draw_key = jax.random.fold_in(key, draw_index * len(temperatures) + chain_index)
            chains[chain_index], _ = _hmc_step(
                chains[chain_index],
                key=draw_key,
                energy_fn=energy_fn,
                variance=variance,
                leapfrog_steps=config.hmc_leapfrog_steps,
                step_size=config.hmc_step_size,
                temperature=temperature,
            )
        if (draw_index + 1) % config.hmc_swap_interval == 0:
            for chain_index in range(len(temperatures) - 1):
                key_swap = jax.random.fold_in(
                    key,
                    total_draws * len(temperatures) + draw_index * len(temperatures) + chain_index,
                )
                energy_i = float(np.asarray(energy_fn(chains[chain_index])))
                energy_j = float(np.asarray(energy_fn(chains[chain_index + 1])))
                temp_i = temperatures[chain_index]
                temp_j = temperatures[chain_index + 1]
                delta = (1.0 / temp_j - 1.0 / temp_i) * (energy_j - energy_i)
                if np.isfinite(delta):
                    accept_swap = bool(
                        np.asarray(
                            jax.random.uniform(key_swap, (), dtype=jnp.float64)
                            < np.exp(min(delta, 0.0))
                        )
                    )
                    if accept_swap:
                        chains[chain_index], chains[chain_index + 1] = (
                            chains[chain_index + 1],
                            chains[chain_index],
                        )
        if draw_index >= config.hmc_warmup:
            cold_samples.append(sample_fn(chains[-1]))
    return jnp.mean(jnp.stack(cold_samples, axis=0), axis=0)


def _finite_difference_jacobian(
    residual_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
) -> jax.Array:
    x_array = np.asarray(x, dtype=np.float64)
    base = np.asarray(residual_fn(jnp.asarray(x_array, dtype=jnp.float64)), dtype=np.float64)
    jacobian = np.zeros((base.size, x_array.size), dtype=np.float64)
    for idx in range(x_array.size):
        step_scale = max(1.0, abs(float(x_array[idx])))
        step_size = max(np.sqrt(np.finfo(np.float64).eps) * step_scale, 1e-7)
        perturbed = x_array.copy()
        perturbed[idx] += step_size
        shifted = np.asarray(
            residual_fn(jnp.asarray(perturbed, dtype=jnp.float64)),
            dtype=np.float64,
        )
        jacobian[:, idx] = (shifted - base) / step_size
    return jnp.asarray(jacobian, dtype=jnp.float64)


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
    jacobian_fn: Optional[SEPJacobianFn] = None,
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
        jacobian_fn=jacobian_fn,
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
    jacobian_fn: Optional[SEPJacobianFn] = None,
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
        jacobian_fn=jacobian_fn,
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
    jacobian_fn: Optional[SEPJacobianFn],
) -> SEPSolution:
    _validate_sep_config(config, shock_dim=shock_dim)
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

    use_hmc = config.expectation_method == "hmc"
    jacobian_method = config.jacobian_method
    if jacobian_method == "auto":
        if use_hmc:
            jacobian_method_used = "finite_difference"
        elif jacobian_fn is not None:
            jacobian_method_used = "subgradient"
        else:
            jacobian_method_used = "autodiff"
    elif jacobian_method == "subgradient":
        if jacobian_fn is None:
            raise ValueError(
                "SEPConfig.jacobian_method='subgradient' requires a jacobian_fn."
            )
        jacobian_method_used = "subgradient"
    else:
        jacobian_method_used = jacobian_method
    rule = (
        GaussHermiteRule(
            nodes=jnp.zeros((1, shock_dim), dtype=jnp.float64),
            weights=jnp.ones((1,), dtype=jnp.float64),
        )
        if use_hmc
        else (
            _gauss_hermite_sparse_rule(config.nnodes, shock_dim, config.shock_scale)
            if config.sparse_tree
            else gauss_hermite_rule(config.nnodes, shock_dim, config.shock_scale)
        )
    )
    num_nodes = int(rule.weights.shape[0])
    counts = _group_counts(
        config.periods,
        0 if use_hmc else config.branching_order,
        num_nodes,
        sparse_tree=(config.sparse_tree and not use_hmc),
    )
    probabilities = _group_probabilities(
        rule,
        config.periods,
        0 if use_hmc else config.branching_order,
        sparse_tree=(config.sparse_tree and not use_hmc),
    )
    runtime_sparse_tree = config.sparse_tree and not use_hmc
    tree_metadata = _precompute_sep_tree_metadata(
        rule=rule,
        deterministic=deterministic,
        counts=counts,
        periods=config.periods,
        branching_order=config.branching_order,
        num_nodes=num_nodes,
        shock_dim=shock_dim,
        sparse_tree=runtime_sparse_tree,
        use_hmc=use_hmc,
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

    expected_stacked_size = time_offsets[-1]
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
        if guess.shape != (expected_stacked_size,):
            raise ValueError(
                "initial_guess must flatten to shape "
                f"({expected_stacked_size},), got {guess.shape}."
            )

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
            period_current_shocks = tree_metadata.current_shocks[t - 1]
            period_parent_indices = tree_metadata.parent_indices[t - 1]
            period_child_groups = tree_metadata.child_groups[t - 1]
            period_child_shocks = tree_metadata.child_shocks[t - 1]
            for g in range(counts[t]):
                if t == 1:
                    prev_state = prev_states[0]
                else:
                    assert period_parent_indices is not None
                    parent = int(period_parent_indices[g])
                    prev_state = prev_states[parent]

                deterministic_shock = deterministic[t - 1]
                current_shock = period_current_shocks[g]

                if conditional_residual_fn is None:
                    if use_hmc and t < config.periods and t <= config.branching_order and shock_dim > 0:
                        next_state = next_states[0]

                        def sample_residual(sampled_shock: jax.Array) -> jax.Array:
                            expected_term = expectation_fn(
                                next_state,
                                deterministic[t] + sampled_shock,
                                params,
                            )
                            return residual_fn(
                                prev_state,
                                current_states[g],
                                expected_term,
                                current_shock,
                                params,
                            )

                        sample_key = jax.random.PRNGKey(config.hmc_seed)
                        sample_key = jax.random.fold_in(sample_key, t)
                        sample_key = jax.random.fold_in(sample_key, g)
                        sample_key = jax.random.fold_in(sample_key, 17)
                        residuals.append(
                            _hmc_expectation_mean(
                                sample_residual,
                                shock_dim=shock_dim,
                                config=config,
                                key=sample_key,
                            )
                        )
                        continue

                    if t == config.periods:
                        expected_term = terminal_expectation
                    else:
                        assert period_child_groups is not None
                        assert period_child_shocks is not None
                        child_groups = period_child_groups[g]
                        child_shocks = period_child_shocks[g]
                        if len(child_groups) == 1:
                            expected_term = expectation_fn(
                                next_states[child_groups[0]],
                                child_shocks[0],
                                params,
                            )
                        else:
                            child_terms = []
                            for local_idx, child in enumerate(child_groups):
                                child_terms.append(
                                    rule.weights[local_idx]
                                    * expectation_fn(
                                        next_states[child],
                                        child_shocks[local_idx],
                                        params,
                                    )
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

                if use_hmc and t <= config.branching_order and shock_dim > 0:
                    next_state = terminal_state_arr if t == config.periods else next_states[0]

                    def sample_residual(sampled_shock: jax.Array) -> jax.Array:
                        return conditional_residual_fn(
                            prev_state,
                            current_states[g],
                            next_state,
                            deterministic_shock + sampled_shock,
                            params,
                        )

                    sample_key = jax.random.PRNGKey(config.hmc_seed)
                    sample_key = jax.random.fold_in(sample_key, t)
                    sample_key = jax.random.fold_in(sample_key, g)
                    residuals.append(
                        _hmc_expectation_mean(
                            sample_residual,
                            shock_dim=shock_dim,
                            config=config,
                            key=sample_key,
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

                assert period_child_groups is not None
                child_groups = period_child_groups[g]
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
    current_lambda = float(config.newton_regularization)
    active_solver = config.linear_solver
    best_err = np.inf
    stall_count = 0

    for iteration in range(1, config.max_iter + 1):
        residual = residual_vector(current)
        residual_norm = float(np.asarray(jnp.linalg.norm(residual, ord=jnp.inf)))
        if residual_norm < config.tol:
            converged = True
            iterations = iteration - 1
            break

        if jacobian_method_used == "finite_difference":
            jacobian = _finite_difference_jacobian(residual_vector, current)
        elif jacobian_method_used == "subgradient":
            jacobian = jnp.asarray(jacobian_fn(current), dtype=jnp.float64)
        else:
            jacobian = jax.jacobian(residual_vector)(current)

        accepted = False
        candidate = current
        err_after = residual_norm
        lambda_value = current_lambda
        while lambda_value <= config.lm_lambda_max:
            step = _solve_sep_newton_direction(
                jacobian,
                residual,
                lambda_value=lambda_value,
                solver=active_solver,
            )
            if not bool(jnp.all(jnp.isfinite(step))):
                lambda_value *= config.lm_lambda_scale
                continue

            if config.line_search:
                alpha = 1.0
                line_iterations = 0
                while True:
                    trial = current + alpha * step
                    trial_residual = residual_vector(trial)
                    trial_norm = float(np.asarray(jnp.linalg.norm(trial_residual, ord=jnp.inf)))
                    if np.isfinite(trial_norm) and trial_norm < residual_norm:
                        candidate = trial
                        err_after = trial_norm
                        if active_solver == "normal_equations":
                            current_lambda = max(
                                lambda_value / config.lm_lambda_scale,
                                config.lm_lambda_min,
                            )
                        accepted = True
                        break
                    line_iterations += 1
                    if (
                        line_iterations >= config.line_search_maxit
                        or alpha <= config.line_search_min_alpha
                    ):
                        break
                    alpha *= config.line_search_factor
            else:
                alpha = 1.0
                trial = current + alpha * step
                trial_residual = residual_vector(trial)
                trial_norm = float(np.asarray(jnp.linalg.norm(trial_residual, ord=jnp.inf)))
                if np.isfinite(trial_norm):
                    candidate = trial
                    err_after = trial_norm
                    if active_solver == "normal_equations":
                        if trial_norm < residual_norm:
                            current_lambda = max(
                                lambda_value / config.lm_lambda_scale,
                                config.lm_lambda_min,
                            )
                        else:
                            current_lambda = min(
                                lambda_value * config.lm_lambda_scale,
                                config.lm_lambda_max,
                            )
                    accepted = True
            if active_solver == "normal_equations" and accepted and config.line_search:
                break

            if accepted:
                break
            lambda_value *= config.lm_lambda_scale

        if not accepted:
            iterations = iteration - 1
            break

        current = candidate
        iterations = iteration
        if not np.isfinite(best_err):
            best_err = err_after
            stall_count = 0
        else:
            improvement_tol = max(
                config.stall_abs_tol,
                config.stall_rel_tol * best_err,
            )
            if err_after + improvement_tol < best_err:
                best_err = err_after
                stall_count = 0
            else:
                stall_count += 1

        if (
            active_solver == "normal_equations"
            and config.fallback_solver is not None
            and stall_count >= config.stall_iters
        ):
            active_solver = config.fallback_solver
            stall_count = 0

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
        jacobian_method=jacobian_method_used,
    )


def _solve_sep_newton_direction(
    jacobian: jax.Array,
    residual: jax.Array,
    *,
    lambda_value: float,
    solver: str,
) -> jax.Array:
    jacobian_arr = jnp.asarray(jacobian, dtype=jnp.float64)
    residual_arr = jnp.asarray(residual, dtype=jnp.float64).reshape(-1)
    ncols = int(jacobian_arr.shape[1])
    if solver == "normal_equations":
        normal_matrix = jacobian_arr.T @ jacobian_arr
        gradient = jacobian_arr.T @ residual_arr
        eye = jnp.eye(normal_matrix.shape[0], dtype=normal_matrix.dtype)
        return jnp.linalg.solve(normal_matrix + float(lambda_value) * eye, -gradient)
    if solver == "qr":
        eye = jnp.eye(ncols, dtype=jacobian_arr.dtype)
        sqrt_lambda = jnp.sqrt(jnp.asarray(lambda_value, dtype=jacobian_arr.dtype))
        augmented_matrix = jnp.concatenate([jacobian_arr, sqrt_lambda * eye], axis=0)
        augmented_rhs = jnp.concatenate(
            [-residual_arr, jnp.zeros((ncols,), dtype=residual_arr.dtype)],
            axis=0,
        )
        q, r = jnp.linalg.qr(augmented_matrix, mode="reduced")
        return jnp.linalg.solve(r, q.T @ augmented_rhs)
    raise ValueError(
        f"Unknown SEP linear_solver={solver!r}. Use 'normal_equations' or 'qr'."
    )
