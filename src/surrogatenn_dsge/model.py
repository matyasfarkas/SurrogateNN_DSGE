from __future__ import annotations

from time import perf_counter
from dataclasses import dataclass, replace as dataclass_replace
from functools import cached_property
import re
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
import scipy.optimize as scipy_optimize
import scipy.special as scipy_special
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.utilities.lambdify import implemented_function

from .dsge import (
    analyze_first_order_dsge_determinacy,
    DSGETimings,
    FirstOrderDeterminacyResult,
    FirstOrderDSGEResult,
    SecondOrderDSGEResult,
    SecondOrderStochasticSteadyStateResult,
    ThirdOrderDSGEResult,
    ThirdOrderStochasticSteadyStateResult,
    first_order_state_update,
    rollout_first_order_solution,
    linear_state_space_from_first_order_solution,
    solve_first_order_dsge_solution,
    solve_second_order_dsge_solution,
    solve_second_order_stochastic_steady_state,
    solve_third_order_dsge_solution,
    solve_third_order_stochastic_steady_state,
)
from .inversion import (
    first_order_inversion_loglikelihood,
    first_order_inversion_loglikelihood_per_period,
    get_sep_inversion_last_diagnostics,
    reset_sep_inversion_last_diagnostics,
    sep_inversion_loglikelihood,
    sep_inversion_loglikelihood_per_period,
)
from .sep import (
    _child_groups,
    _gauss_hermite_sparse_rule,
    _group_counts,
    _group_shock_at_time,
    _parent_group,
    SEPConfig,
    SEPSolution,
    gauss_hermite_rule,
    solve_stochastic_extended_path_residual_expectation,
)
from .statespace import (
    LinearGaussianStateSpace,
    kalman_loglikelihood as _statespace_kalman_loglikelihood,
    kalman_loglikelihood_per_period as _statespace_kalman_loglikelihood_per_period,
)
from .switching import (
    LinearGateStatsResult,
    SwitchingLikelihoodConfig,
    SwitchingLikelihoodResult,
    compute_gate_stats,
    compute_gate_stat_series,
    compute_switching_loglikelihood,
    evaluate_gate_budget_frontier,
    evaluate_gate_decisions,
    evaluate_gate_probabilities,
    evaluate_likelihood_surface_alignment,
    evaluate_switching_surface_alignment,
    evaluate_switching_vs_fom,
    summarize_loglik_decomposition,
    summarize_runtime,
)
from .linalg import solve_discrete_lyapunov_direct

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_NEWTON_GEOMETRIC_RESTART_SCALES = (0.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
_STEADY_STATE_HOMOTOPY_LEVELS = (0.0, 0.05, 0.125, 0.25, 0.5, 0.75, 1.0)

_STEADY_STATE_ALIASES = {"ss", "stst", "steady", "steadystate", "steady_state"}
_EXOGENOUS_ALIASES = {"x", "ex", "exo", "exogenous"}

_MODEL_BLOCK_RE = re.compile(
    r"@model\s+(?P<name>[^\s]+)(?P<options>.*?)\bbegin\b",
    re.DOTALL,
)
_PARAMETERS_BLOCK_RE = re.compile(
    r"@parameters\s+(?P<name>[^\s]+)(?P<options>.*?)\bbegin\b",
    re.DOTALL,
)
_IDENTIFIER_CHAR_CLASS = r"[\w\u02b0-\u02ff\u0300-\u036f\u1d2c-\u1dbf\u2070-\u209f]"
_IDENTIFIER_PATTERN = rf"(?!\d)(?:{_IDENTIFIER_CHAR_CLASS})+(?:\{{[^{{}}\[\]]+\}})*"
_INDEXED_IDENTIFIER_PATTERN = rf"(?!\d)(?:{_IDENTIFIER_CHAR_CLASS})+(?:\{{[^{{}}\[\]]+\}})+"
_REFERENCE_RE = re.compile(
    rf"(?P<name>{_IDENTIFIER_PATTERN})\s*\[\s*(?P<index>[^\]]+)\s*\]",
    re.UNICODE,
)
_IDENTIFIER_RE = re.compile(_IDENTIFIER_PATTERN, re.UNICODE)
_INDEXED_IDENTIFIER_RE = re.compile(_INDEXED_IDENTIFIER_PATTERN, re.UNICODE)

LoopIndex = Union[int, str]
LoopCollections = Mapping[str, tuple[LoopIndex, ...]]


class SteadyStateResult(NamedTuple):
    steady_state: jax.Array
    base_steady_state: jax.Array
    parameter_values: jax.Array
    converged: bool
    iterations: int
    residual_norm: float


class ParsedModelFirstOrderResult(NamedTuple):
    steady_state: jax.Array
    parameter_values: jax.Array
    jacobian: jax.Array
    solution: FirstOrderDSGEResult


class ParsedModelFirstOrderDeterminacyResult(NamedTuple):
    steady_state: jax.Array
    parameter_values: jax.Array
    jacobian: jax.Array
    determinacy: FirstOrderDeterminacyResult


class ParsedModelSecondOrderResult(NamedTuple):
    steady_state: jax.Array
    parameter_values: jax.Array
    jacobian: jax.Array
    hessian: jax.Array
    first_order_solution: FirstOrderDSGEResult
    second_order_solution: SecondOrderDSGEResult
    stochastic_steady_state: SecondOrderStochasticSteadyStateResult


class ParsedModelThirdOrderResult(NamedTuple):
    steady_state: jax.Array
    parameter_values: jax.Array
    jacobian: jax.Array
    hessian: jax.Array
    third_order_derivatives: jax.Array
    first_order_solution: FirstOrderDSGEResult
    second_order_solution: SecondOrderDSGEResult
    third_order_solution: ThirdOrderDSGEResult
    stochastic_steady_state: ThirdOrderStochasticSteadyStateResult


class ParsedModelSEPResult(NamedTuple):
    steady_state: jax.Array
    parameter_values: jax.Array
    solution: SEPSolution


class _SEPPathSimulationResult(NamedTuple):
    state_path: np.ndarray
    shocks: np.ndarray
    sep_result: ParsedModelSEPResult


class HomotopySEPResult(NamedTuple):
    success: bool
    result: ParsedModelSEPResult
    sigma_path: tuple[float, ...]


class HomotopyChainedTrajectoryResult(NamedTuple):
    trajectory: jax.Array
    shocks: jax.Array
    success: bool
    periods_completed: int
    sigma_paths: tuple[tuple[float, ...], ...]
    steady_state: jax.Array
    parameter_values: jax.Array


class OBCViolationPathResult(NamedTuple):
    state_path: jax.Array
    shocks: jax.Array
    violations: jax.Array


class ModelSimulationResult(NamedTuple):
    variables: tuple[str, ...]
    data: jax.Array
    state_path: jax.Array
    shocks: jax.Array
    algorithm_used: str
    steady_state: jax.Array
    parameter_values: jax.Array


class ModelIRFResult(NamedTuple):
    variables: tuple[str, ...]
    shock_names: tuple[str, ...]
    responses: jax.Array
    state_paths: jax.Array
    shocks: jax.Array
    algorithm_used: str
    steady_state: jax.Array
    parameter_values: jax.Array


class ParsedParameterBlock(NamedTuple):
    target_names: tuple[str, ...]
    calibrated_target_names: tuple[str, ...]
    equation_texts: tuple[str, ...]
    initial_values: dict[str, float]
    bounds: dict[str, tuple[float, float]]


class _JaxNewtonState(NamedTuple):
    x: jax.Array
    residual: jax.Array
    residual_norm: jax.Array
    converged: jax.Array
    done: jax.Array
    iterations: jax.Array


class _FirstOrderOBCProjectionSpec(NamedTuple):
    variable_name: str
    variable_index: int
    current_symbol: sp.Symbol
    operator: str
    left_expr: sp.Expr
    right_expr: sp.Expr
    target_symbol: sp.Symbol
    inverse_expr: sp.Expr
    mode: str = "branch_target"


class _FirstOrderOBCSimulationResult(NamedTuple):
    state_path: np.ndarray
    shocks: np.ndarray


class _FirstOrderOBCWindowOptimizationResult(NamedTuple):
    shock_window: np.ndarray
    state_window: np.ndarray


@dataclass(frozen=True)
class MacroModel:
    name: str
    equations: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: jax.Array
    calibrated_parameter_names: tuple[str, ...]
    default_initial_guess: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    model_options: dict[str, object]
    parameter_options: dict[str, object]
    max_obc_horizon: int
    has_obc: bool
    timings: DSGETimings
    steady_state_names: tuple[str, ...]
    steady_state_reference_names: tuple[str, ...]
    dynamic_symbol_names: tuple[str, ...]
    _dynamic_expressions: tuple[sp.Expr, ...]
    _steady_state_expressions: tuple[sp.Expr, ...]
    _parameter_expressions: tuple[sp.Expr, ...]
    _parameter_symbols: tuple[sp.Symbol, ...]
    _steady_state_symbols: tuple[sp.Symbol, ...]
    _dynamic_symbols: tuple[sp.Symbol, ...]
    _dynamic_input_symbols: tuple[sp.Symbol, ...]
    _parameter_equations_depend_on_steady_state: bool

    @cached_property
    def _steady_state_matrix(self) -> sp.Matrix:
        return sp.Matrix(self._steady_state_expressions)

    @cached_property
    def _steady_state_jacobian(self) -> sp.Matrix:
        return self._steady_state_matrix.jacobian(self._steady_state_symbols)

    @cached_property
    def _steady_state_parameter_jacobian(self) -> sp.Matrix:
        return self._steady_state_matrix.jacobian(self._parameter_symbols)

    @cached_property
    def _parameter_matrix(self) -> sp.Matrix:
        return sp.Matrix(self._parameter_expressions)

    @cached_property
    def _parameter_equation_jacobian(self) -> sp.Matrix:
        return self._parameter_matrix.jacobian(self._parameter_symbols)

    @cached_property
    def _dynamic_matrix(self) -> sp.Matrix:
        return sp.Matrix(self._dynamic_expressions)

    @cached_property
    def _dynamic_jacobian(self) -> sp.Matrix:
        return self._dynamic_matrix.jacobian(self._dynamic_symbols)

    @cached_property
    def _dynamic_hessian(self) -> sp.Matrix:
        return sp.Matrix(
            [_flatten_hessian(expr, self._dynamic_symbols) for expr in self._dynamic_expressions]
        )

    @cached_property
    def _dynamic_third(self) -> sp.Matrix:
        return sp.Matrix(
            [
                _flatten_third_order(expr, self._dynamic_symbols)
                for expr in self._dynamic_expressions
            ]
        )

    @cached_property
    def _joint_unknown_symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(list(self._steady_state_symbols) + list(self._parameter_symbols))

    @cached_property
    def _joint_steady_state_matrix(self) -> sp.Matrix:
        return sp.Matrix(list(self._steady_state_expressions) + list(self._parameter_expressions))

    @cached_property
    def _joint_steady_state_jacobian(self) -> sp.Matrix:
        return self._joint_steady_state_matrix.jacobian(self._joint_unknown_symbols)

    @cached_property
    def _steady_state_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._steady_state_matrix,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _steady_state_jacobian_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._steady_state_jacobian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _steady_state_parameter_jacobian_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._steady_state_parameter_jacobian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _parameter_equation_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._parameter_matrix,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _parameter_equation_jacobian_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._parameter_equation_jacobian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _joint_steady_state_fn(self) -> object:
        return sp.lambdify(
            self._joint_unknown_symbols,
            self._joint_steady_state_matrix,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _joint_steady_state_jacobian_fn(self) -> object:
        return sp.lambdify(
            self._joint_unknown_symbols,
            self._joint_steady_state_jacobian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _steady_state_residual_jax_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._steady_state_matrix,
            modules=_jax_lambdify_modules(),
        )

    @cached_property
    def _parameter_equation_residual_jax_fn(self) -> object:
        return sp.lambdify(
            list(self._steady_state_symbols) + list(self._parameter_symbols),
            self._parameter_matrix,
            modules=_jax_lambdify_modules(),
        )

    @cached_property
    def _joint_steady_state_residual_jax_fn(self) -> object:
        return sp.lambdify(
            self._joint_unknown_symbols,
            self._joint_steady_state_matrix,
            modules=_jax_lambdify_modules(),
        )

    @cached_property
    def _dynamic_residual_fn(self) -> object:
        return sp.lambdify(
            self._dynamic_input_symbols,
            self._dynamic_matrix,
            modules=_jax_lambdify_modules(),
        )

    @cached_property
    def _steady_state_solution_cache(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return []

    @cached_property
    def _symbolic_steady_state_seed_entries(self) -> tuple[tuple[int, sp.Expr], ...]:
        if not bool(self.parameter_options.get("symbolic", False)):
            return ()
        return _build_symbolic_steady_state_seed_entries(
            self._steady_state_expressions,
            self._steady_state_symbols,
            self._parameter_symbols,
        )

    @cached_property
    def _symbolic_steady_state_seed_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self._symbolic_steady_state_seed_entries)

    @cached_property
    def _symbolic_steady_state_seed_matrix(self) -> sp.Matrix:
        expressions = [expr for _, expr in self._symbolic_steady_state_seed_entries]
        if not expressions:
            return sp.Matrix.zeros(0, 1)
        return sp.Matrix(expressions)

    @cached_property
    def _symbolic_steady_state_seed_fn(self) -> object:
        return sp.lambdify(
            self._parameter_symbols,
            self._symbolic_steady_state_seed_matrix,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _symbolic_steady_state_seed_jax_fn(self) -> object:
        return sp.lambdify(
            self._parameter_symbols,
            self._symbolic_steady_state_seed_matrix,
            modules=_jax_lambdify_modules(),
        )

    @cached_property
    def _steady_state_expansion_indices(self) -> tuple[int, ...]:
        base_lookup = {name: idx for idx, name in enumerate(self.steady_state_names)}
        indices: list[int] = []
        for name in self.timings.var:
            if name in base_lookup:
                indices.append(base_lookup[name])
                continue
            if name in self.timings.exo_present:
                indices.append(0)
                continue
            stripped = _strip_auxiliary_suffix(name)
            if stripped in base_lookup:
                indices.append(base_lookup[stripped])
                continue
            if stripped in self.timings.exo:
                indices.append(0)
                continue
            raise ValueError(f"Could not expand steady state value for `{name}`.")
        return tuple(indices)

    @cached_property
    def _steady_state_expansion_zero_mask(self) -> tuple[bool, ...]:
        zero_mask: list[bool] = []
        for name in self.timings.var:
            if name in self.timings.exo_present:
                zero_mask.append(True)
                continue
            stripped = _strip_auxiliary_suffix(name)
            zero_mask.append(stripped in self.timings.exo)
        return tuple(zero_mask)

    @cached_property
    def _steady_reference_indices(self) -> tuple[int, ...]:
        index_lookup = {name: idx for idx, name in enumerate(self.timings.var)}
        indices: list[int] = []
        for name in self.steady_state_reference_names:
            if name in index_lookup:
                indices.append(index_lookup[name])
                continue
            stripped = _strip_auxiliary_suffix(name)
            if stripped in index_lookup:
                indices.append(index_lookup[stripped])
                continue
            raise ValueError(f"Missing steady-state reference value for `{name}`.")
        return tuple(indices)

    @cached_property
    def _base_steady_state_indices(self) -> tuple[int, ...]:
        index_lookup = {name: idx for idx, name in enumerate(self.timings.var)}
        return tuple(index_lookup[name] for name in self.steady_state_names)

    @cached_property
    def _dynamic_jacobian_fn(self) -> object:
        return sp.lambdify(
            self._dynamic_input_symbols,
            self._dynamic_jacobian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _dynamic_hessian_fn(self) -> object:
        return sp.lambdify(
            self._dynamic_input_symbols,
            self._dynamic_hessian,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _dynamic_third_order_fn(self) -> object:
        return sp.lambdify(
            self._dynamic_input_symbols,
            self._dynamic_third,
            modules=_numpy_lambdify_modules(),
        )

    @cached_property
    def _obc_violation_expressions(self) -> tuple[sp.Expr, ...]:
        if not self.has_obc:
            return ()
        expressions: list[sp.Expr] = []
        for expr in self._dynamic_expressions:
            expressions.extend(_transform_obc_residual_to_violation_expressions(expr))
        return tuple(expressions)

    @cached_property
    def _obc_violation_matrix(self) -> sp.Matrix:
        return sp.Matrix(self._obc_violation_expressions)

    @cached_property
    def _first_order_obc_projection_specs(
        self,
    ) -> Optional[tuple[_FirstOrderOBCProjectionSpec, ...]]:
        if not self.has_obc:
            return ()
        current_symbols = tuple(
            self._dynamic_input_symbols[
                self.timings.nFuture_not_past_and_mixed : (
                    self.timings.nFuture_not_past_and_mixed + self.timings.nVars
                )
            ]
        )
        current_symbol_to_index = {
            symbol: idx for idx, symbol in enumerate(current_symbols)
        }
        current_symbol_set = frozenset(current_symbols)
        future_symbols = frozenset(
            self._dynamic_input_symbols[: self.timings.nFuture_not_past_and_mixed]
        )
        specs: list[_FirstOrderOBCProjectionSpec] = []

        def _build_zero_binding_spec(
            branch_expr: sp.Expr,
            *,
            operator: str,
            left_expr: sp.Expr,
            right_expr: sp.Expr,
        ) -> Optional[_FirstOrderOBCProjectionSpec]:
            branch_symbols = branch_expr.free_symbols
            if branch_symbols & future_symbols:
                return None
            branch_current_symbols = tuple(
                symbol for symbol in current_symbols if symbol in branch_symbols
            )
            if not branch_current_symbols:
                return None

            candidate_specs: list[
                tuple[sp.Symbol, sp.Expr, int]
            ] = []
            for candidate_symbol in branch_current_symbols:
                target_symbol = sp.Symbol("__obc_projection_target__", real=True)
                try:
                    inverse_solutions = sp.solve(
                        sp.Eq(branch_expr, target_symbol),
                        candidate_symbol,
                    )
                except Exception:
                    continue
                if len(inverse_solutions) != 1:
                    continue
                inverse_expr = sp.simplify(inverse_solutions[0])
                inverse_symbols = inverse_expr.free_symbols - {target_symbol}
                if candidate_symbol in inverse_symbols:
                    continue
                if inverse_symbols & future_symbols:
                    continue
                derivative = sp.simplify(sp.diff(branch_expr, candidate_symbol))
                derivative_sign = 0
                if not derivative.free_symbols:
                    try:
                        derivative_value = float(derivative)
                    except (TypeError, ValueError):
                        derivative_sign = 0
                    else:
                        if derivative_value < 0.0:
                            derivative_sign = -1
                        elif derivative_value > 0.0:
                            derivative_sign = 1
                candidate_specs.append(
                    (candidate_symbol, inverse_expr, derivative_sign)
                )

            if len(candidate_specs) == 1:
                chosen_symbol, inverse_expr, _ = candidate_specs[0]
            else:
                negative_candidates = [
                    candidate for candidate in candidate_specs if candidate[2] < 0
                ]
                if len(negative_candidates) != 1:
                    return None
                chosen_symbol, inverse_expr, _ = negative_candidates[0]

            variable_index = current_symbol_to_index[chosen_symbol]
            return _FirstOrderOBCProjectionSpec(
                variable_name=self.timings.var[variable_index],
                variable_index=variable_index,
                current_symbol=chosen_symbol,
                operator=operator,
                left_expr=left_expr,
                right_expr=right_expr,
                target_symbol=sp.Symbol("__obc_projection_target__", real=True),
                inverse_expr=inverse_expr,
                mode="zero_binding",
            )

        for expr in self._dynamic_expressions:
            obc_calls = _obc_calls_in_expression(expr)
            if not obc_calls:
                continue
            if len(obc_calls) != 1:
                return None
            obc_call = obc_calls[0]
            if len(obc_call.args) != 2:
                return None
            placeholder = sp.Symbol("__obc_projection_placeholder__", real=True)
            try:
                solutions = sp.solve(
                    sp.Eq(expr.xreplace({obc_call: placeholder}), 0),
                    placeholder,
                )
            except Exception:
                return None
            if len(solutions) != 1:
                return None
            solved_expr = sp.simplify(solutions[0])
            left_expr = sp.simplify(obc_call.args[0])
            right_expr = sp.simplify(obc_call.args[1])
            solved_symbols = solved_expr.free_symbols
            current_symbols_in_solution = tuple(
                symbol for symbol in current_symbols if symbol in solved_symbols
            )
            operator = "max" if obc_call.func is sp.Max else "min"
            if len(current_symbols_in_solution) == 1:
                current_symbol = current_symbols_in_solution[0]
                if solved_symbols & future_symbols:
                    return None
                if solved_symbols & (current_symbol_set - {current_symbol}):
                    return None
                target_symbol = sp.Symbol("__obc_projection_target__", real=True)
                try:
                    inverse_solutions = sp.solve(
                        sp.Eq(solved_expr, target_symbol),
                        current_symbol,
                    )
                except Exception:
                    return None
                if len(inverse_solutions) != 1:
                    return None
                inverse_expr = sp.simplify(inverse_solutions[0])
                inverse_symbols = inverse_expr.free_symbols - {target_symbol}
                if current_symbol in inverse_symbols:
                    return None
                if inverse_symbols & future_symbols:
                    return None
                if inverse_symbols & (current_symbol_set - {current_symbol}):
                    return None
                branch_symbols = left_expr.free_symbols | right_expr.free_symbols
                if current_symbol in branch_symbols:
                    return None
                if branch_symbols & future_symbols:
                    return None
                variable_index = current_symbol_to_index[current_symbol]
                specs.append(
                    _FirstOrderOBCProjectionSpec(
                        variable_name=self.timings.var[variable_index],
                        variable_index=variable_index,
                        current_symbol=current_symbol,
                        operator=operator,
                        left_expr=left_expr,
                        right_expr=right_expr,
                        target_symbol=target_symbol,
                        inverse_expr=inverse_expr,
                    )
                )
                continue

            if solved_expr != 0:
                return None
            zero_specs = [
                _build_zero_binding_spec(
                    left_expr,
                    operator=operator,
                    left_expr=left_expr,
                    right_expr=right_expr,
                ),
                _build_zero_binding_spec(
                    right_expr,
                    operator=operator,
                    left_expr=left_expr,
                    right_expr=right_expr,
                ),
            ]
            if any(spec is None for spec in zero_specs):
                return None
            specs.extend(spec for spec in zero_specs if spec is not None)

        if not specs:
            return None
        variable_indices = [spec.variable_index for spec in specs]
        if len(set(variable_indices)) != len(variable_indices):
            return None
        return tuple(specs)

    def _coerce_parameter_values(
        self,
        parameter_values: Optional[Sequence[float]],
    ) -> np.ndarray:
        if parameter_values is None:
            values = np.asarray(self.parameter_values, dtype=np.float64)
        else:
            values = np.asarray(parameter_values, dtype=np.float64)
        if values.shape != (len(self.parameter_names),):
            raise ValueError(
                "parameter_values must have shape "
                f"({len(self.parameter_names)},), got {values.shape}."
            )
        return values

    def _coerce_base_parameter_values(
        self,
        base_parameter_values: Optional[Sequence[float] | Mapping[str, float]],
    ) -> np.ndarray:
        if base_parameter_values is None:
            return np.asarray(self.parameter_values, dtype=np.float64)
        if isinstance(base_parameter_values, Mapping):
            unknown = tuple(
                sorted(set(base_parameter_values).difference(self.parameter_names))
            )
            if unknown:
                raise ValueError(
                    "Unknown parameter names in `base_parameter_values`: "
                    + ", ".join(unknown)
                    + "."
                )
            base = np.asarray(self.parameter_values, dtype=np.float64).copy()
            index_lookup = {name: idx for idx, name in enumerate(self.parameter_names)}
            for name, value in base_parameter_values.items():
                base[index_lookup[name]] = float(value)
            return base
        return self._coerce_parameter_values(base_parameter_values)

    def _coerce_parameter_draw_matrix(
        self,
        parameter_draws: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    ) -> np.ndarray:
        if isinstance(parameter_draws, Mapping):
            unknown = tuple(sorted(set(parameter_draws).difference(self.parameter_names)))
            if unknown:
                raise ValueError(
                    "Unknown parameter names in `parameter_draws`: "
                    + ", ".join(unknown)
                    + "."
                )
            if not parameter_draws:
                raise ValueError("parameter_draws mapping must not be empty.")
            lengths = {
                np.asarray(values, dtype=np.float64).reshape(-1).shape[0]
                for values in parameter_draws.values()
            }
            if len(lengths) != 1:
                raise ValueError("All parameter draw series must share the same length.")
            n_draws = int(next(iter(lengths)))
            base = self._coerce_base_parameter_values(base_parameter_values)
            draw_matrix = np.repeat(base[None, :], n_draws, axis=0)
            index_lookup = {name: idx for idx, name in enumerate(self.parameter_names)}
            for name, values in parameter_draws.items():
                draw_matrix[:, index_lookup[name]] = np.asarray(
                    values,
                    dtype=np.float64,
                ).reshape(-1)
            return draw_matrix

        draw_matrix = np.asarray(parameter_draws, dtype=np.float64)
        if draw_matrix.ndim == 1:
            draw_matrix = draw_matrix[None, :]
        expected_shape = (len(self.parameter_names),)
        if draw_matrix.ndim != 2 or draw_matrix.shape[1] != expected_shape[0]:
            raise ValueError(
                "parameter_draws must have shape (n_draws, "
                f"{expected_shape[0]}), got {draw_matrix.shape}."
            )
        return draw_matrix

    def _coerce_parameter_values_jax(
        self,
        parameter_values: Optional[Sequence[float]],
    ) -> jax.Array:
        if parameter_values is None:
            values = jnp.asarray(self.parameter_values, dtype=jnp.float64)
        else:
            values = jnp.asarray(parameter_values, dtype=jnp.float64)
        expected_shape = (len(self.parameter_names),)
        if values.shape != expected_shape:
            raise ValueError(
                "parameter_values must have shape "
                f"{expected_shape}, got {values.shape}."
            )
        return values

    def _coerce_steady_state_guess(
        self,
        initial_guess: Optional[Sequence[float] | Mapping[str, float]],
    ) -> np.ndarray:
        n = len(self.steady_state_names)

        def default_value(name: str) -> float:
            compact = name.replace("_", "").lower()
            if "log" in compact:
                return 0.0
            if name == "efficiency":
                return 0.0
            if compact in {"l", "n"} or "labour" in compact or "labor" in compact or "lab" in compact:
                return 0.3
            if compact in {"k", "capital"}:
                return 10.0
            if compact in {"c", "consumption", "y", "output", "i", "investment"}:
                return 1.2
            return 1.0

        default_guess = np.asarray(
            [default_value(name) for name in self.steady_state_names],
            dtype=np.float64,
        )
        if initial_guess is None:
            if not self.default_initial_guess:
                return default_guess
            guess = default_guess.copy()
            for idx, name in enumerate(self.steady_state_names):
                if name in self.default_initial_guess:
                    guess[idx] = float(self.default_initial_guess[name])
            return guess
        if isinstance(initial_guess, Mapping):
            guess = default_guess.copy()
            for idx, name in enumerate(self.steady_state_names):
                if name in initial_guess:
                    guess[idx] = float(initial_guess[name])
            return guess
        guess = np.asarray(initial_guess, dtype=np.float64)
        if guess.shape != (n,):
            raise ValueError(
                "initial_guess must have shape "
                f"({n},), got {guess.shape}."
            )
        return guess

    def _cached_steady_state_guess_for_parameters(
        self,
        parameter_values: Sequence[float],
    ) -> Optional[np.ndarray]:
        cache = self._steady_state_solution_cache
        if not cache:
            return None
        target = np.asarray(parameter_values, dtype=np.float64)
        best_guess: Optional[np.ndarray] = None
        best_distance: Optional[float] = None
        for cached_parameters, cached_guess in cache:
            if cached_parameters.shape != target.shape or cached_guess.shape != (
                len(self.steady_state_names),
            ):
                continue
            if not np.isfinite(cached_parameters).all() or not np.isfinite(cached_guess).all():
                continue
            distance = float(np.sum((cached_parameters - target) ** 2))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_guess = np.asarray(cached_guess, dtype=np.float64).copy()
        return best_guess

    def _remember_steady_state_solution(
        self,
        parameter_values: Sequence[float],
        base_steady_state: Sequence[float],
    ) -> None:
        cached_parameters = np.asarray(parameter_values, dtype=np.float64)
        cached_guess = np.asarray(base_steady_state, dtype=np.float64)
        if cached_parameters.shape != (len(self.parameter_names),):
            return
        if cached_guess.shape != (len(self.steady_state_names),):
            return
        if not np.isfinite(cached_parameters).all() or not np.isfinite(cached_guess).all():
            return
        cache = self._steady_state_solution_cache
        cache.append((cached_parameters.copy(), cached_guess.copy()))
        if len(cache) > 32:
            del cache[: len(cache) - 32]

    def _apply_symbolic_steady_state_seed(
        self,
        guess: np.ndarray,
        parameter_values: Sequence[float],
    ) -> np.ndarray:
        if not self._symbolic_steady_state_seed_indices:
            return guess
        values = np.asarray(
            self._symbolic_steady_state_seed_fn(*np.asarray(parameter_values, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        updated = np.asarray(guess, dtype=np.float64).copy()
        for idx, value in zip(self._symbolic_steady_state_seed_indices, values):
            if np.isfinite(value):
                updated[idx] = float(value)
        return updated

    def _apply_symbolic_steady_state_seed_jax(
        self,
        guess: jax.Array,
        parameter_values: Sequence[float],
    ) -> jax.Array:
        if not self._symbolic_steady_state_seed_indices:
            return guess
        indices = jnp.asarray(self._symbolic_steady_state_seed_indices, dtype=jnp.int32)
        values = jnp.asarray(
            self._symbolic_steady_state_seed_jax_fn(*parameter_values),
            dtype=jnp.float64,
        ).reshape(-1)
        seeded = guess[indices]
        return guess.at[indices].set(jnp.where(jnp.isfinite(values), values, seeded))

    def _expand_to_full_steady_state(
        self,
        base_steady_state: Sequence[float],
    ) -> np.ndarray:
        base = np.asarray(base_steady_state, dtype=np.float64)
        if base.shape != (len(self.steady_state_names),):
            raise ValueError(
                "base_steady_state must have shape "
                f"({len(self.steady_state_names)},), got {base.shape}."
            )
        values = dict(zip(self.steady_state_names, base))
        full = np.zeros(self.timings.nVars, dtype=np.float64)
        for idx, name in enumerate(self.timings.var):
            if name in values:
                full[idx] = values[name]
                continue
            if name in self.timings.exo_present:
                full[idx] = 0.0
                continue
            stripped = _strip_auxiliary_suffix(name)
            if stripped in values:
                full[idx] = values[stripped]
                continue
            if stripped in self.timings.exo:
                full[idx] = 0.0
                continue
            raise ValueError(f"Could not expand steady state value for `{name}`.")
        return full

    def _expand_to_full_steady_state_jax(
        self,
        base_steady_state: Sequence[float],
    ) -> jax.Array:
        base = jnp.asarray(base_steady_state, dtype=jnp.float64)
        expected_shape = (len(self.steady_state_names),)
        if base.shape != expected_shape:
            raise ValueError(
                "base_steady_state must have shape "
                f"{expected_shape}, got {base.shape}."
            )
        source = (
            base
            if len(self.steady_state_names) > 0
            else jnp.zeros((1,), dtype=jnp.float64)
        )
        indices = jnp.asarray(self._steady_state_expansion_indices, dtype=jnp.int32)
        zero_mask = jnp.asarray(self._steady_state_expansion_zero_mask)
        expanded = source[indices]
        return jnp.where(zero_mask, jnp.zeros_like(expanded), expanded)

    def _extract_base_steady_state(
        self,
        full_steady_state: Sequence[float],
    ) -> np.ndarray:
        full = np.asarray(full_steady_state, dtype=np.float64)
        if full.shape != (self.timings.nVars,):
            raise ValueError(
                "full_steady_state must have shape "
                f"({self.timings.nVars},), got {full.shape}."
            )
        lookup = dict(zip(self.timings.var, full))
        return np.asarray(
            [lookup[name] for name in self.steady_state_names],
            dtype=np.float64,
        )

    def _extract_base_steady_state_jax(
        self,
        full_steady_state: Sequence[float],
    ) -> jax.Array:
        full = jnp.asarray(full_steady_state, dtype=jnp.float64)
        expected_shape = (self.timings.nVars,)
        if full.shape != expected_shape:
            raise ValueError(
                "full_steady_state must have shape "
                f"{expected_shape}, got {full.shape}."
            )
        if not self.steady_state_names:
            return jnp.zeros((0,), dtype=full.dtype)
        indices = jnp.asarray(self._base_steady_state_indices, dtype=jnp.int32)
        return full[indices]

    def _coerce_full_steady_state(
        self,
        steady_state: Optional[Sequence[float]],
        *,
        parameter_values: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if steady_state is None:
            return np.asarray(
                self.solve_steady_state(parameter_values=parameter_values).steady_state,
                dtype=np.float64,
            )
        state = np.asarray(steady_state, dtype=np.float64)
        if state.shape == (len(self.steady_state_names),):
            return self._expand_to_full_steady_state(state)
        if state.shape == (self.timings.nVars,):
            return state
        raise ValueError(
            "steady_state must have shape "
            f"({len(self.steady_state_names)},) or ({self.timings.nVars},), got {state.shape}."
        )

    def _steady_reference_values(self, full_steady_state: np.ndarray) -> np.ndarray:
        lookup = dict(zip(self.timings.var, full_steady_state))
        refs = np.zeros(len(self.steady_state_reference_names), dtype=np.float64)
        for idx, name in enumerate(self.steady_state_reference_names):
            if name in lookup:
                refs[idx] = lookup[name]
                continue
            stripped = _strip_auxiliary_suffix(name)
            if stripped in lookup:
                refs[idx] = lookup[stripped]
                continue
            raise ValueError(f"Missing steady-state reference value for `{name}`.")
        return refs

    @cached_property
    def _obc_shock_indices(self) -> np.ndarray:
        return np.asarray(
            [idx for idx, name in enumerate(self.timings.exo) if "ᵒᵇᶜ" in str(name)],
            dtype=np.int64,
        )

    @cached_property
    def _dynamic_exogenous_symbols(self) -> tuple[sp.Symbol, ...]:
        start = (
            self.timings.nFuture_not_past_and_mixed
            + self.timings.nVars
            + self.timings.nPast_not_future_and_mixed
        )
        end = start + self.timings.nExo
        return tuple(self._dynamic_input_symbols[start:end])

    def _obc_shocks_included(
        self,
        shocks: np.ndarray,
        *,
        tol: float = 1e-10,
    ) -> bool:
        if not self.has_obc or self._obc_shock_indices.size == 0:
            return False
        shock_values = np.asarray(shocks, dtype=np.float64)
        if shock_values.ndim not in {2, 3}:
            raise ValueError(
                "shocks must be rank-2 or rank-3 for OBC shock detection, "
                f"got shape {shock_values.shape}."
            )
        relevant = shock_values[self._obc_shock_indices]
        return bool(relevant.size and np.max(np.abs(relevant)) > tol)

    def _effective_ignore_obc_flag(
        self,
        shocks: np.ndarray,
        *,
        ignore_obc: bool,
    ) -> bool:
        if not self.has_obc or not ignore_obc:
            return False
        if self._obc_shocks_included(shocks):
            return False
        return True

    def _steady_reference_values_jax(
        self,
        full_steady_state: Sequence[float],
    ) -> jax.Array:
        state = jnp.asarray(full_steady_state, dtype=jnp.float64)
        expected_shape = (self.timings.nVars,)
        if state.shape != expected_shape:
            raise ValueError(
                "full_steady_state must have shape "
                f"{expected_shape}, got {state.shape}."
            )
        if not self.steady_state_reference_names:
            return jnp.zeros((0,), dtype=state.dtype)
        indices = jnp.asarray(self._steady_reference_indices, dtype=jnp.int32)
        return state[indices]

    def _apply_default_calibrated_parameter_guess_jax(
        self,
        parameter_values: jax.Array,
        *,
        parameter_values_provided: bool,
    ) -> jax.Array:
        if parameter_values_provided or not self.default_initial_guess:
            return parameter_values
        calibrated = set(self.calibrated_parameter_names)
        updated = jnp.asarray(parameter_values, dtype=jnp.float64)
        for idx, name in enumerate(self.parameter_names):
            if name in calibrated and name in self.default_initial_guess:
                updated = updated.at[idx].set(
                    jnp.asarray(self.default_initial_guess[name], dtype=updated.dtype)
                )
        return updated

    def _coerce_dynamic_state_vector(
        self,
        state: Sequence[float],
        *,
        label: str,
    ) -> jax.Array:
        values = np.asarray(state, dtype=np.float64)
        if values.shape == (self.timings.nVars,):
            return jnp.asarray(values, dtype=jnp.float64)
        if values.shape == (len(self.steady_state_names),) and self.timings.nVars == len(
            self.steady_state_names
        ):
            return jnp.asarray(values, dtype=jnp.float64)
        raise ValueError(
            f"{label} must have shape ({self.timings.nVars},), got {values.shape}."
        )

    def _coerce_sep_deterministic_shocks(
        self,
        deterministic_shocks: Optional[
            Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ],
        *,
        periods: int,
    ) -> Optional[jax.Array]:
        if deterministic_shocks is None:
            return None
        if isinstance(deterministic_shocks, Mapping):
            unexpected = sorted(
                set(deterministic_shocks).difference(self.timings.exo)
            )
            if unexpected:
                raise ValueError(
                    "Unknown deterministic shock names: "
                    + ", ".join(unexpected)
                )
            values = np.zeros((periods, self.timings.nExo), dtype=np.float64)
            for idx, name in enumerate(self.timings.exo):
                if name not in deterministic_shocks:
                    continue
                series = np.asarray(deterministic_shocks[name], dtype=np.float64)
                if series.shape != (periods,):
                    raise ValueError(
                        f"Deterministic shock `{name}` must have shape ({periods},), got {series.shape}."
                    )
                values[:, idx] = series
            return jnp.asarray(values, dtype=jnp.float64)
        values = np.asarray(deterministic_shocks, dtype=np.float64)
        if values.shape == (self.timings.nExo, periods):
            return jnp.asarray(values.T, dtype=jnp.float64)
        if values.shape != (periods, self.timings.nExo):
            raise ValueError(
                "deterministic_shocks must have shape "
                f"({periods}, {self.timings.nExo}) or ({self.timings.nExo}, {periods}), got {values.shape}."
            )
        return jnp.asarray(values, dtype=jnp.float64)

    def _resolve_variable_selection(
        self,
        variables: Optional[Sequence[str] | str],
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if variables is None:
            selected = tuple(self.timings.var)
        elif isinstance(variables, str):
            token = _strip_selector_prefix(variables)
            if token == "all":
                selected = tuple(self.timings.var)
            elif token == "all_excluding_obc":
                selected = tuple(
                    name for name in self.timings.var if "ᵒᵇᶜ" not in str(name)
                )
            elif token == "all_excluding_auxiliary_and_obc":
                selected = tuple(
                    name
                    for name in self.timings.var
                    if "ᵒᵇᶜ" not in str(name)
                    and name not in self.timings.aux
                    and name not in self.timings.exo_present
                )
            else:
                selected = (token,)
        else:
            selected = _flatten_named_selection(variables)
        if not selected:
            raise ValueError("variables must contain at least one variable name.")
        lookup = {name: idx for idx, name in enumerate(self.timings.var)}
        unexpected = sorted(set(selected).difference(lookup))
        if unexpected:
            raise ValueError("Unknown variable names: " + ", ".join(unexpected))
        indices = np.asarray([lookup[name] for name in selected], dtype=np.int64)
        return selected, indices

    def _coerce_simulation_shocks(
        self,
        shocks: Optional[str | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]],
        *,
        periods: int,
        shock_size: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        if isinstance(shocks, str):
            token = shocks.strip()
            token = token[1:] if token.startswith(":") else token
            if token == "none":
                return np.zeros((self.timings.nExo, periods), dtype=np.float64)
            if token == "simulate":
                return self._draw_random_simulation_shocks(
                    periods=periods,
                    shock_size=shock_size,
                    random_seed=random_seed,
                )
        shock_matrix = self._coerce_sep_deterministic_shocks(shocks, periods=periods)
        if shock_matrix is None:
            return np.zeros((self.timings.nExo, periods), dtype=np.float64)
        return np.asarray(shock_matrix, dtype=np.float64).T

    def _draw_random_simulation_shocks(
        self,
        *,
        periods: int,
        shock_size: float,
        random_seed: Optional[int],
    ) -> np.ndarray:
        if self.timings.nExo == 0:
            return np.zeros((0, periods), dtype=np.float64)
        rng = np.random.default_rng(random_seed)
        shock_matrix = rng.standard_normal((self.timings.nExo, periods)) * float(shock_size)
        obc_mask = np.asarray(
            ["ᵒᵇᶜ" in str(name) for name in self.timings.exo],
            dtype=bool,
        )
        shock_matrix[obc_mask, :] = 0.0
        return np.asarray(shock_matrix, dtype=np.float64)

    def _coerce_irf_shock_scenarios(
        self,
        shocks: Optional[
            str | Sequence[str] | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ],
        *,
        periods: int,
        shock_size: float,
        negative_shock: bool,
        random_seed: Optional[int] = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if self.timings.nExo == 0:
            return ("none",), np.zeros((0, periods, 1), dtype=np.float64)

        sign = -1.0 if negative_shock else 1.0
        amplitude = sign * float(shock_size)

        if shocks is None:
            shocks = "all"

        if isinstance(shocks, str):
            token = _strip_selector_prefix(shocks)
            if token == "all":
                selected = tuple(self.timings.exo)
            elif token == "all_excluding_obc":
                selected = tuple(
                    name for name in self.timings.exo if "ᵒᵇᶜ" not in str(name)
                )
            elif token == "simulate":
                shock_matrix = self._draw_random_simulation_shocks(
                    periods=periods,
                    shock_size=shock_size,
                    random_seed=random_seed,
                )
                return ("simulate",), shock_matrix[:, :, None]
            elif token == "none":
                return ("none",), np.zeros((self.timings.nExo, periods, 1), dtype=np.float64)
            else:
                selected = (token,)
            lookup = {name: idx for idx, name in enumerate(self.timings.exo)}
            unexpected = sorted(set(selected).difference(lookup))
            if unexpected:
                raise ValueError("Unknown shock names: " + ", ".join(unexpected))
            scenarios = np.zeros(
                (self.timings.nExo, periods, len(selected)),
                dtype=np.float64,
            )
            for scenario_idx, name in enumerate(selected):
                scenarios[lookup[name], 0, scenario_idx] = amplitude
            return selected, scenarios

        grouped_selection = _flatten_named_selection(shocks)
        if grouped_selection:
            selected = grouped_selection
            lookup = {name: idx for idx, name in enumerate(self.timings.exo)}
            unexpected = sorted(set(selected).difference(lookup))
            if unexpected:
                raise ValueError("Unknown shock names: " + ", ".join(unexpected))
            scenarios = np.zeros(
                (self.timings.nExo, periods, len(selected)),
                dtype=np.float64,
            )
            for scenario_idx, name in enumerate(selected):
                scenarios[lookup[name], 0, scenario_idx] = amplitude
            return selected, scenarios

        shock_matrix = self._coerce_simulation_shocks(shocks, periods=periods)
        return ("custom",), shock_matrix[:, :, None]

    def _simulate_first_order_path(
        self,
        shocks: np.ndarray,
        *,
        first_order_result: ParsedModelFirstOrderResult,
        steady_state: np.ndarray,
        initial_state: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        periods = shocks.shape[1]
        initial_state_values = (
            np.asarray(steady_state, dtype=np.float64)
            if initial_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(initial_state, label="initial_state"),
                dtype=np.float64,
            )
        )
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        reduced_initial_state = (
            initial_state_values[state_indices] - steady_state[state_indices]
        )
        deviations = np.asarray(
            rollout_first_order_solution(
                first_order_result.solution.solution_matrix,
                self.timings,
                shocks,
                initial_reduced_state=reduced_initial_state,
            ),
            dtype=np.float64,
        )
        if deviations.shape != (self.timings.nVars, periods):
            raise ValueError(
                "First-order rollout returned an unexpected shape "
                f"{deviations.shape}, expected ({self.timings.nVars}, {periods})."
            )
        return deviations + steady_state[:, None]

    def _project_first_order_obc_state(
        self,
        lag_state: Sequence[float],
        current_state: Sequence[float],
        shock: Sequence[float],
        *,
        parameter_values: Sequence[float],
        steady_reference_values: Sequence[float],
    ) -> np.ndarray:
        specs = self._first_order_obc_projection_specs
        if specs is None:
            raise ValueError(
                "The parsed OBC equations are not currently supported by the "
                "dedicated first-order enforcement path."
            )
        projected = np.asarray(current_state, dtype=np.float64).copy()
        lag = np.asarray(lag_state, dtype=np.float64)
        shock_values = np.asarray(shock, dtype=np.float64)
        params = np.asarray(parameter_values, dtype=np.float64)
        steady_refs = np.asarray(steady_reference_values, dtype=np.float64)

        for _ in range(max(1, len(specs) + 1)):
            previous = projected.copy()
            args = self._dynamic_input_args_from_context(
                lag,
                projected,
                projected,
                shock_values,
                parameter_values=params,
                steady_reference_values=steady_refs,
            )
            numeric_args = _coerce_symbolic_numeric_args(args)
            symbol_values = {
                symbol: value
                for symbol, value in zip(self._dynamic_input_symbols, numeric_args)
            }
            for spec in specs:
                if spec.mode == "zero_binding":
                    target_value = 0.0
                else:
                    left_value = _evaluate_obc_expression_value(spec.left_expr, symbol_values)
                    right_value = _evaluate_obc_expression_value(spec.right_expr, symbol_values)
                    target_value = (
                        max(left_value, right_value)
                        if spec.operator == "max"
                        else min(left_value, right_value)
                    )
                projected_value = _evaluate_obc_expression_value(
                    spec.inverse_expr,
                    {
                        **symbol_values,
                        spec.target_symbol: target_value,
                    },
                )
                projected[spec.variable_index] = projected_value
                symbol_values[spec.current_symbol] = projected_value
            if np.allclose(projected, previous, rtol=0.0, atol=1e-12):
                break
        return projected

    def _simulate_first_order_obc_path(
        self,
        shocks: np.ndarray,
        *,
        first_order_result: ParsedModelFirstOrderResult,
        steady_state: np.ndarray,
        initial_state: Optional[Sequence[float]] = None,
    ) -> _FirstOrderOBCSimulationResult:
        specs = self._first_order_obc_projection_specs
        if specs is None:
            raise ValueError(
                "The parsed OBC equations are not currently supported by the "
                "dedicated first-order enforcement path."
            )
        periods = shocks.shape[1]
        initial_state_values = (
            np.asarray(steady_state, dtype=np.float64)
            if initial_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(initial_state, label="initial_state"),
                dtype=np.float64,
            )
        )
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        reduced_state = (
            initial_state_values[state_indices] - steady_state[state_indices]
        )
        steady_refs = self._steady_reference_values(np.asarray(steady_state, dtype=np.float64))
        parameter_values = np.asarray(first_order_result.parameter_values, dtype=np.float64)
        solution_matrix = np.asarray(
            first_order_result.solution.solution_matrix,
            dtype=np.float64,
        )
        shock_matrix = np.asarray(shocks, dtype=np.float64).copy()
        exogenous_impact = solution_matrix[:, self.timings.nPast_not_future_and_mixed :]
        exo_symbol_to_index = {
            symbol: idx for idx, symbol in enumerate(self._dynamic_exogenous_symbols)
        }
        spec_shock_indices: list[tuple[int, ...]] = []
        obc_shock_index_set = set(int(idx) for idx in np.asarray(self._obc_shock_indices, dtype=np.int64))
        for spec in specs:
            relevant_symbols = spec.left_expr.free_symbols | spec.right_expr.free_symbols
            indices = tuple(
                sorted(
                    idx
                    for symbol, idx in exo_symbol_to_index.items()
                    if idx in obc_shock_index_set and symbol in relevant_symbols
                )
            )
            spec_shock_indices.append(indices)
        state_path = np.zeros((self.timings.nVars, periods), dtype=np.float64)
        lag_state = np.asarray(initial_state_values, dtype=np.float64)

        for period in range(periods):
            period_shocks = np.asarray(shock_matrix[:, period], dtype=np.float64).copy()
            constrained_state: Optional[np.ndarray] = None
            if self.max_obc_horizon > 0 and self._obc_shock_indices.size > 0:
                window_result = self._optimize_first_order_obc_shock_window(
                    shock_matrix,
                    start_period=period,
                    lag_state=lag_state,
                    first_order_result=first_order_result,
                    steady_state=steady_state,
                )
                if window_result is not None:
                    horizon = window_result.shock_window.shape[1]
                    shock_matrix[:, period : period + horizon] = window_result.shock_window
                    period_shocks = np.asarray(
                        window_result.shock_window[:, 0],
                        dtype=np.float64,
                    ).copy()
                    constrained_state = np.asarray(
                        window_result.state_window[:, 0],
                        dtype=np.float64,
                    ).copy()
            for _ in range(max(1, len(specs) + len(spec_shock_indices) + 1)):
                if constrained_state is not None:
                    break
                deviation = np.asarray(
                    first_order_state_update(
                        solution_matrix,
                        self.timings,
                        reduced_state,
                        period_shocks,
                    ),
                    dtype=np.float64,
                )
                unconstrained_state = deviation + steady_state
                projected_state = self._project_first_order_obc_state(
                    lag_state,
                    unconstrained_state,
                    period_shocks,
                    parameter_values=parameter_values,
                    steady_reference_values=steady_refs,
                )
                constrained_state = projected_state
                desired_change = projected_state[[spec.variable_index for spec in specs]] - unconstrained_state[
                    [spec.variable_index for spec in specs]
                ]
                if np.allclose(desired_change, 0.0, rtol=0.0, atol=1e-12):
                    break
                active_shock_indices = tuple(
                    sorted(
                        {
                            idx
                            for spec_idx, spec in enumerate(specs)
                            if abs(desired_change[spec_idx]) > 1e-12
                            for idx in spec_shock_indices[spec_idx]
                        }
                    )
                )
                if not active_shock_indices:
                    break
                coefficient_matrix = exogenous_impact[
                    [spec.variable_index for spec in specs], :
                ][:, list(active_shock_indices)]
                if coefficient_matrix.size == 0:
                    break
                delta, *_ = np.linalg.lstsq(
                    coefficient_matrix,
                    desired_change,
                    rcond=None,
                )
                if not np.all(np.isfinite(delta)):
                    break
                updated_shocks = period_shocks.copy()
                updated_shocks[list(active_shock_indices)] += np.asarray(delta, dtype=np.float64)
                if np.allclose(updated_shocks, period_shocks, rtol=0.0, atol=1e-12):
                    break
                period_shocks = updated_shocks
            if constrained_state is None:
                deviation = np.asarray(
                    first_order_state_update(
                        solution_matrix,
                        self.timings,
                        reduced_state,
                        period_shocks,
                    ),
                    dtype=np.float64,
                )
                constrained_state = deviation + steady_state
            state_path[:, period] = constrained_state
            shock_matrix[:, period] = period_shocks
            lag_state = constrained_state
            reduced_state = constrained_state[state_indices] - steady_state[state_indices]
        return _FirstOrderOBCSimulationResult(state_path=state_path, shocks=shock_matrix)

    def _optimize_first_order_obc_shock_window(
        self,
        shock_matrix: np.ndarray,
        *,
        start_period: int,
        lag_state: np.ndarray,
        first_order_result: ParsedModelFirstOrderResult,
        steady_state: np.ndarray,
        tol: float = 1e-10,
    ) -> Optional[_FirstOrderOBCWindowOptimizationResult]:
        if self.max_obc_horizon <= 0 or self._obc_shock_indices.size == 0:
            return None
        remaining_periods = shock_matrix.shape[1] - start_period
        if remaining_periods <= 0:
            return None
        horizon = min(remaining_periods, self.max_obc_horizon + 1)
        obc_indices = np.asarray(self._obc_shock_indices, dtype=np.int64)
        if obc_indices.size == 0:
            return None

        base_window = np.asarray(
            shock_matrix[:, start_period : start_period + horizon],
            dtype=np.float64,
        ).copy()
        parameter_values = np.asarray(first_order_result.parameter_values, dtype=np.float64)
        horizon_columns = np.arange(horizon, dtype=np.int64)

        def _unpack_window(flat_window: np.ndarray) -> np.ndarray:
            candidate = base_window.copy()
            candidate[np.ix_(obc_indices, horizon_columns)] = np.asarray(
                flat_window,
                dtype=np.float64,
            ).reshape(obc_indices.size, horizon)
            return candidate

        def _simulate_window(
            candidate_window: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            state_window = np.asarray(
                self._simulate_first_order_path(
                    candidate_window,
                    first_order_result=first_order_result,
                    steady_state=steady_state,
                    initial_state=lag_state,
                ),
                dtype=np.float64,
            )
            full_path = np.concatenate(
                [np.asarray(lag_state, dtype=np.float64)[:, None], state_window],
                axis=1,
            )
            violations = np.asarray(
                self.evaluate_obc_violations_along_path(
                    full_path,
                    shocks=candidate_window,
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                ),
                dtype=np.float64,
            )
            return state_window, violations

        initial_state_window, initial_violations = _simulate_window(base_window)
        if not initial_violations.size or np.max(initial_violations) <= tol:
            return _FirstOrderOBCWindowOptimizationResult(
                shock_window=base_window,
                state_window=initial_state_window,
            )

        x0 = np.asarray(base_window[obc_indices, :], dtype=np.float64).reshape(-1)

        def _objective(flat_window: np.ndarray) -> float:
            flat = np.asarray(flat_window, dtype=np.float64)
            return float(np.dot(flat, flat))

        def _objective_jac(flat_window: np.ndarray) -> np.ndarray:
            return 2.0 * np.asarray(flat_window, dtype=np.float64)

        def _constraint_fun(flat_window: np.ndarray) -> np.ndarray:
            _, violations = _simulate_window(_unpack_window(flat_window))
            return (tol - violations.reshape(-1)).astype(np.float64)

        try:
            result = scipy_optimize.minimize(
                _objective,
                x0,
                jac=_objective_jac,
                method="SLSQP",
                constraints=({"type": "ineq", "fun": _constraint_fun},),
                options={"ftol": tol, "maxiter": 500},
            )
        except Exception:
            return None

        candidate_vector = np.asarray(getattr(result, "x", x0), dtype=np.float64)
        if candidate_vector.shape != x0.shape or not np.all(np.isfinite(candidate_vector)):
            return None
        candidate_window = _unpack_window(candidate_vector)
        candidate_state_window, candidate_violations = _simulate_window(candidate_window)
        if candidate_violations.size and np.max(candidate_violations) > tol:
            return None
        return _FirstOrderOBCWindowOptimizationResult(
            shock_window=candidate_window,
            state_window=candidate_state_window,
        )

    def _build_sep_linear_initial_guess(
        self,
        *,
        parameter_values: np.ndarray,
        steady_state: np.ndarray,
        initial_state: np.ndarray,
        deterministic_shocks: np.ndarray,
        config: SEPConfig,
    ) -> Optional[np.ndarray]:
        try:
            jacobian = self.calculate_jacobian(
                parameter_values=parameter_values,
                steady_state=steady_state,
            )
            first_order_solution = solve_first_order_dsge_solution(
                jacobian,
                self.timings,
                qme_algorithm="schur",
            )
        except Exception:
            return None
        if not bool(np.asarray(first_order_solution.converged)):
            return None

        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        initial_reduced_state = (
            np.asarray(initial_state, dtype=np.float64)[state_indices]
            - np.asarray(steady_state, dtype=np.float64)[state_indices]
        )
        shock_matrix = np.asarray(deterministic_shocks, dtype=np.float64)
        if shock_matrix.shape == (config.periods, self.timings.nExo):
            shock_matrix = shock_matrix.T
        if shock_matrix.shape != (self.timings.nExo, config.periods):
            return None
        try:
            linear_path = np.asarray(
                rollout_first_order_solution(
                    first_order_solution.solution_matrix,
                    self.timings,
                    shock_matrix,
                    initial_reduced_state=initial_reduced_state,
                ),
                dtype=np.float64,
            )
        except Exception:
            return None
        if linear_path.shape != (self.timings.nVars, config.periods):
            return None
        if not np.isfinite(linear_path).all():
            return None

        if config.expectation_method == "hmc":
            group_counts = tuple(1 for _ in range(config.periods + 1))
        else:
            rule = (
                _gauss_hermite_sparse_rule(
                    config.nnodes,
                    self.timings.nExo,
                    config.shock_scale,
                )
                if config.sparse_tree
                else gauss_hermite_rule(
                    config.nnodes,
                    self.timings.nExo,
                    config.shock_scale,
                )
            )
            group_counts = _group_counts(
                config.periods,
                config.branching_order,
                int(rule.weights.shape[0]),
                sparse_tree=config.sparse_tree,
            )

        level_path = linear_path + np.asarray(steady_state, dtype=np.float64)[:, None]
        guess_blocks = [
            np.tile(level_path[:, period_idx], (group_counts[period_idx + 1], 1))
            for period_idx in range(config.periods)
        ]
        return np.vstack(guess_blocks)

    def _sep_obc_maxiter(self) -> int:
        for key in ("zlb_obc_maxiter", "sep_obc_maxiter"):
            raw_value = self.model_options.get(key)
            if raw_value is None:
                continue
            try:
                parsed_value = int(raw_value)
            except Exception:
                continue
            if parsed_value >= 0:
                return parsed_value
        return 3

    def _evaluate_sep_obc_max_violation(
        self,
        *,
        sep_result: ParsedModelSEPResult,
        deterministic_shocks: np.ndarray,
        parameter_values: np.ndarray,
        steady_state: np.ndarray,
        terminal_state: np.ndarray,
    ) -> float:
        if not self.has_obc:
            return 0.0
        try:
            violations = np.asarray(
                self.evaluate_obc_violations_along_path(
                    np.asarray(sep_result.solution.mean_path, dtype=np.float64).T,
                    shocks=deterministic_shocks,
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                    terminal_state=terminal_state,
                ),
                dtype=np.float64,
            )
        except Exception:
            return float("inf")
        if violations.size == 0:
            return 0.0
        if not np.all(np.isfinite(violations)):
            return float("inf")
        return max(float(np.max(violations)), 0.0)

    def _solve_stochastic_extended_path_core(
        self,
        *,
        full_steady_state: np.ndarray,
        parameter_values: np.ndarray,
        initial_state: jax.Array,
        terminal_state: jax.Array,
        config: SEPConfig,
        deterministic_shocks: Optional[jax.Array],
        initial_guess: Optional[Sequence[Sequence[float]]] = None,
    ) -> ParsedModelSEPResult:
        parameter_array = jnp.asarray(parameter_values, dtype=jnp.float64)
        steady_reference_values = jnp.asarray(
            self._steady_reference_values(full_steady_state),
            dtype=jnp.float64,
        )
        effective_config = config
        sep_jacobian_fn = None
        if self.has_obc:
            if config.expectation_method == "gauss_hermite":
                if config.jacobian_method == "auto":
                    effective_config = dataclass_replace(
                        config,
                        jacobian_method="subgradient",
                    )
                if effective_config.jacobian_method == "subgradient":
                    sep_jacobian_fn = self._build_sep_subgradient_jacobian(
                        initial_state=initial_state,
                        terminal_state=terminal_state,
                        parameter_values=parameter_array,
                        steady_reference_values=steady_reference_values,
                        config=effective_config,
                        deterministic_shocks=deterministic_shocks,
                    )
            elif config.jacobian_method == "auto":
                effective_config = dataclass_replace(
                    config,
                    jacobian_method="finite_difference",
                )

        def conditional_residual(
            lag_state: jax.Array,
            current_state: jax.Array,
            lead_state: jax.Array,
            current_shock: jax.Array,
            _params: object,
        ) -> jax.Array:
            return self._evaluate_dynamic_residual_with_context(
                lag_state,
                current_state,
                lead_state,
                current_shock,
                parameter_values=parameter_array,
                steady_reference_values=steady_reference_values,
            )

        deterministic_array = (
            np.zeros((config.periods, self.timings.nExo), dtype=np.float64)
            if deterministic_shocks is None
            else np.asarray(deterministic_shocks, dtype=np.float64)
        )
        sep_initial_guess = initial_guess
        if sep_initial_guess is None:
            sep_initial_guess = self._build_sep_linear_initial_guess(
                parameter_values=np.asarray(parameter_values, dtype=np.float64),
                steady_state=np.asarray(full_steady_state, dtype=np.float64),
                initial_state=np.asarray(initial_state, dtype=np.float64),
                deterministic_shocks=deterministic_array,
                config=effective_config,
            )

        solution = solve_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=initial_state,
            terminal_state=terminal_state,
            shock_dim=self.timings.nExo,
            config=effective_config,
            deterministic_shocks=deterministic_shocks,
            initial_guess=sep_initial_guess,
            jacobian_fn=sep_jacobian_fn,
        )

        return ParsedModelSEPResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=parameter_array,
            solution=solution,
        )

    def _solve_stochastic_extended_path_with_obc_enforcement(
        self,
        *,
        full_steady_state: np.ndarray,
        parameter_values: np.ndarray,
        initial_state: jax.Array,
        terminal_state: jax.Array,
        config: SEPConfig,
        deterministic_shocks: Optional[jax.Array],
        initial_guess: Optional[Sequence[Sequence[float]]] = None,
    ) -> tuple[ParsedModelSEPResult, np.ndarray]:
        current_shocks = (
            np.zeros((config.periods, self.timings.nExo), dtype=np.float64)
            if deterministic_shocks is None
            else np.asarray(deterministic_shocks, dtype=np.float64)
        )
        sep_result = self._solve_stochastic_extended_path_core(
            full_steady_state=np.asarray(full_steady_state, dtype=np.float64),
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            initial_state=initial_state,
            terminal_state=terminal_state,
            config=config,
            deterministic_shocks=jnp.asarray(current_shocks, dtype=jnp.float64),
            initial_guess=initial_guess,
        )
        if (
            not self.has_obc
            or self._obc_shock_indices.size == 0
            or self._first_order_obc_projection_specs is None
        ):
            return sep_result, current_shocks

        violation_tol = max(float(config.tol), 1e-10)
        max_violation = self._evaluate_sep_obc_max_violation(
            sep_result=sep_result,
            deterministic_shocks=current_shocks,
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            steady_state=np.asarray(full_steady_state, dtype=np.float64),
            terminal_state=np.asarray(terminal_state, dtype=np.float64),
        )
        if sep_result.solution.accepted and max_violation <= violation_tol:
            return sep_result, current_shocks

        try:
            first_order_result = self.solve_first_order(
                parameter_values=np.asarray(parameter_values, dtype=np.float64),
                steady_state=np.asarray(full_steady_state, dtype=np.float64),
                qme_algorithm="schur",
            )
        except Exception:
            return sep_result, current_shocks
        if not bool(np.asarray(first_order_result.solution.converged)):
            return sep_result, current_shocks

        current_guess = np.asarray(
            sep_result.solution.stacked_states,
            dtype=np.float64,
        )
        for _ in range(max(1, self._sep_obc_maxiter())):
            linear_obc_result = self._simulate_first_order_obc_path(
                current_shocks.T,
                first_order_result=first_order_result,
                steady_state=np.asarray(full_steady_state, dtype=np.float64),
                initial_state=np.asarray(initial_state, dtype=np.float64),
            )
            updated_shocks = np.asarray(linear_obc_result.shocks.T, dtype=np.float64)
            if (
                np.allclose(updated_shocks, current_shocks, rtol=0.0, atol=1e-12)
                and sep_result.solution.accepted
                and max_violation <= violation_tol
            ):
                break
            current_shocks = updated_shocks
            sep_result = self._solve_stochastic_extended_path_core(
                full_steady_state=np.asarray(full_steady_state, dtype=np.float64),
                parameter_values=np.asarray(parameter_values, dtype=np.float64),
                initial_state=initial_state,
                terminal_state=terminal_state,
                config=config,
                deterministic_shocks=jnp.asarray(current_shocks, dtype=jnp.float64),
                initial_guess=current_guess,
            )
            current_guess = np.asarray(
                sep_result.solution.stacked_states,
                dtype=np.float64,
            )
            max_violation = self._evaluate_sep_obc_max_violation(
                sep_result=sep_result,
                deterministic_shocks=current_shocks,
                parameter_values=np.asarray(parameter_values, dtype=np.float64),
                steady_state=np.asarray(full_steady_state, dtype=np.float64),
                terminal_state=np.asarray(terminal_state, dtype=np.float64),
            )
            if sep_result.solution.accepted and max_violation <= violation_tol:
                break
        return sep_result, current_shocks

    def _simulate_sep_path_with_shocks(
        self,
        shocks: np.ndarray,
        *,
        parameter_values: np.ndarray,
        steady_state: np.ndarray,
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig,
    ) -> _SEPPathSimulationResult:
        periods = shocks.shape[1]
        runtime_periods = max(int(config.periods), periods)
        runtime_shocks = shocks
        if runtime_periods > periods:
            runtime_shocks = np.zeros((shocks.shape[0], runtime_periods), dtype=np.float64)
            runtime_shocks[:, :periods] = shocks
        initial_state_values = (
            np.asarray(steady_state, dtype=np.float64)
            if initial_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(initial_state, label="initial_state"),
                dtype=np.float64,
            )
        )
        terminal_state_values = (
            np.asarray(steady_state, dtype=np.float64)
            if terminal_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(terminal_state, label="terminal_state"),
                dtype=np.float64,
            )
        )
        runtime_config = config
        if runtime_config.periods != runtime_periods:
            runtime_config = dataclass_replace(runtime_config, periods=runtime_periods)
        sep_initial_guess = self._build_sep_linear_initial_guess(
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            steady_state=np.asarray(steady_state, dtype=np.float64),
            initial_state=np.asarray(initial_state_values, dtype=np.float64),
            deterministic_shocks=np.asarray(runtime_shocks, dtype=np.float64),
            config=runtime_config,
        )
        sep_result, used_shocks = self._solve_stochastic_extended_path_with_obc_enforcement(
            full_steady_state=np.asarray(steady_state, dtype=np.float64),
            parameter_values=np.asarray(parameter_values, dtype=np.float64),
            initial_state=jnp.asarray(initial_state_values, dtype=jnp.float64),
            terminal_state=jnp.asarray(terminal_state_values, dtype=jnp.float64),
            config=runtime_config,
            deterministic_shocks=jnp.asarray(runtime_shocks.T, dtype=jnp.float64),
            initial_guess=sep_initial_guess,
        )
        if not sep_result.solution.accepted:
            raise ValueError(
                "Stochastic extended path solve did not meet the configured acceptance tolerance."
            )
        state_path = np.asarray(
            sep_result.solution.mean_path[:, 1 : runtime_periods + 1],
            dtype=np.float64,
        )
        if state_path.shape != (self.timings.nVars, runtime_periods):
            raise ValueError(
                "SEP path returned an unexpected shape "
                f"{state_path.shape}, expected ({self.timings.nVars}, {runtime_periods})."
            )
        return _SEPPathSimulationResult(
            state_path=state_path[:, :periods],
            shocks=np.asarray(used_shocks[:periods].T, dtype=np.float64),
            sep_result=sep_result,
        )

    def _simulate_sep_path(
        self,
        shocks: np.ndarray,
        *,
        parameter_values: np.ndarray,
        steady_state: np.ndarray,
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig,
    ) -> np.ndarray:
        return self._simulate_sep_path_with_shocks(
            shocks,
            parameter_values=parameter_values,
            steady_state=steady_state,
            initial_state=initial_state,
            terminal_state=terminal_state,
            config=config,
        ).state_path


    def _evaluate_dynamic_residual_with_context(
        self,
        lag_state: jax.Array,
        current_state: jax.Array,
        lead_state: jax.Array,
        shock: jax.Array,
        *,
        parameter_values: jax.Array,
        steady_reference_values: jax.Array,
    ) -> jax.Array:
        args = self._dynamic_input_args_from_context(
            lag_state,
            current_state,
            lead_state,
            shock,
            parameter_values=parameter_values,
            steady_reference_values=steady_reference_values,
        )
        return jnp.asarray(self._dynamic_residual_fn(*args), dtype=jnp.float64).reshape(-1)

    def _dynamic_input_args_from_context(
        self,
        lag_state: Sequence[float],
        current_state: Sequence[float],
        lead_state: Sequence[float],
        shock: Sequence[float],
        *,
        parameter_values: Sequence[float],
        steady_reference_values: Sequence[float],
    ) -> tuple[float, ...]:
        lag = jnp.asarray(lag_state, dtype=jnp.float64)
        current = jnp.asarray(current_state, dtype=jnp.float64)
        lead = jnp.asarray(lead_state, dtype=jnp.float64)
        shock_values = jnp.asarray(shock, dtype=jnp.float64)
        parameters = jnp.asarray(parameter_values, dtype=jnp.float64)
        steady_refs = jnp.asarray(steady_reference_values, dtype=jnp.float64)
        args = (
            tuple(lead[idx] for idx in self.timings.future_not_past_and_mixed_idx)
            + tuple(current[idx] for idx in range(self.timings.nVars))
            + tuple(lag[idx] for idx in self.timings.past_not_future_and_mixed_idx)
            + tuple(shock_values[idx] for idx in range(self.timings.nExo))
            + tuple(
                steady_refs[idx]
                for idx in range(len(self.steady_state_reference_names))
            )
            + tuple(parameters[idx] for idx in range(len(self.parameter_names)))
        )
        return args

    def _evaluate_dynamic_jacobian_with_context(
        self,
        lag_state: Sequence[float],
        current_state: Sequence[float],
        lead_state: Sequence[float],
        shock: Sequence[float],
        *,
        parameter_values: Sequence[float],
        steady_reference_values: Sequence[float],
    ) -> jax.Array:
        args = self._dynamic_input_args_from_context(
            lag_state,
            current_state,
            lead_state,
            shock,
            parameter_values=parameter_values,
            steady_reference_values=steady_reference_values,
        )
        if self.has_obc:
            numeric_args = _coerce_symbolic_numeric_args(args)
            symbol_values = {
                symbol: value
                for symbol, value in zip(self._dynamic_input_symbols, numeric_args)
            }
            preferred_symbols = frozenset(self._dynamic_symbols)
            resolved = tuple(
                _freeze_obc_expression(
                    expr,
                    symbol_values=symbol_values,
                    preferred_symbols=preferred_symbols,
                )
                for expr in self._dynamic_expressions
            )
            matrix = sp.Matrix(resolved).jacobian(self._dynamic_symbols)
            values = _evaluate_symbolic_matrix(
                matrix,
                self._dynamic_input_symbols,
                numeric_args,
            )
            return jnp.asarray(values, dtype=jnp.float64)
        values = np.asarray(
            self._dynamic_jacobian_fn(*_coerce_symbolic_numeric_args(args)),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def evaluate_dynamic_residual(
        self,
        lag_state: Sequence[float],
        current_state: Sequence[float],
        lead_state: Sequence[float],
        *,
        shock: Optional[Sequence[float]] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        lag = self._coerce_dynamic_state_vector(lag_state, label="lag_state")
        current = self._coerce_dynamic_state_vector(
            current_state,
            label="current_state",
        )
        lead = self._coerce_dynamic_state_vector(lead_state, label="lead_state")
        if shock is None:
            shock_values = jnp.zeros((self.timings.nExo,), dtype=jnp.float64)
        else:
            shock_values = np.asarray(shock, dtype=np.float64)
            if shock_values.shape != (self.timings.nExo,):
                raise ValueError(
                    "shock must have shape "
                    f"({self.timings.nExo},), got {shock_values.shape}."
                )
            shock_values = jnp.asarray(shock_values, dtype=jnp.float64)

        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
            )
            full_steady_state = np.asarray(
                steady_state_result.steady_state,
                dtype=np.float64,
            )
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )

        return self._evaluate_dynamic_residual_with_context(
            lag,
            current,
            lead,
            shock_values,
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            steady_reference_values=jnp.asarray(
                self._steady_reference_values(full_steady_state),
                dtype=jnp.float64,
            ),
        )

    def evaluate_obc_violations(
        self,
        lag_state: Sequence[float],
        current_state: Sequence[float],
        lead_state: Sequence[float],
        *,
        shock: Optional[Sequence[float]] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        if not self.has_obc:
            return jnp.zeros((0,), dtype=jnp.float64)

        lag = self._coerce_dynamic_state_vector(lag_state, label="lag_state")
        current = self._coerce_dynamic_state_vector(current_state, label="current_state")
        lead = self._coerce_dynamic_state_vector(lead_state, label="lead_state")
        if shock is None:
            shock_values = jnp.zeros((self.timings.nExo,), dtype=jnp.float64)
        else:
            shock_array = np.asarray(shock, dtype=np.float64)
            if shock_array.shape != (self.timings.nExo,):
                raise ValueError(
                    "shock must have shape "
                    f"({self.timings.nExo},), got {shock_array.shape}."
                )
            shock_values = jnp.asarray(shock_array, dtype=jnp.float64)

        if steady_state is None:
            steady_state_result = self.solve_steady_state(parameter_values=parameter_values)
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )

        args = self._dynamic_input_args_from_context(
            lag,
            current,
            lead,
            shock_values,
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            steady_reference_values=jnp.asarray(
                self._steady_reference_values(full_steady_state),
                dtype=jnp.float64,
            ),
        )
        values = _evaluate_symbolic_matrix(
            self._obc_violation_matrix,
            self._dynamic_input_symbols,
            _coerce_symbolic_numeric_args(args),
        ).reshape(-1)
        return jnp.asarray(values, dtype=jnp.float64)

    def evaluate_obc_violations_along_path(
        self,
        state_path: Sequence[Sequence[float]],
        *,
        shocks: Optional[Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        if not self.has_obc:
            path = np.asarray(state_path, dtype=np.float64)
            periods = path.shape[1] - 1 if path.ndim == 2 and path.shape[1] > 0 else 0
            return jnp.zeros((0, max(periods, 0)), dtype=jnp.float64)

        states = np.asarray(state_path, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError(f"state_path must be rank-2, got shape {states.shape}.")
        if states.shape[0] == self.timings.nVars:
            state_matrix = states
        elif states.shape[1] == self.timings.nVars:
            state_matrix = states.T
        else:
            raise ValueError(
                "state_path must have one axis equal to the number of present variables "
                f"({self.timings.nVars}), got {states.shape}."
            )
        if state_matrix.shape[1] < 2:
            raise ValueError(
                "state_path must contain at least two time points (initial plus one period)."
            )

        periods = state_matrix.shape[1] - 1
        shock_matrix = self._coerce_sep_deterministic_shocks(shocks, periods=periods)
        if shock_matrix is None:
            shock_values = np.zeros((periods, self.timings.nExo), dtype=np.float64)
        else:
            shock_values = np.asarray(shock_matrix, dtype=np.float64)

        if steady_state is None:
            steady_state_result = self.solve_steady_state(parameter_values=parameter_values)
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        terminal_state_values = (
            full_steady_state
            if terminal_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(terminal_state, label="terminal_state"),
                dtype=np.float64,
            )
        )

        violations = np.zeros((len(self._obc_violation_expressions), periods), dtype=np.float64)
        for t in range(periods):
            lead_state = (
                state_matrix[:, t + 2]
                if t + 2 < state_matrix.shape[1]
                else terminal_state_values
            )
            violations[:, t] = np.asarray(
                self.evaluate_obc_violations(
                    state_matrix[:, t],
                    state_matrix[:, t + 1],
                    lead_state,
                    shock=shock_values[t],
                    parameter_values=resolved_parameters,
                    steady_state=full_steady_state,
                ),
                dtype=np.float64,
            )
        return jnp.asarray(violations, dtype=jnp.float64)

    def resolve_parameter_values(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        tol: float = 1e-12,
        max_iter: int = 100,
        line_search_min_step: float = 2.0**-16,
    ) -> jax.Array:
        parameters = self._coerce_parameter_values(parameter_values)

        if len(self.parameter_names) == 0:
            return jnp.asarray(parameters, dtype=jnp.float64)
        parameters = self._apply_default_calibrated_parameter_guess(
            parameters,
            parameter_values_provided=parameter_values is not None,
        )
        lower_bounds, upper_bounds = self._bounds_vector(self.parameter_names)

        if steady_state is None:
            if self._parameter_equations_depend_on_steady_state:
                raise ValueError(
                    "Resolving parameter values requires `steady_state` when the "
                    "`@parameters` block contains calibration equations."
                )
            base_steady_state = self._coerce_steady_state_guess(None)
            residual_fn = lambda x: np.asarray(
                    self._parameter_equation_fn(*base_steady_state, *x),
                    dtype=np.float64,
                ).reshape(-1)

            def symbolic_jacobian_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._parameter_equation_jacobian_fn(*base_steady_state, *x),
                    dtype=np.float64,
                )
        else:
            state = np.asarray(steady_state, dtype=np.float64)
            if state.shape == (len(self.steady_state_names),):
                base_steady_state = state
            elif state.shape == (self.timings.nVars,):
                base_steady_state = self._extract_base_steady_state(state)
            else:
                raise ValueError(
                    "steady_state must have shape "
                    f"({len(self.steady_state_names)},) or ({self.timings.nVars},), got {state.shape}."
                )
            residual_fn = lambda x: np.concatenate(
                    [
                        np.asarray(
                            self._steady_state_fn(*base_steady_state, *x),
                            dtype=np.float64,
                        ).reshape(-1),
                        np.asarray(
                            self._parameter_equation_fn(*base_steady_state, *x),
                            dtype=np.float64,
                        ).reshape(-1),
                    ]
                )

            def symbolic_jacobian_fn(x: np.ndarray) -> np.ndarray:
                return np.concatenate(
                    [
                        (
                            self._evaluate_steady_state_obc_parameter_jacobian(
                                base_steady_state,
                                x,
                            )
                            if self.has_obc
                            else np.asarray(
                                self._steady_state_parameter_jacobian_fn(
                                    *base_steady_state,
                                    *x,
                                ),
                                dtype=np.float64,
                            )
                        ),
                        np.asarray(
                            self._parameter_equation_jacobian_fn(*base_steady_state, *x),
                            dtype=np.float64,
                        ),
                    ],
                    axis=0,
                )

        resolved, _, _, _ = _solve_newton_system_with_restarts(
            parameters,
            residual_fn=residual_fn,
            jacobian_fn=_make_safe_jacobian_fn(residual_fn, symbolic_jacobian_fn),
            default_guess=parameters,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            tol=tol,
            max_iter=max_iter,
            line_search_min_step=line_search_min_step,
            nonfinite_message="Initial parameter guess produced non-finite residuals.",
        )
        return jnp.asarray(resolved, dtype=jnp.float64)

    def resolve_parameter_values_jax(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        tol: float = 1e-12,
        max_iter: int = 100,
        line_search_min_step: float = 2.0**-16,
    ) -> jax.Array:
        parameters = self._coerce_parameter_values_jax(parameter_values)

        if len(self.parameter_names) == 0:
            return jnp.asarray(parameters, dtype=jnp.float64)
        parameters = self._apply_default_calibrated_parameter_guess_jax(
            parameters,
            parameter_values_provided=parameter_values is not None,
        )
        lower_bounds, upper_bounds = self._bounds_vector(self.parameter_names)

        if steady_state is None:
            if self._parameter_equations_depend_on_steady_state:
                raise ValueError(
                    "Resolving parameter values requires `steady_state` when the "
                    "`@parameters` block contains calibration equations."
                )
            base_steady_state = jnp.asarray(
                self._coerce_steady_state_guess(None),
                dtype=jnp.float64,
            )

            def residual_fn(x: jax.Array) -> jax.Array:
                return jnp.asarray(
                    self._parameter_equation_residual_jax_fn(*base_steady_state, *x),
                    dtype=jnp.float64,
                ).reshape(-1)
        else:
            state = jnp.asarray(steady_state, dtype=jnp.float64)
            if state.shape == (len(self.steady_state_names),):
                base_steady_state = state
            elif state.shape == (self.timings.nVars,):
                base_steady_state = self._extract_base_steady_state_jax(state)
            else:
                raise ValueError(
                    "steady_state must have shape "
                    f"({len(self.steady_state_names)},) or ({self.timings.nVars},), got {state.shape}."
                )

            def residual_fn(x: jax.Array) -> jax.Array:
                return jnp.concatenate(
                    [
                        jnp.asarray(
                            self._steady_state_residual_jax_fn(*base_steady_state, *x),
                            dtype=jnp.float64,
                        ).reshape(-1),
                        jnp.asarray(
                            self._parameter_equation_residual_jax_fn(*base_steady_state, *x),
                            dtype=jnp.float64,
                        ).reshape(-1),
                    ]
                )

        resolved, _, _, _ = _solve_newton_system_jax_with_restarts(
            parameters,
            residual_fn=residual_fn,
            jacobian_fn=_make_safe_jacobian_fn_jax(
                residual_fn,
                jax.jacrev(residual_fn),
            ),
            default_guess=parameters,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            tol=tol,
            max_iter=max_iter,
            line_search_min_step=line_search_min_step,
        )
        return jnp.asarray(resolved, dtype=jnp.float64)

    def solve_steady_state(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        tol: float = 1e-12,
        max_iter: int = 100,
        line_search_min_step: float = 2.0**-16,
    ) -> SteadyStateResult:
        guess = self._coerce_steady_state_guess(initial_guess)
        initial_parameters = self._apply_default_calibrated_parameter_guess(
            self._coerce_parameter_values(parameter_values),
            parameter_values_provided=parameter_values is not None,
        )
        cached_guess = (
            None
            if initial_guess is not None
            else self._cached_steady_state_guess_for_parameters(initial_parameters)
        )
        if cached_guess is not None:
            guess = cached_guess
        if initial_guess is None:
            if cached_guess is None:
                guess = self._apply_symbolic_steady_state_seed(guess, initial_parameters)
        default_guess = (
            np.asarray(cached_guess, dtype=np.float64).copy()
            if cached_guess is not None
            else self._coerce_steady_state_guess(None)
        )
        if cached_guess is None:
            default_guess = self._apply_symbolic_steady_state_seed(
                default_guess,
                initial_parameters,
            )

        if self._parameter_equations_depend_on_steady_state:
            joint_initial = np.concatenate([guess, initial_parameters])
            lower_bounds, upper_bounds = self._bounds_vector(
                tuple(self.steady_state_names) + tuple(self.parameter_names)
            )

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._joint_steady_state_fn(*x),
                    dtype=np.float64,
                ).reshape(-1)

            def symbolic_jacobian_fn(x: np.ndarray) -> np.ndarray:
                if self.has_obc:
                    return self._evaluate_joint_steady_state_obc_jacobian(x)
                return np.asarray(
                    self._joint_steady_state_jacobian_fn(*x),
                    dtype=np.float64,
                )

            solution, converged, iterations, residual_norm = _solve_newton_system_with_restarts(
                joint_initial,
                residual_fn=residual_fn,
                jacobian_fn=_make_safe_jacobian_fn(residual_fn, symbolic_jacobian_fn),
                default_guess=np.concatenate(
                    [
                        default_guess,
                        initial_parameters,
                    ]
                ),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
                nonfinite_message="Initial steady-state guess produced non-finite residuals.",
            )
            base_steady_state = solution[: len(self.steady_state_names)]
            resolved_parameters = solution[len(self.steady_state_names) :]
        else:
            resolved_parameters = np.asarray(
                self.resolve_parameter_values(
                    parameter_values=parameter_values,
                    tol=tol,
                    max_iter=max_iter,
                    line_search_min_step=line_search_min_step,
                ),
                dtype=np.float64,
            )
            if initial_guess is None and cached_guess is None:
                cached_guess = self._cached_steady_state_guess_for_parameters(
                    resolved_parameters,
                )
                if cached_guess is not None:
                    guess = cached_guess
            if initial_guess is None:
                if cached_guess is None:
                    guess = self._apply_symbolic_steady_state_seed(
                        guess,
                        resolved_parameters,
                    )
            default_guess = (
                np.asarray(cached_guess, dtype=np.float64).copy()
                if cached_guess is not None
                else self._coerce_steady_state_guess(None)
            )
            if cached_guess is None:
                default_guess = self._apply_symbolic_steady_state_seed(
                    default_guess,
                    resolved_parameters,
                )

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._steady_state_fn(*x, *resolved_parameters),
                    dtype=np.float64,
                ).reshape(-1)

            def symbolic_jacobian_fn(x: np.ndarray) -> np.ndarray:
                if self.has_obc:
                    return self._evaluate_steady_state_obc_jacobian(
                        x,
                        resolved_parameters,
                    )
                return np.asarray(
                    self._steady_state_jacobian_fn(*x, *resolved_parameters),
                    dtype=np.float64,
                )

            base_steady_state, converged, iterations, residual_norm = _solve_newton_system_with_restarts(
                guess,
                residual_fn=residual_fn,
                jacobian_fn=_make_safe_jacobian_fn(residual_fn, symbolic_jacobian_fn),
                default_guess=default_guess,
                lower_bounds=self._bounds_vector(self.steady_state_names)[0],
                upper_bounds=self._bounds_vector(self.steady_state_names)[1],
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
                nonfinite_message="Initial steady-state guess produced non-finite residuals.",
            )

        full = self._expand_to_full_steady_state(base_steady_state)
        converged_python = _coerce_optional_python_bool(converged)
        if converged_python:
            self._remember_steady_state_solution(
                resolved_parameters,
                base_steady_state,
            )
        return SteadyStateResult(
            steady_state=jnp.asarray(full, dtype=jnp.float64),
            base_steady_state=jnp.asarray(base_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
        )

    def solve_steady_state_jax(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        tol: float = 1e-12,
        max_iter: int = 100,
        line_search_min_step: float = 2.0**-16,
    ) -> SteadyStateResult:
        guess = jnp.asarray(
            self._coerce_steady_state_guess(initial_guess),
            dtype=jnp.float64,
        )
        initial_parameters = self._apply_default_calibrated_parameter_guess_jax(
            self._coerce_parameter_values_jax(parameter_values),
            parameter_values_provided=parameter_values is not None,
        )
        cached_guess = (
            None
            if initial_guess is not None
            else self._cached_steady_state_guess_for_parameters(initial_parameters)
        )
        if cached_guess is not None:
            guess = jnp.asarray(cached_guess, dtype=jnp.float64)
        if initial_guess is None:
            if cached_guess is None:
                guess = self._apply_symbolic_steady_state_seed_jax(
                    guess,
                    initial_parameters,
                )
        default_guess = (
            jnp.asarray(cached_guess, dtype=jnp.float64)
            if cached_guess is not None
            else jnp.asarray(self._coerce_steady_state_guess(None), dtype=jnp.float64)
        )
        if cached_guess is None:
            default_guess = self._apply_symbolic_steady_state_seed_jax(
                default_guess,
                initial_parameters,
            )

        if self._parameter_equations_depend_on_steady_state:
            joint_initial = jnp.concatenate([guess, initial_parameters])
            lower_bounds, upper_bounds = self._bounds_vector(
                tuple(self.steady_state_names) + tuple(self.parameter_names)
            )

            def residual_fn(x: jax.Array) -> jax.Array:
                return jnp.asarray(
                    self._joint_steady_state_residual_jax_fn(*x),
                    dtype=jnp.float64,
                ).reshape(-1)

            solution, converged, iterations, residual_norm = _solve_newton_system_jax_with_restarts(
                joint_initial,
                residual_fn=residual_fn,
                jacobian_fn=_make_safe_jacobian_fn_jax(
                    residual_fn,
                    jax.jacrev(residual_fn),
                ),
                default_guess=jnp.concatenate([default_guess, initial_parameters]),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
            base_steady_state = solution[: len(self.steady_state_names)]
            resolved_parameters = solution[len(self.steady_state_names) :]
        else:
            resolved_parameters = self.resolve_parameter_values_jax(
                parameter_values=parameter_values,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
            if initial_guess is None and cached_guess is None:
                cached_guess = self._cached_steady_state_guess_for_parameters(
                    resolved_parameters,
                )
                if cached_guess is not None:
                    guess = jnp.asarray(cached_guess, dtype=jnp.float64)
            if initial_guess is None:
                if cached_guess is None:
                    guess = self._apply_symbolic_steady_state_seed_jax(
                        guess,
                        resolved_parameters,
                    )
            default_guess = (
                jnp.asarray(cached_guess, dtype=jnp.float64)
                if cached_guess is not None
                else jnp.asarray(self._coerce_steady_state_guess(None), dtype=jnp.float64)
            )
            if cached_guess is None:
                default_guess = self._apply_symbolic_steady_state_seed_jax(
                    default_guess,
                    resolved_parameters,
                )

            def residual_fn(x: jax.Array) -> jax.Array:
                return jnp.asarray(
                    self._steady_state_residual_jax_fn(*x, *resolved_parameters),
                    dtype=jnp.float64,
                ).reshape(-1)

            base_steady_state, converged, iterations, residual_norm = _solve_newton_system_jax_with_restarts(
                guess,
                residual_fn=residual_fn,
                jacobian_fn=_make_safe_jacobian_fn_jax(
                    residual_fn,
                    jax.jacrev(residual_fn),
                ),
                default_guess=default_guess,
                lower_bounds=self._bounds_vector(self.steady_state_names)[0],
                upper_bounds=self._bounds_vector(self.steady_state_names)[1],
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
        full = self._expand_to_full_steady_state_jax(base_steady_state)
        converged_python = _coerce_optional_python_bool(converged)
        if converged_python:
            self._remember_steady_state_solution(
                resolved_parameters,
                base_steady_state,
            )
        return SteadyStateResult(
            steady_state=full,
            base_steady_state=base_steady_state,
            parameter_values=resolved_parameters,
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
        )

    def _coerce_observable_names(self, observables: Sequence[str] | str) -> tuple[str, ...]:
        names = (observables,) if isinstance(observables, str) else tuple(observables)
        if not names:
            raise ValueError("observables must contain at least one variable name.")
        duplicates = tuple(
            name for idx, name in enumerate(names) if name in names[:idx]
        )
        if duplicates:
            raise ValueError(
                "observables must not contain duplicates, got "
                + ", ".join(duplicates)
                + "."
            )
        return tuple(str(name) for name in names)

    def resolve_observable_indices(
        self,
        observables: Sequence[str] | str,
    ) -> tuple[int, ...]:
        names = self._coerce_observable_names(observables)
        available = set(self.steady_state_names)
        unknown = tuple(name for name in names if name not in available)
        if unknown:
            raise ValueError(
                "Unknown observable names: "
                + ", ".join(unknown)
                + ". Available observables: "
                + ", ".join(self.steady_state_names)
                + "."
            )
        index_lookup = {name: idx for idx, name in enumerate(self.timings.var)}
        return tuple(index_lookup[name] for name in names)

    def build_linear_state_space(
        self,
        observables: Sequence[str] | str,
        *,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    ) -> LinearGaussianStateSpace:
        observable_indices = self.resolve_observable_indices(observables)
        resolved_first_order = first_order_result
        if resolved_first_order is None:
            resolved_first_order = self.solve_first_order(
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                qme_algorithm=qme_algorithm,
            )

        state_space = linear_state_space_from_first_order_solution(
            resolved_first_order.solution.solution_matrix,
            self.timings,
            observable_indices=observable_indices,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=(
                0.0 if measurement_error_covariance is not None else measurement_error_scale
            ),
        )
        if measurement_error_covariance is None:
            return state_space

        observation_covariance = np.asarray(
            measurement_error_covariance,
            dtype=np.float64,
        )
        expected_shape = (len(observable_indices), len(observable_indices))
        if observation_covariance.shape != expected_shape:
            raise ValueError(
                "measurement_error_covariance must have shape "
                f"{expected_shape}, got {observation_covariance.shape}."
            )
        if not np.isfinite(observation_covariance).all():
            raise ValueError(
                "measurement_error_covariance must contain only finite values."
            )
        return state_space._replace(
            observation_noise_covariance=jnp.asarray(
                observation_covariance,
                dtype=state_space.transition_matrix.dtype,
            )
        )

    def _coerce_observations(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        if isinstance(observations, Mapping):
            series_lookup = {
                str(name): np.asarray(series, dtype=np.float64)
                for name, series in observations.items()
            }
            observable_names = (
                tuple(sorted(series_lookup))
                if observables is None
                else tuple(sorted(self._coerce_observable_names(observables)))
            )
            missing = tuple(name for name in observable_names if name not in series_lookup)
            if missing:
                raise ValueError(
                    "Missing observable series for "
                    + ", ".join(missing)
                    + "."
                )
            if observables is not None:
                unexpected = tuple(
                    sorted(set(series_lookup).difference(observable_names))
                )
                if unexpected:
                    raise ValueError(
                        "observations contains unexpected observable names: "
                        + ", ".join(unexpected)
                        + "."
                    )
            stacked: list[np.ndarray] = []
            periods: Optional[int] = None
            for name in observable_names:
                values = series_lookup[name]
                if values.ndim != 1:
                    raise ValueError(
                        f"Observable `{name}` must be one-dimensional, got shape {values.shape}."
                    )
                if periods is None:
                    periods = int(values.shape[0])
                elif values.shape != (periods,):
                    raise ValueError(
                        "All observable series must share the same length, got "
                        f"{values.shape[0]} for `{name}` and {periods} elsewhere."
                    )
                stacked.append(values)
            data = np.vstack(stacked)
        else:
            if observables is None:
                raise ValueError(
                    "observables must be provided when observations are array-like."
                )
            requested_names = self._coerce_observable_names(observables)
            observable_names = tuple(sorted(requested_names))
            data = np.asarray(observations, dtype=np.float64)
            if data.ndim != 2:
                raise ValueError(
                    f"observations must be rank-2, got shape {data.shape}."
                )
            if data.shape[0] != len(observable_names):
                raise ValueError(
                    "observations must have one row per observable, got "
                    f"{data.shape[0]} rows for {len(observable_names)} observables."
                )
            if observable_names != requested_names:
                row_lookup = {name: idx for idx, name in enumerate(requested_names)}
                data = data[[row_lookup[name] for name in observable_names], :]

        if not np.isfinite(data).all():
            raise ValueError("observations must contain only finite values.")
        return observable_names, data

    def _observable_steady_state_values(
        self,
        observables: Sequence[str],
        full_steady_state: np.ndarray,
    ) -> np.ndarray:
        lookup = dict(zip(self.timings.var, full_steady_state))
        return np.asarray([lookup[name] for name in observables], dtype=np.float64)

    def _coerce_named_values(
        self,
        values: Sequence[float] | Mapping[str, float],
        names: Sequence[str],
        *,
        label: str,
    ) -> np.ndarray:
        ordered_names = tuple(str(name) for name in names)
        if isinstance(values, Mapping):
            unexpected = tuple(sorted(set(values).difference(ordered_names)))
            if unexpected:
                raise ValueError(
                    f"{label} contains unexpected names: " + ", ".join(unexpected) + "."
                )
            missing = tuple(name for name in ordered_names if name not in values)
            if missing:
                raise ValueError(
                    f"{label} is missing values for " + ", ".join(missing) + "."
                )
            vector = np.asarray(
                [values[name] for name in ordered_names],
                dtype=np.float64,
            )
        else:
            vector = np.asarray(values, dtype=np.float64)
            if vector.shape != (len(ordered_names),):
                raise ValueError(
                    f"{label} must have shape ({len(ordered_names)},), got {vector.shape}."
                )
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} must contain only finite values.")
        return vector

    def _parameter_values_within_bounds(
        self,
        parameter_values: np.ndarray,
    ) -> bool:
        lower, upper = self._bounds_vector(self.parameter_names)
        return bool(
            np.logical_and(parameter_values >= lower, parameter_values <= upper).all()
        )

    def _prepare_steady_state_and_parameters_for_runtime(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        provided_parameters = None
        if parameter_values is not None:
            provided_parameters = self._coerce_parameter_values(parameter_values)
            if not self._parameter_values_within_bounds(provided_parameters):
                return None, None

        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            if not steady_state_result.converged:
                return None, None
            return (
                np.asarray(steady_state_result.steady_state, dtype=np.float64),
                np.asarray(steady_state_result.parameter_values, dtype=np.float64),
            )

        full_steady_state = self._coerce_full_steady_state(
            steady_state,
            parameter_values=parameter_values,
        )
        resolved_parameters = np.asarray(
            self.resolve_parameter_values(
                parameter_values=provided_parameters,
                steady_state=full_steady_state,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            ),
            dtype=np.float64,
        )
        if not self._parameter_values_within_bounds(resolved_parameters):
            return full_steady_state, None
        return full_steady_state, resolved_parameters

    def _prepare_first_order_solution_for_likelihood(
        self,
        *,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
    ) -> tuple[Optional[ParsedModelFirstOrderResult], Optional[np.ndarray]]:
        if first_order_result is not None:
            full_steady_state = np.asarray(first_order_result.steady_state, dtype=np.float64)
            if not first_order_result.solution.converged:
                return None, full_steady_state
            return first_order_result, full_steady_state

        full_steady_state, resolved_parameters = (
            self._prepare_steady_state_and_parameters_for_runtime(
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
            )
        )
        if full_steady_state is None or resolved_parameters is None:
            return None, full_steady_state

        jacobian = self.calculate_jacobian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        solution = solve_first_order_dsge_solution(
            jacobian,
            self.timings,
            qme_algorithm=qme_algorithm,
        )
        if not solution.converged:
            return None, full_steady_state

        return (
            ParsedModelFirstOrderResult(
                steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
                parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
                jacobian=jacobian,
                solution=solution,
            ),
            full_steady_state,
        )

    def _prepare_first_order_state_space_for_likelihood(
        self,
        observables: Sequence[str] | str,
        *,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    ) -> tuple[Optional[LinearGaussianStateSpace], Optional[np.ndarray]]:
        observable_names = self._coerce_observable_names(observables)
        parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
        )
        if parsed_result is None or full_steady_state is None:
            return None, full_steady_state
        return (
            self.build_linear_state_space(
                observable_names,
                first_order_result=parsed_result,
                qme_algorithm=qme_algorithm,
                initial_covariance_strategy=initial_covariance_strategy,
                measurement_error_scale=measurement_error_scale,
                measurement_error_covariance=measurement_error_covariance,
            ),
            full_steady_state,
        )

    def kalman_loglikelihood(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
        presample_periods: int = 0,
        jitter: float = 1e-9,
        on_failure_loglikelihood: float = -np.inf,
    ) -> jax.Array:
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        state_space, full_steady_state = self._prepare_first_order_state_space_for_likelihood(
            observable_names,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
        )
        if state_space is None or full_steady_state is None:
            return jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)

        demeaned_observations = observation_data - self._observable_steady_state_values(
            observable_names,
            full_steady_state,
        )[:, None]
        return _statespace_kalman_loglikelihood(
            state_space,
            demeaned_observations,
            presample_periods=presample_periods,
            jitter=jitter,
        )

    def kalman_loglikelihood_per_period(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
        presample_periods: int = 0,
        jitter: float = 1e-9,
        on_failure_loglikelihood: float = -np.inf,
    ) -> jax.Array:
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        state_space, full_steady_state = self._prepare_first_order_state_space_for_likelihood(
            observable_names,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
        )
        if state_space is None or full_steady_state is None:
            return jnp.full(
                (observation_data.shape[1],),
                on_failure_loglikelihood,
                dtype=jnp.float64,
            )

        demeaned_observations = observation_data - self._observable_steady_state_values(
            observable_names,
            full_steady_state,
        )[:, None]
        return _statespace_kalman_loglikelihood_per_period(
            state_space,
            demeaned_observations,
            presample_periods=presample_periods,
            jitter=jitter,
        )

    def inversion_loglikelihood(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        algorithm: str = "first_order",
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: Optional[bool] = None,
        sep_maxit: Optional[int] = None,
        sep_tol: Optional[float] = None,
        sep_accept_tol: float = 1e-3,
        sep_shock_scale: Optional[float] = None,
        sep_inv_maxit: int = 8,
        sep_inv_step_tol: float = 1e-6,
        sep_inv_resid_tol: float = 1e-6,
        sep_inv_lambda: float = 1e-4,
        warmup_iterations: int = 0,
        presample_periods: int = 0,
        on_failure_loglikelihood: float = -np.inf,
    ) -> jax.Array:
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        observable_indices = self.resolve_observable_indices(observable_names)
        if algorithm == "first_order":
            parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
                first_order_result=first_order_result,
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                qme_algorithm=qme_algorithm,
            )
            if parsed_result is None or full_steady_state is None:
                return jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
            initial_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if initial_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        initial_state,
                        label="initial_state",
                    ),
                    dtype=np.float64,
                )
            )
            demeaned_observations = observation_data - self._observable_steady_state_values(
                observable_names,
                full_steady_state,
            )[:, None]
            return first_order_inversion_loglikelihood(
                parsed_result.solution.solution_matrix,
                self.timings,
                demeaned_observations,
                observable_indices,
                initial_state=initial_state_values - full_steady_state,
                warmup_iterations=warmup_iterations,
                presample_periods=presample_periods,
                on_failure_loglikelihood=on_failure_loglikelihood,
            )

        if algorithm in {"stochastic_extended_path", "sep"}:
            full_steady_state, resolved_parameters = (
                self._prepare_steady_state_and_parameters_for_runtime(
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                    steady_state_initial_guess=steady_state_initial_guess,
                    steady_state_tol=steady_state_tol,
                    steady_state_max_iter=steady_state_max_iter,
                )
            )
            if full_steady_state is None or resolved_parameters is None:
                return jnp.asarray(on_failure_loglikelihood, dtype=jnp.float64)
            initial_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if initial_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        initial_state,
                        label="initial_state",
                    ),
                    dtype=np.float64,
                )
            )
            terminal_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if terminal_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        terminal_state,
                        label="terminal_state",
                    ),
                    dtype=np.float64,
                )
            )
            demeaned_observations = observation_data - self._observable_steady_state_values(
                observable_names,
                full_steady_state,
            )[:, None]
            return sep_inversion_loglikelihood(
                self,
                demeaned_observations,
                observable_indices,
                parameter_values=resolved_parameters,
                steady_state=full_steady_state,
                initial_state=initial_state_values,
                terminal_state=terminal_state_values,
                config=config,
                sep_periods=sep_periods,
                sep_order=sep_order,
                sep_nnodes=sep_nnodes,
                sep_sparse_tree=sep_sparse_tree,
                sep_maxit=sep_maxit,
                sep_tol=sep_tol,
                sep_accept_tol=sep_accept_tol,
                sep_shock_scale=sep_shock_scale,
                sep_inv_maxit=sep_inv_maxit,
                sep_inv_step_tol=sep_inv_step_tol,
                sep_inv_resid_tol=sep_inv_resid_tol,
                sep_inv_lambda=sep_inv_lambda,
                presample_periods=presample_periods,
                on_failure_loglikelihood=on_failure_loglikelihood,
            )

        raise ValueError(
            f"Unknown inversion algorithm {algorithm!r}. "
            "Use 'first_order' or 'stochastic_extended_path'."
        )

    def inversion_loglikelihood_per_period(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        algorithm: str = "first_order",
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: Optional[bool] = None,
        sep_maxit: Optional[int] = None,
        sep_tol: Optional[float] = None,
        sep_accept_tol: float = 1e-3,
        sep_shock_scale: Optional[float] = None,
        sep_inv_maxit: int = 8,
        sep_inv_step_tol: float = 1e-6,
        sep_inv_resid_tol: float = 1e-6,
        sep_inv_lambda: float = 1e-4,
        warmup_iterations: int = 0,
        presample_periods: int = 0,
        on_failure_loglikelihood: float = -np.inf,
    ) -> jax.Array:
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        observable_indices = self.resolve_observable_indices(observable_names)
        if algorithm == "first_order":
            parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
                first_order_result=first_order_result,
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                qme_algorithm=qme_algorithm,
            )
            if parsed_result is None or full_steady_state is None:
                return jnp.full(
                    (observation_data.shape[1],),
                    on_failure_loglikelihood,
                    dtype=jnp.float64,
                )
            initial_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if initial_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        initial_state,
                        label="initial_state",
                    ),
                    dtype=np.float64,
                )
            )
            demeaned_observations = observation_data - self._observable_steady_state_values(
                observable_names,
                full_steady_state,
            )[:, None]
            return first_order_inversion_loglikelihood_per_period(
                parsed_result.solution.solution_matrix,
                self.timings,
                demeaned_observations,
                observable_indices,
                initial_state=initial_state_values - full_steady_state,
                warmup_iterations=warmup_iterations,
                presample_periods=presample_periods,
                on_failure_loglikelihood=on_failure_loglikelihood,
            )

        if algorithm in {"stochastic_extended_path", "sep"}:
            full_steady_state, resolved_parameters = (
                self._prepare_steady_state_and_parameters_for_runtime(
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                    steady_state_initial_guess=steady_state_initial_guess,
                    steady_state_tol=steady_state_tol,
                    steady_state_max_iter=steady_state_max_iter,
                )
            )
            if full_steady_state is None or resolved_parameters is None:
                return jnp.full(
                    (observation_data.shape[1],),
                    on_failure_loglikelihood,
                    dtype=jnp.float64,
                )
            initial_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if initial_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        initial_state,
                        label="initial_state",
                    ),
                    dtype=np.float64,
                )
            )
            terminal_state_values = (
                np.asarray(full_steady_state, dtype=np.float64)
                if terminal_state is None
                else np.asarray(
                    self._coerce_dynamic_state_vector(
                        terminal_state,
                        label="terminal_state",
                    ),
                    dtype=np.float64,
                )
            )
            demeaned_observations = observation_data - self._observable_steady_state_values(
                observable_names,
                full_steady_state,
            )[:, None]
            return sep_inversion_loglikelihood_per_period(
                self,
                demeaned_observations,
                observable_indices,
                parameter_values=resolved_parameters,
                steady_state=full_steady_state,
                initial_state=initial_state_values,
                terminal_state=terminal_state_values,
                config=config,
                sep_periods=sep_periods,
                sep_order=sep_order,
                sep_nnodes=sep_nnodes,
                sep_sparse_tree=sep_sparse_tree,
                sep_maxit=sep_maxit,
                sep_tol=sep_tol,
                sep_accept_tol=sep_accept_tol,
                sep_shock_scale=sep_shock_scale,
                sep_inv_maxit=sep_inv_maxit,
                sep_inv_step_tol=sep_inv_step_tol,
                sep_inv_resid_tol=sep_inv_resid_tol,
                sep_inv_lambda=sep_inv_lambda,
                presample_periods=presample_periods,
                on_failure_loglikelihood=on_failure_loglikelihood,
            )

        raise ValueError(
            f"Unknown inversion algorithm {algorithm!r}. "
            "Use 'first_order' or 'stochastic_extended_path'."
        )

    def switching_loglikelihood(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        gate_probs: Optional[Sequence[float]] = None,
        hard_mask: Optional[Sequence[bool]] = None,
        fom_algorithm: str = "stochastic_extended_path",
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
        presample_periods: int = 0,
        jitter: float = 1e-9,
        on_failure_loglikelihood: float = -np.inf,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: Optional[bool] = None,
        sep_maxit: Optional[int] = None,
        sep_tol: Optional[float] = None,
        sep_accept_tol: float = 1e-3,
        sep_shock_scale: Optional[float] = None,
        sep_inv_maxit: int = 8,
        sep_inv_step_tol: float = 1e-6,
        sep_inv_resid_tol: float = 1e-6,
        sep_inv_lambda: float = 1e-4,
        switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
    ) -> SwitchingLikelihoodResult:
        rom = self.kalman_loglikelihood_per_period(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )
        fom = self.inversion_loglikelihood_per_period(
            observations,
            observables=observables,
            algorithm=fom_algorithm,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            terminal_state=terminal_state,
            config=config,
            sep_periods=sep_periods,
            sep_order=sep_order,
            sep_nnodes=sep_nnodes,
            sep_sparse_tree=sep_sparse_tree,
            sep_maxit=sep_maxit,
            sep_tol=sep_tol,
            sep_accept_tol=sep_accept_tol,
            sep_shock_scale=sep_shock_scale,
            sep_inv_maxit=sep_inv_maxit,
            sep_inv_step_tol=sep_inv_step_tol,
            sep_inv_resid_tol=sep_inv_resid_tol,
            sep_inv_lambda=sep_inv_lambda,
            presample_periods=presample_periods,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )
        return compute_switching_loglikelihood(
            rom,
            fom,
            hard_mask=hard_mask,
            gate_probs=gate_probs,
            config=switching_config,
        )

    def switching_pipeline_report(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        gate_probs: Optional[Sequence[float]] = None,
        hard_mask: Optional[Sequence[bool]] = None,
        fom_algorithm: str = "stochastic_extended_path",
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
        presample_periods: int = 0,
        jitter: float = 1e-9,
        on_failure_loglikelihood: float = -np.inf,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: Optional[bool] = None,
        sep_maxit: Optional[int] = None,
        sep_tol: Optional[float] = None,
        sep_accept_tol: float = 1e-3,
        sep_shock_scale: Optional[float] = None,
        sep_inv_maxit: int = 8,
        sep_inv_step_tol: float = 1e-6,
        sep_inv_resid_tol: float = 1e-6,
        sep_inv_lambda: float = 1e-4,
        switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
        gate_gain_tol: float = 0.0,
        gate_hard_threshold: float = 0.5,
        budget_frontier_budgets: Optional[Sequence[int]] = None,
        budget_frontier_points: int = 11,
        benchmark_reps: int = 0,
    ) -> dict[str, Any]:
        common_kwargs = dict(
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            terminal_state=terminal_state,
            presample_periods=presample_periods,
            on_failure_loglikelihood=on_failure_loglikelihood,
            config=config,
            sep_periods=sep_periods,
            sep_order=sep_order,
            sep_nnodes=sep_nnodes,
            sep_sparse_tree=sep_sparse_tree,
            sep_maxit=sep_maxit,
            sep_tol=sep_tol,
            sep_accept_tol=sep_accept_tol,
            sep_shock_scale=sep_shock_scale,
            sep_inv_maxit=sep_inv_maxit,
            sep_inv_step_tol=sep_inv_step_tol,
            sep_inv_resid_tol=sep_inv_resid_tol,
            sep_inv_lambda=sep_inv_lambda,
        )
        kalman_kwargs = dict(
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )

        rom = np.asarray(
            self.kalman_loglikelihood_per_period(observations, **kalman_kwargs),
            dtype=np.float64,
        )
        reset_sep_inversion_last_diagnostics()
        fom = np.asarray(
            self.inversion_loglikelihood_per_period(
                observations,
                algorithm=fom_algorithm,
                **common_kwargs,
            ),
            dtype=np.float64,
        )
        fom_sep_diagnostics = get_sep_inversion_last_diagnostics()
        reset_sep_inversion_last_diagnostics()
        switching = self.switching_loglikelihood(
            observations,
            gate_probs=gate_probs,
            hard_mask=hard_mask,
            fom_algorithm=fom_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            jitter=jitter,
            switching_config=switching_config,
            **common_kwargs,
        )
        switching_sep_diagnostics = get_sep_inversion_last_diagnostics()

        runtime_fom_s: Optional[float] = None
        runtime_switching_s: Optional[float] = None
        if benchmark_reps < 0:
            raise ValueError("benchmark_reps must be >= 0.")
        if benchmark_reps > 0:
            start = perf_counter()
            for _ in range(benchmark_reps):
                reset_sep_inversion_last_diagnostics()
                self.inversion_loglikelihood_per_period(
                    observations,
                    algorithm=fom_algorithm,
                    **common_kwargs,
                )
            runtime_fom_s = (perf_counter() - start) / benchmark_reps

            start = perf_counter()
            for _ in range(benchmark_reps):
                reset_sep_inversion_last_diagnostics()
                self.switching_loglikelihood(
                    observations,
                    gate_probs=gate_probs,
                    hard_mask=hard_mask,
                    fom_algorithm=fom_algorithm,
                    initial_covariance_strategy=initial_covariance_strategy,
                    measurement_error_scale=measurement_error_scale,
                    measurement_error_covariance=measurement_error_covariance,
                    jitter=jitter,
                    switching_config=switching_config,
                    **common_kwargs,
                )
            runtime_switching_s = (perf_counter() - start) / benchmark_reps

        hard_mask_array = np.asarray(switching.hard_mask, dtype=bool)
        ll_switching = np.asarray(switching.per_period, dtype=np.float64)
        comparison = evaluate_switching_vs_fom(
            ll_switching,
            fom,
            runtime_switching=runtime_switching_s,
            runtime_fom=runtime_fom_s,
        )
        decision_quality = evaluate_gate_decisions(
            rom,
            fom,
            hard_mask_array,
            gain_tol=gate_gain_tol,
        )
        probability_quality = evaluate_gate_probabilities(
            rom,
            fom,
            np.asarray(switching.gate_probs, dtype=np.float64),
            gain_tol=gate_gain_tol,
            hard_threshold=gate_hard_threshold,
        )
        periods = int(rom.size)
        if budget_frontier_points <= 0:
            raise ValueError("budget_frontier_points must be >= 1.")
        if budget_frontier_budgets is None:
            if periods == 0:
                frontier_budgets = np.asarray([0], dtype=np.int64)
            else:
                frontier_count = min(max(int(budget_frontier_points), 2), periods + 1)
                frontier_budgets = np.unique(
                    np.rint(
                        np.linspace(0, periods, num=frontier_count, dtype=np.float64)
                    ).astype(np.int64)
                )
        else:
            frontier_budgets = np.asarray(budget_frontier_budgets, dtype=np.int64)
        budget_frontier = evaluate_gate_budget_frontier(
            rom,
            fom,
            np.asarray(switching.gate_probs, dtype=np.float64),
            budgets=frontier_budgets,
            gain_tol=gate_gain_tol,
        )
        return {
            "ll_rom": rom,
            "ll_fom": fom,
            "ll_switching": ll_switching,
            "switching_total": float(np.asarray(switching.total, dtype=np.float64)),
            "gate_probs": np.asarray(switching.gate_probs, dtype=np.float64),
            "hard_mask": hard_mask_array,
            "comparison": comparison,
            "decomposition": summarize_loglik_decomposition(rom, fom, hard_mask_array),
            "gate_stats": compute_gate_stats(hard_mask_array),
            "decision_quality": decision_quality,
            "probability_quality": probability_quality,
            "budget_frontier": budget_frontier,
            "runtime": summarize_runtime(
                runtime_switching_s=runtime_switching_s,
                runtime_fom_s=runtime_fom_s,
            ),
            "fom_sep_diagnostics": fom_sep_diagnostics,
            "switching_sep_diagnostics": switching_sep_diagnostics,
        }

    def likelihood_surface_report(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        parameter_draws: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
        observables: Optional[Sequence[str] | str] = None,
        gate_probs: Optional[Sequence[float]] = None,
        hard_mask: Optional[Sequence[bool]] = None,
        fom_algorithm: str = "stochastic_extended_path",
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        initial_covariance_strategy: str = "theoretical",
        measurement_error_scale: float = 1e-9,
        measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
        presample_periods: int = 0,
        jitter: float = 1e-9,
        on_failure_loglikelihood: float = -np.inf,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: Optional[bool] = None,
        sep_maxit: Optional[int] = None,
        sep_tol: Optional[float] = None,
        sep_accept_tol: float = 1e-3,
        sep_shock_scale: Optional[float] = None,
        sep_inv_maxit: int = 8,
        sep_inv_step_tol: float = 1e-6,
        sep_inv_resid_tol: float = 1e-6,
        sep_inv_lambda: float = 1e-4,
        switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
        top_share: float = 0.1,
    ) -> dict[str, Any]:
        draw_matrix = self._coerce_parameter_draw_matrix(
            parameter_draws,
            base_parameter_values=base_parameter_values,
        )
        common_kwargs = dict(
            observables=observables,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_state=initial_state,
            terminal_state=terminal_state,
            presample_periods=presample_periods,
            on_failure_loglikelihood=on_failure_loglikelihood,
            config=config,
            sep_periods=sep_periods,
            sep_order=sep_order,
            sep_nnodes=sep_nnodes,
            sep_sparse_tree=sep_sparse_tree,
            sep_maxit=sep_maxit,
            sep_tol=sep_tol,
            sep_accept_tol=sep_accept_tol,
            sep_shock_scale=sep_shock_scale,
            sep_inv_maxit=sep_inv_maxit,
            sep_inv_step_tol=sep_inv_step_tol,
            sep_inv_resid_tol=sep_inv_resid_tol,
            sep_inv_lambda=sep_inv_lambda,
        )
        kalman_kwargs = dict(
            observables=observables,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_covariance_strategy=initial_covariance_strategy,
            measurement_error_scale=measurement_error_scale,
            measurement_error_covariance=measurement_error_covariance,
            presample_periods=presample_periods,
            jitter=jitter,
            on_failure_loglikelihood=on_failure_loglikelihood,
        )

        ll_rom = np.zeros((draw_matrix.shape[0],), dtype=np.float64)
        ll_fom = np.zeros((draw_matrix.shape[0],), dtype=np.float64)
        ll_switching = None if gate_probs is None and hard_mask is None else np.zeros(
            (draw_matrix.shape[0],),
            dtype=np.float64,
        )

        for draw_idx, parameters in enumerate(draw_matrix):
            ll_rom[draw_idx] = float(
                self.kalman_loglikelihood(
                    observations,
                    parameter_values=parameters,
                    **kalman_kwargs,
                )
            )
            ll_fom[draw_idx] = float(
                self.inversion_loglikelihood(
                    observations,
                    algorithm=fom_algorithm,
                    parameter_values=parameters,
                    **common_kwargs,
                )
            )
            if ll_switching is not None:
                ll_switching[draw_idx] = float(
                    np.asarray(
                        self.switching_loglikelihood(
                            observations,
                            gate_probs=gate_probs,
                            hard_mask=hard_mask,
                            fom_algorithm=fom_algorithm,
                            parameter_values=parameters,
                            initial_covariance_strategy=initial_covariance_strategy,
                            measurement_error_scale=measurement_error_scale,
                            measurement_error_covariance=measurement_error_covariance,
                            jitter=jitter,
                            switching_config=switching_config,
                            **common_kwargs,
                        ).total,
                        dtype=np.float64,
                    )
                )

        rom_vs_fom = evaluate_likelihood_surface_alignment(
            ll_fom,
            ll_rom,
            top_share=top_share,
        )
        switching_surface = (
            None
            if ll_switching is None
            else evaluate_switching_surface_alignment(
                ll_rom,
                ll_fom,
                ll_switching,
                top_share=top_share,
            )
        )

        return {
            "parameter_draws": draw_matrix,
            "ll_rom": ll_rom,
            "ll_fom": ll_fom,
            "ll_switching": ll_switching,
            "rom_vs_fom": rom_vs_fom,
            "switching_surface": switching_surface,
        }

    def _normalize_linear_filter_options(
        self,
        *,
        filter: str,
        algorithm: str,
    ) -> tuple[str, str]:
        filter_name = str(filter)
        if filter_name not in {"kalman", "inversion"}:
            raise ValueError(
                f"Unsupported filter {filter!r}. Use 'kalman' or 'inversion'."
            )
        algorithm_name = str(algorithm)
        if algorithm_name != "first_order":
            raise ValueError(
                "Only the first-order filter helper path is currently ported. "
                f"Got algorithm={algorithm!r}."
            )
        return filter_name, algorithm_name

    def _estimate_first_order_inversion_filter_paths(
        self,
        observation_deviations: np.ndarray,
        observable_indices: Sequence[int],
        solution_matrix: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    ) -> tuple[np.ndarray, np.ndarray]:
        solution = np.asarray(solution_matrix, dtype=np.float64)
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        observable_rows = np.asarray(observable_indices, dtype=np.int64)
        n_past = self.timings.nPast_not_future_and_mixed
        periods = int(observation_deviations.shape[1])

        state = np.zeros((self.timings.nVars,), dtype=np.float64)
        variables = np.zeros((self.timings.nVars, periods), dtype=np.float64)
        shocks = np.zeros((self.timings.nExo, periods), dtype=np.float64)

        jacobian = solution[observable_rows, n_past:]
        if self.timings.nExo == len(observable_rows):
            try:
                inverse_jacobian = np.linalg.inv(jacobian)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Inversion filter failed: observable shock Jacobian is singular."
                ) from exc
        else:
            inverse_jacobian = np.linalg.pinv(jacobian)
        if not np.isfinite(inverse_jacobian).all():
            raise ValueError(
                "Inversion filter failed: could not construct a finite shock map."
            )

        observable_transition = solution[observable_rows, :n_past]
        for period in range(periods):
            reduced_state = state[state_indices]
            residual = (
                observation_deviations[:, period]
                - observable_transition @ reduced_state
            )
            shock_t = inverse_jacobian @ residual
            next_state = solution @ np.concatenate([reduced_state, shock_t], axis=0)
            if not np.isfinite(shock_t).all() or not np.isfinite(next_state).all():
                raise ValueError(
                    "Inversion filter produced non-finite shocks or state estimates."
                )
            shocks[:, period] = shock_t
            variables[:, period] = next_state
            state = next_state

        return shocks, variables

    def _estimate_first_order_kalman_filter_paths(
        self,
        observation_deviations: np.ndarray,
        observable_names: Sequence[str],
        observable_indices: Sequence[int],
        parsed_result: ParsedModelFirstOrderResult,
        *,
        smooth: bool,
        initial_covariance_strategy: str = "theoretical",
        jitter: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        solution = np.asarray(parsed_result.solution.solution_matrix, dtype=np.float64)
        n_past = self.timings.nPast_not_future_and_mixed
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        transition = np.zeros((self.timings.nVars, self.timings.nVars), dtype=np.float64)
        transition[:, state_indices] = solution[:, :n_past]
        shock_impact = solution[:, n_past:]
        observation = np.zeros(
            (len(observable_indices), self.timings.nVars),
            dtype=np.float64,
        )
        observation[
            np.arange(len(observable_indices), dtype=np.int64),
            np.asarray(observable_indices, dtype=np.int64),
        ] = 1.0
        process_covariance = shock_impact @ shock_impact.T
        if initial_covariance_strategy == "theoretical":
            covariance = np.asarray(
                solve_discrete_lyapunov_direct(
                    transition,
                    process_covariance,
                ).solution,
                dtype=np.float64,
            )
        elif initial_covariance_strategy == "diagonal":
            covariance = 10.0 * np.eye(self.timings.nVars, dtype=np.float64)
        else:
            raise ValueError(
                "initial_covariance_strategy must be 'theoretical' or 'diagonal'."
            )

        periods = int(observation_deviations.shape[1])
        innovations = np.zeros((observation.shape[0], periods), dtype=np.float64)
        predicted_states = np.zeros((self.timings.nVars, periods + 1), dtype=np.float64)
        inverse_innovation_covariances = np.zeros(
            (periods, observation.shape[0], observation.shape[0]),
            dtype=np.float64,
        )
        kalman_l_matrices = np.zeros(
            (periods, self.timings.nVars, self.timings.nVars),
            dtype=np.float64,
        )
        predicted_covariances = np.zeros(
            (periods + 1, self.timings.nVars, self.timings.nVars),
            dtype=np.float64,
        )
        predicted_covariances[0] = covariance
        filtered_shocks = np.zeros((self.timings.nExo, periods), dtype=np.float64)

        for period in range(periods):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                innovation = (
                    observation_deviations[:, period]
                    - observation @ predicted_states[:, period]
                )
                innovation_covariance = (
                    observation @ predicted_covariances[period] @ observation.T
                )
                if jitter > 0.0:
                    innovation_covariance = innovation_covariance + jitter * np.eye(
                        innovation.shape[0],
                        dtype=np.float64,
                    )
                try:
                    inverse_innovation = np.linalg.inv(innovation_covariance)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(
                        "Kalman filter helper failed: innovation covariance is singular."
                    ) from exc
                if not np.isfinite(inverse_innovation).all():
                    raise ValueError(
                        "Kalman filter helper failed: innovation covariance inverse is non-finite."
                    )
                projected_gain = (
                    predicted_covariances[period] @ observation.T @ inverse_innovation
                )
                kalman_l = transition - transition @ projected_gain @ observation
                next_covariance = (
                    transition @ predicted_covariances[period] @ kalman_l.T
                    + process_covariance
                )
                next_state = transition @ (
                    predicted_states[:, period] + projected_gain @ innovation
                )
                shock_t = (
                    shock_impact.T @ observation.T @ inverse_innovation @ innovation
                )
            if (
                not np.isfinite(innovation).all()
                or not np.isfinite(next_covariance).all()
                or not np.isfinite(next_state).all()
                or not np.isfinite(shock_t).all()
            ):
                raise ValueError(
                    "Kalman filter helper produced non-finite shocks or state estimates."
                )
            innovations[:, period] = innovation
            inverse_innovation_covariances[period] = inverse_innovation
            kalman_l_matrices[period] = kalman_l
            predicted_covariances[period + 1] = next_covariance
            predicted_states[:, period + 1] = next_state
            filtered_shocks[:, period] = shock_t

        filtered_variables = predicted_states[:, 1:]
        if not smooth:
            return filtered_shocks, filtered_variables

        smoothed_variables = np.zeros((self.timings.nVars, periods), dtype=np.float64)
        smoothed_shocks = np.zeros((self.timings.nExo, periods), dtype=np.float64)
        r_vector = np.zeros((self.timings.nVars,), dtype=np.float64)
        n_matrix = np.zeros((self.timings.nVars, self.timings.nVars), dtype=np.float64)

        for period in range(periods - 1, -1, -1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                inverse_innovation = inverse_innovation_covariances[period]
                kalman_l = kalman_l_matrices[period]
                innovation = innovations[:, period]
                r_vector = (
                    observation.T @ inverse_innovation @ innovation
                    + kalman_l.T @ r_vector
                )
                smoothed_variables[:, period] = (
                    predicted_states[:, period]
                    + predicted_covariances[period] @ r_vector
                )
                n_matrix = (
                    observation.T @ inverse_innovation @ observation
                    + kalman_l.T @ n_matrix @ kalman_l
                )
                smoothed_shocks[:, period] = shock_impact.T @ r_vector

        return smoothed_shocks, smoothed_variables

    def _estimate_observed_shocks_and_variables_matrix(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        data_in_levels: bool = True,
        levels: bool = True,
        smooth: bool = False,
        initial_covariance_strategy: str = "theoretical",
        jitter: float = 1e-9,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        filter_name, _ = self._normalize_linear_filter_options(
            filter=filter,
            algorithm=algorithm,
        )
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        observable_indices = self.resolve_observable_indices(observable_names)
        parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
        )
        if parsed_result is None or full_steady_state is None:
            raise ValueError(
                "Could not prepare a converged first-order solution for linear filtering."
            )

        steady_observables = self._observable_steady_state_values(
            observable_names,
            full_steady_state,
        )
        observation_deviations = (
            observation_data - steady_observables[:, None]
            if data_in_levels
            else observation_data
        )
        if observation_deviations.size == 0 or np.max(np.abs(observation_deviations)) <= 1e-14:
            shocks = np.zeros(
                (self.timings.nExo, observation_data.shape[1]),
                dtype=np.float64,
            )
            variables = np.zeros(
                (self.timings.nVars, observation_data.shape[1]),
                dtype=np.float64,
            )
            if levels:
                variables = variables + full_steady_state[:, None]
            return observable_names, shocks, variables

        if filter_name == "inversion":
            shocks, variables = self._estimate_first_order_inversion_filter_paths(
                observation_deviations,
                observable_indices,
                parsed_result.solution.solution_matrix,
            )
        else:
            shocks, variables = self._estimate_first_order_kalman_filter_paths(
                observation_deviations,
                observable_names,
                observable_indices,
                parsed_result,
                smooth=smooth,
                initial_covariance_strategy=initial_covariance_strategy,
                jitter=jitter,
            )

        if levels:
            variables = variables + full_steady_state[:, None]
        return observable_names, shocks, variables

    def estimate_observed_shocks_matrix(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        data_in_levels: bool = True,
        smooth: bool = False,
        verbose: bool = False,
        expected_rows: Optional[int] = None,
        expected_cols: Optional[int] = None,
        label: str = "Estimated shocks",
        initial_covariance_strategy: str = "theoretical",
        jitter: float = 1e-9,
    ) -> jax.Array:
        del verbose
        _, shocks, _ = self._estimate_observed_shocks_and_variables_matrix(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=data_in_levels,
            levels=False,
            smooth=smooth,
            initial_covariance_strategy=initial_covariance_strategy,
            jitter=jitter,
        )
        if expected_rows is not None and shocks.shape[0] != int(expected_rows):
            raise ValueError(
                f"{label} row mismatch: got {shocks.shape[0]}, expected {int(expected_rows)}."
            )
        if expected_cols is not None and shocks.shape[1] != int(expected_cols):
            raise ValueError(
                f"{label} length mismatch: got {shocks.shape[1]}, expected {int(expected_cols)}."
            )
        return jnp.asarray(shocks, dtype=jnp.float64)

    def estimate_observed_variables_matrix(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        data_in_levels: bool = True,
        levels: bool = True,
        smooth: bool = False,
        verbose: bool = False,
        expected_rows: Optional[int] = None,
        expected_cols: Optional[int] = None,
        label: str = "Estimated variables",
        initial_covariance_strategy: str = "theoretical",
        jitter: float = 1e-9,
    ) -> tuple[jax.Array, tuple[str, ...]]:
        del verbose
        _, _, variables = self._estimate_observed_shocks_and_variables_matrix(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=data_in_levels,
            levels=levels,
            smooth=smooth,
            initial_covariance_strategy=initial_covariance_strategy,
            jitter=jitter,
        )
        variable_names = tuple(self.timings.var)
        if expected_rows is not None and variables.shape[0] != int(expected_rows):
            raise ValueError(
                f"{label} row mismatch: got {variables.shape[0]}, expected {int(expected_rows)}."
            )
        if expected_cols is not None and variables.shape[1] != int(expected_cols):
            raise ValueError(
                f"{label} length mismatch: got {variables.shape[1]}, expected {int(expected_cols)}."
            )
        if len(variable_names) != variables.shape[0]:
            raise ValueError(
                f"{label} variable-name count mismatch: got {len(variable_names)} names for {variables.shape[0]} rows."
            )
        return jnp.asarray(variables, dtype=jnp.float64), variable_names

    def linear_filter_initial_state(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        state_names: Sequence[str] | str,
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        smooth: bool = False,
        label: str = "Linear filter variables",
    ) -> jax.Array:
        names = self._coerce_observable_names(state_names)
        variables, variable_names = self.estimate_observed_variables_matrix(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=True,
            levels=True,
            smooth=smooth,
            expected_cols=np.asarray(self._coerce_observations(
                observations,
                observables=observables,
            )[1]).shape[1],
            label=label,
        )
        variable_lookup = {name: idx for idx, name in enumerate(variable_names)}
        missing = tuple(name for name in names if name not in variable_lookup)
        if missing:
            raise ValueError(
                "state_names not found in linear filter output: "
                + ", ".join(missing)
                + "."
            )
        selected = np.asarray(variables, dtype=np.float64)[
            [variable_lookup[name] for name in names],
            -1,
        ]
        return jnp.asarray(selected, dtype=jnp.float64)

    def linear_filter_full_state_initial(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        smooth: bool = False,
        label: str = "Linear filter variables",
    ) -> jax.Array:
        observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )[1]
        variables, _ = self.estimate_observed_variables_matrix(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=True,
            levels=True,
            smooth=smooth,
            expected_cols=observation_data.shape[1],
            label=label,
        )
        return jnp.asarray(np.asarray(variables, dtype=np.float64)[:, 0], dtype=jnp.float64)

    def compute_linear_gate_stats_from_filter(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        obs_sigma: Sequence[float] | Mapping[str, float],
        shock_sigmas: Sequence[float] | Mapping[str, float],
        state_names: Optional[Sequence[str] | str] = None,
        *,
        observables: Optional[Sequence[str] | str] = None,
        periods: Optional[int] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        filter: str = "kalman",
        algorithm: str = "first_order",
        smooth: bool = False,
        shock_norm: str = "l2",
        error_norm: str = "l2",
        label: str = "Linear gate stats",
    ) -> LinearGateStatsResult:
        del state_names
        if periods is not None and int(periods) <= 0:
            raise ValueError(f"periods must be positive, got {periods}.")

        observable_names, shocks, variables = self._estimate_observed_shocks_and_variables_matrix(
            observations,
            observables=observables,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            filter=filter,
            algorithm=algorithm,
            data_in_levels=True,
            levels=True,
            smooth=smooth,
        )
        return self.compute_linear_gate_stats_from_shocks(
            observations,
            shocks,
            obs_sigma,
            shock_sigmas,
            observables=observable_names,
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
            initial_state=variables[:, 0],
            shock_norm=shock_norm,
            error_norm=error_norm,
        )

    def compute_linear_gate_stats_from_shocks(
        self,
        observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        obs_sigma: Sequence[float] | Mapping[str, float],
        shock_sigmas: Sequence[float] | Mapping[str, float],
        *,
        observables: Optional[Sequence[str] | str] = None,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        shock_norm: str = "l2",
        error_norm: str = "l2",
    ) -> LinearGateStatsResult:
        observable_names, observation_data = self._coerce_observations(
            observations,
            observables=observables,
        )
        observable_indices = self.resolve_observable_indices(observable_names)
        parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
        )
        if parsed_result is None or full_steady_state is None:
            raise ValueError(
                "Could not prepare a converged first-order solution for linear gate statistics."
            )

        periods = int(observation_data.shape[1])
        shock_values = self._coerce_sep_deterministic_shocks(
            shocks,
            periods=periods,
        )
        if shock_values is None:
            raise ValueError("shocks must be provided for linear gate statistics.")
        shock_matrix = np.asarray(shock_values, dtype=np.float64).T
        obs_sigma_vector = self._coerce_named_values(
            obs_sigma,
            observable_names,
            label="obs_sigma",
        )
        shock_sigma_vector = self._coerce_named_values(
            shock_sigmas,
            self.timings.exo,
            label="shock_sigmas",
        )
        initial_state_values = (
            np.asarray(full_steady_state, dtype=np.float64)
            if initial_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(
                    initial_state,
                    label="initial_state",
                ),
                dtype=np.float64,
            )
        )
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        reduced_initial_state = (
            initial_state_values[state_indices] - full_steady_state[state_indices]
        )
        linear_deviations = np.asarray(
            rollout_first_order_solution(
                parsed_result.solution.solution_matrix,
                self.timings,
                shock_matrix,
                initial_reduced_state=reduced_initial_state,
            ),
            dtype=np.float64,
        )
        linear_observations = linear_deviations[list(observable_indices), :] + (
            self._observable_steady_state_values(
                observable_names,
                full_steady_state,
            )[:, None]
        )
        e_stat, f_stat = compute_gate_stat_series(
            observation_data,
            linear_observations,
            shock_matrix,
            obs_sigma_vector,
            shock_sigma_vector,
            shock_norm=shock_norm,
            error_norm=error_norm,
        )
        return LinearGateStatsResult(
            linear_observations=jnp.asarray(linear_observations, dtype=jnp.float64),
            shocks=jnp.asarray(shock_matrix, dtype=jnp.float64),
            e_stat=jnp.asarray(e_stat, dtype=jnp.float64),
            f_stat=jnp.asarray(f_stat, dtype=jnp.float64),
        )

    def compute_first_order_obc_violation_path(
        self,
        shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
    ) -> OBCViolationPathResult:
        periods = None
        if isinstance(shocks, Mapping):
            if shocks:
                periods = len(next(iter(shocks.values())))
        else:
            shock_array = np.asarray(shocks, dtype=np.float64)
            if shock_array.ndim != 2:
                raise ValueError(f"shocks must be rank-2, got shape {shock_array.shape}.")
            periods = shock_array.shape[1] if shock_array.shape[0] == self.timings.nExo else shock_array.shape[0]
        if periods is None:
            raise ValueError("shocks must contain at least one period.")

        shock_values = self._coerce_sep_deterministic_shocks(shocks, periods=periods)
        if shock_values is None:
            raise ValueError("shocks must contain at least one period.")
        shock_matrix = np.asarray(shock_values, dtype=np.float64).T

        parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
            first_order_result=first_order_result,
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            qme_algorithm=qme_algorithm,
        )
        if parsed_result is None or full_steady_state is None:
            raise ValueError(
                "Could not prepare a converged first-order solution for OBC violations."
            )

        initial_state_values = (
            np.asarray(full_steady_state, dtype=np.float64)
            if initial_state is None
            else np.asarray(
                self._coerce_dynamic_state_vector(initial_state, label="initial_state"),
                dtype=np.float64,
            )
        )
        state_indices = np.asarray(
            self.timings.past_not_future_and_mixed_idx,
            dtype=np.int64,
        )
        reduced_initial_state = (
            initial_state_values[state_indices] - full_steady_state[state_indices]
        )
        linear_deviations = np.asarray(
            rollout_first_order_solution(
                parsed_result.solution.solution_matrix,
                self.timings,
                shock_matrix,
                initial_reduced_state=reduced_initial_state,
            ),
            dtype=np.float64,
        )
        state_path = np.concatenate(
            [
                initial_state_values[:, None],
                linear_deviations + full_steady_state[:, None],
            ],
            axis=1,
        )
        violations = self.evaluate_obc_violations_along_path(
            state_path,
            shocks=shock_matrix,
            parameter_values=np.asarray(parsed_result.parameter_values, dtype=np.float64),
            steady_state=full_steady_state,
            terminal_state=terminal_state,
        )
        return OBCViolationPathResult(
            state_path=jnp.asarray(state_path, dtype=jnp.float64),
            shocks=jnp.asarray(shock_matrix, dtype=jnp.float64),
            violations=violations,
        )

    def simulate(
        self,
        *,
        periods: int,
        shocks: Optional[str | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
        variables: Optional[Sequence[str] | str] = None,
        shock_size: float = 1.0,
        random_seed: Optional[int] = None,
        algorithm: str = "first_order",
        ignore_obc: bool = False,
        levels: bool = True,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
    ) -> ModelSimulationResult:
        if periods < 1:
            raise ValueError(f"periods must be positive, got {periods}.")
        selected_variables, variable_indices = self._resolve_variable_selection(variables)
        shock_matrix = self._coerce_simulation_shocks(
            shocks,
            periods=periods,
            shock_size=shock_size,
            random_seed=random_seed,
        )
        effective_ignore_obc = self._effective_ignore_obc_flag(
            shock_matrix,
            ignore_obc=ignore_obc,
        )

        algorithm_token = algorithm.lower()
        if algorithm_token not in {"first_order", "stochastic_extended_path", "sep"}:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. "
                "Use 'first_order' or 'stochastic_extended_path'."
            )
        use_first_order_obc = (
            algorithm_token == "first_order"
            and self.has_obc
            and not effective_ignore_obc
            and terminal_state is None
            and self._first_order_obc_projection_specs is not None
        )
        use_sep = algorithm_token in {"stochastic_extended_path", "sep"} or (
            algorithm_token == "first_order"
            and self.has_obc
            and not effective_ignore_obc
            and not use_first_order_obc
        )

        if use_sep:
            full_steady_state, resolved_parameters = (
                self._prepare_steady_state_and_parameters_for_runtime(
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                    steady_state_initial_guess=steady_state_initial_guess,
                    steady_state_tol=steady_state_tol,
                    steady_state_max_iter=steady_state_max_iter,
                )
            )
            if full_steady_state is None or resolved_parameters is None:
                raise ValueError("Could not prepare a converged steady state for simulation.")
            runtime_config = config
            if algorithm_token == "first_order":
                runtime_config = dataclass_replace(
                    runtime_config,
                    periods=max(runtime_config.periods, periods, self.max_obc_horizon),
                    branching_order=0,
                )
            sep_runtime = self._simulate_sep_path_with_shocks(
                shock_matrix,
                parameter_values=resolved_parameters,
                steady_state=np.asarray(full_steady_state, dtype=np.float64),
                initial_state=initial_state,
                terminal_state=terminal_state,
                config=runtime_config,
            )
            state_path = sep_runtime.state_path
            algorithm_used = "stochastic_extended_path"
            parameter_output = np.asarray(resolved_parameters, dtype=np.float64)
            steady_output = np.asarray(full_steady_state, dtype=np.float64)
            shock_output = np.asarray(sep_runtime.shocks, dtype=np.float64)
        else:
            parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
                first_order_result=first_order_result,
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                qme_algorithm=qme_algorithm,
            )
            if parsed_result is None or full_steady_state is None:
                raise ValueError("Could not prepare a converged first-order solution for simulation.")
            if use_first_order_obc:
                obc_result = self._simulate_first_order_obc_path(
                    shock_matrix,
                    first_order_result=parsed_result,
                    steady_state=np.asarray(full_steady_state, dtype=np.float64),
                    initial_state=initial_state,
                )
                state_path = obc_result.state_path
                shock_output = obc_result.shocks
            else:
                state_path = self._simulate_first_order_path(
                    shock_matrix,
                    first_order_result=parsed_result,
                    steady_state=np.asarray(full_steady_state, dtype=np.float64),
                    initial_state=initial_state,
                )
                shock_output = np.asarray(shock_matrix, dtype=np.float64)
            algorithm_used = "first_order"
            parameter_output = np.asarray(parsed_result.parameter_values, dtype=np.float64)
            steady_output = np.asarray(full_steady_state, dtype=np.float64)

        data = state_path[variable_indices]
        if not levels:
            data = data - steady_output[variable_indices][:, None]
        return ModelSimulationResult(
            variables=selected_variables,
            data=jnp.asarray(data, dtype=jnp.float64),
            state_path=jnp.asarray(state_path, dtype=jnp.float64),
            shocks=jnp.asarray(shock_output, dtype=jnp.float64),
            algorithm_used=algorithm_used,
            steady_state=jnp.asarray(steady_output, dtype=jnp.float64),
            parameter_values=jnp.asarray(parameter_output, dtype=jnp.float64),
        )

    def get_irf(
        self,
        *,
        periods: int,
        variables: Optional[Sequence[str] | str] = None,
        shocks: Optional[
            str | Sequence[str] | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ] = "all",
        shock_size: float = 1.0,
        negative_shock: bool = False,
        random_seed: Optional[int] = None,
        algorithm: str = "first_order",
        ignore_obc: bool = False,
        levels: bool = False,
        first_order_result: Optional[ParsedModelFirstOrderResult] = None,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
    ) -> ModelIRFResult:
        if periods < 1:
            raise ValueError(f"periods must be positive, got {periods}.")
        selected_variables, variable_indices = self._resolve_variable_selection(variables)
        shock_names, shock_scenarios = self._coerce_irf_shock_scenarios(
            shocks,
            periods=periods,
            shock_size=shock_size,
            negative_shock=negative_shock,
            random_seed=random_seed,
        )
        effective_ignore_obc = self._effective_ignore_obc_flag(
            shock_scenarios,
            ignore_obc=ignore_obc,
        )

        algorithm_token = algorithm.lower()
        if algorithm_token not in {"first_order", "stochastic_extended_path", "sep"}:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. "
                "Use 'first_order' or 'stochastic_extended_path'."
            )
        use_first_order_obc = (
            algorithm_token == "first_order"
            and self.has_obc
            and not effective_ignore_obc
            and terminal_state is None
            and self._first_order_obc_projection_specs is not None
        )
        use_sep = algorithm_token in {"stochastic_extended_path", "sep"} or (
            algorithm_token == "first_order"
            and self.has_obc
            and not effective_ignore_obc
            and not use_first_order_obc
        )

        state_paths = np.zeros(
            (self.timings.nVars, periods, shock_scenarios.shape[2]),
            dtype=np.float64,
        )
        algorithm_used = "stochastic_extended_path" if use_sep else "first_order"

        if use_sep:
            full_steady_state, resolved_parameters = (
                self._prepare_steady_state_and_parameters_for_runtime(
                    parameter_values=parameter_values,
                    steady_state=steady_state,
                    steady_state_initial_guess=steady_state_initial_guess,
                    steady_state_tol=steady_state_tol,
                    steady_state_max_iter=steady_state_max_iter,
                )
            )
            if full_steady_state is None or resolved_parameters is None:
                raise ValueError("Could not prepare a converged steady state for IRFs.")
            runtime_config = config
            if algorithm_token == "first_order":
                runtime_config = dataclass_replace(
                    runtime_config,
                    periods=max(runtime_config.periods, periods, self.max_obc_horizon),
                    branching_order=0,
                )
            shock_output = np.asarray(shock_scenarios, dtype=np.float64).copy()
            for scenario_idx in range(shock_scenarios.shape[2]):
                sep_runtime = self._simulate_sep_path_with_shocks(
                    shock_scenarios[:, :, scenario_idx],
                    parameter_values=np.asarray(resolved_parameters, dtype=np.float64),
                    steady_state=np.asarray(full_steady_state, dtype=np.float64),
                    initial_state=initial_state,
                    terminal_state=terminal_state,
                    config=runtime_config,
                )
                state_paths[:, :, scenario_idx] = sep_runtime.state_path
                shock_output[:, :, scenario_idx] = sep_runtime.shocks
            parameter_output = np.asarray(resolved_parameters, dtype=np.float64)
            steady_output = np.asarray(full_steady_state, dtype=np.float64)
        else:
            parsed_result, full_steady_state = self._prepare_first_order_solution_for_likelihood(
                first_order_result=first_order_result,
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                qme_algorithm=qme_algorithm,
            )
            if parsed_result is None or full_steady_state is None:
                raise ValueError("Could not prepare a converged first-order solution for IRFs.")
            shock_output = np.asarray(shock_scenarios, dtype=np.float64).copy()
            for scenario_idx in range(shock_scenarios.shape[2]):
                if use_first_order_obc:
                    obc_result = self._simulate_first_order_obc_path(
                        shock_scenarios[:, :, scenario_idx],
                        first_order_result=parsed_result,
                        steady_state=np.asarray(full_steady_state, dtype=np.float64),
                        initial_state=initial_state,
                    )
                    state_paths[:, :, scenario_idx] = obc_result.state_path
                    shock_output[:, :, scenario_idx] = obc_result.shocks
                else:
                    state_paths[:, :, scenario_idx] = self._simulate_first_order_path(
                        shock_scenarios[:, :, scenario_idx],
                        first_order_result=parsed_result,
                        steady_state=np.asarray(full_steady_state, dtype=np.float64),
                        initial_state=initial_state,
                    )
            parameter_output = np.asarray(parsed_result.parameter_values, dtype=np.float64)
            steady_output = np.asarray(full_steady_state, dtype=np.float64)

        responses = state_paths[variable_indices]
        if not levels:
            responses = responses - steady_output[variable_indices][:, None, None]
        return ModelIRFResult(
            variables=selected_variables,
            shock_names=shock_names,
            responses=jnp.asarray(responses, dtype=jnp.float64),
            state_paths=jnp.asarray(state_paths, dtype=jnp.float64),
            shocks=jnp.asarray(shock_output, dtype=jnp.float64),
            algorithm_used=algorithm_used,
            steady_state=jnp.asarray(steady_output, dtype=jnp.float64),
            parameter_values=jnp.asarray(parameter_output, dtype=jnp.float64),
        )

    def _apply_default_calibrated_parameter_guess(
        self,
        parameter_values: np.ndarray,
        *,
        parameter_values_provided: bool,
    ) -> np.ndarray:
        if parameter_values_provided or not self.default_initial_guess:
            return parameter_values
        calibrated = set(self.calibrated_parameter_names)
        updated = np.asarray(parameter_values, dtype=np.float64).copy()
        for idx, name in enumerate(self.parameter_names):
            if name in calibrated and name in self.default_initial_guess:
                updated[idx] = float(self.default_initial_guess[name])
        return updated

    def _bounds_vector(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full((len(names),), -np.inf, dtype=np.float64)
        upper = np.full((len(names),), np.inf, dtype=np.float64)
        for idx, name in enumerate(names):
            if name not in self.bounds:
                continue
            lower[idx], upper[idx] = self.bounds[name]
        return lower, upper

    def _dynamic_evaluation_args(
        self,
        full_steady_state: np.ndarray,
        parameter_values: np.ndarray,
    ) -> list[float]:
        steady_refs = self._steady_reference_values(full_steady_state)
        full_lookup = dict(zip(self.timings.var, full_steady_state))
        args: list[float] = []
        for name in self.timings.future_not_past_and_mixed:
            args.append(float(full_lookup[name]))
        for name in self.timings.var:
            args.append(float(full_lookup[name]))
        for name in self.timings.past_not_future_and_mixed:
            args.append(float(full_lookup[name]))
        args.extend([0.0] * self.timings.nExo)
        args.extend(float(x) for x in steady_refs)
        args.extend(float(x) for x in parameter_values)
        return args

    def _steady_state_symbol_value_map(
        self,
        base_steady_state: Sequence[float],
        parameter_values: Sequence[float],
    ) -> dict[sp.Symbol, float]:
        values = list(base_steady_state) + list(parameter_values)
        symbols = list(self._steady_state_symbols) + list(self._parameter_symbols)
        return {
            symbol: float(value)
            for symbol, value in zip(symbols, values)
        }

    def _resolve_steady_state_obc_expressions(
        self,
        base_steady_state: Sequence[float],
        parameter_values: Sequence[float],
    ) -> tuple[sp.Expr, ...]:
        symbol_values = self._steady_state_symbol_value_map(
            base_steady_state,
            parameter_values,
        )
        preferred_symbols = frozenset(self._steady_state_symbols)
        return tuple(
            _freeze_obc_expression(
                expr,
                symbol_values=symbol_values,
                preferred_symbols=preferred_symbols,
            )
            for expr in self._steady_state_expressions
        )

    def _evaluate_steady_state_obc_jacobian(
        self,
        base_steady_state: Sequence[float],
        parameter_values: Sequence[float],
    ) -> np.ndarray:
        resolved = self._resolve_steady_state_obc_expressions(
            base_steady_state,
            parameter_values,
        )
        matrix = sp.Matrix(resolved).jacobian(self._steady_state_symbols)
        values = list(base_steady_state) + list(parameter_values)
        symbols = list(self._steady_state_symbols) + list(self._parameter_symbols)
        return _evaluate_symbolic_matrix(matrix, symbols, values)

    def _evaluate_steady_state_obc_parameter_jacobian(
        self,
        base_steady_state: Sequence[float],
        parameter_values: Sequence[float],
    ) -> np.ndarray:
        resolved = self._resolve_steady_state_obc_expressions(
            base_steady_state,
            parameter_values,
        )
        matrix = sp.Matrix(resolved).jacobian(self._parameter_symbols)
        values = list(base_steady_state) + list(parameter_values)
        symbols = list(self._steady_state_symbols) + list(self._parameter_symbols)
        return _evaluate_symbolic_matrix(matrix, symbols, values)

    def _evaluate_joint_steady_state_obc_jacobian(
        self,
        joint_values: Sequence[float],
    ) -> np.ndarray:
        n_steady = len(self._steady_state_symbols)
        base_steady_state = joint_values[:n_steady]
        parameter_values = joint_values[n_steady:]
        resolved = self._resolve_steady_state_obc_expressions(
            base_steady_state,
            parameter_values,
        )
        matrix = sp.Matrix(list(resolved) + list(self._parameter_expressions)).jacobian(
            self._joint_unknown_symbols
        )
        return _evaluate_symbolic_matrix(matrix, self._joint_unknown_symbols, joint_values)

    def _resolve_dynamic_obc_expressions(
        self,
        full_steady_state: Sequence[float],
        parameter_values: Sequence[float],
    ) -> tuple[sp.Expr, ...]:
        evaluation_args = self._dynamic_evaluation_args(
            np.asarray(full_steady_state, dtype=np.float64),
            np.asarray(parameter_values, dtype=np.float64),
        )
        symbol_values = {
            symbol: float(value)
            for symbol, value in zip(self._dynamic_input_symbols, evaluation_args)
        }
        preferred_symbols = frozenset(self._dynamic_symbols)
        return tuple(
            _freeze_obc_expression(
                expr,
                symbol_values=symbol_values,
                preferred_symbols=preferred_symbols,
            )
            for expr in self._dynamic_expressions
        )

    def _evaluate_dynamic_obc_derivatives(
        self,
        full_steady_state: Sequence[float],
        parameter_values: Sequence[float],
        *,
        order: int,
    ) -> np.ndarray:
        full_state = np.asarray(full_steady_state, dtype=np.float64)
        parameters = np.asarray(parameter_values, dtype=np.float64)
        evaluation_args = self._dynamic_evaluation_args(full_state, parameters)
        resolved = self._resolve_dynamic_obc_expressions(full_state, parameters)
        if order == 1:
            matrix = sp.Matrix(resolved).jacobian(self._dynamic_symbols)
        elif order == 2:
            matrix = sp.Matrix(
                [_flatten_hessian(expr, self._dynamic_symbols) for expr in resolved]
            )
        elif order == 3:
            matrix = sp.Matrix(
                [_flatten_third_order(expr, self._dynamic_symbols) for expr in resolved]
            )
        else:
            raise ValueError(f"Unsupported derivative order `{order}`.")
        return _evaluate_symbolic_matrix(
            matrix,
            self._dynamic_input_symbols,
            evaluation_args,
        )

    def _build_sep_subgradient_jacobian(
        self,
        *,
        initial_state: Sequence[float],
        terminal_state: Sequence[float],
        parameter_values: Sequence[float],
        steady_reference_values: Sequence[float],
        config: SEPConfig,
        deterministic_shocks: Optional[jax.Array],
    ) -> callable:
        if config.expectation_method != "gauss_hermite":
            raise ValueError(
                "Parsed-model SEP subgradient Jacobians currently support only "
                "expectation_method='gauss_hermite'."
            )

        state_dim = self.timings.nVars
        initial_state_arr = jnp.asarray(initial_state, dtype=jnp.float64)
        terminal_state_arr = jnp.asarray(terminal_state, dtype=jnp.float64)
        parameter_array = jnp.asarray(parameter_values, dtype=jnp.float64)
        steady_refs = jnp.asarray(steady_reference_values, dtype=jnp.float64)
        if deterministic_shocks is None:
            deterministic = jnp.zeros((config.periods, self.timings.nExo), dtype=jnp.float64)
        else:
            deterministic = jnp.asarray(deterministic_shocks, dtype=jnp.float64)

        rule = (
            _gauss_hermite_sparse_rule(config.nnodes, self.timings.nExo, config.shock_scale)
            if config.sparse_tree
            else gauss_hermite_rule(config.nnodes, self.timings.nExo, config.shock_scale)
        )
        num_nodes = int(rule.weights.shape[0])
        counts = _group_counts(
            config.periods,
            config.branching_order,
            num_nodes,
            sparse_tree=config.sparse_tree,
        )
        time_offsets = [0]
        for t in range(1, config.periods + 1):
            time_offsets.append(time_offsets[-1] + counts[t] * state_dim)

        future_dim = len(self.timings.future_not_past_and_mixed_idx)
        past_dim = len(self.timings.past_not_future_and_mixed_idx)

        def unflatten(stacked: jax.Array) -> tuple[jax.Array, ...]:
            values = []
            for t in range(1, config.periods + 1):
                start = time_offsets[t - 1]
                end = time_offsets[t]
                values.append(jnp.reshape(stacked[start:end], (counts[t], state_dim)))
            return tuple(values)

        def block_start(time_index: int, group_index: int) -> int:
            return time_offsets[time_index - 1] + group_index * state_dim

        def jacobian_fn(stacked: jax.Array) -> jax.Array:
            states_by_time = unflatten(stacked)
            zero_shock = jnp.zeros((self.timings.nExo,), dtype=jnp.float64)
            jacobian = np.zeros((time_offsets[-1], time_offsets[-1]), dtype=np.float64)
            row_cursor = 0
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
                        parent = None
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
                        if t <= stochastic_time_limit and self.timings.nExo > 0
                        else zero_shock
                    )
                    current_shock = deterministic_shock + stochastic_shock
                    row_slice = slice(row_cursor, row_cursor + state_dim)
                    current_start = block_start(t, g)

                    def accumulate_block(
                        *,
                        weight: float,
                        lead_state: jax.Array,
                        lead_group: Optional[int],
                    ) -> None:
                        dynamic_jacobian = np.asarray(
                            self._evaluate_dynamic_jacobian_with_context(
                                prev_state,
                                current_states[g],
                                lead_state,
                                current_shock,
                                parameter_values=parameter_array,
                                steady_reference_values=steady_refs,
                            ),
                            dtype=np.float64,
                        )
                        current_block = dynamic_jacobian[
                            :,
                            future_dim : future_dim + state_dim,
                        ]
                        jacobian[row_slice, current_start : current_start + state_dim] += (
                            weight * current_block
                        )

                        if t > 1 and parent is not None and past_dim > 0:
                            lag_block = dynamic_jacobian[
                                :,
                                future_dim + state_dim : future_dim + state_dim + past_dim,
                            ]
                            prev_start = block_start(t - 1, parent)
                            for local_idx, var_idx in enumerate(
                                self.timings.past_not_future_and_mixed_idx
                            ):
                                jacobian[row_slice, prev_start + var_idx] += (
                                    weight * lag_block[:, local_idx]
                                )

                        if lead_group is not None and future_dim > 0:
                            lead_block = dynamic_jacobian[:, :future_dim]
                            lead_start = block_start(t + 1, lead_group)
                            for local_idx, var_idx in enumerate(
                                self.timings.future_not_past_and_mixed_idx
                            ):
                                jacobian[row_slice, lead_start + var_idx] += (
                                    weight * lead_block[:, local_idx]
                                )

                    if t == config.periods:
                        accumulate_block(weight=1.0, lead_state=terminal_state_arr, lead_group=None)
                    else:
                        child_groups = _child_groups(
                            g,
                            t,
                            config.branching_order,
                            num_nodes,
                            sparse_tree=config.sparse_tree,
                        )
                        if len(child_groups) == 1:
                            child = child_groups[0]
                            accumulate_block(
                                weight=1.0,
                                lead_state=next_states[child],
                                lead_group=child,
                            )
                        else:
                            for local_idx, child in enumerate(child_groups):
                                accumulate_block(
                                    weight=float(rule.weights[local_idx]),
                                    lead_state=next_states[child],
                                    lead_group=child,
                                )

                    row_cursor += state_dim
            return jnp.asarray(jacobian, dtype=jnp.float64)

        return jacobian_fn

    def calculate_jacobian(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        if steady_state is None:
            steady_state_result = self.solve_steady_state(parameter_values=parameter_values)
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        if self.has_obc:
            values = self._evaluate_dynamic_obc_derivatives(
                full_steady_state,
                resolved_parameters,
                order=1,
            )
            return jnp.asarray(values, dtype=jnp.float64)
        values = np.asarray(
            self._dynamic_jacobian_fn(
                *self._dynamic_evaluation_args(full_steady_state, resolved_parameters)
            ),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def calculate_hessian(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        if steady_state is None:
            steady_state_result = self.solve_steady_state(parameter_values=parameter_values)
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        if self.has_obc:
            values = self._evaluate_dynamic_obc_derivatives(
                full_steady_state,
                resolved_parameters,
                order=2,
            )
            return jnp.asarray(values, dtype=jnp.float64)
        values = np.asarray(
            self._dynamic_hessian_fn(
                *self._dynamic_evaluation_args(full_steady_state, resolved_parameters)
            ),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def calculate_third_order_derivatives(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        if steady_state is None:
            steady_state_result = self.solve_steady_state(parameter_values=parameter_values)
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        if self.has_obc:
            values = self._evaluate_dynamic_obc_derivatives(
                full_steady_state,
                resolved_parameters,
                order=3,
            )
            return jnp.asarray(values, dtype=jnp.float64)
        values = np.asarray(
            self._dynamic_third_order_fn(
                *self._dynamic_evaluation_args(full_steady_state, resolved_parameters)
            ),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def solve_stochastic_extended_path(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
        deterministic_shocks: Optional[
            Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
        ] = None,
        initial_guess: Optional[Sequence[Sequence[float]]] = None,
    ) -> ParsedModelSEPResult:
        if len(self._dynamic_expressions) != self.timings.nVars:
            raise ValueError(
                "SEP solution requires as many dynamic equations as present variables. "
                f"Got {len(self._dynamic_expressions)} equations and {self.timings.nVars} variables."
            )

        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            full_steady_state = np.asarray(
                steady_state_result.steady_state,
                dtype=np.float64,
            )
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )

        initial_state_values = (
            jnp.asarray(full_steady_state, dtype=jnp.float64)
            if initial_state is None
            else self._coerce_dynamic_state_vector(
                initial_state,
                label="initial_state",
            )
        )
        terminal_state_values = jnp.asarray(
            self._coerce_full_steady_state(
                terminal_state,
                parameter_values=resolved_parameters,
            )
            if terminal_state is not None
            else full_steady_state,
            dtype=jnp.float64,
        )
        deterministic_shock_values = self._coerce_sep_deterministic_shocks(
            deterministic_shocks,
            periods=config.periods,
        )
        sep_result, _ = self._solve_stochastic_extended_path_with_obc_enforcement(
            full_steady_state=np.asarray(full_steady_state, dtype=np.float64),
            parameter_values=np.asarray(resolved_parameters, dtype=np.float64),
            initial_state=initial_state_values,
            terminal_state=terminal_state_values,
            config=config,
            deterministic_shocks=deterministic_shock_values,
            initial_guess=initial_guess,
        )
        return sep_result

    def solve_first_order(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
    ) -> ParsedModelFirstOrderResult:
        if len(self._dynamic_expressions) != self.timings.nVars:
            raise ValueError(
                "First-order solution requires as many dynamic equations as present variables. "
                f"Got {len(self._dynamic_expressions)} equations and {self.timings.nVars} variables."
            )
        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        jacobian = self.calculate_jacobian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        solution = solve_first_order_dsge_solution(
            jacobian,
            self.timings,
            qme_algorithm=qme_algorithm,
        )
        return ParsedModelFirstOrderResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            jacobian=jacobian,
            solution=solution,
        )

    def analyze_first_order_determinacy(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_acceptance_tol: float = 1e-8,
    ) -> ParsedModelFirstOrderDeterminacyResult:
        if len(self._dynamic_expressions) != self.timings.nVars:
            raise ValueError(
                "First-order determinacy analysis requires as many dynamic equations as "
                f"present variables. Got {len(self._dynamic_expressions)} equations and "
                f"{self.timings.nVars} variables."
            )
        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        jacobian = self.calculate_jacobian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        determinacy = analyze_first_order_dsge_determinacy(
            jacobian,
            self.timings,
            qme_acceptance_tol=qme_acceptance_tol,
        )
        return ParsedModelFirstOrderDeterminacyResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            jacobian=jacobian,
            determinacy=determinacy,
        )

    def solve_second_order(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        pruning: bool = False,
        sylvester_algorithm: str = "doubling",
        sylvester_tol: float = 1e-14,
        sylvester_acceptance_tol: float = 1e-10,
        sylvester_max_iter: int = 500,
        stochastic_steady_state_tol: float = 1e-14,
        stochastic_steady_state_max_iter: int = 100,
    ) -> ParsedModelSecondOrderResult:
        if len(self._dynamic_expressions) != self.timings.nVars:
            raise ValueError(
                "Second-order solution requires as many dynamic equations as present variables. "
                f"Got {len(self._dynamic_expressions)} equations and {self.timings.nVars} variables."
            )
        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        jacobian = self.calculate_jacobian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        hessian = self.calculate_hessian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        first_order_solution = solve_first_order_dsge_solution(
            jacobian,
            self.timings,
            qme_algorithm=qme_algorithm,
        )
        second_order_solution = solve_second_order_dsge_solution(
            jacobian,
            hessian,
            first_order_solution,
            self.timings,
            sylvester_algorithm=sylvester_algorithm,
            sylvester_tol=sylvester_tol,
            sylvester_acceptance_tol=sylvester_acceptance_tol,
            sylvester_max_iter=sylvester_max_iter,
        )
        stochastic_steady_state = solve_second_order_stochastic_steady_state(
            first_order_solution,
            second_order_solution,
            self.timings,
            pruning=pruning,
            tol=stochastic_steady_state_tol,
            max_iter=stochastic_steady_state_max_iter,
        )
        return ParsedModelSecondOrderResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            jacobian=jacobian,
            hessian=hessian,
            first_order_solution=first_order_solution,
            second_order_solution=second_order_solution,
            stochastic_steady_state=stochastic_steady_state,
        )

    def solve_third_order(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
        qme_algorithm: str = "schur",
        pruning: bool = False,
        sylvester_algorithm: str = "doubling",
        sylvester_tol: float = 1e-14,
        sylvester_acceptance_tol: float = 1e-10,
        sylvester_max_iter: int = 500,
        stochastic_steady_state_tol: float = 1e-14,
        stochastic_steady_state_max_iter: int = 100,
    ) -> ParsedModelThirdOrderResult:
        if len(self._dynamic_expressions) != self.timings.nVars:
            raise ValueError(
                "Third-order solution requires as many dynamic equations as present variables. "
                f"Got {len(self._dynamic_expressions)} equations and {self.timings.nVars} variables."
            )
        if steady_state is None:
            steady_state_result = self.solve_steady_state(
                parameter_values=parameter_values,
                initial_guess=steady_state_initial_guess,
                tol=steady_state_tol,
                max_iter=steady_state_max_iter,
            )
            full_steady_state = np.asarray(steady_state_result.steady_state, dtype=np.float64)
            resolved_parameters = np.asarray(
                steady_state_result.parameter_values,
                dtype=np.float64,
            )
        else:
            full_steady_state = self._coerce_full_steady_state(steady_state)
            resolved_parameters = (
                self._coerce_parameter_values(parameter_values)
                if parameter_values is not None
                else np.asarray(
                    self.resolve_parameter_values(steady_state=full_steady_state),
                    dtype=np.float64,
                )
            )
        jacobian = self.calculate_jacobian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        hessian = self.calculate_hessian(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        third_order_derivatives = self.calculate_third_order_derivatives(
            parameter_values=resolved_parameters,
            steady_state=full_steady_state,
        )
        first_order_solution = solve_first_order_dsge_solution(
            jacobian,
            self.timings,
            qme_algorithm=qme_algorithm,
        )
        second_order_solution = solve_second_order_dsge_solution(
            jacobian,
            hessian,
            first_order_solution,
            self.timings,
            sylvester_algorithm=sylvester_algorithm,
            sylvester_tol=sylvester_tol,
            sylvester_acceptance_tol=sylvester_acceptance_tol,
            sylvester_max_iter=sylvester_max_iter,
        )
        third_order_solution = solve_third_order_dsge_solution(
            jacobian,
            hessian,
            third_order_derivatives,
            first_order_solution,
            second_order_solution,
            self.timings,
            sylvester_algorithm=sylvester_algorithm,
            sylvester_tol=sylvester_tol,
            sylvester_acceptance_tol=sylvester_acceptance_tol,
            sylvester_max_iter=sylvester_max_iter,
        )
        stochastic_steady_state = solve_third_order_stochastic_steady_state(
            first_order_solution,
            second_order_solution,
            third_order_solution,
            self.timings,
            pruning=pruning,
            tol=stochastic_steady_state_tol,
            max_iter=stochastic_steady_state_max_iter,
        )
        return ParsedModelThirdOrderResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            jacobian=jacobian,
            hessian=hessian,
            third_order_derivatives=third_order_derivatives,
            first_order_solution=first_order_solution,
            second_order_solution=second_order_solution,
            third_order_solution=third_order_solution,
            stochastic_steady_state=stochastic_steady_state,
        )


def parse_macro_model(source: str) -> MacroModel:
    model_block = _extract_block(_MODEL_BLOCK_RE, source, "model")
    parameter_block = _extract_block(_PARAMETERS_BLOCK_RE, source, "parameters")
    if parameter_block["name"] != model_block["name"]:
        raise ValueError(
            "The `@parameters` block must target the same model name as `@model`."
        )
    model_options = _parse_block_options(model_block["options"])
    raw_default_guess = _parse_parameter_block_guess(parameter_block["options"])
    parameter_options = _parse_block_options(parameter_block["options"])
    loop_collections = _parse_source_loop_collections(
        source,
        (
            (model_block["start"], model_block["end"]),
            (parameter_block["start"], parameter_block["end"]),
        ),
    )

    equations = tuple(
        _split_model_body_lines(
            model_block["body"],
            loop_collections=loop_collections,
        )
    )
    if not equations:
        raise ValueError("Model block does not contain any equations.")

    dynamic_texts: list[str] = []
    steady_state_texts: list[str] = []
    timed_symbols: dict[str, sp.Symbol] = {}
    timed_metadata: dict[str, tuple[str, str, int]] = {}
    exogenous_names: set[str] = set()
    aux_equations_added: set[tuple[str, int]] = set()
    shifted_shocks_added: set[str] = set()

    for equation in equations:
        expr_text = _equation_to_difference(equation)
        transformed = _transform_dynamic_expression(
            expr_text,
            dynamic_texts,
            timed_symbols,
            timed_metadata,
            exogenous_names,
            aux_equations_added,
            shifted_shocks_added,
        )
        dynamic_texts.append(transformed)
        steady_state_texts.append(
            _transform_steady_state_expression(
                expr_text,
                timed_symbols,
                timed_metadata,
                exogenous_names,
            )
        )

    available_indexed_names = _available_indexed_parameter_names(
        dynamic_texts + steady_state_texts,
        timed_symbols,
        timed_metadata,
    )
    parsed_parameter_block = _parse_parameter_block(
        parameter_block["body"],
        timed_symbols,
        timed_metadata,
        exogenous_names,
        available_indexed_names,
    )
    timings = _build_timings(timed_metadata)
    parameter_names = tuple(
        sorted(
            _extract_parameter_names(
                dynamic_texts
                + steady_state_texts
                + list(parsed_parameter_block.equation_texts),
                timed_symbols,
            )
        )
    )
    missing = [
        name
        for name in parameter_names
        if name not in parsed_parameter_block.target_names
    ]
    if missing:
        raise ValueError(
            "Missing parameter assignments for: " + ", ".join(sorted(missing))
        )
    parameter_values = jnp.asarray(
        [parsed_parameter_block.initial_values.get(name, 1.0) for name in parameter_names],
        dtype=jnp.float64,
    )

    parameter_parse_names = tuple(
        _parameter_parse_name(name) for name in parameter_names
    )
    parameter_name_map = dict(zip(parameter_names, parameter_parse_names))
    parameter_symbols = tuple(
        sp.Symbol(name, real=True) for name in parameter_parse_names
    )
    parameter_symbol_map = dict(zip(parameter_parse_names, parameter_symbols))
    unknown_function_names = _collect_unknown_function_names(
        dynamic_texts + steady_state_texts + list(parsed_parameter_block.equation_texts),
        timed_symbols,
        parameter_names,
    )
    parse_locals = {
        **timed_symbols,
        **parameter_symbol_map,
        **{name: sp.Function(name) for name in unknown_function_names},
        **_function_locals(),
    }

    dynamic_exprs = tuple(
        parse_expr(
            _substitute_parameter_identifiers(text, parameter_name_map),
            local_dict=parse_locals,
            transformations=_TRANSFORMATIONS,
        )
        for text in dynamic_texts
    )
    steady_state_exprs = tuple(
        parse_expr(
            _substitute_parameter_identifiers(text, parameter_name_map),
            local_dict=parse_locals,
            transformations=_TRANSFORMATIONS,
        )
        for text in steady_state_texts
    )
    parameter_exprs = tuple(
        parse_expr(
            _substitute_parameter_identifiers(text, parameter_name_map),
            local_dict=parse_locals,
            transformations=_TRANSFORMATIONS,
        )
        for text in parsed_parameter_block.equation_texts
    )

    steady_state_names = tuple(
        name
        for name in timings.var
        if name not in timings.aux and name not in timings.exo_present
    )
    steady_state_symbols = tuple(
        timed_symbols[_steady_state_token_name(name)] for name in steady_state_names
    )
    steady_state_reference_names = tuple(
        name for name, kind, _ in _iter_metadata_by_name(timed_metadata) if kind == "steady"
    )
    dynamic_symbol_names = (
        timings.future_not_past_and_mixed
        + timings.var
        + timings.past_not_future_and_mixed
        + timings.exo
    )
    dynamic_symbols = tuple(
        [timed_symbols[_time_token_name(name, 1)] for name in timings.future_not_past_and_mixed]
        + [timed_symbols[_time_token_name(name, 0)] for name in timings.var]
        + [timed_symbols[_time_token_name(name, -1)] for name in timings.past_not_future_and_mixed]
        + [timed_symbols[_exo_token_name(name)] for name in timings.exo]
    )
    dynamic_input_symbols = tuple(
        list(dynamic_symbols)
        + [timed_symbols[_steady_state_token_name(name)] for name in steady_state_reference_names]
        + list(parameter_symbols)
    )
    parameter_equations_depend_on_steady_state = any(
        bool(expr.free_symbols & set(steady_state_symbols))
        for expr in parameter_exprs
    )
    default_initial_guess = _expand_default_initial_guess(
        raw_default_guess,
        steady_state_names=steady_state_names,
        calibrated_parameter_names=parsed_parameter_block.calibrated_target_names,
    )
    has_obc = any(_expression_contains_obc(expr) for expr in dynamic_exprs)
    max_obc_horizon = int(model_options.get("max_obc_horizon", 0) or 0)
    if max_obc_horizon < 0:
        raise ValueError(
            f"`max_obc_horizon` must be non-negative, got {max_obc_horizon}."
        )

    model = MacroModel(
        name=model_block["name"],
        equations=equations,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        calibrated_parameter_names=tuple(
            sorted(parsed_parameter_block.calibrated_target_names)
        ),
        default_initial_guess=default_initial_guess,
        bounds=parsed_parameter_block.bounds,
        model_options=model_options,
        parameter_options=parameter_options,
        max_obc_horizon=max_obc_horizon,
        has_obc=has_obc,
        timings=timings,
        steady_state_names=steady_state_names,
        steady_state_reference_names=steady_state_reference_names,
        dynamic_symbol_names=dynamic_symbol_names,
        _dynamic_expressions=dynamic_exprs,
        _steady_state_expressions=steady_state_exprs,
        _parameter_expressions=parameter_exprs,
        _parameter_symbols=parameter_symbols,
        _steady_state_symbols=steady_state_symbols,
        _dynamic_symbols=dynamic_symbols,
        _dynamic_input_symbols=dynamic_input_symbols,
        _parameter_equations_depend_on_steady_state=parameter_equations_depend_on_steady_state,
    )
    if _should_precompile_parsed_model(model_options, parameter_options):
        _precompile_parsed_model(model)
    return model


def _should_precompile_parsed_model(
    model_options: Mapping[str, object],
    parameter_options: Mapping[str, object],
) -> bool:
    return bool(
        _coerce_optional_python_bool(model_options.get("precompile"))
        or _coerce_optional_python_bool(parameter_options.get("precompile"))
    )


def _precompile_parsed_model(model: MacroModel) -> None:
    _ = model._steady_state_matrix
    _ = model._steady_state_jacobian
    _ = model._steady_state_parameter_jacobian
    _ = model._steady_state_fn
    _ = model._steady_state_jacobian_fn
    _ = model._steady_state_parameter_jacobian_fn
    _ = model._steady_state_residual_jax_fn
    _ = model._parameter_matrix
    _ = model._parameter_equation_jacobian
    _ = model._parameter_equation_fn
    _ = model._parameter_equation_jacobian_fn
    _ = model._parameter_equation_residual_jax_fn
    if model._parameter_equations_depend_on_steady_state:
        _ = model._joint_unknown_symbols
        _ = model._joint_steady_state_matrix
        _ = model._joint_steady_state_jacobian
        _ = model._joint_steady_state_fn
        _ = model._joint_steady_state_jacobian_fn
        _ = model._joint_steady_state_residual_jax_fn
    _ = model._dynamic_matrix
    _ = model._dynamic_jacobian
    _ = model._dynamic_residual_fn
    _ = model._dynamic_jacobian_fn
    perturbation_order = int(model.parameter_options.get("perturbation_order", 1) or 1)
    if perturbation_order >= 2:
        _ = model._dynamic_hessian
        _ = model._dynamic_hessian_fn
    if perturbation_order >= 3:
        _ = model._dynamic_third
        _ = model._dynamic_third_order_fn
    if bool(model.parameter_options.get("symbolic", False)):
        _ = model._symbolic_steady_state_seed_entries
        _ = model._symbolic_steady_state_seed_indices
        _ = model._symbolic_steady_state_seed_matrix
        _ = model._symbolic_steady_state_seed_fn
        _ = model._symbolic_steady_state_seed_jax_fn
    if model.has_obc:
        _ = model._obc_violation_expressions
        _ = model._obc_violation_matrix
        _ = model._first_order_obc_projection_specs


def solve_steady_state(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> SteadyStateResult:
    return model.solve_steady_state(
        parameter_values=parameter_values,
        initial_guess=initial_guess,
        tol=tol,
        max_iter=max_iter,
    )


def solve_steady_state_jax(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
    line_search_min_step: float = 2.0**-16,
) -> SteadyStateResult:
    return model.solve_steady_state_jax(
        parameter_values=parameter_values,
        initial_guess=initial_guess,
        tol=tol,
        max_iter=max_iter,
        line_search_min_step=line_search_min_step,
    )


def resolve_parameter_values(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> jax.Array:
    return model.resolve_parameter_values(
        parameter_values=parameter_values,
        steady_state=steady_state,
        tol=tol,
        max_iter=max_iter,
    )


def resolve_parameter_values_jax(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
    line_search_min_step: float = 2.0**-16,
) -> jax.Array:
    return model.resolve_parameter_values_jax(
        parameter_values=parameter_values,
        steady_state=steady_state,
        tol=tol,
        max_iter=max_iter,
        line_search_min_step=line_search_min_step,
    )


def calculate_jacobian(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.calculate_jacobian(
        parameter_values=parameter_values,
        steady_state=steady_state,
    )


def calculate_hessian(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.calculate_hessian(
        parameter_values=parameter_values,
        steady_state=steady_state,
    )


def calculate_third_order_derivatives(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.calculate_third_order_derivatives(
        parameter_values=parameter_values,
        steady_state=steady_state,
    )


def evaluate_dynamic_residual(
    model: MacroModel,
    lag_state: Sequence[float],
    current_state: Sequence[float],
    lead_state: Sequence[float],
    *,
    shock: Optional[Sequence[float]] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.evaluate_dynamic_residual(
        lag_state,
        current_state,
        lead_state,
        shock=shock,
        parameter_values=parameter_values,
        steady_state=steady_state,
    )


def evaluate_obc_violations(
    model: MacroModel,
    lag_state: Sequence[float],
    current_state: Sequence[float],
    lead_state: Sequence[float],
    *,
    shock: Optional[Sequence[float]] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.evaluate_obc_violations(
        lag_state,
        current_state,
        lead_state,
        shock=shock,
        parameter_values=parameter_values,
        steady_state=steady_state,
    )


def evaluate_obc_violations_along_path(
    model: MacroModel,
    state_path: Sequence[Sequence[float]],
    *,
    shocks: Optional[Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
) -> jax.Array:
    return model.evaluate_obc_violations_along_path(
        state_path,
        shocks=shocks,
        parameter_values=parameter_values,
        steady_state=steady_state,
        terminal_state=terminal_state,
    )


def simulate_model(
    model: MacroModel,
    *,
    periods: int,
    shocks: Optional[str | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
    variables: Optional[Sequence[str] | str] = None,
    shock_size: float = 1.0,
    random_seed: Optional[int] = None,
    algorithm: str = "first_order",
    ignore_obc: bool = False,
    levels: bool = True,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
) -> ModelSimulationResult:
    return model.simulate(
        periods=periods,
        shocks=shocks,
        variables=variables,
        shock_size=shock_size,
        random_seed=random_seed,
        algorithm=algorithm,
        ignore_obc=ignore_obc,
        levels=levels,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
    )


def get_irf(
    model: MacroModel,
    *,
    periods: int,
    variables: Optional[Sequence[str] | str] = None,
    shocks: Optional[
        str | Sequence[str] | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
    ] = "all",
    shock_size: float = 1.0,
    negative_shock: bool = False,
    random_seed: Optional[int] = None,
    algorithm: str = "first_order",
    ignore_obc: bool = False,
    levels: bool = False,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
) -> ModelIRFResult:
    return model.get_irf(
        periods=periods,
        variables=variables,
        shocks=shocks,
        shock_size=shock_size,
        negative_shock=negative_shock,
        random_seed=random_seed,
        algorithm=algorithm,
        ignore_obc=ignore_obc,
        levels=levels,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
    )


def resolve_observable_indices(
    model: MacroModel,
    observables: Sequence[str] | str,
) -> tuple[int, ...]:
    return model.resolve_observable_indices(observables)


def build_linear_state_space_from_model(
    model: MacroModel,
    observables: Sequence[str] | str,
    *,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
) -> LinearGaussianStateSpace:
    return model.build_linear_state_space(
        observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
    )


def kalman_loglikelihood_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    return model.kalman_loglikelihood(
        observations,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )


def kalman_loglikelihood_per_period_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    return model.kalman_loglikelihood_per_period(
        observations,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )


def inversion_loglikelihood_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    algorithm: str = "first_order",
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    warmup_iterations: int = 0,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    return model.inversion_loglikelihood(
        observations,
        observables=observables,
        algorithm=algorithm,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        warmup_iterations=warmup_iterations,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )


def inversion_loglikelihood_per_period_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    algorithm: str = "first_order",
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    warmup_iterations: int = 0,
    presample_periods: int = 0,
    on_failure_loglikelihood: float = -np.inf,
) -> jax.Array:
    return model.inversion_loglikelihood_per_period(
        observations,
        observables=observables,
        algorithm=algorithm,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        warmup_iterations=warmup_iterations,
        presample_periods=presample_periods,
        on_failure_loglikelihood=on_failure_loglikelihood,
    )


def switching_loglikelihood_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "stochastic_extended_path",
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
) -> SwitchingLikelihoodResult:
    return model.switching_loglikelihood(
        observations,
        observables=observables,
        gate_probs=gate_probs,
        hard_mask=hard_mask,
        fom_algorithm=fom_algorithm,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        switching_config=switching_config,
    )


def switching_pipeline_report_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "stochastic_extended_path",
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
    gate_gain_tol: float = 0.0,
    gate_hard_threshold: float = 0.5,
    budget_frontier_budgets: Optional[Sequence[int]] = None,
    budget_frontier_points: int = 11,
    benchmark_reps: int = 0,
) -> dict[str, Any]:
    return model.switching_pipeline_report(
        observations,
        observables=observables,
        gate_probs=gate_probs,
        hard_mask=hard_mask,
        fom_algorithm=fom_algorithm,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        switching_config=switching_config,
        gate_gain_tol=gate_gain_tol,
        gate_hard_threshold=gate_hard_threshold,
        budget_frontier_budgets=budget_frontier_budgets,
        budget_frontier_points=budget_frontier_points,
        benchmark_reps=benchmark_reps,
    )


def likelihood_surface_report_from_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    parameter_draws: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    base_parameter_values: Optional[Sequence[float] | Mapping[str, float]] = None,
    observables: Optional[Sequence[str] | str] = None,
    gate_probs: Optional[Sequence[float]] = None,
    hard_mask: Optional[Sequence[bool]] = None,
    fom_algorithm: str = "stochastic_extended_path",
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    initial_covariance_strategy: str = "theoretical",
    measurement_error_scale: float = 1e-9,
    measurement_error_covariance: Optional[Sequence[Sequence[float]]] = None,
    presample_periods: int = 0,
    jitter: float = 1e-9,
    on_failure_loglikelihood: float = -np.inf,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: Optional[bool] = None,
    sep_maxit: Optional[int] = None,
    sep_tol: Optional[float] = None,
    sep_accept_tol: float = 1e-3,
    sep_shock_scale: Optional[float] = None,
    sep_inv_maxit: int = 8,
    sep_inv_step_tol: float = 1e-6,
    sep_inv_resid_tol: float = 1e-6,
    sep_inv_lambda: float = 1e-4,
    switching_config: SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(),
    top_share: float = 0.1,
) -> dict[str, Any]:
    return model.likelihood_surface_report(
        observations,
        parameter_draws,
        base_parameter_values=base_parameter_values,
        observables=observables,
        gate_probs=gate_probs,
        hard_mask=hard_mask,
        fom_algorithm=fom_algorithm,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
        initial_covariance_strategy=initial_covariance_strategy,
        measurement_error_scale=measurement_error_scale,
        measurement_error_covariance=measurement_error_covariance,
        presample_periods=presample_periods,
        jitter=jitter,
        on_failure_loglikelihood=on_failure_loglikelihood,
        config=config,
        sep_periods=sep_periods,
        sep_order=sep_order,
        sep_nnodes=sep_nnodes,
        sep_sparse_tree=sep_sparse_tree,
        sep_maxit=sep_maxit,
        sep_tol=sep_tol,
        sep_accept_tol=sep_accept_tol,
        sep_shock_scale=sep_shock_scale,
        sep_inv_maxit=sep_inv_maxit,
        sep_inv_step_tol=sep_inv_step_tol,
        sep_inv_resid_tol=sep_inv_resid_tol,
        sep_inv_lambda=sep_inv_lambda,
        switching_config=switching_config,
        top_share=top_share,
    )


def compute_linear_gate_stats_from_shocks_model(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    shock_norm: str = "l2",
    error_norm: str = "l2",
) -> LinearGateStatsResult:
    return model.compute_linear_gate_stats_from_shocks(
        observations,
        shocks,
        obs_sigma,
        shock_sigmas,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        shock_norm=shock_norm,
        error_norm=error_norm,
    )


def compute_linear_gate_stats_from_shocks(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    shock_norm: str = "l2",
    error_norm: str = "l2",
) -> LinearGateStatsResult:
    return model.compute_linear_gate_stats_from_shocks(
        observations,
        shocks,
        obs_sigma,
        shock_sigmas,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        shock_norm=shock_norm,
        error_norm=error_norm,
    )


def compute_first_order_obc_violation_path(
    model: MacroModel,
    shocks: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
) -> OBCViolationPathResult:
    return model.compute_first_order_obc_violation_path(
        shocks,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        initial_state=initial_state,
        terminal_state=terminal_state,
    )


def estimate_observed_shocks_matrix(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    data_in_levels: bool = True,
    smooth: bool = False,
    verbose: bool = False,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    label: str = "Estimated shocks",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
) -> jax.Array:
    return model.estimate_observed_shocks_matrix(
        observations,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=data_in_levels,
        smooth=smooth,
        verbose=verbose,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        label=label,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
    )


def estimate_observed_variables_matrix(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    data_in_levels: bool = True,
    levels: bool = True,
    smooth: bool = False,
    verbose: bool = False,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    label: str = "Estimated variables",
    initial_covariance_strategy: str = "theoretical",
    jitter: float = 1e-9,
) -> tuple[jax.Array, tuple[str, ...]]:
    return model.estimate_observed_variables_matrix(
        observations,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        data_in_levels=data_in_levels,
        levels=levels,
        smooth=smooth,
        verbose=verbose,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        label=label,
        initial_covariance_strategy=initial_covariance_strategy,
        jitter=jitter,
    )


def linear_filter_initial_state(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    state_names: Sequence[str] | str,
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    label: str = "Linear filter variables",
) -> jax.Array:
    return model.linear_filter_initial_state(
        observations,
        state_names,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        smooth=smooth,
        label=label,
    )


def linear_filter_full_state_initial(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    observables: Optional[Sequence[str] | str] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    label: str = "Linear filter variables",
) -> jax.Array:
    return model.linear_filter_full_state_initial(
        observations,
        observables=observables,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        smooth=smooth,
        label=label,
    )


def compute_linear_gate_stats_from_filter(
    model: MacroModel,
    observations: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    obs_sigma: Sequence[float] | Mapping[str, float],
    shock_sigmas: Sequence[float] | Mapping[str, float],
    state_names: Optional[Sequence[str] | str] = None,
    *,
    observables: Optional[Sequence[str] | str] = None,
    periods: Optional[int] = None,
    first_order_result: Optional[ParsedModelFirstOrderResult] = None,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    filter: str = "kalman",
    algorithm: str = "first_order",
    smooth: bool = False,
    shock_norm: str = "l2",
    error_norm: str = "l2",
    label: str = "Linear gate stats",
) -> LinearGateStatsResult:
    return model.compute_linear_gate_stats_from_filter(
        observations,
        obs_sigma,
        shock_sigmas,
        state_names,
        observables=observables,
        periods=periods,
        first_order_result=first_order_result,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        filter=filter,
        algorithm=algorithm,
        smooth=smooth,
        shock_norm=shock_norm,
        error_norm=error_norm,
        label=label,
    )


def solve_first_order_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
) -> ParsedModelFirstOrderResult:
    return model.solve_first_order(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
    )


def analyze_first_order_model_determinacy(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_acceptance_tol: float = 1e-8,
) -> ParsedModelFirstOrderDeterminacyResult:
    return model.analyze_first_order_determinacy(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_acceptance_tol=qme_acceptance_tol,
    )


def solve_stochastic_extended_path_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    deterministic_shocks: Optional[
        Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
    ] = None,
    initial_guess: Optional[Sequence[Sequence[float]]] = None,
) -> ParsedModelSEPResult:
    return model.solve_stochastic_extended_path(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        deterministic_shocks=deterministic_shocks,
        initial_guess=initial_guess,
    )


def solve_sep_at_noise_level(
    model: MacroModel,
    *,
    sigma: float = 1.0,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    deterministic_shocks: Optional[
        Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
    ] = None,
    initial_guess: Optional[Sequence[Sequence[float]]] = None,
) -> ParsedModelSEPResult:
    if not np.isfinite(float(sigma)) or sigma < 0.0:
        raise ValueError(f"sigma must be a finite non-negative float, got {sigma!r}.")
    effective_config = (
        dataclass_replace(config, branching_order=0) if sigma < 1e-10 else config
    )
    scaled_deterministic_shocks = None
    if deterministic_shocks is not None:
        coerced_shocks = np.asarray(
            model._coerce_sep_deterministic_shocks(
                deterministic_shocks,
                periods=effective_config.periods,
            ),
            dtype=np.float64,
        )
        scaled_deterministic_shocks = float(sigma) * coerced_shocks
    return model.solve_stochastic_extended_path(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=effective_config,
        deterministic_shocks=scaled_deterministic_shocks,
        initial_guess=initial_guess,
    )


def homotopy_sep(
    model: MacroModel,
    *,
    n_steps: int = 10,
    adaptive: bool = True,
    max_retries: int = 3,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    deterministic_shocks: Optional[
        Sequence[Sequence[float]] | Mapping[str, Sequence[float]]
    ] = None,
    initial_guess: Optional[Sequence[Sequence[float]]] = None,
) -> HomotopySEPResult:
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")

    sigma_targets = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
    sigma_actual = [0.0]

    result_prev = solve_sep_at_noise_level(
        model,
        sigma=0.0,
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        initial_state=initial_state,
        terminal_state=terminal_state,
        config=config,
        deterministic_shocks=deterministic_shocks,
        initial_guess=initial_guess,
    )
    if not result_prev.solution.accepted:
        return HomotopySEPResult(
            success=False,
            result=result_prev,
            sigma_path=tuple(sigma_actual),
        )

    for sigma_target in sigma_targets[1:]:
        sigma_target_value = float(sigma_target)
        sigma_prev = float(sigma_actual[-1])
        sigma_current = sigma_target_value
        retry_count = 0
        while True:
            warm_state = np.asarray(result_prev.solution.mean_path[:, 0], dtype=np.float64)
            if result_prev.solution.mean_path.shape[1] >= 2:
                warm_state = np.asarray(
                    result_prev.solution.mean_path[:, 1],
                    dtype=np.float64,
                )

            result_current = solve_sep_at_noise_level(
                model,
                sigma=sigma_current,
                parameter_values=parameter_values,
                steady_state=steady_state,
                steady_state_initial_guess=steady_state_initial_guess,
                steady_state_tol=steady_state_tol,
                steady_state_max_iter=steady_state_max_iter,
                initial_state=warm_state,
                terminal_state=terminal_state,
                config=config,
                deterministic_shocks=deterministic_shocks,
                initial_guess=initial_guess,
            )

            if result_current.solution.accepted:
                result_prev = result_current
                sigma_actual.append(sigma_current)
                if adaptive and sigma_current + 1e-12 < sigma_target_value:
                    sigma_prev = sigma_current
                    sigma_current = sigma_target_value
                    retry_count = 0
                    continue
                break

            if adaptive and retry_count < max_retries:
                sigma_current = 0.5 * (sigma_prev + sigma_current)
                retry_count += 1
                continue

            return HomotopySEPResult(
                success=False,
                result=result_prev,
                sigma_path=tuple(float(value) for value in sigma_actual),
            )

    return HomotopySEPResult(
        success=True,
        result=result_prev,
        sigma_path=tuple(float(value) for value in sigma_actual),
    )


def homotopy_chained_trajectory(
    model: MacroModel,
    *,
    periods: int,
    n_steps: int = 10,
    adaptive: bool = True,
    max_retries: int = 3,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    shocks: Optional[str | Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
    shock_size: float = 1.0,
    random_seed: Optional[int] = None,
    config: SEPConfig = SEPConfig(),
) -> HomotopyChainedTrajectoryResult:
    if periods < 1:
        raise ValueError(f"periods must be positive, got {periods}.")

    full_steady_state, resolved_parameters = (
        model._prepare_steady_state_and_parameters_for_runtime(
            parameter_values=parameter_values,
            steady_state=steady_state,
            steady_state_initial_guess=steady_state_initial_guess,
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
        )
    )
    if full_steady_state is None or resolved_parameters is None:
        raise ValueError(
            "Could not prepare a converged steady state for homotopy trajectory simulation."
        )

    shock_matrix = model._coerce_simulation_shocks(
        shocks,
        periods=periods,
        shock_size=shock_size,
        random_seed=random_seed,
    )
    initial_state_values = (
        np.asarray(full_steady_state, dtype=np.float64)
        if initial_state is None
        else np.asarray(
            model._coerce_dynamic_state_vector(initial_state, label="initial_state"),
            dtype=np.float64,
        )
    )

    trajectory = np.zeros((model.timings.nVars, periods + 1), dtype=np.float64)
    trajectory[:, 0] = initial_state_values
    sigma_paths: list[tuple[float, ...]] = []
    periods_completed = 0

    for period_idx in range(periods):
        period_shocks = np.zeros((model.timings.nExo, config.periods), dtype=np.float64)
        if model.timings.nExo > 0:
            period_shocks[:, 0] = shock_matrix[:, period_idx]

        result = homotopy_sep(
            model,
            n_steps=n_steps,
            adaptive=adaptive,
            max_retries=max_retries,
            parameter_values=np.asarray(resolved_parameters, dtype=np.float64),
            steady_state=np.asarray(full_steady_state, dtype=np.float64),
            steady_state_tol=steady_state_tol,
            steady_state_max_iter=steady_state_max_iter,
            initial_state=trajectory[:, period_idx],
            terminal_state=terminal_state,
            config=config,
            deterministic_shocks=period_shocks,
        )
        sigma_paths.append(result.sigma_path)
        if not result.success:
            return HomotopyChainedTrajectoryResult(
                trajectory=jnp.asarray(trajectory, dtype=jnp.float64),
                shocks=jnp.asarray(shock_matrix, dtype=jnp.float64),
                success=False,
                periods_completed=period_idx,
                sigma_paths=tuple(sigma_paths),
                steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
                parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            )

        trajectory[:, period_idx + 1] = np.asarray(
            result.result.solution.mean_path[:, 1],
            dtype=np.float64,
        )
        periods_completed = period_idx + 1

    return HomotopyChainedTrajectoryResult(
        trajectory=jnp.asarray(trajectory, dtype=jnp.float64),
        shocks=jnp.asarray(shock_matrix, dtype=jnp.float64),
        success=True,
        periods_completed=periods_completed,
        sigma_paths=tuple(sigma_paths),
        steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
        parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
    )


def solve_second_order_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    pruning: bool = False,
    sylvester_algorithm: str = "doubling",
    sylvester_tol: float = 1e-14,
    sylvester_acceptance_tol: float = 1e-10,
    sylvester_max_iter: int = 500,
    stochastic_steady_state_tol: float = 1e-14,
    stochastic_steady_state_max_iter: int = 100,
) -> ParsedModelSecondOrderResult:
    return model.solve_second_order(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        pruning=pruning,
        sylvester_algorithm=sylvester_algorithm,
        sylvester_tol=sylvester_tol,
        sylvester_acceptance_tol=sylvester_acceptance_tol,
        sylvester_max_iter=sylvester_max_iter,
        stochastic_steady_state_tol=stochastic_steady_state_tol,
        stochastic_steady_state_max_iter=stochastic_steady_state_max_iter,
    )


def solve_third_order_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
    qme_algorithm: str = "schur",
    pruning: bool = False,
    sylvester_algorithm: str = "doubling",
    sylvester_tol: float = 1e-14,
    sylvester_acceptance_tol: float = 1e-10,
    sylvester_max_iter: int = 500,
    stochastic_steady_state_tol: float = 1e-14,
    stochastic_steady_state_max_iter: int = 100,
) -> ParsedModelThirdOrderResult:
    return model.solve_third_order(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
        qme_algorithm=qme_algorithm,
        pruning=pruning,
        sylvester_algorithm=sylvester_algorithm,
        sylvester_tol=sylvester_tol,
        sylvester_acceptance_tol=sylvester_acceptance_tol,
        sylvester_max_iter=sylvester_max_iter,
        stochastic_steady_state_tol=stochastic_steady_state_tol,
        stochastic_steady_state_max_iter=stochastic_steady_state_max_iter,
    )


def _solve_newton_system(
    initial: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
    nonfinite_message: str,
) -> tuple[np.ndarray, bool, int, float]:
    x = np.asarray(initial, dtype=np.float64)
    if lower_bounds is not None or upper_bounds is not None:
        lower = (
            np.asarray(lower_bounds, dtype=np.float64)
            if lower_bounds is not None
            else np.full_like(x, -np.inf)
        )
        upper = (
            np.asarray(upper_bounds, dtype=np.float64)
            if upper_bounds is not None
            else np.full_like(x, np.inf)
        )
        x = np.clip(x, lower, upper)
    else:
        lower = upper = None
    residual = np.asarray(residual_fn(x), dtype=np.float64).reshape(-1)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    if not np.isfinite(residual_norm):
        raise ValueError(nonfinite_message)

    for iteration in range(1, max_iter + 1):
        if residual_norm < tol:
            return x, True, iteration - 1, residual_norm

        jacobian = np.asarray(jacobian_fn(x), dtype=np.float64)
        try:
            direction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            try:
                direction, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            except np.linalg.LinAlgError:
                normal_matrix = jacobian.T @ jacobian
                rhs = -(jacobian.T @ residual)
                ridge = 1e-8 * max(1.0, float(np.linalg.norm(normal_matrix, ord=np.inf)))
                direction = np.linalg.solve(
                    normal_matrix + ridge * np.eye(normal_matrix.shape[0], dtype=np.float64),
                    rhs,
                )

        step = 1.0
        accepted = False
        while step >= line_search_min_step:
            candidate = x + step * direction
            if lower is not None and upper is not None:
                candidate = np.clip(candidate, lower, upper)
            if np.isfinite(candidate).all():
                candidate_residual = np.asarray(
                    residual_fn(candidate),
                    dtype=np.float64,
                ).reshape(-1)
                candidate_norm = float(np.linalg.norm(candidate_residual, ord=np.inf))
                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    x = candidate
                    residual = candidate_residual
                    residual_norm = candidate_norm
                    accepted = True
                    break
            step *= 0.5

        if not accepted:
            x = x + direction
            if lower is not None and upper is not None:
                x = np.clip(x, lower, upper)
            residual = np.asarray(residual_fn(x), dtype=np.float64).reshape(-1)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if not np.isfinite(residual_norm):
                break

    return x, residual_norm < tol, max_iter, residual_norm


def _solver_result_is_better(
    candidate: tuple[np.ndarray, bool, int, float],
    current: tuple[np.ndarray, bool, int, float],
) -> bool:
    _, current_converged, _, current_residual = current
    _, candidate_converged, _, candidate_residual = candidate
    if candidate_converged and not current_converged:
        return True
    if candidate_converged == current_converged:
        if np.isfinite(candidate_residual) and not np.isfinite(current_residual):
            return True
        if (
            np.isfinite(candidate_residual)
            and np.isfinite(current_residual)
            and candidate_residual < current_residual
        ):
            return True
    return False


def _solve_least_squares_system(
    initial: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
) -> Optional[tuple[np.ndarray, bool, int, float]]:
    x0 = np.asarray(initial, dtype=np.float64)
    lower = (
        np.asarray(lower_bounds, dtype=np.float64)
        if lower_bounds is not None
        else np.full_like(x0, -np.inf)
    )
    upper = (
        np.asarray(upper_bounds, dtype=np.float64)
        if upper_bounds is not None
        else np.full_like(x0, np.inf)
    )
    x0 = np.clip(x0, lower, upper)
    residual0 = np.asarray(residual_fn(x0), dtype=np.float64).reshape(-1)
    if not np.isfinite(residual0).all():
        return None

    try:
        result = scipy_optimize.least_squares(
            residual_fn,
            x0,
            jac=jacobian_fn,
            bounds=(lower, upper),
            method="trf",
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale="jac",
            max_nfev=max(100, min(400, max_iter * 2)),
        )
    except Exception:
        return None

    solution = np.asarray(result.x, dtype=np.float64)
    residual = np.asarray(residual_fn(solution), dtype=np.float64).reshape(-1)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    return (
        solution,
        residual_norm < tol,
        int(getattr(result, "nfev", 0)),
        residual_norm,
    )


def _finite_difference_jacobian(
    residual_fn: object,
    x: Sequence[float],
    *,
    step_scale: float = 1e-6,
) -> np.ndarray:
    point = np.asarray(x, dtype=np.float64)
    base = np.asarray(residual_fn(point), dtype=np.float64).reshape(-1)
    jacobian = np.zeros((base.shape[0], point.shape[0]), dtype=np.float64)

    for idx in range(point.shape[0]):
        step = step_scale * max(1.0, abs(float(point[idx])))
        forward = point.copy()
        forward[idx] += step
        backward = point.copy()
        backward[idx] -= step

        forward_residual = np.asarray(residual_fn(forward), dtype=np.float64).reshape(-1)
        backward_residual = np.asarray(residual_fn(backward), dtype=np.float64).reshape(-1)
        if np.isfinite(forward_residual).all() and np.isfinite(backward_residual).all():
            jacobian[:, idx] = (forward_residual - backward_residual) / (2.0 * step)
            continue
        if np.isfinite(forward_residual).all():
            jacobian[:, idx] = (forward_residual - base) / step
            continue
        if np.isfinite(backward_residual).all():
            jacobian[:, idx] = (base - backward_residual) / step
            continue
        raise ValueError(
            "Finite-difference Jacobian evaluation produced non-finite residuals."
        )

    return jacobian


def _make_safe_jacobian_fn(
    residual_fn: object,
    symbolic_jacobian_fn: object,
) -> object:
    jacobian_mode = {"use_finite_difference": False}

    def jacobian_fn(x: np.ndarray) -> np.ndarray:
        if jacobian_mode["use_finite_difference"]:
            return _finite_difference_jacobian(residual_fn, x)
        try:
            jacobian = np.asarray(symbolic_jacobian_fn(x), dtype=np.float64)
        except Exception:
            jacobian_mode["use_finite_difference"] = True
            return _finite_difference_jacobian(residual_fn, x)
        if not np.isfinite(jacobian).all():
            jacobian_mode["use_finite_difference"] = True
            return _finite_difference_jacobian(residual_fn, x)
        return jacobian

    return jacobian_fn


def _finite_difference_jacobian_jax(
    residual_fn: object,
    x: Sequence[float],
    *,
    step_scale: float = 1e-6,
) -> jax.Array:
    point = jnp.asarray(x, dtype=jnp.float64)
    base = jnp.asarray(residual_fn(point), dtype=jnp.float64).reshape(-1)
    steps = step_scale * jnp.maximum(1.0, jnp.abs(point))

    def _column(idx: int) -> jax.Array:
        step = steps[idx]
        basis = jax.nn.one_hot(idx, point.shape[0], dtype=point.dtype)
        forward = point + step * basis
        backward = point - step * basis
        forward_residual = jnp.asarray(
            residual_fn(forward),
            dtype=jnp.float64,
        ).reshape(-1)
        backward_residual = jnp.asarray(
            residual_fn(backward),
            dtype=jnp.float64,
        ).reshape(-1)
        forward_finite = jnp.all(jnp.isfinite(forward_residual))
        backward_finite = jnp.all(jnp.isfinite(backward_residual))
        central = (forward_residual - backward_residual) / (2.0 * step)
        forward_only = (forward_residual - base) / step
        backward_only = (base - backward_residual) / step
        return jnp.where(
            forward_finite & backward_finite,
            central,
            jnp.where(
                forward_finite,
                forward_only,
                jnp.where(
                    backward_finite,
                    backward_only,
                    jnp.full_like(base, jnp.nan),
                ),
            ),
        )

    return jax.vmap(_column)(jnp.arange(point.shape[0])).T


def _make_safe_jacobian_fn_jax(
    residual_fn: object,
    symbolic_jacobian_fn: object,
) -> object:
    def jacobian_fn(x: jax.Array) -> jax.Array:
        jacobian = jnp.asarray(symbolic_jacobian_fn(x), dtype=jnp.float64)
        return lax.cond(
            jnp.all(jnp.isfinite(jacobian)),
            lambda _: jacobian,
            lambda _: _finite_difference_jacobian_jax(residual_fn, x),
            operand=None,
        )

    return jacobian_fn


def _newton_restart_candidates(
    initial: Sequence[float],
    *,
    default_guess: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, ...]:
    initial_arr = np.asarray(initial, dtype=np.float64)
    default_arr = (
        np.asarray(default_guess, dtype=np.float64)
        if default_guess is not None
        else initial_arr
    )
    lower = (
        np.asarray(lower_bounds, dtype=np.float64)
        if lower_bounds is not None
        else np.full_like(initial_arr, -np.inf)
    )
    upper = (
        np.asarray(upper_bounds, dtype=np.float64)
        if upper_bounds is not None
        else np.full_like(initial_arr, np.inf)
    )

    candidates: list[np.ndarray] = []
    candidate_keys: list[np.ndarray] = []

    def _add(candidate: np.ndarray) -> None:
        raw = np.asarray(candidate, dtype=np.float64)
        clipped = np.clip(raw, lower, upper)
        if any(np.allclose(clipped, existing, rtol=0.0, atol=1e-12) for existing in candidate_keys):
            return
        candidates.append(raw)
        candidate_keys.append(clipped)

    _add(initial_arr)
    _add(default_arr)

    positive_seed = np.where(np.abs(default_arr) > 0.0, np.abs(default_arr), 1.0)
    unit_seed = np.where(default_arr >= 0.0, 1.0, -1.0)
    for scale in (1.0, 0.5, 0.1):
        _add(unit_seed * scale)
    for scale in _NEWTON_GEOMETRIC_RESTART_SCALES:
        _add(default_arr * scale)
        _add(np.where(default_arr >= 0.0, positive_seed * scale, -positive_seed * scale))

    nudged_bounds = default_arr.copy()
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    nudged_bounds[finite_lower] = np.maximum(
        nudged_bounds[finite_lower],
        lower[finite_lower] + np.maximum(1e-3, 0.05 * np.maximum(1.0, np.abs(lower[finite_lower]))),
    )
    nudged_bounds[finite_upper] = np.minimum(
        nudged_bounds[finite_upper],
        upper[finite_upper] - np.maximum(1e-3, 0.05 * np.maximum(1.0, np.abs(upper[finite_upper]))),
    )
    _add(nudged_bounds)

    midpoint = default_arr.copy()
    finite_box = finite_lower & finite_upper
    midpoint[finite_box] = 0.5 * (lower[finite_box] + upper[finite_box])
    _add(midpoint)

    return tuple(candidates)


def _newton_restart_candidates_jax(
    initial: Sequence[float],
    *,
    default_guess: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
) -> jax.Array:
    initial_arr = jnp.asarray(initial, dtype=jnp.float64)
    default_arr = (
        jnp.asarray(default_guess, dtype=jnp.float64)
        if default_guess is not None
        else initial_arr
    )
    lower = (
        jnp.asarray(lower_bounds, dtype=jnp.float64)
        if lower_bounds is not None
        else jnp.full_like(initial_arr, -jnp.inf)
    )
    upper = (
        jnp.asarray(upper_bounds, dtype=jnp.float64)
        if upper_bounds is not None
        else jnp.full_like(initial_arr, jnp.inf)
    )

    positive_seed = jnp.where(jnp.abs(default_arr) > 0.0, jnp.abs(default_arr), 1.0)
    unit_seed = jnp.where(default_arr >= 0.0, 1.0, -1.0)
    sign_preserving_candidates = []
    for scale in (1.0, 0.5, 0.1):
        sign_preserving_candidates.append(unit_seed * scale)
    for scale in _NEWTON_GEOMETRIC_RESTART_SCALES:
        sign_preserving_candidates.append(default_arr * scale)
        sign_preserving_candidates.append(
            jnp.where(default_arr >= 0.0, positive_seed * scale, -positive_seed * scale)
        )

    finite_lower = jnp.isfinite(lower)
    finite_upper = jnp.isfinite(upper)
    lower_nudge = lower + jnp.maximum(1e-3, 0.05 * jnp.maximum(1.0, jnp.abs(lower)))
    upper_nudge = upper - jnp.maximum(1e-3, 0.05 * jnp.maximum(1.0, jnp.abs(upper)))
    nudged_bounds = jnp.where(
        finite_lower,
        jnp.maximum(default_arr, lower_nudge),
        default_arr,
    )
    nudged_bounds = jnp.where(
        finite_upper,
        jnp.minimum(nudged_bounds, upper_nudge),
        nudged_bounds,
    )
    midpoint = jnp.where(
        finite_lower & finite_upper,
        0.5 * (lower + upper),
        default_arr,
    )

    candidates = jnp.stack(
        [initial_arr, default_arr, *sign_preserving_candidates, nudged_bounds, midpoint],
        axis=0,
    )
    return jnp.clip(candidates, lower, upper)


def _solve_homotopy_system(
    anchor: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
    nonfinite_message: str,
    levels: Sequence[float] = _STEADY_STATE_HOMOTOPY_LEVELS,
) -> Optional[tuple[np.ndarray, bool, int, float]]:
    anchor_arr = np.asarray(anchor, dtype=np.float64)
    residual_shape = np.asarray(residual_fn(anchor_arr), dtype=np.float64).reshape(-1).shape[0]
    if residual_shape != anchor_arr.shape[0]:
        return None
    lower = (
        np.asarray(lower_bounds, dtype=np.float64)
        if lower_bounds is not None
        else np.full_like(anchor_arr, -np.inf)
    )
    upper = (
        np.asarray(upper_bounds, dtype=np.float64)
        if upper_bounds is not None
        else np.full_like(anchor_arr, np.inf)
    )
    current = np.clip(anchor_arr, lower, upper)
    total_iterations = 0
    final_result: Optional[tuple[np.ndarray, bool, int, float]] = None

    for lam in tuple(levels)[1:]:
        lam_float = float(lam)

        def homotopy_residual_fn(x: np.ndarray) -> np.ndarray:
            point = np.asarray(x, dtype=np.float64)
            residual = np.asarray(residual_fn(point), dtype=np.float64).reshape(-1)
            return lam_float * residual + (1.0 - lam_float) * (point - current)

        def homotopy_jacobian_fn(x: np.ndarray) -> np.ndarray:
            point = np.asarray(x, dtype=np.float64)
            jacobian = np.asarray(jacobian_fn(point), dtype=np.float64)
            identity = np.eye(point.shape[0], dtype=np.float64)
            return lam_float * jacobian + (1.0 - lam_float) * identity

        try:
            stage_result = _solve_newton_system(
                current,
                residual_fn=homotopy_residual_fn,
                jacobian_fn=homotopy_jacobian_fn,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
                nonfinite_message=nonfinite_message,
            )
        except ValueError:
            return None
        total_iterations += int(stage_result[2])
        if not bool(stage_result[1]):
            return None
        current = np.asarray(stage_result[0], dtype=np.float64)
        final_result = (
            current,
            bool(stage_result[1]),
            total_iterations,
            float(stage_result[3]),
        )

    return final_result


def _solve_homotopy_system_jax(
    anchor: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    levels = jnp.asarray(_STEADY_STATE_HOMOTOPY_LEVELS[1:], dtype=jnp.float64)
    anchor_arr = jnp.asarray(anchor, dtype=jnp.float64)
    residual_shape = jnp.asarray(residual_fn(anchor_arr), dtype=jnp.float64).reshape(-1).shape[0]
    if residual_shape != anchor_arr.shape[0]:
        return (
            anchor_arr,
            jnp.asarray(False),
            jnp.asarray(0),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )

    def _stage_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        lam: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        current_x, path_ok, total_iterations, last_residual_norm = carry

        def _advance(_: None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            def homotopy_residual_fn(x: jax.Array) -> jax.Array:
                residual = jnp.asarray(residual_fn(x), dtype=jnp.float64).reshape(-1)
                return lam * residual + (1.0 - lam) * (x - current_x)

            def homotopy_jacobian_fn(x: jax.Array) -> jax.Array:
                jacobian = jnp.asarray(jacobian_fn(x), dtype=jnp.float64)
                identity = jnp.eye(x.shape[0], dtype=jnp.float64)
                return lam * jacobian + (1.0 - lam) * identity

            stage_x, stage_converged, stage_iterations, stage_residual_norm = _solve_newton_system_jax(
                current_x,
                residual_fn=homotopy_residual_fn,
                jacobian_fn=homotopy_jacobian_fn,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
            return (
                stage_x,
                stage_converged,
                total_iterations + stage_iterations,
                stage_residual_norm,
            )

        updated = lax.cond(
            path_ok,
            _advance,
            lambda _: (current_x, path_ok, total_iterations, last_residual_norm),
            operand=None,
        )
        return updated, None

    init = (
        anchor_arr,
        jnp.asarray(True),
        jnp.asarray(0),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    final_state, _ = lax.scan(_stage_step, init, levels)
    return final_state


def _solve_newton_system_with_restarts(
    initial: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    default_guess: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
    nonfinite_message: str,
) -> tuple[np.ndarray, bool, int, float]:
    best_result: tuple[np.ndarray, bool, int, float] | None = None
    last_error: Exception | None = None
    for candidate in _newton_restart_candidates(
        initial,
        default_guess=default_guess,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    ):
        try:
            result = _solve_newton_system(
                candidate,
                residual_fn=residual_fn,
                jacobian_fn=jacobian_fn,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
                nonfinite_message=nonfinite_message,
            )
        except ValueError as err:
            last_error = err
            continue
        if best_result is None:
            best_result = result
        else:
            if _solver_result_is_better(result, best_result):
                best_result = result
        if result[1]:
            return result
    if (
        best_result is not None
        and not best_result[1]
        and np.isfinite(best_result[3])
        and best_result[3] <= 1.0
    ):
        least_squares_result = _solve_least_squares_system(
            best_result[0],
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            tol=tol,
            max_iter=max_iter,
        )
        if least_squares_result is not None:
            if _solver_result_is_better(least_squares_result, best_result):
                best_result = least_squares_result
            if not best_result[1]:
                try:
                    refined_result = _solve_newton_system(
                        best_result[0],
                        residual_fn=residual_fn,
                        jacobian_fn=jacobian_fn,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                        tol=tol,
                        max_iter=max(max_iter * 2, max_iter + 25),
                        line_search_min_step=line_search_min_step,
                        nonfinite_message=nonfinite_message,
                    )
                except ValueError:
                    refined_result = None
                if refined_result is not None and _solver_result_is_better(
                    refined_result,
                    best_result,
                ):
                    best_result = refined_result
            if best_result[1]:
                return best_result
    if best_result is not None and not best_result[1] and np.isfinite(best_result[3]):
        homotopy_result = _solve_homotopy_system(
            best_result[0],
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            tol=tol,
            max_iter=max_iter,
            line_search_min_step=line_search_min_step,
            nonfinite_message=nonfinite_message,
        )
        if homotopy_result is not None and _solver_result_is_better(
            homotopy_result,
            best_result,
        ):
            best_result = homotopy_result
        if best_result[1]:
            return best_result
    if best_result is not None:
        return best_result
    if last_error is not None:
        raise last_error
    raise ValueError(nonfinite_message)


def _solve_newton_system_jax_with_restarts(
    initial: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    default_guess: Optional[Sequence[float]] = None,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    candidates = _newton_restart_candidates_jax(
        initial,
        default_guess=default_guess,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    first_result = _solve_newton_system_jax(
        candidates[0],
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        tol=tol,
        max_iter=max_iter,
        line_search_min_step=line_search_min_step,
    )

    def _maybe_update_best(
        best: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        candidate: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        best_x, best_converged, best_iterations, best_residual_norm = best
        cand_x, cand_converged, cand_iterations, cand_residual_norm = candidate
        candidate_better = (cand_converged & (~best_converged)) | (
            (cand_converged == best_converged)
            & (
                (jnp.isfinite(cand_residual_norm) & (~jnp.isfinite(best_residual_norm)))
                | (
                    jnp.isfinite(cand_residual_norm)
                    & jnp.isfinite(best_residual_norm)
                    & (cand_residual_norm < best_residual_norm)
                )
            )
        )
        return (
            jnp.where(candidate_better, cand_x, best_x),
            jnp.where(candidate_better, cand_converged, best_converged),
            jnp.where(candidate_better, cand_iterations, best_iterations),
            jnp.where(candidate_better, cand_residual_norm, best_residual_norm),
        )

    def _scan_body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        candidate: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        best_result = carry
        best_converged = best_result[1]

        def _solve_and_update(
            _: None,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            candidate_result = _solve_newton_system_jax(
                candidate,
                residual_fn=residual_fn,
                jacobian_fn=jacobian_fn,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
            return _maybe_update_best(best_result, candidate_result)

        updated = lax.cond(
            best_converged,
            lambda _: best_result,
            _solve_and_update,
            operand=None,
        )
        return updated, None

    final_result, _ = lax.scan(_scan_body, first_result, candidates[1:])
    residual_is_finite = jnp.isfinite(final_result[3])
    homotopy_result = _solve_homotopy_system_jax(
        final_result[0],
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        tol=tol,
        max_iter=max_iter,
        line_search_min_step=line_search_min_step,
    )
    use_homotopy = (~final_result[1]) & residual_is_finite & homotopy_result[1]
    return (
        jnp.where(use_homotopy, homotopy_result[0], final_result[0]),
        jnp.where(use_homotopy, homotopy_result[1], final_result[1]),
        jnp.where(use_homotopy, homotopy_result[2], final_result[2]),
        jnp.where(use_homotopy, homotopy_result[3], final_result[3]),
    )


def _solve_newton_system_jax(
    initial: Sequence[float],
    *,
    residual_fn: object,
    jacobian_fn: object,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    tol: float,
    max_iter: int,
    line_search_min_step: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x0 = jnp.asarray(initial, dtype=jnp.float64)
    if lower_bounds is not None or upper_bounds is not None:
        lower = (
            jnp.asarray(lower_bounds, dtype=jnp.float64)
            if lower_bounds is not None
            else jnp.full_like(x0, -jnp.inf)
        )
        upper = (
            jnp.asarray(upper_bounds, dtype=jnp.float64)
            if upper_bounds is not None
            else jnp.full_like(x0, jnp.inf)
        )
        x0 = jnp.clip(x0, lower, upper)
    else:
        lower = upper = None

    tol_arr = jnp.asarray(tol, dtype=jnp.float64)
    backtracking_steps = max(
        1,
        int(np.ceil(np.log2(1.0 / line_search_min_step))) + 1,
    )

    def _clip(x: jax.Array) -> jax.Array:
        if lower is None or upper is None:
            return x
        return jnp.clip(x, lower, upper)

    residual0 = jnp.asarray(residual_fn(x0), dtype=jnp.float64).reshape(-1)
    residual_norm0 = jnp.linalg.norm(residual0, ord=jnp.inf)
    converged0 = jnp.isfinite(residual_norm0) & (residual_norm0 < tol_arr)
    initial_state = _JaxNewtonState(
        x=x0,
        residual=residual0,
        residual_norm=residual_norm0,
        converged=converged0,
        done=converged0 | (~jnp.isfinite(residual_norm0)),
        iterations=jnp.asarray(0),
    )

    def body(iteration: int, state: _JaxNewtonState) -> _JaxNewtonState:
        def _active(current_state: _JaxNewtonState) -> _JaxNewtonState:
            jacobian = jnp.asarray(jacobian_fn(current_state.x), dtype=jnp.float64)
            direction = -(jnp.linalg.pinv(jacobian) @ current_state.residual)

            def line_search_body(
                search_step: int,
                search_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                candidate_x, candidate_residual, candidate_norm, accepted = search_state
                step_scale = jnp.asarray(0.5**search_step, dtype=current_state.x.dtype)
                proposed_x = _clip(current_state.x + step_scale * direction)
                proposed_residual = jnp.asarray(
                    residual_fn(proposed_x),
                    dtype=jnp.float64,
                ).reshape(-1)
                proposed_norm = jnp.linalg.norm(proposed_residual, ord=jnp.inf)
                improve = (
                    jnp.isfinite(proposed_norm)
                    & (proposed_norm < current_state.residual_norm)
                )
                accept_now = (~accepted) & improve
                return (
                    jnp.where(accept_now, proposed_x, candidate_x),
                    jnp.where(accept_now, proposed_residual, candidate_residual),
                    jnp.where(accept_now, proposed_norm, candidate_norm),
                    accepted | improve,
                )

            line_search_init = (
                current_state.x,
                current_state.residual,
                current_state.residual_norm,
                jnp.asarray(False),
            )
            candidate_x, candidate_residual, candidate_norm, accepted = lax.fori_loop(
                0,
                backtracking_steps,
                line_search_body,
                line_search_init,
            )

            fallback_x = _clip(current_state.x + direction)
            fallback_residual = jnp.asarray(
                residual_fn(fallback_x),
                dtype=jnp.float64,
            ).reshape(-1)
            fallback_norm = jnp.linalg.norm(fallback_residual, ord=jnp.inf)

            next_x = jnp.where(accepted, candidate_x, fallback_x)
            next_residual = jnp.where(accepted, candidate_residual, fallback_residual)
            next_norm = jnp.where(accepted, candidate_norm, fallback_norm)
            finite_norm = jnp.isfinite(next_norm)
            converged = finite_norm & (next_norm < tol_arr)
            return _JaxNewtonState(
                x=next_x,
                residual=next_residual,
                residual_norm=next_norm,
                converged=converged,
                done=converged | (~finite_norm),
                iterations=jnp.asarray(iteration + 1),
            )

        return lax.cond(
            state.done,
            lambda current_state: current_state,
            _active,
            state,
        )

    final_state = lax.fori_loop(0, max_iter, body, initial_state)
    return (
        final_state.x,
        final_state.converged,
        final_state.iterations,
        final_state.residual_norm,
    )


def _extract_block(
    pattern: re.Pattern[str],
    source: str,
    label: str,
) -> dict[str, str | int]:
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Could not find `@{label}` block in source.")
    block = match.groupdict()
    body_lines: list[str] = []
    nested_depth = 0
    cursor = match.end()
    for raw_line in source[match.end() :].splitlines(keepends=True):
        visible = _strip_comment(raw_line).strip()
        if visible == "end" and nested_depth == 0:
            block["body"] = "".join(body_lines).rstrip("\n")
            block["start"] = match.start()
            block["end"] = cursor + len(raw_line)
            return block
        body_lines.append(raw_line)
        cursor += len(raw_line)
        nested_depth += _block_line_delta(visible)
        if nested_depth < 0:
            raise ValueError(f"Encountered unmatched `end` while parsing `@{label}`.")
    raise ValueError(f"Could not find closing `end` for `@{label}` block.")


def _split_body_lines(body: str) -> list[str]:
    lines = []
    for raw_line in body.splitlines():
        line = _strip_comment(raw_line).strip()
        if line:
            lines.append(line)
    return lines


def _parse_source_loop_collections(
    source: str,
    block_spans: Sequence[tuple[int, int]],
) -> dict[str, tuple[LoopIndex, ...]]:
    spans = sorted((int(start), int(end)) for start, end in block_spans)
    cursor = 0
    outside_parts: list[str] = []
    for start, end in spans:
        if cursor < start:
            outside_parts.append(source[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(source):
        outside_parts.append(source[cursor:])

    collections: dict[str, tuple[LoopIndex, ...]] = {}
    for raw_line in "\n".join(outside_parts).splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        match = re.fullmatch(
            rf"(?P<name>{_IDENTIFIER_PATTERN})\s*=\s*(?P<value>.+)",
            line,
            re.UNICODE,
        )
        if match is None:
            continue
        parsed = _parse_loop_collection_value(match.group("value"))
        if parsed is not None:
            collections[match.group("name")] = parsed
    return collections


def _parse_loop_collection_value(value_text: str) -> tuple[LoopIndex, ...] | None:
    token = value_text.strip()
    if not token:
        return None
    if token.startswith("[") and token.endswith("]"):
        entries = _split_top_level(token[1:-1], ",")
        return tuple(
            _parse_for_loop_index_token(entry)
            for entry in entries
            if entry.strip()
        )

    range_parts = _split_top_level(token, ":")
    if len(range_parts) == 2:
        start = _evaluate_integer_expression(range_parts[0])
        stop = _evaluate_integer_expression(range_parts[1])
        step = 1 if stop >= start else -1
        return tuple(range(start, stop + step, step))
    return None


def _split_model_body_lines(
    body: str,
    *,
    loop_collections: Optional[LoopCollections] = None,
) -> list[str]:
    lines = _split_body_lines(body)
    statements: list[str] = []
    current: list[str] = []
    loop_depth = 0
    resolved_loop_collections = loop_collections or {}

    for line in lines:
        stripped = line.strip()
        current.append(stripped)
        loop_depth += _block_line_delta(stripped)
        if loop_depth < 0:
            raise ValueError("Encountered `end` without a matching `for` in `@model`.")

        if (
            loop_depth == 0
            and _delimiter_balance(" ".join(current)) == 0
            and not _model_line_requires_continuation(stripped)
        ):
            if _is_model_for_block_statement(current):
                statements.extend(
                    _expand_model_for_block(
                        current,
                        loop_collections=resolved_loop_collections,
                    )
                )
            else:
                statements.append(
                    _expand_inline_for_loops(
                        " ".join(current),
                        loop_collections=resolved_loop_collections,
                    )
                )
            current = []

    if loop_depth != 0:
        raise ValueError("Unbalanced `for` / `end` blocks in `@model`.")
    if current:
        if _is_model_for_block_statement(current):
            statements.extend(
                _expand_model_for_block(
                    current,
                    loop_collections=resolved_loop_collections,
                )
            )
        else:
            statements.append(
                _expand_inline_for_loops(
                    " ".join(current),
                    loop_collections=resolved_loop_collections,
                )
            )
    return statements


def _split_parameter_body_lines(body: str) -> list[str]:
    lines = _split_body_lines(body)
    statements: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        current.append(stripped)
        if (
            _delimiter_balance(" ".join(current)) == 0
            and not _model_line_requires_continuation(stripped)
        ):
            statements.append(" ".join(current))
            current = []

    if current:
        statements.append(" ".join(current))
    return _resolve_parameter_tail_conditionals(statements)


def _resolve_parameter_tail_conditionals(statements: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    environment: dict[str, float] = {}
    conditional_active: Optional[bool] = None
    branch_taken = False
    tail_conditional_seen = False
    statement_targets: dict[str, int] = {}

    for statement in statements:
        stripped = statement.strip()
        if stripped.startswith("if "):
            conditional_active = _evaluate_parameter_condition_expression(
                stripped[len("if") :].strip(),
                environment,
            )
            branch_taken = conditional_active
            tail_conditional_seen = True
            continue
        if stripped.startswith("elseif "):
            if not tail_conditional_seen:
                raise ValueError("Encountered `elseif` before `if` in `@parameters`.")
            if branch_taken:
                conditional_active = False
            else:
                conditional_active = _evaluate_parameter_condition_expression(
                    stripped[len("elseif") :].strip(),
                    environment,
                )
                branch_taken = conditional_active
            continue
        if stripped == "else":
            if not tail_conditional_seen:
                raise ValueError("Encountered `else` before `if` in `@parameters`.")
            conditional_active = not branch_taken
            branch_taken = True
            continue

        if conditional_active is False:
            continue

        target_name = _parameter_direct_assignment_target(statement)
        if (
            tail_conditional_seen
            and target_name is not None
            and target_name in statement_targets
        ):
            filtered[statement_targets[target_name]] = statement
        else:
            if target_name is not None:
                statement_targets[target_name] = len(filtered)
            filtered.append(statement)

        _update_parameter_condition_environment(statement, environment)

    return filtered


def _evaluate_parameter_condition_expression(
    expression_text: str,
    environment: Mapping[str, float],
) -> bool:
    for operator in ("==", "!=", ">=", "<=", ">", "<"):
        if operator not in expression_text:
            continue
        lhs_text, rhs_text = (part.strip() for part in expression_text.split(operator, 1))
        lhs_value = _evaluate_parameter_condition_operand(lhs_text, environment)
        rhs_value = _evaluate_parameter_condition_operand(rhs_text, environment)
        if operator == "==":
            return bool(np.isclose(lhs_value, rhs_value, rtol=0.0, atol=1e-12))
        if operator == "!=":
            return not bool(np.isclose(lhs_value, rhs_value, rtol=0.0, atol=1e-12))
        if operator == ">=":
            return lhs_value >= rhs_value
        if operator == "<=":
            return lhs_value <= rhs_value
        if operator == ">":
            return lhs_value > rhs_value
        if operator == "<":
            return lhs_value < rhs_value
    raise ValueError(
        "Parameter conditionals must use one of `==`, `!=`, `>=`, `<=`, `>`, or `<`, "
        f"got `{expression_text}`."
    )


def _evaluate_parameter_condition_operand(
    operand_text: str,
    environment: Mapping[str, float],
) -> float:
    parse_name_map = {
        name: _parameter_parse_name(name)
        for name in environment
        if _is_indexed_identifier(name)
    }
    local_env = {
        parse_name_map.get(key, key): value
        for key, value in environment.items()
    }
    expr = parse_expr(
        _sanitize_indexed_identifiers(operand_text, parse_name_map),
        local_dict={**local_env, **_function_locals()},
        transformations=_TRANSFORMATIONS,
    )
    if getattr(expr, "free_symbols", set()):
        unresolved = ", ".join(sorted(str(symbol) for symbol in expr.free_symbols))
        raise ValueError(
            "Parameter conditional expression depends on unresolved identifiers: "
            + unresolved
        )
    return _evaluate_constant_parameter_expression(expr)


def _parameter_direct_assignment_target(statement: str) -> Optional[str]:
    if _is_parameter_bounds_line(statement) or "|" in statement or "=" not in statement:
        return None
    target_text = statement.split("=", 1)[0].strip()
    if not re.fullmatch(_IDENTIFIER_PATTERN, target_text, re.UNICODE):
        return None
    return target_text


def _update_parameter_condition_environment(
    statement: str,
    environment: dict[str, float],
) -> None:
    target_name = _parameter_direct_assignment_target(statement)
    if target_name is None:
        return

    parse_name_map = {
        name: _parameter_parse_name(name)
        for name in environment
        if _is_indexed_identifier(name)
    }
    local_env = {
        parse_name_map.get(key, key): value
        for key, value in environment.items()
    }
    try:
        expr = parse_expr(
            _sanitize_indexed_identifiers(
                statement.split("=", 1)[1].strip(),
                parse_name_map,
            ),
            local_dict={**local_env, **_function_locals()},
            transformations=_TRANSFORMATIONS,
        )
    except Exception:
        return
    if getattr(expr, "free_symbols", set()):
        return
    try:
        environment[target_name] = _evaluate_constant_parameter_expression(expr)
    except Exception:
        return


def _model_line_requires_continuation(line: str) -> bool:
    return bool(re.search(r"[+\-*/=]$", line))


def _delimiter_balance(text: str) -> int:
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def _block_line_delta(line: str) -> int:
    if not line:
        return 0
    return len(re.findall(r"\bfor\b", line)) - len(re.findall(r"\bend\b", line))


def _is_model_for_block_statement(lines: Sequence[str]) -> bool:
    return (
        bool(lines)
        and lines[0].startswith("for ")
        and lines[-1] == "end"
        and len(lines) > 1
    )


def _expand_model_for_block(
    lines: Sequence[str],
    *,
    loop_collections: Optional[LoopCollections] = None,
) -> list[str]:
    if not _is_model_for_block_statement(lines):
        raise ValueError("Expected a top-level `for` block in `@model`.")
    _, loop_var, indices_text, inline_body = _parse_for_loop_header(
        lines[0][len("for") :].strip(),
        lines[0],
        allow_empty_body=True,
    )
    body_lines = list(lines[1:-1])
    if inline_body:
        body_lines.insert(0, inline_body)
    if not body_lines:
        return []

    expanded: list[str] = []
    for idx in _parse_for_loop_indices(
        indices_text,
        loop_collections=loop_collections,
    ):
        substituted_lines = [
            _substitute_loop_variable(line, loop_var, idx)
            for line in body_lines
        ]
        expanded.extend(
            _split_model_body_lines(
                "\n".join(substituted_lines),
                loop_collections=loop_collections,
            )
        )
    return expanded


def _expand_inline_for_loops(
    statement: str,
    *,
    loop_collections: Optional[LoopCollections] = None,
) -> str:
    expanded = statement
    while re.search(r"\bfor\b", expanded):
        start, end = _find_innermost_for_segment(expanded)
        replacement = _expand_single_inline_for_loop(
            expanded[start:end],
            loop_collections=loop_collections,
        )
        expanded = expanded[:start] + replacement + expanded[end:]
    return expanded


def _find_innermost_for_segment(statement: str) -> tuple[int, int]:
    stack: list[int] = []
    for match in re.finditer(r"\bfor\b|\bend\b", statement):
        token = match.group(0)
        if token == "for":
            stack.append(match.start())
            continue
        if not stack:
            raise ValueError("Encountered `end` without a matching `for` in equation.")
        start = stack.pop()
        return start, match.end()
    raise ValueError("Encountered `for` without a matching `end` in equation.")


def _expand_single_inline_for_loop(
    segment: str,
    *,
    loop_collections: Optional[LoopCollections] = None,
) -> str:
    segment_text = segment.strip()
    if not re.search(r"\bend\s*$", segment_text):
        raise ValueError(f"Could not find closing `end` in `for` loop `{segment}`.")
    loop_text = re.sub(r"\s*\bend\s*$", "", segment_text)
    operator, loop_var, indices_text, body = _parse_for_loop_header(
        loop_text[len("for") :].strip(),
        segment,
        allow_empty_body=False,
    )

    indices = _parse_for_loop_indices(
        indices_text,
        loop_collections=loop_collections,
    )
    terms = [
        _expand_inline_for_loops(
            _substitute_loop_variable(body, loop_var, idx),
            loop_collections=loop_collections,
        )
        for idx in indices
    ]
    joiner = f" {operator} "
    return "(" + joiner.join(f"({term})" for term in terms) + ")"


def _parse_for_loop_header(
    rest: str,
    segment: str,
    *,
    allow_empty_body: bool,
) -> tuple[str, str, str, str]:
    operator = "+"
    header_rest = rest
    if header_rest.startswith("operator"):
        operator_match = re.match(
            r"operator\s*=\s*:(?P<op>[+*])\s*,\s*(?P<rest>.*)",
            header_rest,
        )
        if operator_match is None:
            raise ValueError(f"Could not parse `for`-loop operator in `{segment}`.")
        operator = operator_match.group("op")
        header_rest = operator_match.group("rest").strip()

    header_match = re.match(
        rf"(?P<var>{_IDENTIFIER_PATTERN})\s+in\s+(?P<rest>.*)",
        header_rest,
    )
    if header_match is None:
        raise ValueError(f"Could not parse `for`-loop header in `{segment}`.")
    loop_var = header_match.group("var")
    indices_text, body = _consume_loop_indices(header_match.group("rest"))
    if not allow_empty_body and not body:
        raise ValueError(f"Missing `for`-loop body in `{segment}`.")
    return operator, loop_var, indices_text, body


def _consume_loop_indices(rest: str) -> tuple[str, str]:
    text = rest.lstrip()
    if not text:
        raise ValueError("Missing `for`-loop indices.")
    if text[0] == "[":
        end = _find_matching_delimiter(text, 0, "[", "]")
        return text[: end + 1], text[end + 1 :].strip()

    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char.isspace() and depth == 0:
            return text[:idx], text[idx:].strip()
    return text, ""


def _find_matching_delimiter(
    text: str,
    start_idx: int,
    opening: str,
    closing: str,
) -> int:
    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Could not find matching `{closing}` in `{text}`.")


def _parse_for_loop_indices(
    indices_text: str,
    *,
    loop_collections: Optional[LoopCollections] = None,
) -> list[LoopIndex]:
    token = indices_text.strip()
    resolved_loop_collections = loop_collections or {}
    if token in resolved_loop_collections:
        return list(resolved_loop_collections[token])
    if token.startswith("[") and token.endswith("]"):
        entries = _split_top_level(token[1:-1], ",")
        return [
            _parse_for_loop_index_token(entry)
            for entry in entries
            if entry.strip()
        ]

    range_parts = _split_top_level(token, ":")
    if len(range_parts) == 2:
        start = _evaluate_integer_expression(range_parts[0])
        stop = _evaluate_integer_expression(range_parts[1])
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    try:
        return [_evaluate_integer_expression(token)]
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "Symbolic/indexed `for` loops in `@model` are only supported for "
            "explicit identifier lists like `[H, F]`, named collections, and "
            "integer ranges."
        ) from exc


def _parse_for_loop_index_token(token: str) -> LoopIndex:
    stripped = token.strip()
    if stripped.startswith(":"):
        stripped = stripped[1:].strip()
    try:
        return _evaluate_integer_expression(stripped)
    except (TypeError, ValueError):
        if re.fullmatch(_IDENTIFIER_PATTERN, stripped, re.UNICODE):
            return stripped
    raise NotImplementedError(
        "Symbolic/indexed `for` loops in `@model` are only supported for "
        "explicit identifier lists like `[H, F]`, named collections, and "
        "integer ranges."
    )


def _split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return parts


def _split_top_level_operator(text: str, operator: str) -> tuple[str, str]:
    depth = 0
    idx = 0
    while idx <= len(text) - len(operator):
        char = text[idx]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if depth == 0 and text.startswith(operator, idx):
            return text[:idx].strip(), text[idx + len(operator) :].strip()
        idx += 1
    raise ValueError(f"Could not find top-level operator `{operator}` in `{text}`.")


def _parse_parameter_block_guess(options_text: str) -> dict[str, float]:
    match = re.search(r"\bguess\s*=\s*Dict\s*\(", options_text)
    if match is None:
        return {}

    open_idx = options_text.find("(", match.start())
    close_idx = _find_matching_delimiter(options_text, open_idx, "(", ")")
    body = options_text[open_idx + 1 : close_idx].strip()
    if not body:
        return {}

    guesses: dict[str, float] = {}
    for entry in _split_top_level(body, ","):
        if not entry:
            continue
        key_text, value_text = _split_top_level_operator(entry, "=>")
        guesses[_parse_guess_key(key_text)] = _parse_guess_value(value_text)
    return guesses


def _strip_guess_option(options_text: str) -> str:
    match = re.search(r"\bguess\s*=\s*Dict\s*\(", options_text)
    if match is None:
        return options_text
    open_idx = options_text.find("(", match.start())
    close_idx = _find_matching_delimiter(options_text, open_idx, "(", ")")
    return (options_text[: match.start()] + " " + options_text[close_idx + 1 :]).strip()


def _parse_block_options(options_text: str) -> dict[str, object]:
    stripped = _strip_guess_option(options_text).strip()
    if not stripped:
        return {}

    assignment_re = re.compile(
        rf"(?P<name>{_IDENTIFIER_PATTERN})\s*=\s*",
        re.UNICODE,
    )
    parsed: dict[str, object] = {}
    cursor = 0
    while True:
        match = assignment_re.search(stripped, cursor)
        if match is None:
            break
        value_start = match.end()
        next_match = assignment_re.search(stripped, value_start)
        value_end = next_match.start() if next_match is not None else len(stripped)
        value_text = stripped[value_start:value_end].strip().rstrip(",")
        if value_text:
            parsed[match.group("name")] = _parse_block_option_value(value_text)
        cursor = value_end
    return parsed


def _parse_block_option_value(text: str) -> object:
    token = text.strip()
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if token.startswith(":") and len(token) > 1:
        return token[1:]
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    try:
        value = parse_expr(
            token,
            local_dict=_function_locals(),
            transformations=_TRANSFORMATIONS,
        )
    except Exception:
        return token
    if isinstance(value, sp.Integer):
        return int(value)
    if isinstance(value, (sp.Float, sp.Rational)):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    if value is sp.true:
        return True
    if value is sp.false:
        return False
    if isinstance(value, sp.Symbol):
        return str(value)
    return token


def _parse_guess_key(text: str) -> str:
    token = text.strip()
    if token.startswith(":") and len(token) > 1:
        return token[1:]
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    if re.fullmatch(_IDENTIFIER_PATTERN, token, re.UNICODE):
        return token
    raise ValueError(f"Unsupported guess key `{text}` in `@parameters` options.")


def _parse_guess_value(text: str) -> float:
    value = parse_expr(
        text.strip(),
        local_dict=_function_locals(),
        transformations=_TRANSFORMATIONS,
    )
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"Guess value `{text}` must evaluate to a finite number.")
    return numeric_value


def _expand_default_initial_guess(
    raw_guess: Mapping[str, float],
    *,
    steady_state_names: Sequence[str],
    calibrated_parameter_names: Sequence[str],
) -> dict[str, float]:
    if not raw_guess:
        return {}

    indexed_targets: dict[str, list[str]] = {}
    for name in tuple(steady_state_names) + tuple(calibrated_parameter_names):
        if _is_indexed_identifier(name):
            indexed_targets.setdefault(_identifier_base_name(name), []).append(name)

    expanded: dict[str, float] = {}
    for key, value in raw_guess.items():
        if key in steady_state_names or key in calibrated_parameter_names:
            expanded[key] = float(value)
            continue
        if not _is_indexed_identifier(key) and key in indexed_targets:
            for name in indexed_targets[key]:
                expanded[name] = float(value)
    return expanded


def _parse_parameter_bounds_line(
    line: str,
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[float, float]]:
    tokens = [
        token.strip()
        for token in re.split(r"\s*(<=|>=|<|>)\s*", line.strip())
        if token.strip()
    ]
    if len(tokens) == 3:
        return _parse_single_parameter_bound(
            tokens[0],
            tokens[1],
            tokens[2],
            available_indexed_names,
        )
    if len(tokens) == 5:
        bounds = _parse_single_parameter_bound(
            tokens[0],
            tokens[1],
            tokens[2],
            available_indexed_names,
        )
        for name, bound in _parse_single_parameter_bound(
            tokens[2],
            tokens[3],
            tokens[4],
            available_indexed_names,
        ).items():
            bounds[name] = _merge_bounds(bounds.get(name), bound)
        return bounds
    raise ValueError(f"Unsupported bound syntax `{line}` in `@parameters`.")


def _parse_single_parameter_bound(
    lhs_text: str,
    operator: str,
    rhs_text: str,
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[float, float]]:
    lhs_is_identifier = _is_bound_identifier(lhs_text)
    rhs_is_identifier = _is_bound_identifier(rhs_text)

    if lhs_is_identifier == rhs_is_identifier:
        raise ValueError(
            "Bounds in `@parameters` must compare one identifier against one "
            f"numeric expression, got `{lhs_text} {operator} {rhs_text}`."
        )

    if lhs_is_identifier:
        target_names = _expand_bound_target_names(lhs_text, available_indexed_names)
        bound = _bound_from_comparison(
            target_on_left=True,
            operator=operator,
            value=_parse_bound_value(rhs_text),
        )
    else:
        target_names = _expand_bound_target_names(rhs_text, available_indexed_names)
        bound = _bound_from_comparison(
            target_on_left=False,
            operator=operator,
            value=_parse_bound_value(lhs_text),
        )

    return {name: bound for name in target_names}


def _is_bound_identifier(text: str) -> bool:
    return bool(re.fullmatch(_IDENTIFIER_PATTERN, text.strip(), re.UNICODE))


def _expand_bound_target_names(
    name: str,
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    target_name = name.strip()
    if _is_indexed_identifier(target_name):
        return (target_name,)
    return available_indexed_names.get(target_name, (target_name,))


def _parse_bound_value(text: str) -> float:
    value = parse_expr(
        text.strip(),
        local_dict=_function_locals(),
        transformations=_TRANSFORMATIONS,
    )
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"Bound value `{text}` must evaluate to a finite number.")
    return numeric_value


def _expression_contains_obc(expr: sp.Expr) -> bool:
    return bool(expr.has(sp.Max, sp.Min))


def _build_symbolic_steady_state_seed_entries(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    parameter_symbols: Sequence[sp.Symbol],
) -> tuple[tuple[int, sp.Expr], ...]:
    remaining_equations = [sp.simplify(expr) for expr in equations]
    remaining_variables = list(variables)
    remaining_variable_set = set(remaining_variables)
    parameter_symbol_set = set(parameter_symbols)
    solved: dict[sp.Symbol, sp.Expr] = {}

    while remaining_equations and remaining_variables:
        best_candidate: Optional[tuple[int, sp.Symbol, sp.Expr, tuple[int, int, str]]] = None
        for eq_idx, equation in enumerate(remaining_equations):
            free_symbols = equation.free_symbols & remaining_variable_set
            if not free_symbols:
                continue
            for variable in tuple(remaining_variables):
                if variable not in free_symbols:
                    continue
                try:
                    solutions = sp.solve(sp.Eq(equation, 0), variable)
                except Exception:
                    continue
                normalized: list[sp.Expr] = []
                seen: set[str] = set()
                for solution in solutions:
                    simplified = sp.simplify(solution)
                    if variable in simplified.free_symbols:
                        continue
                    unresolved = simplified.free_symbols & remaining_variable_set
                    if unresolved - {variable}:
                        continue
                    if simplified.free_symbols - parameter_symbol_set - remaining_variable_set:
                        continue
                    key = sp.srepr(simplified)
                    if key in seen:
                        continue
                    seen.add(key)
                    normalized.append(simplified)
                if len(normalized) != 1:
                    continue
                candidate = normalized[0]
                score = (
                    len(candidate.free_symbols),
                    int(sp.count_ops(candidate, visual=False)),
                    str(variable),
                )
                if best_candidate is None or score < best_candidate[3]:
                    best_candidate = (eq_idx, variable, candidate, score)
        if best_candidate is None:
            break

        eq_idx, variable, expression, _ = best_candidate
        solved[variable] = expression
        remaining_variables.remove(variable)
        remaining_variable_set.remove(variable)

        next_equations: list[sp.Expr] = []
        for current_idx, equation in enumerate(remaining_equations):
            if current_idx == eq_idx:
                continue
            substituted = sp.simplify(equation.subs(variable, expression))
            if substituted != 0:
                next_equations.append(substituted)
        remaining_equations = next_equations

    return tuple(
        (idx, solved[symbol])
        for idx, symbol in enumerate(variables)
        if symbol in solved
    )


def _obc_calls_in_expression(expr: sp.Expr) -> tuple[sp.Expr, ...]:
    return tuple(
        node
        for node in sp.preorder_traversal(expr)
        if isinstance(node, sp.Expr) and node.func in (sp.Max, sp.Min)
    )


def _transform_obc_residual_to_violation_expressions(
    expr: sp.Expr,
) -> tuple[sp.Expr, ...]:
    obc_calls = _obc_calls_in_expression(expr)
    if not obc_calls:
        return ()
    if len(obc_calls) > 1:
        raise NotImplementedError(
            "OBC violation evaluation currently supports at most one min/max call per equation."
        )
    obc_call = obc_calls[0]
    if len(obc_call.args) != 2:
        raise NotImplementedError(
            "OBC violation evaluation currently supports binary min/max constraints only."
        )
    placeholder = sp.Symbol("__obc_placeholder__", real=True)
    substituted = expr.xreplace({obc_call: placeholder})
    solutions = sp.solve(sp.Eq(substituted, 0), placeholder)
    if not solutions:
        raise ValueError(
            f"Could not transform OBC residual into violation expressions: {expr}"
        )
    solution = sp.simplify(solutions[0])
    left = sp.simplify(obc_call.args[0] - solution)
    right = sp.simplify(obc_call.args[1] - solution)
    product = sp.simplify(left * right)
    if obc_call.func is sp.Max:
        return (product, left, right)
    return (product, sp.simplify(-left), sp.simplify(-right))


def _obc_branch_preference_score(
    expr: sp.Expr,
    preferred_symbols: frozenset[sp.Symbol],
) -> tuple[int, int, int, str]:
    free_symbols = expr.free_symbols
    preferred_count = sum(1 for symbol in free_symbols if symbol in preferred_symbols)
    return (
        preferred_count,
        len(free_symbols),
        int(sp.count_ops(expr, visual=False)),
        str(expr),
    )


def _evaluate_obc_expression_value(
    expr: sp.Expr,
    symbol_values: Mapping[sp.Symbol, float],
) -> float:
    numeric = complex(sp.N(expr.subs(symbol_values)))
    if not np.isfinite(numeric.real) or abs(numeric.imag) > 1e-10:
        raise ValueError(f"OBC branch expression did not evaluate to a finite real value: {expr}")
    return float(numeric.real)


def _coerce_optional_python_bool(value: object) -> Optional[bool]:
    try:
        return bool(np.asarray(value))
    except Exception:
        return None


def _freeze_obc_expression(
    expr: sp.Expr,
    *,
    symbol_values: Mapping[sp.Symbol, float],
    preferred_symbols: frozenset[sp.Symbol],
    tie_tolerance: float = 1e-10,
) -> sp.Expr:
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr

    resolved_args = tuple(
        _freeze_obc_expression(
            arg,
            symbol_values=symbol_values,
            preferred_symbols=preferred_symbols,
            tie_tolerance=tie_tolerance,
        )
        if isinstance(arg, sp.Basic)
        else arg
        for arg in expr.args
    )
    resolved_expr = expr.func(*resolved_args)
    if resolved_expr.func not in (sp.Max, sp.Min):
        return resolved_expr

    numeric_values = [
        _evaluate_obc_expression_value(arg, symbol_values) for arg in resolved_args
    ]
    target_value = (
        max(numeric_values) if resolved_expr.func is sp.Max else min(numeric_values)
    )
    candidate_args = [
        arg
        for arg, value in zip(resolved_args, numeric_values)
        if abs(value - target_value) <= tie_tolerance
    ]
    if len(candidate_args) == 1:
        return candidate_args[0]
    return min(
        candidate_args,
        key=lambda arg: _obc_branch_preference_score(arg, preferred_symbols),
    )


def _evaluate_symbolic_matrix(
    matrix: sp.Matrix,
    symbols: Sequence[sp.Symbol],
    values: Sequence[float],
) -> np.ndarray:
    fn = sp.lambdify(symbols, matrix, modules=_numpy_lambdify_modules())
    return np.asarray(fn(*values), dtype=np.float64)


def _coerce_symbolic_numeric_args(values: Sequence[object]) -> list[float]:
    return [float(np.asarray(value, dtype=np.float64)) for value in values]


def _bound_from_comparison(
    *,
    target_on_left: bool,
    operator: str,
    value: float,
) -> tuple[float, float]:
    if target_on_left:
        if operator == "<":
            return (-np.inf, _open_upper_bound(value))
        if operator == "<=":
            return (-np.inf, value)
        if operator == ">":
            return (_open_lower_bound(value), np.inf)
        if operator == ">=":
            return (value, np.inf)
    else:
        if operator == "<":
            return (_open_lower_bound(value), np.inf)
        if operator == "<=":
            return (value, np.inf)
        if operator == ">":
            return (-np.inf, _open_upper_bound(value))
        if operator == ">=":
            return (-np.inf, value)
    raise ValueError(f"Unsupported bound operator `{operator}`.")


def _open_lower_bound(value: float) -> float:
    return float(np.nextafter(value, np.inf))


def _open_upper_bound(value: float) -> float:
    return float(np.nextafter(value, -np.inf))


def _merge_bounds(
    current: Optional[tuple[float, float]],
    update: tuple[float, float],
) -> tuple[float, float]:
    if current is None:
        lower, upper = update
    else:
        lower = max(current[0], update[0])
        upper = min(current[1], update[1])
    if lower > upper:
        raise ValueError(f"Invalid bounds after merging: ({lower}, {upper}).")
    return (lower, upper)


def _is_parameter_bounds_line(line: str) -> bool:
    if not re.search(r"<=|>=|<|>", line):
        return False
    return re.search(r"(?<![<>=])=(?![=>])", line) is None


def _evaluate_integer_expression(text: str) -> int:
    value = parse_expr(
        text.strip(),
        local_dict=_function_locals(),
        transformations=_TRANSFORMATIONS,
    )
    numeric_value = float(value)
    rounded = int(round(numeric_value))
    if not np.isfinite(numeric_value) or abs(numeric_value - rounded) > 1e-12:
        raise ValueError(f"`{text}` does not evaluate to an integer loop bound.")
    return rounded


def _substitute_loop_variable(statement: str, loop_var: str, value: int | str) -> str:
    return re.sub(rf"\b{re.escape(loop_var)}\b", str(value), statement)


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0]


def _equation_to_difference(equation: str) -> str:
    if "=" not in equation:
        return f"({equation.strip()})"
    lhs, rhs = equation.split("=", 1)
    return f"({lhs.strip()}) - ({rhs.strip()})"


def _identifier_is_function_call(
    expression: str,
    match: re.Match[str],
) -> bool:
    return expression[match.end() :].lstrip().startswith("(")


def _identifier_is_numeric_exponent(
    expression: str,
    match: re.Match[str],
) -> bool:
    name = match.group(0)
    if name not in {"e", "E"}:
        return False
    previous = expression[match.start() - 1] if match.start() > 0 else ""
    following = expression[match.end() :]
    return bool(
        previous
        and (previous.isdigit() or previous == ".")
        and re.match(r"[+-]?\d", following)
    )


def _transform_dynamic_expression(
    expr_text: str,
    dynamic_texts: list[str],
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
    aux_equations_added: set[tuple[str, int]],
    shifted_shocks_added: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        base_name = match.group("name").strip()
        kind, offset = _parse_reference_index(match.group("index"))
        if kind == "steady":
            return _register_steady_state_token(base_name, timed_symbols, timed_metadata)
        if kind == "exo":
            exogenous_names.add(base_name)
            if offset == 0:
                return _register_exogenous_token(base_name, timed_symbols, timed_metadata)
            if base_name not in shifted_shocks_added:
                shifted_shocks_added.add(base_name)
                dynamic_texts.append(
                    f"{_register_time_token(base_name, 0, timed_symbols, timed_metadata)}"
                    f" - "
                    f"{_register_exogenous_token(base_name, timed_symbols, timed_metadata)}"
                )
            return _transform_endogenous_reference(
                base_name,
                offset,
                dynamic_texts,
                timed_symbols,
                timed_metadata,
                aux_equations_added,
            )
        return _transform_endogenous_reference(
            base_name,
            offset,
            dynamic_texts,
            timed_symbols,
            timed_metadata,
            aux_equations_added,
        )

    return _REFERENCE_RE.sub(replace, expr_text)


def _transform_steady_state_expression(
    expr_text: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        base_name = match.group("name").strip()
        kind, offset = _parse_reference_index(match.group("index"))
        if kind == "exo":
            exogenous_names.add(base_name)
            _ = offset
            return "0"
        return _register_steady_state_token(base_name, timed_symbols, timed_metadata)

    return _REFERENCE_RE.sub(replace, expr_text)


def _parse_reference_index(index_text: str) -> tuple[str, int]:
    normalized = index_text.strip().lower().replace(" ", "")
    if normalized in _STEADY_STATE_ALIASES:
        return "steady", 0
    if normalized in _EXOGENOUS_ALIASES:
        return "exo", 0
    for prefix in sorted(_EXOGENOUS_ALIASES, key=len, reverse=True):
        if normalized.startswith(prefix) and normalized[len(prefix) :] and re.fullmatch(
            r"[+-]\d+",
            normalized[len(prefix) :],
        ):
            return "exo", int(normalized[len(prefix) :])
    try:
        return "time", _evaluate_integer_expression(normalized)
    except ValueError:
        pass
    raise ValueError(f"Unsupported time index `{index_text}`.")


def _transform_endogenous_reference(
    base_name: str,
    offset: int,
    dynamic_texts: list[str],
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    aux_equations_added: set[tuple[str, int]],
) -> str:
    if -1 <= offset <= 1:
        return _register_time_token(base_name, offset, timed_symbols, timed_metadata)
    if offset > 1:
        for lead in range(offset - 1, 0, -1):
            key = (base_name, lead)
            if key not in aux_equations_added:
                aux_equations_added.add(key)
                current = _auxiliary_name(base_name, lead)
                nxt = _auxiliary_name(base_name, lead - 1) if lead > 1 else base_name
                dynamic_texts.append(
                    f"{_register_time_token(current, 0, timed_symbols, timed_metadata)}"
                    f" - "
                    f"{_register_time_token(nxt, 1, timed_symbols, timed_metadata)}"
                )
        return _register_time_token(
            _auxiliary_name(base_name, offset - 1),
            1,
            timed_symbols,
            timed_metadata,
        )
    for lag in range(abs(offset) - 1, 0, -1):
        key = (base_name, -lag)
        if key not in aux_equations_added:
            aux_equations_added.add(key)
            current = _auxiliary_name(base_name, -lag)
            nxt = _auxiliary_name(base_name, -(lag - 1)) if lag > 1 else base_name
            dynamic_texts.append(
                f"{_register_time_token(current, 0, timed_symbols, timed_metadata)}"
                f" - "
                f"{_register_time_token(nxt, -1, timed_symbols, timed_metadata)}"
            )
    return _register_time_token(
        _auxiliary_name(base_name, offset + 1),
        -1,
        timed_symbols,
        timed_metadata,
    )


def _register_time_token(
    name: str,
    offset: int,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
) -> str:
    token = _time_token_name(name, offset)
    if token not in timed_symbols:
        timed_symbols[token] = sp.Symbol(token, real=True)
        timed_metadata[token] = (name, "time", offset)
    return token


def _register_exogenous_token(
    name: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
) -> str:
    token = _exo_token_name(name)
    if token not in timed_symbols:
        timed_symbols[token] = sp.Symbol(token, real=True)
        timed_metadata[token] = (name, "exo", 0)
    return token


def _register_steady_state_token(
    name: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
) -> str:
    token = _steady_state_token_name(name)
    if token not in timed_symbols:
        timed_symbols[token] = sp.Symbol(token, real=True)
        timed_metadata[token] = (name, "steady", 0)
    return token


def _build_timings(
    timed_metadata: Mapping[str, tuple[str, str, int]],
) -> DSGETimings:
    dyn_var_future = sorted(
        {
            name
            for name, kind, offset in timed_metadata.values()
            if kind == "time" and offset > 0
        }
    )
    dyn_var_present = sorted(
        {
            name
            for name, kind, offset in timed_metadata.values()
            if kind == "time" and offset == 0
        }
    )
    dyn_var_past = sorted(
        {
            name
            for name, kind, offset in timed_metadata.values()
            if kind == "time" and offset < 0
        }
    )
    dyn_var_ss = sorted(
        {
            name
            for name, kind, _ in timed_metadata.values()
            if kind == "steady"
        }
    )
    exo = sorted(
        {
            name
            for name, kind, _ in timed_metadata.values()
            if kind == "exo"
        }
    )

    all_dyn_vars = set(dyn_var_future) | set(dyn_var_present) | set(dyn_var_past)
    if set(dyn_var_ss) - all_dyn_vars:
        raise ValueError(
            "The following variables are defined only in steady state: "
            + ", ".join(sorted(set(dyn_var_ss) - all_dyn_vars))
        )

    present_only = sorted(set(dyn_var_present) - set(dyn_var_past) - set(dyn_var_future))
    future_not_past = sorted(set(dyn_var_future) - set(dyn_var_past))
    past_not_future = sorted(set(dyn_var_past) - set(dyn_var_future))
    mixed = sorted(
        set(dyn_var_present)
        - set(present_only)
        - set(future_not_past)
        - set(past_not_future)
    )
    future_not_past_and_mixed = sorted(set(future_not_past) | set(mixed))
    past_not_future_and_mixed = sorted(set(past_not_future) | set(mixed))
    present_but_not_only = sorted(set(dyn_var_present) - set(present_only))
    mixed_in_past = sorted(set(dyn_var_past) & set(mixed))
    not_mixed_in_past = sorted(set(dyn_var_past) - set(mixed_in_past))
    mixed_in_future = sorted(set(dyn_var_future) & set(mixed))
    var = tuple(dyn_var_present)
    aux_tmp = sorted(name for name in dyn_var_present if _is_auxiliary_name(name))
    aux = tuple(name for name in aux_tmp if _strip_auxiliary_suffix(name) not in exo)
    exo_present = tuple(
        sorted(name for name in dyn_var_present if _strip_auxiliary_suffix(name) in exo)
    )

    if any(name not in dyn_var_present for name in future_not_past_and_mixed):
        missing = sorted(set(future_not_past_and_mixed) - set(dyn_var_present))
        raise ValueError(
            "The following variables appear in the future but not in the present: "
            + ", ".join(missing)
        )
    if any(name not in dyn_var_present for name in past_not_future_and_mixed):
        missing = sorted(set(past_not_future_and_mixed) - set(dyn_var_present))
        raise ValueError(
            "The following variables appear in the past but not in the present: "
            + ", ".join(missing)
        )

    present_only_idx = _index_positions(present_only, dyn_var_present)
    present_but_not_only_idx = _index_positions(present_but_not_only, dyn_var_present)
    future_not_past_and_mixed_idx = _index_positions(
        future_not_past_and_mixed,
        dyn_var_present,
    )
    past_not_future_and_mixed_idx = _index_positions(
        past_not_future_and_mixed,
        dyn_var_present,
    )
    mixed_in_future_idx = _index_positions(mixed_in_future, dyn_var_future)
    mixed_in_past_idx = _index_positions(mixed_in_past, dyn_var_past)
    not_mixed_in_past_idx = _index_positions(not_mixed_in_past, dyn_var_past)
    past_not_future_idx = _index_positions(past_not_future, dyn_var_present)
    reorder = _index_positions(
        dyn_var_present,
        present_only + past_not_future + future_not_past_and_mixed,
    )
    dynamic_order = _index_positions(
        present_but_not_only,
        past_not_future + future_not_past_and_mixed,
    )

    return DSGETimings(
        present_only=tuple(present_only),
        future_not_past=tuple(future_not_past),
        past_not_future=tuple(past_not_future),
        mixed=tuple(mixed),
        future_not_past_and_mixed=tuple(future_not_past_and_mixed),
        past_not_future_and_mixed=tuple(past_not_future_and_mixed),
        present_but_not_only=tuple(present_but_not_only),
        mixed_in_past=tuple(mixed_in_past),
        not_mixed_in_past=tuple(not_mixed_in_past),
        mixed_in_future=tuple(mixed_in_future),
        exo=tuple(exo),
        var=tuple(dyn_var_present),
        aux=aux,
        exo_present=exo_present,
        nPresent_only=len(present_only),
        nMixed=len(mixed),
        nFuture_not_past_and_mixed=len(future_not_past_and_mixed),
        nPast_not_future_and_mixed=len(past_not_future_and_mixed),
        nPresent_but_not_only=len(present_but_not_only),
        nVars=len(dyn_var_present),
        nExo=len(exo),
        present_only_idx=tuple(present_only_idx),
        present_but_not_only_idx=tuple(present_but_not_only_idx),
        future_not_past_and_mixed_idx=tuple(future_not_past_and_mixed_idx),
        not_mixed_in_past_idx=tuple(not_mixed_in_past_idx),
        past_not_future_and_mixed_idx=tuple(past_not_future_and_mixed_idx),
        mixed_in_past_idx=tuple(mixed_in_past_idx),
        mixed_in_future_idx=tuple(mixed_in_future_idx),
        past_not_future_idx=tuple(past_not_future_idx),
        reorder=tuple(reorder),
        dynamic_order=tuple(dynamic_order),
    )


def _extract_parameter_names(
    expressions: Sequence[str],
    timed_symbols: Mapping[str, sp.Symbol],
) -> set[str]:
    function_names = set(_function_locals()) | {"E", "pi"}
    parameters: set[str] = set()
    for expression in expressions:
        for match in _IDENTIFIER_RE.finditer(expression):
            name = match.group(0)
            if name in timed_symbols or name in function_names:
                continue
            if _identifier_is_function_call(expression, match):
                continue
            if _identifier_is_numeric_exponent(expression, match):
                continue
            parameters.add(name)
    return parameters


def _collect_unknown_function_names(
    expressions: Sequence[str],
    timed_symbols: Mapping[str, sp.Symbol],
    parameter_names: Sequence[str],
) -> tuple[str, ...]:
    function_names = set(_function_locals()) | {"E", "pi"}
    known_identifiers = set(timed_symbols) | set(parameter_names) | function_names
    unknown_functions: list[str] = []
    seen: set[str] = set()

    for expression in expressions:
        for match in _IDENTIFIER_RE.finditer(expression):
            name = match.group(0)
            if name in known_identifiers:
                continue
            if not _identifier_is_function_call(expression, match):
                continue
            if name in seen:
                continue
            seen.add(name)
            unknown_functions.append(name)
    return tuple(unknown_functions)


def _parse_parameter_block(
    body: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> ParsedParameterBlock:
    target_names: list[str] = []
    calibrated_target_names: list[str] = []
    equation_texts: list[str] = []
    direct_definition_texts: dict[str, str] = {}
    bounds: dict[str, tuple[float, float]] = {}
    seen_targets: set[str] = set()

    for line in _split_parameter_body_lines(body):
        if _is_parameter_bounds_line(line):
            for name, bound in _parse_parameter_bounds_line(
                line,
                available_indexed_names,
            ).items():
                bounds[name] = _merge_bounds(bounds.get(name), bound)
            continue
        if "=" not in line:
            continue
        for (
            target_name,
            calibrated_target_name,
            equation_text,
            direct_definition_text,
        ) in _parse_parameter_line(
            line,
            timed_symbols,
            timed_metadata,
            exogenous_names,
            available_indexed_names,
        ):
            if target_name in seen_targets:
                raise ValueError(f"Parameter `{target_name}` is defined more than once.")
            seen_targets.add(target_name)
            target_names.append(target_name)
            equation_texts.append(equation_text)
            if calibrated_target_name is not None:
                calibrated_target_names.append(calibrated_target_name)
            if direct_definition_text is not None:
                direct_definition_texts[target_name] = direct_definition_text

    return ParsedParameterBlock(
        target_names=tuple(target_names),
        calibrated_target_names=tuple(calibrated_target_names),
        equation_texts=tuple(equation_texts),
        initial_values=_initial_parameter_guesses(target_names, direct_definition_texts),
        bounds=bounds,
    )


def _parse_parameter_line(
    line: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> list[tuple[str, Optional[str], str, Optional[str]]]:
    lhs_text, rhs_text = (part.strip() for part in line.split("=", 1))
    if "|" in lhs_text:
        target_text, equation_lhs = (part.strip() for part in lhs_text.split("|", 1))
        return [
            (
                target_name,
                target_name,
                f"({transformed_lhs}) - ({transformed_rhs})",
                None,
            )
            for target_name, transformed_lhs, transformed_rhs in _expand_calibration_parameter_line(
                target_text,
                equation_lhs,
                rhs_text,
                timed_symbols,
                timed_metadata,
                exogenous_names,
                available_indexed_names,
            )
        ]
    if "|" in rhs_text:
        equation_rhs, target_text = (part.strip() for part in rhs_text.rsplit("|", 1))
        return [
            (
                target_name,
                target_name,
                f"({transformed_lhs}) - ({transformed_rhs})",
                None,
            )
            for target_name, transformed_lhs, transformed_rhs in _expand_calibration_parameter_line(
                target_text,
                lhs_text,
                equation_rhs,
                timed_symbols,
                timed_metadata,
                exogenous_names,
                available_indexed_names,
            )
        ]

    target_name = _validate_parameter_target(lhs_text)
    expanded_target_names = _expand_direct_parameter_targets(
        target_name,
        available_indexed_names,
    )
    transformed_rhs = _transform_parameter_expression(
        rhs_text,
        timed_symbols,
        timed_metadata,
        exogenous_names,
    )
    return [
        (
            expanded_target_name,
            None,
            f"({expanded_target_name}) - ({transformed_rhs})",
            rhs_text,
        )
        for expanded_target_name in expanded_target_names
    ]


def _validate_parameter_target(target_text: str) -> str:
    if not re.fullmatch(_IDENTIFIER_PATTERN, target_text, re.UNICODE):
        raise ValueError(
            "Parameter targets in `@parameters` must be identifiers with optional "
            "curly-brace indices, "
            f"got `{target_text}`."
        )
    return target_text


def _transform_parameter_expression(
    expr_text: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        base_name = match.group("name").strip()
        kind, _ = _parse_reference_index(match.group("index"))
        if kind == "steady":
            return _register_steady_state_token(base_name, timed_symbols, timed_metadata)
        if kind == "exo":
            exogenous_names.add(base_name)
            return "0"
        raise ValueError(
            "Expressions in `@parameters` may only use steady-state references "
            f"like `{base_name}[ss]`, got `{match.group(0)}`."
        )

    return _REFERENCE_RE.sub(replace, expr_text)


def _initial_parameter_guesses(
    target_names: Sequence[str],
    direct_definition_texts: Mapping[str, str],
) -> dict[str, float]:
    environment: dict[str, float] = {}
    unresolved = dict(direct_definition_texts)
    parse_name_map = {
        name: _parameter_parse_name(name)
        for name in target_names
        if _is_indexed_identifier(name)
    }

    progress = True
    while progress and unresolved:
        progress = False
        for name in list(unresolved):
            try:
                local_env = {
                    parse_name_map.get(key, key): value
                    for key, value in environment.items()
                }
                value = parse_expr(
                    _sanitize_indexed_identifiers(unresolved[name], parse_name_map),
                    local_dict={**local_env, **_function_locals()},
                    transformations=_TRANSFORMATIONS,
                )
                if getattr(value, "free_symbols", set()):
                    continue
                numeric_value = _evaluate_constant_parameter_expression(value)
            except Exception:
                continue
            if np.isfinite(numeric_value):
                environment[name] = numeric_value
                del unresolved[name]
                progress = True

    return {name: environment.get(name, 1.0) for name in target_names}


def _evaluate_constant_parameter_expression(expr: object) -> float:
    reduced = sp.sympify(expr)

    def is_implemented_function(node: sp.Basic) -> bool:
        return (
            getattr(node, "is_Function", False)
            and hasattr(node.func, "_imp_")
            and all(not getattr(arg, "free_symbols", set()) for arg in node.args)
        )

    def evaluate_implemented_function(node: sp.Basic) -> sp.Float:
        numeric_args = [float(sp.N(arg)) for arg in node.args]
        return sp.Float(node.func._imp_(*numeric_args))

    while True:
        updated = reduced.replace(is_implemented_function, evaluate_implemented_function)
        if updated == reduced:
            break
        reduced = updated
    return float(sp.N(reduced))


def _available_indexed_parameter_names(
    expressions: Sequence[str],
    timed_symbols: Mapping[str, sp.Symbol],
    timed_metadata: Mapping[str, tuple[str, str, int]],
) -> dict[str, tuple[str, ...]]:
    function_names = set(_function_locals()) | {"E", "pi"}
    ordered_names: list[str] = []
    seen: set[str] = set()

    for name, kind, _ in timed_metadata.values():
        if kind not in {"time", "steady"}:
            continue
        base_name = _strip_auxiliary_suffix(name)
        if _is_indexed_identifier(base_name) and base_name not in seen:
            seen.add(base_name)
            ordered_names.append(base_name)

    for expression in expressions:
        for match in _IDENTIFIER_RE.finditer(expression):
            name = match.group(0)
            if name in timed_symbols or name in function_names:
                continue
            if _identifier_is_function_call(expression, match):
                continue
            if _is_indexed_identifier(name) and name not in seen:
                seen.add(name)
                ordered_names.append(name)

    grouped: dict[str, list[str]] = {}
    for name in ordered_names:
        grouped.setdefault(_identifier_base_name(name), []).append(name)
    return {base_name: tuple(names) for base_name, names in grouped.items()}


def _expand_direct_parameter_targets(
    target_name: str,
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if _is_indexed_identifier(target_name):
        return (target_name,)
    return available_indexed_names.get(target_name, (target_name,))


def _expand_calibration_parameter_line(
    target_text: str,
    equation_lhs_text: str,
    equation_rhs_text: str,
    timed_symbols: Mapping[str, sp.Symbol],
    timed_metadata: Mapping[str, tuple[str, str, int]],
    exogenous_names: set[str],
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> list[tuple[str, str, str]]:
    target_name = _validate_parameter_target(target_text)
    generic_bases = _generic_indexed_identifier_bases(
        (equation_lhs_text, equation_rhs_text),
        available_indexed_names,
    )
    suffix_groups: list[tuple[str, ...]] = []

    if not _is_indexed_identifier(target_name) and target_name in available_indexed_names:
        suffix_groups.append(_indexed_name_suffixes(available_indexed_names[target_name]))
    for base_name in generic_bases:
        suffixes = _indexed_name_suffixes(available_indexed_names[base_name])
        if suffixes not in suffix_groups:
            suffix_groups.append(suffixes)

    if len(suffix_groups) > 1:
        raise ValueError(
            "Calibration equations cannot have more than one indexed family in the "
            "equation or parameter target."
        )

    if _is_indexed_identifier(target_name):
        if not generic_bases:
            suffixes = (None,)
        else:
            target_suffix = _identifier_suffix(target_name)
            if suffix_groups and target_suffix not in suffix_groups[0]:
                raise ValueError(
                    "Explicitly indexed calibration targets must match the indexed "
                    "family used in the calibration equation."
                )
            suffixes = (target_suffix,)
    elif target_name in available_indexed_names:
        suffixes = suffix_groups[0] if suffix_groups else _indexed_name_suffixes(
            available_indexed_names[target_name]
        )
    elif suffix_groups:
        raise ValueError(
            "Calibration equations with indexed references require an indexed "
            "parameter target."
        )
    else:
        suffixes = (None,)

    expanded: list[tuple[str, str, str]] = []
    for suffix in suffixes:
        replacements = (
            {}
            if suffix is None
            else {
                base_name: _indexed_name_for_suffix(
                    base_name,
                    suffix,
                    available_indexed_names,
                )
                for base_name in generic_bases
            }
        )
        expanded_target_name = (
            target_name
            if suffix is None or _is_indexed_identifier(target_name)
            else _indexed_name_for_suffix(
                target_name,
                suffix,
                available_indexed_names,
            )
        )
        transformed_lhs = _transform_parameter_expression(
            _substitute_parameter_identifiers(equation_lhs_text, replacements),
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        transformed_rhs = _transform_parameter_expression(
            _substitute_parameter_identifiers(equation_rhs_text, replacements),
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        expanded.append((expanded_target_name, transformed_lhs, transformed_rhs))
    return expanded


def _generic_indexed_identifier_bases(
    expressions: Sequence[str],
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    ordered_bases: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        for name in _IDENTIFIER_RE.findall(expression):
            if _is_indexed_identifier(name):
                continue
            if name not in available_indexed_names or name in seen:
                continue
            seen.add(name)
            ordered_bases.append(name)
    return tuple(ordered_bases)


def _indexed_name_suffixes(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(_identifier_suffix(name) for name in names)


def _identifier_base_name(name: str) -> str:
    return name.split("{", 1)[0]


def _identifier_suffix(name: str) -> str:
    return name[len(_identifier_base_name(name)) :]


def _indexed_name_for_suffix(
    base_name: str,
    suffix: str,
    available_indexed_names: Mapping[str, tuple[str, ...]],
) -> str:
    for name in available_indexed_names.get(base_name, ()):
        if _identifier_suffix(name) == suffix:
            return name
    raise ValueError(
        f"Could not expand indexed name for `{base_name}` with suffix `{suffix}`."
    )


def _substitute_parameter_identifiers(
    expression: str,
    replacements: Mapping[str, str],
) -> str:
    if not replacements:
        return expression
    return _IDENTIFIER_RE.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        expression,
    )


def _flatten_hessian(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> list[sp.Expr]:
    hessian = sp.hessian(expr, symbols)
    return [hessian[i, j] for j in range(len(symbols)) for i in range(len(symbols))]


def _flatten_third_order(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> list[sp.Expr]:
    third_order: list[sp.Expr] = []
    for k in range(len(symbols)):
        for j in range(len(symbols)):
            for i in range(len(symbols)):
                third_order.append(sp.diff(expr, symbols[i], symbols[j], symbols[k]))
    return third_order


def _qmipf_solve_ss(
    zeta: float,
    zeta_st: float,
    s_gy: float,
    s_gy_st: float,
    ss_y: float,
    ss_c: float,
    chi: float,
    sigma: float,
    eta_0: float,
    k_n_st: float,
    alpha: float,
    theta_p: float,
    chi_0_st: float,
    ss_tau_n_st: float,
    ss_tau_c_st: float,
    varkappa_st: float,
    ss_nu_st: float,
    ss_n: float,
    theta_w: float,
    tau_p: float,
    tau_w: float,
) -> float:
    values = np.asarray(
        [
            zeta,
            zeta_st,
            s_gy,
            s_gy_st,
            ss_y,
            ss_c,
            chi,
            sigma,
            eta_0,
            k_n_st,
            alpha,
            theta_p,
            chi_0_st,
            ss_tau_n_st,
            ss_tau_c_st,
            varkappa_st,
            ss_nu_st,
            ss_n,
            theta_w,
            tau_p,
            tau_w,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("QMIPF_solve_SS requires only finite numeric arguments.")

    exponent = chi * sigma
    if zeta_st == 0.0 or chi_0_st == 0.0 or 1 + theta_p == 0.0 or 1 + theta_w == 0.0:
        raise ValueError("QMIPF_solve_SS received invalid zero-denominator arguments.")
    if 1 - varkappa_st - ss_nu_st == 0.0:
        raise ValueError("QMIPF_solve_SS received a singular denominator.")

    constant_term = (
        ((1 + tau_p) * (1 + tau_w) * (1 - alpha) / (1 + theta_p))
        * (k_n_st**alpha)
        * (1 / chi_0_st / (1 + theta_w))
        * (1 - ss_tau_n_st)
        / (1 + ss_tau_c_st)
    ) ** sigma / (1 - varkappa_st - ss_nu_st)

    def objective(candidate: float) -> float:
        if not np.isfinite(candidate) or candidate <= 0.0:
            return np.inf
        return (
            zeta / zeta_st * ((1 - s_gy) * ss_y - ss_c) * candidate**exponent
            + (1 - s_gy_st + eta_0 * s_gy_st) * (k_n_st**alpha) * candidate ** (exponent + 1.0)
            - constant_term
        )

    initial = float(ss_n) if np.isfinite(ss_n) and ss_n > 0.0 else 0.3
    initial_value = objective(initial)
    if np.isfinite(initial_value) and abs(initial_value) < 1e-12:
        return initial

    lower = max(1e-10, initial * 0.5)
    upper = max(1.0, initial * 2.0)
    lower_value = objective(lower)
    upper_value = objective(upper)

    for _ in range(50):
        if np.isfinite(lower_value) and lower_value == 0.0:
            return lower
        if np.isfinite(upper_value) and upper_value == 0.0:
            return upper
        if (
            np.isfinite(lower_value)
            and np.isfinite(upper_value)
            and np.sign(lower_value) != np.sign(upper_value)
        ):
            result = scipy_optimize.root_scalar(
                objective,
                bracket=(lower, upper),
                method="brentq",
            )
            if result.converged and np.isfinite(result.root) and result.root > 0.0:
                return float(result.root)
            break
        lower = max(1e-12, lower * 0.5)
        upper *= 2.0
        lower_value = objective(lower)
        upper_value = objective(upper)

    secant_upper = initial * 1.1 if initial > 0.0 else 0.33
    if secant_upper <= initial:
        secant_upper = initial + 0.1
    result = scipy_optimize.root_scalar(
        objective,
        x0=initial,
        x1=secant_upper,
        method="secant",
        maxiter=200,
    )
    if result.converged and np.isfinite(result.root) and result.root > 0.0:
        return float(result.root)
    raise RuntimeError("QMIPF_solve_SS failed to find a positive root.")


def _function_locals() -> dict[str, object]:
    normcdf = lambda x: sp.Rational(1, 2) * (1 + sp.erf(x / sp.sqrt(2)))
    norminv = lambda x: sp.sqrt(2) * sp.erfinv(2 * x - 1)
    normpdf = lambda x: sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
    normlogpdf = lambda x: -(x**2) / 2 - sp.log(sp.sqrt(2 * sp.pi))
    qmipf_solve_ss = implemented_function("QMIPF_solve_SS", _qmipf_solve_ss)
    qmipf_solve_ss_lower = implemented_function("QMIPF_solve_ss", _qmipf_solve_ss)
    return {
        "abs": sp.Abs,
        "dnorm": normpdf,
        "erfinv": sp.erfinv,
        "erfcinv": sp.erfcinv,
        "exp": sp.exp,
        "log": sp.log,
        "max": sp.Max,
        "min": sp.Min,
        "normcdf": normcdf,
        "normlogpdf": normlogpdf,
        "normpdf": normpdf,
        "norminvcdf": norminv,
        "norminv": norminv,
        "pnorm": normcdf,
        "QMIPF_solve_SS": qmipf_solve_ss,
        "QMIPF_solve_ss": qmipf_solve_ss_lower,
        "qnorm": norminv,
        "sqrt": sp.sqrt,
    }


def _numpy_lambdify_modules() -> list[object]:
    return [
        {
            "QMIPF_solve_SS": _qmipf_solve_ss,
            "QMIPF_solve_ss": _qmipf_solve_ss,
            "erfcinv": scipy_special.erfcinv,
            "erfinv": scipy_special.erfinv,
        },
        "numpy",
    ]


def _jax_lambdify_modules() -> list[object]:
    return [
        {
            "QMIPF_solve_SS": _qmipf_solve_ss,
            "QMIPF_solve_ss": _qmipf_solve_ss,
            "erf": jsp_special.erf,
            "erfc": jsp_special.erfc,
            "erfcinv": lambda x: jsp_special.erfinv(1 - x),
            "erfinv": jsp_special.erfinv,
        },
        "jax",
    ]


def _is_indexed_identifier(name: str) -> bool:
    return "{" in name and "}" in name


def _parameter_parse_name(name: str) -> str:
    return f"par__{_encode_name(name)}"


def _sanitize_indexed_identifiers(
    expression: str,
    replacements: Mapping[str, str],
) -> str:
    if not replacements:
        return expression
    return _INDEXED_IDENTIFIER_RE.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        expression,
    )


def _encode_name(name: str) -> str:
    encoded = []
    for char in name:
        if ("a" <= char <= "z") or ("A" <= char <= "Z") or ("0" <= char <= "9") or char == "_":
            encoded.append(char)
        else:
            encoded.append(f"_u{ord(char):04x}_")
    if not encoded:
        return "sym"
    if encoded[0][0].isdigit():
        encoded.insert(0, "s_")
    return "".join(encoded)


def _time_token_name(name: str, offset: int) -> str:
    prefix = _encode_name(name)
    if offset == 0:
        suffix = "t0"
    elif offset > 0:
        suffix = f"tp{offset}"
    else:
        suffix = f"tm{abs(offset)}"
    return f"{prefix}__{suffix}"


def _exo_token_name(name: str) -> str:
    return f"{_encode_name(name)}__x"


def _steady_state_token_name(name: str) -> str:
    return f"{_encode_name(name)}__ss"


def _auxiliary_name(name: str, offset: int) -> str:
    return f"{name}__L{offset}"


def _is_auxiliary_name(name: str) -> bool:
    return "__L" in name


def _strip_auxiliary_suffix(name: str) -> str:
    return name.split("__L", 1)[0]


def _strip_selector_prefix(name: str) -> str:
    token = str(name).strip()
    return token[1:] if token.startswith(":") else token


def _flatten_named_selection(
    values: object,
) -> tuple[str, ...]:
    flattened: list[str] = []

    def visit(current: object) -> bool:
        if isinstance(current, str):
            flattened.append(_strip_selector_prefix(current))
            return True
        if isinstance(current, (list, tuple)):
            success = True
            for item in current:
                success = visit(item) and success
            return success
        return False

    if not visit(values):
        return tuple()

    deduped: list[str] = []
    seen: set[str] = set()
    for name in flattened:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return tuple(deduped)


def _index_positions(values: Sequence[str], reference: Sequence[str]) -> list[int]:
    mapping = {value: idx for idx, value in enumerate(reference)}
    return [mapping[value] for value in values]


def _iter_metadata_by_name(
    timed_metadata: Mapping[str, tuple[str, str, int]],
) -> list[tuple[str, str, int]]:
    seen: dict[tuple[str, str, int], None] = {}
    for name, kind, offset in timed_metadata.values():
        seen[(name, kind, offset)] = None
    return list(sorted(seen, key=lambda item: (item[0], item[1], item[2])))
