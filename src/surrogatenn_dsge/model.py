from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import re
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import scipy.special as scipy_special
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .dsge import (
    DSGETimings,
    FirstOrderDSGEResult,
    SecondOrderDSGEResult,
    SecondOrderStochasticSteadyStateResult,
    ThirdOrderDSGEResult,
    ThirdOrderStochasticSteadyStateResult,
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
    sep_inversion_loglikelihood,
    sep_inversion_loglikelihood_per_period,
)
from .sep import (
    SEPConfig,
    SEPSolution,
    solve_stochastic_extended_path_residual_expectation,
)
from .statespace import (
    LinearGaussianStateSpace,
    kalman_loglikelihood as _statespace_kalman_loglikelihood,
    kalman_loglikelihood_per_period as _statespace_kalman_loglikelihood_per_period,
)
from .switching import (
    SwitchingLikelihoodConfig,
    SwitchingLikelihoodResult,
    compute_switching_loglikelihood,
)

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

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
_REFERENCE_RE = re.compile(
    r"(?P<name>(?!\d)\w+(?:\{[^{}\[\]]+\})*)\s*\[\s*(?P<index>[^\]]+)\s*\]",
    re.UNICODE,
)
_IDENTIFIER_RE = re.compile(r"(?!\d)\w+(?:\{[^{}\[\]]+\})*", re.UNICODE)
_INDEXED_IDENTIFIER_RE = re.compile(r"(?!\d)\w+(?:\{[^{}\[\]]+\})+", re.UNICODE)

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


@dataclass(frozen=True)
class MacroModel:
    name: str
    equations: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: jax.Array
    calibrated_parameter_names: tuple[str, ...]
    default_initial_guess: dict[str, float]
    bounds: dict[str, tuple[float, float]]
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
        if initial_guess is None:
            if not self.default_initial_guess:
                return np.ones(n, dtype=np.float64)
            guess = np.ones(n, dtype=np.float64)
            for idx, name in enumerate(self.steady_state_names):
                if name in self.default_initial_guess:
                    guess[idx] = float(self.default_initial_guess[name])
            return guess
        if isinstance(initial_guess, Mapping):
            guess = np.ones(n, dtype=np.float64)
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
        if values.shape != (periods, self.timings.nExo):
            raise ValueError(
                "deterministic_shocks must have shape "
                f"({periods}, {self.timings.nExo}), got {values.shape}."
            )
        return jnp.asarray(values, dtype=jnp.float64)

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
        args = (
            tuple(lead_state[idx] for idx in self.timings.future_not_past_and_mixed_idx)
            + tuple(current_state[idx] for idx in range(self.timings.nVars))
            + tuple(lag_state[idx] for idx in self.timings.past_not_future_and_mixed_idx)
            + tuple(shock[idx] for idx in range(self.timings.nExo))
            + tuple(
                steady_reference_values[idx]
                for idx in range(len(self.steady_state_reference_names))
            )
            + tuple(parameter_values[idx] for idx in range(len(self.parameter_names)))
        )
        return jnp.asarray(self._dynamic_residual_fn(*args), dtype=jnp.float64).reshape(-1)

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
            base_steady_state = np.ones(len(self.steady_state_names), dtype=np.float64)
            residual_fns = (
                lambda x: np.asarray(
                    self._parameter_equation_fn(*base_steady_state, *x),
                    dtype=np.float64,
                ).reshape(-1),
            )
            jacobian_fns = (
                lambda x: np.asarray(
                    self._parameter_equation_jacobian_fn(*base_steady_state, *x),
                    dtype=np.float64,
                ),
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
            residual_fns = (
                lambda x: np.concatenate(
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
                ),
            )
            jacobian_fns = (
                lambda x: np.concatenate(
                    [
                        np.asarray(
                            self._steady_state_parameter_jacobian_fn(
                                *base_steady_state,
                                *x,
                            ),
                            dtype=np.float64,
                        ),
                        np.asarray(
                            self._parameter_equation_jacobian_fn(*base_steady_state, *x),
                            dtype=np.float64,
                        ),
                    ],
                    axis=0,
                ),
            )

        resolved, _, _, _ = _solve_newton_system(
            parameters,
            residual_fn=residual_fns[0],
            jacobian_fn=jacobian_fns[0],
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
            base_steady_state = jnp.ones(
                (len(self.steady_state_names),),
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

        resolved, _, _, _ = _solve_newton_system_jax(
            parameters,
            residual_fn=residual_fn,
            jacobian_fn=jax.jacrev(residual_fn),
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

            def jacobian_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._joint_steady_state_jacobian_fn(*x),
                    dtype=np.float64,
                )

            solution, converged, iterations, residual_norm = _solve_newton_system(
                joint_initial,
                residual_fn=residual_fn,
                jacobian_fn=jacobian_fn,
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

            def residual_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._steady_state_fn(*x, *resolved_parameters),
                    dtype=np.float64,
                ).reshape(-1)

            def jacobian_fn(x: np.ndarray) -> np.ndarray:
                return np.asarray(
                    self._steady_state_jacobian_fn(*x, *resolved_parameters),
                    dtype=np.float64,
                )

            base_steady_state, converged, iterations, residual_norm = _solve_newton_system(
                guess,
                residual_fn=residual_fn,
                jacobian_fn=jacobian_fn,
                lower_bounds=self._bounds_vector(self.steady_state_names)[0],
                upper_bounds=self._bounds_vector(self.steady_state_names)[1],
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
                nonfinite_message="Initial steady-state guess produced non-finite residuals.",
            )

        full = self._expand_to_full_steady_state(base_steady_state)
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

            solution, converged, iterations, residual_norm = _solve_newton_system_jax(
                joint_initial,
                residual_fn=residual_fn,
                jacobian_fn=jax.jacrev(residual_fn),
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

            def residual_fn(x: jax.Array) -> jax.Array:
                return jnp.asarray(
                    self._steady_state_residual_jax_fn(*x, *resolved_parameters),
                    dtype=jnp.float64,
                ).reshape(-1)

            base_steady_state, converged, iterations, residual_norm = _solve_newton_system_jax(
                guess,
                residual_fn=residual_fn,
                jacobian_fn=jax.jacrev(residual_fn),
                lower_bounds=self._bounds_vector(self.steady_state_names)[0],
                upper_bounds=self._bounds_vector(self.steady_state_names)[1],
                tol=tol,
                max_iter=max_iter,
                line_search_min_step=line_search_min_step,
            )
        full = self._expand_to_full_steady_state_jax(base_steady_state)
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
                else self._coerce_observable_names(observables)
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
            observable_names = self._coerce_observable_names(observables)
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
        resolved_parameters = (
            provided_parameters
            if provided_parameters is not None
            else np.asarray(
                self.resolve_parameter_values(steady_state=full_steady_state),
                dtype=np.float64,
            )
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
        solution = solve_first_order_dsge_solution(jacobian, self.timings)
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
        )
        if parsed_result is None or full_steady_state is None:
            return None, full_steady_state
        return (
            self.build_linear_state_space(
                observable_names,
                first_order_result=parsed_result,
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
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: bool = False,
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
        initial_state: Optional[Sequence[float]] = None,
        terminal_state: Optional[Sequence[float]] = None,
        config: SEPConfig = SEPConfig(),
        sep_periods: Optional[int] = None,
        sep_order: Optional[int] = None,
        sep_nnodes: Optional[int] = None,
        sep_sparse_tree: bool = False,
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
        sep_sparse_tree: bool = False,
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

        parameter_array = jnp.asarray(resolved_parameters, dtype=jnp.float64)
        steady_reference_values = jnp.asarray(
            self._steady_reference_values(full_steady_state),
            dtype=jnp.float64,
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

        solution = solve_stochastic_extended_path_residual_expectation(
            conditional_residual,
            initial_state=initial_state_values,
            terminal_state=terminal_state_values,
            shock_dim=self.timings.nExo,
            config=config,
            deterministic_shocks=deterministic_shock_values,
            initial_guess=initial_guess,
        )

        return ParsedModelSEPResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=parameter_array,
            solution=solution,
        )

    def solve_first_order(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
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
        solution = solve_first_order_dsge_solution(jacobian, self.timings)
        return ParsedModelFirstOrderResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
            parameter_values=jnp.asarray(resolved_parameters, dtype=jnp.float64),
            jacobian=jacobian,
            solution=solution,
        )

    def solve_second_order(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
        steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        steady_state_tol: float = 1e-12,
        steady_state_max_iter: int = 100,
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
        first_order_solution = solve_first_order_dsge_solution(jacobian, self.timings)
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
        first_order_solution = solve_first_order_dsge_solution(jacobian, self.timings)
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
    raw_default_guess = _parse_parameter_block_guess(parameter_block["options"])
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
    parse_locals = {**timed_symbols, **parameter_symbol_map, **_function_locals()}

    dynamic_exprs = tuple(
        parse_expr(
            _sanitize_indexed_identifiers(text, parameter_name_map),
            local_dict=parse_locals,
            transformations=_TRANSFORMATIONS,
        )
        for text in dynamic_texts
    )
    steady_state_exprs = tuple(
        parse_expr(
            _sanitize_indexed_identifiers(text, parameter_name_map),
            local_dict=parse_locals,
            transformations=_TRANSFORMATIONS,
        )
        for text in steady_state_texts
    )
    parameter_exprs = tuple(
        parse_expr(
            _sanitize_indexed_identifiers(text, parameter_name_map),
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

    return MacroModel(
        name=model_block["name"],
        equations=equations,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        calibrated_parameter_names=tuple(
            sorted(parsed_parameter_block.calibrated_target_names)
        ),
        default_initial_guess=default_initial_guess,
        bounds=parsed_parameter_block.bounds,
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
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: bool = False,
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
    initial_state: Optional[Sequence[float]] = None,
    terminal_state: Optional[Sequence[float]] = None,
    config: SEPConfig = SEPConfig(),
    sep_periods: Optional[int] = None,
    sep_order: Optional[int] = None,
    sep_nnodes: Optional[int] = None,
    sep_sparse_tree: bool = False,
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
    sep_sparse_tree: bool = False,
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


def solve_first_order_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
) -> ParsedModelFirstOrderResult:
    return model.solve_first_order(
        parameter_values=parameter_values,
        steady_state=steady_state,
        steady_state_initial_guess=steady_state_initial_guess,
        steady_state_tol=steady_state_tol,
        steady_state_max_iter=steady_state_max_iter,
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


def solve_second_order_model(
    model: MacroModel,
    *,
    parameter_values: Optional[Sequence[float]] = None,
    steady_state: Optional[Sequence[float]] = None,
    steady_state_initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
    steady_state_tol: float = 1e-12,
    steady_state_max_iter: int = 100,
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
            direction, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)

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
        match = re.fullmatch(r"(?P<name>(?!\d)\w+)\s*=\s*(?P<value>.+)", line)
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

        if loop_depth == 0 and not _model_line_requires_continuation(stripped):
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


def _model_line_requires_continuation(line: str) -> bool:
    return bool(re.search(r"[+\-*/=]$", line))


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

    header_match = re.match(r"(?P<var>(?!\d)\w+)\s+in\s+(?P<rest>.*)", header_rest)
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
        if re.fullmatch(r"(?!\d)\w+(?:\{[^{}\[\]]+\})*", stripped):
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


def _parse_guess_key(text: str) -> str:
    token = text.strip()
    if token.startswith(":") and len(token) > 1:
        return token[1:]
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    if re.fullmatch(r"(?!\d)\w+(?:\{[^{}\[\]]+\})*", token):
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
    return bool(re.fullmatch(r"(?!\d)\w+(?:\{[^{}\[\]]+\})*", text.strip()))


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
        raise ValueError(f"Equation must contain `=`, got `{equation}`.")
    lhs, rhs = equation.split("=", 1)
    return f"({lhs.strip()}) - ({rhs.strip()})"


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
        for name in _IDENTIFIER_RE.findall(expression):
            if name in timed_symbols or name in function_names:
                continue
            parameters.add(name)
    return parameters


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

    for line in _split_body_lines(body):
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
    if not re.fullmatch(r"(?!\d)\w+(?:\{[^{}\[\]]+\})*", target_text):
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
                numeric_value = float(value)
            except Exception:
                continue
            if np.isfinite(numeric_value):
                environment[name] = numeric_value
                del unresolved[name]
                progress = True

    return {name: environment.get(name, 1.0) for name in target_names}


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
        for name in _IDENTIFIER_RE.findall(expression):
            if name in timed_symbols or name in function_names:
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


def _function_locals() -> dict[str, object]:
    normcdf = lambda x: sp.Rational(1, 2) * (1 + sp.erf(x / sp.sqrt(2)))
    norminv = lambda x: sp.sqrt(2) * sp.erfinv(2 * x - 1)
    normpdf = lambda x: sp.exp(-(x**2) / 2) / sp.sqrt(2 * sp.pi)
    normlogpdf = lambda x: -(x**2) / 2 - sp.log(sp.sqrt(2 * sp.pi))
    return {
        "abs": sp.Abs,
        "dnorm": normpdf,
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
        "qnorm": norminv,
        "sqrt": sp.sqrt,
    }


def _numpy_lambdify_modules() -> list[object]:
    return [{"erfcinv": scipy_special.erfcinv}, "numpy"]


def _jax_lambdify_modules() -> list[object]:
    return [{"erfcinv": scipy_special.erfcinv}, "jax"]


def _is_indexed_identifier(name: str) -> bool:
    return "{" in name and "}" in name


def _parameter_parse_name(name: str) -> str:
    return f"par__{_encode_name(name)}" if _is_indexed_identifier(name) else name


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
        if char.isalnum() or char == "_":
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
