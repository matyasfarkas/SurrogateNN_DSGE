from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
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
    solve_first_order_dsge_solution,
    solve_second_order_dsge_solution,
    solve_second_order_stochastic_steady_state,
    solve_third_order_dsge_solution,
    solve_third_order_stochastic_steady_state,
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
    r"(?P<name>(?!\d)\w+(?:\{[^{}\[\]]+\})?)\s*\[\s*(?P<index>[^\]]+)\s*\]",
    re.UNICODE,
)
_IDENTIFIER_RE = re.compile(r"\b(?!\d)\w+\b", re.UNICODE)


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


class ParsedParameterBlock(NamedTuple):
    target_names: tuple[str, ...]
    calibrated_target_names: tuple[str, ...]
    equation_texts: tuple[str, ...]
    initial_values: dict[str, float]


@dataclass(frozen=True)
class MacroModel:
    name: str
    equations: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: jax.Array
    calibrated_parameter_names: tuple[str, ...]
    timings: DSGETimings
    steady_state_names: tuple[str, ...]
    steady_state_reference_names: tuple[str, ...]
    dynamic_symbol_names: tuple[str, ...]
    _dynamic_expressions: tuple[sp.Expr, ...]
    _steady_state_expressions: tuple[sp.Expr, ...]
    _parameter_symbols: tuple[sp.Symbol, ...]
    _steady_state_symbols: tuple[sp.Symbol, ...]
    _dynamic_symbols: tuple[sp.Symbol, ...]
    _dynamic_input_symbols: tuple[sp.Symbol, ...]
    _steady_state_fn: object
    _steady_state_jacobian_fn: object
    _steady_state_parameter_jacobian_fn: object
    _parameter_equations_depend_on_steady_state: bool
    _parameter_equation_fn: object
    _parameter_equation_jacobian_fn: object
    _joint_steady_state_fn: object
    _joint_steady_state_jacobian_fn: object
    _dynamic_jacobian_fn: object
    _dynamic_hessian_fn: object
    _dynamic_third_order_fn: object

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

    def _coerce_steady_state_guess(
        self,
        initial_guess: Optional[Sequence[float] | Mapping[str, float]],
    ) -> np.ndarray:
        n = len(self.steady_state_names)
        if initial_guess is None:
            return np.ones(n, dtype=np.float64)
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
            tol=tol,
            max_iter=max_iter,
            line_search_min_step=line_search_min_step,
            nonfinite_message="Initial parameter guess produced non-finite residuals.",
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
        initial_parameters = self._coerce_parameter_values(parameter_values)

        if self._parameter_equations_depend_on_steady_state:
            joint_initial = np.concatenate([guess, initial_parameters])

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

    equations = tuple(_split_model_body_lines(model_block["body"]))
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

    parsed_parameter_block = _parse_parameter_block(
        parameter_block["body"],
        timed_symbols,
        timed_metadata,
        exogenous_names,
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

    parameter_symbols = tuple(sp.Symbol(name, real=True) for name in parameter_names)
    parameter_symbol_map = dict(zip(parameter_names, parameter_symbols))
    parse_locals = {**timed_symbols, **parameter_symbol_map, **_function_locals()}

    dynamic_exprs = tuple(
        parse_expr(text, local_dict=parse_locals, transformations=_TRANSFORMATIONS)
        for text in dynamic_texts
    )
    steady_state_exprs = tuple(
        parse_expr(text, local_dict=parse_locals, transformations=_TRANSFORMATIONS)
        for text in steady_state_texts
    )
    parameter_exprs = tuple(
        parse_expr(text, local_dict=parse_locals, transformations=_TRANSFORMATIONS)
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

    steady_matrix = sp.Matrix(steady_state_exprs)
    steady_jacobian = steady_matrix.jacobian(steady_state_symbols)
    steady_parameter_jacobian = steady_matrix.jacobian(parameter_symbols)
    parameter_matrix = sp.Matrix(parameter_exprs)
    parameter_jacobian = parameter_matrix.jacobian(parameter_symbols)
    dynamic_matrix = sp.Matrix(dynamic_exprs)
    dynamic_jacobian = dynamic_matrix.jacobian(dynamic_symbols)
    dynamic_hessian = sp.Matrix(
        [_flatten_hessian(expr, dynamic_symbols) for expr in dynamic_exprs]
    )
    dynamic_third = sp.Matrix(
        [_flatten_third_order(expr, dynamic_symbols) for expr in dynamic_exprs]
    )
    joint_unknown_symbols = tuple(list(steady_state_symbols) + list(parameter_symbols))
    joint_matrix = sp.Matrix(list(steady_state_exprs) + list(parameter_exprs))
    joint_jacobian = joint_matrix.jacobian(joint_unknown_symbols)
    parameter_equations_depend_on_steady_state = any(
        bool(expr.free_symbols & set(steady_state_symbols))
        for expr in parameter_exprs
    )

    return MacroModel(
        name=model_block["name"],
        equations=equations,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        calibrated_parameter_names=tuple(
            sorted(parsed_parameter_block.calibrated_target_names)
        ),
        timings=timings,
        steady_state_names=steady_state_names,
        steady_state_reference_names=steady_state_reference_names,
        dynamic_symbol_names=dynamic_symbol_names,
        _dynamic_expressions=dynamic_exprs,
        _steady_state_expressions=steady_state_exprs,
        _parameter_symbols=parameter_symbols,
        _steady_state_symbols=steady_state_symbols,
        _dynamic_symbols=dynamic_symbols,
        _dynamic_input_symbols=dynamic_input_symbols,
        _steady_state_fn=sp.lambdify(
            list(steady_state_symbols) + list(parameter_symbols),
            steady_matrix,
            modules="numpy",
        ),
        _steady_state_jacobian_fn=sp.lambdify(
            list(steady_state_symbols) + list(parameter_symbols),
            steady_jacobian,
            modules="numpy",
        ),
        _steady_state_parameter_jacobian_fn=sp.lambdify(
            list(steady_state_symbols) + list(parameter_symbols),
            steady_parameter_jacobian,
            modules="numpy",
        ),
        _parameter_equations_depend_on_steady_state=parameter_equations_depend_on_steady_state,
        _parameter_equation_fn=sp.lambdify(
            list(steady_state_symbols) + list(parameter_symbols),
            parameter_matrix,
            modules="numpy",
        ),
        _parameter_equation_jacobian_fn=sp.lambdify(
            list(steady_state_symbols) + list(parameter_symbols),
            parameter_jacobian,
            modules="numpy",
        ),
        _joint_steady_state_fn=sp.lambdify(
            joint_unknown_symbols,
            joint_matrix,
            modules="numpy",
        ),
        _joint_steady_state_jacobian_fn=sp.lambdify(
            joint_unknown_symbols,
            joint_jacobian,
            modules="numpy",
        ),
        _dynamic_jacobian_fn=sp.lambdify(dynamic_input_symbols, dynamic_jacobian, modules="numpy"),
        _dynamic_hessian_fn=sp.lambdify(dynamic_input_symbols, dynamic_hessian, modules="numpy"),
        _dynamic_third_order_fn=sp.lambdify(dynamic_input_symbols, dynamic_third, modules="numpy"),
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
    tol: float,
    max_iter: int,
    line_search_min_step: float,
    nonfinite_message: str,
) -> tuple[np.ndarray, bool, int, float]:
    x = np.asarray(initial, dtype=np.float64)
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
            residual = np.asarray(residual_fn(x), dtype=np.float64).reshape(-1)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if not np.isfinite(residual_norm):
                break

    return x, residual_norm < tol, max_iter, residual_norm


def _extract_block(
    pattern: re.Pattern[str],
    source: str,
    label: str,
) -> dict[str, str]:
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Could not find `@{label}` block in source.")
    block = match.groupdict()
    body_lines: list[str] = []
    nested_depth = 0
    for raw_line in source[match.end() :].splitlines():
        visible = _strip_comment(raw_line).strip()
        if visible == "end" and nested_depth == 0:
            block["body"] = "\n".join(body_lines)
            return block
        body_lines.append(raw_line)
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


def _split_model_body_lines(body: str) -> list[str]:
    lines = _split_body_lines(body)
    statements: list[str] = []
    current: list[str] = []
    loop_depth = 0

    for line in lines:
        stripped = line.strip()
        current.append(stripped)
        loop_depth += _block_line_delta(stripped)
        if loop_depth < 0:
            raise ValueError("Encountered `end` without a matching `for` in `@model`.")

        if loop_depth == 0 and not _model_line_requires_continuation(stripped):
            if _is_unsupported_block_for_statement(current):
                raise NotImplementedError(
                    "Top-level symbolic/indexed `for`-loop blocks in `@model` are not "
                    "ported yet. Time-index expression loops are supported."
                )
            statements.append(_expand_inline_for_loops(" ".join(current)))
            current = []

    if loop_depth != 0:
        raise ValueError("Unbalanced `for` / `end` blocks in `@model`.")
    if current:
        if _is_unsupported_block_for_statement(current):
            raise NotImplementedError(
                "Top-level symbolic/indexed `for`-loop blocks in `@model` are not "
                "ported yet. Time-index expression loops are supported."
            )
        statements.append(_expand_inline_for_loops(" ".join(current)))
    return statements


def _model_line_requires_continuation(line: str) -> bool:
    return bool(re.search(r"[+\-*/=]$", line))


def _block_line_delta(line: str) -> int:
    if not line:
        return 0
    return len(re.findall(r"\bfor\b", line)) - len(re.findall(r"\bend\b", line))


def _is_unsupported_block_for_statement(lines: Sequence[str]) -> bool:
    return (
        bool(lines)
        and lines[0].startswith("for ")
        and lines[-1] == "end"
        and len(lines) > 1
    )


def _expand_inline_for_loops(statement: str) -> str:
    expanded = statement
    while re.search(r"\bfor\b", expanded):
        start, end = _find_innermost_for_segment(expanded)
        replacement = _expand_single_inline_for_loop(expanded[start:end])
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


def _expand_single_inline_for_loop(segment: str) -> str:
    segment_text = segment.strip()
    if not re.search(r"\bend\s*$", segment_text):
        raise ValueError(f"Could not find closing `end` in `for` loop `{segment}`.")
    loop_text = re.sub(r"\s*\bend\s*$", "", segment_text)
    rest = loop_text[len("for") :].strip()
    operator = "+"
    if rest.startswith("operator"):
        operator_match = re.match(
            r"operator\s*=\s*:(?P<op>[+*])\s*,\s*(?P<rest>.*)",
            rest,
        )
        if operator_match is None:
            raise ValueError(f"Could not parse `for`-loop operator in `{segment}`.")
        operator = operator_match.group("op")
        rest = operator_match.group("rest").strip()

    header_match = re.match(r"(?P<var>(?!\d)\w+)\s+in\s+(?P<rest>.*)", rest)
    if header_match is None:
        raise ValueError(f"Could not parse `for`-loop header in `{segment}`.")
    loop_var = header_match.group("var")
    indices_text, body = _consume_loop_indices(header_match.group("rest"))
    if not body:
        raise NotImplementedError(
            "Top-level symbolic/indexed `for`-loop blocks in `@model` are not "
            "ported yet. Time-index expression loops are supported."
        )

    indices = _parse_for_loop_indices(indices_text)
    terms = [
        _expand_inline_for_loops(_substitute_loop_variable(body, loop_var, idx))
        for idx in indices
    ]
    joiner = f" {operator} "
    return "(" + joiner.join(f"({term})" for term in terms) + ")"


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


def _parse_for_loop_indices(indices_text: str) -> list[int]:
    token = indices_text.strip()
    if token.startswith("[") and token.endswith("]"):
        entries = _split_top_level(token[1:-1], ",")
        try:
            return [_evaluate_integer_expression(entry) for entry in entries if entry.strip()]
        except ValueError as exc:
            raise NotImplementedError(
                "Symbolic/indexed `for` loops in `@model` are not ported yet."
            ) from exc

    range_parts = _split_top_level(token, ":")
    if len(range_parts) == 2:
        start = _evaluate_integer_expression(range_parts[0])
        stop = _evaluate_integer_expression(range_parts[1])
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    try:
        return [_evaluate_integer_expression(token)]
    except ValueError as exc:
        raise NotImplementedError(
            "Symbolic/indexed `for` loops in `@model` are not ported yet."
        ) from exc


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


def _substitute_loop_variable(statement: str, loop_var: str, value: int) -> str:
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
    if re.fullmatch(r"[+-]?\d+", normalized):
        return "time", int(normalized)
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
) -> ParsedParameterBlock:
    target_names: list[str] = []
    calibrated_target_names: list[str] = []
    equation_texts: list[str] = []
    direct_definition_texts: dict[str, str] = {}
    seen_targets: set[str] = set()

    for line in _split_body_lines(body):
        if "=" not in line:
            continue
        (
            target_name,
            calibrated_target_name,
            equation_text,
            direct_definition_text,
        ) = _parse_parameter_line(
            line,
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
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
    )


def _parse_parameter_line(
    line: str,
    timed_symbols: dict[str, sp.Symbol],
    timed_metadata: dict[str, tuple[str, str, int]],
    exogenous_names: set[str],
) -> tuple[str, Optional[str], str, Optional[str]]:
    lhs_text, rhs_text = (part.strip() for part in line.split("=", 1))
    if "|" in lhs_text:
        target_text, equation_lhs = (part.strip() for part in lhs_text.split("|", 1))
        target_name = _validate_parameter_target(target_text)
        transformed_lhs = _transform_parameter_expression(
            equation_lhs,
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        transformed_rhs = _transform_parameter_expression(
            rhs_text,
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        return (
            target_name,
            target_name,
            f"({transformed_lhs}) - ({transformed_rhs})",
            None,
        )
    if "|" in rhs_text:
        equation_rhs, target_text = (part.strip() for part in rhs_text.rsplit("|", 1))
        target_name = _validate_parameter_target(target_text)
        transformed_lhs = _transform_parameter_expression(
            lhs_text,
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        transformed_rhs = _transform_parameter_expression(
            equation_rhs,
            timed_symbols,
            timed_metadata,
            exogenous_names,
        )
        return (
            target_name,
            target_name,
            f"({transformed_lhs}) - ({transformed_rhs})",
            None,
        )

    target_name = _validate_parameter_target(lhs_text)
    transformed_rhs = _transform_parameter_expression(
        rhs_text,
        timed_symbols,
        timed_metadata,
        exogenous_names,
    )
    return (
        target_name,
        None,
        f"({target_name}) - ({transformed_rhs})",
        rhs_text,
    )


def _validate_parameter_target(target_text: str) -> str:
    if not re.fullmatch(r"(?!\d)\w+", target_text):
        raise ValueError(
            "Parameter targets in `@parameters` must be plain identifiers, "
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

    progress = True
    while progress and unresolved:
        progress = False
        for name in list(unresolved):
            try:
                value = parse_expr(
                    unresolved[name],
                    local_dict={**environment, **_function_locals()},
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
    return {
        "abs": sp.Abs,
        "exp": sp.exp,
        "log": sp.log,
        "max": sp.Max,
        "min": sp.Min,
        "normcdf": normcdf,
        "norminvcdf": norminv,
        "norminv": norminv,
        "qnorm": norminv,
        "sqrt": sp.sqrt,
    }


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
