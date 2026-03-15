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
    solve_first_order_dsge_solution,
    solve_second_order_dsge_solution,
    solve_second_order_stochastic_steady_state,
)

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

_STEADY_STATE_ALIASES = {"ss", "stst", "steady", "steadystate", "steady_state"}
_EXOGENOUS_ALIASES = {"x", "ex", "exo", "exogenous"}

_MODEL_BLOCK_RE = re.compile(
    r"@model\s+(?P<name>[^\s]+)(?P<options>.*?)\bbegin\b(?P<body>.*?)\bend\b",
    re.DOTALL,
)
_PARAMETERS_BLOCK_RE = re.compile(
    r"@parameters\s+(?P<name>[^\s]+)(?P<options>.*?)\bbegin\b(?P<body>.*?)\bend\b",
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
    converged: bool
    iterations: int
    residual_norm: float


class ParsedModelFirstOrderResult(NamedTuple):
    steady_state: jax.Array
    jacobian: jax.Array
    solution: FirstOrderDSGEResult


class ParsedModelSecondOrderResult(NamedTuple):
    steady_state: jax.Array
    jacobian: jax.Array
    hessian: jax.Array
    first_order_solution: FirstOrderDSGEResult
    second_order_solution: SecondOrderDSGEResult
    stochastic_steady_state: SecondOrderStochasticSteadyStateResult


@dataclass(frozen=True)
class MacroModel:
    name: str
    equations: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: jax.Array
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

    def solve_steady_state(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        initial_guess: Optional[Sequence[float] | Mapping[str, float]] = None,
        tol: float = 1e-12,
        max_iter: int = 100,
        line_search_min_step: float = 2.0**-16,
    ) -> SteadyStateResult:
        parameters = self._coerce_parameter_values(parameter_values)
        guess = self._coerce_steady_state_guess(initial_guess)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(self._steady_state_fn(*x, *parameters), dtype=np.float64).reshape(-1)

        def jacobian_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray(
                self._steady_state_jacobian_fn(*x, *parameters),
                dtype=np.float64,
            )

        x = guess
        residual = residual_fn(x)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if not np.isfinite(residual_norm):
            raise ValueError("Initial steady-state guess produced non-finite residuals.")

        for iteration in range(1, max_iter + 1):
            if residual_norm < tol:
                full = self._expand_to_full_steady_state(x)
                return SteadyStateResult(
                    steady_state=jnp.asarray(full, dtype=jnp.float64),
                    base_steady_state=jnp.asarray(x, dtype=jnp.float64),
                    converged=True,
                    iterations=iteration - 1,
                    residual_norm=residual_norm,
                )

            jacobian = jacobian_fn(x)
            try:
                direction = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                direction, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)

            step = 1.0
            accepted = False
            while step >= line_search_min_step:
                candidate = x + step * direction
                if np.isfinite(candidate).all():
                    candidate_residual = residual_fn(candidate)
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
                residual = residual_fn(x)
                residual_norm = float(np.linalg.norm(residual, ord=np.inf))
                if not np.isfinite(residual_norm):
                    break

        full = self._expand_to_full_steady_state(x)
        return SteadyStateResult(
            steady_state=jnp.asarray(full, dtype=jnp.float64),
            base_steady_state=jnp.asarray(x, dtype=jnp.float64),
            converged=residual_norm < tol,
            iterations=max_iter,
            residual_norm=residual_norm,
        )

    def _dynamic_evaluation_args(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> list[float]:
        parameters = self._coerce_parameter_values(parameter_values)
        full_steady_state = self._coerce_full_steady_state(
            steady_state,
            parameter_values=parameter_values,
        )
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
        args.extend(float(x) for x in parameters)
        return args

    def calculate_jacobian(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        values = np.asarray(
            self._dynamic_jacobian_fn(*self._dynamic_evaluation_args(
                parameter_values=parameter_values,
                steady_state=steady_state,
            )),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def calculate_hessian(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        values = np.asarray(
            self._dynamic_hessian_fn(*self._dynamic_evaluation_args(
                parameter_values=parameter_values,
                steady_state=steady_state,
            )),
            dtype=np.float64,
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def calculate_third_order_derivatives(
        self,
        *,
        parameter_values: Optional[Sequence[float]] = None,
        steady_state: Optional[Sequence[float]] = None,
    ) -> jax.Array:
        values = np.asarray(
            self._dynamic_third_order_fn(*self._dynamic_evaluation_args(
                parameter_values=parameter_values,
                steady_state=steady_state,
            )),
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
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
        jacobian = self.calculate_jacobian(
            parameter_values=parameter_values,
            steady_state=full_steady_state,
        )
        solution = solve_first_order_dsge_solution(jacobian, self.timings)
        return ParsedModelFirstOrderResult(
            steady_state=jnp.asarray(full_steady_state, dtype=jnp.float64),
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
        else:
            full_steady_state = self._coerce_full_steady_state(
                steady_state,
                parameter_values=parameter_values,
            )
        jacobian = self.calculate_jacobian(
            parameter_values=parameter_values,
            steady_state=full_steady_state,
        )
        hessian = self.calculate_hessian(
            parameter_values=parameter_values,
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
            jacobian=jacobian,
            hessian=hessian,
            first_order_solution=first_order_solution,
            second_order_solution=second_order_solution,
            stochastic_steady_state=stochastic_steady_state,
        )


def parse_macro_model(source: str) -> MacroModel:
    model_block = _extract_block(_MODEL_BLOCK_RE, source, "model")
    parameter_block = _extract_block(_PARAMETERS_BLOCK_RE, source, "parameters")
    if parameter_block["name"] != model_block["name"]:
        raise ValueError(
            "The `@parameters` block must target the same model name as `@model`."
        )

    equations = tuple(_split_body_lines(model_block["body"]))
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

    timings = _build_timings(timed_metadata)
    parameter_names = tuple(
        sorted(
            _extract_parameter_names(
                dynamic_texts + steady_state_texts,
                timed_symbols,
            )
        )
    )
    parameter_defaults = _parse_parameter_defaults(parameter_block["body"])
    missing = [name for name in parameter_names if name not in parameter_defaults]
    if missing:
        raise ValueError(
            "Missing parameter assignments for: " + ", ".join(sorted(missing))
        )
    parameter_values = jnp.asarray(
        [parameter_defaults[name] for name in parameter_names],
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
    dynamic_matrix = sp.Matrix(dynamic_exprs)
    dynamic_jacobian = dynamic_matrix.jacobian(dynamic_symbols)
    dynamic_hessian = sp.Matrix(
        [_flatten_hessian(expr, dynamic_symbols) for expr in dynamic_exprs]
    )
    dynamic_third = sp.Matrix(
        [_flatten_third_order(expr, dynamic_symbols) for expr in dynamic_exprs]
    )

    return MacroModel(
        name=model_block["name"],
        equations=equations,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
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


def _extract_block(
    pattern: re.Pattern[str],
    source: str,
    label: str,
) -> dict[str, str]:
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Could not find `@{label}` block in source.")
    return match.groupdict()


def _split_body_lines(body: str) -> list[str]:
    lines = []
    for raw_line in body.splitlines():
        line = _strip_comment(raw_line).strip()
        if line:
            lines.append(line)
    return lines


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


def _parse_parameter_defaults(body: str) -> dict[str, float]:
    assignments = {}
    environment: dict[str, float] = {}
    for line in _split_body_lines(body):
        if "|" in line:
            raise NotImplementedError(
                "Calibration equations in `@parameters` blocks are not ported yet."
            )
        if "=" not in line:
            continue
        name, expr = line.split("=", 1)
        parameter_name = name.strip()
        value = parse_expr(
            expr.strip(),
            local_dict={**environment, **_function_locals()},
            transformations=_TRANSFORMATIONS,
        )
        numeric_value = float(value)
        environment[parameter_name] = numeric_value
        assignments[parameter_name] = numeric_value
    return assignments


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
