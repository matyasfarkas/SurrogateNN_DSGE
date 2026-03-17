from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from surrogatenn_dsge import (
    calculate_jacobian,
    linear_state_space_from_first_order_solution,
    parse_macro_model,
    resolve_parameter_values,
    resolve_parameter_values_jax,
    simulate_linear_gaussian_state_space,
    solve_first_order_model,
    solve_steady_state,
    solve_steady_state_jax,
)


PARAMETER_DEFINITION_SOURCE = """
@model parameter_defs begin
    x[0] = zeta
end

@parameters parameter_defs begin
    zeta = eta + 1
    eta = Pi_bar / 2
    Pi_bar = 1.0025
end
"""


END_TARGET_CALIBRATION_SOURCE = """
@model end_target begin
    x[0] = a
end

@parameters end_target begin
    target = theta + 1
    x[ss] = target | a
    theta = 2
end
"""


RBC_CME_CALIBRATION_SOURCE = """
@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME verbose = true begin
    alpha | k[ss] / (4 * y[ss]) = cap_share
    cap_share = 1.66

    beta | R[ss] = R_ss
    R_ss = 1.0035

    delta = .0226

    Pibar | Pi[ss] = Pi_ss
    Pi_ss = R_ss - Pi_real
    Pi_real = 1/1000

    phi_pi = 1.5
    rhoz = 9 / 10
    std_eps = .0068
    rho_z_delta = rhoz
    std_z_delta = .005
end
"""


def _parameter_mapping(model, values) -> dict[str, float]:
    return dict(zip(model.parameter_names, np.asarray(values, dtype=np.float64).tolist()))


def test_parameter_definitions_resolve_in_arbitrary_order() -> None:
    model = parse_macro_model(PARAMETER_DEFINITION_SOURCE)

    resolved = resolve_parameter_values(model)
    resolved_map = _parameter_mapping(model, resolved)
    steady_state_result = solve_steady_state(model)

    np.testing.assert_allclose(resolved_map["Pi_bar"], 1.0025, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resolved_map["eta"], 0.50125, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resolved_map["zeta"], 1.50125, rtol=0, atol=1e-12)
    assert steady_state_result.converged
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        jnp.asarray([1.50125], dtype=jnp.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        steady_state_result.parameter_values,
        resolved,
        rtol=0,
        atol=1e-12,
    )


def test_end_target_calibration_syntax_resolves_parameter_from_steady_state() -> None:
    model = parse_macro_model(END_TARGET_CALIBRATION_SOURCE)

    resolved = resolve_parameter_values(model, steady_state=jnp.asarray([3.0], dtype=jnp.float64))
    resolved_map = _parameter_mapping(model, resolved)
    steady_state_result = solve_steady_state(model)

    np.testing.assert_allclose(resolved_map["theta"], 2.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resolved_map["target"], 3.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resolved_map["a"], 3.0, rtol=0, atol=1e-12)
    assert steady_state_result.converged
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        jnp.asarray([3.0], dtype=jnp.float64),
        rtol=0,
        atol=1e-12,
    )


def test_jax_end_target_calibration_matches_numpy_path() -> None:
    model = parse_macro_model(END_TARGET_CALIBRATION_SOURCE)

    resolved_jax = resolve_parameter_values_jax(
        model,
        steady_state=jnp.asarray([3.0], dtype=jnp.float64),
    )
    steady_state_jax = solve_steady_state_jax(model)
    steady_state_numpy = solve_steady_state(model)

    np.testing.assert_allclose(
        resolved_jax,
        resolve_parameter_values(model, steady_state=jnp.asarray([3.0], dtype=jnp.float64)),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        steady_state_jax.steady_state,
        steady_state_numpy.steady_state,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        steady_state_jax.parameter_values,
        steady_state_numpy.parameter_values,
        rtol=0,
        atol=1e-12,
    )


def test_rbc_calibration_equations_match_julia_fixture() -> None:
    model = parse_macro_model(RBC_CME_CALIBRATION_SOURCE)

    assert model.calibrated_parameter_names == ("Pibar", "alpha", "beta")

    steady_state_result = solve_steady_state(
        model,
        initial_guess={
            "A": 1.0,
            "Pi": 1.0025,
            "R": 1.0035,
            "c": 1.2,
            "k": 9.4,
            "y": 1.42,
            "z_delta": 1.0,
        },
    )

    assert steady_state_result.converged
    np.testing.assert_allclose(
        steady_state_result.steady_state,
        jnp.asarray(
            [
                1.0,
                1.0025000000000002,
                1.0035,
                1.2082172240118771,
                9.439019370210064,
                1.4215390617786243,
                1.0,
            ],
            dtype=jnp.float64,
        ),
        rtol=1e-9,
        atol=1e-9,
    )

    parameter_map = _parameter_mapping(model, steady_state_result.parameter_values)
    expected_parameters = {
        "Pi_real": 0.001,
        "Pibar": 1.0008326398517904,
        "Pi_ss": 1.0025,
        "R_ss": 1.0035,
        "alpha": 0.15668744139650842,
        "beta": 0.9990034877927255,
        "cap_share": 1.66,
        "delta": 0.0226,
        "phi_pi": 1.5,
        "rho_z_delta": 0.9,
        "rhoz": 0.9,
        "std_eps": 0.0068,
        "std_z_delta": 0.005,
    }
    for name, value in expected_parameters.items():
        np.testing.assert_allclose(parameter_map[name], value, rtol=1e-9, atol=1e-9)

    first_order_result = solve_first_order_model(
        model,
        steady_state_initial_guess={
            "A": 1.0,
            "Pi": 1.0025,
            "R": 1.0035,
            "c": 1.2,
            "k": 9.4,
            "y": 1.42,
            "z_delta": 1.0,
        },
    )

    np.testing.assert_allclose(
        first_order_result.steady_state,
        steady_state_result.steady_state,
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        first_order_result.parameter_values,
        steady_state_result.parameter_values,
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        first_order_result.solution.solution_matrix,
        jnp.asarray(
            [
                [0.9, -1.8735013540549517e-16, -4.5196157792000816e-17, 0.0, 0.0068],
                [
                    0.022362567671995476,
                    -0.0036606251688372483,
                    0.001213132036479368,
                    6.739622424885366e-6,
                    0.00016896162241063402,
                ],
                [
                    0.03357731170899882,
                    -0.0054964149978974815,
                    0.001821513214873372,
                    1.0119517860407679e-5,
                    0.000253695244023548,
                ],
                [
                    0.28751705126421906,
                    0.04970268320293412,
                    -0.06588592331797216,
                    -0.00036603290732206754,
                    0.0021723510539963234,
                ],
                [
                    0.991868104336543,
                    0.9512948230314798,
                    -0.12610373067210068,
                    -0.0007005762815116696,
                    0.007494114566098323,
                ],
                [
                    1.2793851556007614,
                    0.02359750623441366,
                    -6.41750984171976e-17,
                    -0.0,
                    0.009666465620094647,
                ],
                [0.0, 0.0, 0.9000000000000004, 0.005, -0.0],
            ],
            dtype=jnp.float64,
        ),
        rtol=1e-8,
        atol=1e-8,
    )

    jacobian = calculate_jacobian(model, steady_state=steady_state_result.steady_state)
    np.testing.assert_allclose(
        jacobian,
        first_order_result.jacobian,
        rtol=1e-9,
        atol=1e-9,
    )


def test_jax_rbc_calibrated_steady_state_matches_numpy_path() -> None:
    model = parse_macro_model(RBC_CME_CALIBRATION_SOURCE)
    initial_guess = {
        "A": 1.0,
        "Pi": 1.0025,
        "R": 1.0035,
        "c": 1.2,
        "k": 9.4,
        "y": 1.42,
        "z_delta": 1.0,
    }

    jax_result = solve_steady_state_jax(
        model,
        initial_guess=initial_guess,
    )
    numpy_result = solve_steady_state(
        model,
        initial_guess=initial_guess,
    )

    assert bool(np.asarray(jax_result.converged))
    np.testing.assert_allclose(
        jax_result.steady_state,
        numpy_result.steady_state,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        jax_result.parameter_values,
        numpy_result.parameter_values,
        rtol=1e-8,
        atol=1e-8,
    )


def test_calibrated_first_order_outputs_are_jittable_and_gpu_accessible() -> None:
    model = parse_macro_model(RBC_CME_CALIBRATION_SOURCE)
    result = solve_first_order_model(
        model,
        steady_state_initial_guess={
            "A": 1.0,
            "Pi": 1.0025,
            "R": 1.0035,
            "c": 1.2,
            "k": 9.4,
            "y": 1.42,
            "z_delta": 1.0,
        },
    )
    state_space = linear_state_space_from_first_order_solution(
        result.solution.solution_matrix,
        model.timings,
        observable_indices=(2, 4),
    )

    jit_simulate = jax.jit(
        simulate_linear_gaussian_state_space,
        static_argnames=("num_periods",),
    )
    simulation = jit_simulate(state_space, jax.random.PRNGKey(0), num_periods=4)

    assert simulation.states.shape == (state_space.transition_matrix.shape[0], 4)
    assert simulation.observations.shape == (state_space.observation_matrix.shape[0], 4)

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if gpu_devices:
        gpu_state_space = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, gpu_devices[0]),
            state_space,
        )
        gpu_key = jax.device_put(jax.random.PRNGKey(0), gpu_devices[0])
        gpu_simulation = jit_simulate(gpu_state_space, gpu_key, num_periods=4)
        assert gpu_simulation.states.shape == simulation.states.shape
        assert gpu_simulation.observations.shape == simulation.observations.shape
