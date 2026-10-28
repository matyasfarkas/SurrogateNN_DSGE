#!/usr/bin/env julia

using Dates
using JSON
using LinearAlgebra
using MacroModelling
using SHA


const SCHEMA_VERSION = "surrogatenn-dsge-gali-julia-sep-likelihood-export/v1"


function sha256_file(path::AbstractString)
    return bytes2hex(sha256(read(path)))
end


function row_major_json(matrix::AbstractMatrix)
    return [collect(matrix[row, :]) for row in axes(matrix, 1)]
end


function inject_parameters(model, overrides::AbstractDict{String,<:Any})
    values = Float64.(model.parameter_values)
    for (name, raw_value) in overrides
        index = findfirst(==(Symbol(name)), model.parameters)
        isnothing(index) && error("Unknown parameter override: $name")
        values[index] = Float64(raw_value)
    end
    return values
end


function checked_indices(names::Vector{String}, requested::Vector{String}, label::String)
    length(unique(requested)) == length(requested) || error("Duplicate $label name.")
    indices = Int[]
    for name in requested
        index = findfirst(==(name), names)
        isnothing(index) && error("Unknown $label name: $name")
        push!(indices, index)
    end
    return indices
end


function predict_step!(
    model,
    parameters::Vector{Float64},
    steady_state::Vector{Float64},
    state_deviation::Vector{Float64},
    shocks::Vector{Float64},
    observable_indices::Vector{Int},
    config::AbstractDict{String,<:Any},
    accept_tol::Float64,
    initial_guess::Union{Nothing,Vector{Float64}},
)
    periods = Int(config["periods"])
    deterministic_shocks = zeros(Float64, periods, length(model.exo))
    deterministic_shocks[1, :] .= shocks
    initial_state = steady_state + state_deviation

    Base.invokelatest(() -> solve!(
        model;
        parameters = parameters,
        algorithm = :stochastic_extended_path,
        silent = true,
        sep_periods = periods,
        sep_order = Int(config["branching_order"]),
        sep_nnodes = Int(config["nnodes"]),
        sep_maxit = Int(config["max_iter"]),
        sep_tol = Float64(config["tol"]),
        sep_sparse_tree = Bool(config["sparse_tree"]),
        sep_linear_solver = Symbol(config["linear_solver"]),
        sep_fallback_solver = isnothing(config["fallback_solver"]) ? nothing : Symbol(config["fallback_solver"]),
        sep_stall_iters = Int(config["stall_iters"]),
        sep_stall_rel_tol = Float64(config["stall_rel_tol"]),
        sep_stall_abs_tol = Float64(config["stall_abs_tol"]),
        sep_initial_guess = initial_guess,
        sep_initial_state = initial_state,
        sep_deterministic_shocks = deterministic_shocks,
    ))
    solution = model.solution.perturbation.stochastic_extended_path
    isnothing(solution) && error("MacroModelling did not retain an SEP solution.")
    error_norm = Float64(solution.final_error)
    flag = Int(solution.convergence_flag)
    accepted = flag == 0 || (isfinite(error_norm) && error_norm <= accept_tol)
    layout = solution.layout
    layout.G[2] == 1 || error("The first SEP transition must contain one state group.")
    indices = layout.voff[2] .+ (1:layout.ny_)
    next_state = Float64.(solution.Y[indices])
    next_deviation = next_state - steady_state
    observable_deviation = next_deviation[observable_indices]
    finite = all(isfinite, next_state) && all(isfinite, observable_deviation)
    return (
        ok = accepted && finite,
        obs_dev = observable_deviation,
        state_next_dev = next_deviation,
        solution_guess = Float64.(solution.Y),
        flag = flag,
        error = error_norm,
        runtime_seconds = Float64(solution.runtime_seconds),
    )
end


function fd_jacobian!(
    model,
    parameters::Vector{Float64},
    steady_state::Vector{Float64},
    state_deviation::Vector{Float64},
    eps_structural::Vector{Float64},
    observable_indices::Vector{Int},
    shock_sigmas::Vector{Float64},
    config::AbstractDict{String,<:Any},
    accept_tol::Float64,
    initial_guess::Union{Nothing,Vector{Float64}},
)
    n_observables = length(observable_indices)
    n_structural = length(eps_structural)
    jacobian = zeros(Float64, n_observables, n_structural)
    calls = Vector{Any}()
    base_eval = predict_step!(
        model,
        parameters,
        steady_state,
        state_deviation,
        eps_structural,
        observable_indices,
        config,
        accept_tol,
        initial_guess,
    )
    push!(calls, Dict(
        "kind" => "base",
        "shocks" => copy(eps_structural),
        "convergence_flag" => base_eval.flag,
        "final_error" => base_eval.error,
        "runtime_seconds" => base_eval.runtime_seconds,
        "accepted" => base_eval.ok,
    ))
    base_eval.ok || return false, jacobian, base_eval, calls

    for column in 1:n_structural
        step_scale = max(1.0, abs(eps_structural[column]), shock_sigmas[column])
        step_size = max(cbrt(eps(Float64)) * step_scale, 1.0e-6)
        perturbed = copy(eps_structural)
        perturbed[column] += step_size
        perturbed_eval = predict_step!(
            model,
            parameters,
            steady_state,
            state_deviation,
            perturbed,
            observable_indices,
            config,
            accept_tol,
            base_eval.solution_guess,
        )
        push!(calls, Dict(
            "kind" => "finite_difference",
            "column" => column - 1,
            "step_size" => step_size,
            "shocks" => perturbed,
            "convergence_flag" => perturbed_eval.flag,
            "final_error" => perturbed_eval.error,
            "runtime_seconds" => perturbed_eval.runtime_seconds,
            "accepted" => perturbed_eval.ok,
        ))
        perturbed_eval.ok || return false, jacobian, base_eval, calls
        jacobian[:, column] .= (perturbed_eval.obs_dev - base_eval.obs_dev) / step_size
    end
    return true, jacobian, base_eval, calls
end


function run_inversion(base_model, spec::AbstractDict{String,<:Any})
    started_ns = time_ns()
    model = deepcopy(base_model)
    parameters = inject_parameters(model, spec["parameters"])
    steady_state_result = Base.invokelatest(() -> get_steady_state(
        model;
        parameters = parameters,
        return_variables_only = true,
        derivatives = false,
        silent = true,
        verbose = false,
    ))
    steady_state = vec(Float64.(Array(steady_state_result)))
    all(isfinite, steady_state) || error("Steady state contains a non-finite value.")

    variable_names = string.(model.var)
    shock_names = string.(model.exo)
    observable_names = String.(spec["observable_names"])
    structural_shocks = String.(spec["structural_shocks"])
    structural_shocks == shock_names || error(
        "The adapter currently requires every shock in the model's declared order."
    )
    observable_indices = checked_indices(variable_names, observable_names, "observable")
    observations = Float64.(spec["observation_levels"])
    length(observations) == length(observable_indices) || error(
        "observation_levels must contain one value per observable."
    )
    observation_deviation = observations - steady_state[observable_indices]
    shock_sigmas = Float64.(spec["shock_sigmas"])
    length(shock_sigmas) == length(shock_names) || error(
        "shock_sigmas must contain one value per model shock."
    )
    all(value -> isfinite(value) && value > 0.0, shock_sigmas) || error(
        "shock_sigmas must be finite and strictly positive."
    )

    config = spec["config"]
    inversion = spec["inversion"]
    accept_tol = Float64(config["accept_tol"])
    max_iterations = Int(inversion["max_iter"])
    step_tol = Float64(inversion["step_tol"])
    residual_tol = Float64(inversion["residual_tol"])
    damping = Float64(inversion["lambda"])
    state_deviation = zeros(Float64, length(steady_state))
    eps_structural = zeros(Float64, length(shock_names))
    base_guess = nothing
    final_eval = nothing
    final_jacobian = nothing
    final_evaluated_shocks = nothing
    iteration_trace = Vector{Any}()
    predict_calls = 0
    stop_reason = "max_iterations"

    for iteration in 1:max_iterations
        evaluated_shocks = copy(eps_structural)
        ok, jacobian, base_eval, calls = fd_jacobian!(
            model,
            parameters,
            steady_state,
            state_deviation,
            eps_structural,
            observable_indices,
            shock_sigmas,
            config,
            accept_tol,
            base_guess,
        )
        predict_calls += length(calls)
        ok || error("Finite-difference SEP prediction failed at iteration $iteration.")
        residual = observation_deviation - base_eval.obs_dev
        all(isfinite, residual) || error("Inversion residual is non-finite.")
        all(isfinite, jacobian) || error("Inversion Jacobian is non-finite.")
        final_eval = base_eval
        final_jacobian = copy(jacobian)
        final_evaluated_shocks = evaluated_shocks
        base_guess = base_eval.solution_guess
        residual_norm = norm(residual)
        record = Dict{String,Any}(
            "iteration" => iteration,
            "evaluated_shocks" => evaluated_shocks,
            "predicted_observation_deviation" => copy(base_eval.obs_dev),
            "residual" => residual,
            "residual_norm" => residual_norm,
            "jacobian" => row_major_json(jacobian),
            "predict_calls" => calls,
        )
        if residual_norm <= residual_tol
            record["stop_reason"] = "residual_tolerance"
            push!(iteration_trace, record)
            stop_reason = "residual_tolerance"
            break
        end

        lhs = transpose(jacobian) * jacobian + damping * I
        rhs = transpose(jacobian) * residual
        step = try
            lhs \ rhs
        catch
            pinv(jacobian) * residual
        end
        all(isfinite, step) || error("Inversion step is non-finite.")
        eps_structural .+= step
        step_norm = norm(step)
        record["step"] = step
        record["step_norm"] = step_norm
        if step_norm <= step_tol * (1.0 + norm(eps_structural))
            record["stop_reason"] = "step_tolerance"
            push!(iteration_trace, record)
            stop_reason = "step_tolerance"
            break
        end
        push!(iteration_trace, record)
    end

    isnothing(final_eval) && error("The inversion produced no final prediction.")
    isnothing(final_jacobian) && error("The inversion produced no final Jacobian.")
    residual = observation_deviation - final_eval.obs_dev
    standardized_jacobian = final_jacobian .* reshape(shock_sigmas, 1, :)
    log_abs_det, determinant_sign = LinearAlgebra.logabsdet(standardized_jacobian)
    determinant_sign == 0 && error("The standardized Jacobian is singular.")
    isfinite(log_abs_det) || error("The standardized Jacobian log determinant is non-finite.")
    standardized_shocks = eps_structural ./ shock_sigmas
    shocks2 = sum(abs2, standardized_shocks)
    loglikelihood = -0.5 * (
        log_abs_det + shocks2 + length(observable_names) * log(2.0 * pi)
    )

    return Dict(
        "status" => "ok",
        "elapsed_s" => (time_ns() - started_ns) / 1.0e9,
        "parameter_values" => parameters,
        "steady_state" => steady_state,
        "observable_indices_zero_based" => observable_indices .- 1,
        "observation_levels" => observations,
        "observation_deviation" => observation_deviation,
        "recovered_shocks" => eps_structural,
        "final_evaluated_shocks" => final_evaluated_shocks,
        "predicted_observation_deviation" => final_eval.obs_dev,
        "residual" => residual,
        "residual_norm" => norm(residual),
        "jacobian" => row_major_json(final_jacobian),
        "standardized_jacobian" => row_major_json(standardized_jacobian),
        "logabsdet" => log_abs_det,
        "shocks2" => shocks2,
        "loglikelihood" => loglikelihood,
        "iterations" => length(iteration_trace),
        "predict_calls" => predict_calls,
        "stop_reason" => stop_reason,
        "final_sep_convergence_flag" => final_eval.flag,
        "final_sep_error" => final_eval.error,
        "iteration_trace" => iteration_trace,
    )
end


function main(args::Vector{String})
    length(args) == 3 || error(
        "Usage: export_gali_full_julia_sep_likelihood.jl MODEL_PATH SPEC_JSON OUTPUT_JSON"
    )
    model_path = abspath(args[1])
    spec_path = abspath(args[2])
    output_path = abspath(args[3])
    spec = JSON.parsefile(spec_path)
    model_symbol = Symbol(spec["model_symbol"])

    Base.include(Main, model_path)
    Base.invokelatest(() -> isdefined(Main, model_symbol)) || error(
        "Model symbol $(String(model_symbol)) was not defined by $model_path."
    )
    model = Base.invokelatest(() -> getfield(Main, model_symbol))
    result = try
        run_inversion(model, spec)
    catch err
        Dict(
            "status" => "error",
            "error" => sprint(showerror, err, catch_backtrace()),
        )
    end
    payload = Dict(
        "schema_version" => SCHEMA_VERSION,
        "generated_at_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        "julia_version" => string(VERSION),
        "model" => Dict(
            "symbol" => String(model_symbol),
            "path" => model_path,
            "sha256" => sha256_file(model_path),
            "variable_names" => string.(model.var),
            "shock_names" => string.(model.exo),
            "parameter_names" => string.(model.parameters),
        ),
        "spec" => spec,
        "result" => result,
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, payload, 2)
        write(io, '\n')
    end
    result["status"] == "ok" || exit(2)
end


main(ARGS)
