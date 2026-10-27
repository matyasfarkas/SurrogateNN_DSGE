#!/usr/bin/env julia

using Dates
using JSON
using MacroModelling
using SHA


const SCHEMA_VERSION = "surrogatenn-dsge-gali-julia-sep-export/v1"


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


function extract_tree_states(solution)
    layout = solution.layout
    output = Vector{Any}(undef, layout.T + 1)
    for t in 0:layout.T
        groups = layout.G[t + 1]
        values = Matrix{Float64}(undef, groups, layout.ny_)
        for g in 1:groups
            indices = layout.voff[t + 1] + (g - 1) * layout.ny_ .+ (1:layout.ny_)
            values[g, :] .= solution.Y[indices]
        end
        all(isfinite, values) || error("SEP tree contains a non-finite state.")
        output[t + 1] = row_major_json(values)
    end
    return output
end


function solve_case(base_model, case::AbstractDict{String,<:Any}, config)
    started_ns = time_ns()
    try
        model = deepcopy(base_model)
        parameters = inject_parameters(model, case["parameters"])
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
        deterministic_shocks = reduce(
            vcat,
            [permutedims(Float64.(row)) for row in case["deterministic_shocks"]],
        )

        Base.invokelatest(() -> solve!(
            model;
            parameters = parameters,
            algorithm = :stochastic_extended_path,
            silent = true,
            sep_periods = Int(config["periods"]),
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
            sep_initial_state = steady_state,
            sep_deterministic_shocks = deterministic_shocks,
        ))
        solution = model.solution.perturbation.stochastic_extended_path
        isnothing(solution) && error("MacroModelling did not retain an SEP solution.")

        return Dict(
            "case_id" => String(case["case_id"]),
            "theta_index" => Int(case["theta_index"]),
            "parameters" => Dict(
                String(name) => Float64(value)
                for (name, value) in case["parameters"]
            ),
            "status" => "ok",
            "elapsed_s" => (time_ns() - started_ns) / 1.0e9,
            "parameter_values" => parameters,
            "steady_state" => steady_state,
            "convergence_flag" => Int(solution.convergence_flag),
            "final_error" => Float64(solution.final_error),
            "runtime_seconds" => Float64(solution.runtime_seconds),
            "group_counts" => Int.(solution.layout.G[1:(solution.periods + 1)]),
            "tree_states" => extract_tree_states(solution),
        )
    catch err
        return Dict(
            "case_id" => String(case["case_id"]),
            "theta_index" => Int(case["theta_index"]),
            "parameters" => case["parameters"],
            "status" => "error",
            "elapsed_s" => (time_ns() - started_ns) / 1.0e9,
            "error" => sprint(showerror, err, catch_backtrace()),
        )
    end
end


function main(args::Vector{String})
    length(args) == 3 || error(
        "Usage: export_gali_full_julia_sep_parity.jl MODEL_PATH CASES_JSON OUTPUT_JSON"
    )
    model_path = abspath(args[1])
    cases_path = abspath(args[2])
    output_path = abspath(args[3])
    spec = JSON.parsefile(cases_path)
    model_symbol = Symbol(spec["model_symbol"])

    Base.include(Main, model_path)
    Base.invokelatest(() -> isdefined(Main, model_symbol)) || error(
        "Model symbol $(String(model_symbol)) was not defined by $model_path."
    )
    model = Base.invokelatest(() -> getfield(Main, model_symbol))
    results = [solve_case(model, case, spec["config"]) for case in spec["cases"]]
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
        "config" => spec["config"],
        "cases" => results,
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, payload, 2)
        write(io, '\n')
    end
    all(case -> case["status"] == "ok", results) || exit(2)
end


main(ARGS)
