#!/usr/bin/env julia

using AxisKeys
using Dates
using JSON
using MacroModelling
using SHA


const SCHEMA_VERSION = "surrogatenn-dsge-gali-julia-core-export/v1"


function sha256_file(path::AbstractString)
    return bytes2hex(sha256(read(path)))
end


function finite_matrix(value)
    matrix = Float64.(Array(value))
    all(isfinite, matrix) || error("Solver output contains a non-finite value.")
    return matrix
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


function solve_case(model, case::AbstractDict{String,<:Any})
    started_ns = time_ns()
    try
        parameters = inject_parameters(model, case["parameters"])
        steady_state = Base.invokelatest(() -> get_steady_state(
            model;
            parameters = parameters,
            return_variables_only = true,
            derivatives = false,
            silent = true,
            verbose = false,
        ))
        first_order = Base.invokelatest(() -> get_solution(
            model;
            parameters = parameters,
            algorithm = :first_order,
            quadratic_matrix_equation_algorithm = :schur,
            silent = true,
            verbose = false,
        ))

        steady_values = vec(finite_matrix(steady_state))
        first_order_raw = finite_matrix(first_order)
        size(first_order_raw, 1) >= 2 || error(
            "The first-order result has no state/shock response columns."
        )
        solution_values = permutedims(first_order_raw[2:end, :])
        solution_ss = vec(first_order_raw[1, :])
        steady_labels = string.(only(axiskeys(steady_state)))
        solution_axes = axiskeys(first_order)

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
            "steady_state" => Dict(
                "names" => steady_labels,
                "values" => steady_values,
            ),
            "first_order" => Dict(
                "row_names" => string.(solution_axes[2]),
                "column_names_raw" => string.(solution_axes[1][2:end]),
                "solution_matrix" => row_major_json(solution_values),
                "embedded_steady_state" => solution_ss,
            ),
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
        "Usage: export_gali_full_julia_core_parity.jl MODEL_PATH CASES_JSON OUTPUT_JSON"
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
    results = [solve_case(model, case) for case in spec["cases"]]
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
