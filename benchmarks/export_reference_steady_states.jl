using JSON
using MacroModelling
using TOML

function quiet(f::Function)
    redirect_stdout(devnull) do
        f()
    end
end

function main()
    length(ARGS) == 2 || error("Usage: export_reference_steady_states.jl <config.toml> <output.json>")
    config_path = ARGS[1]
    output_path = ARGS[2]

    config = TOML.parsefile(config_path)
    entries = Vector{Dict{String, Any}}()

    for case in config["case"]
        quiet() do
            Base.include(Main, case["model_path"])
        end
        model = Base.invokelatest(() -> getfield(Main, Symbol(case["model_symbol"])))
        steady_state = Float64.(model.solution.non_stochastic_steady_state[1:length(model.var)])
        push!(
            entries,
            Dict(
                "name" => case["name"],
                "model_symbol" => case["model_symbol"],
                "var" => string.(model.var),
                "steady_state" => steady_state,
            ),
        )
    end

    payload = Dict(
        "julia_version" => string(VERSION),
        "entries" => entries,
    )
    open(output_path, "w") do io
        write(io, JSON.json(payload))
    end
end

main()
