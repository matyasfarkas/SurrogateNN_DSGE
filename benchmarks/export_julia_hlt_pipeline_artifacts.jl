#!/usr/bin/env julia
# Export Julia HLT SEP/surrogate artifacts to the canonical JSON parity schema.
#
# This script deliberately lives in the Python port repository and only reads
# serialized Julia outputs. It does not mutate the Julia source tree or rerun SEP.

using Dates
using Serialization
using SHA

const DEFAULT_JULIA_REPO = normpath(joinpath(@__DIR__, "..", "..", "SurrogateNN_Estimation.jl"))

function print_help()
    println("""
Export Julia HLT pipeline artifacts to JSON.

Required:
  --dataset PATH          Serialized Julia dataset with X, Y, Y_rom1/Y_rom.
  --out PATH              Output JSON file.

Optional:
  --case NAME             Case name in the parity JSON payload. Default: hlt_pipeline.
  --julia-repo PATH       Julia source repo root. Default: sibling SurrogateNN_Estimation.jl.
  --surrogate PATH        Serialized Julia surrogate bundle with a frozen model.
  --extra-payload PATH    Additional serialized Dict/summary payload to mine for ROM/gate/likelihood fields.
                          May be repeated.
  --first-column INT      First dataset column to export, using Julia 1-based indexing. Default: 1.
  --max-columns INT       Maximum columns to export. Default: 0 means all remaining columns.

Examples:
  julia benchmarks/export_julia_hlt_pipeline_artifacts.jl \\
    --dataset ../SurrogateNN_Estimation.jl/data/hlt_moderate/hlt_sep_surrogate_dataset.jls \\
    --surrogate ../SurrogateNN_Estimation.jl/data/hlt_moderate/hlt_sep_surrogate_trained_rom1_resid_obs.jls \\
    --out benchmarks/results/hlt_pipeline_parity/julia_pipeline_artifacts.json
""")
end

function parse_cli(args::Vector{String})
    opts = Dict{String, Any}(
        "case" => "hlt_pipeline",
        "julia-repo" => DEFAULT_JULIA_REPO,
        "extra-payload" => String[],
        "first-column" => "1",
        "max-columns" => "0",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_help()
            exit(0)
        end
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        body = arg[3:end]
        key = body
        value = nothing
        if occursin("=", body)
            parts = split(body, "=", limit = 2)
            key = parts[1]
            value = parts[2]
        else
            i < length(args) || error("Option --$key requires a value.")
            i += 1
            value = args[i]
        end
        if key == "extra-payload"
            push!(opts["extra-payload"], String(value))
        else
            opts[key] = String(value)
        end
        i += 1
    end
    haskey(opts, "dataset") || error("Missing required --dataset PATH.")
    haskey(opts, "out") || error("Missing required --out PATH.")
    opts["first-column"] = parse(Int, opts["first-column"])
    opts["max-columns"] = parse(Int, opts["max-columns"])
    opts["first-column"] >= 1 || error("--first-column must be >= 1.")
    opts["max-columns"] >= 0 || error("--max-columns must be >= 0.")
    return opts
end

function _dict_haskey(d, key::AbstractString)
    return d isa AbstractDict && (haskey(d, key) || haskey(d, Symbol(key)))
end

function _dict_get(d, key::AbstractString, default = nothing)
    if !(d isa AbstractDict)
        return default
    elseif haskey(d, key)
        return d[key]
    elseif haskey(d, Symbol(key))
        return d[Symbol(key)]
    else
        return default
    end
end

function _get_or_field(x, key::AbstractString, default = nothing)
    if x isa AbstractDict
        return _dict_get(x, key, default)
    end
    sym = Symbol(key)
    return hasproperty(x, sym) ? getproperty(x, sym) : default
end

function _lookup_path(payload, path::Vector{String})
    current = payload
    for key in path
        next = _get_or_field(current, key, nothing)
        next === nothing && return nothing
        current = next
    end
    return current
end

function _first_alias(payload, aliases::Vector{Vector{String}})
    for alias in aliases
        value = _lookup_path(payload, alias)
        value === nothing || return value
    end
    return nothing
end

function _require_matrix(payload, keys::Vector{String}; label::String)
    for key in keys
        value = _dict_get(payload, key, nothing)
        if value !== nothing
            matrix = Matrix{Float64}(value)
            ndims(matrix) == 2 || error("$label must be a matrix.")
            return matrix
        end
    end
    error("Dataset is missing required matrix $label. Accepted keys: $(join(keys, ", ")).")
end

function _column_selection(n::Integer, first_col::Integer, max_cols::Integer)
    first_col <= n || error("--first-column=$first_col exceeds dataset column count $n.")
    last_col = max_cols == 0 ? n : min(n, first_col + max_cols - 1)
    first_col <= last_col || error("Empty column selection.")
    return first_col:last_col
end

function _metadata_subset(meta)
    out = Dict{String, Any}()
    meta isa AbstractDict || return out
    keys_to_copy = String[
        "source",
        "created_at",
        "base_dataset",
        "hmc_path",
        "surrogate_path",
        "data_path",
        "selection",
        "ood_z_threshold",
        "max_points",
        "repeat_active",
        "draw_count",
        "draw_seed",
        "selected_draw_indices",
        "selected_local_periods",
        "selected_global_periods",
        "selected_max_z",
        "selected_p95_z",
        "selected_top_label",
        "sep_horizon",
        "sep_order",
        "sep_nnodes",
        "sep_maxit",
        "sep_tol",
        "sep_accept_tol",
        "sep_recovery",
        "theta_names",
        "theta",
    ]
    for key in keys_to_copy
        value = _dict_get(meta, key, nothing)
        value === nothing || (out[key] = value)
    end
    return out
end

function _support_indices(dataset, meta, cols, total_columns::Integer)
    candidates = Any[
        _dict_get(dataset, "selected_indices", nothing),
        _dict_get(meta, "selected_global_periods", nothing),
        _dict_get(meta, "selected_draw_indices", nothing),
        _dict_get(dataset, "sample_idx", nothing),
    ]
    for value in candidates
        value === nothing && continue
        vec_value = vec(value)
        if length(vec_value) == total_columns
            return Int.(vec_value)[cols]
        end
    end
    return collect(Int, cols)
end

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _ensure_section!(artifacts::Dict{String, Any}, section::String)
    if !haskey(artifacts, section)
        artifacts[section] = Dict{String, Any}()
    end
    return artifacts[section]
end

function _assign_if_present!(artifacts::Dict{String, Any},
                             section::String,
                             field::String,
                             payload,
                             aliases::Vector{Vector{String}};
                             transform = identity)
    value = _first_alias(payload, aliases)
    value === nothing && return false
    _ensure_section!(artifacts, section)[field] = transform(value)
    return true
end

function _float_array(value)
    return Array{Float64}(value)
end

function _int_array(value)
    return Int.(vec(value))
end

function _bool_array(value)
    return Bool.(vec(value))
end

function _float_scalar(value)
    values = value isa AbstractArray ? vec(value) : [value]
    length(values) == 1 || error("Expected scalar value, got $(length(values)) values.")
    return Float64(values[1])
end

const ROM_STATE_ALIASES = Vector{Vector{String}}([
    ["artifacts", "rom", "states"],
    ["rom", "states"],
    ["rom_state_path"],
    ["states"],
    ["filtered_variables"],
    ["stages", "rom_path", "result", "states"],
    ["stages", "kalman_paths", "result", "filtered_variables"],
])
const ROM_OBS_ALIASES = Vector{Vector{String}}([
    ["artifacts", "rom", "observations"],
    ["rom", "observations"],
    ["rom_observation_path"],
    ["observations"],
    ["linear_observations"],
    ["stages", "gate_stats", "result", "linear_observations"],
])
const ROM_SHOCK_ALIASES = Vector{Vector{String}}([
    ["artifacts", "rom", "shocks"],
    ["rom", "shocks"],
    ["rom_shock_path"],
    ["shocks"],
    ["filtered_shocks"],
    ["stages", "gate_stats", "result", "shocks"],
    ["stages", "kalman_paths", "result", "filtered_shocks"],
])
const GATE_E_ALIASES = Vector{Vector{String}}([
    ["artifacts", "gate", "e_stat"],
    ["gate", "e_stat"],
    ["gate_e_stat"],
    ["e_stat"],
    ["stages", "gate_stats", "result", "e_stat"],
])
const GATE_F_ALIASES = Vector{Vector{String}}([
    ["artifacts", "gate", "f_stat"],
    ["gate", "f_stat"],
    ["gate_f_stat"],
    ["f_stat"],
    ["stages", "gate_stats", "result", "f_stat"],
])
const GATE_PROB_ALIASES = Vector{Vector{String}}([
    ["artifacts", "gate", "probs"],
    ["artifacts", "gate", "gate_probs"],
    ["gate", "probs"],
    ["gate", "gate_probs"],
    ["gate_probs"],
    ["shared_gate_probs"],
    ["switching_result", "gate_probs"],
])
const GATE_MASK_ALIASES = Vector{Vector{String}}([
    ["artifacts", "gate", "mask"],
    ["artifacts", "gate", "hard_mask"],
    ["gate", "mask"],
    ["gate", "hard_mask"],
    ["gate_mask"],
    ["hard_mask"],
    ["switching_result", "hard_mask"],
])
const LIKELIHOOD_ALIASES = Vector{Vector{String}}([
    ["artifacts", "likelihood", "switching_per_period"],
    ["artifacts", "likelihood", "per_period"],
    ["likelihood", "switching_per_period"],
    ["likelihood", "per_period"],
    ["switching_loglik_per_period"],
    ["ll_switching"],
    ["switching_result", "per_period"],
])
const POSTERIOR_ALIASES = Vector{Vector{String}}([
    ["artifacts", "posterior", "log_density"],
    ["posterior", "log_density"],
    ["posterior_log_density"],
    ["log_density"],
    ["logposterior"],
])

function extract_known_fields!(artifacts::Dict{String, Any}, payload)
    _assign_if_present!(artifacts, "rom", "states", payload, ROM_STATE_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "rom", "observations", payload, ROM_OBS_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "rom", "shocks", payload, ROM_SHOCK_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "gate", "e_stat", payload, GATE_E_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "gate", "f_stat", payload, GATE_F_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "gate", "gate_probs", payload, GATE_PROB_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "gate", "hard_mask", payload, GATE_MASK_ALIASES; transform = _bool_array)
    _assign_if_present!(artifacts, "likelihood", "switching_per_period", payload, LIKELIHOOD_ALIASES; transform = _float_array)
    _assign_if_present!(artifacts, "posterior", "log_density", payload, POSTERIOR_ALIASES; transform = _float_scalar)
    return artifacts
end

function include_surrogate_utils!(julia_repo::AbstractString)
    utils = joinpath(julia_repo, "scripts", "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl")
    isfile(utils) || error("Cannot find Julia surrogate utility file: $utils")
    include(utils)
    return nothing
end

function load_frozen_surrogate(path::AbstractString)
    isfile(path) || error("Surrogate bundle not found: $path")
    payload = try
        Serialization.deserialize(path)
    catch err
        error(
            "Failed to deserialize surrogate bundle $path. The bundle may have been " *
            "written with incompatible Julia type definitions; rerun the Julia exporter " *
            "with the matching source checkout/environment, or regenerate the bundle. " *
            "Original error: $(sprint(showerror, err))"
        )
    end
    frozen = _get_or_field(payload, "frozen", nothing)
    frozen === nothing && error("Surrogate bundle $path does not contain a frozen model.")
    return frozen
end

function add_surrogate_predictions!(artifacts::Dict{String, Any},
                                    surrogate_path::AbstractString,
                                    X::Matrix{Float64})
    frozen = load_frozen_surrogate(surrogate_path)
    frozen_dim = hasproperty(frozen, :d_in) ? getproperty(frozen, :d_in) : nothing
    frozen_dim == size(X, 1) ||
        error("Surrogate input dimension does not match dataset X rows: d_in=$frozen_dim rows=$(size(X, 1)).")
    predictions = Base.invokelatest(predict_frozen_batch, frozen, X)
    _ensure_section!(artifacts, "surrogate")["predictions"] = Matrix{Float64}(predictions)
    return artifacts
end

function json_escape(s::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for ch in s
        if ch == '"'
            print(io, "\\\"")
        elseif ch == '\\'
            print(io, "\\\\")
        elseif ch == '\b'
            print(io, "\\b")
        elseif ch == '\f'
            print(io, "\\f")
        elseif ch == '\n'
            print(io, "\\n")
        elseif ch == '\r'
            print(io, "\\r")
        elseif ch == '\t'
            print(io, "\\t")
        elseif Int(ch) < 0x20
            print(io, "\\u", lpad(string(Int(ch), base = 16), 4, '0'))
        else
            print(io, ch)
        end
    end
    print(io, '"')
    return String(take!(io))
end

function write_json_value(io::IO, value)
    if value === nothing || value isa Missing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        if isfinite(value)
            print(io, repr(Float64(value)))
        else
            print(io, "null")
        end
    elseif value isa Real
        number = Float64(value)
        if isfinite(number)
            print(io, repr(number))
        else
            print(io, "null")
        end
    elseif value isa AbstractString
        print(io, json_escape(value))
    elseif value isa Symbol
        print(io, json_escape(String(value)))
    elseif value isa Dates.AbstractTime
        print(io, json_escape(string(value)))
    elseif value isa AbstractDict
        write_json_dict(io, value)
    elseif value isa NamedTuple
        write_json_dict(io, Dict(String(k) => getfield(value, k) for k in keys(value)))
    elseif value isa Tuple
        write_json_vector(io, collect(value))
    elseif value isa AbstractArray
        write_json_array(io, value)
    else
        error("Cannot encode value of type $(typeof(value)) as JSON.")
    end
end

function write_json_vector(io::IO, value)
    print(io, "[")
    first = true
    for item in value
        first || print(io, ",")
        write_json_value(io, item)
        first = false
    end
    print(io, "]")
end

function write_json_array(io::IO, value::AbstractArray)
    if ndims(value) == 0
        write_json_value(io, value[])
    elseif ndims(value) == 1
        write_json_vector(io, value)
    else
        print(io, "[")
        first = true
        for i in axes(value, 1)
            first || print(io, ",")
            write_json_array(io, selectdim(value, 1, i))
            first = false
        end
        print(io, "]")
    end
end

function write_json_dict(io::IO, value::AbstractDict)
    print(io, "{")
    first = true
    for key in sort(collect(keys(value)); by = x -> string(x))
        first || print(io, ",")
        print(io, json_escape(string(key)), ":")
        write_json_value(io, value[key])
        first = false
    end
    print(io, "}")
end

function write_json_file(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        write_json_value(io, payload)
        println(io)
    end
end

function build_artifacts(opts::Dict{String, Any})
    dataset_path = opts["dataset"]
    isfile(dataset_path) || error("Dataset not found: $dataset_path")
    dataset = Serialization.deserialize(dataset_path)
    dataset isa AbstractDict || error("Dataset must deserialize to a Dict-like object, got $(typeof(dataset)).")

    X_all = _require_matrix(dataset, ["X"]; label = "X")
    Y_all = _require_matrix(dataset, ["Y"]; label = "Y")
    Y_rom_all = _require_matrix(dataset, ["Y_rom1", "Y_rom"]; label = "Y_rom1/Y_rom")
    size(Y_all) == size(Y_rom_all) || error("Y and Y_rom1/Y_rom shapes differ: $(size(Y_all)) vs $(size(Y_rom_all)).")
    size(X_all, 2) == size(Y_all, 2) || error("X and Y column counts differ: $(size(X_all, 2)) vs $(size(Y_all, 2)).")

    cols = _column_selection(size(X_all, 2), opts["first-column"], opts["max-columns"])
    X = X_all[:, cols]
    Y = Y_all[:, cols]
    Y_rom = Y_rom_all[:, cols]
    residual_labels = Y .- Y_rom
    meta = _dict_get(dataset, "meta", Dict{String, Any}())

    artifacts = Dict{String, Any}(
        "support" => Dict{String, Any}(
            "features" => X,
            "selected_indices" => _support_indices(dataset, meta, cols, size(X_all, 2)),
        ),
        "sep" => Dict{String, Any}(
            "targets" => Y,
            "rom_targets" => Y_rom,
            "residual_labels" => residual_labels,
        ),
        "diagnostics" => Dict{String, Any}(
            "dataset_path" => String(dataset_path),
            "dataset_sha256" => _sha256_file(dataset_path),
            "total_columns" => size(X_all, 2),
            "first_column" => first(cols),
            "last_column" => last(cols),
            "exported_columns" => length(cols),
            "created_at" => string(now()),
            "julia_repo" => String(opts["julia-repo"]),
            "dataset_meta" => _metadata_subset(meta),
        ),
    )

    sep_residuals = _dict_get(dataset, "sep_residuals", nothing)
    if sep_residuals !== nothing && length(vec(sep_residuals)) == size(X_all, 2)
        artifacts["diagnostics"]["sep_residuals"] = Float64.(vec(sep_residuals))[cols]
    end

    if haskey(opts, "surrogate")
        add_surrogate_predictions!(artifacts, opts["surrogate"], X)
    end

    extract_known_fields!(artifacts, dataset)
    for path in opts["extra-payload"]
        isfile(path) || error("Extra payload not found: $path")
        extra = Serialization.deserialize(path)
        extract_known_fields!(artifacts, extra)
    end

    return artifacts
end

function main(opts::Dict{String, Any})
    artifacts = build_artifacts(opts)
    case_name = String(opts["case"])
    payload = Dict{String, Any}(
        "cases" => Dict{String, Any}(
            case_name => Dict{String, Any}(
                "artifacts" => artifacts,
            ),
        ),
    )
    write_json_file(opts["out"], payload)
    println("Wrote HLT pipeline parity artifact: $(opts["out"])")
    return 0
end

const CLI_OPTIONS = parse_cli(ARGS)
if haskey(CLI_OPTIONS, "surrogate")
    include_surrogate_utils!(CLI_OPTIONS["julia-repo"])
end
main(CLI_OPTIONS)
