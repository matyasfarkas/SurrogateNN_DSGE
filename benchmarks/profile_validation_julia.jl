using AxisKeys
using JSON
using MacroModelling
using LinearAlgebra
using Statistics
using Zygote

function quiet(f::Function)
    redirect_stdout(devnull) do
        f()
    end
end

function timed_call(f::Function)
    t0 = time_ns()
    value = f()
    return value, (time_ns() - t0) / 1.0e9
end

function steady_stats(times::Vector{Float64})
    isempty(times) && return Dict("reps" => 0)
    return Dict(
        "reps" => length(times),
        "mean_s" => Statistics.mean(times),
        "median_s" => Statistics.median(times),
        "min_s" => minimum(times),
        "max_s" => maximum(times),
        "std_s" => length(times) > 1 ? Statistics.std(times) : 0.0,
    )
end

function inject_subset(base_theta::Vector{Float64}, subset_idx::Vector{Int}, x)
    return [
        begin
            pos = findfirst(==(i), subset_idx)
            isnothing(pos) ? base_theta[i] : x[pos]
        end
        for i in eachindex(base_theta)
    ]
end

function scalar_result(value)
    return Dict("value" => Float64(value))
end

function first_order_result(value)
    matrix = permutedims(Array(value))
    return Dict(
        "converged" => true,
        "solution_matrix" => matrix,
        "shape" => collect(size(matrix)),
    )
end

function gradient_result(value)
    grad = Float64.(value[2])
    return Dict(
        "value" => Float64(value[1]),
        "grad" => grad,
        "grad_l2" => Float64(norm(grad)),
    )
end

function paths_result(value)
    return Dict(
        "filtered_variables" => permutedims(value["filtered_variables"]),
        "smoothed_variables" => permutedims(value["smoothed_variables"]),
        "filtered_shocks" => permutedims(value["filtered_shocks"]),
        "smoothed_shocks" => permutedims(value["smoothed_shocks"]),
    )
end

function gate_stats_result(value)
    return Dict(
        "linear_observations" => permutedims(value["linear_observations"]),
        "shocks" => permutedims(value["shocks"]),
        "e_stat" => value["e_stat"],
        "f_stat" => value["f_stat"],
    )
end

function measure_stage(f::Function, steady_reps::Int; serializer = scalar_result)
    try
        first_value, first_s = timed_call(f)
        steady_times = Float64[]
        last_value = first_value
        for _ in 1:steady_reps
            last_value, elapsed_s = timed_call(f)
            push!(steady_times, elapsed_s)
        end
        return Dict(
            "status" => "ok",
            "first_call_s" => first_s,
            "steady" => steady_stats(steady_times),
            "result" => serializer(last_value),
        )
    catch err
        return Dict(
            "status" => "error",
            "error" => sprint(showerror, err),
        )
    end
end

function keyed_observations(case::Dict{String,Any})
    names = Symbol.(case["observables"])
    rows = [Float64.(row) for row in case["observations"]]
    matrix = permutedims(reduce(hcat, rows))
    return KeyedArray(matrix; Variable = names, Time = collect(1:size(matrix, 2)))
end

function keyed_observations_subset(case::Dict{String,Any}, periods::Int)
    names = Symbol.(case["observables"])
    rows = [Float64.(row[1:periods]) for row in case["observations"]]
    matrix = permutedims(reduce(hcat, rows))
    return KeyedArray(matrix; Variable = names, Time = collect(1:size(matrix, 2)))
end

function case_result(case::Dict{String,Any})
    model = nothing
    load_stage = measure_stage(
        () -> begin
            quiet() do
                Base.include(Main, case["model_path"])
            end
            model_local = Base.invokelatest(() -> getfield(Main, Symbol(case["model_symbol"])))
            model_local
        end,
        0;
        serializer = value -> Dict(
            "n_vars" => length(value.var),
            "n_exo" => length(value.exo),
            "parameter_count" => length(value.parameters),
        ),
    )
    if load_stage["status"] != "ok"
        return Dict("model_info" => Dict(), "stages" => Dict("model_load" => load_stage))
    end

    model = Base.invokelatest(() -> getfield(Main, Symbol(case["model_symbol"])))
    params = Float64.(model.parameter_values)
    obs_data = keyed_observations(case)
    obs_subset = keyed_observations_subset(case, Int(case["sep_eval_periods"]))
    obs_names = Symbol.(case["observables"])
    state_names = Symbol.(case["state_names"])
    obs_sigma = Float64.([case["obs_sigma"][name] for name in case["observables"]])
    shock_sigmas = Float64.([case["shock_sigmas"][name] for name in case["shock_names"]])
    shared_gate_probs = Float64.(case["shared_gate_probs"])
    subset_names = Symbol.(case["parameter_subset"])
    subset_idx = indexin(subset_names, model.parameters)
    any(isnothing, subset_idx) && error("Missing parameter subset indices.")
    subset_idx = Int.(subset_idx)
    x0 = params[subset_idx]
    regime_switch_config = MacroModelling.RegimeSwitchConfig(
        gate_mode = Symbol(case["gate_mode"]),
        tau_eps = Float64(case["regime_switch_config"]["tau_eps"]),
        tau_y = Float64(case["regime_switch_config"]["tau_y"]),
        beta_eps = Float64(case["regime_switch_config"]["beta_eps"]),
        beta_y = Float64(case["regime_switch_config"]["beta_y"]),
        hard_threshold = Float64(case["gate_hard_threshold"]),
        prob_floor = Float64(case["gate_prob_floor"]),
        prob_ceiling = Float64(case["gate_prob_ceiling"]),
        soft_mixture = Symbol(case["soft_mixture"]),
    )
    switching_config = MacroModelling.SwitchingLikelihoodConfig(
        gate_mode = Symbol(case["gate_mode"]),
        hard_threshold = Float64(case["gate_hard_threshold"]),
        prob_floor = Float64(case["gate_prob_floor"]),
        prob_ceiling = Float64(case["gate_prob_ceiling"]),
        soft_mixture = Symbol(case["soft_mixture"]),
    )

    first_order_fn = () -> Base.invokelatest(() -> quiet() do
        get_solution(
            model;
            parameters = params,
            algorithm = :first_order,
            quadratic_matrix_equation_algorithm = :schur,
            silent = true,
            verbose = false,
        )
    end)
    kalman_fn = () -> get_loglikelihood(
        model,
        obs_data,
        params;
        algorithm = :first_order,
        filter = :kalman,
        quadratic_matrix_equation_algorithm = :schur,
        initial_covariance = :theoretical,
        presample_periods = 0,
        verbose = false,
    )
    kalman_per_period_fn = () -> MacroModelling.get_loglikelihood_per_period(
        model,
        obs_data,
        params;
        algorithm = :first_order,
        filter = :kalman,
        quadratic_matrix_equation_algorithm = :schur,
        initial_covariance = :theoretical,
        presample_periods = 0,
        verbose = false,
    )
    kalman_grad_fn = () -> begin
        objective = x -> begin
            theta = inject_subset(params, subset_idx, x)
            get_loglikelihood(
                model,
                obs_data,
                theta;
                algorithm = :first_order,
                filter = :kalman,
                quadratic_matrix_equation_algorithm = :doubling,
                initial_covariance = :theoretical,
                presample_periods = 0,
                verbose = false,
            )
        end
        value = objective(x0)
        grad = only(Zygote.gradient(objective, x0))
        return value, grad
    end
    kalman_paths_fn = () -> begin
        filtered_variables, _ = Base.invokelatest(
            MacroModelling.estimate_observed_variables_matrix,
            model,
            Array(obs_data),
            obs_names;
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            data_in_levels = true,
            levels = true,
            smooth = false,
            verbose = false,
        )
        smoothed_variables, _ = Base.invokelatest(
            MacroModelling.estimate_observed_variables_matrix,
            model,
            Array(obs_data),
            obs_names;
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            data_in_levels = true,
            levels = true,
            smooth = true,
            verbose = false,
        )
        filtered_shocks = Base.invokelatest(
            MacroModelling.estimate_observed_shocks_matrix,
            model,
            Array(obs_data),
            obs_names;
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            data_in_levels = true,
            smooth = false,
            verbose = false,
        )
        smoothed_shocks = Base.invokelatest(
            MacroModelling.estimate_observed_shocks_matrix,
            model,
            Array(obs_data),
            obs_names;
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            data_in_levels = true,
            smooth = true,
            verbose = false,
        )
        return Dict(
            "filtered_variables" => filtered_variables,
            "smoothed_variables" => smoothed_variables,
            "filtered_shocks" => filtered_shocks,
            "smoothed_shocks" => smoothed_shocks,
        )
    end
    gate_stats_fn = () -> begin
        lin_obs, gate_shocks, e_stat, f_stat = Base.invokelatest(
            MacroModelling.compute_linear_gate_stats_from_filter,
            model,
            Array(obs_data),
            obs_names,
            obs_sigma,
            shock_sigmas,
            state_names;
            periods = size(obs_data, 2),
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            shock_norm = :l2,
            error_norm = :l2,
        )
        return Dict(
            "linear_observations" => lin_obs,
            "shocks" => gate_shocks,
            "e_stat" => e_stat,
            "f_stat" => f_stat,
        )
    end
    switching_fixed_fn = () -> begin
        ll_rom = MacroModelling.get_loglikelihood_per_period(
            model,
            obs_data,
            params;
            algorithm = :first_order,
            filter = :kalman,
            quadratic_matrix_equation_algorithm = :schur,
            initial_covariance = :theoretical,
            presample_periods = 0,
            verbose = false,
        )
        ll_fom = MacroModelling.get_loglikelihood_per_period(
            model,
            obs_data,
            params;
            algorithm = :first_order,
            filter = :inversion,
            quadratic_matrix_equation_algorithm = :schur,
            presample_periods = 0,
            verbose = false,
        )
        MacroModelling.mix_loglikelihood(ll_fom, ll_rom, shared_gate_probs; config = switching_config)
    end
    switching_fn = () -> Base.invokelatest(() -> begin
        lin_obs, gate_shocks, e_stat, f_stat = MacroModelling.compute_linear_gate_stats_from_filter(
            model,
            Array(obs_data),
            obs_names,
            obs_sigma,
            shock_sigmas,
            state_names;
            periods = size(obs_data, 2),
            parameters = params,
            filter = :kalman,
            algorithm = :first_order,
            shock_norm = :l2,
            error_norm = :l2,
        )
        gate_probs = MacroModelling.gate_probabilities(e_stat, f_stat, regime_switch_config)
        ll_rom = MacroModelling.get_loglikelihood_per_period(
            model,
            obs_data,
            params;
            algorithm = :first_order,
            filter = :kalman,
            quadratic_matrix_equation_algorithm = :schur,
            initial_covariance = :theoretical,
            presample_periods = 0,
            verbose = false,
        )
        ll_fom = MacroModelling.get_loglikelihood_per_period(
            model,
            obs_data,
            params;
            algorithm = :first_order,
            filter = :inversion,
            quadratic_matrix_equation_algorithm = :schur,
            presample_periods = 0,
            verbose = false,
        )
        MacroModelling.mix_loglikelihood(ll_fom, ll_rom, gate_probs; config = switching_config)
    end)
    sep_fn = () -> get_loglikelihood(
        model,
        obs_subset,
        params;
        algorithm = :stochastic_extended_path,
        filter = :inversion,
        quadratic_matrix_equation_algorithm = :schur,
        presample_periods = 0,
        sep_periods = Int(case["sep_periods"]),
        sep_order = Int(case["sep_branching_order"]),
        sep_nnodes = Int(case["sep_nnodes"]),
        sep_sparse_tree = Bool(case["sep_sparse_tree"]),
        sep_maxit = Int(case["sep_maxit"]),
        sep_tol = Float64(case["sep_tol"]),
        sep_accept_tol = Float64(case["sep_accept_tol"]),
        sep_inv_maxit = Int(case["sep_inv_maxit"]),
        sep_inv_step_tol = Float64(case["sep_inv_step_tol"]),
        sep_inv_resid_tol = Float64(case["sep_inv_resid_tol"]),
        sep_inv_lambda = Float64(case["sep_inv_lambda"]),
        verbose = false,
    )

    return Dict(
        "model_info" => Dict(
            "n_vars" => length(model.var),
            "n_exo" => length(model.exo),
            "parameter_count" => length(model.parameters),
        ),
        "stages" => Dict(
            "model_load" => load_stage,
            "first_order_solve" => measure_stage(
                first_order_fn,
                Int(case["solve_reps"]);
                serializer = first_order_result,
            ),
            "kalman_value" => measure_stage(kalman_fn, Int(case["kalman_value_reps"])),
            "kalman_per_period" => measure_stage(
                kalman_per_period_fn,
                Int(case["kalman_per_period_reps"]);
                serializer = value -> Dict("per_period" => value),
            ),
            "kalman_paths" => measure_stage(
                kalman_paths_fn,
                Int(case["kalman_paths_reps"]);
                serializer = paths_result,
            ),
            "kalman_grad" => measure_stage(
                kalman_grad_fn,
                Int(case["kalman_grad_reps"]);
                serializer = gradient_result,
            ),
            "gate_stats" => measure_stage(
                gate_stats_fn,
                Int(case["gate_stats_reps"]);
                serializer = gate_stats_result,
            ),
            "switching_fixed" => measure_stage(
                switching_fixed_fn,
                Int(case["switching_fixed_reps"]),
            ),
            "switching_value" => measure_stage(switching_fn, Int(case["switching_reps"])),
            "sep_inversion" => measure_stage(sep_fn, Int(case["sep_reps"])),
        ),
    )
end

function main()
    length(ARGS) == 2 || error("Usage: profile_validation_julia.jl <payload.json> <output.json>")
    payload = JSON.parsefile(ARGS[1])
    results = Dict(
        "language" => "julia",
        "julia_version" => string(VERSION),
        "cases" => Dict{String, Any}(),
    )
    for case in payload["cases"]
        results["cases"][case["name"]] = case_result(case)
    end
    open(ARGS[2], "w") do io
        write(io, JSON.json(results))
    end
end

main()
