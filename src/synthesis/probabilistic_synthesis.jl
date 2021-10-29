using ProgressLogging: @progress
import Plots
using Setfield
using Plots.PlotMeasures: cm

@kwdef(
struct DynamicsFittingSettings
    optim_options::Optim.Options=Optim.Options(f_abstol=1e-4)
    use_bijectors::Bool=true
    evals_per_program::Int=10
    use_distributed::Bool=false
    n_threads::Int=6
end)

compute_avg_score(scores, avg_window, iter) = let
    start = max(iter - avg_window, 1)
    mean(scores[start:iter])
end

struct IterativeSynthesisResult
    dyn_est::NamedTuple
    dyn_history::Vector{<:NamedTuple}
    logp_history::Vector{Vector{Float64}}
    improve_pred_hisotry::Vector{Float64}
end

function Base.show(io::IO, iter_result::IterativeSynthesisResult; 
    first_rows::Int=5, max_rows::Int=15,
)
    let rows=[], step=(length(iter_result.dyn_history)-first_rows)÷max_rows
        for (i, (; comps, params)) in enumerate(iter_result.dyn_history)
            (i <= first_rows || i % step == 0) && push!(rows, (; i, comps, params))
        end
        println(io, "=====IterativeSynthesisResult=====")
        show(io, DataFrame(rows), truncate=100)
    end
end
propertynames(to_measurement([1,2,3]))
function Plots.plot(iter_result::IterativeSynthesisResult; start_idx=1)
    (; logp_history, improve_pred_hisotry) = iter_result
    xs = start_idx:length(logp_history)
    mean_history = mean.(logp_history)
    std_history = std.(logp_history) .* 2
    change_points = _dynamics_change_points(iter_result)
    p1 = plot(
        xs, mean_history[xs], ribbon=(std_history[xs], std_history[xs]),
        title="log p(observations)", label="est. mean")
    vline!(change_points, label="structure change", linestyle=:dash)
    p2 = let 
        improve_actual = mean_history[2:end] .- mean_history[1:end-1]
        data = hcat(improve_actual, improve_pred_hisotry) |> specific_elems
        xs1 = xs[1:end-1]
        plot(xs1, data[xs1], title="iteration improvement",
            label=["est. actual" "predicted"], ylims=[-0.5, 0.5])
        hline!([0.0], label="y=0", line=:dash)
    end
    plot(p1, p2, layout=(2,1), size=(600,800))
end

function _dynamics_change_points(iter_result::IterativeSynthesisResult)
    dh = iter_result.dyn_history
    filter(1:length(dh)-1) do i
        dh[i].comps != dh[i+1].comps
    end |> map(x -> x+0.5)
end

function plot_params(iter_result::IterativeSynthesisResult)
    dh = iter_result.dyn_history
    change_points = _dynamics_change_points(iter_result)
    param_names = keys(dh[1].params)
    N = length(dh[1].params)
    plts = map(1:N) do i 
        ys = (x -> x.params[i]).(dh)
        plot(1:length(dh), ys, title=param_names[i], label="value")
        vline!(change_points, label="structure change", linestyle=:dash)
    end
    W, H = 2, ceil(Int, N / 2)
    plot(plts..., layout=(W, H), size=(400W, 300H), left_margin=1cm)
end

"""
Synthesize the dynamics using an Expectation Maximization-style loop.  

## Arguments
- `transition_dynamics((;x...,u...,params...,Δt), holes) -> distribution_of_x′`.
- `obs_model(x) -> distribtion_of_y`.
- `particle_smoother(stochastic_system, obs_data) -> sample_result`.
"""
function fit_dynamics_iterative(
    senum::SynthesisEnumerationResult,
    obs_data::NamedTuple{(:times, :obs_frames, :observations, :controls)},
    comps_guess::NamedTuple, 
    params_guess::Union{NamedTuple, Nothing};
    obs_model::Function,
    program_logp::Function,
    sampler::PosteriorSampler,
    iteration_callback = (data::NamedTuple{(:iter, :trajectories, :dyn_est)}) -> nothing,
    fit_settings::DynamicsFittingSettings = DynamicsFittingSettings(),
    max_iters::Int=100, n_quick_prune_keep=10,
    p_structure_update_min=0.1,
)
    (; vdata, sketch) = senum
    @assert keys(comps_guess) == tuple((v.name for v in sketch.outputs)...)

    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(sketch)
    empty!(sampler)

    function sample_data((; p_motion_model, params))
        local mm = p_motion_model(params)
        local system = MarkovSystem(x0_dist, mm, obs_model)
        sample_posterior(sampler, system, obs_data)
    end

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        compile_ctx = (; senum.shape_env, senum.comp_env, sketch, hide_type=false)
        (p_motion_model = compile_motion_model(comps_guess, compile_ctx), 
            params = p_est, comps = comps_guess)
    end
    update_id = 1
    
    dyn_history, logp_history, improve_pred_hisotry = NamedTuple[], Vector{Float64}[], Float64[]
    n_structure_synthesis = n_structure_updates = 1

    @progress "fit_dynamics_iterative" for iter in 1:max_iters+1
        (; p_motion_model, params, comps) = dyn_est
        (; trajectories, log_obs) = sample_data(dyn_est)
        trajectories::Vector{<:Vector}
        log_prior = program_logp(comps) + logpdf(params_dist, params)
        log_p = log_prior .+ (log_obs::Vector{Float64})

        push!(dyn_history, dyn_est)
        push!(logp_history, log_p)

        score_old = let
            system = (; motion_model=p_motion_model(params), x0_dist)
            state_scores = states_log_score.(
                Ref(system), Ref(obs_data), trajectories, Float64)
            mean(state_scores) + log_prior
        end

        iteration_callback((;iter, trajectories, dyn_est))

        (iter == max_iters+1) && break
        
        update_id = rand(1:length(comps))
        mode = SingleFittingMode(comps, update_id)
        comp_to_update = keys(comps)[update_id]
        @info "Iteration $iter started." comp_to_update
        p_structure_update = 
            max(p_structure_update_min, n_structure_updates/n_structure_synthesis) |>
            x -> round(x, digits=3)

        if rand() < p_structure_update
            n_structure_synthesis += 1
            println("Use the first trajectory to perform quick pruning...")
            @time begin
                all_comps = _candidate_comps(senum, mode)
                # use the first trajectory to quickly prune away bad candidates
                traj_subsets = trajectories[1:1]
                params_map = Dict{NamedTuple, typeof(params)}()
                fit_r = fit_dynamics(
                    all_comps, senum, traj_subsets, obs_data, params_map;
                    program_logp, fit_settings=@set(fit_settings.evals_per_program=1),
                    specialize_motion_model=false,
                )
            end

            println("Use all trajectories to perform precise synthesis...")
            @time begin
                subset_comps = Iterators.take(fit_r.sorted_results, n_quick_prune_keep) |>
                    map(x -> x.comps) |> collect
                # always keep the previous best in the candidates to ensure robustness
                comps ∈ subset_comps || push!(subset_comps, comps)
                params_map = Dict{NamedTuple, typeof(params)}(comps => params)
                fit_r = fit_dynamics(
                    subset_comps, senum, trajectories, obs_data, params_map;
                    program_logp, fit_settings, specialize_motion_model=true,
                )
            end
            display(fit_r)
            show_top_results(fit_r, 5)
            sol = fit_r.best_result
        
            structure_updated = sol.comps != dyn_est.comps
            n_structure_updates += structure_updated
            dyn_est = (; sol.p_motion_model, sol.params, sol.comps)
        else
            structure_updated = false
            params_map = Dict{NamedTuple, typeof(params)}(comps => params)
            fit_r = fit_dynamics(
                [dyn_est.comps], senum, trajectories, obs_data, params_map;
                program_logp, fit_settings, specialize_motion_model=true,
            )
            show_top_results(fit_r, 1)
            sol = fit_r.best_result
            dyn_est = (; sol.p_motion_model, sol.params, sol.comps)
        end

        improve_pred = sol.score - score_old
        push!(improve_pred_hisotry, improve_pred)
        @info("Iteration $iter finished.", 
            log_p=to_measurement(log_p), improve_pred, 
            p_structure_update, structure_updated)
    end
    empty!(sampler)
    IterativeSynthesisResult(dyn_est, dyn_history, logp_history, improve_pred_hisotry)
end

abstract type DynamicsFittingMode end
struct CombinatorialFittingMode <: DynamicsFittingMode end
struct SingleFittingMode <: DynamicsFittingMode 
    prev_comps::NamedTuple
    update_id::Int
end

"""
Enumerate a single component while keeping others unchanged
"""
function _candidate_comps(senum, mode::SingleFittingMode)
    (; prev_comps, update_id) = mode
    _candidate_comps(senum, prev_comps, Val(update_id))
end

function _candidate_comps(
    senum, prev_comps::NamedTuple, ::Val{update_id},
) where {update_id}
    @assert update_id isa Integer
    update_var = senum.sketch.outputs[update_id]
    update_key = keys(prev_comps)[update_id]
    map(senum.enum_result[update_var.type]) do comp
        merge(prev_comps, NamedTuple{(update_key,)}(tuple(comp)))
    end |> collect
end

function _candidate_comps(senum, ::CombinatorialFittingMode)
    hole_names = tuple((v.name for v in sketch.outputs)...)
    _candidate_comps(senum, Val(hole_names))
end

function _candidate_comps(senum, ::Val{hole_names}) where {hole_names}
    Iterators.product((senum.enum_result[v.type] for v in senum.sketch.outputs)...) |>
        map(comps -> NamedTuple{hole_names}(comps)) |> collect
end

function fit_dynamics(
    all_comps::Vector{<:NamedTuple},
    senum::SynthesisEnumerationResult,
    particles::Vector{<:Vector},
    (; times, observations, controls), 
    params_map::Dict; 
    program_logp::Function,
    fit_settings::DynamicsFittingSettings,
    specialize_motion_model::Bool,
    progress_offset::Int=0,
)::MapSynthesisResult
    (; vdata, sketch, shape_env, comp_env) = senum

    params_dist = params_distribution(sketch)
    x0_dist = init_state_distribution(vdata)

    (; evals_per_program, use_bijectors, n_threads, use_distributed) = fit_settings
    
    compile_ctx = (; shape_env, comp_env, sketch, hide_type=!specialize_motion_model)

    ctx = (; compile_ctx, 
        particles, x0_dist, params_dist, program_logp, optim_options, 
        obs_data=(; times, controls, observations), specialize_motion_model,
        params_map, use_bijectors, evals_per_program)

    progress = Progress(length(all_comps), desc="fit_dynamics", 
        offset=progress_offset, showspeed=true, output=stdout)

    eval_results = parallel_map(evaluate_dynamics, all_comps, ctx; 
        progress, n_threads, use_distributed)

    err_progs = eval_results |> filter(r -> r.status == :errored)
    succeeded = eval_results |> filter(r -> r.status == :success) |> map(r -> r.result)
    isempty(succeeded) && error("Synthesis failed to produce any valid solution.\n" * 
        "First 5 errored programs: $(Iterators.take(err_progs, 5) |> collect)")
    sorted_results = succeeded |> sort_by(r -> -r.score)
    best_result = sorted_results[1]
    @unzip_named (times, :time), (iterations, :iterations), (converges, :converged) =
        map(r -> r.stats, succeeded)
    opt_time, opt_iters, opt_converge_rate = to_measurement.((times, iterations, converges))
    stats = (; n_progs=length(all_comps), n_err_progs=length(err_progs),
        opt_time, opt_iters, opt_converge_rate)
    MapSynthesisResult(best_result, stats, err_progs, sorted_results)
end


function evaluate_dynamics(ctx, comps)
    p_motion_model = compile_motion_model(comps, ctx.compile_ctx)
    n = ctx.evals_per_program
    (; params_map, params_dist) = ctx
    results = map(1:n) do i
        params_guess = 
            if i == 1 || !haskey(params_map, comps)
                rand(params_dist)
            else
                params_map[comps]
            end
        evaluate_dynamics_once(ctx, comps, p_motion_model, params_guess)
    end
    succeeded = results |> filter(x -> x.status == :success)
    if isempty(succeeded)
        errors = results |> map(x -> x.error) |> unique
        (; status= :errored, program=comps, errors)
    else
        result = succeeded |> map(x -> x.result) |> max_by(x -> x.score)
        params_map[result.comps] = result.params
        (; status= :success, result)
    end
end

function evaluate_dynamics_once(ctx, comps, p_motion_model::Function, params_guess)
    (; particles, x0_dist, params_dist, program_logp, obs_data) = ctx
    (; use_bijectors, optim_options) = ctx
    
    function failed(info::String)
        (status= :errored, program=comps, error=ErrorException(info))
    end

    try
        local (; value, time) = @ltimed fit_dynamics_params(
            p_motion_model, particles, 
            x0_dist, params_dist, obs_data, params_guess; 
            use_bijectors, optim_options)

        !isnan(value.f_final) || return failed("logp = NaN.")
        all(isfinite, value.params) || return failed("bad params = $(value.params).")

        stats = merge(value.stats, (time=time,))
        score = program_logp(comps) + value.f_final
        result=(; score, comps, p_motion_model, value.params, value.f_init, value.f_final, stats)
        return (status= :success, result)
    catch err
        if err isa Union{OverflowError, DomainError, BadLossError}
            return (status= :errored, program=comps, error=err)
        else
            @error "Unexpected error encountered during dynamics evaluation" comps params_guess
            rethrow()
        end
    end
end

"""
## Arguments
- `p_motion_model(params::NamedTuple)(x::NamedTuple, u::NamedTuple, Δt::Real) -> distribution_of_x`
"""
function fit_dynamics_params(
    p_motion_model::WrappedFunc,
    trajectories::Vector{<:Vector},
    x0_dist, params_dist,
    obs_data, 
    params_guess::NamedTuple; 
    optim_options::Optim.Options,
    use_bijectors=true,
)
    @assert isfinite(logpdf(params_dist, params_guess))

    if use_bijectors
        p_bj = bijector(params_dist)
        p_inv = inv(p_bj)
        guess_original = params_guess
        guess = p_bj(guess_original)
        @assert structure_to_vec(p_inv(guess)) ≈ structure_to_vec(guess_original)
    else
        p_bj = identity
        lower, upper = _compute_bounds(params_dist)
    end

    vec_to_params = (vec) -> let
        local values = structure_from_vec(params_guess, vec)
        use_bijectors && (values = p_inv(values))
        values
    end

    function state_liklihood(system, ::Type{T})::T where T
        local r::T = 0.0
        for tr in trajectories
            r += states_log_score(system, obs_data, tr, T)
        end
        r /= length(trajectories)
    end

    # Maximize the EM objective
    function loss(vec::Vector{T})::T where T
        local params = vec_to_params(vec)
        all(map(isfinite, params)) || return Inf
        local prior = log_score(params_dist, params, T)
        motion_model = p_motion_model(params)
        local sys_new = (; motion_model, x0_dist)
        local likelihood = state_liklihood(sys_new, T)
        
        -(likelihood + prior)
    end

    pvec_guess = structure_to_vec(p_bj(params_guess))
    f_init = -loss(pvec_guess)
    
    if use_bijectors
        sol = optimize_no_tag(loss, pvec_guess, optim_options)
    else
        sol = optimize_bounded(loss, pvec_guess, (lower, upper), optim_options)
    end
    
    pvec_final = Optim.minimizer(sol)
    f_final = -loss(pvec_final)
    params = vec_to_params(pvec_final)
    stats = (converged=Optim.converged(sol), iterations=Optim.iterations(sol))
    (; f_init, f_final, params, stats)
end

