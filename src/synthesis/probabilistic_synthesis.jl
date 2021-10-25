using ProgressLogging: @progress
import Plots

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
    logp_history::Vector{Float64}
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

function Plots.plot(iter_result::IterativeSynthesisResult; start_idx=1)
    (; logp_history, improve_pred_hisotry) = iter_result
    xs = start_idx:length(logp_history)
    p1 = plot(
        xs, logp_history[xs], xlabel="iterations", 
        title="log p(observations)", legend=false)
    p2 = let 
        improve_actual = logp_history[2:end] .- logp_history[1:end-1]
        data = hcat(improve_actual, improve_pred_hisotry) |> specific_elems
        plot(data, xlabel="iterations", title="iteration improvement",
            label=["est. actual" "predicted"], ylims=[-0.5, 0.5])
        hline!([0.0], label="y=0", line=:dash)
    end
    plot(p1, p2, layout=(2,1), size=(600,800))
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
    particle_smoother::Function = (system, obs_data) -> 
        filter_smoother(system, obs_data; n_particles=1_000_000, n_trajs=64),
    iteration_callback = (data::NamedTuple{(:iter, :trajectories, :dyn_est)}) -> nothing,
    fit_settings::DynamicsFittingSettings = DynamicsFittingSettings(),
    max_iters::Int=100,
)
    (; vdata, sketch) = senum
    @assert keys(comps_guess) == tuple((v.name for v in sketch.outputs)...)

    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(sketch)

    function sample_data((; p_motion_model, params)) 
        local mm = p_motion_model(params)
        local system = MarkovSystem(x0_dist, mm, obs_model)
        particle_smoother(system, obs_data)
    end

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        compile_ctx = (; senum.shape_env, senum.comp_env, sketch)
        (p_motion_model = compile_motion_model(comps_guess, compile_ctx), 
            params = p_est, comps = comps_guess)
    end
    update_id = 1
    
    dyn_history, logp_history, improve_pred_hisotry = NamedTuple[], Float64[], Float64[]
    # maps programs into best previous parameters
    params_map = Dict{NamedTuple, typeof(dyn_est.params)}()

    @progress "fit_dynamics_iterative" for iter in 1:max_iters+1
        (; p_motion_model, params, comps) = dyn_est
        (; trajectories, log_obs) = sample_data(dyn_est)
        log_prior = program_logp(comps) + logpdf(params_dist, params)
        log_p::Float64 = log_prior + log_obs

        push!(dyn_history, dyn_est)
        push!(logp_history, log_p)

        score_old = let
            system = (; motion_model=p_motion_model(params), x0_dist)
            state_scores = states_log_score.(
                Ref(system), Ref(obs_data), get_rows(trajectories), Float64)
            mean(state_scores) + log_prior
        end

        Threads.@spawn iteration_callback((;iter, trajectories, dyn_est))

        if iter == max_iters+1
            break
        end
        
        mode = SingleFittingMode(comps, update_id)
        comp_to_update = keys(comps)[update_id]
        @info "Iteration $iter started." comp_to_update
        fit_r = fit_dynamics(
            mode, senum, trajectories, obs_data, params_map;
            program_logp, fit_settings, 
        )
        display(fit_r)
        show_top_results(fit_r, 4)

        sol = fit_r.best_result
        
        dyn_est = (; sol.p_motion_model, sol.params, sol.comps)
        update_id = mod1(update_id + 1, length(comps))
        previous_result = fit_r

        improve_pred = sol.score - score_old
        push!(improve_pred_hisotry, improve_pred)
        @info "Iteration $iter finished." log_p improve_pred
    end

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
function _candidate_comps(senum, sketch, mode::SingleFittingMode)
    (; prev_comps, update_id) = mode
    _candidate_comps(senum, sketch, prev_comps, Val(update_id))
end

function _candidate_comps(
    senum, sketch, prev_comps::NamedTuple, ::Val{update_id},
) where {update_id}
    @assert update_id isa Integer
    update_var = sketch.outputs[update_id]
    update_key = keys(prev_comps)[update_id]
    map(senum.enum_result[update_var.type]) do comp
        merge(prev_comps, NamedTuple{(update_key,)}(tuple(comp)))
    end |> collect
end

function _candidate_comps(senum, sketch, ::CombinatorialFittingMode)
    hole_names = tuple((v.name for v in sketch.outputs)...)
    _candidate_comps(senum, sketch, Val(hole_names))
end

function _candidate_comps(senum, sketch, ::Val{hole_names}) where {hole_names}
    Iterators.product((senum.enum_result[v.type] for v in sketch.outputs)...) |>
        map(comps -> NamedTuple{hole_names}(comps)) |> collect
end

function fit_dynamics(
    mode::DynamicsFittingMode,
    senum::SynthesisEnumerationResult,
    particles::Matrix,
    (; times, observations, controls), 
    params_map::Dict; 
    program_logp::Function,
    fit_settings::DynamicsFittingSettings,
    progress_offset::Int=0,
)
    (; vdata, sketch, shape_env, comp_env, enum_result) = senum

    all_comps = _candidate_comps(senum, sketch, mode)
    params_dist = params_distribution(sketch    )
    x0_dist = init_state_distribution(vdata)

    (; evals_per_program, use_bijectors, n_threads, use_distributed) = fit_settings
    
    compile_ctx = (; shape_env, comp_env, sketch)

    ctx = (; compile_ctx, 
        particles, x0_dist, params_dist, program_logp, optim_options, 
        obs_data=(; times, controls, observations), 
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
    @nospecialize p_motion_model
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
    trajectories::Matrix,
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

    traj_list = get_rows(trajectories)
    function state_liklihood(system, ::Type{T})::T where T
        local r::T = 0.0
        for tr in traj_list
            r += states_log_score(system, obs_data, tr, T)
        end
        r /= length(traj_list)
    end

    # Maximize the EM objective
    function loss(vec::Vector{T})::T where T
        local params = vec_to_params(vec)
        all(map(isfinite, params)) || return Inf
        local prior = log_score(params_dist, params, T)
        local sys_new = (; motion_model = p_motion_model(params), x0_dist)
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

