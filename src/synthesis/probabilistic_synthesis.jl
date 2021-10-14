using ProgressLogging: @progress

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

"""
Synthesize the dynamics using an Expectation Maximization-style loop.  

## Arguments
- `transition_dynamics((;x...,u...,params...,Δt), holes) -> distribution_of_x′`.
- `obs_model(x) -> distribtion_of_y`.
- `particle_smoother(stochastic_system, obs_data) -> sample_result`.
"""
function fit_dynamics_iterative(
    senum::SynthesisEnumerationResult,
    obs_data::NamedTuple{(:times, :observations, :controls)},
    comps_guess::NamedTuple, 
    params_guess::Union{NamedTuple, Nothing};
    obs_model::Function,
    program_logp::Function,
    particle_smoother::Function = (system, obs_data) -> 
        ffbs_smoother(system, obs_data; n_particles=10_000, n_trajs=100),
    fit_settings::DynamicsFittingSettings = DynamicsFittingSettings(),
    max_iters::Int=100,
)
    (; vdata, sketch) = senum
    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(vdata)

    function sample_data((; motion_model, params)) 
        local mm = motion_model(params)
        local system = MarkovSystem(x0_dist, mm, obs_model)
        particle_smoother(system, obs_data)
    end

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        compile_ctx = (; senum.shape_env, senum.comp_env, sketch.combine_holes)
        (motion_model = compile_motion_model(comps_guess, compile_ctx), 
            params = p_est, comps = comps_guess)
    end
    
    dyn_history, logp_history, improve_pred_hisotry = NamedTuple[], Float64[], Float64[]
    previous_result = nothing

    @progress "fit_dynamics_iterative" for iter in 1:max_iters+1
        (; motion_model, params, comps) = dyn_est
        (; particles, log_weights, log_obs) = sample_data(dyn_est)
        log_prior = program_logp(comps) + logpdf(params_dist, params)
        log_p::Float64 = log_prior + log_obs

        push!(dyn_history, dyn_est)
        push!(logp_history, log_p)

        score_old = let
            system = (; motion_model=motion_model(params), x0_dist)
            state_scores = states_likelihood.(
                Ref(system), Ref(obs_data), rows(particles))
            local weights = log_weights === nothing ? 
                1 / length(state_scores) : softmax(log_weights)
            sum(state_scores .* weights) + log_prior
        end

        plot_freq = 4
        if iter % plot_freq == 1
            # TODO: generalize this
            plot_particles(particles, times, "iteration $iter", ex_data.states) |> display
        end

        if iter == max_iters+1
            break
        end
        
        fit_r = fit_dynamics(
            senum, particles, log_weights, obs_data, previous_result;
            program_logp, fit_settings, 
        )
        display(fit_r)
        show_top_results(fit_r, 4)

        sol = fit_r.best_result
        
        dyn_est = (; sol.motion_model, sol.params, sol.comps)
        previous_result = fit_r

        improve_pred = sol.score - score_old
        push!(improve_pred_hisotry, improve_pred)
        @info "Iteration $iter:" log_p improve_pred
    end

    (; dyn_est, dyn_history, logp_history, improve_pred_hisotry)
end

function fit_dynamics(
    senum::SynthesisEnumerationResult,
    particles::Matrix, log_weights::Optional{Vector},
    (; times, observations, controls), 
    previous_result::Union{MapSynthesisResult, Nothing}; 
    program_logp::Function,
    fit_settings::DynamicsFittingSettings,
    progress_offset::Int=0,
)
    (; vdata, sketch, shape_env, comp_env, enum_result) = senum

    hole_names = tuple((v.name for v in sketch.holes)...)
    all_comps = Iterators.product((enum_result[v.type] for v in sketch.holes)...) |>
        map(comps -> NamedTuple{hole_names}(comps)) |> collect
    params_dist = params_distribution(vdata)
    x0_dist = init_state_distribution(vdata)

    ast_to_params = map_optional(previous_result) do pv
        Dict(comps => params for (; comps, params) in pv.sorted_results)
    end

    (; evals_per_program, use_bijectors, n_threads, use_distributed) = fit_settings
    
    compile_ctx = (; shape_env, comp_env, sketch.combine_holes)

    ctx = (; compile_ctx, 
        particles, log_weights, x0_dist, params_dist, program_logp, optim_options, 
        obs_data=(; times, controls, observations), 
        ast_to_params, use_bijectors, evals_per_program)

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
    motion_model = compile_motion_model(comps, ctx.compile_ctx)
    n = ctx.evals_per_program
    (; ast_to_params, params_dist) = ctx
    results = map(1:n) do i
        params_guess = if ast_to_params !== nothing && i == 1
            # try the previous parameters first
            ast_to_params[comps]
        else
            rand(params_dist)
        end
        evaluate_dynamics_once(ctx, comps, motion_model, params_guess)
    end
    succeeded = results |> filter(x -> x.status == :success)
    if isempty(succeeded)
        errors = results |> map(x -> x.error) |> unique
        (; status= :errored, program=comps, errors)
    else
        result = succeeded |> map(x -> x.result) |> max_by(x -> x.score)
        (; status= :success, result)
    end
end

function evaluate_dynamics_once(ctx, comps, motion_model, params_guess)
    (; particles, log_weights, x0_dist, params_dist, program_logp, obs_data) = ctx
    (; use_bijectors, optim_options) = ctx
    try
        local (; value, time) = @ltimed fit_dynamics_params(
            motion_model, particles, log_weights, 
            x0_dist, params_dist, obs_data, params_guess; 
            use_bijectors, optim_options)
        if isnan(value.f_final)
            return (status= :errored, program=comps, error=ErrorException("logp = NaN."))
        end
        stats = merge(value.stats, (time=time,))
        score = program_logp(comps) + value.f_final
        result=(; score, comps, motion_model, value.params, value.f_init, value.f_final, stats)
        return (status= :success, result)
    catch err
        if err isa Union{OverflowError, DomainError, BadLossError}
            return (status= :errored, program=comps, error=err)
        else
            @error "Unexpected error encountered during dynamics evaluation" comps f_holes
            rethrow()
        end
    end
end

"""
## Arguments
- `motion_model(params)(x::NamedTuple, u::NamedTuple, Δt::Real) -> distribution_of_x`
"""
function fit_dynamics_params(
    motion_model::Function,
    particles::Matrix, log_weights::Optional{Vector},
    x0_dist, params_dist,
    obs_data, 
    params_guess::NamedTuple; 
    optim_options::Optim.Options,
    use_bijectors=false,
)
    @assert isfinite(logpdf(params_dist, params_guess))

    trajectories = rows(particles)
    traj_weights = log_weights === nothing ? 1 / length(trajectories) : softmax(log_weights)
    
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

    # Maximize the EM objective
    function loss(vec)
        local params = vec_to_params(vec)
        local prior = logpdf(params_dist, params)
        local sys_new = (; motion_model = motion_model(params), x0_dist)
        local state_scores′ = states_likelihood.(
            Ref(sys_new), Ref(obs_data), trajectories)

        -(sum(state_scores′ .* traj_weights) + prior)
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

function compile_motion_model(
    comps::NamedTuple,
    (; shape_env, comp_env, combine_holes),
)
    funcs = map(comp -> compile(comp, shape_env, comp_env; check_gradient=false), comps)
    to_motion_model(funcs, combine_holes)
end

function to_motion_model(hole_fs::NamedTuple, combine_holes)
    (params) -> (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
        local inputs = merge(x, u, params, (;Δt))
        local holes = map(f -> f(inputs), hole_fs)
        combine_holes(inputs, holes)::GDistr
    end
end