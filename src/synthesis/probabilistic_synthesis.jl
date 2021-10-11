using ProgressLogging: @progress

struct DynamicsEnumerator
    comp_env::ComponentEnv
    vdata::VariableData
    action_vars::Vector{Var}
end

function synthesis_enumeration(enumerator::DynamicsEnumerator, inputs, max_ast_size)
    (; comp_env, vdata, action_vars) = enumerator
    pruner = IOPruner(; inputs, comp_env)
    synthesis_enumeration(
        vdata, sketch, action_vars, comp_env, max_ast_size; pruner)
end

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
"""
function fit_dynamics_iterative(
    senum::SynthesisEnumerationResult,
    obs_data::NamedTuple{(:times, :observations, :controls)},
    comps_guess::Tuple{Vararg{TAST}}, 
    params_guess::Union{NamedTuple, Nothing},
    σ_guess::NamedTuple; 
    program_logp::Function,
    n_trajs::Int=100, n_particles::Int=10_000, resample_threshold=0.5,
    fit_settings::DynamicsFittingSettings = DynamicsFittingSettings(),
    max_iters::Int=100,
)
    vdata = senum.vdata
    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(vdata)

    # TODO: generalize this
    function sample_data((; f_x′′, params)) 
        system = car1d_system(params, x0_dist, f_x′′, σ_guess)
        ffbs_smoother(system, obs_data; n_particles, n_trajs, resample_threshold)
    end

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        compile_ctx = mk_compile_ctx(senum.sketch, senum.vdata.state′′_vars, senum.comp_env)
        f_x′′_guess = compile_dynamics(compile_ctx, comps_guess)
        (f_x′′ = f_x′′_guess, params = p_est, comps = comps_guess)
    end
    
    dyn_history, logp_history = [], [], []
    previous_result = nothing

    @progress "fit_dynamics_iterative" for iter in 1:max_iters
        (; params, comps) = dyn_est
        (; particles, log_weights, log_obs) = sample_data(dyn_est)
        log_prior = program_logp(comps) + logpdf(params_dist, params)
        log_p::Float64 = log_prior + log_obs

        push!(dyn_history, dyn_est)
        push!(logp_history, log_p)

        plot_freq = 4
        if iter % plot_freq == 1
            # TODO: generalize this
            plot_particles(particles, times, "iteration $iter", ex_data.states) |> display
        end

        if iter == max_iters
            break
        end
        
        fit_r = fit_dynamics(
            senum, particles, log_weights, obs_data, previous_result;
            σ_guess, program_logp, fit_settings, 
        )
        display(fit_r)
        show_top_results(fit_r, 4)

        sol = fit_r.best_result
        
        dyn_est = (; sol.f_x′′, sol.params, sol.comps)
        previous_result = fit_r

        @info "Iteration $iter:" log_p sol.f_init sol.f_final
    end

    (; dyn_est, dyn_history, logp_history)
end

function fit_dynamics(
    senum::SynthesisEnumerationResult,
    particles::Matrix, log_weights::Vector, 
    (; times, observations, controls), 
    previous_result::Union{MapSynthesisResult, Nothing}; 
    σ_guess::NamedTuple,
    program_logp::Function,
    fit_settings::DynamicsFittingSettings,
    progress_offset::Int=0,
)
    (; vdata, sketch, comp_env, enum_result) = senum
    (; state′′_vars) = vdata

    all_comps = collect(Iterators.product((enum_result[v.type] for v in sketch.holes)...))
    params_dist = params_distribution(vdata)
    x0_dist = init_state_distribution(vdata)

    ast_to_params = map_optional(previous_result) do pv
        Dict(f_x′′.ast => params for (; f_x′′, params) in pv.sorted_results)
    end

    (; evals_per_program, use_bijectors, n_threads, use_distributed) = fit_settings
    
    compile_ctx = mk_compile_ctx(sketch, state′′_vars, comp_env)

    ctx = (; compile_ctx, 
        particles, log_weights, x0_dist, params_dist, program_logp, optim_options, 
        obs_data=(; times, controls, observations), 
        ast_to_params, σ_guess, use_bijectors, evals_per_program)

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

function compile_dynamics(compile_ctx, comps)
    (; hole_names, output_names) = compile_ctx
    compile_sketch(comps, hole_names, output_names, compile_ctx)
end

function mk_compile_ctx(sketch, state′′_vars, comp_env)
    (
        hole_names = Val(Tuple(v.name for v in sketch.holes)),
        output_names = Val(tuple((v.name for v in state′′_vars)...)),
        shape_env, comp_env, sketch.combine, check_gradient=false,
    )
end

function evaluate_dynamics(ctx, comps)
    f_x′′ = compile_dynamics(ctx.compile_ctx, comps)
    evaluate_dynamics(ctx, comps, f_x′′)
end

function evaluate_dynamics(ctx, comps, f_x′′)
    n = ctx.evals_per_program
    (; ast_to_params, params_dist) = ctx
    results = map(1:n) do i
        params_guess = if ast_to_params !== nothing && i == 1
            # try the previous parameters first
            ast_to_params[f_x′′.ast]
        else
            rand(params_dist)
        end
        evaluate_dynamics_once(ctx, comps, f_x′′, params_guess)
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

function evaluate_dynamics_once(ctx, comps, f_x′′, params_guess)
    (; particles, log_weights, x0_dist, params_dist, program_logp, obs_data) = ctx
    (; use_bijectors, optim_options, σ_guess) = ctx
    try
        local (; value, time) = @ltimed fit_dynamics_params(
            f_x′′, particles, log_weights, 
            x0_dist, params_dist, obs_data, params_guess, σ_guess; 
            use_bijectors, optim_options)
        if isnan(value.f_final)
            return (status= :errored, program=comps, error=ErrorException("logp = NaN."))
        end
        stats = merge(value.stats, (time=time,))
        score = program_logp(comps) + value.f_final
        result=(; score, f_x′′, comps, value.params, value.f_init, value.f_final, stats)
        return (status= :success, result)
    catch err
        if err isa Union{OverflowError, DomainError, BadLossError}
            return (status= :errored, program=comps, error=err)
        else
            @error "Unexpected error encountered during dynamics evaluation" comps f_x′′
            rethrow()
        end
    end
end

function fit_dynamics_params(
    f_x′′::Function,
    particles::Matrix, log_weights::Vector, 
    x0_dist, params_dist,
    obs_data, 
    params_guess::NamedTuple, 
    σ_guess::NamedTuple; 
    optim_options::Optim.Options,
    use_bijectors=false,
)
    @assert isfinite(logpdf(params_dist, params_guess))

    trajectories = rows(particles)
    traj_weights = softmax(log_weights)
    
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
        local sys_new = car1d_system(params, x0_dist, f_x′′, σ_guess)
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