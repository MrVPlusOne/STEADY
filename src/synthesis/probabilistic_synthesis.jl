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

"""
Synthesize the dynamics using an Expectation Maximization-style loop.  
"""
function fit_dynamics_iterative(
    senum::SynthesisEnumerationResult,
    obs_data::NamedTuple{(:times, :observations, :controls)},
    f_x′′_guess::Function, 
    params_guess::Union{NamedTuple, Nothing}; 
    program_logp::Function,
    optim_options::Optim.Options,
    n_trajs::Int=100, n_particles::Int=10_000,
    trust_thresholds=(0.25, 0.75), λ_multiplier=2.0,
    max_iters::Int=100, patience::Int=10,
    λ0::Float64=1.0, 
    use_bijectors=false,
    evals_per_program=10,
    use_distributed::Bool=false,
    n_threads=6,
)
    vdata = senum.vdata
    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(vdata)

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        (f_x′′_guess, p_est)
    end
    λ = λ0

    dyn_history, score_history, λ_history = [], [], []
    best_dyn=nothing
    best_iter=0

    # TODO: generalize this
    function sample_data((f_x′′, params); debug=false) 
        system = params_to_system(params, x0_dist, f_x′′, σ=(σ_pos=0.2, σ_pos′=0.6))
        sample_posterior_data(system, obs_data; n_particles, max_trajs=n_trajs, debug, 
        use_ffbs=true)
    end

    for iter in 1:max_iters
        if iter - best_iter > patience
            @info "Synthesis stopped because no better solutino was found in the \
                last $patience iterations."
            break
        end

        # TODO: record time taken
        (; particles, log_weights, perf) = sample_data(dyn_est)
        # perf2 = sample_data(dyn_est).perf
        # perfΔ = abs(perf2 - perf)
        # if perfΔ > 1.0
        #     @info "High performance variance" perfΔ perf perf2
        # end

        fit_r = fit_dynamics(
            senum,
            particles, log_weights, 
            obs_data,
            params_guess;
            program_logp, λ, use_bijectors, optim_options,
            evals_per_program, use_distributed, n_threads
        )
        println("Iteration $iter:")
        display(fit_r)
        show_top_results(fit_r, 4)
        sol = fit_r.best_result
        dyn_new = (sol.f_x′′, sol.params)
        perf_new = sample_data(dyn_new).perf

        if best_dyn === nothing || perf_new > best_dyn.perf
            best_dyn = (dyn=dyn_new, perf=perf_new)
            best_iter = iter
        end

        improve_actual = perf_new - perf
        improve_pred = sol.f_final.logp - sol.f_init.logp
        ρ = improve_actual / improve_pred
        
        if ρ > trust_thresholds[2]
            λ /= λ_multiplier
        elseif ρ < trust_thresholds[1]
            λ *= λ_multiplier
        end
        if ρ < 0
            @warn("Failed to improve the objective.")
        end
        dyn_est = dyn_new

        @info "Optimization details:" sol.f_final improve_pred improve_actual ρ
        @info "Optimal synthesis finished." λ perf_old=perf perf_new

        push!(dyn_history, dyn_est)
        push!(score_history, perf)
        push!(λ_history, λ)

        plot_freq = 4
        if iter % plot_freq == 1
            # TODO: generalize this
            plot_particles(particles, times, "iteration $iter", ex_data.states) |> display
        end
    end
    (; best_dyn, dyn_history, score_history, λ_history)
end

function fit_dynamics(
    senum::SynthesisEnumerationResult,
    particles::Matrix, log_weights::Vector, 
    (; times, observations, controls), 
    params_guess::Union{NamedTuple, Nothing}; 
    program_logp::Function,
    λ::Float64, use_bijectors=false,
    optim_options::Optim.Options,
    evals_per_program=10,
    use_distributed::Bool=false,
    n_threads=6,
)
    (; vdata, sketch, comp_env, enum_result, param_vars) = senum
    (; state_vars, state′′_vars) = vdata
    sketch
    enum_result[]

    all_comps = collect(Iterators.product((enum_result[v.type] for v in sketch.holes)...))
    params_dist = params_distribution(vdata)
    x0_dist = init_state_distribution(vdata)

    compile_ctx = (
        hole_names = Val(Tuple(v.name for v in sketch.holes)),
        output_names = Val(tuple((v.name for v in state′′_vars)...)),
        shape_env, comp_env, sketch.combine, check_gradient=false,
    )

    ctx = (; compile_ctx, 
        particles, log_weights, x0_dist, params_dist, program_logp, optim_options, 
        evals_per_program, obs_data=(; times, controls, observations), 
        params_guess, λ, use_bijectors)

    progress = Progress(length(all_comps), desc="fit_dynamics", 
        showspeed=true, output=stdout)

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
    compile_ctx = ctx.compile_ctx
    (; hole_names, output_names) = compile_ctx
    f_x′′ = compile_sketch(comps, hole_names, output_names, compile_ctx)
    evaluate_dynamics(ctx, comps, f_x′′)
end

function evaluate_dynamics(ctx, comps, f_x′′)
    n = ctx.evals_per_program
    results = map(1:n) do _
        evaluate_dynamics_once(ctx, comps, f_x′′)
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

function evaluate_dynamics_once(ctx, comps, f_x′′)
    (; particles, log_weights, x0_dist, params_dist, program_logp, obs_data) = ctx
    (;  params_guess, λ, use_bijectors, optim_options) = ctx
    p₀ = params_guess === nothing ? rand(params_dist) : params_guess
    try
        local (; value, time) = @ltimed fit_dynamics_params(f_x′′, particles, log_weights, 
            x0_dist, params_dist, obs_data, p₀; λ, use_bijectors, optim_options)
        if isnan(value.f_final.score)
            return (status= :errored, program=comps, error=ErrorException("logp = NaN."))
        end
        stats = merge(value.stats, (time=time,))
        score = program_logp(comps) + value.f_final.score
        result=(; score, f_x′′, value.params, value.f_init, value.f_final, stats)
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

function expected_data_p(log_prior, log_data_p, log_state_p; debug = false)
    # compute the importance wights of the samples under the new dynamics        
    local log_weights = log_softmax(log_prior .- log_data_p)
    if debug
        n_samples = length(log_prior)
        max_ratio = exp(maximum(log_weights) - minimum(log_weights))
        ess_prior = effective_particles(softmax(log_prior))
        ess_data_inv = effective_particles(softmax(-log_data_p))
        ess = effective_particles(softmax(log_weights .+ log_data_p))
        @info expected_data_p n_samples ess ess_data_inv ess_prior max_ratio
    end
    # compute the (weighted) expectated data log probability
    logsumexp(log_weights .+ log_data_p)
end

"""
Compute the posterior average log data likelihood, ``∫ log(p(y|x)) p(x|y,f) dx``.
"""
function expected_log_p(log_prior, log_data_p, log_state_p; debug::Bool = false)
    # compute the importance wights of the samples under the new dynamics        
    local weights = softmax(log_prior)
    # compute the (weighted) expectated data log probability
    # perf = sum(weights .* (log_data_p + log_state_p))
    perf = mean(log_data_p + log_state_p)
    if debug
        max_piror_ratio = maximum(weights)/minimum(weights)
        ess_prior = effective_particles(weights)
        ess_data = effective_particles(softmax(log_data_p))
        ess_state = effective_particles(softmax(log_state_p))
        ess_total = effective_particles(softmax(log_data_p + log_state_p))
        stats = (;max_piror_ratio, ess_prior, ess_data, ess_state, ess_total)
    else
        stats = ()
    end
    (; perf, stats)
end

function sample_performance(log_prior, log_data_p, log_state_p; debug = false)
    expected_log_p(log_prior, log_data_p, log_state_p; debug)
    # expected_data_p(log_prior, log_data_p, log_state_p; debug)
end

function fit_dynamics_params(
    f_x′′::Function,
    particles::Matrix, log_weights::Vector, 
    x0_dist, params_dist,
    (;times, observations, controls), 
    params_guess::NamedTuple; 
    λ::Float64,
    optim_options::Optim.Options,
    use_bijectors=false,
)
    @assert isfinite(logpdf(params_dist, params_guess))

    trajectories = rows(particles)
    data_scores = data_likelihood.(
        trajectories, Ref(car_obs_model), Ref(observations))
    state_scores = states_likelihood.(
        trajectories, Ref(times), Ref(x0_dist), 
        Ref(car_motion_model(params_guess; f_x′′)), Ref(controls))
    
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

    function loss(vec; return_details::Bool=false)
        local params = vec_to_params(vec)
        local prior = logpdf(params_dist, params)
        local state_scores′ = states_likelihood.(
            trajectories, Ref(times), Ref(x0_dist), 
            Ref(car_motion_model(params; f_x′′)), Ref(controls))

        local logp = sample_performance(state_scores′ - state_scores + log_weights, 
            data_scores, state_scores′).perf + prior
        local kl = 0.5mean((state_scores′ .- state_scores).^2)
        local regularization = -λ * kl
        local score = logp + regularization
        if return_details
            (; score, regularization, logp)
        else
            -score
        end
    end

    pvec_guess = structure_to_vec(p_bj(params_guess))
    f_init = loss(pvec_guess; return_details=true)
    
    if use_bijectors
        sol = optimize_no_tag(loss, pvec_guess, optim_options)
    else
        sol = optimize_bounded(loss, pvec_guess, (lower, upper), optim_options)
    end
    
    pvec_final = Optim.minimizer(sol)
    f_final = loss(pvec_final; return_details=true)
    params = vec_to_params(pvec_final)
    stats = (converged=Optim.converged(sol), iterations=Optim.iterations(sol))
    (; f_init, f_final, params, stats)
end