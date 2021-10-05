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

function fit_dynamics(
    senum::SynthesisEnumerationResult,
    particles::Matrix, log_weights::Vector, 
    times, (; controls, observations), 
    params_guess::Union{NamedTuple, Nothing}; 
    λ::Float64, use_bijectors=false,
    n_threads=6,
    use_distributed::Bool=false,
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
        particles, log_weights, x0_dist, params_dist, times, 
        obs_data=(; controls, observations), params_guess, λ, use_bijectors)

    progress = Progress(length(all_comps), desc="fit_dynamics", 
        showspeed=true, output=stdout)

    eval_results = parallel_map(evaluate_dynamics, all_comps, ctx; 
        progress, n_threads, use_distributed)

    err_progs = eval_results |> filter(r -> r.status == :errored) |> collect
    succeeded = eval_results |> filter(r -> r.status == :success) |> map(r -> r.result)
    isempty(succeeded) && error("Synthesis failed to produce any valid solution.\n" * 
        "First 5 errored programs: $(Iterators.take(err_progs, 5) |> collect)")
    sorted_results = succeeded |> sort_by(r -> -r.logp)
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
    (; particles, log_weights, x0_dist, params_dist, times, obs_data) = ctx
    (;  params_guess, λ, use_bijectors) = ctx
    p₀ = params_guess === nothing ? rand(params_dist) : params_guess
    try
        local (; value, time) = @ltimed fit_dynamics_params(f_x′′, particles, log_weights, 
            x0_dist, params_dist, times, obs_data, p₀; λ, use_bijectors)
        if isnan(value.f_final)
            return (status= :errored, program=comps, error=ErrorException("logp = NaN."))
        end
        stats = merge(value.stats, (time=time,))
        return (status= :success, result=(; logp=value.f_final, f_x′′, value.params, stats))
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
function expected_log_p(log_prior, log_data_p, log_state_p; debug = false)
    # compute the importance wights of the samples under the new dynamics        
    local weights = softmax(log_prior)
    if debug
        max_ratio = maximum(weights)/minimum(weights)
        ess = effective_particles(weights)
        @info expected_log_p ess max_ratio
    end
    # compute the (weighted) expectated data log probability
    sum(weights .* (log_data_p + log_state_p))
end

function sample_performance(log_prior, log_data_p, log_state_p; debug = false)
    expected_log_p(log_prior, log_data_p, log_state_p; debug)
    # expected_data_p(log_prior, log_data_p, log_state_p; debug)
end

function fit_dynamics_params(
    f_x′′::Function,
    particles::Matrix, log_weights::Vector, 
    x0_dist, params_dist,
    times, (; controls, observations), 
    params_guess::NamedTuple; λ::Float64,
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

    function loss(vec; use_λ::Bool=true)
        local params = vec_to_params(vec)
        local prior = logpdf(params_dist, params)
        local state_scores′ = states_likelihood.(
            trajectories, Ref(times), Ref(x0_dist), 
            Ref(car_motion_model(params; f_x′′)), Ref(controls))

        local perf = sample_performance(state_scores′ - state_scores + log_weights, 
            data_scores, state_scores′)
        local kl = 0.5mean((state_scores′ .- state_scores).^2)
        -(prior + perf - use_λ * λ * kl)
    end

    pvec_guess = structure_to_vec(p_bj(params_guess))
    f_init = -loss(pvec_guess; use_λ=false)
    alg = Optim.LBFGS()
    optim_options = Optim.Options(
        x_abstol=1e-3, f_abstol=1e-3, iterations=500,
        outer_x_abstol=1e-3, outer_f_abstol=0.001, outer_iterations=50)
    
    if use_bijectors
        sol = optimize_no_tag(loss, pvec_guess, optim_options)
    else
        sol = optimize_bounded(loss, pvec_guess, (lower, upper), optim_options)
    end
    
    Optim.converged(sol) || display(sol)
    pvec_final = Optim.minimizer(sol)
    f_final = -loss(pvec_final; use_λ=false)
    params = vec_to_params(pvec_final)
    stats = (converged=Optim.converged(sol), iterations=Optim.iterations(sol))
    (; f_init, f_final, params, stats)
end