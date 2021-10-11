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

"""
Synthesize the dynamics using an Expectation Maximization-style loop.  
"""
function fit_dynamics_iterative(
    senum::SynthesisEnumerationResult,
    obs_data::NamedTuple{(:times, :observations, :controls)},
    f_x′′_guess::Function, 
    params_guess::Union{NamedTuple, Nothing},
    σ_guess::NamedTuple; 
    program_logp::Function,
    n_trajs::Int=100, n_particles::Int=10_000, resample_threshold=0.5,
    fit_settings::DynamicsFittingSettings = DynamicsFittingSettings(),
    trust_thresholds=(0.25, 0.75), λ_multiplier=2.0,
    max_iters::Int=100, patience::Int=10,
    λ0::Float64=1.0, 
)
    vdata = senum.vdata
    x0_dist = init_state_distribution(vdata)
    params_dist = params_distribution(vdata)

    # TODO: generalize this
    function sample_data((f_x′′, params); debug=false) 
        system = car1d_system(params, x0_dist, f_x′′, σ_guess)
        sample_posterior_data(system, obs_data; n_particles, max_trajs=n_trajs, 
        resample_threshold, debug, use_ffbs=true)
    end

    dyn_est = let p_est = params_guess === nothing ? rand(params_dist) : params_guess
        (f_x′′_guess, p_est)
    end
    λ = λ0
    
    dyn_history, score_history, λ_history = [], [], []
    previous_result = nothing
    best_dyn=nothing
    best_iter=0

    @progress "fit_dynamics_iterative" for iter in 1:max_iters
        if iter - best_iter > patience
            @info "Synthesis stopped because no better solutino was found in the \
                last $patience iterations."
            break
        end

        (; particles, log_weights, perf) = sample_data(dyn_est)

        fit_r = fit_dynamics(
            senum, particles, log_weights, obs_data, previous_result;
            σ_guess, program_logp, λ, fit_settings, 
        )
        println("Iteration $iter:")
        display(fit_r)
        show_top_results(fit_r, 4)
        println()
        sol = fit_r.best_result
        dyn_new = (sol.f_x′′, sol.params)
        perf_new = sample_data(dyn_new).perf

        if best_dyn === nothing || perf_new > best_dyn.perf
            best_dyn = (dyn=dyn_new, perf=perf_new)
            best_iter = iter
        end

        improve_actual = perf_new - perf
        improve_pred = sol.f_final.perf - sol.f_init.perf
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
        previous_result = fit_r

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
    previous_result::Union{MapSynthesisResult, Nothing}; 
    σ_guess::NamedTuple,
    program_logp::Function,
    λ::Float64, 
    fit_settings::DynamicsFittingSettings,
    progress_offset::Int=0,
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

    ast_to_params = map_optional(previous_result) do pv
        Dict(f_x′′.ast => params for (; f_x′′, params) in pv.sorted_results)
    end

    (; evals_per_program, use_bijectors, n_threads, use_distributed) = fit_settings

    ctx = (; compile_ctx, 
        particles, log_weights, x0_dist, params_dist, program_logp, optim_options, 
        obs_data=(; times, controls, observations), 
        ast_to_params, λ, σ_guess, use_bijectors, evals_per_program)

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
    compile_ctx = ctx.compile_ctx
    (; hole_names, output_names) = compile_ctx
    f_x′′ = compile_sketch(comps, hole_names, output_names, compile_ctx)
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
    (; λ, use_bijectors, optim_options, σ_guess) = ctx
    try
        local (; value, time) = @ltimed fit_dynamics_params(
            f_x′′, particles, log_weights, 
            x0_dist, params_dist, obs_data, params_guess, σ_guess; 
            λ, use_bijectors, optim_options)
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
    obs_data, 
    params_guess::NamedTuple, 
    σ_guess::NamedTuple; 
    λ::Float64,
    optim_options::Optim.Options,
    use_bijectors=false,
)
    @assert isfinite(logpdf(params_dist, params_guess))

    trajectories = rows(particles)
    sys_guess = car1d_system(params_guess, x0_dist, f_x′′, σ_guess)
    data_scores = data_likelihood.(
        Ref(sys_guess), Ref(obs_data), trajectories)
    state_scores = states_likelihood.(
        Ref(sys_guess), Ref(obs_data), trajectories)
    
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

    # Maximize
    function loss(vec; return_details::Bool=false)
        local params = vec_to_params(vec)
        local prior = logpdf(params_dist, params)
        local sys_new = car1d_system(params, x0_dist, f_x′′, σ_guess)
        local state_scores′ = states_likelihood.(
            Ref(sys_new), Ref(obs_data), trajectories)

        local perf = sample_performance(state_scores′ - state_scores + log_weights, 
            data_scores, state_scores′).perf
        local kl = 0.5mean((state_scores′ .- state_scores).^2)
        local regularization = -λ * kl
        local score = perf + prior + regularization
        if return_details
            (; score, perf, prior, regularization)
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