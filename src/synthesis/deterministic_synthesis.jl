export map_synthesis
"""
Perform Maximum a posteriori (MAP) synthesis to find the joint assignment of the motion 
model *and* the trajecotry that maximizes the posterior probability.

The system dynamics are assuemd to be 2nd-order.

- `program_logp(prog::TAST) -> logp` should return the log piror probability of a given 
dynamics program. 
- `data_likelihood(trajectory, other_vars) -> logp_score` 
should return the log probability density (plus some fixed constant) of the observation.
- `max_size`: the maximal AST size of each component program to consider.
"""
function map_synthesis(
    senum::SynthesisEnumerationResult,
    shape_env::ShapeEnv,
    actions::TimeSeries{<:NamedTuple},
    times::AbstractVector,
    program_logp::Function,
    data_likelihood::Function;
    evals_per_program::Int = 10,
    trials_per_eval::Int = 10,
    optim_options=Optim.Options(),
    use_bijectors::Bool=true,
    n_threads::Int=min(Sys.CPU_THREADS ÷ 2, Threads.nthreads()),
    use_distributed::Bool=false,
    check_gradient=false,
)::MapSynthesisResult
    (; vdata, sketch, comp_env, enum_result, param_vars) = senum
    (; state_vars, state′′_vars) = vdata
    _check_variable_types(vdata, shape_env)

    all_comps = collect(Iterators.product((enum_result[v.type] for v in sketch.holes)...))
    @info "number of programs: $(length(all_comps))"

    prior_dists = to_distribution(vdata)
    compile_ctx = (
        hole_names = Val(Tuple(v.name for v in sketch.holes)),
        output_names = Val(tuple((v.name for v in state′′_vars)...)),
        shape_env, comp_env, sketch.combine, check_gradient,
    )
    ctx = (; compile_ctx, 
        evals_per_program, trials_per_eval, use_bijectors, optim_options, 
        prior_dists, times, actions, program_logp, data_likelihood,
    )
    
    progress = Progress(length(all_comps), desc="evaluating", 
        showspeed=true, output=stdout)

    eval_results = parallel_map(evaluate_prog, all_comps, ctx; 
        progress, n_threads, use_distributed)
    err_progs = eval_results |> Filter(r -> r.status == :errored) |> collect
    succeeded = filter(r -> r.status == :success, eval_results)
    isempty(succeeded) && error("Synthesis failed to produce any valid solution.\n" * 
        "First 5 errored programs: $(Iterators.take(err_progs, 5) |> collect)")
    sorted_results = map(r -> r.result, succeeded) |> sort_by(r -> -r.logp)
    best_result = sorted_results[1]

    all_times = succeeded |> Map(r -> r.optimize_times) |> collect
    first_opt_time, mean_opt_time = 
        (f.(all_times) |> to_measurement for f in (first, mean))
    opt_iters = succeeded |> Map(r -> mean(r.optimize_iters)) |> to_measurement
    opt_converge_rate = succeeded |> Map(r -> mean(r.optimize_converges)) |> to_measurement
    opt_success_rate = (succeeded |> Map(r -> length(r.optimize_times)/evals_per_program) 
        |> to_measurement)

    stats = (; n_progs=length(all_comps), n_err_progs=length(err_progs),
        first_opt_time, mean_opt_time, opt_iters, opt_success_rate, opt_converge_rate)
    MapSynthesisResult(best_result, stats, err_progs, sorted_results)
end


function evaluate_prog(ctx, comps)
    compile_ctx = ctx.compile_ctx
    (; hole_names, output_names) = compile_ctx
    f_x′′ = compile_sketch(comps, hole_names, output_names, compile_ctx)
    evaluate_prog(ctx, comps, f_x′′)
end

function evaluate_prog(ctx, comps, f_x′′)
    (; evals_per_program, trials_per_eval, use_bijectors, optim_options) = ctx
    (; prior_dists, times, actions, data_likelihood, program_logp) = ctx

    local solutions = []
    local optimize_times = Float64[]
    local optimize_iters = Int[]
    local optimize_converges = Bool[]
    local errors = []
    for _ in 1:evals_per_program
        for _ in 1:trials_per_eval
            try
                local (;value, time) = @ltimed map_trajectory(
                    f_x′′, prior_dists, 
                    times, actions, data_likelihood, 
                    optim_options, use_bijectors,
                )
                if isnan(value.logp)
                    push!(errors, ErrorException("logp = NaN."))
                else
                    push!(solutions, value)
                    push!(optimize_times, time)
                    push!(optimize_iters, value.iters)
                    push!(optimize_converges, value.converged)
                    break # successfully evaluated
                end
            catch err
                if err isa Union{OverflowError, DomainError, BadLossError}
                    push!(errors, err)
                else
                    @error "Unexpected error encountered when evaluating f_x′′ = $(f_x′′)" 
                    rethrow()
                end
            end
        end
    end

    if isempty(solutions)
        return (status= :errored, program=comps, errors=unique(errors))
    end
    sol = solutions |> max_by(s->s.logp)
    MAP_est = (
        sol.params,
        sol.others,
        states=sol.traj,
        logp=sol.logp,
    )
    logp = sol.logp + program_logp(comps)
    # only return the AST to prevent passing generated functions in a distributed setting.
    (; status= :success,
        result=(; logp, f_x′′, MAP_est),
        optimize_times, optimize_iters, optimize_converges)
end

struct BadLossError{I} <: Exception
    msg::AbstractString
    info::I
end

export map_trajectory
function map_trajectory(
    f_x′′::Function,
    prior_dist::DistrIterator{<: NamedTuple{(:x₀, :x′₀, :params, :others)}},
    times::AbstractVector,
    actions::TimeSeries{<:NamedTuple},
    data_likelihood,
    optim_options::Optim.Options,
    use_bijectors=true, # if false, will use box constrained optimization
)
    actions = specific_elems(actions) # important for type stability

    guess = rand(prior_dist)
    if use_bijectors
        bj = bijector(prior_dist)
        bj_inv = inv(bj)
        guess_original = guess
        guess = bj(guess_original)
        @assert structure_to_vec(bj_inv(guess)) ≈ structure_to_vec(guess_original)
    else
        lower, upper = _compute_bounds(prior_dist)
    end
    guess_vec::Vector{Float64} = structure_to_vec(guess)
    @assert structure_from_vec(guess, guess_vec) == guess
    
    vec_to_traj = (vec) -> let
        values = structure_from_vec(guess, vec)
        use_bijectors && (values = bj_inv(values))
        local (; x₀, x′₀, params, others) = values
        local p = merge(params, others)
        simulate(x₀, x′₀, f_x′′, p, times, actions), values
    end
    loss = (vec) -> let
        is_bad_dual(vec) && error("Bad dual in loss input vec: $vec")
        local traj, values = vec_to_traj(vec)
        prior = score(prior_dist, values)
        dl = data_likelihood(traj, values.others)
        v = -(prior + dl)
        if isnan(v) || is_bad_dual(v)
            info = (; vec, values.params, values.others, traj, prior, likelihood=dl, loss=v)
            throw(BadLossError("bad loss detected: loss = $v", info))
        end
        v
    end

    if use_bijectors
        sol = optimize_no_tag(loss, guess_vec, optim_options)
    else
        sol = optimize_bounded(loss, guess_vec, (lower, upper), optim_options)
    end
    iters = Optim.iterations(sol)
    let
        traj, values = vec_to_traj(Optim.minimizer(sol))
        logp = -Optim.minimum(sol)
        converged = Optim.converged(sol) && iters > 1
        (; traj, values.params, values.others, logp, iters, converged)
    end
end

"""
Numerically integrate the trajecotry using 
[Leapfrog integration](https://en.wikipedia.org/wiki/Leapfrog_integration).

Returns a vector of named tuples containing the next state for each time step.

# Arguments
- `x::NamedTuple`: the initial pose ``(s_1=v_1, s_2=v_2,...,s_n=v_n)``.
- `x′::NamedTuple`: the initial velocity ``(s′_1=v_1, s′_2=v_2,...,s′_n=v_n)``.
- `f_x′′::Tuple{Vararg{Function}}`: the acceleration tuple ``(s′′_1, s′′_2,...,s′′_n)``.
- `params::NamedTuple`: the dynamics parameters.
- `times::AbstractVector`: the time steps.
- `actions::TimeSeries{<:NamedTuple}`: The actions for each time step.
"""
function simulate(
    x₀::NamedTuple{x_keys, X},
    x′₀::NamedTuple{x′_keys, X},
    f_x′′::Function,
    params::NamedTuple,
    times::AbstractVector,
    actions::TimeSeries{<:NamedTuple},
) where {x_keys, x′_keys, X}
    i_ref = Ref(1)
    next_time_action!() = begin
        i = i_ref[]::Int
        i_ref[] += 1
        times[i], actions[i]
    end
    should_stop() = i_ref[] > length(times)

    result = NamedTuple[]
    record_state!(s) = begin
        push!(result, s)
    end

    simulate(x₀, x′₀, f_x′′, params, should_stop, next_time_action!, record_state!)
    return result
end

"""
# Arguments
- `should_stop() -> bool`: whether to continue the simulation to the next time step.
- `next_time_action!() -> (t, act)`: return the time and action of the next time step.
- `record_state!(::NamedTuple)`: callback to handle the current state.
"""
function simulate(
    x₀::NamedTuple{x_keys, X},
    x′₀::NamedTuple{x′_keys, X},
    f_x′′::Function,
    params::NamedTuple,
    should_stop::Function,
    next_time_action!::Function,
    record_state!::Function,
)::Nothing where {x_keys, x′_keys, X}
    common_keys = intersect(x_keys, x′_keys)
    @assert isempty(common_keys) "overlapping keys: $common_keys"
    x′′_keys = derivative.(x′_keys)

    acc(x, x′, action) = begin
        input = merge(NamedTuple{x_keys}(x), NamedTuple{x′_keys}(x′), action, params)
        call_T(f_x′′, input, NamedTuple{x′′_keys, X}) |> values
    end
    to_named(x, x′) = merge(NamedTuple{x_keys}(x), NamedTuple{x′_keys}(x′))

    x = values(x₀)
    x′ = values(x′₀)
    should_stop() && return
    t_act = next_time_action!()::Tuple{Float64, NamedTuple}
    a = acc(x, x′, t_act[2])
    t = t_act[1]
    record_state!(to_named(x, x′))

    while !should_stop()
        (t1, act) = next_time_action!()
        Δt = t1 - t
        t = t1
        x, x′, a = leap_frog_step((x, x′, a), (x, x′) -> acc(x, x′, act), Δt)
        record_state!(to_named(x, x′))
    end
end

function leap_frog_step((x, v, a)::Tuple{X,X,X}, a_f, Δt) where X
    v_half = @. v + (Δt/2) * a 
    x1 = @.(x + Δt * v_half)::X
    a1 = a_f(x1, @.(v_half + (Δt/2) * a))::X
    v1 = @.(v + (Δt/2) * (a + a1))::X
    (x1, v1, a1)
end