using ProgressMeter: Progress, next!, @showprogress, progress_pmap
using Distributed

const TimeSeries{T} = Vector{T}

export DynamicsSketch, no_sketch
"""
A sketch for second-order system dynamics.
"""
struct DynamicsSketch{F}
    holes::Vector{Var}
    "combine(inputs::NamedTuple, hole_values::Tuple) -> accelerations::Tuple"
    combine::F
end

function no_sketch(state′′_vars::Vector{Var})
    DynamicsSketch(state′′_vars, (inputs, hole_values) -> hole_values)
end

export VariableData, map_synthesis
"""
The robot dynamics are assumed to be of the form `f(state, action, params) -> next_state`.

## Fields
- `states`: maps each state variable (e.g. `position`) to the distribution of its initial 
value and initial derivative (e.g. the distribution of `position` and `velocity`). 
- `actions`: maps each action variable to its time-series data.
- `dynamics_params`: maps each dynamics parameters to its prior distribution.
- `others`: maps other random variables to their prior distributions. Unlike 
`dynamics_params`, these variables cannot affect the state of the robot but 
can still affect the observations. e.g., these can be (unknown) landmark locations or the 
position of the camera.
- `var_types`: maps each variable to their `PType`.
"""
@kwdef(
struct VariableData
    states::OrderedDict{Var, Tuple{GDistr, GDistr}}  
    dynamics_params::OrderedDict{Var, GDistr}
    # Should map each symbol to a value that supports the trait `Is{GDistr}`
    others::OrderedDict{Symbol, GDistr}  
    t_unit::PUnit=PUnits.Time
    state_vars::Vector{Var} = keys(states) |> collect
    state′_vars::Vector{Var} = derivative.(state_vars, Ref(t_unit))
    state′′_vars::Vector{Var} = derivative.(state′_vars, Ref(t_unit))
end)

to_distribution(vdata::VariableData) = let
    (; state_vars) = vdata
    all_dists = (
        x₀ = (;(s.name => vdata.states[s][1] for s in state_vars)...),
        x′₀ = (;(derivative(s.name) => vdata.states[s][2] for s in state_vars)...),
        params = (;(v.name => dist for (v, dist) in vdata.dynamics_params)...),
        others = (;(name => dist for (name, dist) in vdata.others)...),
    )
    DistrIterator(map(DistrIterator, all_dists))
end

params_distribution(vdata::VariableData) = begin
    DistrIterator((;(v.name => dist for (v, dist) in vdata.dynamics_params)...))
end

_check_variable_types(vdata::VariableData, shape_env::ShapeEnv) = begin
    check_type(var, value) = begin
        t = shape_env[var.type]
        @assert value isa t "Prior distribution for \
            $var produced value $value, expected type: $t"
    end

    (; states, dynamics_params) = vdata
    foreach(states) do (v, (d1, d2))
        check_type(v, rand(d1))
        check_type(v, rand(d2))
    end
    foreach(dynamics_params) do (v, d)
        check_type(v, rand(d))
    end
end

export SynthesisEnumerationResult, synthesis_enumeration

"""
Group the enumeration data needed for `map_synthesis`. Can be created by `bottom_up_enum`.
"""
@kwdef(
struct SynthesisEnumerationResult{Combine}
    vdata::VariableData
    sketch::DynamicsSketch{Combine}
    comp_env::ComponentEnv
    action_vars::Vector{Var}
    param_vars::Vector{Var}
    enum_result::EnumerationResult
end)

Base.show(io::IO, mime::MIME"text/plain", r::SynthesisEnumerationResult) = begin
    (; comp_env, action_vars, param_vars, enum_result, sketch) = r
    (; state_vars, state′′_vars) = r.vdata
    holes = sketch.holes

    n_interest = prod(count_len(enum_result[v.type]) for v in holes)
    search_details = join((count_len(enum_result[v.type]) for v in holes), " * ")

    println(io, "===== Synthesis enumeration result =====")
    println(io, "search_space: $(pretty_number(n_interest)) = $search_details")
    println(io, "n_components: ", length(comp_env.signatures))
    println(io, "holes: $holes")
    println(io, "states: $state_vars")
    println(io, "actions: $action_vars")
    println(io, "params: $param_vars")

    show(io, mime, enum_result)
end

function synthesis_enumeration(
    vdata::VariableData, sketch::DynamicsSketch, action_vars::Vector{Var},
    comp_env::ComponentEnv, max_size; 
    pruner=NoPruner(), type_pruning=true,
)
    (; state_vars, state′_vars, state′′_vars) = vdata
    param_vars = keys(vdata.dynamics_params) |> collect
    dyn_vars = [state_vars; state′_vars; action_vars; param_vars]
    if type_pruning
        types_needed, _ = enumerate_types(
            comp_env, 
            Set(v.type for v in dyn_vars), 
            Set(v.type for v in sketch.holes), 
            max_size,
        )
    else 
        types_needed = nothing
    end
    dyn_varset = Set(v for v in dyn_vars)
    enum_result = enumerate_terms(comp_env, dyn_varset, max_size; types_needed, pruner)
    SynthesisEnumerationResult(;
        vdata,
        sketch,
        comp_env,
        action_vars,
        param_vars,
        enum_result,
    )
end

export MapSynthesisResult, map_synthesis
struct MapSynthesisResult{R}
    best_result::R
    stats::NamedTuple
    errored_programs::Vector
    sorted_results::Vector
end

get_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rows = map(Iterators.take(r.sorted_results, top_k)) do (; logp, f_x′′, MAP_est)
        (; params, others) = MAP_est
        (; logp, f_x′′=(x -> x.ast).(f_x′′) , params, others)
    end
    rows
end

show_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rs = get_top_results(r, top_k)
    println("Top $top_k solutions:")
    for (i, r) in enumerate(rs)
        println("---- solution $i ----")
        println(r)
    end
end

Base.show(io::IO, r::MapSynthesisResult) =
    print(io, "MapSynthesisResult(best=$(r.best_result))")

Base.show(io::IO, ::MIME"text/plain", r::MapSynthesisResult) = begin
    io = IOIndents.IOIndent(io)
    println(io, "==== MAP synthesis result ====")
    println(io, "Stats:", Indent())
    for (k, v) in pairs(r.stats)
        println(io, "$k: ", pretty_number(v))
    end
    print(io, Dedent())
    println(io, "Best estimation found:", Indent())
    for (k, v) in pairs(r.best_result)
        k ∈ (:logp, :f_x′′) && println(io, "$k: $v")
    end
end

Base.size(v::Transducers.ProgressLoggingFoldable) = size(v.foldable)

import LinearAlgebra.BLAS

export prog_size_prior
function prog_size_prior(decay::Float64)
    (comps) -> log(decay) * sum(ast_size.(comps); init=0) 
end

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
    
    to_eval = all_comps
    progress = Progress(length(all_comps), desc="evaluating", 
        showspeed=true, output=stdout)

    if use_distributed
        length(workers()) > 1 || error("distributed enabled with less than 2 workers.")
        @everywhere SEDL._eval_ctx[] = $ctx
        eval_results = progress_pmap(evaluate_prog_distributed, to_eval; progress)
    else
        eval_task(e) = let
            evaluate_prog(ctx, e)
            next!(progress)
        end
        if n_threads <= 1
            eval_results = map(eval_task, to_eval)
        else
            pool = ThreadPools.QueuePool(2, n_threads)
            eval_results = ThreadPools.tmap(eval_task, pool, to_eval)
            close(pool)
        end
    end
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

const _eval_ctx = Ref{Any}(nothing)
function evaluate_prog_distributed(comps)
    ctx = _eval_ctx[]
    evaluate_prog(ctx, comps)
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


"Combine sketch hole expressions into an executable Julia function."
compile_sketch(
    comps::Tuple{Vararg{TAST}}, 
    ::Val{hole_names},
    ::Val{output_names},
    (;shape_env, comp_env, combine, check_gradient),
) where {hole_names, output_names} = begin
    # TODO: Check the inferred return types 
    funcs = map(comp -> compile(comp, shape_env, comp_env; check_gradient), comps)
    ast = (; (=>).(hole_names, comps)...)
    julia = Expr(:tuple, (x -> x.julia).(funcs))
    f = input -> let
        hole_values = NamedTuple{hole_names}(map(f -> f(input), funcs))
        NamedTuple{output_names}(combine(input, hole_values))
    end
    CompiledFunc(ast, julia, f)
end

const DebugMode = Ref(false)

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
        use_bijectors && (guess = bj(guess_original))
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


function Distributions.logpdf(dist::NamedTuple{ks}, v::NamedTuple{ks})::Real where ks
    sum(logpdf.(values(dist), values(v)))
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

function structure_to_vec!(arr, v::Union{AbstractVector, Tuple, NamedTuple})
    i = Ref(0)
    rec(r) = let
        if r isa Real
            arr[i[]+=1] = r
        elseif r isa Union{AbstractVector, Tuple, NamedTuple}
            foreach(rec, r)
        end
        nothing
    end
    rec(v)
    @assert i[] == length(arr)
    arr
end

function _compute_bounds(prior_dist)
    lower, upper = Float64[], Float64[]
    rec(d) = let
        if d isa UnivariateDistribution
            (l, u) = Distributions.extrema(d)
            push!(lower, l)
            push!(upper, u)
        elseif d isa DistrIterator
            foreach(rec, d.distributions)
        elseif d isa SMvUniform
            foreach(rec, d.uniforms)
        else
            error("Don't know how to compute bounds for $d")
        end
        nothing
    end
    rec(prior_dist)
    lower, upper
end

"""
Count how many numbers there are in the given NamedTuple.

```jldoctest
julia> n_numbers((0.0, @SVector[0.0, 0.0]))
3
```
"""
n_numbers(v::Union{Tuple, NamedTuple}) = sum(n_numbers, v)
n_numbers(::Real) = 1
n_numbers(v::AbstractVector{<:Real}) = length(v)
n_numbers(v::AbstractVector) = sum(n_numbers, v)

function promote_numbers_type(v::Union{Tuple, NamedTuple})
    rec(x::Real) = typeof(x)
    rec(::AbstractVector{T}) where {T <: Real} = T
    rec(v::AbstractVector) = rec(v[1])
    rec(v::Union{Tuple, NamedTuple}) = Base.promote_eltype(rec.(values(v))...)
    Base.promote_eltype(rec.(values(v))...)
end

function structure_to_vec(v::Union{Tuple, NamedTuple})
    T = promote_numbers_type(v)
    vec = Vector{T}(undef, n_numbers(v))
    structure_to_vec!(vec, v)
    vec
end

function structure_from_vec(template::Tuple, vec)::Tuple
    i::Ref{Int} = Ref(0)

    map(template) do x
        _read_structure(x, i, vec)
    end
end

_read_structure(x, i::Ref{Int}, vec) = let
    if x isa Real
        vec[i[]+=1]
    elseif x isa Union{AbstractVector, Tuple, NamedTuple}
        map(x) do x′
            _read_structure(x′, i, vec)
        end
    else
        error("don't know how to handle the template: $x")
    end
end

function structure_from_vec(template::NamedTuple{S}, vec)::NamedTuple{S} where S
    NamedTuple{keys(template)}(structure_from_vec(values(template), vec))
end
