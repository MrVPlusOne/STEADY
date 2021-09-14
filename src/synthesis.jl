const TimeSeries{T} = Vector{T}

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
    states::Dict{Var, Tuple{GDistr, GDistr}}  
    dynamics_params::Dict{Var, GDistr}
    # Should map each symbol to a value that supports the trait `Is{GDistr}`
    others::Dict{Symbol, GDistr}  
    t_unit::PUnit=PUnits.Time
end)


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

Base.rand(vdata::VariableData) = begin
    x₀ = (;(v.name => rand(dist) for (v, (dist, _)) in vdata.states)...)
    x′₀ = (;(derivative(v.name) => rand(dist) for (v, (_, dist)) in vdata.states)...)
    params = (;(v.name => rand(dist) for (v, dist) in vdata.dynamics_params)...)
    others = (;vdata.others...)
    (; x₀, x′₀, params, others)
end

"""
Group the enumeration data needed for `map_synthesis`. Can be created by `bottom_up_enum`.
"""
@kwdef(
struct SynthesisEnumerationResult
    vdata::VariableData
    comp_env::ComponentEnv
    state_vars::Vector{Var}
    state′_vars::Vector{Var}
    state′′_vars::Vector{Var}
    action_vars::Vector{Var}
    param_vars::Vector{Var}
    enum_result::EnumerationResult
end)

Base.show(io::IO, mime::MIME"text/plain", r::SynthesisEnumerationResult) = begin
    (; comp_env, state_vars, state′′_vars, action_vars, param_vars, enum_result) = r

    comp_types = Set(v.type for v in state′′_vars)
    n_interest = prod(count_len(enum_result[ty]) for ty in comp_types)

    println(io, "===== Synthesis enumeration result =====")
    println(io, "search_space: $(pretty_number(n_interest))")
    println(io, "n_components: ", length(comp_env.signatures))
    println(io, "states: $state_vars")
    println(io, "actions: $action_vars")
    println(io, "params: $param_vars")
    show(io, mime, enum_result)
end

function synthesis_enumeration(
    vdata::VariableData, action_vars::Vector{Var}, comp_env::ComponentEnv, max_size; 
    pruner=NoPruner(), type_pruning=true,
)
    (; t_unit) = vdata
    state_vars = keys(vdata.states) |> collect
    state′_vars = derivative.(state_vars, Ref(t_unit))
    state′′_vars = derivative.(state′_vars, Ref(t_unit))
    param_vars = keys(vdata.dynamics_params) |> collect
    dyn_vars = [state_vars; state′_vars; action_vars; param_vars]
    output_types = Set(v.type for v in state′′_vars)
    if type_pruning
        types_needed, _ = enumerate_types(
            comp_env, 
            Set(v.type for v in dyn_vars), 
            output_types, 
            max_size,
        )
    else 
        types_needed = nothing
    end
    dyn_varset = Set(v for v in dyn_vars)
    enum_result = enumerate_terms(comp_env, dyn_varset, max_size; types_needed, pruner)
    SynthesisEnumerationResult(;
        vdata,
        comp_env,
        state_vars,
        state′_vars,
        state′′_vars,
        action_vars,
        param_vars,
        enum_result,
    )
end

function synthesis_enumeration_staged(
    vdata::VariableData, action_vars::Vector{Var},
    comp_env::ComponentEnv, max_size,
)
    (; t_unit) = vdata
    state_vars = keys(vdata.states) |> collect
    state′_vars = derivative.(state_vars, Ref(t_unit))
    state′′_vars = derivative.(state′_vars, Ref(t_unit))
    param_vars = keys(vdata.dynamics_params) |> collect
    dyn_vars = [state_vars; state′_vars; action_vars; param_vars]
    output_types = Set(v.type for v in state′′_vars)
    enumerate_types(
        comp_env, 
        Set(v.type for v in dyn_vars), 
        output_types, 
        max_size,
    )
end

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

"""
Perform Maximum a posteriori (MAP) synthesis to find the joint assignment of the motion 
model *and* the trajecotry that maximizes the posterior probability.

The system dynamics are assuemd to be 2nd-order.

- `program_logp(prog::TAST) -> logp` should return the log piror probability of a given 
dynamics program. 
- `data_likelihood(trajectory::Dict{Var, TimeSeries}, other_vars::Dict{Var, Any}) -> logp` 
should return the log probability density of the observation.
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
    optim_options = Optim.Options(),
    n_threads = min(Sys.CPU_THREADS ÷ 2, Threads.nthreads()),
)::MapSynthesisResult
    (; vdata, comp_env, enum_result, state_vars, state′′_vars, param_vars) = senum
    _check_variable_types(vdata, shape_env)
    
    output_types = [v.type for v in state′′_vars]
    all_comps = collect(Iterators.product((enum_result[ty] for ty in output_types)...))

    @info "number of programs: $(length(all_comps))"
    prog_compile_time = @elapsed begin
        # cache = Dict{TAST, CompiledFunc}()
        compiled = map(all_comps) do comps
            # TODO: update this to a single compiled function that returns a tuple 
            funcs = map(comp -> compile(comp, shape_env, comp_env), comps)
            ast = (; (=>).((v.name for v in state′′_vars), comps)...)
            julia = Expr(:tuple, (x -> x.julia).(funcs))
            f = input -> map(f -> f(input), funcs)
            CompiledFunc(ast, julia, f)
        end
    end

    x₀_dist = (;(s.name => vdata.states[s][1] for s in state_vars)...)
    x′₀_dist = (;(derivative(s.name) => vdata.states[s][2] for s in state_vars)...)
    params_dist = (;(p.name => vdata.dynamics_params[p] for p in param_vars)...)
    others_dist = (;(name => dist for (name, dist) in vdata.others)...)
    prior_dists = merge(params_dist, others_dist)

    function evaluate((comps, f_x′′))
        local solutions = []
        local optimize_times = Float64[]
        local errors = []
        for _ in 1:evals_per_program
            try
                timed_r = @ltimed map_trajectory(
                    x₀_dist, x′₀_dist, f_x′′, prior_dists, 
                    times, actions, data_likelihood, 
                    optim_options,
                )
                if isnan(timed_r.value.logp)
                    push!(errors, ErrorException("logp = NaN."))
                else
                    push!(solutions, timed_r.value)
                    push!(optimize_times, timed_r.time)
                end
            catch err
                if err isa ArgumentError || err isa DomainError
                    push!(errors, err)
                else
                    @error "f_x′′ = $(f_x′′)" 
                    rethrow()
                end
            end
        end

        if isempty(solutions)
            return (status= :errored, program=comps, errors=unique(errors))
        end
        sol = solutions |> max_by(s->s.logp)
        MAP_est = (
            params=subtuple(sol.params, keys(params_dist)),
            others=subtuple(sol.params, keys(others_dist)),
            states=sol.traj,
            logp=sol.logp,
        )
        logp = sol.logp + program_logp(comps)
        (; status= :success,
            result=(; logp, f_x′′, MAP_est),
            optimize_times)
    end

    # blas_threads = BLAS.get_num_threads()
    # BLAS.set_num_threads(1)
    to_eval = withprogress(zip(all_comps, compiled); interval=0.1)
    eval_results = if n_threads > 1
        ThreadsX.mapi(evaluate, to_eval)
    else
        to_eval |> Map(evaluate) |> collect
    end
    # BLAS.set_num_threads(blas_threads)

    err_progs = eval_results |> Filter(r -> r.status == :errored) |> collect
    succeeded = filter(r -> r.status == :success, eval_results)
    isempty(succeeded) && error("Synthesis failed to produce any valid solution.\n" * 
        "First 5 errored programs: $(Iterators.take(err_progs, 5) |> collect)")
    sorted_results = map(r -> r.result, succeeded) |> sort_by(r -> -r.logp)
    best_result = sorted_results[1]

    all_times = succeeded |> Map(r -> r.optimize_times) |> collect
    first_opt_time, mean_opt_time = 
        (f.(all_times) |> to_measurement for f in (first, mean))

    stats = (; n_progs=length(all_comps), n_err_progs=length(err_progs),
        prog_compile_time, first_opt_time, mean_opt_time)
    MapSynthesisResult(best_result, stats, err_progs, sorted_results)
end

function transpose_series(
    len::Int, vars, series_comps::Dict{Var, TimeSeries}
)::TimeSeries{<:NamedTuple}
    collect(let 
        as = (a.name => series_comps[a][t] for a in vars)
        (; as...)
    end for t in 1:len)
end

export map_trajectory
function map_trajectory(
    x₀_dist::NamedTuple,
    x′₀_dist::NamedTuple,
    f_x′′::Function,
    params_dist::NamedTuple,
    times::AbstractVector,
    actions::TimeSeries{<:NamedTuple},
    data_likelihood,
    optim_options::Optim.Options,
)
    x_bj, x′_bj, p_bj = (ds -> map(bijector, values(ds))).((x₀_dist, x′₀_dist, params_dist))
    x_inv, x′_inv, p_inv = (bs -> map(inv, bs)).((x_bj, x′_bj, p_bj))
    
    x_guess = zipmap(x_bj, map(rand, x₀_dist))
    x′_guess = zipmap(x′_bj, map(rand, x′₀_dist))
    params_guess = zipmap(p_bj, map(rand, params_dist))
    
    x_size = n_numbers(x_guess)
    function vec_to_traj(vec) 
        local x₀ = zipmap(x_inv, structure_from_vec(x_guess, vec))
        local x′₀ = zipmap(x′_inv, structure_from_vec(x′_guess, @views vec[x_size+1:2x_size]))
        local p = zipmap(p_inv, structure_from_vec(params_guess, @views vec[2x_size+1:end]))
        simulate(x₀, x′₀, f_x′′, p, times, actions), (; x₀, x′₀, p)
    end
    function loss(vec)
        traj, (x₀, x′₀, p) = vec_to_traj(vec)
        prior = logpdf(x₀_dist, x₀) + logpdf(x′₀_dist, x′₀) + logpdf(params_dist, p)
        -(prior + data_likelihood(traj, p))
    end
    vec_guess::Vector{Float64} = vcat(
        structure_to_vec(x_guess), structure_to_vec(x′_guess), structure_to_vec(params_guess))
    # sol = Optim.optimize(loss, vec_guess, LBFGS(), optim_options; autodiff = :forward)
    sol = optimize_no_tag(loss, vec_guess, optim_options)
    if !Optim.converged(sol)
        @warn "Optim not converged." f_x′′ sol
    end
    traj, (x₀, x′₀, params) = vec_to_traj(Optim.minimizer(sol))
    logp = -Optim.minimum(sol)
    (;params, traj, x₀, x′₀, logp)
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
        i = i_ref[]
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

    acc(x, x′, action) = begin
        input = merge(NamedTuple{x_keys}(x), NamedTuple{x′_keys}(x′), action, params)
        call_T(f_x′′, input, X)
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

function structure_to_vec!(arr, v::Union{Tuple, NamedTuple})
    i = Ref(0)
    function rec(r::Real)
        arr[i[]+=1] = r
        nothing
    end
    function rec(v::AbstractVector{<:Real})
        j = i[]+=1
        arr[j:j+length(v)-1] .= v
        nothing
    end
    function rec(v::AbstractVector)
        foreach(rec, v)
    end
    foreach(rec, v)
    arr
end

"""
Count how many numbers there are in the given NamedTuple.

```jldoctest
julia> n_numbers((0.0, @SVector[0.0, 0.0]))
3
```
"""
function n_numbers(v::Union{Tuple, NamedTuple})
    count(::Real) = 1
    count(v::AbstractVector{<:Real}) = length(v)
    count(v::AbstractVector) = sum(count, v)
    sum(map(count, v))
end

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
    i::Int = 0
    read(::Real) = begin
        vec[i+=1]
    end
    read(::SVector{n, <:Real}) where n = begin
        SVector{n}(read(0.0) for _ in Base.OneTo(n))
    end
    read(v::AbstractVector) = map(read, v)

    read.(template)
end

function structure_from_vec(template::NamedTuple{S}, vec)::NamedTuple{S} where S
    NamedTuple{keys(template)}(structure_from_vec(values(template), vec))
end
