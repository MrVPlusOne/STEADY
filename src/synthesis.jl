"""
Use the syntax `result[pshape]` to get an iterator over all found programs
of the given [`PShape`](@ref).
"""
struct EnumerationResult
    programs::Dict{PShape, Dict{Int, Dict{PUnit, Vector{TAST}}}}
    stats::Dict{String, Any}
end

Base.getindex(r::EnumerationResult, shape::PShape) = let
    d1 = r.programs[shape]
    (p 
    for size in sort(collect(keys(d1)))
    for d2 in values(d1[size]) 
    for p in d2)
end


Base.getindex(r::EnumerationResult, type::PType) = begin
    @unpack shape, unit = type
    d1 = get(r.programs, shape, Dict())
    (p 
    for size in sort(collect(keys(d1)))
    for p in get(d1[size], unit, []))
end

function show_programs(io::IO, r::EnumerationResult; max_programs::Int=20)
    for s in keys(r.programs)
        println(io, "------- $(s.name) -------")
        i = 1
        for p in r[s]
            if i > max_programs
                println(io, "...")
                break
            end
            println(io, p, "::", p.type)
            i+=1
        end
    end
end

Base.show(io::IO, ::MIME"text/plain", r::EnumerationResult) = begin
    io = IOIndents.IOIndent(io)
    println(io, "==== Enumeration result ====")
    println(io, "Stats:", Indent())
    for (k, v) in r.stats
        println(io, "$k: $v")
    end
    print(io, Dedent())
    println(io, "Found programs: ", Indent())
    show_programs(io, r)
end

Base.show(io::IO, r::EnumerationResult) = begin
    stats = join(["$k: $v" for (k, v) in r.stats], ", ")
    print(io, "EnumerationResult($stats)")
end

function bottom_up_enum(env::ComponentEnv, vars::Vector{Var}, max_size)::EnumerationResult
    @assert allunique(vars) "Duplicate variables in the input."

    "stores all found programs, indexed by: shape -> ast_size -> unit"
    found = Dict{PShape, Dict{Int, Dict{PUnit, Vector{TAST}}}}()
    n_found = 0

    function get_progs!(shape, size, unit)
        s1 = get!(found, shape) do ; Dict{Int, Dict{PUnit, Vector{TAST}}}() end
        s2 = get!(s1, size) do ; Dict{PUnit, Vector{TAST}}() end
        get!(s2, unit) do ; TAST[] end
    end

    function insert_prog!(size, prog::TAST)
        @unpack shape, unit = prog.type
        ps = get_progs!(shape, size, unit)
        push!(ps, prog)
        n_found += 1
    end

    start_time = time()
    # size 1 programs consist of all variables
    foreach(vars) do v
        insert_prog!(1, v)
    end

    signatures = env.signatures
    # construct larger programs from smaller ones
    for size in 2:max_size, (f, sig) in signatures
        arg_shapes = sig.arg_shapes
        sizes_for_arg = [keys(found[s]) for s in arg_shapes]
        # iterate over all possible size combinations s.t. the sum = size-1
        for arg_sizes in size_combinations(length(arg_shapes), sizes_for_arg, size-1)
            arg_dicts = [found[arg_shapes[i]][s] for (i, s) in enumerate(arg_sizes)]
            # iterate over all unit combinations
            for arg_units in Iterators.product((keys(d) for d in arg_dicts)...)
                runit = sig.result_unit(arg_units...)
                (runit === nothing) && continue # skip invalid combination
                arg_candidates = (d[u] for (u, d) in zip(arg_units, arg_dicts))
                # iterate over all argument AST combinations
                for args in Iterators.product(arg_candidates...)
                    prog = Call(f, collect(args), PType(sig.result_shape, runit::PUnit))
                    insert_prog!(size, prog)
                end
            end
        end
    end

    stats = Dict("time_taken" => time() - start_time, "n_programs" => n_found)
    EnumerationResult(found, stats)
end

"""
Iterate over all possible size combinations such that they have the specified total size.

## Example
```jldoctest
julia> size_combinations(3, [[1,2],[2,3],[1,2,3,4]], 6) |> collect
4-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [1, 3, 2]
 [2, 2, 2]
 [2, 3, 1]

julia> size_combinations(3, [1:1, 1:2, 1:4], 6) |> collect
2-element Vector{Vector{Int64}}:
[1, 1, 4]
[1, 2, 3]
```
"""
function size_combinations(n_args, sizes_for_arg, total_size)
    function rec(i, size_left)
        if i == n_args
            (size_left in sizes_for_arg[i]) ? [[size_left]] : Vector{Int}[]
        else
            # the maximal possible size for argument i
            si_max = size_left-(n_args-i)
            si_sizes = filter(s -> s <= si_max, sizes_for_arg[i])
            Iterators.flatten(
                (push!(sizes, s1) for sizes in rec(i + 1, size_left - s1))
                for s1 in si_sizes
            )
        end
    end
    (reverse!(v) for v in rec(1, total_size))
end

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
Base.@kwdef(
struct VariableData
    states::Dict{Var, Tuple{Distribution, Distribution}}  
    dynamics_params::Dict{Var, Distribution}
    others::Dict{Var, Distribution}
    t_unit:: PUnit = PUnits.Time
end)

Base.rand(vdata::VariableData) = begin
    x₀ = (;(v.name => rand(dist) for (v, (dist, _)) in vdata.states)...)
    x′₀ = (;(derivative(v.name) => rand(dist) for (v, (_, dist)) in vdata.states)...)
    params = (;(v.name => rand(dist) for (v, dist) in vdata.dynamics_params)...)
    others = (;(v.name => rand(dist) for (v, dist) in vdata.others)...)
    (; x₀, x′₀, params, others)
end

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
    shape_env::ShapeEnv,
    comp_env::ComponentEnv,
    vdata::VariableData,
    action_vars::Vector{Var},
    actions::TimeSeries{<:NamedTuple},
    times::AbstractVector,
    program_logp::Function,
    data_likelihood::Function;
    max_size::Int,
    evals_per_program::Int = 10,
    optim_options = Optim.Options(),
)
    @unpack t_unit = vdata
    state_vars = keys(vdata.states) |> collect
    state′_vars = derivative.(state_vars, Ref(t_unit))
    state′′_vars = derivative.(state′_vars, Ref(t_unit))
    param_vars = keys(vdata.dynamics_params) |> collect
    dyn_vars = [state_vars; state′_vars; action_vars; param_vars]
    enum_result = bottom_up_enum(comp_env, dyn_vars, max_size)
    
    output_types = [v.type for v in state′′_vars]
    all_comps = Iterators.product((enum_result[ty] for ty in output_types)...) |> collect
    @info "number of programs: $(length(all_comps))"
    x₀_dist = (;(s.name => vdata.states[s][1] for s in state_vars)...)
    x′₀_dist = (;(derivative(s.name) => vdata.states[s][2] for s in state_vars)...)
    params_dist = (;(p.name => vdata.dynamics_params[p] for p in param_vars)...)
    others_dist = (;(p.name => dist for (p, dist) in vdata.others)...)
    best_prog = missing
    @progress for comps in all_comps
        f_x′′ = map(comp -> compile(comp, dyn_vars, shape_env, comp_env), comps)::Tuple
        sols = [map_trajectory(
            x₀_dist, x′₀_dist, f_x′′, merge(params_dist, others_dist), 
            times, actions, data_likelihood, 
            optim_options,
        ) for _ in 1:evals_per_program]
        _, s_id = findmax([s.logp for s in sols])
        sol = sols[s_id]
        logp = sol.logp + program_logp(comps)
        if best_prog === missing || best_prog.logp < logp
            best_prog = (; logp, f_x′′, sol)
        end
    end
    best_prog
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
    x₀_dist::NamedTuple{x_keys},
    x′₀_dist::NamedTuple{x′_keys},
    f_x′′::Tuple,
    params_dist::NamedTuple{p_keys},
    times::AbstractVector,
    actions::TimeSeries{<:NamedTuple},
    data_likelihood,
    optim_options::Optim.Options,
) where {x_keys, x′_keys, p_keys}
    x_guess = map(rand, x₀_dist)
    x′_guess = map(rand, x′₀_dist)
    params_guess = map(rand, params_dist) 
    x_size = n_numbers(x_guess)
    function vec_to_traj(vec) 
        local x₀ = NamedTuple{x_keys}(tuple_from_vec(x_guess, vec))
        local x′₀ = NamedTuple{x′_keys}(tuple_from_vec(x′_guess, @views vec[x_size+1:2x_size]))
        local p = NamedTuple{p_keys}(tuple_from_vec(params_guess, @views vec[2x_size+1:end]))
        simulate(x₀, x′₀, f_x′′, p, times, actions), (; x₀, x′₀, p)
    end
    function loss(vec)
        traj, (x₀, x′₀, p) = vec_to_traj(vec)
        prior = logpdf(x₀_dist, x₀) + logpdf(x′₀_dist, x′₀) + logpdf(params_dist, p)
        -(prior + data_likelihood(traj, p))
    end
    vec_guess::Vector{Float64} = vcat(tuple_to_vec(x_guess), tuple_to_vec(x′_guess), tuple_to_vec(params_guess))
    sol = Optim.optimize(loss, vec_guess, LBFGS(), optim_options; autodiff = :forward)
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
    f_x′′::Tuple{Vararg{Function}},
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
    f_x′′::Tuple{Vararg{Function}},
    params::NamedTuple,
    should_stop::Function,
    next_time_action!::Function,
    record_state!::Function,
)::Nothing where {x_keys, x′_keys, X}
    common_keys = intersect(x_keys, x′_keys)
    @assert isempty(common_keys) "overlapping keys: $common_keys"

    acc(x, x′, action) = begin
        input = merge(NamedTuple{x_keys}(x), NamedTuple{x′_keys}(x′), action, params)
        map(f -> f(input), f_x′′)
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

function leap_frog_step((x, v, a), a_f, Δt)
    v_half = @. v + (Δt/2) * a 
    x1 = @. x + Δt * v_half
    a1 = a_f(x1, @.(v_half + (Δt/2) * a))
    v1 = @. v + (Δt/2) * (a + a1)
    (x1, v1, a1)
end

## === below are unused functions and may be removed in the future ===
using OrdinaryDiffEq

function simulate_ode(
    x::NamedTuple{x_keys, X},
    x′::NamedTuple{x′_keys, X},
    f_x′′::NamedTuple,
    params::NamedTuple,
    times::TimeSeries,
    actions::TimeSeries{<:NamedTuple},
) where {x_keys, x′_keys, X}
    u_keys = (x_keys..., x′_keys...)
    n = sum(length(z) for z in values(x))
    p = (params=params, action=actions[1])
    u0_tuple = merge(x, x′)
    state_types = typeof(u0_tuple)
    u0 = tuple_to_vec(values(u0_tuple))
    function dynamics!(du, u, p, t)
        s = tuple_from_vec(NamedTuple{x_keys, X}, @views u[1:n])
        s′ = tuple_from_vec(NamedTuple{x′_keys, X}, @views u[n+1:2n])
        # input = (s, s′, p.action, p.params)
        du[1:n] .= u[n+1:end]
        tuple_to_vec!(@views(u[n+1:2n]), map(f -> f(s, s′, p.action, p.params), values(f_x′′)))
    end

    prob = ODEProblem{true}(dynamics!, u0, (times[1], times[end]), p)
    integrator = init(
        prob, BS3(),
        dtmin=0.01, force_dtmin=true, # we want to guarantee good solver speed
        dt=0.1,
        abstol=1e-3, save_everystep=false,
    )

    map(0:length(times)-1) do t
        if t != 0
            Δt = times[t+1]-times[t]
            p = (params=params, action=actions[t])
            integrator.p = p
            step!(integrator, Δt, true)
        end
        integrator.u
        # NamedTuple{u_keys}(tuple_from_vec(state_types, integrator.u))
    end
end

function tuple_to_vec!(arr, v::Union{Tuple, NamedTuple})
    i = Ref(0)
    function rec(r::Real)
        arr[i[]+=1] = r
        nothing
    end
    function rec(v::AbstractVector)
        j = i[]+=1
        arr[j:j+length(v)-1] .= v
        nothing
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
    count(::SVector{n}) where n = n
    sum(map(count, v))
end


function promote_numbers_type(v::Union{Tuple, NamedTuple})
    types = typeof.(values(v))
    Base.promote_eltype(types...)
end

function tuple_to_vec(v::Union{Tuple, NamedTuple})
    T = promote_numbers_type(v)
    vec = Vector{T}(undef, n_numbers(v))
    tuple_to_vec!(vec, v)
    vec
end

function tuple_from_vec(template::Tuple, vec)
    i::Int = 0
    read(::Real) = begin
        vec[i+=1]
    end
    read(::SVector{n}) where n = begin
        i+=1
        SVector{n}(vec[i:i+n-1])
    end
    read.(template)
end

function tuple_from_vec(template::NamedTuple, vec)
    NamedTuple{keys(template)}(tuple_from_vec(values(template), vec))
end


function to_named_tuple(var_dict)::NamedTuple
    (; (k.name => v for (k, v) in var_dict)...)
end