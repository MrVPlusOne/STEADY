using ProgressMeter: Progress, next!, @showprogress, progress_pmap
using Distributed

const TimeSeries{T} = Vector{T}

export DynamicsSketch
"""
A sketch for second-order system dynamics.
"""
struct DynamicsSketch{F}
    holes::Vector{Var}
    "combine_holes(inputs::NamedTuple, hole_values::NamedTuple) -> state_distribution"
    combine_holes::F
end


export compile_sketch
"Combine sketch hole expressions into an executable Julia function."
compile_sketch(
    comps::Tuple{Vararg{TAST}}, 
    hn::Val{hole_names},
    on::Val{output_names},
    (;shape_env, comp_env, combine, check_gradient),
) where {hole_names, output_names} = begin
    funcs = map(comp -> compile(comp, shape_env, comp_env; check_gradient), comps)
    julia = Expr(:tuple, (x -> x.julia).(funcs))
    ast = (; (=>).(hole_names, comps)...)
    compile_sketch(funcs, julia, ast, hn, on, combine)
end

compile_sketch(
    funcs::Tuple{Vararg{Function}}, julia, ast, 
    ::Val{hole_names}, ::Val{output_names}, combine,
) where {hole_names, output_names} = begin
    f = input -> let
        hole_values = NamedTuple{hole_names}(map(f -> f(input), funcs))
        combine(input::NamedTuple, hole_values)
    end
    CompiledFunc(f, ast, julia, any_return_type)
end

export VariableData
"""
The robot dynamics are assumed to be of the form 
`f(state, action, params) -> state_derivatives`.

## Fields
- `states`: maps each state variable (e.g. `position`) to the distribution of its initial 
value.
- `dynamics_params`: maps each dynamics parameters to its prior distribution.
- `action_vars`: a vector holding the action variables.
"""
@kwdef(
struct VariableData{system_order}
    states::OrderedDict{Var, GDistr}  
    dynamics_params::OrderedDict{Var, GDistr}
    action_vars::Vector{Var}
    param_vars::Vector{Var} = keys(dynamics_params) |> collect
    state_vars::Vector{Var} = keys(states) |> collect
end)

function VariableData(
    ::Val{order};
    states::OrderedDict{Var, <:NTuple{order, GDistr}},
    dynamics_params::OrderedDict{Var, <:GDistr},
    action_vars::Vector{Var},
    t_unit::PUnit=PUnits.Time,
) where order
    s_vars = keys(states) |> collect
    new_states = Tuple{Var, GDistr}[]
    for i in 1:order
        dists = [d[i] for (_, d) in states]
        append!(new_states, zip(s_vars, dists))
        s_vars = derivative.(s_vars, Ref(t_unit))
    end
    VariableData{order}(; 
        states = OrderedDict(new_states),
        dynamics_params,
        action_vars,
    )
end

to_distribution(vdata::VariableData) = let
    (; state_vars) = vdata
    all_dists = (
        x₀ = (;(s.name => vdata.states[s] for s in state_vars)...),
        params = (;(v.name => dist for (v, dist) in vdata.dynamics_params)...),
    )
    DistrIterator(map(DistrIterator, all_dists))
end

params_distribution(vdata::VariableData) = begin
    DistrIterator((;(v.name => dist for (v, dist) in vdata.dynamics_params)...))
end

init_state_distribution(vdata::VariableData) = begin
    (; state_vars) = vdata
    DistrIterator((;(s.name => vdata.states[s] for s in state_vars)...))
end

_check_variable_types(vdata::VariableData, shape_env::ShapeEnv) = begin
    check_type(var, value) = begin
        t = shape_env[var.type]
        @assert value isa t "Prior distribution for \
            $var produced value $value, expected type: $t"
    end

    (; states, dynamics_params) = vdata
    foreach([states; dynamics_params]) do (v, d)
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
    shape_env::ShapeEnv
    comp_env::ComponentEnv
    enum_result::EnumerationResult
end)

Base.show(io::IO, mime::MIME"text/plain", r::SynthesisEnumerationResult) = begin
    (; comp_env, enum_result, sketch) = r
    (; state_vars, action_vars, param_vars) = r.vdata
    holes = sketch.holes

    n_interest = prod(count_len(enum_result[v.type]) for v in holes)
    search_details = join((count_len(enum_result[v.type]) for v in holes), " * ")

    io = IOIndent(io)
    println(io, "===== Synthesis enumeration result =====")
    println(io, "search_space: $(pretty_number(n_interest)) = $search_details")
    println(io, "n_components: ", length(comp_env.signatures))
    println(io, "holes: $holes")
    println(io, "states: $state_vars")
    println(io, "actions: $action_vars")
    println(io, "params: $param_vars")

    println(io, "Candidate expressions:", Indent())
    for h in holes
        println(io, "---- For $h ----")
        show(io, DataFrame((expression=e,) for e in enum_result[h.type]))
        println(io)
    end
    print(io, Dedent())

    show(io, mime, enum_result)
end

"""
Enuemrate all programs with the correct return types up to some max AST size.
"""
function synthesis_enumeration(
    vdata::VariableData, sketch::DynamicsSketch,
    shape_env::ShapeEnv, comp_env::ComponentEnv, max_size; 
    pruner=NoPruner(), type_pruning=true,
)
    (; state_vars, action_vars, param_vars) = vdata
    dyn_vars = [state_vars; action_vars; param_vars]
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
        shape_env,
        comp_env,
        enum_result,
    )
end

export MapSynthesisResult
struct MapSynthesisResult{R}
    best_result::R
    stats::NamedTuple
    errored_programs::Vector
    sorted_results::Vector
end

get_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rows = map(Iterators.take(r.sorted_results, top_k)) do (; score, comps, params)
        (; score, comps , params)
    end
    rows
end

export get_top_results, show_top_results
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
    let (; score, comps, params) = r.best_result
        println(io, "score: $score")
        println(io, "expressions: $(comps)")
        println(io, "params: $(params)")
    end
end

include("synthesis_utils.jl")
include("deterministic_synthesis.jl")
include("probabilistic_synthesis.jl")