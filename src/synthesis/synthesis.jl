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
        NamedTuple{output_names}(combine(input, hole_values)::NamedTuple)
    end
    CompiledFunc(f, ast, julia, any_return_type)
end

export VariableData
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

init_state_distribution(vdata::VariableData) = begin
    (; state_vars) = vdata
    x₀ = (;(s.name => vdata.states[s][1] for s in state_vars)...)
    x′₀ = (;(derivative(s.name) => vdata.states[s][2] for s in state_vars)...)
    DistrIterator(merge(x₀, x′₀))
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

export MapSynthesisResult
struct MapSynthesisResult{R}
    best_result::R
    stats::NamedTuple
    errored_programs::Vector
    sorted_results::Vector
end

get_top_results(r::MapSynthesisResult, top_k::Int) = begin
    rows = map(Iterators.take(r.sorted_results, top_k)) do (; score, f_x′′, params)
        (; score, f_x′′=(x -> x.ast).(f_x′′) , params)
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
    let (; score, f_x′′, params) = r.best_result
        println(io, "score: $score")
        println(io, "expressions: $(f_x′′.ast)")
        println(io, "params: $(params)")
    end
end

Base.size(v::Transducers.ProgressLoggingFoldable) = size(v.foldable)

export prog_size_prior
function prog_size_prior(decay::Float64)
    (comps) -> log(decay) * sum(ast_size.(comps); init=0) 
end

function Distributions.logpdf(dist::NamedTuple{ks}, v::NamedTuple{ks})::Real where ks
    sum(logpdf.(values(dist), values(v)))
end

include("synthesis_utils.jl")
include("deterministic_synthesis.jl")
include("probabilistic_synthesis.jl")