using ProgressMeter: Progress, next!, @showprogress, progress_pmap
using Distributed

const TimeSeries{T} = Vector{T}
const ObservationData = 
    NamedTuple{(:times, :obs_frames, :observations, :controls, :x0_dist)}

export DynamicsSketch, params_distribution, compile_motion_model, to_p_motion_model


"""
The sketch of a motion model that enables supervised learning from input-output data. 
The output transformation therefore needs to be a bijection.

Use [mk_motion_model](@ref) to construct the motion model from the sketch and components.
"""
@kwdef(
struct MotionModelSketch{F1, F2, F3}
    input_vars::Vector{Var}
    output_vars::Vector{Var}
    "genereate sketch inputs from state and control"
    inputs_transform::F1
    "transform (state, outputs, Δt) into next state"
    outputs_transform::F2
    "inversely compute sketch outputs from (state, next_state, Δt)"
    outputs_inv_transform::F3
end)

function Base.show(io::IO, ::Type{<:MotionModelSketch})
    print(io, "MotionModelSketch{...}")
end

function mk_motion_model(sketch::MotionModelSketch, comps::NamedTuple{names}) where names
    (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
        local inputs = sketch.inputs_transform(x, u)
        local outputs_dist = DistrIterator(map(f -> f(inputs), values(comps)))
        GenericSamplable(
            rng -> begin
                local out = NamedTuple{names}(rand(rng, outputs_dist))
                sketch.outputs_transform(x, out, Δt)# |> assert_finite
            end, 
            x1 -> begin 
                local out = sketch.outputs_inv_transform(x, x1, Δt)# |> assert_finite
                logpdf(outputs_dist, values(out))
            end
        )
    end
end

"""
A regression algorith specifies how to synthesize the dynamics from a given set of 
state transition data.

Need to implement:
 - [`fit_candidate_dynamics`](@ref)
"""
abstract type AbstractRegerssionAlgorithm end

"""
Fit multiple dynamics and return the best one according to the validation set.
"""
function fit_best_dynamics(
    alg::AbstractRegerssionAlgorithm,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)::NamedTuple{(:dynamics, :model_info, :optimizer_info, :display_info)}
    error("Not implemented.")
end

"""
Split the provided trajectories into training and validation set. 
Fit multiple dynamics and return the best one according to the validation set.
"""
function fit_best_dynamics(
    alg::AbstractRegerssionAlgorithm, 
    sketch::MotionModelSketch, 
    trajectories::Matrix{<:Vector},
    obs_data_list::Vector{<:ObservationData},
    train_split::Int,
    comps_σ_guess::Vector{Float64};
    n_fit_trajs::Int,
    n_valid_trajs::Int=size(trajectories, 1),
)
    @smart_assert train_split < size(trajectories, 2)
    @smart_assert n_fit_trajs <= size(trajectories, 1)
    @smart_assert n_valid_trajs <= size(trajectories, 1)

    train_data = trajectories[1:n_fit_trajs, 1:train_split]
    valid_data = trajectories[1:n_valid_trajs, train_split+1:end]

    inputs, outputs = construct_inputs_outputs(
        train_data, obs_data_list[1:train_split], sketch)
    valid_inputs, valid_outputs = construct_inputs_outputs(
        valid_data, obs_data_list[train_split+1:end], sketch)

    fit_best_dynamics(
        alg, sketch, (inputs, outputs), (valid_inputs, valid_outputs), comps_σ_guess)
end

"""
A sketch for some missing dynamics. 

With the missing dynamics provided (by e.g., a synthesizer), this skech can then be 
compiled into a parameterized dynamics of the form
`params -> (state, control, Δt) -> distribution_of_state`.
"""
@kwdef(
struct DynamicsSketch{F, G}
    inputs::Vector{Var}
    outputs::Vector{Var}
    params::OrderedDict{Var, GDistr}
    "state_to_inputs(state::NamedTuple, control::NamedTuple) -> inputs"
    state_to_inputs::F
    "outputs_to_state_dist(outputs, (; state..., control..., inputs...), Δt) -> distribution_of_state"
    outputs_to_state_dist::G
end)

params_distribution(sketch::DynamicsSketch) = begin
    DistrIterator((;(v.name => dist for (v, dist) in sketch.params)...))
end

inputs_distribution(sketch::DynamicsSketch) = begin
    DistrIterator((;(v.name => dist_for_shape[v.type.shape] for v in sketch.inputs)...))
end

function check_params_logp(x0, x0_dist, params, params_dist)
    x0_logp = logpdf(x0_dist, x0)
    isfinite(x0_logp) || error("x0_logp = $x0_logp, some value may be out of its support.\n
        x0=$x0,\nx0_dist=$x0_dist")
    params_logp = logpdf(params_dist, params)
    isfinite(params_logp) || error("params_logp = $params_logp, some value may be out of its support.")
end

function compile_motion_model(
    comps::NamedTuple,
    (; shape_env, comp_env, sketch, hide_type);
)
    funcs = map(comp -> compile(comp, shape_env, comp_env; hide_type), comps)
    sketch_core = _build_sketch_core(funcs)
    # hide_type && (sketch_core = WrappedFunc(sketch_core))
    WrappedFunc(to_p_motion_model(sketch_core, sketch))
end

function _build_sketch_core(funcs)
    x -> map(f->f(x), funcs)
end

function to_p_motion_model(
    sketch_core::Function, (; state_to_inputs, outputs_to_state_dist),
)
    (params::NamedTuple) -> (x::NamedTuple, u::NamedTuple, Δt::Real) -> begin
        local inputs = state_to_inputs(x, u)
        local prog_inputs = merge(inputs, params)
        local outputs = sketch_core(prog_inputs)
        local others = merge(x, u, params, prog_inputs)
        outputs_to_state_dist(outputs, others, Δt)::GDistr
    end
end

export VariableData

struct VariableData
    state_vars::Vector{Var}
    action_vars::Vector{Var}
end

#TODO: fix this
function VariableData(
    ::Val{order};
    states::OrderedDict{Var, <:NTuple{order, GDistr}},
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
        action_vars,
    )
end

_check_variable_types(vdata::VariableData, shape_env::ShapeEnv) = begin
    check_type(var, value) = begin
        t = shape_env[var.type]
        @smart_assert value isa t "Prior distribution for \
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
    (; comp_env, enum_result, vdata, sketch) = r
    (; state_vars, action_vars) = vdata
    param_vars = sketch.params |> keys |> collect
    hole_inputs, holes = sketch.inputs, sketch.outputs

    n_interest = prod(count_len(enum_result[v.type]) for v in holes)
    search_details = join((count_len(enum_result[v.type]) for v in holes), " * ")

    io = IOIndent(io)
    println(io, "===== Synthesis enumeration result =====")
    println(io, "search_space: $(pretty_number(n_interest)) = $search_details")
    println(io, "n_components: ", length(comp_env.signatures))
    println(io, "holes: $holes")
    println(io, "hole_inputs: $hole_inputs")
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
    param_vars = sketch.params |> keys |> collect
    dyn_vars = [sketch.inputs; param_vars]
    if type_pruning
        types_needed, _ = enumerate_types(
            comp_env, 
            Set(v.type for v in dyn_vars), 
            Set(v.type for v in sketch.outputs), 
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

include("deterministic_synthesis.jl")
include("posterior_sampling.jl")
include("probabilistic_synthesis.jl")

include("sparse_regression.jl")
include("em_synthesis.jl")
include("sindy_regression.jl")

include("synthesis_utils.jl")